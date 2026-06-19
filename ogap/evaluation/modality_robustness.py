"""Modality-robustness evaluation and multi-model statistical analysis.

Answers the deployment question: how does segmentation accuracy degrade when
only a subset of the four MRI modalities is available, and does that degradation
differ between models (e.g. teacher vs. full-precision student vs. INT8 student)?

Pipeline:
1. :func:`evaluate_modality_combinations` masks each of the 15 modality subsets
   and evaluates a model per case, reusing
   :func:`ogap.evaluation.brats_metrics.compute_brats_metrics`.
2. :func:`modality_robustness_table` aggregates mean/std per model x combo x metric.
3. :func:`friedman_test` / :func:`multimodel_significance_matrix` /
   :func:`full_significance_sweep` provide the omnibus + post-hoc paired tests.
4. :func:`modality_degradation_analysis` quantifies the degradation curve.
5. :func:`quantize_student_int8` is a CPU INT8 *proxy* (Linear heads only,
   strict-by-default - it refuses conv-heavy students); the real INT8 conv model
   for published comparisons is the ONNX/onnxruntime export.

Channel order follows the project convention; verify against the data loader
before use. ``compute_brats_metrics`` may emit ``np.nan`` HD95 when a subregion
is absent in target or prediction; downstream tests/aggregation handle NaN.
"""
from __future__ import annotations

import itertools
import logging
from typing import Dict, FrozenSet, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from .brats_metrics import compute_brats_metrics
from .stats import benjamini_hochberg, holm_correction, rank_biserial

logger = logging.getLogger(__name__)

# BraTS standard channel order - verify against the data loader's mapping.
MODALITY_CHANNELS: Dict[str, int] = {"T1": 0, "T1ce": 1, "T2": 2, "FLAIR": 3}
CHANNEL_NAMES: Dict[int, str] = {v: k for k, v in MODALITY_CHANNELS.items()}

# All 15 non-empty modality subsets: C(4,1)=4 + C(4,2)=6 + C(4,3)=4 + C(4,4)=1.
# The four single-modality cases are realistic LMIC acquisitions (a site with
# only T1, or only FLAIR) and the hardest missing-modality stress test, so they
# are part of the sweep - not dropped.
ALL_COMBINATIONS: List[FrozenSet[int]] = [
    frozenset({0}), frozenset({1}), frozenset({2}), frozenset({3}),
    frozenset({0, 1}), frozenset({0, 2}), frozenset({0, 3}),
    frozenset({1, 2}), frozenset({1, 3}), frozenset({2, 3}),
    frozenset({0, 1, 2}), frozenset({0, 1, 3}),
    frozenset({0, 2, 3}), frozenset({1, 2, 3}),
    frozenset({0, 1, 2, 3}),
]

_DEFAULT_METRICS = ("dsc_wt", "dsc_tc", "dsc_et", "hd95_wt", "hd95_tc", "hd95_et")


def combo_label(combo: FrozenSet[int]) -> str:
    """Human-readable label sorted by channel index, e.g. ``'T1+T1ce+FLAIR'``."""
    return "+".join(CHANNEL_NAMES[i] for i in sorted(combo))


def apply_modality_mask(x: torch.Tensor, combo: FrozenSet[int],
                        fill: str = "zero") -> torch.Tensor:
    """Return a copy of ``x`` (B, C, ...) with channels outside ``combo`` filled.

    Args:
        x: input volume ``(B, C, D, H, W)``.
        combo: channel indices to keep.
        fill: ``"zero"`` sets missing channels to 0; ``"mean"`` sets them to the
            per-voxel mean across the present channels.

    Returns:
        A new tensor; ``x`` is not modified.
    """
    if fill not in ("zero", "mean"):
        raise ValueError(f"fill must be 'zero' or 'mean', got {fill!r}.")
    present = sorted(combo)
    missing = [c for c in range(x.shape[1]) if c not in combo]
    out = x.clone()
    if not missing:
        return out
    if fill == "zero":
        out[:, missing] = 0
    else:
        mean_present = x[:, present].mean(dim=1, keepdim=True)  # (B,1,D,H,W)
        for c in missing:
            out[:, c:c + 1] = mean_present
    return out


def _logits_to_pred(out) -> torch.Tensor:
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out.argmax(dim=1)


@torch.no_grad()
def evaluate_modality_combinations(
    model: nn.Module,
    dataloader,
    device: str = "cpu",
    fill: str = "zero",
    voxel_spacing_mm: Sequence[float] = (1.0, 1.0, 1.0),
    hd95_empty_policy: str = "nan",
) -> Dict[str, Dict[str, dict]]:
    """Evaluate a model over all 11 modality subsets.

    Args:
        model: callable mapping ``(B, C, D, H, W)`` -> logits (or a tuple whose
            first element is logits).
        dataloader: yields ``(volume, label)`` or ``(volume, label, case_ids)``
            with ``volume`` shape ``(B, 4, D, H, W)`` and integer ``label``
            ``(B, D, H, W)``; ``case_ids`` (length B) become the result keys.
        device: torch device string.
        fill: missing-channel strategy (see :func:`apply_modality_mask`).
        voxel_spacing_mm: passed to :func:`compute_brats_metrics` for HD95.
        hd95_empty_policy: "nan" (default; absent subregions drop from aggregation) or
            "penalty" (BraTS-leaderboard convention - substitute the penalty distance so
            missed ET/TC regions are penalised, not silently dropped). "penalty" is
            recommended for the degradation study, where absent regions are the signal.

    Returns:
        ``{combo_label: {case_id: {metric: value}}}``. If the dataloader yields a
        3-tuple ``(volume, label, case_ids)`` those real ids key the results (so
        they can be joined to per-case metadata); otherwise ``case_id`` is
        assigned in iteration order as ``case_{index:04d}``.
    """
    model = model.to(device).eval()
    results: Dict[str, Dict[str, dict]] = {combo_label(c): {} for c in ALL_COMBINATIONS}
    case_idx = 0
    for batch in dataloader:
        # Accept (volume, label) or (volume, label, case_ids). Real case ids let
        # the robustness table be joined to per-case metadata (field strength /
        # cohort) for the LMIC-vs-HIC disparity analysis; a 2-tuple loader falls
        # back to positional case_{idx} ids.
        if len(batch) == 3:
            volume, label, case_ids = batch
            batch_ids = [str(c) for c in case_ids]
        else:
            volume, label = batch
            batch_ids = [f"case_{case_idx + i:04d}" for i in range(volume.shape[0])]
        volume = volume.to(device)
        label = label.to(device)
        case_idx += volume.shape[0]
        for combo in ALL_COMBINATIONS:
            masked = apply_modality_mask(volume, combo, fill=fill)
            pred = _logits_to_pred(model(masked))
            lbl = combo_label(combo)
            for i, cid in enumerate(batch_ids):
                results[lbl][cid] = compute_brats_metrics(
                    pred[i], label[i], voxel_spacing_mm=voxel_spacing_mm,
                    hd95_empty_policy=hd95_empty_policy)
    return results


def _n_modalities_from_label(lbl: str) -> int:
    return len(lbl.split("+"))


def _combo_sort_key(lbl: str):
    return (_n_modalities_from_label(lbl), lbl)


def modality_robustness_table(
    results_per_model: Dict[str, Dict[str, dict]],
    metrics: Optional[Sequence[str]] = None,
):
    """Aggregate mean and std per model x modality combination x metric.

    Args:
        results_per_model: ``{model_name: output_of_evaluate_modality_combinations}``.
        metrics: metric keys to aggregate (defaults to DSC + HD95 per region).

    Returns:
        A pandas DataFrame indexed by ``combo_label`` (sorted by n_modalities then
        alphabetically) with MultiIndex columns ``(stat_metric, model_name)`` where
        ``stat_metric`` is e.g. ``dsc_et_mean`` / ``dsc_et_std``, plus a flat
        ``n_modalities`` column.
    """
    import pandas as pd

    metrics = list(metrics or _DEFAULT_METRICS)
    models = list(results_per_model)
    combos = sorted({lbl for m in models for lbl in results_per_model[m]},
                    key=_combo_sort_key)
    col_tuples, data = [], {}
    for metric in metrics:
        for stat in ("mean", "std"):
            for model in models:
                col_tuples.append((f"{metric}_{stat}", model))
    for lbl in combos:
        row = {}
        for metric in metrics:
            for model in models:
                vals = [v.get(metric) for v in results_per_model[model].get(lbl, {}).values()]
                arr = np.array([x for x in vals if x is not None and np.isfinite(x)], dtype=float)
                row[(f"{metric}_mean", model)] = float(arr.mean()) if arr.size else float("nan")
                # Sample std (ddof=1), NaN for n<2 - population std (ddof=0) returns
                # 0.0 for a single case, which reads as "zero variability", not "n=1".
                row[(f"{metric}_std", model)] = float(arr.std(ddof=1)) if arr.size >= 2 else float("nan")
        data[lbl] = row
    df = pd.DataFrame.from_dict(data, orient="index")
    df = df.reindex(columns=pd.MultiIndex.from_tuples(col_tuples))
    df["n_modalities"] = [_n_modalities_from_label(lbl) for lbl in df.index]
    return df


def friedman_test(per_case_by_model: Dict[str, List[float]]) -> dict:
    """Friedman omnibus test for >=3 related samples.

    Args:
        per_case_by_model: ``{model_name: [per-case metric values]}`` with the
            same case order across models.

    Returns:
        ``{"statistic": float, "p_value": float, "significant_at_0.05": bool}``.

    Raises:
        ValueError: if fewer than 3 models are provided (Friedman requires >=3
            related groups), the per-model arrays differ in length, or there are
            fewer than 3 paired cases.
    """
    from scipy.stats import friedmanchisquare

    models = list(per_case_by_model)
    if len(models) < 3:
        raise ValueError(
            f"Friedman test requires >=3 related samples, got {len(models)}.")
    arrays = [np.asarray(per_case_by_model[m], dtype=float) for m in models]
    lengths = {a.size for a in arrays}
    if len(lengths) != 1:
        raise ValueError(f"per-model arrays must share length, got sizes {lengths}.")
    if arrays[0].size < 3:
        raise ValueError("Friedman test requires >=3 paired cases.")
    stat, p = friedmanchisquare(*arrays)
    return {"statistic": float(stat), "p_value": float(p),
            "significant_at_0.05": bool(p < 0.05)}


def _aligned_values(results_per_model: Dict[str, Dict[str, dict]],
                    combo_lbl: str, metric: str):
    """Return ``{model: np.array}`` aligned on the cases finite for all models."""
    models = list(results_per_model)
    per_model_cases = {m: results_per_model[m].get(combo_lbl, {}) for m in models}
    common = set.intersection(*[set(c) for c in per_model_cases.values()]) if models else set()
    keep = []
    for cid in sorted(common):
        vals = [per_model_cases[m][cid].get(metric) for m in models]
        if all(v is not None and np.isfinite(v) for v in vals):
            keep.append(cid)
    return {m: np.array([per_model_cases[m][cid][metric] for cid in keep], dtype=float)
            for m in models}, keep


def _significance_stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def _correct(pvals: Sequence[float], method: str) -> List[float]:
    if method == "none":
        return list(pvals)
    if method == "holm":
        return holm_correction(pvals)
    if method in ("bh", "fdr", "benjamini_hochberg"):
        return benjamini_hochberg(pvals)
    raise ValueError(f"Unknown correction {method!r}; expected holm | bh | none.")


def multimodel_significance_matrix(
    results_per_model: Dict[str, Dict[str, dict]],
    combo: FrozenSet[int],
    correction: str = "holm",
    metrics: Optional[Sequence[str]] = None,
):
    """Friedman omnibus + all pairwise Wilcoxon tests for one modality combination.

    For each metric a Friedman row is emitted (omnibus across all models), then
    one row per model pair with a paired Wilcoxon test and rank-biserial effect
    size. Pairwise rows are always emitted (deterministic row count); the omnibus
    result is reported alongside via ``friedman_p`` / ``friedman_significant`` so
    consumers can gate interpretation without the row set changing. Multiple-
    comparison correction is applied across the pairwise tests for this combo
    (``n_pairs x n_metrics``).

    Args:
        results_per_model: output of :func:`evaluate_modality_combinations` per model.
        combo: the modality subset to test.
        correction: ``holm`` | ``bh`` | ``none`` applied to pairwise p-values.
        metrics: metric family (defaults to DSC + HD95 per region).

    Returns:
        A pandas DataFrame with columns ``combo_label, metric, pair, friedman_p,
        friedman_significant, statistic, p_value_raw, p_value_corrected,
        significant, stars, effect_size_r`` (one Friedman row per metric + one
        row per (metric, pair)).
    """
    import pandas as pd
    from scipy.stats import wilcoxon

    metrics = list(metrics or _DEFAULT_METRICS)
    lbl = combo_label(combo)
    models = list(results_per_model)
    pairs = list(itertools.combinations(models, 2))

    friedman_rows, pairwise_rows, raw_ps = [], [], []
    for metric in metrics:
        aligned, _ = _aligned_values(results_per_model, lbl, metric)
        fp = float("nan")
        if len(models) >= 3 and aligned and min(a.size for a in aligned.values()) >= 3:
            try:
                fp = friedman_test({m: aligned[m].tolist() for m in models})["p_value"]
            except ValueError:
                fp = float("nan")
        friedman_rows.append({
            "combo_label": lbl, "metric": metric, "pair": "FRIEDMAN",
            "friedman_p": fp, "friedman_significant": bool(np.isfinite(fp) and fp < 0.05),
            "statistic": float("nan"), "p_value_raw": float("nan"),
            "effect_size_r": float("nan"),
        })
        for a_name, b_name in pairs:
            a, b = aligned.get(a_name), aligned.get(b_name)
            stat, p, eff = float("nan"), 1.0, 0.0
            if a is not None and a.size >= 1 and not np.all(a - b == 0):
                try:
                    stat, p = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
                except ValueError as exc:
                    # All-zero diffs handled above; an unexpected failure must not be a
                    # silent p=1.0 in a published table. Conservative (no false
                    # significance) but logged so it is auditable.
                    logger.warning("combo %s metric %s pair %s vs %s: wilcoxon failed "
                                   "(%s); reporting conservative p=1.0.", lbl, metric,
                                   a_name, b_name, exc)
                    stat, p = float("nan"), 1.0
                eff = rank_biserial(a - b)
            pairwise_rows.append({
                "combo_label": lbl, "metric": metric, "pair": f"{a_name} vs {b_name}",
                "friedman_p": float("nan"),
                "friedman_significant": bool(np.isfinite(fp) and fp < 0.05),
                "statistic": float(stat), "p_value_raw": float(p), "effect_size_r": float(eff),
            })
            raw_ps.append(float(p))

    corrected = _correct(raw_ps, correction)
    for row, pc in zip(pairwise_rows, corrected):
        row["p_value_corrected"] = float(pc)
        row["significant"] = bool(pc < 0.05)
        row["stars"] = _significance_stars(pc)
    for row in friedman_rows:
        row["p_value_corrected"] = float("nan")
        row["significant"] = row["friedman_significant"]
        row["stars"] = _significance_stars(row["friedman_p"])

    cols = ["combo_label", "metric", "pair", "friedman_p", "friedman_significant",
            "statistic", "p_value_raw", "p_value_corrected", "significant",
            "stars", "effect_size_r"]
    pr_by_metric: Dict[str, list] = {m: [] for m in metrics}
    for r in pairwise_rows:
        pr_by_metric[r["metric"]].append(r)
    ordered = []
    for fr in friedman_rows:
        ordered.append(fr)
        ordered.extend(pr_by_metric[fr["metric"]])
    return pd.DataFrame(ordered, columns=cols)


def full_significance_sweep(
    results_per_model: Dict[str, Dict[str, dict]],
    correction: str = "holm",
    metrics: Optional[Sequence[str]] = None,
):
    """Run :func:`multimodel_significance_matrix` for all 11 modality combinations.

    Correction is applied *per combination*: each modality combination is treated as a
    SEPARATE confirmatory family (FWER controlled within its own n_pairs x n_metrics
    block), NOT across the whole 11-combo sweep. Interpret each combination independently
    and gate pairwise claims on that combination's Friedman omnibus; the concatenated
    table does NOT control a single family-wise error rate over all combinations. Returns
    a single concatenated DataFrame.
    """
    import pandas as pd
    frames = [multimodel_significance_matrix(results_per_model, combo,
                                             correction=correction, metrics=metrics)
              for combo in ALL_COMBINATIONS]
    return pd.concat(frames, ignore_index=True)


def modality_degradation_analysis(
    results_per_model: Dict[str, Dict[str, dict]],
    metric: str = "dsc_et",
):
    """Quantify how a metric degrades as modalities are removed.

    For each model and each n-modality level, reports the mean/std metric over
    all combinations at that level, the absolute drop from the 4-modality
    baseline, and the percentage degradation. Within each n-level it also adds
    Holm-corrected pairwise Wilcoxon p-values between models, paired on PER-CASE
    values pooled across the combinations at that level (not per-combination means).

    Args:
        results_per_model: output of :func:`evaluate_modality_combinations` per model.
        metric: metric to analyse (default ET Dice - the most sensitive region).

    Returns:
        A pandas DataFrame with columns ``model, n_modalities, combo_label,
        mean_metric, std_metric, delta_from_4mod, pct_degradation`` plus
        ``p_<a>_vs_<b>`` columns added per n-level.
    """
    import pandas as pd
    from scipy.stats import wilcoxon

    models = list(results_per_model)
    combo_means: Dict[str, Dict[str, float]] = {m: {} for m in models}
    for m in models:
        for lbl, cases in results_per_model[m].items():
            vals = np.array([c.get(metric) for c in cases.values()
                             if c.get(metric) is not None and np.isfinite(c.get(metric))],
                            dtype=float)
            combo_means[m][lbl] = float(vals.mean()) if vals.size else float("nan")

    full_lbl = combo_label(frozenset({0, 1, 2, 3}))
    rows = []
    for m in models:
        base = combo_means[m].get(full_lbl, float("nan"))
        by_n: Dict[int, List[float]] = {}
        for lbl, mu in combo_means[m].items():
            by_n.setdefault(_n_modalities_from_label(lbl), []).append(mu)
        for n in sorted(by_n):
            arr = np.array(by_n[n], dtype=float)
            mean_n = float(np.nanmean(arr)) if arr.size else float("nan")
            std_n = float(np.nanstd(arr)) if arr.size else float("nan")
            delta = (base - mean_n) if (np.isfinite(base) and np.isfinite(mean_n)) else float("nan")
            pct = (delta / base * 100.0) if (np.isfinite(delta) and np.isfinite(base) and base != 0.0) else float("nan")
            rows.append({"model": m, "n_modalities": n, "combo_label": f"{n}-mod-mean",
                         "mean_metric": mean_n, "std_metric": std_n,
                         "delta_from_4mod": (0.0 if (n == 4 and np.isfinite(base)) else delta),
                         "pct_degradation": (0.0 if (n == 4 and np.isfinite(base)) else pct)})

    df = pd.DataFrame(rows)
    pairs = list(itertools.combinations(models, 2))
    # Pairwise model comparison at each n-level on PER-CASE paired values, POOLED across
    # the combos at that level - not per-combination MEANS. The means version used the
    # number of combinations as the sample size (6/4/1), which conflates between-combo
    # variance for within-method case variance and degenerates to a single observation
    # (scipy returns p=1.0) at the 4-modality level. _aligned_values pairs cases finite
    # for every model; combos sourced from the intersection so a combo one model skipped
    # does not KeyError.
    common_combos = set.intersection(*[set(combo_means[m]) for m in models]) if models else set()
    for n in sorted({r["n_modalities"] for r in rows}):
        combos_n = sorted([lbl for lbl in common_combos
                           if _n_modalities_from_label(lbl) == n])
        pooled: Dict[str, list] = {m: [] for m in models}
        for c in combos_n:
            aligned, _ = _aligned_values(results_per_model, c, metric)
            for m in models:
                pooled[m].extend(aligned[m].tolist())
        raw, labels = [], []
        for a, b in pairs:
            va = np.asarray(pooled[a], dtype=float)
            vb = np.asarray(pooled[b], dtype=float)
            ok = np.isfinite(va) & np.isfinite(vb)
            if ok.sum() >= 1 and not np.all((va - vb)[ok] == 0):
                try:
                    _, p = wilcoxon(va[ok], vb[ok])
                except ValueError as exc:
                    logger.warning("degradation n=%d pair %s vs %s: wilcoxon failed "
                                   "(%s); reporting conservative p=1.0.", n, a, b, exc)
                    p = 1.0
            else:
                p = 1.0
            raw.append(float(p)); labels.append(f"p_{a}_vs_{b}")
        corr = holm_correction(raw) if raw else []
        for lab, pc in zip(labels, corr):
            df.loc[df["n_modalities"] == n, lab] = pc
    return df


def quantize_student_int8(student_model: nn.Module, strict: bool = True) -> nn.Module:
    """Dynamic INT8 quantization for CPU-side testing - Linear/RNN layers only.

    IMPORTANT: ``torch.quantization.quantize_dynamic`` only has dynamic INT8
    implementations for ``Linear`` (and the RNN family); **3D convolutions are NOT
    dynamically quantizable**. For OGAP's conv-heavy student this therefore
    quantizes essentially nothing beyond the small ``Linear`` heads (e.g. the RANO
    head) - it is a coarse CPU *proxy*, NOT a faithful INT8 conv model. The real
    deployment INT8 path (with quantized convolutions) is the ONNX export +
    onnxruntime/OpenVINO pipeline; use THAT for any INT8-vs-FP32 number that backs
    a published claim. The original model is not modified.

    Args:
        student_model: the model to (proxy-)quantize.
        strict: **default True** - *raise* when the model contains convolutions
            instead of returning a misleading FP32 proxy. Safe-by-default so a
            conv-heavy student can never be reported as "INT8" from this proxy;
            the real INT8 conv model is the ONNX/onnxruntime export. Pass
            ``strict=False`` only for a quick CPU smoke-test of the Linear heads,
            never for a number that backs a published claim. [C1]

    Raises:
        RuntimeError: if dynamic quantization / a backend engine is unavailable, or
            (when ``strict``) if the model contains conv layers this path cannot
            quantize.
    """
    import copy
    qd = getattr(getattr(torch, "quantization", None), "quantize_dynamic", None)
    if qd is None and getattr(torch, "ao", None) is not None:
        qd = getattr(torch.ao.quantization, "quantize_dynamic", None)
    if qd is None:
        raise RuntimeError("torch dynamic quantization is unavailable in this build.")
    engines = [e for e in getattr(torch.backends.quantized, "supported_engines", ())
               if e != "none"]
    if not engines:
        raise RuntimeError(
            "dynamic INT8 quantization requires a quantized backend engine "
            "(fbgemm/qnnpack); none is available in this torch build.")
    # Weight prepacking needs an *active* engine; the default can be "none" even
    # when one is compiled in. Select a real engine (fbgemm preferred on x86,
    # the only listed option elsewhere) without overriding an explicit choice.
    if torch.backends.quantized.engine == "none":
        torch.backends.quantized.engine = engines[0]
    if any(isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d))
           for m in student_model.modules()):
        msg = ("quantize_student_int8: dynamic quantization does not quantize conv "
               "layers, so the returned model keeps FP32 convolutions - a CPU proxy "
               "only. Use the ONNX/onnxruntime path for a true INT8 conv model.")
        if strict:
            raise RuntimeError(
                msg + " (strict=True: refusing to return a proxy that could be "
                "mistaken for a real INT8 conv model in reported results). [C1]")
        logger.warning(msg)
    model = copy.deepcopy(student_model).eval()
    # Only Linear (and RNN-family) layers have dynamic INT8 kernels; passing
    # Conv3d here was a silent no-op, so we quantise the layers that actually can.
    return qd(model, {nn.Linear}, dtype=torch.qint8)
