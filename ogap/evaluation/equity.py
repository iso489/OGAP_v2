"""Health-equity evaluation: per-subgroup risk disparity for segmentation.

OGAP's mission is to bring brain-tumour segmentation to low-field / LMIC MRI,
where the under-served population is exactly the *distribution-shifted, under-
represented* subgroup. "Fairness Without Harm" (Pang et al., NeurIPS 2024,
Def. 3.1) gives the right metric for that mission: **risk disparity**, the gap
between a subgroup's risk and the overall risk,

    Δ_k = R_{Q_k}(model) - R_Q(model),

where ``R`` is the expected task error. For segmentation we instantiate the risk
of a higher-is-better metric (Dice) as ``R = 1 - metric`` and of a distance
metric (HD95, lower-is-better) as ``R = metric`` directly, so a **positive**
disparity always means "this subgroup is served worse than the cohort average".

The clinically actionable subgroup axis for OGAP is the acquisition domain:
field strength (e.g. 64 mT vs 1.5/3 T), scanner / site, or vendor. Reporting
``Δ_k`` per acquisition group - with bootstrap confidence intervals and a
Mann-Whitney test of the gap - makes "does the low-field group receive the same
quality of service?" a measured result rather than an aspiration. Theorem 5.2 of
the same paper bounds ``Δ_k`` by distribution shift + group gap + group size,
which is precisely what OGAP's physics augmentation, cross-scanner distillation
and domain-invariance machinery are designed to shrink - *without harming* the
high-field group ("fairness without harm").

Pure ``numpy`` (+ optional ``scipy`` for the test); CPU-only, data-free.
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# Metric directionality. Anything not listed is assumed higher-is-better.
_LOWER_IS_BETTER_PREFIXES: Tuple[str, ...] = ("hd95", "hd", "assd", "asd", "nsd_dist")


def _is_higher_better(metric: str) -> bool:
    m = metric.lower()
    return not any(m.startswith(p) for p in _LOWER_IS_BETTER_PREFIXES)


def _risk(values: np.ndarray, higher_is_better: bool) -> float:
    """Expected risk (error) of a metric sample: ``1 - metric`` for Dice-like
    scores, the value itself for distance-like errors."""
    if values.size == 0:
        return float("nan")
    mean = float(np.nanmean(values))
    return (1.0 - mean) if higher_is_better else mean


def _finite(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    return arr[np.isfinite(arr)]


def _bootstrap_ci(values: np.ndarray, *, reps: int, alpha: float,
                  seed: int) -> Tuple[float, float]:
    """Percentile bootstrap CI of the mean (NaN if too few samples)."""
    if values.size < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(reps, values.size))
    means = values[idx].mean(axis=1)
    lo = float(np.percentile(means, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(means, 100.0 * (1.0 - alpha / 2.0)))
    return (lo, hi)


def risk_disparity(
    per_group_metric: Mapping[str, Sequence[float]],
    metric_name: str = "dsc_et",
    higher_is_better: Optional[bool] = None,
) -> Dict[str, float]:
    """Per-group risk disparity ``Δ_k = R_k - R_overall`` for one metric.

    Args:
        per_group_metric: ``{group_label: [per-case metric values]}``. Groups are
            independent subgroups (different patients), e.g. ``{"64mT": [...],
            "1.5T": [...], "3T": [...]}``.
        metric_name: which metric these values are (controls directionality).
        higher_is_better: override the auto-detected direction.

    Returns:
        ``{group: Δ_k}`` plus ``"_overall_risk"`` and ``"_worst_group"`` /
        ``"_worst_disparity"``. Positive ``Δ_k`` ⇒ that group is served worse than
        the pooled cohort. Empty/all-NaN groups map to ``nan``.
    """
    hib = _is_higher_better(metric_name) if higher_is_better is None else higher_is_better
    finite = {g: _finite(v) for g, v in per_group_metric.items()}
    pooled = np.concatenate([a for a in finite.values() if a.size]) if any(
        a.size for a in finite.values()) else np.array([])
    overall = _risk(pooled, hib)
    out: Dict[str, float] = {"_overall_risk": overall}
    worst_g, worst_d = None, -np.inf
    for g, arr in finite.items():
        disp = _risk(arr, hib) - overall if (arr.size and np.isfinite(overall)) else float("nan")
        out[g] = disp
        if np.isfinite(disp) and disp > worst_d:
            worst_d, worst_g = disp, g
    out["_worst_group"] = worst_g if worst_g is not None else float("nan")
    out["_worst_disparity"] = worst_d if np.isfinite(worst_d) else float("nan")
    return out


def risk_disparity_ci(
    per_group_metric: Mapping[str, Sequence[float]],
    metric_name: str = "dsc_et",
    higher_is_better: Optional[bool] = None,
    *, reps: int = 2000, alpha: float = 0.05, seed: int = 0,
) -> Dict[str, Tuple[float, float, float]]:
    """Stratified-bootstrap CI on each Δ_k = R_k - R_overall.

    The headline equity quantity is the disparity Δ_k, but a point Δ_k with no
    interval can over- or under-state how much worse a subgroup is served. We
    resample WITHIN each group (preserving group sizes - groups are independent
    patients), recompute R_k and the pooled R_overall on each resample, and take
    the percentile interval of Δ_k. Returns ``{group: (Δ_k, lo, hi)}`` (lo/hi NaN
    for a group with <2 finite cases).
    """
    hib = _is_higher_better(metric_name) if higher_is_better is None else higher_is_better
    finite = {g: _finite(v) for g, v in per_group_metric.items()}
    point = risk_disparity(per_group_metric, metric_name, hib)
    groups = [g for g, a in finite.items() if a.size]
    if not groups:
        return {g: (float("nan"), float("nan"), float("nan")) for g in finite}
    rng = np.random.default_rng(seed)
    boot: Dict[str, list] = {g: [] for g in groups}
    for _ in range(reps):
        resampled = {g: finite[g][rng.integers(0, finite[g].size, finite[g].size)]
                     for g in groups}
        overall_b = _risk(np.concatenate(list(resampled.values())), hib)
        for g in groups:
            boot[g].append(_risk(resampled[g], hib) - overall_b)
    res: Dict[str, Tuple[float, float, float]] = {}
    for g in finite:
        d = float(point.get(g, float("nan")))
        if g in groups and finite[g].size >= 2:
            arr = np.asarray(boot[g])
            lo = float(np.percentile(arr, 100.0 * alpha / 2.0))
            hi = float(np.percentile(arr, 100.0 * (1.0 - alpha / 2.0)))
        else:
            lo = hi = float("nan")
        res[g] = (d, lo, hi)
    return res


def subgroup_equity_table(
    results_by_case: Mapping[str, Mapping[str, float]],
    group_of_case: Mapping[str, str],
    metrics: Sequence[str] = ("dsc_wt", "dsc_tc", "dsc_et", "hd95_wt", "hd95_tc", "hd95_et"),
    reference_group: Optional[str] = None,
    *,
    bootstrap_reps: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
):
    """Per-(group, metric) mean, bootstrap CI, risk disparity, and a gap test.

    Args:
        results_by_case: ``{case_id: {metric: value}}`` (e.g. the per-case output
            of :func:`ogap.evaluation.brats_metrics.compute_brats_metrics`).
        group_of_case: ``{case_id: group_label}`` - the acquisition subgroup of
            each case (field strength / scanner / site).
        metrics: metric keys to report.
        reference_group: if given, the disparity and Mann-Whitney test compare each
            group to THIS reference group (e.g. ``"3T"``); otherwise disparity is
            vs the pooled cohort and the test compares each group to "all others".
        bootstrap_reps, alpha, seed: CI controls.

    Returns:
        A pandas DataFrame with columns ``group, metric, n, mean, ci_low, ci_high,
        risk, risk_disparity, gap_p_value`` sorted by metric then group. Requires
        pandas; raises ImportError with guidance if unavailable.
    """
    try:
        import pandas as pd
    except Exception as exc:  # pragma: no cover
        raise ImportError("subgroup_equity_table requires pandas") from exc

    # Bucket per-case values by group.
    groups: Dict[str, Dict[str, List[float]]] = {}
    for cid, mvals in results_by_case.items():
        g = group_of_case.get(cid)
        if g is None:
            continue
        gd = groups.setdefault(g, {})
        for m in metrics:
            gd.setdefault(m, []).append(mvals.get(m))

    rows = []
    for m in metrics:
        hib = _is_higher_better(m)
        per_group = {g: _finite(groups[g].get(m, [])) for g in groups}
        pooled = np.concatenate([a for a in per_group.values() if a.size]) if any(
            a.size for a in per_group.values()) else np.array([])
        overall_risk = _risk(pooled, hib)
        for g in sorted(per_group):
            arr = per_group[g]
            mean = float(np.nanmean(arr)) if arr.size else float("nan")
            lo, hi = _bootstrap_ci(arr, reps=bootstrap_reps, alpha=alpha, seed=seed)
            if reference_group is not None and reference_group in per_group:
                ref_risk = _risk(per_group[reference_group], hib)
                disparity = (_risk(arr, hib) - ref_risk) if arr.size else float("nan")
                other = per_group[reference_group]
            else:
                disparity = (_risk(arr, hib) - overall_risk) if arr.size else float("nan")
                other = np.concatenate([per_group[o] for o in per_group if o != g and per_group[o].size]) \
                    if any(per_group[o].size for o in per_group if o != g) else np.array([])
            rows.append({
                "group": g, "metric": m, "n": int(arr.size),
                "mean": mean, "ci_low": lo, "ci_high": hi,
                "risk": _risk(arr, hib), "risk_disparity": disparity,
                "gap_p_value": _gap_test(arr, other),
            })
    return pd.DataFrame(rows, columns=["group", "metric", "n", "mean", "ci_low",
                                       "ci_high", "risk", "risk_disparity",
                                       "gap_p_value"])


def _gap_test(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided Mann-Whitney U p-value for an independent-group gap (NaN if
    either side is too small or scipy is unavailable)."""
    if a.size < 1 or b.size < 1:
        return float("nan")
    try:
        from scipy.stats import mannwhitneyu
    except Exception:  # pragma: no cover
        return float("nan")
    if np.all(a == a[0]) and np.all(b == b[0]) and a[0] == b[0]:
        return 1.0
    try:
        _, p = mannwhitneyu(a, b, alternative="two-sided")
        return float(p)
    except ValueError:
        return float("nan")


def fairness_without_harm_check(
    baseline_per_group: Mapping[str, Sequence[float]],
    candidate_per_group: Mapping[str, Sequence[float]],
    metric_name: str = "dsc_et",
    reference_group: Optional[str] = None,
    higher_is_better: Optional[bool] = None,
    harm_tol: float = 0.0,
) -> Dict[str, object]:
    """Does ``candidate`` reduce subgroup disparity *without harming* the
    reference (well-served) group? - the "fairness without harm" criterion.

    Args:
        baseline_per_group / candidate_per_group: ``{group: [metric values]}`` for
            the two models (e.g. without vs with physics augmentation + KD).
        metric_name: metric + directionality.
        reference_group: the well-served group that must not be harmed (e.g.
            ``"3T"`` / high-field). If None, uses the group with the lowest
            baseline risk.
        harm_tol: allowed worsening of the reference group's mean metric before it
            counts as "harm" (e.g. ``0.005`` Dice).

    Returns:
        ``{"baseline_worst_disparity", "candidate_worst_disparity",
        "disparity_reduced", "reference_group", "reference_delta",
        "reference_harmed", "fairness_without_harm"}``.
    """
    hib = _is_higher_better(metric_name) if higher_is_better is None else higher_is_better
    base = risk_disparity(baseline_per_group, metric_name, hib)
    cand = risk_disparity(candidate_per_group, metric_name, hib)

    if reference_group is None:
        # Lowest-risk (best-served) group in the baseline.
        ref_candidates = {g: base[g] for g in baseline_per_group if np.isfinite(base.get(g, np.nan))}
        reference_group = min(ref_candidates, key=ref_candidates.get) if ref_candidates else None

    def _mean(d: Mapping[str, Sequence[float]], g: Optional[str]) -> float:
        if g is None or g not in d:
            return float("nan")
        arr = _finite(d[g])
        return float(np.nanmean(arr)) if arr.size else float("nan")

    ref_base_mean = _mean(baseline_per_group, reference_group)
    ref_cand_mean = _mean(candidate_per_group, reference_group)
    # Positive ref_delta == reference group improved (Dice up / distance down).
    ref_delta = (ref_cand_mean - ref_base_mean) if hib else (ref_base_mean - ref_cand_mean)
    reference_harmed = bool(np.isfinite(ref_delta) and ref_delta < -abs(harm_tol))

    disparity_reduced = bool(
        np.isfinite(cand["_worst_disparity"]) and np.isfinite(base["_worst_disparity"])
        and cand["_worst_disparity"] < base["_worst_disparity"]
    )
    return {
        "baseline_worst_disparity": base["_worst_disparity"],
        "candidate_worst_disparity": cand["_worst_disparity"],
        "disparity_reduced": disparity_reduced,
        "reference_group": reference_group,
        "reference_delta": ref_delta,
        "reference_harmed": reference_harmed,
        "fairness_without_harm": bool(disparity_reduced and not reference_harmed),
    }
