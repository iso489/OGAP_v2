"""Robust paired statistics for comparing two segmentation methods.

Segmentation metrics on a shared set of cases are *paired*, so comparisons use
the Wilcoxon signed-rank test (non-parametric, paired). Every comparison reports:
the test statistic, the raw p-value, a multiplicity-corrected p-value (Holm or
Benjamini-Hochberg across the metric family), the matched-pairs rank-biserial
effect size, and a bias-corrected-accelerated (BCa) bootstrap confidence interval
on the median paired difference.

Design choices that make the output defensible:

* **Pairing is explicit** — methods are aligned by case id; only cases present in
  both with finite values for a given metric enter that metric's test.
* **Per-metric NaN handling** — e.g. cases with an empty reference enhancing
  region (HD95 = NaN) drop from that metric only, and the dropped count is
  reported, never silently ignored.
* **Multiplicity control** across the whole metric family (default Holm, FWER).
* **Effect size + CI**, not just a p-value.
* **Lower-is-better awareness** for distance metrics (HD95).
* **Degenerate guards** — zero post-filter pairs, all-zero differences, and
  exact-test fallbacks are surfaced, not hidden.

Returns plain lists of dicts (no hard pandas dependency); callers may wrap in a
DataFrame. Depends only on numpy + scipy.stats.
"""
from __future__ import annotations

import logging
import math
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    from scipy import stats as _sps
    _SCIPY = True
except Exception:  # pragma: no cover
    _SCIPY = False

logger = logging.getLogger(__name__)

_DEFAULT_METRICS = ("dsc_wt", "dsc_tc", "dsc_et", "hd95_wt", "hd95_tc", "hd95_et")
_LOWER_IS_BETTER = {"hd95_wt", "hd95_tc", "hd95_et"}


def holm_correction(pvals: Sequence[float]) -> List[float]:
    """Holm-Bonferroni step-down adjusted p-values (controls FWER)."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    adj = np.empty(n, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (n - rank) * p[idx])
        adj[idx] = min(running, 1.0)
    return adj.tolist()


def benjamini_hochberg(pvals: Sequence[float]) -> List[float]:
    """Benjamini-Hochberg adjusted p-values (controls FDR)."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    adj = np.empty(n, dtype=float)
    prev = 1.0
    for rank in range(n - 1, -1, -1):
        idx = order[rank]
        prev = min(prev, p[idx] * n / (rank + 1))
        adj[idx] = min(prev, 1.0)
    return adj.tolist()


def _correct(pvals: Sequence[float], method: str) -> List[float]:
    if method == "none":
        return list(pvals)
    if method == "holm":
        return holm_correction(pvals)
    if method in ("bh", "fdr", "benjamini_hochberg"):
        return benjamini_hochberg(pvals)
    raise ValueError(f"Unknown correction {method!r}; expected holm | bh | none.")


def _rankdata(a: np.ndarray) -> np.ndarray:
    if _SCIPY:
        return _sps.rankdata(a)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(a.size, dtype=float)
    ranks[order] = np.arange(1, a.size + 1)
    return ranks


def rank_biserial(diff: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation effect size in [-1, 1]."""
    d = diff[diff != 0]
    if d.size == 0:
        return 0.0
    ranks = _rankdata(np.abs(d))
    total = ranks.sum()
    return float((ranks[d > 0].sum() - ranks[d < 0].sum()) / total) if total > 0 else 0.0


def _norm_ppf(p: float) -> float:
    if _SCIPY:
        return float(_sps.norm.ppf(p))
    # Acklam rational approximation
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def _norm_cdf(z: float) -> float:
    if _SCIPY:
        return float(_sps.norm.cdf(z))
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def bootstrap_median_diff_ci(diff: np.ndarray, n_boot: int = 10000, alpha: float = 0.05,
                             seed: int = 0, bca: bool = True) -> Tuple[float, float, float]:
    """BCa bootstrap CI for the median paired difference.

    The bias-corrected and accelerated interval is more accurate than the
    percentile interval for the skewed difference distributions typical of
    Dice/HD95. Falls back to percentile if the acceleration term is degenerate.
    Returns ``(point_estimate, ci_low, ci_high)``.
    """
    d = diff[np.isfinite(diff)]
    point = float(np.median(d)) if d.size else float("nan")
    if d.size < 2:
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot = np.median(d[rng.integers(0, d.size, size=(n_boot, d.size))], axis=1)
    if not bca:
        return point, float(np.percentile(boot, 100*alpha/2)), float(np.percentile(boot, 100*(1-alpha/2)))
    prop = min(max(np.mean(boot < point), 1e-6), 1 - 1e-6)
    z0 = _norm_ppf(prop)
    jack = np.array([np.median(np.delete(d, i)) for i in range(d.size)])
    jbar = jack.mean()
    den = 6.0 * (np.sum((jbar - jack) ** 2) ** 1.5)
    a = np.sum((jbar - jack) ** 3) / den if den != 0 else 0.0
    zl, zu = _norm_ppf(alpha/2), _norm_ppf(1 - alpha/2)

    def _adj(z):
        return _norm_cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))

    pl, pu = _adj(zl), _adj(zu)
    if not (0 < pl < pu < 1):
        pl, pu = alpha/2, 1 - alpha/2
    return point, float(np.percentile(boot, 100*pl)), float(np.percentile(boot, 100*pu))


def _significance_stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def _paired_arrays(results_a: Dict, results_b: Dict, metric: str
                   ) -> Tuple[np.ndarray, np.ndarray, int, int, int, int]:
    # Report roster sizes so a reduction below the full cohort is auditable: cases
    # present in only ONE method are silently outside the paired comparison, and a
    # reader must be able to see when n_pairs < |A|, |B| because the rosters differ.
    n_a, n_b = len(results_a), len(results_b)
    common = [cid for cid in results_a if cid in results_b]
    a_vals, b_vals, dropped = [], [], 0
    for cid in common:
        av, bv = results_a[cid].get(metric), results_b[cid].get(metric)
        if av is None or bv is None or not np.isfinite(av) or not np.isfinite(bv):
            dropped += 1
            continue
        a_vals.append(float(av))
        b_vals.append(float(bv))
    if n_a != n_b:
        logger.warning("paired test (%s): method rosters differ (|A|=%d, |B|=%d, "
                       "|A_and_B|=%d); %d case(s) lie outside the paired comparison.",
                       metric, n_a, n_b, len(common), abs(n_a - n_b))
    return np.asarray(a_vals), np.asarray(b_vals), len(a_vals), dropped, n_a, n_b


def paired_significance_table(
    results_a: Dict[str, Dict[str, float]],
    results_b: Dict[str, Dict[str, float]],
    metrics: Sequence[str] = _DEFAULT_METRICS,
    alpha: float = 0.05,
    correction: str = "holm",
    n_boot: int = 10000,
    seed: int = 0,
) -> List[Dict[str, object]]:
    """Paired Wilcoxon comparison of method A vs B across a metric family.

    Args:
        results_a, results_b: ``{case_id: {metric: value}}`` for the two methods.
        metrics: metric keys to test (the multiplicity family).
        alpha: significance level.
        correction: ``holm`` (FWER) | ``bh`` (FDR) | ``none``.
        n_boot, seed: bootstrap resamples and RNG seed for the difference CI.

    Returns:
        One row dict per metric with keys ``metric, n_pairs, n_dropped,
        statistic, p_value_raw, p_value_corrected, significant, stars,
        effect_size_r, median_diff, ci_low, ci_high, direction`` (``direction``
        accounts for HD95 being lower-is-better).
    """
    if not _SCIPY:
        raise RuntimeError("scipy is required for paired_significance_table.")
    rows: List[Dict[str, object]] = []
    raw_ps: List[float] = []
    for metric in metrics:
        a, b, n, dropped, n_a, n_b = _paired_arrays(results_a, results_b, metric)
        diff = a - b
        row: Dict[str, object] = {"metric": metric, "n_pairs": n, "n_dropped": dropped,
                                  "n_roster_a": n_a, "n_roster_b": n_b}
        if n < 1 or np.all(diff == 0):
            row.update(statistic=float("nan"), p_value_raw=1.0, effect_size_r=0.0,
                       median_diff=float(np.median(diff)) if n else float("nan"),
                       ci_low=float("nan"), ci_high=float("nan"), direction="tie")
            if n < 1:
                logger.warning("metric %s: no paired finite cases; reported n.s.", metric)
            raw_ps.append(1.0)
            rows.append(row)
            continue
        try:
            stat, p = _sps.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        except ValueError as exc:
            # Degenerate inputs (all-zero diffs / n<1) are already handled above; an
            # unexpected wilcoxon failure must NOT be silent in a published table. Log
            # loudly and report a conservative non-significant p with an auditable flag
            # (a NaN here would break the Holm/BH correction below).
            logger.warning("metric %s: wilcoxon raised ValueError (%s); reporting "
                           "conservative p=1.0 with a compute_error flag.", metric, exc)
            stat, p = float("nan"), 1.0
            row["compute_error"] = str(exc)
        point, lo, hi = bootstrap_median_diff_ci(diff, n_boot=n_boot, alpha=alpha, seed=seed)
        better = "tie" if point == 0 else ("A" if (point > 0) ^ (metric in _LOWER_IS_BETTER) else "B")
        row.update(statistic=float(stat), p_value_raw=float(p),
                   effect_size_r=rank_biserial(diff), median_diff=point,
                   ci_low=lo, ci_high=hi, direction=better)
        raw_ps.append(float(p))
        rows.append(row)

    for row, pc in zip(rows, _correct(raw_ps, correction)):
        row["p_value_corrected"] = float(pc)
        row["significant"] = bool(pc < alpha)
        row["stars"] = _significance_stars(pc)
    return rows


def wilcoxon_significance_table(results_a, results_b, alpha: float = 0.05,
                                correction: str = "holm", **kwargs):
    """Alias for :func:`paired_significance_table` over the default metric set."""
    return paired_significance_table(results_a, results_b, alpha=alpha,
                                     correction=correction, **kwargs)
