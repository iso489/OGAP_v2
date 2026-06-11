"""Validate that zero-cost proxies actually rank OGAP *segmentation* students.

The proxies in :mod:`ogap.nas.proxies` (SNIP, SynFlow, NASWOT/``jacob_cov``,
DisWOT) and the search loop are excellent, but every one of them was validated on
**image classification**. DisWOT's authors say so explicitly ("we evaluate
DisWOT in classification tasks; … future work will expand to … semantic
segmentation"), and the same caveat applies to the others. Using a proxy to rank
3-D segmentation students whose *final distillation Dice* is what we care about
is therefore an **unvalidated extrapolation** until we measure the proxy↔Dice
rank correlation ourselves.

This module is that measurement. Given, for a handful of candidate
architectures, (a) their zero-cost proxy scores and (b) a real held-out metric
(e.g. validation Dice after a short distillation run), it computes Spearman and
Kendall rank correlations and returns a verdict:

* ``"trustworthy"``  — strong, significant positive rank correlation; the proxy
  may be used to drive the search and reported as a validated estimator.
* ``"weak"``         — significant but modest; use only as a coarse pre-filter.
* ``"exploratory"``  — not significant / negative; NAS results driven by this
  proxy must be reported as exploratory, not as validated architecture claims.

This is the guard that keeps a reviewer from rejecting a NAS claim as an
unvalidated proxy extrapolation. Pure ``numpy`` (+ optional ``scipy`` for exact
p-values; a permutation fallback otherwise).
"""
from __future__ import annotations

from typing import Dict, Mapping, Optional

import numpy as np

# A proxy that drives the search must be POSITIVELY rank-correlated with the metric
# (the search maximizes it). The verdict is therefore gated on SIGNED rho and the best
# proxy is selected by signed rho — a strongly anti-correlated proxy is flagged, not
# certified (param_count is a capacity prior, not an accuracy predictor).
_TRUSTWORTHY_RHO = 0.7
_WEAK_RHO = 0.3
_ALPHA = 0.05


def _rank(x: np.ndarray) -> np.ndarray:
    """Average (tie-corrected) ranks, so the scipy-free Spearman fallback is
    correct under tied proxy / metric values, not merely ordinal."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=float)
    ranks[order] = np.arange(x.size, dtype=float)
    # Average the ranks within each group of equal values.
    sx = x[order]
    i = 0
    while i < sx.size:
        j = i + 1
        while j < sx.size and sx[j] == sx[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = ranks[order[i:j]].mean()
        i = j
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return float("nan")
    ra, rb = _rank(a), _rank(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def _kendall_tau(a: np.ndarray, b: np.ndarray) -> float:
    n = a.size
    if n < 2:
        return float("nan")
    c = d = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
            if s > 0:
                c += 1
            elif s < 0:
                d += 1
    total = c + d
    return float((c - d) / total) if total > 0 else float("nan")


def _perm_pvalue(a: np.ndarray, b: np.ndarray, observed: float,
                 reps: int = 5000, seed: int = 0) -> float:
    """Two-sided permutation p-value for a Spearman correlation."""
    if not np.isfinite(observed) or a.size < 3:
        return float("nan")
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(reps):
        perm = rng.permutation(b)
        if abs(_spearman(a, perm)) >= abs(observed) - 1e-12:
            count += 1
    return float((count + 1) / (reps + 1))


def proxy_metric_rank_correlation(
    proxy_scores: Mapping[str, Mapping[str, float]],
    true_metric: Mapping[str, float],
    higher_metric_is_better: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Per-proxy rank correlation against a real metric across architectures.

    Args:
        proxy_scores: ``{arch_id: {proxy_name: score}}`` (e.g. the per-arch output
            of :func:`ogap.nas.proxies.compute_proxies`).
        true_metric: ``{arch_id: measured_metric}`` — e.g. validation Dice from a
            short distillation run for each architecture.
        higher_metric_is_better: if False (e.g. HD95), the metric is negated so a
            *positive* reported correlation always means "proxy ranks good
            architectures high".

    Returns:
        ``{proxy_name: {"spearman", "kendall", "p_value", "n", "verdict"}}``.
    """
    archs = [a for a in proxy_scores if a in true_metric and np.isfinite(true_metric[a])]
    if len(archs) < 3:
        # Not enough architectures to say anything; everything is exploratory.
        proxies = sorted({p for a in proxy_scores for p in proxy_scores[a]})
        return {p: {"spearman": float("nan"), "kendall": float("nan"),
                    "p_value": float("nan"), "n": len(archs),
                    "verdict": "exploratory"} for p in proxies}

    metric = np.array([true_metric[a] for a in archs], dtype=float)
    if not higher_metric_is_better:
        metric = -metric
    proxies = sorted({p for a in archs for p in proxy_scores[a]})

    out: Dict[str, Dict[str, float]] = {}
    for p in proxies:
        vals = np.array([proxy_scores[a].get(p, np.nan) for a in archs], dtype=float)
        ok = np.isfinite(vals)
        if ok.sum() < 3:
            out[p] = {"spearman": float("nan"), "kendall": float("nan"),
                      "p_value": float("nan"), "n": int(ok.sum()),
                      "verdict": "exploratory"}
            continue
        a, m = vals[ok], metric[ok]
        rho = _spearman(a, m)
        tau = _kendall_tau(a, m)
        pval = _scipy_spearman_p(a, m)
        if not np.isfinite(pval):
            pval = _perm_pvalue(a, m, rho)
        out[p] = {"spearman": rho, "kendall": tau, "p_value": pval,
                  "n": int(ok.sum()), "verdict": _verdict(rho, pval)}
    return out


def _scipy_spearman_p(a: np.ndarray, m: np.ndarray) -> float:
    try:
        from scipy.stats import spearmanr
        _, p = spearmanr(a, m)
        return float(p)
    except Exception:
        return float("nan")


def _verdict(rho: float, pval: float) -> str:
    # Use SIGNED rho: the search MAXIMIZES the proxy, so only a POSITIVE proxy<->metric
    # rank correlation certifies it. Ranking by |rho| would certify a strongly
    # anti-correlated proxy as "trustworthy" — exactly a wrong-sign proxy, which would
    # drive the search toward the WORST architectures.
    if not np.isfinite(rho):
        return "exploratory"
    significant = np.isfinite(pval) and pval < _ALPHA
    if significant and rho >= _TRUSTWORTHY_RHO:
        return "trustworthy"
    if significant and rho >= _WEAK_RHO:
        return "weak"
    if significant and rho <= -_WEAK_RHO:
        return "anti_correlated"
    return "exploratory"


def validate_proxies(
    proxy_scores: Mapping[str, Mapping[str, float]],
    true_metric: Mapping[str, float],
    higher_metric_is_better: bool = True,
    primary_proxy: Optional[str] = None,
) -> Dict[str, object]:
    """End-to-end proxy-validity report with an overall recommendation.

    Returns the per-proxy correlations plus a top-level
    ``{"best_proxy", "best_spearman", "recommendation", "report"}``. The
    recommendation is a plain-English gate for the methods section: whether the
    NAS architecture selection may be reported as *validated* or must be labelled
    *exploratory*. If ``primary_proxy`` is given (the one actually used to drive
    the search), the recommendation is keyed to its verdict.
    """
    report = proxy_metric_rank_correlation(proxy_scores, true_metric, higher_metric_is_better)
    finite = {p: r for p, r in report.items() if np.isfinite(r["spearman"])}
    best_proxy = max(finite, key=lambda p: finite[p]["spearman"]) if finite else None
    best_rho = finite[best_proxy]["spearman"] if best_proxy else float("nan")

    keyed = report.get(primary_proxy) if primary_proxy else (report.get(best_proxy) if best_proxy else None)
    verdict = keyed["verdict"] if keyed else "exploratory"
    rec = {
        "trustworthy": "Proxy-Dice rank correlation is strong and significant - the "
                       "NAS architecture selection may be reported as validated.",
        "weak": "Proxy-Dice rank correlation is modest - use the proxy only as a "
                "coarse pre-filter and report final architectures with a short "
                "validation run, not on the proxy alone.",
        "exploratory": "Proxy-Dice rank correlation is not established on this "
                       "segmentation task - report NAS results as exploratory, "
                       "not as validated architecture claims.",
        "anti_correlated": "Proxy is significantly ANTI-correlated with Dice on this "
                           "task - maximizing it would select the WORST architectures. "
                           "Do not use it to drive the search; its sign is likely wrong.",
    }[verdict]
    return {
        "best_proxy": best_proxy,
        "best_spearman": best_rho,
        "primary_proxy": primary_proxy,
        "recommendation": rec,
        "verdict": verdict,
        "report": report,
    }
