"""Conformal prediction with honest small-dataset and group-balanced guarantees.

OGAP already flags distribution shift (Mahalanobis / CNF density, see
``ogap.ood.cnf``) and needs to turn that into a *calibrated abstention rule* for
clinical deployment: "is this low-field scan one the model can be trusted on?".
Split conformal prediction (Vovk; Angelopoulos & Bates) gives a distribution-
free coverage guarantee for that decision. But two facts, emphasised by
Sánchez-Domínguez et al. (2025, "Reliable Statistical Guarantees for Conformal
Predictors with Small Datasets"), make the *textbook* guarantee misleading for
exactly OGAP's setting:

1. **The standard guarantee is marginal (ensemble-averaged over calibration
   draws).** For a *single* predictor calibrated on a *small* set, the realised
   coverage is a random variable ``Cov ~ Beta(k, n+1-k)`` with
   ``k = ceil((n+1)(1-alpha))`` - and for small ``n`` this Beta has a wide
   spread, so the coverage a deployed model actually achieves can sit well below
   the nominal ``1-alpha``. BraTS-Africa is small, so we must report this spread
   (a coverage CI), not just the nominal level.
2. **Coverage must be group-balanced.** When error is heterogeneous across the
   input space - precisely OGAP's cross-scanner / cross-field reality - a single
   marginal threshold can hold on average yet fail badly on the most out-of-
   distribution subgroup (the low-field African scans we most care about). The
   fix is to calibrate a *separate* threshold per group (field strength / scanner
   / site) so each subgroup gets its own valid guarantee.

This module provides: the finite-sample corrected quantile, the Beta coverage
distribution + CI, a :class:`SplitConformalPredictor`, and a
:class:`GroupBalancedConformalPredictor` that exposes the **worst-group**
coverage - the number that the marginal guarantee hides. Pure ``numpy`` (+
optional ``scipy`` for exact Beta intervals).

Convention: lower nonconformity score = more in-distribution / more confident.
A test point is *accepted* (prediction trusted) iff ``score <= threshold``.
"""
from __future__ import annotations

import math
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


def conformal_quantile(cal_scores: Sequence[float], alpha: float = 0.1) -> float:
    """Finite-sample split-conformal threshold at miscoverage ``alpha``.

    Uses the rank ``k = ceil((n+1)(1-alpha))`` so the marginal coverage is
    guaranteed ``>= 1-alpha`` (Vovk; Angelopoulos & Bates). If ``k > n`` the
    calibration set is too small to guarantee the level and ``+inf`` is returned
    (accept everything - the only honest choice).

    Args:
        cal_scores: nonconformity scores on a held-out calibration set (lower =
            more confident).
        alpha: target miscoverage (e.g. ``0.1`` for 90% coverage).
    """
    s = np.sort(np.asarray([v for v in cal_scores if v is not None and np.isfinite(v)], dtype=float))
    n = s.size
    if n == 0:
        return float("inf")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0,1), got {alpha}")
    k = math.ceil((n + 1) * (1.0 - alpha))
    if k > n:
        return float("inf")  # too few calibration points to certify this level
    return float(s[k - 1])   # k-th smallest (1-indexed)


def coverage_beta_params(n_cal: int, alpha: float = 0.1) -> Tuple[int, int]:
    """``(a, b)`` of the per-realisation coverage law ``Cov ~ Beta(a, b)``.

    With ``k = ceil((n+1)(1-alpha))`` the coverage of a single split-conformal
    predictor (conditioned on its calibration draw) is ``Beta(k, n+1-k)``
    (Vovk 2012). Equivalent to Sánchez-Domínguez et al. Eq. 6 up to the
    off-by-one quantile convention.
    """
    n = int(n_cal)
    k = math.ceil((n + 1) * (1.0 - alpha))
    k = min(max(k, 1), n + 1)
    return k, (n + 1 - k)


def coverage_distribution(n_cal: int, alpha: float = 0.1, ci: float = 0.90
                          ) -> Dict[str, float]:
    """Mean and central-``ci`` interval of the realised coverage for a single
    predictor calibrated on ``n_cal`` points - the small-dataset spread that the
    marginal guarantee hides.

    Returns ``{"nominal", "mean", "lo", "hi", "n_cal", "alpha", "certifiable"}``.
    ``mean`` is the marginal coverage ``E[Cov] = k/(n+1) >= 1-alpha``; ``[lo, hi]``
    is the central ``ci`` interval of ``Beta(k, n+1-k)`` (wide for small
    ``n_cal``). When ``n_cal`` is too small to certify the level at this
    convention (``k = ceil((n+1)(1-alpha)) > n``, i.e. ``alpha < 1/(n+1)``), the
    split-conformal threshold is ``+inf`` (accept-all): the realised coverage is
    deterministically 1, the Beta degenerates (``b = 0``), and ``certifiable`` is
    ``False`` - surfacing that this group cannot back a ``1-alpha`` guarantee
    rather than silently emitting a NaN interval (the exact small-dataset failure
    this module exists to flag).
    """
    a, b = coverage_beta_params(n_cal, alpha)
    if b <= 0 or int(n_cal) <= 0:
        # Degenerate Beta(k, 0): threshold is +inf, coverage is trivially 1.0 and
        # the level is NOT certifiable at this calibration size.
        return {"nominal": 1.0 - alpha, "mean": 1.0, "lo": 1.0, "hi": 1.0,
                "n_cal": int(n_cal), "alpha": float(alpha), "certifiable": False}
    mean = a / (a + b)
    tail = (1.0 - ci) / 2.0
    try:
        from scipy.stats import beta as _beta
    except ImportError:
        _beta = None
    if _beta is not None:
        # Exact Beta quantiles. a/b are already validated above, so a computation
        # error should not occur here - and is deliberately NOT swallowed, so it would
        # surface rather than be silently replaced by a Monte-Carlo approximation.
        lo = float(_beta.ppf(tail, a, b))
        hi = float(_beta.ppf(1.0 - tail, a, b))
    else:
        # Monte-Carlo fallback only when scipy is genuinely unavailable.
        rng = np.random.default_rng(0)
        draws = rng.beta(a, b, size=20000)
        lo = float(np.percentile(draws, 100 * tail))
        hi = float(np.percentile(draws, 100 * (1.0 - tail)))
    # Guard residual non-finite ppf output (e.g. extreme params).
    if not (np.isfinite(lo) and np.isfinite(hi)):
        lo, hi = float(mean), float(mean)
    return {"nominal": 1.0 - alpha, "mean": float(mean), "lo": lo, "hi": hi,
            "n_cal": int(n_cal), "alpha": float(alpha), "certifiable": True}


class SplitConformalPredictor:
    """Marginal split-conformal predictor with a small-dataset coverage report."""

    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = float(alpha)
        self.threshold_: float = float("inf")
        self.n_cal_: int = 0

    def fit(self, cal_scores: Sequence[float]) -> "SplitConformalPredictor":
        scores = [v for v in cal_scores if v is not None and np.isfinite(v)]
        self.n_cal_ = len(scores)
        self.threshold_ = conformal_quantile(scores, self.alpha)
        return self

    def accept(self, scores) -> np.ndarray:
        """Boolean mask: True where the prediction is trusted (score <= τ)."""
        arr = np.asarray(scores, dtype=float)
        return arr <= self.threshold_

    def empirical_coverage(self, test_scores: Sequence[float]) -> float:
        arr = np.asarray([v for v in test_scores if v is not None and np.isfinite(v)], dtype=float)
        if arr.size == 0:
            return float("nan")
        return float((arr <= self.threshold_).mean())

    def coverage_report(self, ci: float = 0.90) -> Dict[str, float]:
        """Honest coverage report: nominal level + the Beta spread for this
        calibration size (see :func:`coverage_distribution`)."""
        rep = coverage_distribution(self.n_cal_, self.alpha, ci=ci)
        rep["threshold"] = self.threshold_
        return rep


class GroupBalancedConformalPredictor:
    """Per-group (field strength / scanner / site) split-conformal calibration.

    Fits an independent threshold for each acquisition subgroup so each gets its
    own valid ``1-alpha`` guarantee, and reports the **worst-group** coverage -
    the quantity a single marginal threshold can silently violate on the most
    out-of-distribution low-field subgroup (Sánchez-Domínguez et al., §"group-
    balanced"). Groups with too few calibration points to certify the level fall
    back to ``threshold = +inf`` (accept all) and are flagged.
    """

    def __init__(self, alpha: float = 0.1, min_group: int = 1) -> None:
        self.alpha = float(alpha)
        self.min_group = int(min_group)
        self.per_group_: Dict[str, SplitConformalPredictor] = {}
        self.undercertified_: Dict[str, int] = {}

    def fit(self, cal_scores_by_group: Mapping[str, Sequence[float]]
            ) -> "GroupBalancedConformalPredictor":
        self.per_group_ = {}
        self.undercertified_ = {}
        for g, scores in cal_scores_by_group.items():
            cp = SplitConformalPredictor(self.alpha).fit(scores)
            self.per_group_[g] = cp
            # ceil((n+1)(1-alpha)) > n  =>  cannot certify this level for the group.
            if not math.isfinite(cp.threshold_) or cp.n_cal_ < self.min_group:
                self.undercertified_[g] = cp.n_cal_
        return self

    def threshold(self, group: str) -> float:
        cp = self.per_group_.get(group)
        return cp.threshold_ if cp is not None else float("inf")

    def accept(self, score: float, group: str) -> bool:
        return float(score) <= self.threshold(group)

    def coverage_report(self, test_scores_by_group: Optional[Mapping[str, Sequence[float]]] = None,
                        ci: float = 0.90) -> Dict[str, object]:
        """Per-group coverage report + the worst-group summary.

        If ``test_scores_by_group`` is given, also reports the *empirical*
        coverage each group achieves on held-out test scores (the real check that
        the per-group guarantee holds out of sample).
        """
        per_group: Dict[str, Dict[str, float]] = {}
        worst_g, worst_cov = None, np.inf
        for g, cp in self.per_group_.items():
            rep = cp.coverage_report(ci=ci)
            if test_scores_by_group is not None and g in test_scores_by_group:
                rep["empirical"] = cp.empirical_coverage(test_scores_by_group[g])
            per_group[g] = rep
            # Rank groups by the quantity that actually degrades for the under-served
            # subgroup: the EMPIRICAL out-of-sample coverage when test scores are given,
            # else the LOWER BOUND of the Beta coverage CI (which widens downward for
            # small calibration sets). The marginal mean k/(n+1) is >= 1-alpha for every
            # group by construction, so ranking by it would always flag the LARGEST
            # group rather than the least-reliable one.
            basis = rep.get("empirical")
            if basis is None or not np.isfinite(basis):
                basis = rep.get("lo", rep["mean"])
            if basis < worst_cov:
                worst_cov, worst_g = basis, g
        return {
            "alpha": self.alpha,
            "per_group": per_group,
            "worst_group": worst_g if worst_g is not None else float("nan"),
            "worst_group_coverage": float(worst_cov) if np.isfinite(worst_cov) else float("nan"),
            "worst_group_coverage_basis": ("empirical" if test_scores_by_group is not None
                                           else "beta_ci_lower"),
            "undercertified_groups": dict(self.undercertified_),
        }
