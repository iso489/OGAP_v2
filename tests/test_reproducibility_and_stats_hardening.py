"""Tests for the 2026-06-15 reproducibility + stats-honesty hardening pass.

  * bootstrap_median_diff_ci reports the interval method it ACTUALLY used
    (bca / bias_corrected / percentile / undefined), not a mislabelled "BCa"
  * paired_significance_table reports n_zero_diff / n_effective (the ties the
    Wilcoxon test drops) + ci_method per row
  * modality-robustness std is NaN for n<2 (not a misleading 0.0)
  * risk_disparity_ci gives a stratified-bootstrap CI on Δ_k (not just a point)
  * run provenance records git SHA / file hash / torch+numpy versions
"""
import numpy as np

import ogap.legacy as L
from ogap.evaluation.equity import risk_disparity, risk_disparity_ci
from ogap.evaluation.stats import bootstrap_median_diff_ci, paired_significance_table


def test_bootstrap_ci_reports_actual_method():
    rng = np.random.default_rng(0)
    diff = rng.normal(0.1, 0.05, 40)
    p, lo, hi, method = bootstrap_median_diff_ci(diff, n_boot=2000, return_method=True)
    assert method in ("bca", "bias_corrected", "percentile")
    assert lo <= p <= hi
    # degenerate n<2 -> undefined, NaN interval
    p2, lo2, hi2, m2 = bootstrap_median_diff_ci(np.array([0.3]), return_method=True)
    assert m2 == "undefined" and np.isnan(lo2) and np.isnan(hi2)
    # backward-compatible 3-tuple when return_method is not requested
    assert len(bootstrap_median_diff_ci(diff, n_boot=500)) == 3


def test_paired_table_reports_zero_ties_and_ci_method():
    # method A consistently better on even cases; exact ties on odd cases
    A = {f"c{i}": {"dsc_wt": (0.9 if i % 2 else 0.8)} for i in range(20)}
    B = {f"c{i}": {"dsc_wt": (0.9 if i % 2 else 0.7)} for i in range(20)}
    rows = paired_significance_table(A, B, metrics=["dsc_wt"], n_boot=1000)
    r = rows[0]
    assert r["n_pairs"] == 20
    assert r["n_zero_diff"] == 10 and r["n_effective"] == 10   # 10 ties the test drops
    assert r["ci_method"] in ("bca", "bias_corrected", "percentile", "undefined")


def test_risk_disparity_ci_brackets_point_and_signs():
    g = {"3T":  [0.90, 0.92, 0.88, 0.91, 0.90],
         "64mT": [0.70, 0.72, 0.68, 0.71, 0.69, 0.70]}
    point = risk_disparity(g, "dsc_et")
    ci = risk_disparity_ci(g, "dsc_et", reps=1000, seed=0)
    for grp in ("3T", "64mT"):
        d, lo, hi = ci[grp]
        assert abs(d - point[grp]) < 1e-9     # point matches risk_disparity
        assert lo <= d <= hi                   # CI brackets the point
    # the low-field group is served worse -> positive Δ; 3T better -> negative Δ
    assert ci["64mT"][0] > 0 > ci["3T"][0]


def test_code_provenance_has_expected_keys():
    prov = L._code_provenance()
    for k in ("git_sha", "legacy_sha256", "torch_version", "numpy_version"):
        assert k in prov and isinstance(prov[k], str)
