import numpy as np

from ogap.evaluation.stats import (
    paired_significance_table, holm_correction, benjamini_hochberg,
    rank_biserial, bootstrap_median_diff_ci,
)


def test_identical_not_significant():
    rng = np.random.default_rng(0)
    a = {f"c{i}": {"dsc_wt": float(rng.uniform(0.7, 0.9))} for i in range(30)}
    b = {k: dict(v) for k, v in a.items()}
    rows = paired_significance_table(a, b, metrics=("dsc_wt",), correction="none")
    assert rows[0]["p_value_raw"] >= 0.05 and not rows[0]["significant"]


def test_clear_difference_significant():
    rng = np.random.default_rng(1)
    a, b = {}, {}
    for i in range(40):
        base = float(rng.uniform(0.5, 0.7))
        a[f"c{i}"] = {"dsc_wt": base + 0.15}
        b[f"c{i}"] = {"dsc_wt": base}
    r = paired_significance_table(a, b, metrics=("dsc_wt",), correction="holm")[0]
    assert r["p_value_corrected"] < 0.001 and r["significant"] and r["stars"] == "***"
    assert r["effect_size_r"] > 0.9 and r["direction"] == "A"


def test_hd95_lower_is_better_direction():
    a = {f"c{i}": {"hd95_wt": 2.0} for i in range(20)}
    b = {f"c{i}": {"hd95_wt": 5.0} for i in range(20)}
    rows = paired_significance_table(a, b, metrics=("hd95_wt",), correction="none")
    assert rows[0]["direction"] == "A"


def test_nan_cases_dropped_and_counted():
    a = {f"c{i}": {"hd95_et": (np.nan if i < 5 else 3.0)} for i in range(15)}
    b = {f"c{i}": {"hd95_et": 4.0} for i in range(15)}
    r = paired_significance_table(a, b, metrics=("hd95_et",), correction="none")[0]
    assert r["n_dropped"] == 5 and r["n_pairs"] == 10


def test_holm_and_bh_monotone():
    p = [0.001, 0.02, 0.03, 0.5]
    h, bh = holm_correction(p), benjamini_hochberg(p)
    assert all(0 <= x <= 1 for x in h + bh)
    assert h[0] >= p[0] and bh[0] >= p[0]


def test_rank_biserial_bounds():
    assert rank_biserial(np.array([1.0, 2.0, 3.0])) == 1.0
    assert rank_biserial(np.array([-1.0, -2.0, -3.0])) == -1.0
    assert rank_biserial(np.zeros(5)) == 0.0


def test_bca_ci_brackets_point():
    rng = np.random.default_rng(3)
    point, lo, hi = bootstrap_median_diff_ci(rng.normal(0.1, 0.05, size=50), n_boot=2000, seed=0)
    assert lo <= point <= hi


def test_multiple_metrics_family_correction():
    rng = np.random.default_rng(4)
    a, b = {}, {}
    for i in range(30):
        a[f"c{i}"] = {"dsc_wt": 0.8, "dsc_tc": 0.7, "dsc_et": float(rng.uniform(0.5, 0.6))}
        b[f"c{i}"] = {"dsc_wt": 0.79, "dsc_tc": 0.69, "dsc_et": float(rng.uniform(0.5, 0.6))}
    rows = paired_significance_table(a, b, metrics=("dsc_wt", "dsc_tc", "dsc_et"))
    assert len(rows) == 3
    for r in rows:
        assert "p_value_corrected" in r and "effect_size_r" in r and "ci_low" in r
