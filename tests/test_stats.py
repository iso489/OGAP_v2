import numpy as np
import pytest

from ogap.evaluation.stats import (
    paired_significance_table, holm_correction, benjamini_hochberg,
    rank_biserial, bootstrap_median_diff_ci, noninferiority_table,
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


# ------------------------------------------------------- non-inferiority (INT8)
def test_noninferiority_identical_is_noninferior():
    # FP32 reference == INT8 candidate -> zero loss -> CI entirely below margin.
    a = {f"c{i}": {"dsc_et": 0.80} for i in range(30)}
    b = {k: dict(v) for k, v in a.items()}
    r = noninferiority_table(a, b, margins=0.01, metrics=("dsc_et",))[0]
    assert r["non_inferior"] is True
    assert r["ci_high"] < 0.01
    assert r["margin"] == 0.01


def test_noninferiority_clearly_worse_candidate_fails():
    # Candidate (INT8) is 0.15 Dice worse than reference -> not non-inferior at 0.01.
    a = {f"c{i}": {"dsc_et": 0.85} for i in range(30)}
    b = {f"c{i}": {"dsc_et": 0.70} for i in range(30)}
    r = noninferiority_table(a, b, margins=0.01, metrics=("dsc_et",))[0]
    assert r["non_inferior"] is False
    assert r["median_loss"] > 0.1          # positive loss == candidate worse


def test_noninferiority_small_loss_within_margin():
    # Candidate 0.004 Dice worse everywhere -> well inside a 0.01 margin.
    a, b = {}, {}
    for i in range(40):
        base = 0.70 + 0.005 * i
        a[f"c{i}"] = {"dsc_et": base}
        b[f"c{i}"] = {"dsc_et": base - 0.004}
    r = noninferiority_table(a, b, margins=0.01, metrics=("dsc_et",))[0]
    assert r["non_inferior"] is True
    assert 0.0 < r["median_loss"] < 0.01


def test_noninferiority_hd95_lower_is_better_direction():
    # Lower-is-better: loss = candidate - reference. +0.4 mm is within a 1.0 mm margin.
    a = {f"c{i}": {"hd95_et": 4.0} for i in range(25)}
    b = {f"c{i}": {"hd95_et": 4.4} for i in range(25)}
    r = noninferiority_table(a, b, margins=1.0, metrics=("hd95_et",))[0]
    assert r["median_loss"] > 0
    assert r["non_inferior"] is True
    worse = {f"c{i}": {"hd95_et": 7.0} for i in range(25)}
    r2 = noninferiority_table(a, worse, margins=1.0, metrics=("hd95_et",))[0]
    assert r2["non_inferior"] is False


def test_noninferiority_per_metric_margins():
    a, b = {}, {}
    for i in range(30):
        a[f"c{i}"] = {"dsc_et": 0.80, "hd95_et": 4.0}
        b[f"c{i}"] = {"dsc_et": 0.795, "hd95_et": 4.3}
    rows = noninferiority_table(a, b, margins={"dsc_et": 0.01, "hd95_et": 1.0},
                                metrics=("dsc_et", "hd95_et"))
    assert {r["metric"] for r in rows} == {"dsc_et", "hd95_et"}
    assert all(r["non_inferior"] for r in rows)


def test_noninferiority_nonpositive_margin_raises():
    a = {f"c{i}": {"dsc_et": 0.8} for i in range(5)}
    b = {f"c{i}": {"dsc_et": 0.8} for i in range(5)}
    with pytest.raises(ValueError):
        noninferiority_table(a, b, margins=0.0, metrics=("dsc_et",))


def test_noninferiority_table_exported_from_package():
    # API parity with paired_significance_table: callable off the package root.
    import ogap.evaluation as ev
    assert hasattr(ev, "noninferiority_table")
    assert "noninferiority_table" in ev.__all__
