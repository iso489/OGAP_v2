"""§9.8.9 — modality-robustness suite + multi-model significance tests.

Verifies the 11-combination enumeration, channel masking (zero/mean fill), the
Friedman omnibus, the deterministic multi-model significance matrix row layout,
the full sweep, degradation analysis, the evaluation integration over a dummy
model/loader, and the CPU INT8 quantize helper.
"""
import numpy as np
import pytest
import torch
import torch.nn as nn

from ogap.evaluation.modality_robustness import (
    ALL_COMBINATIONS,
    apply_modality_mask,
    combo_label,
    evaluate_modality_combinations,
    friedman_test,
    full_significance_sweep,
    modality_degradation_analysis,
    modality_robustness_table,
    multimodel_significance_matrix,
    quantize_student_int8,
)

METRICS = ("dsc_wt", "dsc_tc", "dsc_et", "hd95_wt", "hd95_tc", "hd95_et")


# ---------------------------------------------------------------- combinations
def test_all_combinations_count_and_distinct():
    assert len(ALL_COMBINATIONS) == 11
    assert all(isinstance(c, frozenset) for c in ALL_COMBINATIONS)
    assert len(set(ALL_COMBINATIONS)) == 11                       # distinct subsets
    labels = [combo_label(c) for c in ALL_COMBINATIONS]
    assert len(set(labels)) == 11                                 # distinct labels
    # C(4,2)=6 pairs, C(4,3)=4 triples, C(4,4)=1 full
    assert sorted(len(c) for c in ALL_COMBINATIONS) == [2] * 6 + [3] * 4 + [4]


def test_combo_label_sorted_by_channel_index():
    assert combo_label(frozenset({0, 1, 2, 3})) == "T1+T1ce+T2+FLAIR"
    assert combo_label(frozenset({3, 0})) == "T1+FLAIR"


# ----------------------------------------------------------------------- masks
def _channel_constant_volume():
    """(B=1, C=4, 2, 2, 2) with channel c filled with the constant (c + 1)."""
    x = torch.zeros(1, 4, 2, 2, 2)
    for c in range(4):
        x[:, c] = float(c + 1)
    return x


def test_apply_modality_mask_zero_fill():
    x = _channel_constant_volume()
    out = apply_modality_mask(x, frozenset({0, 1}), fill="zero")
    assert torch.all(out[:, 0] == 1.0) and torch.all(out[:, 1] == 2.0)
    assert torch.all(out[:, 2] == 0.0) and torch.all(out[:, 3] == 0.0)
    assert torch.all(x[:, 2] == 3.0)                              # input not mutated


def test_apply_modality_mask_mean_fill():
    x = _channel_constant_volume()                                # channels 1,2,3,4
    out = apply_modality_mask(x, frozenset({0, 1}), fill="mean")
    assert torch.all(out[:, 0] == 1.0) and torch.all(out[:, 1] == 2.0)
    expected = torch.full_like(out[:, 2], 1.5)                    # (1 + 2) / 2
    assert torch.allclose(out[:, 2], expected)
    assert torch.allclose(out[:, 3], expected)


def test_apply_modality_mask_full_combo_is_unmodified_copy():
    x = _channel_constant_volume()
    out = apply_modality_mask(x, frozenset({0, 1, 2, 3}))
    assert out is not x
    assert torch.equal(out, x)


def test_apply_modality_mask_invalid_fill_raises():
    with pytest.raises(ValueError):
        apply_modality_mask(_channel_constant_volume(), frozenset({0}), fill="bogus")


# -------------------------------------------------------------------- Friedman
def test_friedman_identical_models_not_significant():
    vals = [0.70, 0.82, 0.61, 0.88, 0.75, 0.69]
    res = friedman_test({"A": list(vals), "B": list(vals), "C": list(vals)})
    assert not res["significant_at_0.05"]


def test_friedman_one_model_consistently_higher_is_significant():
    rng = np.random.default_rng(0)
    base = rng.uniform(0.5, 0.7, size=20)
    res = friedman_test({
        "A": base.tolist(),
        "B": (base + 0.01).tolist(),
        "C": (base + 0.20).tolist(),
    })
    assert res["p_value"] < 0.001 and res["significant_at_0.05"]


def test_friedman_requires_three_models():
    with pytest.raises(ValueError):
        friedman_test({"A": [1.0, 2.0, 3.0], "B": [1.0, 2.0, 3.0]})


def test_friedman_unequal_lengths_raises():
    with pytest.raises(ValueError):
        friedman_test({"A": [1.0, 2.0, 3.0], "B": [1.0, 2.0], "C": [1.0, 2.0, 3.0]})


# ------------------------------------------------------- multi-model machinery
def _synth_results(models, combos, n=5, seed=0):
    """{model: {combo_label: {case_id: {metric: value}}}} of finite synthetic data."""
    rng = np.random.default_rng(seed)
    out = {}
    for m in models:
        out[m] = {}
        for combo in combos:
            lbl = combo_label(combo)
            out[m][lbl] = {
                f"case_{i:04d}": {k: float(rng.uniform(0.4, 0.9)) for k in METRICS}
                for i in range(n)
            }
    return out


def test_multimodel_significance_matrix_row_layout():
    combo = frozenset({0, 1, 2, 3})
    res = _synth_results(["teacher", "student", "student_int8"], [combo], n=6)
    df = multimodel_significance_matrix(res, combo, correction="holm")
    # 6 metrics x (1 Friedman + C(3,2)=3 pairwise) = 24 rows
    assert len(df) == 24
    assert (df["pair"] == "FRIEDMAN").sum() == 6
    assert (df["pair"] != "FRIEDMAN").sum() == 18
    for col in ("combo_label", "metric", "pair", "friedman_p",
                "p_value_corrected", "significant", "stars", "effect_size_r"):
        assert col in df.columns
    assert set(df["combo_label"]) == {combo_label(combo)}


def test_full_significance_sweep_row_count():
    res = _synth_results(["a", "b", "c"], ALL_COMBINATIONS, n=4)
    df = full_significance_sweep(res, correction="holm")
    assert len(df) == 11 * 24                                     # 264


def test_modality_robustness_table_shape_and_sort():
    res = _synth_results(["teacher", "student"], ALL_COMBINATIONS, n=3)
    df = modality_robustness_table(res, metrics=("dsc_et",))
    assert len(df) == 11
    assert "n_modalities" in df.columns
    assert ("dsc_et_mean", "teacher") in df.columns
    assert ("dsc_et_std", "student") in df.columns
    assert list(df["n_modalities"]) == sorted(df["n_modalities"])  # ascending by n


def test_degradation_delta_zero_at_full_modality():
    res = _synth_results(["teacher", "student"], ALL_COMBINATIONS, n=4, seed=1)
    df = modality_degradation_analysis(res, metric="dsc_et")
    full = df[df["n_modalities"] == 4]
    assert not full.empty
    assert (full["delta_from_4mod"] == 0.0).all()
    assert (full["pct_degradation"] == 0.0).all()


# ----------------------------------------------------------------- integration
class _TinyModel(nn.Module):
    def __init__(self, n_classes=4):
        super().__init__()
        self.conv = nn.Conv3d(4, n_classes, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


def _dummy_loader(n_batches=2, batch=2, d=4):
    for _ in range(n_batches):
        vol = torch.randn(batch, 4, d, d, d)
        lbl = torch.randint(0, 4, (batch, d, d, d))
        yield vol, lbl


def test_evaluate_modality_combinations_structure():
    torch.manual_seed(0)
    results = evaluate_modality_combinations(
        _TinyModel(), _dummy_loader(n_batches=2, batch=2), device="cpu", fill="zero")
    assert set(results.keys()) == {combo_label(c) for c in ALL_COMBINATIONS}
    for cases in results.values():
        assert len(cases) == 4                                    # 2 batches x 2
        for metrics in cases.values():
            assert "dsc_wt" in metrics and "hd95_et" in metrics


# ------------------------------------------------------------------- quantize
def test_quantize_student_int8_quantizes_or_reports_unavailable():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8, 4)

        def forward(self, x):
            return self.fc(x)

    m = M().eval()
    real_engines = [e for e in torch.backends.quantized.supported_engines if e != "none"]
    if not real_engines:
        # No quantized backend compiled in (e.g. CPU dev box): documented RuntimeError.
        with pytest.raises(RuntimeError):
            quantize_student_int8(m)
        return
    q = quantize_student_int8(m)                                  # capable host (e.g. fbgemm)
    assert isinstance(q, nn.Module) and q is not m
    assert isinstance(m.fc, nn.Linear)                            # original untouched
    assert "quantized" in type(q.fc).__module__.lower()           # target layer quantized
