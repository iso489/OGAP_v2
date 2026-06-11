"""Ablation-study helpers: component-state introspection, config loading, and
the variant-vs-baseline results table.

Exercises the 8 experiments/ablation_configs overlays (full + 7 single-component
ablations). No GPU/data needed — these test the bookkeeping, not training.
"""
import os

import pytest

from ogap.evaluation.ablation import (
    COMPONENTS,
    ablation_matrix,
    ablation_results_table,
    component_states,
    load_ablation_configs,
)

CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "experiments", "ablation_configs",
)


def test_components_are_seven_distinct_names():
    assert isinstance(COMPONENTS, tuple)
    assert len(COMPONENTS) == 7 and len(set(COMPONENTS)) == 7
    assert "region_aware_loss" in COMPONENTS
    assert "distillation" in COMPONENTS
    assert "tissue_priors" in COMPONENTS


def test_load_ablation_configs_count_and_order():
    configs = load_ablation_configs(CONFIG_DIR)
    assert len(configs) == 8
    names = [name for name, _ in configs]
    assert names == sorted(names)              # deterministic, filename-sorted
    assert names[0] == "0_full"


def test_full_variant_has_all_components_on():
    configs = dict(load_ablation_configs(CONFIG_DIR))
    states = component_states(configs["0_full"])
    assert set(states) == set(COMPONENTS)
    assert all(states.values())                # everything enabled in the full run


def test_each_ablation_disables_exactly_one_named_component():
    configs = dict(load_ablation_configs(CONFIG_DIR))
    for name, cfg in configs.items():
        if name == "0_full":
            continue
        target = name.split("_no_", 1)[1]       # e.g. '2_no_distillation' -> 'distillation'
        assert target in COMPONENTS
        states = component_states(cfg)
        off = {c for c, on in states.items() if not on}
        assert off == {target}                  # exactly the one named component is off


def test_component_states_accepts_namespace():
    from ogap.config.loader import load_config
    ns_cfg = load_config(os.path.join(CONFIG_DIR, "0_full.yaml"), as_namespace=True)
    assert all(component_states(ns_cfg).values())


def test_ablation_matrix_shape_and_single_toggles():
    df = ablation_matrix(CONFIG_DIR)
    assert df.shape == (8, 7)                   # 8 variants x 7 components
    assert list(df.columns) == list(COMPONENTS)
    assert df.loc["0_full"].all()               # full row all True
    # each ablation flips exactly one off => 7 False cells total across the matrix
    assert int((~df.astype(bool)).to_numpy().sum()) == 7


def test_ablation_results_table_computes_delta_vs_baseline():
    results = {
        "full": {"dsc_wt": 0.90, "dsc_et": 0.80},
        "no_distillation": {"dsc_wt": 0.85, "dsc_et": 0.70},
        "no_physics": {"dsc_wt": 0.88, "dsc_et": 0.82},
    }
    rows = ablation_results_table(results, baseline="full", metrics=("dsc_wt", "dsc_et"))
    by_variant = {r["variant"]: r for r in rows}
    assert set(by_variant) == set(results)
    assert by_variant["full"]["dsc_wt_delta"] == 0.0
    assert by_variant["full"]["dsc_et_delta"] == 0.0
    assert by_variant["no_distillation"]["dsc_wt_delta"] == pytest.approx(-0.05)
    assert by_variant["no_distillation"]["dsc_et_delta"] == pytest.approx(-0.10)
    assert by_variant["no_physics"]["dsc_et_delta"] == pytest.approx(0.02)


def test_ablation_results_table_unknown_baseline_raises():
    with pytest.raises(ValueError):
        ablation_results_table({"a": {"dsc_wt": 0.5}}, baseline="missing",
                               metrics=("dsc_wt",))
