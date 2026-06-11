"""Comparison-baseline registry: names, descriptions, and not-yet-runnable stubs.

The baselines need cluster GPU + dataset + external method code, so on the dev
box every runner raises NotImplementedError. These tests pin the registry shape
and the failure contract, not any training behaviour.
"""
import pytest

from ogap.evaluation.baselines import (
    BASELINES,
    baseline_description,
    list_baselines,
    run_baseline,
)


def test_registry_is_non_empty_and_distinct():
    names = list_baselines()
    assert len(names) >= 3
    assert len(names) == len(set(names))
    assert names == sorted(names)              # deterministic ordering
    assert set(names) == set(BASELINES)


def test_known_baselines_present():
    names = set(list_baselines())
    assert "dense_teacher" in names
    assert "vanilla_kd" in names
    assert "student_no_distillation" in names


def test_baseline_description_returns_text_for_known():
    for name in list_baselines():
        desc = baseline_description(name)
        assert isinstance(desc, str) and desc.strip()


def test_baseline_description_unknown_raises_keyerror():
    with pytest.raises(KeyError):
        baseline_description("does_not_exist")


def test_run_baseline_known_raises_not_implemented():
    for name in list_baselines():
        with pytest.raises(NotImplementedError):
            run_baseline(name)


def test_run_baseline_message_names_the_method():
    with pytest.raises(NotImplementedError) as excinfo:
        run_baseline("dense_teacher")
    assert "dense_teacher" in str(excinfo.value)


def test_run_baseline_unknown_raises_keyerror_before_notimplemented():
    # Validation must reject an unknown name rather than reporting it unimplemented.
    with pytest.raises(KeyError):
        run_baseline("not_a_baseline")
