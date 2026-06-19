"""Regression: a blank region-weight spec must not silently disable KD.

Fresh-eyes audit finding. The config loader yields an *empty* namespace/dict for
a YAML key written with nothing under it (`region_weights:`). Previously
``_as_weight_dict`` only substituted the default for ``None``, so an empty spec
produced an empty weight map and ``RegionKDLoss.forward`` returned 0 KD with no
error - a silent failure. The fix mirrors region_aware's ``x or default``: an
empty spec falls back to a non-empty default, while an empty default (which is
legitimate for ``per_region_temperature``) stays empty.
"""
from types import SimpleNamespace

import torch

from ogap.losses.distillation import RegionKDLoss, _as_weight_dict, _DEFAULT_KD_REGION_WEIGHTS


def _kd(weights):
    torch.manual_seed(0)
    loss = RegionKDLoss(region_weights=weights)
    s = torch.randn(2, 4, 8, 8, 8)
    t = torch.randn(2, 4, 8, 8, 8)        # differing logits => KD must be > 0
    return float(loss(s, t))


def test_empty_dict_region_weights_falls_back_to_default():
    assert _as_weight_dict({}, _DEFAULT_KD_REGION_WEIGHTS) == _DEFAULT_KD_REGION_WEIGHTS
    # and the loss is genuinely active, not a silent zero
    assert _kd({}) > 0.0


def test_empty_namespace_region_weights_falls_back_to_default():
    assert _as_weight_dict(SimpleNamespace(), _DEFAULT_KD_REGION_WEIGHTS) == _DEFAULT_KD_REGION_WEIGHTS
    assert _kd(SimpleNamespace()) > 0.0


def test_none_region_weights_uses_default():
    assert _as_weight_dict(None, _DEFAULT_KD_REGION_WEIGHTS) == _DEFAULT_KD_REGION_WEIGHTS
    assert _kd(None) > 0.0


def test_explicit_weights_are_respected():
    custom = {"WT": 2.0, "ET": 5.0}
    assert _as_weight_dict(custom, _DEFAULT_KD_REGION_WEIGHTS) == custom


def test_empty_default_stays_empty_for_per_region_temperature():
    # per_region_temperature legitimately means "no per-region override" -> {}.
    assert _as_weight_dict({}, {}) == {}
    assert _as_weight_dict(None, {}) == {}
    # a populated per-region temperature still passes through
    assert _as_weight_dict({"ET": 2.0}, {}) == {"ET": 2.0}


def test_per_region_temperature_empty_does_not_crash_loss():
    torch.manual_seed(0)
    loss = RegionKDLoss(temperature=3.0, per_region_temperature={},
                        use_per_region_temperature=True)
    out = float(loss(torch.randn(2, 4, 8, 8, 8), torch.randn(2, 4, 8, 8, 8)))
    assert out > 0.0  # falls back to the global temperature per region
