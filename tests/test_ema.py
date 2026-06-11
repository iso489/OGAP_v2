"""Weight-EMA utility tests (CPU)."""
from types import SimpleNamespace

import torch
import torch.nn as nn

from ogap.utils.ema import ModelEMA, build_model_ema


def _model():
    return nn.Sequential(nn.Conv3d(2, 3, 3, padding=1), nn.GroupNorm(1, 3), nn.Conv3d(3, 2, 1))


def test_ema_tracks_and_keys_match():
    m = _model()
    ema = ModelEMA(m, decay=0.9)
    # state_dict keys are identical to the source model (checkpoint-compatible).
    assert set(ema.state_dict().keys()) == set(m.state_dict().keys())
    # After a weight change + update, EMA moves toward the new weights but lags.
    with torch.no_grad():
        for p in m.parameters():
            p.add_(1.0)
    before = ema.state_dict()["0.weight"].clone()
    ema.update(m)
    after = ema.state_dict()["0.weight"]
    assert not torch.equal(before, after)             # moved
    assert not torch.allclose(after, m.state_dict()["0.weight"])  # but lags (decay)


def test_ema_decay_one_is_frozen():
    m = _model()
    ema = ModelEMA(m, decay=1.0)
    snap = ema.state_dict()["0.weight"].clone()
    with torch.no_grad():
        for p in m.parameters():
            p.add_(5.0)
    ema.update(m)
    assert torch.equal(ema.state_dict()["0.weight"], snap)  # decay=1 -> never updates


def test_build_model_ema_gated_by_config():
    m = _model()
    off = SimpleNamespace(training=SimpleNamespace(ema=SimpleNamespace(enabled=False)))
    assert build_model_ema(m, off) is None
    on = SimpleNamespace(training=SimpleNamespace(ema=SimpleNamespace(enabled=True, decay=0.99)))
    e = build_model_ema(m, on)
    assert e is not None and abs(e.decay - 0.99) < 1e-9
    assert build_model_ema(m, None) is None  # no config -> no EMA
