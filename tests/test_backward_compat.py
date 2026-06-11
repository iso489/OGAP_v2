import os
import tempfile

import torch
import yaml

from ogap.config import load_config
from ogap.config.loader import load_config as lc
from ogap.losses import build_task_loss
from ogap.losses.distillation import build_kd_loss


def test_defaults_load_and_legacy_type():
    cfg = load_config()
    assert cfg.loss.task.type == "legacy"
    assert cfg.data.in_channels == 4
    assert cfg.distillation.enabled is True


def test_tissue_priors_set_7_channels():
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump({"data": {"use_tissue_priors": True}}, f)
        p = f.name
    try:
        assert lc(p).data.in_channels == 7
    finally:
        os.unlink(p)


def test_unknown_key_raises():
    import pytest
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump({"loss": {"task": {"not_a_real_key": 1}}}, f)
        p = f.name
    try:
        with pytest.raises(ValueError):
            load_config(p)
    finally:
        os.unlink(p)


def test_legacy_loss_via_config_matches_direct():
    from ogap.legacy import DiceCELoss
    cfg = load_config()
    torch.manual_seed(0)
    logits = torch.randn(1, 4, 16, 16, 16)
    target = torch.randint(0, 4, (1, 16, 16, 16))
    assert torch.allclose(build_task_loss(cfg)(logits, target), DiceCELoss(4)(logits, target), atol=1e-5)


def test_kd_no_teacher_equals_task_under_defaults():
    cfg = load_config()
    loss = build_kd_loss(cfg)
    s = torch.randn(1, 4, 8, 8, 8)
    tgt = torch.randint(0, 4, (1, 8, 8, 8))
    out = loss(s, tgt, teacher_logits=None)
    assert torch.allclose(out["loss_total"], out["loss_task"], atol=1e-6)


def test_set_all_seeds_runs():
    from ogap.utils import set_all_seeds
    set_all_seeds(123)
    a = torch.randn(3)
    set_all_seeds(123)
    b = torch.randn(3)
    assert torch.allclose(a, b)
