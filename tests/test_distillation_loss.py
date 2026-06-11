import torch
from types import SimpleNamespace

from ogap.losses.distillation import build_kd_loss, RegionKDLoss


def _cfg(**dist):
    d = dict(enabled=True, alpha=1.0, beta=0.5, gamma=0.1, eta=0.05, temperature=3.0)
    d.update(dist)
    return SimpleNamespace(
        loss=SimpleNamespace(task=SimpleNamespace(type="region_weighted_gdl", num_classes=4)),
        distillation=SimpleNamespace(**d),
    )


def test_dict_keys_and_backward():
    loss = build_kd_loss(_cfg())
    s = torch.randn(2, 4, 8, 8, 8, requires_grad=True)
    t = torch.randn(2, 4, 8, 8, 8)
    tgt = torch.randint(0, 4, (2, 8, 8, 8))
    out = loss(s, tgt, teacher_logits=t)
    assert set(out) == {"loss_total", "loss_task", "loss_kd", "loss_feat", "loss_attn"}
    out["loss_total"].backward()
    assert s.grad is not None and torch.isfinite(s.grad).all()


def test_beta_zero_equals_task_only():
    s = torch.randn(1, 4, 8, 8, 8)
    t = torch.randn(1, 4, 8, 8, 8)
    tgt = torch.randint(0, 4, (1, 8, 8, 8))
    out = build_kd_loss(_cfg(beta=0.0))(s, tgt, teacher_logits=t)
    assert torch.allclose(out["loss_total"], out["loss_task"], atol=1e-6)
    assert float(out["loss_kd"]) == 0.0


def test_kl_nonneg_and_zero_when_equal():
    torch.manual_seed(0)
    kd = RegionKDLoss()
    x = torch.randn(2, 4, 8, 8, 8)
    same = kd(x, x.clone())
    # KL(p||p) == 0 up to float32 reduction error over the volume.
    assert abs(same.item()) < 1e-4
    s, t = torch.randn(2, 4, 8, 8, 8), torch.randn(2, 4, 8, 8, 8)
    assert kd(s, t).item() >= -1e-4


def test_temperature_changes_value():
    s, t = torch.randn(1, 4, 8, 8, 8), torch.randn(1, 4, 8, 8, 8)
    a = RegionKDLoss(temperature=1.0)(s, t)
    b = RegionKDLoss(temperature=5.0)(s, t)
    assert not torch.allclose(a, b)


def test_feature_distillation_term():
    loss = build_kd_loss(_cfg(feature_distillation=True))
    s = torch.randn(1, 4, 8, 8, 8, requires_grad=True)
    t = torch.randn(1, 4, 8, 8, 8)
    tgt = torch.randint(0, 4, (1, 8, 8, 8))
    sf = [torch.randn(1, 16, 4, 4, 4, requires_grad=True)]
    tf = [torch.randn(1, 16, 4, 4, 4)]
    out = loss(s, tgt, teacher_logits=t, student_features=sf, teacher_features=tf)
    assert float(out["loss_feat"]) > 0.0


def test_feature_distillation_accepts_feature_dicts():
    loss = build_kd_loss(_cfg(feature_distillation=True, attention_transfer=True))
    s = torch.randn(1, 4, 8, 8, 8, requires_grad=True)
    t = torch.randn(1, 4, 8, 8, 8)
    tgt = torch.randint(0, 4, (1, 8, 8, 8))
    sf = {"bottleneck": torch.randn(1, 16, 4, 4, 4, requires_grad=True)}
    tf = {"bottleneck": torch.randn(1, 16, 4, 4, 4)}
    out = loss(s, tgt, teacher_logits=t, student_features=sf, teacher_features=tf)
    assert float(out["loss_feat"]) > 0.0
    assert float(out["loss_attn"]) >= 0.0
