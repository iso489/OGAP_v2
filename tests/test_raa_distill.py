"""Tests for the Region-Aware Attention (RAA) heterogeneous-KD ablation arm (PAT).

CPU-only. Verifies (a) RAA produces a finite scalar across heterogeneous teacher/
student channels AND spatial sizes, (b) gradients flow to the student features and
the RAA params but NOT the (detached) teacher, (c) the lazy per-stage projections
materialise on first forward, (d) disjoint stages -> 0 and bad dim/heads -> error,
(e) KDLoss is byte-identical (same 5 keys) when RAA is OFF (default) and gains a
`loss_raa` term only when enabled.
"""
from types import SimpleNamespace

import pytest
import torch

from ogap.losses.distillation import (
    RegionAwareAttentionDistill,
    build_kd_loss,
)


# --------------------------------------------------------------------------
# RAA module
# --------------------------------------------------------------------------
def _hetero_feats():
    # student and teacher differ in BOTH channel count and spatial size per stage.
    sf = {"bottleneck": torch.randn(2, 32, 8, 8, 8, requires_grad=True),
          "dec3": torch.randn(2, 16, 16, 16, 16, requires_grad=True)}
    tf = {"bottleneck": torch.randn(2, 64, 6, 6, 6, requires_grad=True),
          "dec3": torch.randn(2, 24, 12, 12, 12, requires_grad=True)}
    return sf, tf


def test_raa_forward_scalar_finite_heterogeneous():
    raa = RegionAwareAttentionDistill(dim=32, token_grid=2, heads=4)
    sf, tf = _hetero_feats()
    loss = raa(sf, tf)
    assert loss.ndim == 0 and torch.isfinite(loss) and float(loss.detach()) >= 0.0


def test_raa_grad_to_student_and_params_not_teacher():
    raa = RegionAwareAttentionDistill(dim=16, token_grid=2, heads=2)
    sf = {"b": torch.randn(2, 8, 4, 4, 4, requires_grad=True)}
    tf = {"b": torch.randn(2, 12, 4, 4, 4, requires_grad=True)}
    raa(sf, tf).backward()
    assert sf["b"].grad is not None and torch.isfinite(sf["b"].grad).all()
    assert tf["b"].grad is None  # teacher tokens are detached targets
    assert any(p.grad is not None for p in raa.parameters())


def test_raa_params_materialise_on_first_forward():
    raa = RegionAwareAttentionDistill(dim=16, token_grid=2, heads=2)
    n_before = sum(p.numel() for p in raa.parameters())   # attn only
    assert len(raa.in_proj) == 0 and len(raa.out_proj) == 0
    sf = {"b": torch.randn(1, 8, 4, 4, 4)}
    tf = {"b": torch.randn(1, 12, 4, 4, 4)}
    with torch.no_grad():
        raa(sf, tf)
    assert sum(p.numel() for p in raa.parameters()) > n_before
    assert len(raa.in_proj) == 1 and len(raa.out_proj) == 1


def test_raa_fp32_params_even_under_bf16_features():
    raa = RegionAwareAttentionDistill(dim=16, token_grid=2, heads=2)
    sf = {"b": torch.randn(1, 8, 4, 4, 4, dtype=torch.bfloat16)}
    tf = {"b": torch.randn(1, 12, 4, 4, 4, dtype=torch.bfloat16)}
    loss = raa(sf, tf)
    assert torch.isfinite(loss)
    assert raa.in_proj["b"].weight.dtype == torch.float32   # master params stay fp32


def test_raa_no_shared_stages_returns_zero():
    raa = RegionAwareAttentionDistill(dim=16, token_grid=2, heads=2)
    sf = {"only_student": torch.randn(1, 8, 4, 4, 4)}
    tf = {"only_teacher": torch.randn(1, 8, 4, 4, 4)}
    assert float(raa(sf, tf)) == 0.0


def test_raa_validation_errors():
    with pytest.raises(ValueError):
        RegionAwareAttentionDistill(dim=30, heads=4)        # 30 % 4 != 0
    with pytest.raises(ValueError):
        RegionAwareAttentionDistill(dim=16, token_grid=0)
    with pytest.raises(TypeError):
        RegionAwareAttentionDistill(dim=16)([torch.randn(1, 8, 4, 4, 4)], [torch.randn(1, 8, 4, 4, 4)])


# --------------------------------------------------------------------------
# KDLoss integration
# --------------------------------------------------------------------------
def _cfg(raa=None):
    dist = dict(enabled=True, alpha=1.0, beta=0.5, gamma=0.1, eta=0.05, temperature=3.0,
                feature_distillation=False, attention_transfer=False)
    if raa is not None:
        dist["raa"] = SimpleNamespace(**raa)
    return SimpleNamespace(
        loss=SimpleNamespace(task=SimpleNamespace(type="region_weighted_gdl", num_classes=4)),
        distillation=SimpleNamespace(**dist),
    )


def _logits_targets():
    s = torch.randn(2, 4, 8, 8, 8, requires_grad=True)
    t = torch.randn(2, 4, 8, 8, 8)
    tgt = torch.randint(0, 4, (2, 8, 8, 8))
    return s, t, tgt


def test_kdloss_raa_off_is_baseline_keys():
    """Default (no raa) -> exactly the 5 baseline keys, no loss_raa."""
    loss = build_kd_loss(_cfg())                 # raa absent -> disabled
    s, t, tgt = _logits_targets()
    out = loss(s, tgt, teacher_logits=t)
    assert set(out) == {"loss_total", "loss_task", "loss_kd", "loss_feat", "loss_attn"}
    assert loss.raa is None


def test_kdloss_raa_on_adds_loss_raa_term():
    loss = build_kd_loss(_cfg(raa=dict(enabled=True, dim=32, token_grid=2, heads=4, weight=0.5)))
    s, t, tgt = _logits_targets()
    sf = {"bottleneck": torch.randn(2, 16, 4, 4, 4, requires_grad=True)}
    tf = {"bottleneck": torch.randn(2, 24, 4, 4, 4)}
    out = loss(s, tgt, teacher_logits=t, student_features=sf, teacher_features=tf)
    assert "loss_raa" in out and float(out["loss_raa"]) >= 0.0
    assert torch.isfinite(out["loss_total"])
    out["loss_total"].backward()
    assert s.grad is not None
    # RAA params received gradient (they are part of the optimised graph).
    assert any(p.grad is not None for p in loss.raa.parameters())


def test_kdloss_raa_on_without_features_is_safe():
    loss = build_kd_loss(_cfg(raa=dict(enabled=True, dim=16, token_grid=2, heads=2)))
    s, t, tgt = _logits_targets()
    out = loss(s, tgt, teacher_logits=t)         # no features supplied
    assert "loss_raa" in out and float(out["loss_raa"]) == 0.0
    assert torch.isfinite(out["loss_total"])
