"""Auxiliary tissue-segmentation head (opt-in multitask regulariser).

A small conv head that predicts tissue classes from shared features, plus a
weighted-CE helper for adding it to the main loss. All CPU, no data.
"""
import torch

from ogap.models.auxiliary_heads import AuxiliaryTissueHead, auxiliary_tissue_loss


def test_head_output_shape_matches_input_spatial():
    head = AuxiliaryTissueHead(in_channels=16, num_tissue=3)
    out = head(torch.randn(2, 16, 6, 6, 6))
    assert out.shape == (2, 3, 6, 6, 6)


def test_head_upsamples_to_out_size():
    head = AuxiliaryTissueHead(in_channels=8, num_tissue=3)
    out = head(torch.randn(1, 8, 4, 4, 4), out_size=(8, 8, 8))
    assert out.shape == (1, 3, 8, 8, 8)


def test_head_forward_is_finite_and_has_parameters():
    head = AuxiliaryTissueHead(in_channels=4, num_tissue=2)
    out = head(torch.randn(1, 4, 5, 5, 5))
    assert torch.isfinite(out).all()
    assert sum(p.numel() for p in head.parameters()) > 0


def test_auxiliary_tissue_loss_is_scalar_and_scales_with_weight():
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 4, 4, 4, requires_grad=True)
    target = torch.randint(0, 3, (2, 4, 4, 4))
    l_full = auxiliary_tissue_loss(logits, target, weight=1.0)
    l_half = auxiliary_tissue_loss(logits, target, weight=0.5)
    assert l_full.ndim == 0                              # scalar
    assert torch.allclose(l_half, 0.5 * l_full)
    l_full.backward()                                    # differentiable
    assert logits.grad is not None


def test_auxiliary_tissue_loss_zero_weight_is_zero():
    logits = torch.randn(1, 3, 4, 4, 4)
    target = torch.randint(0, 3, (1, 4, 4, 4))
    assert float(auxiliary_tissue_loss(logits, target, weight=0.0)) == 0.0
