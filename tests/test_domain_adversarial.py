"""Domain-adversarial pieces: gradient-reversal layer + domain classifier.

Forward is the identity; the backward pass negates and scales the gradient by
lambda (the DANN gradient-reversal trick). All CPU, no data.
"""
import torch
import torch.nn as nn

from ogap.models.domain_adversarial import (
    DomainClassifier,
    GradientReversalLayer,
    grad_reverse,
)


def test_grad_reverse_forward_is_identity():
    x = torch.randn(2, 3, 4)
    assert torch.equal(grad_reverse(x, 1.0), x)


def test_grad_reverse_negates_and_scales_gradient():
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    grad_reverse(x, 2.0).sum().backward()
    assert torch.allclose(x.grad, torch.full((3,), -2.0))


def test_layer_forward_identity_backward_reversed():
    layer = GradientReversalLayer(lambda_=0.5)
    x = torch.randn(4, requires_grad=True)
    y = layer(x)
    assert torch.equal(y, x)
    y.sum().backward()
    assert torch.allclose(x.grad, torch.full((4,), -0.5))


def test_layer_lambda_zero_blocks_gradient():
    layer = GradientReversalLayer(lambda_=1.0)
    layer.set_lambda(0.0)
    x = torch.randn(3, requires_grad=True)
    layer(x).sum().backward()
    assert torch.allclose(x.grad, torch.zeros(3))


def test_domain_classifier_output_shape():
    clf = DomainClassifier(in_channels=8, num_domains=3)
    out = clf(torch.randn(2, 8, 5, 5, 5))
    assert out.shape == (2, 3)


def test_domain_classifier_with_grl_propagates_reversed_gradient():
    clf = DomainClassifier(in_channels=4, num_domains=2)
    grl = GradientReversalLayer(lambda_=1.0)
    feats = torch.randn(2, 4, 3, 3, 3, requires_grad=True)
    logits = clf(grl(feats))
    assert logits.shape == (2, 2)
    logits.sum().backward()
    assert feats.grad is not None and torch.isfinite(feats.grad).all()
