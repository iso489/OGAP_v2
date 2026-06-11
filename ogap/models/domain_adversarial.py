"""Domain-adversarial components (opt-in, gated by ``domain.enabled`` in config).

A gradient-reversal layer (GRL) plus a small domain classifier let the encoder
learn domain-invariant features: the classifier tries to predict the domain
(e.g. scanner / site), while the GRL flips the sign of its gradient so the
encoder is pushed to *fool* it. Forward is the identity; only the backward pass
is modified (Ganin & Lempitsky's DANN trick). Pure ``torch``; CPU-friendly.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _GradientReversalFn(torch.autograd.Function):
    """Identity forward; gradient is negated and scaled by lambda on the backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambda_ * grad_output, None        # None for the lambda arg


def grad_reverse(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    """Functional gradient reversal: returns ``x`` unchanged, reverses its grad."""
    return _GradientReversalFn.apply(x, lambda_)


class GradientReversalLayer(nn.Module):
    """Module wrapper around :func:`grad_reverse` with an adjustable ``lambda_``.

    ``lambda_`` is typically ramped from 0 to 1 over training; use
    :meth:`set_lambda` from the schedule. ``lambda_=0`` blocks the adversarial
    gradient entirely (forward still identity).
    """

    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = float(lambda_)

    def set_lambda(self, lambda_: float) -> None:
        self.lambda_ = float(lambda_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return grad_reverse(x, self.lambda_)

    def extra_repr(self) -> str:
        return f"lambda_={self.lambda_}"


class DomainClassifier(nn.Module):
    """Predict the domain from pooled feature maps.

    Args:
        in_channels: channel count of the input feature map ``(B, C, D, H, W)``.
        num_domains: number of domain labels.
        hidden: hidden width of the 2-layer MLP head.
    """

    def __init__(self, in_channels: int, num_domains: int = 2, hidden: int = 64):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_domains),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.pool(x))                  # (B, C, D, H, W) -> (B, num_domains)
