"""Auxiliary tissue-segmentation head (opt-in, gated by
``loss.auxiliary_tissue.enabled``).

Predicting healthy-tissue classes (e.g. WM / GM / CSF) from the shared encoder
features is a multitask regulariser: it gives the backbone an extra anatomical
signal without changing the primary tumour-segmentation output. The head is a
small conv stack that can optionally upsample its logits back to the label
resolution; :func:`auxiliary_tissue_loss` is a weighted cross-entropy for adding
it to the main objective. Pure ``torch``; CPU-friendly.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class AuxiliaryTissueHead(nn.Module):
    """Conv head mapping shared features to tissue-class logits.

    Args:
        in_channels: channels of the input feature map ``(B, C, D, H, W)``.
        num_tissue: number of tissue classes to predict.
        hidden: width of the intermediate conv (defaults to ``in_channels // 2``,
            at least ``num_tissue``).
    """

    def __init__(self, in_channels: int, num_tissue: int = 3,
                 hidden: Optional[int] = None):
        super().__init__()
        hidden = hidden or max(in_channels // 2, num_tissue)
        self.head = nn.Sequential(
            nn.Conv3d(in_channels, hidden, kernel_size=3, padding=1),
            nn.InstanceNorm3d(hidden, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, num_tissue, kernel_size=1),
        )

    def forward(self, x: torch.Tensor,
                out_size: Optional[Sequence[int]] = None) -> torch.Tensor:
        """Return tissue logits ``(B, num_tissue, D, H, W)``.

        If ``out_size`` is given and differs from the feature resolution, the
        logits are trilinearly resampled to it (to match the label volume).
        """
        y = self.head(x)
        if out_size is not None and tuple(y.shape[2:]) != tuple(out_size):
            y = F.interpolate(y, size=tuple(out_size), mode="trilinear",
                              align_corners=False)
        return y


def auxiliary_tissue_loss(logits: torch.Tensor, target: torch.Tensor,
                          weight: float = 0.1) -> torch.Tensor:
    """Weighted cross-entropy for the auxiliary tissue task.

    Args:
        logits: ``(B, num_tissue, D, H, W)`` tissue logits.
        target: ``(B, D, H, W)`` integer tissue labels.
        weight: scalar multiplier (``loss.auxiliary_tissue.weight``); ``0`` cleanly
            disables the contribution.

    Returns:
        A scalar tensor ``weight * CE(logits, target)``.
    """
    return weight * F.cross_entropy(logits, target)
