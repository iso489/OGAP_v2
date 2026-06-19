"""Once-for-All (OFA) elastic supernet for OGAP's deployable student.

OGAP's hardest deployment problem is *hardware heterogeneity*: African clinics
span old CPUs, weak GPUs and edge boxes. The survey's one-shot / weight-sharing
methods (White et al. 2023, §4) and specifically Once-for-All (Cai et al. 2020)
answer this: train **one** supernet, then extract a specialised subnet per
hardware target **without retraining** - "one training run → a right-sized INT8
student per clinic".

Two elastic axes:

* **width** - channel-sliced :class:`DynamicConv3d` / :class:`DynamicGroupNorm`
  share weights across widths (slimmable-network mechanism, Yu et al. 2019).
* **depth** - the bottleneck refinement block is applied ``depth`` times with
  **shared weights** - precisely the weight-tied Euler view of a Neural ODE
  (:mod:`ogap.models.student_ode`). Elastic depth costs zero extra parameters.

Design note: skip connections are **additive**, not concatenated. With a concat
+ contiguous-slice scheme, an elastic-width conv whose input is the concat of two
elastic tensors would mis-align channel groups. Additive fusion gives every conv
a single elastic input source, so the top-left weight slice is exactly correct -
which is what makes ``export_subnet()`` reproduce the supernet output bit-for-bit
(verified in tests).
"""
from __future__ import annotations

import random
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.blocks import _group_count

WIDTH_MULTS: Tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
DEPTHS: Tuple[int, ...] = (2, 4, 6, 8)


def make_divisible(v: float, divisor: int = 4, min_val: int = 4) -> int:
    return int(max(min_val, int(v + divisor / 2) // divisor * divisor))


class DynamicConv3d(nn.Module):
    """3D conv using the top-left ``[:active_out, :in_ch]`` weight slice."""

    def __init__(self, max_in: int, max_out: int, k: int,
                 stride: int = 1, padding: int = 0, bias: bool = False) -> None:
        super().__init__()
        self.conv = nn.Conv3d(max_in, max_out, k, stride=stride, padding=padding, bias=bias)
        self.stride = stride
        self.padding = padding
        self.active_out = max_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_ch = x.shape[1]
        w = self.conv.weight[:self.active_out, :in_ch]
        b = self.conv.bias[:self.active_out] if self.conv.bias is not None else None
        return F.conv3d(x, w, b, stride=self.stride, padding=self.padding)


class DynamicGroupNorm(nn.Module):
    """GroupNorm over the first ``C`` channels (C read at runtime)."""

    def __init__(self, max_ch: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(max_ch))
        self.bias = nn.Parameter(torch.zeros(max_ch))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = x.shape[1]
        return F.group_norm(x, _group_count(c), self.weight[:c], self.bias[:c], self.eps)


class DynamicConvBlock(nn.Module):
    """Residual conv block with elastic output width (single input source)."""

    def __init__(self, max_in: int, max_out: int) -> None:
        super().__init__()
        self.conv1 = DynamicConv3d(max_in, max_out, 3, padding=1)
        self.norm1 = DynamicGroupNorm(max_out)
        self.conv2 = DynamicConv3d(max_out, max_out, 3, padding=1)
        self.norm2 = DynamicGroupNorm(max_out)
        self.skip = DynamicConv3d(max_in, max_out, 1)
        self.act = nn.GELU()
        self.active_out = max_out

    def set_active_out(self, c: int) -> None:
        self.active_out = c
        self.conv1.active_out = c
        self.conv2.active_out = c
        self.skip.active_out = c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.norm2(self.conv2(self.act(self.norm1(self.conv1(x)))))
        return self.act(out + self.skip(x))


class DynamicFuseBlock(nn.Module):
    """Decoder block that *additively* fuses an upsampled path and a skip path.

    Two separate elastic convs (one per source) avoid the concat channel-group
    mis-alignment, so width elasticity stays exact.
    """

    def __init__(self, max_up: int, max_skip: int, max_out: int) -> None:
        super().__init__()
        self.conv_up = DynamicConv3d(max_up, max_out, 3, padding=1)
        self.conv_skip = DynamicConv3d(max_skip, max_out, 3, padding=1)
        self.norm1 = DynamicGroupNorm(max_out)
        self.conv2 = DynamicConv3d(max_out, max_out, 3, padding=1)
        self.norm2 = DynamicGroupNorm(max_out)
        self.act = nn.GELU()
        self.active_out = max_out

    def set_active_out(self, c: int) -> None:
        self.active_out = c
        self.conv_up.active_out = c
        self.conv_skip.active_out = c
        self.conv2.active_out = c

    def forward(self, up: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv_up(up) + self.conv_skip(skip)))
        return self.act(self.norm2(self.conv2(h)))


def _upsample_to(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return F.interpolate(x, size=ref.shape[2:], mode="trilinear", align_corners=False)


class OFASupernet3D(nn.Module):
    """Elastic-width / elastic-depth UNet supernet for the OGAP student."""

    def __init__(self, in_channels: int, num_classes: int,
                 stage_chs: Sequence[int] = (16, 32, 64, 128)) -> None:
        super().__init__()
        c1, c2, c3, c4 = stage_chs
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.stage_chs_max: Tuple[int, ...] = (c1, c2, c3, c4)

        self.enc1 = DynamicConvBlock(in_channels, c1)
        self.enc2 = DynamicConvBlock(c1, c2)
        self.enc3 = DynamicConvBlock(c2, c3)
        self.pool = nn.MaxPool3d(2)
        self.bott_lift = DynamicConvBlock(c3, c4)
        self.bott_refine = DynamicConvBlock(c4, c4)   # applied `depth` times (tied)

        self.dec3 = DynamicFuseBlock(max_up=c4, max_skip=c3, max_out=c3)
        self.dec2 = DynamicFuseBlock(max_up=c3, max_skip=c2, max_out=c2)
        self.dec1 = DynamicFuseBlock(max_up=c2, max_skip=c1, max_out=c1)
        self.head = DynamicConv3d(c1, num_classes, 1)
        self.head.active_out = num_classes

        self.active_depth: int = max(DEPTHS)
        self._active: Tuple[int, ...] = (c1, c2, c3, c4)
        self.set_active_subnet(1.0, max(DEPTHS))

    def set_active_subnet(self, width_mult: float, depth: int) -> None:
        a1, a2, a3, a4 = (make_divisible(width_mult * c) for c in self.stage_chs_max)
        self.enc1.set_active_out(a1)
        self.enc2.set_active_out(a2)
        self.enc3.set_active_out(a3)
        self.bott_lift.set_active_out(a4)
        self.bott_refine.set_active_out(a4)
        self.dec3.set_active_out(a3)
        self.dec2.set_active_out(a2)
        self.dec1.set_active_out(a1)
        self.head.active_out = self.num_classes
        self.active_depth = int(depth)
        self._active = (a1, a2, a3, a4)

    def active_stage_chs(self) -> Tuple[int, ...]:
        return self._active

    def random_subnet(self, rng: Optional[random.Random] = None) -> Tuple[float, int]:
        r = rng or random
        w, d = r.choice(WIDTH_MULTS), r.choice(DEPTHS)
        self.set_active_subnet(w, d)
        return w, d

    def active_param_count(self) -> int:
        return int(sum(p.numel() for p in self.export_subnet().parameters()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        bn = self.bott_lift(self.pool(e3))
        for _ in range(self.active_depth):
            bn = self.bott_refine(bn)                 # weight-tied elastic depth
        d3 = self.dec3(_upsample_to(bn, e3), e3)
        d2 = self.dec2(_upsample_to(d3, e2), e2)
        d1 = self.dec1(_upsample_to(d2, e1), e1)
        return self.head(d1)

    def export_subnet(self) -> "OFASupernet3D":
        """Standalone, smaller model with the active subnet's (sliced) weights.

        The slices are exactly those the dynamic forward uses, so the export
        reproduces the supernet output at this config (verified in tests). This
        is the artifact you INT8-quantise and ship to a clinic.
        """
        sub = OFASupernet3D(self.in_channels, self.num_classes, stage_chs=self._active)
        sub.set_active_subnet(1.0, self.active_depth)
        src = dict(self.named_parameters())
        for name, p in sub.named_parameters():
            s = src[name]
            sl = tuple(slice(0, dim) for dim in p.shape)
            p.data.copy_(s.data[sl])
        return sub
