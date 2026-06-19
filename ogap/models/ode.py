"""Continuous-depth (Neural ODE) teacher for OGAP knowledge distillation.

Motivation
----------
OGAP distils a heavy *teacher* into a cheap INT8 *student*. The teacher is never
exported, so its only cost is training-time GPU memory - exactly the bottleneck
the Neural ODE paper removes. Chen et al. (NeurIPS 2018, §2) show that an
ODE-defined hidden state can be back-propagated with the **adjoint sensitivity
method at O(1) memory** regardless of effective depth. So we can give the
teacher a near-"infinite-depth" bottleneck refinement at *constant* memory,
producing richer features/logits for the KD signal without changing the
student or its export path.

``ODEBottleneckTeacher`` is a drop-in replacement for ``UNet3DTeacher``: same
constructor surface, same ``forward(x, return_features=...)`` contract, same
``bottleneck_channels`` / ``dec3_channels`` attributes that the KD
``FeatureProjector`` relies on. The only change is that the bottleneck is
refined by an :class:`ODEBlock3D` instead of a single conv block.

By default the teacher requests ``ODESolverConfig(method="dopri5",
adjoint=True)`` and therefore requires ``torchdiffeq``. This is intentional:
without the adjoint backend, the model is only a fixed-step differentiable ODE
block and does not satisfy the paper's constant-memory claim. Pass an explicit
``ODESolverConfig(method="rk4", adjoint=False)`` only for a non-adjoint ablation.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..numerics import ODESolverConfig, integrate
from .blocks import MixStyle3D, _group_count, _teacher_block_class


class ConvODEFunc3D(nn.Module):
    """The learned dynamics f(t, h) of the bottleneck ODE.

    Maps a hidden state of ``channels`` back to a derivative of the same shape
    (an ODE field must be shape-preserving). Time ``t`` is injected as an extra
    constant channel (the standard "augment with t" trick) so the field is
    genuinely non-autonomous. The body is a **dense** 3×3×3 conv → GroupNorm →
    GELU → pointwise expand → pointwise project → GroupNorm. (The teacher is never
    exported, so it can afford a full conv here; the *student* ODE field
    ``_StudentODEFunc3D`` is the depthwise-separable, INT8-friendly counterpart.)

    ``nfe`` counts function evaluations per solve - the continuous-depth analogue
    of "number of layers" (paper Fig. 3d).
    """

    def __init__(self, channels: int, hidden_mult: int = 2,
                 time_dependent: bool = True) -> None:
        super().__init__()
        self.time_dependent = bool(time_dependent)
        in_ch = channels + (1 if self.time_dependent else 0)
        hidden = max(channels * hidden_mult, channels)
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
            nn.GELU(),
            nn.Conv3d(channels, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv3d(hidden, channels, 1, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
        )
        self.nfe = 0

    def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        self.nfe += 1
        if self.time_dependent:
            # broadcast scalar t to a (B,1,D,H,W) channel
            tt = torch.ones_like(h[:, :1]) * t.to(h.dtype)
            h = torch.cat([h, tt], dim=1)
        return self.net(h)


class ODEBlock3D(nn.Module):
    """Integrate learned dynamics from t=0 to t=1 - a continuous residual block.

    A fixed-step Euler solve of this block is *exactly* a weight-tied residual
    network of depth ``steps``; an adaptive solve adapts depth per input. Either
    way the parameter count is that of a single block, independent of effective
    depth.
    """

    def __init__(self, channels: int, *, hidden_mult: int = 2,
                 time_dependent: bool = True,
                 solver: Optional[ODESolverConfig] = None) -> None:
        super().__init__()
        self.func = ConvODEFunc3D(channels, hidden_mult, time_dependent)
        self.solver = solver or ODESolverConfig(method="dopri5", adjoint=True)

    @property
    def nfe(self) -> int:
        return self.func.nfe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.func.nfe = 0
        return integrate(self.func, x, t0=0.0, t1=1.0, config=self.solver)


class ODEBottleneckTeacher(nn.Module):
    """UNet3DTeacher with a continuous-depth ODE bottleneck.

    Drop-in for ``UNet3DTeacher`` in the KD pipeline. Constructor and forward
    signature match, so ``build_teacher(arch="ode", ...)`` (see
    :func:`ogap.models.build_ode_teacher`) can supply it to
    :func:`pretrain_teacher` / the distillation loop unchanged. ``feature_dr``
    (MixStyle/DSU) injection mirrors ``UNet3DTeacher`` so the KD signal still
    carries cross-scanner invariance.
    """

    def __init__(
        self, in_channels: int, num_classes: int, base: int = 32,
        attention: str = "none",
        block_style: str = "mednext",
        feature_dr: str = "none",
        feature_dr_p: float = 0.5,
        feature_dr_alpha: float = 0.1,
        ode_hidden_mult: int = 2,
        ode_solver: Optional[ODESolverConfig] = None,
    ) -> None:
        super().__init__()
        b = base
        Block = _teacher_block_class(block_style)
        self.block_style = block_style
        self.enc1 = Block(in_channels, b, attention=attention)
        self.pool1 = nn.Conv3d(b, b, 2, stride=2, groups=b, bias=False)
        self.enc2 = Block(b, b * 2, attention=attention)
        self.pool2 = nn.Conv3d(b * 2, b * 2, 2, stride=2, groups=b * 2, bias=False)
        self.enc3 = Block(b * 2, b * 4, attention=attention)
        self.pool3 = nn.Conv3d(b * 4, b * 4, 2, stride=2, groups=b * 4, bias=False)
        # Bottleneck: conv lift to b*8, then continuous-depth ODE refinement.
        self.bottleneck = Block(b * 4, b * 8, attention=attention)
        self.bottleneck_ode = ODEBlock3D(
            b * 8, hidden_mult=ode_hidden_mult, solver=ode_solver
        )
        self.bottleneck_norm = nn.GroupNorm(_group_count(b * 8), b * 8)

        self.up3 = nn.ConvTranspose3d(b * 8, b * 4, 2, stride=2)
        self.dec3 = Block(b * 8, b * 4, attention=attention)
        self.up2 = nn.ConvTranspose3d(b * 4, b * 2, 2, stride=2)
        self.dec2 = Block(b * 4, b * 2, attention=attention)
        self.up1 = nn.ConvTranspose3d(b * 2, b, 2, stride=2)
        self.dec1 = Block(b * 2, b, attention=attention)

        if feature_dr in ("mixstyle", "dsu"):
            self.mixstyle_e1 = MixStyle3D(p=feature_dr_p, alpha=feature_dr_alpha, mode=feature_dr)
            self.mixstyle_e2 = MixStyle3D(p=feature_dr_p, alpha=feature_dr_alpha, mode=feature_dr)
        else:
            self.mixstyle_e1 = nn.Identity()
            self.mixstyle_e2 = nn.Identity()

        self.head = nn.Conv3d(b, num_classes, 1)
        self.bottleneck_channels: int = b * 8
        self.dec3_channels: int = b * 4

    @property
    def last_nfe(self) -> int:
        """Function evaluations used by the most recent bottleneck solve."""
        return self.bottleneck_ode.nfe

    def forward(self, x: torch.Tensor, return_features: bool = False):
        e1 = self.mixstyle_e1(self.enc1(x))
        e2 = self.mixstyle_e2(self.enc2(self.pool1(e1)))
        e3 = self.enc3(self.pool2(e2))
        bn_conv = self.bottleneck(self.pool3(e3))
        # Continuous-depth refinement at the bottleneck (residual via the ODE).
        bn = self.bottleneck_norm(self.bottleneck_ode(bn_conv)) + bn_conv

        up3 = self.up3(bn)
        e3 = e3[..., :up3.shape[2], :up3.shape[3], :up3.shape[4]]
        d3 = self.dec3(torch.cat([up3, e3], 1))
        up2 = self.up2(d3)
        e2 = e2[..., :up2.shape[2], :up2.shape[3], :up2.shape[4]]
        d2 = self.dec2(torch.cat([up2, e2], 1))
        up1 = self.up1(d2)
        e1 = e1[..., :up1.shape[2], :up1.shape[3], :up1.shape[4]]
        d1 = self.dec1(torch.cat([up1, e1], 1))
        logits = self.head(d1)
        if return_features:
            return logits, {"bottleneck": bn, "dec3": d3}
        return logits


def build_ode_teacher(in_channels: int, num_classes: int, base: int = 32,
                      **kwargs) -> ODEBottleneckTeacher:
    """Builder mirroring :func:`ogap.models.build_teacher` for ``arch='ode'``."""
    return ODEBottleneckTeacher(in_channels, num_classes, base=base, **kwargs)
