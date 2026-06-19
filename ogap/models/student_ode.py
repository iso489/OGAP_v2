"""Weight-tied ODE student block - train continuous, deploy discrete (INT8).

The deployable OGAP student must (a) be tiny, (b) quantise to INT8, and (c)
export to a *static* ONNX graph at a *predictable* latency. A naive Neural ODE
violates (b) and (c). But there is an elegant reconciliation, noted in the
review: a **fixed-step Euler solve of a Neural ODE is exactly a residual network
whose blocks all share one set of weights**

    h_{k+1} = h_k + (1/N) · f(h_k)          k = 0 .. N-1

So we can:

* **Train** with continuous-depth semantics (optionally an adaptive solver via
  the shared :mod:`ogap.numerics` backend) for the regularisation / expressivity
  benefit, and
* **Deploy** the *unrolled Euler* form - a plain ``for`` loop of ``N`` shared-
  weight residual updates. That graph is static, ONNX-traceable, and INT8-
  quantisable, yet its parameter count is that of **one** block regardless of
  ``N``. More effective depth, no extra weights - friendly to weak hardware.

``WeightTiedODEBlock3D`` is a drop-in replacement for ``ResConvBlock3D``
(``in_ch, out_ch`` constructor, ``set_epoch`` no-op-compatible). Use
``deploy_mode="euler_unrolled"`` (default) for an export-safe student;
``deploy_mode="solver"`` routes through the numerics backend for research/adjoint
training.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..numerics import ODESolverConfig, integrate
from .blocks import _group_count


class _StudentODEFunc3D(nn.Module):
    """Lightweight, channels-preserving dynamics for the student (depthwise)."""

    def __init__(self, channels: int, time_dependent: bool = True) -> None:
        super().__init__()
        self.time_dependent = bool(time_dependent)
        in_ch = channels + (1 if self.time_dependent else 0)
        # depthwise 3x3x3 (cheap) on the time-augmented state, then pointwise.
        self.dw = nn.Conv3d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False)
        self.norm = nn.GroupNorm(_group_count(in_ch), in_ch)
        self.pw = nn.Conv3d(in_ch, channels, 1, bias=False)
        self.act = nn.GELU()
        self.nfe = 0

    def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        self.nfe += 1
        if self.time_dependent:
            tt = torch.ones_like(h[:, :1]) * t.to(h.dtype)
            x = torch.cat([h, tt], dim=1)
        else:
            x = h
        return self.pw(self.act(self.norm(self.dw(x))))


class WeightTiedODEBlock3D(nn.Module):
    """ResConvBlock3D-compatible block whose depth is weight-tied ODE integration.

    Parameters mirror ``ResConvBlock3D`` so it can be swapped into
    ``UNet3DStudent`` for ablations / the OFA supernet. ``steps`` sets the
    effective depth (number of Euler updates) at **no parameter cost**.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        attention: str = "none",          # accepted for signature parity (unused)
        curriculum_dropout_p: float = 0.0,  # accepted for signature parity
        total_epochs: int = 300,
        *,
        steps: int = 4,
        time_dependent: bool = True,
        deploy_mode: str = "euler_unrolled",  # "euler_unrolled" | "solver"
    ) -> None:
        super().__init__()
        if deploy_mode not in ("euler_unrolled", "solver"):
            raise ValueError(f"deploy_mode must be euler_unrolled|solver, got {deploy_mode!r}")
        if deploy_mode == "solver":
            import logging
            logging.getLogger(__name__).warning(
                "WeightTiedODEBlock3D(deploy_mode='solver') integrates with RK4, but "
                "the export path uses an explicit Euler unroll - a model trained in "
                "'solver' mode is a DIFFERENT function from its exported 'euler_unrolled' "
                "form. Use the default 'euler_unrolled' for an export-consistent "
                "(train==deploy) student. [audit S3-B]"
            )
        self.steps = int(steps)
        self.deploy_mode = deploy_mode
        # Channel projection (the only place in_ch != out_ch is handled).
        self.proj = (
            nn.Identity() if in_ch == out_ch
            else nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, bias=False),
                nn.GroupNorm(_group_count(out_ch), out_ch),
            )
        )
        self.func = _StudentODEFunc3D(out_ch, time_dependent=time_dependent)
        self.solver = ODESolverConfig(method="rk4", steps=self.steps, adjoint=False)

    def set_epoch(self, epoch: int) -> None:  # parity with ResConvBlock3D
        pass

    @property
    def nfe(self) -> int:
        return self.func.nfe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        self.func.nfe = 0
        if self.deploy_mode == "solver":
            return integrate(self.func, h, t0=0.0, t1=1.0, config=self.solver)
        # Export-safe explicit Euler unroll (static graph, INT8-friendly).
        dt = 1.0 / self.steps
        t = h.new_zeros(())
        for _ in range(self.steps):
            h = h + dt * self.func(t, h)
            t = t + dt
        return h

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
