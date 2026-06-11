"""OGAP missing-modality student network (UNet3DStudent).

EXTRACTED VERBATIM from the original v9 monolith — now ``ogap/legacy.py``; the
top-level ``OGAP_source_code_experimental_v9.py`` is a thin shim — by the
v9.1 strangler-fig refactor. Class bodies are byte-identical to the monolith so
existing state_dict checkpoints load without remapping. Do not "improve" these
classes here without a matching note in the legacy shim — checkpoint compat
depends on attribute names and submodule order staying fixed.

``block_style`` (added v9.1) selects the residual block family:
* ``"res"`` (default) — the historical ``ResConvBlock3D`` blocks; the default
  path is constructed identically to the monolith, so existing checkpoints load
  unchanged.
* ``"ode"`` — ``WeightTiedODEBlock3D`` continuous-depth blocks (the export-safe
  student distilled from a SegMamba/ODE teacher). Effective depth is set by
  ``ode_steps`` at **no extra parameters**.
"""
from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import MixStyle3D, ResConvBlock3D
from .student_ode import WeightTiedODEBlock3D


class UNet3DStudent(nn.Module):
    """Missing-modality MedNeXt-S style student with deep supervision.

    Input  : 2C channels (C masked volumes + C binary availability maps)
    Output : main logits + auxiliary logits from dec3, dec2
    base=16 → ~0.24M parameters (depthwise-separable MedNeXt-S; base=32 → ~0.88M),
    <500 MB RAM at inference

    FIX #5 (partial): attention and deep_supervision are now configurable
    via constructor for clean ablation studies.
    feature_dr in {"none","mixstyle","dsu"} inserts a MixStyle3D module
    after enc1 to provide free feature-space domain randomization (Zhou+
    2021 / Li+ 2022). At eval time MixStyle3D is an identity.

    block_style in {"res","ode"} selects the residual block family (see module
    docstring). "res" is the default and is checkpoint-compatible; "ode" builds a
    weight-tied continuous-depth student whose effective depth is ``ode_steps``.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        base: int = 16,
        attention: str = "none",          # "eca", "se", "none"
        enable_deep_supervision: bool = True,
        curriculum_dropout_p: float = 0.1,
        total_epochs: int = 300,
        feature_dr: str = "none",
        feature_dr_p: float = 0.5,
        feature_dr_alpha: float = 0.1,
        block_style: str = "res",
        ode_steps: int = 4,
    ) -> None:
        super().__init__()
        if block_style not in ("res", "ode"):
            raise ValueError(
                f"block_style must be 'res' or 'ode', got {block_style!r}."
            )
        b = base
        self._enable_ds = enable_deep_supervision
        self.block_style = block_style
        self.ode_steps = int(ode_steps)
        # Residual block family. The default "res" path binds Block to
        # ResConvBlock3D so construction is byte-identical to the monolith
        # (checkpoint compatibility). "ode" swaps in the export-safe weight-tied
        # ODE block; ``steps`` is bound here so the per-block call sites are
        # unchanged and ResConvBlock3D-compatible.
        if block_style == "ode":
            Block = partial(WeightTiedODEBlock3D, steps=self.ode_steps)
        else:
            Block = ResConvBlock3D

        # Curriculum dropout only on deeper layers (bottleneck + dec3)
        # to avoid disrupting low-level feature learning
        self.enc1 = Block(in_channels, b, attention=attention)
        # Feature-space domain randomization (Zhou+ 2021 MixStyle / Li+ 2022 DSU).
        # Inserted after the first encoder block — the standard early-layer
        # placement that the MixStyle paper found most effective.
        self.feature_dr = MixStyle3D(p=feature_dr_p, alpha=feature_dr_alpha, mode=feature_dr)
        self.pool1 = nn.Conv3d(b, b, 2, stride=2, groups=b, bias=False)
        self.enc2 = Block(b, b * 2, attention=attention)
        self.pool2 = nn.Conv3d(b * 2, b * 2, 2, stride=2, groups=b * 2, bias=False)
        self.enc3 = Block(b * 2, b * 4, attention=attention)
        self.pool3 = nn.Conv3d(b * 4, b * 4, 2, stride=2, groups=b * 4, bias=False)
        self.bottleneck = Block(
            b * 4, b * 8, attention=attention,
            curriculum_dropout_p=curriculum_dropout_p,
            total_epochs=total_epochs,
        )

        self.up3 = nn.ConvTranspose3d(b * 8, b * 4, 2, stride=2)
        self.dec3 = Block(
            b * 8, b * 4, attention=attention,
            curriculum_dropout_p=curriculum_dropout_p,
            total_epochs=total_epochs,
        )
        self.up2 = nn.ConvTranspose3d(b * 4, b * 2, 2, stride=2)
        self.dec2 = Block(b * 4, b * 2, attention=attention)
        self.up1 = nn.ConvTranspose3d(b * 2, b, 2, stride=2)
        self.dec1 = Block(b * 2, b, attention=attention)

        self.head = nn.Conv3d(b, num_classes, 1)
        self.uncertainty_head = nn.Conv3d(b, num_classes, 1)
        self.rano_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(b * 8, 12),
        )

        # Deep supervision heads (only created if enabled)
        if enable_deep_supervision:
            # Learned upsampling keeps auxiliary supervision on conv kernels
            # instead of large trilinear interpolation passes.
            self.aux_head3 = nn.ConvTranspose3d(b * 4, num_classes, 4, stride=4)
            self.aux_head2 = nn.ConvTranspose3d(b * 2, num_classes, 2, stride=2)
        else:
            # Register as None so state_dict is consistent
            self.aux_head3 = None
            self.aux_head2 = None

        self.bottleneck_channels: int = b * 8
        self.dec3_channels: int = b * 4

    def set_epoch(self, epoch: int) -> None:
        """Update curriculum dropout schedule for all blocks."""
        for module in self.modules():
            if isinstance(module, (ResConvBlock3D, WeightTiedODEBlock3D)):
                module.set_epoch(epoch)

    def forward(
        self, x: torch.Tensor,
        return_features: bool = False,
        deep_supervision: bool = False,
        return_aux_outputs: bool = False,
    ):
        e1 = self.enc1(x)
        # Feature-space domain randomization (no-op if mode == "none" or eval mode).
        e1 = self.feature_dr(e1)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        bn = self.bottleneck(self.pool3(e3))

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
        aux_outputs = None
        if return_aux_outputs:
            aux_outputs = {
                # self.rano_head is untrained (no per-case RANO targets exist for any
                # OGAP cohort) and is NOT emitted — exposing an untrained head's output as
                # "rano" risks it being read as a clinical measurement. The reported RANO
                # comes from the geometric _region_clinical_measurements; the head is kept
                # in __init__ only for checkpoint compatibility.
                "uncertainty_logits": self.uncertainty_head(d1),
                "ood_features": F.adaptive_avg_pool3d(bn.float(), 1).flatten(1),
            }

        if deep_supervision and self._enable_ds:
            aux3 = self.aux_head3(d3)
            aux2 = self.aux_head2(d2)
            if return_features:
                if return_aux_outputs:
                    return logits, aux3, aux2, {"bottleneck": bn, "dec3": d3, **aux_outputs}
                return logits, aux3, aux2, {"bottleneck": bn, "dec3": d3}
            if return_aux_outputs:
                return logits, aux3, aux2, aux_outputs
            return logits, aux3, aux2

        if return_features:
            if return_aux_outputs:
                return logits, {"bottleneck": bn, "dec3": d3, **aux_outputs}
            return logits, {"bottleneck": bn, "dec3": d3}
        if return_aux_outputs:
            return logits, aux_outputs
        return logits
