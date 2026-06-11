"""OGAP teacher networks.

``UNet3DTeacher`` preserves the historical checkpoint-compatible teacher. The
optional paper-backed teachers are additive: ``SegMambaTeacher`` follows the
SegMamba encoder/decoder design, and ``SwinUNETRTeacher`` delegates to MONAI's
Swin UNETR implementation.
"""
from __future__ import annotations

import logging
from typing import Dict, Tuple

import torch
import torch.nn as nn

from .blocks import (
    DenseConvBlock3D,
    MixStyle3D,
    ResConvBlock3D,
    _group_count,
    _teacher_block_class,
)

LOG = logging.getLogger("OGAP")


class UNet3DTeacher(nn.Module):
    """Unified full-modality teacher with a 4-level encoder-decoder.

    block_style="mednext" (default): MedNeXt-S depthwise-separable blocks,
        ~1M params @ base=32. Quantization-friendly but capacity-limited.
    block_style="dense": full 3x3x3 conv blocks, ~6M @ base=32, ~15M @
        base=48. Use for heavy KD teachers; the student stays MedNeXt-S
        for INT8 export so this only affects teacher capacity.
    """

    def __init__(
        self, in_channels: int, num_classes: int, base: int = 32,
        attention: str = "none",
        block_style: str = "mednext",
        feature_dr: str = "none",
        feature_dr_p: float = 0.5,
        feature_dr_alpha: float = 0.1,
    ) -> None:
        super().__init__()
        b = base
        Block = _teacher_block_class(block_style)
        self.block_style: str = block_style
        self.enc1 = Block(in_channels, b, attention=attention)
        self.pool1 = nn.Conv3d(b, b, 2, stride=2, groups=b, bias=False)
        self.enc2 = Block(b, b * 2, attention=attention)
        self.pool2 = nn.Conv3d(b * 2, b * 2, 2, stride=2, groups=b * 2, bias=False)
        self.enc3 = Block(b * 2, b * 4, attention=attention)
        self.pool3 = nn.Conv3d(b * 4, b * 4, 2, stride=2, groups=b * 4, bias=False)
        self.bottleneck = Block(b * 4, b * 8, attention=attention)

        self.up3 = nn.ConvTranspose3d(b * 8, b * 4, 2, stride=2)
        self.dec3 = Block(b * 8, b * 4, attention=attention)
        self.up2 = nn.ConvTranspose3d(b * 4, b * 2, 2, stride=2)
        self.dec2 = Block(b * 4, b * 2, attention=attention)
        self.up1 = nn.ConvTranspose3d(b * 2, b, 2, stride=2)
        self.dec1 = Block(b * 2, b, attention=attention)

        # FIX (Gap A, 2026 audit): Inject MixStyle3D after encoder block 1 and
        # block 2.  Only the student previously had feature-level domain
        # randomization, so the teacher logits used for KD encoded no
        # cross-scanner invariance — distillation then transferred only the
        # data-augmentation noise distribution to the student.  Putting
        # MixStyle3D in the teacher pushes domain-invariant features into the
        # KD signal itself.  MixStyle3D is an identity at eval, so this only
        # affects training.
        if feature_dr in ("mixstyle", "dsu"):
            self.mixstyle_e1 = MixStyle3D(p=feature_dr_p, alpha=feature_dr_alpha, mode=feature_dr)
            self.mixstyle_e2 = MixStyle3D(p=feature_dr_p, alpha=feature_dr_alpha, mode=feature_dr)
        else:
            self.mixstyle_e1 = nn.Identity()
            self.mixstyle_e2 = nn.Identity()

        self.head = nn.Conv3d(b, num_classes, 1)
        self.bottleneck_channels: int = b * 8
        self.dec3_channels: int = b * 4

    def forward(self, x: torch.Tensor, return_features: bool = False):
        e1 = self.mixstyle_e1(self.enc1(x))
        e2 = self.mixstyle_e2(self.enc2(self.pool1(e1)))
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

        if return_features:
            return logits, {"bottleneck": bn, "dec3": d3}
        return logits


class _ChannelLayerNorm3D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.permute(0, 2, 3, 4, 1).contiguous()
        y = self.norm(y)
        return y.permute(0, 4, 1, 2, 3).contiguous()


class _GatedSpatialConv3D(nn.Module):
    """SegMamba GSC: z + C3x3(C3x3(z) * C1x1(z))."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.branch3 = nn.Sequential(
            nn.GroupNorm(_group_count(channels), channels),
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            nn.GELU(),
        )
        self.branch1 = nn.Sequential(
            nn.GroupNorm(_group_count(channels), channels),
            nn.Conv3d(channels, channels, 1, bias=False),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fuse(self.branch3(x) * self.branch1(x))


class _TriOrientedMamba3D(nn.Module):
    """ToM: forward, reverse, and inter-slice Mamba scans fused in 3D."""

    def __init__(self, channels: int, mamba_cls) -> None:
        super().__init__()
        self.forward_mamba = mamba_cls(d_model=channels, d_state=16, d_conv=4, expand=2)
        self.reverse_mamba = mamba_cls(d_model=channels, d_state=16, d_conv=4, expand=2)
        self.slice_mamba = mamba_cls(d_model=channels, d_state=16, d_conv=4, expand=2)

    @staticmethod
    def _spatial_to_sequence(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int, int, int]]:
        b, c, d, h, w = x.shape
        return x.flatten(2).transpose(1, 2).contiguous(), (b, c, d, h, w)

    @staticmethod
    def _sequence_to_spatial(seq: torch.Tensor, shape: Tuple[int, int, int, int, int]) -> torch.Tensor:
        b, c, d, h, w = shape
        return seq.transpose(1, 2).contiguous().view(b, c, d, h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq, shape = self._spatial_to_sequence(x)
        fwd = self._sequence_to_spatial(self.forward_mamba(seq), shape)
        rev = torch.flip(self.reverse_mamba(torch.flip(seq, dims=[1])), dims=[1])
        rev = self._sequence_to_spatial(rev, shape)

        b, c, d, h, w = x.shape
        inter = x.permute(0, 3, 4, 2, 1).contiguous().view(b, h * w * d, c)
        inter = self.slice_mamba(inter).view(b, h, w, d, c).permute(0, 4, 3, 1, 2).contiguous()
        return (fwd + rev + inter) / 3.0


class _TSMambaBlock3D(nn.Module):
    """Tri-orientated Spatial Mamba block from SegMamba §2.1."""

    def __init__(self, channels: int, mamba_cls, mlp_ratio: int = 2) -> None:
        super().__init__()
        hidden = channels * mlp_ratio
        self.gsc = _GatedSpatialConv3D(channels)
        self.norm1 = _ChannelLayerNorm3D(channels)
        self.tom = _TriOrientedMamba3D(channels, mamba_cls)
        self.norm2 = _ChannelLayerNorm3D(channels)
        self.mlp = nn.Sequential(
            nn.Conv3d(channels, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv3d(hidden, channels, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_hat = self.gsc(x)
        z_tilde = self.tom(self.norm1(z_hat)) + z_hat
        return self.mlp(self.norm2(z_tilde)) + z_tilde


class _FeatureUncertaintyEnhancement3D(nn.Module):
    """FUE skip filtering from SegMamba §2.2 (Xing et al., MICCAI 2024).

    Implements the paper's Eq. 4 exactly: a channel-mean sigmoid gate
    ``z̄ = σ(mean_C(z))``, uncertainty ``u = -z̄·log(z̄)``, and re-weighted skip
    feature ``z̃ = z + z·(1 - u)``. The single-term ``-z̄·log z̄`` form IS the
    SegMamba formulation — not a reduced Bernoulli entropy and not an
    approximation — so this module is faithful to the reference implementation.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_bar = torch.sigmoid(x.mean(dim=1, keepdim=True))
        uncertainty = -z_bar * torch.log(z_bar.clamp_min(1e-6))
        return x + x * (1.0 - uncertainty)


def _crop_like(skip: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return skip[..., :ref.shape[2], :ref.shape[3], :ref.shape[4]]


class SegMambaTeacher(nn.Module):
    """SegMamba teacher aligned with Xing et al. MICCAI 2024.

    The encoder is a stem plus multi-scale TSMamba blocks. Each TSMamba block
    uses GSC before tri-orientated Mamba scans, and decoder skip features pass
    through FUE. This replaces the previous bottleneck-only Mamba approximation.
    """

    def __init__(
        self, in_channels: int, num_classes: int, base: int = 32,
        block_style: str = "dense",
        feature_dr: str = "none",
        feature_dr_p: float = 0.5,
        feature_dr_alpha: float = 0.1,
    ) -> None:
        super().__init__()
        try:
            from mamba_ssm import Mamba
        except ImportError as exc:
            raise ImportError(
                "SegMambaTeacher requires the 'mamba-ssm' package. Install with:\n"
                "  pip install mamba-ssm causal-conv1d\n"
                f"(import error: {exc})"
            )
        b = base
        ch = (b, b * 2, b * 4, b * 8, b * 16)
        Block = _teacher_block_class(block_style)
        self.block_style = block_style

        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, 7, stride=2, padding=3, groups=in_channels, bias=False),
            nn.Conv3d(in_channels, ch[0], 1, bias=False),
            nn.GroupNorm(_group_count(ch[0]), ch[0]),
            nn.GELU(),
        )
        self.stage0 = _TSMambaBlock3D(ch[0], Mamba)
        self.down1 = nn.Conv3d(ch[0], ch[1], 2, stride=2, bias=False)
        self.stage1 = _TSMambaBlock3D(ch[1], Mamba)
        self.down2 = nn.Conv3d(ch[1], ch[2], 2, stride=2, bias=False)
        self.stage2 = _TSMambaBlock3D(ch[2], Mamba)
        self.down3 = nn.Conv3d(ch[2], ch[3], 2, stride=2, bias=False)
        self.stage3 = _TSMambaBlock3D(ch[3], Mamba)
        self.down4 = nn.Conv3d(ch[3], ch[4], 2, stride=2, bias=False)
        self.stage4 = _TSMambaBlock3D(ch[4], Mamba)

        self.fue0 = _FeatureUncertaintyEnhancement3D()
        self.fue1 = _FeatureUncertaintyEnhancement3D()
        self.fue2 = _FeatureUncertaintyEnhancement3D()
        self.fue3 = _FeatureUncertaintyEnhancement3D()

        self.up4 = nn.ConvTranspose3d(ch[4], ch[3], 2, stride=2)
        self.dec4 = Block(ch[3] * 2, ch[3])
        self.up3 = nn.ConvTranspose3d(ch[3], ch[2], 2, stride=2)
        self.dec3 = Block(ch[2] * 2, ch[2])
        self.up2 = nn.ConvTranspose3d(ch[2], ch[1], 2, stride=2)
        self.dec2 = Block(ch[1] * 2, ch[1])
        self.up1 = nn.ConvTranspose3d(ch[1], ch[0], 2, stride=2)
        self.dec1 = Block(ch[0] * 2, ch[0])
        self.up0 = nn.ConvTranspose3d(ch[0], ch[0], 2, stride=2)
        self.head = nn.Conv3d(ch[0], num_classes, 1)

        if feature_dr in ("mixstyle", "dsu"):
            self.mixstyle_e1 = MixStyle3D(p=feature_dr_p, alpha=feature_dr_alpha, mode=feature_dr)
            self.mixstyle_e2 = MixStyle3D(p=feature_dr_p, alpha=feature_dr_alpha, mode=feature_dr)
        else:
            self.mixstyle_e1 = nn.Identity()
            self.mixstyle_e2 = nn.Identity()

        self.bottleneck_channels = ch[4]
        self.dec3_channels = ch[2]

    def forward(self, x: torch.Tensor, return_features: bool = False):
        z0 = self.mixstyle_e1(self.stage0(self.stem(x)))
        z1 = self.mixstyle_e2(self.stage1(self.down1(z0)))
        z2 = self.stage2(self.down2(z1))
        z3 = self.stage3(self.down3(z2))
        z4 = self.stage4(self.down4(z3))

        u4 = self.up4(z4)
        d4 = self.dec4(torch.cat([u4, _crop_like(self.fue3(z3), u4)], dim=1))
        u3 = self.up3(d4)
        d3 = self.dec3(torch.cat([u3, _crop_like(self.fue2(z2), u3)], dim=1))
        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat([u2, _crop_like(self.fue1(z1), u2)], dim=1))
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, _crop_like(self.fue0(z0), u1)], dim=1))
        logits = self.head(self.up0(d1))
        logits = logits[..., :x.shape[2], :x.shape[3], :x.shape[4]]

        if return_features:
            return logits, {"bottleneck": z4, "dec3": d3}
        return logits


class SwinUNETRTeacher(nn.Module):
    """MONAI Swin UNETR teacher matching Hatamizadeh et al. §3."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        base: int = 48,
        img_size: Tuple[int, int, int] = (128, 128, 128),
        **_: object,
    ) -> None:
        super().__init__()
        try:
            from monai.networks.nets import SwinUNETR
        except ImportError as exc:
            raise ImportError(
                "SwinUNETRTeacher requires MONAI. Install with:\n"
                "  pip install monai\n"
                f"(import error: {exc})"
            )
        feature_size = max(12, int((base + 11) // 12) * 12)
        kwargs = dict(in_channels=in_channels, out_channels=num_classes,
                      feature_size=feature_size, use_checkpoint=False)
        try:
            self.net = SwinUNETR(**kwargs)
        except TypeError:
            self.net = SwinUNETR(img_size=img_size, **kwargs)
        self.bottleneck_channels = feature_size * 16
        # MONAI SwinUNETR.decoder3 (UnetrUpBlock) emits feature_size*2 channels, not
        # *4 — verified empirically (feature_size=24 -> decoder3 = 48). The KD pipeline
        # sizes the dec3 FeatureProjector from this attribute, so a wrong value crashes
        # multi_scale_feature_distillation (cosine over dim=1) for any swin teacher with
        # lambda_feat > 0.
        self.dec3_channels = feature_size * 2
        self._features: Dict[str, torch.Tensor] = {}
        for attr, key in (("encoder10", "bottleneck"), ("decoder3", "dec3")):
            module = getattr(self.net, attr, None)
            if module is not None:
                module.register_forward_hook(self._save_feature(key))

    def _save_feature(self, key: str):
        def hook(_module, _inputs, output):
            if isinstance(output, torch.Tensor):
                self._features[key] = output
        return hook

    def forward(self, x: torch.Tensor, return_features: bool = False):
        self._features = {}
        logits = self.net(x)
        if not return_features:
            return logits
        return logits, dict(self._features)


def build_teacher(
    in_channels: int,
    num_classes: int,
    base: int,
    arch: str = "unet3d",
    attention: str = "none",
    block_style: str = "mednext",
    feature_dr: str = "none",
    feature_dr_p: float = 0.5,
    feature_dr_alpha: float = 0.1,
) -> nn.Module:
    """Construct a teacher network of the requested architecture.

    If the requested architecture's optional dependency is unavailable, the
    builder raises immediately. Silently changing architecture would make a
    later checkpoint load fail with a much less useful state_dict mismatch.

    block_style controls per-block parameter count (see _teacher_block_class):
    "mednext" (default) keeps backward compatibility with all existing
    teacher checkpoints; "dense" produces a heavy teacher for KD setups.

    feature_dr controls teacher-side domain randomization (2026 audit, Gap A):
        "none"      → identity (legacy behaviour)
        "mixstyle"  → MixStyle3D (Zhou et al. 2021)
        "dsu"       → distribution-uncertainty modeling (Li et al. 2022)
    """
    arch = (arch or "unet3d").lower()
    if arch in ("unet3d", "default", ""):
        return UNet3DTeacher(
            in_channels, num_classes, base,
            attention=attention, block_style=block_style,
            feature_dr=feature_dr,
            feature_dr_p=feature_dr_p,
            feature_dr_alpha=feature_dr_alpha,
        )
    if arch == "ode":
        # Continuous-depth ODE-bottleneck teacher. Lazy import avoids any
        # module-load coupling (ode.py imports only blocks.py + numerics).
        # Routing it here makes the package build_teacher API complete and
        # consistent with ogap.legacy.build_teacher, which already dispatches
        # 'ode'. [resolves the audit S2-C divergence named in
        # tests/test_legacy_package_parity.py]
        from .ode import build_ode_teacher
        return build_ode_teacher(
            in_channels, num_classes, base,
            attention=attention, block_style=block_style,
            feature_dr=feature_dr,
            feature_dr_p=feature_dr_p,
            feature_dr_alpha=feature_dr_alpha,
        )
    if arch == "segmamba":
        try:
            return SegMambaTeacher(
                in_channels, num_classes, base, block_style=block_style,
                feature_dr=feature_dr,
                feature_dr_p=feature_dr_p,
                feature_dr_alpha=feature_dr_alpha,
            )
        except ImportError as exc:
            LOG.error(
                "[teacher] --teacher_arch=segmamba requested but mamba-ssm "
                "is not available. Refusing to substitute UNet3DTeacher "
                "because SegMamba checkpoints would not be load-compatible. (%s)",
                exc,
            )
            raise
    if arch in ("swin_unetr", "swinunetr"):
        try:
            return SwinUNETRTeacher(in_channels, num_classes, base=base)
        except ImportError as exc:
            LOG.error(
                "[teacher] --teacher_arch=swin_unetr requested but MONAI is "
                "not available. Refusing to substitute another architecture. (%s)",
                exc,
            )
            raise
    raise ValueError(
        f"Unknown --teacher_arch {arch!r}; expected one of "
        "unet3d / ode / segmamba / swin_unetr."
    )
