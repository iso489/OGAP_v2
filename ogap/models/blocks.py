"""OGAP conv building blocks (attention, residual, MedNeXt, MixStyle).

EXTRACTED VERBATIM from the original v9 monolith - now ``ogap/legacy.py``; the
top-level ``OGAP_source_code_experimental_v9.py`` is a thin shim - by the
v9.1 strangler-fig refactor. Class bodies are byte-identical to the monolith so
existing state_dict checkpoints load without remapping. Do not "improve" these
classes here without a matching note in the legacy shim - checkpoint compat
depends on attribute names and submodule order staying fixed.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ECABlock3D(nn.Module):
    """Efficient Channel Attention - quantisation-friendly replacement for SE.

    Uses 1D convolution to learn inter-channel dependencies with
    near-zero parameter overhead (~5-7 params per block vs ~8K for SE).
    Google removed SE blocks when creating quantisation-friendly
    EfficientNet-Lite; we follow the same principle for INT8 deployment.

    Ref: Wang et al., "ECA-Net" CVPR 2020
    """

    def __init__(self, channels: int, gamma: int = 2, b: int = 1) -> None:
        super().__init__()
        # Adaptive kernel size based on channel count (ECA formula)
        t = int(abs(math.log2(channels) + b) / gamma)
        k = max(t if t % 2 else t + 1, 3)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.act = nn.Hardsigmoid(inplace=True)  # quantisation-friendly vs sigmoid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _, _ = x.shape
        w = self.pool(x).view(b, 1, c)         # (B, 1, C)
        w = self.act(self.conv(w))              # (B, 1, C)
        return x * w.view(b, c, 1, 1, 1)


class SEBlock3D(nn.Module):
    """Squeeze-and-Excitation block for ablation comparison.

    Included only for ablation studies. The OGAP 2.0 deployable student
    defaults to no attention because that path is friendliest to INT8.

    Ref: Hu et al., "Squeeze-and-Excitation Networks" CVPR 2018
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1, 1)
        return x * w


# ── NEW #9: Curriculum Dropout (Morerio et al., ICCV 2017) ──
class CurriculumDropout3D(nn.Module):
    """Spatial dropout with curriculum scheduling.

    θ(t) = (1 - θ̄) · exp(-γt) + θ̄,  γ = 10/T

    Starts with nearly no dropout (θ(0) ≈ 1), gradually increases to
    the target drop rate. This implements curriculum learning applied to
    regularisation: easy learning first, then progressive challenge.

    Uses Dropout3d (spatial dropout) which drops entire feature map
    channels - more effective than element-wise dropout for CNNs.

    Ref: Morerio et al., "Curriculum Dropout" ICCV 2017 (arXiv:1703.06229)
         Li et al., "Disharmony Between Dropout and BN" CVPR 2019
    """

    def __init__(self, p_final: float = 0.1, total_epochs: int = 300) -> None:
        super().__init__()
        self.theta_bar = 1.0 - p_final  # target retain probability
        self.gamma = 10.0 / max(total_epochs, 1)
        self.current_epoch: int = 0
        self.p_final = p_final

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = epoch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p_final <= 0:
            return x
        # Curriculum retain probability: starts at ~1, decays to theta_bar
        retain_p = (1.0 - self.theta_bar) * math.exp(
            -self.gamma * self.current_epoch
        ) + self.theta_bar
        drop_p = 1.0 - retain_p
        # Clamp to valid range
        drop_p = max(0.0, min(drop_p, self.p_final))
        return F.dropout3d(x, p=drop_p, training=True)


def _group_count(channels: int, max_groups: int = 8) -> int:
    """Largest GroupNorm group count <= max_groups that divides channels."""
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ResConvBlock3D(nn.Module):
    """MedNeXt-style 3D block with residual skip and optional attention.

    The PDF explicitly asks for MedNeXt-S/MedNeXt-L style blocks rather than
    dense nnU-Net convolutions: depth-wise 3x3x3 convolution, GroupNorm, GELU,
    and point-wise expansion/projection. Keeping the historical class name
    preserves the surrounding training/export code while changing the actual
    block to the requested quantization-friendly architecture.

    Supports no attention (default), ECA, or SE for ablation.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        attention: str = "none",  # "eca", "se", "none"
        curriculum_dropout_p: float = 0.0,
        total_epochs: int = 300,
    ) -> None:
        super().__init__()
        expansion_ch = max(out_ch * 2, in_ch)
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False),
            nn.GroupNorm(_group_count(in_ch), in_ch),
            nn.GELU(),
            nn.Conv3d(in_ch, expansion_ch, 1, bias=False),
            nn.GELU(),
            nn.Conv3d(expansion_ch, out_ch, 1, bias=False),
            nn.GroupNorm(_group_count(out_ch), out_ch),
        )
        # Configurable attention for clean ablation
        if attention == "eca":
            self.attn = ECABlock3D(out_ch)
        elif attention == "se":
            self.attn = SEBlock3D(out_ch)
        else:
            self.attn = nn.Identity()

        self.skip = (
            nn.Identity()
            if in_ch == out_ch
            else nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, bias=False),
                nn.GroupNorm(_group_count(out_ch), out_ch),
            )
        )

        # Curriculum dropout (applied after residual add + attention)
        self.cdropout = CurriculumDropout3D(
            p_final=curriculum_dropout_p, total_epochs=total_epochs
        ) if curriculum_dropout_p > 0 else None

    def set_epoch(self, epoch: int) -> None:
        if self.cdropout is not None:
            self.cdropout.set_epoch(epoch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.attn(self.block(x) + self.skip(x))
        if self.cdropout is not None:
            out = self.cdropout(out)
        return out


class DenseConvBlock3D(nn.Module):
    """Dense 3D conv block for the HEAVY teacher only.

    Two full 3x3x3 convolutions with GroupNorm + GELU and a residual skip,
    matching the canonical SegMamba / nnU-Net block design. Roughly 4-5x
    more parameters per block than ResConvBlock3D (MedNeXt-S), giving the
    teacher meaningfully more capacity for knowledge distillation without
    affecting the student or its INT8 export path.

    Constructor signature is intentionally identical to ResConvBlock3D so
    teacher classes can switch between block styles by class reference.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        attention: str = "none",
        curriculum_dropout_p: float = 0.0,
        total_epochs: int = 300,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_ch), out_ch),
            nn.GELU(),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_ch), out_ch),
        )
        if attention == "eca":
            self.attn = ECABlock3D(out_ch)
        elif attention == "se":
            self.attn = SEBlock3D(out_ch)
        else:
            self.attn = nn.Identity()
        self.skip = (
            nn.Identity()
            if in_ch == out_ch
            else nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, bias=False),
                nn.GroupNorm(_group_count(out_ch), out_ch),
            )
        )
        self.cdropout = CurriculumDropout3D(
            p_final=curriculum_dropout_p, total_epochs=total_epochs
        ) if curriculum_dropout_p > 0 else None

    def set_epoch(self, epoch: int) -> None:
        if self.cdropout is not None:
            self.cdropout.set_epoch(epoch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.attn(self.block(x) + self.skip(x))
        if self.cdropout is not None:
            out = self.cdropout(out)
        return out


class LargeKernelConvBlock3D(nn.Module):
    """MedNeXt-faithful 3D block with a LARGE depthwise kernel (default 5×5×5).

    The original ``ResConvBlock3D`` uses a 3×3×3 depthwise kernel, which discards
    MedNeXt's defining ingredient - the large depthwise kernel that gives a wide
    receptive field (MedNeXt uses 5³/7³). That wide context matters most exactly
    where low-field segmentation is hard: blurry, low-contrast boundaries that need
    surrounding context to localise. This is the depthwise → GN → GELU → 1×1 expand
    → GELU → 1×1 project → GN inverted bottleneck (4× expansion, ConvNeXt/MedNeXt
    style) with a residual. New (additive) class: existing checkpoints are unaffected;
    this is selected only via ``block_style="mednext_large"`` on the heavy teacher.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        attention: str = "none",
        curriculum_dropout_p: float = 0.0,
        total_epochs: int = 300,
        kernel_size: int = 5,
        expansion: int = 4,
    ) -> None:
        super().__init__()
        pad = kernel_size // 2
        expansion_ch = max(out_ch * expansion, in_ch)
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, kernel_size, padding=pad, groups=in_ch, bias=False),
            nn.GroupNorm(_group_count(in_ch), in_ch),
            nn.GELU(),
            nn.Conv3d(in_ch, expansion_ch, 1, bias=False),
            nn.GELU(),
            nn.Conv3d(expansion_ch, out_ch, 1, bias=False),
            nn.GroupNorm(_group_count(out_ch), out_ch),
        )
        if attention == "eca":
            self.attn = ECABlock3D(out_ch)
        elif attention == "se":
            self.attn = SEBlock3D(out_ch)
        else:
            self.attn = nn.Identity()
        self.skip = (
            nn.Identity()
            if in_ch == out_ch
            else nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, bias=False),
                nn.GroupNorm(_group_count(out_ch), out_ch),
            )
        )
        self.cdropout = CurriculumDropout3D(
            p_final=curriculum_dropout_p, total_epochs=total_epochs
        ) if curriculum_dropout_p > 0 else None

    def set_epoch(self, epoch: int) -> None:
        if self.cdropout is not None:
            self.cdropout.set_epoch(epoch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.attn(self.block(x) + self.skip(x))
        if self.cdropout is not None:
            out = self.cdropout(out)
        return out


def _teacher_block_class(block_style: str) -> type:
    """Return the conv block class for a given teacher block style.

    "mednext"       → ResConvBlock3D (depthwise-separable 3³, quantization-friendly,
                      ~1M params per teacher @ base=32). Default.
    "dense"         → DenseConvBlock3D (full 3x3x3 convs, ~4-5x heavier).
    "mednext_large" → LargeKernelConvBlock3D (5³ depthwise + 4× inverted bottleneck;
                      wide receptive field for blurry low-field boundaries).

    Student always uses ResConvBlock3D regardless of this setting - only
    the teacher's capacity is configurable, since only the student is
    exported to INT8.
    """
    style = (block_style or "mednext").lower()
    if style == "dense":
        return DenseConvBlock3D
    if style == "mednext_large":
        return LargeKernelConvBlock3D
    if style in ("mednext", "default", ""):
        return ResConvBlock3D
    raise ValueError(
        f"Unknown teacher block_style {block_style!r}; expected 'mednext', 'dense', "
        "or 'mednext_large'."
    )


class MixStyle3D(nn.Module):
    """Feature-statistics domain randomization for 3D segmentation.

    During training, with probability ``p``, replaces each instance's
    per-channel (μ, σ) feature statistics with a Beta-mixed combination of
    its own and a permuted batchmate's. At eval time it is a no-op identity.

    Two modes (controlled by ``mode``):
      * ``"mixstyle"`` - Zhou et al., "Domain Generalization with MixStyle",
        ICLR 2021. Sample λ ∼ Beta(α, α) and mix with a permuted batchmate.
        Recommended default for cross-site segmentation.
      * ``"dsu"`` - Li et al., "Uncertainty Modeling for Out-of-Distribution
        Generalization in Visual Recognition" (DSU), ICLR 2022. Sample
        per-instance perturbed (μ, σ) from a Gaussian fitted to the
        within-batch (μ, σ) distribution. Slightly stronger; needs ≥4 batch.

    The 3D version normalizes over (D, H, W) per (batch, channel) and is
    cheap (one mean / std reduction + scatter-affine).

    Refs:
        Zhou+ ICLR 2021 (MixStyle).
        Li+ ICLR 2022 (DSU).
    """

    def __init__(self, p: float = 0.5, alpha: float = 0.1, mode: str = "mixstyle",
                 eps: float = 1e-6) -> None:
        super().__init__()
        if mode not in ("mixstyle", "dsu", "none"):
            raise ValueError(f"MixStyle3D mode must be one of mixstyle/dsu/none, got {mode!r}")
        self.p = float(p)
        self.alpha = float(alpha)
        self.mode = str(mode)
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.training) or self.mode == "none" or self.p <= 0.0:
            return x
        if x.shape[0] < 2:
            return x
        if torch.rand(1, device=x.device).item() >= self.p:
            return x
        b, c = x.shape[0], x.shape[1]
        # Per-instance, per-channel statistics over spatial dims.
        mu = x.mean(dim=(2, 3, 4), keepdim=True)
        var = x.var(dim=(2, 3, 4), keepdim=True, unbiased=False)
        sig = (var + self.eps).sqrt()
        x_norm = (x - mu) / sig

        if self.mode == "mixstyle":
            perm = torch.randperm(b, device=x.device)
            mu2 = mu[perm]
            sig2 = sig[perm]
            lam = torch.distributions.Beta(self.alpha, self.alpha).sample(
                (b, c, 1, 1, 1)
            ).to(x.device)
            mu_mix = lam * mu + (1.0 - lam) * mu2
            sig_mix = lam * sig + (1.0 - lam) * sig2
            return x_norm * sig_mix + mu_mix

        # DSU: estimate the (μ, σ) distribution across the batch and resample
        # per-instance perturbations. We use the batch's std-of-mean as the
        # uncertainty proxy (Li+ 2022 §3.2).
        mu_std = mu.std(dim=0, keepdim=True, unbiased=False) + self.eps
        sig_std = sig.std(dim=0, keepdim=True, unbiased=False) + self.eps
        eps_mu = torch.randn_like(mu) * mu_std
        eps_sig = torch.randn_like(sig) * sig_std
        return x_norm * (sig + eps_sig) + (mu + eps_mu)


class FeatureProjector(nn.Module):
    """1×1×1 projection: student feature space → teacher feature space.

    Fixes channel mismatch when student_base != teacher_base.
    """

    def __init__(self, s_channels: int, t_channels: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv3d(s_channels, t_channels, 1, bias=False),
            nn.GroupNorm(_group_count(t_channels), t_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)
