"""Differentiable Bloch low-field MRI simulator (training-time augmentation).

Renders a physically-consistent ~64 mT appearance of the four BraTS channels
(T1, T1Gd, T2, FLAIR) from **tissue partial-volume maps** (FSL FAST CSF/WM/GM) and
the **BraTS label**, using the field-dependent relaxation constants in
:mod:`ogap.augmentations.relaxation_constants`.

Physics
-------
Per voxel the signal is the **partial-volume sum of per-tissue Bloch signals**

    S(v) = sum_t  f_t(v) * S_Bloch(T1_t(B0), T2_t(B0), PD_t ; TR, TE, TI)

(NOT a parameter average — averaging T1/T2 is unphysical). FSL FAST is reliable
only in healthy brain, so voxels inside the BraTS label are **overridden** with the
tumour-sub-region relaxation values (TC/ED/ET), and Gd enhancement shortens T1 of the
enhancing tumour on the post-contrast channel only.

Realism layer (training)
------------------------
A naive partial-volume render is piecewise-constant and has razor-sharp label
boundaries — *sharper* than a real low-field scan, which has more partial-volume
blur and within-tissue heterogeneity. :class:`BlochLowFieldSimulator` therefore adds,
on top of the physics render:

* **within-tissue heterogeneity** — a smooth, low-frequency multiplicative field so
  tissue/tumour regions are not perfectly flat;
* **low-field PSF blur** — a small Gaussian blur so lesion boundaries match the
  partial-volume blur of a real low-resolution acquisition;
* **real-texture blend** — adds back the high-frequency texture of the real scan so
  fine anatomy (sulci/gyri/vessels) the PVE maps do not capture survives;
* **acquisition randomisation** — optional per-sample jitter of TR/TE/TI and the
  target field, so the network sees a *distribution* of low-field appearances.

Properties
----------
* **Label-preserving** — the segmentation mask is never modified.
* **Differentiable / device-native** — pure ``torch`` ops, so the render is autograd-
  friendly w.r.t. the input partial-volume fractions and composes with the H100 batch
  path. (The relaxation constants currently enter as Python scalars and are ``float()``-
  cast inside ``_bloch_signal_torch``, so gradients do NOT flow to T1/T2/PD/TR/TE. To
  enable gradient-based matching of sim params, change that signature to accept tensors
  and drop the ``float(...)`` casts — passing tensors alone is not enough.)
* **Training-time only** — deployment/inference needs no FSL and no simulator.
* **Background-preserving** — skull-stripped zero background stays zero (no train/test
  shift, and the downstream ``v != 0`` masks stay valid).

This is the principled, physics-exact replacement for the heuristic legacy
``P_FIELD_CONTRAST_WARP`` and the stand-in for scarce native low-field data.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Sequence as Seq, Tuple

import torch
import torch.nn.functional as F

from . import relaxation_constants as rc


@dataclass(frozen=True)
class PulseSequence:
    """Acquisition parameters (ms) for one output channel."""

    name: str
    tr: float
    te: float
    ti: float = 0.0  # >0 => inversion recovery (FLAIR nulls CSF)


# Default BraTS channel order: T1, T1Gd, T2, FLAIR. Representative clinical timings; the
# FLAIR TI nulls CSF using the FINITE-TR inversion-null TI = T1*ln(2/(1+exp(-TR/T1)))
# (T1_CSF=4300, TR=9000 -> ~2481 ms). The simpler T1*ln2 ~ 2980 ms is only the TR>>T1 limit
# and leaves ~12% residual CSF at this TR. CSF T1 is B0-independent, so the null TI holds at
# the target field.
DEFAULT_BRATS_SEQUENCES: Tuple[PulseSequence, ...] = (
    PulseSequence("t1n", tr=500.0, te=10.0, ti=0.0),       # T1-weighted spin-echo
    PulseSequence("t1c", tr=500.0, te=10.0, ti=0.0),       # T1-weighted post-Gd
    PulseSequence("t2w", tr=4000.0, te=100.0, ti=0.0),     # T2-weighted
    PulseSequence("t2f", tr=9000.0, te=100.0, ti=2481.0),  # FLAIR (finite-TR CSF null)
)

# Gd shortens enhancing-tumour T1 on the post-contrast channel (-> brighter) via the
# physically-grounded SBM relaxivity model in `relaxation_constants.gd_shortened_t1`
# (1/T1_post = 1/T1_pre + r1*[Gd]); a fixed multiplier under-enhanced ET at high field.


def _bloch_signal_torch(t1: float, t2: float, pd: float, seq: PulseSequence,
                        device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Scalar Bloch magnitude for one tissue + sequence (0-d tensor).

    SE / saturation-recovery ``S = PD*(1-exp(-TR/T1))*exp(-TE/T2)``; inversion
    recovery (``ti>0``) uses ``|1 - 2 exp(-TI/T1) + exp(-TR/T1)|``. T1/T2/PD/TR/TE/TI
    enter as Python scalars (``float`` casts below), so autograd does NOT flow to them;
    the render is differentiable only w.r.t. the partial-volume fractions it mixes into.
    """
    t1t = torch.as_tensor(float(t1), device=device, dtype=dtype)
    t2t = torch.as_tensor(float(t2), device=device, dtype=dtype)
    if seq.ti and seq.ti > 0.0:
        lon = (1.0 - 2.0 * torch.exp(-float(seq.ti) / t1t) + torch.exp(-float(seq.tr) / t1t)).abs()
    else:
        lon = 1.0 - torch.exp(-float(seq.tr) / t1t)
    return float(pd) * lon * torch.exp(-float(seq.te) / t2t)


def _smooth_field(shape: Tuple[int, int, int], amplitude: float,
                  device: torch.device, dtype: torch.dtype,
                  generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """A smooth, ~zero-mean low-frequency multiplicative field ``1 + a*g`` (g ~ smooth N(0,1))."""
    d, h, w = shape
    low = torch.randn(1, 1, max(2, d // 8), max(2, h // 8), max(2, w // 8),
                      generator=generator, device="cpu").to(device=device, dtype=dtype)
    field = F.interpolate(low, size=(d, h, w), mode="trilinear", align_corners=False)[0, 0]
    return 1.0 + float(amplitude) * field


def _gaussian_blur3d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable 3D Gaussian blur of a ``(D, H, W)`` volume (low-field PSF)."""
    if sigma <= 0.0:
        return x
    radius = max(1, int(round(3.0 * sigma)))
    coords = torch.arange(-radius, radius + 1, device=x.device, dtype=torch.float32)
    kernel = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
    kernel = (kernel / kernel.sum()).to(x.dtype)
    vol = x[None, None]
    for axis in range(3):
        shape = [1, 1, 1, 1, 1]
        shape[2 + axis] = kernel.numel()
        pad = [0, 0, 0, 0, 0, 0]
        pad[(2 - axis) * 2] = radius
        pad[(2 - axis) * 2 + 1] = radius
        vol = F.conv3d(F.pad(vol, pad, mode="replicate"), kernel.view(*shape))
    return vol[0, 0]


def render_lowfield(
    pve: torch.Tensor,
    label: Optional[torch.Tensor],
    *,
    target_b0_t: float = 0.064,
    sequences: Seq[PulseSequence] = DEFAULT_BRATS_SEQUENCES,
    normalise: bool = True,
    heterogeneity: float = 0.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Render synthesized low-field MRI channels from tissue maps + label.

    Args:
        pve: ``(3, D, H, W)`` partial-volume fractions in [0,1], ordered CSF, WM, GM
            (FSL FAST). Need not sum to 1 (the residual is treated as non-brain/air).
        label: ``(D, H, W)`` integer BraTS label (0 healthy, 1 TC, 2 ED, 3 ET) or None.
        target_b0_t: target field in Tesla (default 64 mT).
        sequences: one :class:`PulseSequence` per output channel (default 4 BraTS).
        normalise: per-channel min-max to [0,1] so the output matches the network's
            expected intensity scale.
        heterogeneity: if > 0, multiply each channel by a smooth, ~zero-mean
            low-frequency field so tissue/tumour regions are not perfectly flat
            (default 0 keeps the render deterministic for the physics-consistency tests).
        generator: optional RNG for the heterogeneity field (reproducibility).

    Returns:
        ``(len(sequences), D, H, W)`` synthesized signal on ``pve``'s device/dtype.
        The label tensor is returned unchanged by the caller (never touched here).
    """
    if pve.dim() != 4 or pve.shape[0] < 3:
        raise ValueError(f"pve must be (>=3, D, H, W) CSF/WM/GM maps, got {tuple(pve.shape)}")
    if label is not None and tuple(label.shape) != tuple(pve.shape[1:]):
        raise ValueError(
            f"label shape {tuple(label.shape)} must match the pve spatial shape "
            f"{tuple(pve.shape[1:])} (co-registered to the same volume).")
    device, dtype = pve.device, pve.dtype
    f_csf, f_wm, f_gm = pve[0], pve[1], pve[2]
    b0 = float(target_b0_t)
    out = []
    for seq in sequences:
        s_csf = _bloch_signal_torch(rc.CSF.t1(b0), rc.CSF.t2(b0), rc.CSF.proton_density, seq, device, dtype)
        s_wm = _bloch_signal_torch(rc.WHITE_MATTER.t1(b0), rc.WHITE_MATTER.t2(b0),
                                   rc.WHITE_MATTER.proton_density, seq, device, dtype)
        s_gm = _bloch_signal_torch(rc.GRAY_MATTER.t1(b0), rc.GRAY_MATTER.t2(b0),
                                   rc.GRAY_MATTER.proton_density, seq, device, dtype)
        chan = f_csf * s_csf + f_wm * s_wm + f_gm * s_gm  # (D,H,W) partial-volume sum
        if label is not None:
            is_t1c = seq.name == "t1c"
            for lab, tissue in rc.TUMOUR_BY_LABEL.items():
                t1 = tissue.t1(b0)
                if is_t1c and lab == 3:        # Gd enhancement on post-contrast T1 (SBM)
                    t1 = rc.gd_shortened_t1(t1)
                s_t = _bloch_signal_torch(t1, tissue.t2(b0), tissue.proton_density, seq, device, dtype)
                chan = torch.where(label == lab, s_t.to(chan.dtype), chan)
        if heterogeneity > 0.0:
            chan = chan * _smooth_field(tuple(chan.shape), heterogeneity, device, dtype, generator)
        out.append(chan)
    sig = torch.stack(out, dim=0).to(device=device, dtype=dtype)
    if normalise:
        flat = sig.flatten(1)
        lo = flat.min(dim=1).values.view(-1, 1, 1, 1)
        hi = flat.max(dim=1).values.view(-1, 1, 1, 1)
        sig = (sig - lo) / (hi - lo).clamp_min(1e-6)
    return sig


class BlochLowFieldSimulator:
    """Callable wrapper: replace MRI channels with a synthesized low-field render.

    Designed to run in the dataset where the FSL FAST tissue priors and the BraTS
    label are both available (i.e. when ``data.use_tissue_priors=True``). It replaces
    the ``n_mri`` MRI channels of ``x`` with the physics render and leaves the label
    and any trailing tissue-prior channels untouched (channel-safe). The render is
    made realistic (within-tissue heterogeneity, low-field PSF blur, real-texture
    blend) and the skull-stripped zero background is preserved.
    """

    def __init__(self, target_b0_t: float = 0.064,
                 sequences: Seq[PulseSequence] = DEFAULT_BRATS_SEQUENCES,
                 n_mri: int = 4, prob: float = 1.0,
                 heterogeneity: float = 0.10,
                 boundary_blur_sigma: float = 0.6,
                 texture_weight: float = 0.25,
                 acquisition_jitter: float = 0.0) -> None:
        self.target_b0_t = float(target_b0_t)
        self.sequences = tuple(sequences)
        self.n_mri = int(n_mri)
        self.prob = float(prob)
        self.heterogeneity = float(heterogeneity)
        self.boundary_blur_sigma = float(boundary_blur_sigma)
        self.texture_weight = float(texture_weight)
        self.acquisition_jitter = float(acquisition_jitter)

    def _jittered_sequences(self, generator: Optional[torch.Generator]
                            ) -> Tuple[PulseSequence, ...]:
        """Per-sample TR/TE/TI randomisation within ``acquisition_jitter`` (domain rand)."""
        j = self.acquisition_jitter
        if j <= 0.0:
            return self.sequences
        out = []
        for seq in self.sequences:
            f = lambda v: float(v) * float(torch.empty(1).uniform_(1.0 - j, 1.0 + j,
                                          generator=generator).item())
            out.append(replace(seq, tr=f(seq.tr), te=f(seq.te),
                               ti=(f(seq.ti) if seq.ti > 0 else 0.0)))
        return tuple(out)

    def _target_b0(self, generator: Optional[torch.Generator]) -> float:
        if self.acquisition_jitter <= 0.0:
            return self.target_b0_t
        # ±1 octave around the target field broadens the low-field appearance.
        scale = float(torch.empty(1).uniform_(0.7, 1.4, generator=generator).item())
        return self.target_b0_t * scale

    def __call__(self, x: torch.Tensor, pve: torch.Tensor,
                 label: Optional[torch.Tensor] = None,
                 generator: Optional[torch.Generator] = None) -> torch.Tensor:
        if self.prob < 1.0:
            if float(torch.rand((), generator=generator)) >= self.prob:
                return x
        if tuple(pve.shape[1:]) != tuple(x.shape[1:]):
            raise ValueError(
                f"pve spatial shape {tuple(pve.shape[1:])} must match the MRI spatial "
                f"shape {tuple(x.shape[1:])} (FSL FAST priors co-registered to the MRI).")
        sig = render_lowfield(pve, label, target_b0_t=self._target_b0(generator),
                              sequences=self._jittered_sequences(generator), normalise=True,
                              heterogeneity=self.heterogeneity, generator=generator)
        n = min(self.n_mri, x.shape[0], sig.shape[0])
        out = x.clone()
        for c in range(n):
            s = sig[c].to(device=x.device, dtype=x.dtype)
            xc = x[c]
            # Low-field PSF: blur the render so lesion/tissue boundaries are not razor
            # sharp (real low-resolution acquisitions have MORE partial-volume blur than
            # the high-field source, never less).
            if self.boundary_blur_sigma > 0.0:
                s = _gaussian_blur3d(s, self.boundary_blur_sigma)
            # Real-texture blend: add back the high-frequency detail of the real scan
            # (sulci/gyri/vessels) that the piecewise PVE render cannot represent.
            if self.texture_weight > 0.0:
                detail = xc - _gaussian_blur3d(xc, 1.0)
                s = s + self.texture_weight * detail
            # Match the render to the input channel's FOREGROUND statistics (the cohort
            # is skull-stripped, z-scored over foreground) and force the background back
            # to zero, so the augmentation injects no train/test shift and keeps the
            # `v != 0` masks valid downstream. See the regression test.
            brain = xc != 0
            if bool(brain.any()):
                xfg, sfg = xc[brain], s[brain]
                s = (s - sfg.mean()) / sfg.std().clamp_min(1e-6) \
                    * xfg.std().clamp_min(1e-6) + xfg.mean()
                s = torch.where(brain, s, s.new_zeros(()))
            else:
                s = (s - s.mean()) / s.std().clamp_min(1e-6) \
                    * xc.std().clamp_min(1e-6) + xc.mean()
            out[c] = s
        return out
