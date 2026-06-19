"""SynthSeg-style fully-randomized contrast generator (training augmentation).

Where the Bloch simulator (:mod:`ogap.augmentations.bloch_lowfield`) renders a
*physically-grounded* low-field appearance, this module renders a **fully
contrast-randomized** appearance from the same inputs (FSL FAST tissue PVE maps +
BraTS label). Following SynthSeg (Billot et al., Med. Image Anal. 2023), each output
channel samples *random* per-region Gaussian intensities, then applies random bias,
blur (random effective resolution), and gamma. Because the contrast is re-sampled
every step, the network cannot rely on any fixed intensity mapping and learns
intensity-agnostic anatomy - the strongest known lever for cross-scanner / low-field
generalisation. It is a complementary training arm to the physics render, not a
replacement.

Contract (identical to the Bloch simulator)
-------------------------------------------
* Replaces the ``n_mri`` MRI channels of ``x`` with the synthetic render; the label
  and any trailing tissue-prior channels are untouched (channel-safe).
* Label-preserving; differentiable / device-native; training-time only.
* Skull-stripped zero background is preserved.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from . import relaxation_constants as rc
from .bloch_lowfield import _gaussian_blur3d, _smooth_field


def _u(lo: float, hi: float, g: Optional[torch.Generator]) -> float:
    return float(torch.empty(1).uniform_(lo, hi, generator=g).item())


# Region order for the randomized GMM: CSF, WM, GM (PVE) + TC(1)/ED(2)/ET(3) labels.
_TUMOUR_LABELS: Tuple[int, ...] = tuple(sorted(rc.TUMOUR_BY_LABEL.keys()))


class SynthSegGenerator:
    """Callable: replace MRI channels with a fully contrast-randomized GMM render."""

    def __init__(self, n_mri: int = 4, prob: float = 0.5,
                 intensity_std_max: float = 0.15,
                 bias_strength: float = 0.30,
                 blur_sigma_max: float = 1.5,
                 gamma_range: Tuple[float, float] = (0.7, 1.5)) -> None:
        self.n_mri = int(n_mri)
        self.prob = float(prob)
        self.intensity_std_max = float(intensity_std_max)
        self.bias_strength = float(bias_strength)
        self.blur_sigma_max = float(blur_sigma_max)
        self.gamma_range = (float(gamma_range[0]), float(gamma_range[1]))

    def _render_channel(self, pve: torch.Tensor, label: Optional[torch.Tensor],
                        device: torch.device, dtype: torch.dtype,
                        g: Optional[torch.Generator]) -> torch.Tensor:
        f_csf, f_wm, f_gm = pve[0], pve[1], pve[2]
        # Random per-region means (μ) and stds (σ).
        mus = {t: _u(0.0, 1.0, g) for t in ("csf", "wm", "gm")}
        sigs = {t: _u(0.0, self.intensity_std_max, g) for t in ("csf", "wm", "gm")}
        mean_img = f_csf * mus["csf"] + f_wm * mus["wm"] + f_gm * mus["gm"]
        std_img = f_csf * sigs["csf"] + f_wm * sigs["wm"] + f_gm * sigs["gm"]
        if label is not None:
            for lab in _TUMOUR_LABELS:
                m = label == lab
                mean_img = torch.where(m, mean_img.new_full((), _u(0.0, 1.0, g)), mean_img)
                std_img = torch.where(m, std_img.new_full((), _u(0.0, self.intensity_std_max, g)), std_img)
        # GMM sample, then random multiplicative bias, blur (random resolution), gamma.
        noise = torch.randn(mean_img.shape, generator=g, device="cpu").to(device=device, dtype=dtype)
        img = mean_img + std_img * noise
        if self.bias_strength > 0.0:
            img = img * _smooth_field(tuple(img.shape), self.bias_strength, device, dtype, g)
        sigma = _u(0.0, self.blur_sigma_max, g)
        if sigma > 0.05:
            img = _gaussian_blur3d(img, sigma)
        lo = img.min()
        rng = (img.max() - lo).clamp_min(1e-6)
        img = ((img - lo) / rng).clamp(0, 1).pow(_u(*self.gamma_range, g))
        return img

    def __call__(self, x: torch.Tensor, pve: torch.Tensor,
                 label: Optional[torch.Tensor] = None,
                 generator: Optional[torch.Generator] = None) -> torch.Tensor:
        if self.prob < 1.0 and float(torch.rand((), generator=generator)) >= self.prob:
            return x
        if tuple(pve.shape[1:]) != tuple(x.shape[1:]):
            raise ValueError(
                f"pve spatial shape {tuple(pve.shape[1:])} must match the MRI spatial "
                f"shape {tuple(x.shape[1:])} (FSL FAST priors co-registered to the MRI).")
        device, dtype = x.device, x.dtype
        n = min(self.n_mri, x.shape[0])
        out = x.clone()
        for c in range(n):
            s = self._render_channel(pve, label, device, dtype, generator)
            xc = x[c]
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


def build_synth_generator(config) -> Optional["SynthSegGenerator"]:
    """Build a :class:`SynthSegGenerator` from ``augmentation.physics.*``, or None."""
    try:
        phys = config.augmentation.physics
        if not getattr(phys, "synth_randomization", False):
            return None
    except AttributeError:
        return None

    def _get(name, default):
        # Only substitute the default when the key is ABSENT or explicitly null.
        # A `... or default` short-circuit would silently turn a deliberate 0.0
        # ablation (e.g. synth_prob: 0.0, synth_bias_strength: 0.0) back into the
        # default, so the ablated run would not reflect the described condition.
        v = getattr(phys, name, default)
        return default if v is None else v

    data = getattr(config, "data", None)
    n_mri = int(getattr(data, "n_mri_channels", 0) or getattr(data, "in_channels", 4) or 4)
    return SynthSegGenerator(
        n_mri=n_mri,
        prob=float(_get("synth_prob", 0.5)),
        intensity_std_max=float(_get("synth_intensity_std_max", 0.15)),
        bias_strength=float(_get("synth_bias_strength", 0.30)),
        blur_sigma_max=float(_get("synth_blur_sigma_max", 1.5)),
    )
