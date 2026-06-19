"""Physics-informed augmentations emulating real MRI acquisition variation.

Each transform models a physically-motivated source of inter-scanner / inter-
site variation (B1 bias field, intensity gain, gamma, contrast-distribution
shift, Rician noise, thick-slice partial-volume effects, k-space artifacts, FOV
wrap-around aliasing, and Fourier-basis AFA). They operate on a float tensor of
shape ``(C, D, H, W)`` -
one channel per MRI modality - and return the same shape.

GPU residency
-------------
Every transform is wrapped by :func:`_batched`, so the *same* function also
accepts a batched ``(B, C, D, H, W)`` tensor that already lives on the GPU. This
is what lets the whole physics pipeline run on the H100 inside the CUDA
prefetcher (:class:`GPUBatchedPhysicsAugmentor`) instead of on the CPU inside a
DataLoader worker - removing the single biggest one-GPU throughput bottleneck.
The 4-D (single-sample) path is byte-for-byte unchanged, so the CPU dataset path
and every existing test behave identically; only a 5-D input takes the new
batched route. Heavy ops (FFTs, 3-D interpolation, polynomial fields) execute on
whichever device the tensor is on.

:func:`build_physics_augmentations` returns a single callable applying the
enabled transforms in sequence (identity if disabled), so it drops into any
per-sample transform pipeline. Apply after intensity normalisation/cropping but
before geometric augmentation.
"""
from __future__ import annotations

import functools
import logging
import math
from typing import Any, Callable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _batched(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """Let a per-sample ``(C, D, H, W)`` transform also accept a GPU batch.

    The v9.1 physics pipeline was written for a single ``(C, D, H, W)`` sample so
    it could run inside a CPU DataLoader worker. To move augmentation onto the
    H100 (the biggest one-GPU DataLoader bottleneck), the same transforms must
    accept a batched ``(B, C, D, H, W)`` tensor *resident on the device*. This
    wrapper makes both work with **zero change to the 4-D path**: a 4-D input
    flows straight through the original implementation (byte-for-byte identical,
    so every existing test and the CPU dataset path are unaffected), while a 5-D
    input is split along the batch dim and each sample is processed on its own
    device-resident tensor (independent per-sample random parameters, matching
    how the CPU path drew them one sample at a time).
    """

    @functools.wraps(fn)
    def wrapper(x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if x.dim() == 5:
            if x.shape[0] == 0:
                return x  # empty batch: nothing to stack, return as-is
            return torch.stack(
                [fn(x[b], *args, **kwargs) for b in range(x.shape[0])], dim=0
            )
        return fn(x, *args, **kwargs)

    return wrapper


def _cfg_get(config: Any, path: Sequence[str], default: Any) -> Any:
    node = config
    for key in path:
        if node is None:
            return default
        node = node.get(key, None) if isinstance(node, dict) else getattr(node, key, None)
    return default if node is None else node


def _cpu_generator(g: Optional[torch.Generator]) -> Optional[torch.Generator]:
    if g is None:
        return None
    try:
        return g if torch.device(getattr(g, "device", "cpu")).type == "cpu" else None
    except (TypeError, RuntimeError):
        return None


def _u(lo: float, hi: float, g: Optional[torch.Generator]) -> float:
    return float(torch.empty(1).uniform_(lo, hi, generator=_cpu_generator(g)).item())


def _fft3(x: torch.Tensor) -> torch.Tensor:
    # Inputs are real-valued MRI volumes. rfftn keeps only the non-redundant
    # half-spectrum, cutting k-space memory and compute on the augmentation path.
    return torch.fft.rfftn(x.float(), dim=(-3, -2, -1))


def _ifft_abs(kspace: torch.Tensor, dtype: torch.dtype, shape: Sequence[int]) -> torch.Tensor:
    return torch.fft.irfftn(kspace, s=tuple(int(v) for v in shape), dim=(-3, -2, -1)).abs().to(dtype=dtype)


def _compatible_generator(ref: torch.Tensor, g: Optional[torch.Generator]) -> Optional[torch.Generator]:
    if g is None:
        return None
    try:
        g_device = torch.device(getattr(g, "device", "cpu"))
    except (TypeError, RuntimeError):
        g_device = torch.device("cpu")
    return g if g_device.type == ref.device.type else None


def _haar_lll_hhh(volume: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """One-level 3D Haar LLL and HHH subbands.

    The robust Rician estimator paper uses object voxels in the highest HHH
    wavelet subband for MAD noise estimation. Implementing the first Haar level
    directly keeps the dependency footprint small and avoids pywt availability
    issues on the cluster.
    """
    d, h, w = (int(s) - int(s) % 2 for s in volume.shape[-3:])
    v = volume[:d, :h, :w].float()
    a = v[0::2, 0::2, 0::2]
    b = v[0::2, 0::2, 1::2]
    c = v[0::2, 1::2, 0::2]
    d0 = v[0::2, 1::2, 1::2]
    e = v[1::2, 0::2, 0::2]
    f = v[1::2, 0::2, 1::2]
    g0 = v[1::2, 1::2, 0::2]
    h0 = v[1::2, 1::2, 1::2]
    scale = math.sqrt(8.0)
    lll = (a + b + c + d0 + e + f + g0 + h0) / scale
    hhh = (a - b - c + d0 - e + f + g0 - h0) / scale
    return lll, hhh


def estimate_rician_sigma_mad_hhh(x: torch.Tensor) -> float:
    """Robust object-based Rician sigma estimate from the HHH wavelet subband.

    This follows the robust Rician-noise paper's estimator family (Coupé et al.,
    Med. Image Anal. 2010): segment the object on the low-frequency LLL subband,
    take only matching coefficients in the highest HHH subband, and estimate
    magnitude noise with MAD. A small Koay-style low-SNR correction is applied
    from the object mean/SNR; at high SNR it approaches the Gaussian MAD estimate.
    """
    if x.numel() < 64 or torch.max(x) <= torch.min(x):
        return 1e-6
    lll, hhh = _haar_lll_hhh(x.detach())
    if lll.numel() == 0:
        return 1e-6
    threshold = torch.quantile(lll.flatten(), 0.35)
    obj = lll > threshold
    if int(obj.sum()) < 32:
        obj = lll > lll.mean()
    if int(obj.sum()) < 32:
        obj = torch.ones_like(lll, dtype=torch.bool)

    # Remove high-gradient object voxels, as described in the paper, so edges do
    # not inflate the HHH noise estimate at low noise levels.
    dz = torch.zeros_like(lll)
    dy = torch.zeros_like(lll)
    dx = torch.zeros_like(lll)
    dz[1:] = (lll[1:] - lll[:-1]).abs()
    dy[:, 1:] = (lll[:, 1:] - lll[:, :-1]).abs()
    dx[:, :, 1:] = (lll[:, :, 1:] - lll[:, :, :-1]).abs()
    grad = dz + dy + dx
    keep = obj & (grad <= torch.median(grad[obj]))
    coeff = hhh[keep]
    if coeff.numel() < 16:
        coeff = hhh[obj]
    if coeff.numel() == 0:
        return 1e-6

    sigma_mag = torch.median(coeff.abs()).item() / 0.6745
    object_mean = float(torch.mean(lll[obj]).abs().item())
    if not math.isfinite(sigma_mag) or sigma_mag <= 0:
        return 1e-6
    snr = object_mean / max(sigma_mag, 1e-6)
    # Low-SNR Rician magnitude noise is wider than the underlying complex
    # Gaussian sigma. This bounded correction approximates the Koay iterative
    # correction without depending on scipy.special in the light augmentor.
    correction = 1.0 if snr >= 3.0 else math.sqrt(max(0.25, (2.0 + snr * snr) / (4.0 + snr * snr)))
    return float(max(sigma_mag * correction, 1e-6))


def rician_added_sigma(signal_mean_abs: float, snr: float,
                       sigma_observed: float) -> float:
    """Gaussian sigma to ADD (in quadrature) to reach a target SNR. [audit S2-A]

    The input is already a magnitude (already-Rician) volume carrying noise of
    level ``sigma_observed``. To reach a *total* noise of
    ``sigma_target = signal_mean_abs / snr`` the added complex-Gaussian sigma must
    satisfy ``sigma_add^2 + sigma_observed^2 = sigma_target^2`` (variances add).
    Returns ``sqrt(max(0, sigma_target^2 - sigma_observed^2))`` - clamped at 0
    because adding noise can only lower SNR, never raise it. Using
    ``max(sigma_target, sigma_observed)`` (the previous behaviour) over-injects on
    already-noisy low-field inputs.
    """
    sigma_target = float(signal_mean_abs) / max(float(snr), 1e-3)
    return math.sqrt(max(0.0, sigma_target * sigma_target - float(sigma_observed) ** 2))


# Physical magnitude bound on the multiplicative log-bias. exp(±0.5) -> a
# 0.61x-1.65x B1 gain envelope, the realistic span for 0.064-3 T head coils;
# clamping keeps the polynomial field from producing non-physical extremes.
_BIAS_LOG_CLAMP = 0.5


@_batched
def random_bias_field(x: torch.Tensor, g: Optional[torch.Generator] = None,
                      coeff_range: float = 0.1, degree: int = 3) -> torch.Tensor:
    """Multiplicative low-order polynomial B1 inhomogeneity, per channel.

    The field is treated as the **log** of the multiplicative bias and applied as
    ``x * exp(field)`` - the N4ITK / N3 model (Tustison et al. 2010), where the
    observed signal equals the true signal times ``exp`` of a smooth field. This
    replaces the earlier ``x * (1 + field)`` form, which (i) is only the
    first-order Taylor term of the true exponential and (ii) could drive the
    multiplier negative for large fields. The log-field is clamped to
    ``±_BIAS_LOG_CLAMP`` so the gain stays in a physical envelope. [F7]
    """
    c, d, h, w = x.shape
    work_dtype = torch.float32 if not x.dtype.is_floating_point else x.dtype
    gd, gh, gw = torch.meshgrid(
        torch.linspace(-1, 1, d, device=x.device, dtype=work_dtype),
        torch.linspace(-1, 1, h, device=x.device, dtype=work_dtype),
        torch.linspace(-1, 1, w, device=x.device, dtype=work_dtype),
        indexing="ij",
    )
    out = x.clone()
    for ch in range(c):
        field = torch.zeros(d, h, w, device=x.device, dtype=work_dtype)
        for i in range(degree + 1):
            for j in range(degree + 1 - i):
                for k in range(degree + 1 - i - j):
                    field = field + _u(-coeff_range, coeff_range, g) * (gd ** i) * (gh ** j) * (gw ** k)
        # N4/N3 exponential bias model (Tustison 2010): multiply by exp(log-field).
        out[ch] = x[ch] * torch.exp(field.clamp(-_BIAS_LOG_CLAMP, _BIAS_LOG_CLAMP))
    return out


@_batched
def random_intensity_shift(x: torch.Tensor, g: Optional[torch.Generator] = None,
                           scale_range: Tuple[float, float] = (0.9, 1.1),
                           shift_range: Tuple[float, float] = (-0.1, 0.1)) -> torch.Tensor:
    out = x.clone()
    for ch in range(x.shape[0]):
        rng = (x[ch].max() - x[ch].min()).clamp_min(1e-6)
        shifted = _u(*scale_range, g) * x[ch] + _u(*shift_range, g) * rng
        # preserve the skull-stripped zero background (the x==0 deployment invariant)
        out[ch] = torch.where(x[ch] != 0, shifted, x[ch])
    return out


@_batched
def random_gamma(x: torch.Tensor, g: Optional[torch.Generator] = None,
                 gamma_range: Tuple[float, float] = (0.7, 1.5)) -> torch.Tensor:
    out = x.clone()
    for ch in range(x.shape[0]):
        lo = x[ch].min()
        rng = (x[ch].max() - lo).clamp_min(1e-6)
        mapped = ((x[ch] - lo) / rng).clamp(0, 1).pow(_u(*gamma_range, g)) * rng + lo
        # preserve the skull-stripped zero background (the x==0 deployment invariant)
        out[ch] = torch.where(x[ch] != 0, mapped, x[ch])
    return out


@_batched
def gmm_contrast(x: torch.Tensor, g: Optional[torch.Generator] = None,
                 k: int = 3, mean_jitter: float = 0.15) -> torch.Tensor:
    """Strictly-monotone piecewise-linear intensity remap (GMM-contrast surrogate).

    The histogram between the lowest and highest foreground quantile is re-stretched
    by sampling a positive per-bin width scale ``exp(U(-jitter, jitter))`` and rebuilding
    monotone knot positions from the cumulative widths (rescaled to keep the endpoints
    fixed). Because the new knots are the cumulative sum of POSITIVE widths, the map is
    strictly increasing - no intensity-ordering inversion at bin edges (the previous
    per-bin affine ``lo + (v-lo)*(1+jitter)`` could map the top of one bin above the
    bottom of the next, contradicting "monotone"). Background (exact 0) is preserved.
    """
    out = x.clone()
    for ch in range(x.shape[0]):
        v = x[ch]
        fg = v != 0
        nz = v[fg]
        if nz.numel() < k + 1:
            continue
        edges = torch.quantile(nz.float(), torch.linspace(0, 1, k + 1, device=v.device))
        widths = (edges[1:] - edges[:-1]).clamp_min(1e-8)
        scales = torch.tensor([math.exp(_u(-mean_jitter, mean_jitter, g)) for _ in range(k)],
                              device=v.device, dtype=widths.dtype)
        new_widths = widths * scales
        new_widths = new_widths * (widths.sum() / new_widths.sum().clamp_min(1e-8))
        new_edges = torch.cat([edges[:1], edges[:1] + torch.cumsum(new_widths, 0)])
        vv = nz.float().clamp(float(edges[0]), float(edges[-1]))
        idx = (torch.searchsorted(edges, vv, right=True).clamp(1, k) - 1)
        lo, hi = edges[idx], edges[idx + 1]
        nlo, nhi = new_edges[idx], new_edges[idx + 1]
        frac = (vv - lo) / (hi - lo).clamp_min(1e-8)
        mapped = nlo + frac * (nhi - nlo)
        new = v.clone()
        new[fg] = mapped.to(v.dtype)
        out[ch] = new
    return out


@_batched
def rician_noise(x: torch.Tensor, g: Optional[torch.Generator] = None,
                 snr_range: Tuple[float, float] = (5.0, 30.0)) -> torch.Tensor:
    """Rician noise at a target SNR (low end emulates low-field MRI)."""
    out = x.clone()
    for ch in range(x.shape[0]):
        v = x[ch]
        mask = v != 0
        if mask.sum() == 0:
            continue
        snr = max(_u(*snr_range, g), 1e-3)
        # Signal level for the SNR target is the mean MAGNITUDE of the foreground,
        # not |mean(foreground)|. The pipeline z-scores before augmenting, so the
        # foreground mean is ~0 and |mean| collapses the target sigma to ~0 - i.e.
        # rician_noise would inject essentially no noise on the real (z-scored)
        # inputs. abs().mean() is the magnitude scale (~0.8 for z-scored data) and
        # is identical to mean() for raw non-negative magnitude images.
        signal_mean_abs = float(v[mask].abs().mean().item())
        sigma_observed = estimate_rician_sigma_mad_hhh(v)
        # Add noise in quadrature to reach the target SNR (not the max() of the
        # target and existing sigmas, which over-injects on noisy inputs). [S2-A]
        sigma = rician_added_sigma(signal_mean_abs, snr, sigma_observed)
        if sigma <= 0.0:
            continue  # already at/below the target SNR; adding noise can't raise it
        rand_gen = _compatible_generator(v, g)
        n1 = torch.randn(v.shape, device=v.device, dtype=v.dtype, generator=rand_gen) * sigma
        # The Rician magnitude form sqrt((v+n1)^2 + n2^2) is only physical on a NON-NEGATIVE
        # magnitude image. The pipeline z-scores before augmenting, so the foreground is
        # signed (~half negative); applying the magnitude operator there RECTIFIES the
        # signal (folds the distribution) instead of adding noise. On signed/z-scored input
        # we therefore add Gaussian noise (Rician ~= Gaussian at SNR>3, which holds here)
        # and reserve the true Rician magnitude form for genuinely magnitude-valued (>=0)
        # volumes.
        if float(v[mask].min().item()) < 0.0:
            noisy = v + n1
        else:
            n2 = torch.randn(v.shape, device=v.device, dtype=v.dtype, generator=rand_gen) * sigma
            noisy = torch.sqrt((v + n1) ** 2 + n2 ** 2)
        # Foreground only; keep the skull-stripped zero background at exactly 0 so the
        # downstream `x != 0` brain mask / availability map invariant (and the Bloch
        # zero-background contract) stay valid.
        out[ch] = torch.where(mask, noisy, v)
    return out


@_batched
def resolution_jitter(x: torch.Tensor, g: Optional[torch.Generator] = None,
                      downsample_range: Tuple[float, float] = (0.5, 1.0),
                      per_axis_prob: float = 0.5) -> torch.Tensor:
    """Down/up-sample to emulate thick-slice partial-volume blur."""
    c, d, h, w = x.shape
    factors = [_u(*downsample_range, g) if _u(0, 1, g) < per_axis_prob else 1.0 for _ in range(3)]
    new = [max(1, int(round(n * f))) for n, f in zip((d, h, w), factors)]
    if new == [d, h, w]:
        new[0] = max(1, int(round(d * _u(*downsample_range, g))))
    down = F.interpolate(x.unsqueeze(0), size=new, mode="trilinear", align_corners=False)
    # Up-sample with trilinear (smooth partial-volume averaging), NOT nearest: a real
    # thick-slice / low-resolution acquisition reconstructs to a smoothly-interpolated
    # volume, whereas nearest leaves blocky voxel replication that no scanner produces.
    return F.interpolate(down, size=(d, h, w), mode="trilinear", align_corners=False).squeeze(0)


@_batched
def kspace_partial_fourier(x: torch.Tensor, g: Optional[torch.Generator] = None,
                           keep_range: Tuple[float, float] = (0.62, 0.88)) -> torch.Tensor:
    """Asymmetric k-space truncation / subsampling proxy."""
    out = x.clone()
    dtype = x.dtype
    for ch in range(x.shape[0]):
        kspace = _fft3(out[ch])
        axis = int(torch.randint(0, 3, (1,), generator=_cpu_generator(g)).item())
        n = kspace.shape[axis]
        keep = max(2, min(n, int(round(n * _u(*keep_range, g)))))
        start = 0 if _u(0.0, 1.0, g) < 0.5 else n - keep
        mask = torch.zeros(n, dtype=torch.bool, device=kspace.device)
        mask[start:start + keep] = True
        shape = [1, 1, 1]
        shape[axis] = n
        kspace = torch.where(mask.view(*shape), kspace, torch.zeros_like(kspace))
        out[ch] = _ifft_abs(kspace, dtype, out[ch].shape)
    return out


@_batched
def kspace_ghosting(x: torch.Tensor, g: Optional[torch.Generator] = None,
                    line_drop_range: Tuple[float, float] = (0.15, 0.45)) -> torch.Tensor:
    """Periodic phase-encode line attenuation causing ghost replicas."""
    out = x.clone()
    dtype = x.dtype
    for ch in range(x.shape[0]):
        kspace = _fft3(out[ch])
        axis = int(torch.randint(0, 3, (1,), generator=_cpu_generator(g)).item())
        n = kspace.shape[axis]
        period = int(torch.randint(2, 6, (1,), generator=_cpu_generator(g)).item())
        offset = int(torch.randint(0, period, (1,), generator=_cpu_generator(g)).item())
        attenuation = _u(*line_drop_range, g)
        factors = torch.ones(n, device=kspace.device, dtype=kspace.real.dtype)
        factors[offset::period] = attenuation
        shape = [1, 1, 1]
        shape[axis] = n
        kspace = kspace * factors.view(*shape)
        out[ch] = _ifft_abs(kspace, dtype, out[ch].shape)
    return out


@_batched
def kspace_motion(x: torch.Tensor, g: Optional[torch.Generator] = None,
                  max_phase: float = math.pi) -> torch.Tensor:
    """Segment-wise random k-space phase shifts, a simple motion artifact model."""
    out = x.clone()
    dtype = x.dtype
    for ch in range(x.shape[0]):
        kspace = _fft3(out[ch])
        axis = int(torch.randint(0, 3, (1,), generator=_cpu_generator(g)).item())
        n = kspace.shape[axis]
        n_segments = int(torch.randint(3, 8, (1,), generator=_cpu_generator(g)).item())
        seg = max(1, n // n_segments)
        for start in range(0, n, seg):
            end = min(start + seg, n)
            phase = _u(-max_phase, max_phase, g)
            factor = torch.complex(
                torch.tensor(math.cos(phase), device=kspace.device, dtype=kspace.real.dtype),
                torch.tensor(math.sin(phase), device=kspace.device, dtype=kspace.real.dtype),
            )
            slc = [slice(None), slice(None), slice(None)]
            slc[axis] = slice(start, end)
            kspace[tuple(slc)] = kspace[tuple(slc)] * factor
        out[ch] = _ifft_abs(kspace, dtype, out[ch].shape)
    return out


@_batched
def wraparound_aliasing(x: torch.Tensor, g: Optional[torch.Generator] = None,
                        fold_fraction_range: Tuple[float, float] = (0.05, 0.20),
                        attenuation_range: Tuple[float, float] = (0.10, 0.50)) -> torch.Tensor:
    """FOV wrap-around (aliasing) artifact - a dominant low-field brain artifact.

    When the field-of-view is smaller than the head along the phase-encode
    direction, anatomy beyond one FOV edge folds back onto the opposite edge.
    Jimeno et al. 2022 ("ArtifactID") identify wrap-around and Gibbs ringing as
    the two characteristic artifacts of low-field brain MRI; Gibbs is already
    produced by :func:`kspace_partial_fourier` (k-space truncation), so this adds
    the missing wrap-around. Modelled as a faint, attenuated copy of an edge slab
    added to the opposite edge: a pure intensity overlay that leaves the true
    anatomy - and therefore the segmentation labels - unchanged. No FFT and no
    even-dimension requirement, so it runs on any device and any spatial size.
    """
    _, d, h, w = x.shape
    out = x.clone()
    spatial = (d, h, w)
    axis = int(torch.randint(0, 3, (1,), generator=_cpu_generator(g)).item())
    dim = axis + 1  # dim 0 is the channel axis
    n = spatial[axis]
    if n < 2:
        return out  # degenerate axis: nothing to fold
    fold = max(1, min(n - 1, int(round(n * _u(*fold_fraction_range, g)))))
    atten = _u(*attenuation_range, g)
    if _u(0.0, 1.0, g) < 0.5:
        # near-edge slab aliases onto the far edge (and vice versa)
        out.narrow(dim, n - fold, fold).add_(atten * x.narrow(dim, 0, fold))
    else:
        out.narrow(dim, 0, fold).add_(atten * x.narrow(dim, n - fold, fold))
    return out


@_batched
def auxiliary_fourier_basis(x: torch.Tensor, g: Optional[torch.Generator] = None,
                            mean_strength: float = 0.06) -> torch.Tensor:
    """Auxiliary Fourier-basis augmentation (AFA).

    AFA adds a sampled sinusoidal Fourier-basis function per channel, with the
    magnitude sampled from an exponential distribution. This is not donor
    amplitude swapping; it is the additive Fourier-basis perturbation described
    by Vaish et al. and by the MRI OOD-generalisation paper.
    """
    c, d, h, w = x.shape
    zz, yy, xx = torch.meshgrid(
        torch.linspace(0, 1, d, device=x.device),
        torch.linspace(0, 1, h, device=x.device),
        torch.linspace(0, 1, w, device=x.device),
        indexing="ij",
    )
    out = x.clone()
    max_freq = max(1, min(d, h, w) // 2)
    for ch in range(c):
        freq = _u(1.0, float(max_freq), g)
        theta = _u(0.0, math.pi, g)
        phi = _u(0.0, math.pi, g)
        direction = (
            math.cos(theta),
            math.sin(theta) * math.cos(phi),
            math.sin(theta) * math.sin(phi),
        )
        basis = torch.sin(2.0 * math.pi * freq * (direction[0] * zz + direction[1] * yy + direction[2] * xx - 0.25))
        # Normalise to UNIT PEAK (max-abs), not unit L2 norm. An L2 norm over all
        # voxels scales the basis by ~1/sqrt(N), so for a 128^3 volume the added
        # perturbation was ~1000x too small - the AFA arm was effectively a no-op.
        # Unit-peak keeps the perturbation magnitude controlled by `sigma` (the
        # sampled exponential strength) and the channel range, as Vaish et al. intend.
        basis = basis / basis.abs().max().clamp_min(1e-6)
        sigma = float(
            torch.empty(1, device=x.device)
            .exponential_(1.0 / max(mean_strength, 1e-6), generator=_compatible_generator(x, g))
            .item()
        )
        rng = (out[ch].max() - out[ch].min()).clamp_min(1e-6)
        pert = out[ch] + sigma * rng * basis
        # preserve the skull-stripped zero background (the x==0 deployment invariant)
        out[ch] = torch.where(x[ch] != 0, pert, x[ch])
    return out


_TIER1 = {
    "bias_field": random_bias_field,
    "intensity_shift": random_intensity_shift,
    "gamma_correction": random_gamma,
    "gmm_contrast": gmm_contrast,
    "rician_noise": rician_noise,
    "resolution_jitter": resolution_jitter,
    "kspace_partial_fourier": kspace_partial_fourier,
    "kspace_ghosting": kspace_ghosting,
    "kspace_motion": kspace_motion,
    "wraparound_aliasing": wraparound_aliasing,
    "auxiliary_fourier_basis": auxiliary_fourier_basis,
}


class PhysicsAugmentor:
    """Composed, individually-toggleable physics augmentation callable.

    Accepts either a single ``(C, D, H, W)`` sample (CPU dataset path) or a
    batched ``(B, C, D, H, W)`` tensor on any device (GPU prefetcher path) - the
    underlying transforms are :func:`_batched`, so both work transparently.
    """

    def __init__(self, config: Any) -> None:
        p = ("augmentation", "physics")
        self.enabled = bool(_cfg_get(config, p + ("enabled",), False))
        self.snr_range = tuple(_cfg_get(config, p + ("rician_snr_range",), (5.0, 30.0)))
        self.down_range = tuple(_cfg_get(config, p + ("resolution_downsample_range",), (0.5, 1.0)))
        self.afa_strength = float(_cfg_get(config, p + ("afa_mean_strength",), 0.06))
        self.stacked_n = int(_cfg_get(config, p + ("stacked_n",), 0))
        self.active: List[str] = [n for n in _TIER1 if bool(_cfg_get(config, p + (n,), False))]
        # Number of leading channels that are real acquired MRI signal. Any extra
        # trailing channels (e.g. FSL FAST CSF/WM/GM partial-volume priors appended
        # when data.use_tissue_priors=True) are anatomical probability maps, NOT
        # signal, and must be shielded from every intensity/noise/k-space transform.
        # None disables the shield (legacy behaviour); the default (data.in_channels)
        # makes a tissue-prior-free 4-channel input a no-op slice (byte-identical).
        # Prefer the explicit MRI modality count; fall back to in_channels for configs
        # that predate it. Using in_channels alone would be WRONG when use_tissue_priors
        # inflates it (4 MRI + 3 priors = 7), which would leave the shield covering ZERO
        # channels and silently corrupt the tissue priors. See ogap.config.loader.
        _n_mri = int(_cfg_get(config, ("data", "n_mri_channels"), 0))
        _in = int(_cfg_get(config, ("data", "in_channels"), 0))
        _eff = _n_mri if _n_mri > 0 else _in
        self.n_mri_channels: Optional[int] = _eff if _eff > 0 else None

    def has_work(self) -> bool:
        """True if the augmentor would actually transform an input."""
        return self.enabled and (bool(self.active) or self.stacked_n > 0)

    def _apply(self, name: str, x: torch.Tensor, g: Optional[torch.Generator]) -> torch.Tensor:
        if name == "rician_noise":
            return rician_noise(x, g, self.snr_range)
        if name == "resolution_jitter":
            return resolution_jitter(x, g, self.down_range)
        if name == "auxiliary_fourier_basis":
            return auxiliary_fourier_basis(x, g, self.afa_strength)
        return _TIER1[name](x, g)

    def __call__(self, x: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        if not self.enabled:
            return x
        n = self.n_mri_channels
        if n is not None and x.shape[-4] > n:
            # Shield trailing non-MRI channels (tissue-prior probability maps) from
            # intensity/noise/k-space transforms; they carry anatomy, not signal.
            # The channel axis is -4 for both (C,D,H,W) and (B,C,D,H,W), so this is
            # correct on the CPU per-sample path and the GPU batched path alike.
            mri = self._run(x[..., :n, :, :, :].contiguous(), generator)
            extra = x[..., n:, :, :, :]
            return torch.cat([mri, extra], dim=-4)
        return self._run(x, generator)

    def _run(self, x: torch.Tensor, generator: Optional[torch.Generator]) -> torch.Tensor:
        names = list(self.active)
        if self.stacked_n > 0:
            pool = list(_TIER1)
            idx = torch.randperm(len(pool), generator=_cpu_generator(generator))[: self.stacked_n]
            names = [pool[i] for i in idx.tolist()]
        for name in names:
            x = self._apply(name, x, generator)
        return x


class GPUBatchedPhysicsAugmentor:
    """On-device, batched v9.1 physics augmentation for the CUDA prefetcher.

    Wraps :class:`PhysicsAugmentor` with the ``(x, y) -> (x, y)`` signature the
    training prefetcher expects, so the entire physics pipeline runs on the GPU
    over a whole ``(B, C, D, H, W)`` batch instead of per-sample on the CPU. The
    label volume ``y`` is returned **unchanged**: every v9.1 transform is an
    intensity / acquisition operation (bias, gamma, Rician, k-space, AFA, …) with
    no geometric component, so the segmentation labels are unaffected - which is
    exactly why the v9.1 engine is safe to run after host->device copy without
    re-warping the mask.

    This is the GPU counterpart of the CPU dataset path: when it is active the
    dataset must pass ``physics_aug=None`` so the transforms are applied once, on
    the device, not twice (the CPU/GPU double-application this resolves is finding
    F1).

    Channel-aware geometric augmentation (flip / affine / elastic) runs FIRST, on
    **all** channels + the label, so MRI, appended tissue priors, and the mask stay
    co-registered; the intensity/noise/k-space physics then runs on the MRI channels
    only (the ``n_mri`` shield). This is the spec-§10 split that restores the spatial
    augmentation the legacy augmentor used to provide.
    """

    def __init__(self, config: Any, generator: Optional[torch.Generator] = None,
                 geometric: Optional[Any] = None) -> None:
        self.aug = PhysicsAugmentor(config)
        self.generator = generator
        self.geometric = geometric

    @property
    def enabled(self) -> bool:
        return self.aug.has_work() or (self.geometric is not None
                                       and self.geometric.has_work())

    def __call__(self, x: torch.Tensor, y: Optional[torch.Tensor] = None
                 ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not self.enabled:
            return x, y
        with torch.no_grad():
            if self.geometric is not None and self.geometric.has_work():
                x, y = self.geometric(x, y, self.generator)
            if self.aug.has_work():
                x = self.aug(x, self.generator)
        return x, y


def build_physics_augmentations(config: Any) -> Callable[..., torch.Tensor]:
    """Return the composed augmentation callable (identity if disabled)."""
    aug = PhysicsAugmentor(config)
    if not aug.has_work():
        return lambda x, generator=None: x
    return aug


def build_gpu_physics_augmentor(config: Any,
                                generator: Optional[torch.Generator] = None
                                ) -> Optional["GPUBatchedPhysicsAugmentor"]:
    """Return a GPU batched augmentor, or ``None`` when nothing is enabled.

    Used by the training loop to run v9.1 physics augmentation inside the CUDA
    prefetcher. ``None`` signals "no v9.1 GPU augmentation", so the caller leaves
    the legacy GPU augmentor / CPU path untouched. The augmentor also carries the
    channel-aware geometric augmentor (flip / affine / elastic on all channels +
    label), built from ``augmentation.geometric.*``; it runs before the intensity
    shield.
    """
    from .geometric import build_geometric_augmentor
    geometric = build_geometric_augmentor(config)
    gpu_aug = GPUBatchedPhysicsAugmentor(config, generator, geometric=geometric)
    return gpu_aug if gpu_aug.enabled else None
