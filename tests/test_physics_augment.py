import torch
from types import SimpleNamespace

from ogap.augmentations import build_physics_augmentations
from ogap.augmentations.physics_augment import (
    random_bias_field, random_intensity_shift, random_gamma,
    gmm_contrast, rician_noise, resolution_jitter,
    kspace_partial_fourier, kspace_ghosting, kspace_motion,
    auxiliary_fourier_basis, estimate_rician_sigma_mad_hhh,
)

_FNS = [random_bias_field, random_intensity_shift, random_gamma,
        gmm_contrast, rician_noise, resolution_jitter,
        kspace_partial_fourier, kspace_ghosting, kspace_motion,
        auxiliary_fourier_basis]


def _x():
    g = torch.Generator().manual_seed(0)
    return torch.rand(4, 16, 16, 16, generator=g).abs() + 0.1


def test_each_tier1_shape_finite_changed():
    x = _x()
    for fn in _FNS:
        g = torch.Generator().manual_seed(1)
        y = fn(x, g)
        assert y.shape == x.shape, fn.__name__
        assert torch.isfinite(y).all(), fn.__name__
        if fn is rician_noise:
            # With quadrature SNR targeting [S2-A], rician_noise correctly adds
            # ~0 noise when the input is already noisier than the target SNR
            # (as here: uniform-random input). Shape/finite only for this fn;
            # the noise-adding case is covered by the dedicated test below.
            continue
        assert not torch.allclose(y, x), fn.__name__


def test_rician_noise_changes_clean_volume():
    """On a clean (low intrinsic noise) volume below the target SNR, rician_noise
    does add noise and change the input. [audit S2-A]"""
    g = torch.Generator().manual_seed(2)
    base = (torch.linspace(1.0, 5.0, 8).view(1, 8, 1, 1)
            .expand(1, 8, 8, 8).contiguous())
    y = rician_noise(base, g, (3.0, 3.0))     # demanding SNR vs a smooth volume
    assert y.shape == base.shape and torch.isfinite(y).all()
    assert not torch.allclose(y, base)


def test_gmm_contrast_is_monotone():
    """The GMM-contrast remap must preserve intensity ordering (monotone) so it
    cannot locally invert tissue contrast, and must keep the zero background."""
    g = torch.Generator().manual_seed(11)
    # Strictly increasing foreground ramp + a zero-background slab.
    fg = torch.linspace(0.1, 5.0, 8 * 8 * 8).reshape(1, 8, 8, 8)
    x = torch.cat([fg, torch.zeros(1, 8, 8, 8)], dim=0)  # ch0 fg ramp, ch1 background
    y = gmm_contrast(x, g)
    v, w = x[0].flatten(), y[0].flatten()
    order = torch.argsort(v)
    remapped = w[order]
    # Sorted-by-input output is non-decreasing (allow tiny float slack).
    assert bool((remapped[1:] - remapped[:-1] >= -1e-5).all())
    # Background channel stays exactly zero.
    assert torch.count_nonzero(y[1]) == 0


def _cfg(enabled, **toggles):
    p = dict(enabled=enabled, rician_snr_range=(5.0, 30.0),
             resolution_downsample_range=(0.5, 1.0), stacked_n=0)
    for name in ("bias_field", "intensity_shift", "gamma_correction",
                 "gmm_contrast", "rician_noise", "resolution_jitter",
                 "kspace_partial_fourier", "kspace_ghosting", "kspace_motion",
                 "auxiliary_fourier_basis"):
        p[name] = toggles.get(name, False)
    return SimpleNamespace(augmentation=SimpleNamespace(physics=SimpleNamespace(**p)))


def test_disabled_is_identity():
    aug = build_physics_augmentations(_cfg(False, bias_field=True))
    x = _x()
    assert torch.equal(aug(x), x)


def test_all_toggles_false_is_identity():
    aug = build_physics_augmentations(_cfg(True))
    x = _x()
    assert torch.equal(aug(x), x)


def test_enabled_changes_input():
    aug = build_physics_augmentations(_cfg(True, gamma_correction=True, rician_noise=True))
    x = _x()
    y = aug(x, generator=torch.Generator().manual_seed(3))
    assert y.shape == x.shape and torch.isfinite(y).all()
    assert not torch.allclose(y, x)


def test_production_kspace_toggles_are_active():
    aug = build_physics_augmentations(_cfg(True, kspace_partial_fourier=True,
                                           kspace_ghosting=True, kspace_motion=True))
    x = _x()
    y = aug(x, generator=torch.Generator().manual_seed(4))
    assert y.shape == x.shape and torch.isfinite(y).all()
    assert not torch.allclose(y, x)


def test_kspace_ops_preserve_shape_device_and_dtype():
    x = _x()
    for fn in (kspace_partial_fourier, kspace_ghosting, kspace_motion):
        y = fn(x, torch.Generator().manual_seed(5))
        assert y.shape == x.shape
        assert y.device == x.device
        assert y.dtype == x.dtype


def test_bias_and_noise_preserve_cuda_device_if_available():
    if not torch.cuda.is_available():
        return
    x = _x().cuda()
    for fn in (random_bias_field, rician_noise):
        y = fn(x, None)
        assert y.shape == x.shape
        assert y.device.type == "cuda"
        assert y.dtype == x.dtype


def test_rician_sigma_uses_object_wavelet_coefficients():
    x = _x()[0]
    sigma = estimate_rician_sigma_mad_hhh(x)
    assert sigma > 0 and sigma < float(x.std()) * 2.0


def test_rician_added_sigma_quadrature():
    """Reaching a target SNR adds noise in quadrature with existing noise,
    never the max() of the two (which would over-inject). [audit S2-A]"""
    from ogap.augmentations.physics_augment import rician_added_sigma

    # No pre-existing noise -> add the full target sigma (mean/snr = 10/5 = 2).
    assert abs(rician_added_sigma(10.0, 5.0, 0.0) - 2.0) < 1e-6
    # Quadrature: target sigma = 10/2 = 5, observed = 3 -> sqrt(25-9) = 4.
    assert abs(rician_added_sigma(10.0, 2.0, 3.0) - 4.0) < 1e-6
    # Existing noise already exceeds target -> add nothing (cannot raise SNR).
    assert rician_added_sigma(10.0, 5.0, 5.0) == 0.0
    assert rician_added_sigma(10.0, 5.0, 9.0) == 0.0


def test_rician_does_not_overinject_beyond_target_snr():
    """A nearly-noise-free volume corrupted at a known target SNR should end up
    with noise close to the target, not target+existing. [audit S2-A]"""
    g = torch.Generator().manual_seed(7)
    # Smooth, high-signal volume (very low intrinsic noise).
    v = torch.full((1, 24, 24, 24), 100.0)
    target_snr = 10.0
    y = rician_noise(v, g, (target_snr, target_snr))
    # Empirical noise std of the magnitude image vs the expected target sigma
    # (mean/snr = 100/10 = 10). Quadrature with ~0 existing noise => ~10.
    emp = float((y - v).std())
    assert 6.0 < emp < 14.0, emp  # within ~40% of target, not ~2x (old max())
