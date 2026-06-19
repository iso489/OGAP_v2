"""Regression tests for the 2026-06-10 line-by-line audit fixes.

Each test encodes the *correct* behaviour and fails on the pre-fix code:

* A  - AFA augmentation must perturb by a meaningful fraction of the intensity
       range, not ~1/sqrt(N) (whole-volume L2 normalisation made it a no-op).
* B1 - build_synth_generator must honour an explicit 0.0 ablation value
       (the ``x or default`` short-circuit silently restored the default).
* B2 - build_model_ema must honour an explicit decay=0.0.
* C  - _deep_merge must reject replacing a dict subtree with a scalar.
* D2 - modality_degradation_analysis must not KeyError when models evaluated
       different modality combinations.
* E  - the latent-ODE recognition net must pair each (reverse-time) step with
       the gap to the *previously processed* observation (backward gaps).
* BG - rician_noise / random_intensity_shift / random_gamma must preserve the
       skull-stripped zero background (the ``x != 0`` deployment invariant).
* Gd - enhancing-tumour T1 shortening must follow the SBM relaxivity model
       (1/T1_post = 1/T1_pre + r1*[Gd]), not a fixed multiplier.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn


# ── A: AFA is not a vanishing no-op ──────────────────────────────────────────
def test_afa_perturbation_is_meaningful_not_vanishing():
    from ogap.augmentations.physics_augment import auxiliary_fourier_basis

    g = torch.Generator().manual_seed(0)
    x = torch.randn(8, 24, 24, 24)  # 8 channels to average out the random magnitude
    out = auxiliary_fourier_basis(x, g, 0.3)
    rels = []
    for c in range(x.shape[0]):
        rng = float(x[c].max() - x[c].min())
        rels.append(float((out[c] - x[c]).pow(2).mean().sqrt()) / rng)
    rel = sum(rels) / len(rels)
    # pre-fix: ~mean_strength / sqrt(N) ~ 2.6e-3 ; fixed: ~0.7 * mean_strength
    assert rel > 0.05, f"AFA perturbation is vanishing (rel={rel:.4e})"


# ── B1: synth builder honours explicit zero ──────────────────────────────────
def test_build_synth_generator_keeps_explicit_zero():
    from ogap.augmentations.synth import build_synth_generator

    cfg = SimpleNamespace(
        augmentation=SimpleNamespace(physics=SimpleNamespace(
            synth_randomization=True,
            synth_prob=0.0,
            synth_bias_strength=0.0,
            synth_intensity_std_max=0.0,
            synth_blur_sigma_max=0.0,
        )),
        data=SimpleNamespace(n_mri_channels=4, in_channels=4),
    )
    gen = build_synth_generator(cfg)
    assert gen is not None
    assert gen.prob == 0.0
    assert gen.bias_strength == 0.0
    assert gen.intensity_std_max == 0.0
    assert gen.blur_sigma_max == 0.0


# ── B2: EMA builder honours explicit zero decay ──────────────────────────────
def test_build_model_ema_keeps_explicit_zero_decay():
    from ogap.utils.ema import build_model_ema

    model = nn.Conv3d(1, 1, 1)
    cfg = SimpleNamespace(training=SimpleNamespace(
        ema=SimpleNamespace(enabled=True, decay=0.0)))
    ema = build_model_ema(model, cfg)
    assert ema is not None
    assert ema.decay == 0.0


# ── C: config merge rejects dict->scalar replacement ─────────────────────────
def test_deep_merge_rejects_dict_replaced_by_scalar():
    from ogap.config.loader import _deep_merge

    base = {"training": {"compile": {"enabled": False, "mode": "max-autotune"}}}
    over = {"training": {"compile": True}}
    with pytest.raises(ValueError):
        _deep_merge(base, over)


def test_deep_merge_still_merges_nested_dicts():
    from ogap.config.loader import _deep_merge

    base = {"a": {"b": 1, "c": 2}}
    assert _deep_merge(base, {"a": {"b": 9}}) == {"a": {"b": 9, "c": 2}}


def test_deep_merge_allows_null_to_disable_a_block():
    # `block: null` is the legitimate "disable this sub-config" idiom and must be
    # allowed (only a non-null scalar over a dict is the silent-subtree-loss bug).
    from ogap.config.loader import _deep_merge

    base = {"domain": {"enabled": True, "lambda": 0.1}}
    assert _deep_merge(base, {"domain": None}) == {"domain": None}


# ── D2: degradation analysis tolerates ragged combo coverage ─────────────────
def test_modality_degradation_no_keyerror_on_ragged_combos():
    from ogap.evaluation.modality_robustness import (
        modality_degradation_analysis, combo_label,
    )

    def case(d):
        return {"dsc_et": d}

    full = combo_label(frozenset({0, 1, 2, 3}))
    a_drop = combo_label(frozenset({1, 2, 3}))   # model A dropped modality 0
    b_drop = combo_label(frozenset({0, 2, 3}))   # model B dropped a *different* modality
    results = {
        "A": {full: {"c1": case(0.80), "c2": case(0.70), "c3": case(0.75)},
              a_drop: {"c1": case(0.60), "c2": case(0.50), "c3": case(0.55)}},
        "B": {full: {"c1": case(0.82), "c2": case(0.72), "c3": case(0.77)},
              b_drop: {"c1": case(0.58), "c2": case(0.48), "c3": case(0.53)}},
    }
    df = modality_degradation_analysis(results, metric="dsc_et")
    assert df is not None and len(df) > 0


# ── E: latent-ODE recognition net uses backward gaps ─────────────────────────
def test_latent_ode_encoder_input_uses_backward_gaps():
    from ogap.longitudinal.latent_ode import LatentODERANOTracker

    tr = LatentODERANOTracker(obs_dim=1)
    values = torch.zeros(1, 3, 1)
    times = torch.tensor([0.0, 1.0, 10.0])
    inp = tr._encoder_input(values, times)        # reversed (latest-first), [value, dt]
    dt = inp[0, :, -1]
    # processed latest-first: t=10 (first -> dt=0), t=1 (gap 10->1 = 9), t=0 (gap 1->0 = 1)
    assert torch.allclose(dt, torch.tensor([0.0, 9.0, 1.0]))


def test_latent_ode_encoder_input_backward_gaps_batched():
    from ogap.longitudinal.latent_ode import LatentODERANOTracker

    tr = LatentODERANOTracker(obs_dim=1)
    values = torch.zeros(2, 3, 1)
    times = torch.tensor([[0.0, 2.0, 5.0], [0.0, 1.0, 10.0]])
    inp = tr._encoder_input(values, times)
    assert torch.allclose(inp[0, :, -1], torch.tensor([0.0, 3.0, 2.0]))
    assert torch.allclose(inp[1, :, -1], torch.tensor([0.0, 9.0, 1.0]))


# ── BG: intensity/noise transforms preserve the zero background ──────────────
def _vol_with_background():
    torch.manual_seed(0)
    x = torch.randn(3, 8, 8, 8)   # z-scored-like foreground (has negatives -> min < 0)
    x[:, :2] = 0.0                # explicit zero-background slab
    return x


def test_rician_noise_preserves_zero_background():
    from ogap.augmentations.physics_augment import rician_noise

    x = _vol_with_background()
    out = rician_noise(x, torch.Generator().manual_seed(1), (5.0, 10.0))
    assert torch.all(out[:, :2] == 0.0)
    assert not torch.all(out[:, 2:] == x[:, 2:])  # foreground actually changed


def test_rician_noise_injects_noise_on_zscored_data():
    # The signal level for the SNR target must be the mean MAGNITUDE of the
    # foreground, not |mean| (~0 on z-scored / mean-centred data) - otherwise
    # rician_noise injects ~nothing on the real (z-scored) pipeline inputs. Use a
    # SMOOTH, mean-centred volume (low intrinsic noise, like z-scored anatomy) so
    # the target sigma can exceed the small existing sigma and noise is injected.
    import torch.nn.functional as F

    from ogap.augmentations.physics_augment import rician_noise

    torch.manual_seed(0)
    low = torch.randn(1, 1, 4, 4, 4)
    x = F.interpolate(low, size=(16, 16, 16), mode="trilinear", align_corners=False)[0]
    x = x - x.mean()                 # mean-centred (|mean| ~ 0), smooth foreground
    out = rician_noise(x, torch.Generator().manual_seed(1), (1.0, 2.0))
    assert not torch.equal(out, x)


def test_intensity_shift_preserves_zero_background():
    from ogap.augmentations.physics_augment import random_intensity_shift

    x = _vol_with_background()
    out = random_intensity_shift(x, torch.Generator().manual_seed(1))
    assert torch.all(out[:, :2] == 0.0)


def test_gamma_preserves_zero_background():
    from ogap.augmentations.physics_augment import random_gamma

    x = _vol_with_background()
    out = random_gamma(x, torch.Generator().manual_seed(1))
    assert torch.all(out[:, :2] == 0.0)


# ── Legacy-path mirrors of the same bug classes ─────────────────────────────
def test_legacy_afa_is_meaningful_and_preserves_background():
    import numpy as np

    from ogap.legacy import MRIPhysicsAugmentor

    np.random.seed(0)
    aug = MRIPhysicsAugmentor(p_fourier_amplitude_mix=1.0, fourier_beta_range=(0.2, 0.3))
    x = np.zeros((2, 16, 16, 16), np.float32)
    x[:, 4:] = np.random.standard_normal((2, 12, 16, 16)).astype(np.float32)
    out = aug._fourier_amplitude_mix(x.copy())
    fg = x != 0
    rel = float(np.sqrt(((out - x) ** 2)[fg].mean())) / float(x[fg].max() - x[fg].min())
    assert rel > 0.02                          # not the ~1/sqrt(N) vanishing no-op
    assert np.all(out[x == 0] == 0.0)          # zero background preserved


def test_legacy_bloch_prob_keeps_explicit_zero():
    from ogap.legacy import _v91_bloch_prob

    v = SimpleNamespace(augmentation=SimpleNamespace(
        physics=SimpleNamespace(bloch_lowfield_prob=0.0)))
    assert _v91_bloch_prob(v) == 0.0


# ── Gd: SBM relaxivity model for post-contrast enhancing-tumour T1 ───────────
def test_gd_shortened_t1_follows_sbm_relaxivity():
    from ogap.augmentations import relaxation_constants as rc

    t1_3t = rc.ENHANCING_TUMOUR.t1(rc._B0_3)     # long pre-contrast T1 (high field)
    t1_lf = rc.ENHANCING_TUMOUR.t1(rc._B0_LOW)   # short pre-contrast T1 (low field)
    post_3t = rc.gd_shortened_t1(t1_3t)
    post_lf = rc.gd_shortened_t1(t1_lf)

    assert post_3t < t1_3t and post_lf < t1_lf                 # Gd always shortens T1
    # SBM shortens the rate, so the *relative* drop is larger where T1_pre is long:
    assert (1.0 - post_3t / t1_3t) > (1.0 - post_lf / t1_lf)
    # and it is brighter (shorter post-T1) at 3 T than the fixed 0.45x model it replaces:
    assert post_3t < 0.45 * t1_3t
