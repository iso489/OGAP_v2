"""Physics-consistency tests for the Bloch low-field simulator (CPU, no native data).

These assert the *robust* qualitative facts of brain MR contrast and the
field-dependent GM/WM T1 contrast collapse - the in-silico validation that stands in
for scarce native 64 mT data.
"""
import torch

from ogap.augmentations.bloch_lowfield import (
    render_lowfield, BlochLowFieldSimulator,
)


def _slabs():
    """Pure-tissue slabs: CSF / WM / GM, plus an enhancing-tumour (ET=3) slab."""
    d = 4
    pve = torch.zeros(3, d, d, d)
    pve[0, 0] = 1.0  # CSF
    pve[1, 1] = 1.0  # WM
    pve[2, 2] = 1.0  # GM
    label = torch.zeros(d, d, d, dtype=torch.long)
    label[3] = 3     # ET
    return pve, label


def test_bloch_lowfield_contrast_is_physical():
    pve, label = _slabs()
    s = render_lowfield(pve, label, target_b0_t=0.064, normalise=False)  # t1n,t1c,t2w,t2f
    m = lambda c, region: float(s[c, region].mean())
    CSF, WM, GM, ET = 0, 1, 2, 3
    # T2-weighted: CSF brightest, then GM, then WM.
    assert m(2, CSF) > m(2, GM) > m(2, WM)
    # FLAIR nulls CSF -> CSF is the darkest of the three normal tissues.
    flair = [m(3, CSF), m(3, WM), m(3, GM)]
    assert flair[0] == min(flair)
    # Post-contrast T1 (Gd) makes the enhancing tumour brighter than on plain T1.
    assert m(1, ET) > m(0, ET)


def test_bloch_t1w_contrast_collapses_at_low_field():
    d = 4
    pve = torch.zeros(3, d, d, d)
    pve[1, 1] = 1.0  # WM
    pve[2, 2] = 1.0  # GM

    def rel_contrast(b0):
        s = render_lowfield(pve, None, target_b0_t=b0, normalise=False)
        wm, gm = float(s[0, 1].mean()), float(s[0, 2].mean())
        return abs(wm - gm) / (wm + gm)

    # 3 T: strong, conventional T1w contrast (WM brighter than GM).
    s3 = render_lowfield(pve, None, target_b0_t=3.0, normalise=False)
    assert float(s3[0, 1].mean()) > float(s3[0, 2].mean())
    # The relative GM/WM T1w contrast COLLAPSES toward low field.
    assert rel_contrast(3.0) > rel_contrast(0.064)


def test_bloch_simulator_label_preserving_and_finite():
    pve, label = _slabs()
    x = torch.rand(4, 4, 4, 4)
    label0 = label.clone()
    sim = BlochLowFieldSimulator(n_mri=4)
    out = sim(x, pve, label)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert torch.equal(label, label0)  # the simulator never touches the label
    assert not torch.equal(out[:4], x[:4])  # MRI channels replaced by the render


def test_bloch_quantitative_contrast_ratios_within_published_envelope():
    """Quantitative in-silico validation (the headline 'validated in-silico' evidence):
    the GM/WM T1 ratio must COLLAPSE from clinical field to 64 mT, landing in the small
    low-field gap O'Reilly 2022 (50 mT) and Volegov (46 µT) report, while staying well
    separated at 3 T. Validates contrast RATIOS, not absolute values (method-dependent)."""
    from ogap.augmentations.relaxation_constants import LOWFIELD_VALIDATION
    r64 = LOWFIELD_VALIDATION["gm_wm_t1_ratio_64mT"]
    r3 = LOWFIELD_VALIDATION["gm_wm_t1_ratio_3T"]
    # Low-field GM/WM T1 contrast is small (published 50 mT: WM~275 / GM~327 -> ~1.0-1.3).
    assert 1.0 < r64 < 1.35
    # 3 T retains strong GM/WM T1 contrast (Wansapura: 1331/832 ~ 1.6).
    assert 1.45 < r3 < 1.75
    # The collapse is the single most important low-field fact.
    assert r3 - r64 > 0.3

    # And the rendered T2w image keeps CSF >> WM (a robust, method-independent ratio).
    pve, label = _slabs()
    s = render_lowfield(pve, label, target_b0_t=0.064, normalise=False)
    csf_t2 = float(s[2, 0].mean()); wm_t2 = float(s[2, 1].mean())
    assert csf_t2 / max(wm_t2, 1e-6) > 1.5


def test_bloch_render_preserves_skull_stripped_background():
    """A skull-stripped (exact-zero background) input must stay zero-background
    after the render, so the augmentation does not inject a non-zero background
    that no real deployment scan has and that breaks the downstream `v != 0`
    foreground masks (Rician sigma, z-score)."""
    pve, label = _slabs()
    # Build a skull-stripped input: zero everywhere except a small brain region.
    x = torch.zeros(4, 4, 4, 4)
    brain = torch.zeros(4, 4, 4, dtype=torch.bool)
    brain[1:3] = True  # only WM/GM slabs are "brain"
    x[:, brain] = torch.randn(4, int(brain.sum()))
    sim = BlochLowFieldSimulator(n_mri=4)
    out = sim(x, pve, label)
    # Background (where the input was exactly zero) must remain exactly zero.
    bg = ~brain
    assert torch.count_nonzero(out[:, bg]) == 0
    # Brain voxels are re-rendered (not all left untouched).
    assert not torch.equal(out[:, brain], x[:, brain])


def test_bloch_config_build_and_scale_match():
    """The legacy config builders gate the simulator and the render matches the
    input intensity scale (so it drops into the z-scored preprocessing the net expects)."""
    from types import SimpleNamespace
    import ogap.legacy as L
    off = SimpleNamespace(
        augmentation=SimpleNamespace(physics=SimpleNamespace(bloch_lowfield=False)),
        data=SimpleNamespace(in_channels=4))
    assert L._v91_bloch_sim(off) is None            # default OFF
    on = SimpleNamespace(
        augmentation=SimpleNamespace(physics=SimpleNamespace(
            bloch_lowfield=True, bloch_lowfield_target_b0=0.064, bloch_lowfield_prob=0.5)),
        data=SimpleNamespace(in_channels=4))
    sim = L._v91_bloch_sim(on)
    assert sim is not None and sim.target_b0_t == 0.064 and sim.n_mri == 4
    assert L._v91_bloch_prob(on) == 0.5
    # statistics matching: a z-scored input stays ~zero-mean after the render replaces it
    pve, label = _slabs()
    x = torch.randn(4, 4, 4, 4)
    out = sim(x, pve, label)
    assert torch.isfinite(out).all()
    assert all(abs(float(out[c].mean()) - float(x[c].mean())) < 0.5 for c in range(4))
