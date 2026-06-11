"""SynthSeg-style randomized contrast generator tests (CPU)."""
from types import SimpleNamespace

import torch

from ogap.augmentations.synth import SynthSegGenerator, build_synth_generator


def _inputs():
    pve = torch.zeros(3, 6, 6, 6)
    pve[0, 0] = 1.0
    pve[1, 1] = 1.0
    pve[2, 2] = 1.0
    label = torch.zeros(6, 6, 6, dtype=torch.long)
    label[3] = 3
    x = torch.zeros(4, 6, 6, 6)
    brain = torch.zeros(6, 6, 6, dtype=torch.bool)
    brain[:4] = True
    x[:, brain] = torch.randn(4, int(brain.sum()))
    return x, pve, label, brain


def test_synth_replaces_channels_and_preserves_label_and_background():
    x, pve, label, brain = _inputs()
    label0 = label.clone()
    gen = SynthSegGenerator(n_mri=4, prob=1.0)
    out = gen(x, pve, label, generator=torch.Generator().manual_seed(0))
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert torch.equal(label, label0)             # label untouched
    assert not torch.equal(out, x)                # channels re-rendered
    assert torch.count_nonzero(out[:, ~brain]) == 0  # zero background preserved


def test_synth_prob_zero_is_identity():
    x, pve, label, _ = _inputs()
    gen = SynthSegGenerator(n_mri=4, prob=0.0)
    out = gen(x, pve, label, generator=torch.Generator().manual_seed(1))
    assert out is x  # never fires -> returns the same object (lets the dataset fall through)


def test_build_synth_generator_gated_by_config():
    off = SimpleNamespace(augmentation=SimpleNamespace(physics=SimpleNamespace(
        synth_randomization=False)), data=SimpleNamespace(n_mri_channels=4, in_channels=7))
    assert build_synth_generator(off) is None
    on = SimpleNamespace(augmentation=SimpleNamespace(physics=SimpleNamespace(
        synth_randomization=True, synth_prob=1.0)),
        data=SimpleNamespace(n_mri_channels=4, in_channels=7))
    gen = build_synth_generator(on)
    assert gen is not None and gen.n_mri == 4
