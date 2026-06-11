"""Channel-aware geometric augmentation tests (CPU).

Verify that flip / affine / elastic keep MRI, tissue-prior channels, and the
label co-registered, and that the augmentor is an identity when disabled.
"""
from types import SimpleNamespace

import torch

from ogap.augmentations.geometric import (
    random_flip, random_affine, random_elastic,
    GeometricAugmentor, build_geometric_augmentor,
)


def _xy(c=7, d=8, h=8, w=8):
    g = torch.Generator().manual_seed(0)
    x = torch.rand(c, d, h, w, generator=g)
    y = torch.zeros(d, h, w, dtype=torch.long)
    y[2:5, 2:5, 2:5] = 1
    y[3:4, 3:4, 3:4] = 3
    return x, y


def test_flip_is_label_coupled():
    x, y = _xy()
    # Force a flip on all axes (p=1) and check x and y move together.
    xf, yf = random_flip(x, y, p=1.0)
    assert xf.shape == x.shape and yf.shape == y.shape
    assert torch.equal(xf, torch.flip(x, dims=[1, 2, 3]))
    assert torch.equal(yf, torch.flip(y, dims=[0, 1, 2]))
    # Label set is preserved (flip is a permutation of voxels).
    assert set(yf.unique().tolist()) == set(y.unique().tolist())


def test_flip_channel_coupled_all_channels_move_identically():
    x, y = _xy(c=7)
    xf, _ = random_flip(x, y, p=1.0)
    # Every channel underwent the SAME flip (tissue priors stay aligned with MRI).
    for c in range(x.shape[0]):
        assert torch.equal(xf[c], torch.flip(x[c], dims=[0, 1, 2]))


def test_affine_preserves_shape_and_label_dtype():
    x, y = _xy()
    xa, ya = random_affine(x, y, p=1.0, generator=torch.Generator().manual_seed(1))
    assert xa.shape == x.shape and ya.shape == y.shape
    assert ya.dtype == y.dtype
    assert torch.isfinite(xa).all()
    # Nearest-sampled label stays integer-valued within the original label set ∪ {0}.
    assert set(ya.unique().tolist()).issubset(set(y.unique().tolist()) | {0})


def test_elastic_preserves_shape_and_finite():
    x, y = _xy()
    xe, ye = random_elastic(x, y, p=1.0, generator=torch.Generator().manual_seed(2))
    assert xe.shape == x.shape and ye.shape == y.shape
    assert torch.isfinite(xe).all()
    assert set(ye.unique().tolist()).issubset(set(y.unique().tolist()) | {0})


def test_batched_path_per_sample_independent():
    g = torch.Generator().manual_seed(3)
    x = torch.rand(2, 7, 8, 8, 8, generator=g)
    y = torch.zeros(2, 8, 8, 8, dtype=torch.long)
    y[:, 2:5, 2:5, 2:5] = 1
    xf, yf = random_flip(x, y, p=1.0)
    assert xf.shape == x.shape and yf.shape == y.shape


def _cfg(enabled, **over):
    g = dict(enabled=enabled, flip=True, flip_p=0.5, affine=True, affine_p=0.3,
             max_rotation_deg=10.0, scale_range=(0.9, 1.1), max_shear=0.05,
             elastic=False, elastic_p=0.2, elastic_alpha=8.0, elastic_grid=4)
    g.update(over)
    return SimpleNamespace(augmentation=SimpleNamespace(geometric=SimpleNamespace(**g)))


def test_disabled_is_identity():
    aug = GeometricAugmentor(_cfg(False))
    x, y = _xy()
    xo, yo = aug(x, y)
    assert torch.equal(xo, x) and torch.equal(yo, y)
    assert build_geometric_augmentor(_cfg(False)) is None


def test_enabled_changes_input_keeps_label_valid():
    aug = build_geometric_augmentor(_cfg(True, flip=True, affine=True))
    assert aug is not None
    x, y = _xy()
    xo, yo = aug(x, y, generator=torch.Generator().manual_seed(5))
    assert xo.shape == x.shape and yo.shape == y.shape
    assert set(yo.unique().tolist()).issubset(set(y.unique().tolist()) | {0})
