"""Integration tests that exercise the pipeline on a real BraTS sample case.

Runs the actual pipeline pieces on a local sample directory (4 modalities +
segmentation), cropped to the tumour bounding box for speed. Point the
``OGAP_SAMPLE_DATA`` environment variable at a directory containing ``Images/`` and
``Segmentation/`` NIfTI files (or drop a fixture under ``tests/data/sample``):

* preprocessing (foreground z-score) keeps the skull-stripped background at 0;
* the fixed additive-intensity augmentations preserve that zero background and are
  finite on a real volume;
* the AFA augmentation produces a meaningful (non-vanishing) perturbation;
* BraTS metrics are perfect for seg-vs-seg and degrade sensibly under erosion;
* the Bloch low-field render runs on the real label geometry and the SBM Gd model
  makes the enhancing tumour brighter on the post-contrast (T1c) channel.

Skips automatically when the sample data is not present.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pytest
import torch

_SAMPLE = os.environ.get(
    "OGAP_SAMPLE_DATA", os.path.join(os.path.dirname(__file__), "data", "sample"))
pytestmark = pytest.mark.skipif(
    not os.path.isdir(_SAMPLE),
    reason="OGAP sample data not present (set OGAP_SAMPLE_DATA to enable)")


def _load_case():
    import nibabel as nib

    imgs = sorted(glob.glob(os.path.join(_SAMPLE, "Images", "*.nii.gz")))
    seg = glob.glob(os.path.join(_SAMPLE, "Segmentation", "*.nii.gz"))[0]
    vols = np.stack([nib.load(p).get_fdata().astype(np.float32) for p in imgs], axis=0)
    lab = np.rint(nib.load(seg).get_fdata()).astype(np.int64)
    return vols, lab


def _tumour_crop(vols, lab, margin=12):
    idx = np.argwhere(lab > 0)
    lo = np.maximum(idx.min(0) - margin, 0)
    hi = np.minimum(idx.max(0) + margin + 1, np.array(lab.shape))
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    return vols[(slice(None),) + sl].copy(), lab[sl].copy()


@pytest.fixture(scope="module")
def case():
    vols, lab = _load_case()
    assert vols.shape[0] == 4 and set(np.unique(lab)).issubset({0, 1, 2, 3})
    return _tumour_crop(vols, lab)


def _zscored(vols):
    from ogap.legacy import zscore_nonzero
    return torch.from_numpy(np.stack([zscore_nonzero(vols[c]) for c in range(vols.shape[0])], 0))


def test_preprocess_preserves_zero_background_on_real_case(case):
    from ogap.legacy import zscore_nonzero

    vols, _ = case
    for c in range(vols.shape[0]):
        x = vols[c]
        bg = x == 0
        z = zscore_nonzero(x)
        assert np.isfinite(z).all()
        assert np.all(z[bg] == 0.0)                       # skull-stripped background stays 0
        fg = z[~bg]
        assert abs(float(fg.mean())) < 1e-3               # foreground z-scored
        assert abs(float(fg.std()) - 1.0) < 2e-2


def test_fixed_intensity_augmentations_preserve_background_on_real_case(case):
    from ogap.augmentations.physics_augment import (
        rician_noise, random_intensity_shift, random_gamma, auxiliary_fourier_basis,
    )

    vols, _ = case
    x = _zscored(vols)
    bg = x == 0
    fg = x != 0
    # A low SNR forces rician_noise to actually inject noise (at high SNR the [S2-A]
    # quadrature rule correctly no-ops because the z-scored input already exceeds the
    # target SNR - you cannot raise SNR by adding noise).
    outputs = {
        "rician_noise": rician_noise(x, torch.Generator().manual_seed(0), (0.8, 1.5)),
        "intensity_shift": random_intensity_shift(x, torch.Generator().manual_seed(0)),
        "gamma": random_gamma(x, torch.Generator().manual_seed(0)),
        "afa": auxiliary_fourier_basis(x, torch.Generator().manual_seed(0), 0.2),
    }
    for name, out in outputs.items():
        assert out.shape == x.shape, name
        assert torch.isfinite(out).all(), name
        assert torch.all(out[bg] == 0.0), f"{name} corrupted the zero background"
        assert not torch.equal(out[fg], x[fg]), f"{name} did not change the foreground"


def test_afa_is_meaningful_on_real_case(case):
    from ogap.augmentations.physics_augment import auxiliary_fourier_basis

    vols, _ = case
    x = _zscored(vols)
    out = auxiliary_fourier_basis(x, torch.Generator().manual_seed(0), 0.2)
    rels = []
    for c in range(x.shape[0]):
        fg = x[c] != 0
        rng = float(x[c][fg].max() - x[c][fg].min())
        rels.append(float((out[c] - x[c])[fg].pow(2).mean().sqrt()) / rng)
    assert sum(rels) / len(rels) > 0.02   # not the ~1/sqrt(N) vanishing no-op


def test_brats_metrics_on_real_segmentation(case):
    from scipy import ndimage as ndi

    from ogap.evaluation.brats_metrics import compute_brats_metrics

    _, lab = case
    perfect = compute_brats_metrics(lab, lab, voxel_spacing_mm=(1.0, 1.0, 1.0))
    for r in ("wt", "tc", "et"):
        assert perfect[f"dsc_{r}"] == 1.0
        assert perfect[f"hd95_{r}"] == 0.0

    # Erode the ET region -> imperfect but non-zero overlap, finite positive HD95.
    et = lab == 3
    shell = et & ~ndi.binary_erosion(et)
    pred = lab.copy()
    pred[shell] = 0
    deg = compute_brats_metrics(pred, lab, voxel_spacing_mm=(1.0, 1.0, 1.0))
    assert 0.0 < deg["dsc_et"] < 1.0
    assert np.isfinite(deg["hd95_et"]) and deg["hd95_et"] > 0.0


def test_bloch_render_sbm_gd_enhancement_on_real_label(case):
    from ogap.augmentations.bloch_lowfield import render_lowfield

    vols, lab = case
    t1 = vols[0]
    brain = vols.sum(0) != 0
    # Crude but valid 3-class PVE (CSF/WM/GM) from T1 terciles inside the brain mask.
    pve = np.zeros((3,) + lab.shape, dtype=np.float32)
    fg = t1[brain]
    q1, q2 = np.percentile(fg, [33, 66])
    pve[0][brain & (t1 <= q1)] = 1.0              # darkest -> CSF
    pve[2][brain & (t1 > q1) & (t1 <= q2)] = 1.0  # mid -> GM
    pve[1][brain & (t1 > q2)] = 1.0               # brightest -> WM
    pve_t = torch.from_numpy(pve)
    lab_t = torch.from_numpy(lab)
    lab_before = lab_t.clone()

    sig64 = render_lowfield(pve_t, lab_t, target_b0_t=0.064, normalise=False)
    sig3t = render_lowfield(pve_t, lab_t, target_b0_t=3.0, normalise=False)
    assert sig64.shape == (4,) + lab.shape
    assert torch.isfinite(sig64).all() and torch.isfinite(sig3t).all()
    assert torch.equal(lab_t, lab_before)         # render is label-preserving

    et = lab_t == 3
    assert et.any()
    # T1c (post-Gd) vs T1n (pre-Gd) enhancement ratio on the enhancing tumour.
    ratio_64 = float(sig64[1][et].mean()) / float(sig64[0][et].mean())
    ratio_3t = float(sig3t[1][et].mean()) / float(sig3t[0][et].mean())
    assert ratio_64 > 1.1                          # Gd brightens ET on T1c at low field
    # SBM signature: the *relative* enhancement is stronger where pre-contrast T1 is
    # long (high field 1900 ms -> ~330 ms) than at low field (560 ms -> ~233 ms).
    assert ratio_3t > ratio_64
