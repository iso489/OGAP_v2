import numpy as np
import pytest
import torch

from ogap.evaluation.brats_metrics import compute_brats_metrics


def test_perfect_prediction():
    t = np.zeros((16, 16, 16), dtype=np.int64)
    t[4:8, 4:8, 4:8] = 1
    t[5:7, 5:7, 5:7] = 3
    m = compute_brats_metrics(t.copy(), t.copy())
    assert abs(m["dsc_wt"] - 1.0) < 1e-4
    assert m["hd95_wt"] == 0.0
    assert abs(m["dsc_et"] - 1.0) < 1e-4


def test_all_wrong_zero_dsc():
    t = np.zeros((12, 12, 12), dtype=np.int64); t[2:6, 2:6, 2:6] = 1
    p = np.zeros((12, 12, 12), dtype=np.int64)
    assert compute_brats_metrics(p, t)["dsc_wt"] == 0.0


def test_empty_region_both_dsc_one():
    z = np.zeros((8, 8, 8), dtype=np.int64)
    m = compute_brats_metrics(z, z)
    assert m["dsc_et"] == 1.0 and m["hd95_et"] == 0.0


def test_empty_one_side_hd95_nan():
    t = np.zeros((10, 10, 10), dtype=np.int64); t[3:6, 3:6, 3:6] = 3
    p = np.zeros((10, 10, 10), dtype=np.int64)
    m = compute_brats_metrics(p, t)
    assert m["dsc_et"] == 0.0 and np.isnan(m["hd95_et"])


def test_single_cube_hd95_finite():
    t = np.zeros((16, 16, 16), dtype=np.int64); t[6:9, 6:9, 6:9] = 3
    p = np.zeros((16, 16, 16), dtype=np.int64); p[6:9, 6:9, 7:10] = 3
    m = compute_brats_metrics(p, t)
    assert np.isfinite(m["hd95_et"]) and m["hd95_et"] > 0


def test_accepts_torch_tensors():
    t = torch.zeros(8, 8, 8, dtype=torch.int64); t[2:5, 2:5, 2:5] = 1
    assert abs(compute_brats_metrics(t, t)["dsc_wt"] - 1.0) < 1e-4


def _dir_p95(a, b, spacing=(1.0, 1.0, 1.0)):
    """Independent oracle: 95th percentile of distances from surface(a) to surface(b)."""
    from scipy import ndimage as ndi
    sa = a ^ ndi.binary_erosion(a)
    sb = b ^ ndi.binary_erosion(b)
    dt = ndi.distance_transform_edt(~sb, sampling=spacing)
    return float(np.percentile(dt[sa], 95))


def _pooled_p95(a, b, spacing=(1.0, 1.0, 1.0)):
    from scipy import ndimage as ndi
    sa = a ^ ndi.binary_erosion(a)
    sb = b ^ ndi.binary_erosion(b)
    da = ndi.distance_transform_edt(~sb, sampling=spacing)[sa]
    db = ndi.distance_transform_edt(~sa, sampling=spacing)[sb]
    return float(np.percentile(np.concatenate([da, db]), 95))


def test_hd95_is_max_of_directional_not_pooled():
    # Asymmetric geometry: a small detached appendage on pred only. Its few, far
    # surface voxels survive that direction's 95th percentile (so max is large),
    # but are diluted below the pooled 95th percentile (so concatenation collapses
    # to ~0). This separates BraTS-standard HD95 from the old pooled version.
    ref = np.zeros((32, 32, 32), dtype=np.int64)
    ref[4:12, 4:12, 4:12] = 3                      # ET region (label 3)
    pred = ref.copy()
    pred[4:7, 4:7, 24:27] = 3                       # small far block, pred only

    pred_et, ref_et = (pred == 3), (ref == 3)
    oracle_max = max(_dir_p95(pred_et, ref_et), _dir_p95(ref_et, pred_et))
    pooled = _pooled_p95(pred_et, ref_et)
    assert oracle_max > pooled + 1.0                # geometry actually distinguishes

    m = compute_brats_metrics(pred, ref)
    assert m["hd95_et"] == pytest.approx(oracle_max, abs=1e-6)
    assert m["hd95_et"] > pooled + 1.0              # not the underestimating pooled value
