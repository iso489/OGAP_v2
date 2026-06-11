"""Standard 3D segmentation metrics for the overlapping tumour subregions.

Per subregion (WT = {1,2,3}, TC = {1,3}, ET = {3}) we report DSC, 95th-
percentile Hausdorff distance (mm), sensitivity and specificity, computed on the
full 3D volume. Edge cases follow the conventions users expect:

* Both prediction and reference empty for a region  -> DSC = 1.0, HD95 = 0.0
  (perfect agreement on a true negative).
* Exactly one of prediction/reference empty        -> DSC = 0.0, HD95 = NaN
  (a distance is undefined; NaN is propagated, never silently zeroed).

HD95 uses a SciPy distance transform (no hard MedPy dependency).

NOTE on spacing: ``voxel_spacing_mm`` defaults to isotropic ``(1,1,1)``. Low-field
/ BraTS-Africa volumes are often anisotropic, so callers MUST pass the real
NIfTI spacing for HD95 to be in millimetres. The production pipeline computes
HD95 via ``ogap.legacy`` (which threads ``header.get_zooms()`` spacing through
``load_nifti_with_spacing``); this module is the library/test surface and does
not infer spacing on its own. [audit S2-D]
"""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np

try:
    from scipy import ndimage as _ndi
    _SCIPY = True
except ImportError:  # pragma: no cover
    import logging as _logging
    _logging.getLogger(__name__).warning(
        "scipy not available: HD95 surface-distance metrics will be NaN for every "
        "case. Install scipy to compute boundary metrics.")
    _SCIPY = False

_REGIONS: Dict[str, Tuple[int, ...]] = {"WT": (1, 2, 3), "TC": (1, 3), "ET": (3,)}


def _to_np(a) -> np.ndarray:
    if hasattr(a, "detach"):
        a = a.detach().cpu().numpy()
    return np.asarray(a)


def _region_mask(label: np.ndarray, region: str) -> np.ndarray:
    return np.isin(label, _REGIONS[region])


def _dsc(pred: np.ndarray, ref: np.ndarray, eps: float = 1e-5) -> float:
    p, r = pred.sum(), ref.sum()
    if p == 0 and r == 0:
        return 1.0          # both empty: perfect agreement on a true negative
    if p == 0 or r == 0:
        return 0.0          # exactly one empty: no overlap possible (BraTS convention)
    inter = np.logical_and(pred, ref).sum()
    return float((2.0 * inter + eps) / (p + r + eps))


def _surface_distances(a: np.ndarray, b: np.ndarray, spacing: Sequence[float]) -> np.ndarray:
    a_surf = a ^ _ndi.binary_erosion(a)
    b_surf = b ^ _ndi.binary_erosion(b)
    if a_surf.sum() == 0 or b_surf.sum() == 0:
        return np.array([np.nan])
    dt = _ndi.distance_transform_edt(~b_surf, sampling=spacing)
    return dt[a_surf]


def _hd95(pred: np.ndarray, ref: np.ndarray, spacing: Sequence[float],
          empty_policy: str = "nan", penalty_mm: float = 373.1287) -> float:
    p, r = pred.sum(), ref.sum()
    if p == 0 and r == 0:
        return 0.0
    if p == 0 or r == 0:
        # Exactly one of prediction/reference is empty: the surface distance is
        # undefined. Two conventions are supported via ``empty_policy``:
        #   "nan"     -> propagate NaN (a distance truly does not exist; never
        #                silently zeroed). Default; matches this module's history.
        #   "penalty" -> the BraTS *leaderboard* convention, which substitutes a
        #                large fixed distance so a missed / false-positive region is
        #                penalised rather than dropped from the average. The default
        #                373.1287 mm is the BraTS-standard cap.
        # State the chosen convention in the paper's Methods; the two give
        # materially different ET/TC numbers when small regions are absent. [F6]
        return float(penalty_mm) if empty_policy == "penalty" else float("nan")
    if not _SCIPY:
        return float("nan")
    # BraTS-standard HD95: the *max* of the two directional 95th percentiles, not
    # the 95th percentile of the pooled distances (pooling underestimates, because
    # a large-distance direction with few surface voxels gets diluted below the
    # 95th percentile of the union).
    d_ab = _surface_distances(pred, ref, spacing)
    d_ba = _surface_distances(ref, pred, spacing)
    d_ab = d_ab[~np.isnan(d_ab)]
    d_ba = d_ba[~np.isnan(d_ba)]
    if d_ab.size == 0 or d_ba.size == 0:
        return float("nan")
    return float(max(np.percentile(d_ab, 95), np.percentile(d_ba, 95)))


def _sens_spec(pred: np.ndarray, ref: np.ndarray) -> Tuple[float, float]:
    tp = np.logical_and(pred, ref).sum()
    fn = np.logical_and(~pred, ref).sum()
    fp = np.logical_and(pred, ~ref).sum()
    tn = np.logical_and(~pred, ~ref).sum()
    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    return sens, spec


def compute_brats_metrics(pred, target,
                          voxel_spacing_mm: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                          hd95_empty_policy: str = "nan",
                          hd95_penalty_mm: float = 373.1287,
                          ) -> Dict[str, float]:
    """Compute DSC / HD95 / sensitivity / specificity per tumour subregion.

    Args:
        pred: integer label volume ``(D, H, W)`` (0=bg, 1/2/3 foreground).
        target: integer reference label volume of the same shape.
        voxel_spacing_mm: physical voxel size for HD95 in millimetres. Pass the
            real NIfTI spacing for anisotropic low-field volumes.
        hd95_empty_policy: how to score HD95 when exactly one of prediction /
            reference is empty for a region — ``"nan"`` (default; the distance is
            undefined) or ``"penalty"`` (BraTS-leaderboard convention; substitutes
            ``hd95_penalty_mm``). Choose deliberately and report it. [F6]
        hd95_penalty_mm: the penalty distance used when ``hd95_empty_policy ==
            "penalty"`` (default 373.1287 mm, the BraTS-standard cap).

    Returns:
        Dict with ``dsc_*``, ``hd95_*``, ``sens_*``, ``spec_*`` for
        ``* in {wt, tc, et}``.
    """
    p = _to_np(pred).astype(np.int64)
    t = _to_np(target).astype(np.int64)
    if p.shape != t.shape:
        raise ValueError(f"pred shape {p.shape} != target shape {t.shape}")
    out: Dict[str, float] = {}
    for region in ("WT", "TC", "ET"):
        pm, tm = _region_mask(p, region), _region_mask(t, region)
        key = region.lower()
        out[f"dsc_{key}"] = _dsc(pm, tm)
        out[f"hd95_{key}"] = _hd95(pm, tm, voxel_spacing_mm,
                                   empty_policy=hd95_empty_policy,
                                   penalty_mm=hd95_penalty_mm)
        sens, spec = _sens_spec(pm, tm)
        out[f"sens_{key}"], out[f"spec_{key}"] = sens, spec
    return out
