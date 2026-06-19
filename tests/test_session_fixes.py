"""Regression tests for the dataset-informed + verified-finding fixes (audit).

Covers behaviours that previously had no test and that this audit changed:
  * SAGER sex-breakdown robustness - missing-value sentinels (incl. the Erasmus
    `-1`), the no-sex-column case (BraTS-Africa), and empty CSVs;
  * `_apply_label_remap` malformed-JSON guard (warn + pass through, never raise);
  * `save_torch_atomic` write + tmp-cleanup contract;
  * `save_nifti_like` dtype guard (a float input is rounded, not truncated);
  * Bloch `render_lowfield` finiteness on an all-zero (background) PVE map.
"""
import csv

import numpy as np
import torch


def _write_csv(tmp_path, rows, header):
    p = tmp_path / "meta.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    return str(p)


# ── SAGER sex breakdown (matches the real cohort metadata patterns) ──────────
def test_sex_breakdown_handles_minus1_sentinel(tmp_path):
    import ogap_compliance as C
    rows = [{"Sex": s} for s in ["M", "F", "M", "-1", "F", "Unknown"]]
    r = C.compute_sex_breakdown(_write_csv(tmp_path, rows, ["Sex"]))
    assert r["status"] == "ok"
    assert r["sex_column_used"] == "Sex"
    assert r["breakdown_count"].get("male") == 2
    assert r["breakdown_count"].get("female") == 2
    # The Erasmus -1 sentinel (and "Unknown") must be counted as unknown, never a
    # spurious sex category.
    assert "-1" not in r["breakdown_count"]
    assert r["n_unknown"] == 2


def test_sex_breakdown_no_column_flags_status(tmp_path):
    import ogap_compliance as C
    # BraTS-Africa metadata has no sex column.
    r = C.compute_sex_breakdown(_write_csv(tmp_path, [{"Center": "CRV"}], ["Center"]))
    assert r["status"] == "no_sex_column"
    assert r["sex_column_used"] is None
    assert r["breakdown_count"] == {}


def test_sex_breakdown_empty_csv_is_no_data(tmp_path):
    import ogap_compliance as C
    r = C.compute_sex_breakdown(_write_csv(tmp_path, [], ["Sex"]))
    assert r["status"] == "no_data"


# ── label remap ──────────────────────────────────────────────────────────────
def test_label_remap_valid_malformed_and_empty():
    import ogap.legacy as L
    remap = L.MultiModalSegDataset._apply_label_remap
    y = np.array([[0, 1, 2, 4]], dtype=np.int16)
    out = remap(y, '{"4": 3}')
    assert (out == np.array([[0, 1, 2, 3]], dtype=np.int16)).all()
    # Malformed JSON: pass through unchanged, must NOT raise.
    assert (remap(y, "{bad json") == y).all()
    # Empty remap: no-op.
    assert (remap(y, "") == y).all()


# ── atomic checkpoint save ───────────────────────────────────────────────────
def test_save_torch_atomic_roundtrip_and_cleanup(tmp_path):
    import ogap.legacy as L
    path = tmp_path / "ckpt.pth"
    L.save_torch_atomic({"a": torch.zeros(3), "b": 7}, path)
    assert path.exists()
    assert torch.load(path, weights_only=False)["b"] == 7
    assert not list(tmp_path.glob("*.tmp")), "atomic save left a tmp file behind"


# ── save_nifti_like dtype guard ──────────────────────────────────────────────
def test_save_nifti_like_float_input_rounds_not_truncates(tmp_path):
    import ogap.legacy as L
    import nibabel as nib
    ref = tmp_path / "ref.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.int16), np.eye(4)), str(ref))
    out = tmp_path / "out.nii.gz"
    # 2.9999998 (float32 arithmetic noise) must round to 3 (ET), not truncate to 2.
    arr = np.full((4, 4, 4), np.float32(2.9999998), dtype=np.float32)
    L.save_nifti_like(arr, str(ref), str(out))
    saved = np.asarray(nib.load(str(out)).dataobj)
    assert saved.dtype == np.int16
    assert int(saved.flat[0]) == 3


# ── Bloch render finiteness on an all-zero PVE (pure-background crop) ─────────
def test_render_lowfield_all_zero_pve_is_finite():
    from ogap.augmentations.bloch_lowfield import render_lowfield
    out = render_lowfield(torch.zeros(3, 4, 4, 4), None)
    assert torch.isfinite(out).all()


# ── RANO Feret bidimensional measurement ─────────────────────────────────────
def test_feret_long_axis_uses_diagonal_not_bounding_box():
    import ogap.legacy as L
    # A 1-voxel-thick diagonal lesion: the longest diameter is the diagonal (~sqrt(2)*9),
    # NOT a 9-voxel axis span, and the perpendicular caliper width is ~0.
    coords = np.array([[i, i] for i in range(10)], dtype=np.int64)
    long_mm, perp_mm = L._feret_bidimensional_mm(coords, 1.0, 1.0)
    assert long_mm > 12.0
    assert perp_mm < 1.5


def test_region_clinical_measurements_feret_and_slice_position():
    import ogap.legacy as L
    pred = np.zeros((4, 8, 8), dtype=np.int16)
    pred[2, 2:6, 2:6] = 1                      # 4x4 TC block on slice z=2
    m = L._region_clinical_measurements(pred, (2.0, 1.0, 1.0))   # 2 mm slice thickness
    tc = m["TC"]
    assert tc["rano_slice_index"] == 2
    assert tc["rano_slice_position_mm"] == 4.0          # 2 * 2.0 mm
    assert tc["rano_long_mm"] > 4.0                     # diagonal of the 4x4 block (~4.24)


def test_student_does_not_emit_untrained_rano():
    from ogap.models import build_student
    m = build_student(8, 4, base=8).eval()
    with torch.no_grad():
        out = m(torch.randn(1, 8, 16, 16, 16), return_aux_outputs=True)
    aux = next((o for o in (out if isinstance(out, tuple) else (out,))
                if isinstance(o, dict)), None)
    assert aux is not None, "expected an aux_outputs dict"
    assert "rano" not in aux, "untrained rano_head output must not be emitted"
    assert "uncertainty_logits" in aux
