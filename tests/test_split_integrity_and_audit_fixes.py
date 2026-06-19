"""Regression tests for the 2026-06-15 NMI-robustness audit fixes.

Locks the load-bearing fixes so they cannot silently regress:
  * patient-level split keying (no longitudinal train/val/test leak)
  * 3-way held-out test split (test = reported-only, disjoint from val)
  * ET connected-component gate no longer keeps every component when no prob map
  * Hölder pseudo-divergence is non-negative (>= 0) at convergence
  * per-epoch HD95 uses NaN (not a misleading 0.0) when not computed
  * the production KD trainer defaults holder_alpha to the documented 1.5 (HPD),
    not 1.0 (KL)
"""
import dataclasses
import importlib.util
from pathlib import Path

import numpy as np
import torch

import ogap.legacy as L

_REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "bsm", _REPO / "workflow" / "build_split_manifests.py")
bsm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bsm)


# ───────────────────── split integrity (C-1 / C-2) ─────────────────────

def test_patient_id_strips_brats_timepoint_only():
    assert bsm.patient_id("BraTS-GLI-00020-001") == "BraTS-GLI-00020"
    assert bsm.patient_id("BraTS-GLI-00020-000") == "BraTS-GLI-00020"
    assert bsm.patient_id("BraTS-SSA-00002-000") == "BraTS-SSA-00002"
    # non-BraTS cohorts are 1:1 case:patient - must NOT be altered
    assert bsm.patient_id("MR_EGD-0001") == "MR_EGD-0001"   # Erasmus (would over-merge if stripped)
    assert bsm.patient_id("BT0001") == "BT0001"             # UTSW


def test_three_way_split_is_patient_disjoint():
    """Two timepoints of the same patient must land in the SAME band."""
    seed = 2026
    test_hi = 0.15
    val_hi = 0.30
    case_ids = [f"BraTS-GLI-{p:05d}-{t:03d}" for p in range(400) for t in (0, 1)]
    band = {}
    for cid in case_ids:
        s = bsm._hash_split(bsm.patient_id(cid), seed)
        band[cid] = "test" if s < test_hi else ("val" if s < val_hi else "train")
    for p in range(400):
        a = band[f"BraTS-GLI-{p:05d}-000"]
        b = band[f"BraTS-GLI-{p:05d}-001"]
        assert a == b, (p, a, b)
    assert set(band.values()) == {"train", "val", "test"}


def test_integrity_gate_catches_a_leak(tmp_path):
    """The script must abort (non-zero exit) when a case_id spans two splits."""
    import csv
    import subprocess
    import sys
    man = tmp_path / "m.csv"
    cols = ["case_id", "t1n", "t1c", "t2w", "t2f", "label", "dataset_tag", "field_strength"]
    base = dict.fromkeys(cols, "x")
    with open(man, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i in range(40):
            w.writerow(dict(base, case_id=f"BraTS-GLI-{i:05d}-000",
                            dataset_tag="brats_glioma", label="lbl"))
        # external row reusing an internal case_id -> guaranteed leak
        w.writerow(dict(base, case_id="BraTS-GLI-00000-000",
                        dataset_tag="erasmus", label="lbl"))
    proc = subprocess.run(
        [sys.executable, str(_REPO / "workflow" / "build_split_manifests.py"),
         "--manifest", str(man), "--out_dir", str(tmp_path / "out"), "--seed", "1"],
        capture_output=True, text=True, timeout=120)
    assert proc.returncode != 0
    assert "leak" in (proc.stdout + proc.stderr).lower()


# ───────────────────── reported-number bug fixes ─────────────────────

def test_et_gate_no_longer_keeps_tiny_components_without_probs():
    """With class_prob=None the ET gate falls back to SIZE-ONLY gating; a tiny
    (<et_min) ET island must be REMOVED, not kept (comp_conf used to default 1.0)."""
    pred = np.zeros((40, 40, 40), dtype=np.int64)
    pred[6:18, 6:18, 6:18] = 3          # large ET blob (1728 vox)
    pred[35:37, 35:37, 35:36] = 3       # tiny ET island (4 vox < et_min=10)
    out = L.connected_component_postprocessing(pred, class_prob=None)
    assert out[36, 36, 35] == 0         # island removed (size-only gating)
    assert out[10, 10, 10] == 3         # main blob kept


def test_holder_divergence_is_nonnegative_at_convergence():
    t = torch.tensor([[2.0, 0.5, -1.0, 0.3]])
    d2 = float(L.holder_divergence(t.clone(), t, temperature=1.0, alpha=2.0))
    assert 0.0 <= d2 < 1e-6             # proper at α=2 -> exactly 0 (was -1e-8)
    g = torch.Generator().manual_seed(0)
    for _ in range(20):
        s = torch.randn(1, 4, generator=g)
        assert float(L.holder_divergence(s, t, temperature=2.0, alpha=1.5)) >= 0.0


def test_brats_metrics_hd95_is_nan_when_skipped_not_zero():
    p = torch.zeros(8, 8, 8, dtype=torch.long)
    p[2:5, 2:5, 2:5] = 1
    m = L.brats_metrics(p, p.clone(), spacing=np.array([1.0, 1.0, 1.0]), compute_hd95=False)
    assert np.isnan(m["hd95_WT"]) and np.isnan(m["hd95_brats_mean"])   # NOT 0.0
    assert m["dice_WT"] > 0.99                                          # dice still real


def test_trainconfig_holder_alpha_default_is_the_hpd_method():
    fields = {f.name: f for f in dataclasses.fields(L.TrainConfig)}
    assert fields["holder_alpha"].default == 1.5   # documented HPD method, not 1.0 (KL)
