#!/usr/bin/env python3
"""Build the OGAP master manifest from the four raw glioma datasets on /project.

Emits a CSV in the exact MultiModalSegDataset schema (see ogap/legacy.py
``MultiModalSegDataset`` docstring):

    case_id, t1n, t1c, t2w, t2f, label, dataset_tag, field_strength, label_remap

``workflow/build_split_manifests.py`` then partitions this into
train.csv / val.csv / external_erasmus.csv / external_brats_africa.csv.

────────────────────────────────────────────────────────────────────────────
Label policy (VERIFIED against actual voxel values on 2026-06-13; do not change
without re-verifying and amending the analysis protocol):

  brats_glioma (BraTS-2023, 1251 cases)
      seg files are ALREADY {0,1,2,3} on disk → label_remap="" (no remap).
  brats_africa (BraTS-Africa Glioma sheet, 95 cases) [EXTERNAL]
      seg files are ALREADY {0,1,2,3} → label_remap="".
  utsw_glioma (UT Southwestern, 625 cases on disk)
      modalities = brain_{t1,t1ce,t2,flair}.nii.gz (all 240x240x155, SRI space).
      label = rtumorseg_manual_correction.nii.gz (the manual correction
              RESAMPLED onto the 240^3 modality grid; the un-prefixed
              tumorseg_manual_correction is in a cropped per-case geometry and
              is NOT voxel-aligned to the modalities for ~44% of cases) - but
              ONLY when its label palette ⊆ {0,1,2,4} (standard BraTS/FeTS
              convention; the 4→3 ET remap is then applied automatically by
              MultiModalSegDataset._canonicalise_label_values).
      Otherwise fall back to tumorseg_FeTS.nii.gz (Federated Tumor
      Segmentation; uniformly {0,1,2,4} for all 625 cases). 121/362 manual
      segmentations use an UNDOCUMENTED extended palette containing labels
      3 and/or 5 (real annotated regions, not interpolation artifacts) whose
      mapping to the canonical {0=BG,1=NCR,2=ED,3=ET} scheme is unknown; those
      cases use FeTS rather than silently colliding label-3 with ET or crashing
      on label-5. label_remap="" in both cases.
  erasmus (Erasmus EGD, 774 cases) [EXTERNAL]
      modalities = {1_T1,2_T1GD,3_T2,4_FLAIR}/NIFTI/*.nii.gz
      label = 5_MASK/NIFTI/MASK.nii.gz  - binary whole-tumour {0,1}.
      label_remap='{"1":"1"}'  (explicit identity; only the WT/Tumor region is
      defined on Erasmus, per the pre-registered analysis plan).

Field strength (Tesla; empty string when unknown):
  BraTS-2023   = 3.0   (study-level assumption; US-acquired, no per-case meta)
  BraTS-Africa = 1.5   (data-provider statement)
  UTSW         = METADATA/USTW_glioma TSV 'Scanner Strength' ('Not Reported' → "")
  Erasmus      = per-case metadata.json Scan_characteristics.Field (<=0 → "")

dataset_tag values match build_split_manifests.py INTERNAL/EXTERNAL tags:
  brats_glioma + utsw_glioma → train/internal-val pool
  erasmus + brats_africa     → external validation (never trained on)

A case is SKIPPED (and logged) if any of the 4 modalities or the chosen label
is missing, if the chosen label has values outside {0,1,2,3,4}, or if the
chosen label is entirely background (no tumour) - each is a data-integrity
problem that should not enter training silently.
────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import nibabel as nib

FIELDNAMES = ["case_id", "t1n", "t1c", "t2w", "t2f",
              "label", "dataset_tag", "field_strength", "label_remap"]

STANDARD_PALETTE = {0, 1, 2, 4}          # BraTS/FeTS source convention (4→3 auto)
ALLOWED_AFTER_AUTO = {0, 1, 2, 3, 4}     # canonicaliser tolerates these (4→3); 5+ crashes


def label_values(path: str) -> Set[int]:
    arr = np.asanyarray(nib.load(path).dataobj)
    return {int(v) for v in np.unique(arr)}


def load_utsw_field_strength(tsv_path: Path) -> Dict[str, str]:
    """Map UTSW 'Subject ID' (BT####) → field strength string ('' if unknown)."""
    out: Dict[str, str] = {}
    if not tsv_path.exists():
        print(f"[utsw] WARNING: metadata TSV not found: {tsv_path}", file=sys.stderr)
        return out
    with open(tsv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            sid = (row.get("Subject ID") or "").strip()
            if not sid:
                continue
            raw = (row.get("Scanner Strength") or "").strip()
            try:
                val = float(raw)
                out[sid] = f"{val:g}"
            except ValueError:
                out[sid] = ""   # 'Not Reported' / blank
    return out


def erasmus_field_strength(case_dir: Path) -> str:
    mj = case_dir / "metadata.json"
    if not mj.exists():
        return ""
    try:
        meta = json.loads(mj.read_text())
        field = meta.get("Scan_characteristics", {}).get("Field")
        if field is None:
            return ""
        val = float(field)
        return f"{val:g}" if val > 0 else ""
    except Exception:
        return ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path,
                    default=Path("/project/def-rdiaz/ilyaso/Datasets"))
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent.parent / "master_manifest.csv")
    args = ap.parse_args()

    D = args.data_root
    rows: List[Dict[str, str]] = []
    skipped: List[Tuple[str, str]] = []
    label_source_counter: Counter = Counter()
    fs_known: Counter = Counter()

    def add(case_id, t1n, t1c, t2w, t2f, label, tag, fs, remap, *, check_label=True,
            src_note="") -> None:
        for p in (t1n, t1c, t2w, t2f, label):
            if not os.path.exists(p):
                skipped.append((case_id, f"missing file: {os.path.basename(p)}"))
                return
        if check_label:
            vals = label_values(label)
            bad = sorted(v for v in vals if v not in ALLOWED_AFTER_AUTO)
            if bad:
                skipped.append((case_id, f"label values {bad} outside {{0,1,2,3,4}}"))
                return
            if vals <= {0}:
                skipped.append((case_id, "label is entirely background (no tumour)"))
                return
        rows.append({"case_id": case_id, "t1n": t1n, "t1c": t1c, "t2w": t2w,
                     "t2f": t2f, "label": label, "dataset_tag": tag,
                     "field_strength": fs, "label_remap": remap})
        if src_note:
            label_source_counter[src_note] += 1
        if fs:
            fs_known[tag] += 1

    # ── BraTS-2023 (brats_glioma) ──────────────────────────────────────────
    for c in sorted((D / "BraTS_2023").glob("BraTS-GLI-*")):
        if not c.is_dir():
            continue
        cid = c.name
        add(cid, *[str(c / f"{cid}-{m}.nii.gz") for m in ("t1n", "t1c", "t2w", "t2f")],
            str(c / f"{cid}-seg.nii.gz"), "brats_glioma", "3.0", "",
            check_label=False, src_note="brats_glioma:seg")

    # ── BraTS-Africa (brats_africa) [EXTERNAL] ─────────────────────────────
    for c in sorted((D / "BraTS_Africa").glob("BraTS-SSA-*")):
        if not c.is_dir():
            continue
        cid = c.name
        add(cid, *[str(c / f"{cid}-{m}.nii.gz") for m in ("t1n", "t1c", "t2w", "t2f")],
            str(c / f"{cid}-seg.nii.gz"), "brats_africa", "1.5", "",
            check_label=False, src_note="brats_africa:seg")

    # ── UTSW (utsw_glioma) ─────────────────────────────────────────────────
    utsw_fs = load_utsw_field_strength(
        D / "METADATA" / "USTW_glioma" / "UTSW_Glioma_Metadata-2-1.tsv")
    utsw_dirs = sorted([c for c in (D / "USTW_glioma").glob("BT*") if c.is_dir()])
    print(f"[utsw] scanning {len(utsw_dirs)} case dirs + choosing manual/FeTS labels ...",
          flush=True)
    for c in utsw_dirs:
        cid = c.name
        mods = [str(c / f"brain_{m}.nii.gz") for m in ("t1", "t1ce", "t2", "flair")]
        rmanual = c / "rtumorseg_manual_correction.nii.gz"
        fets = c / "tumorseg_FeTS.nii.gz"
        label = None
        note = ""
        if rmanual.exists():
            try:
                if label_values(str(rmanual)) <= STANDARD_PALETTE:
                    label, note = str(rmanual), "utsw:manual(rtumorseg)"
            except Exception as exc:
                skipped.append((cid, f"rmanual unreadable: {exc}"))
        if label is None and fets.exists():
            label, note = str(fets), "utsw:FeTS(fallback)"
        if label is None:
            skipped.append((cid, "no usable manual or FeTS label"))
            continue
        fs = utsw_fs.get(cid, "")
        add(cid, *mods, label, "utsw_glioma", fs, "", check_label=True, src_note=note)

    # ── Erasmus (erasmus) [EXTERNAL] ───────────────────────────────────────
    for c in sorted((D / "Erasmus_Glioma").glob("MR_EGD-*")):
        if not c.is_dir():
            continue
        cid = c.name
        mods = [str(c / "1_T1" / "NIFTI" / "T1.nii.gz"),
                str(c / "2_T1GD" / "NIFTI" / "T1GD.nii.gz"),
                str(c / "3_T2" / "NIFTI" / "T2.nii.gz"),
                str(c / "4_FLAIR" / "NIFTI" / "FLAIR.nii.gz")]
        label = str(c / "5_MASK" / "NIFTI" / "MASK.nii.gz")
        fs = erasmus_field_strength(c)
        add(cid, *mods, label, "erasmus", fs, '{"1":"1"}',
            check_label=False, src_note="erasmus:MASK")

    # ── Write ──────────────────────────────────────────────────────────────
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(rows)

    # ── Summary ──────────────────────────────────────────────────────────
    by_tag = Counter(r["dataset_tag"] for r in rows)
    print("\n" + "=" * 64)
    print(f"master_manifest: {len(rows)} cases written → {args.out}")
    for tag in ("brats_glioma", "utsw_glioma", "erasmus", "brats_africa"):
        n = by_tag.get(tag, 0)
        print(f"  {tag:14s} n={n:5d}   field_strength known: {fs_known.get(tag, 0)}/{n}")
    print("  UTSW label source:")
    for k, v in label_source_counter.items():
        if k.startswith("utsw"):
            print(f"      {k:26s} {v}")
    print(f"  SKIPPED: {len(skipped)} cases")
    skip_reasons = Counter(reason for _, reason in skipped)
    for reason, n in skip_reasons.most_common():
        print(f"      {n:4d}  {reason}")
    if skipped:
        print("  (first 12 skipped):")
        for cid, reason in skipped[:12]:
            print(f"      {cid}: {reason}")
    print("=" * 64)

    # Provenance JSON
    prov = {
        "data_root": str(D),
        "n_written": len(rows),
        "by_tag": dict(by_tag),
        "field_strength_known": dict(fs_known),
        "utsw_label_source": {k: v for k, v in label_source_counter.items()},
        "n_skipped": len(skipped),
        "skip_reasons": dict(skip_reasons),
        "skipped_cases": skipped,
        "label_policy": "see module docstring; verified 2026-06-13",
    }
    prov_path = args.out.with_name("master_manifest_provenance.json")
    prov_path.write_text(json.dumps(prov, indent=2))
    print(f"provenance → {prov_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
