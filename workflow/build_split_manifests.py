#!/usr/bin/env python3
"""Build OGAP train / internal-val / external-val CSVs (2026 audit, Step 3).

Policy (PRE-REGISTERED - DO NOT CHANGE WITHOUT A PROTOCOL AMENDMENT):

  Training + internal validation pool:
      dataset_tag in {brats_glioma, utsw_glioma}
      → randomly partitioned 85/15 with a hash-stable seed.

  External validation #1 (HIC reference):  dataset_tag == 'erasmus'
  External validation #2 (LMIC stress):    dataset_tag == 'brats_africa'

Rationale:
  Erasmus and BraTS-Africa are NEVER seen at training time so the HIC-vs-LMIC
  statistical comparison (Scripts/eval_hic_vs_lmic.py) is an honest external
  validation in the TRIPOD+AI / SAGER sense.

Hash-stable, PATIENT-level split:
  Each case's train/val assignment is a deterministic function of
  sha256("<seed>:<patient_id>"), so (a) adding new rows later does NOT shuffle
  existing rows between train and val (auditable incremental ingestion), and
  (b) every scan/timepoint of one PATIENT lands on the same side. The latter is
  essential: BraTS ids are ``BraTS-{cohort}-{patient}-{timepoint}``, so keying on
  the raw case_id (as the original code did) put 28 BraTS-GLI patients in BOTH
  train and val (same patient, timepoints -000 and -001) - a longitudinal leak
  that inflates internal-val metrics. See ``patient_id()``.

Outputs (in --out_dir):
  train.csv                 - BraTS-2023 ∪ UTSW-glioma training rows
  val.csv                   - BraTS-2023 ∪ UTSW-glioma internal-val rows
  external_erasmus.csv      - Erasmus external-val rows
  external_brats_africa.csv - BraTS-Africa external-val rows
  split_provenance.json     - record of the seed, counts, policy version

Integrity gate:
  Aborts if case_ids leak between any pair of splits.

Usage:
  python build_split_manifests.py \
      --manifest /home/.../master_manifest.csv \
      --out_dir  /home/.../Scripts \
      --internal_val_frac 0.15 \
      --seed 2026 \
      --strict
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

POLICY_VERSION = "2026-05-OGAP-v9.1-step3"
INTERNAL_TAGS = {"brats_glioma", "utsw_glioma"}
EXTERNAL_TAGS = {"erasmus", "brats_africa"}
EXTERNAL_FILENAME = {
    "erasmus":      "external_erasmus.csv",
    "brats_africa": "external_brats_africa.csv",
}


def _read(path: Path) -> List[Dict[str, str]]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _hash_split(key: str, seed: int) -> float:
    """Deterministic per-key [0, 1) - hash-stable across incremental data adds."""
    h = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def patient_id(case_id: str) -> str:
    """Map a case_id to its PATIENT id so all scans of one patient share a split.

    The split MUST be keyed on the patient, not the scan: BraTS ids are
    ``BraTS-{cohort}-{patient:5d}-{timepoint:3d}`` (e.g. ``BraTS-GLI-00020-001``),
    so a single patient with two timepoints has two distinct case_ids. Keying the
    hash split on the raw case_id therefore let the same patient land in BOTH
    train and val (a longitudinal leak: 28 BraTS-GLI patients in the 2026 split).
    We strip the trailing 3-digit BraTS timepoint so every timepoint of a patient
    hashes identically. Non-BraTS cohorts (UTSW ``BT####``, Erasmus
    ``MR_EGD-####``) are 1:1 case:patient, so the case_id is returned unchanged
    (verified: this rule collapses 0 Erasmus / 0 BraTS-Africa / 0 UTSW ids).
    """
    return re.sub(r"-\d{3}$", "", case_id) if case_id.startswith("BraTS-") else case_id


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--manifest", required=True, type=Path,
                    help="Master CSV with case_id, t1n, t1c, t2w, t2f, label, "
                         "dataset_tag, field_strength (and any other columns "
                         "consumed by MultiModalSegDataset).")
    ap.add_argument("--out_dir", required=True, type=Path,
                    help="Directory to write {train,val,external_*}.csv into.")
    ap.add_argument("--internal_val_frac", type=float, default=0.15,
                    help="Fraction of the BraTS-2023 ∪ UTSW-glioma pool held "
                         "out as internal validation (model selection). Default 0.15.")
    ap.add_argument("--internal_test_frac", type=float, default=0.15,
                    help="Fraction held out as the internal TEST set - used ONLY for "
                         "final reported numbers, NEVER for checkpoint selection or "
                         "tuning. 0.0 disables it (val-only legacy behaviour). "
                         "Default 0.15 (→ 70/15/15 train/val/test, by patient).")
    ap.add_argument("--seed", type=int, default=2026,
                    help="Seed for the hash-stable per-case split.")
    ap.add_argument("--strict", action="store_true",
                    help="Error out if any row has dataset_tag outside the "
                         "known four. By default unknown tags are logged and "
                         "dropped.")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not (0.0 < args.internal_val_frac < 1.0):
        sys.exit(f"--internal_val_frac must be in (0, 1); got {args.internal_val_frac}")
    if not (0.0 <= args.internal_test_frac < 1.0):
        sys.exit(f"--internal_test_frac must be in [0, 1); got {args.internal_test_frac}")
    if args.internal_val_frac + args.internal_test_frac >= 1.0:
        sys.exit("val_frac + test_frac must be < 1 (rows must remain for train).")

    rows = _read(args.manifest)
    by_tag: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    unknown: List[Dict[str, str]] = []
    for r in rows:
        tag = (r.get("dataset_tag") or "").strip().lower()
        if tag in INTERNAL_TAGS or tag in EXTERNAL_TAGS:
            by_tag[tag].append(r)
        else:
            unknown.append(r)

    print(f"Manifest:        {len(rows)} rows")
    for tag in sorted(INTERNAL_TAGS | EXTERNAL_TAGS):
        print(f"  {tag:20s}  n={len(by_tag.get(tag, []))}")
    if unknown:
        print(f"  {len(unknown)} rows with unrecognised dataset_tag: "
              f"{sorted({(r.get('dataset_tag') or '?') for r in unknown})}",
              file=sys.stderr)
        if args.strict:
            sys.exit("--strict and unknown tags present; aborting.")

    # ── Internal pool split (stratified by dataset_tag), patient-keyed ────
    # 3-way: TEST (reported numbers only) | VAL (model selection) | TRAIN, via
    # disjoint hash bands so all of a patient's timepoints share one band.
    train_rows: List[Dict[str, str]] = []
    val_rows: List[Dict[str, str]] = []
    test_rows: List[Dict[str, str]] = []
    per_tag_counts: Dict[str, Dict[str, int]] = {}
    _test_hi = args.internal_test_frac
    _val_hi = args.internal_test_frac + args.internal_val_frac
    for tag in sorted(INTERNAL_TAGS):
        tag_rows = by_tag.get(tag, [])
        tr, va, te = 0, 0, 0
        for r in tag_rows:
            # Key on the PATIENT, not the scan, so all of a patient's timepoints
            # land in the SAME band (no longitudinal leak across train/val/test).
            score = _hash_split(patient_id(str(r.get("case_id", ""))), args.seed)
            if score < _test_hi:
                test_rows.append(r); te += 1
            elif score < _val_hi:
                val_rows.append(r); va += 1
            else:
                train_rows.append(r); tr += 1
        per_tag_counts[tag] = {"train": tr, "val": va, "test": te, "total": tr + va + te}

    def _write(name: str, rows_out: List[Dict[str, str]]) -> None:
        if not rows_out:
            print(f"  skipping {name}: 0 rows")
            return
        out = args.out_dir / name
        # Preserve original column order from the manifest's first row.
        fieldnames = list(rows_out[0].keys())
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows_out)
        print(f"  wrote {out}  ({len(rows_out)} rows)")

    _write("train.csv", train_rows)
    _write("val.csv",   val_rows)
    _write("test.csv",  test_rows)
    for tag in sorted(EXTERNAL_TAGS):
        _write(EXTERNAL_FILENAME[tag], by_tag.get(tag, []))

    # ── Integrity gate: no case_id appears in more than one split ────────
    sets = {
        "train":            {str(r.get("case_id", "")) for r in train_rows},
        "val":              {str(r.get("case_id", "")) for r in val_rows},
        "test":             {str(r.get("case_id", "")) for r in test_rows},
        "ext_erasmus":      {str(r.get("case_id", "")) for r in by_tag.get("erasmus", [])},
        "ext_brats_africa": {str(r.get("case_id", "")) for r in by_tag.get("brats_africa", [])},
    }
    for a in sets:
        for b in sets:
            if a >= b:
                continue
            overlap = sets[a] & sets[b]
            if overlap:
                sys.exit(
                    f"FATAL: case_id leak between {a} and {b}: "
                    f"{sorted(overlap)[:5]}...  Aborting before any model "
                    f"training can see the leak."
                )
    # PATIENT-level disjointness - the leak the case_id check alone MISSES: a
    # BraTS patient's two timepoints have distinct case_ids but are the same
    # patient, so a case_id-disjoint split can still be patient-overlapping.
    pat_sets = {k: {patient_id(c) for c in v} for k, v in sets.items()}
    for a in pat_sets:
        for b in pat_sets:
            if a >= b:
                continue
            overlap = pat_sets[a] & pat_sets[b]
            if overlap:
                sys.exit(
                    f"FATAL: PATIENT leak between {a} and {b}: {sorted(overlap)[:5]}... "
                    f"(same patient - different timepoint/scan - in two splits). "
                    f"Aborting before any model training can see the leak."
                )
    print("OK - all splits have disjoint case_ids AND patients.")

    # ── Provenance ───────────────────────────────────────────────────────
    provenance = {
        "policy_version": POLICY_VERSION,
        "seed": args.seed,
        "internal_val_frac": args.internal_val_frac,
        "internal_test_frac": args.internal_test_frac,
        "split_key": "patient_id (BraTS timepoint stripped)",
        "manifest": str(args.manifest),
        "out_dir": str(args.out_dir),
        "counts": {
            "manifest_total": len(rows),
            "internal_per_tag": per_tag_counts,
            "external": {tag: len(by_tag.get(tag, [])) for tag in sorted(EXTERNAL_TAGS)},
            "unknown_dropped": len(unknown),
        },
        "internal_tags": sorted(INTERNAL_TAGS),
        "external_tags": sorted(EXTERNAL_TAGS),
        "generated_at_unix": int(time.time()),
    }
    (args.out_dir / "split_provenance.json").write_text(json.dumps(provenance, indent=2))
    print(f"  wrote {args.out_dir / 'split_provenance.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
