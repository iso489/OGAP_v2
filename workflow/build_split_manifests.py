#!/usr/bin/env python3
"""Build OGAP train / internal-val / external-val CSVs (2026 audit, Step 3).

Policy (PRE-REGISTERED — DO NOT CHANGE WITHOUT A PROTOCOL AMENDMENT):

  Training + internal validation pool:
      dataset_tag in {brats_glioma, utsw_glioma}
      → randomly partitioned 85/15 with a hash-stable seed.

  External validation #1 (HIC reference):  dataset_tag == 'erasmus'
  External validation #2 (LMIC stress):    dataset_tag == 'brats_africa'

Rationale:
  Erasmus and BraTS-Africa are NEVER seen at training time so the HIC-vs-LMIC
  statistical comparison (Scripts/eval_hic_vs_lmic.py) is an honest external
  validation in the TRIPOD+AI / SAGER sense.

Hash-stable split:
  Each case's train/val assignment is a deterministic function of
  sha256("<seed>:<case_id>"), so adding new training rows to the master
  manifest later does NOT shuffle existing rows between train and val.  This
  makes incremental data ingestion auditable.

Outputs (in --out_dir):
  train.csv                 — BraTS-2023 ∪ UTSW-glioma training rows
  val.csv                   — BraTS-2023 ∪ UTSW-glioma internal-val rows
  external_erasmus.csv      — Erasmus external-val rows
  external_brats_africa.csv — BraTS-Africa external-val rows
  split_provenance.json     — record of the seed, counts, policy version

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


def _hash_split(case_id: str, seed: int) -> float:
    """Deterministic per-case [0, 1) — hash-stable across incremental data adds."""
    h = hashlib.sha256(f"{seed}:{case_id}".encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


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
                         "out as internal validation. Default 0.15.")
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

    # ── Internal pool split (stratified by dataset_tag) ──────────────────
    train_rows: List[Dict[str, str]] = []
    val_rows: List[Dict[str, str]] = []
    per_tag_counts: Dict[str, Dict[str, int]] = {}
    for tag in sorted(INTERNAL_TAGS):
        tag_rows = by_tag.get(tag, [])
        tr, va = 0, 0
        for r in tag_rows:
            score = _hash_split(str(r.get("case_id", "")), args.seed)
            if score < args.internal_val_frac:
                val_rows.append(r); va += 1
            else:
                train_rows.append(r); tr += 1
        per_tag_counts[tag] = {"train": tr, "val": va, "total": tr + va}

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
    for tag in sorted(EXTERNAL_TAGS):
        _write(EXTERNAL_FILENAME[tag], by_tag.get(tag, []))

    # ── Integrity gate: no case_id appears in more than one split ────────
    sets = {
        "train":            {str(r.get("case_id", "")) for r in train_rows},
        "val":              {str(r.get("case_id", "")) for r in val_rows},
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
    print("OK — all splits have disjoint case_ids.")

    # ── Provenance ───────────────────────────────────────────────────────
    provenance = {
        "policy_version": POLICY_VERSION,
        "seed": args.seed,
        "internal_val_frac": args.internal_val_frac,
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
