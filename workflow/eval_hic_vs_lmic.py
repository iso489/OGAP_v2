#!/usr/bin/env python3
"""HIC (Erasmus) vs LMIC (BraTS-Africa) external-validation statistical comparison.

This is Step 4 of the 2026 audit.  Inputs are the per-case metrics JSONs
produced by the standard ``evaluate`` subcommand for each external cohort;
outputs are a single report JSON and a per-case CSV that drop directly
into the TRIPOD+AI / SAGER external-validation section.

Pre-registration (DO NOT CHANGE WITHOUT A PROTOCOL AMENDMENT):

  H0:  median Dice (and HD95) on Erasmus equals median on BraTS-Africa.
  H1:  two-sided difference.
  Test:           Mann-Whitney U  (cohorts are independent, not paired).
  Correction:     Holm-Bonferroni over 6 primary metrics
                  (dice_{WT,TC,ET}, hd95_{WT,TC,ET}).
  Alpha:          0.05.
  Effect size:    rank-biserial correlation; bootstrap-CI on median diff.

Equity metrics (NIHMS RAND REACH / TRIPOD+AI 14 / SAGER 2022):

  - subgroup-disparity table by cohort and by field-strength bucket
  - concentration index of Dice_WT ranked by field-strength availability
  - sex breakdown per cohort if metadata TSVs are supplied

Usage:
  python eval_hic_vs_lmic.py \
      --hic_per_case  /…/utsw_full_evaluation_tta/erasmus/per_case_metrics.json \
      --lmic_per_case /…/utsw_full_evaluation_tta/brats_africa/per_case_metrics.json \
      --out_dir       /…/utsw_full_evaluation_tta/hic_vs_lmic \
      [--hic_metadata_csv  /…/erasmus_metadata.csv] \
      [--lmic_metadata_csv /…/brats_africa_metadata.csv] \
      [--alpha 0.05]
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import stats

# Reuse the compliance helpers - they encode the TRIPOD+AI / SAGER / NIHMS
# definitions verbatim, so re-implementing them here would risk drift.
_REPO_ROOT = Path(__file__).resolve().parent.parent  # parent of workflow/
sys.path.insert(0, str(_REPO_ROOT))
try:
    from ogap_compliance import (  # noqa: E402
        compute_fairness_metrics,
        compute_concentration_index,
        compute_sex_breakdown,
    )
except ImportError as exc:  # pragma: no cover - import-environment guard
    raise RuntimeError(
        f"Could not import ogap_compliance from the repo root ({_REPO_ROOT}). "
        "Run this from a checkout where ogap_compliance.py sits at the repository "
        "root (one level above workflow/)."
    ) from exc

PRIMARY_METRICS = ("dice_WT", "dice_TC", "dice_ET", "hd95_WT", "hd95_TC", "hd95_ET")
HIGHER_IS_BETTER = {"dice_WT", "dice_TC", "dice_ET"}


def _load_per_case(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load per-case metrics JSON written by the evaluate subcommand."""
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "per_case" in data:
        per_case = data["per_case"]
    elif isinstance(data, list):
        per_case = data
    else:
        raise ValueError(
            f"{path} is not in the expected per-case JSON shape "
            "(top-level 'per_case' list or a bare list)."
        )
    return {row["case_id"]: row for row in per_case}


def _classify_field(value: Any) -> str:
    """Bucket field strength for the concentration-index sort key."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if v < 0.5:
        return "ulf"      # ultra-low: 0.064-0.3 T
    if v < 1.0:
        return "low"      # 0.55, 0.7 T
    if v < 2.5:
        return "1p5T"     # 1.0-1.5 T
    return "3T_plus"


# Higher rank = more resourced site (high-field availability).
FIELD_RANK = {"ulf": 1, "low": 2, "1p5T": 3, "3T_plus": 4, "unknown": 2}


def _mann_whitney(x: np.ndarray, y: np.ndarray, *, higher_is_better: bool,
                  n_boot: int = 1000, seed: int = 2026) -> Dict[str, Any]:
    """Two-sided Mann-Whitney U + rank-biserial + bootstrap median-diff CI."""
    x = np.asarray([v for v in x if np.isfinite(v)], dtype=np.float64)
    y = np.asarray([v for v in y if np.isfinite(v)], dtype=np.float64)
    if len(x) < 3 or len(y) < 3:
        return {
            "n_hic": int(len(x)), "n_lmic": int(len(y)),
            "p_value": float("nan"), "U": float("nan"),
            "median_hic": float("nan"), "median_lmic": float("nan"),
            "effect_size_rbc": float("nan"),
            "ci95_diff_lo": float("nan"), "ci95_diff_hi": float("nan"),
            "alternative": "two-sided",
            "higher_is_better": higher_is_better,
            "warning": "n < 3 in one cohort; test undefined",
        }
    U, p = stats.mannwhitneyu(x, y, alternative="two-sided")
    # Rank-biserial correlation = 1 - 2U/(n1*n2); interpret as effect size.
    rbc = 1.0 - 2.0 * U / (len(x) * len(y))
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        bx = rng.choice(x, size=len(x), replace=True)
        by = rng.choice(y, size=len(y), replace=True)
        diffs[i] = np.median(bx) - np.median(by)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "n_hic": int(len(x)), "n_lmic": int(len(y)),
        "median_hic": float(np.median(x)), "median_lmic": float(np.median(y)),
        "mean_hic":   float(np.mean(x)),   "mean_lmic":   float(np.mean(y)),
        "U": float(U), "p_value": float(p),
        "effect_size_rbc": float(rbc),
        "median_diff_hic_minus_lmic": float(np.median(x) - np.median(y)),
        "ci95_diff_lo": float(lo), "ci95_diff_hi": float(hi),
        "alternative": "two-sided",
        "higher_is_better": higher_is_better,
    }


def _holm_bonferroni(p_values: List[float]) -> List[float]:
    """Step-down Holm-Bonferroni adjusted p-values for k tests."""
    m = len(p_values)
    if m == 0:
        return []
    order = np.argsort(p_values)
    adj = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        p = p_values[idx]
        if not np.isfinite(p):
            adj[idx] = float("nan")
            continue
        v = min(1.0, p * (m - rank))
        running_max = max(running_max, v)
        adj[idx] = running_max
    return adj


def _interpret(tests: Dict[str, Dict[str, Any]], alpha: float) -> str:
    sig = [k for k, v in tests.items()
           if v.get("significant_after_correction")]
    if not sig:
        return (
            f"No primary metric showed a statistically-significant HIC vs LMIC "
            f"difference after Holm-Bonferroni correction (alpha={alpha})."
        )
    lines = [f"Significant HIC vs LMIC differences after Holm correction (alpha={alpha}):"]
    for k in sig:
        v = tests[k]
        if v["higher_is_better"]:
            direction = "HIC > LMIC" if v["median_diff_hic_minus_lmic"] > 0 else "LMIC > HIC"
        else:
            # Lower is better for HD95.
            direction = "HIC better (lower HD95)" if v["median_diff_hic_minus_lmic"] < 0 else "LMIC better (lower HD95)"
        lines.append(
            f"  {k}: {direction}, median diff (HIC - LMIC) = "
            f"{v['median_diff_hic_minus_lmic']:+.4f} "
            f"(95% CI [{v['ci95_diff_lo']:+.4f}, {v['ci95_diff_hi']:+.4f}]), "
            f"p_holm = {v['p_value_holm']:.2e}, "
            f"rank-biserial = {v['effect_size_rbc']:+.3f}"
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--hic_per_case",  required=True, type=Path,
                    help="per-case metrics JSON from `evaluate` on Erasmus.")
    ap.add_argument("--lmic_per_case", required=True, type=Path,
                    help="per-case metrics JSON from `evaluate` on BraTS-Africa.")
    ap.add_argument("--out_dir",       required=True, type=Path)
    ap.add_argument("--hic_metadata_csv",  type=Path, default=None,
                    help="optional CSV with a sex/gender column (SAGER 2022).")
    ap.add_argument("--lmic_metadata_csv", type=Path, default=None)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--bootstrap_n", type=int, default=1000,
                    help="Bootstrap resamples for the median-diff CI.")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    hic = _load_per_case(args.hic_per_case)
    lmic = _load_per_case(args.lmic_per_case)
    print(f"HIC (Erasmus):       n={len(hic)}")
    print(f"LMIC (BraTS-Africa): n={len(lmic)}")

    # ── Primary tests ────────────────────────────────────────────────
    tests: Dict[str, Dict[str, Any]] = {}
    raw_p: List[float] = []
    test_keys: List[str] = []
    for m in PRIMARY_METRICS:
        higher_is_better = m in HIGHER_IS_BETTER
        x = np.array([r[m] for r in hic.values()  if m in r], dtype=np.float64)
        y = np.array([r[m] for r in lmic.values() if m in r], dtype=np.float64)
        result = _mann_whitney(x, y, higher_is_better=higher_is_better,
                               n_boot=args.bootstrap_n)
        tests[m] = result
        raw_p.append(result["p_value"])
        test_keys.append(m)
    adj = _holm_bonferroni(raw_p)
    for k, p in zip(test_keys, adj):
        tests[k]["p_value_holm"] = float(p) if np.isfinite(p) else float("nan")
        tests[k]["significant_after_correction"] = bool(np.isfinite(p) and p < args.alpha)

    # ── Pool for equity metrics ──────────────────────────────────────
    pooled_cases: List[Dict[str, Any]] = []
    for r in hic.values():
        r2 = dict(r); r2["__cohort__"] = "HIC_erasmus"
        r2["__field_bucket__"] = _classify_field(r.get("field_strength"))
        pooled_cases.append(r2)
    for r in lmic.values():
        r2 = dict(r); r2["__cohort__"] = "LMIC_brats_africa"
        r2["__field_bucket__"] = _classify_field(r.get("field_strength"))
        pooled_cases.append(r2)

    fairness: Dict[str, Any] = {}
    for m in ("dice_WT", "dice_TC", "dice_ET"):
        # Disparity across cohorts (HIC vs LMIC).
        vals, cohorts = [], []
        for r in pooled_cases:
            v = r.get(m)
            if v is None or not np.isfinite(float(v)):
                continue
            vals.append(float(v)); cohorts.append(r["__cohort__"])
        fairness[f"{m}_cohort_disparity"] = compute_fairness_metrics(
            vals, cohorts, metric_name=m, min_group_n=10,
        )
        # Disparity across field-strength buckets (a separate equity lens).
        vals_f, buckets = [], []
        for r in pooled_cases:
            v = r.get(m)
            if v is None or not np.isfinite(float(v)):
                continue
            vals_f.append(float(v)); buckets.append(r["__field_bucket__"])
        fairness[f"{m}_field_strength_disparity"] = compute_fairness_metrics(
            vals_f, buckets, metric_name=m, min_group_n=5,
        )

    # Concentration index over field-strength rank for Dice_WT.
    h_vals, r_vals = [], []
    for r in pooled_cases:
        v = r.get("dice_WT")
        if v is None or not np.isfinite(float(v)):
            continue
        h_vals.append(float(v))
        r_vals.append(float(FIELD_RANK[r["__field_bucket__"]]))
    concentration = compute_concentration_index(
        np.asarray(h_vals, dtype=np.float64),
        np.asarray(r_vals, dtype=np.float64),
    )

    # SAGER-style sex breakdown per cohort (optional).
    sex_breakdown = None
    if args.hic_metadata_csv or args.lmic_metadata_csv:
        sex_breakdown = {}
        if args.hic_metadata_csv:
            sex_breakdown["HIC_erasmus"] = compute_sex_breakdown(args.hic_metadata_csv)
        if args.lmic_metadata_csv:
            sex_breakdown["LMIC_brats_africa"] = compute_sex_breakdown(args.lmic_metadata_csv)

    # ── Report ───────────────────────────────────────────────────────
    report = {
        "hypothesis": {
            "h0": "median Dice / HD95 on Erasmus (HIC) equals median on BraTS-Africa (LMIC)",
            "h1": "two-sided difference",
            "test": "Mann-Whitney U (independent cohorts)",
            "alpha": args.alpha,
            "correction": "Holm-Bonferroni over 6 primary metrics",
            "effect_size": "rank-biserial correlation",
            "ci": f"{args.bootstrap_n}-resample bootstrap on median (HIC - LMIC)",
        },
        "n_hic": len(hic),
        "n_lmic": len(lmic),
        "primary_tests": tests,
        "equity_metrics": {
            "subgroup_disparity": fairness,
            "field_strength_concentration_index_dice_WT": concentration,
            "sex_breakdown": sex_breakdown,
        },
        "interpretation": _interpret(tests, args.alpha),
    }
    out_json = args.out_dir / "hic_vs_lmic_report.json"
    out_json.write_text(json.dumps(report, indent=2, default=str))

    # Per-case CSV for the paper appendix.
    out_csv = args.out_dir / "hic_vs_lmic_per_case.csv"
    if pooled_cases:
        all_keys = set()
        for r in pooled_cases:
            all_keys.update(r.keys())
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=sorted(all_keys))
            w.writeheader()
            w.writerows(pooled_cases)

    print(f"[hic_vs_lmic] wrote {out_json}")
    print(f"[hic_vs_lmic] wrote {out_csv}")
    print()
    print(report["interpretation"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
