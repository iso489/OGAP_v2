# OGAP - Reader study, on-site feasibility & compression non-inferiority: analysis plan

**Date:** 2026-06-15  **Status:** PRE-REGISTRATION DRAFT - *freeze before any African-cohort
or reader data is unblinded.* **Version:** 0.1

This plan operationalises the three planned validation efforts on top of the existing OGAP
evaluation code. It is written to satisfy the external-validation, equity, deployment and
clinical-evaluation requirements of the target venues (Radiology: AI / Lancet Digital Health /
npj Digital Medicine, or MedIA/TMI for the methods framing).

## 0. Reporting-guideline alignment

| Guideline | Scope here |
|---|---|
| **TRIPOD+AI (2024)** | model description + external validation (Arm A) |
| **CLAIM 2024** | the imaging-AI technical paper checklist (Arms A+B) |
| **DECIDE-AI** | early-stage *live clinical evaluation* on real African hardware (Arms B+C) |
| **SAGER (2022)** | sex/gender reporting (already wired in `eval_hic_vs_lmic.py`) |

## 1. Cohorts, models, and code map

- **HIC cohort:** Erasmus. **LMIC cohort:** BraTS-Africa. Field-strength buckets: `ulf` (<0.5 T,
  incl. 0.064 T Swoop), `low` (0.55-0.7 T), `1p5T` (1.0-1.5 T), `3T_plus` (≥2.5 T) - per
  `workflow/eval_hic_vs_lmic.py::_classify_field`.
- **Models compared (fixed roster):**
  1. `teacher` - performance ceiling / KD source (NOT deployed).
  2. `student_fp32` - full-precision deployable model.
  3. `student_int8` - **the deployed artifact**, produced via the **ONNX export + onnxruntime/
     OpenVINO** path. The torch `quantize_student_int8` proxy is now `strict=True` and refuses
     conv models, so it can never silently supply an "INT8" number for the paper.

| Analysis | Implementing function |
|---|---|
| Paired two-method difference | `ogap.evaluation.stats.paired_significance_table` |
| **Compression non-inferiority** | `ogap.evaluation.stats.noninferiority_table` *(new)* |
| 15-way modality robustness | `ogap.evaluation.modality_robustness.evaluate_modality_combinations` → `…modality_robustness_table` / `…full_significance_sweep` / `…modality_degradation_analysis` |
| Subgroup risk disparity | `ogap.evaluation.equity.subgroup_equity_table`, `…fairness_without_harm_check` |
| HIC-vs-LMIC external validation | `workflow/eval_hic_vs_lmic.py` |

> **Metric-name convention:** `stats.py`/`modality_robustness.py`/`equity.py` use
> `dsc_{wt,tc,et}` / `hd95_{wt,tc,et}`; `eval_hic_vs_lmic.py` consumes `dice_{WT,TC,ET}` /
> `hd95_{WT,TC,ET}`. Keep one mapping in the harness so per-case JSONs feed both without drift.

---

## 2. Study Arm A - 15-way modality robustness × LMIC/HIC disparity

**Question.** How far does the deployed student degrade as MRI sequences are missing, and is that
degradation *worse for the LMIC cohort* than the HIC cohort?

**Combinations (now 15, was 11).** All non-empty subsets of {T1, T1ce, T2, FLAIR}:
C(4,1)=4 single-sequence + C(4,2)=6 + C(4,3)=4 + C(4,4)=1. The four single-sequence cases are the
realistic LMIC stress test (a site with only T1 or only FLAIR) and are now in
`ALL_COMBINATIONS`.

**Case-ID linkage.** Run `evaluate_modality_combinations` with a 3-tuple loader
`(volume, label, case_ids)` so each result keys on the real case id and joins to per-case
metadata (cohort, field bucket, sex). Per-case metrics JSON schema (synthetic):

```json
{"case_id": "BraTS-Africa-007", "cohort": "LMIC", "field_strength": 0.064,
 "dice_WT": 0.88, "dice_TC": 0.81, "dice_ET": 0.77,
 "hd95_WT": 4.2, "hd95_TC": 6.0, "hd95_ET": 8.1}
```

**Primary metric:** ET Dice (`dsc_et`) - the most modality-sensitive region. **Secondary:** WT, TC
Dice and HD95.

**Missing-channel fill:** `fill="zero"` for the primary analysis (matches train-time modality
dropout); report `fill="mean"` as a sensitivity analysis. **HD95 absent-region policy:**
`hd95_empty_policy="penalty"` for the degradation study (absent regions are the signal).

**Significance.** Per combination: Friedman omnibus across the 3 models, then Holm-corrected
pairwise Wilcoxon (this is `full_significance_sweep`’s per-combo FWER block). **Multiplicity
decision (pre-register):** the 4-modality and the four 1-modality combos are the *confirmatory*
family (global Holm across those 5 × 3 metrics); the 6 two- and 4 three-modality combos are
*exploratory* and labelled as such. `full_significance_sweep` does **not** impose a single FWER
across all 15 combos - do not claim it does.

**Equity crosstab (the headline).** For each (combo, region) compute the LMIC-vs-HIC gap with
`subgroup_equity_table(reference_group="HIC")` (mean, bootstrap CI, risk disparity Δ, Mann-Whitney
gap p). Then `fairness_without_harm_check` (baseline = no-physics-aug model, candidate = shipped
model) to demonstrate disparity reduction **without harming** the HIC group. Also report the
field-strength **concentration index** from `eval_hic_vs_lmic.py`.

---

## 3. Study Arm B - INT8 compression non-inferiority (technical + on-site)

**Claim to support:** "INT8 compression preserves segmentation accuracy." A non-significant
two-sided difference does **not** establish this - non-inferiority with a *pre-specified margin*
does.

**INT8 source:** ONNX + onnxruntime (or OpenVINO) **only**. The torch dynamic-quant proxy is
strict-by-default and excluded from reported numbers.

**Margins (pre-register; CI-based decision via `noninferiority_table`):**

| Metric | Margin δ | Rationale |
|---|---|---|
| Dice WT/TC/ET | **0.01** Dice | below typical inter-rater Dice noise; clinically negligible |
| HD95 WT/TC/ET | **2.0 mm** | ≤ 2× nominal voxel; sub-clinical boundary shift |

**Decision rule.** For each metric, `noninferiority_table(results_reference=student_fp32,
results_candidate=student_int8, margins=…)` reports the median *loss* (positive = INT8 worse) and
its BCa 95% CI. INT8 is declared **non-inferior** for that metric iff `ci_high < margin`
(a conservative CI rule = one-sided test at α/2). Run this **twice**: (a) on the held-out internal
val set, and (b) on each African site's local hardware (numbers must reproduce on-device).

**On-site hardware feasibility (DECIDE-AI).** On each site's actual machine, record per case:

| Metric | Acceptance threshold (pre-register) |
|---|---|
| Inference latency (wall-clock, CPU) | report median + IQR; target site-defined |
| Peak RAM | must fit the site's smallest machine |
| Model size on disk (INT8 .onnx) | report; the compression headline |
| Energy / thermals (if measurable) | report |
| Run completion / failure rate | report; any crash is a reportable event |
| INT8-vs-FP32 Dice on-device | non-inferior per the margins above |

---

## 4. Study Arm C - Reader study

**Design.** Multi-reader review of segmentation outputs. Pre-specify: number of readers
(≥3 neuroradiologists/neuro-oncologists recommended for agreement stats), case sample (stratified
by cohort × field bucket; power-justified - see below), **blinding** (reader blind to model
identity and to cohort where feasible), and **randomised** case/model presentation order.

**Scoring instrument (per reader × case × model):**

```
reader_id, case_id, cohort, field_bucket, model(fp32|int8),
acceptability_1to5,            # 5=use as-is … 1=unusable
edit_required(0/1),            # would you correct before clinical use?
region_flags(WT/TC/ET free-text), notes
```

**Endpoints.**
- **Primary:** proportion of cases rated *clinically acceptable* (acceptability ≥ 4). Pre-specify a
  **non-inferiority margin** for INT8 vs FP32 acceptability (e.g. ≤ 5 percentage-point drop) and,
  separately, an absolute acceptability floor for the LMIC cohort.
- **Secondary:** inter-reader agreement; per-region acceptability; edit-required rate.

**Statistics.**
- **Acceptability (ordinal, repeated measures):** a **cumulative-link mixed model** (proportional-
  odds) with fixed effects for model (INT8 vs FP32), cohort (LMIC vs HIC) and their interaction,
  and **random intercepts for reader and for case**. Do *not* average Likert to a mean and t-test.
- **Inter-reader agreement:** **ICC(2,k)** for the ordinal score, plus **Krippendorff's α**
  (ordinal) or **Fleiss'/weighted Cohen's κ** for the acceptable/not dichotomy.
- **AI-assisted vs unassisted (only if that arm is run):** a **multi-reader multi-case (MRMC)**
  design analysed with **Obuchowski-Rockette / DBM** - this is the design that supports "the AI
  *helps* clinicians," and is a stronger claim than standalone acceptability.

**Sample size.** Power the primary non-inferiority comparison (acceptability proportion) at the
chosen margin and α=0.05; inflate for the reader random effect (design effect from the expected
ICC). Record the assumptions here before recruiting.

---

## 5. Multiplicity & pre-registration freeze

- **Confirmatory families:** (Arm A) full + single-modality combos × 3 regions, global Holm;
  (Arm B) the 6 INT8-vs-FP32 non-inferiority tests; (Arm C) the primary acceptability endpoint.
- **Exploratory (labelled, not corrected globally):** 2- and 3-modality combos; per-region reader
  breakdowns; field-bucket sub-analyses below `min_group_n`.
- **Freeze:** margins (§3), the model roster (§1), the metric mapping, and the endpoint definitions
  are frozen at v1.0 before unblinding. Any later change is a documented protocol amendment, as in
  `eval_hic_vs_lmic.py`’s pre-registration block.

## 6. Deliverables

1. `…/modality_robustness/` - 15-combo tables + per-combo significance + degradation curves.
2. `…/hic_vs_lmic/hic_vs_lmic_report.json` + per-case CSV (existing).
3. `…/noninferiority/int8_vs_fp32_{internal,onsite}.csv` - `noninferiority_table` output.
4. `…/onsite/<site>_hardware_feasibility.json` - DECIDE-AI metrics.
5. `…/reader_study/` - raw scores, CLMM fit, agreement statistics.
