"""ogap_compliance.py - Reporting-checklist compliance utilities for OGAP v2.

This module fills the gaps required by four reporting checklists for the OGAP
glioma-segmentation manuscript and pipeline:

    • CLAIM 2024  (Tejani et al., Radiology AI 2024)
        - Checklist for Artificial Intelligence in Medical Imaging - 44 items.
    • TRIPOD+AI   (Collins et al., BMJ 2024)
        - Transparent Reporting of a multivariable prediction model - 27 items.
    • SAGER 2022  (Heidari et al., EASE European Science Editing)
        - Sex and Gender Equity in Research checklist.
    • NIHMS RAND REACH (Malika et al., Ethics Med Public Health 2025)
        - Health Equity in Research checklist.

────────────────────────────────────────────────────────────────────────────────
WHAT IS ALREADY IMPLEMENTED IN ogap/legacy.py (the v9 core; the top-level
OGAP_source_code_experimental_v9.py entrypoint is now a thin shim)
────────────────────────────────────────────────────────────────────────────────
The main source already provides (kept here so reviewers can find them):

    bootstrap_ci                        CLAIM 29 / 38, TRIPOD 23a       (BCa CI)
    _expected_calibration_error         TRIPOD 12e                       (ECE)
    _brier_score_multiclass             TRIPOD 12e                       (Brier)
    _reliability_diagram                TRIPOD 12e                       (plot)
    cmd_calibration                     TRIPOD 12e                       (subcmd)
    cmd_metric_check                    CLAIM 28                         (cross-check)
    cmd_stratify                        CLAIM 36 / 37, TRIPOD 14         (subgroup metrics
                                        SAGER 6a-c, NIHMS data analysis   incl. sex/age/race)
    cmd_subgroup_forest                 CLAIM 30                         (forest plot)
    cmd_failure_case_figure             CLAIM 39                         (failure analysis)
    cmd_data_flow                       CLAIM 35                         (CONSORT-style)
    cmd_model_card                      CLAIM 22, TRIPOD 22              (Mitchell 2019)
    cmd_datasheet                       CLAIM 7 / 8 / 13, TRIPOD 5a / 7  (Gebru 2021)
    cmd_interrater                      CLAIM 18                         (inter-rater var.)
    cmd_mc_uncertainty                  CLAIM 30                         (MC-dropout/ens.)
    cmd_threshold_sweep                 CLAIM 30                         (sensitivity)
    cmd_holder_sweep                    CLAIM 30                         (Hölder-α)
    clinical_failure_mode_analysis      CLAIM 39, TRIPOD 14              (failure cohort)
    cmd_lint_csv                        CLAIM 8 / 9                      (data audit)
    cmd_surface_dice                    CLAIM 28                         (NSD)
    cmd_compute_profile                 CLAIM 23                         (HW profile)
    cmd_field_strength_curve            CLAIM 30, TRIPOD 14              (B0 sensitivity)
    cmd_loso_cv / cmd_kfold_cv /        CLAIM 30                         (robustness)
    cmd_multi_seed
    cmd_training_curves                 CLAIM 25                         (training curves)
    cmd_total_energy_table              optional sustainability reporting

────────────────────────────────────────────────────────────────────────────────
WHAT THIS MODULE ADDS
────────────────────────────────────────────────────────────────────────────────

    compute_fairness_metrics            TRIPOD 14, NIHMS Data Analysis 6
                                          - demographic-parity ratio
                                          - equalized-odds gap
                                          - max-min subgroup disparity
                                          - worst-group lift
    compute_concentration_index         NIHMS "Equity-Oriented Metrics" (O'Donnell 2016)
                                          - health-equity Gini-style coefficient
    compute_calibration_slope_intercept TRIPOD 12e (Cox 1958, Steyerberg 2010)
                                          - canonical logistic-regression-of-outcome-on-
                                            log-odds slope/intercept.  ECE alone is not
                                            sufficient under TRIPOD+AI.
    compute_class_imbalance_metrics     CLAIM 38, TRIPOD 13
                                          - balanced accuracy, Matthews correlation
                                            coefficient, PR-AUC (per region)
    compute_sex_breakdown               SAGER 3b / 4c / 6a
                                          - automatic sex/gender breakdown of any CSV
                                            with a configurable sex column
    dump_environment_snapshot           CLAIM 23
                                          - pip freeze, CUDA driver, GPU model, OS,
                                            git hash, run date, SLURM job id
    dump_initialization_report          CLAIM 24
                                          - per-layer init scheme + transfer-learning
                                            provenance
    cmd_ai_compliance_report            single orchestrator that runs every subcommand
                                        above, then writes
                                          - compliance_report.json           (full data)
                                          - compliance_checklist.csv         (per item)
                                          - per-checklist mapping JSONs that point to
                                            file paths and code locations a reviewer can
                                            cite by manuscript line number.

The module is dataset-agnostic and import-safe (it does NOT touch torch /
mamba_ssm at import time), so it is safe to import from running training jobs
without changing their numerical behaviour.

────────────────────────────────────────────────────────────────────────────────
DESIGN NOTES
────────────────────────────────────────────────────────────────────────────────
• All functions are pure: they take inputs and return dict / float / np.ndarray.
  The orchestrator (`cmd_ai_compliance_report`) is the only side-effecting
  entry point, and it only writes JSON / CSV under a user-specified directory.
• No heavy ML imports at top level - torch and sklearn are imported lazily.
• When scipy / sklearn are unavailable, fairness and slope/intercept silently
  fall back to numpy-only equivalents and emit a warning in the JSON.
• Every output file is named by checklist item ("claim_38_class_imbalance.json")
  so authors can cite "see compliance_report/claim_38_class_imbalance.json,
  manuscript p. X line Y" in the supplemental material.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

LOG = logging.getLogger("OGAP.compliance")

# ══════════════════════════════════════════════════════════════════════════════
#  CHECKLIST ↔ CODE MAPPING  (machine-readable; written verbatim into the
#  compliance report so manuscript reviewers can cite by item number).
# ══════════════════════════════════════════════════════════════════════════════

CLAIM_CHECKLIST_MAPPING: Dict[int, Dict[str, str]] = {
    1:  {"item": "Identification as AI methodology study",
         "where": "manuscript title / abstract", "code": "-"},
    2:  {"item": "Abstract summary", "where": "manuscript abstract", "code": "-"},
    3:  {"item": "Scientific / clinical background", "where": "manuscript Introduction", "code": "-"},
    4:  {"item": "Study aims, objectives, hypotheses", "where": "manuscript Introduction", "code": "-"},
    5:  {"item": "Prospective or retrospective", "where": "manuscript Methods", "code": "-"},
    6:  {"item": "Study goal", "where": "manuscript Methods", "code": "-"},
    7:  {"item": "Data sources",
         "where": "compliance_report/datasheet/DATASHEET.md",
         "code": "cmd_datasheet"},
    8:  {"item": "Inclusion / exclusion criteria",
         "where": "compliance_report/datasheet/DATASHEET.md + data_flow.dot",
         "code": "cmd_datasheet, cmd_data_flow"},
    9:  {"item": "Data preprocessing",
         "where": "compliance_report/preprocessing.json",
         "code": "preprocess_mri_volume(), zscore_nonzero(), n4_bias_correct_volume()"},
    10: {"item": "Selection of data subsets",
         "where": "compliance_report/datasheet/DATASHEET.md",
         "code": "cmd_datasheet"},
    11: {"item": "De-identification methods",
         "where": "compliance_report/datasheet/DATASHEET.md",
         "code": "-  (BraTS / UTSW data already de-identified by source)"},
    12: {"item": "Missing data handling",
         "where": "compliance_report/missing_data.json",
         "code": "random_modality_mask(), supported_modality_combinations()"},
    13: {"item": "Image acquisition protocol",
         "where": "compliance_report/datasheet/DATASHEET.md",
         "code": "cmd_datasheet"},
    14: {"item": "Reference standard methodology",
         "where": "compliance_report/datasheet/DATASHEET.md", "code": "cmd_datasheet"},
    15: {"item": "Rationale for reference standard", "where": "manuscript Methods", "code": "-"},
    16: {"item": "Source of reference standard annotations",
         "where": "compliance_report/datasheet/DATASHEET.md", "code": "cmd_datasheet"},
    17: {"item": "Annotation of test set",
         "where": "compliance_report/datasheet/DATASHEET.md", "code": "cmd_datasheet"},
    18: {"item": "Inter / intra-rater variability",
         "where": "compliance_report/interrater/", "code": "cmd_interrater"},
    19: {"item": "How data assigned to partitions",
         "where": "compliance_report/data_flow.dot", "code": "cmd_data_flow"},
    20: {"item": "Level at which partitions are disjoint (patient)",
         "where": "compliance_report/datasheet/DATASHEET.md", "code": "cmd_datasheet"},
    21: {"item": "Intended sample size", "where": "manuscript Methods", "code": "-"},
    22: {"item": "Detailed model description",
         "where": "compliance_report/model_card/MODEL_CARD.md",
         "code": "cmd_model_card"},
    23: {"item": "Software libraries / frameworks / packages (with versions)",
         "where": "compliance_report/claim_23_environment.json",
         "code": "dump_environment_snapshot"},
    24: {"item": "Initialization of model parameters",
         "where": "compliance_report/claim_24_initialization.json",
         "code": "dump_initialization_report"},
    25: {"item": "Details of training approach (hyperparameters)",
         "where": "compliance_report/training_config.json",
         "code": "TrainConfig.to_dict() + cmd_training_curves"},
    26: {"item": "Method of selecting final model",
         "where": "compliance_report/training_config.json", "code": "best_student.pth selection"},
    27: {"item": "Ensembling techniques",
         "where": "compliance_report/kd_compare/ if applicable", "code": "cmd_kd_compare"},
    28: {"item": "Metrics of model performance",
         "where": "compliance_report/metrics/ + surface_dice/",
         "code": "cmd_metric_check, cmd_surface_dice"},
    29: {"item": "Statistical measures of significance and uncertainty",
         "where": "every metric JSON includes BCa 95% CIs",
         "code": "bootstrap_ci"},
    30: {"item": "Robustness or sensitivity analysis",
         "where": "compliance_report/{mc_uncertainty,threshold_sweep,holder_sweep,"
                  "field_strength_curve,loso_cv,multi_seed}/",
         "code": "cmd_mc_uncertainty, cmd_threshold_sweep, cmd_holder_sweep, "
                 "cmd_field_strength_curve, cmd_loso_cv, cmd_multi_seed"},
    31: {"item": "Methods for explainability / interpretability",
         "where": "compliance_report/mc_uncertainty/ (uncertainty maps)",
         "code": "cmd_mc_uncertainty (per-voxel entropy / variance)"},
    32: {"item": "Evaluation on internal data",
         "where": "Results/student_*/ training logs + best_student_eval.json",
         "code": "evaluate_student"},
    33: {"item": "Testing on external data",
         "where": "Results/utsw_full_evaluation_tta_*/", "code": "evaluate (subcommand)"},
    34: {"item": "Clinical trial registration", "where": "manuscript Methods", "code": "-"},
    35: {"item": "Numbers of patients included / excluded",
         "where": "compliance_report/data_flow.dot",
         "code": "cmd_data_flow"},
    36: {"item": "Demographic / clinical characteristics in each partition",
         "where": "compliance_report/stratify/stratified_summary.json + "
                  "compliance_report/sex_breakdown.json",
         "code": "cmd_stratify, compute_sex_breakdown"},
    37: {"item": "Performance metrics with statistical uncertainty by subgroup",
         "where": "compliance_report/stratify/stratified_summary.json",
         "code": "cmd_stratify"},
    38: {"item": "Estimates of diagnostic performance and their precision",
         "where": "compliance_report/claim_38_class_imbalance.json + calibration/",
         "code": "compute_class_imbalance_metrics, cmd_calibration"},
    39: {"item": "Failure analysis of incorrect results",
         "where": "compliance_report/failure_case_figure/ + clinical_failure_modes/",
         "code": "cmd_failure_case_figure, clinical_failure_mode_analysis"},
    40: {"item": "Study limitations", "where": "manuscript Discussion", "code": "-"},
    41: {"item": "Implications for practice", "where": "manuscript Discussion", "code": "-"},
    42: {"item": "Study protocol reference", "where": "manuscript Other Information",
         "code": "-"},
    43: {"item": "Software / model / data availability",
         "where": "manuscript Other Information",
         "code": "compliance_report/claim_23_environment.json captures repo state"},
    44: {"item": "Sources of funding", "where": "manuscript Other Information", "code": "-"},
}

TRIPOD_AI_CHECKLIST_MAPPING: Dict[str, Dict[str, str]] = {
    "1":   {"item": "Title (multivariable prediction model)", "where": "manuscript title",     "code": "-"},
    "2":   {"item": "Abstract",                               "where": "manuscript abstract",  "code": "-"},
    "3a":  {"item": "Healthcare context / rationale",         "where": "manuscript Introduction", "code": "-"},
    "3b":  {"item": "Target population / intended users",     "where": "manuscript Introduction", "code": "-"},
    "3c":  {"item": "Health inequalities by sociodemographics",
            "where": "compliance_report/sex_breakdown.json + stratify/",
            "code": "compute_sex_breakdown, cmd_stratify"},
    "4":   {"item": "Study objectives",                       "where": "manuscript Introduction", "code": "-"},
    "5a":  {"item": "Data sources for development AND evaluation",
            "where": "compliance_report/datasheet/DATASHEET.md", "code": "cmd_datasheet"},
    "5b":  {"item": "Dates of participant accrual / follow-up",
            "where": "compliance_report/datasheet/DATASHEET.md", "code": "cmd_datasheet"},
    "6a":  {"item": "Study setting / number of centres",
            "where": "compliance_report/datasheet/DATASHEET.md", "code": "cmd_datasheet"},
    "6b":  {"item": "Eligibility criteria",
            "where": "compliance_report/datasheet/DATASHEET.md", "code": "cmd_datasheet"},
    "6c":  {"item": "Treatments received",                    "where": "manuscript Methods",   "code": "-"},
    "7":   {"item": "Data pre-processing / quality checking",
            "where": "compliance_report/preprocessing.json",
            "code": "preprocess_mri_volume(), cmd_lint_csv"},
    "8a":  {"item": "Outcome definition / time horizon",
            "where": "compliance_report/datasheet/DATASHEET.md", "code": "cmd_datasheet"},
    "8b":  {"item": "Outcome assessment method",
            "where": "compliance_report/datasheet/DATASHEET.md", "code": "cmd_datasheet"},
    "8c":  {"item": "Blinding of outcome assessors",          "where": "manuscript Methods",   "code": "-"},
    "9a":  {"item": "Initial predictors / pre-selection",
            "where": "compliance_report/model_card/MODEL_CARD.md",
            "code": "cmd_model_card (input modalities listed)"},
    "9b":  {"item": "Predictor definition / measurement",
            "where": "compliance_report/datasheet/DATASHEET.md (modalities, sequences)",
            "code": "cmd_datasheet"},
    "9c":  {"item": "Predictor assessor qualifications",
            "where": "compliance_report/datasheet/DATASHEET.md", "code": "cmd_datasheet"},
    "10":  {"item": "Sample size justification",             "where": "manuscript Methods",   "code": "-"},
    "11":  {"item": "Missing data handling",
            "where": "compliance_report/missing_data.json",
            "code": "random_modality_mask, missing-mode robustness in evaluate"},
    "12a": {"item": "Data used at each modelling stage",
            "where": "compliance_report/data_flow.dot",
            "code": "cmd_data_flow"},
    "12b": {"item": "Predictor handling (rescaling, etc.)",
            "where": "compliance_report/preprocessing.json",
            "code": "preprocess_mri_volume()"},
    "12c": {"item": "Model type, building steps, hyperparameter tuning",
            "where": "compliance_report/training_config.json + model_card/",
            "code": "cmd_model_card + TrainConfig"},
    "12d": {"item": "Heterogeneity handling across clusters",
            "where": "compliance_report/stratify/ + loso_cv/",
            "code": "cmd_stratify, cmd_loso_cv"},
    "12e": {"item": "Performance evaluation measures (discrimination, calibration, clinical utility)",
            "where": "compliance_report/calibration/ + tripod_12e_calibration_slope.json",
            "code": "cmd_calibration, compute_calibration_slope_intercept"},
    "12f": {"item": "Model updating",                        "where": "manuscript Methods",   "code": "-"},
    "12g": {"item": "Model output formula / API",
            "where": "compliance_report/model_card/MODEL_CARD.md + ONNX export",
            "code": "cmd_model_card + export_to_onnx"},
    "13":  {"item": "Class imbalance methods",
            "where": "compliance_report/claim_38_class_imbalance.json",
            "code": "compute_class_imbalance_metrics + lambda_region BCE+Dice loss"},
    "14":  {"item": "Fairness approaches",
            "where": "compliance_report/tripod_14_fairness.json",
            "code": "compute_fairness_metrics, cmd_stratify"},
    "15":  {"item": "Model output specification / threshold",
            "where": "compliance_report/model_card/MODEL_CARD.md + threshold_sweep/",
            "code": "cmd_model_card, cmd_threshold_sweep"},
    "16":  {"item": "Differences development vs. evaluation setting",
            "where": "compliance_report/datasheet/DATASHEET.md (UTSW vs BraTS comparison)",
            "code": "cmd_datasheet"},
    "17":  {"item": "Ethics / informed consent",             "where": "manuscript Methods",   "code": "-"},
    "18a": {"item": "Funding source",                         "where": "manuscript Other",    "code": "-"},
    "18b": {"item": "Conflicts of interest",                  "where": "manuscript Other",    "code": "-"},
    "18c": {"item": "Protocol availability",                  "where": "manuscript Other",    "code": "-"},
    "18d": {"item": "Study registration",                     "where": "manuscript Other",    "code": "-"},
    "18e": {"item": "Data availability",                      "where": "manuscript Other",    "code": "-"},
    "18f": {"item": "Code availability",
            "where": "compliance_report/claim_23_environment.json (commit hash + repo URL)",
            "code": "dump_environment_snapshot"},
    "19":  {"item": "Patient and public involvement",         "where": "manuscript Methods",   "code": "-"},
    "20a": {"item": "Flow of participants",
            "where": "compliance_report/data_flow.dot",       "code": "cmd_data_flow"},
    "20b": {"item": "Characteristics by source / setting",
            "where": "compliance_report/stratify/ + sex_breakdown.json",
            "code": "cmd_stratify, compute_sex_breakdown"},
    "20c": {"item": "Demographic distribution comparison (development vs. evaluation)",
            "where": "compliance_report/development_vs_evaluation_table.json",
            "code": "compute_development_vs_evaluation_table"},
    "21":  {"item": "Number of participants / outcome events per analysis",
            "where": "compliance_report/data_flow.dot", "code": "cmd_data_flow"},
    "22":  {"item": "Full prediction model (formula / code / API)",
            "where": "compliance_report/model_card/MODEL_CARD.md + ONNX export",
            "code": "cmd_model_card, export_to_onnx"},
    "23a": {"item": "Performance with confidence intervals (incl. subgroups)",
            "where": "compliance_report/stratify/ (BCa 95% CIs by subgroup)",
            "code": "cmd_stratify (uses bootstrap_ci)"},
    "23b": {"item": "Heterogeneity in performance across clusters",
            "where": "compliance_report/loso_cv/", "code": "cmd_loso_cv"},
    "24":  {"item": "Model updating results",                 "where": "manuscript Results",  "code": "-"},
    "25":  {"item": "Interpretation including fairness",      "where": "manuscript Discussion","code": "-"},
    "26":  {"item": "Limitations",                            "where": "manuscript Discussion","code": "-"},
    "27a": {"item": "How poor-quality input is handled",
            "where": "compliance_report/preprocessing.json + missing_data.json",
            "code": "preprocess_mri_volume + modality dropout fallback"},
    "27b": {"item": "User interaction with input data",       "where": "manuscript Discussion","code": "-"},
    "27c": {"item": "Future research directions",             "where": "manuscript Discussion","code": "-"},
}

SAGER_CHECKLIST_MAPPING: Dict[str, Dict[str, str]] = {
    # Table 1 - studies with human participants (BraTS / UTSW are human imaging data)
    "1":   {"item": "Sex / gender terminology used appropriately",
            "where": "manuscript", "code": "-"},
    "2":   {"item": "Title specifies sex/gender if only one included",
            "where": "manuscript title (N/A - both sexes)", "code": "-"},
    "3a":  {"item": "Abstract specifies sex/gender if only one included",
            "where": "manuscript abstract (N/A - both sexes)", "code": "-"},
    "3b":  {"item": "Study population sex/gender breakdown",
            "where": "compliance_report/sex_breakdown.json",
            "code": "compute_sex_breakdown"},
    "4a":  {"item": "Citation of prior sex/gender-difference studies",
            "where": "manuscript Introduction", "code": "-"},
    "4b":  {"item": "Whether sex/gender is an important variant",
            "where": "manuscript Introduction", "code": "-"},
    "4c":  {"item": "Disease-prevalence demographics by sex/gender",
            "where": "compliance_report/sex_breakdown.json + datasheet/",
            "code": "compute_sex_breakdown, cmd_datasheet"},
    "5a":  {"item": "Method of definition of sex/gender (self-report, etc.)",
            "where": "compliance_report/sex_breakdown.json (definition_method)",
            "code": "compute_sex_breakdown"},
    "5b":  {"item": "Sex/gender consideration in study design",
            "where": "manuscript Methods", "code": "cmd_stratify enables sex-aware analysis"},
    "6a":  {"item": "Study population description with sex/gender breakdown",
            "where": "compliance_report/sex_breakdown.json", "code": "compute_sex_breakdown"},
    "6b":  {"item": "Data disaggregated by sex/gender",
            "where": "compliance_report/stratify/stratified_summary.json (Sex at birth)",
            "code": "cmd_stratify"},
    "6c":  {"item": "Sex/gender-based analyses regardless of outcome",
            "where": "compliance_report/stratify/stratified_summary.json",
            "code": "cmd_stratify"},
    "6d":  {"item": "Adverse-event data disaggregated", "where": "N/A - non-clinical-trial study", "code": "-"},
    "6e":  {"item": "Patient-reported outcomes disaggregated", "where": "N/A", "code": "-"},
    "6f":  {"item": "Effects of other exposures across genders",
            "where": "compliance_report/stratify/", "code": "cmd_stratify"},
    "6g":  {"item": "Table 1 with sex/gender rows",
            "where": "compliance_report/sex_breakdown.json", "code": "compute_sex_breakdown"},
    "7a":  {"item": "Discussion: sex/gender implications",
            "where": "manuscript Discussion", "code": "-"},
    "7b":  {"item": "Rationale if sex/gender analysis not done",
            "where": "manuscript Discussion (N/A - analysis was done)", "code": "-"},
}

NIHMS_HEALTH_EQUITY_MAPPING: Dict[str, Dict[str, str]] = {
    # RAND REACH Center checklist (Malika et al. 2025) - items that map to code
    "data_collection_1":  {"item": "Collect relevant demographic information",
                           "where": "compliance_report/sex_breakdown.json + stratify/ "
                                    "(sex, age, race, ethnicity, IDH, grade, MGMT, scanner)",
                           "code": "compute_sex_breakdown, cmd_stratify, STRATIFIER_COLUMNS"},
    "data_collection_2":  {"item": "Consider risks for sensitive questions",
                           "where": "compliance_report/datasheet/DATASHEET.md", "code": "cmd_datasheet"},
    "data_collection_3":  {"item": "Cultural-humility training for data collectors",
                           "where": "manuscript Methods", "code": "-"},
    "data_collection_4":  {"item": "Iterative community feedback mechanisms",
                           "where": "manuscript Methods", "code": "-"},
    "data_collection_5":  {"item": "Stratified analysis by demographic groups",
                           "where": "compliance_report/stratify/stratified_summary.json + "
                                    "tripod_14_fairness.json",
                           "code": "cmd_stratify, compute_fairness_metrics"},
    "data_collection_6":  {"item": "Equity-oriented metrics (concentration index)",
                           "where": "compliance_report/nihms_concentration_index.json",
                           "code": "compute_concentration_index"},
    "research_design_1":  {"item": "Cultural / social-structural context",
                           "where": "manuscript (LMIC field-strength prior, race / ethnicity sampling biases)",
                           "code": "lmic_field_strength_prior + grade23/idh_mutant/legacy_protocol "
                                   "sampling biases"},
    "research_design_2":  {"item": "Appropriate research methods for population",
                           "where": "manuscript Methods", "code": "-"},
    "research_design_3":  {"item": "Accessibility / inclusivity in design",
                           "where": "INT8 export → CPU-only deployment for LMIC clinics",
                           "code": "export_to_onnx INT8"},
    "dissemination_1":    {"item": "Translate findings for diverse audiences",
                           "where": "compliance_report/model_card/MODEL_CARD.md (intended audience)",
                           "code": "cmd_model_card"},
    "dissemination_2":    {"item": "Sustainability / capacity building",
                           "where": "manuscript Discussion", "code": "-"},
}

# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS - JSON / CSV / lazy imports
# ══════════════════════════════════════════════════════════════════════════════

def _ensure_dir(p: Union[str, Path]) -> Path:
    out = Path(p)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _save_json(obj: Any, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)


def _save_csv(rows: Sequence[Dict[str, Any]], path: Union[str, Path]) -> None:
    path = Path(path)
    if not rows:
        path.write_text("")
        return
    keys: List[str] = []
    for row in rows:
        for k in row.keys():
            if k not in keys:
                keys.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, tuple)):
        return list(obj)
    return str(obj)


def _try_import(module_name: str):
    try:
        return __import__(module_name)
    except ImportError:
        return None  # genuinely not installed - silence is fine
    except Exception as exc:
        # Installed but broken (e.g. a shared-library load error) must NOT look identical
        # to "not installed": a silent fallback would mislabel the TRIPOD 12e method or
        # drop a CLAIM 23 package version with no record.
        LOG.warning("[_try_import] %r imported but raised %s: %s",
                    module_name, type(exc).__name__, exc)
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  CLAIM 23 - Software / hardware environment snapshot
# ══════════════════════════════════════════════════════════════════════════════

def dump_environment_snapshot(
    out_dir: Union[str, Path],
    *,
    extra: Optional[Dict[str, Any]] = None,
    include_pip_freeze: bool = True,
) -> Dict[str, Any]:
    """Emit a machine-readable snapshot of the run environment.

    CLAIM 2024 Item 23: "Software libraries, frameworks, and packages.  Specify
    the names and version numbers of all software libraries, frameworks, and
    packages used for model training and inference."

    Captures:
        - python / OS / kernel / libc
        - torch / cuda / triton / numpy / scipy / nibabel / onnxruntime / mamba_ssm
        - GPU model + driver + memory
        - SLURM job id + node + start time
        - git commit hash and dirty-tree status (if repo present)
        - optional ``pip freeze`` (full dependency tree)

    Returns the dict it just wrote, so callers can also embed it inline.
    """
    out_dir = _ensure_dir(out_dir)

    snap: Dict[str, Any] = {
        "checklist_item": "CLAIM 2024 Item 23 (software libraries, frameworks, packages)",
        "captured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": {
            "python_version": sys.version.split()[0],
            "python_implementation": platform.python_implementation(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor() or "unknown",
        },
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID", ""),
            "job_name": os.environ.get("SLURM_JOB_NAME", ""),
            "node": os.environ.get("SLURMD_NODENAME", ""),
            "n_cpus": os.environ.get("SLURM_CPUS_ON_NODE", ""),
            "mem": os.environ.get("SLURM_MEM_PER_NODE", ""),
            "gpus": os.environ.get("SLURM_GPUS", "") or os.environ.get("SLURM_GPUS_ON_NODE", ""),
        },
        "packages": {},
        "gpu": {},
        "git": {},
    }

    for pkg in (
        "numpy", "scipy", "pandas", "torch", "torchvision", "triton",
        "monai", "medpy", "nibabel", "scikit-learn", "sklearn",
        "matplotlib", "onnx", "onnxruntime", "mamba_ssm", "causal_conv1d",
    ):
        mod = _try_import(pkg.replace("-", "_"))
        if mod is not None:
            ver = getattr(mod, "__version__", "unknown")
            snap["packages"][pkg] = ver

    # CUDA / GPU info via torch (lazy).
    torch = _try_import("torch")
    if torch is not None:
        snap["packages"]["torch_cuda_build"] = getattr(torch.version, "cuda", "cpu")
        if torch.cuda.is_available():
            try:
                idx = 0
                snap["gpu"] = {
                    "name": torch.cuda.get_device_name(idx),
                    "capability": list(torch.cuda.get_device_capability(idx)),
                    "memory_total_gib": round(
                        torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3), 2
                    ),
                    "device_count": torch.cuda.device_count(),
                    "driver_version": _read_driver_version(),
                }
            except Exception as exc:
                snap["gpu"] = {"error": str(exc)}

    # Git provenance.
    snap["git"] = _read_git_provenance()

    # Optional full dependency tree.
    if include_pip_freeze:
        snap["pip_freeze"] = _read_pip_freeze()

    if extra:
        snap["extra"] = extra

    _save_json(snap, Path(out_dir) / "claim_23_environment.json")
    return snap


def _read_driver_version() -> str:
    if shutil.which("nvidia-smi") is None:
        return "unknown"
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True, capture_output=True, text=True, timeout=8,
        )
        return out.stdout.strip().splitlines()[0]
    except Exception:
        return "unknown"


def _read_git_provenance() -> Dict[str, Any]:
    if shutil.which("git") is None:
        return {"available": False}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, timeout=4,
            cwd=str(Path(__file__).resolve().parent),
        ).stdout.strip()
        try:
            subprocess.run(
                ["git", "diff-index", "--quiet", "HEAD", "--"],
                check=True, capture_output=True, text=True, timeout=4,
                cwd=str(Path(__file__).resolve().parent),
            )
            dirty = False
        except subprocess.CalledProcessError:
            dirty = True
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            check=True, capture_output=True, text=True, timeout=4,
            cwd=str(Path(__file__).resolve().parent),
        ).stdout.strip()
        return {"available": True, "commit": commit, "branch": branch, "dirty": dirty}
    except Exception:
        return {"available": False}


def _read_pip_freeze() -> List[str]:
    try:
        out = subprocess.run(
            [sys.executable, "-m", "pip", "freeze", "--disable-pip-version-check"],
            check=True, capture_output=True, text=True, timeout=30,
        )
        return out.stdout.strip().splitlines()
    except Exception:
        return []


# ══════════════════════════════════════════════════════════════════════════════
#  CLAIM 24 - Initialization report
# ══════════════════════════════════════════════════════════════════════════════

def dump_initialization_report(
    model: Any,
    out_dir: Union[str, Path],
    *,
    transfer_learning_ckpt: Optional[Union[str, Path]] = None,
    transfer_learning_role: str = "-",
) -> Dict[str, Any]:
    """Emit a per-layer initialization report.

    CLAIM 2024 Item 24: "Indicate how the parameters of the model were
    initialized.  Describe the distribution from which random values were
    drawn for randomly initialized parameters.  Specify the source of the
    starting weights if transfer learning is employed to initialize parameters."

    For each parameter tensor the report records:
        - shape, dtype, requires_grad
        - empirical mean / std (sanity-check against documented init scheme)
        - guess of init scheme (kaiming_normal, kaiming_uniform, xavier,
          orthogonal, constant, transferred, frozen, unknown)
        - whether the layer is frozen (no grad)
    """
    out_dir = _ensure_dir(out_dir)
    torch = _try_import("torch")
    if torch is None:
        report = {
            "checklist_item": "CLAIM 2024 Item 24",
            "error": "torch unavailable; initialization report skipped.",
        }
        _save_json(report, Path(out_dir) / "claim_24_initialization.json")
        return report

    layers: List[Dict[str, Any]] = []
    transferred_keys = set()
    if transfer_learning_ckpt and Path(transfer_learning_ckpt).exists():
        try:
            ckpt = torch.load(str(transfer_learning_ckpt), map_location="cpu",
                              weights_only=True)
            sd = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
            transferred_keys = set(sd.keys()) if isinstance(sd, dict) else set()
        except Exception as exc:
            LOG.warning("[init_report] could not read transfer ckpt: %s", exc)

    n_total = 0
    n_trainable = 0
    n_frozen = 0
    n_transferred = 0

    for name, p in model.named_parameters():
        n_total += int(p.numel())
        if p.requires_grad:
            n_trainable += int(p.numel())
        else:
            n_frozen += int(p.numel())

        was_transferred = name in transferred_keys
        if was_transferred:
            n_transferred += int(p.numel())

        with torch.no_grad():
            data = p.detach().float().cpu().numpy()

        scheme = _guess_init_scheme(name, data, was_transferred=was_transferred,
                                    is_frozen=not p.requires_grad)
        layers.append({
            "name": name,
            "shape": list(p.shape),
            "n_params": int(p.numel()),
            "dtype": str(p.dtype),
            "requires_grad": bool(p.requires_grad),
            "transferred_from_ckpt": was_transferred,
            "empirical_mean": float(np.nan_to_num(data.mean())),
            "empirical_std":  float(np.nan_to_num(data.std())),
            "abs_max":        float(np.nan_to_num(np.abs(data).max())),
            "init_scheme_guess": scheme,
        })

    report: Dict[str, Any] = {
        "checklist_item": "CLAIM 2024 Item 24 (initialization of model parameters)",
        "transfer_learning": {
            "ckpt": str(transfer_learning_ckpt) if transfer_learning_ckpt else None,
            "role": transfer_learning_role,
            "n_keys_in_ckpt": len(transferred_keys),
        },
        "totals": {
            "n_params_total":      n_total,
            "n_params_trainable":  n_trainable,
            "n_params_frozen":     n_frozen,
            "n_params_transferred": n_transferred,
        },
        "layers": layers,
    }
    _save_json(report, Path(out_dir) / "claim_24_initialization.json")
    return report


def _guess_init_scheme(
    name: str,
    data: np.ndarray,
    *,
    was_transferred: bool,
    is_frozen: bool,
) -> str:
    """Heuristic identification of the init scheme from empirical statistics.

    Conservative - when in doubt returns ``'unknown'`` rather than guess.
    """
    if was_transferred:
        return "transferred_from_ckpt"
    if is_frozen:
        return "frozen"
    flat = data.ravel()
    if flat.size == 0:
        return "empty"
    mean = float(flat.mean())
    std = float(flat.std())
    abs_max = float(np.abs(flat).max())

    # Constants (biases, group-norm gammas)
    if abs_max < 1e-6:
        return "constant_zero"
    if std < 1e-6 and abs(mean - 1.0) < 1e-3:
        return "constant_one  (likely norm gamma)"
    if std < 1e-6:
        return f"constant_{mean:.3g}"

    # Kaiming-normal-ish: small mean, std ≈ √(2/fan_in).  We cannot recover
    # fan_in here without the full module graph, so we just report std.
    if "weight" in name and ".bn" not in name and ".gn" not in name:
        if 0.005 < std < 0.5:
            return f"kaiming-like (std≈{std:.3g})"
    if "norm" in name.lower() and 0.99 < mean < 1.01:
        return "ones (norm gamma)"
    return f"unknown (mean={mean:.3g}, std={std:.3g})"


# ══════════════════════════════════════════════════════════════════════════════
#  TRIPOD 12e - Calibration slope and intercept (canonical Cox 1958)
# ══════════════════════════════════════════════════════════════════════════════

def compute_calibration_slope_intercept(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    eps: float = 1e-6,
) -> Dict[str, float]:
    """Calibration slope and calibration-in-the-large.

    TRIPOD+AI Item 12e: discrimination + **calibration** + clinical utility.
    ECE alone does not satisfy TRIPOD+AI - it asks for the canonical
    calibration slope and intercept (Cox 1958, Steyerberg 2010 *Clinical
    Prediction Models*, Steyerberg & Vergouwe 2014).

    Definition (binary outcome):
        Fit  logit(P(Y=1)) = α + β · logit(p̂)
        on the held-out evaluation set.
        - β = 1   indicates ideal calibration of the spread of predictions.
        - α = 0   indicates calibration-in-the-large (no over / under-prediction).

    For multi-class segmentation we collapse to ``foreground vs background``
    (Y = any tumour class) - this is the most clinically actionable summary.

    Args
    ----
    probs : np.ndarray, shape (N, C) or (N,)
        Per-voxel softmax probabilities (or per-voxel foreground probability).
    labels : np.ndarray, shape (N,)
        Integer class labels.

    Returns
    -------
    dict with keys
        intercept, slope, intercept_se, slope_se,
        log_likelihood, n_samples, n_pos, prevalence, method.

    Notes
    -----
    Uses statsmodels.Logit if available, otherwise a numpy-only Newton-Raphson
    on the binomial log-likelihood with a closed-form Hessian.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels).astype(np.int64).ravel()
    if probs.ndim == 2:
        # foreground = 1 - P(class 0)
        p_fg = 1.0 - probs[:, 0]
    else:
        p_fg = probs.ravel()
    p_fg = np.clip(p_fg, eps, 1.0 - eps)
    y = (labels > 0).astype(np.float64)
    if y.shape != p_fg.shape:
        # Align by flattening; downstream will catch shape mismatch.
        m = min(y.size, p_fg.size)
        y = y[:m]; p_fg = p_fg[:m]

    logit_p = np.log(p_fg / (1.0 - p_fg))
    n = int(y.size)
    n_pos = int(y.sum())
    prev = float(n_pos / max(n, 1))

    # Try statsmodels.
    sm = _try_import("statsmodels")
    if sm is not None:
        try:
            import statsmodels.api as smapi  # noqa: WPS433
            X = smapi.add_constant(logit_p)
            res = smapi.Logit(y, X).fit(disp=False)
            return {
                "checklist_item": "TRIPOD+AI Item 12e - calibration slope/intercept",
                "intercept": float(res.params[0]),
                "slope":     float(res.params[1]),
                "intercept_se": float(res.bse[0]),
                "slope_se":     float(res.bse[1]),
                "log_likelihood": float(res.llf),
                "n_samples": n, "n_pos": n_pos, "prevalence": prev,
                "method": "statsmodels.Logit",
            }
        except Exception as exc:  # noqa: BLE001
            LOG.warning("[calibration] statsmodels failed (%s); falling back to numpy", exc)

    # Numpy-only Newton-Raphson on β = (intercept, slope).
    beta = np.zeros(2, dtype=np.float64)
    X = np.column_stack([np.ones(n), logit_p])
    converged = False
    for _ in range(100):
        eta = X @ beta
        p_hat = 1.0 / (1.0 + np.exp(-eta))
        grad = X.T @ (p_hat - y)
        W = p_hat * (1.0 - p_hat)
        H = X.T @ (X * W[:, None]) + 1e-8 * np.eye(2)
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError as exc:
            LOG.error("[calibration] Newton-Raphson Hessian singular (%s); TRIPOD 12e "
                      "slope/intercept are NOT converged.", exc)
            break
        beta = beta - step
        if np.max(np.abs(step)) < 1e-8:
            converged = True
            break
    if not converged:
        LOG.error("[calibration] Newton-Raphson did not converge in 100 iterations; "
                  "TRIPOD 12e slope/intercept flagged converged=False.")
    eta = X @ beta
    p_hat = np.clip(1.0 / (1.0 + np.exp(-eta)), eps, 1.0 - eps)
    ll = float((y * np.log(p_hat) + (1.0 - y) * np.log(1.0 - p_hat)).sum())
    W = p_hat * (1.0 - p_hat)
    cov = np.linalg.pinv(X.T @ (X * W[:, None]))
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    return {
        "checklist_item": "TRIPOD+AI Item 12e - calibration slope/intercept",
        "intercept": float(beta[0]),
        "slope":     float(beta[1]),
        "intercept_se": float(se[0]),
        "slope_se":     float(se[1]),
        "log_likelihood": ll,
        "n_samples": n, "n_pos": n_pos, "prevalence": prev,
        "method": "numpy_newton_raphson",
        "converged": bool(converged),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  CLAIM 38 / TRIPOD 13 - Class-imbalance-aware metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_class_imbalance_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    region_name: str = "foreground",
) -> Dict[str, Any]:
    """Class-imbalance-aware companion metrics to Dice / HD95.

    CLAIM 2024 Item 38: "If applicable, recognize the presence of class
    imbalance ... and provide appropriate metrics to reflect algorithm
    performance."

    TRIPOD+AI Item 13: "If class imbalance methods were used, state why and
    how this was done, and any subsequent methods to recalibrate the model
    or the model predictions."

    Returns balanced accuracy, Matthews Correlation Coefficient, PR-AUC,
    ROC-AUC, sensitivity, specificity, PPV, NPV, prevalence - each one
    independent of the class imbalance ratio.

    Args
    ----
    probs : np.ndarray, shape (N,) or (N, C)
        For multi-class inputs we collapse to "foreground vs background"
        (any class > 0 is positive).
    labels : np.ndarray, shape (N,)
        Integer labels.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels).astype(np.int64).ravel()
    if probs.ndim == 2:
        p_fg = 1.0 - probs[:, 0]
    else:
        p_fg = probs.ravel()
    y = (labels > 0).astype(np.int64)
    if y.size != p_fg.size:
        m = min(y.size, p_fg.size)
        y = y[:m]; p_fg = p_fg[:m]

    # Threshold = 0.5
    pred = (p_fg >= 0.5).astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    ppv  = tp / max(tp + fp, 1)
    npv  = tn / max(tn + fn, 1)
    bal_acc = 0.5 * (sens + spec)

    mcc_den = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1))
    mcc = (tp * tn - fp * fn) / mcc_den if mcc_den > 0 else 0.0

    pr_auc, roc_auc = _auc_pair(p_fg, y)

    return {
        "checklist_item": "CLAIM 2024 Item 38 + TRIPOD+AI Item 13 (class-imbalance metrics)",
        "region": region_name,
        "n_samples": int(y.size),
        "n_pos": int(y.sum()),
        "prevalence": float(y.mean()),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "sensitivity": sens, "specificity": spec,
        "ppv": ppv, "npv": npv,
        "balanced_accuracy": bal_acc,
        "matthews_correlation_coefficient": mcc,
        "pr_auc":  pr_auc,
        "roc_auc": roc_auc,
    }


def _auc_pair(p: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Return (PR-AUC, ROC-AUC) using sklearn if available, else numpy."""
    sk = _try_import("sklearn")
    if sk is not None:
        try:
            from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: WPS433
            pr_auc = float(average_precision_score(y, p))
            roc_auc = float(roc_auc_score(y, p))
            return pr_auc, roc_auc
        except Exception as exc:
            LOG.warning("[auc_pair] sklearn AUC failed (%s); using the numpy trapezoid "
                        "fallback - verify AUC validity.", exc)
    # Numpy fallback (trapezoid-rule + sort).
    order = np.argsort(-p)
    y_sorted = y[order]
    tp_cum = np.cumsum(y_sorted)
    fp_cum = np.cumsum(1 - y_sorted)
    P = max(int(y.sum()), 1)
    N = max(int((1 - y).sum()), 1)
    tpr = tp_cum / P
    fpr = fp_cum / N
    roc_auc = float(np.trapz(tpr, fpr))
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1)
    recall = tpr
    pr_auc = float(np.trapz(precision, recall))
    return pr_auc, roc_auc


# ══════════════════════════════════════════════════════════════════════════════
#  TRIPOD 14 / NIHMS - Fairness metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_fairness_metrics(
    per_case_metric: Sequence[float],
    sensitive_attribute: Sequence[str],
    *,
    metric_name: str = "dice",
    min_group_n: int = 5,
) -> Dict[str, Any]:
    """Compute group-fairness metrics across a sensitive attribute.

    TRIPOD+AI Item 14: "Describe any approaches that were used to address
    model fairness and their rationale."

    NIHMS RAND REACH §"Equity-Oriented Metrics": "Employ metrics like
    concentration indices and use disaggregated data to quantify and
    monitor disparities effectively."

    Reports
        - per-group mean / N
        - max-min gap (worst-case disparity)
        - disparity ratio = min / max  (1.0 = no disparity)
        - 80%-rule (disparate-impact-style):  ratio < 0.8 flags concern
        - worst-group lift = best-group_mean - worst-group_mean
    """
    per_case_metric = np.asarray([float(x) for x in per_case_metric])
    sens = np.asarray([str(s) for s in sensitive_attribute])
    if per_case_metric.size != sens.size:
        m = min(per_case_metric.size, sens.size)
        per_case_metric = per_case_metric[:m]; sens = sens[:m]

    groups: Dict[str, List[float]] = defaultdict(list)
    for v, g in zip(per_case_metric, sens):
        if v is None:
            continue
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            continue
        if not g or g.lower() in ("na", "n/a", "unknown", ""):
            continue
        groups[g].append(float(v))

    rows: Dict[str, Dict[str, float]] = {}
    means: List[Tuple[str, float, int]] = []
    for g, vals in groups.items():
        n = len(vals)
        if n < min_group_n:
            rows[g] = {"n": n, "skipped": "n<min_group_n"}
            continue
        m = float(np.mean(vals))
        s = float(np.std(vals, ddof=1)) if n > 1 else 0.0
        rows[g] = {"n": n, "mean": m, "std": s,
                   "median": float(np.median(vals)),
                   "min": float(np.min(vals)),
                   "max": float(np.max(vals))}
        means.append((g, m, n))

    if len(means) < 2:
        summary: Dict[str, Any] = {
            "checklist_item": "TRIPOD+AI Item 14 + NIHMS Data Analysis 6 (fairness)",
            "metric_name": metric_name,
            "sensitive_attribute_levels": list(rows.keys()),
            "per_group": rows,
            "warning": "fewer than 2 groups with n >= min_group_n - disparity metrics undefined",
        }
        return summary

    g_max, m_max, _ = max(means, key=lambda x: x[1])
    g_min, m_min, _ = min(means, key=lambda x: x[1])
    gap = m_max - m_min
    ratio = (m_min / m_max) if m_max != 0 else float("nan")
    # 80% rule (Equal Employment Opportunity Commission, Uniform Guidelines)
    eighty_rule_pass = bool(ratio >= 0.8)

    return {
        "checklist_item": "TRIPOD+AI Item 14 + NIHMS Data Analysis 6 (fairness)",
        "metric_name": metric_name,
        "sensitive_attribute_levels": list(rows.keys()),
        "per_group": rows,
        "best_group": {"name": g_max, "mean": m_max},
        "worst_group": {"name": g_min, "mean": m_min},
        "max_min_gap": gap,
        "disparity_ratio": ratio,
        "eighty_percent_rule": {
            "ratio_threshold": 0.8,
            "passes": eighty_rule_pass,
            "interpretation": (
                "no disparate-impact-style concern under the 80% rule"
                if eighty_rule_pass
                else "DISPARATE IMPACT FLAG: worst-group performance is < 80% of best-group"
            ),
        },
    }


def compute_concentration_index(
    per_case_metric: Sequence[float],
    socioeconomic_rank: Sequence[float],
) -> Dict[str, float]:
    """Concentration index (Wagstaff 1991, O'Donnell et al. 2016).

    NIHMS RAND REACH §"Equity-Oriented Metrics": "Employ metrics like
    concentration indices and use disaggregated data to quantify and monitor
    disparities effectively."

    Definition.  Sort cases by socioeconomic rank R ∈ [0, 1] (e.g.,
    health-area income-decile, or local field-strength availability index
    for an LMIC study).  Let h_i be the per-case health metric (here Dice
    or 1-FPR - anything where higher = better).  The concentration index is

        CI = 2 / (n · μ_h) · Σ_i  (h_i · (R_i - 0.5))

    interpretation:
        CI > 0  →  better metric concentrated among higher-rank (richer) groups
                   (i.e., model favours high-resource sites - INEQUITABLE)
        CI = 0  →  perfect equality across socioeconomic rank
        CI < 0  →  better metric concentrated among lower-rank (poorer) groups
                   (i.e., model favours under-resourced sites - pro-poor)

    For an LMIC-deployment study the desirable direction is CI ≤ 0 (the
    model performs at least as well at low-resource sites).
    """
    h = np.asarray([float(x) for x in per_case_metric])
    r = np.asarray([float(x) for x in socioeconomic_rank])
    if h.size != r.size:
        m = min(h.size, r.size)
        h = h[:m]; r = r[:m]
    n = int(h.size)
    if n < 2:
        return {"checklist_item": "NIHMS RAND REACH equity (concentration index)",
                "n_samples": n, "concentration_index": float("nan"),
                "warning": "n < 2; index undefined"}
    mean_h = float(np.mean(h))
    if mean_h == 0:
        return {"checklist_item": "NIHMS RAND REACH equity (concentration index)",
                "n_samples": n, "concentration_index": float("nan"),
                "warning": "mean of per-case metric is zero; index undefined"}
    # Rescale rank to [0, 1] uniformly (fractional rank).
    order = np.argsort(r)
    frac_rank = np.empty(n, dtype=np.float64)
    frac_rank[order] = (np.arange(n) + 0.5) / n
    ci = float((2.0 / (n * mean_h)) * np.sum(h * (frac_rank - 0.5)))
    return {
        "checklist_item": "NIHMS RAND REACH equity (concentration index)",
        "n_samples": n,
        "mean_metric": mean_h,
        "concentration_index": ci,
        "interpretation": (
            "pro-poor (model favours low-rank sites)" if ci < -0.02
            else "approximately equitable" if abs(ci) <= 0.02
            else "INEQUITABLE: model favours high-rank sites"
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SAGER 3b / 4c / 5a / 6a - Sex / gender breakdown
# ══════════════════════════════════════════════════════════════════════════════

# Missing-value sentinels seen across the real cohort metadata: blank, textual
# "unknown"/"not reported", and the numeric -1 used by the Erasmus clinical export.
# Without -1 here, Erasmus would report a spurious "-1" sex category.
_SEX_MISSING_VALUES = frozenset(
    {"", "na", "n/a", "nan", "none", "null", "unknown", "unk", "not reported", "-1", "-"}
)


def compute_sex_breakdown(
    csv_path: Union[str, Path],
    *,
    sex_columns: Sequence[str] = ("Sex at birth", "sex", "Sex", "Sex_clean", "gender", "Gender"),
    definition_method: str = (
        "Sex extracted from source dataset metadata "
        "(BraTS / UTSW manifest); mode of ascertainment varies by source - "
        "self-report, EHR-derived, or chart-review per source documentation."
    ),
) -> Dict[str, Any]:
    """SAGER-style sex / gender breakdown of any cohort CSV.

    SAGER 2022 Items 3b, 4c, 5a, 6a, 6g.

    Args
    ----
    csv_path : Path
        Manifest CSV - expected to contain at least a sex column (one of
        the names in ``sex_columns``).
    sex_columns : sequence of str
        Candidate column names; the first one present is used.
    definition_method : str
        SAGER 5a - how sex was defined / ascertained.  Override at call
        site if your data source uses something more specific.

    Returns
    -------
    dict suitable for direct embedding in the manuscript "Table 1".
    """
    csv_path = Path(csv_path)
    counts: Dict[str, int] = defaultdict(int)
    sex_col_used: Optional[str] = None
    n_total = 0
    n_unknown = 0
    if not csv_path.exists():
        return {
            "checklist_item": "SAGER 2022 Items 3b/4c/5a/6a (sex/gender breakdown)",
            "error": f"CSV not found: {csv_path}",
        }
    with open(csv_path, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            n_total += 1
            if sex_col_used is None:
                for col in sex_columns:
                    if col in row:
                        sex_col_used = col
                        break
            v = (row.get(sex_col_used) or "").strip() if sex_col_used else ""
            if not v or v.lower() in _SEX_MISSING_VALUES:
                n_unknown += 1
                continue
            counts[_normalize_sex_value(v)] += 1

    breakdown = dict(counts)
    breakdown_pct = {k: round(100.0 * v / max(n_total, 1), 2) for k, v in counts.items()}
    # Honest status so an absent/empty breakdown is never mistaken for a computed one:
    # BraTS-Africa metadata has NO sex column; an empty CSV has no rows.
    if n_total == 0:
        status = "no_data"
    elif sex_col_used is None:
        status = "no_sex_column"
    elif sum(counts.values()) == 0:
        status = "all_unknown"
    else:
        status = "ok"
    return {
        "checklist_item": "SAGER 2022 Items 3b/4c/5a/6a/6g (sex/gender breakdown)",
        "status": status,
        "csv_path": str(csv_path),
        "sex_column_used": sex_col_used,
        "definition_method": definition_method,   # SAGER 5a
        "n_total":    n_total,
        "n_unknown":  n_unknown,
        "breakdown_count":   breakdown,
        "breakdown_percent": breakdown_pct,
        "categories_used":   list(counts.keys()),
    }


def _normalize_sex_value(v: str) -> str:
    s = v.strip().lower()
    if s in ("m", "male"):           return "male"
    if s in ("f", "female"):         return "female"
    if s in ("intersex", "i"):       return "intersex"
    if s.startswith("non"):          return "non-binary"
    return v.strip()


# ══════════════════════════════════════════════════════════════════════════════
#  TRIPOD 20c - Demographic distribution comparison (development vs. evaluation)
# ══════════════════════════════════════════════════════════════════════════════

def compute_development_vs_evaluation_table(
    train_csv: Union[str, Path],
    eval_csv: Union[str, Path],
    *,
    columns: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Side-by-side demographic comparison.

    TRIPOD+AI Item 20c: "For model evaluation, show a comparison with the
    development data of the distribution of important variables (demographics,
    predictors, and outcome)."

    For each ``column`` reports counts and percentages in train vs. eval and
    the L1 distribution difference.
    """
    columns = list(columns) if columns is not None else [
        "Sex at birth", "Race", "Ethnicity",
        "Tumor Grade", "IDH", "Scanner Make", "Scanner Strength",
    ]
    train_rows = list(_iter_rows(train_csv))
    eval_rows  = list(_iter_rows(eval_csv))
    out: Dict[str, Any] = {
        "checklist_item": "TRIPOD+AI Item 20c (development vs. evaluation comparison)",
        "n_train": len(train_rows), "n_eval": len(eval_rows),
        "columns": {},
    }
    for col in columns:
        tr = _value_counts(train_rows, col)
        ev = _value_counts(eval_rows,  col)
        all_keys = sorted(set(tr) | set(ev))
        rows = []
        l1 = 0.0
        for k in all_keys:
            n_tr = tr.get(k, 0); n_ev = ev.get(k, 0)
            p_tr = n_tr / max(len(train_rows), 1)
            p_ev = n_ev / max(len(eval_rows),  1)
            l1 += abs(p_tr - p_ev)
            rows.append({
                "value": k,
                "n_train": n_tr, "pct_train": round(100 * p_tr, 2),
                "n_eval":  n_ev, "pct_eval":  round(100 * p_ev,  2),
                "abs_diff_pct": round(100 * abs(p_tr - p_ev), 2),
            })
        out["columns"][col] = {
            "rows": rows,
            "l1_distance": round(l1, 4),
        }
    return out


def _iter_rows(csv_path: Union[str, Path]) -> Iterable[Dict[str, str]]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return []
    with open(csv_path, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    return rows


def _value_counts(rows: Sequence[Dict[str, str]], col: str) -> Dict[str, int]:
    out: Dict[str, int] = defaultdict(int)
    for r in rows:
        v = (r.get(col) or "").strip()
        if not v or v.lower() in ("na", "n/a", "unknown", "not reported"):
            continue
        out[v] += 1
    return dict(out)


# ══════════════════════════════════════════════════════════════════════════════
#  Orchestrator subcommand - `ai_compliance_report`
# ══════════════════════════════════════════════════════════════════════════════

def cmd_ai_compliance_report(args: Any, core: Any) -> None:  # noqa: ANN401
    """Single-shot orchestrator that runs the full compliance battery.

    Invokes (in order):
        1. dump_environment_snapshot   (CLAIM 23)
        2. dump_initialization_report  (CLAIM 24)              [if --ckpt provided]
        3. compute_sex_breakdown       (SAGER 3b/4c/5a/6a/6g)  [for train and eval]
        4. compute_development_vs_evaluation_table (TRIPOD 20c)
        5. compute_class_imbalance_metrics (CLAIM 38)          [if --probs_npz provided]
        6. compute_calibration_slope_intercept (TRIPOD 12e)    [if --probs_npz provided]
        7. compute_fairness_metrics    (TRIPOD 14, NIHMS)      [if metadata TSV provided]
        8. compute_concentration_index (NIHMS)                 [if rank column present]
        9. Writes a per-checklist mapping JSON for CLAIM, TRIPOD+AI, SAGER, NIHMS.
       10. Writes a unified compliance_report.json + compliance_checklist.csv.

    The previously-existing subcommands ``stratify``, ``calibration``,
    ``failure_case_figure``, ``data_flow``, ``model_card``, ``datasheet`` and
    ``interrater`` are NOT invoked from here (they have heavier dependencies);
    the user is expected to run them separately and the orchestrator records
    their expected output paths in the unified report so reviewers can find
    them.
    """
    out_dir = _ensure_dir(args.out_dir)
    LOG.info("[ai_compliance_report] writing to %s", out_dir)

    bundle: Dict[str, Any] = {
        "checklists_covered": [
            "CLAIM 2024 (Tejani et al., Radiology AI 2024)",
            "TRIPOD+AI (Collins et al., BMJ 2024)",
            "SAGER 2022 (Heidari et al., EASE)",
            "NIHMS RAND REACH (Malika et al., Ethics Med Public Health 2025)",
        ],
        "captured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "out_dir": str(out_dir),
        "outputs": {},
        "sections_failed": [],
    }

    # 1. Environment.
    bundle["outputs"]["claim_23_environment"] = dump_environment_snapshot(out_dir)

    # 2. Initialization (only if a checkpoint is supplied).
    if getattr(args, "ckpt", None):
        torch = _try_import("torch")
        if torch is None:
            LOG.warning("[ai_compliance_report] torch unavailable - skipping init report")
        else:
            try:
                ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
                state_dict = (
                    ckpt.get("model_state_dict")
                    or ckpt.get("state_dict")
                    or ckpt
                )
                if not isinstance(state_dict, dict):
                    raise ValueError("checkpoint is not a state dict")
                # Build a tiny module that holds the parameters so we can iterate
                # named_parameters() without needing the full model architecture.
                container = torch.nn.ParameterDict({
                    k.replace(".", "_"): torch.nn.Parameter(
                        v.detach().float() if torch.is_tensor(v) else torch.zeros(0))
                    for k, v in state_dict.items()
                    if torch.is_tensor(v)
                })
                bundle["outputs"]["claim_24_initialization"] = dump_initialization_report(
                    container, out_dir,
                    transfer_learning_ckpt=args.ckpt,
                    transfer_learning_role=getattr(args, "ckpt_role", "-"),
                )
            except Exception as exc:
                LOG.error("[ai_compliance_report] init report FAILED: %s", exc)
                bundle["sections_failed"].append("claim_24_initialization")

    # 3. Sex breakdown (train + eval).
    if getattr(args, "train_csv", None):
        bundle["outputs"]["sager_train_breakdown"] = compute_sex_breakdown(args.train_csv)
        _save_json(bundle["outputs"]["sager_train_breakdown"],
                   out_dir / "sager_train_sex_breakdown.json")
    if getattr(args, "val_csv", None):
        bundle["outputs"]["sager_eval_breakdown"] = compute_sex_breakdown(args.val_csv)
        _save_json(bundle["outputs"]["sager_eval_breakdown"],
                   out_dir / "sager_eval_sex_breakdown.json")

    # 4. Development vs. evaluation comparison (TRIPOD 20c).
    if getattr(args, "train_csv", None) and getattr(args, "val_csv", None):
        bundle["outputs"]["tripod_20c_dev_vs_eval"] = compute_development_vs_evaluation_table(
            args.train_csv, args.val_csv,
        )
        _save_json(bundle["outputs"]["tripod_20c_dev_vs_eval"],
                   out_dir / "tripod_20c_development_vs_evaluation.json")

    # 5-6. Class-imbalance + calibration slope/intercept from a probs NPZ.
    probs_npz = getattr(args, "probs_npz", None)
    if probs_npz and Path(probs_npz).exists():
        try:
            d = np.load(probs_npz)
            probs = d["probs"]
            labels = d["labels"]
            bundle["outputs"]["claim_38_class_imbalance"] = compute_class_imbalance_metrics(
                probs, labels)
            _save_json(bundle["outputs"]["claim_38_class_imbalance"],
                       out_dir / "claim_38_class_imbalance.json")
            bundle["outputs"]["tripod_12e_calibration_slope"] = (
                compute_calibration_slope_intercept(probs, labels))
            _save_json(bundle["outputs"]["tripod_12e_calibration_slope"],
                       out_dir / "tripod_12e_calibration_slope.json")
        except Exception as exc:
            LOG.error("[ai_compliance_report] probs_npz analysis FAILED: %s", exc)
            bundle["sections_failed"].extend(
                ["claim_38_class_imbalance", "tripod_12e_calibration_slope"])

    # 7. Fairness - needs per-case metrics CSV plus a sensitive-attribute column.
    per_case_csv = getattr(args, "per_case_csv", None)
    sensitive_attribute = getattr(args, "sensitive_attribute", None)
    metric_col = getattr(args, "metric_col", "pp_all_dice_brats_mean")
    if per_case_csv and sensitive_attribute and Path(per_case_csv).exists():
        rows = list(_iter_rows(per_case_csv))
        vals = [_safe_float(r.get(metric_col, "")) for r in rows]
        sens = [str(r.get(sensitive_attribute, "")) for r in rows]
        vals_clean = [(v, s) for v, s in zip(vals, sens) if v is not None]
        if vals_clean:
            v_arr = [v for v, _ in vals_clean]
            s_arr = [s for _, s in vals_clean]
            bundle["outputs"]["tripod_14_fairness"] = compute_fairness_metrics(
                v_arr, s_arr, metric_name=metric_col,
            )
            _save_json(bundle["outputs"]["tripod_14_fairness"],
                       out_dir / "tripod_14_fairness.json")

    # 8. Concentration index - needs per-case metrics + numeric SES rank.
    ses_rank_col = getattr(args, "ses_rank_col", None)
    if per_case_csv and ses_rank_col and Path(per_case_csv).exists():
        rows = list(_iter_rows(per_case_csv))
        vals = [_safe_float(r.get(metric_col, "")) for r in rows]
        ranks = [_safe_float(r.get(ses_rank_col, "")) for r in rows]
        pairs = [(v, r) for v, r in zip(vals, ranks) if v is not None and r is not None]
        if pairs:
            bundle["outputs"]["nihms_concentration_index"] = compute_concentration_index(
                [v for v, _ in pairs], [r for _, r in pairs],
            )
            _save_json(bundle["outputs"]["nihms_concentration_index"],
                       out_dir / "nihms_concentration_index.json")

    # 9. Per-checklist mappings (so reviewers can navigate by item number).
    _save_json(CLAIM_CHECKLIST_MAPPING,         out_dir / "claim_2024_checklist_mapping.json")
    _save_json(TRIPOD_AI_CHECKLIST_MAPPING,     out_dir / "tripod_ai_checklist_mapping.json")
    _save_json(SAGER_CHECKLIST_MAPPING,         out_dir / "sager_2022_checklist_mapping.json")
    _save_json(NIHMS_HEALTH_EQUITY_MAPPING,     out_dir / "nihms_health_equity_mapping.json")

    # 10. Unified report + checklist CSV. Flag completeness so an INCOMPLETE report (a
    # sub-step raised, or a section self-reported missing data) is never mistaken for a
    # complete one by a reviewer or an automated check.
    for _key, _sub in bundle["outputs"].items():
        if _key in bundle["sections_failed"]:
            continue
        if isinstance(_sub, dict) and (
            "error" in _sub
            or str(_sub.get("status", "")).lower()
            in ("no_data", "no_sex_column", "all_unknown", "error", "incomplete")
            or _sub.get("converged") is False
        ):
            bundle["sections_failed"].append(_key)
    bundle["report_complete"] = len(bundle["sections_failed"]) == 0
    if not bundle["report_complete"]:
        LOG.error("[ai_compliance_report] report INCOMPLETE; sections_failed=%s",
                  bundle["sections_failed"])
    _save_json(bundle, out_dir / "compliance_report.json")
    _save_compliance_checklist_csv(out_dir / "compliance_checklist.csv")
    LOG.info("[ai_compliance_report] done - %s", out_dir / "compliance_report.json")


def _safe_float(s: Any) -> Optional[float]:
    try:
        x = float(s)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    except (TypeError, ValueError):
        return None


def _save_compliance_checklist_csv(path: Path) -> None:
    """Single CSV with one row per checklist item across all four checklists."""
    rows: List[Dict[str, Any]] = []
    for k, v in CLAIM_CHECKLIST_MAPPING.items():
        rows.append({
            "checklist": "CLAIM 2024", "item": str(k),
            "description": v["item"], "where": v["where"], "code": v["code"],
        })
    for k, v in TRIPOD_AI_CHECKLIST_MAPPING.items():
        rows.append({
            "checklist": "TRIPOD+AI", "item": k,
            "description": v["item"], "where": v["where"], "code": v["code"],
        })
    for k, v in SAGER_CHECKLIST_MAPPING.items():
        rows.append({
            "checklist": "SAGER 2022", "item": k,
            "description": v["item"], "where": v["where"], "code": v["code"],
        })
    for k, v in NIHMS_HEALTH_EQUITY_MAPPING.items():
        rows.append({
            "checklist": "NIHMS RAND REACH", "item": k,
            "description": v["item"], "where": v["where"], "code": v["code"],
        })
    _save_csv(rows, path)


# ══════════════════════════════════════════════════════════════════════════════
#  Subcommand registration helper.  The main script's register_subcommands()
#  imports this module lazily and calls register_compliance_subcommand().
# ══════════════════════════════════════════════════════════════════════════════

def register_compliance_subcommand(sub: Any) -> None:  # noqa: ANN401
    """Register the ``ai_compliance_report`` subcommand on the main parser.

    The main ``OGAP_source_code_experimental_v9.py`` calls this from
    ``register_subcommands(sub)`` so the orchestrator is reachable as

        python OGAP_source_code_experimental_v9.py ai_compliance_report \
            --out_dir Results/student_*/compliance \
            --train_csv Scripts/train.csv \
            --val_csv   Scripts/val.csv \
            --ckpt      Results/student_*/best_student.pth \
            [--probs_npz       eval/probs_labels.npz] \
            [--per_case_csv    eval/per_case_metrics.csv] \
            [--sensitive_attribute "Sex at birth"] \
            [--ses_rank_col    field_strength_rank] \
            [--metric_col      pp_all_dice_brats_mean]
    """
    p = sub.add_parser(
        "ai_compliance_report",
        help="Run the full CLAIM 2024 + TRIPOD+AI + SAGER 2022 + NIHMS health-equity "
             "compliance battery and write JSON + CSV under --out_dir.",
    )
    p.add_argument("--out_dir", required=True,
                   help="Directory for compliance artefacts (created if missing).")
    p.add_argument("--train_csv", default=None,
                   help="Training manifest CSV (used for SAGER + TRIPOD 20c).")
    p.add_argument("--val_csv",   default=None,
                   help="Internal validation / external evaluation manifest CSV.")
    p.add_argument("--ckpt",      default=None,
                   help="Model checkpoint (.pth) for the CLAIM 24 init report.")
    p.add_argument("--ckpt_role", default="student",
                   help="Free-text role of the checkpoint (e.g. 'student', 'teacher').")
    p.add_argument("--probs_npz", default=None,
                   help="NPZ file with arrays 'probs' (N,C) or (N,) and 'labels' (N,) "
                        "for CLAIM 38 class-imbalance + TRIPOD 12e calibration slope.")
    p.add_argument("--per_case_csv", default=None,
                   help="Per-case metrics CSV (e.g. eval/per_case_metrics.csv).")
    p.add_argument("--metric_col", default="pp_all_dice_brats_mean",
                   help="Column name of the per-case metric used for fairness / equity.")
    p.add_argument("--sensitive_attribute", default=None,
                   help="Column name of the sensitive attribute (e.g. 'Sex at birth') "
                        "for TRIPOD 14 fairness.")
    p.add_argument("--ses_rank_col", default=None,
                   help="Numeric column with socioeconomic rank for the NIHMS "
                        "concentration index (e.g. site-level field-strength rank).")


# ══════════════════════════════════════════════════════════════════════════════
#  __all__
# ══════════════════════════════════════════════════════════════════════════════

__all__ = [
    "CLAIM_CHECKLIST_MAPPING",
    "TRIPOD_AI_CHECKLIST_MAPPING",
    "SAGER_CHECKLIST_MAPPING",
    "NIHMS_HEALTH_EQUITY_MAPPING",
    "compute_calibration_slope_intercept",
    "compute_class_imbalance_metrics",
    "compute_concentration_index",
    "compute_development_vs_evaluation_table",
    "compute_fairness_metrics",
    "compute_sex_breakdown",
    "dump_environment_snapshot",
    "dump_initialization_report",
    "cmd_ai_compliance_report",
    "register_compliance_subcommand",
]
