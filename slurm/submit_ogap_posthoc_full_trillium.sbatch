#!/bin/bash
# Full OGAP post-hoc analysis bundle for Trillium.
# Intended to be submitted with a dependency after the main eval job:
#   sbatch --dependency=afterok:<eval_job> --export=ALL submit_ogap_posthoc_full_trillium.sbatch
#
# ── Trillium adaptation (vs the Rorqual original) ───────────────────────────
#   • --account=def-rdiaz; no --tmp (diskless).
#   • Trillium has NO MIG: the Rorqual version used a 2g.20gb MIG slice
#     (--gpus=nvidia_h100_80gb_hbm3_2g.20gb:1).  On Trillium this job takes a
#     FULL H100 (--gpus-per-node=1) - the post-hoc bundle is GPU-light, so
#     this slightly over-provisions, but no fractional GPU exists on Trillium.
#   • 24 cores requested (1/4-node max); memory is auto 188 GiB - do NOT set --mem.

#SBATCH --account=def-rdiaz
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --gpus-per-node=1
#SBATCH --job-name=ogap_posthoc_full
#SBATCH --output=ogap_posthoc_full_%j.out
#SBATCH --error=ogap_posthoc_full_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/project/def-rdiaz/ilyaso/OGAP}"
DATA_ROOT="${DATA_ROOT:-/project/def-rdiaz/ilyaso/Datasets}"
SCRIPT_PATH="${SCRIPT_PATH:-${PROJECT_ROOT}/Scripts/OGAP_source_code_experimental_v9.py}"
ENV_TO_SOURCE="${MAMBA_ENV_PATH:-${ENV_PATH:-${HOME}/ogap/envs/ogap_env_v2}}"

TRAIN_CSV="${TRAIN_CSV:-${PROJECT_ROOT}/Scripts/train.csv}"
TRAIN_VAL_CSV="${TRAIN_VAL_CSV:-${PROJECT_ROOT}/Scripts/val.csv}"
EVAL_VAL_CSV="${EVAL_VAL_CSV:-${PROJECT_ROOT}/Scripts/val.csv}"
# Held-out TEST set: the headline internal Dice/HD95 are reported on THIS set
# (never used for checkpoint selection or tuning). val.csv is selection-only.
TEST_CSV="${TEST_CSV:-${PROJECT_ROOT}/Scripts/test.csv}"
export TEST_CSV
STUDENT_CKPT="${STUDENT_CKPT:-${SCRATCH:-${HOME}/scratch}/ogap/Results/student_external_push/best_student.pth}"
STUDENT_BASE="${STUDENT_BASE:-16}"
TEACHER_OUT_DIR="${TEACHER_OUT_DIR:-${SCRATCH:-${HOME}/scratch}/ogap/Results/teacher}"
# holder_sweep / kd_compare below consume TEACHER_CKPT; define it so the script
# does not abort with "unbound variable" under `set -u`.
TEACHER_CKPT="${TEACHER_CKPT:-${TEACHER_OUT_DIR}/best_teacher.pth}"
STUDENT_OUT_DIR="${STUDENT_OUT_DIR:-${SCRATCH:-${HOME}/scratch}/ogap/Results/student_external_push}"
EVAL_OUT_DIR="${EVAL_OUT_DIR:-${SCRATCH:-${HOME}/scratch}/ogap/Results/utsw_full_evaluation_tta}"
EXPORT_OUT_DIR="${EXPORT_OUT_DIR:-${STUDENT_OUT_DIR}_export_int8}"
QUANT_COMPARE_OUT_DIR="${QUANT_COMPARE_OUT_DIR:-${EXPORT_OUT_DIR}/quantisation_comparison}"

METADATA_ROOT="${METADATA_ROOT:-/project/def-rdiaz/ilyaso/Datasets/METADATA}"
UTSW_METADATA_TSV="${UTSW_METADATA_TSV:-${METADATA_ROOT}/UTSW_Glioma_Metadata-2-1.tsv}"
BRATS_AFRICA_METADATA_XLSX="${BRATS_AFRICA_METADATA_XLSX:-${METADATA_ROOT}/BraTS-Africa_TCIA_datainfo_v2-2.xlsx}"
ERASMUS_CLINICAL_XLSX="${ERASMUS_CLINICAL_XLSX:-${METADATA_ROOT}/Clinical_data.xlsx}"
ERASMUS_GENETIC_HISTOLOGY_XLSX="${ERASMUS_GENETIC_HISTOLOGY_XLSX:-${METADATA_ROOT}/Genetic_and_Histological_labels.xlsx}"
ERASMUS_SCAN_XLSX="${ERASMUS_SCAN_XLSX:-${METADATA_ROOT}/Scan_characteristics.xlsx}"
CLINICAL_METADATA_TSV="${CLINICAL_METADATA_TSV:-${UTSW_METADATA_TSV}}"

POSTHOC_LABEL="${POSTHOC_LABEL:-ogap}"
POSTHOC_OUT="${POSTHOC_OUT:-${SCRATCH:-${HOME}/scratch}/ogap/Results/posthoc_${POSTHOC_LABEL}}"
POSTHOC_NUM_WORKERS="${POSTHOC_NUM_WORKERS:-4}"
POSTHOC_METRIC_WORKERS="${POSTHOC_METRIC_WORKERS:-2}"
POSTHOC_EVAL_CONFIG_BATCH_SIZE="${POSTHOC_EVAL_CONFIG_BATCH_SIZE:-4}"
POSTHOC_N_BOOT="${POSTHOC_N_BOOT:-10000}"

module --force purge
module load StdEnv/2023 python/3.11.5 cuda/12.6 scipy-stack/2024a
source "${ENV_TO_SOURCE}/bin/activate"

if [[ -n "${MAMBA_ENV_PATH:-}" ]]; then
  export EXPECTED_TORCH_PUBLIC_VERSION="${MAMBA_EXPECTED_TORCH_PUBLIC_VERSION:-2.5.1}"
  export OGAP_DISABLE_TORCH_COMPILE=1
fi
export OGAP_DISABLE_N4=1
export OGAP_N4_THREADS=1
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1

mkdir -p "${SCRATCH:-${HOME}/scratch}/ogap/logs" "${POSTHOC_OUT}"
cd "${PROJECT_ROOT}"

for pair in \
  "SCRIPT_PATH=${SCRIPT_PATH}" \
  "TRAIN_CSV=${TRAIN_CSV}" \
  "TRAIN_VAL_CSV=${TRAIN_VAL_CSV}" \
  "EVAL_VAL_CSV=${EVAL_VAL_CSV}" \
  "TEST_CSV=${TEST_CSV}" \
  "STUDENT_CKPT=${STUDENT_CKPT}" \
  "UTSW_METADATA_TSV=${UTSW_METADATA_TSV}"; do
  var="${pair%%=*}"
  path="${pair#*=}"
  if [[ ! -e "${path}" ]]; then
    echo "[ERROR][posthoc] Required path (${var}) not found: ${path}" >&2
    exit 1
  fi
done

echo "[posthoc] output=${POSTHOC_OUT}"

python "${SCRIPT_PATH}" lint_csv --csv "${TRAIN_CSV}" --modalities t1n,t1c,t2w,t2f --train_csv "${TRAIN_CSV}" --check_label_values --out_dir "${POSTHOC_OUT}/lint_train"
python "${SCRIPT_PATH}" lint_csv --csv "${TRAIN_VAL_CSV}" --modalities t1n,t1c,t2w,t2f --train_csv "${TRAIN_CSV}" --check_label_values --out_dir "${POSTHOC_OUT}/lint_internal_val"
python "${SCRIPT_PATH}" lint_csv --csv "${EVAL_VAL_CSV}" --modalities t1n,t1c,t2w,t2f --train_csv "${TRAIN_CSV}" --check_label_values --out_dir "${POSTHOC_OUT}/lint_external_utsw"
python "${SCRIPT_PATH}" sampler_distribution --csv "${TRAIN_CSV}" --out_dir "${POSTHOC_OUT}/sampler_distribution"
python "${SCRIPT_PATH}" validate_augmentor --out_dir "${POSTHOC_OUT}/validate_augmentor"

INTERNAL_EVAL_OUT="${POSTHOC_OUT}/internal_validation_eval"
python "${SCRIPT_PATH}" evaluate \
  --ckpt "${STUDENT_CKPT}" \
  --val_csv "${TEST_CSV}" \
  --out_dir "${INTERNAL_EVAL_OUT}" \
  --modalities t1n,t1c,t2w,t2f \
  --num_classes 4 \
  --patch_size 128,160,112 \
  --student_base "${STUDENT_BASE}" \
  --device cuda \
  --num_workers "${POSTHOC_NUM_WORKERS}" \
  --metric_workers "${POSTHOC_METRIC_WORKERS}" \
  --eval_config_batch_size "${POSTHOC_EVAL_CONFIG_BATCH_SIZE}" \
  --tta \
  --lesion_wise

python - <<'PY'
import csv, json, os, re
from pathlib import Path

import pandas as pd

posthoc = Path(os.environ["POSTHOC_OUT"])
val_csv = Path(os.environ["TEST_CSV"])  # headline reporting is on the held-out TEST set
out_tsv = posthoc / "internal_validation_metadata.tsv"
missing_json = posthoc / "internal_validation_metadata_missingness.json"

def norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())

def digits(s):
    d = re.sub(r"\D+", "", str(s))
    return d.lstrip("0") or d

def load_excel(path, sheet=0):
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_excel(p, sheet_name=sheet)

africa = load_excel(os.environ.get("BRATS_AFRICA_METADATA_XLSX", ""), sheet=0)
erasmus_clin = load_excel(os.environ.get("ERASMUS_CLINICAL_XLSX", ""))
erasmus_gen = load_excel(os.environ.get("ERASMUS_GENETIC_HISTOLOGY_XLSX", ""))
erasmus_scan = load_excel(os.environ.get("ERASMUS_SCAN_XLSX", ""))

africa_by_digits = {}
if not africa.empty and "Subject ID (0x)" in africa.columns:
    for _, r in africa.iterrows():
        africa_by_digits[digits(r.get("Subject ID (0x)", ""))] = r.to_dict()

def by_subject(df):
    out = {}
    if not df.empty and "Subject" in df.columns:
        for _, r in df.iterrows():
            out[norm(r.get("Subject", ""))] = r.to_dict()
    return out

erasmus_clin_by = by_subject(erasmus_clin)
erasmus_gen_by = by_subject(erasmus_gen)
erasmus_scan_by = by_subject(erasmus_scan)

def cohort_for(case_id, tag):
    text = f"{case_id} {tag}".lower()
    if "africa" in text:
        return "BraTS Africa", "LMIC"
    if "erasmus" in text or str(case_id).upper().startswith("EGD"):
        return "ERASMUS glioma", "High-income"
    if "brats" in text:
        return "BraTS 2023", "High-income"
    return str(tag or "Unknown"), ""

rows = []
with open(val_csv, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        case_id = str(r.get("case_id", "")).strip()
        tag = str(r.get("dataset_tag", "")).strip()
        cohort, income = cohort_for(case_id, tag)
        out = {
            "Subject ID": case_id,
            "Cohort": cohort,
            "Income Setting": income,
            "Sex at birth": r.get("Sex at birth") or r.get("sex") or r.get("Sex") or "",
            "Age at Imaging": r.get("Age at Imaging") or r.get("age") or r.get("Age") or "",
            "Race": r.get("Race") or r.get("race") or "",
            "Ethnicity": r.get("Ethnicity") or r.get("ethnicity") or "",
            "Tumor Grade": r.get("Tumor Grade") or r.get("grade") or "",
            "IDH": r.get("IDH") or r.get("idh") or "",
            "1p19Q CODEL": r.get("1p19Q CODEL") or r.get("1p19q") or "",
            "MGMT": r.get("MGMT") or r.get("mgmt") or "",
            "Scanner Make": r.get("Scanner Make") or r.get("scanner_make") or "",
            "Scanner Model": r.get("Scanner Model") or r.get("scanner_model") or "",
            "Scanner Strength": r.get("Scanner Strength") or r.get("field_strength") or "",
            "Manually Refined Segmentation": r.get("Manually Refined Segmentation") or "",
        }
        ncase = norm(case_id)
        if cohort == "ERASMUS glioma":
            c = erasmus_clin_by.get(ncase, {})
            g = erasmus_gen_by.get(ncase, {})
            s = erasmus_scan_by.get(ncase, {})
            out["Age at Imaging"] = out["Age at Imaging"] or c.get("Age", "")
            out["Sex at birth"] = out["Sex at birth"] or c.get("Sex", "")
            out["Tumor Grade"] = out["Tumor Grade"] or g.get("Grade", "")
            out["IDH"] = out["IDH"] or g.get("IDH", "")
            out["1p19Q CODEL"] = out["1p19Q CODEL"] or g.get("1p19q", "")
            out["Scanner Make"] = out["Scanner Make"] or s.get("Manufacturer", "")
            out["Scanner Model"] = out["Scanner Model"] or s.get("System", "")
            out["Scanner Strength"] = out["Scanner Strength"] or s.get("Field", "")
        elif cohort == "BraTS Africa":
            a = africa_by_digits.get(digits(case_id), {})
            out["Scanner Model"] = out["Scanner Model"] or a.get("Scanner Information", "")
            out["Scanner Make"] = out["Scanner Make"] or a.get("Center", "")
        rows.append(out)

fieldnames = [
    "Subject ID", "Cohort", "Income Setting", "Sex at birth", "Age at Imaging",
    "Race", "Ethnicity", "Tumor Grade", "IDH", "1p19Q CODEL", "MGMT",
    "Scanner Make", "Scanner Model", "Scanner Strength", "Manually Refined Segmentation",
]
with open(out_tsv, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
    w.writeheader()
    w.writerows(rows)

missing = {
    col: sum(1 for r in rows if str(r.get(col, "")).strip() == "")
    for col in fieldnames
}
json.dump({"n_rows": len(rows), "missing_by_column": missing}, open(missing_json, "w"), indent=2)
print(f"[posthoc] wrote {out_tsv} rows={len(rows)}")
PY

python "${SCRIPT_PATH}" stratify --per_case "${INTERNAL_EVAL_OUT}/full_evaluation_per_case_metrics.csv" --metadata_tsv "${POSTHOC_OUT}/internal_validation_metadata.tsv" --metadata_id_col "Subject ID" --metric pp_all_dice_brats_mean --min_group_n 1 --n_boot "${POSTHOC_N_BOOT}" --out_dir "${POSTHOC_OUT}/internal_stratify"
python "${SCRIPT_PATH}" subgroup_forest --stratified_summary_json "${POSTHOC_OUT}/internal_stratify/stratified_summary.json" --out_dir "${POSTHOC_OUT}/internal_subgroup_forest"
python "${SCRIPT_PATH}" field_strength_curve --joined_csv "${POSTHOC_OUT}/internal_stratify/joined_per_case.csv" --metric pp_all_dice_brats_mean --n_boot "${POSTHOC_N_BOOT}" --out_dir "${POSTHOC_OUT}/internal_field_strength_curve"

python "${SCRIPT_PATH}" calibration --ckpt "${STUDENT_CKPT}" --val_csv "${EVAL_VAL_CSV}" --modalities t1n,t1c,t2w,t2f --num_classes 4 --patch_size 128,160,112 --student_base "${STUDENT_BASE}" --device cuda --num_workers "${POSTHOC_NUM_WORKERS}" --out_dir "${POSTHOC_OUT}/calibration_external_utsw"
python "${SCRIPT_PATH}" mc_uncertainty --ckpts "${STUDENT_CKPT}" --val_csv "${EVAL_VAL_CSV}" --modalities t1n,t1c,t2w,t2f --num_classes 4 --patch_size 128,160,112 --student_base "${STUDENT_BASE}" --device cuda --num_workers "${POSTHOC_NUM_WORKERS}" --n_mc_samples 10 --save_maps --out_dir "${POSTHOC_OUT}/mc_uncertainty_external_utsw"
python "${SCRIPT_PATH}" compute_profile --ckpt "${STUDENT_CKPT}" --modalities t1n,t1c,t2w,t2f --num_classes 4 --patch_size 128,160,112 --student_base "${STUDENT_BASE}" --device cuda --n_warmup 5 --n_iter 25 --out_dir "${POSTHOC_OUT}/compute_profile"

python "${SCRIPT_PATH}" infer --test_csv "${EVAL_VAL_CSV}" --ckpt "${STUDENT_CKPT}" --out_dir "${POSTHOC_OUT}/infer_external_utsw_full" --modalities t1n,t1c,t2w,t2f --num_classes 4 --patch_size 128,160,112 --student_base "${STUDENT_BASE}" --device cuda --tta

python - <<'PY'
import csv, os
from pathlib import Path

eval_csv = Path(os.environ["EVAL_VAL_CSV"])
pred_dir = Path(os.environ["POSTHOC_OUT"]) / "infer_external_utsw_full"
pairs_csv = Path(os.environ["POSTHOC_OUT"]) / "external_prediction_gt_pairs.csv"
rows = []
with open(eval_csv, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        cid = str(r.get("case_id", "")).strip()
        pred = pred_dir / f"{cid}_pred.nii.gz"
        gt = Path(str(r.get("label", "")))
        if cid and pred.exists() and gt.exists():
            rows.append({"case_id": cid, "pred": str(pred), "gt": str(gt)})
with open(pairs_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["case_id", "pred", "gt"])
    w.writeheader()
    w.writerows(rows)
print(f"[posthoc] external pairs={len(rows)} -> {pairs_csv}")
PY

python "${SCRIPT_PATH}" metric_check --pairs_csv "${POSTHOC_OUT}/external_prediction_gt_pairs.csv" --out_dir "${POSTHOC_OUT}/metric_check_external_utsw"
python "${SCRIPT_PATH}" surface_dice --pairs_csv "${POSTHOC_OUT}/external_prediction_gt_pairs.csv" --tolerances_mm 1,2,3 --out_dir "${POSTHOC_OUT}/surface_dice_external_utsw"

python "${SCRIPT_PATH}" stratify --per_case "${EVAL_OUT_DIR}/full_evaluation_per_case_metrics.csv" --metadata_tsv "${CLINICAL_METADATA_TSV}" --metadata_id_col "Subject ID" --metric pp_all_dice_brats_mean --min_group_n 1 --n_boot "${POSTHOC_N_BOOT}" --out_dir "${POSTHOC_OUT}/external_stratify_utsw"
python "${SCRIPT_PATH}" subgroup_forest --stratified_summary_json "${POSTHOC_OUT}/external_stratify_utsw/stratified_summary.json" --out_dir "${POSTHOC_OUT}/external_subgroup_forest_utsw"
python "${SCRIPT_PATH}" field_strength_curve --joined_csv "${POSTHOC_OUT}/external_stratify_utsw/joined_per_case.csv" --metric pp_all_dice_brats_mean --n_boot "${POSTHOC_N_BOOT}" --out_dir "${POSTHOC_OUT}/external_field_strength_curve_utsw"

python "${SCRIPT_PATH}" failure_case_figure --per_case "${EVAL_OUT_DIR}/full_evaluation_per_case_metrics.csv" --metric pp_all_dice_brats_mean --out_dir "${POSTHOC_OUT}/failure_cases_external_utsw"
python "${SCRIPT_PATH}" training_curves --runs "${TEACHER_OUT_DIR},${STUDENT_OUT_DIR}" --out_dir "${POSTHOC_OUT}/training_curves"

python - <<'PY'
import json, os
from pathlib import Path

src = Path(os.environ["QUANT_COMPARE_OUT_DIR"]) / "quantisation_comparison.json"
dst = Path(os.environ["POSTHOC_OUT"]) / "quantisation_hardware_input.json"
if src.exists():
    data = json.load(open(src))
    rows = []
    for cfg, metrics in data.get("comparisons", {}).items():
        m = metrics.get("dice_brats_mean", {})
        if m:
            rows.append({
                "config": cfg,
                "fp32_mean_dice": m.get("fp32_mean"),
                "int8_mean_dice": m.get("int8_mean"),
            })
    data["per_config"] = rows
    json.dump(data, open(dst, "w"), indent=2)
else:
    json.dump({"per_config": []}, open(dst, "w"), indent=2)
PY
python "${SCRIPT_PATH}" quantization_hardware --compare_json "${POSTHOC_OUT}/quantisation_hardware_input.json" --out_dir "${POSTHOC_OUT}/quantization_hardware"
python "${SCRIPT_PATH}" total_energy_table --root "${SCRATCH:-${HOME}/scratch}/ogap/Results" --out_dir "${POSTHOC_OUT}/energy"

python - <<'PY'
import csv, json, os
from pathlib import Path

def count_rows(path):
    p = Path(path)
    if not p.exists():
        return 0
    with open(p, encoding="utf-8") as f:
        return max(0, sum(1 for _ in f) - 1)

out = Path(os.environ["POSTHOC_OUT"]) / "data_flow_spec.json"
spec = {
    "steps": [
        {"label": "Training manifest", "n": count_rows(os.environ["TRAIN_CSV"])},
        {"label": "Internal validation manifest", "n": count_rows(os.environ["TRAIN_VAL_CSV"])},
        {"label": "External UTSW evaluation manifest", "n": count_rows(os.environ["EVAL_VAL_CSV"])},
        {"label": "Internal validation cases evaluated", "n": count_rows(Path(os.environ["POSTHOC_OUT"]) / "internal_validation_eval/full_evaluation_per_case_metrics.csv")},
        {"label": "External UTSW cases evaluated", "n": count_rows(Path(os.environ["EVAL_OUT_DIR"]) / "full_evaluation_per_case_metrics.csv")},
    ]
}
json.dump(spec, open(out, "w"), indent=2)
PY

python "${SCRIPT_PATH}" data_flow --spec_json "${POSTHOC_OUT}/data_flow_spec.json" --out_dir "${POSTHOC_OUT}/data_flow"
python "${SCRIPT_PATH}" model_card --ckpt "${STUDENT_CKPT}" --student_base "${STUDENT_BASE}" --holder_alpha "${HOLDER_ALPHA:-1.5}" --out_dir "${POSTHOC_OUT}/model_card"
python "${SCRIPT_PATH}" datasheet --out_dir "${POSTHOC_OUT}/datasheet" --cohorts "BraTS 2023, BraTS Africa, ERASMUS glioma, UTSW-glioma external" --n_cases "fill after final cohort count"
python "${SCRIPT_PATH}" ai_compliance_report --out_dir "${POSTHOC_OUT}/ai_compliance_report" --train_csv "${TRAIN_CSV}" --val_csv "${EVAL_VAL_CSV}" --ckpt "${STUDENT_CKPT}" --ckpt_role student --per_case_csv "${POSTHOC_OUT}/external_stratify_utsw/joined_per_case.csv" --metric_col pp_all_dice_brats_mean --sensitive_attribute "Sex at birth"

python "${SCRIPT_PATH}" holder_sweep --train_csv "${TRAIN_CSV}" --val_csv "${TRAIN_VAL_CSV}" --teacher_ckpt "${TEACHER_CKPT}" --modalities t1n,t1c,t2w,t2f --student_base "${STUDENT_BASE}" --out_dir "${POSTHOC_OUT}/runbooks/holder_sweep"
python "${SCRIPT_PATH}" kd_compare --train_csv "${TRAIN_CSV}" --val_csv "${TRAIN_VAL_CSV}" --teacher_ckpt "${TEACHER_CKPT}" --modalities t1n,t1c,t2w,t2f --student_base "${STUDENT_BASE}" --out_dir "${POSTHOC_OUT}/runbooks/kd_compare"
python "${SCRIPT_PATH}" single_modality_scenario --ckpt "${STUDENT_CKPT}" --val_csv "${EVAL_VAL_CSV}" --modalities t1n,t1c,t2w,t2f --student_base "${STUDENT_BASE}" --tta --keep "t2f,t1n+t2f,t1c+t2f,t1n+t1c+t2f" --out_dir "${POSTHOC_OUT}/runbooks/single_modality_scenario"
python "${SCRIPT_PATH}" kfold_cv --train_csv "${TRAIN_CSV}" --modalities t1n,t1c,t2w,t2f --student_base "${STUDENT_BASE}" --out_dir "${POSTHOC_OUT}/runbooks/kfold_cv"
python "${SCRIPT_PATH}" multi_seed --train_csv "${TRAIN_CSV}" --val_csv "${TRAIN_VAL_CSV}" --modalities t1n,t1c,t2w,t2f --student_base "${STUDENT_BASE}" --out_dir "${POSTHOC_OUT}/runbooks/multi_seed"

echo "[posthoc] complete: ${POSTHOC_OUT}"
