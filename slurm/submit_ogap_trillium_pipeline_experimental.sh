#!/bin/bash
# Submit the standard OGAP Trillium workflow with Slurm dependencies:
# teacher -> student -> INT8 export/comparison -> UTSW evaluation
#
# Default usage:
#   bash submit_ogap_trillium_pipeline.sh
#
# Reuse an existing teacher checkpoint:
#   TEACHER_CKPT=/home/.../Results/teacher/best_teacher.pth RUN_TEACHER=0 bash submit_ogap_trillium_pipeline.sh
#
# ── Trillium 24h walltime + afterok chain caveat ────────────────────────────
# Each Trillium job is capped at 24h.  The teacher (and student) train for 1000
# epochs and will NOT finish in a single 24h window, so an afterok chain submitted
# all at once can break: if the teacher hits walltime it exits non-zero (TIMEOUT)
# and the student's afterok dependency is cancelled.  Recommended pattern on
# Trillium: run the teacher to completion first (resubmit until best_teacher.pth
# is final), then launch the rest with RUN_TEACHER=0 TEACHER_CKPT=...  See
# README_TRILLIUM.md "24-hour walltime".
#
# Layout:
#   PROJECT_ROOT = OGAP_v2          (scripts, results, logs)
#   DATA_ROOT    = OGAP_project     (training data + UTSW validation data)
#
# CSV variables:
#   TRAIN_CSV      = internal train split    (default: ${PROJECT_ROOT}/Scripts/train.csv)
#   TRAIN_VAL_CSV  = internal val split      (default: ${PROJECT_ROOT}/Scripts/val.csv)
#   EVAL_VAL_CSV   = internal held-out val split (default: ${PROJECT_ROOT}/Scripts/val.csv).
#                    External validation (Erasmus/BraTS-Africa) is run separately
#                    via external_*.csv per the pre-registered analysis plan.
# NEVER set a bare VAL_CSV in your shell before calling this script - sbatch
# --export=ALL would leak it into the teacher/student jobs.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/project/def-rdiaz/ilyaso/OGAP}"
DATA_ROOT="${DATA_ROOT:-/project/def-rdiaz/ilyaso/Datasets}"
SCRIPTS_DIR="${SCRIPTS_DIR:-${PROJECT_ROOT}/Scripts}"
ENV_PATH="${ENV_PATH:-${HOME}/ogap/envs/ogap_env_v2}"
EXPECTED_TORCH_PUBLIC_VERSION="${EXPECTED_TORCH_PUBLIC_VERSION:-2.6.0}"
TEACHER_ARCH="${TEACHER_ARCH:-unet3d}"
TEACHER_BASE="${TEACHER_BASE:-64}"
# mednext = depthwise-separable MedNeXt-S (~1M @ base=32, default for backward compat).
# dense   = full 3x3x3 conv blocks (~6M @ base=32, ~15M @ base=48, ~26M @ base=64) for heavy KD teacher.
# Only the student is INT8-exported, so the teacher does not need to be quantization-friendly.
TEACHER_BLOCK_STYLE="${TEACHER_BLOCK_STYLE:-mednext}"
# FIX (Gap A + Gap B + Finding 4, 2026 audit): teacher robustness knobs that
# only affect KD training (zero inference cost - they're identity at eval and
# only the student is exported to INT8).
TEACHER_FEATURE_DR="${TEACHER_FEATURE_DR:-none}"
TEACHER_FEATURE_DR_P="${TEACHER_FEATURE_DR_P:-0.5}"
TEACHER_FEATURE_DR_ALPHA="${TEACHER_FEATURE_DR_ALPHA:-0.1}"
TEACHER_MODALITY_DROPOUT_P="${TEACHER_MODALITY_DROPOUT_P:-0.0}"
TEACHER_MODALITY_DROPOUT_MAX_DROP="${TEACHER_MODALITY_DROPOUT_MAX_DROP:-2}"
FORCE_RESUME_PARTIAL="${FORCE_RESUME_PARTIAL:-0}"
RUN_TEACHER="${RUN_TEACHER:-1}"
RUN_STUDENT="${RUN_STUDENT:-1}"
RUN_EXPORT="${RUN_EXPORT:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_NAS="${RUN_NAS:-0}"
RUN_OOD="${RUN_OOD:-0}"
# DDP teacher: USE_DDP_TEACHER=1 trains the teacher on a full 4xH100 node via torchrun.
USE_DDP_TEACHER="${USE_DDP_TEACHER:-0}"
TEACHER_NPROC="${TEACHER_NPROC:-4}"

TEACHER_SBATCH="${TEACHER_SBATCH:-${SCRIPTS_DIR}/slurm/submit_ogap_teacher_trillium_experimental.sbatch}"
STUDENT_SBATCH="${STUDENT_SBATCH:-${SCRIPTS_DIR}/slurm/submit_ogap_student_trillium_experimental.sbatch}"
EXPORT_SBATCH="${EXPORT_SBATCH:-${SCRIPTS_DIR}/slurm/submit_ogap_export_int8_trillium_experimental.sbatch}"
EVAL_SBATCH="${EVAL_SBATCH:-${SCRIPTS_DIR}/slurm/submit_utsw_eval_tta_trillium_experimental.sbatch}"
NAS_SBATCH="${NAS_SBATCH:-${SCRIPTS_DIR}/slurm/submit_ogap_nas_trillium.sbatch}"
OOD_SBATCH="${OOD_SBATCH:-${SCRIPTS_DIR}/slurm/submit_ogap_ood_trillium.sbatch}"
NAS_OUT="${NAS_OUT:-${SCRATCH:-${HOME}/scratch}/ogap/Results/nas/pareto_front.json}"
OOD_OUT_DIR="${OOD_OUT_DIR:-${SCRATCH:-${HOME}/scratch}/ogap/Results/ood}"
OOD_CSV="${OOD_CSV:-${PROJECT_ROOT}/Scripts/external_brats_africa.csv}"

TEACHER_OUT_DIR="${TEACHER_OUT_DIR:-${SCRATCH:-${HOME}/scratch}/ogap/Results/teacher}"
TEACHER_CKPT="${TEACHER_CKPT:-${TEACHER_OUT_DIR}/best_teacher.pth}"
STUDENT_OUT_DIR="${STUDENT_OUT_DIR:-${SCRATCH:-${HOME}/scratch}/ogap/Results/student_external_push}"
STUDENT_CKPT="${STUDENT_CKPT:-${STUDENT_OUT_DIR}/best_student.pth}"
EXPORT_OUT_DIR="${EXPORT_OUT_DIR:-${STUDENT_OUT_DIR}_export_int8}"
EXPORT_ONNX_PATH="${EXPORT_ONNX_PATH:-${EXPORT_OUT_DIR}/best_student.onnx}"
QUANT_COMPARE_OUT_DIR="${QUANT_COMPARE_OUT_DIR:-${EXPORT_OUT_DIR}/quantisation_comparison}"
# NOTE: this used to be VAL_CSV, which collided with the student/teacher sbatch's
# own VAL_CSV via --export=ALL and silently overrode Scripts/val.csv. Renamed to
# EVAL_VAL_CSV so only the eval job ever sees the external UTSW CSV.
EVAL_VAL_CSV="${EVAL_VAL_CSV:-${VAL_CSV:-${PROJECT_ROOT}/Scripts/val.csv}}"
# Do not let a stray VAL_CSV (e.g. from shell history) leak into child sbatch jobs.
unset VAL_CSV
EVAL_OUT_DIR="${EVAL_OUT_DIR:-${SCRATCH:-${HOME}/scratch}/ogap/Results/utsw_full_evaluation_tta}"
TRAIN_CSV="${TRAIN_CSV:-${PROJECT_ROOT}/Scripts/train.csv}"
TRAIN_VAL_CSV="${TRAIN_VAL_CSV:-${PROJECT_ROOT}/Scripts/val.csv}"
NUM_WORKERS="${NUM_WORKERS:-16}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-4}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-8}"
METRIC_WORKERS="${METRIC_WORKERS:-4}"
EVAL_CONFIG_BATCH_SIZE="${EVAL_CONFIG_BATCH_SIZE:-8}"
CALIBRATION_CSV="${CALIBRATION_CSV:-${TRAIN_VAL_CSV:-${PROJECT_ROOT}/Scripts/val.csv}}"
CALIBRATION_SAMPLES="${CALIBRATION_SAMPLES:-200}"
CALIBRATION_MIN_PRESENT="${CALIBRATION_MIN_PRESENT:-1}"
CALIBRATION_MAX_PRESENT="${CALIBRATION_MAX_PRESENT:-4}"
STATIC_CALIBRATION_METHOD="${STATIC_CALIBRATION_METHOD:-percentile}"
STATIC_PER_CHANNEL="${STATIC_PER_CHANNEL:-1}"
STATIC_REDUCE_RANGE="${STATIC_REDUCE_RANGE:-auto}"
SKIP_QUANT_PREPROCESS="${SKIP_QUANT_PREPROCESS:-0}"
RUN_QUANT_COMPARE="${RUN_QUANT_COMPARE:-1}"
COMPARE_NUM_WORKERS="${COMPARE_NUM_WORKERS:-${EVAL_NUM_WORKERS}}"
COMPARE_METRIC_WORKERS="${COMPARE_METRIC_WORKERS:-${METRIC_WORKERS}}"
COMPARE_EVAL_CONFIG_BATCH_SIZE="${COMPARE_EVAL_CONFIG_BATCH_SIZE:-${EVAL_CONFIG_BATCH_SIZE}}"
COMPARE_NUM_THREADS="${COMPARE_NUM_THREADS:-}"
TRACK_COMPARE_ENERGY="${TRACK_COMPARE_ENERGY:-1}"
COMPARE_ENERGY_POLL_INTERVAL="${COMPARE_ENERGY_POLL_INTERVAL:-5.0}"
TEACHER_BATCH_SIZE="${TEACHER_BATCH_SIZE:-2}"
STUDENT_BATCH_SIZE="${STUDENT_BATCH_SIZE:-4}"
STUDENT_BASE="${STUDENT_BASE:-16}"
PATCH_CACHE_START_EPOCH="${PATCH_CACHE_START_EPOCH:-2}"
OGAP_N4_THREADS="${OGAP_N4_THREADS:-1}"
OGAP_N4_ITERATIONS="${OGAP_N4_ITERATIONS:-50,50,30,20}"
OGAP_DISABLE_N4="${OGAP_DISABLE_N4:-1}"
OGAP_PREFETCH_FACTOR="${OGAP_PREFETCH_FACTOR:-1}"
OGAP_MAX_TRAIN_WORKERS="${OGAP_MAX_TRAIN_WORKERS:-16}"
OGAP_MAX_EVAL_WORKERS="${OGAP_MAX_EVAL_WORKERS:-8}"
OGAP_MAX_PREFETCH_FACTOR="${OGAP_MAX_PREFETCH_FACTOR:-1}"
OGAP_MAX_TRAIN_INFLIGHT_CASES="${OGAP_MAX_TRAIN_INFLIGHT_CASES:-32}"
OGAP_MAX_EVAL_INFLIGHT_CASES="${OGAP_MAX_EVAL_INFLIGHT_CASES:-8}"
OGAP_USE_NPY_CACHE="${OGAP_USE_NPY_CACHE:-0}"
OGAP_USE_PATCH_CACHE="${OGAP_USE_PATCH_CACHE:-0}"
OGAP_DROP_NPY_FILE_CACHE="${OGAP_DROP_NPY_FILE_CACHE:-1}"
OGAP_STAGE_DATA_TO_TMP="${OGAP_STAGE_DATA_TO_TMP:-1}"
OGAP_STAGE_DECOMPRESS_GZ="${OGAP_STAGE_DECOMPRESS_GZ:-1}"
OGAP_STAGE_WORKERS="${OGAP_STAGE_WORKERS:-32}"
OGAP_STAGE_COLUMNS="${OGAP_STAGE_COLUMNS:-t1n,t1c,t2w,t2f,label}"
TEACHER_AUTO_BATCH_UPPER_BOUND="${TEACHER_AUTO_BATCH_UPPER_BOUND:-16}"
STUDENT_AUTO_BATCH_UPPER_BOUND="${STUDENT_AUTO_BATCH_UPPER_BOUND:-16}"
export OGAP_N4_THREADS
export OGAP_N4_ITERATIONS
export OGAP_DISABLE_N4
export OGAP_PREFETCH_FACTOR
export OGAP_MAX_TRAIN_WORKERS
export OGAP_MAX_EVAL_WORKERS
export OGAP_MAX_PREFETCH_FACTOR
export OGAP_MAX_TRAIN_INFLIGHT_CASES
export OGAP_MAX_EVAL_INFLIGHT_CASES
export OGAP_USE_NPY_CACHE
export OGAP_USE_PATCH_CACHE
export OGAP_DROP_NPY_FILE_CACHE
export OGAP_STAGE_DATA_TO_TMP
export OGAP_STAGE_DECOMPRESS_GZ
export OGAP_STAGE_WORKERS
export OGAP_STAGE_COLUMNS
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
CLINICAL_METADATA_TSV="${CLINICAL_METADATA_TSV:-}"
FAILURE_MODE_METRIC="${FAILURE_MODE_METRIC:-pp_all_dice_brats_mean}"
FAILURE_MODE_QUANTILE="${FAILURE_MODE_QUANTILE:-0.25}"
FAILURE_MODE_MIN_GROUP_N="${FAILURE_MODE_MIN_GROUP_N:-10}"
LESION_WISE="${LESION_WISE:-1}"
LMIC_FIELD_STRENGTH_PRIOR="${LMIC_FIELD_STRENGTH_PRIOR:-1}"
CONTRAST_MOD_INDICES="${CONTRAST_MOD_INDICES:-1}"
CONTRAST_DROPOUT_EXTRA_PROB="${CONTRAST_DROPOUT_EXTRA_PROB:-0.50}"
P_ANISOTROPIC_2D="${P_ANISOTROPIC_2D:-0.85}"
P_PARTIAL_CONTRAST="${P_PARTIAL_CONTRAST:-0.30}"
P_LABEL_NOISE="${P_LABEL_NOISE:-0.20}"
# Architecture-pass physics augmentations (defaults match source = OFF/static).
LABEL_NOISE_MODE="${LABEL_NOISE_MODE:-morph}"
LABEL_NOISE_BAND_RADIUS="${LABEL_NOISE_BAND_RADIUS:-2}"
LABEL_NOISE_BAND_FLIP_PROB="${LABEL_NOISE_BAND_FLIP_PROB:-0.30}"
P_FIELD_CONTRAST_WARP="${P_FIELD_CONTRAST_WARP:-0.0}"
FIELD_CONTRAST_SOURCE_B0="${FIELD_CONTRAST_SOURCE_B0:-3.0}"
P_FOURIER_AMPLITUDE_MIX="${P_FOURIER_AMPLITUDE_MIX:-0.0}"
AFA_PROB="${AFA_PROB:-0.0}"
P_RECEIVE_COIL_INHOMOGENEITY="${P_RECEIVE_COIL_INHOMOGENEITY:-0.0}"
P_OFF_RESONANCE_VOID="${P_OFF_RESONANCE_VOID:-0.0}"
NOISE_CALIBRATION="${NOISE_CALIBRATION:-static}"
CARVEMIX_PROB="${CARVEMIX_PROB:-0.0}"
CARVEMIX_DILATION="${CARVEMIX_DILATION:-5}"
P_ULF_GAN_SYNTHESIS="${P_ULF_GAN_SYNTHESIS:-0.0}"
ULF_GAN_WEIGHTS="${ULF_GAN_WEIGHTS:-}"
ULF_GAN_TARGET_B0="${ULF_GAN_TARGET_B0:-0.064}"
# Distribution-shift / feature-DR (student only)
LAMBDA_VREX="${LAMBDA_VREX:-0.0}"
FEATURE_DR="${FEATURE_DR:-none}"
FEATURE_DR_P="${FEATURE_DR_P:-0.5}"
FEATURE_DR_ALPHA="${FEATURE_DR_ALPHA:-0.1}"
KEEP_MIN_MODALITIES="${KEEP_MIN_MODALITIES:-1}"
KEEP_MAX_MODALITIES="${KEEP_MAX_MODALITIES:-4}"
AUTO_RESOURCES="${AUTO_RESOURCES:-1}"
# AUTO_RESOURCES_SAFETY is the multiplier applied to the probed maximum batch
# size in OGAP_source_code_experimental_v9.py::auto_tune_resources.  The
# teacher/student split is intentional:
#
#   * Teacher (0.85): single deterministic forward+backward at fixed
#     patch_size (192,192,144).  The probe is reliable; we keep more
#     headroom only for the intermittent allocator spikes from
#     channels_last_3d conversion + AMP cast.
#   * Student (0.70): KD path runs the (compiled) teacher AND the student
#     in the same step, so peak VRAM has a wider distribution; the lower
#     safety factor absorbs that variance.
#
# RUN B in the user playbook overrides student=0.75 (UNet3D dense teacher
# leaves more VRAM headroom than SegMamba); RUN A uses 0.75 / 0.75 because
# Mamba SSM working memory is harder to predict.  Override per-run as needed.
AUTO_RESOURCES_SAFETY_TEACHER="${AUTO_RESOURCES_SAFETY_TEACHER:-0.85}"
AUTO_RESOURCES_SAFETY_STUDENT="${AUTO_RESOURCES_SAFETY_STUDENT:-0.70}"

check_path() {
  # Usage: check_path <LABEL> <path> [hint]
  local label="$1" path="$2" hint="${3:-}"
  if [[ ! -e "${path}" ]]; then
    echo "[ERROR] ${label} not found: ${path}" >&2
    if [[ -n "${hint}" ]]; then
      echo "        hint: ${hint}" >&2
    fi
    # Targeted hint for the recurring UTSW (UT Southwestern) vs USTW typo.
    if [[ "${path}" == *"/UTSW-glioma/"* || "${path}" == *"/UTSW-Glioma/"* ]]; then
      local parent="${path%/UTSW-*}"
      if [[ -d "${parent}/USTW-glioma" ]]; then
        echo "        hint: found '${parent}/USTW-glioma' on disk - that is a typo." >&2
        echo "              The correct spelling is UTSW (UT Southwestern). Rename it with:" >&2
        echo "              mv '${parent}/USTW-glioma' '${parent}/UTSW-glioma'" >&2
      fi
    fi
    exit 1
  fi
}

slurm_export_spec() {
  local __out_var="$1"
  shift
  local export_csv="ALL"
  local pair name value
  for pair in "$@"; do
    name="${pair%%=*}"
    value="${pair#*=}"
    if [[ -z "${value}" ]]; then
      continue
    fi
    if [[ ! "${name}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
      echo "[ERROR] Unsafe Slurm export variable name: ${name}" >&2
      exit 1
    fi
    # Export into THIS environment and let --export=ALL (export_csv stays "ALL")
    # propagate the whole environment to the job. Inlining ~40 NAME=VALUE pairs in
    # the --export argument overflows the SciNet sbatch's length cap on Trillium
    # ("Batch job submission failed: Pathname ... too long"). Exporting also makes
    # comma-containing values (e.g. OGAP_N4_ITERATIONS=50,50,30,20) safe. [Trillium fix]
    export "${name}=${value}"
  done
  printf -v "${__out_var}" '%s' "${export_csv}"
}

if [[ ! -d "${ENV_PATH}" ]]; then
  echo "[ERROR] ENV_PATH does not exist: ${ENV_PATH}" >&2
  echo "        hint: source/create the intended venv, or export ENV_PATH=/path/to/ogap_env_v2." >&2
  exit 1
fi

for required_path in "${TEACHER_SBATCH}" "${STUDENT_SBATCH}" "${EXPORT_SBATCH}" "${EVAL_SBATCH}"; do
  check_path "Workflow file" "${required_path}"
done

# Validate all downstream paths up front so we never queue a job that will abort in preflight.
check_path "PROJECT_ROOT" "${PROJECT_ROOT}"
check_path "DATA_ROOT"    "${DATA_ROOT}"
check_path "TRAIN_CSV"     "${TRAIN_CSV}"
check_path "TRAIN_VAL_CSV" "${TRAIN_VAL_CSV}"
# A reused checkpoint is only required when a stage that CONSUMES it will run:
#   teacher ckpt ← consumed by the student stage (RUN_STUDENT=1)
#   student ckpt ← consumed by export/eval (RUN_EXPORT=1 or RUN_EVAL=1)
# This lets a teacher-only Phase-1 run (RUN_STUDENT=0 RUN_EXPORT=0 RUN_EVAL=0) proceed.
if [[ "${RUN_TEACHER}" != "1" && "${RUN_STUDENT}" == "1" ]]; then
  check_path "TEACHER_CKPT (RUN_TEACHER=0)" "${TEACHER_CKPT}"
fi
if [[ "${RUN_STUDENT}" != "1" && ( "${RUN_EXPORT}" == "1" || "${RUN_EVAL}" == "1" ) ]]; then
  check_path "STUDENT_CKPT (RUN_STUDENT=0)" "${STUDENT_CKPT}"
fi
if [[ "${RUN_EXPORT}" == "1" ]]; then
  check_path "CALIBRATION_CSV" "${CALIBRATION_CSV}"
  if [[ "${RUN_QUANT_COMPARE}" == "1" ]]; then
    check_path "EVAL_VAL_CSV (quantisation comparison)" "${EVAL_VAL_CSV}"
  fi
fi
if [[ "${RUN_EVAL}" == "1" ]]; then
  check_path "EVAL_VAL_CSV" "${EVAL_VAL_CSV}"
fi
if [[ -n "${CLINICAL_METADATA_TSV}" ]]; then
  check_path "CLINICAL_METADATA_TSV" "${CLINICAL_METADATA_TSV}"
fi

mkdir -p "${SCRATCH:-${HOME}/scratch}/ogap/logs"

teacher_job=""
student_job=""
export_job=""
eval_job=""

if [[ "${RUN_TEACHER}" == "1" ]]; then
  if [[ "${USE_DDP_TEACHER}" == "1" ]]; then
    _TEACHER_NPROC="${TEACHER_NPROC}"
    TEACHER_SBATCH_OPTS=(--gpus-per-node=4 --cpus-per-task=96)
    echo "[pipeline] DDP teacher on a full 4xH100 node (torchrun nproc=${TEACHER_NPROC})"
  else
    _TEACHER_NPROC=1
    TEACHER_SBATCH_OPTS=()
  fi
  slurm_export_spec teacher_export \
    "PROJECT_ROOT=${PROJECT_ROOT}" \
    "ENV_PATH=${ENV_PATH}" \
    "EXPECTED_TORCH_PUBLIC_VERSION=${EXPECTED_TORCH_PUBLIC_VERSION}" \
    "TEACHER_ARCH=${TEACHER_ARCH}" \
    "TEACHER_BASE=${TEACHER_BASE}" \
    "TEACHER_BLOCK_STYLE=${TEACHER_BLOCK_STYLE}" \
    "TRAIN_CSV=${TRAIN_CSV}" \
    "VAL_CSV=${TRAIN_VAL_CSV}" \
    "OUT_DIR=${TEACHER_OUT_DIR}" \
    "NUM_WORKERS=${NUM_WORKERS}" \
    "VAL_NUM_WORKERS=${VAL_NUM_WORKERS}" \
    "TEACHER_BATCH_SIZE=${TEACHER_BATCH_SIZE}" \
    "PATCH_CACHE_START_EPOCH=${PATCH_CACHE_START_EPOCH}" \
    "OGAP_AUTO_BATCH_UPPER_BOUND=${TEACHER_AUTO_BATCH_UPPER_BOUND}" \
    "LMIC_FIELD_STRENGTH_PRIOR=${LMIC_FIELD_STRENGTH_PRIOR}" \
    "CONTRAST_MOD_INDICES=${CONTRAST_MOD_INDICES}" \
    "P_ANISOTROPIC_2D=${P_ANISOTROPIC_2D}" \
    "P_PARTIAL_CONTRAST=${P_PARTIAL_CONTRAST}" \
    "P_LABEL_NOISE=${P_LABEL_NOISE}" \
    "LABEL_NOISE_MODE=${LABEL_NOISE_MODE}" \
    "LABEL_NOISE_BAND_RADIUS=${LABEL_NOISE_BAND_RADIUS}" \
    "LABEL_NOISE_BAND_FLIP_PROB=${LABEL_NOISE_BAND_FLIP_PROB}" \
    "P_FIELD_CONTRAST_WARP=${P_FIELD_CONTRAST_WARP}" \
    "FIELD_CONTRAST_SOURCE_B0=${FIELD_CONTRAST_SOURCE_B0}" \
    "P_FOURIER_AMPLITUDE_MIX=${P_FOURIER_AMPLITUDE_MIX}" \
    "AFA_PROB=${AFA_PROB}" \
    "P_RECEIVE_COIL_INHOMOGENEITY=${P_RECEIVE_COIL_INHOMOGENEITY}" \
    "P_OFF_RESONANCE_VOID=${P_OFF_RESONANCE_VOID}" \
    "NOISE_CALIBRATION=${NOISE_CALIBRATION}" \
    "CARVEMIX_PROB=${CARVEMIX_PROB}" \
    "CARVEMIX_DILATION=${CARVEMIX_DILATION}" \
    "P_ULF_GAN_SYNTHESIS=${P_ULF_GAN_SYNTHESIS}" \
    "ULF_GAN_WEIGHTS=${ULF_GAN_WEIGHTS}" \
    "ULF_GAN_TARGET_B0=${ULF_GAN_TARGET_B0}" \
    "OGAP_PREFETCH_FACTOR=${OGAP_PREFETCH_FACTOR}" \
    "AUTO_RESOURCES=${AUTO_RESOURCES}" \
    "AUTO_RESOURCES_SAFETY=${AUTO_RESOURCES_SAFETY_TEACHER}" \
    "TEACHER_FEATURE_DR=${TEACHER_FEATURE_DR}" \
    "TEACHER_FEATURE_DR_P=${TEACHER_FEATURE_DR_P}" \
    "TEACHER_FEATURE_DR_ALPHA=${TEACHER_FEATURE_DR_ALPHA}" \
    "TEACHER_MODALITY_DROPOUT_P=${TEACHER_MODALITY_DROPOUT_P}" \
    "TEACHER_MODALITY_DROPOUT_MAX_DROP=${TEACHER_MODALITY_DROPOUT_MAX_DROP}" \
    "FORCE_RESUME_PARTIAL=${FORCE_RESUME_PARTIAL}" \
    "OGAP_NPROC=${_TEACHER_NPROC}"
  TEACHER_DEP_OPT=()
  if [[ -n "${TEACHER_DEPENDENCY:-}" ]]; then
    TEACHER_DEP_OPT=(--dependency="${TEACHER_DEPENDENCY}")
  fi
  teacher_job="$(sbatch --parsable "${TEACHER_DEP_OPT[@]}" "${TEACHER_SBATCH_OPTS[@]}" --export="${teacher_export}" "${TEACHER_SBATCH}")"
  echo "teacher_job=${teacher_job}"
else
  if [[ ! -e "${TEACHER_CKPT}" ]]; then
    echo "[ERROR] RUN_TEACHER=0 but teacher checkpoint not found: ${TEACHER_CKPT}" >&2
    exit 1
  fi
fi

if [[ "${RUN_STUDENT}" == "1" ]]; then
  slurm_export_spec student_export \
    "PROJECT_ROOT=${PROJECT_ROOT}" \
    "ENV_PATH=${ENV_PATH}" \
    "EXPECTED_TORCH_PUBLIC_VERSION=${EXPECTED_TORCH_PUBLIC_VERSION}" \
    "TEACHER_ARCH=${TEACHER_ARCH}" \
    "TEACHER_BASE=${TEACHER_BASE}" \
    "TEACHER_BLOCK_STYLE=${TEACHER_BLOCK_STYLE}" \
    "TRAIN_CSV=${TRAIN_CSV}" \
    "VAL_CSV=${TRAIN_VAL_CSV}" \
    "TEACHER_CKPT=${TEACHER_CKPT}" \
    "OUT_DIR=${STUDENT_OUT_DIR}" \
    "NUM_WORKERS=${NUM_WORKERS}" \
    "VAL_NUM_WORKERS=${VAL_NUM_WORKERS}" \
    "STUDENT_BATCH_SIZE=${STUDENT_BATCH_SIZE}" \
    "STUDENT_BASE=${STUDENT_BASE}" \
    "PATCH_CACHE_START_EPOCH=${PATCH_CACHE_START_EPOCH}" \
    "OGAP_AUTO_BATCH_UPPER_BOUND=${STUDENT_AUTO_BATCH_UPPER_BOUND}" \
    "GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS}" \
    "LMIC_FIELD_STRENGTH_PRIOR=${LMIC_FIELD_STRENGTH_PRIOR}" \
    "CONTRAST_MOD_INDICES=${CONTRAST_MOD_INDICES}" \
    "CONTRAST_DROPOUT_EXTRA_PROB=${CONTRAST_DROPOUT_EXTRA_PROB}" \
    "P_ANISOTROPIC_2D=${P_ANISOTROPIC_2D}" \
    "P_PARTIAL_CONTRAST=${P_PARTIAL_CONTRAST}" \
    "P_LABEL_NOISE=${P_LABEL_NOISE}" \
    "LABEL_NOISE_MODE=${LABEL_NOISE_MODE}" \
    "LABEL_NOISE_BAND_RADIUS=${LABEL_NOISE_BAND_RADIUS}" \
    "LABEL_NOISE_BAND_FLIP_PROB=${LABEL_NOISE_BAND_FLIP_PROB}" \
    "P_FIELD_CONTRAST_WARP=${P_FIELD_CONTRAST_WARP}" \
    "FIELD_CONTRAST_SOURCE_B0=${FIELD_CONTRAST_SOURCE_B0}" \
    "P_FOURIER_AMPLITUDE_MIX=${P_FOURIER_AMPLITUDE_MIX}" \
    "AFA_PROB=${AFA_PROB}" \
    "P_RECEIVE_COIL_INHOMOGENEITY=${P_RECEIVE_COIL_INHOMOGENEITY}" \
    "P_OFF_RESONANCE_VOID=${P_OFF_RESONANCE_VOID}" \
    "NOISE_CALIBRATION=${NOISE_CALIBRATION}" \
    "CARVEMIX_PROB=${CARVEMIX_PROB}" \
    "CARVEMIX_DILATION=${CARVEMIX_DILATION}" \
    "P_ULF_GAN_SYNTHESIS=${P_ULF_GAN_SYNTHESIS}" \
    "ULF_GAN_WEIGHTS=${ULF_GAN_WEIGHTS}" \
    "ULF_GAN_TARGET_B0=${ULF_GAN_TARGET_B0}" \
    "LAMBDA_VREX=${LAMBDA_VREX}" \
    "FEATURE_DR=${FEATURE_DR}" \
    "FEATURE_DR_P=${FEATURE_DR_P}" \
    "FEATURE_DR_ALPHA=${FEATURE_DR_ALPHA}" \
    "KEEP_MIN_MODALITIES=${KEEP_MIN_MODALITIES}" \
    "KEEP_MAX_MODALITIES=${KEEP_MAX_MODALITIES}" \
    "OGAP_PREFETCH_FACTOR=${OGAP_PREFETCH_FACTOR}" \
    "AUTO_RESOURCES=${AUTO_RESOURCES}" \
    "AUTO_RESOURCES_SAFETY=${AUTO_RESOURCES_SAFETY_STUDENT}"
  if [[ -n "${teacher_job}" ]]; then
    student_job="$(sbatch --parsable --dependency=afterok:${teacher_job} --export="${student_export}" "${STUDENT_SBATCH}")"
  else
    student_job="$(sbatch --parsable --export="${student_export}" "${STUDENT_SBATCH}")"
  fi
  echo "student_job=${student_job}"
else
  # Only require an existing student checkpoint when a DOWNSTREAM stage will
  # consume it. For a teacher-only Phase-1 run (RUN_EXPORT=0 RUN_EVAL=0) there is
  # nothing to consume it, so do not abort. [fix: make README §3 Phase-1 work]
  if [[ "${RUN_EXPORT}" == "1" || "${RUN_EVAL}" == "1" || "${RUN_OOD}" == "1" ]] && [[ ! -e "${STUDENT_CKPT}" ]]; then
    echo "[ERROR] RUN_STUDENT=0 but student checkpoint not found (needed by export/eval/ood): ${STUDENT_CKPT}" >&2
    exit 1
  fi
fi

if [[ "${RUN_EXPORT}" == "1" ]]; then
  slurm_export_spec export_export \
    "PROJECT_ROOT=${PROJECT_ROOT}" \
    "DATA_ROOT=${DATA_ROOT}" \
    "ENV_PATH=${ENV_PATH}" \
    "EXPECTED_TORCH_PUBLIC_VERSION=${EXPECTED_TORCH_PUBLIC_VERSION}" \
    "CKPT=${STUDENT_CKPT}" \
    "OUT_DIR=${EXPORT_OUT_DIR}" \
    "EXPORT_ONNX_PATH=${EXPORT_ONNX_PATH}" \
    "CALIBRATION_CSV=${CALIBRATION_CSV}" \
    "COMPARE_VAL_CSV=${EVAL_VAL_CSV}" \
    "COMPARE_OUT_DIR=${QUANT_COMPARE_OUT_DIR}" \
    "RUN_COMPARE=${RUN_QUANT_COMPARE}" \
    "STUDENT_BASE=${STUDENT_BASE}" \
    "CALIBRATION_SAMPLES=${CALIBRATION_SAMPLES}" \
    "CALIBRATION_MIN_PRESENT=${CALIBRATION_MIN_PRESENT}" \
    "CALIBRATION_MAX_PRESENT=${CALIBRATION_MAX_PRESENT}" \
    "STATIC_CALIBRATION_METHOD=${STATIC_CALIBRATION_METHOD}" \
    "STATIC_PER_CHANNEL=${STATIC_PER_CHANNEL}" \
    "STATIC_REDUCE_RANGE=${STATIC_REDUCE_RANGE}" \
    "SKIP_QUANT_PREPROCESS=${SKIP_QUANT_PREPROCESS}" \
    "COMPARE_NUM_WORKERS=${COMPARE_NUM_WORKERS}" \
    "COMPARE_METRIC_WORKERS=${COMPARE_METRIC_WORKERS}" \
    "COMPARE_EVAL_CONFIG_BATCH_SIZE=${COMPARE_EVAL_CONFIG_BATCH_SIZE}" \
    "COMPARE_NUM_THREADS=${COMPARE_NUM_THREADS}" \
    "TRACK_COMPARE_ENERGY=${TRACK_COMPARE_ENERGY}" \
    "COMPARE_ENERGY_POLL_INTERVAL=${COMPARE_ENERGY_POLL_INTERVAL}"
  if [[ -n "${student_job}" ]]; then
    export_job="$(sbatch --parsable --dependency=afterok:${student_job} --export="${export_export}" "${EXPORT_SBATCH}")"
  else
    export_job="$(sbatch --parsable --export="${export_export}" "${EXPORT_SBATCH}")"
  fi
  echo "export_job=${export_job}"
fi

if [[ "${RUN_EVAL}" == "1" ]]; then
  slurm_export_spec eval_export \
    "PROJECT_ROOT=${PROJECT_ROOT}" \
    "ENV_PATH=${ENV_PATH}" \
    "EXPECTED_TORCH_PUBLIC_VERSION=${EXPECTED_TORCH_PUBLIC_VERSION}" \
    "CKPT=${STUDENT_CKPT}" \
    "VAL_CSV=${EVAL_VAL_CSV}" \
    "OUT_DIR=${EVAL_OUT_DIR}" \
    "NUM_WORKERS=${EVAL_NUM_WORKERS}" \
    "METRIC_WORKERS=${METRIC_WORKERS}" \
    "EVAL_CONFIG_BATCH_SIZE=${EVAL_CONFIG_BATCH_SIZE}" \
    "STUDENT_BASE=${STUDENT_BASE}" \
    "CLINICAL_METADATA_TSV=${CLINICAL_METADATA_TSV}" \
    "FAILURE_MODE_METRIC=${FAILURE_MODE_METRIC}" \
    "FAILURE_MODE_QUANTILE=${FAILURE_MODE_QUANTILE}" \
    "FAILURE_MODE_MIN_GROUP_N=${FAILURE_MODE_MIN_GROUP_N}" \
    "LESION_WISE=${LESION_WISE}"
  if [[ -n "${export_job}" ]]; then
    eval_job="$(sbatch --parsable --dependency=afterok:${export_job} --export="${eval_export}" "${EVAL_SBATCH}")"
  elif [[ -n "${student_job}" ]]; then
    eval_job="$(sbatch --parsable --dependency=afterok:${student_job} --export="${eval_export}" "${EVAL_SBATCH}")"
  else
    eval_job="$(sbatch --parsable --export="${eval_export}" "${EVAL_SBATCH}")"
  fi
  echo "eval_job=${eval_job}"
fi

# ── NAS (offline hardware-aware architecture search; independent of training) ─
if [[ "${RUN_NAS}" == "1" ]]; then
  slurm_export_spec nas_export \
    "PROJECT_ROOT=${PROJECT_ROOT}" \
    "ENV_PATH=${ENV_PATH}" \
    "OUT=${NAS_OUT}"
  nas_job="$(sbatch --parsable --export="${nas_export}" "${NAS_SBATCH}")"
  echo "nas_job=${nas_job}"
fi

# ── Mahalanobis OOD fitting (consumes the student checkpoint) ────────────────
if [[ "${RUN_OOD}" == "1" ]]; then
  slurm_export_spec ood_export \
    "PROJECT_ROOT=${PROJECT_ROOT}" \
    "ENV_PATH=${ENV_PATH}" \
    "CKPT=${STUDENT_CKPT}" \
    "STUDENT_OUT_DIR=${STUDENT_OUT_DIR}" \
    "ID_CSV=${TRAIN_VAL_CSV}" \
    "OOD_CSV=${OOD_CSV}" \
    "OUT_DIR=${OOD_OUT_DIR}" \
    "STUDENT_BASE=${STUDENT_BASE}"
  if [[ -n "${student_job}" ]]; then
    ood_job="$(sbatch --parsable --dependency=afterok:${student_job} --export="${ood_export}" "${OOD_SBATCH}")"
  else
    ood_job="$(sbatch --parsable --export="${ood_export}" "${OOD_SBATCH}")"
  fi
  echo "ood_job=${ood_job}"
fi

echo "teacher_ckpt=${TEACHER_CKPT}"
echo "student_ckpt=${STUDENT_CKPT}"
echo "export_onnx_path=${EXPORT_ONNX_PATH}"
echo "quant_compare_out_dir=${QUANT_COMPARE_OUT_DIR}"
echo "train_csv=${TRAIN_CSV}"
echo "train_val_csv=${TRAIN_VAL_CSV}"
echo "eval_val_csv=${EVAL_VAL_CSV}"
echo "eval_out_dir=${EVAL_OUT_DIR}"
echo "auto_resources=${AUTO_RESOURCES}"
echo "loader_plan=num_workers=${NUM_WORKERS}, eval_num_workers=${EVAL_NUM_WORKERS}, prefetch=${OGAP_PREFETCH_FACTOR}, max_train_workers=${OGAP_MAX_TRAIN_WORKERS}, max_eval_workers=${OGAP_MAX_EVAL_WORKERS}, max_prefetch=${OGAP_MAX_PREFETCH_FACTOR}, max_train_inflight=${OGAP_MAX_TRAIN_INFLIGHT_CASES}, max_eval_inflight=${OGAP_MAX_EVAL_INFLIGHT_CASES}"
echo "batch_probe_upper=teacher:${TEACHER_AUTO_BATCH_UPPER_BOUND}, student:${STUDENT_AUTO_BATCH_UPPER_BOUND}"
echo "cache_plan=npy:${OGAP_USE_NPY_CACHE}, patch:${OGAP_USE_PATCH_CACHE}, drop_npy_file_cache:${OGAP_DROP_NPY_FILE_CACHE}, patch_start_epoch:${PATCH_CACHE_START_EPOCH}"
echo "stage_plan=enabled:${OGAP_STAGE_DATA_TO_TMP}, decompress_gzip:${OGAP_STAGE_DECOMPRESS_GZ}, workers:${OGAP_STAGE_WORKERS}, columns:${OGAP_STAGE_COLUMNS}"
echo "ogap_disable_n4=${OGAP_DISABLE_N4}"
