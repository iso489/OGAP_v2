#!/bin/bash
# Rorqual ONNX + INT8 export job for the OGAP v2 student.
# Runs after student training and before external evaluation.
#
# Default usage is through submit_ogap_rorqual_pipeline_experimental.sh.

#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=200G
#SBATCH --tmp=1000G
#SBATCH --gpus=h100:1
#SBATCH --job-name=ogap_export_int8
#SBATCH --output=logs/ogap_export_int8_%j.out
#SBATCH --error=logs/ogap_export_int8_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${HOME}/ogap/OGAP_v2}"
DATA_ROOT="${DATA_ROOT:-${HOME}/ogap/OGAP_project}"
ENV_PATH="${ENV_PATH:-${HOME}/ogap/envs/ogap_env_v2}"
EXPECTED_TORCH_PUBLIC_VERSION="${EXPECTED_TORCH_PUBLIC_VERSION:-2.6.0}"
SCRIPT_PATH="${SCRIPT_PATH:-${PROJECT_ROOT}/Scripts/OGAP_source_code_experimental_v9.py}"
CKPT="${CKPT:-${PROJECT_ROOT}/Results/student_external_push/best_student.pth}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/Results/student_external_push_export_int8}"
EXPORT_ONNX_PATH="${EXPORT_ONNX_PATH:-${OUT_DIR}/best_student.onnx}"
CALIBRATION_CSV="${CALIBRATION_CSV:-${PROJECT_ROOT}/Scripts/val.csv}"
COMPARE_VAL_CSV="${COMPARE_VAL_CSV:-${DATA_ROOT}/Validation/UTSW-glioma/utsw_manual_val.csv}"
COMPARE_OUT_DIR="${COMPARE_OUT_DIR:-${OUT_DIR}/quantisation_comparison}"
RUN_COMPARE="${RUN_COMPARE:-1}"
STUDENT_BASE="${STUDENT_BASE:-16}"
CALIBRATION_SAMPLES="${CALIBRATION_SAMPLES:-200}"
CALIBRATION_MIN_PRESENT="${CALIBRATION_MIN_PRESENT:-1}"
CALIBRATION_MAX_PRESENT="${CALIBRATION_MAX_PRESENT:-4}"
STATIC_CALIBRATION_METHOD="${STATIC_CALIBRATION_METHOD:-percentile}"
STATIC_PER_CHANNEL="${STATIC_PER_CHANNEL:-1}"
STATIC_REDUCE_RANGE="${STATIC_REDUCE_RANGE:-auto}"
SKIP_QUANT_PREPROCESS="${SKIP_QUANT_PREPROCESS:-0}"
COMPARE_NUM_WORKERS="${COMPARE_NUM_WORKERS:-${EVAL_NUM_WORKERS:-8}}"
COMPARE_METRIC_WORKERS="${COMPARE_METRIC_WORKERS:-${METRIC_WORKERS:-4}}"
COMPARE_EVAL_CONFIG_BATCH_SIZE="${COMPARE_EVAL_CONFIG_BATCH_SIZE:-${EVAL_CONFIG_BATCH_SIZE:-8}}"
COMPARE_NUM_THREADS="${COMPARE_NUM_THREADS:-}"
TRACK_COMPARE_ENERGY="${TRACK_COMPARE_ENERGY:-1}"
COMPARE_ENERGY_POLL_INTERVAL="${COMPARE_ENERGY_POLL_INTERVAL:-5.0}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
OGAP_MAIN_THREADS="${OGAP_MAIN_THREADS:-1}"
OGAP_INTEROP_THREADS="${OGAP_INTEROP_THREADS:-1}"
OGAP_PREFETCH_FACTOR="${OGAP_PREFETCH_FACTOR:-1}"
OGAP_MAX_EVAL_WORKERS="${OGAP_MAX_EVAL_WORKERS:-8}"
OGAP_MAX_PREFETCH_FACTOR="${OGAP_MAX_PREFETCH_FACTOR:-1}"
OGAP_MAX_EVAL_INFLIGHT_CASES="${OGAP_MAX_EVAL_INFLIGHT_CASES:-8}"
OGAP_USE_NPY_CACHE="${OGAP_USE_NPY_CACHE:-0}"
OGAP_DROP_NPY_FILE_CACHE="${OGAP_DROP_NPY_FILE_CACHE:-1}"
OGAP_STAGE_DATA_TO_TMP="${OGAP_STAGE_DATA_TO_TMP:-1}"
OGAP_STAGE_DECOMPRESS_GZ="${OGAP_STAGE_DECOMPRESS_GZ:-1}"
OGAP_STAGE_WORKERS="${OGAP_STAGE_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
OGAP_STAGE_COLUMNS="${OGAP_STAGE_COLUMNS:-t1n,t1c,t2w,t2f,label}"
OGAP_N4_THREADS="${OGAP_N4_THREADS:-1}"
OGAP_N4_ITERATIONS="${OGAP_N4_ITERATIONS:-50,50,30,20}"
OGAP_DISABLE_N4="${OGAP_DISABLE_N4:-1}"
SCRATCH_ROOT="${SCRATCH:-${HOME}/links/scratch}"
SCRATCH_CACHE="${SCRATCH_ROOT}/.ogap_torchinductor_cache"

module --force purge
module load StdEnv/2023 python/3.11.5 cuda/12.6 scipy-stack/2024a
source "${ENV_PATH}/bin/activate"

export SLURM_EXPORT_ENV=ALL
export OMP_NUM_THREADS
export MKL_NUM_THREADS
export OPENBLAS_NUM_THREADS
export NUMEXPR_NUM_THREADS
export OGAP_MAIN_THREADS
export OGAP_INTEROP_THREADS
export OGAP_PREFETCH_FACTOR
export OGAP_MAX_EVAL_WORKERS
export OGAP_MAX_PREFETCH_FACTOR
export OGAP_MAX_EVAL_INFLIGHT_CASES
export OGAP_USE_NPY_CACHE
export OGAP_DROP_NPY_FILE_CACHE
export OGAP_STAGE_DATA_TO_TMP
export OGAP_STAGE_DECOMPRESS_GZ
export OGAP_STAGE_WORKERS
export OGAP_STAGE_COLUMNS
export OGAP_N4_THREADS
export OGAP_N4_ITERATIONS
export OGAP_DISABLE_N4
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="${ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS:-${OGAP_N4_THREADS}}"
export KMP_BLOCKTIME="${KMP_BLOCKTIME:-1}"
export KMP_SETTINGS="${KMP_SETTINGS:-0}"
export TMPDIR="${SLURM_TMPDIR}"
export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR}/pycache"
export TORCHINDUCTOR_CACHE_DIR="${SCRATCH_CACHE}"
export TRITON_CACHE_DIR="${SCRATCH_CACHE}/triton_cache"
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_AUTOGRAD_CACHE=1
export TORCHINDUCTOR_MAX_AUTOTUNE="${TORCHINDUCTOR_MAX_AUTOTUNE:-1}"
export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM="${TORCHINDUCTOR_MAX_AUTOTUNE_GEMM:-1}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:512}"

mkdir -p "${PROJECT_ROOT}/Scripts/logs"
mkdir -p "${OUT_DIR}" "${COMPARE_OUT_DIR}"
mkdir -p "${SCRATCH_CACHE}" "${TRITON_CACHE_DIR}" "${PYTHONPYCACHEPREFIX}"
CACHE_ROOT="${SLURM_TMPDIR}/ogap_export_int8_${SLURM_JOB_ID}"
COMPARE_CACHE_ARGS=()
if [[ "${OGAP_USE_NPY_CACHE}" == "1" ]]; then
  mkdir -p "${CACHE_ROOT}/npy_cache"
  COMPARE_CACHE_ARGS+=(--compare_cache_dir "${CACHE_ROOT}/npy_cache")
fi

echo "[preflight] Verifying export dependencies ..."
python - <<'PYEOF'
import os
import sys
try:
    import torch
    import numpy as np
    expected_torch = os.environ.get("EXPECTED_TORCH_PUBLIC_VERSION", "2.6.0")
    if torch.__version__.split("+", 1)[0] != expected_torch:
        raise RuntimeError(
            f"expected torch {expected_torch}, found {torch.__version__}; "
            "rerun the matching setup_ogap_env script"
        )
    if int(np.__version__.split(".", 1)[0]) >= 2:
        raise RuntimeError(
            f"expected scipy-stack numpy 1.x, found local numpy {np.__version__}; "
            "rerun setup/setup_ogap_env_v2.sh"
        )
    assert torch.cuda.is_available(), (
        "torch.cuda.is_available() is False — CUDA driver not found or GPU not allocated"
    )
    import onnx
    import onnxruntime as ort
    print(
        f"[preflight] OK — torch={torch.__version__} cuda={torch.version.cuda} "
        f"gpu={torch.cuda.get_device_name(0)} onnx={onnx.__version__} "
        f"ort={ort.__version__} providers={ort.get_available_providers()}"
    )
except Exception as exc:
    print(f"[preflight] FATAL: {exc}", file=sys.stderr)
    sys.exit(1)
PYEOF
echo "[preflight] All checks passed."

for pair in \
    "SCRIPT_PATH=${SCRIPT_PATH}" \
    "CKPT=${CKPT}" \
    "CALIBRATION_CSV=${CALIBRATION_CSV}"; do
  var="${pair%%=*}"
  path="${pair#*=}"
  if [[ ! -e "${path}" ]]; then
    echo "[ERROR][export] Required path (${var}) not found: ${path}" >&2
    exit 1
  fi
done
if [[ "${RUN_COMPARE}" == "1" && ! -e "${COMPARE_VAL_CSV}" ]]; then
  echo "[ERROR][export] COMPARE_VAL_CSV not found: ${COMPARE_VAL_CSV}" >&2
  if [[ "${COMPARE_VAL_CSV}" == *"/UTSW-"* && -d "${COMPARE_VAL_CSV%/UTSW-*}/USTW-glioma" ]]; then
    echo "        hint: '${COMPARE_VAL_CSV%/UTSW-*}/USTW-glioma' exists — rename to UTSW-glioma." >&2
  fi
  exit 1
fi

if [[ "${OGAP_STAGE_DATA_TO_TMP}" == "1" ]]; then
  : "${SLURM_TMPDIR:?SLURM_TMPDIR is required for OGAP_STAGE_DATA_TO_TMP=1}"
  STAGE_ROOT="${OGAP_STAGE_ROOT:-${SLURM_TMPDIR}/ogap_stage_export_${SLURM_JOB_ID}}"
  STAGE_FLAGS=()
  if [[ "${OGAP_STAGE_DECOMPRESS_GZ}" == "1" ]]; then
    STAGE_FLAGS+=(--decompress-gzip)
  fi
  STAGE_INPUTS=(--csv "CALIBRATION_CSV=${CALIBRATION_CSV}")
  if [[ "${RUN_COMPARE}" == "1" ]]; then
    STAGE_INPUTS+=(--csv "COMPARE_VAL_CSV=${COMPARE_VAL_CSV}")
  fi
  echo "[stage] export CSV staging enabled: root=${STAGE_ROOT}"
  STAGED_ENV_FILE="$(
    python "${PROJECT_ROOT}/Scripts/workflow/stage_ogap_csv_to_tmp.py" \
      --stage-root "${STAGE_ROOT}" \
      --workers "${OGAP_STAGE_WORKERS}" \
      --columns "${OGAP_STAGE_COLUMNS}" \
      --env-file "${STAGE_ROOT}/staged_env.sh" \
      "${STAGE_FLAGS[@]}" \
      "${STAGE_INPUTS[@]}"
  )"
  # shellcheck source=/dev/null
  source "${STAGED_ENV_FILE}"
  echo "[stage] export staged CALIBRATION_CSV=${CALIBRATION_CSV}"
  if [[ "${RUN_COMPARE}" == "1" ]]; then
    echo "[stage] export staged COMPARE_VAL_CSV=${COMPARE_VAL_CSV}"
  fi
fi

cd "${PROJECT_ROOT}"

EXPORT_ARGS=(
  --ckpt "${CKPT}"
  --out_path "${EXPORT_ONNX_PATH}"
  --modalities t1n,t1c,t2w,t2f
  --num_classes 4
  --patch_size 128,160,112
  --student_base "${STUDENT_BASE}"
  --calibration_csv "${CALIBRATION_CSV}"
  --calibration_samples "${CALIBRATION_SAMPLES}"
  --calibration_min_present "${CALIBRATION_MIN_PRESENT}"
  --calibration_max_present "${CALIBRATION_MAX_PRESENT}"
  --static_calibration_method "${STATIC_CALIBRATION_METHOD}"
)

if [[ "${SKIP_QUANT_PREPROCESS}" == "1" ]]; then
  EXPORT_ARGS+=(--skip_quant_preprocess)
fi
if [[ "${STATIC_PER_CHANNEL}" != "1" ]]; then
  EXPORT_ARGS+=(--no_static_per_channel)
fi
case "${STATIC_REDUCE_RANGE}" in
  auto|"") ;;
  1|true|TRUE|yes|YES) EXPORT_ARGS+=(--static_reduce_range) ;;
  0|false|FALSE|no|NO) EXPORT_ARGS+=(--no_static_reduce_range) ;;
  *)
    echo "[ERROR][export] STATIC_REDUCE_RANGE must be auto, 1, or 0; got ${STATIC_REDUCE_RANGE}" >&2
    exit 1
    ;;
esac

if [[ "${RUN_COMPARE}" == "1" ]]; then
  EXPORT_ARGS+=(
    --run_compare
    --compare_val_csv "${COMPARE_VAL_CSV}"
    --compare_out_dir "${COMPARE_OUT_DIR}"
    --compare_num_workers "${COMPARE_NUM_WORKERS}"
    --compare_eval_config_batch_size "${COMPARE_EVAL_CONFIG_BATCH_SIZE}"
    --compare_metric_workers "${COMPARE_METRIC_WORKERS}"
    --compare_energy_poll_interval "${COMPARE_ENERGY_POLL_INTERVAL}"
    "${COMPARE_CACHE_ARGS[@]}"
  )
  if [[ "${TRACK_COMPARE_ENERGY}" != "1" ]]; then
    EXPORT_ARGS+=(--no_compare_energy_tracking)
  fi
  if [[ -n "${COMPARE_NUM_THREADS}" ]]; then
    EXPORT_ARGS+=(--compare_num_threads "${COMPARE_NUM_THREADS}")
  fi
fi

srun --gpu-bind=closest --cpu-bind=cores --mem-bind=local python "${SCRIPT_PATH}" export \
  "${EXPORT_ARGS[@]}"

echo "export_onnx_path=${EXPORT_ONNX_PATH}"
echo "int8_static_path=${EXPORT_ONNX_PATH%.onnx}_int8_static.onnx"
echo "int8_dynamic_path=${EXPORT_ONNX_PATH%.onnx}_int8_dynamic.onnx"
echo "quantisation_compare_out_dir=${COMPARE_OUT_DIR}"
