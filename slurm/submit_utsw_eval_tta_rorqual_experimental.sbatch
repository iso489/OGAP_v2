#!/bin/bash
# Rorqual UTSW external evaluation job for OGAP v9.
# Default usage:
#   cd ${HOME}/ogap/OGAP_v2/Scripts
#   sbatch submit_utsw_eval_tta_rorqual_experimental.sbatch

#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=200G
#SBATCH --tmp=1000G
#SBATCH --gpus=h100:1
#SBATCH --job-name=utsw_eval_tta
#SBATCH --output=logs/utsw_eval_tta_%j.out
#SBATCH --error=logs/utsw_eval_tta_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${HOME}/ogap/OGAP_v2}"
DATA_ROOT="${DATA_ROOT:-${HOME}/ogap/OGAP_project}"
ENV_PATH="${ENV_PATH:-${HOME}/ogap/envs/ogap_env_v2}"
EXPECTED_TORCH_PUBLIC_VERSION="${EXPECTED_TORCH_PUBLIC_VERSION:-2.6.0}"
SCRIPT_PATH="${SCRIPT_PATH:-${PROJECT_ROOT}/Scripts/OGAP_source_code_experimental_v9.py}"
VAL_CSV="${VAL_CSV:-${DATA_ROOT}/Validation/UTSW-glioma/utsw_manual_val.csv}"
CKPT="${CKPT:-${PROJECT_ROOT}/Results/student_external_push/best_student.pth}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/Results/utsw_full_evaluation_tta}"
NUM_WORKERS="${NUM_WORKERS:-8}"
METRIC_WORKERS="${METRIC_WORKERS:-4}"
EVAL_CONFIG_BATCH_SIZE="${EVAL_CONFIG_BATCH_SIZE:-8}"
STUDENT_BASE="${STUDENT_BASE:-16}"
CLINICAL_METADATA_TSV="${CLINICAL_METADATA_TSV:-}"
FAILURE_MODE_METRIC="${FAILURE_MODE_METRIC:-pp_all_dice_brats_mean}"
FAILURE_MODE_QUANTILE="${FAILURE_MODE_QUANTILE:-0.25}"
FAILURE_MODE_MIN_GROUP_N="${FAILURE_MODE_MIN_GROUP_N:-10}"
LESION_WISE="${LESION_WISE:-1}"
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
# N4 bias-field correction is the dominant cost of the first eval pass when
# the npy cache is cold (slow path through MultiModalSegDataset._load /
# InferenceDataset).  Pin ITK threads here for the same reason as the
# teacher / student jobs - without these caps, the worker pool can dispatch
# hundreds of ITK threads onto one SLURM CPU allocation and the scheduler thrashes.
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
# Mamba evaluation runs use torch 2.5.1 + custom Triton kernels; torch.compile
# is not safe to enable.  Honour OGAP_DISABLE_TORCH_COMPILE if the caller (or
# the Mamba pipeline orchestrator) sets it.
if [[ -n "${OGAP_DISABLE_TORCH_COMPILE:-}" ]]; then
  export OGAP_DISABLE_TORCH_COMPILE
fi
export KMP_BLOCKTIME="${KMP_BLOCKTIME:-1}"
export KMP_SETTINGS="${KMP_SETTINGS:-0}"
export TMPDIR="${SLURM_TMPDIR}"
export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR}/pycache"
export TORCHINDUCTOR_CACHE_DIR="${SCRATCH_CACHE}"
export TRITON_CACHE_DIR="${SCRATCH_CACHE}/triton_cache"
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_AUTOGRAD_CACHE=1
export TORCHINDUCTOR_MAX_AUTOTUNE="${TORCHINDUCTOR_MAX_AUTOTUNE:-0}"
export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM="${TORCHINDUCTOR_MAX_AUTOTUNE_GEMM:-0}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:512}"

mkdir -p "${PROJECT_ROOT}/Scripts/logs"
mkdir -p "${OUT_DIR}"
mkdir -p "${SCRATCH_CACHE}" "${TRITON_CACHE_DIR}"
CACHE_ROOT="${SLURM_TMPDIR}/utsw_eval_${SLURM_JOB_ID}"
mkdir -p "${PYTHONPYCACHEPREFIX}"
CACHE_ARGS=()
if [[ "${OGAP_USE_NPY_CACHE}" == "1" ]]; then
  mkdir -p "${CACHE_ROOT}/npy_cache"
  CACHE_ARGS+=(--cache_dir "${CACHE_ROOT}/npy_cache")
fi

echo "[preflight] Verifying Triton + CUDA ..."
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
        "torch.cuda.is_available() is False - CUDA driver not found or GPU not allocated"
    )
    import triton
    sitk_state = "(SimpleITK absent)"
    try:
        import SimpleITK as _sitk
        try:
            _sitk_v = _sitk.Version.VersionString()
        except Exception:
            _sitk_v = getattr(_sitk, "__version__", "unknown")
        n4_threads_env = os.environ.get("OGAP_N4_THREADS", "1")
        sitk_default = _sitk.ProcessObject.GetGlobalDefaultNumberOfThreads()
        sitk_state = (
            f"sitk={_sitk_v} default_threads={sitk_default} "
            f"OGAP_N4_THREADS={n4_threads_env} "
            f"disabled={os.environ.get('OGAP_DISABLE_N4', '') or '0'} "
            f"iters={os.environ.get('OGAP_N4_ITERATIONS', '50,50,30,20 (default)')}"
        )
    except Exception as _sitk_exc:
        sitk_state = f"(SimpleITK probe failed: {_sitk_exc})"
    print(
        f"[preflight] OK - torch={torch.__version__}  "
        f"cuda={torch.version.cuda}  triton={triton.__version__}  "
        f"gpu={torch.cuda.get_device_name(0)}\n"
        f"[preflight] {sitk_state}"
    )
except AssertionError as exc:
    print(f"[preflight] FATAL: {exc}", file=sys.stderr)
    sys.exit(1)
except ImportError as exc:
    print(
        f"[preflight] FATAL: Triton import failed ({exc}).\n"
        "  Fix: run `bash setup/setup_ogap_env_v2.sh`, then resubmit this job.",
        file=sys.stderr,
    )
    sys.exit(1)
except RuntimeError as exc:
    print(f"[preflight] FATAL: {exc}", file=sys.stderr)
    sys.exit(1)
PYEOF
echo "[preflight] All checks passed."

for pair in \
    "SCRIPT_PATH=${SCRIPT_PATH}" \
    "VAL_CSV=${VAL_CSV}" \
    "CKPT=${CKPT}"; do
  var="${pair%%=*}"
  path="${pair#*=}"
  if [[ ! -e "${path}" ]]; then
    echo "[ERROR][utsw_eval] Required path (${var}) not found: ${path}" >&2
    if [[ "${path}" == *"/UTSW-"* && -d "${path%/UTSW-*}/USTW-glioma" ]]; then
      echo "        hint: '${path%/UTSW-*}/USTW-glioma' exists - rename to UTSW-glioma (UT Southwestern)." >&2
    fi
    exit 1
  fi
done
if [[ -n "${CLINICAL_METADATA_TSV}" && ! -e "${CLINICAL_METADATA_TSV}" ]]; then
  echo "[ERROR][utsw_eval] Required path (CLINICAL_METADATA_TSV) not found: ${CLINICAL_METADATA_TSV}" >&2
  exit 1
fi

if [[ "${OGAP_STAGE_DATA_TO_TMP}" == "1" ]]; then
  : "${SLURM_TMPDIR:?SLURM_TMPDIR is required for OGAP_STAGE_DATA_TO_TMP=1}"
  STAGE_ROOT="${OGAP_STAGE_ROOT:-${SLURM_TMPDIR}/ogap_stage_eval_${SLURM_JOB_ID}}"
  STAGE_FLAGS=()
  if [[ "${OGAP_STAGE_DECOMPRESS_GZ}" == "1" ]]; then
    STAGE_FLAGS+=(--decompress-gzip)
  fi
  echo "[stage] eval CSV staging enabled: root=${STAGE_ROOT}"
  STAGED_ENV_FILE="$(
    python "${PROJECT_ROOT}/Scripts/workflow/stage_ogap_csv_to_tmp.py" \
      --stage-root "${STAGE_ROOT}" \
      --workers "${OGAP_STAGE_WORKERS}" \
      --columns "${OGAP_STAGE_COLUMNS}" \
      --env-file "${STAGE_ROOT}/staged_env.sh" \
      "${STAGE_FLAGS[@]}" \
      --csv "VAL_CSV=${VAL_CSV}"
  )"
  # shellcheck source=/dev/null
  source "${STAGED_ENV_FILE}"
  echo "[stage] eval staged VAL_CSV=${VAL_CSV}"
fi

cd "${PROJECT_ROOT}"

EVAL_ARGS=(
  --ckpt "${CKPT}"
  --val_csv "${VAL_CSV}"
  --out_dir "${OUT_DIR}"
  --modalities t1n,t1c,t2w,t2f
  --num_classes 4
  --patch_size 128,160,112
  --student_base "${STUDENT_BASE}"
  --device cuda
  --num_workers "${NUM_WORKERS}"
  --eval_config_batch_size "${EVAL_CONFIG_BATCH_SIZE}"
  --metric_workers "${METRIC_WORKERS}"
  --tta
  "${CACHE_ARGS[@]}"
)
if [[ "${LESION_WISE}" == "1" ]]; then
  EVAL_ARGS+=(--lesion_wise)
fi
if [[ -n "${CLINICAL_METADATA_TSV}" ]]; then
  EVAL_ARGS+=(
    --clinical_metadata_tsv "${CLINICAL_METADATA_TSV}"
    --failure_mode_metric "${FAILURE_MODE_METRIC}"
    --failure_mode_quantile "${FAILURE_MODE_QUANTILE}"
    --failure_mode_min_group_n "${FAILURE_MODE_MIN_GROUP_N}"
  )
fi

srun --gpu-bind=closest --cpu-bind=cores --mem-bind=local python "${SCRIPT_PATH}" evaluate \
  "${EVAL_ARGS[@]}"
