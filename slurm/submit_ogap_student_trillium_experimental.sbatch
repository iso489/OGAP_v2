#!/bin/bash
# Trillium student KD training job for OGAP v9.
# Default usage:
#   cd /project/def-rdiaz/ilyaso/OGAP/Scripts
#   sbatch submit_ogap_student_trillium_experimental.sbatch
#
# Override teacher checkpoint if needed:
#   sbatch --export=ALL,TEACHER_CKPT=/home/.../Results/teacher_seed2/best_teacher.pth submit_ogap_student_trillium_experimental.sbatch
#
# ── Trillium adaptation (vs the Rorqual original) ───────────────────────────
#   • --account=def-rdiaz
#   • --time capped at 24:00:00 (Trillium hard walltime limit; Rorqual used 48h).
#     A full 1000-epoch student will NOT finish in one 24h window - resubmit to
#     continue from the latest checkpoint in OUT_DIR, or reduce --epochs.
#     See README_TRILLIUM.md §"24-hour walltime".
#   • 1 GPU = 1/4 of a Trillium GPU node → ≤24 cores, ≤188 GiB RAM (auto-assigned); no MIG.
#     We request --gpus-per-node=1; 24 cores; do NOT set --mem (memory auto = 188 GiB).
#   • Diskless nodes: no --tmp; per-job scratch (staging, caches, TMPDIR) lives
#     under $SCRATCH in OGAP_JOB_TMP, removed on exit.
#SBATCH --account=def-rdiaz
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=1
#SBATCH --job-name=ogap_student_ext
#SBATCH --output=/scratch/ilyaso/ogap/logs/ogap_student_%j.out
#SBATCH --error=/scratch/ilyaso/ogap/logs/ogap_student_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/project/def-rdiaz/ilyaso/OGAP}"
ENV_PATH="${ENV_PATH:-${HOME}/ogap/envs/ogap_env_v2}"
EXPECTED_TORCH_PUBLIC_VERSION="${EXPECTED_TORCH_PUBLIC_VERSION:-2.6.0}"
SCRIPT_PATH="${SCRIPT_PATH:-${PROJECT_ROOT}/Scripts/OGAP_source_code_experimental_v9.py}"
TRAIN_CSV="${TRAIN_CSV:-${PROJECT_ROOT}/Scripts/train.csv}"
VAL_CSV="${VAL_CSV:-${PROJECT_ROOT}/Scripts/val.csv}"
TEACHER_CKPT="${TEACHER_CKPT:-${SCRATCH:-${HOME}/scratch}/ogap/Results/teacher/best_teacher.pth}"
OUT_DIR="${OUT_DIR:-${SCRATCH:-${HOME}/scratch}/ogap/Results/student_external_push}"
NUM_WORKERS="${NUM_WORKERS:-16}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-4}"
STUDENT_BATCH_SIZE="${STUDENT_BATCH_SIZE:-4}"
STUDENT_BASE="${STUDENT_BASE:-16}"
TEACHER_BASE="${TEACHER_BASE:-32}"
TEACHER_ARCH="${TEACHER_ARCH:-unet3d}"
# MUST match the block_style the teacher checkpoint was trained with.
# Mismatch will fail load_state_dict because tensor shapes differ between
# mednext (depthwise+pointwise) and dense (full 3x3x3) blocks.
TEACHER_BLOCK_STYLE="${TEACHER_BLOCK_STYLE:-mednext}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
LMIC_FIELD_STRENGTH_PRIOR="${LMIC_FIELD_STRENGTH_PRIOR:-1}"
CONTRAST_MOD_INDICES="${CONTRAST_MOD_INDICES:-1}"
CONTRAST_DROPOUT_EXTRA_PROB="${CONTRAST_DROPOUT_EXTRA_PROB:-0.50}"
P_ANISOTROPIC_2D="${P_ANISOTROPIC_2D:-0.85}"
P_PARTIAL_CONTRAST="${P_PARTIAL_CONTRAST:-0.30}"
P_LABEL_NOISE="${P_LABEL_NOISE:-0.20}"
# Architecture-pass physics augmentations (defaults match source = OFF/static)
LABEL_NOISE_MODE="${LABEL_NOISE_MODE:-morph}"
LABEL_NOISE_BAND_RADIUS="${LABEL_NOISE_BAND_RADIUS:-2}"
LABEL_NOISE_BAND_FLIP_PROB="${LABEL_NOISE_BAND_FLIP_PROB:-0.30}"
P_FIELD_CONTRAST_WARP="${P_FIELD_CONTRAST_WARP:-0.0}"
FIELD_CONTRAST_SOURCE_B0="${FIELD_CONTRAST_SOURCE_B0:-3.0}"
P_FOURIER_AMPLITUDE_MIX="${P_FOURIER_AMPLITUDE_MIX:-0.0}"
# AFA dataset-side donor sourcing probability - MUST be > 0 for the augmentor's
# Fourier-amplitude mix to actually receive donors.  Both knobs together control AFA.
AFA_PROB="${AFA_PROB:-0.0}"
P_RECEIVE_COIL_INHOMOGENEITY="${P_RECEIVE_COIL_INHOMOGENEITY:-0.0}"
# B0 off-resonance signal voids (Campbell-Washburn 2023)
P_OFF_RESONANCE_VOID="${P_OFF_RESONANCE_VOID:-0.0}"
NOISE_CALIBRATION="${NOISE_CALIBRATION:-static}"
CARVEMIX_PROB="${CARVEMIX_PROB:-0.0}"
CARVEMIX_DILATION="${CARVEMIX_DILATION:-5}"
P_ULF_GAN_SYNTHESIS="${P_ULF_GAN_SYNTHESIS:-0.0}"
ULF_GAN_WEIGHTS="${ULF_GAN_WEIGHTS:-}"
ULF_GAN_TARGET_B0="${ULF_GAN_TARGET_B0:-0.064}"
# Distribution-shift / feature-DR knobs (student only)
LAMBDA_VREX="${LAMBDA_VREX:-0.0}"
FEATURE_DR="${FEATURE_DR:-none}"
# MixStyle3D control: P (per-batch activation) and α (Beta(α,α) mix)
FEATURE_DR_P="${FEATURE_DR_P:-0.5}"
FEATURE_DR_ALPHA="${FEATURE_DR_ALPHA:-0.1}"
# Modality dropout guardrails - keep [min, max] modalities per sample
KEEP_MIN_MODALITIES="${KEEP_MIN_MODALITIES:-1}"
KEEP_MAX_MODALITIES="${KEEP_MAX_MODALITIES:-4}"
HOLDER_ALPHA="${HOLDER_ALPHA:-1.5}"
AUTO_RESOURCES="${AUTO_RESOURCES:-1}"
AUTO_RESOURCES_SAFETY="${AUTO_RESOURCES_SAFETY:-0.70}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
OGAP_MAIN_THREADS="${OGAP_MAIN_THREADS:-1}"
OGAP_INTEROP_THREADS="${OGAP_INTEROP_THREADS:-1}"
OGAP_PREFETCH_FACTOR="${OGAP_PREFETCH_FACTOR:-1}"
OGAP_MAX_TRAIN_WORKERS="${OGAP_MAX_TRAIN_WORKERS:-16}"
OGAP_MAX_EVAL_WORKERS="${OGAP_MAX_EVAL_WORKERS:-8}"
OGAP_MAX_PREFETCH_FACTOR="${OGAP_MAX_PREFETCH_FACTOR:-1}"
OGAP_MAX_TRAIN_INFLIGHT_CASES="${OGAP_MAX_TRAIN_INFLIGHT_CASES:-32}"
OGAP_MAX_EVAL_INFLIGHT_CASES="${OGAP_MAX_EVAL_INFLIGHT_CASES:-8}"
OGAP_AUTO_BATCH_UPPER_BOUND="${OGAP_AUTO_BATCH_UPPER_BOUND:-16}"
OGAP_USE_NPY_CACHE="${OGAP_USE_NPY_CACHE:-0}"
OGAP_USE_PATCH_CACHE="${OGAP_USE_PATCH_CACHE:-0}"
OGAP_DROP_NPY_FILE_CACHE="${OGAP_DROP_NPY_FILE_CACHE:-1}"
OGAP_STAGE_DATA_TO_TMP="${OGAP_STAGE_DATA_TO_TMP:-1}"
OGAP_STAGE_DECOMPRESS_GZ="${OGAP_STAGE_DECOMPRESS_GZ:-1}"
OGAP_STAGE_WORKERS="${OGAP_STAGE_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
OGAP_STAGE_COLUMNS="${OGAP_STAGE_COLUMNS:-t1n,t1c,t2w,t2f,label}"
# N4 bias-field correction is the dominant cost of the first epoch
# (slow path through MultiModalSegDataset._load).  Without these caps,
# every DataLoader worker spawns ~num-CPU ITK threads, oversubscribes the
# Slurm CPU allocation, and the OS scheduler thrashes - the same hang that broke the
# teacher run (jobs 11111229 / 11112647).
#
#   OGAP_N4_THREADS    : ITK thread budget per worker (default 1)
#   OGAP_N4_ITERATIONS : per-level multi-resolution schedule
#                        (default in source: 50,50,30,20)
#   OGAP_DISABLE_N4    : default 1 on Trillium so cold-cache CPU N4 cannot
#                        starve the H100 for hours. Set OGAP_DISABLE_N4=0
#                        only for an explicit N4 ablation/preprocessing run.
OGAP_N4_THREADS="${OGAP_N4_THREADS:-1}"
OGAP_N4_ITERATIONS="${OGAP_N4_ITERATIONS:-50,50,30,20}"
OGAP_DISABLE_N4="${OGAP_DISABLE_N4:-1}"
SCRATCH_ROOT="${SCRATCH:-${HOME}/scratch}"
SCRATCH_CACHE="${SCRATCH_ROOT}/.ogap_torchinductor_cache"

# Trillium compute nodes are DISKLESS: there is no node-local NVMe and no
# $SLURM_TMPDIR scratch disk to request via --tmp.  Per-job scratch (data
# staging, npy/patch caches, TMPDIR, __pycache__) lives under $SCRATCH in a
# job-private directory that is removed on exit.
OGAP_JOB_TMP="${OGAP_JOB_TMP:-${SCRATCH_ROOT}/ogap_jobtmp/${SLURM_JOB_NAME:-ogap}_${SLURM_JOB_ID}}"
mkdir -p "${OGAP_JOB_TMP}"
trap 'rm -rf "${OGAP_JOB_TMP}"' EXIT

PATCH_CACHE_START_EPOCH="${PATCH_CACHE_START_EPOCH:-2}"

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
export OGAP_MAX_TRAIN_WORKERS
export OGAP_MAX_EVAL_WORKERS
export OGAP_MAX_PREFETCH_FACTOR
export OGAP_MAX_TRAIN_INFLIGHT_CASES
export OGAP_MAX_EVAL_INFLIGHT_CASES
export OGAP_AUTO_BATCH_UPPER_BOUND
export OGAP_USE_NPY_CACHE
export OGAP_USE_PATCH_CACHE
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
export TMPDIR="${OGAP_JOB_TMP}"
export PYTHONPYCACHEPREFIX="${OGAP_JOB_TMP}/pycache"
export TORCHINDUCTOR_CACHE_DIR="${SCRATCH_CACHE}"
export TRITON_CACHE_DIR="${SCRATCH_CACHE}/triton_cache"
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_AUTOGRAD_CACHE=1
export TORCHINDUCTOR_MAX_AUTOTUNE="${TORCHINDUCTOR_MAX_AUTOTUNE:-1}"
export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM="${TORCHINDUCTOR_MAX_AUTOTUNE_GEMM:-1}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:512}"

mkdir -p "${SCRATCH:-${HOME}/scratch}/ogap/logs"
mkdir -p "${OUT_DIR}"
mkdir -p "${SCRATCH_CACHE}" "${TRITON_CACHE_DIR}"
CACHE_ROOT="${OGAP_JOB_TMP}/ogap_student_${SLURM_JOB_ID}"
if [[ "${OGAP_USE_PATCH_CACHE}" == "1" && -z "${PATCH_CACHE_ROOT:-}" ]]; then
  if [[ -d /dev/shm && -w /dev/shm ]]; then
    PATCH_CACHE_ROOT="/dev/shm/ogap_student_${SLURM_JOB_ID}/patch_cache"
  else
    PATCH_CACHE_ROOT="${CACHE_ROOT}/patch_cache"
    echo "[warning] /dev/shm unavailable or unwritable; patch cache falling back to ${PATCH_CACHE_ROOT}" >&2
  fi
fi
mkdir -p "${PYTHONPYCACHEPREFIX}"
CACHE_ARGS=()
if [[ "${OGAP_USE_NPY_CACHE}" == "1" ]]; then
  mkdir -p "${CACHE_ROOT}/npy_cache"
  CACHE_ARGS+=(--cache_dir "${CACHE_ROOT}/npy_cache")
fi
if [[ "${OGAP_USE_PATCH_CACHE}" == "1" ]]; then
  mkdir -p "${PATCH_CACHE_ROOT}"
  CACHE_ARGS+=(--patch_cache_dir "${PATCH_CACHE_ROOT}" --patch_cache_start_epoch "${PATCH_CACHE_START_EPOCH}")
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
    "TRAIN_CSV=${TRAIN_CSV}" \
    "VAL_CSV=${VAL_CSV}" \
    "TEACHER_CKPT=${TEACHER_CKPT}"; do
  var="${pair%%=*}"
  path="${pair#*=}"
  if [[ ! -e "${path}" ]]; then
    echo "[ERROR][student] Required path (${var}) not found: ${path}" >&2
    if [[ "${var}" == "VAL_CSV" && "${path}" == *"/Validation/"* ]]; then
      echo "        hint: the student expects the *internal* val split (\${PROJECT_ROOT}/Scripts/val.csv)," >&2
      echo "              NOT the external UTSW eval CSV. Check if VAL_CSV leaked from the submitter env," >&2
      echo "              or pass VAL_CSV=\${PROJECT_ROOT}/Scripts/val.csv explicitly." >&2
    fi
    if [[ "${path}" == *"/UTSW-"* && -d "${path%/UTSW-*}/USTW-glioma" ]]; then
      echo "        hint: '${path%/UTSW-*}/USTW-glioma' exists - rename to UTSW-glioma (UT Southwestern)." >&2
    fi
    exit 1
  fi
done

if [[ "${OGAP_STAGE_DATA_TO_TMP}" == "1" ]]; then
  # Trillium is diskless: stage to a job-private $SCRATCH dir (OGAP_JOB_TMP),
  # not $SLURM_TMPDIR.  The decompress-gzip pass (gz -> nii) still pays off
  # across epochs by removing per-read gunzip from every DataLoader worker.
  STAGE_ROOT="${OGAP_STAGE_ROOT:-${OGAP_JOB_TMP}/ogap_stage_student_${SLURM_JOB_ID}}"
  STAGE_FLAGS=()
  if [[ "${OGAP_STAGE_DECOMPRESS_GZ}" == "1" ]]; then
    STAGE_FLAGS+=(--decompress-gzip)
  fi
  echo "[stage] student CSV staging enabled: root=${STAGE_ROOT}"
  STAGED_ENV_FILE="$(
    python "${PROJECT_ROOT}/Scripts/workflow/stage_ogap_csv_to_tmp.py" \
      --stage-root "${STAGE_ROOT}" \
      --workers "${OGAP_STAGE_WORKERS}" \
      --columns "${OGAP_STAGE_COLUMNS}" \
      --env-file "${STAGE_ROOT}/staged_env.sh" \
      "${STAGE_FLAGS[@]}" \
      --csv "TRAIN_CSV=${TRAIN_CSV}" \
      --csv "VAL_CSV=${VAL_CSV}"
  )"
  # shellcheck source=/dev/null
  source "${STAGED_ENV_FILE}"
  echo "[stage] student staged TRAIN_CSV=${TRAIN_CSV}"
  echo "[stage] student staged VAL_CSV=${VAL_CSV}"
fi

# /project is READ-ONLY on Trillium compute nodes; run from writable scratch.
# All inputs (SCRIPT_PATH, CSVs, OUT_DIR) are absolute, so cwd does not affect them.
cd "${OGAP_JOB_TMP}"

REALISM_ARGS=()
if [[ "${LMIC_FIELD_STRENGTH_PRIOR}" == "1" ]]; then
  REALISM_ARGS+=(--lmic_field_strength_prior)
fi
if [[ "${AUTO_RESOURCES}" == "1" ]]; then
  REALISM_ARGS+=(--auto_resources --auto_resources_safety "${AUTO_RESOURCES_SAFETY}")
fi
# Architecture-pass physics augmentations + V-REx + feature_dr.  These match the
# source-code defaults when env vars are absent, so unmodified runs are unchanged.
REALISM_ARGS+=(
  --label_noise_mode "${LABEL_NOISE_MODE}"
  --label_noise_band_radius "${LABEL_NOISE_BAND_RADIUS}"
  --label_noise_band_flip_prob "${LABEL_NOISE_BAND_FLIP_PROB}"
  --p_field_contrast_warp "${P_FIELD_CONTRAST_WARP}"
  --field_contrast_source_b0 "${FIELD_CONTRAST_SOURCE_B0}"
  --p_fourier_amplitude_mix "${P_FOURIER_AMPLITUDE_MIX}"
  --afa_prob "${AFA_PROB}"
  --p_receive_coil_inhomogeneity "${P_RECEIVE_COIL_INHOMOGENEITY}"
  --p_off_resonance_void "${P_OFF_RESONANCE_VOID}"
  --noise_calibration "${NOISE_CALIBRATION}"
  --carvemix_prob "${CARVEMIX_PROB}"
  --carvemix_dilation "${CARVEMIX_DILATION}"
  --p_ulf_gan_synthesis "${P_ULF_GAN_SYNTHESIS}"
  --ulf_gan_target_b0 "${ULF_GAN_TARGET_B0}"
  --lambda_vrex "${LAMBDA_VREX}"
  --feature_dr "${FEATURE_DR}"
  --feature_dr_p "${FEATURE_DR_P}"
  --feature_dr_alpha "${FEATURE_DR_ALPHA}"
  --keep_min_modalities "${KEEP_MIN_MODALITIES}"
  --keep_max_modalities "${KEEP_MAX_MODALITIES}"
)
if [[ -n "${ULF_GAN_WEIGHTS}" ]]; then
  REALISM_ARGS+=(--ulf_gan_weights "${ULF_GAN_WEIGHTS}")
fi

srun --gpu-bind=closest --cpu-bind=cores --mem-bind=local python "${SCRIPT_PATH}" train \
  --train_csv "${TRAIN_CSV}" \
  --val_csv "${VAL_CSV}" \
  --out_dir "${OUT_DIR}" \
  --modalities t1n,t1c,t2w,t2f \
  --num_classes 4 \
  --teacher_ckpt "${TEACHER_CKPT}" \
  --patch_size 128,160,112 \
  --batch_size "${STUDENT_BATCH_SIZE}" \
  --epochs 1000 \
  --lr 1e-3 \
  --weight_decay 3e-5 \
  --grad_accum_steps "${GRAD_ACCUM_STEPS}" \
  --num_workers "${NUM_WORKERS}" \
  --val_num_workers "${VAL_NUM_WORKERS}" \
  --teacher_base "${TEACHER_BASE}" \
  --student_base "${STUDENT_BASE}" \
  --teacher_arch "${TEACHER_ARCH}" \
  --teacher_block_style "${TEACHER_BLOCK_STYLE}" \
  --curriculum_floor_modalities 1 \
  --keep_count_bias_power 0.0 \
  --lambda_kd 2.0 \
  --lambda_feat 0.5 \
  --feature_distill_sparse_decay 1.50 \
  --lambda_ds 1.0 \
  --lambda_region 0.50 \
  --temperature_start 4.0 \
  --temperature_end 4.0 \
  --curriculum_ramp 0.2 \
  --holder_alpha "${HOLDER_ALPHA}" \
  --attention none \
  --confidence_gate 0.7 \
  --field_strength_sampling_power 0.5 \
  --low_field_sampling_bias 1.5 \
  --grade23_sampling_bias 1.5 \
  --idh_mutant_sampling_bias 1.5 \
  --legacy_protocol_sampling_bias 1.25 \
  --legacy_protocol_year_threshold 2013 \
  --focal_b1_inhomogeneity_prob 0.12 \
  --hd95_freq 50 \
  --val_fast_n 9 \
  --val_start_epoch 3 \
  --val_full_freq 25 \
  "${CACHE_ARGS[@]}" \
  --contrast_mod_indices "${CONTRAST_MOD_INDICES}" \
  --contrast_dropout_extra_prob "${CONTRAST_DROPOUT_EXTRA_PROB}" \
  --p_anisotropic_2d "${P_ANISOTROPIC_2D}" \
  --p_partial_contrast "${P_PARTIAL_CONTRAST}" \
  --p_label_noise "${P_LABEL_NOISE}" \
  "${REALISM_ARGS[@]}"
