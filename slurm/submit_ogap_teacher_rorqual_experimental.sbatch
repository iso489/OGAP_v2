#!/bin/bash
# Rorqual teacher pretraining job for OGAP v9.
# Default usage:
#   cd ${HOME}/ogap/OGAP_v2/Scripts
#   sbatch submit_ogap_teacher_rorqual_experimental.sbatch
#
# Optional override:
#   sbatch --export=ALL,OUT_DIR=/home/.../Results/teacher_seed2 submit_ogap_teacher_rorqual_experimental.sbatch

#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=200G
#SBATCH --tmp=1000G
#SBATCH --gpus=h100:1
#SBATCH --job-name=ogap_teacher_ext
#SBATCH --output=logs/ogap_teacher_%j.out
#SBATCH --error=logs/ogap_teacher_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${HOME}/ogap/OGAP_v2}"
ENV_PATH="${ENV_PATH:-${HOME}/ogap/envs/ogap_env_v2}"
EXPECTED_TORCH_PUBLIC_VERSION="${EXPECTED_TORCH_PUBLIC_VERSION:-2.6.0}"
SCRIPT_PATH="${SCRIPT_PATH:-${PROJECT_ROOT}/Scripts/OGAP_source_code_experimental_v9.py}"
TRAIN_CSV="${TRAIN_CSV:-${PROJECT_ROOT}/Scripts/train.csv}"
VAL_CSV="${VAL_CSV:-${PROJECT_ROOT}/Scripts/val.csv}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/Results/teacher}"
NUM_WORKERS="${NUM_WORKERS:-16}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-4}"
TEACHER_BATCH_SIZE="${TEACHER_BATCH_SIZE:-2}"
# 2026 audit: TEACHER_BASE default raised from 32 -> 64.  The teacher is a
# KD-only model and is NEVER exported to INT8 — only the student is.  A heavier
# teacher gives a stronger KD signal at no inference-time cost, since the
# student (UNet3DStudent / MedNeXt-S) is what gets deployed on CPU.  Approximate
# parameter counts at base=64 dense: ~46M (UNet3D) / ~50M (SegMamba); both fit
# comfortably on a single H100 80GB with batch_size=2 at patch_size=192^2x144.
TEACHER_BASE="${TEACHER_BASE:-64}"
TEACHER_ARCH="${TEACHER_ARCH:-unet3d}"
# mednext = depthwise-separable MedNeXt-S blocks (~1M @ base=32, default for
# backward compat with old checkpoints).  dense = full 3x3x3 conv blocks
# (~6M @ base=32, ~15M @ base=48, ~26M @ base=64); use 'dense' for heavy KD
# teachers — only the student is exported to INT8 so the teacher does not need
# to be quantization-friendly.
TEACHER_BLOCK_STYLE="${TEACHER_BLOCK_STYLE:-mednext}"
# FIX (Gap A + Gap B, 2026 audit): teacher-side feature-DR + modality dropout
# defaults.  When set, the teacher's KD logits encode cross-scanner invariance
# AND missing-modality behaviour, which transfers to the student via KD loss.
TEACHER_FEATURE_DR="${TEACHER_FEATURE_DR:-none}"
TEACHER_FEATURE_DR_P="${TEACHER_FEATURE_DR_P:-0.5}"
TEACHER_FEATURE_DR_ALPHA="${TEACHER_FEATURE_DR_ALPHA:-0.1}"
TEACHER_MODALITY_DROPOUT_P="${TEACHER_MODALITY_DROPOUT_P:-0.0}"
TEACHER_MODALITY_DROPOUT_MAX_DROP="${TEACHER_MODALITY_DROPOUT_MAX_DROP:-2}"
LMIC_FIELD_STRENGTH_PRIOR="${LMIC_FIELD_STRENGTH_PRIOR:-1}"
CONTRAST_MOD_INDICES="${CONTRAST_MOD_INDICES:-1}"
P_ANISOTROPIC_2D="${P_ANISOTROPIC_2D:-0.85}"
P_PARTIAL_CONTRAST="${P_PARTIAL_CONTRAST:-0.30}"
P_LABEL_NOISE="${P_LABEL_NOISE:-0.20}"
# Architecture-pass physics augmentations (defaults match source defaults = OFF/static)
LABEL_NOISE_MODE="${LABEL_NOISE_MODE:-morph}"
LABEL_NOISE_BAND_RADIUS="${LABEL_NOISE_BAND_RADIUS:-2}"
LABEL_NOISE_BAND_FLIP_PROB="${LABEL_NOISE_BAND_FLIP_PROB:-0.30}"
P_FIELD_CONTRAST_WARP="${P_FIELD_CONTRAST_WARP:-0.0}"
FIELD_CONTRAST_SOURCE_B0="${FIELD_CONTRAST_SOURCE_B0:-3.0}"
P_FOURIER_AMPLITUDE_MIX="${P_FOURIER_AMPLITUDE_MIX:-0.0}"
# AFA dataset-side donor sourcing probability — MUST be > 0 for the augmentor's
# Fourier-amplitude mix to actually receive donor batches.  Both knobs together
# control AFA: dataset (afa_prob) sources donors; augmentor (p_fourier_amplitude_mix)
# applies the mix.  Setting only one silently disables AFA.
AFA_PROB="${AFA_PROB:-0.0}"
P_RECEIVE_COIL_INHOMOGENEITY="${P_RECEIVE_COIL_INHOMOGENEITY:-0.0}"
# B0 off-resonance signal voids near sinus/mastoid/brainstem (Campbell-Washburn 2023)
P_OFF_RESONANCE_VOID="${P_OFF_RESONANCE_VOID:-0.0}"
NOISE_CALIBRATION="${NOISE_CALIBRATION:-static}"
CARVEMIX_PROB="${CARVEMIX_PROB:-0.0}"
CARVEMIX_DILATION="${CARVEMIX_DILATION:-5}"
# NOTE: ULF GAN synthesis (--p_ulf_gan_synthesis / --ulf_gan_weights / --ulf_gan_target_b0)
# is a STUDENT-ONLY argument in OGAP_source_code_experimental_v9.py — the
# pretrain_teacher subparser does not accept it.  Do NOT add it to REALISM_ARGS here.
AUTO_RESOURCES="${AUTO_RESOURCES:-1}"
AUTO_RESOURCES_SAFETY="${AUTO_RESOURCES_SAFETY:-0.85}"
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
# Slurm CPU allocation, and the OS scheduler thrashes — causing first-epoch
# hangs of >19 hours (see jobs 11111229 / 11112647).
#
#   OGAP_N4_THREADS    : ITK thread budget per worker (default 1)
#   OGAP_N4_ITERATIONS : per-level multi-resolution schedule
#                        (default in source: 50,50,30,20)
#   OGAP_DISABLE_N4    : default 1 on Rorqual so cold-cache CPU N4 cannot
#                        starve the H100 for hours. Set OGAP_DISABLE_N4=0
#                        only for an explicit N4 ablation/preprocessing run.
OGAP_N4_THREADS="${OGAP_N4_THREADS:-1}"
OGAP_N4_ITERATIONS="${OGAP_N4_ITERATIONS:-50,50,30,20}"
OGAP_DISABLE_N4="${OGAP_DISABLE_N4:-1}"
SCRATCH_ROOT="${SCRATCH:-${HOME}/links/scratch}"
SCRATCH_CACHE="${SCRATCH_ROOT}/.ogap_torchinductor_cache"
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
# ITK reads its own env var as a secondary safety net (some ITK paths bypass
# the SimpleITK Python wrapper).  Mirror the explicit cap so both agree.
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="${ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS:-${OGAP_N4_THREADS}}"
export KMP_BLOCKTIME="${KMP_BLOCKTIME:-1}"
export KMP_SETTINGS="${KMP_SETTINGS:-0}"
export TMPDIR="${SLURM_TMPDIR}"
export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR}/pycache"
export TORCHINDUCTOR_CACHE_DIR="${SCRATCH_CACHE}"
# Keep Triton kernel cache persistent across jobs; SLURM_TMPDIR is wiped per job.
export TRITON_CACHE_DIR="${SCRATCH_CACHE}/triton_cache"
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_AUTOGRAD_CACHE=1
export TORCHINDUCTOR_MAX_AUTOTUNE="${TORCHINDUCTOR_MAX_AUTOTUNE:-1}"
export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM="${TORCHINDUCTOR_MAX_AUTOTUNE_GEMM:-1}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:512}"

mkdir -p "${PROJECT_ROOT}/Scripts/logs"
mkdir -p "${OUT_DIR}"
mkdir -p "${SCRATCH_CACHE}" "${TRITON_CACHE_DIR}"
CACHE_ROOT="${SLURM_TMPDIR}/ogap_teacher_${SLURM_JOB_ID}"
if [[ "${OGAP_USE_PATCH_CACHE}" == "1" && -z "${PATCH_CACHE_ROOT:-}" ]]; then
  if [[ -d /dev/shm && -w /dev/shm ]]; then
    PATCH_CACHE_ROOT="/dev/shm/ogap_teacher_${SLURM_JOB_ID}/patch_cache"
  else
    PATCH_CACHE_ROOT="${CACHE_ROOT}/patch_cache"
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
        "torch.cuda.is_available() is False — CUDA driver not found or GPU not allocated"
    )
    import triton
    # SimpleITK / N4 thread budget probe.  N4 first-epoch oversubscription was
    # the cause of the 19h teacher hang — surface the resolved cap here so any
    # regression (e.g. a future env where SimpleITK ignores the env var) is
    # caught at preflight rather than after sbatch queues the job.
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
        f"[preflight] OK — torch={torch.__version__}  "
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
    "VAL_CSV=${VAL_CSV}"; do
  var="${pair%%=*}"
  path="${pair#*=}"
  if [[ ! -e "${path}" ]]; then
    echo "[ERROR][teacher] Required path (${var}) not found: ${path}" >&2
    if [[ "${path}" == *"/UTSW-"* && -d "${path%/UTSW-*}/USTW-glioma" ]]; then
      echo "        hint: '${path%/UTSW-*}/USTW-glioma' exists — rename to UTSW-glioma (UT Southwestern)." >&2
    fi
    exit 1
  fi
done

if [[ "${OGAP_STAGE_DATA_TO_TMP}" == "1" ]]; then
  : "${SLURM_TMPDIR:?SLURM_TMPDIR is required for OGAP_STAGE_DATA_TO_TMP=1}"
  STAGE_ROOT="${OGAP_STAGE_ROOT:-${SLURM_TMPDIR}/ogap_stage_teacher_${SLURM_JOB_ID}}"
  STAGE_FLAGS=()
  if [[ "${OGAP_STAGE_DECOMPRESS_GZ}" == "1" ]]; then
    STAGE_FLAGS+=(--decompress-gzip)
  fi
  echo "[stage] teacher CSV staging enabled: root=${STAGE_ROOT}"
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
  echo "[stage] teacher staged TRAIN_CSV=${TRAIN_CSV}"
  echo "[stage] teacher staged VAL_CSV=${VAL_CSV}"
fi

cd "${PROJECT_ROOT}"

REALISM_ARGS=()
if [[ "${LMIC_FIELD_STRENGTH_PRIOR}" == "1" ]]; then
  REALISM_ARGS+=(--lmic_field_strength_prior)
fi
if [[ "${AUTO_RESOURCES}" == "1" ]]; then
  REALISM_ARGS+=(--auto_resources --auto_resources_safety "${AUTO_RESOURCES_SAFETY}")
fi
# Architecture-pass physics augmentations (only added when explicitly requested
# via env var; default values match the source defaults so behaviour is unchanged
# for unmodified runs).
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
  # FIX (Gap A, Gap B, Finding 4, 2026 audit) — new teacher robustness knobs.
  --teacher_feature_dr "${TEACHER_FEATURE_DR}"
  --teacher_feature_dr_p "${TEACHER_FEATURE_DR_P}"
  --teacher_feature_dr_alpha "${TEACHER_FEATURE_DR_ALPHA}"
  --teacher_modality_dropout_p "${TEACHER_MODALITY_DROPOUT_P}"
  --teacher_modality_dropout_max_drop "${TEACHER_MODALITY_DROPOUT_MAX_DROP}"
)
if [[ "${FORCE_RESUME_PARTIAL:-0}" == "1" ]]; then
  REALISM_ARGS+=(--force_resume_partial)
fi
# NOTE: --p_ulf_gan_synthesis / --ulf_gan_target_b0 / --ulf_gan_weights are
# student-only flags in the source.  They are intentionally NOT appended for
# pretrain_teacher — argparse would reject them with "unrecognized arguments".

srun --gpu-bind=closest --cpu-bind=cores --mem-bind=local python "${SCRIPT_PATH}" pretrain_teacher \
  --train_csv "${TRAIN_CSV}" \
  --val_csv "${VAL_CSV}" \
  --out_dir "${OUT_DIR}" \
  --modalities t1n,t1c,t2w,t2f \
  --num_classes 4 \
  --patch_size 192,192,144 \
  --batch_size "${TEACHER_BATCH_SIZE}" \
  --teacher_arch "${TEACHER_ARCH}" \
  --teacher_block_style "${TEACHER_BLOCK_STYLE}" \
  --epochs 1000 \
  --lr 2e-4 \
  --weight_decay 3e-5 \
  --num_workers "${NUM_WORKERS}" \
  --val_num_workers "${VAL_NUM_WORKERS}" \
  --teacher_base "${TEACHER_BASE}" \
  --warmup_epochs 5 \
  --lambda_region 0.50 \
  --field_strength_sampling_power 0.5 \
  --low_field_sampling_bias 1.5 \
  --grade23_sampling_bias 1.5 \
  --idh_mutant_sampling_bias 1.5 \
  --legacy_protocol_sampling_bias 1.25 \
  --legacy_protocol_year_threshold 2013 \
  --focal_b1_inhomogeneity_prob 0.12 \
  --hd95_freq 50 \
  --val_fast_n 1 \
  --val_start_epoch 3 \
  --val_full_freq 50 \
  "${CACHE_ARGS[@]}" \
  --contrast_mod_indices "${CONTRAST_MOD_INDICES}" \
  --p_anisotropic_2d "${P_ANISOTROPIC_2D}" \
  --p_partial_contrast "${P_PARTIAL_CONTRAST}" \
  --p_label_noise "${P_LABEL_NOISE}" \
  "${REALISM_ARGS[@]}"
