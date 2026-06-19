#!/usr/bin/env bash
# OGAP single-entry training dispatcher (2026 audit, Step 5).
#
# Usage:
#   ./launch_ogap_train.sh unet        # UNet3D dense heavy teacher
#   ./launch_ogap_train.sh mamba       # SegMamba dense heavy teacher
#   ./launch_ogap_train.sh unet --no-n4   # smoke-test with N4 disabled
#
# What this script does:
#   1. Activates the right venv (ogap_env_v2 vs ogap_env_v2_mamba).
#   2. Sets the optimized env-knob playbook from the 2026 audit:
#        - heavy teacher (base=64 dense) - student stays MedNeXt-S base=16
#          for INT8/CPU deployment.
#        - teacher-side MixStyle3D + 0.15 modality dropout (cross-scanner +
#          missing-modality robustness, identity at eval).
#        - npy cache ON, patch cache OFF (preserves crop diversity), prefetch=2
#          with inflight cap = 64 (16 workers x bs=2 x pf=2).
#        - N4 thread cap = 1 + reduced iteration schedule = no first-epoch hang.
#   3. Dispatches to the right pipeline orchestrator
#      (standard or Mamba-force-overridden).
set -euo pipefail

usage() {
  cat >&2 <<EOF
usage: $0 {unet|mamba} [--no-n4]
  unet      Standard (non-Mamba) UNet3D dense teacher pipeline.
  mamba     SegMamba dense teacher pipeline.
  --no-n4   Optional: smoke-test mode with N4 disabled (foreground z-score only).
EOF
  exit 1
}

[[ $# -lt 1 ]] && usage
MODE="$1"; shift
DISABLE_N4=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-n4) DISABLE_N4=1 ;;
    -h|--help) usage ;;
    *) echo "unknown arg: $1" >&2; usage ;;
  esac
  shift
done

PROJECT_ROOT="${PROJECT_ROOT:-/project/def-rdiaz/ilyaso/OGAP}"
export PROJECT_ROOT
export DATA_ROOT="${DATA_ROOT:-/project/def-rdiaz/ilyaso/Datasets}"
cd "${PROJECT_ROOT}/Scripts"

# ── Hygiene: prevent leaks from prior shell sessions ──────────────────
unset VAL_CSV

case "${MODE}" in
  unet|standard)
    source ${HOME}/ogap/envs/ogap_env_v2/bin/activate
    unset OGAP_DISABLE_TORCH_COMPILE
    export ENV_PATH=${HOME}/ogap/envs/ogap_env_v2
    export EXPECTED_TORCH_PUBLIC_VERSION=2.6.0
    export TEACHER_ARCH=unet3d
    export AUTO_RESOURCES_SAFETY_TEACHER=0.90
    export TEACHER_OUT_DIR="${SCRATCH:-${HOME}/scratch}/ogap/Results/teacher_unet_dense"
    export STUDENT_OUT_DIR="${SCRATCH:-${HOME}/scratch}/ogap/Results/student_unet_dense"
    export EVAL_OUT_DIR="${SCRATCH:-${HOME}/scratch}/ogap/Results/utsw_full_evaluation_tta_unet_dense"
    PIPELINE=slurm/submit_ogap_trillium_pipeline_experimental.sh
    ;;
  mamba|segmamba)
    # SegMamba arm. Requires the dedicated torch-2.5.1 venv built by
    # setup/setup_ogap_env_mamba.sh (mamba-ssm is pinned to torch ~2.5). The
    # vars below are forwarded to the teacher/student jobs by the standard
    # pipeline's slurm_export_spec, so the full Mamba environment is set here
    # (this replaces the former submit_ogap_mamba_rorqual_pipeline_experimental.sh).
    MAMBA_ENV="${MAMBA_ENV_PATH:-${HOME}/ogap/envs/ogap_env_v2_mamba}"
    source "${MAMBA_ENV}/bin/activate"
    export ENV_PATH="${MAMBA_ENV}"
    export EXPECTED_TORCH_PUBLIC_VERSION="${MAMBA_EXPECTED_TORCH_PUBLIC_VERSION:-2.5.1}"
    export OGAP_DISABLE_TORCH_COMPILE=1            # Mamba env Triton/Inductor: run eager
    export TEACHER_ARCH=segmamba
    export AUTO_RESOURCES_SAFETY_TEACHER=0.75
    export TEACHER_OUT_DIR="${SCRATCH:-${HOME}/scratch}/ogap/Results/teacher_mamba"
    export STUDENT_OUT_DIR="${SCRATCH:-${HOME}/scratch}/ogap/Results/student_mamba"
    export EVAL_OUT_DIR="${SCRATCH:-${HOME}/scratch}/ogap/Results/utsw_full_evaluation_tta_mamba"
    PIPELINE=slurm/submit_ogap_trillium_pipeline_experimental.sh
    ;;
  *) usage ;;
esac

# ── Shared model + training config ────────────────────────────────────
export TEACHER_BLOCK_STYLE=dense
export TEACHER_BASE=64           # heavy KD teacher (only student is INT8-exported)
export TEACHER_BATCH_SIZE=1
export STUDENT_BASE=16           # CPU-deployable target
export STUDENT_BATCH_SIZE=4
export GRAD_ACCUM_STEPS=1
export AUTO_RESOURCES=1
export AUTO_RESOURCES_SAFETY_STUDENT=0.75

# ── Full experimental stack (2026): DDP teacher (4xH100) + NAS + OOD ──────────
# Override any to 0 to revert (e.g. USE_DDP_TEACHER=0 for the validated single-GPU
# teacher). NAS is independent; OOD runs after the student checkpoint exists.
export USE_DDP_TEACHER="${USE_DDP_TEACHER:-1}"
export TEACHER_NPROC="${TEACHER_NPROC:-4}"
export RUN_NAS="${RUN_NAS:-1}"
export RUN_OOD="${RUN_OOD:-1}"

# ── 2026 audit: teacher robustness knobs (Gap A + Gap B) ──────────────
export TEACHER_FEATURE_DR=mixstyle       # cross-scanner invariance for KD logits
export TEACHER_FEATURE_DR_P=0.5
export TEACHER_FEATURE_DR_ALPHA=0.1
export TEACHER_MODALITY_DROPOUT_P=0.15   # KD logits encode missing-modality behaviour
export TEACHER_MODALITY_DROPOUT_MAX_DROP=2

# ── Staging (Trillium is diskless: copy to a $SCRATCH job dir, decompress) ──
export OGAP_STAGE_DATA_TO_TMP=1
export OGAP_STAGE_DECOMPRESS_GZ=1
export OGAP_STAGE_WORKERS=32

# ── Caches: npy ON (free speed), patch OFF (preserves crop diversity) ─
export OGAP_USE_NPY_CACHE=1
export OGAP_USE_PATCH_CACHE=0
export OGAP_DROP_NPY_FILE_CACHE=0

# ── DataLoader: prefetch=2 with matching inflight ceiling ─────────────
export NUM_WORKERS=16
export VAL_NUM_WORKERS=4
export EVAL_NUM_WORKERS=8
export METRIC_WORKERS=4
export EVAL_CONFIG_BATCH_SIZE=8
export OGAP_PREFETCH_FACTOR=2
export OGAP_MAX_TRAIN_WORKERS=16
export OGAP_MAX_EVAL_WORKERS=8
export OGAP_MAX_PREFETCH_FACTOR=2
export OGAP_MAX_TRAIN_INFLIGHT_CASES=64   # admits prefetch=2 (16w x bs=2 x pf=2 = 64)
export OGAP_MAX_EVAL_INFLIGHT_CASES=8

# ── N4 bias correction (thread cap + reduced iteration schedule) ──────
export OGAP_N4_THREADS=1
export OGAP_N4_ITERATIONS=50,50,30,20
if [[ "${DISABLE_N4}" == "1" ]]; then
  export OGAP_DISABLE_N4=1
  echo "[launch] OGAP_DISABLE_N4=1 (smoke-test mode; N4 skipped)"
else
  unset OGAP_DISABLE_N4 || true
fi

# ── Architectural robustness defaults (LMIC physics pack) ─────────────
export LMIC_FIELD_STRENGTH_PRIOR=1
export LABEL_NOISE_MODE=boundary_band
export LABEL_NOISE_BAND_RADIUS=2
export LABEL_NOISE_BAND_FLIP_PROB=0.30
export P_FIELD_CONTRAST_WARP=0.30
export P_FOURIER_AMPLITUDE_MIX=0.30
export AFA_PROB=0.30
export P_RECEIVE_COIL_INHOMOGENEITY=0.30
export P_OFF_RESONANCE_VOID=0.30
export NOISE_CALIBRATION=aja_fernandez
export CARVEMIX_PROB=0.30
export CARVEMIX_DILATION=5

# ── Student-side feature DR + V-REx (existing knobs) ──────────────────
export LAMBDA_VREX=0.10
export FEATURE_DR=mixstyle
export FEATURE_DR_P=0.5
export FEATURE_DR_ALPHA=0.1
export KEEP_MIN_MODALITIES=1
export KEEP_MAX_MODALITIES=4

# ── Evaluation defaults ───────────────────────────────────────────────
export FAILURE_MODE_METRIC=pp_all_dice_brats_mean
export FAILURE_MODE_QUANTILE=0.25
export FAILURE_MODE_MIN_GROUP_N=10
export LESION_WISE=1

echo "[launch] mode=${MODE}  teacher=${TEACHER_ARCH}/base=${TEACHER_BASE}/${TEACHER_BLOCK_STYLE}"
echo "[launch] teacher_feature_dr=${TEACHER_FEATURE_DR} teacher_modality_dropout_p=${TEACHER_MODALITY_DROPOUT_P}"
echo "[launch] dispatching to ${PIPELINE}"
bash "${PIPELINE}"
