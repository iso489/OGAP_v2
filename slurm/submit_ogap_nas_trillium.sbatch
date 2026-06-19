#!/bin/bash
# Trillium hardware-aware NAS (offline architecture search) for OGAP.
# Independent of training - writes an advisory Pareto front to JSON (no stage
# consumes it automatically). Single H100, short walltime.
#SBATCH --account=def-rdiaz
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=1
#SBATCH --job-name=ogap_nas
#SBATCH --output=/scratch/ilyaso/ogap/logs/ogap_nas_%j.out
#SBATCH --error=/scratch/ilyaso/ogap/logs/ogap_nas_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/project/def-rdiaz/ilyaso/OGAP}"
ENV_PATH="${ENV_PATH:-${HOME}/ogap/envs/ogap_env_v2}"
SCRATCH_ROOT="${SCRATCH:-${HOME}/scratch}"
OUT="${OUT:-${SCRATCH_ROOT}/ogap/Results/nas/pareto_front.json}"
NAS_STRATEGY="${NAS_STRATEGY:-evolution}"
NAS_ITERS="${NAS_ITERS:-50}"
NAS_IN_CHANNELS="${NAS_IN_CHANNELS:-11}"   # 4 modalities + 4 availability + 3 tissue-prior channels
NUM_CLASSES="${NUM_CLASSES:-4}"
NAS_PATCH="${NAS_PATCH:-32,32,32}"
MAX_INT8_MB="${MAX_INT8_MB:-5.0}"
MAX_LATENCY_MS="${MAX_LATENCY_MS:-800.0}"
# Accuracy-proxy batch (jacob_cov/NASWOT). MUST be >1: at B=1 the Jacobian
# correlation matrix is the constant [[1]] and the proxy is dead (~-1.0 for every
# architecture), so the search degrades to a pure params/latency sweep. 16-32 is a
# good 3D compromise; cost/latency stay B=1 (single-volume deployment) regardless.
NAS_PROXY_BATCH="${NAS_PROXY_BATCH:-16}"

module --force purge
module load StdEnv/2023 python/3.11.5 cuda/12.6 scipy-stack/2024a
source "${ENV_PATH}/bin/activate"

# /project is READ-ONLY on Trillium compute nodes: run from scratch, redirect caches.
export PYTHONPATH="${PROJECT_ROOT}/Scripts${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR="${SCRATCH_ROOT}/ogap/jobtmp/nas_${SLURM_JOB_ID}"
export PYTHONPYCACHEPREFIX="${TMPDIR}/pycache"
export TORCHINDUCTOR_CACHE_DIR="${SCRATCH_ROOT}/.ogap_torchinductor_cache"
export TRITON_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR}/triton_cache"
mkdir -p "${TMPDIR}" "${PYTHONPYCACHEPREFIX}" "$(dirname "${OUT}")" \
         "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${SCRATCH_ROOT}/ogap/logs"
cd "${TMPDIR}"

echo "[nas] strategy=${NAS_STRATEGY} n=${NAS_ITERS} patch=${NAS_PATCH} -> ${OUT}"
srun --gpu-bind=closest python -m ogap nas-search \
  --strategy "${NAS_STRATEGY}" \
  --n "${NAS_ITERS}" \
  --in-channels "${NAS_IN_CHANNELS}" \
  --num-classes "${NUM_CLASSES}" \
  --patch "${NAS_PATCH}" \
  --max-int8-mb "${MAX_INT8_MB}" \
  --max-latency-ms "${MAX_LATENCY_MS}" \
  --proxy-batch "${NAS_PROXY_BATCH}" \
  --out "${OUT}"
echo "[nas] done -> ${OUT}"
