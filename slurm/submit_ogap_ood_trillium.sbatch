#!/bin/bash
# Trillium Mahalanobis OOD fitting for the OGAP student (Lee et al. 2018).
# Fits the per-class Gaussian on in-distribution features (internal val), and if
# an OOD CSV is given, reports ID-vs-OOD AUROC. Runs AFTER the student exists.
#SBATCH --account=def-rdiaz
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=1
#SBATCH --job-name=ogap_ood
#SBATCH --output=/scratch/ilyaso/ogap/logs/ogap_ood_%j.out
#SBATCH --error=/scratch/ilyaso/ogap/logs/ogap_ood_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/project/def-rdiaz/ilyaso/OGAP}"
ENV_PATH="${ENV_PATH:-${HOME}/ogap/envs/ogap_env_v2}"
SCRATCH_ROOT="${SCRATCH:-${HOME}/scratch}"
STUDENT_OUT_DIR="${STUDENT_OUT_DIR:-${SCRATCH_ROOT}/ogap/Results/student_unet_dense}"
CKPT="${CKPT:-${STUDENT_OUT_DIR}/best_student.pth}"
ID_CSV="${ID_CSV:-${PROJECT_ROOT}/Scripts/val.csv}"
# LMIC cohort (BraTS-Africa) as the OOD probe vs internal-val ID. Set OOD_CSV="" to skip AUROC.
OOD_CSV="${OOD_CSV:-${PROJECT_ROOT}/Scripts/external_brats_africa.csv}"
OUT_DIR="${OUT_DIR:-${SCRATCH_ROOT}/ogap/Results/ood}"
STUDENT_BASE="${STUDENT_BASE:-16}"
NUM_CLASSES="${NUM_CLASSES:-4}"
OGAP_DISABLE_N4="${OGAP_DISABLE_N4:-1}"

module --force purge
module load StdEnv/2023 python/3.11.5 cuda/12.6 scipy-stack/2024a
source "${ENV_PATH}/bin/activate"

export OGAP_DISABLE_N4
export PYTHONPATH="${PROJECT_ROOT}/Scripts${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR="${SCRATCH_ROOT}/ogap/jobtmp/ood_${SLURM_JOB_ID}"
export PYTHONPYCACHEPREFIX="${TMPDIR}/pycache"
export TORCHINDUCTOR_CACHE_DIR="${SCRATCH_ROOT}/.ogap_torchinductor_cache"
export TRITON_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR}/triton_cache"
mkdir -p "${TMPDIR}" "${PYTHONPYCACHEPREFIX}" "${OUT_DIR}" \
         "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${SCRATCH_ROOT}/ogap/logs"
cd "${TMPDIR}"

for p in "CKPT=${CKPT}" "ID_CSV=${ID_CSV}"; do
  v="${p#*=}"; [[ -e "${v}" ]] || { echo "[ERROR][ood] missing $(echo "$p"|cut -d= -f1): ${v}" >&2; exit 1; }
done

OOD_ARGS=(--ckpt "${CKPT}" --id_csv "${ID_CSV}" --modalities t1n,t1c,t2w,t2f
          --num_classes "${NUM_CLASSES}" --patch_size 128,160,112
          --student_base "${STUDENT_BASE}" --device cuda --out_dir "${OUT_DIR}")
if [[ -n "${OOD_CSV}" && -e "${OOD_CSV}" ]]; then
  OOD_ARGS+=(--ood_csv "${OOD_CSV}")
  echo "[ood] ID=${ID_CSV}  OOD=${OOD_CSV}"
else
  echo "[ood] ID=${ID_CSV}  (no OOD CSV -> fit only, no AUROC)"
fi

srun --gpu-bind=closest python -m ogap mahalanobis_ood "${OOD_ARGS[@]}"
echo "[ood] done -> ${OUT_DIR}"
