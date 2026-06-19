#!/bin/bash
#SBATCH --job-name=ogap_teacher_ddp
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4          # full node: 4x H100 SXM5 (NVLink)
#SBATCH --ntasks-per-node=4             # one task (rank) per GPU
#SBATCH --cpus-per-task=16              # 64 cores / 4 ranks
#SBATCH --mem=498G
#SBATCH --tmp=1000G
#SBATCH --time=24:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=ogap_teacher_ddp_%j.out
# ---------------------------------------------------------------------------
# 4x H100 DistributedDataParallel teacher training for OGAP (Rorqual).
#
# Multi-GPU counterpart of submit_ogap_teacher_rorqual_experimental.sbatch, which
# used a single GPU (--gpus=h100:1) and left 3/4 of a node idle [finding B1]. It
# launches one process per GPU with torchrun; the training entry point uses
# ogap.utils.distributed (init_distributed_from_env / wrap_ddp /
# make_distributed_sampler / is_main_process / reduce_mean). Gradients all-reduce
# over NVLink with the NCCL NVL P2P level set below. ~4x throughput for the
# teacher / student / ablation sweep.
#
# Pair this with GPU-resident physics augmentation (augmentation.physics.enabled:
# true in the v9.1 config) so BOTH model compute AND augmentation run on the
# H100s -- no CPU augmentation bottleneck [GPU-augmentation migration].
# ---------------------------------------------------------------------------
set -euo pipefail

# Match the validated single-GPU script's module + venv (do NOT silence failures
# under set -e: a missing module/venv must abort early, not fail confusingly later).
module load StdEnv/2023 python/3.11.5 cuda/12.6 scipy-stack/2024a
source "${ENV_PATH:-${HOME}/ogap/envs/ogap_env_v2}/bin/activate"

# --- NUMA / threading (match the validated single-GPU script) --------------
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${OMP_NUM_THREADS}
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=${OMP_NUM_THREADS}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OGAP_DISABLE_N4=1

# --- NCCL over NVLink ------------------------------------------------------
export NCCL_P2P_LEVEL=NVL                # intra-node NVLink P2P (matches defaults.yaml)
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

# --- Persistent compile / Triton caches on scratch -------------------------
export TORCHINDUCTOR_CACHE_DIR=${OGAP_SCRATCH:-$SCRATCH}/ogap/torchinductor_cache
export TRITON_CACHE_DIR=${OGAP_SCRATCH:-$SCRATCH}/ogap/triton_cache
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

# --- Manifests (stage to node-local NVMe $SLURM_TMPDIR as in the pipeline) --
: "${OGAP_TRAIN_CSV:?set OGAP_TRAIN_CSV to the training manifest}"
: "${OGAP_VAL_CSV:?set OGAP_VAL_CSV to the validation manifest}"

# --- torchrun: one rank per GPU on this node -------------------------------
# Single-node rendezvous (--standalone). For multi-node, switch to c10d with
# --rdzv_backend=c10d --rdzv_endpoint="$head_node:29500".
MASTER_PORT=${MASTER_PORT:-29500}

srun --cpu-bind=cores --mem-bind=local \
  torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=4 \
    --master_port="${MASTER_PORT}" \
    -m ogap pretrain_teacher \
      --config "${OGAP_CONFIG:-configs/production_v91.yaml}" \
      --train_csv "${OGAP_TRAIN_CSV}" \
      --val_csv "${OGAP_VAL_CSV}" \
      --out_dir "${OGAP_OUT:-$SCRATCH/ogap/teacher_ddp}" \
      "$@"

# DDP IS WIRED into the training loop (ogap/legacy.py pretrain_teacher / train):
# it calls ogap.utils.distributed.init_distributed_from_env() at start, binds each
# rank to its LOCAL_RANK GPU, wraps the teacher with wrap_ddp() (the student syncs
# its multi-module gradients via all_reduce_gradients), shards the metadata-weighted
# sampler per rank with DistributedWeightedSampler (keeps the equity weighting) and
# calls set_epoch(), scales the LR linearly, and gates all checkpoint writes on
# is_main_process(). Activation is automatic from torchrun's WORLD_SIZE - no flag is
# needed; the helpers no-op under a single process, so this script also runs correctly
# with --nproc_per_node=1. The single-process path is CPU-unit-tested; a 1-node 4-GPU
# smoke test is the one remaining cluster validation (multi-GPU cannot run on a dev box).
