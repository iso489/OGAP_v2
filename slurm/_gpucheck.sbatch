#!/bin/bash
#SBATCH --account=def-rdiaz
#SBATCH --time=00:08:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gpus-per-node=4
#SBATCH --job-name=gpucheck
#SBATCH --output=/scratch/ilyaso/ogap/logs/gpucheck_%j.out
#SBATCH --error=/scratch/ilyaso/ogap/logs/gpucheck_%j.err
set -uo pipefail
module --force purge
module load StdEnv/2023 python/3.11.5 cuda/12.6 scipy-stack/2024a
source "$HOME/ogap/envs/ogap_env_v2/bin/activate"
export NCCL_DEBUG=WARN NCCL_P2P_LEVEL=NVL
echo "=== batch ctx: CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-<unset>}' ==="
nvidia-smi -L 2>&1 | head -6
echo "=== [A] BARE python (no srun) - reproduces the preflight failure? ==="
python -c "import torch; print('  bare: cuda_avail=', torch.cuda.is_available(), 'count=', torch.cuda.device_count())"
echo "=== [B] srun python (1 task) - does srun bind the GPU? ==="
srun --ntasks=1 python -c "import torch,os; print('  srun: cuda_avail=', torch.cuda.is_available(), 'count=', torch.cuda.device_count(), 'CVD=', os.environ.get('CUDA_VISIBLE_DEVICES'))"
echo "=== [C] srun torchrun 4 ranks - DDP per-rank binding + NCCL all_reduce ==="
srun --cpu-bind=cores --mem-bind=local torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  /project/def-rdiaz/ilyaso/OGAP/Scripts/slurm/_ddp_probe.py
echo "=== gpucheck done (rc=$?) ==="
