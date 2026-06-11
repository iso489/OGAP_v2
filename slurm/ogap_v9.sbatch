#!/bin/bash
#SBATCH --job-name=ogap_v9
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=32
#SBATCH --hint=nomultithread
#SBATCH --distribution=block:block
#SBATCH --mem=200G
#SBATCH --tmp=500G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
##SBATCH --mail-user=YOUR_EMAIL@example.com  # set and uncomment before submitting
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

# ── Environment ───────────────────────────────────────────────────────────────
module purge
module load StdEnv/2023 python/3.11.5 cuda/12.6 scipy-stack/2024a
source ~/ogap/envs/ogap_env/bin/activate

# ── CPU / runtime tuning ─────────────────────────────────────────────────────
# Single-thread native libraries per process avoids worker oversubscription.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
export KMP_BLOCKTIME=1
export KMP_AFFINITY="granularity=fine,compact,1,0"
export KMP_SETTINGS=0
export MALLOC_ARENA_MAX=2
export PYTHONUNBUFFERED=1

# ── CUDA / H100 tuning ────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_ALLOW_TF32=1
export CUDNN_ALLOW_TF32=1
export CUDNN_BENCHMARK=1
export CUDA_LAUNCH_BLOCKING=0

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT="${HOME}/ogap/OGAP_project"
SCRIPT="$PROJECT/Scripts/OGAP_source_code_v9.py"

TRAIN_CSV="$PROJECT/Scripts/train.csv"
VAL_CSV="$PROJECT/Scripts/val.csv"
ALL_CSV="$PROJECT/Scripts/all.csv"

RESULTS="$PROJECT/Results"
mkdir -p "$RESULTS/teacher" "$RESULTS/student" "$RESULTS/export" \
         "$RESULTS/eval"    "$RESULTS/ablation"

# ── srun helper ───────────────────────────────────────────────────────────────
# Default to local memory binding for lower NUMA latency. If your site rejects
# this, submit with USE_MEM_BIND_LOCAL=0.
# Example: sbatch --export=ALL,USE_MEM_BIND_LOCAL=0,NUM_WORKERS_OVERRIDE=24 ogap_v9.sbatch
run() {
  local mem_bind_flags=()
  if [[ "${USE_MEM_BIND_LOCAL:-1}" == "1" ]]; then
    mem_bind_flags=(--mem-bind=local)
  fi

  srun --unbuffered --cpu-bind=cores --hint=nomultithread "${mem_bind_flags[@]}" \
       -c "${SLURM_CPUS_PER_TASK}" "$@"
}

# ── Worker counts ─────────────────────────────────────────────────────────────
# Conservative default for heavy 3D CPU transforms.
# Override quickly without editing script:
#   sbatch --export=ALL,NUM_WORKERS_OVERRIDE=20,DECOMPRESS_JOBS_OVERRIDE=20 ogap_v9.sbatch
NUM_WORKERS="${NUM_WORKERS_OVERRIDE:-20}"
DECOMPRESS_JOBS="${DECOMPRESS_JOBS_OVERRIDE:-20}"

if (( NUM_WORKERS < 1 || NUM_WORKERS > SLURM_CPUS_PER_TASK )); then
  echo "ERROR: NUM_WORKERS=${NUM_WORKERS} must be in [1, ${SLURM_CPUS_PER_TASK}]"
  exit 1
fi

if (( DECOMPRESS_JOBS < 1 || DECOMPRESS_JOBS > SLURM_CPUS_PER_TASK )); then
  echo "ERROR: DECOMPRESS_JOBS=${DECOMPRESS_JOBS} must be in [1, ${SLURM_CPUS_PER_TASK}]"
  exit 1
fi

echo "============================================="
echo "OGAP Full Pipeline v9 — $(date)"
echo "JobID:  ${SLURM_JOB_ID}"
echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader)"
echo "CPUs:   ${SLURM_CPUS_PER_TASK} physical (workers=${NUM_WORKERS}, OMP=${OMP_NUM_THREADS})"
echo "RAM:    ${SLURM_MEM_PER_NODE:-<unset>} MB"
echo "Torch:  $(python -c 'import torch; print(torch.__version__, "| CUDA:", torch.version.cuda)')"
echo "Tmpdir: ${SLURM_TMPDIR:-<unset>}"
echo "============================================="

: "${SLURM_TMPDIR:?SLURM_TMPDIR not set — job is missing --tmp flag}"

# ── Stage data to local NVMe ──────────────────────────────────────────────────
RAW_BASE="$PROJECT/Training_Datasets/Raw_Data"
BRATS_SRC="${HOME}/scratch/brats_extracted"

if [[ ! -d "$BRATS_SRC" ]]; then
  echo "ERROR: Pre-extracted data not found at $BRATS_SRC"
  exit 1
fi

echo ""
echo ">>> Staging data to local NVMe ..."
rsync -a --info=progress2 "$BRATS_SRC/" "$SLURM_TMPDIR/"
echo "    Done: $(date)"

# ── Pre-decompress .nii.gz -> .nii ───────────────────────────────────────────
echo ""
echo ">>> Pre-decompressing .nii.gz (${DECOMPRESS_JOBS} parallel jobs) ..."
NII_GZ_COUNT=$(find "$SLURM_TMPDIR" -name "*.nii.gz" | wc -l)
echo "    Found: ${NII_GZ_COUNT} files"
if [[ "$NII_GZ_COUNT" -gt 0 ]]; then
  find "$SLURM_TMPDIR" -name "*.nii.gz" -print0 \
    | xargs -0 -P "${DECOMPRESS_JOBS}" -I{} bash -c 'gunzip -f "$1"' _ {}
fi
echo "    Done: $(date)"

# ── Rewrite CSV paths -> NVMe paths ───────────────────────────────────────────
TRAIN_CSV_LOCAL="$SLURM_TMPDIR/train_local.csv"
VAL_CSV_LOCAL="$SLURM_TMPDIR/val_local.csv"
ALL_CSV_LOCAL="$SLURM_TMPDIR/all_local.csv"

sed "s|$RAW_BASE/|$SLURM_TMPDIR/|g; s|\.nii\.gz|\.nii|g" "$TRAIN_CSV" > "$TRAIN_CSV_LOCAL"
sed "s|$RAW_BASE/|$SLURM_TMPDIR/|g; s|\.nii\.gz|\.nii|g" "$VAL_CSV"   > "$VAL_CSV_LOCAL"
sed "s|$RAW_BASE/|$SLURM_TMPDIR/|g; s|\.nii\.gz|\.nii|g" "$ALL_CSV"   > "$ALL_CSV_LOCAL"

TRAIN_CSV="$TRAIN_CSV_LOCAL"
VAL_CSV="$VAL_CSV_LOCAL"
ALL_CSV="$ALL_CSV_LOCAL"

FIRST_NII=$(awk -F',' 'NR==2{print $2}' "$TRAIN_CSV" | tr -d '"')
if [[ ! -f "$FIRST_NII" ]]; then
  echo "ERROR: First NIfTI not found after rewrite: $FIRST_NII"
  exit 1
fi

# ── Step 1: Pre-train teacher ─────────────────────────────────────────────────
echo ""
echo ">>> STEP 1/5: Pre-training teacher (300 epochs) ..."

run python "$SCRIPT" pretrain_teacher \
  --train_csv      "$TRAIN_CSV" \
  --val_csv        "$VAL_CSV" \
  --out_dir        "$RESULTS/teacher" \
  --modalities     t1n,t1c,t2w,t2f \
  --num_classes    4 \
  --patch_size     128,128,128 \
  --batch_size     16 \
  --epochs         300 \
  --lr             4e-4 \
  --num_workers    "$NUM_WORKERS" \
  --teacher_base   32 \
  --warmup_epochs  10 \
  --hd95_freq      50 \
  --val_fast_n     1 \
  --val_full_freq  50 \
  --seed           42 \
  --device         cuda

# ── Step 2: Train student ─────────────────────────────────────────────────────
echo ""
echo ">>> STEP 2/5: Training student with KD (1000 epochs) ..."

run python "$SCRIPT" train \
  --train_csv          "$TRAIN_CSV" \
  --val_csv            "$VAL_CSV" \
  --out_dir            "$RESULTS/student" \
  --teacher_ckpt       "$RESULTS/teacher/best_teacher.pth" \
  --modalities         t1n,t1c,t2w,t2f \
  --num_classes        4 \
  --patch_size         128,128,128 \
  --batch_size         16 \
  --epochs             1000 \
  --lr                 8e-4 \
  --num_workers        "$NUM_WORKERS" \
  --teacher_base       32 \
  --student_base       16 \
  --lambda_kd          0.35 \
  --lambda_feat        0.1 \
  --lambda_ds          0.3 \
  --holder_alpha       1.5 \
  --confidence_gate    0.7 \
  --curriculum_ramp    0.30 \
  --warmup_epochs      10 \
  --hd95_freq          50 \
  --val_fast_n         1 \
  --val_full_freq      50 \
  --seed               42 \
  --device             cuda

# ── Step 3: Export ONNX + INT8 ───────────────────────────────────────────────
echo ""
echo ">>> STEP 3/5: Exporting to ONNX + INT8 ..."

run python "$SCRIPT" export \
  --ckpt             "$RESULTS/student/best_student.pth" \
  --out_path         "$RESULTS/export/student.onnx" \
  --modalities       t1n,t1c,t2w,t2f \
  --num_classes      4 \
  --patch_size       128,128,128 \
  --student_base     16 \
  --calibration_csv  "$ALL_CSV"

# ── Step 4: Full evaluation ───────────────────────────────────────────────────
echo ""
echo ">>> STEP 4/5: Full evaluation ..."

run python "$SCRIPT" evaluate \
  --ckpt         "$RESULTS/student/best_student.pth" \
  --val_csv      "$VAL_CSV" \
  --out_dir      "$RESULTS/eval" \
  --modalities   t1n,t1c,t2w,t2f \
  --num_classes  4 \
  --patch_size   128,128,128 \
  --student_base 16 \
  --device       cuda

# ── Step 5: Ablation study ───────────────────────────────────────────────────
echo ""
echo ">>> STEP 5/5: Ablation study ..."

run python "$SCRIPT" ablation \
  --train_csv    "$TRAIN_CSV" \
  --val_csv      "$VAL_CSV" \
  --out_dir      "$RESULTS/ablation" \
  --teacher_ckpt "$RESULTS/teacher/best_teacher.pth" \
  --modalities   t1n,t1c,t2w,t2f \
  --num_classes  4 \
  --patch_size   128,128,128 \
  --batch_size   16 \
  --epochs       300 \
  --seed         42 \
  --device       cuda

echo ""
echo "============================================="
echo "OGAP v9 Pipeline COMPLETE — $(date)"
echo "============================================="
ls -lh "$RESULTS/teacher/best_teacher.pth"             2>/dev/null && echo "  ✓ Teacher checkpoint"
ls -lh "$RESULTS/student/best_student.pth"             2>/dev/null && echo "  ✓ Student checkpoint"
ls -lh "$RESULTS/export/student.onnx"                  2>/dev/null && echo "  ✓ ONNX model"
ls -lh "$RESULTS/export/student_int8_static.onnx"      2>/dev/null && echo "  ✓ INT8 static model"
ls -lh "$RESULTS/eval/full_evaluation_results.json"    2>/dev/null && echo "  ✓ Full evaluation"
ls -lh "$RESULTS/eval/bootstrap_ci_results.json"       2>/dev/null && echo "  ✓ Bootstrap CIs"
ls -lh "$RESULTS/eval/wilcoxon_results.json"           2>/dev/null && echo "  ✓ Wilcoxon tests"
