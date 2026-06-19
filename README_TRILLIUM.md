# OGAP launch scripts - Trillium (SciNet / Alliance) port

This folder is the **Trillium-adapted launch surface** of `OGAP_v2/`. It contains
the cluster-facing launch + setup scripts plus the `workflow/` utilities the launch
path calls; the Python package (`ogap/`), the legacy source
(`OGAP_source_code_experimental_v9.py`), `configs/`, `experiments/`, and `tests/`
are **unchanged** and are deployed alongside these scripts exactly as on Rorqual.
The `workflow/` utilities are copied **verbatim** from upstream (cluster-agnostic),
except the staging helper's docstring, which was updated to describe the diskless
`$SCRATCH` staging path.

The scripts were ported from the Rorqual versions in `../OGAP_v2/slurm` and
`../OGAP_v2/setup`. Nothing about the model, data, or training math changed -
**only the cluster-specific Slurm directives and the diskless-node scratch
handling.**

---

## 1. Deployment layout (unchanged from Rorqual)

The scripts assume the same on-cluster layout the Rorqual scripts use:
`PROJECT_ROOT/Scripts/` holds the deployed repo, so the launch scripts reference
their siblings as `${PROJECT_ROOT}/Scripts/...`:

```
$HOME/ogap/OGAP_v2/                         ← PROJECT_ROOT
└── Scripts/                                ← the deployed repo (this folder's contents go here)
    ├── OGAP_source_code_experimental_v9.py
    ├── launch_pipeline.py
    ├── ogap/  configs/  experiments/  tests/
    ├── workflow/
    │   ├── stage_ogap_csv_to_tmp.py        ← staging helper (cluster-agnostic; unchanged)
    │   └── launch_ogap_train.sh            ← Trillium variant (points at the Trillium pipeline)
    ├── setup/                              ← Trillium-commented bootstrap + verifier
    └── slurm/                              ← Trillium .sbatch + orchestrator
```

To deploy: copy the contents of `OGAP_v2_trillium/` over the corresponding paths
in your deployed `Scripts/` tree (or rsync the whole repo and overlay this folder).

---

## 2. What changed vs the Rorqual scripts

Every change is a cluster constraint you gave for Trillium; nothing else was touched.

| Trillium fact | Change applied |
|---|---|
| Account `def-rdiaz` | `--account=YOUR_SLURM_ACCOUNT` → `--account=def-rdiaz` (all jobs) |
| `--gpus-per-node=h100:N` | `--gpus=h100:1` → `--gpus-per-node=h100:1`; DDP already used `--gpus-per-node=h100:4` |
| No MIG | post-hoc job's `--gpus=nvidia_h100_80gb_hbm3_2g.20gb:1` (a 2g.20gb MIG slice) → `--gpus-per-node=h100:1` (full H100) |
| 1 GPU = ¼ node (≤24 cores, ≤187 GB) | single-GPU `--mem=200G` → `187G`; `ogap_v9.sbatch` `--cpus-per-task=32` → `24`; all single-GPU core counts already ≤24 |
| Full node = 96 cores + 768 GB | DDP `--cpus-per-task=16` → `24` (4×24 = 96), `--mem=498G` → `748G` (4×187), `OMP` fallback 16 → 24 |
| 24 h walltime cap | `--time=48:00:00`/`72:00:00` → `24:00:00` (see §3) |
| Diskless nodes, stage to `$SCRATCH` | removed all `--tmp=…`; redirected every `$SLURM_TMPDIR` use to a job-private `$SCRATCH` dir (see §4) |
| Modules work as-is | `module load StdEnv/2023 python/3.11.5 cuda/12.6 scipy-stack/2024a` unchanged; env setup scripts unchanged except comments |

Filenames were renamed `…_rorqual_…` → `…_trillium_…`, and every cross-reference
(the orchestrator's four `*_SBATCH` defaults and `launch_ogap_train.sh`'s
`PIPELINE=…`) was updated to match.

### Per-job resource summary (all within the limits you gave)

| Job | GPUs | cores | mem | time |
|---|---|---|---|---|
| teacher / student / export / eval | `h100:1` | 16 | 187 G | 24 h |
| `ogap_v9.sbatch` (legacy single-job) | `h100:1` | 24 | 187 G | 24 h |
| post-hoc | `h100:1` (was MIG) | 12 | 96 G | 24 h |
| teacher DDP | `h100:4` (full node) | 24/rank ×4 = 96 | 748 G | 24 h |

> If the scheduler ever rejects `--mem=187G` (some partitions schedule a hair
> under the nominal per-GPU share), drop it to `--mem=185G`. 748 G = 4 × 187 for
> the full-node DDP job; `--mem=0` (all node memory) is an equivalent alternative.

---

## 3. 24-hour walltime - important for teacher/student

Trillium caps every job at 24 h, but the teacher and student each train for
**1000 epochs**, which will not finish in one window. Two consequences:

1. **Run long trainings as a sequence of 24 h jobs, not one.** Resubmit the
   teacher job until `Results/teacher/best_teacher.pth` is final. The trainer
   writes checkpoints into `OUT_DIR`; a resubmitted job picks up from the latest
   checkpoint there (confirm resume behaviour on your first restart - if a
   resubmit restarts from epoch 0, set `--epochs` to a value that fits 24 h, or
   pass `FORCE_RESUME_PARTIAL=1`).

2. **Do not submit the full dependency chain in one shot for a fresh teacher.**
   The orchestrator chains `teacher → student → export → eval` with
   `--dependency=afterok`. If the teacher hits walltime it exits non-zero
   (TIMEOUT) and the student's `afterok` dependency is **cancelled**. Recommended
   pattern:

   ```bash
   # Phase 1 - train the teacher to completion (resubmit as needed):
   RUN_STUDENT=0 RUN_EXPORT=0 RUN_EVAL=0 bash slurm/submit_ogap_trillium_pipeline_experimental.sh

   # Phase 2 - once best_teacher.pth is final, chain the rest:
   RUN_TEACHER=0 TEACHER_CKPT="$HOME/ogap/OGAP_v2/Results/teacher/best_teacher.pth" \
     bash slurm/submit_ogap_trillium_pipeline_experimental.sh
   ```

   `RUN_TEACHER/RUN_STUDENT/RUN_EXPORT/RUN_EVAL` and the `*_CKPT` overrides are
   the same knobs the Rorqual orchestrator already exposes - nothing new to learn.

The export, eval, and post-hoc jobs are short and comfortably fit 24 h.

---

## 4. Diskless nodes - staging to `$SCRATCH`

Trillium compute nodes have **no node-local disk**, so there is no `$SLURM_TMPDIR`
NVMe to request with `--tmp` and stage onto (the Rorqual scripts requested
`--tmp=1000G` and staged there). The ported scripts instead create a job-private
directory under `$SCRATCH` and remove it on exit:

```bash
OGAP_JOB_TMP="${OGAP_JOB_TMP:-${SCRATCH}/ogap_jobtmp/<jobname>_<jobid>}"
trap 'rm -rf "${OGAP_JOB_TMP}"' EXIT
```

Everything that used `$SLURM_TMPDIR` on Rorqual - `TMPDIR`, `PYTHONPYCACHEPREFIX`,
the npy/patch cache roots, and the CSV staging root - now lives under
`OGAP_JOB_TMP`. The CSV staging path (`OGAP_STAGE_DATA_TO_TMP=1`, default on) calls
the unchanged `workflow/stage_ogap_csv_to_tmp.py` with `--stage-root` pointed at
`$SCRATCH`; the gzip→nii decompression pass it does (`OGAP_STAGE_DECOMPRESS_GZ=1`)
still pays off across 1000 epochs by removing per-read `gunzip` from every
DataLoader worker. The Torch/Inductor and Triton caches stay on a **persistent**
`$SCRATCH` path (`$SCRATCH/.ogap_torchinductor_cache`), not under the per-job dir,
so compiled kernels survive across jobs.

`ogap_v9.sbatch` (the legacy single-job pipeline that `rsync`s a pre-extracted
BraTS tree and pre-decompresses it) was redirected the same way: it stages into
`$SCRATCH/ogap_v9_stage_<jobid>` instead of `$SLURM_TMPDIR`.

**Scratch-usage caveat:** staging (especially with gz→nii decompression, which
inflates volumes ~3-5×) copies data into `$SCRATCH`. Each job cleans up its own
staged dir on exit via the `trap`, so dirs do not accumulate across jobs, but a
running job needs scratch headroom for its staged copy. If your data already
lives on `$SCRATCH` and scratch is tight, set `OGAP_STAGE_DATA_TO_TMP=0` to read
the originals directly (you lose the one-time decompression speedup).

---

## 5. Quickstart (Trillium)

```bash
# 1. Build the environment (modules are unchanged; CC/Alliance wheelhouse on CVMFS)
bash setup/setup_ogap_env_v2.sh            # add --with-mamba for SegMamba deps
python setup/verify_ogap_env.py            # green/red status per dependency

# 2. (optional) dedicated SegMamba env (separate torch-2.5.1 venv)
bash setup/setup_ogap_env_mamba.sh

# 3. End-to-end via the dispatcher (heavy UNet3D dense teacher path)
workflow/launch_ogap_train.sh unet         # or: mamba (needs the mamba env)

# - or - submit individual jobs directly:
sbatch slurm/submit_ogap_teacher_trillium_experimental.sbatch
sbatch slurm/submit_ogap_teacher_trillium_ddp.sbatch          # 4×H100 full node
```

`launch_pipeline.py` (the `python -m ogap` stage launcher) is cluster-agnostic
and is included unchanged from upstream.

---

## 6. File inventory

```
slurm/
  submit_ogap_trillium_pipeline_experimental.sh   orchestrator (teacher→student→export→eval)
  submit_ogap_teacher_trillium_experimental.sbatch
  submit_ogap_teacher_trillium_ddp.sbatch         4×H100 full-node DDP teacher
  submit_ogap_student_trillium_experimental.sbatch
  submit_ogap_export_int8_trillium_experimental.sbatch
  submit_utsw_eval_tta_trillium_experimental.sbatch
  submit_ogap_posthoc_full_trillium.sbatch
  ogap_v9.sbatch                                  legacy single-job pipeline
setup/
  setup_ogap_env_v2.sh   setup_ogap_env_mamba.sh   verify_ogap_env.py
workflow/
  launch_ogap_train.sh                            single-entry train dispatcher (unet|mamba)
  stage_ogap_csv_to_tmp.py                        CSV->$SCRATCH staging helper (called by the sbatch jobs)
  build_split_manifests.py                        train/internal-val/external-val CSV builder (verbatim)
  eval_hic_vs_lmic.py                             HIC-vs-LMIC external-validation stats (verbatim)
launch_pipeline.py                                python -m ogap stage launcher (unchanged)
```
