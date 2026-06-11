# OGAP — Open Glioma Analysis Pipeline (v9.1)

Brain-tumour segmentation for **low-field / LMIC MRI** (BraTS-Africa, 64 mT Hyperfine
Swoop) with teacher→student knowledge distillation and INT8 edge deployment. The
codebase was refactored from a single 16k-line script
(`OGAP_source_code_experimental_v9.py`) into the tested `ogap/` package, and the
supporting scripts are grouped by role for clarity.

The refactor is a **strangler-fig wrap**: the legacy core lives as `ogap/legacy.py` (the
extracted monolith, with audited correctness fixes applied symmetrically to it and the
package, all default-OFF), and every new capability is opt-in and feature-flagged off by
default. Backward compatibility is guaranteed by **state-dict-key parity tests**, not by
byte-identity of the source. Every module ships with CPU unit tests (`pytest tests/` —
**296 passing, 1 skipped**; a further 5 sample-data integration tests run when
`OGAP_SAMPLE_DATA` points at a local BraTS case).

---

## 1. Directory layout

```
OGAP_v2/   (deploys as Scripts/ on Rorqual)
├── OGAP_source_code_experimental_v9.py   ← thin shim → ogap.legacy.main (back-compat CLI)
├── launch_pipeline.py                    ← v9.1 production launcher (chains python -m ogap subcommands)
├── ogap_compliance.py                    ← reporting-checklist utilities (CLAIM / TRIPOD+AI / SAGER / NIHMS)
├── requirements.txt
│
├── ogap/                ← the package (numerics · models · nas · losses · ood ·
│                          longitudinal · augmentations · evaluation · inference · config)
│   └── legacy.py        ← extracted 16k-line core + audited default-OFF fixes
├── configs/             ← baseline_v91.yaml (validated default) · production_v91.yaml (full stack)
├── experiments/         ← ablation_configs/ (0_full … 7_no_domain_adaptation)
├── tests/               ← 296 passing CPU unit tests (+1 optional skip; +5 sample-data integration)
│
├── setup/               ← environment bootstrap + verification
│   ├── setup_ogap_env_v2.sh        (UNet/dense env; --with-mamba, --verify-only)
│   ├── setup_ogap_env_mamba.sh     (dedicated torch-2.5.1 SegMamba env — mamba-ssm is torch-pinned)
│   └── verify_ogap_env.py          (env verifier; honours OGAP_REQUIRE_MAMBA for the SegMamba checks)
├── workflow/            ← pre-registered audit pipeline (numbered steps)
│   ├── build_split_manifests.py    (Step 3 — train / internal-val / external-val CSVs)
│   ├── stage_ogap_csv_to_tmp.py    (stage NIfTI volumes into $SLURM_TMPDIR)
│   ├── eval_hic_vs_lmic.py         (Step 4 — HIC vs LMIC external-validation stats)
│   └── launch_ogap_train.sh        (Step 5 — single-entry train dispatcher: `unet` | `mamba`)
├── slurm/               ← Rorqual job scripts
│   ├── submit_ogap_rorqual_pipeline_experimental.sh   (orchestrator: teacher→student→export→eval)
│   ├── submit_ogap_teacher_rorqual_experimental.sbatch
│   ├── submit_ogap_teacher_rorqual_ddp.sbatch         (4×H100 multi-GPU teacher)
│   ├── submit_ogap_student_rorqual_experimental.sbatch
│   ├── submit_ogap_export_int8_rorqual_experimental.sbatch
│   ├── submit_utsw_eval_tta_rorqual_experimental.sbatch
│   ├── submit_ogap_posthoc_full_rorqual.sbatch
│   └── ogap_v9.sbatch
```

### Backward compatibility (verified)
* `python OGAP_source_code_experimental_v9.py <cmd> …` — **unchanged**; the shim
  re-exports `ogap.legacy.main`. All legacy subcommands present.
* `python -m ogap <cmd> …` — same legacy commands, delegated verbatim, plus `nas-search`.
* Extracted models produce **byte-identical `state_dict` keys** to the monolith;
  existing checkpoints load `strict=True`.

---

## 2. Quickstart (Rorqual)

```bash
# 1. Build the environment (UNet/dense path)
bash setup/setup_ogap_env_v2.sh            # add --with-mamba to also build SegMamba deps in-env
python setup/verify_ogap_env.py            # green/red status per dependency

# 2. (optional) Build the dedicated SegMamba env (separate torch-2.5.1 venv)
bash setup/setup_ogap_env_mamba.sh

# 3. Build the data manifests (pre-registered split policy)
python workflow/build_split_manifests.py …

# 4. Train end-to-end (teacher → student → INT8 export → eval)
workflow/launch_ogap_train.sh unet         # UNet3D dense heavy teacher
workflow/launch_ogap_train.sh mamba        # SegMamba teacher (needs the mamba env)

# 5. External-validation statistics
python workflow/eval_hic_vs_lmic.py --hic_per_case … --lmic_per_case … --out_dir …
```

Submit individual jobs directly from the repo root, e.g.:

```bash
sbatch slurm/submit_ogap_teacher_rorqual_experimental.sbatch
sbatch slurm/submit_ogap_teacher_rorqual_ddp.sbatch          # 4×H100
```

**SegMamba via direct submission** (the standalone mamba wrappers were folded into the
main scripts — `workflow/launch_ogap_train.sh mamba` is the common path):

```bash
sbatch --export=ALL,ENV_PATH=$HOME/…/envs/ogap_env_v2_mamba,\
EXPECTED_TORCH_PUBLIC_VERSION=2.5.1,TEACHER_ARCH=segmamba,OUT_DIR=…/teacher_mamba \
  slurm/submit_ogap_teacher_rorqual_experimental.sbatch
```

---

## 3. New capabilities (all opt-in, all CPU-tested)

| Capability | Module | Flag |
|---|---|---|
| Continuous-depth ODE teacher (Chen 2018 §2 adjoint) | `ogap/models/ode.py` | `--teacher_arch ode` |
| Weight-tied ODE student (train continuous, deploy discrete, INT8-friendly) | `ogap/models/student_ode.py` | model variant |
| Hardware-aware NAS / Once-for-All supernet (White 2023) | `ogap/nas/` | `python -m ogap nas-search …` |
| Latent-ODE longitudinal RANO tracker (Chen 2018 §5) | `ogap/longitudinal/latent_ode.py` | `inference.longitudinal` |
| Continuous-normalizing-flow OOD scorer (Chen 2018 §4) | `ogap/ood/cnf.py` | `inference.ood` |
| Group-balanced conformal abstention | `ogap/ood/conformal.py` | eval-time |
| Equity / fairness-without-harm reporting | `ogap/evaluation/equity.py` | eval-time |
| Field-strength-aware physics augmentation | `ogap/augmentations/physics_augment.py` | `augmentation.physics.*` |

---

## 4. Running the tests

```bash
pip install pytest
python -m pytest tests/ -q          # 296 passing, 1 skipped on CPU
```

---

## 5. Honest status — verified vs. needs the cluster

**Verified here (CPU):** all imports; model forward/backward; monolith→package
checkpoint parity; OFA export equality; zero-cost proxies; the `nas-search` CLI;
latent-ODE fit/extrapolation; CNF OOD separation; conformal + equity statistics;
**after the reorganization the full suite still passes (296 passing, 1 skipped) and every
entry point imports from its new location.**

**Needs Rorqual (GPU + BraTS-Africa data):** the KD-lift benchmark of the ODE/SegMamba
teachers; training the OFA supernet; ONNX/INT8 export numbers (use the ONNX/onnxruntime
path only — the CPU `quantize_student_int8` proxy does not quantise Conv3d); latent-ODE /
CNF on real longitudinal + feature data. DDP is wired into both training loops
(single-process verified here); the multi-GPU run still needs a one-node smoke test.

**Before submission (author TODO):** finalise the `CITATION.cff` author / affiliation /
ORCID fields and confirm the `LICENSE` choice + copyright holder; `ogap/legacy.py` can be
further thinned once a GPU run verifies the extracted models match it.

---

## 6. Paper mapping

| Paper | Applied as |
|---|---|
| *Neural ODEs* (Chen 2018) §2 adjoint | constant-memory ODE **teacher** |
| §2 Euler ≡ weight-tied ResNet | **deployable** weight-tied ODE student |
| §5 latent ODE for irregular series | longitudinal **RANO tracker** |
| §4 continuous normalizing flow | **OOD** density scorer |
| *NAS: 1000 Papers* (White 2023) §4 one-shot/OFA | **elastic supernet** |
| §5 performance estimation | **zero-cost proxies** (validated by `ogap/nas/validation.py`) |
| SegMamba (Xing 2024) | SegMamba **teacher** arm |
| Conformal small-data (Sánchez-Domínguez 2025) | group-balanced **conformal** abstention |
| Fairness without Harm (Pang 2024) | **equity** risk-disparity reporting |
