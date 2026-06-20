# OGAP: Open Glioma Analysis Pipeline

Brain-tumour segmentation for low-field / LMIC MRI (BraTS-Africa, 64 mT Hyperfine
Swoop) with teacher-to-student knowledge distillation and INT8 edge deployment. The
codebase was refactored from a single monolith script
(`OGAP_source_code_experimental_v9.py`) into the tested `ogap/` package, and the
supporting scripts are grouped by role.

The refactor is a strangler-fig wrap: the legacy core lives as `ogap/legacy.py` (the
extracted monolith, with audited correctness fixes applied symmetrically to it and the
package, all default-OFF), and every new capability is opt-in and feature-flagged off by
default. Backward compatibility is guaranteed by state-dict-key parity tests, not by
byte-identity of the source. Every module ships with CPU unit tests (`pytest tests/`).

---

## System requirements

- Operating system: Linux (developed and validated on the Alliance Canada CVMFS
  software stack, `StdEnv/2023`).
- Python: 3.11 (validated on 3.11.5).
- Training hardware: an NVIDIA GPU with >= 40 GB memory (validated on H100, CUDA 12.6).
- Inference: CPU-only deployment is supported through the INT8 ONNX / onnxruntime path.
- Python dependencies: see `pyproject.toml` and `requirements.txt`. The optional
  SegMamba teacher uses a separate environment (`setup/setup_ogap_env_mamba.sh`).

## Installation

Local / non-cluster:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .                  # the ogap package + the `ogap` console command
pip install -e '.[onnx,ode,dev]'  # optional: INT8 export, ODE teacher, tests
```

PyTorch note: `requirements.txt` pins `torch==2.6.0+computecanada`, the Alliance
Canada wheelhouse build, which is only resolvable on those clusters. Off-cluster,
install a standard PyTorch >= 2.5 from pytorch.org or PyPI first, then `pip install -e .`.

Cluster (Alliance Canada), using the bootstrap scripts:

```bash
bash setup/setup_ogap_env_v2.sh    # add --with-mamba to also build SegMamba deps
python setup/verify_ogap_env.py    # green/red status per dependency
```

## Repository layout

```
OGAP_v2/
|-- OGAP_source_code_experimental_v9.py   back-compat CLI shim -> ogap.legacy.main
|-- launch_pipeline.py                     production launcher (chains python -m ogap stages)
|-- ogap_compliance.py                     reporting-checklist utilities (CLAIM / TRIPOD+AI / SAGER / NIHMS)
|-- pyproject.toml  requirements.txt
|-- ogap/            the package: numerics, models, nas, losses, ood, longitudinal,
|                    augmentations, evaluation, inference, config, adaptation, utils
|   `-- legacy.py    the extracted ~16k-line core + audited default-OFF fixes
|-- configs/         baseline_v91.yaml (validated default), production_v91.yaml (full stack)
|-- experiments/     ablation_configs/ (0_full ... 7_no_domain_adaptation)
|-- tests/           CPU unit tests
|-- setup/           environment bootstrap + verification
|-- workflow/        pre-registered data + evaluation steps (build_master_manifest,
|                    build_split_manifests, stage_ogap_csv_to_tmp, eval_hic_vs_lmic,
|                    launch_ogap_train)
|-- slurm/           cluster job scripts (Rorqual and Trillium families, see below)
`-- docs/            methods specification + pre-registered analysis plan
```

Backward compatibility:

- `python OGAP_source_code_experimental_v9.py <cmd> ...`: unchanged; the shim
  re-exports `ogap.legacy.main`. All legacy subcommands present.
- `python -m ogap <cmd> ...`: the same legacy commands, delegated verbatim, plus
  `nas-search` and `--validate-config`.
- Extracted models produce byte-identical `state_dict` keys to the monolith, so
  existing checkpoints load with `strict=True`.

## Quickstart

```bash
# Validate the shipped config and print which capability gates are active
python -m ogap --validate-config --config configs/baseline_v91.yaml

# Build the data manifests (pre-registered split policy; data not shipped, see Data Availability)
python workflow/build_master_manifest.py ...
python workflow/build_split_manifests.py ...

# Train end-to-end (teacher -> student -> INT8 export -> eval)
workflow/launch_ogap_train.sh unet     # UNet3D dense heavy teacher
workflow/launch_ogap_train.sh mamba    # SegMamba teacher (needs the SegMamba env)

# Hardware-aware architecture search
python -m ogap nas-search --strategy evolution --n 64 --out nas/pareto.json

# External-validation statistics (HIC vs LMIC)
python workflow/eval_hic_vs_lmic.py --hic_per_case ... --lmic_per_case ... --out_dir ...
```

## Cluster deployment: Rorqual and Trillium

The pipeline runs on two Alliance Canada clusters. The model, data, and training math
are identical; only the Slurm directives and the scratch handling differ.

- Rorqual: `slurm/submit_ogap_*_rorqual_*` plus `slurm/ogap_v9.sbatch`.
- Trillium (SciNet): `slurm/submit_ogap_*_trillium_*`, plus the helpers
  `slurm/_gpucheck.sbatch`, `slurm/_ddp_probe.py`, `slurm/auto_resubmit_teacher.sh`,
  `slurm/submit_ogap_nas_trillium.sbatch`, and `slurm/submit_ogap_ood_trillium.sbatch`.
  See **[README_TRILLIUM.md](README_TRILLIUM.md)** for the authoritative Trillium guide
  (24 h walltime, diskless `$SCRATCH` staging, per-node GPU rules).

Before submitting on your own allocation, edit the Slurm `--account` and
`--output`/`--error` directives and the `PROJECT_ROOT` / `DATA_ROOT` defaults at the top
of each script (they currently default to the authors' cluster paths). For example:

```bash
sbatch slurm/submit_ogap_teacher_trillium_experimental.sbatch    # 1xH100
sbatch slurm/submit_ogap_teacher_trillium_ddp.sbatch             # 4xH100 full node
```

## New capabilities (all opt-in)

| Capability | Module | Flag |
|---|---|---|
| Continuous-depth ODE teacher (Chen 2018, adjoint) | `ogap/models/ode.py` | `--teacher_arch ode` |
| Weight-tied ODE student (train continuous, deploy discrete, INT8-friendly) | `ogap/models/student_ode.py` | model variant |
| Hardware-aware NAS / Once-for-All supernet (White 2023) | `ogap/nas/` | `python -m ogap nas-search ...` |
| Latent-ODE longitudinal RANO tracker (Chen 2018) | `ogap/longitudinal/latent_ode.py` | `inference.longitudinal` |
| Continuous-normalizing-flow OOD scorer (Chen 2018) | `ogap/ood/cnf.py` | `inference.ood` |
| Group-balanced conformal abstention | `ogap/ood/conformal.py` | eval-time |
| Equity / fairness-without-harm reporting | `ogap/evaluation/equity.py` | eval-time |
| Field-strength-aware physics augmentation | `ogap/augmentations/physics_augment.py` | `augmentation.physics.*` |

## Running the tests

```bash
pip install -e '.[dev]'    # or: pip install pytest
python -m pytest tests/ -q
```

The suite is CPU-only and needs no GPU or downloaded data (337 passing; a few tests
skip when an optional dependency such as `monai` or `mamba-ssm` is absent, or under
older PyTorch).

## Reproducing the paper results

The headline tables and figures come from the post-hoc reporting bundle plus the
external-validation statistics:

- `slurm/submit_ogap_posthoc_full_trillium.sbatch` runs, against a trained
  student/teacher checkpoint: the Holder alpha sweep (`holder_sweep`), the distillation
  comparison including the alpha=2 proper Cauchy-Schwarz arm (`kd_compare`), lesion-wise
  and field-strength-stratified HD95 and Dice (`evaluate --lesion_wise --stratify`), the
  INT8 non-inferiority tables, the group-balanced conformal coverage tables, the equity
  / fairness-without-harm tables, and the reporting-checklist compliance report
  (`ogap_compliance.py`).
- `python workflow/eval_hic_vs_lmic.py ...` produces the HIC-vs-LMIC external-validation
  comparison.

The mathematical specification is in `docs/2026-06-15-methods.md`; the pre-registered
evaluation, non-inferiority, and reader-study plan is in
`docs/2026-06-15-reader-study-and-noninferiority-analysis-plan.md`.

## Paper mapping

| Paper | Applied as |
|---|---|
| Neural ODEs (Chen 2018), adjoint | constant-memory ODE teacher |
| Euler step equals weight-tied ResNet | deployable weight-tied ODE student |
| Latent ODE for irregular series | longitudinal RANO tracker |
| Continuous normalizing flow | OOD density scorer |
| NAS: 1000 Papers (White 2023), one-shot / OFA | elastic supernet |
| Performance estimation | zero-cost proxies (validated by `ogap/nas/validation.py`) |
| SegMamba (Xing 2024) | SegMamba teacher arm |
| Conformal small-data (Sanchez-Dominguez 2025) | group-balanced conformal abstention |
| Fairness without Harm (Pang 2024) | equity risk-disparity reporting |

## Data Availability

OGAP is trained and evaluated on publicly governed and restricted-access cohorts; no
patient data are redistributed in this repository.

- BraTS-2023 (glioma) and BraTS-Africa: available through the BraTS challenge / Synapse
  under the challenge data-use agreement.
- UTSW-glioma and Erasmus cohorts: restricted clinical data, available from the
  respective institutions under a data-sharing agreement / ethics approval.

Field-strength buckets and the HIC vs LMIC split policy are defined in
`docs/2026-06-15-reader-study-and-noninferiority-analysis-plan.md`. The split manifests
are not shipped; regenerate the train / internal-val / external-val CSVs from your local
copies with `workflow/build_master_manifest.py` then `workflow/build_split_manifests.py`.
The pre-registered split provenance (seed, counts, policy version) is recorded in
`split_provenance.json`.

## Code Availability

The OGAP source code is openly available at https://github.com/iso489/OGAP_v2 under the
MIT License (see `LICENSE`). The exact version reported in the paper is archived at
Zenodo: https://doi.org/10.5281/zenodo.XXXXXXX (release `v9.1.0`). All results can be
regenerated with the released code and the reporting bundle described in "Reproducing
the paper results"; the CPU unit-test suite (`python -m pytest tests/`) checks the
load-bearing numerics. Cite the software through `CITATION.cff`.

## License

MIT. See [LICENSE](LICENSE).

## How to cite

See [CITATION.cff](CITATION.cff) for machine-readable citation metadata. Once a release
is archived (see Code Availability), cite the versioned Zenodo DOI.
