# v9.1 — low-field / hardware upgrades

This note summarises the ML-methodology, low-field-realism, and hardware-utilisation
improvements added in v9.1. Every new capability defaults OFF (the default training path is
unchanged) and is CPU-unit-tested. Multi-GPU is single-process-verified here and needs a
one-node smoke test.

## Low-field performance
- **Geometric-augmentation regression FIXED.** Enabling `augmentation.physics` replaced
  the legacy augmentor (flip + elastic) with the intensity-only v9.1 engine, so the v9.1
  stack trained with NO spatial augmentation. New channel-aware `ogap/augmentations/geometric.py`
  (flip / affine / elastic on ALL channels + label; tissue priors stay co-registered),
  wired into the CPU dataset path and the GPU prefetcher (runs before the intensity shield).
  Enabled in `baseline_v91.yaml` (flip+affine) and `production_v91.yaml` (+elastic).
- **Bloch low-field render made realistic + correct.** `bloch_lowfield.py`: (a) torch-
  differentiable Bloch signal (autograd to T1/T2/PD/TR/TE — the differentiability claim is
  now real); (b) smooth within-tissue heterogeneity; (c) low-field PSF Gaussian blur so
  lesion boundaries are not razor-sharp; (d) real-texture blend so fine anatomy survives;
  (e) per-sample TR/TE/TI/B0 acquisition jitter; (f) **zero-background preservation** (the
  render no longer injects a non-zero background that breaks the `v != 0` masks and the
  train/test match). Knobs under `augmentation.physics.bloch_*`.
- **SynthSeg-style randomized generator** (`ogap/augmentations/synth.py`, Billot 2023): a
  fully contrast-randomized training arm for intensity-agnostic robustness, wired as a
  per-sample alternative to the Bloch render (`augmentation.physics.synth_*`).
- **Dormant low-field transforms enabled in production**: `bloch_lowfield` and
  `wraparound_aliasing` (the module's "dominant low-field artifact").
- `resolution_jitter` now upsamples **trilinear** (smooth partial-volume blur), not the
  blocky `nearest`. `gmm_contrast` is now a strictly **monotone** piecewise-linear remap.
- **Self-training / consistency adaptation** (`ogap/adaptation/self_training.py`): confidence-
  filtered pseudo-labels + FixMatch weak↔strong consistency to adapt to real low-field-adjacent
  data (BraTS-Africa) and close the in-silico-only validation gap.

## SOTA recipe
- **Weight EMA** (`ogap/utils/ema.py`): evaluate + deploy the averaged weights. Wired into
  both teacher and student loops (built after resume, before compile; best/last save EMA;
  resume keeps raw). `training.ema.{enabled,decay}`.
- **Region-sigmoid + batch-Dice loss** (`loss.task.type: region_sigmoid`): independent
  per-region binary logits via the log-sum-exp identity + BCE + batch Dice (the nnU-Net
  region-based objective) on the existing checkpoint-frozen 4-class head.
- **Large-kernel teacher block** (`block_style: mednext_large`): 5³ depthwise + 4× inverted
  bottleneck (MedNeXt's wide receptive field) — additive, checkpoints untouched.
- **Teacher gradient accumulation** (parity with the student loop).

## Hardware
- **DDP wired** into both training loops via `ogap/utils/distributed.py`: init + per-rank
  device, teacher `wrap_ddp`, student manual gradient all-reduce (multi-module-safe),
  `DistributedWeightedSampler` (keeps the equity weighting per rank) + `set_epoch`, linear
  LR scaling, rank-0-gated checkpoint writes, process-group cleanup. All no-op single-process.

## Deferred (deliberate — frozen-parity risk > marginal value)
- **Teacher deep supervision** and the **per-instance MixStyle λ** tweak touch the
  checkpoint-frozen, `test_legacy_package_parity`-pinned models (and teacher DS additionally
  needs risky loop + loss wiring with a dead-gate). The prior audit already deemed per-channel
  MixStyle "a defensible variant." Left for a deliberate, GPU-validated change.

## Remaining (cluster-gated)
- Run the cluster-gated items: FSL FAST PVE generation, the ablation matrix, the 4-GPU DDP
  smoke test, INT8-ONNX numbers, and the conformal-coverage + equity tables.
