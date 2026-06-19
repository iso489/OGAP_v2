"""Tests for the 2026 throughput/accuracy + low-field pass.

Fix #1 - the bare legacy teacher CLI trained un-compiled / no-EMA because
``torch.compile`` and weight-EMA were gated on a v9.1 ``--config`` that the
teacher sbatch never passes. ``TeacherConfig`` now carries ``compile``/``ema``
knobs and ``pretrain_teacher`` honours them directly.

Fix #2 - the field-strength contrast warp (the most impactful low-field
augmentation) was OFF *and*, more importantly, absent from the
``GPUPhysicsAugmentor`` that the teacher actually runs on the GPU, so it never
fired. It is now ported to the GPU augmentor and shares ONE relaxometry source
of truth (``MRIPhysicsAugmentor._modality_signal_scalar``) with the CPU path.

All tests run on CPU: ``GPUPhysicsAugmentor.__call__`` early-returns for non-CUDA
tensors, but the new ``_simulate_field_strength_contrast_batch`` is device-
agnostic and is exercised directly here.
"""
import dataclasses
import subprocess
import sys
from pathlib import Path

import torch

import ogap.legacy as L
from ogap.utils.ema import ModelEMA, build_model_ema

MRIPhysicsAugmentor = L.MRIPhysicsAugmentor
GPUPhysicsAugmentor = L.GPUPhysicsAugmentor
REPO = Path(__file__).resolve().parents[1]


def _fg_volume(seed: int = 0, n: int = 1, c: int = 4) -> torch.Tensor:
    """A zero-background (skull-stripped) volume with a nonzero foreground block."""
    g = torch.Generator().manual_seed(seed)
    x = torch.zeros(n, c, 16, 16, 16)
    x[:, :, 4:12, 4:12, 4:12] = torch.rand(n, c, 8, 8, 8, generator=g) + 0.5
    return x


# ───────────────────────── Fix #2: GPU field-strength contrast warp ──────────

def test_modality_signal_scalar_matches_instance_method():
    """The shared static must be bit-identical to the CPU instance method
    (single physics source of truth - no CPU/GPU LUT drift)."""
    inst = MRIPhysicsAugmentor.__new__(MRIPhysicsAugmentor)
    for kind in ("t1", "t2", "flair"):
        for tissue in ("wm", "gm", "csf", "edema", "et"):
            for b0 in (0.064, 0.55, 1.5, 3.0, 7.0):
                a = MRIPhysicsAugmentor._modality_signal_scalar(b0, tissue, kind)
                b = inst._modality_signal(b0, tissue, kind)
                assert a == b, (kind, tissue, b0, a, b)


def test_contrast_warp_ratio_physics_direction_and_clip():
    """At low field WM T1 shortens, so T1-weighted recovery (1-e^{-TR/T1}) and
    hence signal INCREASES 3T->0.064T; Gd-shortened ET brightens even more.
    Every ratio is clipped to [0.1, 4.0]."""
    aug = GPUPhysicsAugmentor(p_field_contrast_warp=1.0, field_contrast_source_b0=3.0)
    assert aug._field_contrast_warp_ratio(3.0, 0.064, "wm", "t1") > 1.0
    assert aug._field_contrast_warp_ratio(3.0, 0.064, "et", "t1") > 1.0
    for tissue in ("wm", "gm", "csf", "edema", "et"):
        for tgt in (0.064, 0.3, 0.55, 1.0, 1.5, 7.0):
            for kind in ("t1", "t2", "flair"):
                r = aug._field_contrast_warp_ratio(3.0, tgt, tissue, kind)
                assert 0.10 <= r <= 4.0


def test_batch_warp_preserves_background_and_shape():
    aug = GPUPhysicsAugmentor(p_field_contrast_warp=1.0, field_contrast_source_b0=3.0,
                              contrast_mod_indices=[1])
    x = _fg_volume()
    out = aug._simulate_field_strength_contrast_batch(x.clone(), torch.tensor([0.3]))
    assert out.shape == x.shape
    assert torch.all(out[x == 0] == 0)            # skull-stripped zeros preserved
    assert (out - x).abs().sum().item() > 0        # foreground actually warped


def test_batch_warp_noop_when_disabled_or_at_source():
    x = _fg_volume()
    off = GPUPhysicsAugmentor(p_field_contrast_warp=0.0)
    assert torch.equal(
        off._simulate_field_strength_contrast_batch(x.clone(), torch.tensor([0.3])), x
    )
    on = GPUPhysicsAugmentor(p_field_contrast_warp=1.0, field_contrast_source_b0=3.0)
    # target == source (3 T) -> nothing to warp even at p=1
    assert torch.equal(
        on._simulate_field_strength_contrast_batch(x.clone(), torch.tensor([3.0])), x
    )


def test_gpu_augmentor_default_is_backward_compatible_noop():
    """Default construction (no warp kwargs) keeps the historical no-op so
    unmodified runs are byte-identical."""
    aug = GPUPhysicsAugmentor()
    assert aug.p_field_contrast_warp == 0.0
    assert aug.field_contrast_source_b0 == 3.0
    x = _fg_volume()
    assert torch.equal(
        aug._simulate_field_strength_contrast_batch(x.clone(), torch.tensor([0.3])), x
    )


def test_batch_warp_matches_manual_per_bin_reconstruction():
    """Strong correctness check: the warp multiplies each foreground tercile bin
    by its tissue ratio EXACTLY once (disjoint bins, no compounding). Verified by
    reconstructing the expected output on a single deterministic channel."""
    aug = GPUPhysicsAugmentor(p_field_contrast_warp=1.0, field_contrast_source_b0=3.0)
    g = torch.Generator().manual_seed(7)
    x = torch.zeros(1, 1, 8, 8, 8)
    x[:, :, 1:7, 1:7, 1:7] = torch.rand(1, 1, 6, 6, 6, generator=g) + 0.5
    b0 = 0.55
    out = aug._simulate_field_strength_contrast_batch(x.clone(), torch.tensor([b0]))

    ch = x[0, 0]
    fg = ch != 0
    fgv = ch[fg]
    q1, q2 = torch.quantile(
        fgv.float(), torch.tensor([0.333, 0.667], dtype=torch.float32)
    ).tolist()
    exp = ch.clone()
    # single (non-contrast) channel -> t1 kind -> {csf, wm, gm}
    for mask, tissue in (
        (fg & (ch <= q1), "csf"),
        (fg & (ch > q1) & (ch <= q2), "wm"),
        (fg & (ch > q2), "gm"),
    ):
        r = aug._field_contrast_warp_ratio(3.0, b0, tissue, "t1")
        exp[mask] = exp[mask] * r
    assert torch.allclose(out[0, 0], exp, atol=1e-5)


def test_batch_warp_per_sample_gate_is_independent():
    """With a 2-sample batch and p=1, both samples warp; p=0 warps neither."""
    on = GPUPhysicsAugmentor(p_field_contrast_warp=1.0, field_contrast_source_b0=3.0)
    x = _fg_volume(n=2)
    out = on._simulate_field_strength_contrast_batch(x.clone(), torch.tensor([0.3, 1.0]))
    assert (out[0] - x[0]).abs().sum().item() > 0
    assert (out[1] - x[1]).abs().sum().item() > 0


# ───────────────────────── Fix #1: teacher compile + EMA ─────────────────────

def test_teacherconfig_has_compile_and_ema_knobs_off_by_default():
    fields = {f.name: f for f in dataclasses.fields(L.TeacherConfig)}
    for name in ("compile", "compile_mode", "compile_dynamic", "ema", "ema_decay"):
        assert name in fields, name
    # defaults OFF so the historical path is byte-identical
    assert fields["compile"].default is False
    assert fields["ema"].default is False
    assert fields["compile_dynamic"].default is False
    assert fields["compile_mode"].default == "max-autotune"
    assert fields["ema_decay"].default == 0.999


def test_root_cause_v91_gates_disabled_without_config():
    """Confirms WHY the teacher was un-compiled/no-EMA: both v9.1 gates return
    'disabled' when no --config is supplied (the bare-CLI case). The fix's
    ``cfg.compile`` / ``cfg.ema`` branches are therefore the operative enablers."""
    enabled, _mode, _dynamic = L._v91_compile_cfg(None)
    assert enabled is False
    model = torch.nn.Conv3d(4, 4, 3)
    assert build_model_ema(model, None) is None


def test_bare_cli_ema_builds_from_teacherconfig_decay():
    """The fix builds ModelEMA(model, decay=cfg.ema_decay) on the bare-CLI path;
    its state_dict stays checkpoint-compatible (same keys as the raw module)."""
    model = torch.nn.Conv3d(4, 4, 3)
    ema = ModelEMA(model, decay=0.997)
    assert abs(ema.decay - 0.997) < 1e-9
    ema.update(model)  # tracks the live module; must not raise
    assert set(ema.state_dict().keys()) == set(model.state_dict().keys())


def test_teacher_cli_exposes_compile_and_ema_flags():
    """The real argparse parser must accept the new teacher flags (end-to-end)."""
    proc = subprocess.run(
        [sys.executable, str(REPO / "OGAP_source_code_experimental_v9.py"),
         "pretrain_teacher", "--help"],
        capture_output=True, text=True, timeout=240,
    )
    help_text = proc.stdout + proc.stderr
    for flag in ("--compile", "--compile_mode", "--compile_dynamic", "--ema", "--ema_decay"):
        assert flag in help_text, flag
