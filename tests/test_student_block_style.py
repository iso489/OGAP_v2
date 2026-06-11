"""Tests for the swappable student block style (conv vs weight-tied ODE).

The deployable student can be built either with the historical ``ResConvBlock3D``
blocks (``block_style="res"``, the default, checkpoint-compatible) or with
``WeightTiedODEBlock3D`` continuous-depth blocks (``block_style="ode"``). The ODE
student is the export-friendly target distilled from a (e.g. SegMamba) teacher.
"""
import torch

from ogap.models import UNet3DStudent, build_student, WeightTiedODEBlock3D
from ogap.models.blocks import ResConvBlock3D
from ogap.losses.distillation import RegionKDLoss

_ENCDEC = ("enc1", "enc2", "enc3", "bottleneck", "dec3", "dec2", "dec1")


def _blocks(model):
    return [getattr(model, n) for n in _ENCDEC]


def test_default_block_style_is_resconv():
    # Default must stay byte-compatible with existing checkpoints (conv blocks).
    m = UNet3DStudent(in_channels=8, num_classes=4, base=8)
    assert all(isinstance(b, ResConvBlock3D) for b in _blocks(m))


def test_ode_block_style_uses_weight_tied_ode_blocks():
    m = UNet3DStudent(in_channels=8, num_classes=4, base=8, block_style="ode")
    assert all(isinstance(b, WeightTiedODEBlock3D) for b in _blocks(m))


def test_ode_student_forward_shape():
    m = UNet3DStudent(in_channels=8, num_classes=4, base=8, block_style="ode").eval()
    logits = m(torch.randn(1, 8, 32, 32, 32))
    assert logits.shape == (1, 4, 32, 32, 32)


def test_ode_steps_control_effective_depth_not_params():
    # The whole point of the weight-tied ODE: more steps = deeper, SAME params.
    shallow = UNet3DStudent(8, 4, base=8, block_style="ode", ode_steps=2)
    deep = UNet3DStudent(8, 4, base=8, block_style="ode", ode_steps=8)
    assert sum(p.numel() for p in shallow.parameters()) == sum(
        p.numel() for p in deep.parameters()
    )
    assert deep.bottleneck.steps == 8


def test_build_student_passes_block_style_through():
    m = build_student(8, 4, base=8, block_style="ode")
    assert isinstance(m.bottleneck, WeightTiedODEBlock3D)
    assert m(torch.randn(1, 8, 16, 16, 16)).shape == (1, 4, 16, 16, 16)


def test_saved_student_arch_config_round_trips_ode_style(tmp_path):
    from ogap.legacy import _load_saved_student_arch_config
    import json

    ckpt = tmp_path / "best_student.pth"
    ckpt.write_bytes(b"")
    (tmp_path / "train_config.json").write_text(
        json.dumps({
            "student_base": 12,
            "attention": "eca",
            "enable_deep_supervision": False,
            "curriculum_dropout_p": 0.2,
            "student_block_style": "ode",
            "student_ode_steps": 6,
        }),
        encoding="utf-8",
    )
    assert _load_saved_student_arch_config(ckpt, 16) == (12, "eca", False, 0.2, "ode", 6)


def test_invalid_block_style_raises():
    try:
        UNet3DStudent(8, 4, base=8, block_style="nope")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown block_style")


def test_ode_student_trains_under_heterogeneous_kd():
    # Logit KD is architecture-agnostic: a (conv/Mamba) teacher's logits distil
    # into the ODE student and gradients must reach the ODE block parameters.
    student = UNet3DStudent(8, 4, base=8, block_style="ode")
    teacher_logits = torch.randn(1, 4, 16, 16, 16)
    s_logits = student(torch.randn(1, 8, 16, 16, 16))
    RegionKDLoss()(s_logits, teacher_logits).backward()
    ode_params = [
        p for m in student.modules() if isinstance(m, WeightTiedODEBlock3D)
        for p in m.parameters()
    ]
    assert ode_params and any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in ode_params
    )
