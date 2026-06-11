"""Regression tests pinning baseline behaviour before additive feature work.

All run on CPU with synthetic data and fixed seeds.
"""
import torch

from ogap.models import build_ode_teacher, WeightTiedODEBlock3D
from ogap.nas import OFASupernet3D
from ogap.numerics import ODESolverConfig


_FIXED_STEP_ABLATION = ODESolverConfig(method="rk4", steps=2, adjoint=False)


def test_r1_ode_teacher_forward():
    torch.manual_seed(0)
    m = build_ode_teacher(
        in_channels=4,
        num_classes=4,
        base=8,
        ode_solver=_FIXED_STEP_ABLATION,
    ).eval()
    y = m(torch.randn(1, 4, 32, 32, 32))
    assert y.shape == (1, 4, 32, 32, 32)


def test_r2_weight_tied_student_block_forward():
    blk = WeightTiedODEBlock3D(4, 8, steps=4)
    y = blk(torch.randn(1, 4, 32, 32, 32))
    assert y.shape == (1, 8, 32, 32, 32)


def test_r3_task_loss_runs():
    from types import SimpleNamespace
    from ogap.losses import build_task_loss
    cfg = SimpleNamespace(loss=SimpleNamespace(task=SimpleNamespace(type="legacy", num_classes=4)))
    logits = torch.randn(1, 4, 16, 16, 16)
    target = torch.randint(0, 4, (1, 16, 16, 16))
    assert torch.isfinite(build_task_loss(cfg)(logits, target))


def test_r4_kd_step_backward():
    torch.manual_seed(0)
    teacher = build_ode_teacher(4, 4, base=8, ode_solver=_FIXED_STEP_ABLATION).eval()
    student = torch.nn.Conv3d(4, 4, 1)
    x = torch.randn(1, 4, 16, 16, 16)
    with torch.no_grad():
        t = teacher(x)
    s = student(x)
    kl = torch.nn.functional.kl_div(
        torch.log_softmax(s, 1), torch.softmax(t, 1), reduction="batchmean")
    kl.backward()
    assert student.weight.grad is not None


def test_r5_ofa_export_bit_identical():
    torch.manual_seed(0)
    net = OFASupernet3D(8, 4, stage_chs=(8, 16, 32, 64)).eval()
    x = torch.randn(1, 8, 16, 16, 16)
    net.set_active_subnet(0.5, 4)
    with torch.no_grad():
        y_super = net(x)
        y_sub = net.export_subnet().eval()(x)
    assert torch.allclose(y_super, y_sub, atol=1e-5)
