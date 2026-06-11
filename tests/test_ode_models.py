
import torch
from ogap.models import ODEBottleneckTeacher, build_ode_teacher
from ogap.numerics import ODESolverConfig


_FIXED = ODESolverConfig(method="rk4", steps=2, adjoint=False)


def test_ode_teacher_forward_and_features():
    torch.manual_seed(0)
    m = build_ode_teacher(in_channels=4, num_classes=4, base=8, ode_solver=_FIXED).eval()
    x = torch.randn(1, 4, 32, 32, 32)
    with torch.no_grad():
        logits, feats = m(x, return_features=True)
    assert logits.shape == (1, 4, 32, 32, 32)
    assert feats["bottleneck"].shape[1] == m.bottleneck_channels == 64
    assert feats["dec3"].shape[1] == m.dec3_channels == 32
    assert m.last_nfe > 0  # the ODE bottleneck actually integrated


def test_ode_teacher_checkpoint_roundtrip():
    a = ODEBottleneckTeacher(4, 4, base=8, ode_solver=_FIXED)
    b = ODEBottleneckTeacher(4, 4, base=8, ode_solver=_FIXED)
    b.load_state_dict(a.state_dict(), strict=True)  # must not raise


def test_ode_teacher_backprops():
    m = build_ode_teacher(in_channels=4, num_classes=4, base=8, ode_solver=_FIXED)
    x = torch.randn(1, 4, 16, 16, 16, requires_grad=False)
    loss = m(x).pow(2).mean()
    loss.backward()
    g = m.bottleneck_ode.func.net[0].weight.grad
    assert g is not None and torch.isfinite(g).all()
