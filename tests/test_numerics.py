
import math
import torch
from ogap.numerics import odeint, integrate, ODESolverConfig


def test_rk4_matches_exponential():
    # dy/dt = y, y0=1 -> y(1) = e
    f = lambda t, y: y
    y0 = torch.tensor([1.0])
    t = torch.tensor([0.0, 1.0])
    y = odeint(f, y0, t, ODESolverConfig(method="rk4", steps=50))[-1]
    assert abs(y.item() - math.e) < 1e-4


def test_euler_unroll_equivalence():
    # explicit Euler must equal odeint(method=euler)
    torch.manual_seed(0)
    A = torch.randn(3, 3) * 0.1
    f = lambda t, y: y @ A.T
    y0 = torch.randn(2, 3)
    N = 7
    y_solver = integrate(f, y0, t0=0.0, t1=1.0, config=ODESolverConfig(method="euler", steps=N))
    # manual euler
    dt = 1.0 / N
    h = y0.clone()
    for _ in range(N):
        h = h + dt * f(None, h)
    assert torch.allclose(y_solver, h, atol=1e-6)


def test_output_shape():
    f = lambda t, y: -y
    y0 = torch.randn(4, 5)
    t = torch.linspace(0, 1, 6)
    out = odeint(f, y0, t, ODESolverConfig(method="midpoint", steps=3))
    assert out.shape == (6, 4, 5)


def test_steps_validation():
    import pytest
    with pytest.raises(ValueError):
        ODESolverConfig(steps=0)
    with pytest.raises(ValueError):
        ODESolverConfig(method="nope")


def test_adjoint_requires_torchdiffeq_when_requested():
    import pytest
    from ogap.numerics import torchdiffeq_available

    if torchdiffeq_available():
        pytest.skip("torchdiffeq installed; adjoint backend is available")
    with pytest.raises(ImportError):
        odeint(lambda t, y: y, torch.ones(1), torch.tensor([0.0, 1.0]),
               ODESolverConfig(adjoint=True))
