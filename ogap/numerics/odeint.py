"""Fixed-step ODE integrators (pure PyTorch) with an optional adjoint bridge.

We deliberately avoid a hard dependency on ``torchdiffeq``. The OGAP cluster
environment pins a Compute-Canada torch wheel and adding adaptive-solver deps
is friction; more importantly, the *deployable* student must export to a static
graph, which adaptive solvers preclude. So the default solvers here are
**fixed-step** (Euler / midpoint / RK4): the number of function evaluations is a
known constant, the unrolled graph is static, and a fixed-step Euler solve is
*exactly* a weight-tied residual network - which is what lets us train
continuous-depth and deploy a discrete, INT8-friendly model (see
:mod:`ogap.models.student_ode`).

When ``torchdiffeq`` is installed and ``ODESolverConfig.adjoint=True``, we
delegate to ``torchdiffeq.odeint_adjoint`` to get the paper's true O(1)-memory
training (Chen et al., 2018, §2). An adjoint request now fails clearly if the
backend is missing; silently falling back would make the implementation disagree
with the paper's memory guarantee.

The public API mirrors ``torchdiffeq.odeint``::

    ys = odeint(func, y0, t, config)            # func(t, y) -> dy/dt

``func`` is callable ``(scalar_t, state) -> d_state`` and ``t`` is a 1-D tensor
of evaluation times (monotonic). The return has shape ``(len(t), *y0.shape)``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import Tensor

ODEFunc = Callable[[Tensor, Tensor], Tensor]

_FIXED_METHODS = ("euler", "midpoint", "rk4")


def torchdiffeq_available() -> bool:
    """True if the optional ``torchdiffeq`` adjoint backend can be imported."""
    try:
        import torchdiffeq  # noqa: F401
        return True
    except Exception:
        return False


@dataclass
class ODESolverConfig:
    """Configuration for an ODE solve.

    Attributes
    ----------
    method:
        One of ``"euler"``, ``"midpoint"``, ``"rk4"`` (fixed-step, dependency
        free) or ``"dopri5"`` (adaptive; requires ``torchdiffeq``).
    steps:
        Number of fixed integration steps between consecutive entries of ``t``.
        Ignored for adaptive methods. ``steps`` is the lever that trades compute
        for accuracy at *fixed, predictable* cost - the LMIC-deployment-friendly
        analogue of the paper's adaptive "reduce accuracy for low power".
    adjoint:
        If True, use ``torchdiffeq.odeint_adjoint`` for constant-memory
        backprop. Raises ``ImportError`` when the backend is unavailable because
        fixed-step backprop does not satisfy the Neural ODE paper's memory
        guarantee.
    rtol, atol:
        Tolerances forwarded to the adaptive backend only.
    """

    method: str = "rk4"
    steps: int = 4
    adjoint: bool = False
    rtol: float = 1e-4
    atol: float = 1e-5

    def __post_init__(self) -> None:
        if self.method not in _FIXED_METHODS and self.method != "dopri5":
            raise ValueError(
                f"Unknown ODE method {self.method!r}; expected one of "
                f"{_FIXED_METHODS + ('dopri5',)}."
            )
        if self.steps < 1:
            raise ValueError(f"steps must be >= 1, got {self.steps}.")


def _fixed_step(func: ODEFunc, y0: Tensor, t0: Tensor, t1: Tensor,
                steps: int, method: str) -> Tensor:
    """Integrate ``func`` from ``t0`` to ``t1`` with ``steps`` uniform steps."""
    dt = (t1 - t0) / steps
    y = y0
    t = t0
    for _ in range(steps):
        if method == "euler":
            dy = func(t, y)
            y = y + dt * dy
        elif method == "midpoint":
            k1 = func(t, y)
            k2 = func(t + dt * 0.5, y + dt * 0.5 * k1)
            y = y + dt * k2
        else:  # rk4
            k1 = func(t, y)
            k2 = func(t + dt * 0.5, y + dt * 0.5 * k1)
            k3 = func(t + dt * 0.5, y + dt * 0.5 * k2)
            k4 = func(t + dt, y + dt * k3)
            y = y + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        t = t + dt
    return y


def odeint(func: ODEFunc, y0: Tensor, t: Tensor,
           config: Optional[ODESolverConfig] = None) -> Tensor:
    """Solve dy/dt = func(t, y) at the times in ``t``.

    Parameters
    ----------
    func:
        ``func(scalar_time, state) -> d_state`` with ``d_state`` shaped like
        ``state``.
    y0:
        Initial state at ``t[0]`` (any shape).
    t:
        1-D, monotonically increasing tensor of evaluation times.
    config:
        :class:`ODESolverConfig`; defaults to RK4 with 4 steps.

    Returns
    -------
    Tensor of shape ``(len(t), *y0.shape)`` with the state at each time.
    """
    if config is None:
        config = ODESolverConfig()
    if t.ndim != 1 or t.numel() < 2:
        raise ValueError("t must be a 1-D tensor with at least two entries.")

    want_adaptive = config.method == "dopri5" or config.adjoint
    if want_adaptive:
        if torchdiffeq_available():
            import torchdiffeq
            solver = torchdiffeq.odeint_adjoint if config.adjoint else torchdiffeq.odeint
            # The adaptive/adjoint path uses dopri5 (the O(1)-memory adaptive solver). If a
            # fixed-step method was requested alongside adjoint it is NOT honoured here, so
            # warn rather than silently substitute it (use adjoint=False for a fixed-step
            # euler/midpoint/rk4 solve).
            if config.method != "dopri5":
                import logging
                logging.getLogger(__name__).warning(
                    "ODESolverConfig(method=%r, adjoint=%s): the adaptive/adjoint path uses "
                    "dopri5; the fixed-step method is ignored.", config.method, config.adjoint)
            return solver(func, y0, t, method="dopri5", rtol=config.rtol, atol=config.atol)
        if config.adjoint:
            raise ImportError(
                "ODESolverConfig(adjoint=True) requires torchdiffeq for "
                "odeint_adjoint; install torchdiffeq or set adjoint=False for "
                "a fixed-step, non-constant-memory ablation."
            )
        raise ImportError(
            "ODESolverConfig(method='dopri5') requires torchdiffeq; choose one "
            "of euler/midpoint/rk4 for the dependency-free fixed-step solver."
        )
    else:
        method = config.method

    outs = [y0]
    y = y0
    for i in range(t.numel() - 1):
        y = _fixed_step(func, y, t[i], t[i + 1], config.steps, method)
        outs.append(y)
    return torch.stack(outs, dim=0)


def integrate(func: ODEFunc, y0: Tensor, *,
              t0: float = 0.0, t1: float = 1.0,
              config: Optional[ODESolverConfig] = None) -> Tensor:
    """Convenience: return only the final state of an ODE block solve.

    This is the workhorse used by :class:`ogap.models.ode.ODEBlock3D` to replace
    a residual block - integrate the learned dynamics from ``t0`` to ``t1`` and
    return ``y(t1)``.
    """
    if config is None:
        config = ODESolverConfig()
    t = torch.tensor([t0, t1], dtype=y0.dtype, device=y0.device)
    return odeint(func, y0, t, config)[-1]
