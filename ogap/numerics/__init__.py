"""Self-contained ODE numerics for OGAP's continuous-depth models.

The Neural ODE paper (Chen et al., NeurIPS 2018) parameterises the derivative
of a hidden state and obtains the output by an ODE solve. Its headline benefit
is *constant-memory* backpropagation via the adjoint sensitivity method.

This subpackage provides:

* :func:`~ogap.numerics.odeint.odeint` — fixed-step Euler / midpoint / RK4
  integrators with **zero external dependencies** (pure PyTorch), so the ODE
  teacher and ODE student run on the cluster even when ``torchdiffeq`` is not
  installed.
* A bridge to ``torchdiffeq.odeint_adjoint`` for true O(1)-memory adjoint
  training. Requests for adjoint/adaptive solves raise if ``torchdiffeq`` is not
  installed, so fixed-step ablations cannot be mistaken for the paper method.

Keeping these solvers in one place means every continuous-depth component
(ODE bottleneck teacher, weight-tied ODE student, latent-ODE tracker, CNF OOD
scorer) shares one audited, unit-tested integrator.
"""
from __future__ import annotations

from .odeint import ODESolverConfig, odeint, integrate, torchdiffeq_available

__all__ = [
    "ODESolverConfig",
    "odeint",
    "integrate",
    "torchdiffeq_available",
]
