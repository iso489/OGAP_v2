"""Continuous Normalizing Flow for principled OOD scoring on OGAP features.

OGAP already flags distribution shift with Mahalanobis distance
(``cmd_mahalanobis_ood``) and controls risk with conformal prediction. A
Continuous Normalizing Flow (Neural ODE paper §4, "Instantaneous Change of
Variables") adds an *exact density* model: fit a CNF to in-distribution feature
embeddings (e.g. the student's bottleneck ``ood_features``), then score a new
scan by its exact log-likelihood. Low likelihood ⇒ out-of-distribution — exactly
the "is this African low-field scan in-distribution?" question, with a
calibrated probabilistic answer that plugs into the existing selective-
prediction / conformal stack.

This is FFJORD-style with an **exact trace** of the Jacobian (cheap for the
modest feature dimensions OGAP uses; a Hutchinson estimator is the drop-in for
larger dims). The flow is integrated with the shared :mod:`ogap.numerics`
solver so the whole package uses one audited integrator.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from ..numerics import ODESolverConfig, odeint


class _CNFField(nn.Module):
    """Velocity field f(t, z) for the flow (small time-conditioned MLP)."""

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, dim),
        )

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        tt = torch.ones_like(z[:, :1]) * t.to(z.dtype)
        return self.net(torch.cat([z, tt], dim=-1))


def _exact_trace(dz: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """tr(∂f/∂z) per sample, via one autograd.grad per dimension (cost O(dim))."""
    tr = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
    create = torch.is_grad_enabled()
    for i in range(z.shape[1]):
        gi, = torch.autograd.grad(dz[:, i].sum(), z, create_graph=create, retain_graph=True)
        tr = tr + gi[:, i]
    return tr


def _hutchinson_trace(dz: torch.Tensor, z: torch.Tensor,
                      n_samples: int = 1) -> torch.Tensor:
    """Stochastic trace estimate tr(∂f/∂z) ≈ E_ε[εᵀ(∂f/∂z)ε], ε∼Rademacher.

    Cost is O(n_samples) vector-Jacobian products regardless of ``dim`` — the
    FFJORD-recommended estimator for large feature dimensions (Chen 2018 §4 notes
    the trace is the bottleneck; Grathwohl 2019 uses exactly this). Unbiased: for
    a linear field f=a·z it returns a·dim for every sample. [audit S3-C]
    """
    tr = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
    create = torch.is_grad_enabled()
    for _ in range(max(1, int(n_samples))):
        eps = torch.randint(0, 2, z.shape, device=z.device, dtype=z.dtype) * 2.0 - 1.0
        # εᵀ(∂f/∂z) via VJP, then dot with ε.
        vjp, = torch.autograd.grad((dz * eps).sum(), z, create_graph=create,
                                   retain_graph=True)
        tr = tr + (vjp * eps).sum(dim=1)
    return tr / max(1, int(n_samples))


class CNFDensityEstimator(nn.Module):
    """Exact-likelihood density model over a feature vector for OOD scoring."""

    def __init__(self, dim: int, hidden: int = 64,
                 solver: Optional[ODESolverConfig] = None,
                 trace_estimator: str = "exact",
                 hutchinson_samples: int = 1) -> None:
        super().__init__()
        if trace_estimator not in ("exact", "hutchinson"):
            raise ValueError(
                f"trace_estimator must be 'exact' or 'hutchinson', got "
                f"{trace_estimator!r}."
            )
        self.dim = dim
        self.field = _CNFField(dim, hidden)
        # Euler keeps the augmented (z, logp) dynamics simple and stable here.
        self.solver = solver or ODESolverConfig(method="euler", steps=10)
        # "exact" trace is O(dim) autograd passes (accurate, fine for small dim);
        # "hutchinson" is O(samples) VJPs, the right choice for large feature
        # dimensions (Chen 2018 §4 / FFJORD). [audit S3-C]
        self.trace_estimator = trace_estimator
        self.hutchinson_samples = int(hutchinson_samples)

    def _augmented(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        z = state[:, : self.dim]
        with torch.enable_grad():
            z = z.requires_grad_(True)
            dz = self.field(t, z)
            if self.trace_estimator == "hutchinson":
                tr = _hutchinson_trace(dz, z, self.hutchinson_samples)
            else:
                tr = _exact_trace(dz, z)
        dlogp = -tr.unsqueeze(-1)
        return torch.cat([dz, dlogp], dim=-1)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Exact log p(x) under the flow, per sample (shape (B,))."""
        b = x.shape[0]
        state0 = torch.cat([x, torch.zeros(b, 1, device=x.device, dtype=x.dtype)], dim=-1)
        t = torch.tensor([0.0, 1.0], device=x.device, dtype=x.dtype)
        stateT = odeint(self._augmented, state0, t, self.solver)[-1]
        zT = stateT[:, : self.dim]
        delta_logp = stateT[:, self.dim]
        base = -0.5 * (zT.pow(2).sum(dim=-1) + self.dim * math.log(2 * math.pi))
        # Instantaneous change of variables: log p(x) = log p_base(z_T) + ∫₀¹ tr dt.
        # The augmented dynamics accumulate dlogp = -tr (see `_augmented`), so
        # delta_logp = -∫tr; hence we SUBTRACT it (matching the FFJORD reference
        # `logpx = logpz - delta_logp`). Adding it would invert the log-det term.
        return base - delta_logp

    def nll(self, x: torch.Tensor) -> torch.Tensor:
        """Mean negative log-likelihood — the training objective."""
        return -self.log_prob(x).mean()

    def fit(
        self,
        embeddings: torch.Tensor,
        *,
        steps: int = 200,
        lr: float = 1e-3,
        batch_size: Optional[int] = None,
    ) -> dict:
        """Fit the flow to reference embeddings and return training metadata."""
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dim:
            raise ValueError(
                f"embeddings must have shape (N, {self.dim}), got {tuple(embeddings.shape)}"
            )
        if embeddings.shape[0] == 0:
            raise ValueError("embeddings must contain at least one row")
        data = embeddings.detach()
        batch_size = int(batch_size or data.shape[0])
        batch_size = max(1, min(batch_size, data.shape[0]))
        opt = torch.optim.Adam(self.parameters(), lr=float(lr))
        self.train()
        final_loss = None
        for _ in range(max(1, int(steps))):
            if batch_size == data.shape[0]:
                batch = data
            else:
                idx = torch.randint(data.shape[0], (batch_size,), device=data.device)
                batch = data.index_select(0, idx)
            opt.zero_grad(set_to_none=True)
            loss = self.nll(batch)
            loss.backward()
            opt.step()
            final_loss = float(loss.detach().cpu())
        self.eval()
        return {
            "n_embeddings": int(data.shape[0]),
            "dim": int(self.dim),
            "steps": int(max(1, int(steps))),
            "batch_size": int(batch_size),
            "final_nll": float(final_loss if final_loss is not None else 0.0),
        }

    @torch.no_grad()
    def ood_score(self, x: torch.Tensor) -> torch.Tensor:
        """Higher ⇒ more out-of-distribution (negative log-likelihood)."""
        return -self.log_prob(x)
