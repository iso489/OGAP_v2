"""Latent-ODE tracker for irregular glioma follow-up (RANO over time).

OGAP today predicts RANO-style measurements from a *single* scan (the student's
``rano_head``). But glioma management is longitudinal: follow-up MRIs arrive at
*irregular* intervals, and what clinicians need is the tumour-burden
**trajectory** and a forecast. The Neural ODE paper's §5 latent-ODE is built for
exactly this - "irregularly-sampled data such as medical records" - modelling a
continuous latent trajectory with a single ODE shared across all series and an
encoder that maps observations to an initial latent state.

``LatentODERANOTracker``:

* encodes a variable-length, irregularly-timed sequence of per-scan
  measurements with a (time-aware) GRU run backwards in time → q(z0 | obs),
* integrates the latent state with a learned ODE over per-series observation
  times,
* decodes each latent state back to measurements, and
* **extrapolates** to arbitrary future times - the clinically useful part
  (forecast tumour burden / RANO category at the next scan).

Trained by the standard latent-ODE ELBO (reconstruction + KL). ``times`` may be
a shared vector ``(T,)`` or a per-series matrix ``(B, T)`` for irregular clinical
follow-up schedules.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..numerics import ODESolverConfig, odeint


class _LatentODEFunc(nn.Module):
    """dz/dt = MLP(z) - the shared latent dynamics."""

    def __init__(self, latent_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, latent_dim),
        )
        self.nfe = 0

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        self.nfe += 1
        return self.net(z)


@dataclass
class LatentODEOutput:
    pred: torch.Tensor       # (B, T, obs_dim) reconstructed measurements
    z0_mean: torch.Tensor    # (B, latent)
    z0_logvar: torch.Tensor  # (B, latent)
    latents: torch.Tensor    # (T, B, latent)


class LatentODERANOTracker(nn.Module):
    def __init__(self, obs_dim: int, latent_dim: int = 8,
                 ode_hidden: int = 32, rec_hidden: int = 32,
                 solver: Optional[ODESolverConfig] = None) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        # time-aware encoder: input is [measurement, delta_t]
        self.enc = nn.GRU(obs_dim + 1, rec_hidden, batch_first=True)
        self.to_z0 = nn.Linear(rec_hidden, 2 * latent_dim)
        self.func = _LatentODEFunc(latent_dim, ode_hidden)
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, ode_hidden), nn.ReLU(),
            nn.Linear(ode_hidden, obs_dim),
        )
        self.solver = solver or ODESolverConfig(method="rk4", steps=4)

    # ---- encoder ----
    def _encoder_input(self, values: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        """Reverse-time GRU input ``[value, backward_delta_t]`` (latest obs first).

        The latent-ODE recognition net consumes observations from latest to earliest,
        so the delta-t feature at each step must be the time elapsed since the
        **previously processed** (later-in-time) observation - the BACKWARD gap on the
        reversed sequence. The previous code computed FORWARD gaps and only flipped the
        stacked tensor, which misaligned delta-t by one step under irregular sampling
        (each observation carried the gap to the *next* one processed, not the one just
        processed). Backward gaps fix it; with uniform spacing the two agree.
        """
        b, t, _ = values.shape
        values_rev = torch.flip(values, dims=[1])
        if times.ndim == 1:
            times_rev = torch.flip(times, dims=[0])
            dt = torch.zeros_like(times_rev)
            dt[1:] = times_rev[:-1] - times_rev[1:]          # positive backward gaps
            dt_feat = dt.view(1, t, 1).expand(b, t, 1)
        elif times.ndim == 2:
            if times.shape != (b, t):
                raise ValueError(f"times must have shape ({b}, {t}), got {tuple(times.shape)}")
            times_rev = torch.flip(times, dims=[1])
            dt = torch.zeros_like(times_rev)
            dt[:, 1:] = times_rev[:, :-1] - times_rev[:, 1:]
            dt_feat = dt.unsqueeze(-1)
        else:
            raise ValueError("times must have shape (T,) or (B,T)")
        return torch.cat([values_rev, dt_feat], dim=-1)

    def encode(self, values: torch.Tensor, times: torch.Tensor
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """values (B,T,obs_dim), times (T,) or (B,T) -> z0 stats."""
        inp_rev = self._encoder_input(values, times)
        out, _ = self.enc(inp_rev)
        h = out[:, -1]                      # summary after consuming the earliest obs
        mu, logvar = self.to_z0(h).chunk(2, dim=-1)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor,
                       sample: bool) -> torch.Tensor:
        if not sample:
            return mu
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    # ---- integrate + decode ----
    def _rollout(self, z0: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        self.func.nfe = 0
        if times.ndim == 1:
            return odeint(self.func, z0, times, self.solver)     # (T, B, latent)
        if times.ndim != 2 or times.shape[0] != z0.shape[0]:
            raise ValueError("times must have shape (T,) or (B,T)")
        series = []
        for b in range(z0.shape[0]):
            series.append(odeint(self.func, z0[b:b + 1], times[b], self.solver))
        return torch.cat(series, dim=1)                         # (T, B, latent)

    def forward(self, values: torch.Tensor, times: torch.Tensor,
                sample: bool = True) -> LatentODEOutput:
        mu, logvar = self.encode(values, times)
        z0 = self.reparameterize(mu, logvar, sample and self.training)
        zs = self._rollout(z0, times)
        pred = self.dec(zs).transpose(0, 1)                # (B, T, obs_dim)
        return LatentODEOutput(pred=pred, z0_mean=mu, z0_logvar=logvar, latents=zs)

    @torch.no_grad()
    def predict_future(self, values: torch.Tensor, times: torch.Tensor,
                       future_times: torch.Tensor) -> torch.Tensor:
        """Forecast measurements at ``future_times`` (>= times[-1])."""
        mu, logvar = self.encode(values, times)
        z0 = mu
        if times.ndim == 1:
            grid = torch.cat([times[-1:].to(future_times), future_times])
            zs = odeint(self.func, z0, grid, self.solver)[1:]  # drop the anchor point
            return self.dec(zs).transpose(0, 1)
        if future_times.ndim == 1:
            future_times = future_times.view(1, -1).expand(values.shape[0], -1)
        preds = []
        for b in range(values.shape[0]):
            grid = torch.cat([times[b, -1:].to(future_times), future_times[b]])
            zs = odeint(self.func, z0[b:b + 1], grid, self.solver)[1:]
            preds.append(self.dec(zs).transpose(0, 1))
        return torch.cat(preds, dim=0)

    # ---- training objective ----
    def elbo_loss(self, out: LatentODEOutput, target: torch.Tensor,
                  beta: float = 1.0) -> torch.Tensor:
        recon = nn.functional.mse_loss(out.pred, target, reduction="mean")
        kl = -0.5 * torch.mean(
            1 + out.z0_logvar - out.z0_mean.pow(2) - out.z0_logvar.exp()
        )
        return recon + beta * kl
