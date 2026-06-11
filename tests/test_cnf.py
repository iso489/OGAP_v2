
import math

import torch
import torch.nn as nn

from ogap.numerics import ODESolverConfig
from ogap.ood import CNFDensityEstimator


def test_log_prob_shape_and_finite():
    est = CNFDensityEstimator(dim=4, hidden=32)
    x = torch.randn(8, 4)
    lp = est.log_prob(x)
    assert lp.shape == (8,) and torch.isfinite(lp).all()


def test_nll_is_differentiable():
    est = CNFDensityEstimator(dim=4, hidden=32)
    loss = est.nll(torch.randn(16, 4))
    loss.backward()  # double-backward through the augmented trace dynamics
    assert all(p.grad is not None for p in est.field.parameters())


def test_fit_returns_metadata():
    torch.manual_seed(0)
    est = CNFDensityEstimator(dim=4, hidden=16)
    meta = est.fit(torch.randn(8, 4), steps=2, batch_size=4)
    assert meta["n_embeddings"] == 8
    assert meta["dim"] == 4
    assert meta["steps"] == 2
    assert torch.isfinite(torch.tensor(meta["final_nll"]))


def test_ood_separation_after_fit():
    torch.manual_seed(0)
    est = CNFDensityEstimator(dim=4, hidden=32)
    # in-distribution cluster centred at +2; OOD centred at -2
    indist = torch.randn(128, 4) * 0.4 + 2.0
    ood = torch.randn(64, 4) * 0.4 - 2.0
    opt = torch.optim.Adam(est.parameters(), lr=5e-3)
    est.train()
    for _ in range(150):
        opt.zero_grad()
        loss = est.nll(indist)
        loss.backward(); opt.step()
    est.eval()
    s_in = est.ood_score(indist).mean().item()
    s_out = est.ood_score(ood).mean().item()
    assert s_out > s_in  # OOD scored as more anomalous


def test_log_prob_matches_analytic_affine_flow():
    """Pin the SIGN of the FFJORD log-det term against a closed-form value.

    For a linear field f(t, z) = a*z the divergence tr(df/dz) = a*dim is
    *constant*, so the Euler solver integrates the log-det term exactly and the
    instantaneous change of variables gives a closed form with no discretisation
    error:

        log p(x) = log N(z1) + integral_0^1 tr dt = log N(z1) + a*dim,

    where z1 is the flow image of x (also exact under Euler for a linear field:
    z1 = x*(1 + a/steps)**steps). The buggy sign returns log N(z1) - a*dim, so
    this test fails loudly unless the log-det sign is correct.
    """
    a = 0.1
    dim = 2
    steps = 10

    class _LinearField(nn.Module):
        def forward(self, t, z):  # noqa: D401 - tiny test field
            return a * z

    est = CNFDensityEstimator(
        dim=dim, hidden=8, solver=ODESolverConfig(method="euler", steps=steps)
    )
    est.field = _LinearField()

    x = torch.tensor([[0.5, -1.0], [1.5, 0.25]])
    z1 = x * (1.0 + a / steps) ** steps
    base = -0.5 * (z1.pow(2).sum(dim=-1) + dim * math.log(2 * math.pi))
    expected = base + a * dim  # integral_0^1 tr dt = a*dim

    got = est.log_prob(x)
    assert torch.allclose(got, expected, atol=1e-4, rtol=1e-4), (got, expected)


def test_hutchinson_trace_estimator_is_available_and_finite():
    """The CNF must offer a Hutchinson trace estimator for large feature dims,
    not only the O(dim) exact trace. [audit S3-C]"""
    torch.manual_seed(0)
    est = CNFDensityEstimator(dim=16, hidden=16, trace_estimator="hutchinson",
                              hutchinson_samples=4)
    lp = est.log_prob(torch.randn(8, 16))
    assert lp.shape == (8,) and torch.isfinite(lp).all()


def test_hutchinson_trace_unbiased_vs_exact_for_linear_field():
    """For a linear field f=a*z, tr(df/dz)=a*dim is constant, so the Hutchinson
    estimate of the log-det term must match the exact value in expectation."""
    a, dim, steps = 0.1, 8, 10

    class _LinearField(nn.Module):
        def forward(self, t, z):
            return a * z

    x = torch.randn(64, dim)
    exact = CNFDensityEstimator(dim=dim, hidden=8,
                                solver=ODESolverConfig(method="euler", steps=steps))
    exact.field = _LinearField()
    torch.manual_seed(0)
    hutch = CNFDensityEstimator(dim=dim, hidden=8, trace_estimator="hutchinson",
                                hutchinson_samples=64,
                                solver=ODESolverConfig(method="euler", steps=steps))
    hutch.field = _LinearField()
    lp_exact = exact.log_prob(x).mean().detach()
    lp_hutch = hutch.log_prob(x).mean().detach()
    assert abs(float(lp_exact - lp_hutch)) < 0.5, (lp_exact, lp_hutch)
