
import torch
from ogap.longitudinal import LatentODERANOTracker


def _data(b=2, t=5, d=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    times = torch.cumsum(torch.rand(t, generator=g) + 0.1, dim=0)  # irregular, increasing
    # smooth trajectory: linear-in-time per channel + offset
    slope = torch.randn(b, 1, d, generator=g)
    vals = times.view(1, t, 1) * slope + torch.randn(b, 1, d, generator=g)
    return vals, times


def test_forward_shapes_and_kl():
    m = LatentODERANOTracker(obs_dim=3, latent_dim=6, ode_hidden=16, rec_hidden=16)
    vals, times = _data()
    out = m(vals, times, sample=False)
    assert out.pred.shape == (2, 5, 3)
    loss = m.elbo_loss(out, vals)
    assert torch.isfinite(loss)
    loss.backward()  # end-to-end differentiable through the ODE solve


def test_extrapolation_shape():
    m = LatentODERANOTracker(obs_dim=3, latent_dim=6).eval()
    vals, times = _data()
    fut = times[-1] + torch.tensor([0.5, 1.0, 1.5])
    pred = m.predict_future(vals, times, fut)
    assert pred.shape == (2, 3, 3)


def test_per_series_irregular_times():
    m = LatentODERANOTracker(obs_dim=3, latent_dim=6, ode_hidden=16, rec_hidden=16).eval()
    vals, shared = _data(b=2, t=5, d=3, seed=4)
    times = torch.stack([shared, shared * 1.4 + 0.2], dim=0)
    out = m(vals, times, sample=False)
    assert out.pred.shape == vals.shape
    fut = torch.stack([times[0, -1] + torch.tensor([0.5, 1.0]),
                       times[1, -1] + torch.tensor([0.25, 0.75])], dim=0)
    pred = m.predict_future(vals, times, fut)
    assert pred.shape == (2, 2, 3)


def test_can_fit_a_trajectory():
    torch.manual_seed(0)
    m = LatentODERANOTracker(obs_dim=3, latent_dim=8, ode_hidden=32, rec_hidden=32)
    vals, times = _data(b=4, t=6, d=3, seed=1)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    m.train()
    first = None
    for _ in range(120):
        opt.zero_grad()
        out = m(vals, times, sample=False)
        loss = m.elbo_loss(out, vals, beta=0.0)  # pure reconstruction for the fit test
        loss.backward(); opt.step()
        if first is None:
            first = loss.item()
    assert loss.item() < first * 0.6  # learns the trajectory
