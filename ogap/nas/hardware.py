"""Hardware cost model - the second objective for OGAP's constrained NAS.

OGAP's mission is LMIC deployment, so accuracy alone is the wrong objective. The
survey's constrained-NAS section (White et al. 2023, §6.2) optimises accuracy
*subject to* a hardware budget. OGAP already measures real deployment cost in
``benchmark_hardware`` (§18a of the monolith); this module gives the search loop
a fast, in-process estimate of the same quantities so candidates can be ranked
without a full cluster benchmark:

* parameter count and FP32 / INT8 model size (INT8 is the deployed format),
* measured single-volume forward latency on the *current* host CPU (a stand-in
  for an LMIC edge CPU; on a target box this measures that box directly),
* a coarse peak-activation estimate (RAM pressure on weak hardware).

These are estimates for *search*; the existing cluster ``benchmark_hardware``
remains the source of truth for reported numbers.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from typing import Dict, Tuple

import torch
import torch.nn as nn


@dataclass
class HardwareCost:
    params: int
    fp32_mb: float
    int8_mb: float
    latency_ms: float          # median single-volume forward latency on this host
    peak_activation_mb: float  # rough largest intermediate tensor (RAM proxy)

    def to_dict(self) -> Dict:
        return asdict(self)


def _model_size_mb(model: nn.Module, bytes_per_param: float) -> float:
    n = sum(p.numel() for p in model.parameters())
    return n * bytes_per_param / (1024 ** 2)


@torch.no_grad()
def _peak_activation_mb(model: nn.Module, x: torch.Tensor) -> float:
    """Largest single intermediate activation seen during a forward pass (MB)."""
    peak = {"v": 0.0}
    hooks = []

    def hook(_m, _inp, out):
        outs = out if isinstance(out, (tuple, list)) else (out,)
        for o in outs:
            if torch.is_tensor(o):
                mb = o.element_size() * o.nelement() / (1024 ** 2)
                peak["v"] = max(peak["v"], mb)

    for m in model.modules():
        if len(list(m.children())) == 0:  # leaf modules only
            hooks.append(m.register_forward_hook(hook))
    try:
        model(x)
    finally:
        for h in hooks:
            h.remove()
    return float(peak["v"])


@torch.no_grad()
def measure_latency(model: nn.Module, x: torch.Tensor,
                    runs: int = 10, warmup: int = 3) -> float:
    """Median forward latency in ms on the host CPU/GPU of ``x``."""
    model.eval()
    is_cuda = x.device.type == "cuda"
    for _ in range(warmup):
        model(x)
    if is_cuda:
        torch.cuda.synchronize(x.device)
    times = []
    for _ in range(runs):
        if is_cuda:
            torch.cuda.synchronize(x.device)
        t0 = time.perf_counter()
        model(x)
        if is_cuda:
            torch.cuda.synchronize(x.device)
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return times[len(times) // 2]


def cost_of(model: nn.Module, input_shape: Tuple[int, ...],
            *, runs: int = 8, device: str = "cpu") -> HardwareCost:
    """Profile ``model`` on a single dummy volume of ``input_shape`` (B,C,D,H,W)."""
    model = model.to(device).eval()
    x = torch.randn(*input_shape, device=device)
    return HardwareCost(
        params=int(sum(p.numel() for p in model.parameters())),
        fp32_mb=_model_size_mb(model, 4.0),
        int8_mb=_model_size_mb(model, 1.0),
        latency_ms=measure_latency(model, x, runs=runs),
        peak_activation_mb=_peak_activation_mb(model, x),
    )


def meets_budget(cost: HardwareCost, *, max_int8_mb: float = 1e9,
                 max_latency_ms: float = 1e9,
                 max_peak_activation_mb: float = 1e9) -> bool:
    """Hard-constraint check for constrained NAS (§6.2)."""
    return (
        cost.int8_mb <= max_int8_mb
        and cost.latency_ms <= max_latency_ms
        and cost.peak_activation_mb <= max_peak_activation_mb
    )
