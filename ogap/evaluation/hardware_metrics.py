"""Inference-efficiency metrics for the accuracy/efficiency trade-off table.

Thin wrapper over :mod:`ogap.nas.hardware` (the in-process cost model used by the
architecture search) returning the efficiency fields commonly reported for a
deployable model: parameter count, INT8 size, and measured forward latency.
FLOPs are reported when an optional counter (``fvcore`` or ``thop``) is
installed, otherwise NaN.
"""
from __future__ import annotations

import logging
import time
from typing import Dict, Tuple

import torch
import torch.nn as nn

from ..nas.hardware import cost_of

logger = logging.getLogger(__name__)


def _try_flops_g(model: nn.Module, input_shape: Tuple[int, ...]) -> float:
    try:
        from fvcore.nn import FlopCountAnalysis
        return float(FlopCountAnalysis(model.eval(), torch.randn(*input_shape)).total()) / 1e9
    except Exception:
        pass
    try:
        from thop import profile
        flops, _ = profile(model.eval(), inputs=(torch.randn(*input_shape),), verbose=False)
        return float(flops) / 1e9
    except Exception:
        return float("nan")


def measure_hardware_metrics(model: nn.Module,
                             input_shape: Tuple[int, ...] = (1, 4, 128, 128, 128),
                             device: str = "cpu", n_warmup: int = 3, n_runs: int = 10
                             ) -> Dict[str, float]:
    """Profile a model's inference efficiency on one dummy volume.

    Args:
        model: the network to profile.
        input_shape: dummy input shape ``(B, C, D, H, W)``.
        device: torch device string.
        n_warmup, n_runs: latency warm-up and timed-run counts.

    Returns:
        Dict with ``params_M``, ``flops_G`` (NaN if no counter), ``int8_size_mb``,
        ``latency_mean_ms``, ``latency_std_ms``, ``peak_mem_mb`` (NaN on CPU).
    """
    cost = cost_of(model, input_shape, runs=n_runs, device=device)
    model = model.to(device).eval()
    x = torch.randn(*input_shape, device=device)
    is_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    with torch.no_grad():
        for _ in range(n_warmup):
            model(x)
        if is_cuda:
            torch.cuda.synchronize(x.device)
        times = []
        for _ in range(n_runs):
            if is_cuda:
                torch.cuda.synchronize(x.device)
            t0 = time.perf_counter()
            model(x)
            if is_cuda:
                torch.cuda.synchronize(x.device)
            times.append((time.perf_counter() - t0) * 1000.0)
    t = torch.tensor(times)
    peak = float("nan")
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            model(x)
        peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
    return {
        "params_M": cost.params / 1e6,
        "flops_G": _try_flops_g(model, input_shape),
        "int8_size_mb": cost.int8_mb,
        "latency_mean_ms": float(t.mean()),
        "latency_std_ms": float(t.std(unbiased=False)),
        "peak_mem_mb": peak,
    }
