"""Runtime hardware configuration for OGAP training and inference paths."""
from __future__ import annotations

from typing import Any, Dict


def _get(cfg: Any, path: str, default: Any = None) -> Any:
    cur = cfg
    for key in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key, default)
        else:
            cur = getattr(cur, key, default)
    return cur


def configure_hardware(cfg: Any) -> Dict[str, Any]:
    """Apply opt-in hardware settings and return a compact summary.

    The packaged defaults request CPU behaviour, so this is a no-op for the
    legacy smoke path. When ``hardware.device`` is ``"cuda"`` and CUDA is
    available, it enables the H100 runtime switches required by the v9.1 plan.
    """
    import torch

    requested = str(_get(cfg, "hardware.device", "cpu") or "cpu").lower()
    cuda_requested = requested == "cuda"
    cuda_available = bool(torch.cuda.is_available())
    summary = {
        "requested_device": requested,
        "cuda_requested": cuda_requested,
        "cuda_available": cuda_available,
        "h100_runtime_enabled": False,
    }
    if not (cuda_requested and cuda_available):
        return summary

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    summary["h100_runtime_enabled"] = True
    summary["matmul_precision"] = "high"
    summary["allow_tf32"] = True
    summary["cudnn_benchmark"] = True
    return summary
