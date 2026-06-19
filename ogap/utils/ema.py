"""Exponential moving average (EMA) of model weights.

A weight EMA (a.k.a. mean-teacher / model averaging) keeps a slowly-updated copy of
the weights and evaluates / deploys *that* copy. It is one of the most reliable,
near-free accuracy gains in segmentation (used by recent nnU-Net / BraTS winners):
the averaged weights sit in a flatter, lower-variance region of the loss surface than
the last SGD iterate. Default OFF - enabled via ``training.ema.enabled``.

Design notes
------------
* The EMA module is a ``deepcopy`` of the *unwrapped* model, so its ``state_dict`` keys
  match the raw model exactly (checkpoint-compatible - it can be saved as
  ``best_*.pth`` and loaded ``strict=True``).
* Build the EMA **before** ``torch.compile`` / DDP wrapping; :func:`_deep_unwrap`
  strips a compile (``_orig_mod``) or DDP (``.module``) wrapper at update time so the
  source weights always line up with the EMA buffers.
* Both floating-point parameters/buffers (EMA-averaged) and integer buffers
  (e.g. ``num_batches_tracked`` - copied verbatim) are handled.
"""
from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn


def _deep_unwrap(model: nn.Module) -> nn.Module:
    """Return the underlying module beneath DDP and/or ``torch.compile`` wrappers.

    Loops so it handles either nesting order - ``compile(DDP(m))`` or ``DDP(compile(m))``
    - and only strips those two known wrappers (never an OGAP channel-adapter wrapper,
    which is neither a ``DistributedDataParallel`` instance nor has ``_orig_mod``).
    """
    prev = None
    while prev is not model:
        prev = model
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
        if hasattr(model, "_orig_mod") and isinstance(getattr(model, "_orig_mod"), nn.Module):
            model = model._orig_mod
    return model


class ModelEMA:
    """Maintains an exponential moving average of a model's weights."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = float(decay)
        self.ema_module = copy.deepcopy(_deep_unwrap(model)).eval()
        for p in self.ema_module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        src = _deep_unwrap(model).state_dict()
        d = self.decay
        for key, ema_v in self.ema_module.state_dict().items():
            src_v = src[key]
            if ema_v.dtype.is_floating_point:
                ema_v.mul_(d).add_(src_v.detach().to(ema_v.dtype), alpha=1.0 - d)
            else:
                ema_v.copy_(src_v)

    def module(self) -> nn.Module:
        """The EMA model (eval-mode), for validation / inference."""
        return self.ema_module

    def state_dict(self) -> Any:
        """Checkpoint-compatible state_dict (same keys as the unwrapped source model)."""
        return self.ema_module.state_dict()


def build_model_ema(model: nn.Module, v91_cfg: Any) -> "ModelEMA | None":
    """Build a :class:`ModelEMA` when ``training.ema.enabled``, else ``None``.

    Reads ``training.ema.{enabled,decay}`` from the v9.1 config namespace; returns
    ``None`` (no EMA, byte-identical legacy path) when the config is absent or off.
    """
    try:
        ema_cfg = v91_cfg.training.ema
        if not bool(getattr(ema_cfg, "enabled", False)):
            return None
        # Honour an explicit decay (including a deliberate 0.0 = "EMA equals live
        # weights" ablation); only fall back to the default when it is absent/null.
        # `... or 0.999` would silently overwrite decay=0.0 with 0.999.
        decay = getattr(ema_cfg, "decay", 0.999)
        decay = 0.999 if decay is None else float(decay)
    except AttributeError:
        return None
    return ModelEMA(model, decay=decay)
