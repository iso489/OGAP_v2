"""Inference-time utilities (opt-in).

Currently test-time augmentation / adaptation; see :mod:`ogap.inference.tta`.
"""
from __future__ import annotations

from .tta import bn_adapt, flip_tta_predict

__all__ = ["bn_adapt", "flip_tta_predict"]
