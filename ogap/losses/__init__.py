"""OGAP loss functions (task losses + knowledge-distillation assembly).

The ``"legacy"`` task-loss type delegates to the existing loss classes in
:mod:`ogap.legacy` *by import* (not a reimplementation), so legacy behaviour is
bit-exact by construction. All other variants are additive and config-driven.
"""
from __future__ import annotations

from .region_aware import build_task_loss

__all__ = ["build_task_loss"]
