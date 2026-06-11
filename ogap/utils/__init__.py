"""Shared OGAP utilities."""
from __future__ import annotations

from .hardware import configure_hardware
from .reproducibility import set_all_seeds

__all__ = ["configure_hardware", "set_all_seeds"]
