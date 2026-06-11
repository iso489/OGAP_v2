"""Domain-adaptation / semi-supervised utilities for the low-field shift.

Closes the in-silico-only validation gap by adapting to real low-field-adjacent data
(e.g. BraTS-Africa) without labels: confidence-filtered self-training (pseudo-labels)
and FixMatch-style weak/strong consistency.
"""
from .self_training import (
    make_pseudo_labels,
    pseudo_label_loss,
    consistency_loss,
    SelfTrainingConfig,
)

__all__ = [
    "make_pseudo_labels",
    "pseudo_label_loss",
    "consistency_loss",
    "SelfTrainingConfig",
]
