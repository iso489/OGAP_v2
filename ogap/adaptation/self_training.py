"""Confidence-filtered self-training + consistency regularisation (low-field DA).

The paper's central risk is that the low-field claim is validated *in silico* only.
A principled way to add real-data evidence without native 64 mT labels is
semi-supervised domain adaptation on the real low-field-adjacent cohort
(BraTS-Africa, 1.5 T): generate pseudo-labels from the model's own confident
predictions and/or enforce weak↔strong augmentation consistency (FixMatch, Sohn
et al. 2020). These are the building blocks the cluster training loop calls on the
unlabelled / weakly-labelled stream; they are pure, device-native, and CPU-tested.

All functions take ``logits`` of shape ``(B, C, D, H, W)``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SelfTrainingConfig:
    """Knobs for confidence-filtered self-training."""

    enabled: bool = False
    confidence_threshold: float = 0.9   # keep a voxel only if max softmax >= this
    weight: float = 1.0                 # weight of the self-training loss term
    rampup_epochs: int = 20             # linearly ramp the weight over the first N epochs
    consistency: bool = True            # also use weak↔strong consistency
    ignore_index: int = -100


def make_pseudo_labels(
    logits: torch.Tensor, confidence_threshold: float = 0.9,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Hard pseudo-labels + a confidence mask from a (teacher) prediction.

    Returns ``(pseudo_label, mask)`` where ``pseudo_label`` is the argmax class and
    ``mask`` is True only where the max softmax probability ≥ ``confidence_threshold``
    (low-confidence voxels are excluded from the loss). Computed in fp32 for a stable
    threshold under bf16.
    """
    with torch.no_grad():
        probs = F.softmax(logits.float(), dim=1)
        conf, pseudo = probs.max(dim=1)
        mask = conf >= float(confidence_threshold)
    return pseudo.long(), mask


def pseudo_label_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    confidence_threshold: float = 0.9,
) -> torch.Tensor:
    """Cross-entropy of the student against the teacher's *confident* pseudo-labels.

    Voxels below the confidence threshold contribute nothing (masked out). Returns a
    scalar; zero when no voxel is confident enough (cold start) so it never destabilises.
    """
    pseudo, mask = make_pseudo_labels(teacher_logits, confidence_threshold)
    if not bool(mask.any()):
        return student_logits.sum() * 0.0
    ce = F.cross_entropy(student_logits.float(), pseudo, reduction="none")
    return (ce * mask.float()).sum() / mask.float().sum().clamp_min(1.0)


def consistency_loss(
    logits_weak: torch.Tensor,
    logits_strong: torch.Tensor,
    confidence_threshold: float = 0.9,
) -> torch.Tensor:
    """FixMatch weak↔strong consistency: the strongly-augmented view is pushed toward
    the confident pseudo-label of the weakly-augmented view (stop-grad on the weak view).
    """
    pseudo, mask = make_pseudo_labels(logits_weak, confidence_threshold)
    if not bool(mask.any()):
        return logits_strong.sum() * 0.0
    ce = F.cross_entropy(logits_strong.float(), pseudo, reduction="none")
    return (ce * mask.float()).sum() / mask.float().sum().clamp_min(1.0)


def rampup_weight(cfg: SelfTrainingConfig, epoch: int) -> float:
    """Linear ramp-up of the self-training weight (avoids confirmation bias early)."""
    if not cfg.enabled or cfg.rampup_epochs <= 0:
        return cfg.weight if cfg.enabled else 0.0
    return cfg.weight * min(1.0, max(0, epoch) / float(cfg.rampup_epochs))
