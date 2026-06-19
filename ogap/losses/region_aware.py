"""Region-aware task losses for multi-class 3D tumour segmentation.

A small factory (:func:`build_task_loss`) selects among task-loss variants by a
config string. The ``"legacy"`` variant delegates to the existing loss in
:mod:`ogap.legacy` so prior behaviour is reproduced exactly; the other variants
are new region-weighted formulations that weight the smaller, harder subregions
more heavily.

Label convention (consistent with the rest of OGAP): 0 = background, 1/2/3 =
foreground classes that compose the overlapping subregions WT = {1,2,3},
TC = {1,3}, ET = {3}.

All losses take ``logits`` of shape ``(B, C, D, H, W)`` and integer ``targets``
of shape ``(B, D, H, W)`` and return a scalar tensor.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_REGION_CLASSES: Dict[str, Sequence[int]] = {"WT": (1, 2, 3), "TC": (1, 3), "ET": (3,)}
_DEFAULT_REGION_WEIGHTS: Dict[str, float] = {"WT": 1.0, "TC": 2.0, "ET": 4.0}
_DEFAULT_FOCAL_GAMMA: Dict[str, float] = {"WT": 1.0, "TC": 1.5, "ET": 2.0}


def _cfg_get(config: Any, path: Sequence[str], default: Any) -> Any:
    """Read a nested key from a dict-or-namespace config, returning default."""
    node = config
    for key in path:
        if node is None:
            return default
        node = node.get(key, None) if isinstance(node, dict) else getattr(node, key, None)
    return default if node is None else node


def _region_soft_masks(probs: torch.Tensor, target: torch.Tensor, region: str):
    """Return (pred_prob, target_binary) for an overlapping subregion."""
    classes = _REGION_CLASSES[region]
    pred = probs[:, classes, ...].sum(dim=1)
    tgt = torch.zeros_like(target, dtype=probs.dtype)
    for c in classes:
        tgt = tgt + (target == c).to(probs.dtype)
    tgt = tgt.clamp_(0, 1)
    return pred, tgt


class RegionWeightedGDL(nn.Module):
    """Soft Dice loss computed independently per subregion, then weighted.

    L = sum_r w_r * (1 - Dice_r); smaller regions get larger default weights
    (volume-inverse prior). Computation is forced to float32 so the loss is
    unbiased under bf16 autocast (matches the legacy loss's fp32-reduction fix).
    """

    def __init__(
        self,
        region_weights: Optional[Dict[str, float]] = None,
        smooth: float = 1e-5,
        focal_gamma: Optional[Dict[str, float]] = None,
        use_focal: bool = False,
    ) -> None:
        super().__init__()
        self.region_weights = dict(region_weights or _DEFAULT_REGION_WEIGHTS)
        self.smooth = float(smooth)
        self.focal_gamma = dict(focal_gamma or _DEFAULT_FOCAL_GAMMA)
        self.use_focal = bool(use_focal)

    def update_weights(self, dice_by_region: Dict[str, float], eps: float = 1e-3) -> None:
        """Dynamic region weighting: ``w_r = 1 / (Dice_r + eps)``.

        Optional contract for the training loop - call after a validation pass to
        up-weight under-performing regions. Unknown regions are ignored.
        """
        for region, dice in dice_by_region.items():
            if region in self.region_weights:
                self.region_weights[region] = 1.0 / (float(dice) + eps)
        logger.debug("region weights updated -> %s", self.region_weights)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=logits.device.type, enabled=False):
            probs = F.softmax(logits.float(), dim=1)
            total = probs.new_zeros(())
            wsum = 0.0
            for region, weight in self.region_weights.items():
                pred, tgt = _region_soft_masks(probs, target, region)
                dims = tuple(range(1, pred.ndim))
                inter = (pred * tgt).sum(dim=dims)
                denom = pred.sum(dim=dims) + tgt.sum(dim=dims)
                dice = (2.0 * inter + self.smooth) / (denom + self.smooth)
                region_loss = (1.0 - dice).mean()
                if self.use_focal:
                    # Focal-Tversky (Abraham & Khan 2019): up-weight under-performing
                    # regions by exponentiating the (1 - overlap) term. The previous
                    # code instead shrank ``pred`` by (1-pred)**gamma INSIDE the Dice,
                    # which drove a confident-correct region's Dice toward 0 (loss -> 1)
                    # and pushed foreground probabilities DOWN - the opposite of focal.
                    gamma = self.focal_gamma.get(region, 1.0)
                    region_loss = region_loss ** gamma
                total = total + weight * region_loss
                wsum += weight
        return total / max(wsum, 1e-8)


class RegionSigmoidBatchDiceLoss(nn.Module):
    """Independent-sigmoid region loss + **batch** Dice (nnU-Net region-based training).

    The BraTS evaluation regions WT⊇TC⊇ET are nested and overlapping, so the SOTA
    objective treats each as an *independent* binary problem rather than summing
    softmax classes (which couples them through the softmax normalisation). The
    deployed head still emits 4-class logits (checkpoint-frozen), so for each region R
    with class set C_R the equivalent binary logit is the exact

        logit P(R) = logsumexp(z_{C_R}) - logsumexp(z_{complement})

    (so ``sigmoid(logit_R) = P(R) = Σ_{c∈C_R} softmax(z)_c``). The loss is then
    ``BCEWithLogits`` per region + **batch** soft Dice (summed over batch *and* spatial,
    one Dice per region over the whole batch), which is far more stable than per-sample
    Dice for the small, often-absent ET region. Forced to fp32 under bf16 autocast.
    """

    def __init__(
        self,
        region_weights: Optional[Dict[str, float]] = None,
        bce_weight: float = 0.5,
        smooth: float = 1e-5,
    ) -> None:
        super().__init__()
        self.region_weights = dict(region_weights or _DEFAULT_REGION_WEIGHTS)
        self.bce_weight = float(bce_weight)
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        with torch.autocast(device_type=logits.device.type, enabled=False):
            z = logits.float()
            total = z.new_zeros(())
            wsum = 0.0
            for region, weight in self.region_weights.items():
                classes = list(_REGION_CLASSES[region])
                complement = [c for c in range(num_classes) if c not in classes]
                logit_r = torch.logsumexp(z[:, classes], dim=1)
                if complement:
                    logit_r = logit_r - torch.logsumexp(z[:, complement], dim=1)
                tgt_r = torch.zeros_like(logit_r)
                for c in classes:
                    tgt_r = tgt_r + (target == c).to(logit_r.dtype)
                tgt_r = tgt_r.clamp_(0, 1)
                bce = F.binary_cross_entropy_with_logits(logit_r, tgt_r)
                prob_r = torch.sigmoid(logit_r)
                inter = (prob_r * tgt_r).sum()                       # batch-level
                den = prob_r.sum() + tgt_r.sum()
                dice = (2.0 * inter + self.smooth) / (den + self.smooth)
                total = total + weight * ((1.0 - dice) + self.bce_weight * bce)
                wsum += weight
        return total / max(wsum, 1e-8)


class CompositeLoss(nn.Module):
    """Region-weighted GDL + weighted cross-entropy."""

    def __init__(
        self,
        region_weights: Optional[Dict[str, float]] = None,
        dice_weight: float = 0.5,
        ce_weight: float = 0.5,
        class_weights: Optional[torch.Tensor] = None,
        smooth: float = 1e-5,
    ) -> None:
        super().__init__()
        self.gdl = RegionWeightedGDL(region_weights, smooth)
        self.dice_weight = float(dice_weight)
        self.ce_weight = float(ce_weight)
        self.class_weights = class_weights

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        cw = self.class_weights
        if cw is not None:
            cw = cw.to(logits.device, logits.dtype)
        ce = F.cross_entropy(logits, target, weight=cw)
        return self.dice_weight * self.gdl(logits, target) + self.ce_weight * ce


class _LegacyTaskLoss(nn.Module):
    """Adapter delegating to the existing OGAP loss for bit-exact parity."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        from ogap.legacy import DiceCELoss  # lazy: avoid heavy import at module load
        self._impl = DiceCELoss(num_classes)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self._impl(logits, target)


def build_task_loss(config: Any) -> nn.Module:
    """Construct the task loss selected by ``config.loss.task.type``.

    Args:
        config: dict or namespace exposing ``loss.task.type`` and optional
            ``loss.task.{region_weights,focal_gamma,smooth,num_classes,
            composite_dice_weight,composite_ce_weight}``.

    Returns:
        An ``nn.Module`` with ``forward(logits, targets) -> scalar``.

    Raises:
        ValueError: if ``type`` is not one of
            ``legacy | region_weighted_gdl | focal_dice | composite``.
    """
    ttype = str(_cfg_get(config, ("loss", "task", "type"), "legacy"))
    num_classes = int(_cfg_get(config, ("loss", "task", "num_classes"),
                               _cfg_get(config, ("num_classes",), 4)))
    smooth = float(_cfg_get(config, ("loss", "task", "smooth"), 1e-5))
    region_weights = _cfg_get(config, ("loss", "task", "region_weights"), None)
    focal_gamma = _cfg_get(config, ("loss", "task", "focal_gamma"), None)

    if ttype == "legacy":
        return _LegacyTaskLoss(num_classes)
    if ttype == "region_weighted_gdl":
        return RegionWeightedGDL(region_weights, smooth)
    if ttype == "focal_dice":
        return RegionWeightedGDL(region_weights, smooth, focal_gamma, use_focal=True)
    if ttype == "composite":
        return CompositeLoss(
            region_weights,
            dice_weight=float(_cfg_get(config, ("loss", "task", "composite_dice_weight"), 0.5)),
            ce_weight=float(_cfg_get(config, ("loss", "task", "composite_ce_weight"), 0.5)),
            smooth=smooth,
        )
    if ttype == "region_sigmoid":
        return RegionSigmoidBatchDiceLoss(
            region_weights,
            bce_weight=float(_cfg_get(config, ("loss", "task", "region_sigmoid_bce_weight"), 0.5)),
            smooth=smooth,
        )
    raise ValueError(
        f"Unknown loss.task.type {ttype!r}; expected one of "
        "legacy | region_weighted_gdl | focal_dice | composite | region_sigmoid."
    )
