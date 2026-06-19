"""Test-time augmentation / adaptation (opt-in, gated by ``inference.tta``).

Two complementary inference-time robustness tools:

* :func:`flip_tta_predict` - average the model's softmax probabilities over axis
  flips (each flip is un-flipped before averaging), the standard segmentation
  TTA. For a flip-equivariant model it returns the plain prediction exactly.
* :func:`bn_adapt` - recalibrate BatchNorm running statistics to the test batch
  (the ``"bn_adapt"`` method in the config), a cheap source-free domain shift fix.

Both operate on a copy / under ``no_grad`` and leave the input model unchanged.
Pure ``torch``; CPU-friendly.
"""
from __future__ import annotations

import copy
import itertools
import logging
from typing import Sequence, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Normalization layers whose affine (weight/bias) params Tent adapts. Unlike
# BatchNorm these have no running stats, so bn_adapt cannot help - but their
# affine parameters can still be tuned at test time.
_NORM_TYPES = (
    nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
    nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
    nn.GroupNorm, nn.LayerNorm,
)


def _logits(out):
    """First element if the model returns a tuple/list (teacher heads), else out."""
    return out[0] if isinstance(out, (tuple, list)) else out


@torch.no_grad()
def flip_tta_predict(model: nn.Module, x: torch.Tensor,
                     spatial_dims: Sequence[int] = (2, 3, 4),
                     softmax: bool = True) -> torch.Tensor:
    """Flip-augmented prediction over all spatial-axis flip combinations.

    Args:
        model: maps ``(B, C, D, H, W)`` to logits (or a tuple whose first element
            is logits).
        x: input volume ``(B, C, D, H, W)``.
        spatial_dims: tensor dims to flip over (default the three spatial axes).
        softmax: average probabilities (True) or raw logits (False).

    Returns:
        The averaged prediction, same shape as the model output.
    """
    model.eval()
    dims_tuple: Tuple[int, ...] = tuple(int(d) for d in spatial_dims)
    flip_sets = [
        combo
        for r in range(len(dims_tuple) + 1)
        for combo in itertools.combinations(dims_tuple, r)
    ]
    augmented = torch.cat(
        [torch.flip(x, dims) if dims else x for dims in flip_sets],
        dim=0,
    )
    out = _logits(model(augmented))
    if softmax:
        out = torch.softmax(out, dim=1)
    chunks = out.chunk(len(flip_sets), dim=0)
    restored = [
        torch.flip(chunk, dims) if dims else chunk
        for chunk, dims in zip(chunks, flip_sets)
    ]
    return torch.stack(restored, dim=0).mean(dim=0)


def bn_adapt(model: nn.Module, x: torch.Tensor, steps: int = 1,
             momentum=None) -> nn.Module:
    """Adapt BatchNorm running statistics to the test batch.

    Returns a deep copy of ``model`` whose BatchNorm layers have had their
    running mean/var updated by forwarding ``x`` in train mode ``steps`` times;
    the original model is untouched. A model without BatchNorm is returned as a
    copy unchanged.

    Args:
        model: the model to adapt.
        x: a representative test batch ``(B, C, D, H, W)``.
        steps: number of forward passes used to update the running statistics.
        momentum: optional BatchNorm momentum override for the adaptation.
    """
    adapted = copy.deepcopy(model)
    bns = [m for m in adapted.modules()
           if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))]
    if not bns:
        logger.warning(
            "bn_adapt: model has no BatchNorm layers (OGAP uses GroupNorm), so "
            "recalibrating running statistics is a no-op. Use tent_adapt() for "
            "norm-affine test-time adaptation on GroupNorm models. [audit S1-C]"
        )
        return adapted
    for m in bns:
        m.train()
        if momentum is not None:
            m.momentum = momentum
    with torch.no_grad():
        for _ in range(max(1, int(steps))):
            adapted(x)
    adapted.eval()
    return adapted


def tent_adapt(model: nn.Module, x: torch.Tensor, steps: int = 1,
               lr: float = 1e-3) -> nn.Module:
    """Tent test-time adaptation (Wang et al., ICLR 2021) - GroupNorm-compatible.

    Minimises the Shannon entropy of the model's prediction by gradient updates
    to the **affine parameters of the normalization layers only** (all other
    weights frozen). Unlike :func:`bn_adapt` this needs no running statistics, so
    it works for OGAP's GroupNorm networks where ``bn_adapt`` is inert. [audit S1-C]

    Returns a deep copy whose norm affine params have been adapted to ``x``; the
    input model is left untouched. A model with no normalization affine params is
    returned (copied) unchanged, with a warning.

    Args:
        model: maps ``(B, C, D, H, W)`` to logits (or a tuple whose first element
            is logits).
        x: a representative test batch.
        steps: number of entropy-minimization gradient steps.
        lr: SGD learning rate for the affine parameters.
    """
    adapted = copy.deepcopy(model)
    affine = []
    for m in adapted.modules():
        if isinstance(m, _NORM_TYPES):
            for pname in ("weight", "bias"):
                p = getattr(m, pname, None)
                if isinstance(p, nn.Parameter):
                    affine.append(p)
    affine_ids = {id(p) for p in affine}
    for p in adapted.parameters():
        p.requires_grad_(id(p) in affine_ids)
    if not affine:
        logger.warning("tent_adapt: model has no normalization affine params; "
                       "returning unchanged.")
        adapted.eval()
        return adapted
    opt = torch.optim.SGD(affine, lr=float(lr))
    adapted.eval()  # eval mode: GroupNorm is train/eval-identical; grad still flows
    for _ in range(max(1, int(steps))):
        opt.zero_grad(set_to_none=True)
        logp = torch.log_softmax(_logits(adapted(x)), dim=1)
        entropy = -(logp.exp() * logp).sum(dim=1).mean()
        entropy.backward()
        opt.step()
    for p in adapted.parameters():
        p.requires_grad_(True)  # restore default grad flags on the returned copy
    return adapted
