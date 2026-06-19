"""Zero-cost performance proxies for cheap architecture ranking.

Training a 3D segmentation model to convergence to score *one* architecture is
infeasible across a search. The NAS survey (White et al. 2023, §5) surveys
"performance estimation" speedups; the cheapest are **zero-cost proxies** -
scalar scores from a single forward/backward on one minibatch that correlate
with final accuracy. They are the right tool for OGAP's expensive-3D / small-
data regime.

Implemented (all robust to GELU activations, unlike ReLU-code proxies):

* ``param_count``  - capacity prior.
* ``grad_norm``    - total gradient norm from one backward (trainability).
* ``snip``         - connection sensitivity ``sum|g ⊙ w|`` (Lee et al. 2019).
* ``synflow``      - data-independent ``sum|g ⊙ w|`` on |w| with a ones input
                     (Tanaka et al. 2020); avoids label leakage entirely.
* ``jacob_cov``    - input-output Jacobian decorrelation across a batch
                     (Mellor et al. 2021, NASWOT family), GELU-friendly variant.

Use :func:`compute_proxies` to get all scores at once for a model + batch.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def param_count(model: nn.Module) -> float:
    return float(sum(p.numel() for p in model.parameters()))


def _seg_loss(logits: torch.Tensor, target: Optional[torch.Tensor]) -> torch.Tensor:
    """Cross-entropy against a (possibly random) integer label volume."""
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    b, c = logits.shape[:2]
    if target is None:
        target = torch.randint(0, c, (b, *logits.shape[2:]), device=logits.device)
    return F.cross_entropy(logits, target)


def grad_norm(model: nn.Module, x: torch.Tensor,
              target: Optional[torch.Tensor] = None) -> float:
    model.zero_grad(set_to_none=True)
    out = model(x)
    loss = _seg_loss(out, target)
    loss.backward()
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().norm().item())
    model.zero_grad(set_to_none=True)
    return total


def snip(model: nn.Module, x: torch.Tensor,
         target: Optional[torch.Tensor] = None) -> float:
    model.zero_grad(set_to_none=True)
    out = model(x)
    loss = _seg_loss(out, target)
    loss.backward()
    score = 0.0
    for p in model.parameters():
        if p.grad is not None:
            score += float((p.detach() * p.grad.detach()).abs().sum().item())
    model.zero_grad(set_to_none=True)
    return score


@torch.no_grad()
def _abs_params(model: nn.Module):
    saved = []
    for p in model.parameters():
        saved.append(p.data.clone())
        p.data = p.data.abs()
    return saved


@torch.no_grad()
def _restore_params(model: nn.Module, saved) -> None:
    for p, s in zip(model.parameters(), saved):
        p.data = s


def synflow(model: nn.Module, x: torch.Tensor) -> float:
    """Data-independent SynFlow score (uses |w| and an all-ones input)."""
    was_training = model.training
    model.eval()
    saved = _abs_params(model)
    model.zero_grad(set_to_none=True)
    ones = torch.ones((1, *x.shape[1:]), device=x.device, dtype=x.dtype)
    out = model(ones)
    if isinstance(out, (tuple, list)):
        out = out[0]
    out.sum().backward()
    score = 0.0
    for p in model.parameters():
        if p.grad is not None:
            score += float((p.detach() * p.grad.detach()).abs().sum().item())
    model.zero_grad(set_to_none=True)
    _restore_params(model, saved)
    if was_training:
        model.train()
    return score


def jacob_cov(model: nn.Module, x: torch.Tensor) -> float:
    """NASWOT-family: higher when per-sample input gradients are decorrelated.

    Score = -sum(log(eig) + 1/eig) over the eigenvalues of the correlation matrix of
    flattened input-output Jacobians (Mellor 2021). It is maximal for a perfectly
    decorrelated (identity) correlation matrix - i.e. a more expressive net that
    separates inputs better scores HIGHER, matching the search's maximize direction.
    """
    x = x.clone().requires_grad_(True)
    out = model(x)
    if isinstance(out, (tuple, list)):
        out = out[0]
    b = x.shape[0]
    # one scalar per sample = mean logit, grad wrt input -> jacobian row
    y = out.reshape(b, -1).mean(dim=1).sum()
    g, = torch.autograd.grad(y, x, create_graph=False)
    J = g.reshape(b, -1)
    J = J - J.mean(dim=1, keepdim=True)
    norm = J.norm(dim=1, keepdim=True) + 1e-8
    Jn = J / norm
    corr = Jn @ Jn.t()
    eig = torch.linalg.eigvalsh(corr + 1e-3 * torch.eye(b, device=x.device))
    # NASWOT (Mellor 2021): reward DECORRELATED (expressive) Jacobians with a HIGHER
    # score. The previous ``-sum(log|eig|)`` = -log det(corr) was inverted - large for a
    # correlated (non-expressive) net and ~0 for a decorrelated one - so the search
    # (which MAXIMIZES this proxy) selected the LEAST-expressive architectures. The full
    # term ``-sum(log(eig) + 1/eig)`` is most-negative for an ill-conditioned corr and
    # largest for the identity (perfectly decorrelated) matrix.
    ev = eig.abs() + 1e-5
    score = -torch.sum(torch.log(ev) + 1.0 / ev)
    return float(score.item())


def _model_logits_and_features(model: nn.Module, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return logits plus a high-level feature map for DisWOT scoring."""
    try:
        out = model(x, return_features=True)
        if isinstance(out, tuple) and len(out) == 2:
            logits, feats = out
            if isinstance(feats, dict) and feats:
                feat = feats["bottleneck"] if "bottleneck" in feats else next(iter(feats.values()))
                return logits, feat
    except TypeError:
        pass
    logits = model(x)
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    return logits, logits


def _gradcam_semantic_matrix(model: nn.Module, x: torch.Tensor,
                             num_classes: int) -> torch.Tensor:
    logits, feat = _model_logits_and_features(model, x)
    cams = []
    for cls in range(num_classes):
        model.zero_grad(set_to_none=True)
        score = logits[:, cls].mean()
        grad, = torch.autograd.grad(score, feat, retain_graph=True, create_graph=False, allow_unused=False)
        weights = grad.mean(dim=tuple(range(2, grad.ndim)), keepdim=True)
        cam = F.relu((weights * feat).sum(dim=1))
        cams.append(cam.flatten(1))
    M = torch.stack(cams, dim=1).mean(dim=0)  # (classes, voxels)
    M = F.normalize(M, dim=1)
    return M @ M.t()


def _relation_matrix(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    logits, feat = _model_logits_and_features(model, x)
    rep = feat.flatten(1)
    if rep.shape[0] < 2:
        rep = logits.flatten(1)
    rep = F.normalize(rep - rep.mean(dim=1, keepdim=True), dim=1)
    return rep @ rep.t()


def diswot_score(student: nn.Module, teacher: nn.Module, x: torch.Tensor,
                 teacher_x: Optional[torch.Tensor] = None,
                 num_classes: Optional[int] = None) -> float:
    """Training-free DisWOT score for a candidate student and fixed teacher.

    The paper minimizes semantic Grad-CAM similarity distance plus sample
    relation distance. The NAS driver maximizes accuracy proxies, so this
    returns the negative distance: larger is better, and ``-score`` is the
    literal DisWOT distance.
    """
    teacher_x = x if teacher_x is None else teacher_x
    student_was_training = student.training
    teacher_was_training = teacher.training
    try:
        student.train(False)
        teacher.train(False)
        s_logits, _ = _model_logits_and_features(student, x)
        n_cls = int(num_classes or s_logits.shape[1])
        Ms = _gradcam_semantic_matrix(student, x, n_cls)
        Mt = _gradcam_semantic_matrix(teacher, teacher_x, n_cls).to(Ms.device)
        Mr_s = _relation_matrix(student, x)
        Mr_t = _relation_matrix(teacher, teacher_x).to(Mr_s.device)
        semantic = torch.linalg.vector_norm(Ms - Mt)
        relation = torch.linalg.vector_norm(Mr_s - Mr_t)
        return float(-(semantic + relation).detach().cpu())
    finally:
        student.zero_grad(set_to_none=True)
        teacher.zero_grad(set_to_none=True)
        student.train(student_was_training)
        teacher.train(teacher_was_training)


def compute_proxies(model: nn.Module, x: torch.Tensor,
                    target: Optional[torch.Tensor] = None,
                    teacher: Optional[nn.Module] = None,
                    teacher_x: Optional[torch.Tensor] = None,
                    num_classes: Optional[int] = None) -> Dict[str, float]:
    """All proxies for one model + batch. Each is independent and side-effect
    free (parameters/gradients restored)."""
    out = {
        "param_count": param_count(model),
        "grad_norm": grad_norm(model, x, target),
        "snip": snip(model, x, target),
        "synflow": synflow(model, x),
        "jacob_cov": jacob_cov(model, x),
    }
    if teacher is not None:
        out["diswot"] = diswot_score(model, teacher, x, teacher_x, num_classes)
    return out
