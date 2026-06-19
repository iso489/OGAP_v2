"""Knowledge-distillation loss assembly for the ODE-teacher → student setup.

The distillation signal is a temperature-scaled KL divergence between teacher and
student output distributions, computed per overlapping subregion and summed with
per-region weights (the small enhancing-tumour region benefits most from teacher
guidance, so it gets the largest default weight). The full assembly combines the
task loss with the KD term (and optional feature / attention terms) and returns
every component separately so each can be logged. ``loss_total`` is the value to
call ``backward()`` on.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .region_aware import _REGION_CLASSES, _cfg_get, build_task_loss

logger = logging.getLogger(__name__)

_DEFAULT_KD_REGION_WEIGHTS: Dict[str, float] = {"WT": 1.0, "TC": 1.5, "ET": 3.0}


def _as_weight_dict(value, default: Dict[str, float]) -> Dict[str, float]:
    """Coerce a per-region weight spec to a plain ``{region: float}`` dict.

    The config loader yields nested ``SimpleNamespace`` objects, so a region
    weight map can arrive as a namespace, a dict, or ``None``. Normalise all
    three to a dict so downstream ``.items()`` / ``.get()`` calls are safe.
    """
    if value is None:
        return dict(default)
    if isinstance(value, dict):
        coerced = dict(value)
    elif hasattr(value, "__dict__"):  # e.g. types.SimpleNamespace from the loader
        coerced = dict(vars(value))
    else:
        return dict(default)
    # A blank spec (e.g. `region_weights:` with nothing under it in YAML, which
    # the loader yields as an empty namespace/dict) must NOT silently disable the
    # term - fall back to a non-empty default. An empty default is itself
    # legitimate (per_region_temperature = "no per-region override"), so only
    # substitute when the default has content. Mirrors region_aware's `x or default`.
    if not coerced and default:
        return dict(default)
    return coerced


def _region_logits(logits: torch.Tensor, region: str) -> torch.Tensor:
    """Two-channel (background-vs-region) logits via log-sum-exp over classes."""
    classes = _REGION_CLASSES[region]
    fg = torch.logsumexp(logits[:, classes, ...], dim=1, keepdim=True)
    bg_idx = [i for i in range(logits.shape[1]) if i not in classes]
    bg = torch.logsumexp(logits[:, bg_idx, ...], dim=1, keepdim=True) if bg_idx \
        else torch.zeros_like(fg)
    return torch.cat([bg, fg], dim=1)


class RegionKDLoss(nn.Module):
    """Per-region, temperature-scaled KL( teacher || student ).

    L_KD = sum_r w_r * tau_r^2 * KL(softmax(t_r/tau_r) || softmax(s_r/tau_r)).
    The tau^2 factor is the standard Hinton scaling keeping gradient magnitudes
    comparable across temperatures. Computed in float32 for autocast safety.
    """

    def __init__(
        self,
        temperature: float = 3.0,
        region_weights: Optional[Dict[str, float]] = None,
        per_region_temperature: Optional[Dict[str, float]] = None,
        use_per_region_temperature: bool = False,
    ) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.region_weights = _as_weight_dict(region_weights, _DEFAULT_KD_REGION_WEIGHTS)
        self.per_region_temperature = _as_weight_dict(per_region_temperature, {})
        self.use_per_region_temperature = bool(use_per_region_temperature)

    def _tau(self, region: str) -> float:
        if self.use_per_region_temperature:
            return float(self.per_region_temperature.get(region, self.temperature))
        return self.temperature

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=student_logits.device.type, enabled=False):
            s_all = student_logits.float()
            t_all = teacher_logits.float()
            total = s_all.new_zeros(())
            wsum = 0.0
            for region, weight in self.region_weights.items():
                tau = self._tau(region)
                s = _region_logits(s_all, region) / tau
                t = _region_logits(t_all, region) / tau
                # Per-voxel mean KL: sum over the 2 region channels, then average
                # over batch+spatial. "batchmean" divides only by batch size, so
                # its magnitude would scale with the voxel count (patch-size
                # dependent); this normalisation is invariant to patch size.
                kl = F.kl_div(F.log_softmax(s, dim=1), F.softmax(t, dim=1),
                              reduction="none").sum(dim=1).mean()
                total = total + weight * (tau * tau) * kl
                wsum += weight
        return total / max(wsum, 1e-8)


def _normalised_mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(F.normalize(a.flatten(1), dim=1), F.normalize(b.flatten(1), dim=1))


def _feature_pairs(student_features, teacher_features) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
    if isinstance(student_features, dict) and isinstance(teacher_features, dict):
        for key in student_features.keys() & teacher_features.keys():
            yield student_features[key], teacher_features[key]
        return
    yield from zip(student_features, teacher_features)


def _token_pool(feat: torch.Tensor, grid: int) -> torch.Tensor:
    """3-D feature map ``(B,C,D,H,W)`` -> region tokens ``(B, grid^3, C)`` (float32).

    The PAT "patchify" step adapted to 3-D: adaptive-average-pool each stage to a
    fixed ``grid^3`` set of region descriptors so heterogeneous teacher/student
    spatial sizes (CNN vs ViT vs Mamba) are unified to a common token count.
    """
    pooled = F.adaptive_avg_pool3d(feat.float(), grid)   # (B, C, g, g, g)
    return pooled.flatten(2).transpose(1, 2).contiguous()  # (B, g^3, C)


def _token_cosine_mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Mean squared error on L2-normalised tokens (cosine-alignment objective)."""
    return F.mse_loss(F.normalize(a, dim=-1), F.normalize(b, dim=-1))


class RegionAwareAttentionDistill(nn.Module):
    """Region-Aware Attention (RAA) feature distillation - PAT (Lin et al., 2025).

    Opt-in heterogeneous-KD ablation arm. OGAP distils from architecturally diverse
    teachers (UNet3D / SegMamba / SwinUNETR / Neural-ODE) into a CNN student, so the
    teacher and student feature *perspectives* (receptive fields / token geometry)
    differ and naive stage-wise feature MSE underperforms (PAT §1, §3.2). RAA
    reblends the student's per-stage features with self-attention so they
    approximate the teacher's perspective before matching:

      1. pool each student stage to ``grid^3`` region tokens and project to a common
         dim ``d`` (the PAT ``C_i`` projectors, here per-stage ``Linear``s);
      2. concatenate the stage tokens and apply self-attention (PAT Eq. 4) so the
         student blends information across stages and regions;
      3. split back per stage, project ``d`` -> the teacher stage's channel count
         (the PAT ``M_i``), and match the teacher's pooled region tokens.

    The teacher tokens are the (detached) target; the whole module runs in float32
    (autocast disabled) for numerically stable attention and fp32 master params.

    **Learnable & lazily built.** The per-stage projections are created on the first
    forward (keyed by stage name, on the input's device, in float32) and registered
    as submodules. Include ``kd_loss.parameters()`` in the optimiser AFTER one
    warm-up forward, or call :meth:`materialise` once with example features first.

    Faithful-adaptation note: PAT is 2-D and matches the full teacher feature map;
    in 3-D with heterogeneous spatial sizes we match at the shared ``grid^3``
    region-token level - the natural multi-resolution analogue that keeps the
    teacher/student spatial-size mismatch well defined.
    """

    def __init__(self, dim: int = 128, token_grid: int = 4, heads: int = 4) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"RAA dim ({dim}) must be divisible by heads ({heads}).")
        if token_grid < 1:
            raise ValueError(f"RAA token_grid must be >= 1, got {token_grid}.")
        self.dim = int(dim)
        self.token_grid = int(token_grid)
        self.heads = int(heads)
        self.attn = nn.MultiheadAttention(self.dim, self.heads, batch_first=True)
        self.in_proj = nn.ModuleDict()   # stage -> Linear(student_channels -> dim)
        self.out_proj = nn.ModuleDict()  # stage -> Linear(dim -> teacher_channels)

    @staticmethod
    def _key(stage: str) -> str:
        # nn.ModuleDict keys may not contain '.'; sanitise the stage name.
        return str(stage).replace(".", "_")

    def materialise(self, student_features: Dict[str, torch.Tensor],
                    teacher_features: Dict[str, torch.Tensor]
                    ) -> "RegionAwareAttentionDistill":
        """Pre-build per-stage projections (float32) so the optimiser sees RAA
        params before the first step. Idempotent."""
        if not (isinstance(student_features, dict) and isinstance(teacher_features, dict)):
            return self
        for stage in sorted(student_features.keys() & teacher_features.keys()):
            k = self._key(stage)
            ref = student_features[stage]
            if k not in self.in_proj:
                # float32 params (do NOT inherit a possibly-bf16 feature dtype).
                self.in_proj[k] = nn.Linear(int(ref.shape[1]), self.dim).to(ref.device)
            if k not in self.out_proj:
                self.out_proj[k] = nn.Linear(
                    self.dim, int(teacher_features[stage].shape[1])).to(ref.device)
        return self

    def forward(self, student_features: Dict[str, torch.Tensor],
                teacher_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        if not (isinstance(student_features, dict) and isinstance(teacher_features, dict)):
            raise TypeError(
                "RAA distillation requires dict feature maps {stage: (B,C,D,H,W)} "
                "for both student and teacher.")
        stages = sorted(student_features.keys() & teacher_features.keys())
        if not stages:
            return next(iter(student_features.values())).new_zeros(())
        self.materialise(student_features, teacher_features)

        dev_type = student_features[stages[0]].device.type
        with torch.autocast(device_type=dev_type, enabled=False):
            # 1. tokenise + project each student stage to the common dim.
            per_stage = [self.in_proj[self._key(s)](_token_pool(student_features[s], self.token_grid))
                         for s in stages]                       # each (B, T, d)
            n_tok = per_stage[0].shape[1]                       # T = grid^3, equal for all stages
            cat = torch.cat(per_stage, dim=1)                  # (B, S*T, d)
            # 2. self-attention reblend across stages/regions (PAT Eq. 4).
            blended, _ = self.attn(cat, cat, cat, need_weights=False)
            # 3. split back per stage, project to teacher dim, match teacher tokens.
            total = blended.new_zeros(())
            for stage, chunk in zip(stages, blended.split(n_tok, dim=1)):
                s_out = self.out_proj[self._key(stage)](chunk)            # (B, T, Ct)
                t_tok = _token_pool(teacher_features[stage], self.token_grid).detach()
                total = total + _token_cosine_mse(s_out, t_tok)
            return total / len(stages)


class KDLoss(nn.Module):
    """Full loss assembly: task + KD (+ optional feature/attention terms).

    forward(student_logits, targets, teacher_logits=None, student_features=None,
    teacher_features=None) -> dict with ``loss_total`` and component tensors
    ``loss_task/loss_kd/loss_feat/loss_attn``. With ``beta == 0`` or no teacher,
    ``loss_total == loss_task``.
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.task = build_task_loss(config)
        self.alpha = float(_cfg_get(config, ("distillation", "alpha"), 1.0))
        self.beta = float(_cfg_get(config, ("distillation", "beta"), 0.5))
        self.gamma = float(_cfg_get(config, ("distillation", "gamma"), 0.1))
        self.eta = float(_cfg_get(config, ("distillation", "eta"), 0.05))
        self.enabled = bool(_cfg_get(config, ("distillation", "enabled"), True))
        self.feature_distillation = bool(
            _cfg_get(config, ("distillation", "feature_distillation"), False))
        self.attention_transfer = bool(
            _cfg_get(config, ("distillation", "attention_transfer"), False))
        self.kd = RegionKDLoss(
            temperature=float(_cfg_get(config, ("distillation", "temperature"), 3.0)),
            region_weights=_cfg_get(config, ("distillation", "region_weights"), None),
            per_region_temperature=_cfg_get(config, ("distillation", "per_region_temperature"), None),
            use_per_region_temperature=bool(
                _cfg_get(config, ("distillation", "per_region_temperature_enabled"), False)),
        )
        # Region-Aware Attention heterogeneous-KD ablation arm (PAT; default OFF so
        # the validated baseline is byte-identical). When enabled, RAA is learnable -
        # add kd_loss.parameters() to the optimiser after one warm-up forward.
        self.raa_enabled = bool(_cfg_get(config, ("distillation", "raa", "enabled"), False))
        self.raa_weight = float(_cfg_get(config, ("distillation", "raa", "weight"), self.gamma))
        self.raa = (
            RegionAwareAttentionDistill(
                dim=int(_cfg_get(config, ("distillation", "raa", "dim"), 128)),
                token_grid=int(_cfg_get(config, ("distillation", "raa", "token_grid"), 4)),
                heads=int(_cfg_get(config, ("distillation", "raa", "heads"), 4)),
            )
            if self.raa_enabled else None
        )

    def forward(
        self,
        student_logits: torch.Tensor,
        targets: torch.Tensor,
        teacher_logits: Optional[torch.Tensor] = None,
        student_features: Optional[Sequence[torch.Tensor]] = None,
        teacher_features: Optional[Sequence[torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        zero = student_logits.new_zeros(())
        loss_task = self.task(student_logits, targets)
        loss_kd = zero
        loss_feat = zero
        loss_attn = zero

        if self.enabled and self.beta != 0.0 and teacher_logits is not None:
            loss_kd = self.kd(student_logits, teacher_logits)

        if self.feature_distillation and student_features and teacher_features:
            terms = [_normalised_mse(s, t) for s, t in _feature_pairs(student_features, teacher_features)
                     if s.shape == t.shape]
            if terms:
                loss_feat = torch.stack(terms).mean()
            else:
                logger.warning("feature_distillation enabled but no matching-shape "
                               "feature pairs provided; skipping L_feat this step.")

        if self.attention_transfer and student_features and teacher_features:
            terms = []
            for s, t in _feature_pairs(student_features, teacher_features):
                a_s = F.normalize(s.abs().mean(dim=1).flatten(1), dim=1)
                a_t = F.normalize(t.abs().mean(dim=1).flatten(1), dim=1)
                if a_s.shape == a_t.shape:
                    terms.append(F.mse_loss(a_s, a_t))
            if terms:
                loss_attn = torch.stack(terms).mean()
            else:
                logger.warning("attention_transfer enabled but no matching-shape "
                               "feature pairs provided; skipping L_attn this step.")

        # Region-Aware Attention arm (opt-in). Kept entirely off the baseline path:
        # when self.raa is None, loss_total and the returned keys are unchanged.
        loss_raa = zero
        out_extra: Dict[str, torch.Tensor] = {}
        if self.raa is not None:
            if isinstance(student_features, dict) and isinstance(teacher_features, dict):
                loss_raa = self.raa(student_features, teacher_features)
            out_extra["loss_raa"] = loss_raa

        loss_total = (self.alpha * loss_task + self.beta * loss_kd
                      + self.gamma * loss_feat + self.eta * loss_attn)
        if self.raa is not None:
            loss_total = loss_total + self.raa_weight * loss_raa
        # Guard the float() conversions: they force a GPU->CPU sync every step, so
        # only pay it when DEBUG logging is actually enabled.
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("loss_total=%.6f task=%.6f kd=%.6f feat=%.6f attn=%.6f",
                         float(loss_total), float(loss_task), float(loss_kd),
                         float(loss_feat), float(loss_attn))
        return {"loss_total": loss_total, "loss_task": loss_task, "loss_kd": loss_kd,
                "loss_feat": loss_feat, "loss_attn": loss_attn, **out_extra}


def build_kd_loss(config: Any) -> KDLoss:
    """Factory for the full KD loss assembly (see :class:`KDLoss`)."""
    return KDLoss(config)
