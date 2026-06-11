"""Config loader: deep-merge a user YAML over the packaged defaults.

Returns a nested :class:`types.SimpleNamespace` so downstream code can use
attribute access (``cfg.loss.task.type``); the loss/augmentation factories also
accept plain dicts, so either works. Unknown keys not present in the defaults
raise ``ValueError`` to catch typos early.
"""
from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_DEFAULTS_PATH = Path(__file__).with_name("defaults.yaml")


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required to load OGAP configs.") from exc
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: Dict[str, Any], over: Dict[str, Any], path: str = "") -> Dict[str, Any]:
    out = dict(base)
    for key, val in over.items():
        where = f"{path}.{key}" if path else key
        if key not in base:
            raise ValueError(f"Unknown config key {where!r} (not in defaults.yaml).")
        base_is_dict = isinstance(base[key], dict)
        val_is_dict = isinstance(val, dict)
        # A dict-valued default replaced by a non-null SCALAR silently discards the
        # whole sub-tree — e.g. `compile: true` over `compile: {enabled: ..., mode: ...}`
        # — and only surfaces as an AttributeError deep in training. Reject that. An
        # explicit `null`/None is allowed as the "disable this block" idiom (the config
        # readers treat a None node as absent and fall back to defaults).
        if base_is_dict and val is not None and not val_is_dict:
            raise ValueError(
                f"Config key {where!r}: expected a mapping (matching defaults.yaml) or "
                f"null to disable, got {type(val).__name__} ({val!r}).")
        if base_is_dict and val_is_dict:
            out[key] = _deep_merge(base[key], val, where)
        else:
            out[key] = val
    return out


def _to_namespace(obj: Any) -> Any:
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    return obj


def _get_path(obj: Any, path: str, default: Any = None) -> Any:
    cur = obj
    for key in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key, default)
        else:
            cur = getattr(cur, key, default)
    return cur


def _as_bool(obj: Any, path: str) -> bool:
    return bool(_get_path(obj, path, False))


_UNSUPPORTED_ACTIVE_GATES: Tuple[Tuple[str, str], ...] = (
    (
        "loss.task.boundary_loss",
        "boundary loss is declared in the schema but is not applied by the training objective",
    ),
    (
        "loss.task.dynamic_region_weighting.enabled",
        "dynamic region reweighting is not updated by the training loop",
    ),
    (
        "distillation.uncertainty_weighting.enabled",
        "uncertainty-weighted distillation is not implemented in the KD assembly",
    ),
    (
        "training.use_fp8",
        "FP8 is not implemented for the Conv3d-heavy OGAP train path",
    ),
    (
        "training.cuda_graphs.enabled",
        "CUDA graph capture is not implemented for the variable training/eval loop",
    ),
    (
        "model.transformer_engine_fp8.enabled",
        "Transformer Engine FP8 modules are not present in the active model path",
    ),
    (
        "inference.ood.enabled",
        "production OOD fitting is not wired into the launcher/runtime contract",
    ),
    (
        "inference.longitudinal.enabled",
        "production longitudinal fitting is not wired into the launcher/runtime contract",
    ),
)


def validate_runtime_contract(cfg: Any) -> None:
    """Fail fast when a config advertises active gates the runtime cannot honor."""
    errors: List[str] = []
    for path, reason in _UNSUPPORTED_ACTIVE_GATES:
        if _as_bool(cfg, path):
            errors.append(f"{path}: {reason}")

    student_arch = str(_get_path(cfg, "model.student.arch", "standard") or "standard").lower()
    if student_arch not in {"standard", "res", "ode"}:
        errors.append(
            "model.student.arch: expected one of 'standard', 'res', or 'ode', "
            f"got {student_arch!r}"
        )

    if errors:
        joined = "\n  - ".join(errors)
        raise ValueError(
            "Unsupported active config gates. Disable these keys or wire the "
            f"corresponding runtime first:\n  - {joined}"
        )


def load_config(path: Optional[str] = None, as_namespace: bool = True) -> Any:
    """Load config, merging ``path`` (if given) over the packaged defaults.

    Args:
        path: optional YAML override file.
        as_namespace: return a nested SimpleNamespace if True, else a dict.

    Returns:
        The merged config. ``data.in_channels`` is auto-set to 7 when
        ``data.use_tissue_priors`` is True.

    Raises:
        ValueError: if the override contains a key absent from the defaults.
    """
    merged = _load_yaml(_DEFAULTS_PATH)
    if path is not None:
        merged = _deep_merge(merged, _load_yaml(Path(path)))
    # The MRI modality count drives channel-safe augmentation (the trailing FSL FAST
    # CSF/WM/GM prior channels must be shielded from intensity/Bloch transforms). It is
    # the *pre-tissue* in_channels; the model input then grows by the 3 prior channels.
    # Keep n_mri_channels distinct from in_channels so the shield is not defeated when
    # use_tissue_priors inflates in_channels to 7.
    data = merged.setdefault("data", {})
    base_in = int(data.get("in_channels", 4) or 4)
    if data.get("use_tissue_priors"):
        data["n_mri_channels"] = base_in
        data["in_channels"] = base_in + 3
    else:
        data["n_mri_channels"] = base_in
    validate_runtime_contract(merged)
    return _to_namespace(merged) if as_namespace else merged
