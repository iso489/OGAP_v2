"""Ablation-study bookkeeping for the opt-in framework components.

Maps the project config (see ``ogap/config/defaults.yaml``) to a fixed set of
toggleable components, loads the ``experiments/ablation_configs`` overlays, and
builds a component on/off matrix plus a variant-vs-baseline results table.

Training runs on the cluster; this module only performs the configuration
bookkeeping and the post-hoc result summary, both CPU-only and data-free. A
component is reported "on" when its config flag is truthy, except
``region_aware_loss`` which is on whenever the task-loss ``type`` is anything
other than the legacy Dice-CE.
"""
from __future__ import annotations

import glob
import os
from typing import Any, Dict, List, Sequence, Tuple

# Canonical component order used by every output of this module.
COMPONENTS: Tuple[str, ...] = (
    "region_aware_loss",
    "distillation",
    "per_region_temperature",
    "physics_augmentation",
    "auxiliary_tissue_head",
    "tissue_priors",
    "domain_adaptation",
)

# Boolean flag location in the merged config for each component (region_aware_loss
# is handled separately because it is a string-typed selector, not a flag).
_FLAG_PATHS: Dict[str, Tuple[str, ...]] = {
    "distillation": ("distillation", "enabled"),
    "per_region_temperature": ("distillation", "per_region_temperature_enabled"),
    "physics_augmentation": ("augmentation", "physics", "enabled"),
    "auxiliary_tissue_head": ("loss", "auxiliary_tissue", "enabled"),
    "tissue_priors": ("data", "use_tissue_priors"),
    "domain_adaptation": ("domain", "enabled"),
}
_TASK_TYPE_PATH: Tuple[str, ...] = ("loss", "task", "type")


def _get(cfg: Any, keys: Sequence[str]) -> Any:
    """Read a nested key from a dict *or* a SimpleNamespace config."""
    cur = cfg
    for k in keys:
        cur = cur[k] if isinstance(cur, dict) else getattr(cur, k)
    return cur


def component_states(cfg: Any) -> Dict[str, bool]:
    """Return ``{component: enabled}`` for a merged config (dict or namespace)."""
    states = {"region_aware_loss": str(_get(cfg, _TASK_TYPE_PATH)) != "legacy"}
    for comp, path in _FLAG_PATHS.items():
        states[comp] = bool(_get(cfg, path))
    return {c: states[c] for c in COMPONENTS}            # canonical order


def load_ablation_configs(config_dir: str, as_namespace: bool = False
                          ) -> List[Tuple[str, Any]]:
    """Load every ``*.yaml`` overlay in ``config_dir`` merged over the defaults.

    Args:
        config_dir: directory of ablation overlays (e.g. experiments/ablation_configs).
        as_namespace: pass-through to :func:`ogap.config.loader.load_config`.

    Returns:
        ``[(variant_name, merged_config), ...]`` sorted by filename; ``variant_name``
        is the filename stem (e.g. ``"0_full"``).
    """
    from ogap.config.loader import load_config

    paths = sorted(glob.glob(os.path.join(config_dir, "*.yaml")))
    return [(os.path.splitext(os.path.basename(p))[0],
             load_config(p, as_namespace=as_namespace)) for p in paths]


def ablation_matrix(config_dir: str):
    """Component on/off matrix over all overlays in ``config_dir``.

    Returns:
        A pandas DataFrame indexed by variant name with one boolean column per
        entry of :data:`COMPONENTS`.
    """
    import pandas as pd

    configs = load_ablation_configs(config_dir)
    data = {name: component_states(cfg) for name, cfg in configs}
    df = pd.DataFrame.from_dict(data, orient="index")
    return df.reindex(columns=list(COMPONENTS))


def ablation_results_table(
    results_by_variant: Dict[str, Dict[str, float]],
    baseline: str = "full",
    metrics: Sequence[str] = ("dsc_wt", "dsc_tc", "dsc_et"),
) -> List[Dict[str, object]]:
    """Summarise per-variant metrics with the signed change from a baseline.

    Args:
        results_by_variant: ``{variant_name: {metric: aggregated_value}}``.
        baseline: variant whose values the deltas are measured against.
        metrics: metric keys to report.

    Returns:
        One flat row dict per variant: ``{"variant", <metric>, "<metric>_delta"}``,
        where ``delta = variant_value - baseline_value`` (so a negative delta on a
        higher-is-better metric means removing that component hurt). The rows plug
        directly into :func:`ogap.evaluation.tables.rows_to_csv` / ``rows_to_latex``.

    Raises:
        ValueError: if ``baseline`` is not a key of ``results_by_variant``.
    """
    if baseline not in results_by_variant:
        raise ValueError(
            f"baseline {baseline!r} not in results_by_variant "
            f"(have {sorted(results_by_variant)}).")
    metrics = list(metrics)
    base = results_by_variant[baseline]
    rows: List[Dict[str, object]] = []
    for variant, vals in results_by_variant.items():
        row: Dict[str, object] = {"variant": variant}
        for m in metrics:
            v, bv = vals.get(m), base.get(m)
            row[m] = v
            row[f"{m}_delta"] = (float(v) - float(bv)
                                 if v is not None and bv is not None else float("nan"))
        rows.append(row)
    return rows
