"""Registry of comparison baselines for the segmentation/distillation study.

Each baseline is a method we want to compare against. Actually training or
evaluating one needs a cluster GPU, the dataset, and (for external methods)
their reference implementation, none of which exist on the CPU dev box - so the
runners are honest stubs that raise ``NotImplementedError`` describing what is
required. The registry itself (names + neutral one-line descriptions) is
available everywhere so configs, tables, and docs can reference it.

Fill in a runner on the cluster by replacing :func:`run_baseline`'s dispatch for
that name; the registry and tests already pin the expected names.
"""
from __future__ import annotations

from typing import Any, Dict, List

# name -> neutral one-line description (no result claims).
BASELINES: Dict[str, str] = {
    "dense_teacher":
        "Full-capacity teacher network trained without compression.",
    "segmamba_teacher":
        "SegMamba teacher with multi-scale TSMamba/GSC/FUE blocks.",
    "swin_unetr_teacher":
        "MONAI Swin UNETR teacher with shifted-window transformer encoder.",
    "nnunet":
        "Standard self-configuring U-Net segmentation pipeline.",
    "nnunet_resenc":
        "nnU-Net residual-encoder preset (the current BraTS-class strong baseline).",
    "synthseg":
        "SynthSeg-style fully contrast-randomized training for domain-agnostic robustness.",
    "student_no_distillation":
        "Compact student trained from scratch with no teacher signal.",
    "vanilla_kd":
        "Response-based distillation with a single global temperature.",
    "feature_kd":
        "Intermediate feature-map distillation (FitNets-style hint layers).",
    "attention_transfer":
        "Attention-map transfer distillation between teacher and student.",
}


def list_baselines() -> List[str]:
    """Sorted list of registered baseline names."""
    return sorted(BASELINES)


def baseline_description(name: str) -> str:
    """One-line description for ``name``.

    Raises:
        KeyError: if ``name`` is not a registered baseline.
    """
    if name not in BASELINES:
        raise KeyError(
            f"unknown baseline {name!r}; known: {list_baselines()}")
    return BASELINES[name]


def run_baseline(name: str, *args: Any, **kwargs: Any):
    """Run a comparison baseline.

    Args:
        name: a registered baseline (see :func:`list_baselines`).
        *args, **kwargs: forwarded once a real runner is implemented.

    Raises:
        KeyError: if ``name`` is not registered (validated before anything else).
        NotImplementedError: always, for a registered name - these baselines need
            a cluster GPU, the dataset, and the method's reference implementation,
            so they cannot run on the CPU dev box.
    """
    if name not in BASELINES:
        raise KeyError(
            f"unknown baseline {name!r}; known: {list_baselines()}")
    raise NotImplementedError(
        f"baseline {name!r} ({BASELINES[name]}) requires a cluster GPU + dataset "
        f"+ the method's reference implementation; not runnable on the dev box. "
        f"Implement its dispatch in ogap.evaluation.baselines.run_baseline.")
