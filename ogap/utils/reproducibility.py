"""Determinism helpers for reproducible experiments."""
from __future__ import annotations

import logging
import os
import random

logger = logging.getLogger(__name__)


def set_all_seeds(seed: int = 42, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch for reproducible runs.

    Args:
        seed: the integer seed applied to all RNGs.
        deterministic: if True, request deterministic cuDNN/algorithms. 3D conv
            on CUDA may lack deterministic kernels; a WARNING is logged rather
            than failing, and the limitation should be documented when reporting.

    Determinism vs. autotuning tradeoff [F8]: with ``deterministic=True`` this
    sets ``cudnn.benchmark=False``, which **overrides** the ``cudnn.benchmark=
    True`` that ``ogap.utils.hardware.configure_hardware`` enables for kernel
    autotuning. This is the correct precedence for a reproducible publication run
    (determinism wins), but it forgoes cuDNN's per-shape kernel search, so
    fixed-shape training is modestly slower. For a throughput-oriented (non-
    reproducibility) run, set ``deterministic=False`` to keep autotuning. The
    chosen mode should be stated in the paper's Methods.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception as exc:
        logger.warning("could not seed numpy: %s", exc)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            logger.info(
                "deterministic=True: cudnn.benchmark disabled (overrides "
                "configure_hardware autotuning; reproducible but modestly slower "
                "at fixed shapes). [F8]"
            )
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception as exc:
                logger.warning("deterministic algorithms unavailable: %s", exc)
    except Exception as exc:  # pragma: no cover
        logger.warning("could not seed torch: %s", exc)
