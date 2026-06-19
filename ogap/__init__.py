"""OGAP - Open Glioma Analysis Pipeline (package).

This package is the de-monolithed home of the OGAP v9 codebase. It was created
by a strangler-fig refactor of ``OGAP_source_code_experimental_v9.py``: the
legacy 16k-line module is preserved verbatim as :mod:`ogap.legacy` and its
behaviour is unchanged, while cleanly separable subsystems (models, numerics)
and brand-new research capabilities (continuous-depth ODE teacher, weight-tied
ODE student, hardware-aware NAS / Once-for-All supernet, latent-ODE
longitudinal RANO tracker, continuous-normalizing-flow OOD scorer) live in
focused submodules with their own unit tests.

Design rules
------------
* **Backward compatibility is sacred.** The original CLI entrypoint
  ``python OGAP_source_code_experimental_v9.py <cmd> ...`` keeps working via a
  thin shim that re-exports this package. Every existing sbatch/.sh job runs
  unchanged.
* **New capability is opt-in.** The ODE/NAS/latent-ODE/CNF modules are additive
  and feature-flagged off by default; the default training/eval/export path is
  byte-for-byte the legacy behaviour.
* **Checkpoint safety.** Extracted ``nn.Module`` classes preserve attribute
  names and submodule order so existing ``state_dict`` checkpoints load without
  remapping.

Submodules
----------
``ogap.numerics``     self-contained ODE integrators (Euler/RK4 + optional adjoint).
``ogap.models``       conv blocks, teacher, student, and the new ODE variants.
``ogap.nas``          hardware-aware search space, zero-cost proxies, OFA supernet.
``ogap.longitudinal`` latent-ODE tumour-trajectory / RANO tracker.
``ogap.ood``          continuous-normalizing-flow out-of-distribution scoring.
"""
from __future__ import annotations

__version__ = "9.1.0-dev"

__all__ = [
    "__version__",
]
