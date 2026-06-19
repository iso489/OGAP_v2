"""OGAP architecture search space.

The v9 monolith already encodes a *manual* macro search space - the ablation
runner sweeps ``attention ∈ {none, eca, se}``, ``block_style ∈ {mednext,
dense}``, deep-supervision on/off and base width. This module promotes that
implicit space to an explicit, sampled/enumerable :class:`ArchConfig`, which is
the "search space" leg of the NAS trichotomy (White et al. 2023, §1.2/§2:
search space, search strategy, performance estimation).

The space is deliberately a **constrained macro space**: small, deployment-aware,
seeded with the human priors that already work for OGAP. That is exactly the
right call for a small African-data, expensive-3D-training regime (the survey
notes large primitive spaces risk search-space overfitting; §2). Every axis here
maps to a real knob the deployable student exposes.
"""
from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

# Searchable axes. Keep these aligned with what build_student / the supernet
# actually accept so a sampled config is always constructible.
BASE_WIDTHS: Tuple[int, ...] = (8, 12, 16, 24, 32)
ATTENTIONS: Tuple[str, ...] = ("none", "eca", "se")
BLOCK_STYLES: Tuple[str, ...] = ("mednext", "weight_tied_ode")
DEEP_SUPERVISION: Tuple[bool, ...] = (True, False)
FEATURE_DR: Tuple[str, ...] = ("none", "mixstyle", "dsu")
# Effective bottleneck depth - for weight_tied_ode blocks this is the number of
# Euler steps and costs ZERO extra parameters (see ogap.models.student_ode).
BOTTLENECK_DEPTHS: Tuple[int, ...] = (2, 4, 6, 8)


@dataclass(frozen=True)
class ArchConfig:
    """A single point in the OGAP student search space."""

    base: int = 16
    attention: str = "none"
    block_style: str = "mednext"          # "mednext" | "weight_tied_ode"
    deep_supervision: bool = True
    feature_dr: str = "none"
    bottleneck_depth: int = 4             # only meaningful for weight_tied_ode

    def validate(self) -> "ArchConfig":
        if self.base not in BASE_WIDTHS:
            raise ValueError(f"base {self.base} not in {BASE_WIDTHS}")
        if self.attention not in ATTENTIONS:
            raise ValueError(f"attention {self.attention!r} invalid")
        if self.block_style not in BLOCK_STYLES:
            raise ValueError(f"block_style {self.block_style!r} invalid")
        if self.feature_dr not in FEATURE_DR:
            raise ValueError(f"feature_dr {self.feature_dr!r} invalid")
        if self.bottleneck_depth not in BOTTLENECK_DEPTHS:
            raise ValueError(f"bottleneck_depth {self.bottleneck_depth} invalid")
        return self

    def to_dict(self) -> Dict:
        return asdict(self)

    # ---- encoding for search algorithms (categorical -> index vector) ----
    def encode(self) -> Tuple[int, ...]:
        return (
            BASE_WIDTHS.index(self.base),
            ATTENTIONS.index(self.attention),
            BLOCK_STYLES.index(self.block_style),
            DEEP_SUPERVISION.index(self.deep_supervision),
            FEATURE_DR.index(self.feature_dr),
            BOTTLENECK_DEPTHS.index(self.bottleneck_depth),
        )

    @staticmethod
    def decode(vec: Sequence[int]) -> "ArchConfig":
        return ArchConfig(
            base=BASE_WIDTHS[vec[0]],
            attention=ATTENTIONS[vec[1]],
            block_style=BLOCK_STYLES[vec[2]],
            deep_supervision=DEEP_SUPERVISION[vec[3]],
            feature_dr=FEATURE_DR[vec[4]],
            bottleneck_depth=BOTTLENECK_DEPTHS[vec[5]],
        ).validate()

    def student_kwargs(self) -> Dict:
        """Kwargs accepted by ogap.models.build_student for this config."""
        return {
            "base": self.base,
            "attention": self.attention,
            "enable_deep_supervision": self.deep_supervision,
            "feature_dr": self.feature_dr,
        }


_AXES = (BASE_WIDTHS, ATTENTIONS, BLOCK_STYLES, DEEP_SUPERVISION, FEATURE_DR, BOTTLENECK_DEPTHS)


def search_space_size() -> int:
    """Total number of architectures in the (discrete) space."""
    n = 1
    for ax in _AXES:
        n *= len(ax)
    return n


def sample_arch(rng: Optional[random.Random] = None) -> ArchConfig:
    """Uniformly sample a valid architecture (random-search baseline / init)."""
    r = rng or random
    return ArchConfig(
        base=r.choice(BASE_WIDTHS),
        attention=r.choice(ATTENTIONS),
        block_style=r.choice(BLOCK_STYLES),
        deep_supervision=r.choice(DEEP_SUPERVISION),
        feature_dr=r.choice(FEATURE_DR),
        bottleneck_depth=r.choice(BOTTLENECK_DEPTHS),
    )


def enumerate_space() -> List[ArchConfig]:
    """Enumerate the full space (small enough to be exhaustive for baselines)."""
    out = []
    for combo in itertools.product(*_AXES):
        out.append(ArchConfig(*combo))
    return out


def mutate(arch: ArchConfig, rng: Optional[random.Random] = None) -> ArchConfig:
    """Single-axis mutation for the evolutionary search strategy."""
    r = rng or random
    vec = list(arch.encode())
    axis = r.randrange(len(vec))
    domain = len(_AXES[axis])
    if domain > 1:
        choices = [i for i in range(domain) if i != vec[axis]]
        vec[axis] = r.choice(choices)
    return ArchConfig.decode(vec)
