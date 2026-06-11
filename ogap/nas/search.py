"""Multi-objective, hardware-constrained architecture search for OGAP.

Ties the pieces together into the survey's NAS loop (search space → search
strategy → performance estimation; White et al. 2023, Fig. 2):

* **search space**  — :mod:`ogap.nas.search_space` (:class:`ArchConfig`).
* **estimation**    — zero-cost proxies (:mod:`ogap.nas.proxies`) for accuracy
  and an in-process hardware model (:mod:`ogap.nas.hardware`) for cost. Either
  can be swapped for a real validation-Dice callback once you can afford partial
  training — the driver only needs ``accuracy_of(model)`` and ``cost_of(model)``.
* **strategy**      — random search (the survey's recommended baseline, §10.x)
  and aging/regularised evolution (Real et al. 2019).

The objective is explicitly **multi-objective and constrained** (§6.2): maximise
an accuracy proxy while minimising INT8 size and latency, subject to a hard
deployment budget. The result is a Pareto front — "one model per clinic" — not a
single winner.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from ..models import build_student
from .hardware import HardwareCost, cost_of, meets_budget
from .proxies import compute_proxies
from .search_space import ArchConfig, mutate, sample_arch
from .supernet import OFASupernet3D


@dataclass
class Budget:
    max_int8_mb: float = 1e9
    max_latency_ms: float = 1e9
    max_peak_activation_mb: float = 1e9


@dataclass
class SearchConfig:
    in_channels: int = 8
    num_classes: int = 4
    input_shape: Tuple[int, ...] = (1, 8, 32, 32, 32)
    proxy_key: str = "jacob_cov"   # size-robust accuracy proxy by default
    teacher_model: Optional[nn.Module] = None
    teacher_input_shape: Optional[Tuple[int, ...]] = None
    budget: Budget = field(default_factory=Budget)
    seed: int = 0
    cost_runs: int = 4


@dataclass
class Candidate:
    arch: ArchConfig
    accuracy: float
    cost: HardwareCost
    proxies: Dict[str, float]

    def dominates(self, other: "Candidate") -> bool:
        """Pareto: not worse on any objective, strictly better on one.
        Objectives: accuracy ↑, latency ↓, int8 size ↓."""
        not_worse = (
            self.accuracy >= other.accuracy
            and self.cost.latency_ms <= other.cost.latency_ms
            and self.cost.int8_mb <= other.cost.int8_mb
        )
        strictly = (
            self.accuracy > other.accuracy
            or self.cost.latency_ms < other.cost.latency_ms
            or self.cost.int8_mb < other.cost.int8_mb
        )
        return not_worse and strictly


def build_search_model(arch: ArchConfig, in_channels: int, num_classes: int) -> nn.Module:
    """Construct the concrete student described by ``arch``.

    ``mednext``           → the production ``UNet3DStudent``.
    ``weight_tied_ode``   → an OFA subnet at full width and ``bottleneck_depth``
                            (weight-tied elastic depth; the deployable artifact).
    """
    if arch.block_style == "weight_tied_ode":
        b = arch.base
        net = OFASupernet3D(in_channels, num_classes, stage_chs=(b, 2 * b, 4 * b, 8 * b))
        net.set_active_subnet(1.0, arch.bottleneck_depth)
        return net.export_subnet()
    return build_student(in_channels, num_classes, **arch.student_kwargs())


def evaluate(arch: ArchConfig, cfg: SearchConfig) -> Candidate:
    """Score one architecture with proxies + hardware cost (no full training)."""
    torch.manual_seed(cfg.seed)
    model = build_search_model(arch, cfg.in_channels, cfg.num_classes)
    x = torch.randn(*cfg.input_shape)
    teacher_x = None
    if cfg.teacher_model is not None and cfg.teacher_input_shape is not None:
        teacher_x = torch.randn(*cfg.teacher_input_shape)
    proxies = compute_proxies(
        model, x,
        teacher=cfg.teacher_model,
        teacher_x=teacher_x,
        num_classes=cfg.num_classes,
    )
    cost = cost_of(model, cfg.input_shape, runs=cfg.cost_runs)
    accuracy = proxies.get(cfg.proxy_key, proxies["jacob_cov"])
    return Candidate(arch=arch, accuracy=accuracy, cost=cost, proxies=proxies)


def pareto_front(cands: Sequence[Candidate]) -> List[Candidate]:
    """Non-dominated set over (accuracy ↑, latency ↓, int8 ↓)."""
    front: List[Candidate] = []
    for c in cands:
        if any(o.dominates(c) for o in cands if o is not c):
            continue
        front.append(c)
    # de-dup architectures that tie on all objectives
    seen = set()
    uniq = []
    for c in front:
        key = c.arch.encode()
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    return uniq


def _within_budget(c: Candidate, b: Budget) -> bool:
    return meets_budget(
        c.cost, max_int8_mb=b.max_int8_mb, max_latency_ms=b.max_latency_ms,
        max_peak_activation_mb=b.max_peak_activation_mb,
    )


def random_search(cfg: SearchConfig, n: int = 16) -> Dict[str, object]:
    """Random-search baseline. Returns evaluated candidates, budget-feasible
    set, and the Pareto front."""
    rng = random.Random(cfg.seed)
    evaluated = [evaluate(sample_arch(rng), cfg) for _ in range(n)]
    feasible = [c for c in evaluated if _within_budget(c, cfg.budget)]
    pool = feasible or evaluated
    return {
        "evaluated": evaluated,
        "feasible": feasible,
        "pareto": pareto_front(pool),
    }


def evolutionary_search(cfg: SearchConfig, iterations: int = 30,
                        population_size: int = 12,
                        sample_size: int = 4) -> Dict[str, object]:
    """Aging/regularised evolution (Real et al. 2019).

    Tournament-select a parent from a random sample, mutate one axis, evaluate,
    append; evict the oldest. Accuracy proxy is the selection signal, subject to
    the hard budget (infeasible children are kept but ranked last)."""
    rng = random.Random(cfg.seed)
    population: List[Candidate] = [evaluate(sample_arch(rng), cfg)
                                   for _ in range(population_size)]
    history: List[Candidate] = list(population)

    def fitness(c: Candidate) -> float:
        return c.accuracy if _within_budget(c, cfg.budget) else c.accuracy - 1e9

    for _ in range(iterations):
        sample = rng.sample(population, min(sample_size, len(population)))
        parent = max(sample, key=fitness)
        child = evaluate(mutate(parent.arch, rng), cfg)
        population.append(child)
        history.append(child)
        population.pop(0)  # age out the oldest

    feasible = [c for c in history if _within_budget(c, cfg.budget)]
    pool = feasible or history
    best = max(pool, key=fitness)
    return {
        "history": history,
        "feasible": feasible,
        "pareto": pareto_front(pool),
        "best": best,
    }
