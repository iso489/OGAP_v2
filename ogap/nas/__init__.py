"""Hardware-aware Neural Architecture Search for OGAP.

Search space (ArchConfig), zero-cost proxies, in-process hardware cost model,
Once-for-All elastic supernet, a multi-objective constrained search driver, and a
proxy-validity harness that gates whether NAS results may be reported as
validated (vs exploratory) on the segmentation task.
"""
from __future__ import annotations

from .search_space import (
    ArchConfig, sample_arch, enumerate_space, mutate, search_space_size,
)
from .proxies import compute_proxies, diswot_score
from .hardware import HardwareCost, cost_of, measure_latency, meets_budget
from .supernet import OFASupernet3D, WIDTH_MULTS, DEPTHS, make_divisible
from .search import (
    Budget, SearchConfig, Candidate, evaluate, build_search_model,
    pareto_front, random_search, evolutionary_search,
)
from .validation import proxy_metric_rank_correlation, validate_proxies

__all__ = [
    "ArchConfig", "sample_arch", "enumerate_space", "mutate", "search_space_size",
    "compute_proxies", "diswot_score",
    "HardwareCost", "cost_of", "measure_latency", "meets_budget",
    "OFASupernet3D", "WIDTH_MULTS", "DEPTHS", "make_divisible",
    "Budget", "SearchConfig", "Candidate", "evaluate", "build_search_model",
    "pareto_front", "random_search", "evolutionary_search",
    "proxy_metric_rank_correlation", "validate_proxies",
]
