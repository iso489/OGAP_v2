"""Regression tests for the NAS accuracy-proxy batch-size fix.

The logged search (pareto_front.json) had jacob_cov scores collapsed to a near
constant: two *distinct* architectures shared the bit-identical value
-1.0000004768371582. Root cause - the proxy was evaluated at batch size B=1
(search.py SearchConfig.input_shape / cli.py), but jacob_cov (NASWOT) scores the
decorrelation of per-sample input-output Jacobians ACROSS A BATCH; a 1x1
correlation matrix is the constant [[1]], so the proxy is dead and the "search"
reduces to a params/latency Pareto sweep with no accuracy signal.

Fix - a dedicated ``proxy_batch`` (default 16) decoupled from the cost batch
(B=1, single-volume deployment latency).
"""
import torch

from ogap.nas.proxies import jacob_cov
from ogap.nas.search import SearchConfig, build_search_model, evaluate
from ogap.nas.search_space import ArchConfig

# Two distinct architectures (both appeared on the logged Pareto front).
_A1 = ArchConfig(base=12, attention="none", block_style="mednext",
                 deep_supervision=False, feature_dr="none", bottleneck_depth=6)
_A2 = ArchConfig(base=16, attention="none", block_style="mednext",
                 deep_supervision=True, feature_dr="mixstyle", bottleneck_depth=2)


def _two_models():
    torch.manual_seed(0)
    m1 = build_search_model(_A1, 8, 4)
    torch.manual_seed(0)
    m2 = build_search_model(_A2, 8, 4)
    return m1, m2


def test_jacob_cov_is_dead_at_batch_1():
    """Documents the failure mode: B=1 -> 1x1 corr -> constant ~-1.0 for ALL
    architectures (reproduces the logged twin -1.0000004768371582)."""
    m1, m2 = _two_models()
    x = torch.randn(1, 8, 16, 16, 16)
    s1, s2 = jacob_cov(m1, x), jacob_cov(m2, x)
    assert abs(s1 - s2) < 1e-5            # bit-identical -> non-discriminative
    assert abs(s1 - (-1.0)) < 1e-3        # the dead constant


def test_jacob_cov_discriminates_at_production_batch():
    """The fix: at B>=16 the proxy separates distinct architectures."""
    m1, m2 = _two_models()
    x = torch.randn(16, 8, 16, 16, 16)
    s1, s2 = jacob_cov(m1, x), jacob_cov(m2, x)
    assert abs(s1 - s2) > 1e-2            # real spread


def test_searchconfig_default_proxy_batch_is_discriminative():
    """SearchConfig defaults to a proxy batch that actually discriminates, and
    evaluate() yields a non-constant accuracy axis across distinct architectures."""
    cfg = SearchConfig(in_channels=8, num_classes=4,
                       input_shape=(1, 8, 16, 16, 16), cost_runs=1, seed=0)
    assert cfg.proxy_batch >= 8
    c1 = evaluate(_A1, cfg)
    c2 = evaluate(_A2, cfg)
    assert c1.accuracy != c2.accuracy     # accuracy axis is alive


def test_cost_stays_b1_independent_of_proxy_batch():
    """Cost (params / peak activation) is single-volume (B=1) regardless of the
    proxy batch - the deployment objective must not scale with proxy batch."""
    base = dict(in_channels=8, num_classes=4, input_shape=(1, 8, 16, 16, 16),
                cost_runs=1, seed=0)
    c_small = evaluate(_A1, SearchConfig(proxy_batch=2, **base))
    c_big = evaluate(_A1, SearchConfig(proxy_batch=16, **base))
    assert c_small.cost.params == c_big.cost.params
    assert abs(c_small.cost.peak_activation_mb - c_big.cost.peak_activation_mb) < 1e-6
