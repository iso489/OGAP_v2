
import random
import torch
from ogap.nas import (
    ArchConfig, sample_arch, enumerate_space, mutate, search_space_size,
    compute_proxies, diswot_score, cost_of, meets_budget,
    OFASupernet3D, SearchConfig, Budget, build_search_model,
    random_search, evolutionary_search, pareto_front,
)
from ogap.models import build_student


# ---- search space ----
def test_space_size_and_encode_roundtrip():
    assert search_space_size() == 5*3*2*2*3*4
    a = sample_arch(random.Random(0))
    assert ArchConfig.decode(a.encode()) == a

def test_mutate_changes_exactly_one_axis():
    a = ArchConfig()
    m = mutate(a, random.Random(1))
    diffs = sum(x != y for x, y in zip(a.encode(), m.encode()))
    assert diffs == 1

def test_enumerate_is_complete_and_valid():
    space = enumerate_space()
    assert len(space) == search_space_size()
    for a in space[:50]:
        a.validate()


# ---- proxies ----
def test_proxies_finite():
    m = build_student(8, 4, base=8)
    x = torch.randn(2, 8, 16, 16, 16)
    p = compute_proxies(m, x)
    assert set(p) == {"param_count","grad_norm","snip","synflow","jacob_cov"}
    for k, v in p.items():
        assert v == v and abs(v) < float("inf"), (k, v)

def test_proxies_side_effect_free():
    m = build_student(8, 4, base=8)
    before = [p.detach().clone() for p in m.parameters()]
    compute_proxies(m, torch.randn(1, 8, 16, 16, 16))
    after = list(m.parameters())
    assert all(torch.allclose(b, a) for b, a in zip(before, after))  # synflow restored


def test_diswot_proxy_is_teacher_aware():
    student = build_student(8, 4, base=8)
    teacher = build_student(8, 4, base=8).eval()
    x = torch.randn(2, 8, 8, 8, 8)
    p = compute_proxies(student, x, teacher=teacher, num_classes=4)
    assert "diswot" in p
    assert p["diswot"] == p["diswot"] and abs(p["diswot"]) < float("inf")
    direct = diswot_score(student, teacher, x, num_classes=4)
    assert direct == direct and abs(direct) < float("inf")
    assert student.training
    assert not teacher.training


# ---- hardware ----
def test_cost_model_basic():
    m = build_student(8, 4, base=8)
    c = cost_of(m, (1, 8, 16, 16, 16), runs=2)
    assert c.params > 0 and c.latency_ms > 0 and c.int8_mb < c.fp32_mb
    assert meets_budget(c, max_int8_mb=1e9)
    assert not meets_budget(c, max_int8_mb=0.0)


# ---- OFA supernet: the key correctness property ----
def test_export_subnet_matches_supernet_output():
    torch.manual_seed(0)
    net = OFASupernet3D(8, 4, stage_chs=(8, 16, 32, 64)).eval()
    x = torch.randn(1, 8, 16, 16, 16)
    for wm, depth in [(1.0, 8), (0.5, 4), (0.25, 2), (0.75, 6)]:
        net.set_active_subnet(wm, depth)
        with torch.no_grad():
            y_super = net(x)
            sub = net.export_subnet().eval()
            y_sub = sub(x)
        assert y_super.shape == y_sub.shape == (1, 4, 16, 16, 16)
        assert torch.allclose(y_super, y_sub, atol=1e-5), (wm, depth)

def test_depth_is_free_width_costs():
    net = OFASupernet3D(8, 4, stage_chs=(8, 16, 32, 64))
    net.set_active_subnet(0.5, 2); p_shallow = net.active_param_count()
    net.set_active_subnet(0.5, 8); p_deep = net.active_param_count()
    assert p_shallow == p_deep                       # elastic depth is free
    net.set_active_subnet(0.25, 4); p_narrow = net.active_param_count()
    net.set_active_subnet(1.0, 4);  p_wide = net.active_param_count()
    assert p_narrow < p_wide                         # elastic width costs


# ---- end-to-end search ----
def test_build_both_block_styles():
    a_med = ArchConfig(base=8, block_style="mednext")
    a_ode = ArchConfig(base=8, block_style="weight_tied_ode", bottleneck_depth=4)
    for a in (a_med, a_ode):
        m = build_search_model(a, 8, 4)
        y = m(torch.randn(1, 8, 16, 16, 16))
        out = y[0] if isinstance(y, (tuple, list)) else y
        assert out.shape[:2] == (1, 4)

def test_random_search_returns_pareto():
    cfg = SearchConfig(in_channels=8, num_classes=4,
                       input_shape=(1, 8, 16, 16, 16), cost_runs=2, seed=0)
    res = random_search(cfg, n=5)
    assert len(res["evaluated"]) == 5
    assert len(res["pareto"]) >= 1
    # pareto members are mutually non-dominated
    P = res["pareto"]
    for a in P:
        assert not any(b.dominates(a) for b in P if b is not a)

def test_evolution_runs():
    cfg = SearchConfig(in_channels=8, num_classes=4,
                       input_shape=(1, 8, 16, 16, 16), cost_runs=2, seed=1)
    res = evolutionary_search(cfg, iterations=4, population_size=4, sample_size=2)
    assert res["best"] is not None and len(res["pareto"]) >= 1
