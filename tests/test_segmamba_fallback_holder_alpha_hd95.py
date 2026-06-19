"""Tests for the 2026 SegMamba-fallback + Hölder-α-ablation + HD95-reporting pass.

1. The canonical ``ogap.models.teacher.SegMambaTeacher`` gains a pure-PyTorch
   selective-scan fallback (``_MinimalMamba``) so it runs / is CPU-testable /
   is ``torch.compile``-able WITHOUT the ``mamba-ssm`` CUDA kernel.
2. Hölder KD keeps α=1.5 (the flatter renormalised teacher^0.5 target =
   deliberate uncertainty transfer), with an explicit α=2.0 (proper
   Cauchy-Schwarz) ablation arm in ``cmd_kd_compare``.
3. HD95 reporting: the per-epoch ``brats_metrics()`` is RAW (now documented);
   the reported path post-processes (``connected_component_postprocessing``) and
   is lesion-wise, with the 373.1287 mm empty-region convention.

All tests run on CPU. The SegMamba integration test uses the (slow) sequential
fallback scan, so it is the heaviest test here (~seconds).
"""
import inspect

import numpy as np
import pytest
import torch

import ogap.legacy as L
from ogap.models.teacher import (
    SegMambaTeacher,
    _MinimalMamba,
    _resolve_mamba_cls,
    _segmamba_fallback_enabled,
)


# ───────────────────── 1. SegMamba pure-PyTorch fallback ─────────────────────

def test_minimal_mamba_shape_determinism_gradient():
    torch.manual_seed(0)
    m = _MinimalMamba(d_model=8, d_state=16, d_conv=4, expand=2).eval()
    x = torch.randn(2, 24, 8)
    y = m(x)
    assert y.shape == x.shape
    assert torch.allclose(m(x), y, atol=1e-6)            # deterministic in eval
    m.train()
    xg = torch.randn(1, 16, 8, requires_grad=True)
    m(xg).sum().backward()
    assert xg.grad is not None and torch.isfinite(xg.grad).all()


def test_minimal_mamba_is_causal():
    """Output at time t must depend only on inputs <= t - the defining property
    of a correct selective scan."""
    torch.manual_seed(0)
    m = _MinimalMamba(d_model=8).eval()
    x = torch.randn(1, 30, 8)
    y = m(x)
    x2 = x.clone()
    x2[:, 15:] += 7.0                                    # perturb the future only
    y2 = m(x2)
    assert torch.allclose(y[:, :15], y2[:, :15], atol=1e-5)    # past unchanged
    assert (y[:, 15:] - y2[:, 15:]).abs().max() > 1e-3        # future changed


def test_resolver_refuses_silent_substitution_but_honors_optin():
    try:
        import mamba_ssm  # noqa: F401
        pytest.skip("mamba-ssm installed: the raise-vs-fallback branch isn't exercised")
    except ImportError:
        pass
    with pytest.raises(ImportError):
        _resolve_mamba_cls(False)                        # no opt-in -> refuse to substitute
    assert _resolve_mamba_cls(True) is _MinimalMamba     # explicit opt-in -> fallback
    assert _segmamba_fallback_enabled(True) is True


def test_segmamba_teacher_runs_via_fallback_cpu():
    """Full canonical SegMambaTeacher forward + KD feature dict + backward on CPU
    via the pure-PyTorch fallback (slow but correct)."""
    torch.manual_seed(0)
    net = SegMambaTeacher(in_channels=4, num_classes=4, base=4,
                          allow_pure_pytorch_fallback=True)
    x = torch.randn(1, 4, 32, 32, 32)
    logits, feats = net(x, return_features=True)
    assert logits.shape == (1, 4, 32, 32, 32)
    assert set(feats.keys()) == {"bottleneck", "dec3"}   # KD taps preserved
    logits.sum().backward()
    assert any(p.grad is not None for p in net.parameters())


# ───────────────────── 2. Hölder α: keep 1.5 + ablate {1.5, 2.0} ─────────────

def test_kd_compare_has_proper_holder_arm():
    """cmd_kd_compare must offer BOTH the flatter pseudo-divergence (1.5) and the
    proper Cauchy-Schwarz control (2.0) - the pre-registered {1.5, 2.0} ablation."""
    src = inspect.getsource(L.cmd_kd_compare)
    assert '"holder_default"' in src and "1.5" in src
    assert '"holder_proper"' in src and "2.0" in src


def test_holder_alpha_semantics_numeric():
    """α=2.0 is proper (D(teacher:teacher)=0); α=1.5 is improper and its minimiser
    is the flatter renormalised teacher^0.5, NOT the teacher itself."""
    t_logits = torch.tensor([[2.0, 0.5, -1.0, 0.3]])
    d2 = float(L.holder_divergence(t_logits.clone(), t_logits, temperature=1.0, alpha=2.0))
    assert d2 < 1e-5                                       # proper at α=2
    d_teacher = float(L.holder_divergence(t_logits.clone(), t_logits, temperature=1.0, alpha=1.5))
    d_sqrt = float(L.holder_divergence(0.5 * t_logits, t_logits, temperature=1.0, alpha=1.5))
    assert d_teacher > 1e-3                                # student=teacher is NOT optimal
    assert d_sqrt < 1e-5                                   # student ∝ teacher^(α-1)=teacher^0.5 IS optimal


def test_holder_docstring_documents_flatter_target():
    doc = L.holder_divergence.__doc__ or ""
    assert "uncertainty transfer" in doc
    assert "teacher^(alpha - 1)" in doc


# ───────────────────── 3. HD95 reporting: raw vs post-processed ──────────────

def test_hd95_empty_region_penalty_convention():
    """Empty-vs-nonempty region -> the 373.1287 mm BraTS penalty; identical -> 0."""
    sp = np.array([1.0, 1.0, 1.0])
    empty = np.zeros((16, 16, 16), dtype=bool)
    blob = empty.copy()
    blob[4:8, 4:8, 4:8] = True
    assert abs(L._hausdorff95_binary(empty, blob, spacing=sp) - 373.1287) < 1e-3
    assert L._hausdorff95_binary(blob, blob, spacing=sp) == 0.0


def test_postprocessing_removes_far_island_and_improves_hd95():
    """A far-away false-positive island inflates RAW HD95; connected-component
    post-processing removes it and collapses HD95 to 0 - exactly why the reported
    path post-processes and the per-epoch teacher log is RAW/pessimistic."""
    pred = np.zeros((48, 48, 48), dtype=np.int64)
    pred[8:13, 8:13, 8:13] = 1            # main tumour-core blob (125 vox, class 1)
    pred[40:43, 40:43, 40:43] = 1         # far FP island (27 vox < min_component_size=50)
    target = np.zeros_like(pred)
    target[8:13, 8:13, 8:13] = 1          # GT == main blob (no island)

    pp = L.connected_component_postprocessing(pred)
    assert pp[41, 41, 41] == 0            # island removed
    assert pp[10, 10, 10] == 1            # main blob kept

    sp = np.array([1.0, 1.0, 1.0])
    raw = L.brats_metrics(torch.from_numpy(pred), torch.from_numpy(target), spacing=sp)
    post = L.brats_metrics(torch.from_numpy(pp), torch.from_numpy(target), spacing=sp)
    assert raw["hd95_TC"] > 5.0           # raw inflated by the far island
    assert post["hd95_TC"] == 0.0         # post-processed == GT -> 0


def test_brats_metrics_docstring_flags_raw():
    doc = L.brats_metrics.__doc__ or ""
    assert "RAW" in doc and "evaluate --lesion_wise" in doc
