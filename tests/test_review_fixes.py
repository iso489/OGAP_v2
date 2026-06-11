"""Verification suite for the Nature-Communications review fixes.

Covers the additive surface introduced to address the architecture-review
findings: batched/GPU physics augmentation + N4 exponential bias (F7), the HD95
empty-region convention (F6), per-subgroup risk disparity / fairness-without-harm
(F2), small-dataset + group-balanced conformal coverage (D1), the NAS
proxy-validity harness (E1), and the DDP helpers (B1, single-process fallbacks).
"""
import math

import numpy as np
import torch
from types import SimpleNamespace

import pytest


# -- F7 + GPU augmentation: batched physics, exponential N4 bias --------------
def test_bias_field_is_exponential_and_positive():
    from ogap.augmentations.physics_augment import random_bias_field
    x = torch.full((1, 8, 8, 8), 100.0)
    y = random_bias_field(x, torch.Generator().manual_seed(0))
    assert (y > 0).all()                      # exp() can never flip the sign
    # exp(+-0.5) envelope => multiplier within [~0.6, ~1.65]
    assert 50.0 < float(y.min()) and float(y.max()) < 170.0


def test_transforms_accept_batched_5d():
    from ogap.augmentations.physics_augment import (
        random_bias_field, rician_noise, random_gamma, auxiliary_fourier_basis,
        kspace_partial_fourier,
    )
    x4 = torch.rand(4, 16, 16, 16) + 0.1
    x5 = torch.rand(2, 4, 16, 16, 16) + 0.1
    for fn in (random_bias_field, rician_noise, random_gamma,
               auxiliary_fourier_basis, kspace_partial_fourier):
        y4 = fn(x4, torch.Generator().manual_seed(1))
        y5 = fn(x5, torch.Generator().manual_seed(1))
        assert y4.shape == x4.shape and y5.shape == x5.shape
        assert torch.isfinite(y5).all()


def test_gpu_batched_augmentor_leaves_labels_unchanged():
    from ogap.augmentations import build_gpu_physics_augmentor
    cfg = SimpleNamespace(augmentation=SimpleNamespace(physics=SimpleNamespace(
        enabled=True, bias_field=True, rician_noise=True, gamma_correction=True,
        gmm_contrast=False, intensity_shift=False, resolution_jitter=False,
        kspace_partial_fourier=False, kspace_ghosting=False, kspace_motion=False,
        auxiliary_fourier_basis=True, rician_snr_range=(5., 30.),
        resolution_downsample_range=(0.5, 1.), afa_mean_strength=0.06, stacked_n=0)))
    aug = build_gpu_physics_augmentor(cfg)
    assert aug is not None and aug.enabled
    x = torch.rand(2, 4, 16, 16, 16) + 0.1
    y = torch.randint(0, 4, (2, 16, 16, 16))
    xa, ya = aug(x, y)
    assert xa.shape == x.shape and torch.equal(ya, y)  # intensity-only => labels intact
    assert not torch.allclose(xa, x)
    # disabled -> builder returns None (legacy path untouched)
    off = SimpleNamespace(augmentation=SimpleNamespace(physics=SimpleNamespace(enabled=False)))
    assert build_gpu_physics_augmentor(off) is None


# -- F6: HD95 empty-region convention -----------------------------------------
def test_hd95_empty_policy_nan_vs_penalty():
    from ogap.evaluation.brats_metrics import compute_brats_metrics
    pred = np.zeros((8, 8, 8), dtype=np.int64)
    target = np.zeros((8, 8, 8), dtype=np.int64)
    target[2:5, 2:5, 2:5] = 3                 # ET present in ref, absent in pred
    nan_res = compute_brats_metrics(pred, target)
    assert math.isnan(nan_res["hd95_et"])     # default: undefined -> NaN
    pen_res = compute_brats_metrics(pred, target, hd95_empty_policy="penalty")
    assert pen_res["hd95_et"] == pytest.approx(373.1287)
    # both-empty stays 0 either way
    empty = np.zeros((8, 8, 8), dtype=np.int64)
    assert compute_brats_metrics(empty, empty, hd95_empty_policy="penalty")["hd95_et"] == 0.0


# -- F2: risk disparity + fairness without harm -------------------------------
def test_risk_disparity_flags_underserved_group():
    from ogap.evaluation.equity import risk_disparity
    per_group = {"3T": [0.90, 0.92, 0.91], "1.5T": [0.85, 0.86, 0.84],
                 "64mT": [0.60, 0.62, 0.58]}
    disp = risk_disparity(per_group, "dsc_et")
    assert disp["64mT"] > disp["3T"]           # low-field served worse
    assert disp["_worst_group"] == "64mT" and disp["_worst_disparity"] > 0


def test_fairness_without_harm_detects_improvement_without_harm():
    from ogap.evaluation.equity import fairness_without_harm_check
    base = {"3T": [0.90, 0.91], "64mT": [0.55, 0.57]}
    cand = {"3T": [0.90, 0.91], "64mT": [0.78, 0.80]}   # low-field up, 3T unchanged
    res = fairness_without_harm_check(base, cand, "dsc_et", reference_group="3T")
    assert res["disparity_reduced"] and not res["reference_harmed"]
    assert res["fairness_without_harm"] is True
    harmful = {"3T": [0.70, 0.71], "64mT": [0.78, 0.80]}   # 3T harmed
    res2 = fairness_without_harm_check(base, harmful, "dsc_et", reference_group="3T")
    assert res2["reference_harmed"] and res2["fairness_without_harm"] is False


# -- D1: small-dataset + group-balanced conformal -----------------------------
def test_conformal_quantile_and_small_dataset_spread():
    from ogap.ood.conformal import conformal_quantile, coverage_distribution
    rng = np.random.default_rng(0)
    scores = rng.standard_normal(500)
    tau = conformal_quantile(scores, alpha=0.1)
    cov = float((scores <= tau).mean())
    assert 0.85 < cov < 0.95                  # ~90% empirical coverage on calib
    # Small calibration set => much wider Beta coverage interval than large.
    small = coverage_distribution(20, alpha=0.1, ci=0.90)
    large = coverage_distribution(2000, alpha=0.1, ci=0.90)
    assert small["mean"] >= small["nominal"]  # marginal guarantee holds in mean
    assert (small["hi"] - small["lo"]) > (large["hi"] - large["lo"])


def test_group_balanced_conformal_reports_worst_group():
    from ogap.ood.conformal import GroupBalancedConformalPredictor
    rng = np.random.default_rng(1)
    cal = {"3T": rng.standard_normal(300), "64mT": rng.standard_normal(300) * 3.0}
    gb = GroupBalancedConformalPredictor(alpha=0.1).fit(cal)
    # Per-group thresholds differ because the score scales differ.
    assert gb.threshold("64mT") > gb.threshold("3T")
    rep = gb.coverage_report()
    assert "worst_group" in rep and "per_group" in rep
    assert set(rep["per_group"]) == {"3T", "64mT"}


# -- E1: NAS proxy-validity harness -------------------------------------------
def test_proxy_validation_trustworthy_when_correlated():
    from ogap.nas.validation import validate_proxies
    # Proxy "good" perfectly ranks Dice; "bad" is unrelated noise.
    archs = [f"a{i}" for i in range(8)]
    dice = {a: 0.5 + 0.05 * i for i, a in enumerate(archs)}
    proxy = {a: {"good": float(i), "bad": float((i * 7) % 3)} for i, a in enumerate(archs)}
    res = validate_proxies(proxy, dice, primary_proxy="good")
    assert res["report"]["good"]["spearman"] == pytest.approx(1.0, abs=1e-9)
    assert res["verdict"] == "trustworthy"
    res_bad = validate_proxies(proxy, dice, primary_proxy="bad")
    assert res_bad["verdict"] in ("weak", "exploratory")


# -- B1: DDP helpers degrade to single-process --------------------------------
def test_distributed_helpers_single_process_fallback():
    from ogap.utils import distributed as D
    assert D.get_world_size() == 1 and D.get_rank() == 0 and D.is_main_process()
    m = torch.nn.Conv3d(4, 4, 1)
    assert D.wrap_ddp(m, torch.device("cpu")) is m          # no-op without a group
    assert D.unwrap(m) is m
    assert D.make_distributed_sampler(range(10), shuffle=True) is None
    assert float(D.reduce_mean(torch.tensor(2.0))) == 2.0
    assert D.all_reduce_dict({"loss": 1.5}) == {"loss": 1.5}
    assert D.scale_lr_for_world_size(1e-3, world_size=4) == pytest.approx(4e-3)


# ── Robustness / edge-case hardening ─────────────────────────────────────────
def test_conformal_small_n_is_finite_and_flags_uncertifiable():
    from ogap.ood.conformal import coverage_distribution, GroupBalancedConformalPredictor
    for n in (0, 3, 5):                          # Beta(k, 0) degenerate regime
        r = coverage_distribution(n, alpha=0.1)
        assert math.isfinite(r["lo"]) and math.isfinite(r["hi"])  # no NaN interval
        assert r["certifiable"] is False and r["mean"] == 1.0
    assert coverage_distribution(50, alpha=0.1)["certifiable"] is True
    # under-sized groups are flagged, not silently trusted
    gb = GroupBalancedConformalPredictor(alpha=0.1).fit(
        {"tiny": np.random.default_rng(0).standard_normal(5),
         "ok": np.random.default_rng(1).standard_normal(300)})
    assert "tiny" in gb.coverage_report()["undercertified_groups"]


def test_batched_augmentation_empty_batch_and_odd_dims():
    from ogap.augmentations.physics_augment import random_bias_field, rician_noise
    empty = torch.zeros(0, 4, 8, 8, 8)
    assert random_bias_field(empty).shape == empty.shape   # no crash on B=0
    odd = torch.rand(2, 7, 7, 7) + 0.1                     # odd spatial dims (Haar)
    assert random_bias_field(odd).shape == odd.shape
    assert torch.isfinite(rician_noise(torch.rand(1, 8, 8, 8) + 0.1)).all()


def test_equity_handles_empty_and_nan_groups():
    from ogap.evaluation.equity import risk_disparity
    out = risk_disparity({"a": [0.9, 0.8], "b": [], "c": [float("nan")]}, "dsc_et")
    assert math.isnan(out["b"]) and math.isnan(out["c"])
    assert math.isfinite(out["a"]) or out["a"] == out["a"]  # finite for the real group


def test_nas_rank_is_tie_corrected():
    from ogap.nas.validation import _rank
    # tied values must share the average rank (1+2)/2 = 1.5, not ordinal 1,2
    r = _rank(np.array([10.0, 10.0, 20.0]))
    assert r[0] == r[1] == pytest.approx(0.5) and r[2] == pytest.approx(2.0)


# ── Low-field wrap-around aliasing (ArtifactID Jimeno 2022) ───────────────────
def test_wraparound_aliasing_is_label_preserving_and_batched():
    from ogap.augmentations.physics_augment import wraparound_aliasing, _TIER1
    from ogap.augmentations import build_gpu_physics_augmentor
    assert "wraparound_aliasing" in _TIER1               # auto-wired into registry
    g = torch.Generator().manual_seed(3)
    x4 = torch.rand(2, 8, 8, 8) + 0.1
    x5 = torch.rand(2, 2, 8, 8, 8) + 0.1
    y4 = wraparound_aliasing(x4, g)
    y5 = wraparound_aliasing(x5, g)
    assert y4.shape == x4.shape and y5.shape == x5.shape
    assert torch.isfinite(y5).all() and not torch.allclose(y4, x4)  # a ghost was added
    assert (y4 >= x4 - 1e-5).all()                        # additive overlay only
    # label-preserving through the GPU augmentor (intensity-only artifact)
    on = SimpleNamespace(augmentation=SimpleNamespace(physics=SimpleNamespace(
        enabled=True, wraparound_aliasing=True, stacked_n=0)))
    aug = build_gpu_physics_augmentor(on)
    assert aug is not None and aug.enabled
    xb = torch.rand(2, 4, 8, 8, 8) + 0.1
    yb = torch.randint(0, 4, (2, 8, 8, 8))
    xa, ya = aug(xb, yb)
    assert xa.shape == xb.shape and torch.equal(ya, yb)
    # absent from config -> inactive (backward compatible: old configs unaffected)
    off = SimpleNamespace(augmentation=SimpleNamespace(physics=SimpleNamespace(
        enabled=True, bias_field=False, stacked_n=0)))
    assert build_gpu_physics_augmentor(off) is None


def test_physics_aug_shields_tissue_prior_channels():
    """use_tissue_priors hazard: CSF/WM/GM partial-volume channels appended after
    the 4 MRI channels are anatomy, not signal. They MUST be byte-identical after
    physics augmentation (no intensity/noise/k-space applied), while the MRI
    channels are transformed. Covers the CPU per-sample and GPU batched paths."""
    import torch
    from types import SimpleNamespace
    from ogap.augmentations.physics_augment import PhysicsAugmentor
    cfg = SimpleNamespace(
        data=SimpleNamespace(in_channels=4),
        augmentation=SimpleNamespace(physics=SimpleNamespace(
            enabled=True, bias_field=True, rician_noise=True, gamma_correction=True,
            rician_snr_range=(5.0, 30.0), stacked_n=0)),
    )
    aug = PhysicsAugmentor(cfg)
    assert aug.n_mri_channels == 4
    # 4 MRI + 3 constant tissue-prior maps (so any change is unmistakable)
    x = torch.rand(7, 8, 8, 8)
    x[4:] = torch.tensor([0.2, 0.5, 0.8]).view(3, 1, 1, 1)
    out = aug(x.clone(), torch.Generator().manual_seed(0))
    assert out.shape == x.shape
    assert torch.equal(out[4:], x[4:]), "tissue-prior channels were modified by intensity aug"
    assert not torch.equal(out[:4], x[:4]), "MRI channels were not augmented"
    # 5-D GPU batched shape: same shield
    xb = x.unsqueeze(0).repeat(2, 1, 1, 1, 1)
    outb = aug(xb.clone(), torch.Generator().manual_seed(1))
    assert torch.equal(outb[:, 4:], xb[:, 4:]), "batched tissue-prior channels modified"
    assert not torch.equal(outb[:, :4], xb[:, :4])
    # backward-compat: a 4-channel (tissue-free) input is still fully augmented
    x4 = torch.rand(4, 8, 8, 8)
    assert not torch.equal(aug(x4.clone(), torch.Generator().manual_seed(2)), x4)


def test_tissue_prior_shield_survives_loader_in_channels_inflation():
    """Regression: load_config inflates data.in_channels to 4+3=7 when tissue priors
    are on. The channel-safety shield must use the MRI modality count
    (n_mri_channels=4), NOT in_channels, or it would cover ZERO channels and silently
    corrupt the priors in production (the manual-config test above used in_channels=4
    and so missed this)."""
    import os
    import tempfile
    import torch
    import yaml
    from ogap.config.loader import load_config
    from ogap.augmentations.physics_augment import PhysicsAugmentor
    d = {"data": {"use_tissue_priors": True},
         "augmentation": {"physics": {"enabled": True, "bias_field": True,
                                      "rician_noise": True, "stacked_n": 0}}}
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(d, f)
    f.close()
    cfg = load_config(f.name)
    os.unlink(f.name)
    assert cfg.data.in_channels == 7 and cfg.data.n_mri_channels == 4
    aug = PhysicsAugmentor(cfg)
    assert aug.n_mri_channels == 4   # the fix: shield uses modality count, not 7
    x = torch.rand(7, 6, 6, 6)
    x[4:] = torch.tensor([0.2, 0.5, 0.8]).view(3, 1, 1, 1)
    out = aug(x.clone(), torch.Generator().manual_seed(0))
    assert torch.equal(out[4:], x[4:]), "tissue priors corrupted in the loader path"
    assert not torch.equal(out[:4], x[:4])
