"""Regression tests pinning the correct behaviour for the 2026-06-11 review fixes.

Each test encodes the intended post-fix behaviour and fails on the pre-fix code:

* focal_dice must NOT penalise a confident-correct region (the focal modulation
  belongs on the (1 - Dice) term, not on the prediction inside Dice).
* jacob_cov (NASWOT) must score an expressive net HIGHER than a collapsed one
  (the search maximises the proxy).
* the NAS proxy-validity gate must FLAG an anti-correlated proxy, not certify it.
* the group-balanced conformal worst-group must track the under-served (small /
  low-coverage) subgroup, not the largest one.
* SwinUNETRTeacher.dec3_channels must match MONAI decoder3 (feature_size*2).
* rician_noise must add noise WITHOUT rectifying signed (z-scored) foreground.
* the training DataLoader must stay in index order under deterministic mode.
* the field-strength parser must treat a negative/zero sentinel as MISSING.
* the modality-degradation pairwise test must use per-case values (a real p-value
  at the single-combination 4-modality level, not the degenerate p=1.0).
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch


# -- region_aware: focal_dice is no longer inverted ---------------------------
def test_focal_dice_low_loss_on_perfect_prediction():
    from ogap.losses.region_aware import RegionWeightedGDL

    loss = RegionWeightedGDL(use_focal=True)
    target = torch.zeros(1, 8, 8, 8, dtype=torch.long)
    target[0, :4] = 1
    target[0, 4:6] = 2
    target[0, 6:] = 3
    logits = torch.zeros(1, 4, 8, 8, 8)
    for c in range(4):
        logits[0, c][target[0] == c] = 20.0   # confident, correct everywhere
    value = float(loss(logits, target))
    assert value < 0.05, f"focal_dice on a near-perfect prediction should be ~0, got {value}"


# -- proxies: jacob_cov sign rewards expressive networks ----------------------
def test_jacob_cov_rewards_expressive_over_collapsed():
    from ogap.nas.proxies import jacob_cov

    torch.manual_seed(0)
    x = torch.randn(8, 4)
    expressive = torch.nn.Sequential(torch.nn.Linear(4, 32), torch.nn.ReLU(),
                                     torch.nn.Linear(32, 4))
    collapsed = torch.nn.Sequential(torch.nn.Linear(4, 4))
    with torch.no_grad():
        for p in collapsed.parameters():
            p.zero_()                          # maps every input to the same output
    assert jacob_cov(expressive, x.clone()) > jacob_cov(collapsed, x.clone())


# -- validation: an anti-correlated proxy is flagged, not certified -----------
def test_proxy_validation_flags_anticorrelated_proxy():
    from ogap.nas.validation import validate_proxies

    archs = [f"a{i}" for i in range(8)]
    dice = {a: 0.5 + 0.05 * i for i, a in enumerate(archs)}
    # "wrong_sign" is perfectly ANTI-correlated with Dice: maximising it picks the worst.
    proxy = {a: {"wrong_sign": float(-i)} for i, a in enumerate(archs)}
    res = validate_proxies(proxy, dice, primary_proxy="wrong_sign")
    assert res["report"]["wrong_sign"]["spearman"] < 0
    assert res["verdict"] == "anti_correlated"


# -- conformal: worst-group tracks the under-served subgroup -------------------
def test_conformal_worst_group_uses_coverage_basis_not_marginal_mean():
    from ogap.ood.conformal import GroupBalancedConformalPredictor

    rng = np.random.default_rng(0)
    cal = {"big": rng.standard_normal(500), "small": rng.standard_normal(15)}
    gb = GroupBalancedConformalPredictor(alpha=0.1).fit(cal)
    rep = gb.coverage_report()                 # no test scores -> Beta-CI lower bound
    assert rep["worst_group_coverage_basis"] == "beta_ci_lower"
    # The SMALL calibration group has the wider, lower Beta coverage interval, so it is
    # the least-reliable subgroup and must be reported as worst (the marginal mean would
    # have flagged the LARGE group instead).
    assert rep["worst_group"] == "small"


# -- teacher: SwinUNETR dec3 channels match MONAI -----------------------------
def test_swin_unetr_teacher_dec3_channels_match_monai():
    pytest.importorskip("monai")
    from ogap.models.teacher import SwinUNETRTeacher

    teacher = SwinUNETRTeacher(in_channels=4, num_classes=4, base=24)
    assert teacher.dec3_channels == 48        # MONAI decoder3 = feature_size*2


# -- physics_augment / legacy: Rician does not rectify signed (z-scored) data --
def test_rician_noise_does_not_rectify_signed_foreground():
    import torch.nn.functional as F

    from ogap.augmentations.physics_augment import rician_noise

    torch.manual_seed(0)
    # Smooth, mean-centred volume (low intrinsic sigma) so the SNR target injects noise;
    # signed (~half negative), like the real z-scored pipeline input.
    low = torch.randn(1, 2, 4, 4, 4)
    x = F.interpolate(low, size=(16, 16, 16), mode="trilinear", align_corners=False)[0]
    x = x - x.mean()
    frac_neg = float((x < 0).float().mean())
    out = rician_noise(x, torch.Generator().manual_seed(1), (1.0, 2.0))
    assert not torch.equal(out, x)            # noise actually injected
    # Gaussian (not magnitude) on signed input keeps roughly half the voxels negative,
    # rather than folding them all positive as sqrt((v+n1)^2 + n2^2) would.
    assert float((out < 0).float().mean()) > 0.5 * frac_neg


def test_legacy_rician_does_not_rectify_signed_foreground():
    from ogap.legacy import MRIPhysicsAugmentor

    np.random.seed(0)
    aug = MRIPhysicsAugmentor()
    x = np.random.standard_normal((2, 8, 8, 8)).astype(np.float32)
    frac_neg_before = float((x < 0).mean())
    out = aug._add_rician_noise(x.copy())
    assert float((out < 0).mean()) > 0.5 * frac_neg_before, "signed foreground was rectified"


# -- legacy: training DataLoader stays in index order under deterministic mode --
def test_train_loader_in_order_gated_on_determinism():
    from torch.utils.data import DataLoader

    from ogap.legacy import _build_loader_kwargs

    if "in_order" not in inspect.signature(DataLoader).parameters:
        pytest.skip("DataLoader has no in_order (torch < 2.6)")

    def build():
        return _build_loader_kwargs(
            batch_size=1, shuffle=True, num_workers=4, pin_memory=False,
            worker_init=None, train_mode=True, persistent_workers=True)

    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        det = build()
    finally:
        torch.use_deterministic_algorithms(False)
    nondet = build()
    assert det.get("in_order", True) is True          # ordered when deterministic
    assert nondet.get("in_order") is False            # throughput mode otherwise


# -- legacy: a negative/zero field-strength sentinel is treated as missing -----
def test_field_strength_sentinel_treated_as_missing():
    from ogap.legacy import _parse_row_field_strength

    assert _parse_row_field_strength({"field_strength": "-1"}) is None
    assert _parse_row_field_strength({"field_strength": "0"}) is None
    assert _parse_row_field_strength({"field_strength": "1.5"}) == pytest.approx(1.5)


# -- modality_robustness: degradation pairwise test uses per-case values --------
def test_degradation_pairwise_is_per_case_not_combo_means():
    from ogap.evaluation.modality_robustness import (
        combo_label, modality_degradation_analysis,
    )

    full = combo_label(frozenset({0, 1, 2, 3}))

    def cases(vals):
        return {f"c{i}": {"dsc_et": v} for i, v in enumerate(vals)}

    results = {
        "A": {full: cases([0.80, 0.82, 0.79, 0.81, 0.83, 0.80])},
        "B": {full: cases([0.60, 0.62, 0.59, 0.61, 0.63, 0.60])},
    }
    df = modality_degradation_analysis(results, metric="dsc_et")
    pcols = [c for c in df.columns if c.startswith("p_")]
    assert pcols, "expected a pairwise p-value column"
    at_n4 = df.loc[df["n_modalities"] == 4, pcols[0]].dropna()
    # Per-case test over the single 4-modality combination gives a REAL p-value;
    # the old per-combination-means version degenerated to p=1.0 (one observation).
    assert len(at_n4) > 0 and float(at_n4.min()) < 1.0
