import torch
from types import SimpleNamespace

from ogap.losses import build_task_loss
from ogap.losses.region_aware import RegionWeightedGDL


def _cfg(ttype):
    return SimpleNamespace(loss=SimpleNamespace(task=SimpleNamespace(type=ttype, num_classes=4)))


def test_legacy_matches_legacy_dicece():
    from ogap.legacy import DiceCELoss
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 16, 16, 16)
    target = torch.randint(0, 4, (2, 16, 16, 16))
    a = build_task_loss(_cfg("legacy"))(logits, target)
    b = DiceCELoss(4)(logits, target)
    assert torch.allclose(a, b, atol=1e-6)


def test_region_weighted_gdl_scalar_and_grad():
    logits = torch.randn(2, 4, 8, 8, 8, requires_grad=True)
    target = torch.randint(0, 4, (2, 8, 8, 8))
    out = build_task_loss(_cfg("region_weighted_gdl"))(logits, target)
    assert out.ndim == 0
    out.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_perfect_prediction_low_loss():
    target = torch.randint(1, 4, (1, 4, 4, 4))
    logits = torch.zeros(1, 4, 4, 4, 4)
    logits.scatter_(1, target.unsqueeze(1), 20.0)
    assert RegionWeightedGDL()(logits, target).item() < 0.05


def test_region_weights_affect_value():
    torch.manual_seed(1)
    logits = torch.randn(1, 4, 8, 8, 8)
    target = torch.randint(0, 4, (1, 8, 8, 8))
    a = RegionWeightedGDL({"WT": 1.0, "TC": 1.0, "ET": 1.0})(logits, target)
    b = RegionWeightedGDL({"WT": 1.0, "TC": 2.0, "ET": 4.0})(logits, target)
    assert not torch.allclose(a, b)


def test_focal_runs():
    logits = torch.randn(1, 4, 8, 8, 8, requires_grad=True)
    target = torch.randint(0, 4, (1, 8, 8, 8))
    out = build_task_loss(_cfg("focal_dice"))(logits, target)
    out.backward()
    assert torch.isfinite(out)


def test_dynamic_update_weights():
    loss = RegionWeightedGDL()
    loss.update_weights({"ET": 0.1})
    assert loss.region_weights["ET"] > loss.region_weights["WT"]


def test_region_sigmoid_builds_scalar_and_grad():
    logits = torch.randn(2, 4, 8, 8, 8, requires_grad=True)
    target = torch.randint(0, 4, (2, 8, 8, 8))
    out = build_task_loss(_cfg("region_sigmoid"))(logits, target)
    assert out.ndim == 0
    out.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_region_sigmoid_perfect_prediction_low_loss():
    from ogap.losses.region_aware import RegionSigmoidBatchDiceLoss
    target = torch.randint(1, 4, (2, 6, 6, 6))
    logits = torch.zeros(2, 4, 6, 6, 6)
    logits.scatter_(1, target.unsqueeze(1), 30.0)
    assert RegionSigmoidBatchDiceLoss()(logits, target).item() < 0.2


def test_region_sigmoid_prob_equals_softmax_region_sum():
    # sigmoid(logit_R) must equal Σ_{c∈C_R} softmax(z)_c (the log-sum-exp identity).
    torch.manual_seed(3)
    z = torch.randn(1, 4, 2, 2, 2)
    probs = torch.softmax(z, dim=1)
    p_tc_softmax = probs[:, 1] + probs[:, 3]
    logit_tc = torch.logsumexp(z[:, [1, 3]], dim=1) - torch.logsumexp(z[:, [0, 2]], dim=1)
    assert torch.allclose(torch.sigmoid(logit_tc), p_tc_softmax, atol=1e-5)


def test_unknown_type_raises():
    import pytest
    with pytest.raises(ValueError):
        build_task_loss(_cfg("nope"))
