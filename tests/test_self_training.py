"""Self-training / consistency adaptation tests (CPU)."""
import torch

from ogap.adaptation import (
    make_pseudo_labels, pseudo_label_loss, consistency_loss, SelfTrainingConfig,
)
from ogap.adaptation.self_training import rampup_weight


def test_pseudo_labels_confidence_mask():
    # One very confident voxel, one uniform (low-confidence) voxel.
    logits = torch.zeros(1, 4, 1, 1, 2)
    logits[0, 2, 0, 0, 0] = 20.0           # confident class 2
    # voxel 1 left at all-zeros -> uniform softmax (conf 0.25)
    pseudo, mask = make_pseudo_labels(logits, confidence_threshold=0.9)
    assert pseudo[0, 0, 0, 0].item() == 2
    assert bool(mask[0, 0, 0, 0]) is True
    assert bool(mask[0, 0, 0, 1]) is False  # uniform voxel excluded


def test_pseudo_label_loss_zero_when_nothing_confident():
    teacher = torch.zeros(1, 4, 2, 2, 2)    # uniform -> no confident voxel
    student = torch.randn(1, 4, 2, 2, 2, requires_grad=True)
    loss = pseudo_label_loss(student, teacher, confidence_threshold=0.99)
    assert float(loss.detach()) == 0.0
    loss.backward()                          # still differentiable (zero grad)


def test_pseudo_label_loss_drives_student_to_teacher():
    teacher = torch.zeros(1, 4, 2, 2, 2)
    teacher[:, 1] = 20.0                      # confident class 1 everywhere
    student = torch.zeros(1, 4, 2, 2, 2, requires_grad=True)
    loss = pseudo_label_loss(student, teacher, confidence_threshold=0.9)
    assert float(loss.detach()) > 0.0
    loss.backward()
    assert student.grad is not None and torch.isfinite(student.grad).all()


def test_consistency_loss_runs():
    weak = torch.zeros(1, 4, 2, 2, 2)
    weak[:, 3] = 15.0
    strong = torch.randn(1, 4, 2, 2, 2, requires_grad=True)
    loss = consistency_loss(weak, strong)
    assert torch.isfinite(loss)
    loss.backward()
    assert strong.grad is not None


def test_rampup_weight():
    cfg = SelfTrainingConfig(enabled=True, weight=2.0, rampup_epochs=10)
    assert rampup_weight(cfg, 0) == 0.0
    assert abs(rampup_weight(cfg, 5) - 1.0) < 1e-9
    assert rampup_weight(cfg, 100) == 2.0
    assert rampup_weight(SelfTrainingConfig(enabled=False), 5) == 0.0
