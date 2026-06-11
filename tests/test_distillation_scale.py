"""RegionKDLoss must be invariant to patch size (the batchmean->per-voxel fix)."""
import torch

from ogap.losses.distillation import RegionKDLoss


def test_region_kd_is_patch_size_invariant():
    torch.manual_seed(0)
    kd = RegionKDLoss(temperature=2.0)
    s = torch.randn(1, 4, 2, 2, 2)
    t = torch.randn(1, 4, 2, 2, 2)
    small = kd(s, t)
    # Tile each voxel 2x per spatial axis: identical per-voxel KL multiset, 8x more
    # voxels. A batchmean reduction would scale ~8x; per-voxel mean stays equal.
    big = kd(s.repeat(1, 1, 2, 2, 2), t.repeat(1, 1, 2, 2, 2))
    assert torch.allclose(small, big, atol=1e-5)


def test_region_kd_identical_logits_is_zero():
    kd = RegionKDLoss(temperature=3.0)
    x = torch.randn(2, 4, 3, 3, 3)
    assert torch.allclose(kd(x, x), torch.zeros(()), atol=1e-6)
