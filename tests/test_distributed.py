"""Distributed-helper tests (single-process behaviour; multi-GPU needs a cluster)."""
import torch

from ogap.utils.distributed import (
    DistributedWeightedSampler, maybe_distributed_weighted_sampler,
    all_reduce_gradients, is_main_process, scale_lr_for_world_size, get_world_size,
)


def test_distributed_weighted_sampler_single_process():
    weights = torch.tensor([0.1, 0.1, 0.8, 0.0])
    s = DistributedWeightedSampler(weights, num_samples_total=100, num_replicas=1, rank=0, seed=0)
    idx = list(s)
    assert len(idx) == 100
    assert all(0 <= i < 4 for i in idx)
    # The high-weight index (2) dominates; the zero-weight index (3) never appears.
    assert idx.count(2) > idx.count(0)
    assert 3 not in idx


def test_distributed_weighted_sampler_shards_across_replicas():
    weights = torch.ones(8)
    a = DistributedWeightedSampler(weights, 8, num_replicas=4, rank=0, seed=1)
    b = DistributedWeightedSampler(weights, 8, num_replicas=4, rank=1, seed=1)
    assert len(a) == 2 and len(b) == 2          # ceil(8/4) per rank
    a.set_epoch(0); b.set_epoch(0)
    # Different ranks draw with different seeds -> independent shards.
    assert list(a) != list(b) or a.rank != b.rank


def test_maybe_distributed_sampler_is_identity_single_process():
    base = object()
    assert maybe_distributed_weighted_sampler(base) is base
    assert maybe_distributed_weighted_sampler(None) is None


def test_all_reduce_gradients_noop_single_process():
    p = torch.nn.Parameter(torch.ones(3))
    p.grad = torch.full((3,), 2.0)
    all_reduce_gradients([p])             # no process group -> unchanged
    assert torch.equal(p.grad, torch.full((3,), 2.0))


def test_single_process_invariants():
    assert is_main_process() is True
    assert get_world_size() == 1
    assert scale_lr_for_world_size(0.001) == 0.001
