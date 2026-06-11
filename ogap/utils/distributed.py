"""Single-node multi-GPU (DistributedDataParallel) helpers for OGAP.

Rorqual GPU nodes carry **4x H100 SXM5 connected by NVLink**, but the training
loop has historically used a single GPU (``--gpus=h100:1``), leaving three
quarters of a node's compute idle per run. This module is the missing
DistributedDataParallel layer: launched with ``torchrun --nproc_per_node=4`` (or
the matching SLURM ``srun``), it replicates the model across the 4 GPUs, shards
the data with a :class:`~torch.utils.data.distributed.DistributedSampler`, and
synchronises gradients over NVLink — ~4x throughput for the teacher / student /
ablation sweep with the NCCL ``NVL`` P2P level the launch scripts already set.

Every function degrades gracefully to a no-op when run as a single process
(``WORLD_SIZE`` unset / ==1), so the existing single-GPU and CPU paths are
unchanged and the helpers are unit-testable without a GPU. The only place that
must change in the training loop is: call :func:`init_distributed_from_env` once
at start, build the sampler via :func:`make_distributed_sampler`, wrap the model
with :func:`wrap_ddp`, gate logging/checkpointing on :func:`is_main_process`, and
reduce metrics with :func:`reduce_mean` / :func:`all_reduce_dict`.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass(frozen=True)
class DistInfo:
    """Resolved distributed topology for the current process."""
    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    backend: str
    device: torch.device


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def is_dist_available_and_initialized() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def get_world_size() -> int:
    return torch.distributed.get_world_size() if is_dist_available_and_initialized() else 1


def get_rank() -> int:
    return torch.distributed.get_rank() if is_dist_available_and_initialized() else 0


def get_local_rank() -> int:
    if is_dist_available_and_initialized():
        return _env_int("LOCAL_RANK", 0)
    return 0


def is_main_process() -> bool:
    """True on rank 0 (or in any single-process run). Gate logging / checkpoint
    writes / stdout on this so 4 ranks do not quadruple the I/O."""
    return get_rank() == 0


def init_distributed_from_env(prefer_backend: Optional[str] = None) -> DistInfo:
    """Initialise the process group from ``torchrun`` / SLURM env vars.

    Reads ``WORLD_SIZE`` / ``RANK`` / ``LOCAL_RANK`` (set by ``torchrun`` and by
    the SLURM launch script). When ``WORLD_SIZE <= 1`` this is a no-op and returns
    a single-process :class:`DistInfo` — so the historical single-GPU path is
    untouched. On CUDA it uses NCCL (NVLink) and pins the process to its
    ``LOCAL_RANK`` GPU; otherwise gloo.
    """
    world_size = _env_int("WORLD_SIZE", 1)
    rank = _env_int("RANK", 0)
    local_rank = _env_int("LOCAL_RANK", 0)
    cuda = torch.cuda.is_available()

    if world_size <= 1:
        device = torch.device(f"cuda:{local_rank}" if cuda else "cpu")
        return DistInfo(False, 0, 0, 1, "none", device)

    backend = prefer_backend or ("nccl" if cuda else "gloo")
    if not is_dist_available_and_initialized():
        torch.distributed.init_process_group(backend=backend, init_method="env://",
                                             world_size=world_size, rank=rank)
    if cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    return DistInfo(True, rank, local_rank, world_size, backend, device)


def cleanup_distributed() -> None:
    if is_dist_available_and_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def wrap_ddp(model: torch.nn.Module, device: torch.device,
             find_unused_parameters: bool = False) -> torch.nn.Module:
    """Wrap ``model`` in DDP when distributed, else return it unchanged.

    ``find_unused_parameters`` defaults False (faster); set True only if some
    parameters (e.g. an optional auxiliary head not always used in a forward)
    receive no gradient — DDP raises otherwise.
    """
    if not is_dist_available_and_initialized():
        return model
    device_ids = [device.index] if device.type == "cuda" else None
    return torch.nn.parallel.DistributedDataParallel(
        model, device_ids=device_ids,
        output_device=device.index if device.type == "cuda" else None,
        find_unused_parameters=find_unused_parameters,
    )


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying module (``.module`` under DDP) for state_dict / export."""
    return model.module if hasattr(model, "module") else model


def make_distributed_sampler(dataset, *, shuffle: bool, seed: int = 0):
    """A :class:`DistributedSampler` when distributed, else ``None`` (so the
    caller falls back to its existing sampler/shuffle). Remember to call
    ``sampler.set_epoch(epoch)`` each epoch for correct shuffling across ranks.
    """
    if not is_dist_available_and_initialized():
        return None
    from torch.utils.data.distributed import DistributedSampler
    return DistributedSampler(dataset, shuffle=shuffle, seed=seed, drop_last=False)


class DistributedWeightedSampler(torch.utils.data.Sampler):
    """Per-rank shard of a weighted sampler (keeps the metadata/equity weighting).

    A plain :class:`DistributedSampler` would drop OGAP's field-strength / phenotype
    upweighting, which is load-bearing for the equity story. This instead gives each
    rank its own weighted draw of ``ceil(total / world_size)`` indices (the base
    ``WeightedRandomSampler`` is already with-replacement, so per-rank independent
    draws preserve the intended marginal weighting). Call ``set_epoch`` each epoch.
    """

    def __init__(self, weights, num_samples_total: int, *,
                 num_replicas: Optional[int] = None, rank: Optional[int] = None,
                 seed: int = 0, replacement: bool = True) -> None:
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_replicas = int(num_replicas if num_replicas is not None else get_world_size())
        self.rank = int(rank if rank is not None else get_rank())
        self.seed = int(seed)
        self.replacement = bool(replacement)
        self.epoch = 0
        import math as _math
        self.num_samples = int(_math.ceil(int(num_samples_total) / max(1, self.num_replicas)))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch * 131 + self.rank * 977)
        idx = torch.multinomial(self.weights, self.num_samples, self.replacement, generator=g)
        return iter(idx.tolist())


def maybe_distributed_weighted_sampler(base_sampler, *, seed: int = 0):
    """When distributed, return a :class:`DistributedWeightedSampler` sharding the
    base ``WeightedRandomSampler``'s weights; otherwise return ``base_sampler``
    unchanged (byte-identical single-process path)."""
    if not is_dist_available_and_initialized() or base_sampler is None:
        return base_sampler
    weights = getattr(base_sampler, "weights", None)
    if weights is None:
        return base_sampler  # non-weighted base: leave as-is
    total = int(getattr(base_sampler, "num_samples", len(weights)))
    return DistributedWeightedSampler(weights, total, seed=seed)


def reduce_mean(value) -> torch.Tensor:
    """All-reduce a scalar/tensor to its cross-rank mean (identity if single-proc)."""
    if not is_dist_available_and_initialized():
        return value if torch.is_tensor(value) else torch.as_tensor(value)
    t = value.detach().clone() if torch.is_tensor(value) else torch.as_tensor(value)
    t = t.to(torch.cuda.current_device() if torch.cuda.is_available() else "cpu", dtype=torch.float64)
    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
    return t / get_world_size()


def all_reduce_gradients(parameters) -> None:
    """Average gradients across ranks in place (manual DDP for multi-module training).

    For models assembled from several separately-called modules (e.g. the student
    backbone + feature projectors + auxiliary/domain heads), wrapping each in DDP does
    not sync grads for submodules invoked by index rather than via the wrapped
    ``forward``. Averaging the gradients of *all* trainable params after backward is
    structure-independent and always correct. No-op single-process.
    """
    if not is_dist_available_and_initialized():
        return
    ws = get_world_size()
    for p in parameters:
        if p.grad is not None:
            torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.SUM)
            p.grad /= ws


def all_reduce_dict(metrics: Dict[str, float]) -> Dict[str, float]:
    """Cross-rank mean of a ``{name: scalar}`` metric dict (for logging)."""
    if not is_dist_available_and_initialized() or not metrics:
        return dict(metrics)
    keys = sorted(metrics)
    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    t = torch.tensor([float(metrics[k]) for k in keys], dtype=torch.float64, device=device)
    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
    t = t / get_world_size()
    return {k: float(v) for k, v in zip(keys, t.tolist())}


@contextmanager
def main_process_first():
    """Run the body on rank 0 first, then the others — for one-time setup such as
    cache priming that must not race across ranks."""
    if is_dist_available_and_initialized() and not is_main_process():
        torch.distributed.barrier()
    try:
        yield
    finally:
        if is_dist_available_and_initialized() and is_main_process():
            torch.distributed.barrier()


def scale_lr_for_world_size(base_lr: float, world_size: Optional[int] = None) -> float:
    """Linear LR scaling rule (Goyal et al. 2017): multiply the base LR by the
    number of ranks, since the effective batch size grows with the world size."""
    ws = world_size if world_size is not None else get_world_size()
    return float(base_lr) * max(1, int(ws))
