import os, torch, torch.distributed as dist
lr = int(os.environ.get("LOCAL_RANK", 0))
rk = os.environ.get("RANK", "?")
print(f"[rank {rk}] LOCAL_RANK={lr} cuda_avail={torch.cuda.is_available()} "
      f"dev_count={torch.cuda.device_count()} CVD={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
torch.cuda.set_device(lr)
dist.init_process_group("nccl")
t = torch.ones(1, device=f"cuda:{lr}") * dist.get_rank()
dist.all_reduce(t)
exp = sum(range(dist.get_world_size()))
print(f"[rank {dist.get_rank()}/{dist.get_world_size()}] all_reduce={t.item():.0f} expect={exp} "
      f"{'OK' if abs(t.item()-exp)<1e-6 else 'MISMATCH'}", flush=True)
dist.destroy_process_group()
