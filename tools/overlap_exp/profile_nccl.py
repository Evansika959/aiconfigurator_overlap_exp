#!/usr/bin/env python3
"""Launch attributes of NCCL's all-reduce kernel, read by ncu on ONE rank.

The occupancy probe says a resident all-reduce puts its SM pair in a shared-memory
class too small for a cuBLAS block. This reads the kernel's own shared memory per block
and the carveout it runs under. ncu serialises and may replay the kernels of the process
it profiles, and a collective cannot be replayed alone, so only launch attributes are
asked for (they need no replay), on rank 0 only; ranks 1-3 run unprofiled.

  bash run_profile_nccl.sh        # writes data/sim_overlap/ncu_nccl_<ctas>.csv
"""
import os
import sys

os.environ.setdefault("LD_LIBRARY_PATH", "")
os.environ["LD_LIBRARY_PATH"] = ":".join(
    p for p in os.environ["LD_LIBRARY_PATH"].split(":") if p and "gib" not in p)
os.environ.pop("NCCL_NET", None)

import torch
import torch.distributed as dist

rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(rank)
dist.init_process_group("nccl", rank=rank, world_size=world)
buf = torch.ones(int(os.environ.get("AR_MIB", "64")) * 1024 * 1024 // 2, dtype=torch.bfloat16, device="cuda")
for _ in range(3):                       # warm up OUTSIDE the profiled range
    dist.all_reduce(buf)
torch.cuda.synchronize()
if rank == 0:
    torch.cuda.cudart().cudaProfilerStart()
dist.all_reduce(buf)
torch.cuda.synchronize()
if rank == 0:
    torch.cuda.cudart().cudaProfilerStop()
dist.all_reduce(buf)
torch.cuda.synchronize()
dist.destroy_process_group()
print(f"rank {rank} done", file=sys.stderr)
