#!/usr/bin/env python3
"""Is TensorRT-LLM's AllReduce(strategy=NCCL) the same thing as a plain NCCL all-reduce?

The comm sweep was collected through tensorrt_llm._torch.distributed.AllReduce with
AllReduceStrategy.NCCL and AllReduceFusionOp.NONE. Reading the Python source only gets
as far as torch.ops.trtllm.allreduce -- a C++ op inside libtensorrt_llm.so, which is
shipped as a binary. So this answers the question empirically instead: run both paths
on the same buffers and compare the CUDA kernels each one actually launches.

Equivalent means: the same device kernels, the same number of them per call, and the
same latency. Anything TRT-LLM adds on top -- an extra copy, a fusion epilogue, a
different collective algorithm -- would show up as a differing kernel list.

  cd tools/comm_dvfs_char
  LD_LIBRARY_PATH=$(python3 -c "import os;print(':'.join(p for p in os.environ.get('LD_LIBRARY_PATH','').split(':') if p and 'gib' not in p))") \
    mpirun -n 2 --allow-run-as-root ~/trtllm_venv/bin/python verify_nccl_equivalence.py
"""

import os
import sys
import time
from collections import Counter

# the gIB shim on this node rejects NCCL settings and breaks NCCL on A100; unsetting
# the env vars is not enough, the directory has to come off the loader path
os.environ["LD_LIBRARY_PATH"] = ":".join(
    p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p and "gib" not in p)
os.environ.pop("NCCL_NET", None)

import torch
import torch.distributed as dist


SIZES_MIB = [8, 64, 256]
ITERS = 20


def kernels_of(fn, warmup=5, iters=ITERS):
    """-> (Counter of CUDA kernel names, median ms per call). Uses the torch profiler,
    which reports the device-side kernel names, not the host-side op names."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1e3

    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    names = Counter()
    for e in prof.events():
        if getattr(e, "device_type", None) is not None and str(e.device_type).endswith("CUDA"):
            if e.name and not e.name.startswith("cuda"):
                names[e.name] += 1
    return names, ms


def main():
    rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", os.environ.get("RANK", "0")))
    world = int(os.environ.get("OMPI_COMM_WORLD_SIZE", os.environ.get("WORLD_SIZE", "2")))
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29601")
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)

    from tensorrt_llm._torch.distributed import (AllReduce, AllReduceFusionOp,
                                                 AllReduceParams)
    from tensorrt_llm.functional import AllReduceStrategy
    from tensorrt_llm.mapping import Mapping

    mapping = Mapping(world_size=world, tp_size=world, rank=rank)
    ar = AllReduce(mapping=mapping, strategy=AllReduceStrategy.NCCL)
    params = AllReduceParams(strategy=AllReduceStrategy.NCCL,
                             fusion_op=AllReduceFusionOp.NONE)

    for mib in SIZES_MIB:
        n = mib * 1024 * 1024 // 2                      # bf16
        t = torch.ones(n, dtype=torch.bfloat16, device=f"cuda:{rank}")

        k_torch, ms_torch = kernels_of(lambda: dist.all_reduce(t))
        k_trt, ms_trt = kernels_of(lambda: ar(t, all_reduce_params=params))

        if rank == 0:
            print(f"\n{'=' * 78}\n{mib} MiB bf16, world_size={world}\n{'=' * 78}")
            for lab, k, ms in (("torch.distributed.all_reduce", k_torch, ms_torch),
                               ("TRT-LLM AllReduce(NCCL, fusion=NONE)", k_trt, ms_trt)):
                print(f"  {lab}:  {ms:.3f} ms/call")
                for name, c in k.most_common():
                    print(f"      {c / ITERS:>5.1f} launches/call   {name}")
            same = set(k_torch) == set(k_trt)
            print(f"  -> same kernel set: {same};  "
                  f"latency ratio TRT-LLM/torch = {ms_trt / ms_torch:.3f}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
