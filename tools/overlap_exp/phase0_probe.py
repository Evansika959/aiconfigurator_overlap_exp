#!/usr/bin/env python3
"""Phase 0 feasibility probes for the GEMM x all-reduce overlap experiment.

Nothing in Phase 1 is worth collecting until these three pass. Each answers a
question that, if answered the wrong way, makes the whole sweep meaningless.

P0.0  DOES THIS PYTHON PICK THE SAME GEMM KERNEL?
      The 1080-row GEMM baseline was collected under the system torch (2.9.1+cu129).
      The overlap run needs TRT-LLM, which lives in a venv on torch 2.10.0+cu130.
      A different cuBLAS could select a different tile shape, and then neither the
      baseline latencies nor the S_eff model transfer. Checked by name and grid.

P0.1  DOES CONCURRENCY ACTUALLY HAPPEN?
      The dominant failure mode. Two kernels on two streams may still serialise --
      the GEMM can fill every SM before NCCL is scheduled, or a stray sync can
      order them. Total wall time alone cannot tell "overlapped" from "the second
      one was cheap": both look fast. So each kernel is bracketed by CUDA events
      recorded against a COMMON reference event, which puts both on one timeline
      and gives the actual intersection of their intervals.

P0.2  DOES NCCL_MIN/MAX_CTAS STILL BITE UNDER CONTENTION?
      It works when the all-reduce runs alone (measured: 1->32 CTAs is 7-14x on
      latency). Under a co-running GEMM the hardware scheduler may not be able to
      place the CTAs NCCL asked for. If the knob stops working, the sweep has no
      SM axis. NCCL reads these at communicator init, so one process per value.

Prediction on record before running: for M=1024/N=K=4096 the GEMM has 128 tiles and
runs 2 waves on 108 SMs -- ncu measured its time-averaged occupancy at 0.5855, i.e.
~64 busy SMs. Giving 20-40 SMs to NCCL should leave the wave count unchanged and cost
almost nothing. For M=8192/N=K=16384 (4096 tiles, occupancy 0.9845) the same trade
should be expensive. If the two shapes do NOT diverge, the premise is wrong.

  LD_LIBRARY_PATH=<no gib> mpirun -n 2 --allow-run-as-root \
      ~/trtllm_venv/bin/python phase0_probe.py --ctas 8
"""

import argparse
import json
import os
import statistics
import sys
import time

# gIB shim rejects NCCL_{MIN,MAX}_CTAS and breaks NCCL on A100; unsetting the env
# vars is not enough, the directory has to come off the loader path
os.environ["LD_LIBRARY_PATH"] = ":".join(
    p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p and "gib" not in p)
os.environ.pop("NCCL_NET", None)

import torch
import torch.distributed as dist

# (M, N, K) -> chosen so the GEMM's solo latency is within ~2x of the all-reduce's,
# otherwise the shorter one hides entirely and the overlap fraction is uninformative
SHAPES = {
    "small_M": (8192, 4096, 4096),      # 1024 tiles, occupancy 0.935, ~1.19 ms @1200
    "large": (4096, 16384, 16384),      # 2048 tiles, occupancy ~0.95, ~8.7 ms @1200
}
AR_MIB = [64, 256]
REPS = 30


def timeline(fns, streams, reps=REPS):
    """Run fns concurrently on their streams; return each one's [start, end] in ms on a
    COMMON timeline, plus the wall time. Events on different streams are comparable via
    elapsed_time as long as they share a device and a reference event."""
    base = torch.cuda.Event(enable_timing=True)
    evs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
           for _ in fns]
    for f in fns:                                    # warm both paths
        for _ in range(3):
            f()
    torch.cuda.synchronize()

    spans = [[] for _ in fns]
    walls = []
    for _ in range(reps):
        torch.cuda.synchronize()
        base.record(torch.cuda.current_stream())
        t0 = time.perf_counter()
        for (f, s, (e0, e1)) in zip(fns, streams, evs):
            with torch.cuda.stream(s):
                s.wait_event(base)                   # all start from the same point
                e0.record(s)
                f()
                e1.record(s)
        torch.cuda.synchronize()
        walls.append((time.perf_counter() - t0) * 1e3)
        for i, (e0, e1) in enumerate(evs):
            spans[i].append((base.elapsed_time(e0), base.elapsed_time(e1)))
    med = [(statistics.median(s for s, _ in sp), statistics.median(e for _, e in sp))
           for sp in spans]
    return med, statistics.median(walls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctas", type=int, default=8)
    ap.add_argument("--comm-priority", type=int, default=0,
                    help="CUDA stream priority for the collective. Lower is HIGHER "
                         "priority on CUDA; -1..-5 is the usual high-priority range. "
                         "Needed because the hardware does not partition SMs: a GEMM "
                         "submits ~1000 blocks that fill all 108 SMs at once, and "
                         "NCCL's handful of CTAs then wait for one to retire. "
                         "Measured at priority 0: the all-reduce ran 2.2-10.7x slower "
                         "while the GEMM was untouched at 1.00-1.01x.")
    ap.add_argument("--comm-first", action="store_true",
                    help="issue the collective before the GEMM, so its CTAs are "
                         "resident before the GEMM floods the SMs")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", os.environ.get("RANK", "0")))
    world = int(os.environ.get("OMPI_COMM_WORLD_SIZE", os.environ.get("WORLD_SIZE", "2")))
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world))
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29611")
    # must be set BEFORE the communicator is built
    os.environ["NCCL_MIN_CTAS"] = str(a.ctas)
    os.environ["NCCL_MAX_CTAS"] = str(a.ctas)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)

    from tensorrt_llm._torch.distributed import (AllReduce, AllReduceFusionOp,
                                                 AllReduceParams)
    from tensorrt_llm.functional import AllReduceStrategy
    from tensorrt_llm.mapping import Mapping
    os.environ["TLLM_DISABLE_ALLREDUCE_AUTOTUNE"] = "1"

    mapping = Mapping(world_size=world, tp_size=world, rank=rank)
    ar_op = AllReduce(mapping=mapping, strategy=AllReduceStrategy.NCCL).cuda()
    ar_params = AllReduceParams(strategy=AllReduceStrategy.NCCL,
                                fusion_op=AllReduceFusionOp.NONE)

    lo, hi = torch.cuda.Stream.priority_range()
    prio = max(hi, min(lo, a.comm_priority))
    s_gemm = torch.cuda.Stream(priority=0)
    s_comm = torch.cuda.Stream(priority=prio)
    if rank == 0:
        print(f"[cfg] stream priority range {(lo, hi)}; comm stream at {prio}, "
              f"gemm at 0; comm_first={a.comm_first}; CTAs={a.ctas}", flush=True)
    out = {"ctas": a.ctas, "world": world, "rank": rank,
           "comm_priority": prio, "comm_first": a.comm_first, "rows": []}

    # ---- P0.0 -------------------------------------------------------------
    if rank == 0:
        from torch.profiler import ProfilerActivity, profile
        m, n, k = SHAPES["small_M"]
        x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
        w = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
        for _ in range(5):
            torch.nn.functional.linear(x, w)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            torch.nn.functional.linear(x, w)
            torch.cuda.synchronize()
        names = [e.name for e in prof.events()
                 if e.name and "gemm" in e.name.lower()]
        print(f"[P0.0] torch {torch.__version__}  GEMM kernel: "
              f"{names[0] if names else 'NOT FOUND'}", flush=True)
        out["gemm_kernel"] = names[0] if names else None
        del x, w
        torch.cuda.empty_cache()

    # ---- P0.1 / P0.2 ------------------------------------------------------
    for tag, (m, n, k) in SHAPES.items():
        x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
        w = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
        gemm = lambda: torch.nn.functional.linear(x, w)
        for mib in AR_MIB:
            buf = torch.ones(mib * 1024 * 1024 // 2, dtype=torch.bfloat16, device="cuda")
            ar = lambda: ar_op(buf, all_reduce_params=ar_params)

            (g,), _ = timeline([gemm], [s_gemm])
            t_gemm = g[1] - g[0]
            (c,), _ = timeline([ar], [s_comm])
            t_ar = c[1] - c[0]
            order = ([ar, gemm], [s_comm, s_gemm]) if a.comm_first else \
                    ([gemm, ar], [s_gemm, s_comm])
            spans, wall = timeline(*order)
            g2, c2 = (spans[1], spans[0]) if a.comm_first else (spans[0], spans[1])
            tg, tc = g2[1] - g2[0], c2[1] - c2[0]
            inter = max(0.0, min(g2[1], c2[1]) - max(g2[0], c2[0]))
            span = max(g2[1], c2[1]) - min(g2[0], c2[0])

            row = dict(shape=tag, m=m, n=n, k=k, ar_mib=mib, ctas=a.ctas,
                       t_gemm_solo=round(t_gemm, 4), t_ar_solo=round(t_ar, 4),
                       t_gemm_conc=round(tg, 4), t_ar_conc=round(tc, 4),
                       overlap_ms=round(inter, 4), span_ms=round(span, 4),
                       overlap_frac=round(inter / min(tg, tc), 4) if min(tg, tc) > 0 else 0,
                       speedup=round((t_gemm + t_ar) / span, 4) if span > 0 else 0,
                       gemm_slowdown=round(tg / t_gemm, 4),
                       ar_slowdown=round(tc / t_ar, 4))
            out["rows"].append(row)
            if rank == 0:
                print(f"[P0.1] {tag:>8} AR={mib:>3}MiB CTA={a.ctas:>2} | "
                      f"solo g={t_gemm:7.3f} c={t_ar:7.3f} | conc g={tg:7.3f} c={tc:7.3f} | "
                      f"overlap {inter:6.3f}ms ({row['overlap_frac'] * 100:5.1f}% of shorter) | "
                      f"span {span:7.3f} speedup {row['speedup']:.3f} | "
                      f"slow g×{row['gemm_slowdown']:.2f} c×{row['ar_slowdown']:.2f}",
                      flush=True)
            del buf
        del x, w
        torch.cuda.empty_cache()

    if rank == 0 and a.out:
        with open(a.out, "w") as fh:
            json.dump(out, fh, indent=1)
        print(f"wrote {a.out}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
