#!/usr/bin/env python3
"""One self-contained example: a GEMM and an all-reduce genuinely running at once.

    M=8192, N=4096, K=4096 bf16 GEMM  ||  64 MiB all-reduce, world=2, 4 NCCL CTAs

RUN -- no launcher, no venv, nothing to install:

    sudo nvidia-smi -i 0,1 -lgc 1200,1200      # optional; the script reports the clock
    python3 demo_overlap.py
    sudo nvidia-smi -i 0,1 -rgc

WHY torch.distributed AND NOT TensorRT-LLM. An earlier version drove the collective
through tensorrt_llm's AllReduce(strategy=NCCL). `import tensorrt_llm` costs 23.9 s
per process (it pulls in transformers, modelopt, torchao), and two ranks importing at
once pushed the demo past three minutes -- for a collective that is bit-identical
either way. Verified in verify_nccl_equivalence.py: both paths launch exactly one
kernel per call, the same one, `ncclDevKernel_AllReduce_Sum_bf16_RING_LL`, within 4 %
on latency. `--backend trtllm` still selects that path if you want to see it.

THE FOUR THINGS THAT MAKE THE OVERLAP HAPPEN

1.  TWO FRESH, NON-DEFAULT STREAMS. Work on one stream is ordered by the hardware,
    and the default stream also synchronises against the others, so each kernel gets
    its own newly created stream.

2.  THE COLLECTIVE'S STREAM AT PRIORITY -3 (A100's range is 0..-3). Not optional.
    A GEMM submits ~1000 blocks that fill all 108 SMs within microseconds, and a
    resident block runs to completion -- A100 does not preempt at block granularity.
    An NCCL collective is ONE persistent kernel whose n_cta blocks stay resident for
    the whole reduction. So the loser of that race waits. Measured here with equal
    priorities: the all-reduce ran 10.6x slower and the pair finished SLOWER than
    running them back to back.

3.  THE COLLECTIVE IS ISSUED FIRST, so its 4 CTAs land and hold their SMs before the
    GEMM floods the rest.

4.  A COMMON REFERENCE EVENT. `base` is recorded once; each stream does
    `wait_event(base)` before its first kernel, so both start from the same instant.
    Without it the second stream starts later by whatever the first launch cost on
    the host, and the measured "overlap" is partly an artefact of issue order.

WHAT EACH MEASUREMENT PROVES, AND WHAT IT DOES NOT

    CUDA events     exact [start, end] per kernel on one timeline -> their real
                    intersection. Shows they were RESIDENT at the same time.
    speedup > 1     shows they made progress SIMULTANEOUSLY. Time-slicing one set of
                    SMs would land near 1.0; a real speedup cannot come from that.
    torch.profiler  the picture: one row per stream, overlapping bars.
    NVML            board power on BOTH GPUs. A collective is a node-level event, so
                    reading one board under-reports it.

    None of these reports SM *assignment*. ncu could, but it serialises kernels to
    read counters, which destroys the concurrency under study. The SM split is
    therefore INFERRED at the end from the GEMM's slowdown, and labelled as such.
"""

import argparse
import math
import os
import statistics
import threading
import time

os.environ.setdefault("LD_LIBRARY_PATH", "")
# GCP's gIB NCCL shim breaks NCCL on A100 and rejects NCCL_{MIN,MAX}_CTAS; unsetting
# the env vars is not enough, the directory must come off the loader path
os.environ["LD_LIBRARY_PATH"] = ":".join(
    p for p in os.environ["LD_LIBRARY_PATH"].split(":") if p and "gib" not in p)
os.environ.pop("NCCL_NET", None)

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

M, N, K = 8192, 4096, 4096
AR_MIB = 64
CTAS = 4
TOTAL_SM = 108
REPS = 15
POWER_WINDOW_S = 2.5
DISCARD_S = 0.6          # NVML needs ~427 ms to reach 95 % of steady state
TRACE = "trace_overlap.json"


class NodePower:
    """NVML board power on every participating GPU, sampled in a background thread."""

    def __init__(self, n):
        import pynvml
        self.nv = pynvml
        pynvml.nvmlInit()
        self.h = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(n)]
        self._stop, self._rows = threading.Event(), []

    def _loop(self):
        while not self._stop.is_set():
            try:
                self._rows.append(
                    (time.time(),
                     [self.nv.nvmlDeviceGetPowerUsage(x) / 1000 for x in self.h],
                     [self.nv.nvmlDeviceGetClockInfo(x, self.nv.NVML_CLOCK_SM)
                      for x in self.h]))
            except Exception:
                pass
            self._stop.wait(0.05)

    def __enter__(self):
        self._rows.clear(); self._stop.clear()
        self._t = threading.Thread(target=self._loop, daemon=True); self._t.start()
        return self

    def __exit__(self, *_):
        self._stop.set(); self._t.join(timeout=2)

    def summary(self):
        if not self._rows:
            return 0.0, 0
        t0 = self._rows[0][0]
        kept = [r for r in self._rows if r[0] - t0 >= DISCARD_S] or self._rows
        node = sum(statistics.median(r[1][i] for r in kept) for i in range(len(self.h)))
        clk = int(statistics.median(r[2][0] for r in kept))
        return node, clk


def free_port():
    """A fixed MASTER_PORT is a trap: a previous run that hung leaves the rendezvous
    socket bound, and the next one dies with EADDRINUSE on rank 0 while rank 1 fails
    with a confusing NCCL 'remote process exited' error."""
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return str(s.getsockname()[1])


def worker(rank, world, backend, port, q):
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world),
                      MASTER_ADDR="127.0.0.1", MASTER_PORT=port,
                      NCCL_MIN_CTAS=str(CTAS), NCCL_MAX_CTAS=str(CTAS))
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    log = (lambda *a: print(*a, flush=True)) if rank == 0 else (lambda *a: None)

    # ---- 1. two fresh streams; the collective gets the highest priority ----------
    lo, hi = torch.cuda.Stream.priority_range()          # A100 -> (0, -3)
    s_gemm = torch.cuda.Stream(priority=0)
    s_comm = torch.cuda.Stream(priority=hi)
    cur = torch.cuda.current_stream()

    x = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
    w = torch.randn((N, K), dtype=torch.bfloat16, device="cuda")
    buf = torch.ones(AR_MIB * 1024 * 1024 // 2, dtype=torch.bfloat16, device="cuda")
    gemm = lambda: torch.nn.functional.linear(x, w)

    if backend == "trtllm":
        from tensorrt_llm._torch.distributed import (AllReduce, AllReduceFusionOp,
                                                     AllReduceParams)
        from tensorrt_llm.functional import AllReduceStrategy
        from tensorrt_llm.mapping import Mapping
        os.environ["TLLM_DISABLE_ALLREDUCE_AUTOTUNE"] = "1"
        _op = AllReduce(mapping=Mapping(world_size=world, tp_size=world, rank=rank),
                        strategy=AllReduceStrategy.NCCL).cuda()
        _p = AllReduceParams(strategy=AllReduceStrategy.NCCL,
                             fusion_op=AllReduceFusionOp.NONE)
        allreduce = lambda: _op(buf, all_reduce_params=_p)
    else:
        allreduce = lambda: dist.all_reduce(buf)

    for _ in range(10):
        gemm(); allreduce()
    torch.cuda.synchronize()

    def issue(mode):
        """Enqueue one iteration. `base` is returned so spans share a timeline."""
        base = torch.cuda.Event(enable_timing=True)
        base.record(cur)
        if mode == "serial":                    # one stream => the hardware orders them
            return base, [(cur, gemm), (cur, allreduce)]
        # collective first, so its CTAs are resident before the GEMM floods the SMs
        return base, [(s_comm, allreduce), (s_gemm, gemm)]

    def timed(mode, reps=REPS):
        eg = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        ec = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        gs, cs = [], []
        for _ in range(reps):
            torch.cuda.synchronize()
            base, plan = issue(mode)
            for (st, fn) in plan:
                ev = eg if fn is gemm else ec
                if st is cur:
                    ev[0].record(st); fn(); ev[1].record(st)
                else:
                    with torch.cuda.stream(st):
                        st.wait_event(base)     # both streams start from the same instant
                        ev[0].record(st); fn(); ev[1].record(st)
            torch.cuda.synchronize()
            gs.append((base.elapsed_time(eg[0]), base.elapsed_time(eg[1])))
            cs.append((base.elapsed_time(ec[0]), base.elapsed_time(ec[1])))
        m = lambda v: (statistics.median(a for a, _ in v), statistics.median(b for _, b in v))
        return m(gs), m(cs)

    def looped(mode, span_ms):
        """Repeat for POWER_WINDOW_S while sampling both boards: one iteration is a
        couple of ms and NVML cannot resolve that.

        THE ITERATION COUNT MUST BE IDENTICAL ON EVERY RANK. It is derived from a
        measured span, and the two ranks measure spans that differ by microseconds --
        enough for int() to land one apart. The rank that computed the larger count
        then issues one more collective than anyone joins, and the job hangs with one
        GPU pinned at 100% and the other idle. Rank 0 decides and broadcasts."""
        n = torch.tensor([max(4, int(POWER_WINDOW_S / (span_ms / 1e3)))],
                         dtype=torch.int64, device="cuda")
        dist.broadcast(n, src=0)
        iters = int(n.item())
        with NodePower(world) as p:
            t0 = time.perf_counter()
            for _ in range(iters):
                base, plan = issue(mode)
                for (st, fn) in plan:
                    if st is cur:
                        fn()
                    else:
                        with torch.cuda.stream(st):
                            st.wait_event(base); fn()
                torch.cuda.synchronize()
            wall = (time.perf_counter() - t0) / iters * 1e3
            node_w, clk = p.summary()
        return wall, node_w, clk

    # ---- 2. solo references ------------------------------------------------------
    (sg, sc) = timed("serial")
    t_g, t_c = sg[1] - sg[0], sc[1] - sc[0]
    ser_ms, ser_w, clk = looped("serial", sc[1] - sg[0])

    # ---- 3. concurrent -----------------------------------------------------------
    (cg, cc) = timed("concurrent")
    span = max(cg[1], cc[1]) - min(cg[0], cc[0])
    inter = max(0.0, min(cg[1], cc[1]) - max(cg[0], cc[0]))
    con_ms, con_w, _ = looped("concurrent", span)

    # ---- 4. profiler timeline ----------------------------------------------------
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(5):
            base, plan = issue("concurrent")
            for (st, fn) in plan:
                with torch.cuda.stream(st):
                    st.wait_event(base); fn()
            torch.cuda.synchronize()
    if rank == 0:
        prof.export_chrome_trace(TRACE)
        ka = prof.key_averages()

    if rank == 0:
        q.put(dict(t_g=t_g, t_c=t_c, sg=sg, sc=sc, cg=cg, cc=cc, span=span, inter=inter,
                   ser_ms=ser_ms, ser_w=ser_w, con_ms=con_ms, con_w=con_w, clk=clk,
                   prio=(lo, hi),
                   ktable=ka.table(sort_by="self_device_time_total", row_limit=6)))
    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=("torch", "trtllm"), default="torch")
    a = ap.parse_args()
    world = 2
    assert torch.cuda.device_count() >= world, "needs 2 GPUs"

    mp.set_start_method("spawn", force=True)
    q = mp.Queue()
    port = free_port()
    ps = [mp.Process(target=worker, args=(r, world, a.backend, port, q))
          for r in range(world)]
    for p in ps:
        p.start()
    r = q.get(timeout=900)
    for p in ps:
        p.join(timeout=120)

    tiles = math.ceil(M / 128) * math.ceil(N / 256)     # 128x256 tiles on torch 2.10;
    waves = math.ceil(tiles / TOTAL_SM)                 # count is the same for 256x128
    s_eff = tiles / waves
    # per-GPU static floor, measured by zeus's profile_p2p.py at each locked clock
    # (data/p0_static_zeus.csv). Interpolated so any achieved clock is covered.
    ZEUS = [(300, 56.37), (510, 57.60), (705, 59.38), (900, 61.36),
            (1200, 69.66), (1410, 85.37)]
    c = r["clk"]
    if c <= ZEUS[0][0]:
        ps1 = ZEUS[0][1]
    elif c >= ZEUS[-1][0]:
        ps1 = ZEUS[-1][1]
    else:
        ps1 = next(y0 + (y1 - y0) * (c - x0) / (x1 - x0)
                   for (x0, y0), (x1, y1) in zip(ZEUS, ZEUS[1:]) if x0 <= c <= x1)
    p_static = ps1 * world
    E_ser, E_con = r["ser_w"] * r["ser_ms"], r["con_w"] * r["con_ms"]

    print(f"\n{'=' * 76}")
    print(f"GEMM M={M} N={N} K={K} bf16  ||  all-reduce {AR_MIB} MiB, world={world}, "
          f"{CTAS} NCCL CTAs")
    print(f"achieved SM clock {r['clk']} MHz   stream priority range {r['prio']}, "
          f"collective at {r['prio'][1]}, issued first")
    print(f"GEMM grid {tiles} blocks -> {waves} waves on {TOTAL_SM} SMs, S_eff {s_eff:.1f}")
    print("=" * 76)
    print(f"\n{'':>13}{'GEMM [start, end] ms':>26}{'all-reduce [start, end] ms':>30}")
    print(f"{'serial':>13}{f'[{r['sg'][0]:.3f}, {r['sg'][1]:.3f}]':>26}"
          f"{f'[{r['sc'][0]:.3f}, {r['sc'][1]:.3f}]':>30}")
    print(f"{'concurrent':>13}{f'[{r['cg'][0]:.3f}, {r['cg'][1]:.3f}]':>26}"
          f"{f'[{r['cc'][0]:.3f}, {r['cc'][1]:.3f}]':>30}")
    print(f"\n  intersection of the two intervals: {r['inter']:.3f} ms "
          f"({r['inter'] / min(r['cg'][1] - r['cg'][0], r['cc'][1] - r['cc'][0]) * 100:.0f}% "
          f"of the shorter kernel)")
    print(f"  serial takes {r['sc'][1]:.3f} ms end to end; concurrent {r['span']:.3f} ms")

    print(f"\n{'':>27}{'serial':>11}{'concurrent':>13}{'change':>10}")
    rows = [("wall per iteration (ms)", r["ser_ms"], r["con_ms"]),
            ("node power, 2 GPUs (W)", r["ser_w"], r["con_w"]),
            ("energy per iteration (mJ)", E_ser, E_con)]
    if p_static:
        rows += [("  of which static (mJ)", p_static * r["ser_ms"], p_static * r["con_ms"]),
                 ("  of which dynamic (mJ)", E_ser - p_static * r["ser_ms"],
                  E_con - p_static * r["con_ms"])]
    for lab, a_, b_ in rows:
        print(f"{lab:>27}{a_:>11.1f}{b_:>13.1f}{(b_ / a_ - 1) * 100:>9.1f}%")
    print(f"\n  speedup {r['ser_ms'] / r['con_ms']:.3f}x  "
          f"(ceiling if perfectly overlapped: "
          f"{(r['t_g'] + r['t_c']) / max(r['t_g'], r['t_c']):.3f}x)")
    print(f"  energy  {(1 - E_con / E_ser) * 100:.1f}% lower")

    slow = (r["cg"][1] - r["cg"][0]) / r["t_g"]
    print(f"\n  GEMM alone {r['t_g']:.3f} ms -> {r['cg'][1] - r['cg'][0]:.3f} ms while "
          f"sharing ({slow:.3f}x).")
    if slow > 1.01:
        print(f"  INFERRED, not measured: if latency goes as 1/S_eff that is "
              f"{s_eff:.1f} -> {s_eff / slow:.1f}\n  effective SMs, i.e. ~"
              f"{s_eff - s_eff / slow:.0f} conceded to the {CTAS} NCCL CTAs.")
    else:
        print(f"  The GEMM did not slow down, so no SM concession can be inferred from "
              f"timing.\n  With {CTAS} CTAs the collective's footprint is small enough "
              f"to hide in the tail\n  waves the GEMM already leaves idle "
              f"({s_eff:.1f}/{TOTAL_SM} SMs busy on average).")
    print(f"  Kernel-to-SM assignment cannot be read while two kernels run: ncu "
          f"serialises\n  them to collect counters, removing the concurrency in question.")
    print(f"\n--- profiler, top device kernels ---\n{r['ktable']}")
    print(f"chrome trace -> {TRACE}    open at https://ui.perfetto.dev "
          f"(or chrome://tracing)")


if __name__ == "__main__":
    main()
