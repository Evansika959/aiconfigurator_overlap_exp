#!/usr/bin/env python3
"""Phase 1: does overlapping a GEMM with an all-reduce pay, in time and in energy?

WHAT IS MEASURED, PER CONFIGURATION

  solo      GEMM alone, and the all-reduce alone. Reference points, timed with CUDA
            events. No power window (short).
  serial    GEMM then all-reduce on the SAME stream, so the hardware orders them.
            This is the no-overlap baseline. It is MEASURED, not computed as
            t_gemm + t_ar: serial execution has its own launch overheads.
  concurrent  the two on separate streams. Timed AND power-sampled.

  Every timing is a CUDA-event span on a common timeline, so the actual intersection
  of the two kernels' intervals is known -- total wall time alone cannot distinguish
  "they overlapped" from "the second one was cheap".

THE THREE LAUNCH ORDERS, AND WHY EACH IS A DIFFERENT EXPERIMENT

  A GEMM submits ~1000 blocks that fill all 108 SMs at once, and a block runs to
  completion -- there is no preemption. An NCCL collective is ONE persistent kernel
  whose n_cta blocks stay resident for the whole reduction. So:

  serial          the baseline.
  gemm_first      the naive implementation. GEMM floods the SMs; NCCL's CTAs wait for
                  blocks to retire. Measured at equal priority: the all-reduce ran
                  10.6x slower and the whole thing was SLOWER than serial (0.96x).
  comm_first      the collective's CTAs land first and hold their SMs for the whole
                  reduction, so the GEMM gets 108 minus that. This is the actual
                  "reserve N SMs for the collective" configuration.

  Both concurrent orders put the collective on a priority -3 stream (A100's range is
  0..-3). Without it gemm_first starves the collective; the naive priority-0 case is
  kept as `gemm_first_p0` to show what that costs.

ENERGY

  An all-reduce is a NODE-level event: both GPUs burn power, so both are sampled and
  summed. Dynamic energy subtracts the static floor measured by zeus's profile_p2p.py
  at this clock (69.66 W/GPU at 1200 MHz), because that part is drawn whether or not
  anything runs. Power needs a window much longer than one iteration, so each pattern
  is looped for >= WINDOW_S with a sync per iteration.

PREDICTION ON RECORD

  tiles = ceil(M/tile_m)*ceil(N/tile_n) and does NOT depend on K, so K is used to make
  a low-occupancy GEMM long enough to overlap with. LOW_OCC (M=1024,N=4096,K=16384)
  has 128 tiles -> 2 waves on 108 SMs -> S_eff 64, i.e. ~44 SMs idle on average.
  Handing those to NCCL should be nearly free. HIGH_OCC (M=8192,N=4096,K=4096) has
  1024 tiles -> S_eff 102.4 and should pay for every SM it gives up. If the two do not
  diverge, the premise behind this whole experiment is wrong.

  python3 phase1_sweep.py --ctas 8            # one CTA value per process: NCCL reads
                                              # NCCL_{MIN,MAX}_CTAS at communicator init
"""

import argparse
import csv
import json
import math
import os
import statistics
import sys
import threading
import time

os.environ["LD_LIBRARY_PATH"] = ":".join(
    p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p and "gib" not in p)
os.environ.pop("NCCL_NET", None)

import pynvml
import torch
import torch.distributed as dist

# (label, M, N, K). K does not enter the grid, so it lengthens the kernel without
# changing the tile count -- that is what makes a LOW-occupancy shape long enough to
# overlap with a multi-millisecond collective.
SHAPES = [
    ("low_occ", 1024, 4096, 16384),
    ("high_occ", 8192, 4096, 4096),
]
AR_MIB = [64, 256]
MODES = ["serial", "comm_first", "gemm_first", "gemm_first_p0"]
CLOCK_MHZ = 1200
P_STATIC_W = 69.66          # zeus profile_p2p.py @1200 MHz, per GPU (data/p0_static_zeus.csv)
WINDOW_S = 2.0
DISCARD_S = 0.6             # NVML needs ~427 ms to reach 95% of steady state
SAMPLE_S = 0.05
TOTAL_SM = 108
TILE_M, TILE_N = 128, 256   # torch 2.10/cu130 picks ampere_bf16_s16816gemm_bf16_128x256


class DualSampler:
    """NVML power on several GPUs at once. A collective is a node-level event, so
    sampling only rank 0's board under-reports it by roughly the world size."""

    def __init__(self, devs):
        pynvml.nvmlInit()
        self.h = [pynvml.nvmlDeviceGetHandleByIndex(d) for d in devs]
        self.devs = devs
        self._stop = threading.Event()
        self._rows = []

    def _loop(self):
        while not self._stop.is_set():
            try:
                self._rows.append((time.time(),
                                   [pynvml.nvmlDeviceGetPowerUsage(x) / 1000.0 for x in self.h],
                                   [pynvml.nvmlDeviceGetClockInfo(x, pynvml.NVML_CLOCK_SM)
                                    for x in self.h]))
            except Exception:
                pass
            self._stop.wait(SAMPLE_S)

    def start(self):
        self._rows.clear()
        self._stop.clear()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def stop(self):
        self._stop.set()
        self._t.join(timeout=2.0)
        rows = list(self._rows)
        if not rows:
            return None
        t0 = rows[0][0]
        kept = [r for r in rows if r[0] - t0 >= DISCARD_S] or rows
        per = [statistics.median(r[1][i] for r in kept) for i in range(len(self.h))]
        clk = [min(r[2][i] for r in kept) for i in range(len(self.h))]
        return dict(power_per_gpu=[round(p, 2) for p in per],
                    power_node_w=round(sum(per), 2),
                    clock_min=clk, n_samples=len(kept))


def tiles_of(m, n):
    return math.ceil(m / TILE_M) * math.ceil(n / TILE_N)


def s_eff(m, n, sm=TOTAL_SM):
    t = tiles_of(m, n)
    return t / math.ceil(t / sm)


def span_of(events, base):
    return (base.elapsed_time(events[0]), base.elapsed_time(events[1]))


def run_pattern(mode, gemm, ar, s_gemm, s_comm, s_comm_p0, reps):
    """Issue one iteration of `mode` and return (gemm_span, ar_span) on a common
    timeline, in ms, as medians over `reps`."""
    base = torch.cuda.Event(enable_timing=True)
    eg = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
    ec = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
    gs, cs = [], []
    cur = torch.cuda.current_stream()
    for _ in range(reps):
        torch.cuda.synchronize()
        base.record(cur)
        if mode == "serial":
            eg[0].record(cur); gemm(); eg[1].record(cur)
            ec[0].record(cur); ar(); ec[1].record(cur)
        else:
            sc = s_comm_p0 if mode == "gemm_first_p0" else s_comm
            # enqueue order is the knob: whichever is issued first reaches the work
            # distributor first, and NCCL's CTAs hold their SMs for the whole collective
            first, second = ((sc, ec, ar), (s_gemm, eg, gemm)) if mode == "comm_first" \
                else ((s_gemm, eg, gemm), (sc, ec, ar))
            for (st, ev, fn) in (first, second):
                with torch.cuda.stream(st):
                    st.wait_event(base)
                    ev[0].record(st)
                    fn()
                    ev[1].record(st)
        torch.cuda.synchronize()
        gs.append(span_of(eg, base))
        cs.append(span_of(ec, base))
    med = lambda sp: (statistics.median(a for a, _ in sp), statistics.median(b for _, b in sp))
    return med(gs), med(cs)


def power_of(mode, gemm, ar, s_gemm, s_comm, s_comm_p0, sampler, iter_ms):
    """Loop the pattern for >= WINDOW_S and sample both boards over it.

    EVERY RANK must run this loop: it contains a collective, and a rank that skips it
    leaves the others blocked in NCCL forever. Only rank 0 carries a sampler -- that is
    the part that is rank-0-only, not the work. (First version gated the whole call on
    rank 0 and deadlocked: rank 1 sat in barrier() while rank 0 waited for it inside
    all_reduce.)"""
    # THE ITERATION COUNT MUST BE IDENTICAL ON EVERY RANK. It is derived from a
    # measured span, and two ranks measure spans differing by microseconds -- enough
    # for int() to land one apart. The rank with the larger count then issues one more
    # collective than anyone joins and the job hangs, one GPU pinned at 100% and the
    # other idle. (This is what stalled the first sweep at 30/80 configurations.)
    n = torch.tensor([max(3, int((WINDOW_S + DISCARD_S) / max(iter_ms, 1e-6) * 1e3))],
                     dtype=torch.int64, device="cuda")
    dist.broadcast(n, src=0)
    iters = int(n.item())
    cur = torch.cuda.current_stream()
    if sampler:
        sampler.start()
    t0 = time.perf_counter()
    for _ in range(iters):
        if mode == "serial":
            gemm(); ar()
        else:
            sc = s_comm_p0 if mode == "gemm_first_p0" else s_comm
            ev = torch.cuda.Event()
            ev.record(cur)
            order = ((sc, ar), (s_gemm, gemm)) if mode == "comm_first" \
                else ((s_gemm, gemm), (sc, ar))
            for (st, fn) in order:
                with torch.cuda.stream(st):
                    st.wait_event(ev)
                    fn()
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    return (sampler.stop() if sampler else None), wall / iters * 1e3, iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctas", type=int, required=True)
    ap.add_argument("--reps", type=int, default=25)
    ap.add_argument("--out", default="data/phase1.csv")
    a = ap.parse_args()

    rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", os.environ.get("RANK", "0")))
    world = int(os.environ.get("OMPI_COMM_WORLD_SIZE", os.environ.get("WORLD_SIZE", "2")))
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world))
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29621")
    os.environ["NCCL_MIN_CTAS"] = str(a.ctas)     # read at communicator init
    os.environ["NCCL_MAX_CTAS"] = str(a.ctas)
    os.environ["TLLM_DISABLE_ALLREDUCE_AUTOTUNE"] = "1"
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)

    from tensorrt_llm._torch.distributed import (AllReduce, AllReduceFusionOp,
                                                 AllReduceParams)
    from tensorrt_llm.functional import AllReduceStrategy
    from tensorrt_llm.mapping import Mapping

    mapping = Mapping(world_size=world, tp_size=world, rank=rank)
    ar_op = AllReduce(mapping=mapping, strategy=AllReduceStrategy.NCCL).cuda()
    ar_params = AllReduceParams(strategy=AllReduceStrategy.NCCL,
                                fusion_op=AllReduceFusionOp.NONE)

    lo, hi = torch.cuda.Stream.priority_range()
    s_gemm = torch.cuda.Stream(priority=0)
    s_comm = torch.cuda.Stream(priority=hi)       # hi is the most-negative = highest
    s_comm_p0 = torch.cuda.Stream(priority=0)
    sampler = DualSampler(list(range(world))) if rank == 0 else None

    rows = []
    for (tag, m, n, k) in SHAPES:
        x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
        w = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
        gemm = lambda: torch.nn.functional.linear(x, w)
        for mib in AR_MIB:
            buf = torch.ones(mib * 1024 * 1024 // 2, dtype=torch.bfloat16, device="cuda")
            ar = lambda: ar_op(buf, all_reduce_params=ar_params)
            for _ in range(5):
                gemm(); ar()
            torch.cuda.synchronize()

            (g0, g1), _ = run_pattern("serial", gemm, lambda: None, s_gemm, s_comm,
                                      s_comm_p0, 8)
            t_gemm_solo = g1 - g0
            _, (c0, c1) = run_pattern("serial", lambda: None, ar, s_gemm, s_comm,
                                      s_comm_p0, 8)
            t_ar_solo = c1 - c0

            for mode in MODES:
                (gs, ge), (cs, ce) = run_pattern(mode, gemm, ar, s_gemm, s_comm,
                                                 s_comm_p0, a.reps)
                tg, tc = ge - gs, ce - cs
                span = max(ge, ce) - min(gs, cs)
                inter = max(0.0, min(ge, ce) - max(gs, cs))
                st, iter_ms, iters = power_of(mode, gemm, ar, s_gemm, s_comm,
                                              s_comm_p0, sampler, span)

                row = dict(ctas=a.ctas, shape=tag, m=m, n=n, k=k, ar_mib=mib, mode=mode,
                           tiles=tiles_of(m, n), s_eff_solo=round(s_eff(m, n), 1),
                           t_gemm_solo=round(t_gemm_solo, 4), t_ar_solo=round(t_ar_solo, 4),
                           t_gemm=round(tg, 4), t_ar=round(tc, 4),
                           span_ms=round(span, 4), overlap_ms=round(inter, 4),
                           iter_ms=round(iter_ms, 4), iters=iters)
                if st:
                    p_node = st["power_node_w"]
                    p_dyn = p_node - P_STATIC_W * world
                    row.update(power_node_w=p_node,
                               power_per_gpu=json.dumps(st["power_per_gpu"]),
                               p_dyn_node_w=round(p_dyn, 2),
                               e_total_mj=round(p_node * iter_ms, 3),
                               e_dyn_mj=round(p_dyn * iter_ms, 3),
                               clock_min=json.dumps(st["clock_min"]),
                               n_samples=st["n_samples"])
                rows.append(row)
                if rank == 0:
                    print(f"  CTA={a.ctas:>2} {tag:>9} {mib:>3}MiB {mode:>14} | "
                          f"g {tg:7.3f} c {tc:7.3f} span {span:8.3f} ovl {inter:7.3f} | "
                          f"iter {iter_ms:7.3f}ms  P {row.get('power_node_w', 0):6.1f}W  "
                          f"Edyn {row.get('e_dyn_mj', 0):8.1f}mJ", flush=True)
            del buf
        del x, w
        torch.cuda.empty_cache()

    if rank == 0:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        new = not os.path.exists(a.out)
        with open(a.out, "a", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            if new:
                wr.writeheader()
            wr.writerows(rows)
        print(f"appended {len(rows)} rows -> {a.out}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
