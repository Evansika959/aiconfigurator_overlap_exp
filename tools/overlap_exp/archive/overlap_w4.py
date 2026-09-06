#!/usr/bin/env python3
"""Overlap GEMM with an all-reduce at world=4, and test the composed power model.

WHY WORLD=4. The collective coefficients (alpha, B, p0, gamma) were fitted on
../comm_dvfs_char, which ran mpirun -n 4. Applying them to the earlier world=2 overlap
runs required rescaling by the ring volume 2(N-1)/N, and that rescaling was measured to
be wrong by 18% -- the implied ratio was 1.22 against the ring model's 1.00. Running the
overlap at world=4 removes the extrapolation entirely: every coefficient is then used at
the world size it was measured at.

MODEL UNDER TEST, per GPU:

    P(t) = P_static(f)                                   board floor, counted ONCE
         + a(f)*s_gemm(t) + b(N*K)                        GEMM: a staircase over waves
         + [p0(f) + gamma(f)*B(c,f)]                      collective: a rectangle

NVML reports an average over a window far longer than one iteration, so the comparison
is the model's time-average over the MEASURED iteration -- sync gap included, where only
P_static is drawn. Comparing the overlap-interval mean against a loop average instead
over-predicted by 142 W in an earlier attempt.

FOUR MODES, because a bare concurrent number proves nothing:
    gemm_only / comm_only   the two halves on their own
    serial                  both on one stream, hardware-ordered: the no-overlap baseline
    concurrent              collective first on a priority -3 stream, then the GEMM

ONE CTA VALUE PER PROCESS LAUNCH. NCCL parses NCCL_{MIN,MAX}_CTAS once and caches it;
re-setting os.environ and rebuilding the process group inside one process silently keeps
the first value. Measured the hard way -- 32, 16 and 8 CTAs all returned 1.55-1.57 ms for
the same collective.

  python3 overlap_w4.py            # 4 GPUs, ~15 min, locks clocks only
"""

import argparse
import collections
import csv
import math
import os
import socket
import statistics as st
import subprocess
import sys
import threading
import time

os.environ.setdefault("LD_LIBRARY_PATH", "")
# GCP's gIB shim breaks NCCL on A100 and rejects NCCL_{MIN,MAX}_CTAS; unsetting the env
# vars is not enough, the directory has to come off the loader path
os.environ["LD_LIBRARY_PATH"] = ":".join(
    p for p in os.environ["LD_LIBRARY_PATH"].split(":") if p and "gib" not in p)
os.environ.pop("NCCL_NET", None)

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD = 4
CLOCK = 1200
GPUS = "0,1,2,3"
TOTAL_SM = 108
TILE_AREA = 32768
CLOCK_TOL = 20
WINDOW_S = 2.5
DISCARD_S = 0.7
MODES = ["gemm_only", "comm_only", "serial", "concurrent"]

SHAPES = {"low_occ": (1024, 4096, 16384),      # grid 128 -> 2 waves, tail 20
          "high_occ": (8192, 4096, 4096)}      # grid 1024 -> 10 waves, tail 52
AR_MIB = [64, 256]
CTAS = [2, 4, 8, 16]

# ---- coefficients, both campaigns, neither of which saw an overlap run --------------
A_GEMM = {1200: 2.6658}                                    # W per busy SM
P_STATIC = {1200: 69.66}                                   # W per GPU
B_FOOT = {1200: [(32, 0.2), (128, 12.1), (512, 16.3)]}     # W vs weight footprint MB
B_W4 = {1200: {1: 9.6, 2: 19.4, 4: 37.1, 8: 68.7, 16: 119.7, 32: 139.1}}   # GB/s
ALPHA_W4 = {1200: {1: 48.1, 2: 46.4, 4: 44.1, 8: 58.3, 16: 77.0, 32: 71.8}}  # us
P0 = {1200: 24.6}
GAMMA = {1200: 0.258}


def sh(c):
    return subprocess.run(c, capture_output=True, text=True)


def b_gemm(n, k, f=CLOCK):
    mb = n * k * 2 / 2 ** 20
    t = B_FOOT[f]
    if mb <= t[0][0]:
        return t[0][1]
    if mb >= t[-1][0]:
        return t[-1][1]
    return next(y0 + (y1 - y0) * (mb - x0) / (x1 - x0)
                for (x0, y0), (x1, y1) in zip(t, t[1:]) if x0 <= mb <= x1)


def gemm_waves(m, n, k, sm=TOTAL_SM, f=CLOCK):
    grid = m * n // TILE_AREA
    w = -(-grid // sm)
    tail = grid - (w - 1) * sm
    a, b = A_GEMM[f], b_gemm(n, k, f)
    return dict(grid=grid, waves=w, tail=tail,
                p_full=a * min(sm, grid) + b, p_tail=a * tail + b,
                p_mean=a * (grid / w) + b)


def comm_pred(mib, c, f=CLOCK):
    """world=4, so the fitted coefficients apply with NO volume rescaling."""
    gb = mib * 2 ** 20 / 1e9
    return dict(t_ms=ALPHA_W4[f][c] / 1e3 + gb / B_W4[f][c] * 1e3,
                p=P0[f] + GAMMA[f] * B_W4[f][c])


class Boards:
    def __init__(self, n):
        import pynvml
        self.nv = pynvml
        pynvml.nvmlInit()
        self.h = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(n)]
        self._stop, self._rows = threading.Event(), []

    def _loop(self):
        while not self._stop.is_set():
            try:
                self._rows.append((
                    time.time(),
                    [self.nv.nvmlDeviceGetPowerUsage(x) / 1000 for x in self.h],
                    [self.nv.nvmlDeviceGetClockInfo(x, self.nv.NVML_CLOCK_SM)
                     for x in self.h]))
            except Exception:
                pass
            self._stop.wait(0.02)

    def __enter__(self):
        self._rows.clear(); self._stop.clear()
        self._t = threading.Thread(target=self._loop, daemon=True); self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set(); self._t.join(timeout=2)

    def summary(self):
        if not self._rows:
            return None
        t0 = self._rows[0][0]
        kept = [r for r in self._rows if r[0] - t0 >= DISCARD_S] or self._rows
        ng = len(self.h)
        per = [st.median(r[1][i] for r in kept) for i in range(ng)]
        return dict(power_node_w=round(sum(per), 1),
                    power_per_gpu_w=round(sum(per) / ng, 2),
                    per_gpu=[round(p, 1) for p in per],
                    clock_min=min(min(r[2]) for r in kept),
                    n_samples=len(kept))


def worker(rank, port, ctas, q):
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(WORLD),
                      MASTER_ADDR="127.0.0.1", MASTER_PORT=port,
                      NCCL_MIN_CTAS=str(ctas), NCCL_MAX_CTAS=str(ctas))
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=WORLD)
    lo, hi = torch.cuda.Stream.priority_range()
    s_gemm = torch.cuda.Stream(priority=0)
    s_comm = torch.cuda.Stream(priority=hi)
    cur = torch.cuda.current_stream()
    out = []

    for tag, (m, n, k) in SHAPES.items():
        x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
        w = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
        gemm = lambda: torch.nn.functional.linear(x, w)
        for mib in AR_MIB:
            buf = torch.ones(mib * 1024 * 1024 // 2, dtype=torch.bfloat16, device="cuda")
            ar = lambda: dist.all_reduce(buf)
            for _ in range(8):
                gemm(); ar()
            torch.cuda.synchronize()

            # per-kernel spans on a common timeline
            base = torch.cuda.Event(enable_timing=True)
            eg = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            ec = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            gs, cs = [], []
            for _ in range(12):
                torch.cuda.synchronize()
                base.record(cur)
                for (s_, ev, fn) in ((s_comm, ec, ar), (s_gemm, eg, gemm)):
                    with torch.cuda.stream(s_):
                        s_.wait_event(base)
                        ev[0].record(s_); fn(); ev[1].record(s_)
                torch.cuda.synchronize()
                gs.append(base.elapsed_time(eg[1]) - base.elapsed_time(eg[0]))
                cs.append(base.elapsed_time(ec[1]) - base.elapsed_time(ec[0]))
            t_g_conc, t_c_conc = st.median(gs), st.median(cs)

            for mode in MODES:
                def one():
                    if mode == "gemm_only":
                        gemm()
                    elif mode == "comm_only":
                        ar()
                    elif mode == "serial":
                        gemm(); ar()
                    else:
                        ev = torch.cuda.Event(); ev.record(cur)
                        for (s_, fn) in ((s_comm, ar), (s_gemm, gemm)):
                            with torch.cuda.stream(s_):
                                s_.wait_event(ev); fn()

                t0 = time.perf_counter()
                for _ in range(5):
                    one()
                torch.cuda.synchronize()
                it = (time.perf_counter() - t0) / 5
                # the iteration count must be IDENTICAL on every rank, or one rank
                # issues a collective nobody joins and the job hangs
                nt = torch.tensor([max(5, int((WINDOW_S + DISCARD_S) / max(it, 1e-6)))],
                                  dtype=torch.int64, device="cuda")
                dist.broadcast(nt, src=0)
                iters = int(nt.item())

                b = Boards(WORLD) if rank == 0 else None
                if b:
                    b.__enter__()
                t0 = time.perf_counter()
                for _ in range(iters):
                    one()
                    torch.cuda.synchronize()
                wall = (time.perf_counter() - t0) / iters * 1e3
                if b:
                    b.__exit__()
                    s = b.summary()
                    out.append(dict(shape=tag, m=m, n=n, k=k, ar_mib=mib, ctas=ctas,
                                    mode=mode, world=WORLD, iter_ms=round(wall, 4),
                                    iters=iters, t_gemm_conc_ms=round(t_g_conc, 4),
                                    t_comm_conc_ms=round(t_c_conc, 4),
                                    clock_min=s["clock_min"],
                                    clock_held=s["clock_min"] >= CLOCK - CLOCK_TOL,
                                    power_node_w=s["power_node_w"],
                                    power_per_gpu_w=s["power_per_gpu_w"],
                                    n_samples=s["n_samples"]))
                    r = out[-1]
                    print(f"  {tag:>9} {mib:>3}MiB {ctas:>2}CTA {mode:>11} | "
                          f"iter {wall:8.3f} ms  P/GPU {s['power_per_gpu_w']:6.1f} W  "
                          f"clk {s['clock_min']:>4}", flush=True)
            del buf
        del x, w
        torch.cuda.empty_cache()
    dist.destroy_process_group()
    if rank == 0:
        q.put(out)


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--out", default=os.path.join(here, "data", "overlap_w4.csv"))
    a = ap.parse_args()
    assert torch.cuda.device_count() >= WORLD, f"need {WORLD} GPUs"

    pm = sh(["nvidia-smi", "-i", GPUS, "--query-gpu=persistence_mode",
             "--format=csv,noheader"]).stdout.split("\n")[0].strip()
    if pm != "Enabled":
        sh(["sudo", "nvidia-smi", "-i", GPUS, "-pm", "1"])
    sh(["sudo", "nvidia-smi", "-i", GPUS, "-lgc", f"{CLOCK},{CLOCK}"])
    time.sleep(1)
    got = sh(["nvidia-smi", "-i", "0", "--query-gpu=clocks.sm",
              "--format=csv,noheader,nounits"]).stdout.strip()
    print(f"[clocks] {CLOCK} MHz on GPUs {GPUS}, idle read-back {got} MHz "
          f"(persistence was {pm})\n", flush=True)
    rows = []
    try:
        mp.set_start_method("spawn", force=True)
        for c in CTAS:
            with socket.socket() as sk:
                sk.bind(("127.0.0.1", 0)); port = str(sk.getsockname()[1])
            print(f"--- {c} CTA, fresh processes ---", flush=True)
            q = mp.Queue()
            ps = [mp.Process(target=worker, args=(r, port, c, q)) for r in range(WORLD)]
            for p in ps:
                p.start()
            rows += q.get(timeout=2400)
            for p in ps:
                p.join(timeout=120)
                if p.is_alive():
                    p.terminate()
    finally:
        sh(["sudo", "nvidia-smi", "-i", GPUS, "-rgc"])
        if pm != "Enabled":
            sh(["sudo", "nvidia-smi", "-i", GPUS, "-pm", "0"])
        print("[clocks] restored", flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)
    print(f"\nwrote {a.out}  ({len(rows)} rows)\n")
    report(rows)


def report(rows):
    by = collections.defaultdict(dict)
    for r in rows:
        by[(r["shape"], r["ar_mib"], r["ctas"])][r["mode"]] = r

    print("A. Collective latency, world=4 fit at world=4 -- no rescaling")
    print(f"   {'MiB':>5}{'CTA':>5}{'measured':>10}{'predicted':>11}{'err':>8}")
    e = []
    for (shape, mib, c), d in sorted(by.items()):
        if shape != "high_occ" or "comm_only" not in d:
            continue
        meas = d["comm_only"]["iter_ms"]
        pred = comm_pred(mib, c)["t_ms"]
        e.append(abs(pred - meas) / meas)
        print(f"   {mib:>5}{c:>5}{meas:>10.3f}{pred:>11.3f}{(pred - meas) / meas * 100:>7.1f}%")
    print(f"   median |err| {st.median(e) * 100:.1f}%\n")

    print("B. Composed power, model vs measured, per GPU")
    print(f"   {'shape':>9}{'MiB':>5}{'CTA':>4}{'mode':>11}{'iter ms':>9}"
          f"{'meas':>8}{'model':>8}{'err':>8}")
    E = collections.defaultdict(list)
    for (shape, mib, c), d in sorted(by.items()):
        m, n, k = SHAPES[shape]
        g = gemm_waves(m, n, k)
        cm = comm_pred(mib, c)
        tg = d["gemm_only"]["iter_ms"] if "gemm_only" in d else 0
        tc = d["comm_only"]["iter_ms"] if "comm_only" in d else 0
        for mode, r in sorted(d.items()):
            if mode == "gemm_only":
                dyn = g["p_mean"] * tg
            elif mode == "comm_only":
                dyn = cm["p"] * tc
            else:
                dyn = g["p_mean"] * tg + cm["p"] * tc
            model = P_STATIC[CLOCK] + dyn / r["iter_ms"]
            meas = r["power_per_gpu_w"]
            err = (model - meas) / meas
            E[mode].append(abs(err))
            print(f"   {shape:>9}{mib:>5}{c:>4}{mode:>11}{r['iter_ms']:>9.3f}"
                  f"{meas:>8.1f}{model:>8.1f}{err * 100:>7.1f}%")
    print()
    for mode in MODES:
        if E[mode]:
            print(f"   {mode:>11}: median |err| {st.median(E[mode]) * 100:5.1f}%  "
                  f"max {max(E[mode]) * 100:5.1f}%  n={len(E[mode])}")
    allv = [x for v in E.values() for x in v]
    print(f"   {'overall':>11}: median |err| {st.median(allv) * 100:5.1f}%  n={len(allv)}")

    print("\nC. Overlap payoff (concurrent vs serial)")
    print(f"   {'shape':>9}{'MiB':>5}{'CTA':>4}{'speedup':>9}{'E/iter serial':>15}"
          f"{'concurrent':>12}{'change':>9}")
    for (shape, mib, c), d in sorted(by.items()):
        if "serial" not in d or "concurrent" not in d:
            continue
        s_, cc = d["serial"], d["concurrent"]
        es = s_["power_per_gpu_w"] * s_["iter_ms"]
        ec = cc["power_per_gpu_w"] * cc["iter_ms"]
        print(f"   {shape:>9}{mib:>5}{c:>4}{s_['iter_ms'] / cc['iter_ms']:>9.3f}"
              f"{es:>15.1f}{ec:>12.1f}{(ec / es - 1) * 100:>8.1f}%")


if __name__ == "__main__":
    main()
