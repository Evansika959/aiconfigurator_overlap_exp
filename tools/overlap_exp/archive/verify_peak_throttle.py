#!/usr/bin/env python3
"""Does overlap throttling follow the GEMM's PEAK power or its MEAN?

THE CLAIM UNDER TEST. power_model.py argues that when a GEMM and a collective run
concurrently their instantaneous powers add, so what matters is the GEMM's power during
a FULL WAVE, not its kernel-wide mean. A mean-only model would allocate CTAs against
the mean and overcommit by peak-minus-mean -- up to 117 W on few-wave shapes.

Nothing measured so far can adjudicate that: every validation compared against NVML,
which reports a loop average and is blind to sub-millisecond structure. This runs the
configurations where the two models give OPPOSITE verdicts and records the one signal
that separates them -- whether the clock actually held.

  peak model says throttle, mean model says fine   (22 configs at 1200 MHz / 108 SM)
    the GEMM's full-wave power plus the collective is 401-432 W against a 400 W cap,
    while its mean plus the collective is only 284-372 W.

  CONTROLS, without which the result means nothing:
    gemm_only    the same GEMM with no collective. If this throttles, the collective
                 is irrelevant and the row proves nothing.
    comm_only    the same collective with no GEMM.
    serial       both, but ordered. Same total work, never concurrent -- so if the
                 concurrent case throttles and this one does not, the overlap caused it.

READ IT AS: concurrent throttles AND serial does not AND gemm_only does not
            -> the peak model is right, the mean model would have overcommitted
            everything throttles or nothing does
            -> the test is inconclusive, not a confirmation

The collective goes on a priority -3 stream and is issued first, matching
demo_overlap.py -- otherwise the GEMM starves it and the two never really overlap.

  python3 verify_peak_throttle.py             # ~25 min, 2 GPUs, locks clocks only
"""

import argparse
import collections
import csv
import math
import os
import statistics as st
import subprocess
import sys
import threading
import time

os.environ.setdefault("LD_LIBRARY_PATH", "")
os.environ["LD_LIBRARY_PATH"] = ":".join(
    p for p in os.environ["LD_LIBRARY_PATH"].split(":") if p and "gib" not in p)
os.environ.pop("NCCL_NET", None)

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

CLOCK = 1200
SM = 108                       # the disagreement region is all at full SM count
TILE_M, TILE_N = 128, 256
AR_MIB = 256                   # long enough to span many GEMM waves
CLOCK_TOL_MHZ = 20
WINDOW_S = 2.0
DISCARD_S = 0.6
MODES = ["gemm_only", "comm_only", "serial", "concurrent"]

# (M, N, K, CTA) drawn from the disagreement list; K varies the footprint, CTA the
# collective's share. Ordered so the strongest disagreement comes first.
CASES = [
    (1024, 4096, 16384, 32), (1024, 4096, 16384, 16), (1024, 4096, 16384, 8),
    (1024, 4096, 16384, 4),
    (1024, 4096, 4096, 32), (1024, 4096, 4096, 16), (1024, 4096, 4096, 8),
    (1024, 8192, 8192, 32), (1024, 8192, 8192, 8),
    (2048, 4096, 16384, 32), (2048, 4096, 16384, 8),
    (8192, 4096, 4096, 16), (8192, 4096, 4096, 8),
]


def sh(c):
    return subprocess.run(c, capture_output=True, text=True)


class Boards:
    """NVML power and the ACHIEVED clock on every participating GPU. The clock is the
    point of this script; power is recorded only for context."""

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
                    [self.nv.nvmlDeviceGetClockInfo(x, self.nv.NVML_CLOCK_SM) for x in self.h]))
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
            return None
        t0 = self._rows[0][0]
        kept = [r for r in self._rows if r[0] - t0 >= DISCARD_S] or self._rows
        ng = len(self.h)
        return dict(
            power_per_gpu=[round(st.median(r[1][i] for r in kept), 1) for i in range(ng)],
            power_node_w=round(sum(st.median(r[1][i] for r in kept) for i in range(ng)), 1),
            clock_min=min(min(r[2]) for r in kept),
            clock_med=int(st.median([st.median(r[2]) for r in kept])),
            n_samples=len(kept))


def worker(rank, world, port, ctas, cases, q):
    """One CTA value per PROCESS. NCCL parses NCCL_{MIN,MAX}_CTAS once, when it first
    reads its environment, and caches it -- re-setting os.environ and rebuilding the
    process group inside one process does nothing. Measured the hard way: 32, 16 and 8
    CTAs all produced 1.55-1.57 ms for the same collective, i.e. every run silently used
    whichever value was set first. Only a fresh process resets it."""
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world),
                      MASTER_ADDR="127.0.0.1", MASTER_PORT=port,
                      NCCL_MIN_CTAS=str(ctas), NCCL_MAX_CTAS=str(ctas))
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    results = []
    for (m, n, k, _c) in cases:

        lo, hi = torch.cuda.Stream.priority_range()
        s_gemm = torch.cuda.Stream(priority=0)
        s_comm = torch.cuda.Stream(priority=hi)
        cur = torch.cuda.current_stream()

        x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
        w = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
        buf = torch.ones(AR_MIB * 1024 * 1024 // 2, dtype=torch.bfloat16, device="cuda")
        gemm = lambda: torch.nn.functional.linear(x, w)
        allreduce = lambda: dist.all_reduce(buf)

        for _ in range(8):
            gemm(); allreduce()
        torch.cuda.synchronize()

        for mode in MODES:
            def one():
                if mode == "gemm_only":
                    gemm()
                elif mode == "comm_only":
                    allreduce()
                elif mode == "serial":
                    gemm(); allreduce()
                else:
                    ev = torch.cuda.Event(); ev.record(cur)
                    for (s_, f_) in ((s_comm, allreduce), (s_gemm, gemm)):
                        with torch.cuda.stream(s_):
                            s_.wait_event(ev); f_()

            t0 = time.perf_counter()
            for _ in range(5):
                one()
            torch.cuda.synchronize()
            it = (time.perf_counter() - t0) / 5

            # identical iteration count on every rank: deriving it locally puts the
            # ranks one apart and one issues a collective nobody joins
            nt = torch.tensor([max(4, int((WINDOW_S + DISCARD_S) / max(it, 1e-6)))],
                              dtype=torch.int64, device="cuda")
            dist.broadcast(nt, src=0)
            iters = int(nt.item())

            b = Boards(world) if rank == 0 else None
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
                results.append(dict(m=m, n=n, k=k, ctas=ctas, mode=mode,
                                    iter_ms=round(wall, 4), iters=iters,
                                    clock_min=s["clock_min"], clock_med=s["clock_med"],
                                    clock_held=s["clock_min"] >= CLOCK - CLOCK_TOL_MHZ,
                                    power_node_w=s["power_node_w"],
                                    power_per_gpu=str(s["power_per_gpu"]),
                                    n_samples=s["n_samples"]))
                r = results[-1]
                print(f"  M={m:>4} N={n:>5} K={k:>5} {ctas:>2}CTA {mode:>11} | "
                      f"iter {wall:7.3f} ms  clk {s['clock_min']:>4}/{s['clock_med']:>4} "
                      f"held={str(r['clock_held']):>5}  P {s['power_node_w']:6.1f} W",
                      flush=True)
        del x, w, buf
        torch.cuda.empty_cache()
    dist.destroy_process_group()
    if rank == 0:
        q.put(results)


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--out", default=os.path.join(here, "data", "peak_vs_mean.csv"))
    a = ap.parse_args()
    world = 2
    assert torch.cuda.device_count() >= world

    locked = False
    try:
        pm = sh(["nvidia-smi", "-i", "0,1", "--query-gpu=persistence_mode",
                 "--format=csv,noheader"]).stdout.split("\n")[0].strip()
        if pm != "Enabled":
            sh(["sudo", "nvidia-smi", "-i", "0,1", "-pm", "1"])
        # without persistence the lock silently does not hold and the run executes at
        # the boost clock, which invalidates every verdict here
        sh(["sudo", "nvidia-smi", "-i", "0,1", "-lgc", f"{CLOCK},{CLOCK}"])
        locked = True
        time.sleep(1)
        got = sh(["nvidia-smi", "-i", "0", "--query-gpu=clocks.sm",
                  "--format=csv,noheader,nounits"]).stdout.strip()
        print(f"[clocks] locked {CLOCK} MHz, idle read-back {got} MHz\n", flush=True)

        import socket
        mp.set_start_method("spawn", force=True)
        rows = []
        for ctas in sorted({c[3] for c in CASES}):
            sub = [c for c in CASES if c[3] == ctas]
            with socket.socket() as sk:
                sk.bind(("127.0.0.1", 0)); port = str(sk.getsockname()[1])
            print(f"--- {ctas} CTA, {len(sub)} shapes, fresh processes ---", flush=True)
            q = mp.Queue()
            ps = [mp.Process(target=worker, args=(r, world, port, ctas, sub, q))
                  for r in range(world)]
            for p in ps:
                p.start()
            rows += q.get(timeout=1800)
            for p in ps:
                p.join(timeout=120)
                if p.is_alive():
                    p.terminate()
    finally:
        if locked:
            sh(["sudo", "nvidia-smi", "-i", "0,1", "-rgc"])
            if pm != "Enabled":
                sh(["sudo", "nvidia-smi", "-i", "0,1", "-pm", "0"])
            print("[clocks] restored", flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)
    print(f"\nwrote {a.out}  ({len(rows)} rows)\n")

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from power_model import compose

    by = collections.defaultdict(dict)
    for r in rows:
        by[(r["m"], r["n"], r["k"], r["ctas"])][r["mode"]] = r
    print("VERDICT per configuration\n")
    print(f"{'M':>5}{'N':>6}{'K':>6}{'CTA':>4} | {'peak pred':>10}{'mean pred':>10} | "
          f"{'gemm':>6}{'comm':>6}{'serial':>7}{'concur':>7} | {'outcome':>28}")
    tally = collections.Counter()
    for key, d in by.items():
        m, n, k, c = key
        cm = compose(m, n, k, SM, CLOCK, c, 1.0, 1.0)
        pk, mn = cm["p_worst"], cm["p_gemm_mean"] + cm["p_comm"]
        held = {mo: d[mo]["clock_held"] for mo in MODES if mo in d}
        if not held.get("gemm_only", True) or not held.get("comm_only", True):
            out = "inconclusive: a solo run throttled"
        elif not held.get("concurrent", True) and held.get("serial", True):
            out = "PEAK model right"
        elif held.get("concurrent", True):
            out = "MEAN model right (no throttle)"
        else:
            out = "inconclusive: serial throttled too"
        tally[out] += 1
        print(f"{m:>5}{n:>6}{k:>6}{c:>4} | {pk:>9.0f}W{mn:>9.0f}W | "
              + "".join(f"{('ok' if held.get(mo) else 'THR'):>7}" for mo in MODES)
              + f" | {out:>28}")
    print()
    for k_, v in tally.most_common():
        print(f"  {v:>2} x {k_}")


if __name__ == "__main__":
    main()
