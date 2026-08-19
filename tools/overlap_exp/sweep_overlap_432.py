#!/usr/bin/env python3
"""Overlap sweep: 6 clocks x 12 GEMM shapes x 6 CTA counts = 432 concurrent cases, world=4.

The 16 cases measured before covered one clock and two shapes that both turned out to
sit at 95-99% occupancy, so neither the clock axis nor the wave term was ever exercised.
This covers both.

WHAT EACH AXIS TESTS
    clock   a(f), P_static(f), p0(f), gamma(f) -- four coefficients at once.  Previously
            1 of 6.
    shape   S_eff (the wave term) and b (footprint).  The 12 shapes span occupancy
            59% to 100%, so the staircase finally gets a real test.  Previously 2 of 12,
            both at 95-99%.
    CTA     B(c,f) and hence collective power.  Previously 4 of 6.

Message size is fixed at 64 MiB. It does NOT enter the collective's power model --
P_comm = p0 + gamma*B(c,f) depends only on CTA and clock -- so sweeping it would
multiply the run count without covering a new coefficient. It only sets the duty cycle.

GRIDS ARE MEASURED, NOT COMPUTED.  grid12.json holds ncu-measured launch__grid_size for
all 12 shapes. 11 are split-K=1, but (1024,8192,8192) splits 2-way -- assuming otherwise
cost 21% on an earlier shape. Do not replace this with a formula.

MEASUREMENT ECONOMY.  gemm_only does not depend on CTA and comm_only does not depend on
the GEMM shape, so each is measured once rather than once per combination:
    comm_only   6 CTA x 6 clocks                     =  36
    gemm_only   12 shapes x 6 clocks, first CTA only =  72
    serial      432
    concurrent  432
                                                       972 windows, ~50 min of sampling

TRAPS THIS SCRIPT IS BUILT AROUND
  * NCCL caches NCCL_{MIN,MAX}_CTAS on first read -- one process launch per CTA value.
  * The iteration count must be identical on every rank, or one rank issues a collective
    nobody joins and the job hangs with one GPU at 100% and the rest idle.
  * A clock lock does not hold without persistence mode; -lgc still reports success.
  * GCP's gIB shim rejects NCCL_{MIN,MAX}_CTAS; the directory must come off
    LD_LIBRARY_PATH, unsetting NCCL_NET is not enough.

  python3 sweep_overlap_432.py                # ~1.5 h
  python3 sweep_overlap_432.py --resume       # after an interruption
"""

import argparse
import csv
import json
import os
import socket
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

WORLD = 4
GPUS = "0,1,2,3"
CLOCKS = [1410, 1200, 900, 705, 510, 300]
CTAS = [1, 2, 4, 8, 16, 32]
SHAPES = [(m, nk, nk) for nk in (4096, 8192, 16384) for m in (1024, 2048, 4096, 8192)]
# spawn re-imports this module in every child, so a smoke run has to shrink the axes
# through the environment -- assigning to the globals in the parent would not carry
if os.environ.get("SMOKE"):
    CLOCKS, CTAS, SHAPES = [1200, 900], [2, 8], SHAPES[:1] + SHAPES[4:5]
AR_MIB = 64
TOTAL_SM = 108
CLOCK_TOL = 20
WINDOW_S = 2.0
DISCARD_S = 0.6
HERE = os.path.dirname(os.path.abspath(__file__))
GRIDS = json.load(open("/tmp/claude-1013/-home-xinting/b2d7b7e5-61f6-442d-8ca6-fa6405055f67/scratchpad/grid12.json"))

FIELDS = ["ctas", "clock", "m", "n", "k", "mode", "grid", "waves", "s_eff", "ar_mib",
          "iter_ms", "iters", "clock_min", "clock_held", "power_node_w",
          "power_per_gpu_w", "n_samples", "world"]


def sh(c):
    return subprocess.run(c, capture_output=True, text=True)


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
                    clock_min=min(min(r[2]) for r in kept), n_samples=len(kept))


def worker(rank, port, ctas, do_gemm_only, done, out_path, q):
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(WORLD),
                      MASTER_ADDR="127.0.0.1", MASTER_PORT=port,
                      NCCL_MIN_CTAS=str(ctas), NCCL_MAX_CTAS=str(ctas))
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=WORLD)
    lo, hi = torch.cuda.Stream.priority_range()
    s_gemm = torch.cuda.Stream(priority=0)
    s_comm = torch.cuda.Stream(priority=hi)
    cur = torch.cuda.current_stream()
    buf = torch.ones(AR_MIB * 1024 * 1024 // 2, dtype=torch.bfloat16, device="cuda")
    ar = lambda: dist.all_reduce(buf)
    fh = wr = None
    if rank == 0:
        new = not os.path.exists(out_path)
        fh = open(out_path, "a", newline="")
        wr = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            wr.writeheader(); fh.flush()

    def run(mode, fn, meta):
        t0 = time.perf_counter()
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        it = (time.perf_counter() - t0) / 3
        # identical on every rank, or a collective goes unanswered and the job hangs
        nt = torch.tensor([max(4, int((WINDOW_S + DISCARD_S) / max(it, 1e-6)))],
                          dtype=torch.int64, device="cuda")
        dist.broadcast(nt, src=0)
        iters = int(nt.item())
        b = Boards(WORLD) if rank == 0 else None
        if b:
            b.__enter__()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
            torch.cuda.synchronize()
        wall = (time.perf_counter() - t0) / iters * 1e3
        if b:
            b.__exit__()
            s = b.summary()
            row = dict(dict.fromkeys(FIELDS, ""), **meta)
            row.update(mode=mode, iter_ms=round(wall, 5), iters=iters, ar_mib=AR_MIB,
                       world=WORLD, clock_min=s["clock_min"],
                       clock_held=s["clock_min"] >= meta["clock"] - CLOCK_TOL,
                       power_node_w=s["power_node_w"],
                       power_per_gpu_w=s["power_per_gpu_w"], n_samples=s["n_samples"])
            wr.writerow(row); fh.flush()
            print(f"  {ctas:>2}CTA {meta['clock']:>4}MHz {meta['m']:>4}x{meta['n']:>5} "
                  f"{mode:>11} | {wall:8.3f} ms  {s['power_per_gpu_w']:6.1f} W/GPU  "
                  f"clk{s['clock_min']:>5}", flush=True)

    for clock in CLOCKS:
        if rank == 0:
            sh(["sudo", "nvidia-smi", "-i", GPUS, "-lgc", f"{clock},{clock}"])
            time.sleep(0.6)
        dist.barrier()

        if ("comm_only", ctas, clock, 0, 0) not in done:
            for _ in range(5):
                ar()
            torch.cuda.synchronize()
            run("comm_only", ar, dict(clock=clock, m=0, n=0, k=0, grid=0, waves=0,
                                      s_eff=0, ctas=ctas))

        for (m, n, k) in SHAPES:
            g = GRIDS[f"{m}_{n}_{k}"]
            w_ = -(-g // TOTAL_SM)
            meta = dict(clock=clock, m=m, n=n, k=k, grid=g, waves=w_,
                        s_eff=round(g / w_, 2), ctas=ctas)
            todo = [mo for mo in (("gemm_only",) if do_gemm_only else ()) + ("serial", "concurrent")
                    if (mo, ctas, clock, m, n) not in done]
            if not todo:
                continue
            x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
            wt = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
            gemm = lambda: torch.nn.functional.linear(x, wt)
            for _ in range(5):
                gemm(); ar()
            torch.cuda.synchronize()

            def concurrent():
                ev = torch.cuda.Event(); ev.record(cur)
                for (s_, f_) in ((s_comm, ar), (s_gemm, gemm)):
                    with torch.cuda.stream(s_):
                        s_.wait_event(ev); f_()

            for mo in todo:
                run(mo, {"gemm_only": gemm,
                         "serial": (lambda: (gemm(), ar())),
                         "concurrent": concurrent}[mo], meta)
            del x, wt
            torch.cuda.empty_cache()

    if rank == 0:
        fh.close()
    dist.destroy_process_group()
    if rank == 0:
        q.put(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "data", "overlap_432.csv"))
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    assert torch.cuda.device_count() >= WORLD

    done = set()
    if a.resume and os.path.exists(a.out):
        for r in csv.DictReader(open(a.out)):
            done.add((r["mode"], int(r["ctas"]), int(r["clock"]), int(r["m"]), int(r["n"])))
        print(f"resume: {len(done)} rows present", flush=True)
    elif os.path.exists(a.out):
        os.remove(a.out)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    pm = sh(["nvidia-smi", "-i", GPUS, "--query-gpu=persistence_mode",
             "--format=csv,noheader"]).stdout.split("\n")[0].strip()
    if pm != "Enabled":
        sh(["sudo", "nvidia-smi", "-i", GPUS, "-pm", "1"])
    print(f"world={WORLD}  {len(CLOCKS)} clocks x {len(SHAPES)} shapes x {len(CTAS)} CTAs "
          f"= {len(CLOCKS) * len(SHAPES) * len(CTAS)} concurrent cases   "
          f"(persistence was {pm})\n", flush=True)
    t0 = time.time()
    try:
        mp.set_start_method("spawn", force=True)
        for i, c in enumerate(CTAS):
            with socket.socket() as sk:
                sk.bind(("127.0.0.1", 0)); port = str(sk.getsockname()[1])
            print(f"=== {c} CTA  [{(time.time() - t0) / 60:.0f} min elapsed] ===", flush=True)
            q = mp.Queue()
            ps = [mp.Process(target=worker,
                             args=(r, port, c, i == 0, done, a.out, q))
                  for r in range(WORLD)]
            for p in ps:
                p.start()
            # poll rather than one long q.get(): if a rank dies, notice in seconds
            # instead of blocking the whole night on a timeout
            while True:
                try:
                    q.get(timeout=20); break
                except Exception:
                    if not all(p.is_alive() for p in ps):
                        print(f"!! a rank exited early at {c} CTA "
                              f"({[p.exitcode for p in ps]}) -- moving on", flush=True)
                        break
            for p in ps:
                p.join(timeout=180)
                if p.is_alive():
                    p.terminate()
    finally:
        sh(["sudo", "nvidia-smi", "-i", GPUS, "-rgc"])
        if pm != "Enabled":
            sh(["sudo", "nvidia-smi", "-i", GPUS, "-pm", "0"])
        print("[clocks] restored", flush=True)
    n = sum(1 for _ in csv.DictReader(open(a.out)))
    print(f"\nDONE in {(time.time() - t0) / 60:.1f} min -> {a.out}  ({n} rows)", flush=True)


if __name__ == "__main__":
    main()
