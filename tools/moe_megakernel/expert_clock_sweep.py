#!/usr/bin/env python3
"""Does the energy-optimal clock move with expert load? The cheap experiment that can
kill the project's hypothesis before any kernel is written.

WHY THIS COMES FIRST. In `overlap_exp` a second V/f domain paid off because a GEMM and a
collective are different KINDS of work with genuinely different optimal clocks. In MoE
prefill all 64 experts run the same operation -- (n_e x 2048) @ (2048 x 1024) -- and
differ only in M. If M does not move the energy-optimal clock, then routing imbalance
produces duration differences and nothing else, a work-conserving queue absorbs those for
free, and spatial V/f is worth zero. That has to be checked before B1 and B2 get built.

The one reason it might move is OCCUPANCY: the lightest experts are only 8 tiles and
cannot fill 108 SMs, and `overlap_exp` measured a real low-occupancy correction
(eta(S) = 1 + 0.118 (1 - S/108)). So this is a measurement, not a derivation.

M values are the measured percentiles of the real load distribution, not round numbers.
Both GEMMs of the expert are swept: gate/up (N=1024, K=2048) and down (N=2048, K=1024).

Method is inherited from overlap_exp: lock the clock, discard the head of the NVML window,
median over a sustained loop, and drop any point where the clock did not hold.

  python3 expert_clock_sweep.py
"""
import argparse, csv, os, statistics as st, subprocess, time
import torch
from baseline_b0 import Power

HERE = os.path.dirname(os.path.abspath(__file__))
CLOCKS = [300, 510, 705, 900, 1050, 1200, 1305, 1410]
# percentiles of the measured n_e distribution (prefill B=16 and B=128), plus the extremes
MS = [72, 355, 955, 1702, 2598, 4244, 7652, 13581, 30754]
SHAPES = [("gate_up", 1024, 2048), ("down", 2048, 1024)]   # (name, N, K)
SM, TOL = 108, 20


def lock(f):
    subprocess.run(["sudo", "nvidia-smi", "-i", "0", "-lgc", f"{f},{f}"],
                   capture_output=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--secs", type=float, default=2.5)
    ap.add_argument("--out", default=os.path.join(HERE, "data", "expert_clock.csv"))
    a = ap.parse_args()
    subprocess.run(["sudo", "nvidia-smi", "-i", "0", "-pm", "1"], capture_output=True)
    torch.cuda.init()
    rows = []
    try:
        for f in CLOCKS:
            lock(f)
            time.sleep(0.5)
            for name, N, K in SHAPES:
                for M in MS:
                    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
                    Bm = torch.randn(K, N, device="cuda", dtype=torch.float16)
                    fn = lambda: torch.mm(A, Bm)
                    for _ in range(20):
                        fn()
                    torch.cuda.synchronize()
                    with Power() as p:
                        n, t0 = 0, time.time()
                        while time.time() - t0 < a.secs:
                            fn(); n += 1
                        torch.cuda.synchronize()
                        wall = time.time() - t0
                    s = p.summary()
                    ms = wall / n * 1e3
                    grid = -(-M // 128) * -(-N // 128)
                    rows.append(dict(
                        gemm=name, m=M, n=N, k=K, clock=f,
                        clock_med=s["clock_med"], held=abs(s["clock_med"] - f) <= TOL,
                        grid=grid, waves=-(-grid // SM), s_eff=round(grid / max(1, -(-grid // SM)), 1),
                        ms=round(ms, 4), power_w=s["power_w"],
                        energy_mj=round(s["power_w"] * ms, 3), reps=n))
                    del A, Bm
                    torch.cuda.empty_cache()
            r = [x for x in rows if x["clock"] == f]
            print(f"  {f:4d} MHz  {len(r)} shapes   held={all(x['held'] for x in r)}",
                  flush=True)
    finally:
        subprocess.run(["sudo", "nvidia-smi", "-i", "0", "-rgc"], capture_output=True)
        print("  [clocks reset]")
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {a.out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
