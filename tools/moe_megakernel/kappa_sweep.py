#!/usr/bin/env python3
"""Is the voltage floor a property of the RAIL, or of the workload we happened to measure?

kappa = P_dyn/f = C*V^2 is the energy per unit of work. The claim that a slack expert
cannot be made cheaper by slowing it down rests entirely on kappa being FLAT below a knee
near 1020-1080 MHz. But that shape has only ever been measured on large bf16 GEMMs
(gemm_dcfs_char, 12 shapes with N=K in 4096-16384).

If kappa really is the rail's V(f), its SHAPE must be identical for any workload -- only
its LEVEL may move, because the effective switched capacitance C differs. This sweep tests
that against the MoE expert shape, whose N and K are several times smaller.

  row 1  large bf16 GEMM, (4096 x 8192) @ (8192 x 8192)   -- control, matches the campaign
  row 2  MoE expert GEMM, (n_e x 2048) @ (2048 x 1024)    -- two n_e from the measured
         routing distribution: 955 (prefill B=16 median) and 7652 (B=128 median)

TWO THINGS THIS FIXES relative to my earlier 8-clock sweep:

  * CUDA GRAPHS. A tight torch.mm loop carries a 5-10 us launch gap per call, and the
    small expert GEMM at 1050 MHz takes only ~17 us -- so most of what I measured there
    was launch overhead, which faked non-compute-boundness and produced a spurious 2-3%
    "saving". Each measurement here replays a graph of REPS back-to-back GEMMs, so the gap
    is amortised by that factor and the timing is the kernel's.
  * THE MEASUREMENT LOOP IS BOUNDED. `g.replay()` is asynchronous, so a
    "while time < secs: replay()" loop queues hundreds of replays in a couple of seconds
    and the trailing synchronize() then drains that backlog -- the timing still came out
    right, but each clock took minutes instead of seconds. The loop now calibrates one
    replay first and issues only as many as fill the window.
  * P_idle IS MEASURED AT EVERY CLOCK, in the same pass, so P_dyn = P - P_idle(f) uses a
    floor from the same thermal state rather than a table from another day.

Whole-GPU kappa is enough here: only the SHAPE is under test, and the 15 MHz campaign
already showed whole-GPU and per-SM fits agree on shape.

  python3 kappa_sweep.py                 # ~20 min, locks clocks on GPU 0 only
"""
import argparse, csv, os, statistics as st, subprocess, time
import torch
from baseline_b0 import Power

HERE = os.path.dirname(os.path.abspath(__file__))
REPS = 64          # GEMMs captured per graph replay; amortises the launch gap
TOL = 20


def clocks():
    """A100 supports 15 MHz steps. Dense through the knee, coarser far from it, because
    the knee's location is what the experiment is about."""
    c = list(range(510, 901, 45)) + list(range(915, 1141, 15)) + \
        list(range(1155, 1411, 45))
    return sorted(set([300, 405] + c), reverse=True)


def graphed(fn):
    """Capture REPS calls into one graph. Warm up on a side stream first, or capture
    fails on cuBLAS's lazy handle init."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(REPS):
            fn()
    return g


def build():
    W = []
    A = torch.randn(4096, 8192, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    W.append(("gemm_large", 4096, 8192, 8192, lambda: torch.mm(A, B)))
    for ne in (955, 7652):
        a = torch.randn(ne, 2048, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(2048, 1024, device="cuda", dtype=torch.bfloat16)
        W.append((f"moe_expert_n{ne}", ne, 1024, 2048, lambda a=a, b=b: torch.mm(a, b)))
    return W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--secs", type=float, default=2.5)
    ap.add_argument("--idle-secs", type=float, default=2.0)
    ap.add_argument("--out", default=os.path.join(HERE, "data", "kappa_sweep.csv"))
    a = ap.parse_args()
    subprocess.run(["sudo", "nvidia-smi", "-i", "0", "-pm", "1"], capture_output=True)
    torch.cuda.init()
    work = [(n, m, N, K, graphed(fn)) for n, m, N, K, fn in build()]
    print(f"{len(work)} workloads, {len(clocks())} clocks, graph of {REPS} GEMMs\n")
    rows = []
    try:
        for f in clocks():
            subprocess.run(["sudo", "nvidia-smi", "-i", "0", "-lgc", f"{f},{f}"],
                           capture_output=True)
            time.sleep(1.2)                                   # let the rail settle
            with Power(discard=0.6) as p:
                time.sleep(a.idle_secs)
            idle = p.summary()
            for name, m, N, K, g in work:
                g.replay(); torch.cuda.synchronize()
                t0 = time.time(); g.replay(); torch.cuda.synchronize()
                one = time.time() - t0                      # calibrate one replay
                n = max(2, min(400, int(a.secs / max(one, 1e-4))))
                with Power(discard=0.5) as p:
                    torch.cuda.synchronize(); t0 = time.time()
                    for _ in range(n):
                        g.replay()
                    torch.cuda.synchronize()
                    wall = time.time() - t0
                s = p.summary()
                ms = wall / (n * REPS) * 1e3
                pdyn = s["power_w"] - idle["idle_w" if "idle_w" in idle else "power_w"]
                rows.append(dict(
                    workload=name, m=m, n=N, k=K, clock=f, clock_med=s["clock_med"],
                    held=abs(s["clock_med"] - f) <= TOL, ms_per_gemm=round(ms, 6),
                    power_w=s["power_w"], idle_w=idle["power_w"],
                    p_dyn_w=round(pdyn, 3), kappa_mw_per_mhz=round(pdyn / f * 1e3, 4),
                    energy_mj=round(s["power_w"] * ms, 5),
                    e_dyn_mj=round(pdyn * ms, 5), replays=n, reps=REPS))
            r = [x for x in rows if x["clock"] == f]
            print(f"  {f:4d} MHz  idle {idle['power_w']:6.2f} W   "
                  + "  ".join(f"{x['workload'].split('_')[-1]}:{x['kappa_mw_per_mhz']:.1f}"
                              for x in r)
                  + ("" if all(x["held"] for x in r) else "   THROTTLED"), flush=True)
    finally:
        subprocess.run(["sudo", "nvidia-smi", "-i", "0", "-rgc"], capture_output=True)
        print("  [clocks reset]")
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {a.out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
