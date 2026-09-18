#!/usr/bin/env python3
"""If experts do not differ from each other, do the STAGES differ from each other?

`expert_clock_sweep.py` found every expert GEMM wants 1050 MHz regardless of its load,
which kills "experts want different clocks". But a MoE layer is not only expert GEMMs: a
megakernel fuses the router GEMM, the permute/gather, the expert GEMMs and the
scatter/combine into ONE kernel, and those are not the same kind of work.

The permute and the combine are pure data movement -- bandwidth bound. `overlap_exp`
established that a bandwidth-bound kernel and a compute-bound kernel have genuinely
different energy-optimal clocks, which is exactly why a second domain paid off there. So
the hypothesis is not dead, it may just have been pointed at the wrong axis.

Same method: locked clocks, sustained loop, NVML median, held-clock filter.
"""
import argparse, csv, os, subprocess, time
import torch
from baseline_b0 import Power

HERE = os.path.dirname(os.path.abspath(__file__))
CLOCKS = [300, 510, 705, 900, 1050, 1200, 1305, 1410]
T, H, I, E, K = 8192, 2048, 1024, 64, 8      # OLMoE prefill B=16, one layer
TOL = 20


def build():
    """The four stages at their real prefill shapes, each as a callable."""
    x = torch.randn(T, H, device="cuda", dtype=torch.float16)
    gate_w = torch.randn(H, E, device="cuda", dtype=torch.float16)
    idx = torch.randint(0, T, (T * K,), device="cuda")
    xp = torch.randn(T * K, H, device="cuda", dtype=torch.float16)
    w1 = torch.randn(H, I, device="cuda", dtype=torch.float16)
    out = torch.zeros(T, H, device="cuda", dtype=torch.float16)
    yp = torch.randn(T * K, H, device="cuda", dtype=torch.float16)
    return [
        ("router_gemm",  lambda: torch.mm(x, gate_w),          "compute, tiny N"),
        ("permute",      lambda: torch.index_select(x, 0, idx), "pure data movement"),
        ("expert_gemm",  lambda: torch.mm(xp[:65536], w1),     "compute, the 99.7%"),
        ("combine",      lambda: out.index_add_(0, idx, yp),   "pure data movement"),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--secs", type=float, default=2.5)
    ap.add_argument("--out", default=os.path.join(HERE, "data", "stage_clock.csv"))
    a = ap.parse_args()
    subprocess.run(["sudo", "nvidia-smi", "-i", "0", "-pm", "1"], capture_output=True)
    torch.cuda.init()
    stages = build()
    rows = []
    try:
        for f in CLOCKS:
            subprocess.run(["sudo", "nvidia-smi", "-i", "0", "-lgc", f"{f},{f}"],
                           capture_output=True)
            time.sleep(0.5)
            for name, fn, kind in stages:
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
                rows.append(dict(stage=name, kind=kind, clock=f,
                                 clock_med=s["clock_med"],
                                 held=abs(s["clock_med"] - f) <= TOL,
                                 ms=round(ms, 4), power_w=s["power_w"],
                                 energy_mj=round(s["power_w"] * ms, 3), reps=n))
            print(f"  {f:4d} MHz done", flush=True)
    finally:
        subprocess.run(["sudo", "nvidia-smi", "-i", "0", "-rgc"], capture_output=True)
        print("  [clocks reset]")
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {a.out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
