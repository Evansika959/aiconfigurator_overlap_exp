#!/usr/bin/env python3
"""Is the energy-optimal clock a function of BATCH SIZE? Two-dimensional sweep.

THE QUESTION

The same MoE layer is two different machines depending on how many tokens it holds.
At batch 64 in decode it moves 549 MiB of expert weights to process 6.25 MiB of tokens
-- 88x more weight than token -- so the SMs sit waiting on memory and the clock should
barely matter. At 7744 prefill tokens the ratio is 2x, SM throughput is 79% of peak, and
the clock buys latency directly. Between those the layer crosses from weight-bound to
compute-bound.

If the energy-optimal clock moves across that crossing, then a serving system -- which
knows its batch size at every instant -- is leaving energy on the table by running one
fixed clock. If it does not move, that is a clean negative and it explains why the
coarse "run decode slower than prefill" rule is already enough.

WHY THE TIMESCALE WORKS, when per-batch DVFS does not

`PLAN.md` measured an NVML clock change at 37.7-38.5 ms, against MoE layers of tens of
microseconds. Adapting per batch is impossible by three orders of magnitude. Batch SIZE
is different: it tracks offered load, which moves on seconds to minutes. This is the
"statistical" escape of PLAN.md section 1, made into a table a scheduler can index.

THE ROUTING IS REAL, NOT SCALED

Batches are built by POOLING real decode token routings from data/routing_*.npz, which
is valid because in decode each token picks its experts independently of the others in
the batch: 64 tokens x (rep, step) cells. Scaling a histogram instead would invent a
routing distribution that no model ever produced.

  python3 batch_clock_sweep.py --repeats 3          # ~25 min, locks clocks on GPU 0
"""
import argparse, csv, json, os, random, subprocess, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vendor.fused_moe_triton import PreparedMoE
from b1_bench import SHAPES, flops
from baseline_b0 import Power

HERE = os.path.dirname(os.path.abspath(__file__))
TOL = 7
REPS = 4          # layers per graph replay


def clocks():
    c = list(range(510, 901, 90)) + list(range(915, 1291, 45)) + [1290]
    return sorted(set(c), reverse=True)


def decode_batches(spec, layer, cells):
    """Pool `cells` real decode (rep, step) routings into one batch of 64*cells tokens."""
    Z = np.load(os.path.join(HERE, "data", spec["npz"]))
    key = [k for k in Z.files if k.startswith("decode")][0]
    A = Z[key]                       # (reps, steps, layers, experts)
    flat = A[:, :, layer, :].reshape(-1, A.shape[-1])
    return flat[:cells].sum(axis=0).astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen", choices=list(SHAPES))
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--cells", default="1,2,4,8,16,32,64",
                    help="decode (rep,step) cells pooled per batch; 1 cell = 64 tokens")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--secs", type=float, default=1.2)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(HERE, "data", f"batch_clock_{a.model}.csv")

    spec = SHAPES[a.model]
    cfg = json.load(open(os.path.join(HERE, "data", f"b1_config_{a.model}.json")))
    K, I, E, topk = spec["K"], spec["I"], spec["E"], spec["topk"]
    dev, dt = "cuda", torch.bfloat16
    w1 = torch.randn(E, 2 * I, K, device=dev, dtype=dt) / K ** 0.5
    w2 = torch.randn(E, K, I, device=dev, dtype=dt) / I ** 0.5

    jobs = []
    for cells in [int(v) for v in a.cells.split(",")]:
        counts = decode_batches(spec, a.layer, cells)
        tot = int(counts.sum()); tot -= tot % topk
        M = tot // topk
        g = torch.Generator().manual_seed(0)
        flat = torch.repeat_interleave(torch.arange(E, dtype=torch.int32),
                                       torch.from_numpy(counts))[:tot]
        ti = flat[torch.randperm(tot, generator=g)].view(M, topk).contiguous().to(dev)
        x = torch.randn(M, K, device=dev, dtype=dt) / K ** 0.5
        tw = torch.rand(M, topk, device=dev, dtype=torch.float32)
        tw /= tw.sum(1, keepdim=True)
        p = PreparedMoE(x, w1, w2, tw, ti, cfg)
        live = int((counts > 0).sum())
        jobs.append(dict(cells=cells, tokens=M, pairs=tot, live=live,
                         per_expert=tot / max(live, 1), p=p,
                         gflop=2.0 * tot * K * 2 * I + 2.0 * tot * I * K))

    print(f"{a.model} layer {a.layer}: real decode routing pooled into batches")
    print(f"{'tokens':>8s} {'pairs':>8s} {'experts hit':>12s} {'pairs/expert':>13s} "
          f"{'GFLOP':>8s}")
    for j in jobs:
        print(f"{j['tokens']:>8d} {j['pairs']:>8d} {j['live']:>9d}/{E} "
              f"{j['per_expert']:>13.1f} {j['gflop']/1e9:>8.1f}")

    graphs = []
    for j in jobs:
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(4):
                j["p"].run()
        torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            for _ in range(REPS):
                j["p"].run()
        graphs.append(gr)

    subprocess.run(["sudo", "nvidia-smi", "-i", "0", "-pm", "1"], capture_output=True)
    rows, rng = [], random.Random(99)
    try:
        for pas in range(a.repeats):
            grid = clocks()[:]
            rng.shuffle(grid)
            for f in grid:
                subprocess.run(["sudo", "nvidia-smi", "-i", "0", "-lgc", f"{f},{f}"],
                               capture_output=True)
                time.sleep(1.0)
                with Power(discard=0.5) as pw:
                    time.sleep(1.2)
                idle = pw.summary()
                for j, gr in zip(jobs, graphs):
                    gr.replay(); torch.cuda.synchronize()
                    t0 = time.time(); gr.replay(); torch.cuda.synchronize()
                    one = time.time() - t0
                    n = max(2, min(800, int(a.secs / max(one, 1e-4))))
                    with Power(discard=0.4) as pw:
                        torch.cuda.synchronize(); t0 = time.time()
                        for _ in range(n):
                            gr.replay()
                        torch.cuda.synchronize(); wall = time.time() - t0
                    sm = pw.summary()
                    ms = wall / (n * REPS) * 1e3
                    pdyn = sm["power_w"] - idle["power_w"]
                    rows.append(dict(
                        pas=pas, tokens=j["tokens"], pairs=j["pairs"],
                        per_expert=round(j["per_expert"], 2), clock=f,
                        clock_med=sm["clock_med"],
                        held=abs(sm["clock_med"] - f) <= TOL,
                        ms_per_layer=round(ms, 6), power_w=sm["power_w"],
                        idle_w=idle["power_w"], p_dyn_w=round(pdyn, 3),
                        uj_per_token=round(sm["power_w"] * ms / j["tokens"] * 1000, 4),
                        uj_dyn_per_token=round(pdyn * ms / j["tokens"] * 1000, 4),
                        tflops=round(j["gflop"] / ms * 1e-9, 2), replays=n))
                print(f"  p{pas} {f:4d} MHz  " + "  ".join(
                    f"{j['tokens']}:{r['uj_per_token']:.1f}"
                    for j, r in zip(jobs, rows[-len(jobs):])), flush=True)
    finally:
        subprocess.run(["sudo", "nvidia-smi", "-i", "0", "-rgc"], capture_output=True)
        print("  [clocks reset]")

    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
