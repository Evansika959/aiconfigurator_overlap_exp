#!/usr/bin/env python3
"""Measure kappa PER PARTITION and per tile size -- the quantity every earlier
two-domain estimate assumed away.

WHY THIS EXISTS

`predict_two_domain.py` charges each partition's work at the WHOLE LAYER's kappa. That
is the model's weakest assumption, and it is not a small one: it assumes the light and
heavy partitions differ only in how much work they hold, not in what a unit of that work
costs. If that were true a custom kernel would have nothing to offer beyond what vLLM's
grouped GEMM already does, which is the same kernel on the same tile shape for every
expert.

A custom kernel's actual levers are visible in the routing:

  1. TILE SIZE. At BLOCK_M=128 the 16 least-loaded experts of this layer occupy 3200
     padded rows for 1821 real tokens -- 76% waste. At BLOCK_M=16 the same tokens need
     1936 rows, 6% waste. That is a pure work saving at a single clock, no DVFS.
  2. The kappa heterogeneity the small tile CREATES. A narrow tile reads the same expert
     weight panel for fewer rows of A, so its arithmetic intensity is lower and it is
     more memory-bound. Memory-bound work has a different kappa curve -- that is the
     heterogeneity a second voltage domain needs and that a uniform grouped GEMM does
     not have.

Lever 1 is arithmetic and already quantified. Lever 2 has to be measured, which is what
this does: run ONE partition alone on a named number of SMs, at a locked clock, and
divide its dynamic power by (clock x SMs busy). One resident block per SM, so SMs busy
is exact rather than modelled.

  python3 partition_kappa.py --light-experts 16 --repeats 3

DECODE MODE (--decode-cells N). Prefill gave no heterogeneity because every partition
is compute-bound. Decode is where it could appear: batch_clock_sweep.py measured the
whole layer's clock elasticity at ~0.6 for batches up to 1024 tokens and ~0.9 at 4096,
and inside one batch of 1024-2048 the hot experts sit above the crossing while the cold
ones stream weights for a handful of tokens. N pools N real decode (rep, step) cells,
64 tokens each, exactly as batch_clock_sweep.py does. --with-all adds the whole layer on
all 108 SMs at 1 block/SM, to check against that sweep's queue kernel whether the
low occupancy of a persistent kernel changes the elasticity by itself.

  python3 partition_kappa.py --decode-cells 16 --light-experts 64 --block-m 16,64,128 \\
      --with-all --repeats 3
"""
import argparse, csv, json, os, random, subprocess, sys, time
import numpy as np
import torch
import triton

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from persistent_moe import PersistentMoE, _launch
from b1_bench import SHAPES, load_counts, make_inputs
from batch_clock_sweep import decode_batches
from baseline_b0 import Power

HERE = os.path.dirname(os.path.abspath(__file__))
TOL = 7


def clocks():
    return [1290, 1245, 1200, 1155, 1110, 1065, 1035, 990, 930, 870, 780, 690, 600]


def build(p, mb, which):
    """tile ids for one row-block subset, for GEMM 1 or 2."""
    return p.tiles(which, mb, key=("solo", which, int(mb.numel()))).contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--light-experts", type=int, default=16)
    ap.add_argument("--block-m", default="16,32,64,128")
    ap.add_argument("--sms", type=int, default=54,
                    help="SMs the partition runs on, one resident block each")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--secs", type=float, default=1.5)
    ap.add_argument("--decode-cells", type=int, default=0,
                    help="pool this many real decode cells (64 tokens each) instead "
                         "of the prefill batch; 0 = prefill")
    ap.add_argument("--with-all", action="store_true",
                    help="also run the whole layer on all SMs, 1 block/SM")
    ap.add_argument("--all-sms", type=int, default=108)
    ap.add_argument("--clocks", default=None,
                    help="comma-separated MHz; default is the 13-clock grid in clocks(). "
                         "Above 1290 only a partial-width run holds its clock: at full "
                         "width the board hits 400 W and settles near 1320 MHz")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    spec = SHAPES[a.model]
    if a.decode_cells:
        counts = decode_batches(spec, a.layer, a.decode_cells)
        counts[np.argmax(counts)] -= int(counts.sum()) % spec["topk"]
        tag = f"_decode{int(counts.sum()) // spec['topk']}"
    else:
        counts = load_counts(spec, a.rep, a.layer)
        tag = ""
    out = a.out or os.path.join(HERE, "data", f"partition_kappa_{a.model}{tag}.csv")
    x, w1, w2, tw, ti, M = make_inputs(spec, counts)
    base = json.load(open(os.path.join(HERE, "data", f"b1_config_{a.model}.json")))
    light_e = np.argsort(counts, kind="stable")[:a.light_experts]
    wmib = 3 * spec["K"] * spec["I"] * 2 / 2**20      # one expert's w1 + w2, MiB

    jobs = []
    for bm in [int(v) for v in a.block_m.split(",")]:
        cfg = dict(base); cfg["BLOCK_SIZE_M"] = bm
        p = PersistentMoE(x, w1, w2, tw, ti, cfg)
        mask = np.isin(p.rowblock_expert.cpu().numpy(), light_e)
        mb = {"light": p.live_m[torch.from_numpy(mask).to(p.live_m.device)],
              "heavy": p.live_m[torch.from_numpy(~mask).to(p.live_m.device)]}
        if a.with_all:
            mb["all"] = p.live_m
        for name, rows in mb.items():
            if rows.numel() == 0:
                continue
            t1, t2 = build(p, rows, "g1"), build(p, rows, "g2")
            # real token rows this partition carries, and padded rows it computes
            e = p.rowblock_expert[
                torch.isin(p.live_m, rows)].cpu().numpy()
            real = int(counts[np.unique(e)].sum())
            padded = int(rows.numel() * bm)
            nexp = int(len(np.unique(e)))
            nsm_part = a.all_sms if name == "all" else a.sms

            def body(p=p, t1=t1, t2=t2, cfg=cfg, nsm=nsm_part):
                _launch(p.x, p.w1, p.c1, p.tw, p.sorted_ids, p.expert_ids, t1,
                        nsm, False, p.topk, cfg, p.ct)
                _launch(p.c2, p.w2, p.c3, p.tw, p.sorted_ids, p.expert_ids, t2,
                        nsm, True, 1, cfg, p.ct)
            jobs.append(dict(block_m=bm, part=name, real=real, padded=padded,
                             tiles=int(t1.numel() + t2.numel()), fn=body,
                             sms=nsm_part, experts=nexp,
                             weight_mib=round(nexp * wmib, 1)))

    print(f"{a.model} layer{a.layer}{tag or ' prefill'}: {a.light_experts} lightest "
          f"experts are 'light'; each partition runs alone on {a.sms} SMs, 1 block/SM"
          + (f"; 'all' on {a.all_sms}" if a.with_all else ""))
    print(f"  tokens per expert: min {counts.min()} median {np.median(counts):.0f} "
          f"max {counts.max()}; light experts carry {counts[light_e].max()} or fewer")
    for j in jobs:
        print(f"  BLOCK_M={j['block_m']:<4d} {j['part']:<6s} {j['tiles']:>6d} tiles, "
              f"{j['padded']:>6d} padded rows for {j['real']:>6d} real "
              f"({j['padded']/j['real']-1:+.0%}), {j['experts']} experts, "
              f"{j['weight_mib']:.0f} MiB of weights")

    graphs = []
    for j in jobs:
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(4):
                j["fn"]()
        torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(4):
                j["fn"]()
        graphs.append(g)

    subprocess.run(["sudo", "nvidia-smi", "-i", "0", "-pm", "1"], capture_output=True)
    rows_out, rng = [], random.Random(11)
    try:
        for pas in range(a.repeats):
            grid = ([int(c) for c in a.clocks.split(",")] if a.clocks
                    else clocks()[:])
            rng.shuffle(grid)
            for f in grid:
                subprocess.run(["sudo", "nvidia-smi", "-i", "0", "-lgc", f"{f},{f}"],
                               capture_output=True)
                time.sleep(1.0)
                with Power(discard=0.5) as pw:
                    time.sleep(1.2)
                idle = pw.summary()
                for j, g in zip(jobs, graphs):
                    g.replay(); torch.cuda.synchronize()
                    t0 = time.time(); g.replay(); torch.cuda.synchronize()
                    one = time.time() - t0
                    n = max(2, min(500, int(a.secs / max(one, 1e-4))))
                    with Power(discard=0.4) as pw:
                        torch.cuda.synchronize(); t0 = time.time()
                        for _ in range(n):
                            g.replay()
                        torch.cuda.synchronize(); wall = time.time() - t0
                    sm = pw.summary()
                    ms = wall / (n * 4) * 1e3
                    pdyn = sm["power_w"] - idle["power_w"]
                    rows_out.append(dict(
                        pas=pas, block_m=j["block_m"], part=j["part"], sms=j["sms"],
                        tokens=int(counts.sum()) // spec["topk"], experts=j["experts"],
                        weight_mib=j["weight_mib"],
                        clock=f, clock_med=sm["clock_med"],
                        held=abs(sm["clock_med"] - f) <= TOL,
                        real_rows=j["real"], padded_rows=j["padded"], tiles=j["tiles"],
                        ms=round(ms, 5), power_w=sm["power_w"], idle_w=idle["power_w"],
                        p_dyn_w=round(pdyn, 3),
                        kappa_mw_per_mhz_per_sm=round(pdyn / f * 1e3 / j["sms"], 5),
                        e_dyn_mj=round(pdyn * ms, 5), replays=n))
                print(f"  p{pas} {f:4d} MHz  " + "  ".join(
                    f"{j['part'][0]}{j['block_m']}:{r['kappa_mw_per_mhz_per_sm']:.2f}"
                    for j, r in zip(jobs, rows_out[-len(jobs):])), flush=True)
    finally:
        subprocess.run(["sudo", "nvidia-smi", "-i", "0", "-rgc"], capture_output=True)
        print("  [clocks reset]")

    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
        w.writeheader(); w.writerows(rows_out)
    print(f"\nwrote {out} ({len(rows_out)} rows)")


if __name__ == "__main__":
    main()
