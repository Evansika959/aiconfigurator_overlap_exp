#!/usr/bin/env python3
"""The experiment that can kill B2 before a line of DVFS code is written.

A second voltage domain needs blocks PINNED to a set of SMs. A work-conserving queue
has no slack to sell: a block that finishes an expert's tile immediately takes another
expert's tile, so imbalance becomes tail time, not idle SMs. Pinning creates the idle
SMs we want to run slow -- and charges makespan for them, because the layer now ends
when the SLOWER partition ends.

The DVFS prize, from `overlap_exp` and from `RESULTS_B1.md`, is 1-5%. So:

    if pinning alone costs more makespan than a second domain could ever save,
    B2 is dead, and no DVFS hardware was needed to find that out.

This measures exactly that, at ONE locked clock, with no DVFS at all:

  queue      vLLM's kernel, one block per tile, hardware scheduler places them
  persistent one persistent launch, 108 blocks, still one shared work list
  pinned     two concurrent persistent launches on disjoint SM counts, the light
             experts in one partition and the heavy ones in the other

The SM split is swept rather than assumed, so the number reported is the cost of the
BEST partition at each cut, not of a badly chosen one.

  python3 pin_vs_queue.py --clock 1200
"""
import argparse, csv, json, os, random, subprocess, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vendor.fused_moe_triton import PreparedMoE
from persistent_moe import PersistentMoE
from b1_bench import SHAPES, load_counts, make_inputs, flops
from baseline_b0 import Power

HERE = os.path.dirname(os.path.abspath(__file__))


def graphed(fn, streams=()):
    """Capture fn into a CUDA graph. Side streams must exist before capture and be
    joined inside it, which is what PersistentMoE.run does."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return g


def timed_graph(g, iters=50):
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(iters)]
    for a, b in ev:
        a.record(); g.replay(); b.record()
    torch.cuda.synchronize()
    t = np.array([a.elapsed_time(b) for a, b in ev])
    return float(np.median(t)), float(t.std())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen", choices=list(SHAPES))
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--clock", type=int, default=1200,
                    help="locked SM clock; 1200 avoids the 400 W cap region where "
                         "every configuration is throttled to the same state")
    ap.add_argument("--light-experts", default="8,16,32,48,64,96")
    ap.add_argument("--blocks-per-sm", type=int, default=2,
                    help="resident blocks per SM. 1 is the crisp '1 CTA = 1 SM' "
                         "partition overlap_exp verified with ncu, but it forfeits the "
                         "latency hiding the non-persistent kernel gets free and costs "
                         "+24%% against it; 2 closes that to +5%%.")
    ap.add_argument("--repeats", type=int, default=1,
                    help="independent passes; cases are reshuffled each pass. The "
                         "handicap this script produces feeds every point of the "
                         "two-domain composition, so it needs the same 6-pass "
                         "treatment as the B1 clock sweep, not n=1.")
    ap.add_argument("--sm-offsets", default="-8,-4,-2,0,2,4,8",
                    help="SM allocations to try around the work-proportional split")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(HERE, "data", f"pin_vs_queue_{a.model}.csv")

    spec = SHAPES[a.model]
    counts = load_counts(spec, a.rep, a.layer)
    x, w1, w2, tw, ti, M = make_inputs(spec, counts)
    cfg = json.load(open(os.path.join(HERE, "data", f"b1_config_{a.model}.json")))
    F = flops(spec, counts)

    p = PersistentMoE(x, w1, w2, tw, ti, cfg)
    S = p.n_sm
    cnt = torch.from_numpy(counts).to(p.live_m.device)[p.rowblock_expert.long()]
    order = torch.argsort(cnt, descending=True)          # heavy row-blocks first
    ntile_g1 = p.tiles("g1").numel() / p.live_m.numel()  # tiles per row-block

    print(f"{a.model} rep{a.rep} layer{a.layer}: M={M}, {p.live_m.numel()} live "
          f"row-blocks over {spec['E']} experts, {S} SMs, clock {a.clock} MHz")
    print(f"expert load: min={counts.min()} mean={counts.mean():.0f} max={counts.max()}")

    prep = PreparedMoE(x, w1, w2, tw, ti, cfg)
    B = a.blocks_per_sm
    cases = [("queue", lambda: prep.run(), None, None),
             ("persistent_1blk_per_sm", lambda: p.run(blocks_per_sm=1), S, None),
             (f"persistent_{B}blk_per_sm", lambda: p.run(blocks_per_sm=B), S, None)]

    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    for L in [int(v) for v in a.light_experts.split(",")]:
        if L >= spec["E"]:
            continue
        # light = the L least-loaded EXPERTS; find their row-blocks
        light_e = np.argsort(counts)[:L]
        mask = np.isin(p.rowblock_expert.cpu().numpy(), light_e)
        mb_light = p.live_m[torch.from_numpy(mask).to(p.live_m.device)]
        mb_heavy = p.live_m[torch.from_numpy(~mask).to(p.live_m.device)]
        share = mb_light.numel() / p.live_m.numel()      # share of tiles, not of tokens
        base = max(1, min(S - 1, int(round(S * share))))
        for n_light in sorted({max(1, min(S - 1, base + int(d)))
                               for d in a.sm_offsets.split(",")}):
            cases.append((f"pinned_L{L}_sm{n_light}", None, n_light,
                          (mb_heavy, mb_light, L, share)))

    subprocess.run(["sudo", "nvidia-smi", "-i", "0", "-pm", "1"], capture_output=True)
    subprocess.run(["sudo", "nvidia-smi", "-i", "0", "-lgc", f"{a.clock},{a.clock}"],
                   capture_output=True)
    time.sleep(1.5)
    rows = []
    rng = random.Random(4321)
    try:
      for pas in range(a.repeats):
        order = list(range(len(cases)))
        rng.shuffle(order)
        for ci in order:
            name, fn, nsm, part = cases[ci]
            if part is not None:
                mb_heavy, mb_light, L, share = part
                n_light = nsm
                p._tile_cache.clear()
                fn = (lambda mh=mb_heavy, ml=mb_light, nl=n_light:
                      p.run(partition=[(mh, S - nl), (ml, nl)], streams=streams,
                            blocks_per_sm=B))
            g = graphed(fn)
            with Power(discard=0.5) as pw:
                torch.cuda.synchronize(); t0 = time.time()
                n = 0
                while time.time() - t0 < 2.0:
                    for _ in range(20):
                        g.replay()
                    n += 20
                torch.cuda.synchronize(); wall = time.time() - t0
            sm = pw.summary()
            ms = wall / n * 1e3
            rows.append(dict(case=name, pas=pas, clock=a.clock,
                             clock_med=sm["clock_med"],
                             sm_light=(nsm if part is not None else ""),
                             sm_heavy=(S - nsm if part is not None else ""),
                             light_experts=(part[2] if part else ""),
                             tile_share_light=(round(part[3], 4) if part else ""),
                             ms=round(ms, 5), power_w=sm["power_w"],
                             energy_mj=round(sm["power_w"] * ms, 4),
                             tflops=round(F / ms * 1e-9, 2)))
            print(f"  {name:22s} {ms:7.3f} ms  {sm['power_w']:6.1f} W  "
                  f"E={rows[-1]['energy_mj']:7.1f} mJ", flush=True)
            del g
    finally:
        subprocess.run(["sudo", "nvidia-smi", "-i", "0", "-rgc"], capture_output=True)
        print("  [clocks reset]")

    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    agg = {}
    for r in rows:
        agg.setdefault(r["case"], []).append(r)
    def stat(case):
        v = np.array([x["ms"] for x in agg[case]])
        return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0), len(v)
    print(f"\n{'case':24s} {'ms':>8s} {'sd':>7s}  n")
    for c in agg:
        m, s_, n = stat(c)
        print(f"  {c:22s} {m:8.3f} {s_:7.3f}  {n}")
    q = [r for r in rows if r["case"] == "queue"][0]
    pers = [r for r in rows if r["case"] == f"persistent_{B}blk_per_sm"][0]
    pers1 = [r for r in rows if r["case"] == "persistent_1blk_per_sm"][0]
    pin = [r for r in rows if r["case"].startswith("pinned")]
    best = min(pin, key=lambda r: r["ms"]) if pin else None
    print(f"\nwrote {out} ({len(rows)} rows)")
    print(f"  queue (vLLM)                {q['ms']:.3f} ms")
    print(f"  persistent, 1 blk/SM        {pers1['ms']:.3f} ms  "
          f"({pers1['ms']/q['ms']-1:+.1%} vs vLLM)")
    print(f"  persistent, {B} blk/SM        {pers['ms']:.3f} ms  "
          f"({pers['ms']/q['ms']-1:+.1%} vs vLLM -- the cost of persistence alone)")
    if best:
        print(f"  best pinned split   {best['ms']:.3f} ms  ({best['case']})")
        print(f"    vs vLLM        {best['ms']/q['ms']-1:+.1%}")
        print(f"    vs persistent  {best['ms']/pers['ms']-1:+.1%}  "
              f"<- THE PINNING TAX the DVFS saving has to beat")


if __name__ == "__main__":
    main()
