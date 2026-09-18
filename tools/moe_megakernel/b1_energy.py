#!/usr/bin/env python3
"""B1 energy -- the first honest MoE energy measurement in this project.

Sweeps SM clock over a real fused grouped-GEMM MoE layer (vLLM's Triton kernel,
vendored) driven by a REAL traced expert histogram, and records latency, NVML
power, and energy per layer at each clock.

This is what every earlier energy number in `moe_megakernel` was missing: B0 is a
Python loop that is 92% idle, and everything else was composed from `overlap_exp`
micro-benchmarks. This measures the actual thing.

  python3 b1_energy.py --model qwen            # ~10 min, locks clocks on GPU 0
"""
import argparse, csv, json, os, random, subprocess, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vendor.fused_moe_triton import PreparedMoE, default_config
from baseline_b0 import Power
from b1_bench import SHAPES, load_counts, make_inputs, flops

HERE = os.path.dirname(os.path.abspath(__file__))
REPS = 4          # layers per graph replay
TOL = 20          # MHz; clock_med further than this from the request = throttled


def clocks():
    """15 MHz -- the A100's real step -- from 915 all the way to 1290, not just to 1140.

    The first version stepped 45 MHz above 1155, which left 1170/1185/1215/1230/1260/1275
    unmeasured. Those six were later filled in as a separate campaign, and a separate
    campaign cannot be compared with the first at the few-mJ level: the reshuffle exists
    to stop drift aliasing onto clock, and it does not reach across campaigns. So the
    grid is dense here and the whole thing is collected in one run."""
    c = list(range(510, 901, 45)) + list(range(915, 1291, 15)) + [1335, 1380, 1410]
    return sorted(set(c), reverse=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen", choices=list(SHAPES))
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--secs", type=float, default=2.5)
    ap.add_argument("--idle-secs", type=float, default=2.0)
    ap.add_argument("--repeats", type=int, default=1,
                    help="independent passes over the clock grid; the grid is "
                         "reshuffled each pass so slow thermal drift does not "
                         "alias onto clock")
    ap.add_argument("--clocks", default=None,
                    help="comma-separated clock list, overriding the default grid. "
                         "Used to fill in clocks the original sweep skipped: it steps "
                         "45 MHz above 1155, so 1170/1185/1215/1230/1260/1275 were "
                         "never measured and anything claiming to be 'the A100's real "
                         "15 MHz grid' was interpolating there.")
    ap.add_argument("--append", action="store_true",
                    help="append to --out instead of overwriting it")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(HERE, "data", f"b1_energy_{a.model}.csv")

    spec = SHAPES[a.model]
    counts = load_counts(spec, a.rep, a.layer)
    x, w1, w2, tw, ti, M = make_inputs(spec, counts)

    cfg_path = os.path.join(HERE, "data", f"b1_config_{a.model}.json")
    if os.path.exists(cfg_path):
        cfg = json.load(open(cfg_path)); src = "tuned"
    else:
        cfg = default_config(M, spec["E"], spec["I"], spec["K"], spec["topk"])
        src = "vLLM default (UNTUNED)"

    p = PreparedMoE(x, w1, w2, tw, ti, cfg)
    F = flops(spec, counts)
    print(f"{a.model}: M={M} E={spec['E']} K={spec['K']} I={spec['I']} "
          f"topk={spec['topk']}  {F/1e9:.0f} GFLOP/layer")
    print(f"config ({src}): {cfg}")

    # graph capture; warm up on a side stream first
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5):
            p.run()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(REPS):
            p.run()

    def measure(f, pas):
        """f = requested clock, or None for the unlocked default governor."""
        with Power(discard=0.6) as pw:
            time.sleep(a.idle_secs)
        idle = pw.summary()
        g.replay(); torch.cuda.synchronize()
        t0 = time.time(); g.replay(); torch.cuda.synchronize()
        one = time.time() - t0
        n = max(2, min(400, int(a.secs / max(one, 1e-4))))
        with Power(discard=0.5) as pw:
            torch.cuda.synchronize(); t0 = time.time()
            for _ in range(n):
                g.replay()
            torch.cuda.synchronize()
            wall = time.time() - t0
        sm = pw.summary()
        ms = wall / (n * REPS) * 1e3
        pdyn = sm["power_w"] - idle["power_w"]
        held = f is not None and abs(sm["clock_med"] - f) <= TOL
        row = dict(
            model=a.model, pas=pas, rep=a.rep, layer=a.layer, m_tokens=M,
            clock=(f if f is not None else -1), clock_med=sm["clock_med"],
            held=held, locked=(f is not None),
            ms_per_layer=round(ms, 5), power_w=sm["power_w"],
            idle_w=idle["power_w"], p_dyn_w=round(pdyn, 3),
            energy_mj_per_layer=round(sm["power_w"] * ms, 4),
            e_dyn_mj_per_layer=round(pdyn * ms, 4),
            tflops=round(F / ms * 1e-9, 2),
            kappa_mw_per_mhz_per_sm=round(pdyn / sm["clock_med"] * 1e3 / 108, 5),
            replays=n, reps=REPS)
        tag = f"{f:4d}" if f is not None else "gov."
        print(f"  p{pas} {tag} MHz -> {sm['clock_med']:4d}  {ms:7.3f} ms  "
              f"{sm['power_w']:6.1f} W  E={row['energy_mj_per_layer']:8.1f} mJ"
              + ("" if held or f is None else "   NOT HELD"), flush=True)
        return row

    subprocess.run(["sudo", "nvidia-smi", "-i", "0", "-pm", "1"], capture_output=True)
    rows = []
    rng = random.Random(1234)
    try:
        for pas in range(a.repeats):
            # the deployed operating point: no locking at all. This is the only
            # baseline that is a measurement rather than a choice of row.
            if not a.clocks:          # the governor baseline belongs to a full sweep
                subprocess.run(["sudo", "nvidia-smi", "-i", "0", "-rgc"],
                               capture_output=True)
                time.sleep(1.5)
                rows.append(measure(None, pas))
            grid = ([int(x) for x in a.clocks.split(",")] if a.clocks
                    else clocks()[:])
            rng.shuffle(grid)
            for f in grid:
                subprocess.run(["sudo", "nvidia-smi", "-i", "0", "-lgc", f"{f},{f}"],
                               capture_output=True)
                time.sleep(1.2)
                rows.append(measure(f, pas))
    finally:
        subprocess.run(["sudo", "nvidia-smi", "-i", "0", "-rgc"], capture_output=True)
        print("  [clocks reset]")

    if a.append and os.path.exists(out):
        with open(out, newline="") as fh:
            old = list(csv.DictReader(fh))
        base = max(int(r["pas"]) for r in old) + 1
        for r in rows:
            r["pas"] += base          # keep pass ids distinct from the existing sweep
        with open(out, "a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=list(rows[0].keys())).writerows(rows)
        print(f"appended {len(rows)} rows to {out} (now {len(old)+len(rows)})")
    else:
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out} ({len(rows)} rows, {a.repeats} passes)")
    agg = {}
    for r in rows:
        agg.setdefault(r["clock"], []).append(r)
    gov = agg.get(-1)
    for c, rs in sorted(agg.items()):
        e = np.array([x["energy_mj_per_layer"] for x in rs])
        t = np.array([x["ms_per_layer"] for x in rs])
        lbl = "governor" if c < 0 else f"{c} MHz"
        print(f"  {lbl:>9s}  {t.mean():7.3f} +/- {t.std(ddof=1) if len(t)>1 else 0:.3f} ms   "
              f"{e.mean():8.1f} +/- {e.std(ddof=1) if len(e)>1 else 0:5.1f} mJ")
    if gov:
        ge = np.mean([x["energy_mj_per_layer"] for x in gov])
        gt = np.mean([x["ms_per_layer"] for x in gov])
        locked = {c: v for c, v in agg.items() if c > 0}
        bc = min(locked, key=lambda c: np.mean([x["energy_mj_per_layer"] for x in locked[c]]))
        be = np.mean([x["energy_mj_per_layer"] for x in locked[bc]])
        bt = np.mean([x["ms_per_layer"] for x in locked[bc]])
        print(f"\n  vs the DEPLOYED point (unlocked governor, {gt:.3f} ms, {ge:.1f} mJ):")
        print(f"    best locked clock {bc} MHz: {be/ge-1:+.1%} energy, {bt/gt-1:+.1%} latency")


if __name__ == "__main__":
    main()
