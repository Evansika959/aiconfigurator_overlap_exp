#!/usr/bin/env python3
"""The plans of the held-out traces, drawn from a fixed seed BEFORE anything was measured.

Every earlier governor trace was looked at while the simulator's limiter was being built,
so none of them is a clean test any more. These two are: the simulator and its
calibration are frozen first (HELDOUT.md lists the checksums), the plans come out of this
script and not out of anybody's head, and each is run on the A100 and replayed through
the simulator exactly once.

What is new in them, against every sweep and trace so far:

  batch sizes   640, 1280, 1792, 2560, 3584, 5120 and 7168 decode tokens: none was ever run
  routing       decode batches pooled from traced cells 96-191, which no fit ever used
                (the 7168-token batch needs 112 cells and takes 16 of them, 80-95, from
                the fitted range). Prefill from prompt sets nothing has used (0, 2, 4, 7,
                11 and 13 are taken)
  layer         plan A runs MoE layer 17, which no sweep ever ran on its own
  order         segment order, lengths (0.6-3 s) and idle gaps (0.25-4.5 s, on both sides
                of the 2 s after which the governor parks the clock) are random

  python3 heldout_plan.py            # prints both plans, writes data/heldout_plan.json
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20261001
DECODE = ["d10@182", "d20@160", "d28@130", "d40@150", "d56@132", "d80@112", "d112@80"]
PROMPTS_USED = {0, 2, 4, 7, 11, 13}
PROMPTS = [p for p in range(24) if p not in PROMPTS_USED]


def draw(rng, n_busy, n_prefill, busy_s, idle_s, p_idle=0.35):
    """`n_busy` load segments, `n_prefill` of them prefill, never the same load twice in
    a row; after each, an idle gap with probability `p_idle`. Durations are log-uniform."""
    logu = lambda lo, hi: float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    kinds = ["p"] * n_prefill + ["d"] * (n_busy - n_prefill)
    rng.shuffle(kinds)
    prompts = [int(p) for p in rng.choice(PROMPTS, size=n_prefill, replace=False)]
    plan, last = [("idle", 3.0)], None                  # start from a parked clock
    for kind in kinds:
        if kind == "p":
            name = f"p{prompts.pop()}"
        else:
            name = last
            while name == last:
                name = DECODE[int(rng.integers(len(DECODE)))]
        plan.append((name, round(logu(*busy_s), 1)))
        last = name
        if rng.random() < p_idle:
            plan.append(("idle", round(logu(*idle_s), 2)))
    if plan[-1][0] != "idle":
        plan.append(("idle", 1.0))
    return ",".join(f"{n}:{s:g}" for n, s in plan)


def main():
    rng = np.random.default_rng(SEED)
    plans = {
        # one MoE layer, four times per graph replay: 3-16 ms a replay, fine enough to
        # see the limiter move the clock
        "heldout1": dict(plan=draw(rng, 14, 4, (0.6, 3.0), (0.25, 4.5)), layer=17, layers=None),
        # whole forward passes over all 48 MoE layers: 50-200 ms a replay
        "heldout2": dict(plan=draw(rng, 8, 3, (2.5, 4.0), (0.5, 3.0)), layer=0, layers="all"),
    }
    for tag, p in plans.items():
        secs = sum(float(s.split(":")[1]) for s in p["plan"].split(","))
        print(f"{tag}  ({secs:.0f} s a run, " + ("all 48 MoE layers" if p["layers"] else f"MoE layer {p['layer']}") + ")")
        print("  " + p["plan"])
    with open(os.path.join(HERE, "data", "heldout_plan.json"), "w") as fh:
        json.dump(dict(seed=SEED, plans=plans), fh, indent=1)
    print("wrote data/heldout_plan.json")


if __name__ == "__main__":
    main()
