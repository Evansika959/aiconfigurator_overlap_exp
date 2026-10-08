"""Choose per-GPU clocks for a training step with the node engine as the predictor.

  python3 -m sim.node_policy [--budget 0.01 0.03] [--ctas 16]

For a step given as per-GPU programs, three policies are compared, each predicted by
the engine and (after `sweep_spans.py` has run the plan this writes) measured:

  A        no clock lock: the stock governor, the user's baseline
  one      the best single clock for every GPU within the time budget
  per_gpu  a clock for the GPU on the critical path and another for the rest

"Within the budget": step time at most (1 + budget) times A's. "Best": least node
energy per step. The search is exhaustive over a clock grid for two groups of GPUs
(the heaviest program's GPUs, and the others), which is what a controller that knows
only "who is the straggler" could set.
"""

import argparse
import itertools
import json
import os
import sys

from . import node
from .calib import Calib
from .fit_kernels import NEW
from .kernels import Kernels

GRID = [510, 600, 705, 810, 900, 975, 1050, 1125, 1200, 1260, 1305, 1350, 1410]


def predict(k, cal, ops, ctas, clock):
    c = node.program_cell(k, cal, ops, ctas=ctas, clock=clock)
    return c["t_loop"], float(sum(c["power"])), c


def search(k, cal, ops, ctas, budgets, heavy=(0,), verbose=True):
    """ops: one program per GPU. Returns dict(A=..., one={budget: ...}, per_gpu={budget: ...})."""
    world = len(ops)
    tA, pA, cA = predict(k, cal, ops, ctas, 0)
    out = dict(A=dict(clock=0, t=tA, w=pA, j=tA * pA / 1e3, clocks=cA["clock_mean"]), one={}, per_gpu={})
    table = {}
    for fh, fl in itertools.product(GRID, GRID):
        clock = [fh if g in heavy else fl for g in range(world)]
        t, p, c = predict(k, cal, ops, ctas, clock)
        table[(fh, fl)] = (t, p, c["limited"])
    for b in budgets:
        ok = {key: v for key, v in table.items() if v[0] <= tA * (1 + b)}
        for name, sel in (("one", {key: v for key, v in ok.items() if key[0] == key[1]}), ("per_gpu", ok)):
            if not sel:
                continue
            (fh, fl), (t, p, _) = min(sel.items(), key=lambda kv: kv[1][0] * kv[1][1])
            out[name][b] = dict(clock=[fh if g in heavy else fl for g in range(world)], t=t, w=p, j=t * p / 1e3)
    if verbose:
        print(f"  A (no lock): step {tA:.2f} ms, node {pA:.0f} W, {tA * pA / 1e3:.2f} J")
        for b in budgets:
            for name in ("one", "per_gpu"):
                r = out[name].get(b)
                if r:
                    print(
                        f"  budget +{b * 100:.0f}%  {name:8s} {r['clock']!s:26s} step {r['t']:.2f} ms "
                        f"({(r['t'] / tA - 1) * 100:+.1f}%)  "
                        f"node {r['w']:.0f} W  energy {r['j']:.2f} J ({(r['j'] / out['A']['j'] - 1) * 100:+.1f}%)"
                    )
    return out


def steps():
    """The steps the policy is tried on: name -> per-GPU ops. Unbalanced by 4:1 and 2:1,
    and balanced (where the two policies should coincide)."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "overlap_exp"))
    import plan_spans as ps

    return {
        "ddp_overlap 8192 + 3 x 2048": [ps.step("ddp_overlap", 8192)] + [ps.step("ddp_overlap", 2048)] * 3,
        "ddp_overlap 8192 + 3 x 4096": [ps.step("ddp_overlap", 8192)] + [ps.step("ddp_overlap", 4096)] * 3,
        "ddp_overlap 4 x 4096": [ps.step("ddp_overlap", 4096)] * 4,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budget", type=float, nargs="+", default=[0.01, 0.03])
    ap.add_argument("--ctas", type=int, default=16)
    ap.add_argument("--plan", default=os.path.join(NEW, "plan_policy.json"))
    a = ap.parse_args()
    k, cal = Kernels.load(), Calib.load()
    runs, report = [], {}
    for name, ops in steps().items():
        print(name)
        r = search(k, cal, ops, a.ctas, a.budget)
        report[name] = r
        tag = name.replace(" ", "_")
        runs.append(dict(mode="program", name=f"{tag}|A", clock=0, ops=ops))
        seen = set()
        for pol in ("one", "per_gpu"):
            for b, v in r[pol].items():
                key = tuple(v["clock"])
                if key not in seen:
                    seen.add(key)
                    runs.append(dict(mode="program", name=f"{tag}|{pol}+{b * 100:.0f}%", clock=v["clock"], ops=ops))
    with open(a.plan, "w") as fh:
        json.dump(dict(world=4, ctas=[a.ctas], runs=runs), fh)
    with open(a.plan.replace("plan_", "pred_"), "w") as fh:
        json.dump(report, fh, indent=1, default=float)
    print(f"plan with {len(runs)} cells -> {a.plan}")


if __name__ == "__main__":
    main()
