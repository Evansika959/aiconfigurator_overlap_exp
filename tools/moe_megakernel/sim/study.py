#!/usr/bin/env python3
"""The control groups: what each ingredient of a two-domain design is worth.

  python3 -m sim.study --cells 32          # a 2048-token decode batch
  python3 -m sim.study --cells 0           # the prefill batch

A two-V/f-domain design bundles four changes. Reporting only its end point against the
stock governor would credit the second voltage rail with gains that the kernel, the
pinning or simply a better single clock produced. So each change is added alone:

  B0  vLLM queue kernel, stock governor                    what the box does today
  B1  vLLM queue kernel, the best SINGLE clock             no new kernel, no new hardware
  B2  persistent kernel, unpartitioned, best single clock  the kernel a partition needs
  B3  cold experts pinned to an SM set, one clock          partitions, still one rail
  B4  the same partitions, a clock per domain              the second rail

  B1 - B0   a better clock than the governor's
  B2 - B1   what the persistent kernel costs (one block per SM, as measured)
  B3 - B2   what pinning costs or buys: the partitions overlap instead of queueing
  best - worst split within B3: SM reallocation
  B4 - B3   the second V/f domain, and nothing else

B3* and B4* repeat B3 and B4 with vLLM's per-SM costs: a pinned kernel as efficient as
the queue kernel, which nobody has built. They bound what a better megakernel could do.

Each group is given the best configuration it is allowed. Which experts are cold is
software, for B3 and B4 alike: 8 to 120 of 128, in steps of 8. How many SMs the cold set
gets is NOT alike. B3 pins partitions on today's chip and may use any SM count (12 to
96 in steps of 4). B4's cold set is a V/f domain, which is silicon: whole GPCs (16 to
80), and -- reported as B4@<SMs> -- one size for every workload. Searching B3 only on
B4's grid, as an earlier version did, credited the rail with what a finer split gives
for free.

Every row is the steady state of one repeated layer, from the engine's own predictor.
Two questions are asked of each: the fastest it can run under the power wall, and the
least energy at which it still meets a latency target set relative to B0. The wall is
the power B0 itself settles at -- the stock limiter holds the board a fraction of a
watt above 400 W -- so that no group is held to a stricter limit than the baseline.

A POINT below is (clocks, ms, mJ, W) of one layer at one clock setting.
"""

import argparse
import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .calib import Calib
from .control import Fixed, Governor, TdpBisection
from .engine import Engine, Kernel, Shared
from .files import write_csv
from .hardware import Hardware, two_domain
from .workload import Batch, build

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------- points ----
class Probe:
    """Steady-state cost of one layer under a kernel and a hardware, at any clocks."""

    def __init__(self, cal, hw, kernel, layer, shared):
        self.e = Engine(hw, cal, kernel, [], Fixed(hw.f_max), shared=shared, record=False, limiter=False)
        self.layer, self.hw = layer, hw

    def __call__(self, f):
        return self.e.predict(self.layer, f)


def sweep1(probe, clocks):
    """One clock for every domain, at each of `clocks`."""
    out = []
    for f in clocks:
        r = probe([f] * probe.e.D)
        out.append(((f,), r["ms"], r["mj"], r["w"]))
    return out


def sweep2(probe, coarse, fine):
    """Two clocks: a coarse grid, plus the diagonal on the fine grid so that 'a clock per
    domain' always contains 'one clock' (B4 can then never lose to B3 by gridding).
    Returns the points and a function that adds one more clock pair to them."""
    seen, out = set(), []

    def add(fc, fh):
        if (fc, fh) not in seen:
            seen.add((fc, fh))
            r = probe([fc, fh])
            out.append(((fc, fh), r["ms"], r["mj"], r["w"]))

    for fc in coarse:
        for fh in coarse:
            add(fc, fh)
    for f in fine:
        add(f, f)
    return out, add


def bisected(cal, hw, kernel, layer, shared):
    """The tdp_bisection controller's own answer, as one more candidate: the grid above
    steps the hot clock by 30 MHz and can straddle the wall."""
    ctl = TdpBisection()
    e = Engine(hw, cal, kernel, [Batch("b", "x", 0, [layer])], ctl, shared=shared, record=False, limiter=False).run()
    f = tuple(ctl.decisions[-1]["f"])
    r = e.predict(layer, f)
    return (f, r["ms"], r["mj"], r["w"])


def refine(pts, add, hw, tdp, slo, rounds=2):
    """Walk the best point to its 15 MHz neighbours until none is better."""
    for _ in range(rounds):
        b = best(pts, tdp, slo)
        if b is None:
            return
        fc, fh = b[0]
        for dc in (-hw.f_step, 0, hw.f_step):
            for dh in (-hw.f_step, 0, hw.f_step):
                c, h = fc + dc, fh + dh
                if hw.f_min <= c <= hw.f_max and hw.f_min <= h <= hw.f_max:
                    add(c, h)


def governor_point(cal, layer_spec, shared, model):
    """B0 through the full engine: the governor and the limiter settle by themselves."""
    b = build([dict(layer_spec, repeat=1)], model)
    one = (
        Engine(Hardware(), cal, Kernel("queue", 128), b, Fixed(1290), shared=shared, record=False, limiter=False)
        .run()
        .layers[0]["ms"]
    )
    b = build([dict(layer_spec, repeat=int(6000 / one))], model)
    e = Engine(Hardware(), cal, Kernel("queue", 128), b, Governor(), shared=shared).run()
    tail = [x for x in e.layers if x["t_start"] > e.t - 3000]
    ms = float(np.mean([x["ms"] for x in tail]))
    mj = float(np.mean([x["mj"] for x in tail]))
    seg = [(t1 - t0, f[0]) for t0, t1, _, f, _ in e.trace if t0 > e.t - 3000]
    clk = sum(d * f for d, f in seg) / sum(d for d, _ in seg)
    return ((round(clk),), ms, mj, mj / ms)


# ------------------------------------------------------------------- selection ----
def _fastest(p):
    return (round(p[1], 6), p[2])  # ties in time go to the lower energy


def _least_energy(p):
    return p[2]


def best(points, tdp, slo=None):
    """Fastest under TDP, or (with `slo`, ms) the least energy within that latency."""
    ok = [p for p in points if p[3] <= tdp and (slo is None or p[1] <= slo)]
    return min(ok, key=_fastest if slo is None else _least_energy) if ok else None


def front(points, tdp):
    """Pareto front in (time, energy) of the points under TDP: for each layer time, the
    least energy any clock setting reaches at or under it."""
    ok = sorted((p for p in points if p[3] <= tdp), key=lambda p: (p[1], p[2]))
    out, low = [], float("inf")
    for p in ok:
        if p[2] < low - 1e-9:
            out.append(p)
            low = p[2]
    return out


# -------------------------------------------------------------- control groups ----
@dataclass
class Group:
    """One control group: its points, for one configuration (`points`) or for each of
    the configurations it may choose from (`per`: {configuration: points})."""

    name: str
    what: str
    points: Optional[list] = None
    per: Optional[dict] = None

    def all_points(self):
        return self.points if self.points is not None else [p for v in self.per.values() for p in v]

    def pick(self, tdp, slo):
        """(point, configuration, worst configuration and its point) for one objective."""
        if self.points is not None:
            return best(self.points, tdp, slo), None, None
        cand = {k: best(v, tdp, slo) for k, v in self.per.items()}
        cand = {k: v for k, v in cand.items() if v}
        if not cand:
            return None, None, None
        key = lambda kv: (_fastest if slo is None else _least_energy)(kv[1])
        k, p = min(cand.items(), key=key)
        return p, k, (max(cand.items(), key=key) if len(cand) > 1 else None)


def describe(config):
    """A configuration key as text: (tile size,) or (tile size, cold experts, cold SMs)."""
    if len(config) == 1:
        return f"BM{config[0]}"
    return f"BM{config[0]}, {config[1]} cold experts on {config[2]} SMs"


N_COLDS = tuple(range(8, 121, 8))  # cold experts: software, any boundary
PIN_SMS = tuple(range(12, 97, 4))  # B3: SMs of the cold partition, any count
DOMAIN_SMS = (16, 32, 48, 64, 80)  # B4: SMs of the cold V/f domain, whole GPCs
SLOS = (1.0, 1.02, 1.05, 1.10, 1.25)  # latency targets, as multiples of B0's


def control_groups(cal, L, shared, b0, cap, bms, n_colds, pin_sms, domain_sms, slos, step, f_lo):
    """B0-B4 and B3*, B4* for layer L, each with every point it may choose from, and
    B4@<SMs>: B4 on one fixed domain size."""
    one = Hardware(tdp_w=cap)
    fine = [f for f in one.clocks() if f >= 510]
    coarse = [one.snap(f) for f in range(one.f_max, f_lo - 1, -step)]
    targets = [None] + [s * b0[1] for s in slos]
    groups = [
        Group("B0", "queue kernel, stock governor", points=[b0]),
        Group("B1", "queue kernel, one clock", points=sweep1(Probe(cal, one, Kernel("queue", 128), L, shared), fine)),
        Group(
            "B2",
            "persistent kernel, one clock",
            per={(bm,): sweep1(Probe(cal, one, Kernel("persistent", bm), L, shared), fine) for bm in bms},
        ),
    ]
    for star, ktype, kbms in (("", "persistent", bms), ("*", "queue", (128,))):
        per3, per4 = {}, {}
        for kbm in kbms:
            for nc in n_colds:
                for s in pin_sms:
                    per3[(kbm, nc, s)] = sweep1(Probe(cal, one, Kernel(ktype, kbm, nc, s), L, shared), fine)
                for s in domain_sms:
                    hw2 = two_domain(s, tdp_w=cap)
                    pts, add = sweep2(Probe(cal, hw2, Kernel(ktype, kbm, nc), L, shared), coarse, fine)
                    pts.append(bisected(cal, hw2, Kernel(ktype, kbm, nc), L, shared))
                    for slo in targets:
                        refine(pts, add, hw2, cap, slo)
                    per4[(kbm, nc, s)] = pts
        tail = " (vLLM per-SM costs)" if star else ""
        groups.append(Group("B3" + star, "pinned, one clock" + tail, per=per3))
        groups.append(Group("B4" + star, "pinned, a clock per domain" + tail, per=per4))
        for s in domain_sms:
            groups.append(
                Group(
                    f"B4{star}@{s}",
                    f"... on a {s}-SM cold domain" + tail,
                    per={k: v for k, v in per4.items() if k[2] == s},
                )
            )
    return groups


def tabulate(groups, objectives, tdp):
    """One row per (objective, control group): its best point and configuration."""
    table = []
    for oname, slo in objectives:
        for g in groups:
            if g.name == "B0":  # the governor is one point
                p, cfg, worst = g.points[0], None, None
                if slo is not None and p[1] > slo * (1 + 1e-9):
                    p = None
            else:
                p, cfg, worst = g.pick(tdp, slo)
            table.append(
                dict(
                    objective=oname,
                    name=g.name,
                    what=g.what,
                    config=describe(cfg) if cfg else "",
                    f="/".join(str(x) for x in p[0]) if p else "",
                    ms=p[1] if p else None,
                    mj=p[2] if p else None,
                    w=p[3] if p else None,
                    worst_config=describe(worst[0]) if worst else "",
                    worst_ms=worst[1][1] if worst else None,
                    worst_mj=worst[1][2] if worst else None,
                )
            )
    return table


def show(table, objectives, header):
    print(header)
    ref = next(t for t in table if t["name"] == "B0")
    for oname, _ in objectives:
        print(f"\n  {oname}")
        print(
            f"  {'':7s} {'':44s} {'clock MHz':>10} {'ms':>8} {'W':>7} {'mJ':>8} "
            f"{'ms vs B0':>9} {'mJ vs B0':>9}  best configuration"
        )
        for t in [x for x in table if x["objective"] == oname]:
            if t["ms"] is None:
                print(f"  {t['name']:7s} {t['what']:44s} {'cannot meet it':>10}")
                continue
            print(
                f"  {t['name']:7s} {t['what']:44s} {t['f']:>10} {t['ms']:>8.4f} "
                f"{t['w']:>7.1f} {t['mj']:>8.1f} {(t['ms'] / ref['ms'] - 1) * 100:>+8.1f}% "
                f"{(t['mj'] / ref['mj'] - 1) * 100:>+8.1f}%  {t['config']}"
            )


def run(
    cells=32,
    block_m=None,
    model="qwen",
    tdp=400.0,
    step=30,
    f_lo=900,
    n_colds=N_COLDS,
    pin_sms=PIN_SMS,
    domain_sms=DOMAIN_SMS,
    slos=SLOS,
    cal=None,
    layer=0,
    rep=0,
    quiet=False,
):
    """(table, fronts): the control groups under each objective, and each group's
    Pareto front in (time, energy).

    `f_lo` bounds the two-clock grid from below: a layer at 900 MHz takes 1.3 times as
    long as B0's, so no target in `slos` can be met from under it."""
    cal = cal or Calib.load()
    shared = Shared()
    phase = "prefill" if cells == 0 else "decode"
    bms = block_m or ((128,) if cells == 0 else (64, 128))
    spec = dict(phase=phase, cells=cells, rep=rep, layers=[layer])
    L = build([spec], model)[0].layers[0]
    t0 = time.time()
    b0 = governor_point(cal, spec, shared, model)
    cap = max(tdp, b0[3])  # the wall as the baseline itself meets it
    groups = control_groups(cal, L, shared, b0, cap, bms, n_colds, pin_sms, domain_sms, slos, step, f_lo)
    objectives = [("fastest under TDP", None)] + [(f"least energy, <= {s:.2f} x B0 latency", s * b0[1]) for s in slos]
    table = tabulate(groups, objectives, cap)
    if not quiet:
        show(
            table,
            objectives,
            f"{phase} batch, {L.pairs // 8} tokens, layer {layer}; the wall: {cap:.1f} W; {time.time() - t0:.0f} s",
        )
    fronts = [
        dict(name=g.name, f="/".join(str(x) for x in p[0]), ms=round(p[1], 6), mj=round(p[2], 4), w=round(p[3], 3))
        for g in groups
        if "@" not in g.name
        for p in (g.all_points() if g.name == "B0" else front(g.all_points(), cap))
    ]
    return table, fronts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=int, default=32, help="decode cells of 64 tokens; 0 = prefill")
    ap.add_argument("--block-m", default=None, help="tile sizes the persistent kernel may use, e.g. 64,128")
    ap.add_argument("--tdp", type=float, default=400.0)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    bms = tuple(int(v) for v in a.block_m.split(",")) if a.block_m else None
    table, fronts = run(a.cells, bms, tdp=a.tdp, layer=a.layer)
    out = a.out or os.path.join(HERE, "out", f"study_{'prefill' if a.cells == 0 else 'decode' + str(a.cells * 64)}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    write_csv(out, table)
    write_csv(out[:-4] + "_fronts.csv", fronts)
    print(f"\nwrote {out} and {out[:-4]}_fronts.csv")


if __name__ == "__main__":
    main()
