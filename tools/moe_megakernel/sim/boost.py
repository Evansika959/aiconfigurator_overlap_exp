#!/usr/bin/env python3
"""What is it worth to boost the domain still running when the other one finishes?

  python3 -m sim.boost --cells 64 --kernel queue      # a 4096-token decode batch

A layer's two partitions rarely finish a GEMM at the same instant, and the layer waits
at a barrier for the slower one. The moment the faster one finishes is an event; from
then until the barrier the board draws less, and the domain still running could be
raised. `control.BoostOnFinish` does exactly that. This asks what it buys, for one
repeated layer on a two-domain chip, with three policies each at its best clocks:

  one clock      both domains at f
  per domain     (f_cold, f_hot), fixed for the layer
  + boost        the same, and a survivor's domain goes to f_boost until the barrier

and under two readings of the power wall:

  mean      the layer's mean power <= TDP. This is what the A100's limiter enforces: it
            reads 20 ms means, and a layer lasts 1.5-5 ms.
  instant   every phase of the layer <= TDP: a limiter faster than a layer.

The comparison that matters is boost against PER DOMAIN, not against one clock: fixed
per-domain clocks already turn the early finisher's slack into a lower clock. The search
runs with no actuation delay and no limiter, so it is a bound; the last table puts the
delay back.

A POINT is one clock setting of one configuration: its clocks, the layer's time, its
mean and peak power, and the share of GEMM time one domain runs alone.
"""

import argparse
import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .calib import Calib
from .control import BoostOnFinish, Fixed
from .engine import Engine, Kernel, Shared
from .files import write_csv
from .hardware import two_domain
from .workload import build

HERE = os.path.dirname(os.path.abspath(__file__))
N_COLDS = (16, 32, 48, 64, 80, 96, 112)  # cold experts: software, free to choose
DOMAIN_SMS = (16, 32, 48)  # SMs of the cold V/f domain: silicon, whole GPCs
DELAYS_MS = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 38.0)  # 38 ms: NVML on the A100
WALLS = ("mean", "instant")
# The clock grid of the search, MHz. 510 is the lowest clock the rate model was fitted at;
# a cold domain that finishes early is often best run that slow, so the grid must reach it.
F_LO, COARSE, FINE = 510, 30, 15


@dataclass
class Point:
    f: tuple  # (cold, hot) clocks
    boost: Optional[int]
    ms: float
    w: float  # mean board power over the layer
    peak: float  # highest power of any phase
    alone: float  # share of GEMM time with one domain running
    survivor_f: int  # lowest clock among the domains that are ever left running alone

    def under(self, wall, tdp):
        return (self.w if wall == "mean" else self.peak) <= tdp

    def key(self):
        return (round(self.ms, 7), self.w)  # ties in time go to the lower power

    def clocks(self):
        return f"{self.f[0]}/{self.f[1]}" + (f"->{self.boost}" if self.boost else "")


class Config:
    """One (cold experts, cold-domain SMs) configuration of one layer: every clock
    setting it is asked for, simulated once."""

    def __init__(self, cal, shared, batches, kernel, cold_sms, actuation_ms=0.0):
        self.cal, self.shared, self.batches, self.kernel = cal, shared, batches, kernel
        self.hw = two_domain(cold_sms, actuation_ms=actuation_ms)
        self.memo = {}

    def run(self, f, boost=None):
        ctl = BoostOnFinish(list(f), boost) if boost else Fixed(list(f))
        return Engine(self.hw, self.cal, self.kernel, self.batches, ctl, shared=self.shared, limiter=False).run(), ctl

    def point(self, f, boost=None):
        key = (tuple(f), boost)
        if key not in self.memo:
            e, _ = self.run(f, boost)
            L = e.layers[-1]
            phases = [(b - a, p, run) for a, b, p, _, run in e.trace if a >= L["t_start"] - 1e-12 and b > a]
            gemm = [(d, run) for d, _, run in phases if "epilogue" not in run]
            alone = [(d, run) for d, run in gemm if "+" not in run]
            left = [f[self.hw.index(run.split(":")[0])] for _, run in alone]
            self.memo[key] = Point(
                tuple(f),
                boost,
                L["ms"],
                L["mj"] / L["ms"],
                max(p for _, p, _ in phases),
                sum(d for d, _ in alone) / sum(d for d, _ in gemm),
                min(left, default=self.hw.f_max),
            )
        return self.memo[key]


def search(cfg, wall, tdp):
    """The best one-clock, per-domain and boosted Points of a configuration under a wall.
    'Per domain' contains 'one clock' and 'boosted' contains 'per domain', so a policy
    never loses to the one it extends by gridding."""
    hw = cfg.hw
    ok = lambda p: p.under(wall, tdp)
    pick = lambda pts: min((p for p in pts if ok(p)), key=Point.key, default=None)
    coarse = [hw.snap(f) for f in range(hw.f_max, F_LO - 1, -COARSE)]
    one = pick(cfg.point((f, f)) for f in hw.clocks() if f >= F_LO)
    static = [cfg.point((fc, fh)) for fc in coarse for fh in coarse]
    # a boost at or under the survivor's own clock changes nothing
    boosted = [cfg.point(p.f, fb) for p in static for fb in coarse if fb > p.survivor_f]
    two = pick([*static, one])
    bst = pick([*boosted, two])
    near = lambda f: [x for x in (f - FINE, f, f + FINE) if F_LO <= x <= hw.f_max]
    for _ in range(3):  # walk each to its 15 MHz neighbours until none is better
        two = pick([two] + [cfg.point((c, h)) for c in near(two.f[0]) for h in near(two.f[1])])
        cand = [bst, two]
        for c in near(bst.f[0]):
            for h in near(bst.f[1]):
                base = cfg.point((c, h))
                top = bst.boost or base.survivor_f + FINE
                cand += [cfg.point((c, h), b) for b in near(top) if b > base.survivor_f]
        bst = pick(cand)
    return one, two, bst


def delay_sweep(cal, shared, spec, kernel, cold_sms, best, ref_ms, delays=DELAYS_MS, n=60):
    """The boosted setting `best` with the actuation delay put back: (delay, ms, mean W,
    peak W, boosts issued per layer). The same clocks at every delay: what a boost
    planned for an instant actuator does on a slower one."""
    batches = build([dict(spec, repeat=n)])
    out = []
    for d in delays:
        cfg = Config(cal, shared, batches, kernel, cold_sms, actuation_ms=d)
        e, ctl = cfg.run(best.f, best.boost)
        L = e.layers[n // 3 :]
        t0 = L[0]["t_start"]
        out.append(
            dict(
                delay_ms=d,
                ms=float(np.mean([x["ms"] for x in L])),
                w=sum(x["mj"] for x in L) / sum(x["ms"] for x in L),
                peak_w=max(p for a, b, p, _, _ in e.trace if a >= t0 and b > a),
                boosts_per_layer=ctl.boosts / len(e.layers),
            )
        )
    for r in out:
        r["vs_per_domain_pct"] = (r["ms"] / ref_ms - 1) * 100
    return out


def run(cells=64, ktype="queue", n_colds=N_COLDS, domain_sms=DOMAIN_SMS, tdp=400.0, cal=None, quiet=False):
    """(rows, summary, delays): one row per (wall, configuration, policy); per wall the
    best configuration of each policy and the configuration where the boost gains most;
    and the delay sweep at that last one under the instantaneous wall."""
    cal = cal or Calib.load()
    shared = Shared()
    spec = dict(phase="prefill" if cells == 0 else "decode", cells=cells, rep=0, layers=[0])
    batches = build([dict(spec, repeat=2)])
    bm = 128 if (cells == 0 or ktype == "queue") else 64
    name = f"{spec['phase']}{'' if cells == 0 else ' ' + str(cells * 64)}, {ktype} per-SM costs"
    t0 = time.time()
    found = {}  # (n_cold, cold SMs) -> {wall: (one, two, boosted)}
    for sms in domain_sms:
        for nc in n_colds:
            cfg = Config(cal, shared, batches, Kernel(ktype, bm, nc), sms)
            found[(nc, sms)] = {wall: search(cfg, wall, tdp) for wall in WALLS}
    rows, summary, delays = [], {}, []
    for wall in WALLS:
        res = {conf: v[wall] for conf, v in found.items()}
        for (nc, sms), pts in res.items():
            for policy, p in zip(("one clock", "per domain", "boost"), pts, strict=True):
                rows.append(
                    dict(
                        wall=wall,
                        n_cold=nc,
                        cold_sms=sms,
                        policy=policy,
                        clocks=p.clocks(),
                        ms=round(p.ms, 5),
                        w=round(p.w, 2),
                        peak_w=round(p.peak, 2),
                        alone_pct=round(p.alone * 100, 2),
                    )
                )
        best = [min(res.items(), key=lambda kv, i=i: kv[1][i].key()) for i in range(3)]
        gain = lambda kv: kv[1][2].ms / kv[1][1].ms
        most = min(res.items(), key=gain)
        summary[wall] = dict(
            best_one=best[0][1][0],
            best_two=best[1][1][1],
            best_boost=best[2][1][2],
            best_conf=[b[0] for b in best],
            most_conf=most[0],
            most_two=most[1][1],
            most_boost=most[1][2],
        )
        if not quiet:
            show(name, wall, res, summary[wall], tdp)
    s = summary["instant"]
    if s["most_boost"].boost:
        nc, sms = s["most_conf"]
        delays = delay_sweep(cal, shared, spec, Kernel(ktype, bm, nc), sms, s["most_boost"], s["most_two"].ms)
        if not quiet:
            print(
                f"\n  the actuation delay put back: {nc} cold experts on {sms} SMs at {s['most_boost'].clocks()}; "
                f"without the boost {s['most_two'].ms:.4f} ms"
            )
            for r in delays:
                print(
                    f"    {r['delay_ms']:6.2f} ms: {r['ms']:.4f} ms ({r['vs_per_domain_pct']:+.2f}%), mean "
                    f"{r['w']:.1f} W, peak {r['peak_w']:.1f} W, {r['boosts_per_layer']:.1f} boosts per layer"
                )
    if not quiet:
        print(f"[{name}: {time.time() - t0:.0f} s]")
    return rows, summary, delays


def show(name, wall, res, s, tdp):
    cell = lambda p: f"{p.clocks():>15} {p.ms:8.4f} {p.w:6.1f} {p.peak:6.1f} {p.alone * 100:5.1f}%"
    print(f"\n{name}; wall: {wall} power <= {tdp:.0f} W")
    print("  per policy: clocks cold/hot(->boost), ms, mean W, peak W, share of GEMM time one domain runs alone")
    print(
        f"  {'cold experts : SMs':18s} | {'one clock':46s} | {'a clock per domain':46s} | "
        f"{'+ boost the survivor':46s} | boost vs per domain"
    )
    for (nc, sms), (one, two, bst) in res.items():
        gain = (bst.ms / two.ms - 1) * 100
        print(f"  {nc:>4d} : {sms:<11d} | {cell(one)} | {cell(two)} | {cell(bst)} | {gain:+6.2f}% ms")
    conf = lambda c: f"{c[0]} cold experts on {c[1]} SMs"
    one, two, bst = s["best_one"], s["best_two"], s["best_boost"]
    print(
        f"  each policy at its best configuration: one clock {one.ms:.4f} ms ({conf(s['best_conf'][0])}) | per domain "
        f"{two.ms:.4f} ms ({conf(s['best_conf'][1])}), {(two.ms / one.ms - 1) * 100:+.2f}% | + boost {bst.ms:.4f} ms "
        f"({conf(s['best_conf'][2])}), {(bst.ms / two.ms - 1) * 100:+.2f}% more"
    )
    m2, mb = s["most_two"], s["most_boost"]
    print(
        f"  where the boost gains most: {conf(s['most_conf'])}, {(mb.ms / m2.ms - 1) * 100:+.2f}% "
        f"({m2.alone * 100:.0f}% of GEMM time alone); that configuration is {(mb.ms / bst.ms - 1) * 100:+.1f}% "
        f"against the best one"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=int, default=64, help="decode cells of 64 tokens; 0 = prefill")
    ap.add_argument("--kernel", default="queue", choices=["persistent", "queue"])
    a = ap.parse_args()
    rows, _, delays = run(a.cells, a.kernel)
    stem = os.path.join(HERE, "out", f"boost_{'prefill' if a.cells == 0 else 'decode' + str(a.cells * 64)}_{a.kernel}")
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    write_csv(stem + ".csv", rows)
    if delays:
        write_csv(stem + "_delay.csv", delays)
    print(f"wrote {stem}.csv")


if __name__ == "__main__":
    main()
