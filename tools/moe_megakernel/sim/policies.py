#!/usr/bin/env python3
"""One load, several policies: the simulator used for what it was built for.

  python3 -m sim.policies --tag trace2

Takes a scripted load that was actually run on the A100 (governor_trace.py; the same
batches, the same number of layers per segment, the same idle gaps) and runs it under
each policy. The first row is the configuration that was measured, so its error is known
(`sim.validate_trace`); every other row is the same load under a different controller,
kernel or hardware.

  busy s     time the GPU spends in MoE layers: the work is the same in every row, so
             this is how fast the policy did it
  busy J     energy drawn during those layers
  peak W     highest 100 ms board-power reading
  over ms    time the readings spent above the limiter's peak (1.05 x TDP)
  set-points clock changes that reached the chip

Rows marked * use vLLM's per-SM costs for a pinned kernel: a kernel nobody has built.
The domains are fixed at 32 + 76 SMs, as silicon would be. "64 cold experts" keeps the
same experts cold for every batch; "rebalanced" moves experts across the boundary per
batch so the two partitions finish together -- software, and free.
"""

import argparse
import os
from dataclasses import dataclass, field

from .calib import Calib
from .control import make
from .engine import Engine, Kernel, Shared
from .files import read_csv, read_json, write_csv
from .hardware import Hardware, two_domain
from .validate_trace import load, spec_of
from .workload import build

HERE = os.path.dirname(os.path.abspath(__file__))
COLD_SMS = 32  # the cold domain, where there is one: two GPCs


@dataclass
class Policy:
    """One row: a kernel and a controller on a hardware. To try another policy, add one
    to POLICIES."""

    name: str
    kernel: Kernel
    controller: dict  # a controller spec, as in a scenario file
    two_domains: bool = False  # a cold domain of COLD_SMS SMs with its own rail
    hardware: dict = field(default_factory=dict)  # further Hardware arguments

    def hw(self):
        return two_domain(COLD_SMS, **self.hardware) if self.two_domains else Hardware(**self.hardware)


QUEUE = Kernel("queue", 128)
FIXED_COLD = dict(type="queue", block_m=128, n_cold=64)  # the same 64 experts, always
REBALANCED = dict(type="queue", block_m=128, n_cold=-1)  # chosen per batch
BISECT, GOVERNOR = {"type": "tdp_bisection"}, {"type": "governor"}
POLICIES = [
    Policy("queue kernel, stock governor (measured)", QUEUE, GOVERNOR),
    Policy("queue kernel, tdp_bisection", QUEUE, BISECT),
    Policy("queue kernel, fixed 1050 MHz", QUEUE, {"type": "fixed", "f": 1050}),
    Policy("pinned*, 64 cold experts, one clock", Kernel(cold_sms=COLD_SMS, **FIXED_COLD), BISECT),
    Policy("pinned*, 64 cold experts, two domains", Kernel(**FIXED_COLD), BISECT, True),
    Policy("pinned*, experts rebalanced per batch, one clock", Kernel(cold_sms=COLD_SMS, **REBALANCED), BISECT),
    Policy("pinned*, rebalanced, two domains, governor", Kernel(**REBALANCED), GOVERNOR, True),
    Policy("pinned*, rebalanced, two domains, tdp_bisection", Kernel(**REBALANCED), BISECT, True),
    Policy(
        "pinned*, rebalanced, two domains, 0.05 ms actuation",
        Kernel(**REBALANCED),
        BISECT,
        True,
        dict(actuation_ms=0.05),
    ),
    Policy("pinned*, rebalanced, two domains, closed_loop", Kernel(**REBALANCED), {"type": "closed_loop"}, True),
    Policy("pinned, one block per SM, rebalanced, two domains", Kernel("persistent", 128, -1), BISECT, True),
]


def run(tag="trace2", run_index=0, model="qwen", cal=None, quiet=False):
    """Every policy on run `run_index` of the measured load `tag`: one dict per policy."""
    cal = cal or Calib.load()
    seg, _, _ = load(tag, model)
    spec = spec_of(seg[run_index])
    shared, rows = Shared(), []
    for pol in POLICIES:
        hw = pol.hw()
        e = Engine(hw, cal, pol.kernel, build(spec, model), make(pol.controller), model=model, shared=shared).run()
        busy = sum(L["ms"] for L in e.layers)
        peak = hw.limiter.k_peak * hw.tdp_w
        rows.append(
            dict(
                policy=pol.name,
                layers=len(e.layers),
                busy_s=busy / 1e3,
                busy_j=sum(L["mj"] for L in e.layers) / 1e3,
                mean_w=sum(L["mj"] for L in e.layers) / busy,
                peak_w=max(x["p_sensor_w"] for x in e.ticks),
                over_ms=sum(hw.limiter.period_ms for x in e.ticks if x["p_sensor_w"] > peak),
                setpoints=sum(1 for x in e.events if x["event"] == "setpoint"),
            )
        )
    if not quiet:
        ref = rows[0]
        print(
            f"the `{tag}` load: {ref['layers']} MoE layers in "
            f"{sum(1 for s in seg[run_index] if s['load'] != 'idle')} segments"
        )
        print(
            f"  {'policy':54s} {'busy s':>7} {'vs row 1':>8} {'busy J':>8} {'vs row 1':>8} "
            f"{'mean W':>7} {'peak W':>7} {'over ms':>7} {'set-points':>10}"
        )
        for r in rows:
            print(
                f"  {r['policy']:54s} {r['busy_s']:>7.2f} {(r['busy_s'] / ref['busy_s'] - 1) * 100:>+7.1f}% "
                f"{r['busy_j']:>8.0f} {(r['busy_j'] / ref['busy_j'] - 1) * 100:>+7.1f}% "
                f"{r['mean_w']:>7.1f} {r['peak_w']:>7.1f} {r['over_ms']:>7.0f} {r['setpoints']:>10d}"
            )
    return rows


def check_fixed(tag="trace2", clocks=(1050, 1290), quiet=False):
    """The one policy change that CAN be made on the real chip: the same plan was run
    with the clock locked (governor_trace.py --lock). Compare what a fixed clock did to
    the load's busy time and busy energy, measured, with what the simulator says it
    does. Needs sim/out/trace_<tag>.json and trace_lock<f>.json (sim.validate_trace)."""
    path = lambda t: os.path.join(HERE, "out", f"trace_{t}.json")
    gov = read_json(path(tag))["runs"][0]["segments"]
    busy = [
        r
        for r in read_csv(os.path.join(os.path.dirname(HERE), "data", f"governor_trace_{tag}_segments_qwen.csv"))
        if r["run"] == "0" and r["load"] != "idle"
    ]
    n = {s["seg"]: int(r["layers"]) for s, r in zip(gov, busy, strict=True)}
    out = []
    for f in clocks:
        if not os.path.exists(path(f"lock{f}")):
            continue
        lock = {s["seg"]: s for s in read_json(path(f"lock{f}"))["runs"][0]["segments"]}
        tot = lambda segs, k: (
            sum(n[g["seg"]] * segs(g)[k + "_ms"] for g in gov) / 1e3,
            sum(n[g["seg"]] * segs(g)[k + "_ms"] * segs(g)[k + "_w"] for g in gov) / 1e6,
        )
        (tg, eg), (tl, el) = tot(lambda g: g, "meas"), tot(lambda g: lock[g["seg"]], "meas")
        (sg, xg), (sl, xl) = tot(lambda g: g, "sim"), tot(lambda g: lock[g["seg"]], "sim")
        out.append(
            dict(
                clock=f,
                meas_time_pct=(tl / tg - 1) * 100,
                meas_energy_pct=(el / eg - 1) * 100,
                sim_time_pct=(sl / sg - 1) * 100,
                sim_energy_pct=(xl / xg - 1) * 100,
                meas_gov_s=tg,
                meas_gov_kj=eg,
                meas_lock_s=tl,
                meas_lock_kj=el,
            )
        )
        if not quiet:
            print(
                f"  fixed {f} MHz against the governor, the `{tag}` load: measured time "
                f"{out[-1]['meas_time_pct']:+.1f}% energy {out[-1]['meas_energy_pct']:+.1f}%  |  "
                f"simulated time {out[-1]['sim_time_pct']:+.1f}% energy {out[-1]['sim_energy_pct']:+.1f}%"
            )
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="trace2")
    a = ap.parse_args()
    rows = run(a.tag)
    fx = check_fixed(a.tag)
    if fx:
        write_csv(os.path.join(HERE, "out", f"policies_{a.tag}_fixed.csv"), fx)
    out = os.path.join(HERE, "out", f"policies_{a.tag}.csv")
    write_csv(out, rows)
    print(f"wrote {out}")
