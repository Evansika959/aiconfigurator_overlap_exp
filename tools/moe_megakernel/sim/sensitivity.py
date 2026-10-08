#!/usr/bin/env python3
"""What a second V/f domain is worth, under every assumption nobody can measure.

  python3 -m sim.sensitivity --cells 32

The A100 has one V/f domain, so these things about a two-domain chip are parameters of
the hardware description rather than facts:

  uncore                 which clock the shared logic (L2, crossbar) follows
  uncore_share           how much of the clock-dependent idle power belongs to that logic
  cdc_penalty            what crossing between two clock domains costs an L2 delivery
  cold_regulator         whether the lower rail has its own buck converter or hangs off
                         the higher one through an LDO (its saving is then linear in V)
  delivery_uncore_share  how much of an SM's panel-delivery cost follows the shared
                         logic's clock rather than its own
  active_power           whether a loaded die's extra power follows each domain's clock
                         or the faster one
  epilogue_clock         whether the activation and sum run at the SM-weighted mean
                         clock or the slower domain's
  xi_clock               whether the DRAM-to-L2 coupling follows the partition's own
                         clock or the shared logic's

and one thing is a choice of ours: the FORM of the rate model, that is, which of its
parameters are fitted. Three forms were cross-validated (`sim.calibrate --compare`); the
other two are close enough behind the one in use that a conclusion should not depend on
which was picked, so each is a row too (it needs its calibration file:
`python3 -m sim.calibrate --form <name> --out sim/calib/a100_qwen_<name>.json`).

For each setting this compares the SAME pinned kernel with one clock (B3) and with a
clock per domain (B4), so the only thing that differs is the second rail:

  energy   least energy at the layer time B3 reaches at 1290 MHz, and at 1350 MHz
  latency  fastest layer under TDP

If the answer changed sign across these rows, the simulator could not say whether the
design works. It does not.
"""

import argparse
import copy
import os
from dataclasses import dataclass, field
from typing import Optional

from .calib import Calib
from .engine import Kernel, Shared
from .files import write_csv
from .hardware import Hardware, two_domain
from .study import Probe, best, bisected, refine, sweep1, sweep2
from .workload import build

HERE = os.path.dirname(os.path.abspath(__file__))


@dataclass
class Variant:
    """One row: the shipped model with one thing changed. To test another assumption,
    add one to VARIANTS."""

    name: str
    hw: dict = field(default_factory=dict)  # Hardware arguments
    power: dict = field(default_factory=dict)  # PowerParams fields
    form: Optional[str] = None  # a calibration fitted under another form of the rate model


VARIANTS = [
    Variant("as shipped"),
    Variant("uncore follows the SLOWER domain", hw=dict(uncore="min")),
    Variant("no idle power in shared logic", power=dict(uncore_share=0.0)),
    Variant("half the idle power in shared logic", power=dict(uncore_share=0.5)),
    Variant("domain crossing costs 5%", hw=dict(cdc_penalty=0.05)),
    Variant("domain crossing costs 10%", hw=dict(cdc_penalty=0.10)),
    Variant("lower rail through an LDO", hw=dict(cold_regulator="ldo")),
    Variant("half of panel delivery on the uncore clock", hw=dict(delivery_uncore_share=0.5)),
    Variant("all of panel delivery on the uncore clock", hw=dict(delivery_uncore_share=1.0)),
    Variant("DRAM-L2 coupling on the uncore clock", hw=dict(xi_clock="uncore")),
    Variant("loaded-die power at the faster clock", hw=dict(active_power="max")),
    Variant("epilogue at the slower clock", hw=dict(epilogue_clock="min")),
    Variant("rate model: proportional form", form="proportional"),
    Variant("rate model: linear form", form="linear"),
]


def calibration_of(form):
    """The calibration fitted under another form of the rate model, or None if that file
    has not been made."""
    path = os.path.join(HERE, "calib", f"a100_qwen_{form}.json")
    return Calib.load(path) if os.path.exists(path) else None


def run(
    cells=32,
    ktype="persistent",
    bm=None,
    n_cold=64,
    cold_sms=32,
    tdp=400.0,
    ref_clocks=(1290, 1350),
    step=30,
    f_lo=900,
    cal=None,
    quiet=False,
):
    """One row per assumption: B4 against B3 for one fixed partition. Energy is compared
    at the layer time B3 reaches at each clock in `ref_clocks`; latency under TDP."""
    cal0 = cal or Calib.load()
    bm = bm or (128 if cells == 0 or ktype == "queue" else 64)
    L = build([dict(phase="prefill" if cells == 0 else "decode", cells=cells, layers=[0])])[0].layers[0]
    one = Hardware(tdp_w=tdp)
    fine = [f for f in one.clocks() if f >= 510]
    coarse = [one.snap(f) for f in range(one.f_max, f_lo - 1, -step)]
    rows = []
    for v in VARIANTS:
        cal = copy.deepcopy(calibration_of(v.form) if v.form else cal0)
        if cal is None:
            print(f"  ({v.name}: no sim/calib/a100_qwen_{v.form}.json, skipped)")
            continue
        for k, x in v.power.items():
            setattr(cal.power, k, x)
        sh = Shared()
        p3 = sweep1(Probe(cal, one, Kernel(ktype, bm, n_cold, cold_sms), L, sh), fine)
        hw2 = two_domain(cold_sms, tdp_w=tdp, **v.hw)
        slos = [next(p for p in p3 if p[0][0] == f)[1] for f in ref_clocks]
        p4, add = sweep2(Probe(cal, hw2, Kernel(ktype, bm, n_cold), L, sh), coarse, fine)
        p4.append(bisected(cal, hw2, Kernel(ktype, bm, n_cold), L, sh))
        for target in [None, *slos]:
            refine(p4, add, hw2, tdp, target, rounds=3)
        l3, l4 = best(p3, tdp), best(p4, tdp)
        row = dict(assumption=v.name)
        for f, slo in zip(ref_clocks, slos, strict=True):
            e3, e4 = best(p3, tdp, slo), best(p4, tdp, slo)
            row.update({f"b3_mj_{f}": e3[2], f"b4_mj_{f}": e4[2], f"b4_f_{f}": "/".join(map(str, e4[0]))})
            row[f"energy_pct_{f}"] = (e4[2] / e3[2] - 1) * 100
        row.update(b3_ms=l3[1], b4_ms=l4[1], b4_f_tdp="/".join(map(str, l4[0])), latency_pct=(l4[1] / l3[1] - 1) * 100)
        rows.append(row)
    if not quiet:
        print(
            f"{'prefill' if cells == 0 else 'decode'} {L.pairs // 8} tokens; {n_cold} cold experts "
            f"on a {cold_sms}-SM domain; {ktype} per-SM costs, BM{bm}; TDP {tdp:.0f} W"
        )
        head = " | ".join(f"{'energy at B3@' + str(f) + ' latency':>26}" for f in ref_clocks)
        print(f"  {'':44s} | {head} | {'fastest under TDP':>20}")
        for r in rows:
            cols = " | ".join(f"{r[f'energy_pct_{f}']:>+15.2f}% {r[f'b4_f_{f}']:>9}" for f in ref_clocks)
            print(f"  {r['assumption']:44s} | {cols} | {r['latency_pct']:>+9.2f}% {r['b4_f_tdp']:>9}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=int, default=32)
    ap.add_argument("--kernel", default="persistent", choices=["persistent", "queue"])
    ap.add_argument("--n-cold", type=int, default=64)
    ap.add_argument("--cold-sms", type=int, default=32)
    a = ap.parse_args()
    rows = run(a.cells, a.kernel, n_cold=a.n_cold, cold_sms=a.cold_sms)
    out = os.path.join(HERE, "out", f"sensitivity_cells{a.cells}_{a.kernel}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    write_csv(out, rows)
    print(f"wrote {out}")
