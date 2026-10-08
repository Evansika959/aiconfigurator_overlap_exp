#!/usr/bin/env python3
"""Run one scenario through the simulator.

  python3 -m sim.run sim/configs/decode2048_b4.yaml
  python3 -m sim.run sim/configs/decode2048_b4.yaml --set controller.type=governor

INPUT, one YAML or JSON file with four parts (see sim/configs/):

  hardware     the machine: V/f domains, TDP, limiter, actuation delay   (sim/hardware.py)
  kernel       how a layer is launched: queue or persistent, tile size,
               which experts are cold and how many SMs they get          (sim/engine.py)
  controller   who sets the clocks                                       (sim/control.py)
  workload     the batches                                               (sim/workload.py)

OUTPUT, four tables at four time scales, and a summary:

  events.csv    every event: when, what, which batch / layer / stage
  trace.csv     every interval between two events: power, clocks, what was running
  layers.csv    one row per MoE layer: latency, energy, when each partition finished
  windows.csv   one row per reported power reading (100 ms): window power, ceiling, clocks
  summary.json  totals and steady-state figures
"""

import argparse
import copy
import json
import os
import time

import numpy as np

from .calib import DEFAULT, Calib
from .control import make
from .engine import Engine, Kernel
from .files import write_csv
from .hardware import Hardware
from .workload import build

HERE = os.path.dirname(os.path.abspath(__file__))


def load(path):
    with open(path) as fh:
        text = fh.read()
    if path.endswith(".json"):
        return json.loads(text)
    import yaml

    return yaml.safe_load(text)


def override(scn, sets):
    """--set a.b.c=value, value parsed as JSON when it can be."""
    for s in sets or []:
        k, v = s.split("=", 1)
        try:
            v = json.loads(v)
        except ValueError:
            pass
        d = scn
        ks = k.split(".")
        for x in ks[:-1]:
            d = d.setdefault(x, {})
        d[ks[-1]] = v
    return scn


def simulate(scn, cal=None, shared=None, record=True):
    """Scenario dict -> finished Engine."""
    cal = cal or Calib.load(scn.get("calibration", DEFAULT))
    if "uncore_share" in scn.get("hardware", {}):
        scn = copy.deepcopy(scn)
        cal = copy.deepcopy(cal)
        cal.power.uncore_share = scn["hardware"].pop("uncore_share")
    hw = Hardware(**scn.get("hardware", {}))
    model = scn.get("model", "qwen")
    eng = Engine(
        hw,
        cal,
        Kernel(**scn.get("kernel", {})),
        build(scn["workload"], model),
        make(scn.get("controller", {"type": "governor"})),
        model=model,
        shared=shared,
        record=record,
    )
    return eng.run()


def summarise(e, scn=None, warm=0.5):
    """Totals, and the steady state: the last `warm` share of each batch type's layers,
    after clocks and limiter have settled."""
    hw = e.hw
    L = e.layers
    ms = np.array([x["ms"] for x in L])
    tdp = hw.tdp_w
    # a reported reading is a 100 ms mean, not an instant: a GEMM alone may sit above TDP
    # for a millisecond while the layer it belongs to averages below it. And the limiter
    # holds the board AT tdp, so half of all readings are a watt or two above; what
    # matters is how long it was above the peak the limiter tolerates.
    peak = hw.limiter.k_peak * tdp
    over = sum(hw.limiter.period_ms for x in e.ticks if x["p_sensor_w"] > peak)
    dur = np.array([b - a for a, b, *_ in e.trace]) if e.trace else np.array([e.t])
    clk = np.array([f for *_, f, _ in e.trace], float) if e.trace else np.array([e.f], float)
    s = dict(
        name=(scn or {}).get("name", ""),
        controller=e.ctl.describe(),
        kernel=dict(type=e.kernel.type, block_m=e.kernel.block_m, n_cold=e.kernel.n_cold, cold_sms=e.kernel.cold_sms),
        domains=[dict(name=d.name, sms=d.sms) for d in hw.domains],
        time_ms=round(e.t, 4),
        energy_j=round(e.energy / 1e3, 4),
        avg_power_w=round(e.energy / e.t, 2) if e.t else 0.0,
        peak_power_w=round(max((p for _, _, p, *_ in e.trace), default=e.P), 2),
        peak_window_w=round(max((x["p_window_w"] for x in e.ticks), default=0.0), 2),
        peak_reading_w=round(max((x["p_sensor_w"] for x in e.ticks), default=0.0), 2),
        ms_above_k_peak=round(float(over), 1),
        layers=len(L),
        events=len(e.events),
        mean_clock_mhz={d.name: round(float(clk[:, i] @ dur / dur.sum()), 1) for i, d in enumerate(hw.domains)},
        final_clock_mhz={d.name: f for d, f in zip(hw.domains, e.f, strict=True)},
    )
    if len(L):
        s["layer_ms"] = dict(
            mean=round(float(ms.mean()), 5),
            p50=round(float(np.median(ms)), 5),
            p99=round(float(np.percentile(ms, 99)), 5),
            max=round(float(ms.max()), 5),
        )
        steady = {}
        for ph in sorted({x["phase"] for x in L}):
            sel = [x for x in L if x["phase"] == ph]
            tail = sel[int(len(sel) * (1 - warm)) :]
            t = sum(x["ms"] for x in tail)
            mj = sum(x["mj"] for x in tail)
            tok = sum(x["tokens"] for x in tail)
            steady[ph] = dict(
                layers=len(tail),
                layer_ms=round(t / len(tail), 5),
                power_w=round(mj / t, 2),
                mj_per_layer=round(mj / len(tail), 3),
                uj_per_token_layer=round(mj / tok * 1e3, 3) if tok else None,
                tokens_per_s=round(tok / t * 1e3, 1) if t else None,
            )
        s["steady"] = steady
    return s


def write(e, out, scn=None):
    os.makedirs(out, exist_ok=True)
    names = [d.name for d in e.hw.domains]

    def dump(name, rows, fields=None):
        write_csv(os.path.join(out, name), rows, fields or (list(rows[0].keys()) if rows else ["empty"]))

    dump("events.csv", e.events, ["t_ms", "event", "batch", "layer", "stage", "who", "detail"])
    dump(
        "trace.csv",
        [
            dict(
                t0_ms=round(a, 6),
                t1_ms=round(b, 6),
                power_w=round(p, 3),
                **{f"f_{n}": x for n, x in zip(names, f, strict=True)},
                running=run,
            )
            for a, b, p, f, run in e.trace
        ],
        ["t0_ms", "t1_ms", "power_w"] + [f"f_{n}" for n in names] + ["running"],
    )
    rows = []
    for x in e.layers:
        r = dict(
            batch=x["batch"],
            name=x["name"],
            phase=x["phase"],
            tokens=x["tokens"],
            layer=x["layer"],
            t_start_ms=round(x["t_start"], 6),
            t_end_ms=round(x["t_end"], 6),
            ms=round(x["ms"], 6),
            mj=round(x["mj"], 4),
            avg_w=round(x["w"], 2),
        )
        for n, f0, f1 in zip(names, x["f_start"], x["f_end"], strict=True):
            r[f"f_{n}_start"], r[f"f_{n}_end"] = f0, f1
        for (who, g), t in sorted(x["finish"].items(), key=lambda kv: kv[1]):
            r[f"{who}_{g}_done_ms"] = round(t - x["t_start"], 6)
        rows.append(r)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    dump("layers.csv", rows, keys)
    dump("windows.csv", e.ticks)
    s = summarise(e, scn)
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(dict(summary=s, scenario=scn), fh, indent=1, default=str)
    return s


def show(s):
    print(
        f"  controller {s['controller']['type']}, kernel {s['kernel']['type']} "
        f"BLOCK_M {s['kernel']['block_m']}"
        + (f", {s['kernel']['n_cold']} cold experts" if s["kernel"]["n_cold"] else "")
        + ", domains "
        + " + ".join(f"{d['name']}:{d['sms']}" for d in s["domains"])
    )
    print(
        f"  {s['layers']} layers, {s['events']} events, {s['time_ms']:.1f} ms simulated, "
        f"{s['energy_j']:.2f} J, mean {s['avg_power_w']:.1f} W, peak 100 ms reading "
        f"{s['peak_reading_w']:.1f} W, {s['ms_above_k_peak']:.0f} ms above k_peak x TDP"
    )
    print(
        "  mean clock "
        + ", ".join(f"{k} {v:.0f} MHz" for k, v in s["mean_clock_mhz"].items())
        + "; final "
        + ", ".join(f"{k} {v}" for k, v in s["final_clock_mhz"].items())
    )
    for ph, x in s.get("steady", {}).items():
        print(
            f"  steady {ph:8s} {x['layer_ms']:.4f} ms/layer  {x['power_w']:.1f} W  "
            f"{x['mj_per_layer']:.1f} mJ/layer  {x['tokens_per_s']:.0f} tokens/s per layer"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("scenario")
    ap.add_argument("--out", default=None, help="output directory (default sim/out/<name>)")
    ap.add_argument("--set", action="append", help="override, e.g. controller.type=governor")
    a = ap.parse_args()
    scn = override(load(a.scenario), a.set)
    scn.setdefault("name", os.path.splitext(os.path.basename(a.scenario))[0])
    t0 = time.time()
    e = simulate(scn)
    out = a.out or os.path.join(HERE, "out", scn["name"])
    s = write(e, out, scn)
    print(f"{scn['name']}  ({time.time() - t0:.2f} s wall)")
    show(s)
    print(f"  wrote {out}/{{events,trace,layers,windows}}.csv, summary.json")


if __name__ == "__main__":
    main()
