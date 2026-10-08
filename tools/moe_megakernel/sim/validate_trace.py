#!/usr/bin/env python3
"""The top rung: a load that MOVES, under the stock governor, simulated and measured on
one time axis.

  python3 -m sim.validate_trace --tag trace         # needs governor_trace.py's CSVs
  python3 -m sim.validate_trace --tag dynamics

`governor_trace.py` ran a scripted sequence of batches on the A100 with the clock left
to the governor, timing every graph replay and logging every NVML reading. This replays
the SAME script through the simulator -- same batches, same number of layers per
segment, same idle gaps -- with the queue kernel and the `governor` controller, and
compares:

  per segment   mean latency per layer, mean board power, mean clock
  per replay    the first 400 ms after every load step: replay k of the simulation
                against replay k of the measurement, with the segment's steady-state
                error taken out. A replay is 4 layers, 4-16 ms, so this sees the clock
                move where NVML's 100 ms readings cannot.
  whole run     wall time and energy
  over time     clock, power and latency (sim/out/trace_<tag>_run<k>.csv, for fig25)

The rate and power models come from the locked-clock sweeps and never saw a moving
load. The governor and limiter have eight numbers (the limiter's loop period, latency,
gain, peak factor and energy budget; the governor's wake delay, wake gain and park
delay). They were set by hand against the `dynamics` plan, so `dynamics` shows how well
they can be set and the other plans -- other batch sizes, other segment lengths, shorter
idle gaps, whole forward passes -- are the test.
"""

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np

from .calib import Calib
from .control import Fixed, Governor
from .demand import DATA
from .engine import Engine, Kernel, Shared
from .files import read_csv
from .hardware import Hardware
from .workload import build

HERE = os.path.dirname(os.path.abspath(__file__))
REPS = 4  # layers per graph replay in governor_trace.py


def load(tag, model="qwen"):
    def rows(name):
        return read_csv(os.path.join(DATA, f"governor_trace_{tag}_{name}_{model}.csv"))

    seg, rep, nv = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in rows("segments"):
        seg[int(r["run"])].append(
            dict(
                seg=int(r["seg"]),
                load=r["load"],
                tokens=int(r["tokens"]),
                t0=float(r["t_start_ms"]),
                t1=float(r["t_end_ms"]),
                layers=int(r["layers"]),
                replays=int(r["replays"]),
                layer_list=r.get("layer_list") or "0",
            )
        )
    for r in rows("replays"):
        rep[int(r["run"])].append((int(r["seg"]), float(r["t_end_ms"]), float(r["ms_per_layer"])))
    for r in rows("nvml"):
        nv[int(r["run"])].append((float(r["t_ms"]), float(r["power_w"]), int(r["clock_mhz"])))
    return seg, rep, nv


def spec_of(segments):
    """governor_trace segments -> the simulator's workload: counts and idle gaps kept."""
    out, after_idle = [], True
    for s in segments:
        if s["load"] == "idle":
            after_idle = True
            continue
        if s["layer_list"] == "all" or "," in s["layer_list"]:  # whole forward passes
            ll = "all" if s["layer_list"] == "all" else [int(v) for v in s["layer_list"].split(",")]
            b = dict(layers=ll, repeat=s["replays"], name=f"seg{s['seg']}")
        else:
            b = dict(layers=[int(s["layer_list"])], repeat=s["layers"], name=f"seg{s['seg']}")
        if s["load"].startswith("p"):
            b.update(phase="prefill", rep=int(s["load"][1:]))
        else:  # "d20" or "d20@160": cells, first cell
            cells, _, start = s["load"][1:].partition("@")
            b.update(phase="decode", cells=int(cells), start=int(start or 0))
        if after_idle:
            b["at_ms"] = s["t0"]  # an idle gap ends when the load arrives
        after_idle = False
        out.append(b)
    return out


def hold_mean(samples, t0, t1, col):
    """Time-weighted mean of a sample-and-hold series over [t0, t1]."""
    tot = acc = 0.0
    for i, s in enumerate(samples):
        a = max(s[0], t0)
        b = min(samples[i + 1][0] if i + 1 < len(samples) else t1, t1)
        if b > a:
            acc += s[col] * (b - a)
            tot += b - a
    return acc / tot if tot else float("nan")


HORIZON_MS = 400.0  # "the first 400 ms after a load step"
PARKED_MS = 2000.0  # an idle gap this long parks the clock


def step_kind(prev):
    """What came before a busy segment: 'wake' (the clock was parked), 'short idle'
    (it was idling at the top clock) or 'load change' (another batch size)."""
    if prev is None or (prev["load"] == "idle" and prev["t1"] - prev["t0"] >= PARKED_MS):
        return "wake"
    return "short idle" if prev["load"] == "idle" else "load change"


def replays(s, sim_layers):
    """The simulated layers of a segment grouped as the measurement's graph replays
    were: (ms since the segment began, mean ms per layer)."""
    n_rep = s["layers"] // s["replays"]
    t0 = sim_layers[0]["t_start"]
    return [
        (sim_layers[i + n_rep - 1]["t_end"] - t0, sum(x["ms"] for x in sim_layers[i : i + n_rep]) / n_rep)
        for i in range(0, len(sim_layers) - n_rep + 1, n_rep)
    ]


def transient(s, meas, sim):
    """Replay-by-replay ln(simulated / measured) latency, in %, over the first
    HORIZON_MS of a busy segment, minus the segment's own steady-state error."""
    n = min(len(meas), len(sim))
    if n < 6:
        return np.array([])
    d = np.array([np.log(sim[i][1] / meas[i][1]) * 100 for i in range(n)])
    t = np.array([meas[i][0] - s["t0"] for i in range(n)])
    late = d[t > 1000.0] if (t > 1000.0).sum() >= 5 else d[-max(5, n // 3) :]
    return d[t <= HORIZON_MS] - np.median(late)


# A measured segment is scored from this far in: an NVML reading describes the 100 ms
# before it. The simulated segment is scored whole, which is how the pass criteria of
# HELDOUT.md were registered -- and is not like for like when a segment begins with a
# clock transient, as one after a wake does. So the clock and the power are ALSO scored
# with the simulated side cut to the same window (`*_same`), and both are reported.
NVML_LAG_MS = 150.0


def simulate(segments, cal, model="qwen", hw=None, controller=None, shared=None):
    """Replay one measured run's script: same batches, layer counts and idle gaps, on
    the queue kernel, under `controller` (default: the stock governor)."""
    return Engine(
        hw or Hardware(),
        cal,
        Kernel("queue", 128),
        build(spec_of(segments), model),
        controller or Governor(),
        model=model,
        shared=shared,
    ).run()


def by_segment(e):
    """The simulated layers of each measured segment (batches are named seg<k>)."""
    out = defaultdict(list)
    for L in e.layers:
        out[int(L["name"].split(".")[0][3:])].append(L)
    return out


def score_segment(k, s, rep_k, nv_k, e, Ls):
    """One busy segment: mean latency per layer, board power and clock, both sides."""
    m_ms = float(np.mean([x[2] for x in rep_k if x[0] == s["seg"]]))
    m_w = hold_mean(nv_k, s["t0"] + NVML_LAG_MS, s["t1"], 1)
    m_clk = hold_mean(nv_k, s["t0"] + NVML_LAG_MS, s["t1"], 2)
    a, b = Ls[0]["t_start"], Ls[-1]["t_end"]
    s_ms = (b - a) / len(Ls)
    s_w = sum(x["mj"] for x in Ls) / (b - a)
    tr = [(t0, p, f[0]) for t0, t1, p, f, _ in e.trace if t1 > a and t0 < b]
    s_clk = hold_mean(tr, a, b, 2)
    cut = a + NVML_LAG_MS if b - a > 2 * NVML_LAG_MS else a  # the measured side's window
    return dict(
        run=k,
        seg=s["seg"],
        load=s["load"],
        tokens=s["tokens"],
        secs=round((s["t1"] - s["t0"]) / 1e3, 2),
        meas_ms=m_ms,
        sim_ms=s_ms,
        meas_w=m_w,
        sim_w=s_w,
        meas_clock=m_clk,
        sim_clock=s_clk,
        sim_w_same=hold_mean(tr, cut, b, 1),
        sim_clock_same=hold_mean(tr, cut, b, 2),
    )


def score_run(k, seg_k, rep_k, nv_k, e, cal, trans, steps):
    """One run: a row per busy segment, the run's energy and busy time, and -- added to
    `trans` and `steps` -- the replay-by-replay view of every load step."""
    sim_layers = by_segment(e)
    rows, prev = [], None
    for s in seg_k:
        if s["load"] == "idle":
            prev = s
            continue
        kind = step_kind(prev)
        m = [(x[1], x[2]) for x in rep_k if x[0] == s["seg"]]
        sr = replays(s, sim_layers[s["seg"]])
        trans[kind] += list(transient(s, m, sr))
        for i in range(min(len(m), len(sr))):  # the step itself, for fig28
            if min(m[i][0] - s["t0"], sr[i][0]) > 2 * HORIZON_MS:
                break
            steps.append(
                [
                    k,
                    s["seg"],
                    kind,
                    s["tokens"],
                    i,
                    round(m[i][0] - s["t0"], 3),
                    round(m[i][1], 5),
                    round(sr[i][0], 3),
                    round(sr[i][1], 5),
                    round(s["t0"], 3),
                    round(sim_layers[s["seg"]][0]["t_start"], 3),
                ]
            )
        prev = s
        rows.append(score_segment(k, s, rep_k, nv_k, e, sim_layers[s["seg"]]))
    t_end = seg_k[-1]["t1"]
    m_j = hold_mean(nv_k, 0.0, t_end, 1) * t_end / 1e3
    s_end = max(e.t, t_end)  # the simulation idles to the end
    s_j = (e.energy + cal.power.static(e.hw, e.f) * (s_end - e.t)) / 1e3
    return dict(
        run=k,
        segments=rows,
        meas_j=m_j,
        sim_j=s_j,
        meas_busy_ms=sum(s["t1"] - s["t0"] for s in seg_k if s["load"] != "idle"),
        sim_busy_ms=sum(L["ms"] for L in e.layers),
        meas_end_ms=t_end,
        sim_end_ms=e.t,
    )


def write_series(path, seg_k, rep_k, nv_k, e):
    """Both sides on one time axis, for fig25: series, t_ms, value."""
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["series", "t_ms", "value"])
        for t, p, c in nv_k:
            w.writerow(["meas_power_w", t, p])
            w.writerow(["meas_clock_mhz", t, c])
        for sg, t, ms in rep_k:
            w.writerow(["meas_layer_ms", t, ms])
        for x in e.ticks:
            w.writerow(["sim_power_w", x["t_ms"], x["p_sensor_w"]])
            w.writerow(["sim_clock_mhz", x["t_ms"], x["f_all"]])
        for t0, t1, p, f, _ in e.trace:
            w.writerow(["sim_clock_exact_mhz", round(t0, 4), f[0]])
        # layers per graph replay: REPS of one layer, or one pass over the stack
        n_rep = next((s["layers"] // s["replays"] for s in seg_k if s["replays"]), REPS)
        for i in range(0, len(e.layers) - n_rep + 1, n_rep):
            grp = e.layers[i : i + n_rep]
            if grp[0]["name"].split(".")[0] == grp[-1]["name"].split(".")[0]:
                w.writerow(["sim_layer_ms", round(grp[-1]["t_end"], 4), round(sum(x["ms"] for x in grp) / n_rep, 5)])


def _rms(v):
    return float(np.sqrt(np.mean(np.square(v)))) if len(v) else None


def summarise(tag, report, trans, peaks):
    allr = [r for rp in report for r in rp["segments"]]
    et = np.array([np.log(r["sim_ms"] / r["meas_ms"]) * 100 for r in allr])
    ep = np.array([(r["sim_w"] / r["meas_w"] - 1) * 100 for r in allr])
    ec = np.array([r["sim_clock"] - r["meas_clock"] for r in allr])
    ep2 = np.array([(r["sim_w_same"] / r["meas_w"] - 1) * 100 for r in allr])
    ec2 = np.array([r["sim_clock_same"] - r["meas_clock"] for r in allr])
    worst = lambda v: float(v[np.argmax(np.abs(v))])
    return dict(
        tag=tag,
        runs=len(report),
        segments=len(allr),
        latency_rms_pct=float(np.sqrt(np.mean(et**2))),
        latency_worst_pct=worst(et),
        power_rms_pct=float(np.sqrt(np.mean(ep**2))),
        power_worst_pct=worst(ep),
        clock_rms_mhz=float(np.sqrt(np.mean(ec**2))),
        clock_worst_mhz=worst(ec),
        power_rms_pct_same_window=float(np.sqrt(np.mean(ep2**2))),
        clock_rms_mhz_same_window=float(np.sqrt(np.mean(ec2**2))),
        clock_worst_mhz_same_window=worst(ec2),
        energy_err_pct=[float((rp["sim_j"] / rp["meas_j"] - 1) * 100) for rp in report],
        transient_rms_pct={k: _rms(v) for k, v in trans.items()},
        transient_replays={k: len(v) for k, v in trans.items()},
        peak_reading_w=dict(sim=max(p[0] for p in peaks), meas=max(p[1] for p in peaks)),
    )


def show_run(rp):
    rows = rp["segments"]
    print(f"run {rp['run']}: {len(rows)} load segments, {rp['meas_end_ms'] / 1e3:.1f} s")
    print(
        f"  {'seg':>3} {'load':>5} {'tokens':>6} {'s':>5} | {'measured':>27} | "
        f"{'simulated':>27} | {'d ms':>6} {'d W':>6} {'d MHz':>6}"
    )
    for r in rows:
        print(
            f"  {r['seg']:>3} {r['load']:>5} {r['tokens']:>6} {r['secs']:>5.1f} | "
            f"{r['meas_ms']:>7.4f} ms {r['meas_w']:>6.1f} W {r['meas_clock']:>5.0f} MHz | "
            f"{r['sim_ms']:>7.4f} ms {r['sim_w']:>6.1f} W {r['sim_clock']:>5.0f} MHz | "
            f"{(r['sim_ms'] / r['meas_ms'] - 1) * 100:>+5.1f}% "
            f"{r['sim_w'] - r['meas_w']:>+6.1f} {r['sim_clock'] - r['meas_clock']:>+6.0f}"
        )
    print(
        f"  whole run: energy {rp['meas_j']:.1f} J measured, {rp['sim_j']:.1f} J simulated "
        f"({(rp['sim_j'] / rp['meas_j'] - 1) * 100:+.1f}%); busy time "
        f"{rp['meas_busy_ms'] / 1e3:.2f} s measured, {rp['sim_busy_ms'] / 1e3:.2f} s simulated "
        f"({(rp['sim_busy_ms'] / rp['meas_busy_ms'] - 1) * 100:+.1f}%)"
    )


def show_summary(s, trans):
    print(
        f"\n{s['tag']}: {s['segments']} segments over {s['runs']} runs -- latency rms "
        f"{s['latency_rms_pct']:.2f}% (worst {s['latency_worst_pct']:+.1f}%), "
        f"power rms {s['power_rms_pct']:.2f}% (worst {s['power_worst_pct']:+.1f}%), "
        f"clock rms {s['clock_rms_mhz']:.1f} MHz (worst {s['clock_worst_mhz']:+.0f}); run energy "
        + ", ".join(f"{x:+.1f}%" for x in s["energy_err_pct"])
    )
    print(
        f"  the same window on both sides (from {NVML_LAG_MS:.0f} ms into each segment): power rms "
        f"{s['power_rms_pct_same_window']:.2f}%, clock rms {s['clock_rms_mhz_same_window']:.1f} MHz "
        f"(worst {s['clock_worst_mhz_same_window']:+.0f})"
    )
    print(
        f"  first {HORIZON_MS:.0f} ms after a load step, replay by replay: "
        + "; ".join(f"{k} {_rms(v):.2f}% (n={len(v)})" for k, v in sorted(trans.items()) if len(v))
        + f"; highest 100 ms reading {s['peak_reading_w']['sim']:.0f} W simulated, "
        f"{s['peak_reading_w']['meas']:.0f} W measured"
    )


def run(tag="trace", model="qwen", cal=None, quiet=False, hw=None, controller=None):
    """Replay every run of trace `tag`, score it, and leave in sim/out:
    trace_<tag>.json (summary and per-segment rows), trace_<tag>_run<k>.csv (both sides
    over time), trace_<tag>_steps.csv (the replays around every load step).
    `controller` is a factory, e.g. `lambda: Fixed(1050)`; default the stock governor."""
    cal = cal or Calib.load()
    seg, rep, nv = load(tag, model)
    out = os.path.join(HERE, "out")
    os.makedirs(out, exist_ok=True)
    shared, report = Shared(), []
    trans, peaks, steps = defaultdict(list), [], []
    for k in sorted(seg):
        e = simulate(seg[k], cal, model, hw, controller() if controller else None, shared)
        peaks.append((max(x["p_sensor_w"] for x in e.ticks), max(x[1] for x in nv[k])))
        report.append(score_run(k, seg[k], rep[k], nv[k], e, cal, trans, steps))
        write_series(os.path.join(out, f"trace_{tag}_run{k}.csv"), seg[k], rep[k], nv[k], e)
        if not quiet:
            show_run(report[-1])
    summary = summarise(tag, report, trans, peaks)
    if not quiet:
        show_summary(summary, trans)
    with open(os.path.join(out, f"trace_{tag}.json"), "w") as fh:
        json.dump(dict(summary=summary, runs=report), fh, indent=1)
    with open(os.path.join(out, f"trace_{tag}_steps.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "seg", "kind", "tokens", "k", "t_ms", "meas_ms", "sim_t_ms", "sim_ms", "t0_ms", "sim_t0_ms"])
        w.writerows(steps)
    return summary, report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="trace")
    ap.add_argument(
        "--fixed",
        type=int,
        default=None,
        help="the trace was run with the clock locked at this many MHz "
        "(governor_trace.py --lock): replay it under the `fixed` controller",
    )
    a = ap.parse_args()
    run(a.tag, controller=(lambda: Fixed(a.fixed)) if a.fixed else None)
