#!/usr/bin/env python3
"""Replay every measurement through the simulator and report how far off it is.

  python3 -m sim.validate            # prints the ladder, writes sim/out/validation.json

The ladder goes from what the model was fitted on to what it never saw. A rung is only
as good as the rung below it, and the two-domain predictions sit above the top one --
no measurement can reach them on a one-domain chip, which is exactly why every rung
underneath has to be reported and not just the best one.

  0   decomposition     tiles, tensor instructions, DRAM bytes   vs ncu counters
  1   fitted            solo and co-run times and power          the training sweeps
  2   other days        the same model on earlier campaigns      never fitted, but one
  3   other workloads   four other MoE layers, and the pinned    never fitted
                        layer end to end
  3b  other models      three models the fit never saw           never fitted
  4   whole layers      end to end through the event engine      partly fitted
  5   the governor      clock, power, latency it settles at      never fitted
  6   a moving load     scripted traces under the governor       sim.validate_trace
  7   self-consistency  two domains at one clock = one domain; energy = integral of power

Every row says whether the calibration used its records (`fit`: yes, partly, no), so a
fitted row is never read as a test.

Time errors are ln(predicted / measured) in %. Power errors are given twice: as % of
board power, and (`dyn`) as % of board power above idle at that clock. Idle is a
measured table, 55-77 W of every reading, that the model cannot get wrong; the second
number is the error of what the model actually predicts.

Each rung is one function below. To add a check, write a function that calls
`Ladder.row`, and call it from `run`.
"""

import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

from . import meas
from .calib import Calib
from .control import Fixed, Governor
from .demand import DATA, GEMMS, SHAPES, Partition, counts_for, split
from .engine import Engine, Kernel, Shared
from .files import read_csv, read_json
from .hardware import Hardware, two_domain
from .rate import Job, solve_rates
from .workload import build

HERE = os.path.dirname(os.path.abspath(__file__))
MIN_REPLAY_MS = 1.0  # replayed single passes shorter than this are mostly launch gap
# The A100's L2. A partition whose weights fit in it is measured by the sweeps as
# something inference never does: the pass is repeated, so after the first one nothing
# comes from DRAM. The simulator charges DRAM on every pass, as one pass per layer would.
L2_MB = 40.0

# Sweeps from before the calibration campaign: (file, decode cells or 0 = prefill, cold
# experts, label). Replayed single passes; never fitted.
OLD_CAMPAIGNS = [
    ("partition_kappa_qwen.csv", 0, 16, "prefill, BM16-128, 54 SMs, 600-1290 MHz"),
    ("partition_kappa_qwen_decode1024.csv", 16, 64, "decode 1024, BM16-128, 600-1290 MHz"),
    ("partition_kappa_qwen_decode2048.csv", 32, 64, "decode 2048, BM16-128, 600-1290 MHz"),
    ("partition_kappa_L8.csv", 0, 8, "prefill, 8 cold experts"),
    ("partition_kappa_L32.csv", 0, 32, "prefill, 32 cold experts"),
    ("partition_kappa_L64.csv", 0, 64, "prefill, 64 cold experts"),
    ("partition_kappa_L96.csv", 0, 96, "prefill, 96 cold experts"),
    ("partition_kappa_qwen_hiclk54.csv", 0, 16, "prefill, 54 SMs, 1050-1410 MHz"),
]
# The one earlier sweep the calibration did use: its power readings fill in the voltage
# curve between the scan's clocks, and its times set `launch_gap`.
PARTLY_FITTED = {"partition_kappa_qwen_hiclk54.csv"}
OTHER_MODELS = [
    ("olmoe", "OLMoE, 64 experts, I=1024"),
    ("granite", "Granite, 40 experts, K=1536"),
    ("qwen3next", "Qwen3-Next, 512 experts, top-10"),
]
TRACES = ("dynamics", "trace", "trace2", "forward", "forward2", "lock1050", "lock1290", "heldout1", "heldout2")


# ------------------------------------------------------- one record, simulated ----
def launch_gap(f):
    """ms between two kernels of one CUDA-graph replay: from the five configurations
    that both the scan (no gap) and the earlier single-pass sweeps (two gaps per pass)
    measured -- prefill, 16 cold experts, cold partition on 54 SMs, 1050-1410 MHz."""
    return 0.5 * (0.0062 + 37.8 / f)


def solo(cal, r, exact=False):
    """(ms per pass, board W) of a partition alone, continuously repeated."""
    c = cal.rate("persistent").at(r["f"])
    pw = cal.power
    t = e = 0.0
    for g in GEMMS:
        st = r["part"].stage(g, r["n"])
        t += Job(st, c, exact=exact).time()
        e += pw.stage_mj(pw.g_at(r["f"]), st)
    gap = 2 * launch_gap(r["f"]) if r.get("replay") else 0.0
    return t + gap, pw.idle_at(r["f"]) + (pw.act_at(r["f"]) * t + e) / (t + gap)


def corun(cal, r, exact=False):
    """Light beside heavy, both repeated so that they overlap: ((ms, ms), board W)."""
    c = cal.rate("persistent").at(r["f"])
    pw = cal.power
    T, E = {"light": 0.0, "heavy": 0.0}, {"light": 0.0, "heavy": 0.0}
    for g in GEMMS:
        jobs = [Job(r[k]["part"].stage(g, r[k]["n"]), c, exact=exact) for k in ("light", "heavy")]
        solve_rates(jobs)
        for k, j in zip(("light", "heavy"), jobs, strict=True):
            T[k] += j.t
            E[k] += pw.stage_mj(pw.g_at(r["f"]), j.st)
    P = pw.idle_at(r["f"]) + pw.act_at(r["f"]) + sum(E[k] / T[k] for k in T)
    return (T["light"], T["heavy"]), P


def layer(
    cal,
    cells,
    bm,
    f,
    kernel="persistent",
    n_cold=0,
    cold_sms=0,
    hw=None,
    n=3,
    shared=None,
    layer_index=0,
    launch_ms=0.0,
    model="qwen",
):
    """(ms, W) of one whole layer end to end, steady state, through the event engine."""
    b = build([dict(phase="prefill" if cells == 0 else "decode", cells=cells, layers=[layer_index], repeat=n)], model)
    e = Engine(
        hw or Hardware(),
        cal,
        Kernel(kernel, bm, n_cold, cold_sms, launch_ms),
        b,
        Fixed(f),
        model=model,
        shared=shared,
        record=False,
        limiter=False,
    ).run()
    L = e.layers[-1]
    return L["ms"], L["w"]


def fits_l2(r):
    """True if a swept record has a partition whose weights stay resident in L2."""
    parts = [r["part"]] if r["kind"] == "solo" else [r["light"]["part"], r["heavy"]["part"]]
    return any(p.weight_mb < L2_MB for p in parts)


# ------------------------------------------------------------------ the ladder ----
def stats(e):
    e = np.asarray(e, float)
    if not len(e):
        return None
    return dict(
        n=len(e),
        rms=round(float(np.sqrt(np.mean(e**2))), 2),
        bias=round(float(e.mean()), 2),
        worst=round(float(e[np.argmax(np.abs(e))]), 1),
    )


class Ladder:
    """One row per check, and the (measured, simulated) points behind it for fig24."""

    HEADER = (
        f"  {'':44s} {'fit':>6} | {'time: n':>5} {'rms%':>6} {'bias%':>6} {'worst%':>7} | "
        f"{'power: n':>5} {'rms%':>6} {'bias%':>6} {'worst%':>7} {'dyn rms%':>8}"
    )

    def __init__(self):
        self.rows, self.points, self.rung = {}, [], ""

    def section(self, rung, title, header=True):
        self.rung = rung
        print(f"\n{title}")
        if header:
            print(self.HEADER)

    def row(self, key, label, time_err, power_err, fit="no", dyn_err=()):
        """`fit`: did the calibration use these records -- yes, partly, no. `dyn_err`:
        the power errors again, relative to board power above idle."""
        st, sp, sd = stats(time_err), stats(power_err), stats(dyn_err)
        self.rows[key] = dict(rung=self.rung, label=label, fit=fit, time=st, power=sp, power_dyn=sd)
        f = lambda s: (
            f"{s['n']:5d} {s['rms']:6.2f} {s['bias']:+6.2f} {s['worst']:+7.1f}"
            if s
            else f"{'':>5} {'':>6} {'':>6} {'':>7}"
        )
        dyn = f"{sd['rms']:8.2f}" if sd else f"{'':>8}"
        print(f"  {label:44s} {fit:>6} | {f(st)} | {f(sp)} {dyn}")

    def power(self, cal, r, p, group, key, pe, de):
        """Record one power prediction p of record r: the error as % of board power in
        `pe`, as % of the power above idle in `de`, and the point for fig24."""
        pe.append((p - r["power"]) / r["power"] * 100)
        de.append((p - r["power"]) / (r["power"] - cal.power.idle_at(r["f"])) * 100)
        if group:
            self.points.append((group, key, "w", r["power"], p))

    def records(self, cal, recs, key, label, group, fit="no"):
        """Score swept 'solo' and 'corun' records. `group` names them in fig24."""
        te, pe, de = [], [], []
        for r in recs:
            if r.get("replay") and r["ms"] < MIN_REPLAY_MS:
                continue
            if r["kind"] == "solo":
                t, p = solo(cal, r)
                te.append(np.log(t / r["ms"]) * 100)
                self.points.append((group, key, "ms", r["ms"], t))
            elif r["kind"] == "corun":
                (tl, th), p = corun(cal, r)
                te += [np.log(tl / r["light"]["ms"]) * 100, np.log(th / r["heavy"]["ms"]) * 100]
                self.points.append((group, key, "ms", r["light"]["ms"], tl))
                self.points.append((group, key, "ms", r["heavy"]["ms"], th))
            else:
                continue
            self.power(cal, r, p, group, key, pe, de)
        self.row(key, label, te, pe, fit, de)

    def layers(self, cal, recs, key, label, group, shared, point_key=None, fit="no", **kw):
        """Score whole layers: recs carry cells, bm, f, ms, power and a kernel name."""
        te, pe, de = [], [], []
        for r in recs:
            t, p = layer(
                cal, r["cells"], r["bm"], r["f"], r["kernel"], shared=shared, **{**kw, **r.get("layer_kw", {})}
            )
            te.append(np.log(t / r["ms"]) * 100)
            if group:
                self.points.append((group, point_key or key, "ms", r["ms"], t))
            self.power(cal, r, p, group, point_key or key, pe, de)
        self.row(key, label, te, pe, fit, de)


# ------------------------------------------------------------------- the rungs ----
def rung0_decomposition(model):
    """The partition decomposition the simulator runs on against ncu's counters
    (data/ncu_counters_<model>.csv): tiles, tensor instructions, DRAM bytes read."""
    print("0  decomposition vs ncu")
    shape = SHAPES[model]
    err = {"tiles": [], "tensor": [], "dram": []}
    for r in read_csv(os.path.join(DATA, f"ncu_counters_{model}.csv")):
        if r["kernel"] != "persistent":
            continue
        c = counts_for(shape, int(r["cells"]))
        cold = split(c, 64)
        ex = None if r["part"] == "all" else (cold if r["part"] == "light" else ~cold)
        p = Partition(shape, c, int(r["block_m"]), ex)
        assert int(p.rows) == int(r["padded_rows"]), r
        k = p.counters(r["gemm"])
        err["tiles"].append(k["tiles"] / float(r["tiles"]) - 1)
        err["tensor"].append(k["tensor"] / float(r["sm__inst_executed_pipe_tensor_sum"]) - 1)
        err["dram"].append(k["dram_bytes"] / float(r["dram__bytes_read_sum"]) - 1)
    for k, v in err.items():
        v = np.array(v) * 100
        print(
            f"  {k:7s} model vs ncu over {len(v)} kernels: median {np.median(v):+.2f}%, "
            f"worst {v[np.argmax(np.abs(v))]:+.2f}%"
        )
    return err


def rung1_fitted(L, cal, model):
    L.section("1 fitted", "1  fitted: the training sweeps, 16-108 SMs")
    scan = meas.train(model)
    for stem, _ in meas.TRAIN:
        sub = [r for r in scan if r["src"] == stem]
        if sub:
            cl = sorted({r["f"] for r in sub})
            for kind, name in (("solo", "solo"), ("corun", "co-run")):
                L.records(
                    cal,
                    [r for r in sub if r["kind"] == kind],
                    f"{stem}_{kind}",
                    f"{stem[9:]}, {cl[0]}-{cl[-1]} MHz: {name}",
                    "fitted",
                    fit="yes",
                )
    for stem, cv in ((cal.meta.get("cross_validation") or {}).get(cal.meta.get("form")) or {}).items():
        print(
            f"  ({stem[9:]}, scored BEFORE it was fitted, by a fit on the scan alone: time rms "
            f"{cv['rms']:.2f}%, bias {cv['bias']:+.2f}%, worst {cv['worst']:+.1f}%, n={cv['n']})"
        )
    return scan


def rung2_other_days(L, cal, model):
    L.section("2 other days", "2  other days (replayed single passes of 1 ms or longer)")
    for name, cells, nc, label in OLD_CAMPAIGNS:
        if os.path.exists(os.path.join(meas.DATA, name)):
            fit = "partly" if name in PARTLY_FITTED else "no"
            try:
                L.records(cal, meas.kappa(model, name, cells, nc), name[:-4], label, "other days", fit)
            except (ValueError, KeyError) as ex:
                print(f"  {label:44s} | skipped: {ex}")
    first = meas.corun_first(model)
    L.records(cal, [r for r in first if r["kind"] == "solo"], "first_solo", "first co-run campaign: solo", "other days")
    L.records(
        cal,
        [r for r in first if r["kind"] == "corun"],
        "first_corun",
        "first co-run campaign: 54+54, 32+76",
        "other days",
    )


def rung3_other_workloads(L, cal, model, scan):
    """Returns the whole-layer records of the other MoE layers, for rung 4."""
    L.section("3 other workloads", "3  other workloads (never fitted)")
    whole = []
    paths = sorted(
        glob.glob(os.path.join(meas.DATA, f"sm_corun_holdout_L*_{model}.csv")),
        key=lambda p: int(re.search(r"_L(\d+)_", p).group(1)),
    )
    for path in paths:
        name = os.path.basename(path)
        lay = int(re.search(r"_L(\d+)_", name).group(1))
        h = meas.scan(model, name, layer=lay)
        for r in h:
            r["layer"] = lay
        for kind, tail in (("solo", "solo"), ("corun", "co-run")):
            L.records(
                cal,
                [r for r in h if r["kind"] == kind],
                f"{name[:-4]}_{kind}",
                f"MoE layer {lay}: {tail}",
                "new workloads",
            )
        whole += h
    rung3_pinned_layer(L, cal, model, scan)
    return whole


def rung3_pinned_layer(L, cal, model, scan):
    """The configuration a two-domain design executes -- both partitions pinned, a
    barrier after each stage -- measured end to end and never fitted.

    It is mostly a test of COMPOSITION. Where the training sweeps ran the same two
    partitions side by side on the same SM split at the same clock, the partitions are
    not new: what is new is the barriers, the epilogue and the event loop that put the
    layer together. The sub-rows say how many records are of each kind, and every case
    is listed, the worst one included."""
    if not os.path.exists(os.path.join(meas.DATA, f"pinned_layer_{model}.csv")):
        print(f"  {'pinned layer END TO END':44s} | not collected yet")
        return
    swept = {(r["cells"], r["bm"], r["n_cold"], r["split"], r["f"]) for r in scan if r["kind"] == "corun"}
    sh, groups = Shared(), defaultdict(lambda: ([], [], []))
    for r in meas.pinned(model):
        t, p = layer(cal, r["cells"], r["bm"], r["f"], "persistent", r["n_cold"], r["sm_light"], shared=sh)
        seen = (r["cells"], r["bm"], r["n_cold"], r["sm_light"], r["f"]) in swept
        L.points.append(("new workloads", "pinned_layer", "ms", r["ms"], t))
        L.points.append(("new workloads", "pinned_layer", "w", r["power"], p))
        for key in ("all", "seen" if seen else "new", (r["cells"], r["bm"], r["n_cold"])):
            te, pe, de = groups[key]
            te.append(np.log(t / r["ms"]) * 100)
            L.power(cal, r, p, None, None, pe, de)
    row = lambda key, label, g: L.row(key, label, g[0], g[1], dyn_err=g[2])
    row("pinned_layer", "pinned layer END TO END, six SM splits", groups["all"])
    row("pinned_layer_seen", "  the pair was also swept side by side", groups["seen"])
    row("pinned_layer_new", "  the pair was not", groups["new"])
    for key in sorted(k for k in groups if isinstance(k, tuple)):
        cells, bm, nc = key
        what = f"{cells * 64} tokens" if cells else "prefill"
        row(f"pinned_layer_c{cells}_bm{bm}_n{nc}", f"  {what}, BM{bm}, {nc} cold experts", groups[key])


def rung3b_other_models(L, cal):
    L.section("3b other models", "3b other MODELS: fitted on Qwen3-30B-A3B only, kernel config borrowed from it")
    found = False
    for m, desc in OTHER_MODELS:
        name = f"sm_corun_xmodel_{m}.csv"
        if not os.path.exists(os.path.join(meas.DATA, name)):
            continue
        found = True
        h = meas.scan(m, name)
        swept = [r for r in h if r["kind"] in ("solo", "corun")]
        for kind, tail in (("solo", "solo"), ("corun", "co-run")):
            L.records(
                cal,
                [r for r in swept if r["kind"] == kind and not fits_l2(r)],
                f"xmodel_{m}_{kind}",
                f"{desc}: {tail}",
                "other models",
            )
        resident = [r for r in swept if fits_l2(r)]
        if resident:
            L.records(
                cal,
                resident,
                f"xmodel_{m}_l2_resident",
                f"{desc}: a partition that fits in L2",
                "other models, L2-resident",
            )
        L.layers(
            cal,
            _whole(h),
            f"xmodel_{m}_layers",
            f"{desc}: whole layers, both kernels",
            "other models",
            Shared(),
            point_key=f"xmodel_{m}",
            model=m,
        )
    if not found:
        print(f"  {'(no data/sm_corun_xmodel_*.csv)':44s} |")


def _whole(recs):
    """The whole-layer records of a scan, with the kernel each ran on."""
    return [
        dict(r, kernel="queue" if r["kind"] == "queue" else "persistent", layer_kw=dict(layer_index=r.get("layer", 0)))
        for r in recs
        if r["kind"] in ("full", "queue")
    ]


def rung4_whole_layers(L, cal, model, scan, other, sh):
    L.section("4 whole layers", "4  whole layers through the event engine, fixed clock")
    # The whole layers of the training sweeps are in the fit: the persistent kernel's
    # epilogue is (whole layer) - (its launches), and the queue kernel is fitted on its
    # whole layers there and in the two sweeps below.
    groups = defaultdict(lambda: ([], [], []))
    for r in _whole(list(scan) + other):
        t, p = layer(cal, r["cells"], r["bm"], r["f"], r["kernel"], shared=sh, **r["layer_kw"])
        src = "other layers (never fitted)" if "layer" in r else "training sweeps"
        te, pe, de = groups[(r["kernel"], src)]
        te.append(np.log(t / r["ms"]) * 100)
        L.points.append(("whole layers", f"layer_{r['kernel']}", "ms", r["ms"], t))
        L.power(cal, r, p, "whole layers", f"layer_{r['kernel']}", pe, de)
    for (kern, src), (te, pe, de) in sorted(groups.items()):
        fit = "yes" if src == "training sweeps" else "no"
        L.row(f"layer_{kern}_{src.replace(' ', '_')}", f"{kern} kernel, {src}", te, pe, fit, de)
    queue = [dict(r, bm=128, kernel="queue") for r in meas.queue_sweeps(model)]
    L.layers(
        cal, queue, "layer_queue_sweeps", "queue kernel, 64-7744 tokens, 510-1410 MHz", "whole layers", sh, fit="yes"
    )
    held = [
        dict(cells=g["cells"], bm=128, f=1290, kernel="queue", ms=g["ms"], power=g["power"])
        for g in meas.governor(model)
        if g["mode"] == "lock1290" and g["held"]
    ]
    L.layers(cal, held, "layer_queue_big", "queue kernel, 512-12288 tokens, 1290 MHz", None, sh)
    if os.path.exists(os.path.join(meas.DATA, f"occupancy_{model}.csv")):
        for kern, label in (
            ("persistent", "persistent, BM64 and BM128, 512-7744 tokens"),
            ("queue", "queue kernel, 512-7744 tokens"),
        ):
            recs = [r for r in meas.occupancy(model) if r["kernel"] == kern and r["blocks"] <= 1]
            L.layers(
                cal,
                recs,
                "layer_occupancy_" + kern,
                label + " (never fitted)",
                "whole layers",
                sh,
                point_key="occupancy_" + kern,
            )
    pv = meas.pin_vs_queue(model)
    recs = [
        dict(cells=0, bm=128, f=1200, kernel=kern, ms=pv[case][0], power=pv[case][1])
        for case, kern in (("queue", "queue"), ("persistent_1blk_per_sm", "persistent"))
    ]
    L.layers(cal, recs, "layer_pin_vs_queue", "prefill at 1200 MHz, both kernels (Sep 17)", None, sh)


def rung5_governor(cal, model, sh):
    print("\n5  the stock governor: where it settles (queue kernel, never fitted)")
    print(f"  {'batch':>6} | {'measured':>26} | {'simulated':>26} | {'d clock':>7} {'d W':>6} {'d ms':>6}")
    gov = []
    for g in meas.governor(model):
        if g["mode"] != "governor":
            continue
        b = build([dict(phase="decode", cells=g["cells"], layers=[0], repeat=int(6000 / g["ms"]))])
        e = Engine(Hardware(), cal, Kernel("queue", 128), b, Governor(), shared=sh).run()
        tail = [x for x in e.layers if x["t_start"] > e.t - 3000]
        ms = float(np.mean([x["ms"] for x in tail]))
        w = sum(x["mj"] for x in tail) / sum(x["ms"] for x in tail)
        seg = [(b_ - a, f[0]) for a, b_, _, f, _ in e.trace if a > e.t - 3000]
        clk = sum(d * f for d, f in seg) / sum(d for d, _ in seg)
        gov.append(
            dict(
                tokens=g["tokens"],
                meas_clock=g["clock"],
                meas_w=g["power"],
                meas_ms=g["ms"],
                sim_clock=round(clk, 1),
                sim_w=round(w, 1),
                sim_ms=round(ms, 4),
            )
        )
        print(
            f"  {g['tokens']:>6} | {g['clock']:>6.0f} MHz {g['power']:>6.1f} W {g['ms']:>7.4f} ms | "
            f"{clk:>6.0f} MHz {w:>6.1f} W {ms:>7.4f} ms | {clk - g['clock']:>+7.0f} "
            f"{w - g['power']:>+6.1f} {(ms / g['ms'] - 1) * 100:>+5.1f}%"
        )
    return gov


def rung6_traces():
    """The summaries `sim.validate_trace` left in sim/out."""
    print("\n6  a load that moves, under the governor (python3 -m sim.validate_trace)")
    tr = {}
    for tag in TRACES:
        path = os.path.join(HERE, "out", f"trace_{tag}.json")
        if os.path.exists(path):
            s = tr[tag] = read_json(path)["summary"]
            same = s.get("clock_rms_mhz_same_window")
            print(
                f"  {tag:9s} {s['segments']:3d} segments, {s['runs']} runs: latency rms "
                f"{s['latency_rms_pct']:.2f}%, power rms {s['power_rms_pct']:.2f}%, clock rms "
                f"{s['clock_rms_mhz']:.0f} MHz"
                + (f" ({same:.0f} over the same window)" if same is not None else "")
                + ", energy of each run "
                + " ".join(f"{x:+.1f}%" for x in s["energy_err_pct"])
            )
    return tr


def rung7_consistency(cal, scan):
    print("\n7  self-consistency")
    cons = {}
    # (a) two V/f domains at one clock must equal one domain with pinned partitions
    worst = 0.0
    for cells, bm, nc, cold_sms in ((32, 64, 64, 32), (16, 16, 64, 48), (0, 128, 16, 16)):
        for f in (930, 1290, 1410):
            a = layer(cal, cells, bm, f, "persistent", nc, cold_sms, shared=Shared())
            b = layer(cal, cells, bm, [f, f], "persistent", nc, hw=two_domain(cold_sms), shared=Shared())
            worst = max(worst, abs(a[0] / b[0] - 1), abs(a[1] / b[1] - 1))
    cons["two_domains_same_clock_vs_one_domain"] = worst
    print(
        f"  two domains at one clock vs one domain, pinned: worst relative difference "
        f"{worst:.1e} (9 cases, time and power)"
    )
    # (b) energy is the integral of the trace
    b = build([dict(phase="decode", cells=32, layers=[0], repeat=200)])
    e = Engine(two_domain(32), cal, Kernel("persistent", 64, 64), b, Fixed([1050, 1290])).run()
    integ = sum((t1 - t0) * p for t0, t1, p, *_ in e.trace)
    lay = sum(x["mj"] for x in e.layers)
    cons["energy_trace_vs_total"] = abs(integ / e.energy - 1)
    cons["energy_layers_vs_total"] = abs(lay / e.energy - 1)
    print(
        f"  energy: trace integral vs total {abs(integ / e.energy - 1):.1e}, "
        f"sum of layers vs total {abs(lay / e.energy - 1):.1e}"
    )
    # (c) the tabulated job times against the exact window solve
    d = []
    for r in [x for x in scan if x["kind"] == "corun"][::7]:
        a, _ = corun(cal, r)
        b_, _ = corun(cal, r, exact=True)
        d += [a[0] / b_[0] - 1, a[1] / b_[1] - 1]
    cons["table_vs_exact_rms"] = float(np.sqrt(np.mean(np.square(d))))
    cons["table_vs_exact_worst"] = float(np.max(np.abs(d)))
    print(
        f"  tabulated vs exact solve, {len(d)} co-run times: rms "
        f"{cons['table_vs_exact_rms'] * 100:.2f}%, worst {cons['table_vs_exact_worst'] * 100:.2f}%"
    )
    ok = worst < 1e-9 and cons["energy_trace_vs_total"] < 1e-9
    print("  PASS" if ok else "  FAIL")
    return cons, ok


# ---------------------------------------------------------------------- driver ----
def run(model="qwen", cal=None):
    cal = cal or Calib.load()
    L, sh = Ladder(), Shared()
    rung0_decomposition(model)
    scan = rung1_fitted(L, cal, model)
    rung2_other_days(L, cal, model)
    other = rung3_other_workloads(L, cal, model, scan)
    rung3b_other_models(L, cal)
    rung4_whole_layers(L, cal, model, scan, other, sh)
    out = dict(L.rows)
    out["governor"] = rung5_governor(cal, model, sh)
    out["traces"] = rung6_traces()
    out["consistency"], ok = rung7_consistency(cal, scan)

    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    with open(os.path.join(HERE, "out", "validation.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    with open(os.path.join(HERE, "out", "validation_points.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rung", "group", "quantity", "measured", "simulated"])
        w.writerows([(a, b, c, round(m, 6), round(v, 6)) for a, b, c, m, v in L.points])
    print(f"\nwrote {os.path.join(HERE, 'out', 'validation.json')}")
    return out, ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen")
    a = ap.parse_args()
    _, ok = run(a.model)
    raise SystemExit(0 if ok else 1)
