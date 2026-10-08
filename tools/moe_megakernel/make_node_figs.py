#!/usr/bin/env python3
"""Figures of the node engine (GEMM + all-reduce overlap, multi-GPU steps).

  fig29_nccl_occupancy   how many SMs a GEMM loses per NCCL CTA, by launch order   (probe)
  fig30_node_validation  simulated against measured, the earlier overlapped runs
  fig31_overlap_spans    one overlapped iteration: measured and simulated spans
  fig32_unbalanced_step  an unbalanced training step: light GPUs slowed, measured
                         and simulated

  source env.sh && python3 make_node_figs.py [--only 29,30]

Every panel is a measurement, or the simulator beside a measurement. PNG only.
"""
import argparse
import collections
import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import figstyle as fs
from sim.calib import Calib
from sim.fit_kernels import NEW, old_rows, span_rows
from sim.kernels import Kernels
from sim.validate_overlap import Scorer, program_rows

SIM, MEAS = fs.GEMM, fs.COMM  # two series per panel: the simulator and the measurement


def fig29():
    """The probe: SMs a cuBLAS-class block cannot reach while c CTAs are resident."""
    with open(os.path.join(NEW, "nccl_occupancy.csv")) as fh:
        rows = [r for r in csv.DictReader(fh) if r["valid"] == "True"]
    lost, dur = collections.defaultdict(list), collections.defaultdict(list)
    for r in rows:
        if r["smem"] == "147456":
            lost[(r["order"], int(r["ctas"]))].append(int(r["lost"]))
        dur[(r["order"], int(r["ctas"]))].append(float(r["comm_end"]) - float(r["comm_start"]))
    cs = sorted({c for _, c in lost})
    fig, ax = plt.subplots(1, 2, figsize=(fs.TEXT_W, 2.15), gridspec_kw=dict(wspace=0.36))
    a = ax[0]
    for order, color, mk, label in (("comm_first", MEAS, "o", "all-reduce launched first"),
                                    ("gemm_first", SIM, "s", "all-reduce launched behind a GEMM")):
        x = [c for c in cs if (order, c) in lost]
        a.plot(x, [np.median(lost[(order, c)]) for c in x], mk + "-", color=color, ms=3.2, lw=0.9, label=label,
               clip_on=False, zorder=3)
    xx = np.array([0, 54])
    a.plot(xx, 2 * xx, color=fs.MUTED, lw=0.5, ls=(0, (4, 3)), zorder=1)
    a.plot(xx, xx, color=fs.MUTED, lw=0.5, ls=(0, (4, 3)), zorder=1)
    a.text(36, 82, "2 per CTA", color=fs.MUTED, fontsize=7, ha="right")
    a.text(50, 40, "1 per CTA", color=fs.MUTED, fontsize=7, ha="left")
    a.set_xlabel("NCCL CTAs resident")
    a.set_ylabel("SMs a cuBLAS GEMM\nblock cannot use")
    a.set_xlim(0, 66)
    a.set_ylim(0, 128)
    a.set_yticks([0, 27, 54, 81, 108])
    a.legend(loc="upper left", fontsize=6.5, handlelength=1.6, borderaxespad=0.0, bbox_to_anchor=(0.0, 1.04))
    fs.despine(a)
    b = ax[1]
    x = [c for c in cs if ("gemm_first", c) in dur and ("comm_first", c) in dur]
    b.plot(x, [np.median(dur[("gemm_first", c)]) / np.median(dur[("comm_first", c)]) for c in x], "s-", color=SIM,
           ms=3.2, lw=0.9, clip_on=False, zorder=3)
    b.axhline(1.0, color=fs.MUTED, lw=0.5, ls=(0, (4, 3)))
    b.set_xlabel("NCCL CTAs resident")
    b.set_ylabel("all-reduce time, launched behind\na GEMM / launched first")
    b.set_xlim(0, 66)
    b.set_ylim(0.95, 1.4)
    fs.despine(b)
    return fs.save(fig, "fig29_nccl_occupancy", os.path.join(HERE, "figs"))


def fig30(scorer):
    """Earlier overlapped runs: simulated against measured iteration time and power."""
    old = old_rows()
    held = scorer.overlap(old)
    wall = scorer.overlap(old, held=False)
    fig, ax = plt.subplots(1, 3, figsize=(fs.TEXT_W, 1.95), gridspec_kw=dict(wspace=0.42))
    for a, key, sim, unit, lim in ((ax[0], "ms", "t_sim", "iteration (ms)", None),
                                   (ax[1], "w", "w_sim", "power (W per GPU)", (60, 420))):
        x, y = np.array([o[key] for o in held]), np.array([o[sim] for o in held])
        xw, yw = np.array([o[key] for o in wall]), np.array([o[sim] for o in wall])
        lo, hi = (min(x.min(), y.min()) * 0.8, max(x.max(), y.max()) * 1.25) if lim is None else lim
        a.plot([lo, hi], [lo, hi], color=fs.MUTED, lw=0.5, zorder=1)
        a.scatter(x, y, s=2.2, color=MEAS, alpha=0.35, linewidths=0, zorder=2, label=f"clock held ({len(x)})")
        a.scatter(xw, yw, s=4, color=SIM, alpha=0.8, linewidths=0, zorder=3, label=f"throttled ({len(xw)})")
        if lim is None:
            a.set_xscale("log")
            a.set_yscale("log")
        a.set_xlim(lo, hi)
        a.set_ylim(lo, hi)
        a.set_xlabel(f"measured {unit}")
        a.set_ylabel(f"simulated {unit}")
        fs.despine(a, grid_axis="both")
    ax[0].legend(loc="upper left", fontsize=6.3, handletextpad=0.1, borderaxespad=0.1, markerscale=2.2)
    c = ax[2]
    e = np.sort(np.abs([o["et"] for o in held]))
    p = np.sort(np.abs([o["ep"] for o in held]))
    q = np.arange(1, len(e) + 1) / len(e) * 100
    c.plot(e, q, color=MEAS, lw=1.0, label="iteration time")
    c.plot(p, q, color=SIM, lw=1.0, label="power")
    c.set_xlim(0, 12)
    c.set_ylim(0, 100)
    c.set_xlabel("simulation error (%)", x=0.45)
    c.set_ylabel("runs within it (%)")
    c.legend(loc="lower right", fontsize=6.5)
    fs.despine(c, grid_axis="both")
    return fs.save(fig, "fig30_node_validation", os.path.join(HERE, "figs"))


def fig31(scorer):
    """One overlapped iteration, measured spans above simulated spans, in both launch
    orders. Top: a cell the rules explain. Bottom: the case they do not -- a GEMM with
    unsplit K = 8192 that outlasts the collective."""
    cases = [("GEMM 4096 x 4096 x 4096 beside a 256 MiB all-reduce, 16 CTAs, 1200 MHz", 16, 1200, 256, "4096x4096x4096"),
             ("GEMM 8192 x 8192 x 8192 beside 64 MiB, 8 CTAs, 1200 MHz: the case the rules do not explain",
              8, 1200, 64, "8192x8192x8192")]
    cal = [r for r in span_rows("cal") if r["rank"] == 0]
    fig, ax = plt.subplots(2, 2, figsize=(fs.TEXT_W, 2.75), gridspec_kw=dict(wspace=0.1, hspace=1.05), sharey=True)
    for i, (label, ctas, clock, mib, shape) in enumerate(cases):
        for a, mode, order in ((ax[i][0], "concurrent", "all-reduce launched first"),
                               (ax[i][1], "concurrent_gf", "GEMM launched first")):
            m = [r for r in cal if r["mode"] == mode and (r["ctas"], r["clock_req"], r["mib"], r["shape"])
                 == (ctas, clock, mib, shape)][0]
            s = scorer.cell(mode, per_gpu=True, shape=shape, mib=mib, ctas=ctas, clock=clock)
            hi = max(m["gemm_end"], m["comm_end"], s["gemm"][1], s["comm"][1])
            lanes = [((m["gemm_start"], m["gemm_end"]), (m["comm_start"], m["comm_end"])), (s["gemm"], s["comm"])]
            for j, (g, c) in enumerate(lanes):
                y = 1 - j
                a.barh(y + 0.18, g[1] - g[0], left=g[0], height=0.32, color=fs.GEMM, zorder=3)
                a.barh(y - 0.18, c[1] - c[0], left=c[0], height=0.32, color=fs.COMM, zorder=3)
                a.text(g[1] + 0.012 * hi, y + 0.18, f"{g[1]:.2f}", va="center", fontsize=6.5, color=fs.INK)
                a.text(c[1] + 0.012 * hi, y - 0.18, f"{c[1]:.2f}", va="center", fontsize=6.5, color=fs.INK)
            a.set_yticks([1, 0])
            a.set_yticklabels(["measured", "simulated"])
            a.set_ylim(-0.55, 1.55)
            a.set_xlim(0, hi * 1.16)
            a.set_xlabel("ms after the launch gate", labelpad=1.5)
            a.set_title(order, fontsize=7, pad=3, loc="left", color=fs.MUTED)
            fs.despine(a, grid_axis="x")
        ax[i][0].text(0.0, 1.36, label, transform=ax[i][0].transAxes, fontsize=7.5, color=fs.INK, va="bottom")
    h = [plt.Rectangle((0, 0), 1, 1, color=fs.GEMM), plt.Rectangle((0, 0), 1, 1, color=fs.COMM)]
    fig.legend(h, ["GEMM", "all-reduce"], loc="upper right", fontsize=6.5, ncol=2, bbox_to_anchor=(0.9, 1.08),
               borderaxespad=0.0)
    return fs.save(fig, "fig31_overlap_spans", os.path.join(HERE, "figs"))


def fig32(scorer, ctas=16, name="ddp_overlap_b8192+3x2048"):
    """An unbalanced training step: step time, node power and energy per step as the
    three light GPUs are slowed (GPU 0 at 1200 MHz), measured and simulated; the dashed
    line is the same step with no clock lock, measured."""
    from sim import node

    rows = program_rows("train")
    cells = [c for c in rows if c["name"] == name and c["ctas"] == ctas and isinstance(c["clock"], list)
             and c["clock"][0] == 1200]
    cells.sort(key=lambda c: -c["clock"][1])
    free = [c for c in rows if c["name"] == name and c["ctas"] == ctas and c["clock"] == 0]
    if not cells:
        return ()
    f_light = [c["clock"][1] for c in cells]
    node_w = lambda c: sum(c["main"][g]["power_w"] for g in range(4))
    t_m, p_m = [c["main"][0]["t_loop"] for c in cells], [node_w(c) for c in cells]
    sims = [node.program_cell(scorer.k, scorer.cal, c["ops"], ctas=ctas, clock=c["clock"]) for c in cells]
    t_s, p_s = [s["t_loop"] for s in sims], [sum(s["power"]) for s in sims]
    joule = lambda t, p: [a * b / 1e3 for a, b in zip(t, p, strict=True)]
    base = (free[0]["main"][0]["t_loop"], node_w(free[0])) if free else None
    fig, ax = plt.subplots(1, 3, figsize=(fs.TEXT_W, 1.9), gridspec_kw=dict(wspace=0.42))
    x = np.arange(len(cells))
    panels = ((ax[0], t_m, t_s, "step time (ms)", base and base[0]), (ax[1], p_m, p_s, "node power (W)", base and base[1]),
              (ax[2], joule(t_m, p_m), joule(t_s, p_s), "energy per step (J)", base and base[0] * base[1] / 1e3))
    for a, m, s, lab, ref in panels:
        if ref:
            a.axhline(ref, color=fs.MUTED, lw=0.7, ls=(0, (4, 3)), zorder=2)
        a.plot(x, m, "o-", color=MEAS, ms=3.2, lw=0.9, label="measured", zorder=3)
        a.plot(x, s, "s--", color=SIM, ms=3.0, lw=0.9, label="simulated", zorder=3)
        a.set_xticks(x)
        a.set_xticklabels([str(f) for f in f_light])
        a.set_ylabel(lab)
        a.set_ylim(0, max(max(m), max(s), ref or 0) * 1.2)
        a.set_xlim(-0.35, len(cells) - 0.65)
        fs.despine(a)
    ax[1].text(len(cells) - 0.7, base[1] * 1.03, "no clock lock (measured)", color=fs.MUTED, fontsize=6.5, ha="right",
               va="bottom") if base else None
    ax[0].legend(loc="lower left", fontsize=6.5, borderaxespad=0.3)
    fig.supxlabel("clock of the three light GPUs (MHz); GPU 0, four times their batch, at 1200 MHz", fontsize=8, y=-0.08)
    return fs.save(fig, "fig32_unbalanced_step", os.path.join(HERE, "figs"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    fs.use()
    want = lambda n: not a.only or str(n) in a.only.split(",")
    scorer = Scorer(Kernels.load(), Calib.load())
    for n, f in ((29, fig29), (30, lambda: fig30(scorer)), (31, lambda: fig31(scorer)), (32, lambda: fig32(scorer))):
        if want(n):
            out = f()
            print(f"fig{n}:", *out)


if __name__ == "__main__":
    main()
