#!/usr/bin/env python3
"""fig24 -- how far the simulator is from every measurement, fitted or not.

Reads sim/out/validation_points.csv (written by `python3 -m sim.validate`): one row per
measured quantity with what the simulator says for the same configuration.

(a) and (b) are parity plots, time on log axes because it spans 0.2-30 ms. A parity plot
hides a 5% error inside the marker, so (c) shows the ERRORS themselves, as the share of
points within a given distance of the measurement, one curve per rung of the validation
ladder:

  fitted          the calibration sweep itself (in-sample)
  other days      five earlier campaigns the fit never saw
  new workloads   batch sizes, tile sizes, cold-expert counts, clocks and a layer
                  that the fit never saw
  whole layers    end to end through the event engine, both kernels

Two hues carry "seen" against "never seen"; marker shape repeats the distinction so the
figure survives greyscale.

This is the calibration model, Qwen3-30B-A3B. The three other models `sim.validate` also
replays (rung 3b) are in the points file and deliberately not drawn: one of them, with
another hidden size, is 5-8% off, and RESULTS_SIM.md section 3 says so in a table.
"""
import csv, os, sys
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs

HERE = os.path.dirname(os.path.abspath(__file__))
RUNGS = [("fitted", "fitted", fs.MUTED, "o", 5),
         ("other days", "other days", fs.COMM, "s", 6),
         ("new workloads", "new workloads", fs.GEMM, "^", 7),
         ("whole layers", "whole layers", fs.INK, "D", 8)]


def load():
    P = defaultdict(lambda: ([], []))
    for r in csv.DictReader(open(os.path.join(HERE, "sim", "out", "validation_points.csv"))):
        m, s = float(r["measured"]), float(r["simulated"])
        P[(r["rung"], r["quantity"])][0].append(m)
        P[(r["rung"], r["quantity"])][1].append(s)
    return {k: (np.array(v[0]), np.array(v[1])) for k, v in P.items()}


def main():
    P = load()
    fs.use()
    W, H, S = fs.TEXT_W, 2.08, 1.30                 # inches; S = side of each square panel
    fig = plt.figure(figsize=(W, H))
    lefts = [0.44, 0.44 + S + 0.58, 0.44 + 2 * (S + 0.58) + 0.04]
    a1, a2, a3 = [fig.add_axes([x / W, 0.37 / H, S / W, S / H]) for x in lefts]
    rng = np.random.default_rng(3)

    for ax, q, lo, hi, log in ((a1, "ms", 0.18, 40.0, True), (a2, "w", 70.0, 430.0, False)):
        ax.plot([lo, hi], [lo, hi], color=fs.INK, lw=0.6, zorder=2)
        for k in (0.95, 1.05):
            ax.plot([lo, hi], [lo * k, hi * k], color=fs.INK, lw=0.4, ls=(0, (3, 2)), zorder=2)
        for name, _, col, mk, z in RUNGS:
            if (name, q) not in P:
                continue
            m, s = P[(name, q)]
            keep = rng.permutation(len(m))[:220]    # the fitted rung alone has 1200 points
            ax.scatter(m[keep], s[keep], s=5.5, marker=mk,
                       facecolor=col if name != "whole layers" else "none",
                       edgecolor=col, linewidth=0.45, alpha=0.8, zorder=z)
        if log:
            ax.set_xscale("log"); ax.set_yscale("log")
            for axis in (ax.xaxis, ax.yaxis):
                axis.set_major_locator(FixedLocator([0.3, 1, 3, 10, 30]))
                axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
                axis.set_minor_formatter(NullFormatter())
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        fs.despine(ax, grid_axis=None)
    a1.set_xlabel("measured (ms)"); a1.set_ylabel("simulated (ms)")
    a2.set_xlabel("measured (W)"); a2.set_ylabel("simulated (W)")
    a2.set_xticks([100, 200, 300, 400]); a2.set_yticks([100, 200, 300, 400])
    a2.text(0.97, 0.04, "dashed: $\\pm$5%", transform=a2.transAxes, ha="right", va="bottom",
            color=fs.MUTED, fontsize=7)

    # (c) the errors themselves: share of points within a given distance
    x = np.linspace(0, 15, 301)
    for name, lab, col, mk, z in RUNGS:
        for q, ls in (("ms", "-"), ("w", (0, (3, 1.5)))):
            if (name, q) not in P:
                continue
            m, s = P[(name, q)]
            e = np.abs(s / m - 1) * 100
            a3.plot(x, [(e <= v).mean() * 100 for v in x], color=col, lw=1.0, ls=ls, zorder=z)
    a3.set_xlim(0, 15); a3.set_ylim(0, 100)
    a3.set_xticks([0, 5, 10, 15]); a3.set_yticks([0, 25, 50, 75, 100])
    a3.set_xlabel("distance from measurement (%)")
    a3.set_ylabel("points within (%)")
    fs.despine(a3)
    a3.text(0.96, 0.05, "solid: time\ndashed: power", transform=a3.transAxes, ha="right",
            va="bottom", fontsize=7, color=fs.INK, linespacing=1.25)
    for ax, t in ((a1, "(a) time per pass or layer"), (a2, "(b) board power"),
                  (a3, "(c) how far off")):
        ax.set_title(t, loc="left", fontsize=8, weight="bold", pad=3)
    hs = [plt.Line2D([], [], marker=mk, ls="", ms=3.2,
                     markerfacecolor=col if name != "whole layers" else "none",
                     markeredgecolor=col, markeredgewidth=0.5,
                     label=f"{lab} ({sum(len(P[(name, q)][0]) for q in ('ms', 'w') if (name, q) in P)})")
          for name, lab, col, mk, z in RUNGS if (name, "ms") in P]
    fig.legend(handles=hs, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0),
               handletextpad=0.2, columnspacing=1.4)
    out = fs.save(fig, "fig24_sim_validation")
    for name, *_ in RUNGS:
        for q in ("ms", "w"):
            if (name, q) in P:
                m, s = P[(name, q)]
                e = (s / m - 1) * 100
                print(f"  {name:14s} {q:2s} n={len(m):5d}  rms {np.sqrt(np.mean(e ** 2)):.2f}%  "
                      f"within 5%: {(np.abs(e) <= 5).mean() * 100:.0f}%  within 10%: "
                      f"{(np.abs(e) <= 10).mean() * 100:.0f}%")
    print("wrote", *out)


if __name__ == "__main__":
    main()
