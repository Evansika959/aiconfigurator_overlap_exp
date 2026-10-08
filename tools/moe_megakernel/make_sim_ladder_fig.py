#!/usr/bin/env python3
"""fig26 -- the control groups on one plane: what each ingredient of a two-domain design
is worth, at every latency.

Reads sim/out/study_<workload>_fronts.csv (python3 -m sim.study --cells N). Each curve is
a Pareto front: the least energy per layer a configuration reaches at or under a given
layer time, over every clock setting that keeps board power under TDP.

  black    vLLM's queue kernel: B0 the stock governor (dot), B1 the best single clock
  blue     the persistent kernel as measured, one block per SM:
             dotted  B2 unpartitioned      solid  B3 pinned, one clock
             dashed  B4 pinned, a clock per V/f domain
  orange   a pinned kernel with vLLM's per-SM costs, which nobody has built:
             solid   B3*                    dashed B4*

A second rail is worth the gap between a solid curve and the dashed one of the same
colour. The kernel is worth the gap between colours.
"""
import argparse, csv, os, sys
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs

HERE = os.path.dirname(os.path.abspath(__file__))
STYLE = {"B1": (fs.INK, "-", 1.1), "B2": (fs.COMM, (0, (1, 1.4)), 1.0), "B3": (fs.COMM, "-", 1.1),
         "B4": (fs.COMM, (0, (3.2, 1.6)), 1.1), "B3*": (fs.GEMM, "-", 1.1),
         "B4*": (fs.GEMM, (0, (3.2, 1.6)), 1.1)}
LABEL = {"B0": "B0  queue kernel, stock governor", "B1": "B1  queue kernel, one clock",
         "B2": "B2  persistent, unpartitioned", "B3": "B3  pinned, one clock",
         "B4": "B4  pinned, a clock per domain",
         "B3*": "B3*  pinned, vLLM per-SM costs", "B4*": "B4*  same, a clock per domain"}


def load(stem):
    F = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(HERE, "sim", "out", f"study_{stem}_fronts.csv"))):
        F[r["name"]].append((float(r["ms"]), float(r["mj"])))
    return {k: np.array(sorted(v)) for k, v in F.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", default="decode2048,prefill")
    a = ap.parse_args()
    panels = a.panels.split(",")
    fs.use()
    W, H = fs.TEXT_W, 2.75
    fig = plt.figure(figsize=(W, H))
    wid = (W - 0.52 - 0.10 - 0.50 * (len(panels) - 1)) / len(panels)
    for i, stem in enumerate(panels):
        ax = fig.add_axes([(0.52 + i * (wid + 0.50)) / W, 0.40 / H, wid / W, 1.62 / H])
        F = load(stem)
        b0 = F["B0"][0]
        for name in ("B2", "B3", "B4", "B3*", "B4*", "B1"):
            if name not in F:
                continue
            col, ls, lw = STYLE[name]
            v = F[name]
            ax.plot(v[:, 0], v[:, 1], color=col, ls=ls, lw=lw, zorder=3 if "*" in name else 4)
        ax.plot([b0[0]], [b0[1]], marker="o", ms=4.2, color=fs.INK, zorder=6,
                markeredgecolor="white", markeredgewidth=0.6)
        ax.annotate("B0", b0, xytext=(5, 7), textcoords="offset points", ha="left",
                    fontsize=7, color=fs.INK)
        ax.set_xlim(b0[0] * 0.96, b0[0] * 1.62)
        ys = np.concatenate([v[(v[:, 0] <= b0[0] * 1.62), 1] for v in F.values()])
        ax.set_ylim(ys.min() * 0.94, max(ys.max(), b0[1]) * 1.05)
        ax.set_xlabel("time per layer (ms)")
        if i == 0:
            ax.set_ylabel("energy per layer (mJ)")
        t = {"prefill": "prefill, 7744 tokens"}.get(stem, f"decode, {stem[6:]} tokens")
        ax.set_title(f"({'ab'[i]}) {t}", loc="left", fontsize=8, weight="bold", pad=3)
        fs.despine(ax)
    hs = [plt.Line2D([], [], marker="o", ls="", ms=4, color=fs.INK)] + \
         [plt.Line2D([], [], color=STYLE[k][0], ls=STYLE[k][1], lw=STYLE[k][2])
          for k in ("B1", "B2", "B3", "B4", "B3*", "B4*")]
    fig.legend(hs, [LABEL[k] for k in ("B0", "B1", "B2", "B3", "B4", "B3*", "B4*")],
               loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0), columnspacing=1.0,
               handlelength=2.0)
    out = fs.save(fig, "fig26_sim_ladder")
    print("wrote", *out)


if __name__ == "__main__":
    main()
