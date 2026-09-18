#!/usr/bin/env python3
"""fig12 -- the honest comparison: does a second domain move the energy-latency frontier?

The earlier table put "-30% from lowering the single clock" next to "-2.6% from two
domains" as if they were alternatives. They are not. Lowering the clock buys energy WITH
LATENCY -- it moves along the existing trade-off. Two domains is a matched-latency
comparison, so it moves the trade-off itself.

Drawn properly: the single-clock grouped GEMM traces out a frontier as its clock sweeps.
The question is whether the two-domain design lies BELOW that frontier, and by how much.
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import figstyle as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict_expert_dvfs import tiles_of, baseline, best_multi, GRID, SM

HERE = os.path.dirname(os.path.abspath(__file__))
Z = np.load(os.path.join(HERE, "data", "routing_olmoe.npz"))
CASES = [("prefill_b16", "prefill B=16", F.GEMM),
         ("prefill_b128", "prefill B=128", F.COMM)]

F.use()
fig, axes = plt.subplots(1, 2, figsize=(F.TEXT_W, 2.5),
                         gridspec_kw=dict(wspace=0.34))
fig.subplots_adjust(bottom=0.24)

for ax, (key, lab, col) in zip(axes, CASES):
    C = Z[key].reshape(-1, 64).astype(float)
    fs = [f for f in GRID if f >= 600]
    # one layer-sample is NOT representative: the gain is a sawtooth in the tile count,
    # so it swings layer to layer. Show the median and the full spread.
    E1 = np.full((len(C), len(fs)), np.nan)
    E2 = np.full_like(E1, np.nan)
    LT = np.full(len(fs), np.nan)
    for r, row in enumerate(C):
        t = np.sort(tiles_of(row)[row > 0])
        N = int(t.sum())
        for c, f in enumerate(fs):
            j = int(np.searchsorted(GRID, f))
            T, Eb = baseline(N, j)
            bb = best_multi(t, T, 2)
            if bb is None:
                continue
            E1[r, c] = Eb; E2[r, c] = bb[0]
            if r == 0:
                LT[c] = T
    ref = np.nanmedian(E1[:, -1])   # 1380 MHz, the fastest point
    L = LT / LT.min()               # min latency = highest clock
    m1 = np.nanmedian(E1, 0) / ref * 100
    m2 = np.nanmedian(E2, 0) / ref * 100
    gap = (E2 - E1) / E1 * 100
    F.despine(ax)
    ax.plot(L, m1, "-", lw=1.6, color=F.MUTED, label="grouped GEMM, one clock", zorder=4)
    ax.plot(L, m2, "-", lw=1.6, color=col, label="+ two frequency domains", zorder=5)
    ax.fill_between(L, np.nanmin(E2, 0) / ref * 100, np.nanmax(E2, 0) / ref * 100,
                    color=col, alpha=0.20, lw=0, zorder=3)
    ax.annotate("1380 MHz", xy=(1.0, m1[-1]), xycoords="data",
                xytext=(0.30, 0.86), textcoords="axes fraction",
                fontsize=6, color=F.INK, ha="center",
                arrowprops=dict(arrowstyle="->", lw=0.6, color=F.INK))
    k = int(np.nanargmin(m1))
    # axes-fraction placement: the minimum sits at the bottom of the panel, so a
    # data-coordinate offset kept pushing this under the tick labels
    ax.annotate(f"cheapest single clock\n{fs[k]:.0f} MHz", xy=(L[k], m1[k]),
                xycoords="data", xytext=(0.58, 0.28), textcoords="axes fraction",
                fontsize=6, color=F.INK, ha="center",
                arrowprops=dict(arrowstyle="->", lw=0.6, color=F.INK))
    g = np.nanmedian(gap, 0)
    ax.set_title(f"{lab}\ngap: median {np.nanmin(g):+.1f}%, "
                 f"best layer {np.nanmin(gap):+.1f}%", fontsize=7.2, pad=3)
    ax.set_xlabel("latency, relative to 1380 MHz")
    ax.set_ylabel("energy, relative to 1380 MHz (%)")
    ax.legend(fontsize=6, loc="upper right", handlelength=1.5)

fig.text(0.5, -0.02, "The grey curve IS the existing trade-off: sweeping the single "
         "clock buys energy by spending latency, and its minimum (1050 MHz, 1.37x the "
         "time) is the\ncheapest the workload can be run at all. Moving along that curve "
         "is NOT the same kind of thing as the coloured curve, which is the same "
         "workload with two frequency\ndomains at every latency the grey curve reaches. "
         "The vertical GAP is what a second domain actually adds: median 3.5% (B=16) and "
         "1.0% (B=128), and it swings\nlayer to layer because it is a sawtooth in the "
         "tile count -- the band is the full spread over 16 layer-samples. Predicted "
         "from the fitted model, not measured.",
         ha="center", va="top", fontsize=5.6, color=F.INK, linespacing=1.5)
print(F.save(fig, "fig12_pareto"))
