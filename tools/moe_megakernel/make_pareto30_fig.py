#!/usr/bin/env python3
"""fig13 -- the frontier inside a 30% latency budget, which is the region that matters.

The full sweep runs to 2.3x latency, but nobody ships that. Restricted to +30%:

  * the best single clock is 1065 MHz at 1.296x, worth -31.2% against 1380
  * the true energy optimum (1050 MHz) sits at 1.314x, just outside -- so the budget
    costs 1 percentage point of energy, not more
  * whatever a second frequency domain adds is the vertical gap between the curves, and
    it is largest at the LEFT end, where the SLO is tightest and the single clock has no
    room to fall

Left panel is the frontier; right panel is the gap on its own, because at this zoom the
two frontier curves are close enough to be hard to separate by eye.
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import figstyle as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict_expert_dvfs import tiles_of, baseline, best_multi, GRID

HERE = os.path.dirname(os.path.abspath(__file__))
Z = np.load(os.path.join(HERE, "data", "routing_olmoe.npz"))
CASES = [("prefill_b16", "prefill B=16", F.GEMM),
         ("prefill_b128", "prefill B=128", F.COMM)]
LMAX = 1.30

F.use()
fig, (a1, a2) = plt.subplots(1, 2, figsize=(F.TEXT_W, 2.45),
                             gridspec_kw=dict(wspace=0.34))
fig.subplots_adjust(bottom=0.24)

for key, lab, col in CASES:
    C = Z[key].reshape(-1, 64).astype(float)
    fs = [f for f in GRID if f >= 1000]
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
    L = LT / LT.min()
    m = L <= LMAX
    ref = np.nanmedian(E1[:, np.argmin(LT)])
    m1 = np.nanmedian(E1, 0) / ref * 100
    m2 = np.nanmedian(E2, 0) / ref * 100
    gap = (E2 - E1) / E1 * 100
    # the grey baseline curves for the two workloads lie on top of each other, so label
    # only once; the coloured line is that workload WITH the second domain
    a1.plot(L[m], m1[m], "-", lw=2.4, color=F.MUTED, zorder=4,
            label="one clock (both workloads)" if key == CASES[0][0] else None)
    a1.plot(L[m], m2[m], "-", lw=1.5, color=col, zorder=5, label=f"+ 2 domains, {lab.split()[-1]}")
    g = np.nanmedian(gap, 0)
    a2.plot(L[m], -g[m], "-", lw=1.7, color=col, zorder=5, label=lab)
    a2.fill_between(L[m], -np.nanpercentile(gap, 90, axis=0)[m],
                    -np.nanpercentile(gap, 10, axis=0)[m], color=col, alpha=0.20,
                    lw=0, zorder=3)

for ax in (a1, a2):
    F.despine(ax)
    ax.set_xlabel("latency, relative to 1380 MHz")
    ax.set_xlim(1.0, LMAX)
    ax.axvline(1.296, color=F.INK, lw=0.7, ls=(0, (3, 2)), zorder=2)
a1.text(1.291, 88, "best single clock in budget:\n1065 MHz, $-$31%", fontsize=6,
        ha="right", va="top", color=F.INK)
a1.set_ylabel("energy, relative to 1380 MHz (%)")
a1.set_title("(a) the frontier inside a 30% latency budget", fontsize=7.4, pad=3)
a1.legend(fontsize=6, loc="lower left", handlelength=1.5)
a2.set_ylabel("energy the 2nd domain adds (%)")
a2.set_title("(b) the gap on its own", fontsize=7.4, pad=3)
a2.legend(fontsize=6, loc="upper right", handlelength=1.5)
a2.axhline(0, color=F.INK, lw=0.6)

fig.text(0.5, -0.02, "Restricted to the +30% latency band. The grouped GEMM's own optimum "
         "(1050 MHz) falls at 1.314x, just outside, so the budget costs 1 percentage "
         "point of\nenergy -- 31.2% instead of 32.2%. The second domain's contribution is "
         "the (b) curve: biggest at the tight end where the single clock cannot fall, "
         "shrinking to\nnothing once the budget lets it reach the knee. Band is the "
         "10th-90th percentile over 16 layer-samples; the swing is the tile-count "
         "sawtooth. Predicted, not measured.",
         ha="center", va="top", fontsize=5.6, color=F.INK, linespacing=1.5)
print(F.save(fig, "fig13_pareto_30pct"))
