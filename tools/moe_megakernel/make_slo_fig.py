#!/usr/bin/env python3
"""fig6 -- the second V/f domain is real money, and a shared queue takes it first.

There IS large energy between the knee and 1410: kappa rises 71-78% across it, so any
expert forced up there costs far more per operation. The question is who collects it.

x is how tight the SLO is, expressed as the clock a PINNED partition's critical expert
must run at to meet the deadline. Both curves are measured against `one clock, pinned`,
the design that has to run every expert at that clock.

  two V/f domains, pinned    let the light experts fall back to the knee -- worth up to 39%
  one clock, work-conserving no pinning at all; a shared queue spreads the total work over
                             all 108 SMs, so it meets the SAME deadline at 1050 MHz, the
                             knee, in every case tested -- worth up to 41%

The second domain is not wrong, it is second. Pinning is what forces the critical expert
above the knee in the first place; a megakernel that never pins never goes there.
"""
import csv, os, sys
import numpy as np
import matplotlib.pyplot as plt
import figstyle as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slo_tightness import allocate, kap, GRID, SM, KNEE, work_conserving, solve

HERE = os.path.dirname(os.path.abspath(__file__))
Z = np.load(os.path.join(HERE, "data", "routing_olmoe.npz"))
C = Z["prefill_b16"].reshape(-1, 64).astype(float)
FC = [1050, 1110, 1155, 1200, 1245, 1305, 1335, 1380]

two, wc = [], []
for f_crit in FC:
    a = b = c = 0.0
    for row in C:
        e1, e2, _ = solve(row, "proportional", f_crit)
        s_ = allocate(row, "proportional")
        T = (row / np.where(s_ > 0, s_, 1))[row > 0].max() / f_crit
        f_w = GRID[np.searchsorted(GRID, min(max(work_conserving(row, T), KNEE),
                                             GRID[-1]))]
        a += e1; b += e2; c += kap(f_w) * row.sum()
    two.append((b - a) / a * 100); wc.append((c - a) / a * 100)

F.use()
fig, ax = plt.subplots(figsize=(F.TEXT_W * 0.72, 2.4))
F.despine(ax)
ax.plot(FC, two, "o-", ms=3.6, lw=1.4, color=F.GEMM, mec="white", mew=0.5,
        label="two V/f domains, pinned")
ax.plot(FC, wc, "s-", ms=3.6, lw=1.4, color=F.COMM, mec="white", mew=0.5,
        label="one clock, work-conserving")
ax.axhline(0, color=F.INK, lw=0.6, ls=(0, (3, 2)))
ax.set_xlabel("SLO tightness: clock the pinned critical expert needs (MHz)")
ax.set_ylabel("energy vs. one clock, pinned (%)")
ax.set_title("Both collect the knee. The shared queue collects more.",
             fontsize=7.4, pad=4)
ax.legend(fontsize=6.2, loc="lower left", handlelength=1.5)
ax.annotate("loose SLO: everything\nalready sits at the knee,\nnothing to collect",
            xy=(1055, -0.6), xytext=(1120, -8.5), fontsize=5.8, color=F.MUTED,
            arrowprops=dict(arrowstyle="->", lw=0.6, color=F.MUTED))
fig.text(0.5, -0.03, "OLMoE prefill B=16, measured routing; $\\kappa$ from "
         "data/kappa_sweep.csv. Work is fixed, so $E = \\kappa(f)W$ and only the clock "
         "each expert lands on\nmatters. $\\kappa$(1410) is clamped to $\\kappa$(1380) "
         "because 1410 throttled, so both savings are understated at the right-hand end.",
         ha="center", va="top", fontsize=5.6, color=F.INK, linespacing=1.5)
print(F.save(fig, "fig6_slo_tightness"))
