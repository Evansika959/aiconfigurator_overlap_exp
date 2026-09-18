#!/usr/bin/env python3
"""fig8 -- where a two-domain split DOES beat one clock, and why.

I argued from Jensen that a split can never win. That argument assumed throughput is
linear in s*f. It is not: N tiles on S SMs take ceil(N/S) WAVES, a staircase, and the
staircase is what breaks the convexity argument.

When N sits just above a multiple of S, the uniform design pays a whole extra wave for a
handful of tiles and must raise the clock for ALL of them to hit the deadline. A split can
put the remainder on a small dedicated domain and leave the big domain an exact number of
waves, so the 95%+ of tiles it holds run far slower and still finish on time.

The result is a sawtooth: large gains just above each multiple of 108, nothing at the
multiples, and the whole effect decaying as the wave count grows -- one wasted wave out of
four matters, one out of three hundred does not.
"""
import csv, os
import numpy as np
import matplotlib.pyplot as plt
import figstyle as F

HERE = os.path.dirname(os.path.abspath(__file__))
R = list(csv.DictReader(open(os.path.join(HERE, "data", "wave_sawtooth.csv"))))
N = np.array([int(r["tiles"]) for r in R])
G = np.array([float(r["gain_pct"]) for r in R])

F.use()
fig, (a1, a2) = plt.subplots(1, 2, figsize=(F.TEXT_W, 2.4),
                             gridspec_kw=dict(wspace=0.34, width_ratios=[1.35, 1]))
fig.subplots_adjust(bottom=0.22)

F.despine(a1)
for m in range(1, 7):
    a1.axvline(m * 108, color=F.COMM, lw=0.7, ls=(0, (3, 2)), zorder=1)
a1.plot(N, -G, "-", lw=1.3, color=F.GEMM, zorder=4)
a1.fill_between(N, 0, -G, color=F.GEMM, alpha=0.18, zorder=3)
a1.set_xlabel("tiles in the layer  (108 SMs, so a multiple of 108 is an exact wave)")
a1.set_ylabel("energy saved by the best\ntwo-domain split (%)")
a1.set_title("(a) a sawtooth, peaking just above each exact wave",
             fontsize=7.2, pad=3)
a1.text(118, 34, "multiples of 108", fontsize=5.8, color=F.COMM, rotation=90,
        va="top")
j = int(np.argmin(G))
a1.annotate(f"{N[j]} tiles: $-${-G[j]:.0f}%", xy=(N[j], -G[j]), xytext=(N[j] + 60, -G[j] + 3),
            fontsize=6.2, color=F.GEMM, fontweight="bold",
            arrowprops=dict(arrowstyle="->", lw=0.7, color=F.GEMM))
a1.set_xlim(105, 660); a1.set_ylim(0, 46)

# (b) the effect against how many waves deep the layer is
F.despine(a2)
W = np.ceil(N / 108).astype(int)
for w in sorted(set(W)):
    g = -G[W == w]
    a2.plot([w] * len(g), g, "o", ms=2.6, color=F.GEMM, alpha=0.45, mec="none", zorder=3)
    a2.plot([w], [g.max()], "_", ms=13, color=F.INK, mew=1.4, zorder=5)
a2.set_xlabel("waves in the layer,  $\\lceil N/108 \\rceil$")
a2.set_ylabel("energy saved (%)")
a2.set_title("(b) it is a shallow-layer effect", fontsize=7.2, pad=3)
a2.set_xticks(sorted(set(W)))
a2.text(4.3, 41, "bar = best at that depth", fontsize=5.8, color=F.INK)
a2.set_ylim(0, 46)

fig.text(0.5, -0.02, "Both arms meet the SAME deadline, the one the split achieves. "
         "Energy is $\\sum_d [\\,n_d\\,\\kappa(f_d) + s_d\\,P_\\mathrm{leak}(f_d)\\,T\\,]$ "
         "with $\\kappa$ and $P_\\mathrm{leak}$ measured\n(kappa_sweep.csv, "
         "idle_power.csv) and clocks on the A100's real 15 MHz grid -- a coarser grid "
         "invents gains that are pure quantisation. Search: every SM split, every clock "
         "pair,\nfive tile assignments. The opportunity is scheduling granularity; the "
         "flat $\\kappa$ below the knee is only what makes the big domain's lower clock "
         "free.", ha="center", va="top", fontsize=5.5, color=F.INK, linespacing=1.5)
print(F.save(fig, "fig8_wave_sawtooth"))
