#!/usr/bin/env python3
"""fig9 -- the whole idea in one picture: a wasted wave, and how a split removes it.

The previous version of this figure showed a sawtooth over tile count and a scatter over
wave depth. Both were true and neither was legible. The mechanism is concrete, so draw
the concrete thing: the SM x time plane, one rectangle per wave.

325 tiles, 108 SMs, same deadline in both panels.
"""
import csv, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import figstyle as F

HERE = os.path.dirname(os.path.abspath(__file__))
SLOW, FAST = "#617BDC", "#C56604"

F.use()
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(F.TEXT_W, 2.35),
                                 gridspec_kw=dict(wspace=0.38, width_ratios=[1, 1, 0.85]))
fig.subplots_adjust(bottom=0.24)

T = 4 / 1380.0 * 1e3          # ms, the deadline both designs meet

# ---------------- (a) one clock: the 4th wave is almost empty --------------------------
# the wasted part of wave 4 is drawn as an explicit hatched box. Left blank it is just
# absent, and "107 SMs idle" is exactly the thing the reader has to see.
w = T / 4
for i in range(3):
    a1.add_patch(Rectangle((i * w, 0), w * 0.94, 108, facecolor=FAST,
                           edgecolor="white", lw=0.8))
a1.add_patch(Rectangle((3 * w, 0), w * 0.94, 1.6, facecolor=FAST, edgecolor="white",
                       lw=0.8))
a1.add_patch(Rectangle((3 * w, 1.6), w * 0.94, 106.4, facecolor="none",
                       edgecolor=F.MUTED, lw=0.7, hatch="///"))
a1.text(1.5 * w, 54, "3 full waves", ha="center", va="center", color="white",
        fontsize=7.5, fontweight="bold")
a1.text(3.5 * w, 54, "wave 4:\n1 tile,\n107 SMs\nidle", ha="center", va="center",
        fontsize=6.2, color=F.INK)
a1.set_title("(a) one clock, 1380 MHz", fontsize=7.4, pad=3)

# ---------------- (b) split: the big domain gets an exact 3 waves ----------------------
ws = T / 3                    # 107 SMs at 1050: 3 waves fill the same window
wf = T / 4
for i in range(3):
    a2.add_patch(Rectangle((i * ws, 0), ws * 0.94, 104, facecolor=SLOW,
                           edgecolor="white", lw=0.8))
for i in range(4):
    a2.add_patch(Rectangle((i * wf, 105), wf * 0.94, 3, facecolor=FAST,
                           edgecolor="white", lw=0.8))
a2.text(1.5 * ws, 52, "321 tiles on 107 SMs\n= exactly 3 waves\n@ 1050 MHz",
        ha="center", va="center", color="white", fontsize=6.8, fontweight="bold")
a2.set_title("(b) split: $-$36% energy", fontsize=7.4, pad=3)

for ax, note in ((a1, None), (a2, "leftovers on 1 SM @ 1380")):
    ax.set_xlim(0, T * 1.06); ax.set_ylim(0, 126)
    ax.axvline(T, color=F.INK, lw=1.0)
    ax.set_xlabel("time (ms)")
    ax.set_yticks([0, 54, 108])
    F.despine(ax, grid_axis=None)
    ax.text(T * 1.015, 60, "deadline", fontsize=5.8, rotation=90, ha="center",
            va="center", color=F.MUTED)
    if note:
        ax.annotate(note, xy=(2.4 * wf, 108.5), xytext=(T * 0.30, 120),
                    fontsize=6.2, color=F.INK, ha="center",
                    arrowprops=dict(arrowstyle="->", lw=0.7, color=F.INK))
a1.set_ylabel("SM (of 108)")

# ---------------- (c) how much it is worth on the real workloads -----------------------
LAB = ["decode\nB=64", "prefill\nB=16", "prefill\nB=128"]
WAV = [5, 40, 306]
GAIN = [14.1, 1.0, 2.5]
F.despine(a3)
a3.bar(range(3), GAIN, width=0.6, color=[FAST, F.BASE, F.BASE], edgecolor="white",
       lw=0.7, zorder=3)
for i, (g, wv) in enumerate(zip(GAIN, WAV)):
    a3.text(i, g + 0.7, f"{g:.1f}%", ha="center", fontsize=7, fontweight="bold")
    a3.text(i, 0.5, f"{wv} waves", ha="center", fontsize=5.8, color=F.INK)
a3.set_xticks(range(3)); a3.set_xticklabels(LAB, fontsize=6.2)
a3.set_ylabel("energy saved (%)")
a3.set_ylim(0, 17)
a3.set_title("(c) only shallow layers", fontsize=7.4, pad=3)

fig.text(0.5, -0.02, "A wave is one round of work across all 108 SMs. 325 tiles need "
         "$\\lceil 325/108 \\rceil = 4$ waves, and the 4th holds a single tile -- but it "
         "still costs a full\nwave of time, so every tile must run at 1380 MHz to make "
         "the deadline. Hand those leftovers to one dedicated SM and the other 107 have "
         "exactly 3 waves,\nso they meet the same deadline at 1050 MHz. 99% of the work "
         "moves to the cheap clock. The deeper the layer, the less one wasted wave "
         "matters.",
         ha="center", va="top", fontsize=5.6, color=F.INK, linespacing=1.5)
print(F.save(fig, "fig9_wasted_wave"))
