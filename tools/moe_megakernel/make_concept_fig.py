#!/usr/bin/env python3
"""fig1 -- what "spatial, intra-kernel V/f partitioning" actually means.

A schematic, not data. Its job is to make one claim unmissable: **in our scheme frequency
becomes a function of WHERE a block runs on the chip, not of WHEN it runs.** Every panel
is the same SM x time plane, so the three designs are directly comparable; colour is
always frequency, never expert identity, because frequency is the variable under study.

(a) is today. (b) is what the LLM-DVFS literature does, drawn to scale so the 38 ms
transition can be compared against the step it would optimise. (c) is the proposal.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
import figstyle as F

SM = 108
DEADLINE = 1.00          # ms, an SLO-derived deadline for one MoE layer
# frequency is a MAGNITUDE, so it gets a single-hue sequential ramp, light = slow.
RAMP = LinearSegmentedColormap.from_list("f", ["#F6E6D4", "#E9A464", F.GEMM, "#7A3F02"])
FL = [510, 900, 1200, 1410]
COL = {f: RAMP(i / (len(FL) - 1)) for i, f in enumerate(FL)}

F.use()
fig, axes = plt.subplots(1, 3, figsize=(F.TEXT_W, 2.5), sharey=True,
                         gridspec_kw=dict(wspace=0.17))
rng = np.random.default_rng(3)

# ---------------------------------------------------------------- (a) today
ax = axes[0]
# a work-conserving queue: every SM keeps pulling tiles until the queue drains, so the
# plane is dense and the only slack is the ragged tail of the last wave.
y = 0
while y < SM:
    h = 1
    t = 0.0
    while t < 0.80:
        w = rng.uniform(0.06, 0.16)
        ax.add_patch(Rectangle((t, y), min(w, 0.80 - t) * 0.97, h * 0.92,
                               facecolor=COL[1410], edgecolor="none"))
        t += w
    # the tail: some SMs run one tile longer than others
    if rng.random() < 0.55:
        ax.add_patch(Rectangle((0.80, y), rng.uniform(0.04, 0.17), h * 0.92,
                               facecolor=COL[1410], edgecolor="none"))
    y += h
ax.set_title("(a) today: one kernel, one clock", fontsize=7.2, pad=4)
ax.text(0.03, SM * 0.965, "all 108 SMs @ 1410 MHz", fontsize=6, va="top", color="white")
# the annotation has to live in the white margin: over the tiles it is dark text on a
# dark fill and disappears.
ax.annotate("tail =\nthe only\nslack", xy=(0.94, 34), xytext=(1.02, 68),
            fontsize=5.5, color=F.INK, ha="left", va="center", annotation_clip=False,
            arrowprops=dict(arrowstyle="->", lw=0.6, color=F.INK))

# ---------------------------------------------------------------- (b) temporal DVFS
ax = axes[1]
ax.add_patch(Rectangle((0, 0), 0.95, SM, facecolor=COL[1410], edgecolor="none"))
ax.text(0.06, SM * 0.965, "step $n$\n@ 1410", fontsize=6, va="top", color="white")
# the transition, drawn to scale against the thing it would optimise
ax.add_patch(Rectangle((1.02, 0), 2.9, SM, facecolor="none", edgecolor=F.MUTED,
                       hatch="////", lw=0.6))
ax.annotate("", xy=(1.02, SM * 0.52), xytext=(3.92, SM * 0.52),
            arrowprops=dict(arrowstyle="<->", lw=0.7, color=F.INK))
ax.text(2.47, SM * 0.58, "38 ms", fontsize=7, ha="center", fontweight="bold")
ax.text(2.47, SM * 0.44, "DVFS transition\n= 38 steps", fontsize=6, ha="center",
        color=F.MUTED, va="top")
ax.set_title("(b) temporal DVFS (the literature)", fontsize=7.2, pad=4)

# ---------------------------------------------------------------- (c) ours
ax = axes[2]
# four SM partitions, each pinned to a group of experts. Heavy groups keep the high clock;
# light groups are STRETCHED to the same deadline at a low clock instead of finishing
# early and idling. Every partition lands on the deadline together.
parts = [(0,  26, 1410, 0.98),   # (sm_lo, sm_hi, f, busy fraction at 1410)
         (26, 54, 1200, 0.72),
         (54, 82,  900, 0.50),
         (82, SM,  510, 0.29)]
for lo, hi, f, busy in parts:
    # solid: the stretched, low-clock execution that ends on the deadline
    ax.add_patch(Rectangle((0, lo), DEADLINE, (hi - lo) * 0.96,
                           facecolor=COL[f], edgecolor="white", lw=0.6))
    ax.text(0.06, lo + (hi - lo) * 0.52, f"{f} MHz", fontsize=6, va="center",
            color="white" if f >= 1200 else F.INK)
    # dashed: where it WOULD have finished at 1410, i.e. the slack being converted
    if busy < 0.98:
        ax.add_patch(Rectangle((0, lo), busy * DEADLINE, (hi - lo) * 0.96, fill=False,
                               edgecolor=F.INK, lw=0.7, ls=(0, (2.2, 1.6))))
ax.axvline(DEADLINE, color=F.INK, lw=1.0)
ax.text(DEADLINE + 0.035, SM * 0.5, "SLO deadline", fontsize=6, rotation=90,
        ha="center", va="center", color=F.INK)
ax.set_title("(c) spatial, intra-kernel V/f", fontsize=7.2, pad=4)

for i, ax in enumerate(axes):
    ax.set_xlim(0, 4.05 if i == 1 else 1.34)
    ax.set_ylim(0, SM)
    ax.set_xlabel("time (ms)", labelpad=2)
    F.despine(ax, grid_axis=None)
    ax.set_yticks([0, 54, 108])
    ax.set_xticks([0, 1, 2, 3, 4] if i == 1 else [0, 0.5, 1.0])
axes[0].set_ylabel("SM (of 108)")
fig.suptitle("frequency is a function of WHERE a block runs, not WHEN "
             "(colour = clock, one MoE layer)", fontsize=7.5, y=1.03)
# the dashed outlines need a caption, and inside panel (c) it would sit on top of the
# very rectangles it is describing
fig.text(0.5, -0.055, "(c) dashed outline = where that partition would finish at "
         "1410 MHz, after which it idles. Filling the slack with a lower clock instead "
         "is the whole idea.", ha="center", fontsize=6, color=F.INK)
h = [Rectangle((0, 0), 1, 1, facecolor=COL[f]) for f in FL]
fig.legend(h, [f"{f} MHz" for f in FL], loc="lower center", ncol=4,
           bbox_to_anchor=(0.5, -0.13), fontsize=6.5)
print(F.save(fig, "fig1_concept"))
