#!/usr/bin/env python3
"""fig2 -- where routing actually shows up inside one prefill MoE layer.

(a) is the pipeline with real shapes and real FLOP shares for OLMoE at prefill batch 16.
    The point it has to make: routing is DECIDED in a stage worth 0.26% of the layer's
    arithmetic, MATERIALISED as 64 segment lengths, and only then does it become 99.71%
    of the work. The lever and the cost live in different stages.

(b) is the measured consequence -- the 64 segment lengths for one real layer, sorted.
    Same data as Phase 0a, but shown as what a grouped GEMM actually receives.
"""
import json, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import figstyle as F

HERE = os.path.dirname(os.path.abspath(__file__))
Z = np.load(os.path.join(HERE, "data", "routing_olmoe.npz"))
meta = json.loads(str(Z["meta"]))
E, K = meta["experts"], meta["top_k"]
B, T = 16, 16 * meta["prefill_len"]
load = Z[f"prefill_b{B}"][0]                      # layer 0, (E,)
H, I = 2048, 1024

F.use()
fig = plt.figure(figsize=(F.TEXT_W, 3.5))
gs = fig.add_gridspec(2, 1, height_ratios=[0.85, 1.0], hspace=0.34)

# ------------------------------------------------------------------ (a) the pipeline
ax = fig.add_subplot(gs[0]); ax.set_axis_off()
ax.set_xlim(0, 100); ax.set_ylim(4.5, 34)
stages = [
    ("hidden\nstates", f"{T} x {H}", None, F.BASE, 0),
    ("router GEMM\n(the gate)", f"({T}x{H})\n@ ({H}x{E})", "0.26% of FLOPs", F.COMM, 1),
    ("top-8, histogram,\nsort by expert", f"64 segments\n$\\sum n_e$ = {load.sum()}",
     "no FLOPs -- pure data movement.\n$n_0 \\ldots n_{63}$ = the imbalance,\n"
     "now a data structure", F.COMM, 2),
    ("grouped GEMM\n(the experts)", f"64 x ($n_e$x{H})\n@ ({H}x{I}), x3",
     "99.71% of FLOPs", F.GEMM, 3),
    ("unpermute +\nweighted sum", f"back to\n{T} x {H}", "0.03%", F.BASE, 4),
]
w, gap = 16.0, 4.6
for name, shape, note, col, i in stages:
    x = i * (w + gap)
    ax.add_patch(FancyBboxPatch((x, 12), w, 15, boxstyle="round,pad=0.35,rounding_size=1",
                                facecolor=col, edgecolor="white", lw=0.8))
    ink = "white" if col in (F.GEMM, F.COMM) else F.INK
    ax.text(x + w / 2, 23.6, name, ha="center", va="center", fontsize=6.4, color=ink,
            fontweight="bold")
    ax.text(x + w / 2, 16.6, shape, ha="center", va="center", fontsize=5.6, color=ink)
    if note:
        ax.text(x + w / 2, 9.6, note, ha="center", va="top", fontsize=5.6, color=F.MUTED)
    if i < 4:
        ax.add_patch(FancyArrowPatch((x + w + 0.5, 19.5), (x + w + gap - 0.5, 19.5),
                                     arrowstyle="-|>", mutation_scale=7, lw=0.8,
                                     color=F.INK))
ax.annotate("routing DECIDED here", xy=(1 * (w + gap) + w / 2, 28.2), xytext=(28, 32.5),
            fontsize=6, color=F.COMM, ha="center", fontweight="bold",
            arrowprops=dict(arrowstyle="->", lw=0.7, color=F.COMM))
ax.annotate("imbalance BECOMES WORK here", xy=(3 * (w + gap) + w / 2, 28.2),
            xytext=(75, 32.5), fontsize=6, color=F.GEMM, ha="center", fontweight="bold",
            arrowprops=dict(arrowstyle="->", lw=0.7, color=F.GEMM))
ax.set_title("(a) one prefill MoE layer, OLMoE, batch 16 "
             f"({T} tokens, top-{K} of {E})", fontsize=7.2, pad=2)

# ------------------------------------------------------------------ (b) the segments
ax = fig.add_subplot(gs[1])
s = np.sort(load)[::-1]
ax.bar(range(E), s, width=0.86, color=F.GEMM, edgecolor="none", zorder=3)
m = load.mean()
ax.axhline(m, color=F.INK, lw=0.8, ls=(0, (3, 2)), zorder=4)
ax.text(20.5, m * 1.10, f"mean {m:.0f} rows", ha="left", fontsize=6, color=F.INK)
ax.annotate(f"{s[0]} rows\n= {int(np.ceil(s[0]/128)*np.ceil(I/128))} tiles, 2 waves",
            xy=(0.6, s[0]), xytext=(6.5, s[0] * 0.94), fontsize=6, color=F.GEMM,
            va="top", arrowprops=dict(arrowstyle="->", lw=0.7, color=F.GEMM))
ax.annotate(f"{s[-1]} rows\n= {int(np.ceil(s[-1]/128)*np.ceil(I/128))} tiles",
            xy=(E - 1.4, s[-1]), xytext=(E - 15, s[0] * 0.40), fontsize=6, color=F.MUTED,
            arrowprops=dict(arrowstyle="->", lw=0.7, color=F.MUTED))
F.despine(ax)
ax.set_xlim(-1, E); ax.set_ylim(0, s[0] * 1.22)
ax.set_xlabel("the 64 experts, sorted by load (this is the grouped GEMM's segment list)")
ax.set_ylabel("rows $n_e$")
ax.set_title(f"(b) what stage 3 hands to stage 4: busiest expert is "
             f"{s[0]/np.median(load):.1f}x the median", fontsize=7.2, pad=3)
print(F.save(fig, "fig2_prefill_routing"))
