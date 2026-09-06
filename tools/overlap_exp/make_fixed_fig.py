#!/usr/bin/env python3
"""fig10 -- the frequency map under a FIXED SM split.

One panel per split the hardware could be built with. Inside a panel the SM split is no
longer a variable: the only two knobs left are the two domain clocks, so the panel IS the
design space of a chip whose partition is already in silicon. Every cell is a
(f_gemm, f_comm) pair, addressable -- point at it and read the design off the axes.

The value is median energy against the SAME split running at the clock the GPU's own
governor picks, so a cell cannot be showing a split effect in disguise.

f_gemm stops at 1200: at 1410 the full-SM GEMM throttles off its requested clock in every
shape, so there is no held-clock measurement to compose from. That is a measurement fact,
not a modelling choice, and it is why the row is absent rather than blank.
"""
import csv, os, statistics as st
import numpy as np
import matplotlib.pyplot as plt
import figstyle as F

HERE = os.path.dirname(os.path.abspath(__file__))
import json

rows = list(csv.DictReader(open(os.path.join(HERE, "data", "fixed_design.csv"))))
n_wl = len({(r["ar_mib"], r["m"], r["nk"]) for r in rows})
# The panel headline is what the SECOND V/f domain buys at this split: cheapest point
# with one domain (f_gemm == f_comm, what an A100 can do today) vs cheapest with two.
# It was previously "how good is this split versus a better split", which answers a
# different question and read as if a bad partition were an argument against two domains.
GAIN = {int(k): v for k, v in
        json.load(open(os.path.join(HERE, "data",
                                    "fixed_design_gain.json"))).items()}

acc = {}
for r in rows:
    acc.setdefault((int(r["ctas"]), int(r["f_gemm"]), int(r["f_comm"])), []).append(
        float(r["dE_vs_auto_pct"]))

# axes are whatever survived the held-clock filter, not a hard-coded list
SPLITS = sorted({k[0] for k in acc})
FG = sorted({k[1] for k in acc})
FC = sorted({k[2] for k in acc})
# a row or column that is NaN for every split carries no information -- drop it rather
# than print a wall of dashes. f_gemm=1410 is the case: the full-SM GEMM throttles off
# its requested clock in every shape, so there is no held-clock measurement to compose.
FG = [f for f in FG if any(len(acc.get((c, f, fc), [])) == n_wl
                           for c in SPLITS for fc in FC)]
FC = [f for f in FC if any(len(acc.get((c, fg, f), [])) == n_wl
                           for c in SPLITS for fg in FG)]
NC = 2 if len(SPLITS) <= 4 else 3
NR = -(-len(SPLITS) // NC)

F.use()
fig, axes = plt.subplots(NR, NC, figsize=(F.TEXT_W, 1.95 * NR))
fig.subplots_adjust(hspace=0.55, wspace=0.34, right=0.855)
for ax in axes.ravel()[len(SPLITS):]:
    ax.set_visible(False)
for k, (ax, c) in enumerate(zip(axes.ravel(), SPLITS)):
    M = [[st.median(acc[(c, fg, fc)]) if len(acc.get((c, fg, fc), [])) == n_wl
          else np.nan for fc in FC] for fg in FG]
    im = F.matrix(ax, [str(f) for f in FG], [str(f) for f in FC], M,
                  fmt="{:+.0f}", vmin=-30, vmax=30, fontsize=5.0)
    a = np.array(M, float)
    i, j = np.unravel_index(np.nanargmin(a), a.shape)
    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                               edgecolor=F.INK, linewidth=1.3, zorder=5))
    ax.set_title(f"({'abcdefghijkl'[k]}) {c} CTA = {c/108*100:.1f}% of SMs\n"
                 f"2nd V/f domain buys {GAIN[c]:+.1f}%", pad=2.5, fontsize=6.5)
    ax.set_xlabel("$f_\\mathrm{comm}$ (MHz)", labelpad=1.0, fontsize=6.5)
    ax.set_ylabel("$f_\\mathrm{gemm}$ (MHz)", labelpad=1.0, fontsize=6.5)
    # eight clocks will not fit side by side at 1.8 in of panel width; rotating is the
    # only way to keep every tick labelled, and dropping every other tick would make the
    # cells unaddressable, which is the entire point of the form.
    ax.tick_params(labelsize=5.5)
    plt.setp(ax.get_xticklabels(), rotation=90, va="top", ha="center")

cax = fig.add_axes([0.885, 0.20, 0.017, 0.60])
cb = fig.colorbar(im, cax=cax)
cb.set_label("median energy vs. same split, unlocked (%)", fontsize=6.5, labelpad=3)
cb.ax.tick_params(labelsize=6); cb.outline.set_visible(False)
print(F.save(fig, "fig10_fixed_design"))
