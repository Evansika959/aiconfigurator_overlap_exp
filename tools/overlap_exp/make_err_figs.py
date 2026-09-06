#!/usr/bin/env python3
"""fig14 -- the two-clock matrix with an uncertainty on every cell.
fig15 -- the one cell where that uncertainty is MEASURED, not propagated.

fig14's +- is a Monte-Carlo propagation of the measured single-window noise through the
whole selection (matrix_errorbars.py). It is deliberately NOT tuned to match: on the one
cell repeated ten times it predicts +-0.63 pp where the real spread is +-0.30, so treat
every bar as a conservative upper bound. It is reported unscaled because a single
validation point cannot justify a global correction factor.

fig15 carries real error bars: ten independent repeats of every measurement feeding that
cell, each bar the median and the whisker the full range.
"""
import csv, glob, json, os, statistics as st
import numpy as np
import matplotlib.pyplot as plt
import followup_baseline as FB
import figstyle as F

HERE = os.path.dirname(os.path.abspath(__file__))
CELL = (128, 4096, 4096, 32)

# ------------------------------------------------------------------ fig14
E = {(int(r["mib"]), int(r["m"]), int(r["nk"]), int(r["ctas"])): r
     for r in csv.DictReader(open(os.path.join(HERE, "data", "fig7_err.csv")))}
shapes = sorted({(k[1], k[2]) for k in E}, key=lambda s: s[0] * s[1])
cols = [(m, c) for m in (128, 256, 512) for c in (32, 16, 8, 4)]

F.use()
fig, ax = plt.subplots(figsize=(F.TEXT_W, 2.75))
grid = [[float(E[(m, sh[0], sh[1], c)]["dE_pct"]) if (m, sh[0], sh[1], c) in E else np.nan
         for (m, c) in cols] for sh in shapes]
sd = [[float(E[(m, sh[0], sh[1], c)]["sd_pp"]) if (m, sh[0], sh[1], c) in E else np.nan
       for (m, c) in cols] for sh in shapes]
im = F.matrix(ax, [f"{p}x{q}" for p, q in shapes], [str(c) for (m, c) in cols],
              grid, fmt="{:+.0f}", vmin=-30, vmax=30, fontsize=5.6)
a = np.array(grid, float)
for i in range(a.shape[0]):
    for j in range(a.shape[1]):
        if np.isnan(a[i, j]):
            continue
        t = abs(a[i, j]) / 30
        ax.text(j, i + 0.30, f"$\\pm${sd[i][j]:.1f}", ha="center", va="center",
                fontsize=4.4, color="white" if t > 0.55 else F.MUTED)
for i in (1, 2):
    ax.axvline(i * 4 - 0.5, color=F.INK, lw=0.8, zorder=6)
for i, m in enumerate((128, 256, 512)):
    ax.text(i * 4 + 1.5, len(shapes) - 0.10, f"{m} MiB", ha="center", va="top",
            fontsize=7, clip_on=False)
ax.set_xlabel("NCCL CTAs, grouped by all-reduce message size", labelpad=15)
ax.set_ylabel("GEMM  M x N=K")
ax.set_title("energy saved by a second clock domain (%), vs the measured overlap at the\n"
             "tightest-SLA clock.  $\\pm$ is a Monte-Carlo upper bound (see caption)",
             fontsize=7.2, pad=5)
cb = fig.colorbar(im, ax=ax, fraction=0.018, pad=0.012)
cb.ax.tick_params(labelsize=6.5, width=0.5, length=2); cb.outline.set_linewidth(0.5)
print(F.save(fig, "fig14_matrix_err"))

# ------------------------------------------------------------------ fig15
CLOCKS = [300, 510, 705, 900, 1200, 1410]
rows = list(csv.DictReader(open(os.path.join(HERE, "data", "commbound_128mib.csv"))))
g_ = int(next(r for r in rows if r["mode"] == "concurrent"
              and (int(r["m"]), int(r["n"])) == (4096, 4096))["grid"])
per = {"baseline\n(measured overlap,\n1200 MHz)": [], "one domain\n(composed,\nbest clock)": [],
       "two domains\n(composed,\n900 / 1200)": []}
for sf, cf in zip(sorted(glob.glob(os.path.join(HERE, "data/repeat/solo_*.csv"))),
                  sorted(glob.glob(os.path.join(HERE, "data/repeat/spans_*.json")))):
    s = json.load(open(cf))
    Tb, Eb = s["T_loop"], s["power_per_gpu_w"] * s["T_loop"]
    G, C = {}, {}
    for r in csv.DictReader(open(sf)):
        if r["clock_held"] != "True":
            continue
        d = dict(power_per_gpu_w=float(r["power_per_gpu_w"]), iter_ms=float(r["iter_ms"]),
                 ctas=32)
        (G if r["mode"] == "gemm_only" else C)[int(r["clock"])] = d
    one, two = [], []
    for fg in G:
        for fc in C:
            ti, En, pk = FB.compose(G[fg], C[fc], fg, fc, g_)
            if En / ti > 400.0 or ti > Tb * 1.0001:
                continue
            two.append(En)
            if fg == fc:
                one.append(En)
    per["baseline\n(measured overlap,\n1200 MHz)"].append(Eb)
    per["one domain\n(composed,\nbest clock)"].append(min(one))
    per["two domains\n(composed,\n900 / 1200)"].append(min(two))

fig, ax = plt.subplots(figsize=(F.TEXT_W * 0.62, 2.4))
F.despine(ax)
labels = list(per)
med = [st.median(per[k]) for k in labels]
lo = [med[i] - min(per[k]) for i, k in enumerate(labels)]
hi = [max(per[k]) - med[i] for i, k in enumerate(labels)]
cols_ = [F.MUTED, F.COMM, F.GEMM]
hat = ["", F.HATCH["comm"], F.HATCH["gemm"]]
ax.bar(range(3), med, 0.6, color=cols_, hatch=hat, edgecolor="white", linewidth=0.5,
       zorder=3)
ax.errorbar(range(3), med, yerr=[lo, hi], fmt="none", ecolor=F.INK, elinewidth=0.8,
            capsize=2.5, capthick=0.8, zorder=5)
for i, k in enumerate(labels):
    ax.text(i, med[i] + hi[i] + 9, f"{med[i]:.0f}", ha="center", fontsize=7.5,
            fontweight="bold")
    if i:
        b = st.median(per[labels[0]])
        ax.text(i, med[i] + hi[i] + 34, f"{(med[i]-b)/b*100:+.1f}%", ha="center",
                fontsize=7.5, color=F.GEMM, fontweight="bold")
ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=6.2)
ax.set_ylabel("energy per iteration (mJ)")
ax.set_ylim(0, 400)
ax.set_title("GEMM 4096x4096x4096 + 128 MiB, 32 CTAs, world = 4\n"
             "bar = median of 10 repeats, whisker = full range", fontsize=7.2, pad=4)
print(F.save(fig, "fig15_cell_bars"))
