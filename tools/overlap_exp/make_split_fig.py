#!/usr/bin/env python3
"""fig11 -- what a fixed SM split costs, as a function of where you put it.

The point of the extension campaign. With only 4/8/16/32 measured, 32 sat at the right
edge of the sampled range and read +0.0%: indistinguishable from "we stopped looking".
With 2..64 measured the curve has an interior minimum, so the flat bottom is a property
of the hardware and not of where sampling stopped.

One line per message size, because every workload in the original set was comm-bound by
construction -- exactly the regime that rewards a wide comm split. If the minimum moved
with message size, "commit to c" would be a claim about the workload mix, not the chip.
"""
import csv, os, statistics as st
import matplotlib.pyplot as plt
import figstyle as F

HERE = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(HERE, "data", "fixed_design.csv"))))

best = {}
for r in rows:
    k = (int(r["ctas"]), int(r["ar_mib"]), int(r["m"]), int(r["nk"]))
    e = float(r["energy_mj"])
    if k not in best or e < best[k]:
        best[k] = e
SPLITS = sorted({k[0] for k in best})
SIZES = sorted({k[1] for k in best})
WL = sorted({k[1:] for k in best})

pen = {c: {m: [] for m in SIZES} for c in SPLITS}
for w in WL:
    per = {c: best[(c,) + w] for c in SPLITS if (c,) + w in best}
    if len(per) != len(SPLITS):
        continue
    b = min(per.values())
    for c in SPLITS:
        pen[c][w[0]].append((per[c] - b) / b * 100)

F.use()
fig, ax = plt.subplots(figsize=(F.TEXT_W, 2.35))
F.despine(ax)
mk = ["o", "s", "^", "D"]
cols = [F.GEMM, "#E09A4B", "#8FA0E4", F.COMM]
for i, m in enumerate(SIZES):
    y = [st.median(pen[c][m]) for c in SPLITS]
    ax.plot(SPLITS, y, marker=mk[i], ms=3.2, lw=1.0, color=cols[i],
            markeredgecolor="white", markeredgewidth=0.5, label=f"{m} MiB", zorder=3)
    j = min(range(len(SPLITS)), key=lambda k: y[k])
    ax.annotate(f"{SPLITS[j]}", (SPLITS[j], y[j]), textcoords="offset points",
                xytext=(0, -9), ha="center", fontsize=6, color=cols[i])
ax.set_xscale("log", base=2)
ax.set_yscale("symlog", linthresh=3)
ax.set_xticks(SPLITS); ax.set_xticklabels([str(c) for c in SPLITS])
ax.set_yticks([0, 1, 3, 10, 30, 100, 300])
ax.set_yticklabels(["0", "1", "3", "10", "30", "100", "300"])
ax.set_xlabel("SM split: NCCL CTAs = SMs given to the collective (of 108)")
ax.set_ylabel("energy penalty vs. best split for\nthat workload (%, median over 6 shapes)")
ax.axhspan(-0.5, 3.5, color=F.BASE, alpha=0.45, zorder=0, lw=0)
ax.text(2.15, 2.6, "within 3% of the per-workload optimum", fontsize=6,
        color=F.MUTED, va="top")
ax.legend(ncol=2, loc="upper right", fontsize=6.5)
ax.set_ylim(-0.6, 400)
print(F.save(fig, "fig11_split_cost"))
