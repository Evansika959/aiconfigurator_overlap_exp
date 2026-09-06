#!/usr/bin/env python3
"""fig13 -- one cell of the fig7 matrix: the second V/f domain against its baseline.

Cell: GEMM 4096x4096x4096 + 128 MiB all-reduce, 32 CTAs, world = 4.

Rectangle AREA IS ENERGY: height is power, width is the kernel's span inside the
overlapped iteration.

(a) is the fig7 baseline itself -- a REAL overlapped run (two fresh streams, collective
at priority -3 and issued first, both gated on one common event, hardware block scheduler
deciding the SM share), locked at the clock the tightest-SLA rule selects for this cell.
(b) cannot be run: A100 has one clock domain. It is composed from the measured solo
kernels and held to (a)'s own latency, so the saving cannot be bought with time.
"""
import json, os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import figstyle as F

HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, "data", "case3.json")))
P, meta = [d["b"], d["c"]], d["meta"]
E0 = d["b"]["E_meas"]
titles = [f"(a) baseline: one domain, {d['b']['fg']} MHz",
          f"(b) two domains, {d['c']['fg']} / {d['c']['fc']} MHz"]

F.use()
fig, axes = plt.subplots(1, 2, figsize=(F.TEXT_W * 0.76, 2.35), sharey=True,
                         gridspec_kw=dict(wspace=0.10))
for i, (ax, p, title) in enumerate(zip(axes, P, titles)):
    for x0, w, y0, h, col, hat in (
            (0, p["T"], 0, p["ps"], F.BASE, F.HATCH["base"]),
            (0, p["tc"], p["ps"], p["pc"], F.COMM, F.HATCH["comm"]),
            (0, p["tg"], p["ps"] + p["pc"], p["pg"], F.GEMM, F.HATCH["gemm"])):
        ax.add_patch(Rectangle((x0, y0), w, h, facecolor=col, edgecolor="white",
                               linewidth=0.7, hatch=hat, zorder=3))
    ax.text(p["tg"] / 2, p["ps"] + p["pc"] + p["pg"] / 2, f"{p['eg']:.0f}",
            ha="center", va="center", color="white", fontsize=7, zorder=5)
    ax.text(p["tc"] * 0.62, p["ps"] + p["pc"] / 2, f"{p['ec']:.0f}", ha="center",
            va="center", color="white", fontsize=7, zorder=5)
    ax.text(p["T"] * 0.42, p["ps"] / 2, f"{p['es']:.0f}", ha="center", va="center",
            color=F.INK, fontsize=7, zorder=5)
    ax.axvline(p["T"], color=F.INK, lw=0.7, zorder=4)
    # the iteration boundary is itself a measurement: 10 repeats give it a range
    ax.axvspan(p["T_lo"], p["T_hi"], color=F.INK, alpha=0.16, lw=0, zorder=2)
    E = p.get("E_meas", p.get("E"))
    half = (p["E_hi"] - p["E_lo"]) / 2
    ax.text(0.04, 0.965, f"{E:.0f} $\\pm$ {half:.0f} mJ", transform=ax.transAxes,
            fontsize=9, va="top", ha="left", fontweight="bold")
    if i:
        # beside the mJ figure, not under it: under it lands inside the GEMM rectangle,
        # where orange text on an orange fill is invisible.
        ax.text(0.04, 0.855, f"{(E - E0) / E0 * 100:+.1f}%", transform=ax.transAxes,
                fontsize=9, va="top", ha="left", color=F.GEMM, fontweight="bold")
        ax.text(0.04, 0.752, f"({p['sav_hi']:+.1f} .. {p['sav_lo']:+.1f})",
                transform=ax.transAxes, fontsize=6.2, va="top", ha="left",
                color=F.GEMM)
    ax.text(0.96, 0.965, f"$T$ {p['T']:.2f} ms", transform=ax.transAxes,
            fontsize=6.8, va="top", ha="right", color=F.MUTED)
    ax.text(0.96, 0.885, f"$\\pm$ {(p['T_hi']-p['T_lo'])/2*1000:.0f} $\\mu$s",
            transform=ax.transAxes, fontsize=6, va="top", ha="right", color=F.MUTED)
    ax.text(0.96, 0.035,
            f"measured, {d['b'].get('reps', 1)} repeats" if i == 0
            else f"inferred, {d['b'].get('reps', 1)} repeats",
            transform=ax.transAxes,
            fontsize=6.5, va="bottom", ha="right",
            color=F.MUTED if i == 0 else F.GEMM, style="italic")
    ax.set_title(title, fontsize=7.2, pad=4)
    ax.set_xlim(0, 1.50); ax.set_ylim(0, 405)
    ax.set_xlabel("time (ms)")
    F.despine(ax, grid_axis="y")
    ax.set_xticks([0, 0.5, 1.0, 1.5]); ax.set_yticks([0, 100, 200, 300])
# the second domain may not buy its saving with time: (b) is held to (a)'s own latency.
axes[1].axvline(d["b"]["T"], color=F.MUTED, lw=0.7, ls=(0, (3, 2)), zorder=4)
axes[1].text(d["b"]["T"] - 0.03, 190, "(a) deadline", fontsize=6, color=F.MUTED,
             va="center", ha="center", rotation=90)
axes[0].set_ylabel("power (W)")
fig.suptitle(f"GEMM {meta['M']}x{meta['N']}x{meta['N']}  +  {meta['MIB']} MiB "
             f"all-reduce, {meta['CT']} CTAs, world = 4\n"
             f"rectangle area = energy;  bars and shading = full range over "
             f"{d['b'].get('reps', 1)} repeats",
             fontsize=7.5, y=1.10)
h = [Rectangle((0, 0), 1, 1, facecolor=c, hatch=k, edgecolor="white", linewidth=0.5)
     for c, k in ((F.GEMM, F.HATCH["gemm"]), (F.COMM, F.HATCH["comm"]),
                  (F.BASE, F.HATCH["base"]))]
fig.legend(h, ["GEMM", "all-reduce", "static floor"], loc="lower center",
           ncol=3, bbox_to_anchor=(0.5, -0.17))
print(F.save(fig, "fig13_case_4096"))
