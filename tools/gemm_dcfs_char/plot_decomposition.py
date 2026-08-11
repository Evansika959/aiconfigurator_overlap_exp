#!/usr/bin/env python3
"""What is left of the measured energy once the board's static power is removed.

Two figures, both in absolute mJ / W -- nothing is normalised.

  gemm_energy_decomposition.png
      Every measured bar split into the part the GEMM actually caused (dynamic)
      and the part the board would have drawn anyway (static). The point of the
      figure is that the two halves behave completely differently: cut the SM
      count and the static half balloons while the dynamic half barely moves, so
      most of the apparent "cost" of a narrow GEMM is idle silicon being paid for
      over a longer wall-clock, not the GEMM getting less efficient.

  gemm_dynamic_power_model.png
      Dynamic power against SM count, with the fit  P_dyn = a*S + b.
      `a` is what one SM costs while computing; `b` is the part that does not
      scale with SMs at all (L2, memory controller, interconnect). A model that
      assumes "70% of the SMs => 70% of the dynamic power" is the b=0 line, and
      it misses -- b is ~8% of dynamic power at 108 SMs and dominates the error
      at low SM counts.

  python3 plot_decomposition.py
"""

import argparse
import collections
import csv
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

SM_ORDER = [108, 81, 54, 27, 14]
FREQ_ORDER = [300, 510, 705, 900, 1200, 1410]
SHAPES = [(1024, 4096), (4096, 8192), (8192, 16384)]
FIX_FREQ = 900         # the clock where dynamic energy is minimal for 46/60 cells
FIX_SM = 108

C_DYN = "#1f7a8c"      # the GEMM's own consumption
C_STA = "#c9ced6"      # what the board draws regardless
C_DROP = "#e8453c"


def load(path):
    raw = collections.defaultdict(list)
    for r in csv.DictReader(open(path)):
        if r["status"] != "ok":
            continue
        raw[(int(r["m"]), int(r["n"]), int(r["sm_avail"]), int(r["freq_req_mhz"]))].append(r)
    out = {}
    for k, rs in raw.items():
        out[k] = dict(
            e_dyn=statistics.median(float(x["energy_dynamic_mj"]) for x in rs),
            e_sta=statistics.median(float(x["energy_static_mj"]) for x in rs),
            e_tot=statistics.median(float(x["energy_mj"]) for x in rs),
            p_dyn=statistics.median(float(x["p_dynamic_w"]) for x in rs),
            lat=statistics.median(float(x["latency_ms"]) for x in rs),
            clk=int(statistics.median(int(x["clock_sm_med"]) for x in rs)),
            dropped=any(x["clock_held"] != "True" for x in rs),
        )
    return out


def stacked(ax, xs, cells, xlabel, title):
    dyn = [c["e_dyn"] for c in cells]
    sta = [c["e_sta"] for c in cells]
    idx = np.arange(len(xs))
    ax.bar(idx, dyn, 0.68, color=C_DYN, label="dynamic — caused by the GEMM", zorder=2)
    ax.bar(idx, sta, 0.68, bottom=dyn, color=C_STA,
           label="static — drawn anyway at this clock", zorder=2)
    for i, c in enumerate(cells):
        tot = c["e_dyn"] + c["e_sta"]
        ax.text(i, tot * 1.03, f"{tot:,.0f}", ha="center", va="bottom",
                fontsize=8.4, fontweight="bold", color="#222222")
        if c["e_dyn"] / tot > 0.16:
            ax.text(i, c["e_dyn"] / 2, f"{c['e_dyn']:,.0f}", ha="center", va="center",
                    fontsize=8.2, color="white", fontweight="bold", zorder=3)
    ax.set_xticks(idx)
    # the achieved clock goes INSIDE the tick label; as a separate annotation below
    # the axis it collided with the tick text
    ax.set_xticklabels([f"{x}\nran @{c['clk']}" if c["dropped"] else str(x)
                        for x, c in zip(xs, cells)], fontsize=9)
    for i, c in enumerate(cells):
        if c["dropped"]:
            ax.get_xticklabels()[i].set_color(C_DROP)
            ax.get_xticklabels()[i].set_fontsize(8)
    ax.set_xlabel(xlabel, fontsize=9.5)
    ax.set_title(title, fontsize=10.5, pad=6)
    ax.set_ylim(0, max(c["e_dyn"] + c["e_sta"] for c in cells) * 1.20)
    ax.grid(axis="y", color="#e6e6e6", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8.5)


def fig_decomposition(D, out):
    # bottom margin has to clear a TWO-LINE tick label ("1410 / ran @1260") plus the
    # x label plus the legend; at 0.085 the x label landed on top of the legend
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.8),
                             gridspec_kw=dict(left=0.062, right=0.985, top=0.855,
                                              bottom=0.125, wspace=0.20, hspace=0.44))
    for j, (m, nk) in enumerate(SHAPES):
        cells = [D[(m, nk, s, FIX_FREQ)] for s in SM_ORDER]
        stacked(axes[0][j], SM_ORDER, cells, "SMs available to GEMM",
                f"M={m}, N=K={nk}   ·   clock fixed at {FIX_FREQ} MHz")
        r_t = cells[-1]["e_tot"] / cells[0]["e_tot"]
        r_d = cells[-1]["e_dyn"] / cells[0]["e_dyn"]
        axes[0][j].text(0.03, 0.965,
                        f"108 → 14 SM:   total ×{r_t:.2f}   but dynamic only ×{r_d:.2f}",
                        transform=axes[0][j].transAxes, fontsize=9.2, va="top",
                        bbox=dict(fc="#fff8e1", ec="#e0c97f", lw=0.8, pad=3.4))

        cells = [D[(m, nk, FIX_SM, f)] for f in FREQ_ORDER]
        stacked(axes[1][j], FREQ_ORDER, cells, "SM clock (MHz)",
                f"M={m}, N=K={nk}   ·   SM count fixed at {FIX_SM}")
        dyn = [c["e_dyn"] for c in cells]
        tot = [c["e_tot"] for c in cells]
        axes[1][j].text(0.03, 0.965,
                        f"cheapest clock:   by total {FREQ_ORDER[tot.index(min(tot))]} MHz"
                        f"   ·   by dynamic {FREQ_ORDER[dyn.index(min(dyn))]} MHz\n"
                        f"300 MHz costs +{(tot[0] / min(tot) - 1) * 100:.0f}% total "
                        f"but only +{(dyn[0] / min(dyn) - 1) * 100:.0f}% dynamic",
                        transform=axes[1][j].transAxes, fontsize=9.2, va="top",
                        bbox=dict(fc="#fff8e1", ec="#e0c97f", lw=0.8, pad=3.4))
    axes[0][0].set_ylabel("Energy per GEMM call (mJ)", fontsize=10)
    axes[1][0].set_ylabel("Energy per GEMM call (mJ)", fontsize=10)

    fig.suptitle("A100-SXM4-40GB · bf16 GEMM · where the measured energy actually goes",
                 fontsize=15.5, fontweight="bold", y=0.972)
    fig.text(0.5, 0.925,
             "Static = P0 board power at the achieved clock with the same SMs occupied "
             "(measured separately, 6 s windows, 144/144 points held their clock), times the GEMM's latency.\n"
             "Dynamic = what is left. Bars are absolute mJ — nothing normalised. "
             "Red labels: the 400 W cap pulled the clock below the request.",
             ha="center", va="top", fontsize=9.6, color="#333333")
    fig.legend(handles=[Patch(fc=C_DYN, label="dynamic energy — caused by the GEMM"),
                        Patch(fc=C_STA, label="static energy — the board would draw this anyway")],
               loc="lower center", bbox_to_anchor=(0.5, 0.004), ncol=2,
               frameon=False, fontsize=10.5)
    fig.savefig(out, dpi=135, facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


def fig_model(D, out):
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.3),
                             gridspec_kw=dict(left=0.058, right=0.987, top=0.755,
                                              bottom=0.135, wspace=0.19))
    cmap = plt.get_cmap("viridis")
    freqs = [300, 510, 705, 900, 1200]           # clocks that held everywhere
    for j, (m, nk) in enumerate(SHAPES):
        ax = axes[j]
        for i, f in enumerate(freqs):
            col = cmap(0.08 + 0.78 * i / (len(freqs) - 1))
            xs = np.array(SM_ORDER, dtype=float)
            ys = np.array([D[(m, nk, s, f)]["p_dyn"] for s in SM_ORDER])
            a, b = np.polyfit(xs, ys, 1)
            gx = np.linspace(0, 112, 50)
            ax.plot(gx, a * gx + b, color=col, lw=1.7, zorder=2)
            ax.scatter(xs, ys, s=42, color=col, edgecolor="white", lw=1.1, zorder=3,
                       label=f"{f} MHz   a={a:.2f} W/SM,  b={b:.1f} W")
        ax.set_xlim(0, 112)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("SMs available to GEMM", fontsize=9.5)
        ax.set_title(f"M={m}, N=K={nk}", fontsize=10.5, pad=6)
        ax.grid(color="#ececec", zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(fontsize=7.9, loc="upper left", frameon=False, handletextpad=0.4,
                  labelspacing=0.32)
        ax.tick_params(labelsize=8.5)
    axes[0].set_ylabel("Dynamic power (W)", fontsize=10)
    fig.suptitle("Dynamic power is linear in SM count — but the line does not pass "
                 "through the origin", fontsize=15, fontweight="bold", y=0.965)
    fig.text(0.5, 0.885,
             "Fitted  P_dyn = a·S + b.   a = what one SM costs while it computes;  "
             "b = the part that does not scale with SMs at all (L2, memory controller, interconnect).\n"
             "Assuming \"70% of the SMs ⇒ 70% of the dynamic power\" is the b=0 line: "
             "R² 0.979 against 0.999 for the two-term fit, and it goes wrong fastest where S is small.",
             ha="center", va="top", fontsize=9.6, color="#333333")
    fig.savefig(out, dpi=135, facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--csv", default=os.path.join(here, "data", "gemm_dynamic_energy.csv"))
    ap.add_argument("--outdir", default=os.path.join(here, "figs"))
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    D = load(a.csv)
    fig_decomposition(D, os.path.join(a.outdir, "gemm_energy_decomposition.png"))
    fig_model(D, os.path.join(a.outdir, "gemm_dynamic_power_model.png"))


if __name__ == "__main__":
    main()
