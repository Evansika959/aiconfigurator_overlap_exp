#!/usr/bin/env python3
"""The static-power characterisation, in absolute watts.

This is the quantity subtracted from every row of the sweep, so it is worth showing
on its own rather than only as a number in a table.

  (a) P_static vs SM clock -- the curve actually used, from zeus's profile_p2p.py run
      unmodified at each locked clock, next to the two squatter-based readings for
      comparison. Clock is a request, so each point also carries the clock the board
      actually ran at.
  (b) P_static vs the number of SMs held by the __nanosleep squatter. The line being
      FLAT is what makes the whole approach work: a parked SM is indistinguishable
      from an idle one, so occupying SMs to control the GEMM's SM count does not
      itself add power that would be mistaken for the GEMM's.
  (c) P_static vs die temperature, from the post-GEMM cooldown. Included to show the
      size of the effect that is deliberately NOT corrected for (~3 % on E_dynamic).

  python3 plot_static.py            # -> figs/static_power.png
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

FREQS = [300, 510, 705, 900, 1200, 1410]
SQUAT_N = [0, 1, 14, 27, 54, 81, 94, 108]
C_ZEUS = "#1f7a8c"
C_SQUAT = "#c1666b"
C_BARE = "#7f8c8d"


def load_zeus(p):
    by = collections.defaultdict(list)
    for r in csv.DictReader(open(p)):
        by[int(r["freq_req_mhz"])].append(float(r["power_w"]))
    return by


def load_squat(p):
    by = collections.defaultdict(list)
    for r in csv.DictReader(open(p)):
        if r["status"] == "ok":
            by[(int(r["freq_req_mhz"]), int(r["n_squat"]))].append(
                (float(r["power_w"]), int(r["clock_sm_med"]), int(r["temp_c"])))
    return by


def load_temp(p):
    by = collections.defaultdict(list)
    for r in csv.DictReader(open(p)):
        by[int(r["clock_req_mhz"])].append((int(r["temp_c"]), float(r["power_w"])))
    return by


def fig_zeus_only(Z, out):
    """The measurement actually used, on its own: whole-board static power vs clock."""
    fig, ax = plt.subplots(figsize=(9.6, 6.2),
                           gridspec_kw=dict(left=0.098, right=0.972, top=0.80, bottom=0.115))
    y = [statistics.median(Z[f]) for f in FREQS]
    for f in FREQS:                                    # every rep, not just the median
        ax.scatter([f] * len(Z[f]), Z[f], s=26, color=C_ZEUS, alpha=0.40, zorder=3,
                   linewidths=0)
    ax.plot(FREQS, y, color=C_ZEUS, lw=2.6, marker="o", ms=10, zorder=4,
            markeredgecolor="white", markeredgewidth=1.6)
    for f, v in zip(FREQS, y):
        ax.annotate(f"{v:.1f} W", (f, v), textcoords="offset points", xytext=(0, 15),
                    ha="center", fontsize=11, fontweight="bold", color=C_ZEUS)
    ax.set_xticks(FREQS)
    ax.set_xlim(230, 1500)
    ax.set_ylim(52, 92)
    ax.set_xlabel("SM clock — locked with nvidia-smi -lgc and verified by read-back (MHz)",
                  fontsize=11)
    ax.set_ylabel("Whole-board static power (W)", fontsize=11)
    ax.grid(color="#ececec", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=10)
    ax.text(0.03, 0.955,
            "Whole GPU module — all 108 SMs, HBM, L2, memory controllers, interconnect\n"
            "and regulator losses. NVML reports one number for the board; per-SM static\n"
            "power is not separable from it, and does not need to be: this is subtracted\n"
            "from the board power measured during the GEMM, so both sides are the board.",
            transform=ax.transAxes, va="top", fontsize=9.6,
            bbox=dict(fc="#f4f7f8", ec="#c8d4d8", lw=0.9, pad=5))
    fig.suptitle("A100-SXM4-40GB · static (no-work) board power vs SM clock",
                 fontsize=15.5, fontweight="bold", y=0.972)
    fig.text(0.5, 0.895,
             "zeus profile_p2p.py run unmodified at each locked clock: rank 0 parked in a blocking NCCL recv for 60 s while rank 1 sleeps,\n"
             "board energy integrated by NVML.  Dots are the 3 individual reps (spread 0.13–0.33 W); the line is their median.",
             ha="center", va="top", fontsize=10, color="#333333")
    fig.savefig(out, dpi=145, facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--zeus", default=os.path.join(here, "data", "p0_static_zeus.csv"))
    ap.add_argument("--squat", default=os.path.join(here, "data", "p0_static.csv"))
    ap.add_argument("--temp", default=os.path.join(here, "data", "p0_static_vs_temp.csv"))
    ap.add_argument("--out", default=os.path.join(here, "figs", "static_power.png"))
    a = ap.parse_args()
    Z, S, T = load_zeus(a.zeus), load_squat(a.squat), load_temp(a.temp)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig_zeus_only(Z, os.path.join(os.path.dirname(os.path.abspath(a.out)),
                                  "static_power_zeus.png"))

    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.5),
                             gridspec_kw=dict(left=0.052, right=0.988, top=0.735,
                                              bottom=0.135, wspace=0.235))

    # ---- (a) P_static vs clock -------------------------------------------------
    ax = axes[0]
    zy = [statistics.median(Z[f]) for f in FREQS]
    ze = [(max(Z[f]) - min(Z[f])) / 2 for f in FREQS]
    sq = [statistics.median(v[0] for n in (1, 14, 27, 54, 81, 94, 108) for v in S[(f, n)])
          for f in FREQS]
    ba = [statistics.median(v[0] for v in S[(f, 0)]) for f in FREQS]
    # NB: matplotlib silently drops legend entries whose label starts with "_", so the
    # squatter series must not be labelled "__nanosleep ...".
    ax.errorbar(FREQS, zy, yerr=ze, color=C_ZEUS, lw=2.2, marker="o", ms=8,
                capsize=4, zorder=4, label="zeus profile_p2p.py  ← used for the subtraction")
    ax.plot(FREQS, sq, color=C_SQUAT, lw=2, marker="s", ms=7, ls="--", zorder=3,
            label="nanosleep squatter resident (n ≥ 1)")
    ax.plot(FREQS, ba, color=C_BARE, lw=2, marker="^", ms=7, ls=":", zorder=3,
            label="bare board, nothing resident (n = 0)")
    for f, y in zip(FREQS, zy):
        ax.annotate(f"{y:.1f}", (f, y), textcoords="offset points", xytext=(3, -15),
                    ha="left", fontsize=8.4, color=C_ZEUS, fontweight="bold")
    ax.set_xticks(FREQS)
    ax.set_xlim(240, 1490)
    ax.set_ylim(53, 92)
    ax.set_xlabel("SM clock, locked and verified (MHz)", fontsize=10)
    ax.set_ylabel("Static board power (W)", fontsize=10)
    ax.set_title("(a)  P_static vs clock — the curve that gets subtracted",
                 fontsize=11, pad=7)
    ax.legend(fontsize=8.4, loc="upper left", frameon=False)
    # placed in the empty upper-middle wedge; at bottom-right it sat on the 705-900 MHz
    # markers, and the legend already occupies the upper left
    ax.text(0.40, 0.62,
            f"zeus reads {statistics.median([z - s for z, s in zip(zy, sq)]):+.1f} W vs the "
            f"squatter at every clock:\na near-constant offset, so it moves the fit's\n"
            f"intercept b, not its slope a.",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=8.4,
            bbox=dict(fc="#fff8e1", ec="#e0c97f", lw=0.8, pad=3.2))

    # ---- (b) what occupying SMs ADDS -------------------------------------------
    # Plotted as watts ABOVE the same clock's bare-board reading, not as absolute
    # power: on a shared absolute axis the 300-900 MHz curves (58.9-64.4 W) collapse
    # onto each other and the flatness -- the whole point -- is unreadable. This is a
    # difference of two measured absolute values, still in watts; nothing is divided.
    ax = axes[1]
    cmap = plt.get_cmap("viridis")
    for i, f in enumerate(FREQS):
        col = cmap(0.06 + 0.80 * i / (len(FREQS) - 1))
        base = statistics.median(v[0] for v in S[(f, 0)])
        ys = [statistics.median(v[0] for v in S[(f, n)]) - base for n in SQUAT_N]
        ax.plot(SQUAT_N, ys, color=col, lw=1.9, marker="o", ms=5.5, zorder=3,
                label=f"{f} MHz   ({base:.1f} W bare)   n=1→108 spans {max(ys[1:]) - min(ys[1:]):.1f} W")
    ax.axhline(0, color="#999999", lw=1, ls="--", zorder=1)
    ax.set_xlabel("SMs held by the nanosleep squatter", fontsize=10)
    ax.set_ylabel("Power ABOVE the bare board at the same clock (W)", fontsize=10)
    ax.set_title("(b)  a parked SM draws the same as an idle one", fontsize=11, pad=7)
    ax.set_xticks([0, 14, 27, 54, 81, 94, 108])
    ax.set_ylim(-0.9, 11.4)
    ax.legend(fontsize=8.0, loc="upper center", frameon=False, ncol=1, labelspacing=0.3)
    ax.text(0.5, 0.035,
            "The whole step is at n=0→1 — the fixed cost of having ANY resident kernel.\n"
            "From 1 SM to 108 every curve is flat to ≤0.6 W. Were it not, occupying SMs\n"
            "would add power that the subtraction would charge to the GEMM.",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=8.4,
            bbox=dict(fc="#fff8e1", ec="#e0c97f", lw=0.8, pad=3.2))

    # ---- (c) leakage vs temperature --------------------------------------------
    # Same reason as (b): 1200 MHz sits ~12 W above 300 MHz, so on a shared absolute
    # axis the 300 MHz trace is a flat smear. Referenced to each curve's own coldest
    # sample -- again a difference in watts, not a normalisation.
    ax = axes[2]
    for f in sorted(T, reverse=True):
        col = cmap(0.06 + 0.80 * (FREQS.index(f) / (len(FREQS) - 1)))
        by = collections.defaultdict(list)
        for t, p in T[f]:
            by[t].append(p)
        xs = sorted(by)
        ys = np.array([statistics.median(by[t]) for t in xs])
        cnt = [len(by[t]) for t in xs]
        base = ys[0]
        ax.scatter(xs, ys - base, s=[min(95, 14 + 0.4 * k) for k in cnt], color=col,
                   zorder=3, edgecolor="white", lw=0.9)
        g, c = np.polyfit(xs, ys, 1)
        gx = np.linspace(min(xs) - 0.6, max(xs) + 0.6, 20)
        ax.plot(gx, g * gx + c - base, color=col, lw=1.9, zorder=2,
                label=f"{f} MHz  ({base:.1f} W at {xs[0]} °C)   γ = {g:.3f} W/°C")
    ax.axhline(0, color="#999999", lw=1, ls="--", zorder=1)
    ax.set_xlabel("Die temperature (°C)", fontsize=10)
    ax.set_ylabel("Power above the same curve's coldest sample (W)", fontsize=10)
    ax.set_title("(c)  leakage vs temperature — measured, not corrected for",
                 fontsize=11, pad=7)
    ax.legend(fontsize=8.2, loc="upper left", frameon=False, labelspacing=0.35)
    ax.set_ylim(-0.6, 5.0)
    ax.text(0.5, 0.035,
            "Only 34–44 °C is reachable: the die falls 62→38 °C in under 5 s once the\n"
            "GEMM stops. Propagating γ through each sweep row moves E_dynamic by a\n"
            "median 2.8 %, so no temperature correction is applied.",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=8.4,
            bbox=dict(fc="#fff8e1", ec="#e0c97f", lw=0.8, pad=3.2))

    for ax in axes:
        ax.grid(color="#ececec", zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=9)

    fig.suptitle("A100-SXM4-40GB · static (no-work) board power — absolute watts, "
                 "no normalisation", fontsize=15, fontweight="bold", y=0.965)
    fig.text(0.5, 0.885,
             "Static power is what the board draws with the clock up and nothing computing. "
             "It is a median 47 % of the board power measured during these GEMMs "
             "(18 % at the hot corner, 84 % at 300 MHz / 14 SM),\n"
             "so it has to come off before any energy number can be called the kernel's. "
             "zeus's method is used end to end; the squatter curves are shown only to bound "
             "what that choice costs.",
             ha="center", va="top", fontsize=9.6, color="#333333")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=135, facecolor="white")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
