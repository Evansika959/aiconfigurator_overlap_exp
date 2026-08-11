#!/usr/bin/env python3
"""Colormap view of the GEMM DVFS x SM-count sweep.  NO NORMALISATION anywhere.

Design constraints that come straight from what has already gone wrong here:

* NO NORMALISATION.  An earlier row-normalised heatmap inverted the reading --
  0.207 ms rendered as "worst" and 1.96 ms as "best" because each row was divided
  by its own max.  Every cell here carries its ABSOLUTE value, printed, in real
  units.  The colourbars are shared per facet-row and labelled in those same units,
  so colour never means a ratio.

* THE X AXIS IS THE ACHIEVED CLOCK, NOT THE REQUESTED ONE.  `nvidia-smi -lgc f,f`
  is a request.  At a requested 1410 MHz the board clamps to 1245-1380 MHz on
  SwPowerCap for every shape at 108 and 81 SM -- and the clamp is CORRELATED WITH
  THE SM AXIS (fewer SMs draw less power and therefore hold the clock), so inside
  that column the SM effect and the clock effect are not separable.  Those cells
  are hatched and carry their achieved MHz.  Read SM scaling off <= 1200 MHz,
  where all 900 cells held the request exactly.

* 3 reps -> median.  Latency CV was <= 0.6%, power CV <= 3.8%.

  python3 plot_sweep.py                       # -> figs/gemm_{energy,latency,power}_map.png
"""

import argparse
import collections
import csv
import math
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize
from matplotlib.patches import Patch, Rectangle

CLOCK_TOL_MHZ = 20
SM_ORDER = [108, 81, 54, 27, 14]            # y, top -> bottom
FREQ_ORDER = [300, 510, 705, 900, 1200, 1410]  # x, left -> right (requested)
M_ORDER = [1024, 2048, 4096, 8192]
NK_ORDER = [4096, 8192, 16384]

_E = lambda v: f"{v:,.0f}" if v >= 100 else f"{v:.1f}"
QUANTITIES = {
    "energy": dict(col="energy_mj", unit="mJ", label="Energy per GEMM call (mJ)",
                   log=True, fmt=_E, cmap="viridis"),
    # only present once make_dynamic_db.py has joined the P0 static table on
    "energy_dyn": dict(col="energy_dynamic_mj", unit="mJ", optional=True,
                       label="DYNAMIC energy per GEMM call (mJ) — board static power subtracted",
                       log=True, fmt=_E, cmap="viridis"),
    "p_dyn": dict(col="p_dynamic_w", unit="W", optional=True,
                  label="Dynamic power (W) — board power minus P0 static at the achieved clock",
                  log=False, fmt=lambda v: f"{v:.0f}", cmap="magma"),
    "latency": dict(col="latency_ms", unit="ms", label="Latency per GEMM call (ms)",
                    log=True, fmt=lambda v: f"{v:,.0f}" if v >= 100 else
                    (f"{v:.1f}" if v >= 10 else f"{v:.2f}"),
                    cmap="viridis"),
    "power": dict(col="power_w", unit="W", label="Board power, NVML median (W)",
                  log=False, fmt=lambda v: f"{v:.0f}", cmap="magma"),
}


def load(path):
    """-> (cells, quantities present).  cells[(m, nk, sm, freq)] = per-quantity median."""
    raw = collections.defaultdict(list)
    header = None
    for r in csv.DictReader(open(path)):
        header = header or set(r)
        if r["status"] != "ok" or r["probe_ok"] != "True":
            continue
        key = (int(r["m"]), int(r["n"]), int(r["sm_avail"]), int(r["freq_req_mhz"]))
        raw[key].append(r)
    # the dynamic-energy columns only exist after make_dynamic_db.py has run
    present = [q for q, s in QUANTITIES.items()
               if s["col"] in header or not s.get("optional")]
    cells = {}
    for key, rs in raw.items():
        freq_req = key[3]
        clk = int(statistics.median(int(x["clock_sm_med"]) for x in rs))
        # The mark means ONE thing: the achieved clock is not the one the column
        # claims, so the cell is at a different operating point than its label.
        # A raised SwPowerCap bit alone is NOT that -- 1 of 360 cells reported the
        # bit while holding 1410 MHz exactly, and marking it would say "throttled
        # @1410", which is self-contradictory. Frequency held => plain cell.
        cells[key] = dict(
            n_rep=len(rs),
            clock_ach=clk,
            dropped=(clk < freq_req - CLOCK_TOL_MHZ) or any(
                x["clock_held"] != "True" for x in rs),
            **{q: statistics.median(float(x[QUANTITIES[q]["col"]]) for x in rs)
               for q in present},
        )
    return cells, present


def draw(cells, qname, out):
    spec = QUANTITIES[qname]
    fig, axes = plt.subplots(
        len(NK_ORDER), len(M_ORDER), figsize=(24.0, 15.0),
        gridspec_kw=dict(left=0.055, right=0.905, top=0.915, bottom=0.070,
                         wspace=0.14, hspace=0.26))

    for i, nk in enumerate(NK_ORDER):
        # one ABSOLUTE colour scale per facet-row (fixed N=K); units stay real
        vals = [cells[k][qname] for k in cells if k[1] == nk]
        if spec["log"]:
            norm = LogNorm(vmin=min(vals), vmax=max(vals))
        else:
            norm = Normalize(vmin=min(vals), vmax=max(vals))
        cmap = plt.get_cmap(spec["cmap"])

        for j, m in enumerate(M_ORDER):
            ax = axes[i][j]
            grid = np.full((len(SM_ORDER), len(FREQ_ORDER)), np.nan)
            for a, sm in enumerate(SM_ORDER):
                for b, f in enumerate(FREQ_ORDER):
                    c = cells.get((m, nk, sm, f))
                    if c:
                        grid[a, b] = c[qname]
            ax.imshow(grid, cmap=cmap, norm=norm, aspect="auto",
                      interpolation="nearest")

            for a, sm in enumerate(SM_ORDER):
                for b, f in enumerate(FREQ_ORDER):
                    c = cells.get((m, nk, sm, f))
                    if not c:
                        continue
                    # text colour picked against the cell it sits on
                    lum = np.mean(cmap(norm(c[qname]))[:3])
                    tc = "black" if lum > 0.55 else "white"
                    if c["dropped"]:
                        ax.add_patch(Rectangle(
                            (b - .5, a - .5), 1, 1, fill=False, hatch="////",
                            edgecolor="#e8453c", linewidth=1.9, zorder=3))
                        ax.text(b, a - 0.17, spec["fmt"](c[qname]), ha="center",
                                va="center", fontsize=13.5, color=tc,
                                fontweight="bold", zorder=4)
                        ax.text(b, a + 0.25, f"ran @{c['clock_ach']}", ha="center",
                                va="center", fontsize=10.0, color=tc, zorder=4)
                    else:
                        ax.text(b, a, spec["fmt"](c[qname]), ha="center", va="center",
                                fontsize=13.5, color=tc, fontweight="bold", zorder=4)

            ax.set_xticks(range(len(FREQ_ORDER)))
            ax.set_xticklabels([str(f) for f in FREQ_ORDER], fontsize=12.5)
            for b, f in enumerate(FREQ_ORDER):
                if any(cells.get((m, nk, sm, f), {}).get("dropped")
                       for sm in SM_ORDER):
                    ax.get_xticklabels()[b].set_color("#e8453c")
            ax.set_yticks(range(len(SM_ORDER)))
            ax.set_yticklabels([str(s) for s in SM_ORDER], fontsize=12.5)
            gf = 2 * m * nk * nk / 1e9
            ax.set_title(f"M={m},  N=K={nk}    ({gf:,.0f} GFLOP)",
                         fontsize=14.5, pad=7)
            if i == len(NK_ORDER) - 1:
                ax.set_xlabel("SM clock (MHz)", fontsize=13.5)
            if j == 0:
                ax.set_ylabel("SMs available to GEMM", fontsize=13.5)
            ax.tick_params(length=0)
            for s in ax.spines.values():
                s.set_visible(False)

        cax = fig.add_axes([0.917, axes[i][0].get_position().y0, 0.011,
                            axes[i][0].get_position().height])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
        cb.set_label(f"{spec['unit']}   (N=K={nk} row)", fontsize=12.5)
        cb.ax.tick_params(labelsize=11)

    fig.suptitle(
        f"A100-SXM4-40GB · bf16 GEMM · {spec['label']} — absolute values, no normalisation",
        fontsize=21, fontweight="bold", y=0.975)
    fig.legend(handles=[
        Patch(facecolor="none", edgecolor="#e8453c", hatch="////", linewidth=1.9,
              label="clock DROPPED — hit the 400 W SwPowerCap and ran at the MHz printed below the value "
                    "(21/360 cells, all at 108 or 81 SM).  The clamp eases as SMs are removed, so within this "
                    "column the SM effect and the clock effect are not separable.")],
        loc="lower center", bbox_to_anchor=(0.5, 0.006), frameon=False, fontsize=12.5)
    fig.savefig(out, dpi=135, facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--csv", default=os.path.join(here, "data", "gemm_squat_sweep.csv"))
    ap.add_argument("--outdir", default=os.path.join(here, "figs"))
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    cells, present = load(a.csv)
    nd = sum(1 for c in cells.values() if c["dropped"])
    print(f"{len(cells)} cells (median of {min(c['n_rep'] for c in cells.values())}-"
          f"{max(c['n_rep'] for c in cells.values())} reps); {nd} clock-dropped")
    print(f"quantities: {', '.join(present)}")
    for q in present:
        draw(cells, q, os.path.join(a.outdir, f"gemm_{q}_map.png"))


if __name__ == "__main__":
    main()
