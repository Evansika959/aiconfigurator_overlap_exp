#!/usr/bin/env python3
"""Multi-dimensional DVFS operating-point heatmaps for the A100 bf16 GEMM sweep.

For each metric (energy / latency / power) this renders a 4x3 FACET GRID of
freq x SM heatmaps -- using all four sweep dimensions at once:

    facet row     = M            (1024, 2048, 4096, 8192)
    facet column  = (N, K)       (4096^2, 8192^2, 16384^2)
    within a panel: x = SM clock (MHz),  y = active SMs (via MPS %)
    cell color    = the metric

Energy & latency are colored by per-panel min-max normalization (0 = best, 1 = worst
within each shape), so every panel uses the full color range and the DVFS "knee" is
visible.  Power is colored by absolute Watts on a single global scale.  Throttled cells
(clock-lock exceeded by the power cap) are red-hatched.  Every cell is annotated with
its absolute value.

Publication (NeurIPS) house style: sans fonts, top/right spines off, 300-dpi PNG.

  python3 plot_heatmap.py                                   # data/ -> figures/
  python3 plot_heatmap.py --data <csv> --outdir <dir>
"""
import argparse, csv, os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

PALETTE = {"blue_main": "#0F4D92", "highlight": "#FFD700", "red_strong": "#B64342"}

METRICS = {
    "energy":  dict(key="energy",  title="Energy per GEMM", unit="mJ", cmap="viridis", mode="panel_norm"),
    "latency": dict(key="latency", title="Latency",         unit="ms", cmap="magma",   mode="panel_norm"),
    "power":   dict(key="power",   title="Power",            unit="W",  cmap="inferno", mode="abs"),
}


def apply_publication_style():
    plt.rcParams.update({
        # DejaVu Sans first: it ships with matplotlib, so this renders identically
        # everywhere. Swap Arial/Helvetica to the front if they are installed.
        "font.family": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
        "font.size": 15,
        "axes.spines.right": False, "axes.spines.top": False,
        "axes.linewidth": 1.6, "legend.frameon": False,
        "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.dpi": 120,
    })


def load(path):
    rows = []
    for r in csv.DictReader(open(path)):
        rows.append(dict(M=int(r["m"]), N=int(r["n"]), K=int(r["k"]),
                         freq=int(r["freq_mhz"]), sm=int(r["sm_count"]),
                         latency=float(r["latency_ms"]), power=float(r["power_w"]),
                         energy=float(r["energy_mj"]), throttled=(r["throttled"] == "True")))
    return rows


def fmt(v, unit):
    return f"{v:.0f}" if unit == "W" else f"{v:.2g}"


def make_figure(rows, mname, outdir):
    m = METRICS[mname]
    Ms   = sorted(set(r["M"] for r in rows))                  # facet rows
    NKs  = sorted(set((r["N"], r["K"]) for r in rows))        # facet cols
    freqs = sorted(set(r["freq"] for r in rows))              # x, ascending
    sms   = sorted(set(r["sm"] for r in rows))                # y, ascending (origin lower)
    fi = {f: i for i, f in enumerate(freqs)}
    si = {s: i for i, s in enumerate(sms)}

    lookup = defaultdict(dict)
    for r in rows:
        lookup[(r["M"], r["N"], r["K"])][(r["sm"], r["freq"])] = r

    # per-panel value + throttle matrices
    panel = {}
    for M in Ms:
        for (N, K) in NKs:
            val = np.full((len(sms), len(freqs)), np.nan)
            thr = np.zeros_like(val, dtype=bool)
            for (s, f), r in lookup[(M, N, K)].items():
                val[si[s], fi[f]] = r[m["key"]]
                thr[si[s], fi[f]] = r["throttled"]
            panel[(M, (N, K))] = (val, thr)

    # color data + shared normalization
    color = {}
    if m["mode"] == "panel_norm":
        for k, (val, _) in panel.items():
            vmin, vmax = float(np.nanmin(val)), float(np.nanmax(val))
            color[k] = (val - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(val)
        norm = Normalize(vmin=0.0, vmax=1.0)
        cbar_label = f"{m['title']} — normalized within shape\n(0 = best  ·  1 = worst)"
    else:
        allv = np.concatenate([val.ravel() for val, _ in panel.values()])
        norm = Normalize(vmin=float(np.nanmin(allv)), vmax=float(np.nanmax(allv)))
        for k, (val, _) in panel.items():
            color[k] = val
        cbar_label = f"{m['title']} ({m['unit']})"
    cmap = plt.get_cmap(m["cmap"])

    nR, nC = len(Ms), len(NKs)
    fig, axes = plt.subplots(nR, nC, figsize=(4.1 * nC, 3.35 * nR),
                             squeeze=False, constrained_layout=True)
    for ri, M in enumerate(Ms):
        for ci, (N, K) in enumerate(NKs):
            ax = axes[ri][ci]
            val, thr = panel[(M, (N, K))]
            c = color[(M, (N, K))]
            ax.imshow(c, origin="lower", aspect="auto", cmap=cmap, norm=norm)
            for yi in range(len(sms)):
                for xi in range(len(freqs)):
                    if np.isnan(val[yi, xi]):
                        continue
                    tc = "white" if norm(c[yi, xi]) < 0.55 else "black"
                    ax.text(xi, yi, fmt(val[yi, xi], m["unit"]), ha="center", va="center",
                            fontsize=6.2, color=tc)
                    if thr[yi, xi]:
                        ax.add_patch(Rectangle((xi - 0.5, yi - 0.5), 1, 1, fill=False,
                                               edgecolor=PALETTE["red_strong"], lw=1.5, hatch="///"))
            ax.set_xticks(range(len(freqs)))
            ax.set_xticklabels(freqs, fontsize=8, rotation=45)
            ax.set_yticks(range(len(sms)))
            ax.set_yticklabels(sms, fontsize=8)
            ax.tick_params(length=0)
            for sp in ax.spines.values():
                sp.set_visible(False)
            if ri == 0:
                ax.set_title(f"N=K={N}", fontsize=13, pad=6)
            if ci == 0:
                ax.set_ylabel(f"M={M}", fontsize=13, fontweight="bold")

    sm_map = ScalarMappable(norm=norm, cmap=cmap)
    sm_map.set_array([])
    cb = fig.colorbar(sm_map, ax=axes, shrink=0.55, pad=0.015, aspect=32)
    cb.set_label(cbar_label, fontsize=12)
    cb.ax.tick_params(labelsize=9)

    marks = []
    if any(thr.any() for _, thr in panel.values()):
        marks.append("red hatch = throttled (power-capped)")
    title = (f"A100 bf16 GEMM — {m['title']} across DVFS operating points "
             f"(SM clock × active SMs)")
    if marks:
        title += "\n" + "    ·    ".join(marks)
    fig.suptitle(title, fontsize=12.5)
    fig.supxlabel("SM clock (MHz)", fontsize=13)
    fig.supylabel("Active SMs (MPS-capped)", fontsize=13)

    os.makedirs(outdir, exist_ok=True)
    base = os.path.join(outdir, f"gemm_{mname}_opmap")
    fig.savefig(base + ".png", dpi=300)
    plt.close(fig)
    return base


def main():
    HERE = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data", "gemm_char_merged.csv"))
    ap.add_argument("--outdir", default=os.path.join(HERE, "figures"))
    a = ap.parse_args()
    apply_publication_style()
    rows = load(a.data)
    print(f"loaded {len(rows)} rows from {a.data}")
    for mname in METRICS:
        base = make_figure(rows, mname, a.outdir)
        print(f"wrote {base}.png")


if __name__ == "__main__":
    main()
