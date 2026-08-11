#!/usr/bin/env python3
"""Facet-grid operating maps: SM clock x NCCL CTAs, one facet per message size.

CTAs, not channels, not MPS. A CTA (cooperative thread array) is a thread block,
one resident per SM, so NCCL_MIN_CTAS = NCCL_MAX_CTAS = N is a direct request for
N SMs of collective footprint. NCCL supersedes NCCL_MAX_NCHANNELS with it, and it
needs no MPS daemon. Verified on this box: N=1 -> "1 p2p channels", N=32 -> "32".
It governs the NCCL path only; TRT-LLM's own MIN_LATENCY kernels ignore it.

COLOUR IS ABSOLUTE -- no normalization of any kind. One log scale spans every
facet, so a given colour means the same number everywhere in the figure and the
~1000x span across message size and CTA count is visible rather than hidden. An
earlier version normalized within each row, which made a 0.207 ms cell render as
"worst" while a 1.96 ms cell rendered as "best".

Achieved SM clock is recorded per configuration (clock_sm_min/mean, sampled by
PowerMonitor alongside power). A locked clock is a request, not a guarantee; any
cell whose achieved minimum fell more than 20 MHz below the request is outlined
in red. On the shipped data that is zero cells out of 288.

  python3 plot_cta_opmap.py                       # -> figures/comm_cta_{metric}_opmap.png
"""

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import Rectangle

SEQ = LinearSegmentedColormap.from_list(
    "house_blue", ["#F4F8FC", "#CFDFEF", "#93B6D9", "#4C81B8", "#0F4D92", "#08305F"])
INK, MUTED, ACCENT = "#1A1A1A", "#8A8A8A", "#B64342"

METRICS = {
    "energy":  dict(key="ej",  title="Energy per collective", unit="mJ"),
    "latency": dict(key="lat", title="Latency",               unit="ms"),
    "power":   dict(key="pw",  title="Power",                 unit="W"),
    "busbw":   dict(key="bw",  title="Bus bandwidth",         unit="GB/s"),
}


def style():
    plt.rcParams.update({
        "font.family": ["DejaVu Sans", "sans-serif"], "font.size": 9,
        "axes.linewidth": 0.8, "axes.edgecolor": MUTED, "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42, "figure.dpi": 120,
    })


def load(path):
    R = []
    for r in csv.DictReader(open(path)):
        if not r["power_w"] or r["nccl_ctas"] == "":
            continue
        R.append(dict(cta=int(r["nccl_ctas"]), freq=int(r["freq_mhz"]), mib=float(r["msg_mib"]),
                      lat=float(r["latency_ms"]), pw=float(r["power_w"]), ej=float(r["energy_mj"]),
                      bw=float(r["busbw_gbps"]), thr=(r["throttled"] == "True")))
    return R


def fmt(v):
    """2 significant figures: 3 is false precision against a 2.4% run-to-run floor."""
    if v >= 1000:
        return f"{v:,.0f}"
    if v >= 10:
        return f"{v:.0f}"
    return f"{v:.2g}"


def make(R, mname, outdir):
    m = METRICS[mname]
    SZ = sorted({r["mib"] for r in R})
    FR = sorted({r["freq"] for r in R})
    CT = sorted({r["cta"] for r in R})
    g = lambda c, f, z: [r for r in R if r["cta"] == c and r["freq"] == f and r["mib"] == z][0]

    panel = {z: np.array([[g(c, f, z)[m["key"]] for f in FR] for c in CT]) for z in SZ}
    thr = {z: np.array([[g(c, f, z)["thr"] for f in FR] for c in CT]) for z in SZ}

    allv = np.concatenate([v.ravel() for v in panel.values()])
    norm = LogNorm(vmin=float(allv.min()), vmax=float(allv.max()))

    n = len(SZ)
    fig, axes = plt.subplots(1, n, figsize=(1.95 * n + 2.0, 3.85), squeeze=False,
                             gridspec_kw={"wspace": 0.15, "left": 0.062, "right": 0.865,
                                          "top": 0.775, "bottom": 0.285})
    for ci, z in enumerate(SZ):
        ax = axes[0][ci]
        val = panel[z]
        ax.pcolormesh(np.arange(len(FR) + 1), np.arange(len(CT) + 1), val,
                      cmap=SEQ, norm=norm, edgecolors="white", linewidth=1.1)
        for i in range(len(CT)):
            for j in range(len(FR)):
                ax.text(j + .5, i + .5, fmt(val[i, j]), ha="center", va="center", fontsize=6.8,
                        color="white" if norm(val[i, j]) > 0.60 else INK)
                if thr[z][i, j]:
                    ax.add_patch(Rectangle((j, i), 1, 1, fill=False, edgecolor=ACCENT, lw=1.4))
        ax.set_xticks(np.arange(len(FR)) + .5); ax.set_xticklabels(FR, fontsize=7, rotation=90)
        ax.set_yticks(np.arange(len(CT)) + .5)
        ax.set_yticklabels(CT if ci == 0 else [], fontsize=7.5)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"{z:g} MiB", fontsize=9.5, pad=4)
    axes[0][0].set_ylabel("NCCL CTAs  (= SMs occupied\nby the collective)", fontsize=8.5)

    cax = fig.add_axes([0.882, 0.295, 0.010, 0.46])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=SEQ), cax=cax)
    cb.set_label(f"{m['title']} ({m['unit']})", fontsize=8)
    cb.ax.tick_params(labelsize=7.5); cb.outline.set_visible(False)

    fig.suptitle(f"A100 ×4 NVLink · bf16 all_reduce (TensorRT-LLM, strategy=NCCL) — {m['title']} "
                 "across SM clock × NCCL CTAs", fontsize=10, y=0.935)
    fig.text(0.5, 0.115, "SM clock (MHz)", ha="center", fontsize=9)
    nthr = sum(int(t.sum()) for t in thr.values())
    fig.text(0.012, 0.020,
             "Colour is an ABSOLUTE log scale shared by every facet — no normalization, so one colour means one value everywhere.  "
             f"SM footprint set with NCCL_MIN_CTAS = NCCL_MAX_CTAS (no MPS).\n"
             "Achieved SM clock sampled per configuration: "
             f"{nthr}/{len(allv)} cells fell >20 MHz below the locked value"
             + ("" if nthr else " — none; every point ran at its requested frequency") + ".\n"
             "Power and energy are the rank-0 GPU; node total ≈ 4×.  TensorRT-LLM 1.3.0rc10, NCCL 2.27.5.",
             fontsize=7, color=MUTED)

    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f"comm_cta_{mname}_opmap.png")
    fig.savefig(out, dpi=400); plt.close(fig)
    return out


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(here, "data", "comm_trtllm_ar_ctas.csv"))
    ap.add_argument("--outdir", default=os.path.join(here, "figures"))
    ap.add_argument("--metric", default="all", choices=["all", *METRICS])
    a = ap.parse_args()
    style()
    R = load(a.data)
    print(f"loaded {len(R)} rows")
    from PIL import Image
    for mn in (METRICS if a.metric == "all" else [a.metric]):
        f = make(R, mn, a.outdir)
        arr = np.array(Image.open(f).convert("L"))
        edge = [k for k, v in {"top": arr[0].min(), "bottom": arr[-1].min(),
                               "left": arr[:, 0].min(), "right": arr[:, -1].min()}.items() if v < 250]
        print(f"wrote {f}  ({arr.shape[1]/400:.2f} x {arr.shape[0]/400:.2f} in)  edge ink: {edge or 'none'}")


if __name__ == "__main__":
    main()
