#!/usr/bin/env python3
"""How M, N, K become tiles, and how tiles become waves.

The two are constantly confused, so this draws them separately:

  a TILE is a piece of WORK  -- a 256x128 block of the OUTPUT matrix C, computed by
                               one thread block. Its count depends on M and N only.
  a WAVE is a unit of TIME   -- one round in which up to S tiles run at once, S being
                               the SMs available. Its count is ceil(tiles / S).

K never creates tiles. It is the reduction depth *inside* each tile, so it makes every
tile take longer without making more of them. That is exactly why a low-occupancy
shape can be stretched into a long kernel by raising K alone.

  python3 plot_tiling.py       # -> figs/tiling_and_waves.png
"""

import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

TM, TN, SM = 256, 128, 108
C_BUSY, C_IDLE, C_A, C_B, C_C = "#1f7a8c", "#e9edf0", "#c1666b", "#8f9e6b", "#4a6fa5"


def draw_matrices(ax, M, N, K):
    """A[M,K] x B[K,N] -> C[M,N], with C carrying the tile grid."""
    nm, nn = math.ceil(M / TM), math.ceil(N / TN)
    aw, cw, h = 1.5, 3.6, 2.2
    ax.add_patch(Rectangle((0, 0), aw, h, fc=C_A, alpha=.75, ec="white"))
    ax.text(aw / 2, h + .18, f"A  [{M} x {K}]", ha="center", fontsize=10, fontweight="bold")
    ax.text(aw / 2, -.28, f"K = {K}", ha="center", fontsize=9.5, color=C_A,
            fontweight="bold")
    ax.text(-.16, h / 2, f"M = {M}", va="center", ha="right", rotation=90, fontsize=9.5,
            color=C_A, fontweight="bold")
    ax.text(aw + .32, h / 2, "x", ha="center", va="center", fontsize=17)

    bx = aw + .65
    ax.add_patch(Rectangle((bx, 0), 1.5, h, fc=C_B, alpha=.75, ec="white"))
    ax.text(bx + .75, h + .18, f"B  [{K} x {N}]", ha="center", fontsize=10,
            fontweight="bold")
    ax.text(bx + 2.05, h / 2, "=", ha="center", va="center", fontsize=17)

    cx = bx + 2.4
    for i in range(nm):                       # the tile grid IS the launch grid
        for j in range(nn):
            ax.add_patch(Rectangle((cx + j * cw / nn, h - (i + 1) * h / nm),
                                   cw / nn, h / nm, fc=C_C, alpha=.55,
                                   ec="white", lw=1.1))
    ax.text(cx + cw / 2, h + .18, f"C  [{M} x {N}]   <- tiled here", ha="center",
            fontsize=10, fontweight="bold")
    ax.text(cx + cw / 2, -.28,
            f"M = {M} -> {nm} tiles of {TM}      N = {N} -> {nn} tiles of {TN}",
            ha="center", fontsize=9.5, color=C_C, fontweight="bold")
    ax.annotate(f"one tile = {TM}x{TN} of C\n= one thread block\n"
                f"= 147456 B smem\n=> one SM at a time",
                xy=(cx + cw / nn / 2, h - h / nm / 2),
                xytext=(cx + cw + .45, h * .78), fontsize=8.6,
                arrowprops=dict(arrowstyle="->", color="#666", lw=1.2),
                bbox=dict(fc="#fff8e1", ec="#e0c97f", lw=.8, pad=3))
    ax.text(cx + cw / 2, -.72,
            f"tiles = ceil({M}/{TM}) x ceil({N}/{TN}) = {nm} x {nn} = {nm * nn}"
            f"      (K does NOT appear)",
            ha="center", fontsize=10.5, fontweight="bold", color="#222")
    ax.set_xlim(-1.0, cx + cw + 2.5); ax.set_ylim(-1.0, h + .55)
    ax.axis("off")


def draw_waves(ax, M, N, label):
    tiles = math.ceil(M / TM) * math.ceil(N / TN)
    W = math.ceil(tiles / SM)
    s_eff = tiles / W
    for w in range(W):
        busy = min(SM, tiles - w * SM)
        ax.add_patch(Rectangle((w, 0), .9, busy, fc=C_BUSY, ec="none"))
        if busy < SM:
            ax.add_patch(Rectangle((w, busy), .9, SM - busy, fc=C_IDLE, ec="#c9ced6",
                                   lw=.8, hatch="///"))
    ax.axhline(SM, color="#333", lw=1.3, ls="--")
    ax.text(W * .5, SM + 9, f"S = {SM} SMs allocated", ha="center", fontsize=9.5,
            fontweight="bold")
    ax.axhline(s_eff, color="#e8453c", lw=2)
    # when S_eff is close to S the two labels collide; drop the red one below its line
    below = (SM - s_eff) < 20
    ax.text(W * .5, s_eff - 7 if below else s_eff + 4,
            f"S_eff = tiles / waves = {tiles}/{W} = {s_eff:.1f}",
            ha="center", va="top" if below else "bottom", fontsize=9.5,
            color="#e8453c", fontweight="bold",
            bbox=dict(fc="white", ec="#e8453c", lw=.8, alpha=.92, pad=2.5))
    if W <= 6:
        for w in range(W):
            busy = min(SM, tiles - w * SM)
            ax.text(w + .45, busy / 2, f"{busy}\ntiles", ha="center", va="center",
                    fontsize=8.5, color="white", fontweight="bold")
    ax.set_xlim(-.3, W + .3); ax.set_ylim(0, SM * 1.22)
    ax.set_xlabel(f"wave  (each lasts one tile-time)   ->   {W} waves total", fontsize=9.5)
    ax.set_ylabel("SMs busy", fontsize=9.5)
    ax.set_title(f"{label}\ntiles={tiles}, waves=ceil({tiles}/{SM})={W}, "
                 f"time-averaged busy SMs = {s_eff:.1f}/{SM} = {s_eff / SM * 100:.0f}%",
                 fontsize=10.5, pad=7)
    ax.set_xticks(range(0, W, max(1, W // 8)))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8.5)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    fig = plt.figure(figsize=(16.5, 9.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1], top=.795, bottom=.075,
                          left=.055, right=.975, hspace=.42, wspace=.19)
    ax0 = fig.add_subplot(gs[0, :])
    draw_matrices(ax0, 1024, 4096, 4096)
    ax0.set_title("(a)  M and N decide how many tiles.  K only decides how long each "
                  "tile takes.", fontsize=12, fontweight="bold", pad=12)
    draw_waves(fig.add_subplot(gs[1, 0]), 1024, 4096,
               "(b)  M=1024, N=4096  -- the wasteful case")
    draw_waves(fig.add_subplot(gs[1, 1]), 8192, 16384,
               "(c)  M=8192, N=16384  -- the efficient case")
    fig.suptitle("A100 - how a GEMM's M, N, K become tiles, and tiles become waves",
                 fontsize=15.5, fontweight="bold", y=.968)
    fig.text(.5, .885,
             "One 256x128 tile of C per thread block; the block's 147456 B of shared memory "
             "means one block per SM, so at most S tiles run at once.\n"
             "Latency counts WAVES (a half-empty wave still costs a full tile-time). "
             "Dynamic power counts S_eff (an idle SM draws none). Measured by ncu: "
             "0.5855 vs 0.5926 predicted in (b), 0.9845 vs 0.9981 in (c).",
             ha="center", va="top", fontsize=9.7, color="#333")
    out = os.path.join(here, "figs", "tiling_and_waves.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=135, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
