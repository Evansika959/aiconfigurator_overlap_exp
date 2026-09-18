#!/usr/bin/env python3
"""fig22 -- the whole two-domain search space, as measured.

One panel per achievable work split. Within a panel, every (f_hot, f_cold) pair the
hardware can be set to. Colour is the quantity that decides the design: energy against
the best SINGLE clock that meets the same latency. Blue is a win, orange a loss, white
a tie.

Only the lower triangle exists, because f_cold <= f_hot by definition. The diagonal is
f_cold = f_hot, which is not a split at all -- it reproduces the single-clock design
exactly, which is why it is white.

The red box marks the best cell in each panel.

Reading it: the wins are a thin band hugging the diagonal, where the two clocks differ
by one or two grid steps. Move away from the diagonal -- a real voltage separation --
and the design loses, by up to 25%.
"""
import os, sys, importlib.util
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs

HERE = os.path.dirname(os.path.abspath(__file__))
GRID = None       # the full measured grid, filled in from the sweep
PHI = [(0.016, 8), (0.045, 16), (0.120, 32), (0.208, 48), (0.324, 64), (0.596, 96)]
VMAX = 6.0          # % ; clipped, the worst cells reach +25%


def main():
    spec = importlib.util.spec_from_file_location(
        "p", os.path.join(HERE, "predict_two_domain.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    cl, T, E, D, PS = mod.load_sweep()
    # Every clock that was actually measured and held -- NOT a coarse subset. A coarse
    # grid makes the single-clock baseline coarse too, and the two-domain design then
    # scores well simply by landing between the baseline's steps: on a 13-clock grid the
    # best cell reads -3.21%, on the real measured grid the same search gives -1.53%.
    global GRID
    GRID = cl
    Tg, Eg, Dg, Pg = T, E, D, PS
    n = len(GRID)

    fs.use()
    fig, axes = plt.subplots(2, 3, figsize=(fs.TEXT_W, 3.95))
    fig.subplots_adjust(left=0.085, right=0.855, top=0.845, bottom=0.215,
                        wspace=0.30, hspace=0.42)
    im = None
    for ax, (phi, L) in zip(axes.ravel(), PHI):
        M = np.full((n, n), np.nan)
        for ih in range(n):
            for il in range(ih + 1):
                t = Tg[ih] * (phi * GRID[ih] / GRID[il] + 1 - phi)
                s = phi * GRID[ih] / (phi * GRID[ih] + (1 - phi) * GRID[il])
                e = (Dg[il] * phi + Dg[ih] * (1 - phi)
                     + (s * Pg[il] + (1 - s) * Pg[ih]) * t)
                mk = Tg <= t + 1e-9
                if mk.any():
                    M[ih, il] = (e / Eg[mk].min() - 1) * 100
        im = ax.imshow(M, origin="lower", cmap=fs.diverging().reversed(),
                       vmin=-VMAX, vmax=VMAX, interpolation="nearest", aspect="equal")
        ax.plot([0, n - 1], [0, n - 1], "-", color=fs.MUTED, lw=0.4, zorder=4)
        tk = [0, n // 3, 2 * n // 3, n - 1]
        ax.set_xticks(tk); ax.set_yticks(tk)
        ax.set_xticklabels([GRID[i] for i in tk], fontsize=6)
        ax.set_yticklabels([GRID[i] for i in tk], fontsize=6)
        ax.set_title(f"{L} cold experts  ($\\varphi$ = {phi:.3f})", fontsize=7,
                     loc="left", pad=3)
        best = np.nanmin(M)
        bi, bj = np.unravel_index(np.nanargmin(M), M.shape)   # (f_hot idx, f_cold idx)
        ax.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False,
                                   edgecolor="#D01C1C", lw=0.9, zorder=6))
        ax.text(0.97, 0.06,
                f"best {best:+.2f}%\n$f_{{hot}}$={GRID[bi]}, $f_{{cold}}$={GRID[bj]}",
                transform=ax.transAxes, fontsize=6.5, ha="right", va="bottom",
                color=fs.INK, linespacing=1.3)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
    for ax in axes[1]:
        ax.set_xlabel("$f_{cold}$ (MHz)", fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("$f_{hot}$ (MHz)", fontsize=7)

    cax = fig.add_axes([0.885, 0.215, 0.022, 0.630])
    cb = fig.colorbar(im, cax=cax, extend="max")
    cb.set_label("energy vs the best single clock at the same latency (%)", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_visible(False)

    fig.text(0.005, 0.99,
             "A100-SXM4-40GB  |  Qwen3-30B-A3B MoE layer  |  free persistent kernel "
             "(no build cost charged)  |  every measured clock pair, all enumerated.",
             fontsize=7, color=fs.INK, va="top", ha="left")
    fig.text(0.005, 0.012,
             "The grey diagonal is $f_{cold}=f_{hot}$: not a split, and identical to "
             "the single-clock design, so it is white by construction. Red boxes mark "
             "each panel's best cell.\nWins are real but small and confined to the top "
             "edge, where the hot domain runs at or near the fastest clock; the best "
             "anywhere is $-$1.53%. Push the separation\nfurther -- down and to the "
             "left -- and the design loses, by up to 25%. All of this is before the "
             "+5.6% latency that a partitionable kernel was measured to cost.",
             fontsize=7, color=fs.INK, va="bottom", ha="left", linespacing=1.35)
    print(fs.save(fig, "fig22_sweep_heatmap"))


if __name__ == "__main__":
    main()
