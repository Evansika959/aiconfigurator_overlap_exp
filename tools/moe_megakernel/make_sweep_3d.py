#!/usr/bin/env python3
"""fig22 -- the two-domain search space against what is actually deployed.

PILLARS, and the viewing angle is the whole trick. Makespan is tallest at low f_cold and
low f_hot -- the near end of the diagonal -- so at the default azimuth that wall stands in
front and hides the valley behind it, which is the entire interesting region. Viewed from
elev 36 / azim 122 the tall wall falls to the back and every pillar is visible.

BASELINE: the unlocked governor. PyTorch launches the kernel and the GPU chooses its own
clock; measured 6 times it settles at 1322 MHz, 3.759 ms, 1503.6 mJ. No clock tuning is
assumed on the baseline's behalf.

Each surface point is one (f_hot, f_cold) pair at 45 MHz resolution.
  HEIGHT -- makespan relative to the governor. What the design costs.
  COLOUR -- energy relative to the governor. What it buys.

MAKESPAN is the wall time for the whole layer: both partitions run concurrently, so the
layer ends when the slower one ends. The SM split is chosen to make them end together,
so makespan is either one's runtime. 1.0 means "as fast as the deployed baseline".

Nothing is faster than the governor: it runs at 1322 MHz, above the fastest clock the
full-width kernel sustains, and any split adds makespan on top. So the surface never
descends below 1.0 and every saving is bought with latency.

STATIC POWER. Energy is D(f_cold).phi + D(f_hot).(1-phi) + P_static.T, with
P_static = s.PS(f_cold) + (1-s).PS(f_hot) where PS is the measured board idle at that
locked clock. That mix is exact rather than approximate: writing PS(f) = P_shared +
108.L(f), the weights sum to 1, so the shared uncore/HBM floor is counted once and only
the per-SM leakage is split. No linearity in PS(f) is assumed -- which matters, since
PS(f) is measurably non-linear (R^2 = 0.77 against a straight line).
"""
import csv, os, sys, importlib.util
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs

HERE = os.path.dirname(os.path.abspath(__file__))
STEP = 45
SHOW = [(0.045, 16), (0.208, 48), (0.596, 96)]


def main():
    spec = importlib.util.spec_from_file_location(
        "p", os.path.join(HERE, "predict_two_domain.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    cl, T, E, D, PS = mod.load_sweep()

    grid = np.array([c for c in range(int(cl.min()), int(cl.max()) + 1, STEP)
                     if c in set(cl.tolist())])
    Tg, Dg, Pg = [np.interp(grid, cl, v) for v in (T, D, PS)]
    n = len(grid)

    by = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(HERE, "data", "b1_energy_qwen.csv"))):
        by[int(r["clock"])].append(r)
    A = lambda c, k: float(np.mean([float(x[k]) for x in by[c]]))
    Tgov, Egov, Fgov = A(-1, "ms_per_layer"), A(-1, "energy_mj_per_layer"), A(-1, "clock_med")
    bestc = min(cl, key=lambda c: A(c, "energy_mj_per_layer"))
    solo_e = A(bestc, "energy_mj_per_layer") / Egov - 1
    solo_t = A(bestc, "ms_per_layer") / Tgov

    fs.use()
    cmap = fs.diverging().reversed()
    norm = colors.Normalize(vmin=-30, vmax=0)
    fig = plt.figure(figsize=(fs.TEXT_W, 3.4))

    for k, (phi, L) in enumerate(SHOW):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        xs, ys, hs, cs = [], [], [], []
        for ih in range(n):
            for il in range(ih + 1):
                t = Tg[ih] * (phi * grid[ih] / grid[il] + 1 - phi)
                sm = phi * grid[ih] / (phi * grid[ih] + (1 - phi) * grid[il])
                e = (Dg[il] * phi + Dg[ih] * (1 - phi)
                     + (sm * Pg[il] + (1 - sm) * Pg[ih]) * t)
                xs.append(il); ys.append(ih)
                hs.append(t / Tgov - 1); cs.append((e / Egov - 1) * 100)
        xs, ys = np.array(xs), np.array(ys)
        hs, cs = np.array(hs), np.array(cs)
        ax.bar3d(xs - .40, ys - .40, np.zeros(len(xs)), .80, .80, hs,
                 color=cmap(norm(cs)), shade=False, edgecolor="white", linewidth=0.10)
        bi = int(np.argmin(cs))
        ax.plot([xs[bi], xs[bi]], [ys[bi], ys[bi]], [hs[bi], hs[bi] + 0.22],
                color="#D01C1C", lw=1.1, zorder=20)
        ax.scatter([xs[bi]], [ys[bi]], [hs[bi] + 0.22], color="#D01C1C", s=9,
                   depthshade=False, zorder=20)
        best_e, best_t = cs[bi], hs[bi] + 1
        tk = [0, n // 2, n - 1]
        ax.set_xticks(tk); ax.set_yticks(tk)
        ax.set_xticklabels([grid[i] for i in tk], fontsize=5.5)
        ax.set_yticklabels([grid[i] for i in tk], fontsize=5.5)
        ax.set_zlim(0, 1.45)
        ax.set_zticks([0, 0.5, 1.0])
        # z labels only on the last panel: on the others they collide with the next
        # panel's axes
        ax.set_zticklabels(["1.0$\\times$", "1.5$\\times$", "2.0$\\times$"]
                           if k == len(SHOW) - 1 else [""] * 3, fontsize=5.5)
        ax.set_xlabel("$f_{cold}$", fontsize=6.5, labelpad=-6)
        ax.set_ylabel("$f_{hot}$", fontsize=6.5, labelpad=-6)
        if k == len(SHOW) - 1:
            ax.set_zlabel("makespan vs baseline", fontsize=6.5, labelpad=-1)
        ax.tick_params(pad=-2)
        ax.view_init(elev=36, azim=122)
        ax.set_box_aspect((1, 1, 0.85))
        ax.xaxis.pane.set_alpha(0); ax.yaxis.pane.set_alpha(0)
        ax.zaxis.pane.set_alpha(0)
        ax.grid(False)
        ax.set_title(f"{L} cold experts  ($\\varphi$={phi:.3f})\n"
                     f"best {best_e:+.1f}% energy at {best_t:.2f}$\\times$ latency",
                     fontsize=6.8, pad=-12)

    cax = fig.add_axes([0.33, 0.255, 0.36, 0.020])
    cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                      orientation="horizontal")
    cb.set_label("energy vs the baseline (%)", fontsize=7, labelpad=1)
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_visible(False)

    fig.text(0.005, 0.99,
             "A100-SXM4-40GB  |  Qwen3-30B-A3B MoE layer  |  free persistent kernel (no "
             f"build cost charged)  |  45 MHz grid, every pair enumerated.\nBaseline: "
             f"the kernel as deployed -- clocks unlocked, PyTorch launches it, the GPU "
             f"picks its own frequency. Measured 6x: it settles at {Fgov:.0f} MHz, "
             f"{Tgov:.3f} ms, {Egov:.0f} mJ.",
             fontsize=7, color=fs.INK, va="top", ha="left", linespacing=1.35)
    fig.text(0.005, 0.012,
             f"Height is what the design costs, colour what it buys. Every pillar "
             f"stands above 1.0: nothing anywhere is faster than the baseline."
             f"\nThe best pillar reaches {best_e:+.0f}% energy at {best_t:.2f}$\\times$ "
             f"latency -- but ONE fixed clock at {bestc} MHz, no split and no new kernel, "
             f"is {solo_e:+.1%} at {solo_t:.2f}$\\times$.\nThe split is not what saves "
             "the energy; lowering the clock is. The red pin marks each panel's best "
             "pillar.",
             fontsize=7, color=fs.INK, va="bottom", ha="left", linespacing=1.35)
    fig.subplots_adjust(left=0.015, right=0.935, top=0.835, bottom=0.360,
                        wspace=0.01)
    print(fs.save(fig, "fig22_sweep_heatmap"))


if __name__ == "__main__":
    main()
