#!/usr/bin/env python3
"""fig27 -- one MoE layer inside the event-driven simulator, twice.

What "event" means, on the smallest example that has all of them: a 2048-token decode
layer whose 64 least-routed experts are pinned to a 32-SM cold domain and the rest to a
76-SM hot domain.

  left   both domains at 1290 MHz. The cold partition finishes each GEMM early and its
         SMs wait at the barrier for the hot one -- at full voltage.
  right  the cold domain alone slowed to the clock at which it finishes WITH the hot
         partition (the min_energy_slo controller's choice for the same layer time).

Top: who is running, per domain. Bottom: board power; it changes only at events (the
ticks along the top of the power panel), so the simulator integrates it exactly.

The numbers in the titles are the simulator's, from sim/calib/a100_qwen.json. Two V/f
domains do not exist on the A100; the left panel is what the measurements calibrate, the
right panel is the prediction.
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs
from sim.calib import Calib
from sim.control import Fixed, MinEnergySLO
from sim.engine import Engine, Kernel, Shared
from sim.hardware import two_domain
from sim.workload import build

COLD, HOT, EPI = fs.COMM, fs.GEMM, fs.MUTED
CELLS, BM, N_COLD, COLD_SMS = 32, 64, 64, 32


def one_layer(cal, f, shared):
    b = build([dict(phase="decode", cells=CELLS, layers=[0], repeat=3)])
    e = Engine(two_domain(COLD_SMS), cal, Kernel("persistent", BM, N_COLD), b, Fixed(f),
               shared=shared, limiter=False).run()
    L = e.layers[-1]
    seg = [(a - L["t_start"], b_ - L["t_start"], p, run) for a, b_, p, _, run in e.trace
           if a >= L["t_start"] - 1e-9]
    return L, seg


def draw(ax_g, ax_p, L, seg, title, ymax):
    T = L["ms"]
    fin = {k: v - L["t_start"] for k, v in L["finish"].items()}
    lanes = [("cold", 2, COLD, "cold domain, 32 SMs"), ("hot", 1, HOT, "hot domain, 76 SMs")]
    t_act0 = max(fin[("cold", "g1")], fin[("hot", "g1")])
    t_act1 = fin[("epilogue", "act")]
    t_g2 = max(fin[("cold", "g2")], fin[("hot", "g2")])
    start = {"g1": {"cold": 0.0, "hot": 0.02}, "g2": {"cold": t_act1, "hot": t_act1 + 0.02}}
    end_stage = {"g1": t_act0, "g2": t_g2}
    for name, y, col, lab in lanes:
        for g in ("g1", "g2"):
            a, b = start[g][name], fin[(name, g)]
            ax_g.barh(y, b - a, left=a, height=0.62, color=col, edgecolor="white", lw=0.6, zorder=3)
            ax_g.text((a + b) / 2, y, "GEMM 1" if g == "g1" else "GEMM 2", ha="center",
                      va="center", color="white", fontsize=7, zorder=4)
            if end_stage[g] - b > 0.004 * T:            # waiting at the barrier
                ax_g.barh(y, end_stage[g] - b, left=b, height=0.62, facecolor="white",
                          edgecolor=col, hatch="////", lw=0.5, zorder=3)
        for a, b in ((t_act0, t_act1), (t_g2, T)):      # epilogue runs on every SM
            ax_g.barh(y, b - a, left=a, height=0.62, color=EPI, edgecolor="white", lw=0.6,
                      zorder=3)
    ax_g.set_yticks([2, 1]); ax_g.set_yticklabels(["cold\n32 SMs", "hot\n76 SMs"])
    ax_g.set_ylim(0.45, 2.55); ax_g.set_xlim(0, ymax[0])
    ax_g.tick_params(axis="x", labelbottom=False, length=0)
    for sp in ("top", "right", "bottom"):
        ax_g.spines[sp].set_visible(False)
    ax_g.set_title(title, loc="left", fontsize=8, pad=3)
    # power: a step function of the events
    xs, ys = [], []
    for a, b, p, _ in seg:
        xs += [a, b]; ys += [p, p]
    ax_p.fill_between(xs, ys, color=fs.BASE, lw=0, zorder=2)
    ax_p.plot(xs, ys, color=fs.INK, lw=0.8, zorder=3)
    ev = sorted({round(a, 6) for a, *_ in seg} | {round(T, 6)})
    for t in ev:
        ax_p.plot([t, t], [ymax[1] * 0.955, ymax[1]], color=fs.INK, lw=0.6, zorder=4)
    ax_p.set_xlim(0, ymax[0]); ax_p.set_ylim(0, ymax[1])
    ax_p.set_xlabel("time since the layer started (ms)")
    fs.despine(ax_p)
    return T


def main():
    cal, sh = Calib.load(), Shared()
    # the cold clock the SLO controller picks for the one-clock layer time
    La, sa = one_layer(cal, [1290, 1290], sh)
    b = build([dict(phase="decode", cells=CELLS, layers=[0], repeat=3)])
    ctl = MinEnergySLO(slo_ms=La["ms"] * 1.004)
    e = Engine(two_domain(COLD_SMS, actuation_ms=0.0), cal, Kernel("persistent", BM, N_COLD),
               b, ctl, shared=sh, limiter=False).run()
    fb = ctl.decisions[-1]["f"]
    Lb, sb = one_layer(cal, fb, sh)

    fs.use()
    W, H = fs.TEXT_W, 2.55
    fig = plt.figure(figsize=(W, H))
    xmax = max(La["ms"], Lb["ms"]) * 1.02
    pmax = 430.0
    lefts, wid = [0.50, 0.50 + 2.36 + 0.22], 2.36
    axs = []
    for i, x0 in enumerate(lefts):
        g = fig.add_axes([x0 / W, 1.52 / H, wid / W, 0.72 / H])
        p = fig.add_axes([x0 / W, 0.36 / H, wid / W, 1.06 / H])
        axs.append((g, p))
    ta = f"(a) one clock, {int(La['f_start'][0])} MHz:  {La['ms']:.2f} ms, {La['mj']:.0f} mJ"
    tb = f"(b) cold domain at {fb[0]} MHz:  {Lb['ms']:.2f} ms, {Lb['mj']:.0f} mJ"
    draw(*axs[0], La, sa, ta, (xmax, pmax))
    draw(*axs[1], Lb, sb, tb, (xmax, pmax))
    axs[0][1].set_ylabel("board power (W)")
    axs[1][0].set_yticklabels([]); axs[1][1].set_yticklabels([])
    axs[0][1].text(0.985, 0.90, "ticks: events", transform=axs[0][1].transAxes, ha="right",
                   va="top", fontsize=7, color=fs.INK)
    hs = [plt.Rectangle((0, 0), 1, 1, color=COLD), plt.Rectangle((0, 0), 1, 1, color=HOT),
          plt.Rectangle((0, 0), 1, 1, color=EPI),
          plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=fs.INK, hatch="////", lw=0.5)]
    fig.legend(hs, ["cold experts", "hot experts", "activation, top-k sum (all SMs)",
                    "waiting at the barrier"], loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 1.005), handlelength=1.2, columnspacing=1.2)
    out = fs.save(fig, "fig27_sim_layer_events")
    print(f"  one clock {La['f_start']}: {La['ms']:.4f} ms {La['mj']:.1f} mJ {La['w']:.1f} W")
    print(f"  cold slowed {tuple(fb)}: {Lb['ms']:.4f} ms {Lb['mj']:.1f} mJ {Lb['w']:.1f} W  "
          f"({(Lb['mj'] / La['mj'] - 1) * 100:+.1f}% energy, {(Lb['ms'] / La['ms'] - 1) * 100:+.2f}% time)")
    print("wrote", *out)


if __name__ == "__main__":
    main()
