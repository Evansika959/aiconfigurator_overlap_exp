#!/usr/bin/env python3
"""fig23b -- the dynamic model, simplified: one loop, few words.

fig23 is the full architecture (every component's equation, every calibration source,
the validation ladder). This is the version for a talk or an overview paragraph: the
loop and what goes in and comes out, nothing else.

  inputs   -> 1 demand -> 2 rate <-> 3 bandwidth -> 4 power -> 5 controller
                  ^          ^                          ^            |
                  |          +---------- clock ---------+------------+
                  +------------------ expert placement --------------+

Drawing helpers and the layout check are fig23's (make_sim_arch_fig.py): every text
must sit inside its box and no two texts may overlap, or the script refuses to save.
5.5 in wide (one NeurIPS text width).
"""
import os, sys
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs
import make_sim_arch_fig as base

base.W, base.H = 5.5, 3.05          # the helpers read the canvas size from here
W, H = base.W, base.H
I, BLUE, BLUE_FILL = fs.INK, base.BLUE, base.BLUE_FILL
ORANGE, ORANGE_FILL = base.ORANGE, base.ORANGE_FILL
DATA, CTRL, CAL = base.DATA, base.CTRL, base.CAL
mid = lambda a, b: (a + b) / 2


def main():
    F = base.Fig()

    # legend
    for k, (ls, col, lab) in enumerate(((DATA, I, "data"), (CTRL, ORANGE, "control"),
                                        (CAL, BLUE, "calibration"))):
        x = 2.95 + k * 0.86
        F.ax.plot([x, x + 0.26], [2.95, 2.95], color=col, ls=ls, lw=0.9)
        F.text(x + 0.32, 2.95, lab)

    # ---- inputs ------------------------------------------------------------------
    F.text(0.05, 2.78, "Inputs", size=base.T_HEAD, weight="bold")
    ins = [("Workload", ["batch sizes,", "expert routing"]),
           ("Hardware", ["V/f domains per GPC,", "400 W TDP"]),
           ("Targets", ["latency SLOs"])]
    tops = [2.62, 2.08, 1.54]
    BH = 0.44                                        # input / output box height
    for (t, lines), top in zip(ins, tops):
        e = (0.05, top - BH, 1.22, top)
        F.io(e, t, lines, BLUE_FILL, BLUE)
        F.arrow([(e[2], mid(e[1], e[3])), (1.40, mid(e[1], e[3]))])

    # ---- the loop -----------------------------------------------------------------
    cx0, cy0, cx1, cy1 = 1.40, 1.02, 4.16, 2.84
    F.box(cx0, cy0, cx1, cy1, fc="#FCFBF9", ec=fs.MUTED, ls=(0, (4, 2)), lw=0.7,
          r=0.06, z=1)
    F.text(cx0 + 0.09, cy1 - 0.11, "Simulator", size=base.T_HEAD, weight="bold")
    c0, c1, c2, c3, c4, c5 = 1.48, 2.30, 2.46, 3.14, 3.30, 4.08
    r1 = (2.14, 2.54)                                   # top row
    r2 = (1.52, 1.92)                                   # bottom row
    D = (c0, r1[0], c1, r1[1]); R = (c2, r1[0], c3, r1[1]); B = (c4, r1[0], c5, r1[1])
    C = (c0, r2[0], c1, r2[1]); P = (c2, r2[0], c3, r2[1])
    F.component(D, "1", "Demand", ["per partition"])
    F.component(R, "2", "Rate", ["time per clock"])
    F.component(B, "3", "Bandwidth", ["shared HBM"])
    F.component(P, "4", "Power", ["from activity"])
    F.component(C, "5", "Controller", ["clock, placement"], fc=ORANGE_FILL, ec=ORANGE)
    G = (c0, 1.10, c5, 1.38)
    F.box(*G, fc=base.NEUTRAL_FILL, ec=I)
    F.text(c0 + 0.08, mid(G[1], G[3]),
           r"Event engine: jump between events, energy $=\Sigma\, P\,\Delta t$", box=G)

    ym = mid(r1[0], r2[1])
    F.arrow([(D[2], mid(*r1)), (R[0], mid(*r1))])                    # 1 -> 2
    F.arrow([(R[2], mid(*r1)), (B[0], mid(*r1))], both=True)         # 2 <-> 3
    F.arrow([(mid(c2, c3), R[1]), (mid(c2, c3), P[3])])              # 2 -> 4
    F.arrow([(mid(c4, c5), B[1]), (mid(c4, c5), ym), (c3 + 0.05, ym),
             (c3 + 0.05, P[3])])                                     # 3 -> 4
    F.arrow([(P[0], mid(*r2) + 0.07), (C[2], mid(*r2) + 0.07)])      # 4 -> 5
    F.arrow([(C[2], mid(*r2) - 0.09), (P[0], mid(*r2) - 0.09)],      # 5 -> 4 clock
            color=ORANGE, ls=CTRL)
    F.arrow([(mid(c0, c1) - 0.18, C[3]), (mid(c0, c1) - 0.18, D[1])],  # 5 -> 1
            color=ORANGE, ls=CTRL)
    F.arrow([(C[2] - 0.08, C[3]), (R[0] + 0.08, R[1])],              # 5 -> 2 clock
            color=ORANGE, ls=CTRL)
    F.text(mid(c0, c1) - 0.13, ym, "placement", color=ORANGE)
    F.text(c2 + 0.02, ym, "clock", color=ORANGE, ha="left")

    # ---- outputs -----------------------------------------------------------------
    F.text(4.30, 2.78, "Outputs", size=base.T_HEAD, weight="bold")
    outs = [("Latency", ["TTFT, TPOT"]), ("Energy", ["per token"]),
            ("Traces", ["power, clock over time"])]
    for (t, lines), top in zip(outs, tops):
        e = (4.30, top - BH, 5.45, top)
        F.box(*e, fc="white", ec=I)
        F.text(e[0] + 0.08, e[3] - 0.105, t, size=base.T_TITLE, weight="bold", box=e)
        F.text(e[0] + 0.08, e[3] - 0.105 - base.LINE - 0.012, lines[0], box=e)
    F.arrow([(cx1, tops[1] - BH / 2), (4.30, tops[1] - BH / 2)])

    # ---- calibrate and validate ------------------------------------------------
    K = (0.05, 0.06, 2.70, 0.56)
    F.io(K, "Calibrate on a single-domain A100",
         ["measured clock sweeps, idle power,", "co-running partitions"], BLUE_FILL, BLUE)
    F.arrow([(2.05, K[3]), (2.05, cy0)], color=BLUE, ls=CAL)
    F.text(2.11, mid(K[3], cy0), "parameters", color=BLUE)
    V = (2.95, 0.06, 5.45, 0.56)
    F.box(*V, fc="white", ec=I)
    F.text(V[0] + 0.08, V[3] - 0.105, "Validate", size=base.T_TITLE, weight="bold", box=V)
    F.text(V[0] + 0.08, V[3] - 0.105 - base.LINE - 0.012,
           "reproduce the measured single domain,", box=V)
    F.text(V[0] + 0.08, V[3] - 0.105 - 2 * base.LINE - 0.024,
           "only then predict two domains", box=V)
    F.arrow([(4.86, tops[2] - BH), (4.86, V[3])])
    F.arrow([(K[2], mid(K[1], K[3])), (V[0], mid(K[1], K[3]))], color=BLUE, ls=CAL)

    n = base.check_layout(F.fig)
    os.makedirs("figs", exist_ok=True)
    for ext in ("png",):
        F.fig.savefig(f"figs/fig23b_sim_simple.{ext}", bbox_inches=None, pad_inches=0)
    plt.close(F.fig)
    print(f"layout ok ({n} items); wrote figs/fig23b_sim_simple.png at {W} x {H} in")


if __name__ == "__main__":
    main()
