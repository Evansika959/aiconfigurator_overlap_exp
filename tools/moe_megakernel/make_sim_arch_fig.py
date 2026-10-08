#!/usr/bin/env python3
"""fig23 -- architecture of the dynamic (runtime) model: a discrete-event fluid
simulator for spatial V/f domains, with its inputs and outputs by level.

WHY A SIMULATOR AND NOT A LOOKUP TABLE. The design it has to predict -- two SM
partitions at different voltages -- cannot be measured on an A100, which has one SM
rail. A table can only interpolate between points that were measured; this one would be
queried where nothing was. So the model is mechanistic: a few physical parameters,
fitted on the single-domain hardware (bottom band), then run where the hardware cannot
go. The validation ladder is the rule for when that extrapolation is trusted.

READING THE FIGURE
  left    inputs, by level of abstraction: workload, model and kernel, hardware, policy.
          Circled letters name the components that consume each input.
  centre  the runtime. (A) turns a batch into per-partition demand; (B) and (C) are
          solved together, because a partition's rate depends on the bandwidth the other
          one leaves it; (D) computes power from the resulting activity; (E) closes the
          loop by setting each domain's clock. The event engine advances from event to
          event, and between events every rate is constant, so power is piecewise
          constant and energy is a sum.
  right   outputs, by time scale. The engine emits the finest level; every coarser
          level is an aggregate of the one below it.
  bottom  what is measured, and what the outputs are checked against.

LINE STYLE CARRIES THE CATEGORY as well as colour, so the figure survives greyscale:
solid = data flow, dashed = control set-points (the only orange), dotted = calibration.

SIZE. 7.0 in, the full text width of a two-column ISCA/IEEE page (figure*). Drawn in a
coordinate system where one unit is one inch and saved WITHOUT the tight bbox, so the
figure is exactly 7.0 in and 7 pt type prints at 7 pt with \\includegraphics{}.

The script checks its own layout before saving: every text must sit inside its box and
no two texts may overlap. It refuses to save otherwise, rather than relying on a look.
"""
import os, sys
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs

W, H = 7.0, 4.30
BLUE, BLUE_FILL = fs.COMM, "#EEF1FB"
ORANGE, ORANGE_FILL = fs.GEMM, "#FBEFE3"
NEUTRAL_FILL = "#F4F2EE"
DATA, CTRL, CAL = "-", (0, (3.2, 1.8)), (0, (1.0, 1.4))
T_BODY, T_TITLE, T_HEAD = 7.0, 7.5, 8.0
LINE = 0.118                       # baseline-to-baseline at 7 pt, inches

texts = []                         # (artist, container extent or None)
marks = []                         # (label, extent, container) for the circled tags
TAG_R = 0.068


class Fig:
    def __init__(self):
        fs.use()
        self.fig = plt.figure(figsize=(W, H))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, W); self.ax.set_ylim(0, H); self.ax.axis("off")

    def text(self, x, y, s, size=T_BODY, weight="normal", ha="left", va="center",
             color=fs.INK, box=None, rotation=0):
        t = self.ax.text(x, y, s, fontsize=size, fontweight=weight, ha=ha, va=va,
                         color=color, rotation=rotation, zorder=5)
        texts.append((t, box))
        return t

    def box(self, x0, y0, x1, y1, fc, ec, ls="-", lw=0.6, r=0.045, z=2):
        self.ax.add_patch(FancyBboxPatch(
            (x0, y0), x1 - x0, y1 - y0, boxstyle=f"round,pad=0,rounding_size={r}",
            fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))
        return (x0, y0, x1, y1)

    def tag(self, x, y, letter, ec, box=None):
        # registered with the layout check: a tag is ink too, and the first draft of
        # this figure had tags sitting on top of box titles that the check never saw
        marks.append((f"tag {letter}", (x - TAG_R, y - TAG_R, x + TAG_R, y + TAG_R), box))
        self.ax.add_patch(Circle((x, y), TAG_R, fc="white", ec=ec, lw=0.7, zorder=6))
        self.ax.text(x, y - 0.004, letter, fontsize=T_BODY, fontweight="bold",
                     ha="center", va="center", color=fs.INK, zorder=7)

    def arrow(self, pts, color=fs.INK, ls=DATA, lw=0.8, both=False):
        """Polyline; the head sits on the last segment."""
        for a, b in zip(pts[:-2], pts[1:-1]):
            self.ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, ls=ls,
                         solid_capstyle="butt", zorder=4)
        self.ax.add_patch(FancyArrowPatch(
            pts[-2], pts[-1], arrowstyle="<|-|>" if both else "-|>",
            mutation_scale=6.5, color=color, lw=lw, ls=ls, shrinkA=0, shrinkB=0,
            zorder=4))

    def component(self, ext, letter, title, lines, fc=NEUTRAL_FILL, ec=fs.INK,
                  lead=LINE):
        x0, y0, x1, y1 = self.box(*ext, fc=fc, ec=ec)
        top = y1 - 0.105
        self.tag(x0 + 0.11, top, letter, ec, box=ext)
        self.text(x0 + 0.215, top, title, size=T_TITLE, weight="bold", box=ext)
        for i, s in enumerate(lines):
            self.text(x0 + 0.08, top - (i + 1) * lead - 0.012, s, box=ext)

    def io(self, ext, title, lines, fc, ec, tags=(), tag_ec=None):
        x0, y0, x1, y1 = self.box(*ext, fc=fc, ec=ec)
        top = y1 - 0.105
        self.text(x0 + 0.08, top, title, size=T_TITLE, weight="bold", box=ext)
        for k, (letter, e) in enumerate(reversed(tags)):
            self.tag(x1 - 0.11 - k * 0.16, top, letter, e, box=ext)
        for i, s in enumerate(lines):
            self.text(x0 + 0.08, top - (i + 1) * LINE - 0.012, s, box=ext)


def check_layout(fig):
    """Every text inside its box (2 pt inset), and no two texts overlapping."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    dpi, inset = fig.dpi, 2 / 72
    items = []
    for t, b in texts:
        bb = t.get_window_extent(r)
        items.append((repr(t.get_text()), (bb.x0 / dpi, bb.y0 / dpi, bb.x1 / dpi,
                                           bb.y1 / dpi), b))
    items += marks
    ext, bad = [], []
    for name, e, b in items:
        ext.append(e)
        if b is not None and not (e[0] >= b[0] + inset - 1e-3 and e[2] <= b[2] - inset + 1e-3
                                  and e[1] >= b[1] + inset - 1e-3
                                  and e[3] <= b[3] - inset + 1e-3):
            over = max(b[0] + inset - e[0], e[2] - b[2] + inset,
                       b[1] + inset - e[1], e[3] - b[3] + inset)
            bad.append(f"overflows its box by {over:.3f} in: {name}")
        if not (0 <= e[0] and e[2] <= W and 0 <= e[1] and e[3] <= H):
            bad.append(f"leaves the canvas: {name}")
    for i in range(len(ext)):
        for j in range(i + 1, len(ext)):
            a, b = ext[i], ext[j]
            if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
                bad.append(f"overlap: {items[i][0]} and {items[j][0]}")
    if bad:
        raise SystemExit("layout check failed:\n  " + "\n  ".join(bad))
    return len(ext)


def main():
    F = Fig()
    I = fs.INK

    # ---- vertical plan (inches from the bottom), derived once, used everywhere ----
    kal = (0.06, 0.74)                 # calibration band
    core = (0.94, 3.94)                # simulator container
    head_y, legend_y = 4.02, 4.20
    col_top = 3.86                     # top of the input and output columns

    # ---- legend --------------------------------------------------------------
    for k, (ls, col, lab) in enumerate(((DATA, I, "data flow"),
                                        (CTRL, ORANGE, "control set-point"),
                                        (CAL, BLUE, "calibration"))):
        x = 3.55 + k * 1.16
        F.ax.plot([x, x + 0.30], [legend_y, legend_y], color=col, ls=ls, lw=0.9)
        F.text(x + 0.37, legend_y, lab)

    # ---- inputs, by level of abstraction ----------------------------------------
    F.text(0.05, head_y, "Inputs, by abstraction level", size=T_HEAD, weight="bold")
    ix0, ix1, ih = 0.05, 1.72, 0.56
    inputs = [
        ("Workload", ["request trace: arrivals, lengths",
                      "routing trace: tokens per", "expert, per layer, per step"],
         [("A", I)]),
        ("Model and kernel", ["experts, top-k, hidden sizes", "tile shape",
                              "expert-to-domain policy"],
         [("A", I), ("B", I)]),
        ("Hardware", ["SMs, HBM bandwidth, TDP",
                      "V/f domains: count, size", "clock actuation: 38 ms"],
         [("B", I), ("C", I), ("D", I)]),
        ("Policy and targets", ["controller policy", "TTFT and TPOT SLOs"],
         [("E", ORANGE)]),
    ]
    igap = (col_top - core[0] - len(inputs) * ih) / (len(inputs) - 1)
    for k, (title, lines, tags) in enumerate(inputs):
        top = col_top - k * (ih + igap)
        F.io((ix0, top - ih, ix1, top), title, lines, BLUE_FILL, BLUE, tags)
        F.arrow([(ix1, top - ih / 2), (1.95, top - ih / 2)])

    # ---- the runtime ---------------------------------------------------------------
    cx0, cx1 = 1.95, 4.97
    F.box(cx0, core[0], cx1, core[1], fc="#FCFBF9", ec=fs.MUTED, ls=(0, (4, 2)),
          lw=0.7, r=0.06, z=1)
    F.text(cx0 + 0.10, core[1] - 0.12, "Discrete-event fluid simulator (runtime)",
           size=T_HEAD, weight="bold")
    # columns; the right gutter carries the controller's tile-placement line
    L0, L1, R0, R1 = 2.08, 3.24, 3.60, 4.74
    gx = 4.80
    G = (L0, core[0] + 0.06, R1, core[0] + 0.50)
    row3 = (G[3] + 0.08, G[3] + 0.08 + 0.64)
    row2 = (row3[1] + 0.26, row3[1] + 0.26 + 0.54)
    A = (L0, row2[1] + 0.26, R1, row2[1] + 0.26 + 0.48)
    assert A[3] <= core[1] - 0.22, A
    C = (L0, row2[0], L1, row2[1]); B = (R0, row2[0], R1, row2[1])
    D = (L0, row3[0], L1, row3[1]); E = (R0, row3[0], R1, row3[1])

    F.component(A, "A", "Workload decomposer",
                [r"batch $\rightarrow$ per-partition demand vector: FLOPs,",
                 "on-chip bytes, L2 bytes, HBM bytes, tile count"])
    F.component(C, "C", "Bandwidth arbiter",
                ["splits HBM bandwidth $b_p$", "between partitions"])
    F.component(B, "B", "Rate model",
                [r"$T_p = C_p/f_d + M_p(b_p)$", "per domain; SM-aware"])
    F.component(D, "D", "Power model",
                [r"$P = P_{\rm const} + P_{\rm HBM}(b)$",
                 r"$+\Sigma_d n_d\,[\,P_{\rm leak}(V_d)$",
                 r"$+\,V_d^{\,2} f_d \Sigma_k e_k a_k\,]$"], lead=0.142)
    F.component(E, "E", "Controller",
                [r"slow: $f_d$ per window,", "TDP on window average",
                 "fast: tile placement"],
                fc=ORANGE_FILL, ec=ORANGE)
    F.box(*G, fc=NEUTRAL_FILL, ec=I)
    F.text(L0 + 0.08, G[3] - 0.095, "Event engine", size=T_TITLE, weight="bold", box=G)
    F.text(L0 + 0.08, G[3] - 0.095 - LINE - 0.012,
           "events: batch arrival, partition finish, controller tick", box=G)
    F.text(L0 + 0.08, G[3] - 0.095 - 2 * LINE - 0.024,
           r"rates are fixed between events: $E=\Sigma_i P_i\,\Delta t_i$", box=G)

    mid = lambda a, b: (a + b) / 2
    xb = R0 + 0.62                                     # A -> B
    F.arrow([(xb, A[1]), (xb, B[3])])
    F.text(xb + 0.06, mid(A[1], B[3]), "demand")
    yc = mid(*row2) - 0.02                             # B <-> C, a fixed point
    F.arrow([(C[2], yc), (B[0], yc)], both=True)
    F.text(mid(C[2], B[0]), yc + 0.05, "$b_p$", ha="center", va="bottom")
    xc = L0 + 0.60                                     # C -> D
    F.arrow([(xc, C[1]), (xc, D[3])])
    F.text(xc + 0.06, mid(C[1], D[3]), "$b_p$")
    ym = mid(C[1], D[3])                               # B -> D, elbow
    F.arrow([(R0 + 0.28, B[1]), (R0 + 0.28, ym), (L1 - 0.18, ym), (L1 - 0.18, D[3])])
    F.text(mid(L1, R0 + 0.28) + 0.04, ym + 0.025, "activity $a_k$", ha="center",
           va="bottom")
    yp, yv = row3[1] - 0.20, row3[0] + 0.20            # D -> E, E -> D
    F.arrow([(D[2], yp), (E[0], yp)])
    F.text(mid(D[2], E[0]), yp + 0.04, "$P(t)$", ha="center", va="bottom")
    F.arrow([(E[0], yv), (D[2], yv)], color=ORANGE, ls=CTRL)
    F.text(mid(D[2], E[0]), yv - 0.045, "$V_d$", ha="center", va="top")
    xf = R1 - 0.30                                     # E -> B
    F.arrow([(xf, E[3]), (xf, B[1])], color=ORANGE, ls=CTRL)
    F.text(xf + 0.06, ym, "$f_d$")
    yA = mid(A[1], A[3])                               # E -> A, the right gutter
    F.arrow([(E[2], mid(*row3)), (gx, mid(*row3)), (gx, yA), (A[2], yA)],
            color=ORANGE, ls=CTRL)
    F.text(gx + 0.08, mid(mid(*row3), yA), "tile placement", rotation=90,
           ha="center", va="center", color=ORANGE,
           box=(gx, core[0], cx1, core[1]))

    # ---- outputs, by time scale -------------------------------------------------
    F.text(5.20, head_y, "Outputs, by time scale", size=T_HEAD, weight="bold")
    ox0, ox1, oh = 5.20, 6.95, 0.46
    outs = [   # bottom-up: finest first
        ("Event", r"$\mu$s$-$ms", ["per-partition rates, $b_p$, $P(t)$"]),
        ("Layer", "ms", ["makespan; energy split into static,",
                         "dynamic, HBM, per domain"]),
        ("Control window", "10$-$100 ms", ["$f_d(t)$ trajectories; window-",
                                          "average power against TDP"]),
        ("Request", "s", ["TTFT and TPOT at p50 and p99;",
                          "tokens/s, J/token vs baseline"]),
    ]
    ogap = (col_top - core[0] - len(outs) * oh) / (len(outs) - 1)
    bots = [core[0] + k * (oh + ogap) for k in range(len(outs))]
    for k, ((title, scale, lines), b) in enumerate(zip(outs, bots)):
        ext = (ox0, b, ox1, b + oh)
        F.box(*ext, fc="white", ec=I)
        top = b + oh - 0.105
        F.text(ox0 + 0.08, top, title, size=T_TITLE, weight="bold", box=ext)
        F.text(ox1 - 0.08, top, scale, ha="right", color=fs.MUTED, box=ext)
        for i, s in enumerate(lines):
            F.text(ox0 + 0.08, top - (i + 1) * LINE - 0.012, s, box=ext)
        if k < len(outs) - 1:
            F.arrow([(6.62, b + oh), (6.62, bots[k + 1])])
    F.text(6.56, mid(bots[0] + oh, bots[1]), "aggregate", ha="right")
    gy = mid(G[1], G[3])                               # the engine emits the finest
    assert bots[0] < gy < bots[0] + oh
    F.arrow([(G[2], gy), (cx1, gy), (ox0, gy)])

    # ---- offline calibration and validation ------------------------------------
    kx0, kx1 = 0.05, 4.97
    F.box(kx0, kal[0], kx1, kal[1], fc="white", ec=BLUE, ls=CAL, lw=0.9, r=0.06, z=1)
    F.text(kx0 + 0.10, kal[1] - 0.11,
           "Offline calibration on a single-domain A100 (measured)",
           size=T_HEAD, weight="bold")
    cal = [("Clock sweeps", [r"$C_p$, $M_p$, $V(f)$, $e_k$"], [("B", I), ("D", I)]),
           ("Idle vs clock", [r"$P_{\rm leak}(V)$"], [("D", I)]),
           ("Co-run", ["arbitration rule"], [("C", I)]),
           ("Governor runs", ["controller constants"], [("E", ORANGE)])]
    bw_, gap = 1.105, 0.115
    for k, (title, lines, tags) in enumerate(cal):
        x0 = 0.13 + k * (bw_ + gap)
        F.io((x0, kal[0] + 0.06, x0 + bw_, kal[0] + 0.44), title, lines, BLUE_FILL,
             BLUE, tags)
    F.arrow([(3.40, kal[1]), (3.40, core[0])], color=BLUE, ls=CAL)
    F.text(3.47, mid(kal[1], core[0]), "fitted parameters")

    V = (ox0, kal[0], ox1, kal[1])
    F.box(*V, fc="white", ec=I)
    vt = V[3] - 0.095
    F.text(ox0 + 0.08, vt, "Validation ladder", size=T_TITLE, weight="bold", box=V)
    ladder = ["1  held-out locked clocks", "2  governor: 1322 MHz, 400 W",
              "3  held-out co-run splits", "4  only then: two domains"]
    for i, s in enumerate(ladder):
        F.text(ox0 + 0.08, vt - (i + 1) * LINE - 0.012, s, box=V)
    F.arrow([(5.50, bots[0]), (5.50, V[3])])
    F.text(5.56, mid(bots[0], V[3]), "compare")
    F.arrow([(kx1, mid(kal[0] + 0.06, kal[0] + 0.44)),
             (ox0, mid(kal[0] + 0.06, kal[0] + 0.44))], color=BLUE, ls=CAL)

    n = check_layout(F.fig)
    os.makedirs("figs", exist_ok=True)
    for ext in ("png",):
        F.fig.savefig(f"figs/fig23_sim_architecture.{ext}", bbox_inches=None,
                      pad_inches=0)
    plt.close(F.fig)
    print(f"layout ok ({n} items); wrote figs/fig23_sim_architecture.png "
          f"at {W} x {H} in")


if __name__ == "__main__":
    main()
