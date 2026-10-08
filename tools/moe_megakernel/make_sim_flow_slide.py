#!/usr/bin/env python3
"""fig23c -- the simulator's event loop, drawn for a SLIDE, not a paper.

Same block vocabulary as the group's "Dynamic GPU DVFS Simulator" slide -- blue inputs,
grey modules, the controller in orange, white outputs, a dashed validation band -- with
the one thing that slide does not show: the ORDER things happen in.

One turn of the loop, numbered on the figure:
  1  Event queue      take the next event: a batch starts, a partition finishes, a clock
                      set-point lands, or the controller ticks
  2  Advance time     energy += P x dt; remaining work shrinks; the limiter's average moves
  3  handle it        Dispatcher places experts in domains (batch start); Controller
                      issues new set-points (tick, partition done); Frequency Assignment
                      applies f and V once the actuation delay has passed
  4  Bandwidth        partition rates and HBM shares, solved together
  5  Power Estimation from activity -- and that fixes when the next event will happen

SLIDE CONVENTIONS, deliberately not figstyle's: 13.33 x 7.5 in (PowerPoint 16:9, so it
fills a slide at 100%), sans-serif, 11-30 pt. Do not put this one in the paper: its paper
version is fig23d (make_sim_flow_fig.py), the same blocks and connections at 5.5 in.

The script checks that every label fits inside its box before it saves.
"""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Arc, Polygon

W, H = 13.33, 7.5
INK, GREY_T, LINE = "#111111", "#555555", "#555555"
BLUE, GREY, ORANGE, YELLOW = "#D6E0F0", "#EEEEEE", "#F7E6D2", "#FBF4D6"
E_BLUE, E_GREY, E_ORANGE = "#6F7F9C", "#666666", "#A08468"
RED = "#D0392B"
labels = []                                   # (text artist, box extent)


def main():
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

    def box(x0, y0, x1, y1, fc, ec, ls="-", lw=1.2, r=0.10, z=2):
        ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                    boxstyle=f"round,pad=0,rounding_size={r}", fc=fc,
                                    ec=ec, lw=lw, ls=ls, zorder=z))
        return (x0, y0, x1, y1)

    def badge(x, y, num):
        ax.add_patch(Circle((x, y), 0.17, fc="white", ec=INK, lw=1.2, zorder=8))
        ax.text(x, y - 0.01, str(num), fontsize=12, fontweight="bold", ha="center",
                va="center", color=INK, zorder=9)

    def module(ext, title, sub=None, fc=GREY, ec=E_GREY, num=None, size=17, slot=False):
        """slot=True reserves a two-line title slot at the top and a two-line sub-label
        slot at the bottom, so a row of boxes lines up even when one title wraps.
        The step badge sits INSIDE the box: on a corner it collided with every line
        that passed the box."""
        x0, y0, x1, y1 = box(*ext, fc, ec)
        cx = ((x0 + 0.46) + (x1 - 0.06)) / 2 if num else (x0 + x1) / 2
        if slot:
            t = ax.text(cx, y1 - 0.34, title, fontsize=size, ha="center", va="center",
                        color=INK, zorder=5, linespacing=1.05)
        else:
            cy = y0 + 0.26 + (y1 - y0 - 0.26) / 2 if sub else (y0 + y1) / 2
            t = ax.text(cx, cy, title, fontsize=size, ha="center", va="center",
                        color=INK, zorder=5, linespacing=1.05)
        labels.append((t, ext))
        if sub:
            s_ = ax.text(cx, y0 + (0.27 if slot else 0.07), sub, fontsize=SUB,
                         ha="center", va="center" if slot else "bottom", color=SUBC,
                         zorder=5, linespacing=1.1)
            labels.append((s_, ext))
        if num:
            badge(x0 + 0.25, (y0 + y1) / 2, num)
        return ext

    def arrow(pts, color=LINE, lw=1.5, head=True, z=4, both=False, ls="-", ms=16):
        segs = list(zip(pts[:-1], pts[1:]))
        for a, b in (segs[:-1] if head else segs):
            ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, ls=ls,
                    solid_capstyle="butt", zorder=z)
        if head:
            ax.add_patch(FancyArrowPatch(pts[-2], pts[-1],
                                         arrowstyle="<|-|>" if both else "-|>",
                                         mutation_scale=ms, color=color, lw=lw, ls=ls,
                                         shrinkA=0, shrinkB=0, zorder=z))

    def diamond(x, y, w, h, z=6):
        ax.add_patch(Polygon([(x, y + h), (x + w, y), (x, y - h), (x - w, y)],
                             closed=True, fc="white", ec=LINE, lw=1.3, zorder=z))

    mid = lambda a, b: (a + b) / 2
    SUB, SUBC, HEAD_Y = 13.5, "#3A3A3A", 6.17

    # ---- title ----------------------------------------------------------------------
    ax.text(0.45, 6.98, "Dynamic GPU DVFS Simulator: Event Loop", fontsize=30,
            ha="left", va="center", color=INK)
    ax.plot([0.25, W - 0.25], [6.55, 6.55], color=RED, lw=2.2)

    # three rows, equally spaced, shared by inputs, modules and outputs
    rA, rB, rC = (5.12, 5.78), (3.08, 4.22), (1.52, 2.18)
    cx0, cy0, cx1, cy1 = 3.38, 0.90, 10.05, 6.42
    ax.text(1.75, HEAD_Y, "Inputs", fontsize=20, fontweight="bold", ha="center",
            va="center", color=GREY_T, zorder=5)
    ax.text(cx0 + 0.28, HEAD_Y, "One turn", fontsize=20, fontweight="bold",
            ha="left", va="center", color=GREY_T, zorder=5)

    # ---- the loop ---------------------------------------------------------------------
    box(cx0, cy0, cx1, cy1, "white", LINE, ls=(0, (5, 3)), lw=1.3, r=0.25, z=1)
    XL, XR, XM = 4.20, 9.60, 7.45         # one grid: left edge, right edge, right column
    Q = module((XL, rA[0], 6.22, rA[1]), "Event queue", "next event", num=1, size=16)
    A = module((XM, rA[0], XR, rA[1]), "Advance time", "energy += P·Δt", num=2, size=16)
    Pwr = module((XL, rC[0], 6.85, rC[1]), "Power Estimation", "P for next interval",
                 num=5, size=16)
    Bnd = module((XM, rC[0], XR, rC[1]), "Bandwidth", "HBM → rates", num=4, size=16)

    # the loop's only exit: a real decision, not a line off the top of a box
    dx, dw = 6.80, 0.46
    diamond(dx, mid(*rA), dw, 0.31)
    ax.text(dx, mid(*rA) - 0.01, "empty?", fontsize=12.5, ha="center", va="center",
            color=INK, zorder=7)
    arrow([(Q[2], mid(*rA)), (dx - dw, mid(*rA))], ms=12)            # 1 -> ?
    arrow([(dx + dw, mid(*rA)), (A[0], mid(*rA))], ms=12)            # no -> 2
    ax.text(dx + dw + 0.02, mid(*rA) + 0.19, "no", fontsize=SUB, ha="center",
            va="center", color=SUBC, zorder=5)

    # step 3: EXACTLY ONE of three handlers. An arrow that meets the group above or below
    # any one handler is read as going through that handler -- two reviewers, two
    # different handlers. So the distribution is drawn: a bar fans out to all three with
    # "or" between them, and a second bar collects them before the single way out.
    G = box(XL, rB[0] - 0.28, XR, rB[1] + 0.60, "#FAFAFA", E_GREY, lw=1.0, r=0.14, z=1.5)
    yt = rB[1] + 0.40
    badge(G[0] + 0.27, yt, 3)
    ax.text(G[0] + 0.55, yt, "Handle the event: exactly one handler runs", fontsize=14,
            ha="left", va="center", color=INK, zorder=5)
    xc = mid(G[0], G[2])
    bw_ = 1.60                           # handlers left-packed: the spine runs to their right
    xs = [XL + 0.16 + bw_ / 2 + k * (bw_ + 0.06) for k in range(3)]
    hb = lambda x: (x - bw_ / 2, rB[0], x + bw_ / 2, rB[1])
    Ctl = module(hb(xs[0]), "Controller", "on: tick or\npartition end", fc=ORANGE,
                 ec=E_ORANGE, size=16, slot=True)
    Frq = module(hb(xs[1]), "Frequency\nAssignment", "on: set-point\ntakes effect",
                 size=16, slot=True)
    Dsp = module(hb(xs[2]), "Dispatcher", "on: batch\narrival", size=16, slot=True)
    # one vertical spine, to the RIGHT of every handler: on a handler's centreline it
    # reads as an arrow straight into that handler, and behind one it reads as a hidden
    # bypass. It meets each bar at its end.
    xa = XR - 0.16
    ybar, ycol = rB[1] + 0.16, rB[0] - 0.14
    arrow([(xa, A[1]), (xa, ybar), (xs[0], ybar)], head=False)       # 2 -> 3, one line

    for x in xs:
        arrow([(x, ybar), (x, rB[1])], lw=1.2, ms=11)
        arrow([(x, rB[0]), (x, ycol)], lw=1.2, head=False)
    for x in (mid(xs[0], xs[1]), mid(xs[1], xs[2])):
        ax.text(x, ybar + 0.02, "or", fontsize=12, ha="center", va="center",
                color=SUBC, zorder=6, bbox=dict(fc="#FAFAFA", ec="none", pad=1.5))
    arrow([(xs[0], ycol), (xa, ycol), (xa, Bnd[3])])                 # 3 -> 4, one line
    arrow([(Bnd[0], mid(*rC)), (Pwr[2], mid(*rC))])                  # 4 -> 5
    # 5 -> 1: out of Power Estimation's left side, up the gutter, into the queue
    gx, xl, yhi = 3.86, 4.52, mid(G[3], Q[1])
    arrow([(Pwr[0], mid(*rC)), (gx, mid(*rC)), (gx, yhi), (xl, yhi), (xl, Q[1])])
    ax.text(gx, mid(Pwr[3], G[1]), "repeat", fontsize=SUB, ha="center", va="center",
            color=SUBC, zorder=6, bbox=dict(fc="white", ec="none", pad=2.0))
    # handlers schedule future events (a new set-point, the next tick)
    xn = 5.45
    arrow([(xn, G[3]), (xn, Q[1])], ls=(0, (3, 2)), lw=1.2, ms=11)
    ax.text(xn + 0.10, mid(G[3], Q[1]), "handlers add new events", fontsize=SUB, ha="left",
            va="center", color=SUBC, zorder=5)

    # hardware parameters feed every step: a band, not an arrow into one box
    HB = box(XL, 1.02, XR, 1.34, "#F3F6FB", E_BLUE, lw=1.0, r=0.08, z=1.5)
    t = ax.text(mid(HB[0], HB[2]), mid(HB[1], HB[3]),
                "hardware parameters: used by every step", fontsize=SUB, ha="center",
                va="center", color=SUBC, zorder=5)
    labels.append((t, HB))

    # ---- inputs -----------------------------------------------------------------------
    ins = [("Workload", "Batch arrivals over time,\nexpert routing", rA),
           ("Targets", "Min energy under SLO, or\nmin latency under TDP", rB),
           ("Hardware", "V/f domains, #SM,\nTDP, limiter", rC)]
    for t, cap, row in ins:
        y0, ym = mid(*row) - 0.40, mid(*row)
        module((0.60, y0, 2.90, y0 + 0.80), t, fc=BLUE, ec=E_BLUE, size=18)
        ax.text(0.66, y0 - 0.10, cap, fontsize=SUB, ha="left", va="top", color=SUBC,
                linespacing=1.2)
        if row is rA:
            arrow([(2.90, ym), (Q[0], ym)], z=5)
        elif row is rB:              # a hop over the return line: a crossing, not a joint
            arrow([(2.90, ym), (gx - 0.14, ym)], head=False, z=5)
            ax.add_patch(Arc((gx, ym), 0.28, 0.28, theta1=0, theta2=180, color=LINE,
                             lw=1.5, zorder=5))
            arrow([(gx + 0.14, ym), (Ctl[0], ym)], z=5)
        else:                        # turns down OUTSIDE the frame, so it cannot be read
            yb = mid(HB[1], HB[3])   # as continuing into Power Estimation's return line
            arrow([(2.90, ym), (3.14, ym), (3.14, yb), (HB[0], yb)], z=5)

    # ---- outputs: all three are reported when the queue runs dry -------------------
    # The exit lands beside the "Outputs" heading, inside the group -- at any one output's
    # height it was read as feeding only that output.
    OG = box(10.70, rC[0] - 0.24, 13.10, cy1, "#FAFAFA", E_GREY, lw=1.0, r=0.14, z=1.5)
    ax.text(mid(OG[0], OG[2]) + 0.10, HEAD_Y, "Outputs", fontsize=20, fontweight="bold",
            ha="center", va="center", color=GREY_T, zorder=5)
    for name, row in (("E2E Latency", rA), ("Energy", rB), ("Power Traces", rC)):
        y0 = mid(*row) - 0.40
        module((10.88, y0, 12.92, y0 + 0.80), name, fc="white", ec=E_GREY, size=17)
    arrow([(dx, mid(*rA) + 0.31), (dx, HEAD_Y), (OG[0], HEAD_Y)])    # yes -> outputs
    ax.text(dx + 0.10, mid(mid(*rA) + 0.31, HEAD_Y), "yes", fontsize=SUB, ha="left",
            va="center", color=SUBC, zorder=5)
    ax.text(mid(A[2], OG[0]) - 1.10, HEAD_Y - 0.05, "done: report", fontsize=SUB,
            ha="center", va="top", color=SUBC, zorder=5)

    # ---- validation: checked against hardware, not fed by the simulator -------------
    V = box(10.70, 0.20, 13.10, 0.94, YELLOW, "#9A8A4A", lw=1.3, r=0.12)
    t = ax.text(mid(V[0], V[2]), mid(V[1], V[3]), "Execution on\none A100 GPU",
                fontsize=16, ha="center", va="center", color=INK, linespacing=1.1)
    labels.append((t, V))
    arrow([(mid(V[0], V[2]), V[3]), (mid(V[0], V[2]), OG[1])], both=True)
    ax.text(mid(V[0], V[2]) + 0.14, mid(V[3], OG[1]), "compare", fontsize=SUB,
            ha="left", va="center", color=SUBC)

    # ---- legend -----------------------------------------------------------------------
    for k, (fc_, ec_, lab) in enumerate(((ORANGE, E_ORANGE, "policy under study"),
                                         (YELLOW, "#9A8A4A", "measured on hardware"))):
        xk, yk = 4.20 + k * 2.9, 0.52            # under the loop, clear of the captions
        box(xk, yk - 0.11, xk + 0.32, yk + 0.11, fc_, ec_, lw=1.0, r=0.04)
        ax.text(xk + 0.42, yk, lab, fontsize=SUB, ha="left", va="center", color=SUBC)

    # ---- every label inside its box ---------------------------------------------
    fig.canvas.draw(); r = fig.canvas.get_renderer(); bad = []
    for t, b in labels:
        e = t.get_window_extent(r)
        x0, y0, x1, y1 = e.x0 / fig.dpi, e.y0 / fig.dpi, e.x1 / fig.dpi, e.y1 / fig.dpi
        if x0 < b[0] + 0.04 or x1 > b[2] - 0.04 or y0 < b[1] + 0.03 or y1 > b[3] - 0.03:
            bad.append(t.get_text().replace("\n", " "))
    if bad:
        raise SystemExit("labels overflow their boxes: " + "; ".join(bad))

    os.makedirs("figs", exist_ok=True)
    for ext in ("png",):
        fig.savefig(f"figs/fig23c_sim_flow_slide.{ext}", dpi=200, bbox_inches=None,
                    pad_inches=0)
    plt.close(fig)
    print(f"labels ok ({len(labels)}); wrote figs/fig23c_sim_flow_slide.png "
          f"at {W} x {H} in")


if __name__ == "__main__":
    main()
