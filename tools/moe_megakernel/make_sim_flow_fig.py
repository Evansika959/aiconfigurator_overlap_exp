#!/usr/bin/env python3
"""fig23d -- the simulator's event loop, for the PAPER (Figure 1 of paper/sec_simulator.md).

This is fig23c (make_sim_flow_slide.py), the version that went through five rounds of
review as a slide: the same blocks, the same connections, the same numbered turn of the
loop. Two things differ.

  format   5.5 in wide (one NeurIPS text width), serif, 8 pt titles and 7 pt labels, the
           validated two-hue palette, no slide title: the caption carries it.
  content  step 3 has a fourth handler, the power limiter. Since fig23c was drawn the
           limiter became an event of its own in sim/engine.py (its 20 ms loop), so there
           is now one handler per kind of event:
               batch arrival                    -> Dispatcher
               partition end, power reading     -> Controller
               limiter period                   -> Power Limiter
               set-point takes effect           -> Frequency Assignment
           and step 4 names both shared resources the rate solve models, DRAM and L2.

One turn, as numbered on the figure and as Engine.run() executes it:
  1  Event queue       pop the next event; an empty queue ends the run
  2  Advance time      energy += P x dt, remaining work shrinks, the limiter's window moves
  3  Handle the event  exactly one handler runs; handlers push new events
  4  Bandwidth         rates of everything still running, under the shared DRAM and L2
  5  Power Estimation  board power until the next event

COLOUR IS NEVER THE ONLY ENCODING. Inputs are blue, the policy under study is orange,
and the box that stands for the real GPU is hatched, so the figure survives greyscale.

Drawing helpers and the layout check are fig23's (make_sim_arch_fig.py): every text must
sit inside its box and no two texts may overlap, or the script refuses to save.
"""
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyArrowPatch, FancyBboxPatch, Polygon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs
import make_sim_arch_fig as base

base.W, base.H = 5.5, 2.96          # the helpers read the canvas size from here
W, H = base.W, base.H
INK, SUB = fs.INK, "#3A3A3A"
BLUE, BLUE_FILL = base.BLUE, base.BLUE_FILL
ORANGE, ORANGE_FILL = base.ORANGE, base.ORANGE_FILL
NEUTRAL, GROUP, HATCH = base.NEUTRAL_FILL, "#FAF9F6", "#D8D3CA"
T_SUB, T_TITLE = 7.0, 8.0
DASH = (0, (3.0, 2.0))
mid = lambda a, b: (a + b) / 2


def line(F, pts, lw=0.8, ls="-"):
    """A polyline with mitred corners and no head."""
    xs, ys = zip(*pts)
    F.ax.plot(xs, ys, color=INK, lw=lw, ls=ls, solid_capstyle="butt",
              solid_joinstyle="miter", zorder=4)


def arrow(F, pts, lw=0.8, ls="-", both=False, ms=6.5):
    """A polyline whose last segment carries the head."""
    if len(pts) > 2:
        line(F, pts[:-1], lw, ls)
    F.ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="<|-|>" if both else "-|>",
                                   mutation_scale=ms, color=INK, lw=lw, ls=ls, shrinkA=0,
                                   shrinkB=0, zorder=4))


def module(F, ext, title, sub=None, fc=NEUTRAL, ec=INK, num=None, slot=False):
    """A box with a centred title and an optional sub-label.

    `num` puts the step badge inside the box, at its left: on a corner it collides with
    every line that passes the box. `slot` reserves two title lines and two sub-label
    lines, so a row of boxes lines up when one title wraps."""
    x0, y0, x1, y1 = F.box(*ext, fc=fc, ec=ec)
    cx = mid(x0 + (0.20 if num else 0.0), x1)
    if slot:
        F.text(cx, y1 - 0.16, title, size=T_TITLE, ha="center", box=ext).set_linespacing(1.05)
        F.text(cx, y0 + 0.145, sub, size=T_SUB, ha="center", color=SUB, box=ext).set_linespacing(1.1)
    elif sub:
        F.text(cx, y1 - 0.115, title, size=T_TITLE, ha="center", box=ext)
        F.text(cx, y0 + 0.095, sub, size=T_SUB, ha="center", color=SUB, box=ext)
    else:
        F.text(cx, mid(y0, y1), title, size=T_TITLE, ha="center", box=ext).set_linespacing(1.1)
    if num:
        F.tag(x0 + 0.115, mid(y0, y1), str(num), ec, box=ext)
    return ext


def hatched(F, x0, y0, x1, y1, r=0.045):
    """The box that stands for the real GPU: hatched, so it does not depend on a hue."""
    style = f"round,pad=0,rounding_size={r}"
    F.ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle=style, fc="white",
                                  ec=HATCH, lw=0, hatch="/////", zorder=2))
    F.ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle=style, fc="none",
                                  ec=INK, lw=0.6, zorder=2.1))
    return (x0, y0, x1, y1)


def main():
    F = base.Fig()
    label = lambda x, y, s, **kw: F.text(x, y, s, size=T_SUB, color=SUB, **kw)

    # ---- one grid: three rows shared by inputs, loop and outputs ---------------------
    HEAD_Y = H - 0.13
    rA = (H - 0.68, H - 0.34)                     # event queue, empty?, advance time
    g_top = rA[0] - 0.18
    rB = (g_top - 0.92, g_top - 0.34)             # the handlers
    g_bot = rB[0] - 0.11
    rC = (g_bot - 0.17 - 0.34, g_bot - 0.17)      # power estimation, bandwidth
    cx0, cx1, cy1 = 1.17, 4.36, H - 0.02          # the frame of one turn
    XL, XR, XM = 1.52, 4.27, 3.20                 # left edge, right edge, right column
    HB = (XL, rC[0] - 0.24, XR, rC[0] - 0.07)     # hardware parameters
    cy0 = HB[1] - 0.06

    F.text(0.51, HEAD_Y, "Inputs", size=T_TITLE, weight="bold", ha="center")
    F.box(cx0, cy0, cx1, cy1, fc="white", ec=fs.MUTED, ls=(0, (4, 2)), lw=0.7, r=0.07, z=1)
    F.text(cx0 + 0.10, HEAD_Y, "One turn", size=T_TITLE, weight="bold")

    # ---- steps 1, 2, 4, 5 ----------------------------------------------------------
    Q = module(F, (XL, rA[0], XL + 0.90, rA[1]), "Event queue", "next event", num=1)
    A = module(F, (XM, rA[0], XR, rA[1]), "Advance time", "energy += P·Δt", num=2)
    Pwr = module(F, (XL, rC[0], XL + 1.26, rC[1]), "Power Estimation", "P for next interval",
                 num=5)
    Bnd = module(F, (XM, rC[0], XR, rC[1]), "Bandwidth", "DRAM, L2 → rates", num=4)

    # the loop's only exit: a decision, not a line off the top of a box
    yA, dw, dh = mid(*rA), 0.24, 0.15
    dx = Q[2] + 0.11 + dw
    F.ax.add_patch(Polygon([(dx, yA + dh), (dx + dw, yA), (dx, yA - dh), (dx - dw, yA)],
                           closed=True, fc="white", ec=INK, lw=0.7, zorder=6))
    F.ax.text(dx, yA - 0.003, "empty?", fontsize=T_SUB, ha="center", va="center", color=INK,
              zorder=7)
    arrow(F, [(Q[2], yA), (dx - dw, yA)], ms=5.5)                      # 1 -> ?
    arrow(F, [(dx + dw, yA), (A[0], yA)], ms=5.5)                      # no -> 2
    label(mid(dx + dw, A[0]), yA + 0.085, "no", ha="center")

    # ---- step 3: EXACTLY ONE handler. A bar fans out to all of them with "or" between,
    # and a second bar collects them before the single way out: an arrow that met the
    # group above any one handler would be read as going through that handler.
    G = F.box(XL, g_bot, XR, g_top, fc=GROUP, ec=INK, lw=0.5, r=0.06, z=1.5)
    F.tag(G[0] + 0.115, g_top - 0.105, "3", INK)
    F.text(G[0] + 0.225, g_top - 0.105, "Handle the event: exactly one handler runs", size=7.5)
    bw, gap = 0.62, 0.03                 # handlers left-packed: the spine runs to their right
    xs = [XL + 0.06 + bw / 2 + k * (bw + gap) for k in range(4)]
    hb = lambda x: (x - bw / 2, rB[0], x + bw / 2, rB[1])
    Ctl = module(F, hb(xs[0]), "Controller", "on: reading or\npartition end", fc=ORANGE_FILL,
                 ec=ORANGE, slot=True)
    module(F, hb(xs[1]), "Power\nLimiter", "on: its 20 ms\nperiod", slot=True)
    module(F, hb(xs[2]), "Frequency\nAssignment", "on: set-point\ntakes effect", slot=True)
    module(F, hb(xs[3]), "Dispatcher", "on: batch\narrival", slot=True)
    # one vertical spine, to the RIGHT of every handler: on a handler's centreline it
    # reads as an arrow into that handler. It meets each bar at its end.
    xa = XR - 0.06
    ybar, ycol = rB[1] + 0.10, rB[0] - 0.055
    line(F, [(xa, A[1]), (xa, ybar), (xs[0], ybar)])                   # 2 -> 3
    for x in xs:
        arrow(F, [(x, ybar), (x, rB[1])], lw=0.7, ms=5)
        line(F, [(x, rB[0]), (x, ycol)], lw=0.7)
    for a, b in zip(xs[:-1], xs[1:]):
        label(mid(a, b), ybar + 0.004, "or", ha="center").set_bbox(
            dict(fc=GROUP, ec="none", pad=0.8))
    arrow(F, [(xs[0], ycol), (xa, ycol), (xa, Bnd[3])])                # 3 -> 4
    arrow(F, [(Bnd[0], mid(*rC)), (Pwr[2], mid(*rC))])                 # 4 -> 5
    # 5 -> 1: out of Power Estimation's left side, up the gutter, into the queue
    gx, xl, yhi = 1.35, XL + 0.13, mid(g_top, Q[1])
    arrow(F, [(Pwr[0], mid(*rC)), (gx, mid(*rC)), (gx, yhi), (xl, yhi), (xl, Q[1])])
    label(gx, mid(Pwr[3], g_bot), "repeat", ha="center").set_bbox(
        dict(fc="white", ec="none", pad=1.0))
    # handlers schedule future events (a set-point, the next reading)
    xn = XL + 0.58
    arrow(F, [(xn, g_top), (xn, Q[1])], lw=0.7, ls=DASH, ms=5)
    label(xn + 0.06, mid(g_top, Q[1]), "handlers add new events")

    # hardware parameters feed every step: a band, not an arrow into one box
    F.box(*HB, fc=BLUE_FILL, ec=BLUE, lw=0.5, r=0.035, z=1.5)
    label(mid(HB[0], HB[2]), mid(HB[1], HB[3]), "hardware parameters: used by every step",
          ha="center", box=HB)

    # ---- inputs ------------------------------------------------------------------------
    ins = [("Workload", "Batch arrivals over time,\nexpert routing", rA),
           ("Targets", "Min energy under SLO, or\nmin latency under TDP", rB),
           ("Hardware", "V/f domains, #SM,\nTDP, limiter", rC)]
    x_in = (0.03, 0.99)
    for title, cap, row in ins:
        ym = mid(*row)
        module(F, (x_in[0], ym - 0.15, x_in[1], ym + 0.15), title, fc=BLUE_FILL, ec=BLUE)
        label(x_in[0] + 0.02, ym - 0.19, cap, va="top").set_linespacing(1.15)
        if row is rA:
            arrow(F, [(x_in[1], ym), (Q[0], ym)])
        elif row is rB:                  # a hop over the return line: a crossing, not a joint
            line(F, [(x_in[1], ym), (gx - 0.04, ym)])
            F.ax.add_patch(Arc((gx, ym), 0.08, 0.08, theta1=0, theta2=180, color=INK, lw=0.8,
                               zorder=5))
            arrow(F, [(gx + 0.04, ym), (Ctl[0], ym)])
        else:                            # turns down OUTSIDE the frame, so it cannot be
            yb = mid(HB[1], HB[3])       # read as running into the return line
            arrow(F, [(x_in[1], ym), (1.08, ym), (1.08, yb), (HB[0], yb)])

    # ---- outputs: all three are reported when the queue runs dry ----------------------
    # The exit lands beside the "Outputs" heading, inside the group: at any one output's
    # height it would be read as feeding only that output.
    OG = F.box(4.52, mid(*rC) - 0.22, 5.48, cy1, fc=GROUP, ec=INK, lw=0.5, r=0.06, z=1.5)
    F.text(mid(OG[0], OG[2]) + 0.04, HEAD_Y, "Outputs", size=T_TITLE, weight="bold", ha="center")
    for name, row in (("E2E Latency", rA), ("Energy", rB), ("Power Traces", rC)):
        ym = mid(*row)
        module(F, (OG[0] + 0.065, ym - 0.15, OG[2] - 0.065, ym + 0.15), name, fc="white")
    arrow(F, [(dx, yA + dh), (dx, HEAD_Y), (OG[0], HEAD_Y)])           # yes -> outputs
    label(dx + 0.04, mid(yA + dh, HEAD_Y), "yes")
    label(mid(A[0], A[2]), HEAD_Y - 0.03, "done: report", ha="center", va="top")

    # ---- validation: checked against hardware, not fed by the simulator ---------------
    V = hatched(F, OG[0], 0.03, OG[2], 0.35)
    t = F.text(mid(V[0], V[2]), mid(V[1], V[3]), "Execution on\none A100 GPU", size=T_TITLE,
               ha="center", box=V)
    t.set_linespacing(1.05)
    t.set_bbox(dict(fc="white", ec="none", pad=0.6))           # no hatch through the letters
    arrow(F, [(mid(V[0], V[2]), V[3]), (mid(V[0], V[2]), OG[1])], both=True, ms=5.5)
    label(mid(V[0], V[2]) + 0.06, mid(V[3], OG[1]), "compare")

    # ---- legend, under the loop ---------------------------------------------------------
    yk = cy0 - 0.13
    F.box(XL, yk - 0.045, XL + 0.15, yk + 0.045, fc=ORANGE_FILL, ec=ORANGE, r=0.02)
    label(XL + 0.20, yk, "policy under study")
    hatched(F, XL + 1.30, yk - 0.045, XL + 1.45, yk + 0.045, r=0.02)
    label(XL + 1.50, yk, "measured on hardware")

    n = base.check_layout(F.fig)
    os.makedirs("figs", exist_ok=True)
    F.fig.savefig("figs/fig23d_sim_event_loop.png", bbox_inches=None, pad_inches=0)
    plt.close(F.fig)
    print(f"layout ok ({n} items); wrote figs/fig23d_sim_event_loop.png at {W} x {H} in")


if __name__ == "__main__":
    main()
