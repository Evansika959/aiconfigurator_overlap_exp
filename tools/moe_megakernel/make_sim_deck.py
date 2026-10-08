#!/usr/bin/env python3
"""slide1-4 -- the simulator and what it found, as four SLIDES in the style of fig23c.

  slide1_sim_why         the observation, the idea, and why it has to be simulated
  slide2_sim_how         one MoE layer inside the simulator, event by event
  slide3_sim_validation  the blind test, and the errors against the A100
  slide4_sim_results     what a second V/f domain is worth
  slide5_sim_boost       one layer at the power cap, the hot domain boosted when the cold
                         one finishes (a split unbalanced on purpose, so that there is
                         something to boost into)

Nothing on the slides is typed in by hand that a file can supply: the routing is the
traced one, the timeline of slide 2 is the simulator's own run of that layer, the trace
of slide 3 is the registered first-shot replay (data/heldout_first_shot), and the errors
and results are read from sim/calib and sim/out. Rerun after `bash sim/reproduce.sh`.

SLIDE CONVENTIONS are fig23c's (make_sim_flow_slide.py), not figstyle's: 13.33 x 7.5 in
(PowerPoint 16:9), DejaVu Sans, 11.5-30 pt, 200 dpi; blue for inputs and questions, grey
modules, orange for the policy under study, yellow for what was measured on hardware.
Data marks use figstyle's validated colours. Not for the paper.

The script checks, before it saves, that every label fits its box, stays on the slide
and touches no other label.

  source env.sh && python3 make_sim_deck.py [--only N]
"""
import argparse, csv, json, os, sys
from collections import defaultdict
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import figstyle as fs
from sim.calib import Calib
from sim.control import BoostOnFinish, Fixed
from sim.demand import SHAPES, decode_counts
from sim.engine import Engine, Kernel, Shared
from sim.hardware import Hardware, two_domain
from sim.study import governor_point
from sim.workload import build

W, H = 13.33, 7.5
INK, GREY_T, LINE = "#111111", "#555555", "#555555"
BLUE, GREY, ORANGE, YELLOW = "#F3F6FB", "#EEEEEE", "#F7E6D2", "#FBF4D6"
E_BLUE, E_GREY, E_ORANGE, E_YELLOW = "#6F7F9C", "#666666", "#A08468", "#9A8A4A"
RED = "#D0392B"
SUB, SUBC, HEAD_Y = 13.5, "#3A3A3A", 6.07
COLD, HOT, EPI, BASE, GRID = fs.COMM, fs.GEMM, fs.MUTED, fs.BASE, fs.GRID
MEAS, SIM = INK, fs.GEMM
TITLE = "Dynamic GPU DVFS Simulator: "

# the running example of slides 1 and 2: one 2048-token decode step of Qwen3-30B-A3B,
# its 64 least-routed experts on a 32-SM cold domain (make_sim_layer_fig.py's case)
CELLS, BM, N_COLD, COLD_SMS, SMS = 32, 64, 64, 32, 108
minus = lambda s: s.replace("-", "−")            # a real minus sign in every number


def out_path(name):
    return os.path.join(HERE, "sim", "out", name)


# ------------------------------------------------------------------ the page ----
class Slide:
    """A 16:9 page in inches, with fig23c's title bar and block vocabulary."""

    def __init__(self, title):
        self.fig = plt.figure(figsize=(W, H))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, W); self.ax.set_ylim(0, H); self.ax.axis("off")
        self.boxed, self.free = [], []                  # labels to check before saving
        self.text(0.45, 6.98, TITLE + title, size=30, color=INK, va="center")
        self.ax.plot([0.25, W - 0.25], [6.55, 6.55], color=RED, lw=2.2)

    def box(self, x0, y0, x1, y1, fc=GREY, ec=E_GREY, ls="-", lw=1.2, r=0.10, z=2):
        self.ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                         boxstyle=f"round,pad=0,rounding_size={r}", fc=fc,
                                         ec=ec, lw=lw, ls=ls, zorder=z))
        return (x0, y0, x1, y1)

    def text(self, x, y, s, size=SUB, color=SUBC, ha="left", va="center_baseline", weight="normal",
             inside=None, ax=None, z=5, **kw):
        """One label. `inside`: the box it must fit in. `ax`: a chart, x and y in its data.
        Centred on its baseline by default, so that labels on one row line up whether or
        not they have descenders."""
        t = (ax or self.ax).text(x, y, s, fontsize=size, color=color, ha=ha, va=va,
                                 fontweight=weight, zorder=z, linespacing=1.22, **kw)
        if inside:
            self.boxed.append((t, inside))
        else:
            self.free.append(t)
        return t

    def lines(self, x, y, rows, size=SUB, color=SUBC, ha="left", anchor="top", inside=None, **kw):
        """Several lines at ONE pitch; y is the block's top, middle or bottom. A multi-line
        string will not do: matplotlib spaces its lines by each line's own glyphs, so a
        line with "V/f" or "(" in it pushes the next one down. Returns the block's bottom."""
        rows = rows.split("\n") if isinstance(rows, str) else list(rows)
        asc, desc, step = 0.76 * size / 72, 0.22 * size / 72, 1.25 * size / 72
        height = asc + desc + (len(rows) - 1) * step
        top = {"top": y, "middle": y + height / 2, "bottom": y + height}[anchor]
        for k, row in enumerate(rows):
            t = self.text(x, top - asc - k * step, row, size=size, color=color, ha=ha, va="baseline",
                          inside=inside, **kw)
            t.block = id(rows)                          # its lines are spaced by construction
        return top - height

    def head(self, x, s):
        return self.text(x, HEAD_Y, s, size=20, color=GREY_T, weight="bold", va="baseline")

    def badge(self, x, y, num):
        self.ax.add_patch(Circle((x, y), 0.17, fc="white", ec=INK, lw=1.2, zorder=8))
        self.ax.text(x, y - 0.01, str(num), fontsize=12, fontweight="bold", ha="center",
                     va="center", color=INK, zorder=9)

    def card(self, ext, title, body=None, fc=GREY, ec=E_GREY, num=None, tsize=16, bsize=13):
        """A box with a title and a few lines under it. Returns where the body ends, so
        that a caller can put more under it."""
        x0, y0, x1, y1 = self.box(*ext, fc, ec)
        if num:
            self.badge(x0 + 0.33, y1 - 0.16 - 0.49 * tsize / 72, num)
        y = self.lines(x0 + (0.62 if num else 0.2), y1 - 0.16, title, size=tsize, color=INK, inside=ext)
        return self.lines(x0 + 0.2, y - 0.07, body, size=bsize, inside=ext) if body else y - 0.07

    def arrow(self, pts, color=LINE, lw=1.5, z=4, ms=16):
        for a, b in list(zip(pts[:-1], pts[1:]))[:-1]:
            self.ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw,
                         solid_capstyle="butt", zorder=z)
        self.ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=ms,
                                          color=color, lw=lw, shrinkA=0, shrinkB=0, zorder=z))

    def swatch(self, x, y, color, hatch=None, w=0.32, h=0.18):
        self.ax.add_patch(Rectangle((x, y - h / 2), w, h, fc=color, hatch=hatch, lw=0,
                                    ec="white" if hatch else "none", zorder=3))

    def chart(self, x0, y0, x1, y1, grid="y"):
        ax = self.fig.add_axes([x0 / W, y0 / H, (x1 - x0) / W, (y1 - y0) / H])
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(INK); ax.spines[sp].set_linewidth(1.0)
        ax.tick_params(colors=INK, labelsize=12, width=1.0, length=4)
        if grid:
            ax.grid(axis=grid, color=GRID, lw=0.9, zorder=0); ax.set_axisbelow(True)
        return ax

    def save(self, stem):
        self.fig.canvas.draw()
        r, d = self.fig.canvas.get_renderer(), self.fig.dpi
        def ext(t):
            e = t.get_window_extent(r)
            return e.x0 / d, e.y0 / d, e.x1 / d, e.y1 / d
        name = lambda t: repr(t.get_text().replace("\n", " ")[:40])
        bad = []
        for t, b in self.boxed:
            x0, y0, x1, y1 = ext(t)
            if x0 < b[0] + 0.06 or x1 > b[2] - 0.06 or y0 < b[1] + 0.04 or y1 > b[3] - 0.04:
                bad.append(f"{name(t)} overflows its box")
        every = [t for t, _ in self.boxed] + self.free
        for i, t in enumerate(every):
            x0, y0, x1, y1 = ext(t)
            if x0 < 0.2 or x1 > W - 0.2 or y0 < 0.1 or y1 > H - 0.1:
                bad.append(f"{name(t)} leaves the slide")
            for u in every[i + 1:]:
                if getattr(t, "block", 0) == getattr(u, "block", 1):
                    continue                            # matplotlib's line boxes are padded
                a0, b0, a1, b1 = ext(u)
                if min(x1, a1) - max(x0, a0) > 0.02 and min(y1, b1) - max(y0, b0) > 0.02:
                    bad.append(f"{name(t)} runs into {name(u)}")
        if bad:
            raise SystemExit(f"{stem}: " + "; ".join(bad))
        path = os.path.join(HERE, "figs", stem + ".png")
        self.fig.savefig(path, dpi=200, bbox_inches=None, pad_inches=0)
        plt.close(self.fig)
        print(f"  {stem}: {len(every)} labels ok; wrote {path}")


# ------------------------------------------------------------------- slide 1 ----
def slide_why():
    s = Slide("Why We Built It")
    counts = np.sort(decode_counts(SHAPES["qwen"], CELLS, 0))[::-1]      # most routed first
    E = len(counts)
    cold_share = counts[E - N_COLD:].sum() / counts.sum() * 100

    # ---- observation: the traced routing of the running example
    s.head(0.45, "Observation")
    s.lines(0.45, 5.90, f"MoE routing is skewed: half of the\nexperts do {100 - cold_share:.0f}% of the work")
    ax = s.chart(1.30, 2.75, 4.45, 5.05)
    x = np.arange(E)
    ax.bar(x[:E - N_COLD], counts[:E - N_COLD], width=1.0, color=HOT, lw=0, zorder=3)
    ax.bar(x[E - N_COLD:], counts[E - N_COLD:], width=1.0, color=COLD, lw=0, zorder=3)
    ax.set_xlim(-1, E); ax.set_ylim(0, counts[0] * 1.04)
    ax.set_xticks([0, 64, 127]); ax.set_xticklabels(["1", "65", "128"])
    ax.set_yticks([0, 200, 400, 600])
    ax.set_xlabel("experts, most used first", fontsize=13, color=INK)
    ax.set_ylabel("token-expert pairs", fontsize=13, color=INK)
    s.text(26, counts[0] * 0.80, f"{E - N_COLD} hot experts:\n{100 - cold_share:.0f}% of the work",
           size=13, color=INK, ax=ax, va="center")
    s.text(E - 1, counts[0] * 0.34, f"{N_COLD} cold experts:\n{cold_share:.0f}%", size=13,
           color=INK, ax=ax, va="center", ha="right", multialignment="left")
    s.lines(0.45, 1.92, "Qwen3-30B-A3B (128 experts), one MoE\nlayer, one 2048-token decode step, traced",
            size=12, color=GREY_T)

    # ---- idea: the chip as two V/f domains
    s.head(4.95, "Idea")
    s.lines(4.95, 5.90, "Give cold and hot experts their\nown clock: two V/f domains")
    p, c, rows, gx, gy = 0.128, 0.108, 4, 5.07, 3.84
    cols_c, cols_h, gap = COLD_SMS // rows, (SMS - COLD_SMS) // rows, 0.22
    hx = gx + cols_c * p + gap
    x1 = hx + cols_h * p - (p - c)
    top, bot = gy + rows * p - (p - c) + 0.12, gy - 0.12
    s.box(gx - 0.12, bot, x1 + 0.12, top, "white", E_GREY, lw=1.0, r=0.08)
    for x0, n, col in ((gx, cols_c, COLD), (hx, cols_h, HOT)):
        for i in range(n):
            for j in range(rows):
                s.ax.add_patch(Rectangle((x0 + i * p, gy + j * p), c, c, fc=col, ec="none", zorder=3))
    s.lines(gx - 0.12, top + 0.10, f"cold domain: {COLD_SMS} SMs\nlow clock, low voltage", size=13,
            color=INK, anchor="bottom")
    s.lines(x1 + 0.12, top + 0.10, "one GPU, 108 SMs", size=12, color=GREY_T, anchor="bottom", ha="right")
    s.lines(x1 + 0.12, bot - 0.10, f"hot domain: {SMS - COLD_SMS} SMs\nhigh clock", size=13, color=INK,
            ha="right")
    s.card((4.95, 1.17, 8.85, 2.55), "Hoped-for gain",
           "Cold experts cost less energy;\nthe freed power lets hot experts\nrun faster under the 400 W cap.",
           fc=ORANGE, ec=E_ORANGE)

    # ---- problem and approach
    s.head(9.25, "Problem and approach")
    X0, X1 = 9.25, 13.05
    xm = (X0 + X1) / 2
    P = s.box(X0, 5.05, X1, 5.90, "white", E_GREY, ls=(0, (5, 3)), lw=1.3)
    s.lines(xm, 5.475, "No GPU has two V/f domains:\none clock drives all 108 SMs", size=14.5, color=INK,
            ha="center", anchor="middle", inside=P)
    steps = [(YELLOW, E_YELLOW, "Measure the real A100", "5000 timed runs, clocks\nlocked at 510–1410 MHz"),
             (GREY, E_GREY, "Calibrate the simulator", "it reproduces the A100\nwithin 1–4%"),
             (ORANGE, E_ORANGE, "Ask what-if", "add V/f domains and\nclock policies")]
    y1 = 4.55
    s.arrow([(xm, 5.05), (xm, y1)])
    s.text(xm + 0.14, (5.05 + y1) / 2, "so we simulate one", size=12.5)
    for k, (fc, ec, title, sub) in enumerate(steps):
        b = s.box(X0, y1 - 0.98, X1, y1, fc, ec)
        s.badge(X0 + 0.33, y1 - 0.49, k + 1)
        cx = (X0 + 0.55 + X1) / 2
        s.text(cx, y1 - 0.27, title, size=16, color=INK, ha="center", inside=b)
        s.lines(cx, y1 - 0.66, sub, size=12.5, ha="center", anchor="middle", inside=b)
        if k < 2:
            s.arrow([(xm, y1 - 0.98), (xm, y1 - 1.20)])
        y1 -= 1.20

    # ---- the question the simulator is asked
    Q = s.box(0.45, 0.30, 13.05, 0.85, BLUE, E_BLUE, lw=1.0, r=0.08)
    s.text(W / 2, 0.575, "Question: under the same 400 W power wall, what does a second V/f domain buy?",
           size=16, color=INK, ha="center", inside=Q)
    s.save("slide1_sim_why")


# ------------------------------------------------------------------- slide 2 ----
def one_layer(cal):
    """The simulator's own run of the running example, both domains at 1290 MHz: the layer
    record and the trace (start, end, board W) in ms from the layer's start."""
    b = build([dict(phase="decode", cells=CELLS, layers=[0], repeat=3)])
    e = Engine(two_domain(COLD_SMS), cal, Kernel("persistent", BM, N_COLD), b, Fixed([1290, 1290]),
               shared=Shared(), limiter=False).run()
    L = e.layers[-1]
    seg = [(a - L["t_start"], z - L["t_start"], p) for a, z, p, _, _ in e.trace if a >= L["t_start"] - 1e-9]
    return L, seg


def slide_how(cal):
    s = Slide("How It Works")
    L, seg = one_layer(cal)
    T = L["ms"]
    fin = {k: v - L["t_start"] for k, v in L["finish"].items()}

    s.head(0.45, "One MoE layer, event by event")
    s.lines(0.45, 5.90, "The simulator's run of a 2048-token decode step: 64 cold experts on\n"
            "32 SMs, 64 hot experts on 76 SMs, both at 1290 MHz")
    XL, XR, xmax = 1.80, 7.70, T * 1.02
    g = s.chart(XL, 3.72, XR, 4.66, grid=None)
    pw = s.chart(XL, 1.52, XR, 3.14)

    # ---- who runs when: two lanes, stages separated by barriers
    t_act0 = max(fin[("cold", "g1")], fin[("hot", "g1")])           # GEMM 1's barrier
    t_act1 = fin[("epilogue", "act")]
    t_g2 = max(fin[("cold", "g2")], fin[("hot", "g2")])             # GEMM 2's barrier
    start, end = {"g1": 0.0, "g2": t_act1}, {"g1": t_act0, "g2": t_g2}
    for name, y, col in (("cold", 2, COLD), ("hot", 1, HOT)):
        for st in ("g1", "g2"):
            a, b = start[st], fin[(name, st)]
            g.barh(y, b - a, left=a, height=0.66, color=col, edgecolor="white", lw=1.0, zorder=3)
            s.text((a + b) / 2, y, "GEMM 1" if st == "g1" else "GEMM 2", size=13, color="white",
                   weight="bold", ha="center", ax=g)
            if end[st] - b > 0.004 * T:                              # waiting at the barrier
                g.barh(y, end[st] - b, left=b, height=0.66, facecolor="white", edgecolor=col,
                       hatch="////", lw=1.0, zorder=3)
        for a, b in ((t_act0, t_act1), (t_g2, T)):                   # the epilogue is on every SM
            g.barh(y, b - a, left=a, height=0.66, color=EPI, edgecolor="white", lw=1.0, zorder=3)
    g.set_yticks([2, 1]); g.set_yticklabels(["cold experts\n32 SMs", "hot experts\n76 SMs"], fontsize=13)
    g.set_ylim(0.5, 2.5); g.set_xlim(0, xmax)
    g.tick_params(axis="x", labelbottom=False, length=0); g.tick_params(axis="y", length=0)
    for sp in ("bottom", "left"):
        g.spines[sp].set_visible(False)

    # ---- board power: a step function of the events
    xs, ys = [], []
    for a, b, p in seg:
        xs += [a, b]; ys += [p, p]
    pw.fill_between(xs, ys, color=BASE, lw=0, zorder=2)
    pw.plot(xs, ys, color=INK, lw=1.8, zorder=4)
    pw.set_xlim(0, xmax); pw.set_ylim(0, 460); pw.set_yticks([0, 100, 200, 300, 400])
    pw.set_xlabel("time since the layer started (ms)", fontsize=13, color=INK)
    pw.set_ylabel("board power (W)", fontsize=13, color=INK)
    events = sorted({round(a, 6) for a, _, _ in seg if a > 1e-9} | {round(T, 6)})
    for t in events:                                                 # one dashed line per event
        for ax in (g, pw):
            ax.axvline(t, color="#8A8A8A", lw=1.0, ls=(0, (3, 2.5)), zorder=5 if ax is pw else 2)

    # ---- what the picture says
    t1 = fin[("cold", "g1")]
    s.text(t1 + 0.06, 392, "event: the cold partition finishes.\nRates and power are solved again.",
           size=13, color=INK, ax=pw, va="center", z=7, bbox=dict(fc="white", ec="none", pad=2.0))
    pw.annotate("", xy=(t1, ys[0] + 4), xytext=(t1 + 0.055, 350), zorder=8,
                arrowprops=dict(arrowstyle="-|>", color=LINE, lw=1.3, mutation_scale=12))
    s.text(t1 / 2, 150, "between two events\nnothing changes:\nenergy += P · Δt", size=13,
           color=INK, ax=pw, ha="center", va="center")
    xb = XL + (t_act0 / xmax) * (XR - XL)
    s.lines(xb - 0.14, 5.05, "barrier: the next stage starts\nwhen both partitions are done", size=13,
            color=INK, ha="right", anchor="middle")
    s.arrow([(xb, 5.24), (xb, 4.68)], lw=1.3, ms=12)
    ky = 3.42                                                        # the two marks the lanes do not name
    s.swatch(XL, ky, EPI)
    s.text(XL + 0.42, ky, "activation, top-k sum (all SMs)", size=12.5)
    s.ax.add_patch(Rectangle((XL + 3.45, ky - 0.09), 0.32, 0.18, fc="white", ec=LINE, hatch="////", lw=1.0, zorder=3))
    s.text(XL + 3.87, ky, "waiting at the barrier", size=12.5)

    # ---- the four things the engine does
    X0, X1 = 8.15, 13.05
    s.card((X0, 5.05, X1, 6.32), "Jump from event to event",
           "Between events every job runs at a\nconstant rate and power is constant:\n"
           "progress += Δt / T_job      energy += P · Δt")
    s.card((X0, 3.66, X1, 4.93), "How fast is each job?",
           "Per SM: weight delivery + tensor work\n+ DRAM wait. DRAM and L2 are shared,\nso a busy neighbour slows it down.")
    s.card((X0, 2.27, X1, 3.54), "How many watts?",
           "Idle power + MACs and bytes per second\n× their energy. The SM part scales\nwith voltage squared.")
    s.card((X0, 0.88, X1, 2.15), "Who sets the clocks?",
           "A controller, one clock per V/f domain,\nunder the board's own limiter: a 20 ms\n"
           "mean above 400 W brings the clock down.", fc=ORANGE, ec=E_ORANGE)
    B = s.box(0.45, 0.25, 13.05, 0.70, YELLOW, E_YELLOW, lw=1.0, r=0.08)
    s.text(W / 2, 0.475, "Every coefficient is measured on the A100: 9 numbers for time, "
           "4 energies and a voltage curve for power", size=13.5, color=INK, ha="center", inside=B)
    s.save("slide2_sim_how")


# ------------------------------------------------------------------- slide 3 ----
LIMITS = dict(latency_rms_pct=3.0, power_rms_pct=2.0, clock_rms_mhz=20.0, energy=2.0, step=1.5)  # HELDOUT.md


def slide_validation():
    s = Slide("Validation on the A100")
    src = os.path.join(HERE, "data", "heldout_first_shot")
    S = defaultdict(lambda: ([], []))
    with open(os.path.join(src, "trace_heldout1_run0.csv")) as fh:
        for r in csv.DictReader(fh):
            S[r["series"]][0].append(float(r["t_ms"]) / 1e3); S[r["series"]][1].append(float(r["value"]))
    with open(os.path.join(HERE, "data", "governor_trace_heldout1_segments_qwen.csv")) as fh:
        segs = [r for r in csv.DictReader(fh) if r["run"] == "0"]
    tok = [int(r["tokens"]) for r in segs if r["load"] != "idle"]
    t_end = float(segs[-1]["t_end_ms"]) / 1e3
    ho = {}
    for tag in ("heldout1", "heldout2"):
        with open(os.path.join(src, f"trace_{tag}.json")) as fh:
            ho[tag] = json.load(fh)["summary"]

    # ---- the registered trace, as the frozen simulator replayed it
    s.head(0.45, "Blind test on a moving load")
    s.lines(0.45, 5.90, f"{t_end:.0f} s of mixed batches ({min(tok)}–{max(tok)} tokens) under the stock governor.\n"
            "Plan, pass criteria and simulator were frozen before the run.")
    XL, XR = 1.30, 8.10
    ac, ap, al = s.chart(XL, 4.02, XR, 4.92), s.chart(XL, 2.72, XR, 3.80), s.chart(XL, 1.00, XR, 2.50)
    ac.step(*S["meas_clock_mhz"], where="post", color=MEAS, lw=1.6, zorder=3)
    ac.step(*S["sim_clock_exact_mhz"], where="post", color=SIM, lw=1.6, zorder=4)
    ac.set_ylim(1050, 1440); ac.set_yticks([1100, 1250, 1400]); ac.set_ylabel("SM clock\n(MHz)", fontsize=13, color=INK)
    ap.step(*S["meas_power_w"], where="post", color=MEAS, lw=1.6, zorder=3)
    ap.step(*S["sim_power_w"], where="pre", color=SIM, lw=1.6, zorder=4)
    ap.axhline(400, color=INK, lw=0.9, ls=(0, (3, 2)), zorder=2)
    ap.set_ylim(0, 470); ap.set_yticks([0, 200, 400]); ap.set_ylabel("board power\n(W)", fontsize=13, color=INK)
    al.plot(*S["meas_layer_ms"], ls="", marker=".", ms=3.2, color=MEAS, zorder=3)
    al.plot(*S["sim_layer_ms"], ls="", marker=".", ms=2.6, color=SIM, zorder=4)
    al.set_yscale("log"); al.set_yticks([1, 2, 4]); al.set_yticklabels(["1", "2", "4"]); al.minorticks_off()
    al.set_ylim(0.85, 5.2); al.set_ylabel("ms per\nMoE layer", fontsize=13, color=INK)
    al.set_xlabel("time (s)", fontsize=13, color=INK)
    for ax in (ac, ap, al):
        ax.set_xlim(0, t_end)
        for r in segs:
            if r["load"] == "idle":
                ax.axvspan(float(r["t_start_ms"]) / 1e3, float(r["t_end_ms"]) / 1e3, color=GRID, lw=0, zorder=1)
    for ax in (ac, ap):
        ax.tick_params(labelbottom=False)
    s.fig.align_ylabels([ac, ap, al])
    ly = 5.16                                                        # the legend, directly above
    for x0, col, lab in ((4.05, MEAS, "measured on the A100"), (6.75, SIM, "simulated")):
        s.ax.plot([x0, x0 + 0.42], [ly, ly], color=col, lw=2.4, solid_capstyle="butt")
        s.text(x0 + 0.52, ly, lab, size=12.5, color=INK)
    s.swatch(XL, ly, GRID)
    s.text(XL + 0.42, ly, "idle", size=12.5, color=INK)
    s.ax.plot([2.35, 2.77], [ly, ly], color=INK, lw=0.9, ls=(0, (3, 2)))
    s.text(2.87, ly, "400 W cap", size=12.5, color=INK)

    # ---- how far it is from the A100
    X0, X1 = 8.60, 13.05
    s.head(X0, "Error against the A100")
    with open(os.path.join(HERE, "sim", "calib", "a100_qwen.json")) as fh:
        meta = json.load(fh)["meta"]
    with open(out_path("validation.json")) as fh:
        pinned = json.load(fh)["pinned_layer"]["time"]
    energy = max(abs(e) for t in ho.values() for e in t["energy_err_pct"])
    runs = sum(len(t["energy_err_pct"]) for t in ho.values())
    tiles = [(f"{meta['rate']['all']['rms']:.1f}%", f"time, one partition\n{meta['rate']['all']['n']} runs (rms)"),
             (f"{pinned['rms']:.1f}%", f"time, whole layer\n{pinned['n']} runs, not fitted (rms)"),
             (f"{meta['power']['all']['rms']:.1f}%", f"board power\n{meta['power']['all']['n']} records (rms)"),
             (f"≤ {energy:.1f}%", f"energy of a {t_end:.0f} s run\nblind test, {runs} runs")]
    tw, th, gapx, gapy, ty = (X1 - X0 - 0.15) / 2, 1.22, 0.15, 0.15, 5.90
    for k, (big, lab) in enumerate(tiles):
        x0, y1 = X0 + (k % 2) * (tw + gapx), ty - (k // 2) * (th + gapy)
        b = s.box(x0, y1 - th, x0 + tw, y1, "white", E_GREY)
        s.text(x0 + tw / 2, y1 - 0.38, big, size=26, color=INK, weight="bold", ha="center", inside=b)
        s.lines(x0 + tw / 2, y1 - 0.87, lab, size=11.5, ha="center", anchor="middle", inside=b)
    # the registered criteria: four passed on both traces, the clock missed on one
    miss = [(k, ho[t][k]) for t in ho for k in ("latency_rms_pct", "power_rms_pct", "clock_rms_mhz") if ho[t][k] > LIMITS[k]]
    miss += [("energy", e) for t in ho.values() for e in t["energy_err_pct"] if abs(e) > LIMITS["energy"]]
    miss += [("step", t["transient_rms_pct"]["load change"]) for t in ho.values()
             if t["transient_rms_pct"]["load change"] > LIMITS["step"]]
    assert [k for k, _ in miss] == ["clock_rms_mhz"], miss            # the card below says exactly this
    s.card((X0, 1.95, X1, 3.08), "Blind test: 4 of 5 criteria passed",
           f"Missed: the clock on one of two traces\n({miss[0][1]:.1f} MHz rms, limit {LIMITS['clock_rms_mhz']:.0f} MHz).",
           tsize=15)
    with open(out_path("policies_trace2_fixed.csv")) as fh:
        fx = next(r for r in csv.DictReader(fh) if r["clock"] == "1050")
    pct = lambda k: minus(f"{float(fx[k]):+.1f}%")
    C = (X0, 0.40, X1, 1.80)
    yb = s.card(C, "A policy change we can measure", "Fixed 1050 MHz against the governor:",
                fc=YELLOW, ec=E_YELLOW, tsize=15)
    for k, who in enumerate(("meas", "sim")):
        y = yb - 0.20 - k * 0.27
        s.text(X0 + 0.2, y, {"meas": "measured", "sim": "simulated"}[who], size=13, inside=C)
        s.text(X0 + 1.35, y, f"{pct(who + '_time_pct')} time, {pct(who + '_energy_pct')} energy", size=13,
               color=INK, inside=C)
    s.save("slide3_sim_validation")


# ------------------------------------------------------------------- slide 4 ----
WORKLOADS = (("decode1024", "decode\n1024 tokens"), ("decode2048", "decode\n2048 tokens"),
             ("decode4096", "decode\n4096 tokens"), ("prefill", "prefill\n7744 tokens"))
TARGET = "least energy, <= 1.05 x B0 latency"
WALL = "fastest under TDP"
DESIGNS = (("B0", "stock governor", BASE, None), ("B1", "same kernel, one chosen clock", COLD, None),
           ("B3*", "cold/hot pinned, one clock*", HOT, None),
           ("B4*", "cold/hot pinned, two V/f domains*", HOT, "////"))


def study():
    """{workload: {(objective, group): (ms, mJ)}} of the control groups that met their target."""
    out = {}
    for wl, _ in WORKLOADS:
        with open(out_path(f"study_{wl}.csv")) as fh:
            out[wl] = {(r["objective"], r["name"]): (float(r["ms"]), float(r["mj"]))
                       for r in csv.DictReader(fh) if r["mj"]}
    return out


def slide_results():
    s = Slide("What It Says")
    st = study()
    mj = lambda wl, grp: st[wl][(WALL if grp == "B0" else TARGET, grp)][1]
    # the second rail alone, as RESULTS_SIM.md section 6 scores it: B4* against B3*
    objs = sorted({o for d in st.values() for o, g in d if o != WALL and g == "B4*"})
    rail_e = [(d[(o, "B4*")][1] / d[(o, "B3*")][1] - 1) * 100 for d in st.values() for o in objs]
    rail_t = [(d[(WALL, "B4*")][0] / d[(WALL, "B3*")][0] - 1) * 100 for d in st.values()]
    clock = [(1 - mj(wl, "B1") / mj(wl, "B0")) * 100 for wl, _ in WORKLOADS]

    s.head(0.45, "Least energy at ≤ 1.05× the stock latency")
    s.lines(0.45, 5.90, "Energy per MoE layer, simulated; each design at its best setting")
    for k, (_, lab, col, hatch) in enumerate(DESIGNS):                # the legend, two rows
        x0, y = (0.55, 3.55)[k % 2], 5.36 - (k // 2) * 0.34
        s.swatch(x0, y, col, hatch)
        s.text(x0 + 0.42, y, lab, size=12.5, color=INK)
    CL, CR = 1.60, 7.60
    ax = s.chart(CL, 2.52, CR, 4.72)
    off, bw = (-0.3, -0.1, 0.1, 0.3), 0.19
    for i, (wl, _) in enumerate(WORKLOADS):
        for (grp, _, col, hatch), o in zip(DESIGNS, off):
            ax.bar(i + o, mj(wl, grp), width=bw, color=col, hatch=hatch, edgecolor="white", lw=0.6, zorder=3)
    ax.set_xlim(-0.5, len(WORKLOADS) - 0.5); ax.set_ylim(0, 1700)
    ax.set_xticks(range(len(WORKLOADS))); ax.set_xticklabels([lab for _, lab in WORKLOADS], fontsize=12.5)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("energy per layer (mJ)", fontsize=13, color=INK)
    # under each group: the change against the stock governor, one line per design
    s.lines(0.45, 1.44, "energy vs.\nstock\ngovernor", size=12, color=GREY_T, anchor="middle")
    for i, (wl, _) in enumerate(WORKLOADS):
        xc = CL + (CR - CL) * (i + 0.5) / len(WORKLOADS)
        for k, (grp, _, col, hatch) in enumerate(DESIGNS[1:]):
            y = 1.72 - k * 0.28
            s.swatch(xc - 0.52, y, col, hatch, w=0.24, h=0.16)
            s.text(xc - 0.20, y, minus(f"{(mj(wl, grp) / mj(wl, 'B0') - 1) * 100:+.1f}%"), size=13, color=INK)
    s.lines(0.45, 0.93, "* a pinned kernel as efficient as vLLM's. Nobody has built it, so these two\n"
            "   designs are a what-if. The pinned kernel that was measured is 17–20%\n"
            "   slower and cannot meet this target.", size=12, color=GREY_T)

    # ---- what to take away
    X0, X1 = 8.05, 13.05
    s.head(X0, "Takeaways")
    s.card((X0, 4.83, X1, 5.90), "Clock choice is the large term",
           minus(f"{min(clock):.0f}–{max(clock):.0f}% less energy for 5% more latency,\n"
                 "with no new kernel and no new hardware."), num=1)
    C = (X0, 3.02, X1, 4.68)
    yb = s.card(C, "A second V/f domain:\nno detectable gain", fc=ORANGE, ec=E_ORANGE, num=2)
    rows = (("energy at equal latency", minus(f"{min(rail_e):+.1f} to {max(rail_e):+.1f} points")),
            ("latency at the 400 W wall", minus(f"{min(rail_t):+.1f}% to {max(rail_t):+.1f}%")))
    for k, (lab, val) in enumerate(rows):
        s.text(X0 + 0.2, yb - 0.12 - k * 0.27, lab, size=13, inside=C)
        s.text(X1 - 0.2, yb - 0.12 - k * 0.27, val, size=13, color=INK, weight="bold", ha="right", inside=C)
    s.text(X0 + 0.2, yb - 0.12 - 2 * 0.27, "Both inside the simulator's 2–4% error.", size=13, inside=C)
    s.card((X0, 1.42, X1, 2.87), "Why so little?",
           "Which experts are cold and how many SMs\nthey get is software. It already removes\n"
           "the slack a slower cold domain would use.\nBoosting the domain left running: 0.0–0.1%.", num=3)
    Bx = s.box(X0, 0.30, X1, 1.27, BLUE, E_BLUE, lw=1.0, r=0.08)
    s.lines(X0 + 0.2, 0.785, "Limits: two V/f domains are simulated, never\nmeasured. One model (Qwen3-30B-A3B), one\n"
            "A100, MoE layers only.", size=12.5, anchor="middle", inside=Bx)
    s.save("slide4_sim_results")


# ------------------------------------------------------------------- slide 5 ----
BOOST = dict(cells=64, n_cold=68, cold_sms=32)   # 4096-token decode; the cold group finishes early
TDP = 400.0


def boost_runs(cal):
    """One layer of the BOOST example four ways, every phase of the layer at or under the
    cap (a limiter faster than a layer): one clock; one clock with the hot domain boosted
    while it runs alone; the best static clock per domain; and the default kernel under
    the stock governor. An instant regulator, except where `act` says otherwise."""
    sh = Shared()
    spec = dict(phase="decode", cells=BOOST["cells"], layers=[0])
    kernel = Kernel("queue", 128, BOOST["n_cold"])

    def run(f, boost=None, act=0.0):
        ctl = BoostOnFinish(list(f), boost) if boost else Fixed(list(f))
        hw = two_domain(BOOST["cold_sms"], actuation_ms=act)
        e = Engine(hw, cal, kernel, build([dict(spec, repeat=3)]), ctl, shared=sh, limiter=False).run()
        L = e.layers[-1]
        seg = [(a - L["t_start"], z - L["t_start"], p, fq) for a, z, p, fq, _ in e.trace
               if a >= L["t_start"] - 1e-9 and z - a > 1e-9]
        return dict(f=tuple(f), boost=boost, L=L, seg=seg, ms=L["ms"], mj=L["mj"], peak=max(x[2] for x in seg))

    clocks = [f for f in Hardware().clocks() if f >= 1050]
    one = [r for r in (run((f, f)) for f in clocks) if r["peak"] <= TDP][-1]
    base = one["f"][0]
    boosted = [r for r in (run((base, base), fb) for fb in clocks if fb > base) if r["peak"] <= TDP][-1]
    static = min((r for r in (run((fc, fh)) for fc in clocks for fh in clocks if fh >= base - 60) if r["peak"] <= TDP),
                 key=lambda r: r["ms"])
    slow = {act: run((base, base), boosted["boost"], act) for act in (0.1, 0.3, 38.0)}
    a = governor_point(cal, dict(spec, rep=0), sh, "qwen")
    return dict(one=one, boosted=boosted, static=static, slow=slow, A=dict(f=a[0][0], ms=a[1], mj=a[2], w=a[3]))


def slide_boost(cal):
    s = Slide("Boost at the Power Cap")
    R = boost_runs(cal)
    one, bst, sta, A = R["one"], R["boosted"], R["static"], R["A"]
    base, fb = one["f"][0], bst["boost"]
    L, seg = bst["L"], bst["seg"]
    T, xmax = L["ms"], one["ms"] * 1.02
    fin = {k: v - L["t_start"] for k, v in L["finish"].items()}

    s.head(0.45, "One MoE layer at the cap, with a boost")
    s.lines(0.45, 5.90, f"Simulated: a 4096-token decode step, {BOOST['n_cold']} cold experts on a {BOOST['cold_sms']}-SM cold domain,\n"
            f"{128 - BOOST['n_cold']} hot experts on {108 - BOOST['cold_sms']} SMs. Pinned kernel with vLLM's per-SM costs (not built).")
    XL, XR = 1.80, 7.70
    g = s.chart(XL, 4.22, XR, 5.02, grid=None)
    ck = s.chart(XL, 3.05, XR, 3.98)
    pw = s.chart(XL, 1.22, XR, 2.80)

    # ---- the boost windows, on every panel: the hot domain runs alone
    wins = [(a, z) for a, z, _, fq in seg if fq[1] > base]
    for ax in (g, ck, pw):
        for a, z in wins:
            ax.axvspan(a, z, color=ORANGE, lw=0, zorder=1)

    # ---- who runs when
    t_act0 = max(fin[("cold", "g1")], fin[("hot", "g1")])
    t_act1 = fin[("epilogue", "act")]
    t_g2 = max(fin[("cold", "g2")], fin[("hot", "g2")])
    start, end = {"g1": 0.0, "g2": t_act1}, {"g1": t_act0, "g2": t_g2}
    for name, y, col in (("cold", 2, COLD), ("hot", 1, HOT)):
        for st in ("g1", "g2"):
            a, b = start[st], fin[(name, st)]
            g.barh(y, b - a, left=a, height=0.66, color=col, edgecolor="white", lw=1.0, zorder=3)
            s.text((a + b) / 2, y, "GEMM 1" if st == "g1" else "GEMM 2", size=13, color="white",
                   weight="bold", ha="center", ax=g)
            if end[st] - b > 0.004 * T:
                g.barh(y, end[st] - b, left=b, height=0.66, facecolor="white", edgecolor=col,
                       hatch="////", lw=1.0, zorder=3)
        for a, b in ((t_act0, t_act1), (t_g2, T)):
            g.barh(y, b - a, left=a, height=0.66, color=EPI, edgecolor="white", lw=1.0, zorder=3)
    g.set_yticks([2, 1]); g.set_yticklabels(["cold, 32 SMs", "hot, 76 SMs"], fontsize=13)
    g.set_ylim(0.5, 2.5); g.set_xlim(0, xmax)
    g.tick_params(axis="x", labelbottom=False, length=0); g.tick_params(axis="y", length=0)
    for sp in ("bottom", "left"):
        g.spines[sp].set_visible(False)

    # ---- the clock of each domain
    def steps(i):
        xs, ys = [], []
        for a, z, _, fq in seg:
            xs += [a, z]; ys += [fq[i], fq[i]]
        return xs, ys
    ck.plot(*steps(0), color=COLD, lw=4.5, solid_capstyle="butt", zorder=3)
    ck.plot(*steps(1), color=HOT, lw=2.0, zorder=4)
    lo, hi = base - 45, fb + 45
    ck.set_xlim(0, xmax); ck.set_ylim(lo, hi); ck.set_yticks([base, fb])
    ck.tick_params(labelbottom=False)
    ck.set_ylabel("clock\n(MHz)", fontsize=13, color=INK)
    w0 = wins[0]
    xm = (w0[0] + w0[1]) / 2                                          # inside the first boost window
    s.text(xm, fb + 24, "hot", size=12.5, color=INK, ax=ck, ha="center", va="center")
    s.text(xm, base - 24, "cold", size=12.5, color=INK, ax=ck, ha="center", va="center")
    s.text(0.05, base + 26, "cold and hot domain", size=12.5, color=INK, ax=ck, va="center")

    # ---- board power, with and without the boost
    xs, ys = [], []
    for a, z, p, _ in seg:
        xs += [a, z]; ys += [p, p]
    pw.fill_between(xs, ys, color=BASE, lw=0, zorder=2)
    pw.plot(xs, ys, color=INK, lw=1.8, zorder=4)
    x1, y1 = [], []
    for a, z, p, _ in one["seg"]:
        x1 += [a, z]; y1 += [p, p]
    pw.plot(x1, y1, color=INK, lw=1.3, ls=(0, (4, 2.5)), zorder=5)
    pw.axhline(TDP, color=RED, lw=1.2, zorder=3)
    pw.set_xlim(0, xmax); pw.set_ylim(0, 470); pw.set_yticks([0, 100, 200, 300, 400])
    pw.set_xlabel("time since the layer started (ms)", fontsize=13, color=INK)
    pw.set_ylabel("board power (W)", fontsize=13, color=INK)
    s.text(0.05, 437, "400 W cap", size=12, color=INK, ax=pw)
    dip = one["seg"][1]                                               # the hot domain alone, not boosted
    s.text((dip[0] + dip[1]) / 2, dip[2] - 62, "without\nthe boost", size=12, color=INK, ax=pw, ha="center", va="center")
    s.text(0.05, 150, f"both at {base} MHz:\n{seg[0][2]:.0f} W, at the cap", size=12.5, color=INK, ax=pw, va="center")
    ky = 5.24
    s.swatch(XL, ky, EPI)
    s.text(XL + 0.40, ky, "activation, top-k sum", size=12)
    s.ax.add_patch(Rectangle((XL + 2.42, ky - 0.09), 0.32, 0.18, fc="white", ec=LINE, hatch="////", lw=1.0, zorder=3))
    s.text(XL + 2.82, ky, "waiting", size=12)
    s.swatch(XL + 3.75, ky, ORANGE)
    s.text(XL + 4.15, ky, "boost window", size=12)

    # ---- what happens, what it buys, what it takes
    X0, X1 = 8.15, 13.05
    s.card((X0, 4.82, X1, 6.32), "What the controller does",
           f"Both domains run at {base} MHz, the highest\nclock that keeps the board under 400 W.\n"
           f"When the cold experts finish, the hot domain\ngoes to {fb} MHz until the barrier.", fc=ORANGE, ec=E_ORANGE)
    C = s.box(X0, 2.12, X1, 4.70, "white", E_GREY)
    s.text(X1 - 1.22, 4.50, "ms per layer", size=11.5, color=GREY_T, ha="right", inside=C)
    s.text(X1 - 0.20, 4.50, "mJ", size=11.5, color=GREY_T, ha="right", inside=C)
    pct = lambda v, ref: minus(f"{(v / ref - 1) * 100:+.1f}%")
    rows = [("One clock", f"{base} / {base} MHz", one["ms"], one["mj"], False),
            ("One clock + boost", f"hot to {fb} MHz when cold is done", bst["ms"], bst["mj"], True),
            ("A clock per domain, no boost", f"{sta['f'][0]} / {sta['f'][1]} MHz", sta["ms"], sta["mj"], False),
            ("Default kernel, stock governor", "not pinned, one clock", A["ms"], A["mj"], False)]
    for k, (lab, small, ms, mj, bold) in enumerate(rows):
        y = 4.18 - k * 0.52
        s.text(X0 + 0.2, y, lab, size=13, color=INK, weight="bold" if bold else "normal", inside=C)
        s.text(X0 + 0.2, y - 0.23, small, size=11.5, color=GREY_T, inside=C)
        s.text(X1 - 1.22, y, f"{ms:.3f}", size=13, color=INK, ha="right", weight="bold" if bold else "normal", inside=C)
        s.text(X1 - 0.20, y, f"{mj:.0f}", size=13, color=INK, ha="right", weight="bold" if bold else "normal", inside=C)
        if k:
            s.text(X1 - 1.22, y - 0.23, pct(ms, one["ms"]), size=11.5, color=GREY_T, ha="right", inside=C)
            s.text(X1 - 0.20, y - 0.23, pct(mj, one["mj"]), size=11.5, color=GREY_T, ha="right", inside=C)
    gain = lambda r: (r["ms"] / one["ms"] - 1) * 100
    B = s.box(X0, 0.30, X1, 2.00, BLUE, E_BLUE, lw=1.0, r=0.08)
    s.lines(X0 + 0.2, 1.15, f"The gain needs a fast regulator: {minus(f'{gain(bst):.1f}')}% if it\n"
            f"switches at once, {minus(f'{gain(R['slow'][0.3]):.1f}')}% at 0.3 ms, none through\n"
            "NVML (38 ms). And the split is unbalanced on\npurpose: with a balanced one both groups\n"
            "finish together and there is nothing to boost.", size=12.5, anchor="middle", inside=B)
    s.save("slide5_sim_boost")
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, default=None, help="draw one slide (1-5)")
    a = ap.parse_args()
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
                         "pdf.fonttype": 42, "ps.fonttype": 42, "hatch.linewidth": 1.0})
    cal = Calib.load()
    slides = {1: slide_why, 2: lambda: slide_how(cal), 3: slide_validation, 4: slide_results,
              5: lambda: slide_boost(cal)}
    for k, fn in slides.items():
        if a.only in (None, k):
            fn()


if __name__ == "__main__":
    main()
