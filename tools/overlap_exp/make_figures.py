#!/usr/bin/env python3
"""NeurIPS-format figures for the two-DVFS-domain overlap study.

Every figure is built at its final printed size (5.5 in = NeurIPS \\textwidth) so it can
go into LaTeX with \\includegraphics[width=\\textwidth]{...} and no scaling -- scaling is
what makes figure type stop matching body type.

  fig1_mechanism   why two clocks save energy. Power vs time; rectangle AREA is energy,
                   drawn to scale, so the reader can compare areas directly rather than
                   being told a number.
  fig2_objectives  the same workload under min-energy / min-latency / min-EDP, with the
                   energy split into its three terms.
  fig3_rule        when it is worth anything: saving vs t_comm/t_gemm over 48 workloads.

  python3 make_figures.py
"""
import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

import figstyle as F

F.use()
HERE = os.path.dirname(os.path.abspath(__file__))

# --- fig 1: the mechanism -------------------------------------------------------
# GEMM 2048x4096x4096 + 100 MiB all-reduce, world=4, min energy at T = T_min.
# Chosen to be as hard to argue with as this hardware allows:
#   * the GEMM shape is one of the 12 actually measured, the message size one of the 8
#   * 900 and 1200 MHz are both calibration grid points -- nothing is interpolated, and
#     nothing sits in the 900-1200 gap or at the extrapolated 1410
#   * peak power is 303 W and 226 W against a 400 W board cap, so the "wouldn't this
#     throttle?" objection does not arise for either arm
#   * both arms finish at exactly T = 0.825 ms
# An earlier version of this figure used 4096x4096x4096 + 256 MiB at 1410 MHz and showed
# -26.4%. That case is not wrong -- its iteration mean is 282 W and the mean is the
# validated throttle criterion -- but it PEAKS at 486 W, 86 W over the cap, which invites
# an objection the figure cannot answer on its own. The honest trade is a smaller,
# unimpeachable number: every large win requires arm B to be pushed to a clock where it
# peaks over the cap, because "the non-critical kernel is forced to run too fast" is
# simultaneously the mechanism of the win and the cause of the high peak.
# (label, f_gemm, f_comm, t_gemm, t_comm, T, P_static, P_gemm, P_comm)
FIG1 = {
    "1 clock":  (1200, 1200, 0.504, 0.825, 0.825, 69.7, 172.3, 60.6),
    "2 clocks": (900, 1200, 0.670, 0.825, 0.825, 63.8, 101.9, 60.6),
}
CAP_W = 400.0

# --- fig 2: the three objectives ------------------------------------------------
# GEMM 4096x4096x4096 + 256 MiB, kept because comparing objectives needs a case where
# they actually disagree. Peaks are annotated so the reader can see which exceed the cap.
CASES = {
    ("E", "1 clock"):  (900, 900, 1.14, 2.58, 2.58, 61.4, 115.8, 48.3),
    ("E", "2 clocks"): (510, 1200, 2.01, 2.00, 2.01, 61.2, 67.4, 60.6),
    ("T", "1 clock"):  (1410, 1410, 0.73, 1.91, 1.91, 85.4, 329.8, 71.1),
    ("T", "2 clocks"): (705, 1410, 1.45, 1.91, 1.91, 67.1, 91.8, 71.1),
    ("D", "1 clock"):  (1200, 1200, 0.86, 2.00, 2.00, 69.7, 197.0, 60.6),
    ("D", "2 clocks"): (705, 1410, 1.45, 1.91, 1.91, 67.1, 91.8, 71.1),
}


def energies(k, tab=None):
    fg, fc, tg, tc, T, ps, pg, pc = (tab or CASES)[k]
    return ps * T, pg * tg, pc * tc


# ---------------------------------------------------------------- fig 1
def fig_mechanism():
    fig, axes = plt.subplots(1, 2, figsize=(F.TEXT_W, 2.15), sharey=True,
                             gridspec_kw=dict(wspace=0.08))
    for ax, key, title in zip(
            axes, ["1 clock", "2 clocks"],
            ["(a) one clock: both at 1200 MHz",
             "(b) two clocks: GEMM 900, comm 1200 MHz"]):
        fg, fc, tg, tc, T, ps, pg, pc = FIG1[key]
        e_s, e_g, e_c = energies(key, FIG1)
        ax.axhline(CAP_W, color=F.MUTED, lw=0.7, ls=(0, (4, 2)), zorder=2)
        for x0, w, y0, h, col, hat in (
                (0, T, 0, ps, F.BASE, F.HATCH["base"]),
                (0, T, ps, pc, F.COMM, F.HATCH["comm"]),
                (0, tg, ps + pc, pg, F.GEMM, F.HATCH["gemm"])):
            ax.add_patch(Rectangle((x0, y0), w, h, facecolor=col, edgecolor="white",
                                   linewidth=0.7, hatch=hat, zorder=3))
        ax.text(tg / 2, ps + pc + pg / 2, f"{e_g:.0f}", ha="center", va="center",
                color="white", fontsize=7.5, zorder=5)
        ax.text(T * 0.76, ps + pc / 2, f"{e_c:.0f}", ha="center", va="center",
                color="white", fontsize=7.5, zorder=5)
        ax.text(T * 0.76, ps / 2, f"{e_s:.0f}", ha="center", va="center",
                color=F.INK, fontsize=7.5, zorder=5)
        ax.annotate("", xy=(tg, 340), xytext=(T, 340),
                    arrowprops=dict(arrowstyle="<->", lw=0.6, color=F.GEMM,
                                    shrinkA=0, shrinkB=0))
        ax.text((tg + T) / 2, 352, f"GEMM idle {T-tg:.2f} ms", ha="center",
                va="bottom", fontsize=6.5, color=F.GEMM)
        ax.axvline(T, color=F.INK, lw=0.6, zorder=4)
        ax.text(T - 0.015, 305, f"$T$ = {T:.3f} ms", ha="right", va="top", fontsize=7)
        ax.text(0.03, 0.97, f"total {e_s+e_g+e_c:.0f} mJ", transform=ax.transAxes,
                fontsize=8, va="top", ha="left", fontweight="bold")
        ax.text(0.03, 0.845, f"peak {ps+pg+pc:.0f} W", transform=ax.transAxes,
                fontsize=6.8, va="top", ha="left", color=F.MUTED)
        ax.set_title(title, fontsize=7.5, pad=4)
        ax.set_xlim(0, 0.95); ax.set_ylim(0, 470)
        ax.set_xlabel("time (ms)")
        F.despine(ax, grid_axis="y")
        ax.set_yticks([0, 100, 200, 300, 400])
    axes[0].set_ylabel("power (W)")
    axes[1].text(0.93, CAP_W + 9, "400 W board cap", fontsize=6.5, color=F.MUTED,
                 ha="right", va="bottom")
    h = [Rectangle((0, 0), 1, 1, facecolor=c, hatch=k, edgecolor="white", linewidth=0.5)
         for c, k in ((F.GEMM, F.HATCH["gemm"]), (F.COMM, F.HATCH["comm"]),
                      (F.BASE, F.HATCH["base"]))]
    fig.legend(h, ["GEMM", "all-reduce", "static floor"], loc="lower center",
               ncol=3, bbox_to_anchor=(0.5, -0.18))
    return F.save(fig, "fig1_mechanism")


# ---------------------------------------------------------------- fig 2
def fig_objectives():
    fig, ax = plt.subplots(figsize=(F.TEXT_W, 2.15))
    groups = [("E", "min energy"), ("T", "min latency"), ("D", "min EDP")]
    tots = []
    for gi, (k, _) in enumerate(groups):
        for ai, arm in enumerate(("1 clock", "2 clocks")):
            x = gi * 2.5 + ai * 0.90
            e_s, e_g, e_c = energies((k, arm))
            bot = 0
            for v, col, hat in ((e_s, F.BASE, F.HATCH["base"]),
                                (e_g, F.GEMM, F.HATCH["gemm"]),
                                (e_c, F.COMM, F.HATCH["comm"])):
                ax.bar(x, v, 0.78, bottom=bot, facecolor=col, edgecolor="white",
                       linewidth=0.7, hatch=hat, zorder=3)
                bot += v
            ax.text(x, bot + 10, f"{bot:.0f}", ha="center", fontsize=7.5, zorder=4)
            ax.text(x, -30, arm, ha="center", va="top", fontsize=7)
            ax.text(x, -60, f"$T$ {CASES[(k, arm)][4]:.2f} ms", ha="center", va="top",
                    fontsize=6.5, color=F.MUTED)
            tots.append(bot)
    for gi, (k, name) in enumerate(groups):
        cx = gi * 2.5 + 0.45
        d = (tots[gi * 2 + 1] - tots[gi * 2]) / tots[gi * 2] * 100
        ax.text(cx, -100, name, ha="center", va="top", fontsize=7.5)
        ax.text(cx, -130, f"{d:+.1f}%", ha="center", va="top", fontsize=8,
                color=F.GEMM if d < -15 else F.MUTED,
                fontweight="bold" if d < -15 else "normal")
    ax.set_xlim(-0.75, 6.15); ax.set_ylim(0, 700)
    ax.set_ylabel("energy per iteration (mJ)")
    ax.set_xticks([]); ax.set_yticks([0, 100, 200, 300, 400, 500, 600])
    F.despine(ax, grid_axis="y")
    h = [Rectangle((0, 0), 1, 1, facecolor=c, hatch=k, edgecolor="white", linewidth=0.5)
         for c, k in ((F.GEMM, F.HATCH["gemm"]), (F.COMM, F.HATCH["comm"]),
                      (F.BASE, F.HATCH["base"]))]
    ax.legend(h, ["GEMM", "all-reduce", "static floor"], loc="upper left",
              ncol=3, bbox_to_anchor=(0.0, 1.02))
    return F.save(fig, "fig2_objectives")


# ---------------------------------------------------------------- fig 3
def fig_rule():
    rows = list(csv.DictReader(open(os.path.join(HERE, "data", "objectives.csv"))))
    r = np.array([float(x["tc_over_tg"]) for x in rows])
    cols = {"E": "CvB_E_mJ_pct", "T": "CvB_T_mJ_pct", "D": "CvB_EDP_mJ_pct"}
    fig, axes = plt.subplots(1, 3, figsize=(F.TEXT_W, 1.8), sharey=True,
                             gridspec_kw=dict(wspace=0.10))
    for ax, (k, c), name in zip(axes, cols.items(),
                                ["min energy", "min latency", "min EDP"]):
        y = np.array([float(x[c]) for x in rows])
        # threshold chosen by search, not by eye: at 1.0 the rule has 0-1 false
        # positives across the three objectives; at 1.5 it misses 2-4 real wins
        ax.axvline(1.0, color=F.MUTED, lw=0.6, ls=(0, (3, 2)), zorder=2)
        ax.scatter(r, y, s=11, facecolor=F.GEMM, edgecolor="white", linewidth=0.4,
                   zorder=4, clip_on=False)
        ax.set_xscale("log")
        ax.set_xlim(0.006, 14); ax.set_ylim(-28, 2)
        ax.set_title(name, fontsize=7.5, pad=3)
        ax.set_xlabel(r"$t_{\mathrm{comm}}\,/\,t_{\mathrm{gemm}}$")
        F.despine(ax, grid_axis="y")
        ax.set_yticks([0, -5, -10, -15, -20, -25])
        miss = int(((r < 1.0) & (y < -1)).sum())      # wins the rule would miss
        fp = int(((r >= 1.0) & (y >= -1)).sum())       # flagged but no win
        ax.text(0.035, 0.05, f"missed {miss}, false {fp}", fontsize=6.5,
                color=F.MUTED, transform=ax.transAxes, ha="left")
    axes[0].set_ylabel("energy saved (%)")
    return F.save(fig, "fig3_rule")


# ---------------------------------------------------------------- fig 4
def fig_frontier():
    """The energy-latency frontier -- the curve the three objectives are points on.

    ANSWERS TWO QUESTIONS AT ONCE.

    Do the three objectives each need to exist? No. They are three points on one curve.
    min E subject to T <= budget traces the whole thing: budget -> infinity gives the
    min-energy point, budget = T_min gives min-latency, and EDP lands between them with
    no principled reason for landing where it does.

    Why does min latency report a LARGER percentage saving than min energy, when min
    energy is by definition the objective that minimises energy? Because the percentage
    is C against B, and under a tight deadline B collapses while C barely moves. Read the
    curves rather than the percentages: the two-clock frontier is nearly FLAT -- it is
    already near its best energy even at T_min -- while the one-clock frontier rises
    steeply to the left, because with a single clock the only way to meet a tight
    deadline is to run BOTH kernels fast.

    The vertical gaps are measured at a COMMON latency budget, not between each arm's own
    optimum, so they answer "at this deadline, what does the second domain buy" rather
    than comparing two different operating points.
    """
    G = json.load(open(os.path.join(HERE, "data", "grid12.json")))
    grid, nk, sk, gb = G["4096_4096_4096"], 4096, 1, 256 * 2 ** 20 / 1e9
    import objectives as J
    XHI = 3.0

    def front(arm):
        c = sorted(J.feasible(J.candidates(grid, nk, sk, gb, arm)), key=lambda z: z[1])
        out, best = [], float("inf")
        for E, T, *_ in c:
            if E < best - 1e-9:
                best = E; out.append((T, E))
        out.append((XHI, out[-1][1]))     # a looser budget never costs more energy
        return np.array(out)

    def at(fr, t):
        v = fr[fr[:, 0] <= t + 1e-9]
        return v[-1, 1] if len(v) else np.nan

    B, C = front("hard1"), front("hard2")
    fig, ax = plt.subplots(figsize=(F.TEXT_W * 0.66, 2.3))
    ax.step(B[:, 0], B[:, 1], where="post", color=F.MUTED, lw=1.3, ls=(0, (4, 2)),
            label="one clock", zorder=3)
    ax.step(C[:, 0], C[:, 1], where="post", color=F.GEMM, lw=1.5,
            label="two clocks", zorder=4)
    # The one-clock arm has a HARD FLOOR once the instantaneous peak is capped: to beat
    # 1.961 ms it would have to run both kernels at 1410 MHz, which peaks at 486 W. Two
    # clocks reach 1.909 ms because only the collective needs the high clock. Marking
    # that region matters -- an earlier version just dropped the min-latency annotation
    # when at(B, t) came back empty, silently hiding the result.
    XLO = 1.885
    # set the limits BEFORE drawing the span and the callout. axvspan mutates xlim, so
    # reading get_xlim() afterwards gave an anchor 0.006 ms outside the final axes, and
    # matplotlib's default annotation_clip then discarded the whole annotation silently.
    ax.set_xlim(XLO, XHI); ax.set_ylim(340, 610)
    tb_min, tc_min = B[0, 0], C[0, 0]
    if tb_min > tc_min + 1e-6:
        ax.axvspan(XLO, tb_min, color=F.GRID, alpha=0.55, lw=0, zorder=0)
        ax.set_xlim(XLO, XHI)
        ax.annotate(f"one clock cannot reach\nthis deadline: its floor is\n"
                    f"{tb_min:.3f} ms, two clocks\nreach {tc_min:.3f} ms",
                    xy=((XLO + tb_min) / 2, 520), xytext=(2.14, 590),
                    fontsize=6.5, color=F.MUTED, ha="left", va="top", linespacing=1.4,
                    annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", lw=0.5, color=F.MUTED,
                                    shrinkA=2, shrinkB=3))
    for t, name, anchor, off, ha, va in (
            (2.001, "min EDP", "mid", (7, 0), "left", "center"),
            (2.579, "min energy", "mid", (7, 0), "left", "center")):
        eb, ec = at(B, t), at(C, t)
        ax.plot([t, t], [ec, eb], color=F.INK, lw=0.6, zorder=5)
        ax.plot([t, t], [ec, eb], "o", ms=3.4, color=F.INK, zorder=6)
        ax.annotate(f"{name}\n{(ec - eb) / eb * 100:+.1f}%",
                    (t, eb if anchor == "top" else (eb + ec) / 2),
                    textcoords="offset points", xytext=off, fontsize=6.8,
                    color=F.INK, ha=ha, va=va, linespacing=1.35)
    ax.set_xlabel("latency budget $T$ (ms)")
    ax.set_ylabel("energy per iteration (mJ)")
    F.despine(ax, grid_axis="y")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.03))
    return F.save(fig, "fig4_frontier")


# ---------------------------------------------------------------- fig 5
def fig_paired():
    """Every workload, every objective: where two clocks land relative to one clock.

    Each point is one workload, plotted as the RATIO of the two-clock optimum to the
    one-clock optimum. That normalisation is what makes 48 workloads comparable at all --
    they span 0.1 to 24 ms and 50 to 6000 mJ, so raw axes would just show the spread of
    problem sizes and hide the effect entirely.

        (1, 1)          the two arms tie -- the second domain bought nothing
        below y = 1     less energy      left of x = 1   less latency
        lower-left      better on BOTH axes, i.e. the two-clock arm dominates

    Colour is t_comm/t_gemm on a log scale, so the reader can see directly that the
    points which move are the comm-bound ones. Single hue, light to dark, because it
    encodes a magnitude; validated for step separation (min OKLCH dL 0.079 against a
    0.06 floor) and for contrast at both ends.
    """
    G = json.load(open(os.path.join(HERE, "data", "grid12.json")))
    import objectives as J
    import optimise_overlap as O

    # t_c/t_g is encoded as TWO CATEGORIES, not a continuous ramp. What decides whether
    # a workload gains anything is which side of 1.0 it sits on -- is the collective the
    # critical path or not -- and a sequential ramp rendered at 6 pt markers cannot be
    # read to that precision. The two hues are the ones already validated for this
    # project (OKLab CVD delta-E 26.8) and the mapping is semantically consistent with
    # figures 1 and 2: amber where the GEMM dominates, indigo where the collective does.
    rows = []
    for nk in (4096, 8192, 16384):
        for m in (1024, 2048, 4096, 8192):
            grid = G[f"{m}_{nk}_{nk}"]; sk = max(1, grid // (m * nk // 32768))
            for mib in (8, 32, 128, 256):
                gb = mib * 2 ** 20 / 1e9
                cb = J.candidates(grid, nk, sk, gb, "hard1")
                cc = J.candidates(grid, nk, sk, gb, "hard2")
                r = dict(ratio=O.t_comm(900, 16, gb) / O.t_gemm(900, 108, grid, nk, sk))
                for o in ("E", "T", "EDP"):
                    B, C = J.pick(cb, o), J.pick(cc, o)
                    r[o] = (C[1] / B[1], C[0] / B[0])
                rows.append(r)

    fig, axes = plt.subplots(1, 3, figsize=(F.TEXT_W, 2.2), sharey=True,
                             gridspec_kw=dict(wspace=0.10))
    for ax, o, name in zip(axes, ("E", "T", "EDP"),
                           ("min energy", "min latency", "min EDP")):
        ax.axhline(1.0, color=F.MUTED, lw=0.6, ls=(0, (3, 2)), zorder=2)
        ax.axvline(1.0, color=F.MUTED, lw=0.6, ls=(0, (3, 2)), zorder=2)
        for lo, hi, col, mk, lab in ((0, 1.0, F.GEMM, "o", "GEMM-bound"),
                                     (1.0, 1e9, F.COMM, "D", "comm-bound")):
            v = [r for r in rows if lo <= r["ratio"] < hi]
            ax.scatter([r[o][0] for r in v], [r[o][1] for r in v], marker=mk,
                       s=26 if mk == "o" else 22, facecolor=col, edgecolor="white",
                       linewidth=0.5, alpha=0.9, zorder=4, label=lab)
        ax.set_title(name, fontsize=7.5, pad=3)
        ax.set_xlabel(r"latency  $T_{2}/T_{1}$")
        ax.set_xlim(0.735, 1.06); ax.set_ylim(0.70, 1.06)
        ax.set_xticks([0.8, 0.9, 1.0])
        F.despine(ax, grid_axis="")
        gain = sum(1 for r in rows if r[o][1] < 0.99)
        ax.text(0.04, 0.10, f"{gain} of {len(rows)} gain\n"
                            f"{len(rows) - gain} sit at (1, 1)",
                transform=ax.transAxes, fontsize=6.8, color=F.MUTED,
                va="bottom", linespacing=1.4)
    axes[0].set_ylabel(r"energy  $E_{2}/E_{1}$")
    axes[0].set_yticks([0.75, 0.85, 0.95, 1.0])
    axes[1].legend(loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.42),
                   handletextpad=0.3)
    return F.save(fig, "fig5_paired")


# ---------------------------------------------------------------- fig 6
def fig_knee():
    """The voltage curve with the 900-1200 MHz gap filled in.

    kappa = a/f is C*V^2. On the original six-clock grid it is flat to 0.5% CV from 300
    to 900 MHz and then +33% by 1200, so the knee sat inside a span with no calibration
    points -- the span every throttled overlap row lands in, and the one the energy
    optimum sits in. Four new clocks resolve it.

    The knee location is robust to the 3-4 W disagreement between the two static-power
    campaigns, because kappa takes P_loaded and P_idle from the SAME campaign and a
    constant offset cancels.
    """
    import json
    GR = json.load(open(os.path.join(HERE, "data", "grid12.json")))
    SRC = os.path.join(HERE, "..", "gemm_dcfs_char", "data")

    def lsq(xs, ys):
        n = len(xs); mx, my = sum(xs) / n, sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        return (sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx) if sxx else 0.0

    A, src = {}, {}
    for x in csv.DictReader(open(os.path.join(SRC, "gemm_dynamic_energy.csv"))):
        if x.get("clock_held") != "True" or x.get("status") != "ok":
            continue
        g = GR.get(f"{x['m']}_{x['n']}_{x['k']}")
        if g is None:
            continue
        A.setdefault(int(x["freq_req_mhz"]), []).append(
            (g / -(-g // int(x["sm_avail"])), float(x["p_dynamic_w"])))
    for f in A:
        src[f] = "orig"
    idle = {int(x["clock"]): float(x["idle_w"])
            for x in csv.DictReader(open(os.path.join(HERE, "data", "voltage_knee.csv")))}
    N = {}
    for x in csv.DictReader(open(os.path.join(SRC, "gemm_squat_knee.csv"))):
        if x["clock_held"] != "True" or x["status"] != "ok":
            continue
        f = int(x["freq_req_mhz"]); g = GR.get(f"{x['m']}_{x['n']}_{x['k']}")
        if g is None or f not in idle:
            continue
        N.setdefault(f, []).append(
            (g / -(-g // int(x["sm_avail"])), float(x["power_w"]) - idle[f]))
    for f, v in N.items():
        A[f] = v; src[f] = "new"
    K = {f: lsq([p[0] for p in v], [p[1] for p in v]) / f
         for f, v in A.items() if len({p[0] for p in v}) >= 3}

    fs = sorted(K)
    floor = sum(K[f] for f in fs if f <= 900) / len([f for f in fs if f <= 900])
    fig, ax = plt.subplots(figsize=(F.TEXT_W * 0.66, 2.2))
    ax.axhline(floor * 1e3, color=F.MUTED, lw=0.7, ls=(0, (4, 2)), zorder=2)
    ax.axvspan(1020, 1080, color=F.GRID, alpha=0.7, lw=0, zorder=0)
    for tag, mk, lab in (("orig", "o", "original 6-clock grid"),
                         ("new", "D", "added tonight")):
        v = [f for f in fs if src[f] == tag]
        ax.plot(v, [K[f] * 1e3 for f in v], mk, ms=5, color=F.GEMM if tag == "new" else F.MUTED,
                mec="white", mew=0.6, ls="none", label=lab, zorder=4)
    ax.plot(fs, [K[f] * 1e3 for f in fs], "-", lw=1.0, color=F.MUTED, alpha=0.5, zorder=3)
    ax.annotate("voltage knee\n1020-1080 MHz", (1050, floor * 1e3 * 1.05),
                xytext=(760, 2.55), fontsize=6.8, color=F.INK, ha="left",
                linespacing=1.35,
                arrowprops=dict(arrowstyle="-", lw=0.5, color=F.MUTED, shrinkA=2, shrinkB=3))
    ax.text(310, floor * 1e3 * 1.03, "floor, flat to 0.5% CV", fontsize=6.5, color=F.MUTED)
    ax.set_xlabel("SM clock (MHz)")
    ax.set_ylabel(r"$\kappa = a/f$  ($\times 10^{-3}$ W SM$^{-1}$ MHz$^{-1}$)")
    ax.set_xlim(250, 1470); ax.set_ylim(1.5, 3.4)
    F.despine(ax, grid_axis="y")
    ax.legend(loc="upper left")
    return F.save(fig, "fig6_knee")


def _size_band(ax, title):
    """Group label under the CTA ticks. Putting it ABOVE the matrix collided with the
    title -- both wanted the same strip of space and matplotlib drew them on top of
    each other."""
    n = len(ax.get_xticks())
    for i, m in enumerate((128, 256, 512)):
        ax.text(i * 4 + 1.5, n / 2 * 0 + ax.get_ylim()[0] + 0.62, f"{m} MiB",
                ha="center", va="top", fontsize=7.5, clip_on=False)
        if i:
            ax.axvline(i * 4 - 0.5, color=F.INK, lw=0.8, zorder=5)
    ax.set_xlabel("NCCL CTAs, grouped by all-reduce message size", labelpad=16)
    ax.set_ylabel("GEMM  M x N=K")
    ax.set_title(title, fontsize=7.5, pad=6)


# ---------------------------------------------------------------- fig 7
def fig_commbound():
    """The ladder across all 72 comm-bound configurations, against the REAL default.

    Both panels measure against what the GPU does when nobody intervenes: its own
    governor, which picked 1410 MHz on 59 of these 72 and stayed above 1170 on the rest.
    That is the only baseline a saving can honestly be quoted against, and every earlier
    version of this figure used a locked clock instead -- a setting no deployment uses.

    (a) is FREE: choose a better single clock. No new hardware, one nvidia-smi call.
    (b) needs new silicon. Reading them together says how much of the opportunity is
    already available and how much is a hardware argument.

    Every cell is addressable: a named GEMM shape by a named (message size, CTA count).
    Columns run 32 -> 4 CTAs within each size, so the collective slows to the right and
    t_comm/t_gemm rises; rows are ordered by GEMM size so it rises downward.

    Panel (a) is measurement throughout -- default and locked runs are both real
    overlapped executions. Panel (b) inherits the composition for its two-clock arm.
    """
    rows = list(csv.DictReader(open(os.path.join(HERE, "data", "ladder.csv"))))
    shapes = sorted({(int(x["m"]), int(x["nk"])) for x in rows},
                    key=lambda s: s[0] * s[1])
    cols = [(m, c) for m in (128, 256, 512) for c in (32, 16, 8, 4)]
    fig, axes = plt.subplots(2, 1, figsize=(F.TEXT_W, 4.5),
                             gridspec_kw=dict(hspace=0.62))
    for ax, key, title in (
            (axes[0], "lock_vs_auto",
             "(a) FREE: pick a better single clock, vs what the GPU chose (%)"),
            (axes[1], "two_vs_auto",
             "(b) that plus a second clock domain, vs the same default (%)")):
        look = {(int(x["m"]), int(x["nk"]), int(x["mib"]), int(x["ctas"])): float(x[key])
                for x in rows}
        grid = [[look.get((sh[0], sh[1], m, c), float("nan")) for (m, c) in cols]
                for sh in shapes]
        im = F.matrix(ax, [f"{p}x{q}" for p, q in shapes],
                      [f"{c}" for (m, c) in cols], grid, fmt="{:+.0f}",
                      vmin=-40, vmax=40, fontsize=6.0)
        _size_band(ax, title)
        cb = fig.colorbar(im, ax=ax, fraction=0.020, pad=0.012)
        cb.ax.tick_params(labelsize=6.5, width=0.5, length=2)
        cb.outline.set_linewidth(0.5)
    return F.save(fig, "fig7_commbound")


# ---------------------------------------------------------------- fig 9
def fig_case():
    """The ladder, on one workload: what you get today, what one clock can do, what two
    could do. Panels (a) and (b) are both REAL overlapped runs.

    WHY THREE PANELS. Comparing two clocks against a locked one-clock baseline answers a
    narrow question and hides the bigger one, because nobody locks clocks in production.
    Panel (a) is the GPU left alone -- its own governor picks 1410 MHz and holds it, and
    that is the only baseline a saving can honestly be quoted against. Against it, simply
    choosing a better single clock is worth 18.1% and needs no new hardware; the second
    domain adds 5.6 points on top.

    WHAT "REAL OVERLAPPED RUN" MEANS. Two fresh non-default streams, the collective at
    priority -3 and issued first so its CTAs are resident before the GEMM floods the
    chip, both gated on one common event. Nothing is partitioned by hand: the hardware
    block scheduler decides how the SMs are shared, exactly as in demo_overlap.py.

    MEASURED in (a) and (b): the total energy, both kernels' spans inside the overlapped
    iteration (per-stream CUDA events on a common timeline), the iteration time, the
    clock the hardware actually ran at, and the static floor. NOT MEASURED: how the
    non-static energy divides between the two kernels, because NVML reports one number
    for the board -- that split is apportioned by the two solo dynamic energies, so the
    rectangles sum to the measured total exactly rather than approximately.

    THE SPANS ARE SCALED ONTO THE LOOP TIMELINE. Spans come from 60 single-shot reps
    each preceded by a device sync; the power and the iteration time come from a
    sustained loop. Locked, those two phases agree to 1.2%. UNLOCKED they differ by 11%,
    because the governor behaves differently under a stop-start pattern than a sustained
    one -- which is exactly what locking removes. Mixing them let the collective's box
    run 0.08 ms past the iteration containing it. Each span is therefore scaled by
    T_loop / T_events, preserving the fraction of the iteration each kernel was measured
    to occupy, which is what the figure communicates.

    Panel (c) cannot be measured -- A100 has one clock domain -- but its GEMM width uses
    the slowdown measured in (b) rather than the wave model, which under-predicts it.
    """
    d = json.load(open(os.path.join(HERE, "data", "fig9_case.json")))
    P, meta = [d["a"], d["b"], d["c"]], d["meta"]
    E0 = d["a"]["E_meas"]
    titles = [f"(a) no lock: the GPU picks {d['a']['fg']} MHz",
              f"(b) best single clock, {d['b']['fg']} MHz",
              f"(c) two clocks, {d['c']['fg']} / {d['c']['fc']} MHz"]
    fig, axes = plt.subplots(1, 3, figsize=(F.TEXT_W, 2.3), sharey=True,
                             gridspec_kw=dict(wspace=0.06))
    for i, (ax, p, title) in enumerate(zip(axes, P, titles)):
        for x0, w, y0, h, col, hat in (
                (0, p["T"], 0, p["ps"], F.BASE, F.HATCH["base"]),
                (0, p["tc"], p["ps"], p["pc"], F.COMM, F.HATCH["comm"]),
                (0, p["tg"], p["ps"] + p["pc"], p["pg"], F.GEMM, F.HATCH["gemm"])):
            ax.add_patch(Rectangle((x0, y0), w, h, facecolor=col, edgecolor="white",
                                   linewidth=0.7, hatch=hat, zorder=3))
        ax.text(p["tg"] / 2, p["ps"] + p["pc"] + p["pg"] / 2, f"{p['eg']:.0f}",
                ha="center", va="center", color="white", fontsize=7, zorder=5)
        ax.text(p["tc"] * 0.66, p["ps"] + p["pc"] / 2, f"{p['ec']:.0f}", ha="center",
                va="center", color="white", fontsize=7, zorder=5)
        ax.text(p["T"] * 0.66, p["ps"] / 2, f"{p['es']:.0f}", ha="center", va="center",
                color=F.INK, fontsize=7, zorder=5)
        ax.axvline(p["T"], color=F.INK, lw=0.7, zorder=4)
        E = p.get("E_meas", p.get("E"))
        ax.text(0.04, 0.965, f"{E:.0f} mJ", transform=ax.transAxes, fontsize=8.5,
                va="top", ha="left", fontweight="bold")
        if i:
            ax.text(0.04, 0.845, f"{(E - E0) / E0 * 100:+.1f}%", transform=ax.transAxes,
                    fontsize=8.5, va="top", ha="left", color=F.GEMM, fontweight="bold")
        ax.text(0.96, 0.965, f"$T$ {p['T']:.2f} ms", transform=ax.transAxes,
                fontsize=6.8, va="top", ha="right", color=F.MUTED)
        ax.text(0.96, 0.865, "measured" if i < 2 else "inferred", transform=ax.transAxes,
                fontsize=6.5, va="top", ha="right",
                color=F.MUTED if i < 2 else F.GEMM, style="italic")
        ax.set_title(title, fontsize=7.2, pad=4)
        ax.set_xlim(0, 1.32); ax.set_ylim(0, 300)
        ax.set_xlabel("time (ms)")
        F.despine(ax, grid_axis="y")
        ax.set_xticks([0, 0.5, 1.0]); ax.set_yticks([0, 100, 200, 300])
    axes[0].set_ylabel("power (W)")
    fig.suptitle(f"GEMM {meta['M']}x{meta['N']}x{meta['N']}  +  {meta['MIB']} MiB "
                 f"all-reduce, {meta['CT']} CTAs, world = 4   "
                 f"(one real overlapped run per panel)", fontsize=7.5, y=1.045)
    h = [Rectangle((0, 0), 1, 1, facecolor=c, hatch=k, edgecolor="white", linewidth=0.5)
         for c, k in ((F.GEMM, F.HATCH["gemm"]), (F.COMM, F.HATCH["comm"]),
                      (F.BASE, F.HATCH["base"]))]
    fig.legend(h, ["GEMM", "all-reduce", "static floor"], loc="lower center",
               ncol=3, bbox_to_anchor=(0.5, -0.17))
    return F.save(fig, "fig9_case")


if __name__ == "__main__":
    for fn in (fig_mechanism, fig_objectives, fig_rule, fig_frontier,
               fig_paired, fig_knee, fig_commbound, fig_case):
        print("wrote", *fn())
