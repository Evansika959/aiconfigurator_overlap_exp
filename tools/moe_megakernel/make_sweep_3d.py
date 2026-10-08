#!/usr/bin/env python3
"""fig22 -- the two-domain search space against what is actually deployed.

PILLARS, and the viewing angle is the whole trick. Makespan is tallest where both clocks
are low, so at the wrong azimuth that wall stands in front and hides the valley behind it,
which is the entire interesting region. Viewed from elev 36 / azim -58, with f_cold
mirrored in the data, the tall low-clock wall falls to the back, every pillar is visible,
and 1410 MHz sits at the near end of BOTH axes -- the two maxima meet at the corner
nearest the reader.

BASELINE: the unlocked governor. PyTorch launches the kernel and the GPU chooses its own
clock; measured 6 times it settles at 1322 MHz, 3.759 ms, 1503.6 mJ. No clock tuning is
assumed on the baseline's behalf.

Each pillar is one (f_hot, f_cold) pair, 510-1410 MHz at 45 MHz -- all 21 clocks
measured. At full width nothing above 1290 MHz can be: locked higher, the board hits
400 W and settles near 1320. So above 1290 the full-width curve is REBUILT from a 54-SM
sweep that holds every clock to 1410 (analyse_hiclk.full_width_curve). The rebuild is
checked against data it did not use: at 1320 MHz it reads 401.4 W against the
governor's measured 400.0 W.
  HEIGHT -- makespan relative to the governor. What the design costs.
  COLOUR -- energy relative to the governor, linear and centred on it: blue saves,
            orange costs more than the baseline.

MAKESPAN is the wall time for the whole layer: both partitions run concurrently, so the
layer ends when the slower one ends. The SM split is chosen to make them end together,
so makespan is either one's runtime. 1.0 means "as fast as the deployed baseline".

THE 400 W LIMIT IS APPLIED TO THE COMPOSED PAIR, not to the inputs. A pair's mean board
power is its energy over its makespan; pairs over 400 W are drawn grey rather than
dropped, so the region the limit removes stays visible. (An earlier version capped
both domains at 1275 MHz because the INPUT data stopped there, which made "nothing is
faster than the baseline" true by construction: the baseline ran at 1322.)

The fastest pillar under the limit is just below 1.0 -- the uniform clock at 1320 MHz,
faster than the governor only because the governor's clock oscillates on the limit.

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
from analyse_hiclk import full_width_curve

HERE = os.path.dirname(os.path.abspath(__file__))
STEP = 45
F_LO, F_HI = 510, 1410            # the A100's range; every 45 MHz point is measured
TDP, TDP_TOL = 400.0, 1.5         # W; the tolerance is analyse_hiclk.py's check-3 error
N_SM = 108
GREY = (0.60, 0.58, 0.55, 0.45)   # pairs over the power limit: shown, not dropped
SHOW = [(0.016, 8), (0.045, 16), (0.208, 48), (0.596, 96)]


def main(cold_sms=None, round_to=None, layout="grid"):
    """cold_sms=None: the SM split follows the work -- recomputed for every pair so both
    sides finish together (continuous, the original fig22).
    cold_sms=N: the cold V/f domain is a FIXED block of N SMs, as a real domain would be
    (one GPC is 16). Each side's experts are pinned to its domain, so the side that
    finishes first idles and the makespan is the slower side. -> fig22b.
    round_to=Q: the split follows the work again, but is ROUNDED to whole blocks of Q SMs
    (one GPC = 16), between one block and 108 - 12. Work stays pinned, so whatever the
    rounding leaves unbalanced shows up as idle time. -> fig22c."""
    fixed = cold_sms is not None or round_to is not None
    row = layout == "row"
    # one row of four 3D panels needs the two-column text width (figure*, 7.0 in): at
    # the single-column 5.5 in each panel would be 1.37 in wide and the 7 pt floor
    # would no longer fit
    W = 7.0 if row else fs.TEXT_W
    HOT_MIN = 12                                   # SMs the hot side keeps, at least
    # 510-1410 MHz: measured at full width up to 1290, and above it rebuilt from the
    # 54-SM sweep (analyse_hiclk.py, whose check 3 lands on the governor to 1.4 W)
    cl, T, D, PS, rebuilt = full_width_curve()
    grid = np.arange(F_LO, F_HI + 1, STEP)
    have = set(cl.tolist())
    assert all(c in have for c in grid), "every grid clock must be a measured one"
    Tg, Dg, Pg = [np.interp(grid, cl, v) for v in (T, D, PS)]
    n = len(grid)

    by = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(HERE, "data", "b1_energy_qwen.csv"))):
        by[int(r["clock"])].append(r)
    A = lambda c, k: float(np.mean([float(x[k]) for x in by[c]]))
    Tgov, Egov, Fgov = A(-1, "ms_per_layer"), A(-1, "energy_mj_per_layer"), A(-1, "clock_med")
    Pgov = A(-1, "power_w")

    # compose every panel first: the floor and the colour range are shared
    panels = []
    for phi, L in SHOW:
        xs, ys, hs, cs, ok, ncold, lim_hot, idle = [], [], [], [], [], [], [], []
        for ih in range(n):
            for il in range(ih + 1):
                if fixed:
                    # work is pinned; time = work share x full-width time x 108/SMs
                    if round_to:
                        # the balanced share, from the MEASURED T(f) of each side...
                        sb = phi * Tg[il] / (phi * Tg[il] + (1 - phi) * Tg[ih])
                        # ...rounded to whole GPCs
                        q = min(max(round(sb * N_SM / round_to), 1),
                                (N_SM - HOT_MIN) // round_to)
                        sm = q * round_to / N_SM
                    else:
                        sm = cold_sms / N_SM
                    tc = phi * Tg[il] / sm
                    th = (1 - phi) * Tg[ih] / (1 - sm)
                    t = max(tc, th)
                    lim_hot.append(th >= tc); idle.append(1 - min(tc, th) / t)
                else:
                    t = Tg[ih] * (phi * grid[ih] / grid[il] + 1 - phi)
                    sm = phi * grid[ih] / (phi * grid[ih] + (1 - phi) * grid[il])
                    lim_hot.append(True); idle.append(0.0)
                e = (Dg[il] * phi + Dg[ih] * (1 - phi)
                     + (sm * Pg[il] + (1 - sm) * Pg[ih]) * t)
                xs.append(ih); ys.append(n - 1 - il)
                hs.append(t / Tgov - 1); cs.append((e / Egov - 1) * 100)
                ok.append(e / t <= TDP + TDP_TOL)          # mean board power, W
                ncold.append(sm * N_SM)                     # SMs given to the cold side
        panels.append((phi, L, *(np.array(v) for v in (xs, ys, hs, cs, ok, ncold,
                                                        lim_hot, idle))))
    z0 = np.floor(min(p[4].min() for p in panels) * 20) / 20 - 0.03
    e_hi = max(p[5][p[6]].max() for p in panels)

    fs.use()
    cmap = fs.diverging().reversed()
    # neutral AT the baseline: blue saves energy, orange costs more than the baseline.
    # Symmetric and LINEAR -- a two-slope norm put -30..0 and 0..+15 on equal halves of
    # the colour bar, which reads as a scale it is not.
    # Fixed mode: pinned work reaches +170%, and a range that wide washes the other
    # panels to white. Clip at +/-50% and SAY so, with extend arrows on the colour bar.
    lim = 50 if fixed else max(30, np.ceil(e_hi / 10) * 10)
    norm = colors.Normalize(vmin=-lim, vmax=lim)
    # ---- the prose, composed first: its height sets the figure's --------------
    # The dense text lives in ONE block under the plots, separated by a rule, so the
    # plotting area carries only what is read off the plots: panel titles, axes, and
    # the colour key.
    bestc = min((c for c in by if c > 0), key=lambda c: A(c, "energy_mj_per_layer"))
    ref_e = (A(bestc, "energy_mj_per_layer") / Egov - 1) * 100
    ref_t = A(bestc, "ms_per_layer") / Tgov
    setup = (f"A100-SXM4-40GB, Qwen3-30B-A3B MoE layer, persistent kernel with no build "
             f"cost charged. Every (f_cold, f_hot) pair from {F_LO} to {F_HI} MHz at "
             f"{STEP} MHz. Above 1290 MHz the full-width curve is rebuilt from a 54-SM "
             "sweep, which reproduces the baseline at 1320 MHz to within 1.4 W. Baseline: "
             "clocks unlocked, the GPU picks its own frequency; measured 6$\\times$: "
             f"{Fgov:.0f} MHz, {Tgov:.3f} ms, {Egov:.0f} mJ, {Pgov:.0f} W.")
    pinned = ("Experts are pinned to their domain, so the side that finishes first idles "
              "(still drawing static power) and the makespan is the slower side. The "
              "diagonal is two domains at one clock, still partitioned.")
    if round_to:
        split_txt = ("The cold domain gets the SM share that balances the two sides, "
                     f"rounded to whole GPCs of {round_to} SMs ({round_to}$-$"
                     f"{(N_SM - HOT_MIN) // round_to * round_to} SMs). " + pinned)
    elif fixed:
        split_txt = (f"The cold domain is fixed at {cold_sms} SMs (one GPC); the hot "
                     f"domain gets the other {N_SM - cold_sms}. " + pinned)
    else:
        split_txt = ("The SM split follows the work: it is recomputed for every pair so "
                     "both sides finish together, and the cold side's SM count is "
                     "continuous. A real partition rounds it to whole SMs.")
    reading = ("Height: makespan vs the baseline" +
               (" (scales differ between panels)" if fixed else "") +
               ". Colour: energy vs the baseline" +
               (f", clipped at $\\pm${lim:.0f}%" if fixed and e_hi > lim else "") +
               ". Grey: mean board power over 400 W. Red pin: the lowest-energy genuine "
               "split (f_cold $\\neq$ f_hot). Black edge: makespan under 1.0$\\times$. "
               f"Reference, no partition: one clock at {bestc} MHz over all {N_SM} SMs "
               f"gives {ref_e:+.1f}% energy at {ref_t:.2f}$\\times$.")
    if not fixed:
        reading += (" Every pillar under 1.0$\\times$ is slower than the uniform clock at "
                    "1320 MHz, which beats the governor only because the governor's clock "
                    "oscillates on the 400 W limit.")
    # f_cold / f_hot in the same math face the axes use, not as raw underscores
    mth = lambda t: t.replace("f_cold", "$f_{cold}$").replace("f_hot", "$f_{hot}$")
    paras = [("Setup.", mth(setup)), ("SM split.", mth(split_txt)),
             ("Reading.", mth(reading))]
    T_PROSE, LEAD = 7.0, 0.132                     # pt, baseline-to-baseline in inches

    def wrap(paras, width_in):
        """Greedy word wrap measured on the renderer, not by character count: the mix
        of serif text and math glyphs makes a character budget either overflow the
        width or waste it."""
        tmp = plt.figure(figsize=(W, 2)); r = tmp.canvas.get_renderer()
        probe = tmp.text(0, 0, "", fontsize=T_PROSE)
        wid = lambda t: (probe.set_text(t), probe.get_window_extent(r).width
                         / tmp.dpi)[1]
        lines = []
        for head, body in paras:
            # mathtext drops plain spaces; "\ " keeps them ("SM split", not "SMsplit")
            cur = "$\\bf{" + head[:-1].replace(" ", "\\ ") + "}$."
            for w in body.split(" "):
                if wid(cur + " " + w) <= width_in:
                    cur += " " + w
                else:
                    lines.append(cur); cur = w
            lines.append(cur)
        plt.close(tmp)
        return lines

    prose = wrap(paras, W - 0.08)

    # ---- bands, in inches from the bottom -----------------------------------------
    # A 2x2 grid of 3D axes does not survive subplots_adjust: a 3D axes' tick labels
    # fill its rectangle to the bottom edge, so anything placed by fraction lands on
    # the neighbouring row. Each element gets its own band instead.
    TXT_H = len(prose) * LEAD + 0.06
    KEY_B, KEY_H = TXT_H + 0.42, 0.075            # colour bar
    if row:
        AX_H, TTL_H = 1.95, 0.64                   # narrower panels: 4 title lines
        ROW1 = KEY_B + KEY_H + 0.26                # the axes' own margin is the gap
        H = ROW1 + AX_H + TTL_H - 0.02
        fig = plt.figure(figsize=(W, H))
        y = lambda inch: inch / H
        # a gutter between panels: panel k's f_cold ticks sit where panel k+1's f_hot
        # axis starts
        RECT = [(k * 0.254, y(ROW1), 0.238, y(AX_H)) for k in range(4)]
        TITLE_Y = [y(H - 0.03)] * 4
    else:
        AX_H, TTL_H = 2.45, 0.66                   # 3D axes, panel title block
        ROW2 = KEY_B + KEY_H + 0.42
        ROW1 = ROW2 + AX_H + TTL_H + 0.10          # clears row 1's axis labels
        H = ROW1 + AX_H + TTL_H - 0.15
        fig = plt.figure(figsize=(W, H))
        y = lambda inch: inch / H
        RECT = [(0.00, y(ROW1), 0.49, y(AX_H)), (0.505, y(ROW1), 0.49, y(AX_H)),
                (0.00, y(ROW2), 0.49, y(AX_H)), (0.505, y(ROW2), 0.49, y(AX_H))]
        TITLE_Y = ([y(ROW1 + AX_H + TTL_H - 0.18)] * 2
                   + [y(ROW2 + AX_H + TTL_H - 0.15)] * 2)

    summary = []
    for k, (phi, L, xs, ys, hs, cs, ok, ncold, lim_hot, idle) in enumerate(panels):
        ax = fig.add_axes(RECT[k], projection="3d")
        # transparent: a 3D axes' white patch covered the previous panel's z ticks and
        # f_cold label wherever they reach past its rectangle
        ax.set_facecolor("none")
        col = np.where(ok[:, None], cmap(norm(cs)), np.array([GREY]))
        under = ok & (hs < 0)                    # faster than the baseline, under 400 W
        rest = ~under
        ax.bar3d(xs[rest] - .40, ys[rest] - .40, np.full(rest.sum(), z0), .80, .80,
                 hs[rest] - z0, color=col[rest], shade=False, edgecolor="white",
                 linewidth=0.10)
        # makespan below 1.0 is ~1% of this axis -- invisible as height. Those pillars
        # are drawn last with a black edge instead; they all sit on the near side
        if under.any():                          # none at all once the domain is fixed
            ax.bar3d(xs[under] - .40, ys[under] - .40, np.full(under.sum(), z0), .80,
                     .80, hs[under] - z0, color=col[under], shade=False,
                     edgecolor=fs.INK, linewidth=0.45)
        feas = np.where(ok)[0]
        # the red pin is the best GENUINE split: the global optimum is always on the
        # diagonal (f_cold = f_hot, no split) in the balanced model, where phi drops
        # out and all panels would print the same number
        split = feas[xs[feas] != (n - 1 - ys[feas])]
        bi = split[int(np.argmin(cs[split]))]
        di = feas[int(np.argmin(cs[feas]))]
        fi = feas[int(np.argmin(hs[feas]))]
        fast = np.where(under)[0]
        pin = 0.12 * (ax.get_zlim()[1] - z0) if fixed else 0.22
        ax.plot([xs[bi], xs[bi]], [ys[bi], ys[bi]], [hs[bi], hs[bi] + pin],
                color="#D01C1C", lw=1.2, zorder=20)
        ax.scatter([xs[bi]], [ys[bi]], [hs[bi] + pin], color="#D01C1C", s=12,
                   depthshade=False, zorder=20)
        summary.append(dict(L=L, diag_e=cs[di], diag_t=hs[di] + 1,
                            n_fast=len(fast), best_e=cs[bi], best_t=hs[bi] + 1,
                            fast_t=hs[fi] + 1, fast_e=cs[fi],
                            fast_h=grid[xs[fi]], fast_c=grid[n - 1 - ys[fi]],
                            n_bad=int((~ok).sum()), n=len(ok)))
        # AXIS ORIENTATION. x is f_hot, y is f_cold. At this azimuth the near corner of
        # the floor is x-MAX / y-MIN, so f_cold alone is mirrored in the data (index
        # n-1-il) and f_hot is not: the top clock then sits at the near end of BOTH
        # axes and the two maxima meet at the corner nearest the reader. Mirroring in
        # the data, not by reversing the axis limits -- reversing the limits mirrors
        # matplotlib's whole 3D projection and pushes an axis out of view. Makespan is
        # tallest where both clocks are low, which this orientation puts at the far
        # corner, so the tall wall falls to the back and hides nothing.
        tk = [0, n // 2, n - 1]
        ax.set_xticks(tk); ax.set_yticks(tk)
        ax.set_xticklabels([grid[i] for i in tk], fontsize=7)
        ax.set_yticklabels([grid[n - 1 - i] for i in tk], fontsize=7)
        if fixed:
            # pinned work can run several times the baseline, so each panel gets its
            # own height scale; a shared one flattens the others
            top = hs.max() + 0.05
            cand = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 9.0, 11.0, 14.0]
            zt = [z for z in cand if z <= top]
            zt = zt[:1] + zt[1:][-2:]
            ax.set_zlim(z0, top)
        else:
            ax.set_zlim(z0, 1.45)
            zt = [0.0, 0.5, 1.0]
        ax.set_zticks(zt)
        ax.set_zticklabels([f"{1 + z:.1f}$\\times$" for z in zt]
                           if (k % 2 == 1 or fixed) else [""] * len(zt), fontsize=7)
        # in a row the four panels share axes: name them once, on the first panel,
        # where nothing sits to the left of f_hot or under f_cold's neighbour
        if not row or k == 0:
            ax.set_xlabel("$f_{hot}$ (MHz)", fontsize=8, labelpad=-3)
            ax.set_ylabel("$f_{cold}$ (MHz)", fontsize=8, labelpad=-3)
        # no set_zlabel: between the two columns it has nowhere to go and was clipped
        # into the neighbouring panel. The word sits on top of the z axis instead, at
        # the axis's projected position (placed after the view is set, below).
        ax.tick_params(pad=-1.5)
        ax.view_init(elev=36, azim=-58)
        ax.set_box_aspect((1, 1, 0.85), zoom=1.02 if row else 1.0)
        ax.xaxis.pane.set_alpha(0); ax.yaxis.pane.set_alpha(0)
        ax.zaxis.pane.set_alpha(0)
        ax.grid(False)
        # top of the z axis in screen space: the corner that projects furthest right
        from mpl_toolkits.mplot3d import proj3d
        fig.canvas.draw()
        zt_top = ax.get_zlim()[1]
        pts = []
        for cxn in (-0.5, n - 0.5):
            for cyn in (-0.5, n - 0.5):
                u, v, _ = proj3d.proj_transform(cxn, cyn, zt_top, ax.get_proj())
                pts.append(ax.transData.transform((u, v)))
        px, py = max(pts, key=lambda q: q[0])
        fx_, fy_ = fig.transFigure.inverted().transform((px, py))
        fig.text(fx_, fy_ + y(0.06), "makespan", fontsize=8, ha="center", va="bottom",
                 color=fs.INK)
        cx = RECT[k][0] + RECT[k][2] / 2
        fig.text(cx, TITLE_Y[k], f"{L} cold experts ($\\varphi$ = {phi:.3f})",
                 fontsize=8.5, fontweight="bold", ha="center", va="top", color=fs.INK)
        under_txt = (f"; {len(fast)} under 1.0$\\times$" if len(fast) else "")
        if fixed and row:
            # four short lines: a panel is 1.75 in wide
            slow = "hot" if lim_hot[bi] else "cold"
            body = (f"best split: {cs[bi]:+.1f}% energy, {hs[bi] + 1:.2f}$\\times$\n"
                    f"cold on {ncold[bi]:.0f} SMs, {slow} side limits\n"
                    f"other side idles {idle[bi]:.0%}\n"
                    f"fastest {hs[fi] + 1:.2f}$\\times${under_txt}")
        elif fixed:
            slow = "hot" if lim_hot[bi] else "cold"
            body = (f"best split: {cs[bi]:+.1f}% energy, {hs[bi] + 1:.2f}$\\times$ "
                    f"makespan\ncold on {ncold[bi]:.0f} SMs; {slow} side limits, the "
                    f"other idles {idle[bi]:.0%}\nfastest {hs[fi] + 1:.2f}$\\times$"
                    f"{under_txt}")
        else:
            body = (f"best split: {cs[bi]:+.2f}% energy, {hs[bi] + 1:.3f}$\\times$ "
                    f"makespan\ncold side on {ncold[bi]:.1f} SMs (continuous)\n"
                    f"fastest {hs[fi] + 1:.3f}$\\times${under_txt}")
        fig.text(cx, TITLE_Y[k] - y(0.17), body,
                 fontsize=7.5, ha="center", va="top", color=fs.INK, linespacing=1.3)

    # ---- colour key --------------------------------------------------------------
    cax = fig.add_axes([0.24, y(KEY_B), 0.34, y(KEY_H)] if row else
                       [0.16, y(KEY_B), 0.46, y(KEY_H)])
    cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                      orientation="horizontal",
                      extend="both" if fixed and e_hi > lim else "neither")
    cb.set_label("energy vs the baseline (%)", fontsize=8, labelpad=2)
    cb.ax.tick_params(labelsize=7)
    cb.outline.set_visible(False)
    gx = fig.add_axes([0.64, y(KEY_B), 0.024, y(KEY_H)] if row else
                      [0.68, y(KEY_B), 0.03, y(KEY_H)])
    gx.add_patch(plt.Rectangle((0, 0), 1, 1, fc=GREY, ec=fs.MUTED, lw=0.4))
    gx.axis("off")
    fig.text(0.672 if row else 0.72, y(KEY_B + KEY_H / 2), "over the 400 W limit",
             fontsize=8,
             color=fs.INK, va="center", ha="left")

    # ---- the prose block ---------------------------------------------------------
    fig.add_artist(plt.Line2D([0.0, 1.0], [y(TXT_H + 0.02)] * 2, color=fs.MUTED,
                              lw=0.5, transform=fig.transFigure))
    for i, line in enumerate(prose):
        fig.text(0.004, y(TXT_H - 0.06 - i * LEAD), line, fontsize=T_PROSE,
                 color=fs.INK, va="top", ha="left")

    print("per panel:", [(d["L"], round(d["best_e"], 2), round(d["best_t"], 3),
                          round(d["fast_t"], 3), d["fast_h"], d["fast_c"],
                          f"{d['n_bad']}/{d['n']} over TDP") for d in summary])
    print(f"figure {W} x {H:.2f} in, {len(prose)} lines of prose")
    stem = ("fig22c_sweep_gpc_round" if round_to else
            "fig22b_sweep_cold_gpc" if fixed else "fig22_sweep_heatmap")
    print(fs.save(fig, stem + ("_row" if row else "")))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cold-sms", type=int, default=None,
                    help="fix the cold V/f domain at this many SMs (one GPC = 16); "
                         "writes fig22b instead of fig22")
    ap.add_argument("--round-gpc", type=int, default=None,
                    help="round the balanced cold-side SM count to whole blocks of this "
                         "many SMs (one GPC = 16); writes fig22c")
    ap.add_argument("--layout", choices=("grid", "row"), default="grid",
                    help="grid: 2x2 at 5.5 in; row: 1x4 at 7.0 in, saved with a _row "
                         "suffix so the grid version is kept")
    a = ap.parse_args()
    main(a.cold_sms, a.round_gpc, a.layout)
