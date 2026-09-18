#!/usr/bin/env python3
"""fig17 -- B1: measured energy and latency of a real fused MoE layer vs SM clock.

Every earlier energy number in this folder was composed from micro-benchmarks. This one
is the layer itself: vLLM's Triton grouped GEMM, a real traced expert histogram, NVML
integration, clocks locked, 6 independent passes over a reshuffled clock grid.

DESIGN NOTES -- each is a review finding this figure had to answer.

  POOL BY MEASURED CLOCK, NOT BY REQUEST. Requests of 1335, 1380 and 1410 MHz all run
    at 1320 MHz, because the layer is against the 400 W board cap. Keying on the
    request makes the panel multivalued at x = 1320 and draws a line through the stack;
    keying on the measurement collapses them into the single operating point they are
    (n = 18: 3.7240 +/- 0.0027 ms, 1486.54 +/- 5.10 mJ).

  DYNAMIC ENERGY IS PLOTTED ONLY WHERE THE CLOCK WAS HELD. Idle is measured with the
    clock locked to the REQUEST, and at idle there is no power cap, so the request is
    honoured: the 1410 MHz row subtracts a 83.0 W idle from a run that was actually at
    1320 MHz, where idle is ~73 W. Static power is itself strongly clock-dependent here
    (60.4 W at 510 MHz rising to 73.2 W at 1290, r = +0.86), so mismatching the two
    states corrupts the subtraction by several percent. Where the clock held, the idle
    reference and the loaded state are the same operating point and the subtraction is
    sound; where it did not, the point is simply not drawn.

  PANEL (b) IS LOG-LOG. The energy/latency trade here is close to 1:1 (governor ->
    1035 MHz is -23.6% energy for +23.4% latency), but latency spans 2.4x while energy
    spans 1.3x, so on linear axes that trade renders about 5x steeper than it is -- the
    frontier reads as a cliff. On log-log a constant-ratio trade is a 45-degree line,
    so the slope means what it looks like it means. The axes keep absolute mJ and ms;
    nothing is normalised.

  THE BASELINE IS THE UNLOCKED GOVERNOR -- a measured operating point, not a row picked
    out of the sweep. "The fastest clock that holds" is a selection rule, and on this
    data it picks a Pareto-dominated point.
"""
import csv, os, sys
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs

HERE = os.path.dirname(os.path.abspath(__file__))
HELD_TOL = 7            # MHz; one A100 clock step is 15
FLAT = 1.01             # the flat band is where dynamic energy is within this of its min
# (clock, text anchor in DATA coordinates, ha). Anchors, not point offsets: the panel is
# log-log and an offset in points lands somewhere different at each end of it. 1125 is
# labelled because it is where the marginal return collapses -- labelling only the
# endpoints would smuggle in a recommendation.
LABEL = ((1320, (4.30, 1468), "left"),
         (1200, (4.60, 1370), "left"),
         (1125, (4.93, 1283), "left"),
         (1035, (4.16, 1110), "left"))


def load(path):
    """Aggregate the passes. Locked rows are keyed on the clock they MEASURED, so the
    three capped requests collapse into the one operating point they share."""
    rows = list(csv.DictReader(open(path)))
    gov_rows = [r for r in rows if int(r["clock"]) == -1]
    by = defaultdict(list)
    for r in rows:
        if int(r["clock"]) != -1:
            by[int(round(float(r["clock_med"])))].append(r)

    def pack(rs):
        n = len(rs)
        g = lambda k: np.array([float(x[k]) for x in rs])
        sd = lambda v: float(v.std(ddof=1)) if n > 1 else 0.0
        ms, mj, dj = g("ms_per_layer"), g("energy_mj_per_layer"), g("e_dyn_mj_per_layer")
        held = bool(np.all(np.abs(g("clock") - g("clock_med")) <= HELD_TOL))
        # the collector's own `held` column used a 20 MHz tolerance and so called the
        # 1335 MHz row held when it measured 1320; trust the measurement, not the flag
        return dict(n=n, ms=ms.mean(), ms_sd=sd(ms), mj=mj.mean(), mj_sd=sd(mj),
                    dj=dj.mean(), dj_sd=sd(dj), held=held,
                    req=sorted({int(float(x["clock"])) for x in rs}))
    return {c: pack(rs) for c, rs in by.items()}, (pack(gov_rows) if gov_rows else None)


def pareto(ms, mj):
    keep = [i for i in range(len(ms))
            if not np.any((ms <= ms[i]) & (mj <= mj[i]) &
                          ((ms < ms[i]) | (mj < mj[i])))]
    return np.array(sorted(keep, key=lambda i: ms[i]))


def pct(v):
    """One decimal below 2%, so -0.9% and -0.6% do not both print as -1%."""
    return f"{v*100:+.1f}%" if abs(v) < 0.02 else f"{v*100:+.0f}%"


def main():
    d, gov = load(os.path.join(HERE, "data", "b1_energy_qwen.csv"))
    cl = np.array(sorted(d))
    MS = np.array([d[c]["ms"] for c in cl])
    MJ = np.array([d[c]["mj"] for c in cl]); MJE = np.array([d[c]["mj_sd"] for c in cl])
    DJ = np.array([d[c]["dj"] for c in cl]); DJE = np.array([d[c]["dj_sd"] for c in cl])
    HELD = np.array([d[c]["held"] for c in cl])
    npass = max(v["n"] for c, v in d.items() if v["held"])      # 6; 1320 pools 3x6

    fs.use()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(fs.TEXT_W, 3.05))

    # ---- (a) total vs dynamic energy ------------------------------------------
    a1.errorbar(cl, MJ, yerr=MJE, fmt="-o", color=fs.GEMM, ms=2.4, lw=1.0,
                elinewidth=0.6, capsize=1.4, zorder=3, label="total (board)")
    a1.errorbar(cl[HELD], DJ[HELD], yerr=DJE[HELD], fmt="--s", color=fs.COMM, ms=2.2,
                lw=1.0, elinewidth=0.6, capsize=1.4, zorder=3,
                label="dynamic (idle subtracted; held clocks only)")
    a1.scatter(cl[~HELD], MJ[~HELD], s=26, facecolors="none", edgecolors=fs.INK,
               linewidths=0.7, zorder=4, label="400 W cap active")
    # the flat band is derived, not hard-coded: where dynamic energy is within 1% of
    # its own minimum
    flat = HELD & (DJ <= DJ[HELD].min() * FLAT)
    a1.axvspan(cl[flat].min(), cl[flat].max(), color=fs.GRID, alpha=0.55, zorder=0)
    y0, y1 = a1.get_ylim()
    a1.text(0.5 * (cl[flat].min() + cl[flat].max()), DJ[flat].max() + 0.06 * (y1 - y0),
            # short enough to sit INSIDE the shaded band: centred on the band midpoint,
            # a longer label grows past both edges -- into the y-axis on the left and
            # under the rising dynamic curve on the right. What the shading marks is
            # said in the provenance block instead.
            "%d$-$%d MHz:\ntotal varies %.0f%%, dynamic %g%%"
            % (cl[flat].min(), cl[flat].max(),
               (MJ[flat].max() / MJ[flat].min() - 1) * 100, (FLAT - 1) * 100),
            ha="center", va="bottom", color=fs.MUTED, fontsize=7)
    a1.set_ylim(y0, y1)
    a1.set_xlabel("measured SM clock (MHz)")
    a1.set_ylabel("energy per MoE layer (mJ)")
    a1.set_title("(a) below the knee the rise is idle amortisation", loc="left",
                 pad=8)
    h, lab = a1.get_legend_handles_labels()
    o = [lab.index(x) for x in ("total (board)",
                                "dynamic (idle subtracted; held clocks only)",
                                "400 W cap active")]
    a1.legend([h[i] for i in o], [lab[i] for i in o], loc="upper center")
    fs.despine(a1)

    # ---- (b) energy-latency frontier, log-log ---------------------------------
    pf = pareto(MS, MJ)
    dom = np.setdiff1d(np.arange(len(cl)), pf)
    xlo, xhi = MS[pf].min() / 1.02, MS[pf].max() * 1.36
    ylo, yhi = MJ[pf].min() / 1.075, MJ[pf].max() * 1.062
    # direct labels instead of a legend box: three series, all of them adjacent to
    # text that already names them
    a2.plot(MS[dom], MJ[dom], "o", color=fs.BASE, ms=2.6, zorder=2)
    a2.errorbar(MS[pf], MJ[pf], yerr=MJE[pf], fmt="-o", color=fs.GEMM, ms=3.0, lw=1.2,
                elinewidth=0.6, capsize=1.2, zorder=3)
    a2.errorbar(gov["ms"], gov["mj"], xerr=gov["ms_sd"], yerr=gov["mj_sd"], fmt="D",
                color=fs.COMM, ms=4.0, elinewidth=0.6, capsize=1.4, zorder=5)
    a2.annotate("default governor (deployed)", (gov["ms"], gov["mj"]),
                xytext=(gov["ms"] * 1.007, yhi / 1.010), fontsize=7,
                color=fs.COMM, va="center", ha="left",
                arrowprops=dict(arrowstyle="-", color=fs.COMM, lw=0.5, shrinkA=0,
                                shrinkB=3))
    vis = dom[(MS[dom] < xhi / 1.12) & (MS[dom] > MS[pf].max())]
    j = vis[len(vis) // 2]
    a2.annotate("dominated", (MS[j], MJ[j]), xytext=(MS[j] * 1.02, MJ[pf].min() / 1.052),
                fontsize=7, color=fs.MUTED, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=fs.MUTED, lw=0.5, shrinkA=0,
                                shrinkB=3))
    a2.set_xscale("log"); a2.set_yscale("log")
    # Zoom to the frontier. NOT because the tail would distort the geometry -- the
    # box aspect below is derived from whatever limits are in force, so 45 degrees is
    # 1:1 at any zoom. It is because carrying the tail gives a 0.31 box aspect, an
    # awkward shape that crushes the frontier detail this panel is about.
    a2.set_xlim(xlo, xhi); a2.set_ylim(ylo, yhi)
    n_off = int(np.sum(MS > xhi))
    # a constant-ratio trade is a 45-degree line only when a decade is the same length
    # on both axes; that fixes the box aspect
    a2.set_box_aspect(np.log10(yhi / ylo) / np.log10(xhi / xlo))
    for c, anchor, ha in LABEL:
        if c not in d:
            continue
        # 1035 and 1050 tie on energy (dE = 0.28 mJ, t = 0.11) but are cleanly separated
        # on latency (t = 61). Merging them and printing 1035's latency would quote the
        # slower member of the pair, so name 1050 -- the one you would deploy -- and say
        # the other ties.
        name = ("1050 MHz (1035 ties)" if c == 1035 else
                f"{c} MHz (capped)" if not d[c]["held"] else f"{c} MHz")
        ref = 1050 if c == 1035 else c
        a2.annotate(f"{name}\n{pct(d[ref]['mj']/gov['mj']-1)} E, "
                    f"{pct(d[ref]['ms']/gov['ms']-1)} T",
                    (d[c]["ms"], d[c]["mj"]), xytext=anchor, fontsize=7,
                    color=fs.INK, va="center", ha=ha,
                    arrowprops=dict(arrowstyle="-", color=fs.MUTED, lw=0.5,
                                    shrinkA=0, shrinkB=2))
    for ax, ticks, fmtstr in ((a2.xaxis, [3.8, 4.0, 4.2, 4.5, 5.0, 5.5, 6.0], "%g"),
                              (a2.yaxis, [1150, 1200, 1300, 1400, 1500], "%g")):
        ax.set_major_locator(FixedLocator(ticks))
        ax.set_minor_locator(FixedLocator([]))
        ax.set_major_formatter(FuncFormatter(lambda v, _p, f=fmtstr: f % v))
    a2.set_xlabel("latency per MoE layer (ms, log)")
    a2.set_ylabel("energy per MoE layer (mJ, log)")
    a2.set_title("(b) frontier, against the deployed point", loc="left", pad=8)

    fs.despine(a2)

    fig.text(0.005, 0.992,
             "A100-SXM4-40GB, 400 W cap  |  Qwen3-30B-A3B MoE layer, 7744 prefill "
             "tokens, real ShareGPT routing  |  vLLM Triton grouped GEMM (tuned)\n"
             f"{npass} passes per clock over a reshuffled grid, mean $\\pm$ sd (the 1320 MHz "
             "point pools requests of 1335/1380/1410, n$\\,$=$\\,$18)  |  total energy = "
             "NVML\nmedian board power $\\times$ layer time; dynamic = (that power "
             "$-$ idle measured at the same locked clock) $\\times$ layer time.\n"
             "Shading in (a) marks where dynamic energy is within 1% of its minimum. "
             "Per-point sd varies genuinely: 1.20 mJ at 1035 MHz against 8.75 at 1050, "
             "both n$\\,$=$\\,$6.",
             fontsize=7, color=fs.INK, va="top", ha="left", linespacing=1.35)
    fig.tight_layout(pad=0.3, w_pad=1.8, rect=(0, 0, 1, 0.828))
    print(fs.save(fig, "fig17_b1_energy"))


if __name__ == "__main__":
    main()
