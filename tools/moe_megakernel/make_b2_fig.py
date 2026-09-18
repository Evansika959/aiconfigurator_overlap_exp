#!/usr/bin/env python3
"""fig18 -- a second spatial V/f domain buys latency granularity, not energy.

(a) is measured: what a partitionable kernel costs.
(b) and (c) are composed from B1's clock sweep, with BOTH sides scored on the same
clock set under the same deadline rule.

WHY (b) IS GAIN-vs-STEP AND NOT GAIN-vs-LATENCY. Scored like-for-like, the two-domain
design does win -- and the win shrinks with the clock step. That is the finding: the
advantage is the design landing exactly on a deadline that a quantised clock grid
overshoots, not any voltage it harvests. A gain-vs-latency panel shows that only as an
eyeballed difference between curves; a gain-vs-step panel on log axes IS the claim, and
a reader can check the straight line.

Three components of the exponent's uncertainty surfaced in succession, and the first
two were each quoted as though they were the whole: the within-fit CI of one thinning
anchor, then the spread across anchors, and only then the measurement noise, which is
the largest. The figure quotes them combined.

(c) plots a DIFFERENCE in mJ, not a ratio: at matched latency the designs differ by
about 1%, so two absolute-energy curves would overplot. In mJ it stays an absolute
measurement rather than a normalisation.
"""
import csv, os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs

HERE = os.path.dirname(os.path.abspath(__file__))


def agg(rows, case, key="ms"):
    v = np.array([float(r[key]) for r in rows if r["case"] == case])
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0), len(v)


def main():
    pq = list(csv.DictReader(open(os.path.join(HERE, "data",
                                               "pin_vs_queue_qwen.csv"))))
    md = list(csv.DictReader(open(os.path.join(HERE, "data",
                                               "two_domain_matched.csv"))))
    # the exponent and its two uncertainty components are read from the summary the
    # model writes, never transcribed -- a docstring table here went stale once already
    sm = list(csv.DictReader(open(os.path.join(HERE, "data",
                                               "two_domain_summary.csv"))))
    # uncertainty components live in their own file with explicit names: overloading
    # the summary's columns per row makes it a trap for whoever reads it next
    unc = {r["quantity"]: float(r["value"]) for r in
           csv.DictReader(open(os.path.join(HERE, "data",
                                            "two_domain_uncertainty.csv")))}
    expo, esd = unc["exponent"], unc["combined_sd"]
    a_sd, m_sd = unc["anchor_spread_sd"], unc["pass_bootstrap_sd"]
    e_max, n_res = unc["exponent_max_over_all_refits"], int(unc["n_pass_resamples"])
    cases = sorted({r["case"] for r in pq})
    best = min((c for c in cases if c.startswith("pinned")),
               key=lambda c: agg(pq, c)[0])

    fs.use()
    fig = plt.figure(figsize=(fs.TEXT_W, 4.35))
    # Explicit axes rectangles, not a gridspec. The two right-hand panels are stacked and
    # (b) needs its own tick labels and axis label in the gap between them; with a
    # gridspec that gap kept collapsing however hspace was set, and (c)'s axes patch
    # painted over whatever landed in it.
    a1 = fig.add_axes([0.098, 0.185, 0.300, 0.605])
    a2 = fig.add_axes([0.580, 0.560, 0.415, 0.250])
    a3 = fig.add_axes([0.580, 0.178, 0.415, 0.240])

    # ---- (a) measured price of a partitionable kernel ----------------------
    bars = [("vLLM", "queue", fs.BASE, ""),
            ("persist.\n1/SM", "persistent_1blk_per_sm", fs.COMM, fs.HATCH["comm"]),
            ("persist.\n2/SM", "persistent_2blk_per_sm", fs.COMM, "xxx"),
            ("pinned\n2 parts", best, fs.GEMM, fs.HATCH["gemm"])]
    vals = [agg(pq, c) for _, c, _, _ in bars]
    q, n = vals[0][0], vals[0][2]
    xs = np.arange(len(bars))
    a1.bar(xs, [v[0] for v in vals], yerr=[v[1] for v in vals], capsize=2,
           error_kw=dict(elinewidth=0.7, ecolor=fs.INK),
           color=[b[2] for b in bars], hatch=[b[3] for b in bars],
           edgecolor="white", width=0.74, zorder=3)
    for i, (v, b) in enumerate(zip(vals, bars)):
        a1.text(i, v[0] + v[1] + 0.09, "baseline" if i == 0 else f"{(v[0]/q-1)*100:+.1f}%",
                ha="center", va="bottom", fontsize=7, color=fs.INK)
    a1.set_xticks(xs); a1.set_xticklabels([b[0] for b in bars], fontsize=6.9)
    a1.set_ylabel("layer latency (ms)")
    a1.set_ylim(0, max(v[0] for v in vals) * 1.24)
    a1.set_title(f"(a) measured: the price of\na partitionable kernel (n$\\,$=$\\,${n})",
                 loc="left", pad=4)
    fs.despine(a1)

    # ---- (b) how the win scales with the clock grid ------------------------
    sc = list(csv.DictReader(open(os.path.join(HERE, "data",
                                               "two_domain_scaling.csv"))))
    anchors = sorted({int(r["anchor"]) for r in sc})
    steps = sorted({int(r["step_mhz"]) for r in sc})
    G = np.array([[float(next(r["gain_pct"] for r in sc
                              if int(r["anchor"]) == a and int(r["step_mhz"]) == st))
                   for st in steps] for a in anchors])
    for row in G:            # every anchor, faintly: the spread IS the uncertainty
        a2.plot(steps, row, "-", color=fs.BASE, lw=0.6, zorder=2)
    mean = G.mean(axis=0)
    # no error bars on the means: min-max bars would encode exactly what the trace fan
    # already shows, and the fan shows more -- that the anchor effect is systematic
    # across step sizes rather than per-point scatter, and that it collapses to a single
    # point at 15 MHz, where thinning by 15 keeps the whole grid whatever the anchor
    a2.plot(steps, mean, "o", color=fs.GEMM, ms=3.4, zorder=4)
    xs = np.log(steps)
    b, c0 = np.polyfit(xs, np.log(mean), 1)
    xf = np.array([steps[0] * 0.88, steps[-1] * 1.14])
    a2.plot(xf, np.exp(c0) * xf ** b, "--", color=fs.COMM, lw=1.0, zorder=3)
    a2.set_xscale("log"); a2.set_yscale("log")
    for ax, tk in ((a2.xaxis, steps), (a2.yaxis, [1.5, 2, 3, 4, 5, 6, 7])):
        ax.set_major_locator(FixedLocator(tk)); ax.set_minor_locator(FixedLocator([]))
        ax.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
    a2.set_xlim(*xf)
    a2.annotate(f"the A100's real step: {mean[0]:.2f}% at 15 MHz", (steps[0], mean[0]),
                xytext=(steps[-1] * 1.06, mean[0] * 0.90), fontsize=7, color=fs.INK,
                ha="right", va="center",
                arrowprops=dict(arrowstyle="-", color=fs.MUTED, lw=0.5, shrinkA=0,
                                shrinkB=3))
    a2.text(steps[0] * 1.06, mean[-1] * 1.20,
            f"gain $\\propto$ (step)$^{{{expo:.2f}\\,\\pm\\,{esd:.2f}}}$\n"
            f"$\\pm\\,$=$\\,${len(anchors)} anchors ({a_sd:.3f}) "
            f"$\\oplus$ noise ({m_sd:.3f})",
            fontsize=7, color=fs.COMM, ha="left", va="top")
    a2.set_xlabel("clock step the single clock is restricted to (MHz)")
    a2.set_ylabel("two-domain gain (%, log)")
    a2.set_title("(b) the win is granularity: it scales with the clock step",
                 loc="left", pad=4)
    fs.despine(a2)

    # ---- (c) with the measured build cost ---------------------------------
    t = np.array([float(r["latency_ms"]) for r in md])
    a3.axhline(0, color=fs.INK, lw=0.7, zorder=2)
    have = np.array([r["two_cost"] != "" for r in md])
    dc = np.array([float(r["two_cost"]) - float(r["one_15MHz_hi"])
                   if r["two_cost"] != "" else np.nan for r in md])
    a3.plot(t[have], dc[have], "-", color=fs.INK, lw=1.3, zorder=3)
    if (~have).any():
        a3.axvspan(t.min() - 0.01, t[have].min(), color=fs.GRID, alpha=0.8, zorder=0)
        a3.text(0.5 * (t.min() + t[have].min()), 100,
                "unreachable\nwith the\nmeasured\nkernel", fontsize=6.8,
                color=fs.MUTED, ha="center", va="center")
    a3.set_ylabel("two domains $-$ one clock (mJ)", labelpad=3, fontsize=7.5)
    a3.set_ylim(0, 200)
    a3.set_yticks([0, 100, 200])
    a3.set_xlabel("matched latency per MoE layer (ms)")
    a3.text(t.max(), 192, "(c) with the measured build cost  |  100 mJ per division",
            fontsize=6.8, color=fs.MUTED, ha="right", va="top")
    fs.despine(a3)
    # the difference cannot be positive: phi = 0 reproduces every single clock, so the
    # two-domain candidate set is a superset.

    fig.text(0.005, 0.99,
             "A100-SXM4-40GB  |  Qwen3-30B-A3B MoE layer, 7744 prefill tokens, real "
             "ShareGPT routing  |  (a) measured at a locked 1200 MHz.\n(b), (c) composed "
             "from the B1 clock sweep: at each deadline, the best two-domain split "
             "against the best single clock, both scored\non the same clock set under "
             "the same deadline rule. Latencies at or below the single-clock energy "
             "optimum (4.63 ms); what that\nrestriction costs the two-domain side is "
             "measured, not assumed, and reported in data/two_domain_summary.csv.",
             fontsize=7, color=fs.INK, va="top", ha="left", linespacing=1.35)
    fig.text(0.005, 0.004,
             "(b) thins the MEASURED grid, so nothing in it is interpolated. Each faint "
             f"line is one of the {len(anchors)} anchors the thinning can start from. "
             "The exponent's uncertainty combines that\nanchor spread with an exhaustive "
             f"bootstrap over the 6 passes, which is the larger; across all {n_res} "
             f"distinct resamples and all {len(anchors)} anchors it never exceeded "
             f"{e_max:.2f}.\nTwo domains win 41 of the 42 deadlines and tie at exactly "
             "one: the fastest, where no latency slack is left to convert into energy.",
             fontsize=7, color=fs.MUTED, va="bottom", ha="left")
    print(fs.save(fig, "fig18_two_domain_verdict"))


if __name__ == "__main__":
    main()
