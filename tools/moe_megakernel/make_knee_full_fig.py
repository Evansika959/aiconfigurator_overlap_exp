#!/usr/bin/env python3
"""fig14 -- the knee sweep extended to 1410 MHz, and what limits it.

The original campaign (voltage_knee.csv) stopped at 1245 MHz. It was not a choice about
frequency: the largest GEMM hit the 400 W board cap there and throttled off its requested
clock. Rerunning 1260-1410 (voltage_knee_hi.csv) shows exactly that -- the smallest shape
holds all the way to 1410 at 369 W, while the 4096 and 8192 shapes drop out at 1305 and
1290. Throttled points are plotted hollow because their kappa is computed against a clock
the GPU was not actually running.

The extension also answers whether kappa depends on the workload: its LEVEL does, by 1.7x
across these three shapes, because a bigger GEMM switches more capacitance per SM. Its
SHAPE does not -- normalised, the three lie on one curve to within a few percent up to the
knee.
"""
import csv, collections, os
import numpy as np
import matplotlib.pyplot as plt
import figstyle as F

HERE = os.path.dirname(os.path.abspath(__file__))
OE = os.path.join(HERE, "..", "overlap_exp", "data")
SHAPES = [(1024, 4096, "$1024\\times4096^2$", F.COMM),
          (4096, 8192, "$4096\\times8192^2$", "#E09A4B"),
          (8192, 16384, "$8192\\times16384^2$", F.GEMM)]

rows = []
for fn in ("voltage_knee.csv", "voltage_knee_hi.csv"):
    p = os.path.join(OE, fn)
    if os.path.exists(p):
        rows += list(csv.DictReader(open(p)))
d = collections.defaultdict(dict)
for r in rows:
    # PER SM. This campaign runs the whole GPU, so the raw quantity is board-level;
    # kappa = C*V^2 is a per-SM quantity and only becomes comparable with the a(f)/f
    # fits (fig4) after dividing by the SM count.
    k = (float(r["p_poll"]) - float(r["idle_w"])) / int(r["clock"]) * 1000 / 108
    d[(int(r["m"]), int(r["n"]))][int(r["clock"])] = (
        k, r["clock_held"] == "True", float(r["p_poll"]),
        float(r["p_poll"]) * float(r["latency_ms"]))          # total energy per iteration

F.use()
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(F.TEXT_W, 2.05),
                                 gridspec_kw=dict(wspace=0.50))
fig.subplots_adjust(bottom=0.26, left=0.09, right=0.985)

for m, n, lab, col in SHAPES:
    dd = d[(m, n)]
    f = np.array(sorted(dd), float)
    k = np.array([dd[int(x)][0] for x in f])
    ok = np.array([dd[int(x)][1] for x in f])
    a1.plot(f[ok], k[ok], "-", lw=1.5, color=col, label=lab, zorder=4)
    a1.plot(f[~ok], k[~ok], "o", ms=2.8, mfc="none", mec=col, mew=0.8, zorder=3)
    b = dd[900][0]
    a2.plot(f[ok], k[ok] / b, "-", lw=1.5, color=col, zorder=4)
    a2.plot(f[~ok], k[~ok] / b, "o", ms=2.8, mfc="none", mec=col, mew=0.8, zorder=3)

for ax in (a1, a2, a3):
    F.despine(ax)
    ax.axvspan(510, 1035, color=F.BASE, alpha=0.45, lw=0, zorder=0)
    ax.axvspan(1050, 1080, color=F.COMM, alpha=0.16, lw=0, zorder=0)
    ax.set_xlabel("SM clock $f$ (MHz)")
    ax.set_xlim(490, 1425)
a1.set_ylabel("$\\kappa$  (mW per SM per MHz)")
a1.set_title("(a) as measured: levels differ 1.7$\\times$", fontsize=7.4, pad=3)
a1.legend(fontsize=5.5, loc="upper left", handlelength=1.2,
          labelspacing=0.25, borderpad=0.25)
a1.text(0.97, 0.42, "hollow =\nthrottled", transform=a1.transAxes,
        fontsize=5.8, color=F.MUTED, ha="right", va="top")
a2.set_ylabel("$\\kappa(f)\\,/\\,\\kappa(900)$")
a2.set_title("(b) normalised: one curve", fontsize=7.4, pad=3)
a2.axhline(1.0, color=F.INK, lw=0.6, ls=(0, (3, 2)), zorder=1)
kk = d[(1024, 4096)]
a2.annotate(f"1410: {kk[1410][0]/kk[900][0]:.2f}$\\times$",
            xy=(1408, kk[1410][0] / kk[900][0]), xycoords="data",
            xytext=(0.42, 0.90), textcoords="axes fraction",
            fontsize=6.4, color=F.COMM, fontweight="bold", ha="center",
            arrowprops=dict(arrowstyle="->", lw=0.7, color=F.COMM))
a2.text(760, 1.12, "V at its floor", fontsize=6, ha="center", color=F.INK)


# ---- (c) kappa alone cannot answer "how low should I go" ----------------------------
# kappa is the DYNAMIC energy per unit work. Total energy also carries static power, which
# is paid for the whole (longer) runtime:  E = kappa*W + P_static*T,  T ~ 1/f.
# So the total-energy optimum sits ABOVE the point where kappa stops improving.
for m, n, lab, col in SHAPES:
    dd = d[(m, n)]
    f = np.array(sorted(dd), float)
    ok = np.array([dd[int(x)][1] for x in f])
    E = np.array([dd[int(x)][3] for x in f])
    a3.plot(f[ok], E[ok] / np.nanmin(E[ok]), "-", lw=1.5, color=col, zorder=4)
    j = int(np.nanargmin(np.where(ok, E, np.inf)))
    a3.plot([f[j]], [1.0], "v", ms=4, color=col, mec="white", mew=0.5, zorder=6)
a3.set_ylabel("total energy / its own minimum")
a3.set_title("(c) total energy has a floor", fontsize=7.4, pad=3)
a3.set_ylim(0.97, 1.45)
a3.text(0.5, 0.97, "optimum 975-1020 MHz", transform=a3.transAxes, ha="center",
        va="top", fontsize=6, color=F.INK)





fig.text(0.5, -0.02,
    "$\\kappa \\equiv P_\\mathrm{dyn}/(fS) = CV^2$: campaign B, single GPU, all 108 SMs,\n"
    "one GEMM per curve, clocks locked at 15 MHz steps, $P_\\mathrm{idle}$ measured at the\n"
    "same clock. The sweep originally ended at 1245 MHz because the largest shape hit the\n"
    "400 W board cap there -- a POWER limit, not a frequency one: the smallest shape holds\n"
    "1410 at 369 W while the others throttle from 1290 and 1305. (c) is why $\\kappa$ alone\n"
    "cannot say how far to downclock: total energy is $\\kappa W + P_\\mathrm{static}T$ and "
    "the static\nterm grows as $f$ falls, so the measured total optimum (975-1020 MHz) sits "
    "ABOVE the\npoint where $\\kappa$ stops improving (705-975 MHz).",
    ha="center", va="top", fontsize=5.6, color=F.INK, linespacing=1.5)
print(F.save(fig, "fig14_knee_to_1410"))
