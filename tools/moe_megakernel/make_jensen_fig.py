#!/usr/bin/env python3
"""fig7 -- why splitting the clock cannot finish the same work in the same time for less.

THE ARGUMENT, which is a proof once the curve is measured.

To do a fixed amount of work W in a fixed time T on S SMs, the SM-weighted MEAN frequency
is pinned: sum_d s_d f_d = W/(kT). Any set of domains meeting the deadline sits on that
constraint. Dynamic energy is E = T * sum_d s_d g(f_d), with g(f) = f*kappa(f) the per-SM
dynamic power. So the question is purely: for a fixed mean of f, is the mean of g(f)
smallest when all f are equal?

That is Jensen's inequality. It depends only on whether g is convex, and g is measured:
its slope runs from 1.23 mW/MHz/SM on the voltage floor to 8.02 near the top, a 6.5x rise.
(Local wobbles inside the floor are noise on a flat quantity, not real concavity.)

  g convex  =>  mean of g(f) >= g(mean of f),  equality only when all f_d are equal
             =>  ONE clock at the mean is optimal; any split costs more.

Below the knee kappa is flat, so g is linear and the split is exactly NEUTRAL. Crossing
the knee it is strictly worse: the fast domain's extra cost exceeds the slow one's saving.
"""
import csv, os
import numpy as np
import matplotlib.pyplot as plt
import figstyle as F

HERE = os.path.dirname(os.path.abspath(__file__))
d = {}
for r in csv.DictReader(open(os.path.join(HERE, "data", "kappa_sweep.csv"))):
    if r["workload"] == "moe_expert_n7652" and r["held"] == "True":
        d[int(r["clock"])] = float(r["kappa_mw_per_mhz"]) / 108.0
FF = np.array(sorted(d), float)
KK = np.array([d[int(x)] for x in FF])
g = lambda f: f * np.interp(f, FF, KK) / 1000.0        # per-SM dynamic power, W

F.use()
fig, (a1, a2) = plt.subplots(1, 2, figsize=(F.TEXT_W, 2.45),
                             gridspec_kw=dict(wspace=0.45))
fig.subplots_adjust(bottom=0.22)

# ---- (a) the Jensen picture on the measured curve ----
x = np.linspace(700, 1390, 300)
a1.plot(x, g(x), "-", lw=1.5, color=F.INK, zorder=3, label="measured $P_\\mathrm{dyn}$ per SM")
SPL = np.array([800., 1000., 1380.])
fbar = SPL.mean(); gbar = g(SPL).mean()
a1.plot(SPL, g(SPL), "o", ms=5, color=F.GEMM, mec="white", mew=0.7, zorder=6)
a1.plot([SPL[0], SPL[-1]], [g(SPL[0]), g(SPL[-1])], "--", lw=0.9, color=F.GEMM,
        alpha=0.8, zorder=4)
a1.plot([fbar], [gbar], "D", ms=5.5, color=F.GEMM, mec="white", mew=0.7, zorder=7)
a1.plot([fbar], [g(fbar)], "s", ms=5.5, color=F.COMM, mec="white", mew=0.7, zorder=7)
a1.annotate("", xy=(fbar, g(fbar)), xytext=(fbar, gbar),
            arrowprops=dict(arrowstyle="<->", lw=1.1, color=F.MUTED))
a1.text(fbar + 22, (gbar + g(fbar)) / 2, f"+{(gbar-g(fbar))/g(fbar)*100:.0f}%",
        fontsize=7.2, color=F.MUTED, fontweight="bold", va="center")
for f_ in SPL:
    a1.text(f_, g(f_) - 0.16, f"{f_:.0f}", fontsize=6, ha="center", color=F.GEMM)
a1.text(fbar, gbar + 0.10, "three domains\n(their average)", fontsize=6, ha="center",
        color=F.GEMM, va="bottom")
a1.text(fbar + 35, g(fbar) - 0.14, f"one clock at {fbar:.0f} MHz", fontsize=6,
        ha="left", color=F.COMM)
F.despine(a1)
a1.set_xlabel("SM clock $f$ (MHz)")
a1.set_ylabel("$P_\\mathrm{dyn}$ per SM (W)")
a1.set_title("(a) the curve bends up, so the average\nsits above it", fontsize=7.2, pad=3)
a1.set_xlim(700, 1400)

# ---- (b) every split tested, matched to its own uniform equivalent ----
CASES = [((800, 1000, 1380), (36, 36, 36)), ((900, 1050, 1200), (36, 36, 36)),
         ((900, 1380), (78, 30)), ((700, 800, 900), (36, 36, 36)),
         ((800, 900, 1050), (36, 36, 36)), ((1050, 1050, 1050), (36, 36, 36))]
lab, val = [], []
for fs, ss in CASES:
    fs = np.array(fs, float); ss = np.array(ss, float)
    fb = (ss * fs).sum() / ss.sum()
    val.append(((ss * g(fs)).sum() - ss.sum() * g(fb)) / (ss.sum() * g(fb)) * 100)
    lab.append("/".join(str(int(v)) for v in fs))
col = [F.GEMM if v > 0.5 else F.BASE for v in val]
F.despine(a2)
a2.barh(range(len(val)), val, height=0.6, color=col, edgecolor="white", lw=0.6, zorder=3)
a2.set_yticks(range(len(val))); a2.set_yticklabels(lab, fontsize=6)
a2.invert_yaxis()
a2.axvline(0, color=F.INK, lw=0.7)
for i, v in enumerate(val):
    a2.text(v + 0.7, i, f"{v:+.1f}%", va="center", fontsize=6.2,
            color=F.GEMM if v > 0.5 else F.MUTED, fontweight="bold")
a2.set_xlabel("extra power vs. one clock,\nsame mean frequency (%)", labelpad=1)
a2.set_title("(b) splits that stay below the knee are\nfree; splits that cross it are not",
             fontsize=7.2, pad=3)
a2.set_xlim(-3, 34)
fig.text(0.5, -0.02, "Same work, same time, same 108 SMs $\\Rightarrow$ the SM-weighted "
         "mean frequency is fixed. Energy is $T\\sum_d s_d\\,g(f_d)$ with "
         "$g(f)=f\\,\\kappa(f)$, so by Jensen a\nsplit can only beat the uniform clock if "
         "$g$ is concave. Measured, $g$'s slope rises 1.23 $\\to$ 8.02 mW/MHz/SM: convex. "
         "$\\kappa$ from data/kappa_sweep.csv, per SM.",
         ha="center", va="top", fontsize=5.6, color=F.INK, linespacing=1.5)
print(F.save(fig, "fig7_jensen"))
