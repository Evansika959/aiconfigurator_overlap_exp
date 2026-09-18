#!/usr/bin/env python3
"""fig10 -- predicted saving of expert-to-frequency-domain routing vs a grouped GEMM.

The scheme: sort experts by routed token count, give the light ones to a low-frequency
region and the heavy ones to a high-frequency region, every region on the same deadline.
The baseline is a standard grouped GEMM -- one kernel, 108 SMs, one clock, work-conserving
-- allowed the cheapest clock that meets the same deadline.

This is a PREDICTION from the fitted model, not a measurement: no such kernel exists and
no A100 has per-region V/f. kappa and P_leak are measured; the schedule is modelled.

x is how tight the SLO is, read as the clock the baseline grouped GEMM must run at. The
grey band is the alternative that needs no new hardware at all: if the deadline can slip
1.37x, simply lowering that single clock from 1380 to 1005 saves 30%.
"""
import csv, os, collections
import numpy as np
import matplotlib.pyplot as plt
import figstyle as F

HERE = os.path.dirname(os.path.abspath(__file__))
R = list(csv.DictReader(open(os.path.join(HERE, "data", "expert_dvfs_pred.csv"))))
by = collections.defaultdict(dict)
for r in R:
    by[r["workload"]][int(r["baseline_clock"])] = (-float(r["gain2_pct"]),
                                                   -float(r["gain3_pct"]))
SHOW = [("decode_b64", "decode B=64  (5 waves)", F.GEMM),
        ("prefill_b16", "prefill B=16  (40 waves)", "#E09A4B"),
        ("prefill_b128", "prefill B=128  (306 waves)", F.COMM)]

F.use()
fig, (a1, a2) = plt.subplots(1, 2, figsize=(F.TEXT_W, 2.4),
                             gridspec_kw=dict(wspace=0.34, width_ratios=[1.25, 1]))
fig.subplots_adjust(bottom=0.22)

F.despine(a1)
a1.axhspan(0, 30.4, color=F.BASE, alpha=0.5, lw=0, zorder=0)
a1.text(1395, 28.6, "the alternative that needs no new hardware:\n"
        "lower the ONE clock 1380 $\\to$ 1005 $\\Rightarrow$ $-$30%,\n"
        "for 1.37$\\times$ the latency",
        fontsize=6, color=F.INK, va="top", ha="right")
for key, lab, col in SHOW:
    fs = sorted(by[key]); g2 = [by[key][f][0] for f in fs]
    a1.plot(fs, g2, "o-", ms=3.4, lw=1.4, color=col, mec="white", mew=0.5, label=lab)
a1.set_xlabel("SLO tightness: clock the baseline grouped GEMM must run at (MHz)")
a1.set_ylabel("energy saved vs. grouped GEMM\nat the same latency (%)")
a1.set_title("(a) two frequency domains", fontsize=7.4, pad=3)
a1.legend(fontsize=6, loc="upper left", handlelength=1.5,
          bbox_to_anchor=(-0.01, 0.72))
a1.set_ylim(-1, 33); a1.set_xlim(660, 1410)

F.despine(a2)
x = np.arange(len(SHOW))
g2 = [by[k][1380][0] for k, _, _ in SHOW]
g3 = [by[k][1380][1] for k, _, _ in SHOW]
a2.bar(x - 0.19, g2, 0.36, color=F.GEMM, edgecolor="white", lw=0.6, label="2 domains",
       zorder=3)
a2.bar(x + 0.19, g3, 0.36, color=F.COMM, edgecolor="white", lw=0.6, label="3 domains",
       zorder=3)
for i, (a_, b_) in enumerate(zip(g2, g3)):
    a2.text(i - 0.19, a_ + 0.25, f"{a_:.1f}", ha="center", fontsize=6.2)
    a2.text(i + 0.19, b_ + 0.25, f"{b_:.1f}", ha="center", fontsize=6.2)
a2.set_xticks(x)
a2.set_xticklabels(["decode\nB=64", "prefill\nB=16", "prefill\nB=128"], fontsize=6.2)
a2.set_ylabel("energy saved (%)")
a2.set_title("(b) at the tightest SLO (1380 MHz):\na third domain adds nothing",
             fontsize=7.4, pad=3)
a2.legend(fontsize=6, loc="upper right", handlelength=1.2)
a2.set_ylim(0, 13)

fig.text(0.5, -0.02, "PREDICTED, not measured. Tiles per expert "
         "$= \\lceil n_e/128 \\rceil \\times \\lceil 1024/128 \\rceil$ from the measured "
         "OLMoE routing; a domain of $s_d$ SMs holding $N_d$ tiles takes "
         "$\\lceil N_d/s_d \\rceil$\nwaves, so its clock is forced to the lowest 15 MHz "
         "grid point meeting the deadline; energy "
         "$= \\sum_d [\\,N_d \\kappa(f_d) + s_d P_\\mathrm{leak}(f_d) T\\,]$ with $\\kappa$ "
         "and $P_\\mathrm{leak}$ measured.\nSearch covers every contiguous grouping of "
         "load-sorted experts and every SM allocation. The optimum always puts the LIGHT "
         "experts on the low-frequency domain.",
         ha="center", va="top", fontsize=5.5, color=F.INK, linespacing=1.5)
print(F.save(fig, "fig10_expert_dvfs_prediction"))
