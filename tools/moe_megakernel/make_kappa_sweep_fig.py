#!/usr/bin/env python3
"""fig5 -- kappa is the voltage rail, not the workload: three workloads, one shape.

If kappa = P_dyn/f = C*V^2 really measures the supply rail, then changing the workload
changes C -- the effective switched capacitance -- and therefore the LEVEL, while leaving
the SHAPE of V(f) untouched. That is a falsifiable prediction and this is the test.

Measured in one pass (data/kappa_sweep.csv, 33 clocks, CUDA-graph timing, P_idle sampled
at every clock in the same thermal state):

  gemm_large        (4096 x 8192) @ (8192 x 8192) bf16 -- the control, the kind of shape
                    the original characterisation used
  moe_expert_n7652  (7652 x 2048) @ (2048 x 1024) bf16 -- a real expert GEMM, prefill
                    B=128 median load
  moe_expert_n955   ( 955 x 2048) @ (2048 x 1024) bf16 -- prefill B=16 median load

The levels span 2.2x. The knees land within 15 MHz of each other and the floors are flat
to 1.7-2.9%. So the claim "below the knee there is no voltage left to give up" is a
property of the silicon and carries to the MoE shapes, which is what had to be shown.
"""
import csv, collections, os
import numpy as np
import matplotlib.pyplot as plt
import figstyle as F

HERE = os.path.dirname(os.path.abspath(__file__))
REF = 870                      # 900 is not on the grid; nearest point clearly on the floor
NICE = {"gemm_large": "large GEMM  4096$\\times$8192$\\times$8192",
        "moe_expert_n7652": "MoE expert  $n_e$=7652",
        "moe_expert_n955": "MoE expert  $n_e$=955"}
COL = {"gemm_large": F.GEMM, "moe_expert_n7652": "#E09A4B", "moe_expert_n955": F.COMM}

by = collections.defaultdict(dict)
for r in csv.DictReader(open(os.path.join(HERE, "data", "kappa_sweep.csv"))):
    if r["held"] == "True":
        by[r["workload"]][int(r["clock"])] = float(r["kappa_mw_per_mhz"])

F.use()
fig, (a1, a2) = plt.subplots(1, 2, figsize=(F.TEXT_W, 2.4), gridspec_kw=dict(wspace=0.30))
fig.subplots_adjust(bottom=0.22)
for w in ("gemm_large", "moe_expert_n7652", "moe_expert_n955"):
    f = np.array(sorted(by[w]), float); k = np.array([by[w][int(x)] for x in f])
    a1.plot(f, k, "-", lw=1.3, color=COL[w], label=NICE[w], zorder=4)
    a2.plot(f, k / by[w][REF], "-", lw=1.3, color=COL[w], zorder=4)
F.despine(a1); F.despine(a2)
lv = [by[w][REF] for w in by]
a1.set_ylabel("$\\kappa = P_\\mathrm{dyn}/f$  (mW per MHz)")
a1.set_title(f"(a) as measured: levels span {max(lv)/min(lv):.1f}$\\times$",
             fontsize=7.4, pad=3)
a1.legend(fontsize=5.6, loc="upper left", handlelength=1.5)
a1.set_xlim(280, 1400)

a2.axvspan(510, 1050, color=F.BASE, alpha=0.45, lw=0, zorder=0)
a2.axvspan(1080, 1095, color=F.COMM, alpha=0.20, lw=0, zorder=0)
a2.axhline(1.0, color=F.INK, lw=0.6, ls=(0, (3, 2)), zorder=1)
a2.text(760, 1.13, "V at its floor", fontsize=6.4, ha="center", color=F.INK)
a2.text(1088, 1.55, "knee", fontsize=6.2, ha="center", color=F.COMM, fontweight="bold",
        rotation=90)
a2.set_ylabel(f"$\\kappa(f)\\,/\\,\\kappa({REF})$")
a2.set_title("(b) normalised: one curve", fontsize=7.4, pad=3)
a2.set_xlim(280, 1400)
for ax in (a1, a2):
    ax.set_xlabel("SM clock $f$ (MHz)")

rows = []
for w in ("gemm_large", "moe_expert_n7652", "moe_expert_n955"):
    d = by[w]; fl = [v for f_, v in d.items() if 510 <= f_ <= 1035]
    fo = np.mean(fl)
    knee = next((f_ for f_ in sorted(d) if f_ > REF and d[f_] > fo * 1.05), None)
    rows.append((NICE[w].split("  ")[0], np.ptp(fl) / fo * 100, knee,
                 d.get(1380, np.nan) / d[REF]))
CAP = ("$\\kappa \\equiv P_\\mathrm{dyn}/f = CV^2$: dynamic power (board power minus the "
       "idle floor sampled at that same clock) divided by frequency, which is the energy "
       "per unit of\nswitching and therefore a direct read of $V^2$. Changing the "
       "workload changes $C$ and so the level; only the rail can change the shape. "
       "Measured over 33 clocks in one\npass, timing by CUDA graph replay of 64 GEMMs so "
       "the launch gap cannot masquerade as kernel time. Flatness over 510-1035 MHz and "
       "the knee, per workload:\n"
       + "     ".join(f"{n}: $\\pm${s:.1f}%, knee {k} MHz, "
                      f"$\\kappa$(1380)/$\\kappa$({REF})={r:.2f}"
                      for n, s, k, r in rows if not np.isnan(r))
       + f"     {rows[0][0]}: $\\pm${rows[0][1]:.1f}%, knee {rows[0][2]} MHz "
         "(1380 throttled).")
fig.text(0.5, -0.02, CAP, ha="center", va="top", fontsize=5.5, color=F.INK,
         linespacing=1.5)
print(F.save(fig, "fig5_kappa_is_the_rail"))
