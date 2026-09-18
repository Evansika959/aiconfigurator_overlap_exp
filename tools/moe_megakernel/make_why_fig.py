#!/usr/bin/env python3
"""fig3 -- why a second V/f domain has nothing to sell in a MoE prefill layer.

The first version of this figure claimed "every stage wants the same clock". Plotting it
disproved that: the permute is strongly bandwidth bound (T x f spreads 3.19x, more than
the NCCL collective's 2.26x) and it genuinely prefers 705 MHz over 1050. Heterogeneity
EXISTS.

The real reason is the second panel. The stage that wants a different clock is 7.8% of the
layer's energy, and mis-clocking it costs 2.7% of that -- 0.20% of the layer. The 91%
that dominates the budget is compute bound (T x f spread 1.02x), so its optimum is pinned
to the silicon's kappa knee and cannot be moved by anything about the workload.

Heterogeneity is not absent. It is in the wrong place to pay for a voltage domain.
"""
import csv, collections, os
import numpy as np
import matplotlib.pyplot as plt
import figstyle as F

HERE = os.path.dirname(os.path.abspath(__file__))
S = collections.defaultdict(dict)
for r in csv.DictReader(open(os.path.join(HERE, "data", "stage_clock.csv"))):
    if r["held"] == "True":
        S[r["stage"]][int(r["clock"])] = (float(r["ms"]), float(r["energy_mj"]))
E = collections.defaultdict(dict)
for r in csv.DictReader(open(os.path.join(HERE, "data", "expert_clock.csv"))):
    if r["held"] == "True":
        E[(r["gemm"], int(r["m"]))][int(r["clock"])] = (float(r["ms"]),
                                                        float(r["energy_mj"]))
C = {}
for r in csv.DictReader(open(os.path.join(HERE, "data", "collective_ref.csv"))):
    C[int(r["clock"])] = (float(r["iter_ms"]), float(r["power_w"]) * float(r["iter_ms"]))

NICE = {"expert_gemm": "expert GEMM", "combine": "combine (scatter)",
        "permute": "permute (gather)", "router_gemm": "router GEMM"}
ORDER = ["expert_gemm", "combine", "permute", "router_gemm"]
CL = sorted(set.intersection(*[set(d) for d in S.values()]))
tot = {f: sum(S[s][f][1] for s in S) for f in CL}
GF = min(tot, key=tot.get)

F.use()
fig, (a1, a2) = plt.subplots(1, 2, figsize=(F.TEXT_W, 2.4))
fig.subplots_adjust(wspace=0.42, bottom=0.24)

# ---- (a) heterogeneity EXISTS ---------------------------------------------------------
F.despine(a1)
for k, lab, col, lw in (("expert_gemm", "expert GEMM (99.7% of FLOPs)", F.GEMM, 1.4),
                        ("permute", "permute (gather)", "#E09A4B", 1.4)):
    f = sorted(S[k]); v = np.array([S[k][c][0] * c / 1000 for c in f])
    a1.plot(f, v / v[0], "o-", ms=2.8, lw=lw, color=col, mec="white", mew=0.4,
            label=lab, zorder=4)
f = sorted(C); v = np.array([C[c][0] * c / 1000 for c in f])
a1.plot(f, v / v[0], "s--", ms=2.8, lw=1.2, color=F.COMM, mec="white", mew=0.4,
        label="NCCL all-reduce (overlap_exp)", zorder=4)
a1.axhline(1.0, color=F.INK, lw=0.6, ls=(0, (3, 2)), zorder=1)
a1.text(1400, 1.05, "perfect $1/f$", fontsize=6, ha="right", color=F.MUTED)
a1.set_xlabel("clock (MHz)")
a1.set_ylabel("$T \\times f$, normalised\n(flat $\\Rightarrow$ compute bound)")
a1.set_title("(a) heterogeneity exists", fontsize=7.2, pad=3)
a1.legend(fontsize=5.6, loc="upper left", handlelength=1.4)
a1.set_xlim(250, 1460); a1.set_ylim(0.9, 3.5)

# ---- (b) but it is in the wrong place --------------------------------------------------
# in-bar text for the two thin segments collided; the names go in a legend and only the
# segment that matters gets an annotation.
F.despine(a2, grid_axis="x")
share = [S[s_][GF][1] for s_ in ORDER]
best = [min(S[s_].items(), key=lambda kv: kv[1][1]) for s_ in ORDER]
cols = [F.GEMM, "#B8860B", "#E09A4B", F.BASE]
left, handles = 0, []
for s_, w, b, c in zip(ORDER, share, best, cols):
    h = a2.barh(0, w, left=left, height=0.42, color=c, edgecolor="white", lw=0.8,
                zorder=3, label=f"{NICE[s_]} -- wants {b[0]} MHz")
    handles.append(h)
    if w / sum(share) > 0.15:
        a2.text(left + w / 2, 0, f"{w/sum(share)*100:.0f}%", ha="center", va="center",
                fontsize=7, color="white", zorder=5, fontweight="bold")
    left += w
a2.set_xlim(0, sum(share) * 1.02); a2.set_ylim(-1.05, 0.62)
a2.set_yticks([])
a2.set_xlabel("energy of one prefill layer at the best\n"
              f"single clock ({GF} MHz), mJ", labelpad=1)
a2.set_title("(b) but it is in the wrong place", fontsize=7.2, pad=3)
gain = (sum(min(v[1] for v in S[s_].values()) for s_ in S) - tot[GF]) / tot[GF] * 100
a2.annotate(f"{share[2]/sum(share)*100:.0f}%", xy=(left - share[3] - share[2] / 2, -0.24),
            xytext=(sum(share) * 0.80, -0.52), fontsize=7, color=F.GEMM,
            fontweight="bold", ha="center",
            arrowprops=dict(arrowstyle="->", lw=0.7, color=F.GEMM))
a2.legend(fontsize=5.6, loc="lower left", bbox_to_anchor=(-0.02, -0.02),
          handlelength=1.1, labelspacing=0.28)
fig.text(0.5, -0.10, f"The one stage that wants a different clock is "
         f"{share[2]/sum(share)*100:.0f}% of the budget, and mis-clocking it costs 2.7% "
         f"of that.\nA perfect per-stage V/f assignment saves {abs(gain):.2f}% of the "
         f"layer.", ha="center", fontsize=6.5, color=F.GEMM, fontweight="bold")
print(F.save(fig, "fig3_why_blocked"))
