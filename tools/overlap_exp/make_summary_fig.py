#!/usr/bin/env python3
"""fig12 -- the three findings of the extension campaign, one panel each.

(a) WHY the split curve has a bottom.  The collective stops getting faster at ~32 CTAs
    while an overlapped iteration keeps getting slower, because past that point every
    extra CTA is taken from the GEMM and buys the collective nothing. Both series are
    measured, not composed.
(b) WHY the frequency optimum moved.  kappa = P_dyn/f is proportional to C*V^2, so it is
    flat while the voltage rail is flat and climbs once DVFS has to raise it. Measured at
    15 MHz resolution: flat to 1035, then up. The original six-clock grid had 900 and
    1200 and nothing between -- it straddled the knee, so every optimum snapped to a
    point that was never the real one.
(c) WHERE the optima actually land once the grid can express them. 168 of 216 use a
    clock the old grid did not contain.
"""
import csv, glob, os, collections, statistics as st
import matplotlib.pyplot as plt
import figstyle as F

HERE = os.path.dirname(os.path.abspath(__file__))
OLD = [300, 510, 705, 900, 1200, 1410]
NEW = [1050, 1305]

F.use()
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(F.TEXT_W, 1.95))
fig.subplots_adjust(wspace=0.46, bottom=0.30)

# ---- (a) the collective saturates ---------------------------------------------------
rows = []
for f in ("data/commbound_128mib.csv", "data/ext_128mib.csv"):
    rows += list(csv.DictReader(open(os.path.join(HERE, f))))
sel = lambda m: {int(r["ctas"]): float(r["iter_ms"]) for r in rows
                 if r["mode"] == m and int(r["clock"]) == 1200
                 and r["clock_held"] == "True"
                 and (m == "comm_only" or (int(r["m"]), int(r["n"])) == (1024, 4096))}
co, cc = sel("comm_only"), sel("concurrent")
F.despine(a1)
x = sorted(co)
a1.plot(x, [co[c] for c in x], "o-", ms=3, color=F.COMM, mec="white", mew=0.5,
        label="collective alone", zorder=3)
x2 = sorted(cc)
a1.plot(x2, [cc[c] for c in x2], "s-", ms=3, color=F.GEMM, mec="white", mew=0.5,
        label="overlapped iteration", zorder=3)
a1.set_xscale("log", base=2); a1.set_xticks([2, 8, 32, 64])
a1.set_xticklabels(["2", "8", "32", "64"])
a1.set_xlabel("CTAs = SMs to collective", labelpad=2)
a1.set_ylabel("latency (ms)", labelpad=2)
a1.set_title("(a) more CTAs stop helping", fontsize=7, pad=3)
a1.legend(fontsize=6, loc="lower left", handlelength=1.2)
# the finding lives in the last octave, where the full-range curve is visually flat.
# An inset shows it at its own scale rather than normalising the series against each
# other, which would hide that these are two different absolute latencies.
ins = a1.inset_axes([0.44, 0.46, 0.54, 0.38])
xz = [c for c in x if c >= 16]
ins.plot(xz, [co[c] for c in xz], "o-", ms=2.2, color=F.COMM, mec="white", mew=0.4)
xz2 = [c for c in x2 if c >= 16]
ins.plot(xz2, [cc[c] for c in xz2], "s-", ms=2.2, color=F.GEMM, mec="white", mew=0.4)
ins.set_xscale("log", base=2); ins.set_xticks([16, 32, 64])
ins.set_xticklabels(["16", "32", "64"], fontsize=5.5)
ins.tick_params(labelsize=5.5, length=1.8, pad=1)
for sp in ("top", "right"):
    ins.spines[sp].set_visible(False)
ins.text(0.5, 1.02, "zoom on 16-64", transform=ins.transAxes, ha="center",
         va="bottom", fontsize=5.5, color=F.MUTED)

# ---- (b) the voltage knee -----------------------------------------------------------
V = [r for r in csv.DictReader(open(os.path.join(HERE, "data/voltage_knee.csv")))
     if r["clock_held"] == "True"]
by = {}
for r in V:
    by.setdefault(int(r["clock"]), []).append(
        (float(r["p_poll"]) - float(r["idle_w"])) / int(r["clock"]) * 1000)
by = {f: st.median(v) for f, v in by.items() if len(v) >= 3}   # n<3 is one bad window
ks = sorted(by)
F.despine(a2)
a2.plot(ks, [by[f] for f in ks], "-", lw=1.1, color=F.INK, zorder=2)
a2.plot([f for f in OLD if f in by], [by[f] for f in OLD if f in by], "o", ms=3.4,
        color=F.MUTED, mec="white", mew=0.5, zorder=4, label="original grid")
a2.plot([f for f in NEW if f in by], [by[f] for f in NEW if f in by], "D", ms=3.8,
        color=F.GEMM, mec="white", mew=0.5, zorder=5, label="added")
a2.axvline(1035, color=F.COMM, lw=0.8, ls=(0, (3, 2)), zorder=1)
a2.annotate("knee\n1035 MHz", (1035, by[ks[0]]), xytext=(-3, 14),
            textcoords="offset points", ha="right", fontsize=6, color=F.COMM)
# this campaign stopped at 1230; 1305 and 1410 have no kappa point, so the "added"
# marker legitimately shows only 1050. Saying so beats an axis that quietly ends early.
a2.set_xlim(460, 1270)
# the kappa campaign stops at 1230, so 1305 and 1410 have no point here. That belongs
# in the caption, not as a text box fighting the curve for the same pixels.
a2.set_xlabel("clock (MHz)", labelpad=2)
a2.set_ylabel("$\\kappa = P_\\mathrm{dyn}/f$  (mW/MHz)", labelpad=2)
a2.set_title("(b) the old grid straddled the knee", fontsize=7, pad=3)
a2.legend(fontsize=6, loc="upper left", handlelength=1.2)

# ---- (c) where the optima land ------------------------------------------------------
R = list(csv.DictReader(open(os.path.join(HERE, "data/fixed_design.csv"))))
best = {}
for r in R:
    k = (r["ctas"], r["ar_mib"], r["m"], r["nk"]); e = float(r["energy_mj"])
    if k not in best or e < best[k][0]:
        best[k] = (e, int(r["f_gemm"]), int(r["f_comm"]))
cg = collections.Counter(v[1] for v in best.values())
cc2 = collections.Counter(v[2] for v in best.values())
clocks = sorted(set(cg) | set(cc2))
F.despine(a3)
w = 0.38
idx = range(len(clocks))
a3.bar([i - w / 2 for i in idx], [cg[f] for f in clocks], w, color=F.GEMM,
       hatch=F.HATCH["gemm"], edgecolor="white", lw=0.4, label="$f_\\mathrm{gemm}$")
a3.bar([i + w / 2 for i in idx], [cc2[f] for f in clocks], w, color=F.COMM,
       hatch=F.HATCH["comm"], edgecolor="white", lw=0.4, label="$f_\\mathrm{comm}$")
for i, f in enumerate(clocks):
    if f in NEW:
        a3.axvspan(i - 0.5, i + 0.5, color=F.BASE, alpha=0.5, zorder=0, lw=0)
a3.set_xticks(list(idx)); a3.set_xticklabels([str(f) for f in clocks], rotation=90,
                                             fontsize=6)
a3.set_xlabel("chosen clock (MHz)", labelpad=2)
a3.set_ylabel("times optimal (of 216)", labelpad=2)
a3.set_title("(c) optima land on the new clocks", fontsize=7, pad=3)
a3.legend(fontsize=6, loc="upper left", handlelength=1.2)
print(F.save(fig, "fig12_summary"))
