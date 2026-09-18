#!/usr/bin/env python3
"""fig4 -- energy per operation is C*V^2, and below the knee there is no V left to give up.

For compute-bound work, finishing W operations takes T = W/f, so

    E = P_dyn * T = (C V^2 f)(W/f) = C V^2 W

f cancels: energy per operation is C*V^2 and does not depend on frequency. Only VOLTAGE
saves energy. kappa = P_dyn/f is therefore a direct measurement of V^2 against f, and a
flat kappa means the rail is sitting on its floor.

PROVENANCE, because the first version of this figure asserted more than the data supports:

  * two independent fits of a(f) are plotted, not one. The published coefficients
    (overlap_exp/data/coefficients.csv) and a fresh least-squares refit of the SAME raw
    1080 rows (gemm_dcfs_char, squat kernel pinning SM count, 12 shapes x 5 SM counts x
    3 reps x 6 clocks). They differ by a constant 11.2% in level -- a normalisation
    difference, since the published fit used the wave model's S_eff where the refit uses
    the measured probe SM count -- and agree on SHAPE, which is the entire argument.
  * the knee is drawn as the BAND 1020-1080 MHz, not a line. It comes from a different
    and finer sweep (voltage_knee.csv, 29 clocks at 15 MHz steps): kappa stays within 1%
    of its floor to 1035 and first exceeds +5% at 1080.
  * the 1410 point is the weakest in the set -- 118 surviving rows against 180 elsewhere
    because throttling removed the rest, R^2 0.954 -- so the saving from dropping off it
    is given as a range across the two fits, not a single number.
  * 510 MHz is the lowest clock in the fine sweep. Flatness below that rests on one
    coarse-grid point, so the shaded floor is drawn from 510.
"""
import csv, os
import numpy as np
import matplotlib.pyplot as plt
import figstyle as F

HERE = os.path.dirname(os.path.abspath(__file__))
OE = os.path.join(HERE, "..", "overlap_exp", "data")
GC = os.path.join(HERE, "..", "gemm_dcfs_char", "data")


def published():
    R = list(csv.DictReader(open(os.path.join(OE, "coefficients.csv"))))
    d = sorted((int(r["clock_mhz"]), float(r["value"]) * 1e3)
               for r in R if r["coefficient"] == "kappa")
    return np.array([x[0] for x in d], float), np.array([x[1] for x in d])


def refit():
    """P_dyn = a(f) * S + b(f, shape); the per-shape intercept IS the footprint term."""
    R = [r for r in csv.DictReader(open(os.path.join(GC, "gemm_dynamic_energy.csv")))
         if r["clock_held"] == "True" and r["status"] == "ok" and r["probe_ok"] == "True"]
    fs, ks, ns = [], [], []
    for f in sorted({int(r["freq_req_mhz"]) for r in R}):
        rows = [r for r in R if int(r["freq_req_mhz"]) == f]
        sh = sorted({(int(r["m"]), int(r["n"]), int(r["k"])) for r in rows})
        ix = {s: i for i, s in enumerate(sh)}
        X = np.zeros((len(rows), 1 + len(sh))); y = np.zeros(len(rows))
        for i, r in enumerate(rows):
            X[i, 0] = float(r["probe_sm_count"])
            X[i, 1 + ix[(int(r["m"]), int(r["n"]), int(r["k"]))]] = 1.0
            y[i] = float(r["p_dynamic_w"])
        a = np.linalg.lstsq(X, y, rcond=None)[0][0]
        fs.append(f); ks.append(a / f * 1e3); ns.append(len(rows))
    return np.array(fs, float), np.array(ks), ns


def fine():
    import statistics as st
    R = [r for r in csv.DictReader(open(os.path.join(OE, "voltage_knee.csv")))
         if r["clock_held"] == "True"]
    by = {}
    for r in R:
        f = int(r["clock"])
        by.setdefault(f, []).append((float(r["p_poll"]) - float(r["idle_w"])) / f * 1000)
    by = {f: st.median(v) for f, v in by.items() if len(v) >= 3}
    SM = 108        # the `sm` column of that campaign: the kernel had the whole GPU
    return (np.array(sorted(by), float),
            np.array([by[f] for f in sorted(by)]) / SM)


fp, kp = published()
fr, kr, nr = refit()
ff, kf = fine()

F.use()
fig, (a1, a2) = plt.subplots(1, 2, figsize=(F.TEXT_W, 2.4), gridspec_kw=dict(wspace=0.30))
fig.subplots_adjust(bottom=0.22)

SER = [(fp, kp, "o-", F.GEMM, "published fit"),
       (fr, kr, "s--", F.MUTED, "independent refit, same raw rows"),
       (ff, kf, "-", F.COMM, "15 MHz sweep, separate campaign")]

# ---- (a) physical units: the three agree on level too, to about 12% ------------------
F.despine(a1)
for f, k, st_, c, lab in SER:
    a1.plot(f, k, st_, ms=3.4, lw=1.2, color=c, mec="white", mew=0.5, label=lab, zorder=4)
a1.set_xlabel("SM clock $f$ (MHz)")
a1.set_ylabel("$\\kappa = a(f)/f$\n(mW per SM per MHz)")
a1.set_title("(a) as measured", fontsize=7.4, pad=3)
a1.legend(fontsize=5.5, loc="upper left", handlelength=1.5)
a1.set_xlim(280, 1470)
lv = [k[f == 900][0] for f, k, *_ in SER]
a1.annotate(f"levels differ {(max(lv)-min(lv))/np.mean(lv)*100:.0f}%\n"
            "(different definitions of $S$)", xy=(890, 1.60), xytext=(1080, 1.98),
            fontsize=5.8, color=F.INK, ha="center",
            arrowprops=dict(arrowstyle="->", lw=0.6, color=F.INK))

# ---- (b) divided by each curve's own 900 MHz value ----------------------------------
F.despine(a2)
a2.axvspan(510, 1020, color=F.BASE, alpha=0.45, lw=0, zorder=0)
a2.axvspan(1020, 1080, color=F.COMM, alpha=0.16, lw=0, zorder=0)
for f, k, st_, c, lab in SER:
    a2.plot(f, k / k[f == 900][0], st_, ms=3.4, lw=1.2, color=c, mec="white", mew=0.5,
            zorder=4)
a2.axhline(1.0, color=F.INK, lw=0.6, ls=(0, (3, 2)), zorder=1)
a2.set_xlabel("SM clock $f$ (MHz)")
a2.set_ylabel("$\\kappa(f)\\,/\\,\\kappa(900)$")
a2.set_title("(b) each divided by its own 900 MHz value", fontsize=7.4, pad=3)
a2.text(760, 1.10, "V at its floor", fontsize=6.4, ha="center", color=F.INK)
a2.text(1050, 1.62, "knee", fontsize=6.2, ha="center", color=F.COMM, fontweight="bold")
lo = min(kp[fp == 1410][0] / kp[fp == 900][0], kr[fr == 1410][0] / kr[fr == 900][0])
hi = max(kp[fp == 1410][0] / kp[fp == 900][0], kr[fr == 1410][0] / kr[fr == 900][0])
a2.text(1150, 1.93, f"$-${(1-1/hi)*100:.0f} to $-${(1-1/lo)*100:.0f}%", fontsize=7.2,
        ha="center", color=F.COMM, fontweight="bold")
k510 = kf[ff == 510][0]; k900f = kf[ff == 900][0]
a2.text(700, 0.875, f"$-${(1-k510/k900f)*100:.1f}%", fontsize=7.2, ha="center",
        color=F.MUTED, fontweight="bold")
a2.set_xlim(280, 1470); a2.set_ylim(0.80, 2.15)

CAP = (
 "$\\kappa(f) \\equiv a(f)/f$, where $a(f) = \\partial P_\\mathrm{dyn}/\\partial S$ is the "
 "per-busy-SM dynamic power slope in W per SM, obtained by regressing measured dynamic\n"
 "power on the number of busy SMs at fixed $f$. One SM draws $P_\\mathrm{dyn} = CV^2f$, so "
 "$\\kappa = CV^2$: $C$ is fixed by the silicon, hence $\\kappa$ IS a measurement of $V^2$, "
 "and equally\nof the energy per unit of work, since $W$ operations cost "
 "$E = P_\\mathrm{dyn}T = (CV^2f)(W/f) = CV^2W$ with $f$ cancelling.\n"
 "WHY (b) DIVIDES BY 900 MHz: the three curves estimate the same quantity but count $S$ "
 "differently -- the published fit uses the wave model's $S_\\mathrm{eff}$, the refit the "
 "measured\nprobe SM count, the sweep the whole GPU -- so they sit at slightly different "
 "levels, as (a) shows. The claim is about SHAPE (flat, then rising), which a constant "
 "factor\ncannot change, so (b) removes it. 900 MHz is the reference because it is the "
 "highest clock still clearly on the voltage floor; any point in the flat band gives the "
 "same picture.\nSources: 1080-row squat-kernel sweep (12 GEMM shapes $\\times$ 5 SM counts "
 "$\\times$ 3 reps $\\times$ 6 clocks), fitted as $P_\\mathrm{dyn} = a(f)S + b(f,\\mathrm{shape})$; "
 "and a 29-clock 15 MHz sweep.\nKnee band 1020-1080: $\\kappa$ is within 1% of its floor to "
 "1035 and first exceeds +5% at 1080. The 1410 point is the weakest (118 surviving rows "
 "vs 180, $R^2$ 0.954), hence a range.")
fig.text(0.5, -0.02, CAP, ha="center", va="top", fontsize=5.5, color=F.INK,
         linespacing=1.5)
print(F.save(fig, "fig4_voltage_floor"))
