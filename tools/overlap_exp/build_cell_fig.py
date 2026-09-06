#!/usr/bin/env python3
"""Resolve one matrix cell from its 10 repeats and draw it as power x time.

Generalises resolve_case.py + make_case3_fig.py over the CTA count, so any cell of the
row can be opened up without editing two scripts. Everything else is unchanged: the
baseline is a REAL overlapped run at the tightest-SLA clock, the two-domain arm is
composed from the measured solo kernels and held to the baseline's own latency, and every
quantity is the median of ten independent repeats with the full range as its error bar.

  python3 build_cell_fig.py --ctas 4
"""
import argparse, csv, glob, json, os, statistics as st
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import followup_baseline as FB
import table_432 as t
import figstyle as F

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = 400.0
CLOCKS = [300, 510, 705, 900, 1200, 1410]

ap = argparse.ArgumentParser()
ap.add_argument("--ctas", type=int, required=True)
ap.add_argument("--m", type=int, default=4096)
ap.add_argument("--nk", type=int, default=4096)
ap.add_argument("--mib", type=int, default=128)
ap.add_argument("--dir", default=None)
a = ap.parse_args()
CT, M, N, MIB = a.ctas, a.m, a.nk, a.mib
D = a.dir or os.path.join(HERE, "data", f"repeat_cta{CT}")

rows = list(csv.DictReader(open(os.path.join(HERE, "data", f"commbound_{MIB}mib.csv"))))
grid = int(next(r for r in rows if r["mode"] == "concurrent"
                and (int(r["m"]), int(r["n"])) == (M, N))["grid"])

S, per_b, per_c, per_Tb, per_Tc, Gs, Cs = [], [], [], [], [], [], []
for sf, cf in zip(sorted(glob.glob(os.path.join(D, "solo_*.csv"))),
                  sorted(glob.glob(os.path.join(D, "spans_*.json")))):
    s = json.load(open(cf)); S.append(s)
    Tb, Eb = s["T_loop"], s["power_per_gpu_w"] * s["T_loop"]
    G, C = {}, {}
    for r in csv.DictReader(open(sf)):
        if r["clock_held"] != "True":
            continue
        d = dict(power_per_gpu_w=float(r["power_per_gpu_w"]),
                 iter_ms=float(r["iter_ms"]), ctas=CT)
        (G if r["mode"] == "gemm_only" else C)[int(r["clock"])] = d
    Gs.append(G); Cs.append(C)
    cand = [(E, ti, fg, fc)
            for fg in G for fc in C
            for ti, E, pk in [FB.compose(G[fg], C[fc], fg, fc, grid)]
            if E / ti <= CAP and ti <= Tb * 1.0001]
    e, ti, fg, fc = min(cand)
    per_b.append(Eb); per_c.append(e); per_Tb.append(Tb); per_Tc.append(ti)

med = lambda k: st.median([s[k] for s in S])
G = {f: dict(power_per_gpu_w=st.median(g[f]["power_per_gpu_w"] for g in Gs if f in g),
             iter_ms=st.median(g[f]["iter_ms"] for g in Gs if f in g))
     for f in CLOCKS if any(f in g for g in Gs)}
C = {f: dict(power_per_gpu_w=st.median(c[f]["power_per_gpu_w"] for c in Cs if f in c),
             iter_ms=st.median(c[f]["iter_ms"] for c in Cs if f in c), ctas=CT)
     for f in CLOCKS if any(f in c for c in Cs)}
Tb, Eb = med("T_loop"), st.median(per_b)
E2, T2, fg, fc = min((E, ti, x, y) for x in G for y in C
                     for ti, E, pk in [FB.compose(G[x], C[y], x, y, grid)]
                     if E / ti <= CAP and ti <= Tb * 1.0001)
print(f"CTA {CT}: baseline 1200 MHz  T {Tb:.3f}  E {Eb:.1f} mJ   |   "
      f"two-clock {fg}/{fc}  T {T2:.3f}  E {E2:.1f} mJ   -> {(E2-Eb)/Eb*100:+.2f}%")
print(f"  spread over {len(S)} repeats: "
      f"{max((y-x)/x*100 for x, y in zip(per_b, per_c)):+.2f} .. "
      f"{min((y-x)/x*100 for x, y in zip(per_b, per_c)):+.2f}")

sc = med("T_loop") / med("T_events")
ps = t.lin(t.PS, 1200)
dg = (G[1200]["power_per_gpu_w"] - ps) * G[1200]["iter_ms"]
dc = (C[1200]["power_per_gpu_w"] - ps) * C[1200]["iter_ms"]
es = ps * Tb
eg, ec = (Eb - es) * dg / (dg + dc), (Eb - es) * dc / (dg + dc)
tg = (med("gemm_end") - med("gemm_start")) * sc
tc = (med("comm_end") - med("comm_start")) * sc
P = [dict(fg=1200, fc=1200, tg=tg, tc=tc, T=Tb, ps=ps, pg=eg / tg, pc=ec / tc,
          E=Eb, es=es, eg=eg, ec=ec, T_lo=min(per_Tb), T_hi=max(per_Tb),
          E_lo=min(per_b), E_hi=max(per_b))]
psc = t.lin(t.PS, min(t.PS)) + (
    (108 - CT) * (t.lin(t.PS, fg) - t.lin(t.PS, min(t.PS)))
    + CT * (t.lin(t.PS, fc) - t.lin(t.PS, min(t.PS)))) / 108
pg = G[fg]["power_per_gpu_w"] - t.lin(t.PS, fg)
pc = C[fc]["power_per_gpu_w"] - t.lin(t.PS, fc)
P.append(dict(fg=fg, fc=fc, tg=G[fg]["iter_ms"], tc=C[fc]["iter_ms"], T=T2, ps=psc,
              pg=pg, pc=pc, E=E2, es=psc * T2, eg=pg * G[fg]["iter_ms"],
              ec=pc * C[fc]["iter_ms"], T_lo=min(per_Tc), T_hi=max(per_Tc),
              E_lo=min(per_c), E_hi=max(per_c),
              sav_lo=min((y - x) / x * 100 for x, y in zip(per_b, per_c)),
              sav_hi=max((y - x) / x * 100 for x, y in zip(per_b, per_c))))

F.use()
XMAX = max(p["T"] for p in P) * 1.12
YMAX = max(p["ps"] + p["pc"] + p["pg"] for p in P) * 1.38
fig, axes = plt.subplots(1, 2, figsize=(F.TEXT_W * 0.76, 2.35), sharey=True,
                         gridspec_kw=dict(wspace=0.10))
titles = [f"(a) baseline: one domain, {P[0]['fg']} MHz",
          f"(b) two domains, {P[1]['fg']} / {P[1]['fc']} MHz"]
for i, (ax, p, title) in enumerate(zip(axes, P, titles)):
    for w, y0, h, col, hat in ((p["T"], 0, p["ps"], F.BASE, F.HATCH["base"]),
                               (p["tc"], p["ps"], p["pc"], F.COMM, F.HATCH["comm"]),
                               (p["tg"], p["ps"] + p["pc"], p["pg"], F.GEMM,
                                F.HATCH["gemm"])):
        ax.add_patch(Rectangle((0, y0), w, h, facecolor=col, edgecolor="white",
                               linewidth=0.7, hatch=hat, zorder=3))
    ax.text(p["tg"] / 2, p["ps"] + p["pc"] + p["pg"] / 2, f"{p['eg']:.0f}", ha="center",
            va="center", color="white", fontsize=7, zorder=5)
    ax.text(p["tc"] * 0.62, p["ps"] + p["pc"] / 2, f"{p['ec']:.0f}", ha="center",
            va="center", color="white", fontsize=7, zorder=5)
    ax.text(p["T"] * 0.42, p["ps"] / 2, f"{p['es']:.0f}", ha="center", va="center",
            color=F.INK, fontsize=7, zorder=5)
    ax.axvline(p["T"], color=F.INK, lw=0.7, zorder=4)
    ax.axvspan(p["T_lo"], p["T_hi"], color=F.INK, alpha=0.16, lw=0, zorder=2)
    ax.text(0.04, 0.965, f"{p['E']:.0f} $\\pm$ {(p['E_hi']-p['E_lo'])/2:.0f} mJ",
            transform=ax.transAxes, fontsize=9, va="top", ha="left", fontweight="bold")
    if i:
        ax.text(0.04, 0.855, f"{(p['E']-P[0]['E'])/P[0]['E']*100:+.1f}%",
                transform=ax.transAxes, fontsize=9, va="top", ha="left",
                color=F.GEMM, fontweight="bold")
        ax.text(0.04, 0.752, f"({p['sav_hi']:+.1f} .. {p['sav_lo']:+.1f})",
                transform=ax.transAxes, fontsize=6.2, va="top", ha="left", color=F.GEMM)
    ax.text(0.96, 0.965, f"$T$ {p['T']:.2f} ms", transform=ax.transAxes, fontsize=6.8,
            va="top", ha="right", color=F.MUTED)
    ax.text(0.96, 0.885, f"$\\pm$ {(p['T_hi']-p['T_lo'])/2*1000:.0f} $\\mu$s",
            transform=ax.transAxes, fontsize=6, va="top", ha="right", color=F.MUTED)
    ax.text(0.96, 0.035, f"measured, {len(S)} repeats" if i == 0
            else f"inferred, {len(S)} repeats", transform=ax.transAxes, fontsize=6.5,
            va="bottom", ha="right", color=F.MUTED if i == 0 else F.GEMM, style="italic")
    ax.set_title(title, fontsize=7.2, pad=4)
    ax.set_xlim(0, XMAX); ax.set_ylim(0, YMAX)
    ax.set_xlabel("time (ms)")
    F.despine(ax, grid_axis="y")
axes[1].axvline(P[0]["T"], color=F.MUTED, lw=0.7, ls=(0, (3, 2)), zorder=4)
axes[1].text(P[0]["T"] - XMAX * 0.02, YMAX * 0.47, "(a) deadline", fontsize=6,
             color=F.MUTED, va="center", ha="center", rotation=90)
axes[0].set_ylabel("power (W)")
fig.suptitle(f"GEMM {M}x{N}x{N}  +  {MIB} MiB all-reduce, {CT} CTAs, world = 4\n"
             f"rectangle area = energy;  bars and shading = full range over "
             f"{len(S)} repeats", fontsize=7.5, y=1.10)
h = [Rectangle((0, 0), 1, 1, facecolor=c, hatch=k, edgecolor="white", linewidth=0.5)
     for c, k in ((F.GEMM, F.HATCH["gemm"]), (F.COMM, F.HATCH["comm"]),
                  (F.BASE, F.HATCH["base"]))]
fig.legend(h, ["GEMM", "all-reduce", "static floor"], loc="lower center", ncol=3,
           bbox_to_anchor=(0.5, -0.17))
print(F.save(fig, f"fig16_cell_{M}x{N}_{CT}cta"))
