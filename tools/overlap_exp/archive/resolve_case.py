#!/usr/bin/env python3
"""Settle the one cell fig7 and fig13 disagreed on, using medians of 10 repeats.

WHY. The two figures used the same method -- same baseline clock, same winning pair, same
matched-latency rule -- and still reported -16.71 vs -13.66. Tracing it: the baseline's
POWER reading differed 3.3% between two runs of the identical configuration while its
latency agreed to 0.26%. The instrument's own repeatability is 1.25% median / 4.55% max
on power, so one 2 s window cannot resolve a 3 pp difference in the answer. Both sides are
therefore re-measured 10x and reduced by median.

BOTH SIDES, not just the baseline: the composed arm drifted 0.52% too, because the solo
GEMM row it reads was itself a single window.

Output: data/case3.json (rebuilt from medians) and the corrected saving with its spread.
"""
import csv, glob, json, os, statistics as st
import followup_baseline as FB
import table_432 as t

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = 400.0
CLOCKS = [300, 510, 705, 900, 1200, 1410]
M, N, CT, MIB = 4096, 4096, 32, 128

# ---- baseline: 10 real overlapped runs, locked 1200 ---------------------------------
S = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(HERE, "data/repeat/spans_*.json")))]
E_reps = [s["power_per_gpu_w"] * s["T_loop"] for s in S]
med = lambda k: st.median([s[k] for s in S])
print(f"baseline, {len(S)} repeats of the identical configuration")
print(f"  power   {st.median([s['power_per_gpu_w'] for s in S]):7.2f} W   "
      f"spread {(max(s['power_per_gpu_w'] for s in S) - min(s['power_per_gpu_w'] for s in S)) / st.mean([s['power_per_gpu_w'] for s in S]) * 100:.2f}%")
print(f"  T       {med('T_loop'):7.4f} ms  "
      f"spread {(max(s['T_loop'] for s in S) - min(s['T_loop'] for s in S)) / st.mean([s['T_loop'] for s in S]) * 100:.2f}%")
print(f"  energy  {st.median(E_reps):7.2f} mJ  "
      f"spread {(max(E_reps) - min(E_reps)) / st.mean(E_reps) * 100:.2f}%")

# ---- solo kernels: median over the same 10 repeats ----------------------------------
acc = {}
for f in sorted(glob.glob(os.path.join(HERE, "data/repeat/solo_*.csv"))):
    for r in csv.DictReader(open(f)):
        if r["clock_held"] != "True":
            continue
        acc.setdefault((r["mode"], int(r["clock"])), []).append(
            (float(r["power_per_gpu_w"]), float(r["iter_ms"])))
solo = {k: dict(power_per_gpu_w=st.median(p for p, _ in v),
                iter_ms=st.median(l for _, l in v), ctas=CT, n=len(v))
        for k, v in acc.items()}
G = {f: solo[("gemm_only", f)] for f in CLOCKS if ("gemm_only", f) in solo}
C = {f: solo[("comm_only", f)] for f in CLOCKS if ("comm_only", f) in solo}
print(f"\nsolo kernels, median of {min(v['n'] for v in solo.values())}-"
      f"{max(v['n'] for v in solo.values())} repeats;  gemm clocks held: {sorted(G)}")

rows = list(csv.DictReader(open(os.path.join(HERE, "data", "commbound_128mib.csv"))))
grid = int(next(r for r in rows if r["mode"] == "concurrent"
                and (int(r["m"]), int(r["n"])) == (M, N))["grid"])

Tb, Eb = med("T_loop"), st.median(E_reps)
two = []
for fg in G:
    for fc in C:
        ti, E, pk = FB.compose(G[fg], C[fc], fg, fc, grid)
        if E / ti > CAP or ti > Tb * 1.0001:
            continue
        two.append((E, ti, fg, fc))
E2, T2, fg, fc = min(two)
print(f"\n  baseline  1200 MHz      T {Tb:.4f} ms   E {Eb:.2f} mJ")
print(f"  two-clock {fg}/{fc} MHz  T {T2:.4f} ms   E {E2:.2f} mJ")
print(f"  SAVING  {(E2 - Eb) / Eb * 100:+.2f}%")
# what a single-window answer could have been, given the same numerator
lo, hi = [(E2 - e) / e * 100 for e in (max(E_reps), min(E_reps))]
print(f"  had the baseline been ONE window: {lo:+.2f}% .. {hi:+.2f}%  "
      f"(the -16.7 / -13.7 disagreement sits inside this)")

# ---- per-rep spread, so the figure can carry a real error bar ------------------------
# Each rep is recomputed end to end from ITS OWN measurements -- baseline and the composed
# arm alike -- so the spread is the spread of the answer, not of one input.
rows_ = list(csv.DictReader(open(os.path.join(HERE, "data", "commbound_128mib.csv"))))
grid_ = int(next(r for r in rows_ if r["mode"] == "concurrent"
                 and (int(r["m"]), int(r["n"])) == (M, N))["grid"])
per_b, per_c, per_Tb, per_Tc = [], [], [], []
for sf, cf in zip(sorted(glob.glob(os.path.join(HERE, "data/repeat/solo_*.csv"))),
                  sorted(glob.glob(os.path.join(HERE, "data/repeat/spans_*.json")))):
    s_ = json.load(open(cf))
    Tb_, Eb_ = s_["T_loop"], s_["power_per_gpu_w"] * s_["T_loop"]
    Gr, Cr = {}, {}
    for r in csv.DictReader(open(sf)):
        if r["clock_held"] != "True":
            continue
        dd = dict(power_per_gpu_w=float(r["power_per_gpu_w"]),
                  iter_ms=float(r["iter_ms"]), ctas=CT)
        (Gr if r["mode"] == "gemm_only" else Cr)[int(r["clock"])] = dd
    cand = []
    for a_ in Gr:
        for b_ in Cr:
            ti_, E_, pk_ = FB.compose(Gr[a_], Cr[b_], a_, b_, grid_)
            if E_ / ti_ > CAP or ti_ > Tb_ * 1.0001:
                continue
            cand.append((E_, ti_))
    e_, t_ = min(cand)
    per_b.append(Eb_); per_c.append(e_); per_Tb.append(Tb_); per_Tc.append(t_)

# ---- rebuild the figure's json from the medians -------------------------------------
sc = med("T_loop") / med("T_events")
tg = (med("gemm_end") - med("gemm_start")) * sc
tc = (med("comm_end") - med("comm_start")) * sc
ps = t.lin(t.PS, 1200)
es = ps * Tb
dg = (G[1200]["power_per_gpu_w"] - ps) * G[1200]["iter_ms"]
dc = (C[1200]["power_per_gpu_w"] - ps) * C[1200]["iter_ms"]
eg, ec = (Eb - es) * dg / (dg + dc), (Eb - es) * dc / (dg + dc)
b = dict(fg=1200, fc=1200, tg=tg, tc=tc, T=Tb, ps=ps, pg=eg / tg, pc=ec / tc,
         E_meas=Eb, es=es, eg=eg, ec=ec, scale=sc, reps=len(S),
         E_lo=min(per_b), E_hi=max(per_b), T_lo=min(per_Tb), T_hi=max(per_Tb))
g, c = G[fg], C[fc]
psc = t.lin(t.PS, min(t.PS)) + (
    (108 - CT) * (t.lin(t.PS, fg) - t.lin(t.PS, min(t.PS)))
    + CT * (t.lin(t.PS, fc) - t.lin(t.PS, min(t.PS)))) / 108
pg = g["power_per_gpu_w"] - t.lin(t.PS, fg)
pc = c["power_per_gpu_w"] - t.lin(t.PS, fc)
cc = dict(fg=fg, fc=fc, tg=g["iter_ms"], tc=c["iter_ms"], T=T2, ps=psc, pg=pg, pc=pc,
          E=E2, es=psc * T2, eg=pg * g["iter_ms"], ec=pc * c["iter_ms"],
          E_lo=min(per_c), E_hi=max(per_c), T_lo=min(per_Tc), T_hi=max(per_Tc),
          sav_lo=min((y - x) / x * 100 for x, y in zip(per_b, per_c)),
          sav_hi=max((y - x) / x * 100 for x, y in zip(per_b, per_c)))
old = json.load(open(os.path.join(HERE, "data", "case3.json")))
json.dump(dict(a=old["a"], b=b, c=cc, meta=old["meta"]),
          open(os.path.join(HERE, "data", "case3.json"), "w"), indent=1)
print("\n  rewrote data/case3.json from the medians")
