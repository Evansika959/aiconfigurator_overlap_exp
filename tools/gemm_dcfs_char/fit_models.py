#!/usr/bin/env python3
"""Fit delay and power models for bf16 GEMM on A100 from the 1080-row sweep."""
import collections, csv, math, statistics, sys
import numpy as np

TILE_M, TILE_N = 256, 128          # torch 2.9.1+cu129, confirmed by ncu launch__grid_size
TOTAL_SM = 108
PEAK_MAC = 1024                    # A100 bf16 dense: 312 TFLOPS / 108 SM / 1.41 GHz
                                   # = 2049 FLOP/cycle/SM = 1024 MAC/cycle/SM
CSV = "/home/xinting/aiconfigurator_overlap_exp/tools/gemm_dcfs_char2/data/gemm_dynamic_energy.csv"

def tiles_of(m, n): return math.ceil(m/TILE_M) * math.ceil(n/TILE_N)

rows = []
for r in csv.DictReader(open(CSV)):
    if r["status"] != "ok" or r["clock_held"] != "True": continue
    m, n, k, s = int(r["m"]), int(r["n"]), int(r["k"]), int(r["sm_avail"])
    t = tiles_of(m, n); w = math.ceil(t/s)
    rows.append(dict(m=m, n=n, k=k, sm=s, f=int(r["clock_sm_med"]),
                     fbin=int(r["freq_req_mhz"]), tiles=t, waves=w,
                     s_eff=t/w, t_ms=float(r["latency_ms"]),
                     p_dyn=float(r["p_dynamic_w"]), e_dyn=float(r["energy_dynamic_mj"])))

def ols(X, y):
    X, y = np.asarray(X, float), np.asarray(y, float)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b; dof = max(1, len(y) - X.shape[1])
    cov = (res @ res / dof) * np.linalg.pinv(X.T @ X)
    return b, np.sqrt(np.diag(cov)), 1 - res@res/((y-y.mean())@(y-y.mean()))

def pct(p, o):
    e = np.abs(np.asarray(p)-np.asarray(o))/np.asarray(o)*100
    return np.median(e), np.percentile(e,90), e.max()

print(f"{len(rows)} rows (clock held), {len({(r['m'],r['n']) for r in rows})} shapes, "
      f"{len({r['f'] for r in rows})} clocks, {len({r['sm'] for r in rows})} SM points\n")

# ---------------- DELAY ----------------
print("="*78); print("DELAY   t = waves * (tile_M*tile_N*K)/(alpha*f) + t0"); print("="*78)
x = np.array([r["waves"]*r["k"]/r["f"] for r in rows]); t = np.array([r["t_ms"] for r in rows])
for lab, X in (("through origin", x.reshape(-1,1)),
               ("with intercept", np.vstack([x, np.ones_like(x)]).T)):
    b, se, r2 = ols(X, t); al = TILE_M*TILE_N/(1000*b[0])
    med, p90, mx = pct(X@b, t); t0 = b[1] if len(b) > 1 else 0.0
    print(f"  {lab:>15}: alpha={al:6.1f} MAC/cyc/SM ({al/PEAK_MAC*100:3.0f}% of peak)"
          f"  t0={t0*1e3:6.1f}us  R2={r2:.5f}  err med {med:4.1f}% p90 {p90:5.1f}% max {mx:5.1f}%")
xi = np.array([r["tiles"]/r["sm"]*r["k"]/r["f"] for r in rows])
b2,_,r2b = ols(xi.reshape(-1,1), t); med,p90,mx = pct(xi*b2[0], t)
print(f"  {'NO wave ceiling':>15}: (tiles/S instead of ceil)          "
      f"       R2={r2b:.5f}  err med {med:4.1f}% p90 {p90:5.1f}% max {mx:5.1f}%")

by = collections.defaultdict(list)
for r in rows: by[(r["m"],r["n"])].append(r)
print(f"\n  per-shape efficiency:\n     {'M':>6}{'N=K':>7}{'tiles':>7}{'alpha':>8}{'%peak':>7}{'err med':>9}")
als=[]
for kk in sorted(by):
    g=by[kk]; xs=np.array([r['waves']*r['k']/r['f'] for r in g]); ys=np.array([r['t_ms'] for r in g])
    bb,_,_=ols(xs.reshape(-1,1),ys); al=TILE_M*TILE_N/(1000*bb[0]); als.append(al)
    print(f"     {kk[0]:>6}{kk[1]:>7}{g[0]['tiles']:>7}{al:>8.1f}{al/PEAK_MAC*100:>6.0f}%{pct(xs*bb[0],ys)[0]:>8.1f}%")
print(f"     -> alpha {min(als):.0f}-{max(als):.0f} ({(max(als)-min(als))/statistics.median(als)*100:.0f}% spread)")

# ---------------- POWER ----------------
print("\n"+"="*78); print("POWER   P_dyn = a(f)*S_eff + b"); print("="*78)
byf = collections.defaultdict(list)
for r in rows: byf[r['fbin']].append(r)
print(f"\n  {'f':>6}{'a (W/SM)':>11}{'+/-':>8}{'b (W)':>8}{'+/-':>7}{'R2':>9}{'kappa=a/f':>11}{'V/V900':>9}")
kap={}; coef={}
for f in sorted(byf):   # first pass so kappa(900) exists for the ratio column
    g=byf[f]; X=np.vstack([[r['s_eff'] for r in g], np.ones(len(g))]).T
    b,se,r2 = ols(X,[r['p_dyn'] for r in g]); coef[f]=b; kap[f]=b[0]/f*1000
    v = math.sqrt(kap[f]/kap[900]) if 900 in kap else float('nan')
    print(f"  {f:>6}{b[0]:>11.4f}{se[0]:>8.4f}{b[1]:>8.2f}{se[1]:>7.2f}{r2:>9.5f}{kap[f]:>11.4f}{v:>9.3f}")

print(f"\n  b by weight footprint (L2 = 40 MB):\n     {'f':>6}" +
      "".join(f"{'N=K='+str(nk):>17}" for nk in (4096,8192,16384)))
for f in sorted(byf):
    cs=[]
    for nk in (4096,8192,16384):
        g=[r for r in byf[f] if r['n']==nk]
        X=np.vstack([[r['s_eff'] for r in g], np.ones(len(g))]).T
        b,se,_=ols(X,[r['p_dyn'] for r in g]); cs.append(f"{b[1]:6.1f} +/-{se[1]:4.1f}")
    print(f"     {f:>6}" + "".join(f"{c:>17}" for c in cs))

# ---------------- END TO END ----------------
print("\n"+"="*78); print("END TO END   E_dyn = P_dyn(model) * t(model)"); print("="*78)
bt,_,_ = ols(x.reshape(-1,1), t)
te,pe,ee = [],[],[]
for r in rows:
    tp = bt[0]*r['waves']*r['k']/r['f']; pp = coef[r['fbin']][0]*r['s_eff']+coef[r['fbin']][1]
    te.append(abs(tp-r['t_ms'])/r['t_ms']*100); pe.append(abs(pp-r['p_dyn'])/r['p_dyn']*100)
    ee.append(abs(tp*pp-r['e_dyn'])/r['e_dyn']*100)
for lab,e in (("latency",te),("dynamic power",pe),("dynamic ENERGY",ee)):
    print(f"  {lab:>15}: median {np.median(e):5.1f}%  p90 {np.percentile(e,90):5.1f}%  max {max(e):5.1f}%")
