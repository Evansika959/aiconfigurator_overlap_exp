#!/usr/bin/env python3
"""Put an uncertainty on every cell of the two-clock matrix.

WHY. Every cell is computed from single 2 s measurement windows. Repeating one cell ten
times showed its answer moves: -17.00 to -15.96, sd 0.30 pp. A matrix of point estimates
with no spread invites reading a 1 pp difference between neighbouring cells as real.

HOW. Not by repeating all 72 cells (hours). The ten repeats give the noise of ONE window
directly -- power CV 0.52%, latency CV 0.55% on solo kernels, 0.68%/0.43% on the
concurrent run. Propagate that through the ENTIRE selection by Monte Carlo: perturb every
input measurement independently, then redo the baseline pick and the matched-latency
two-clock search from scratch. The selection is a min over candidates, so noise can flip
which clock wins -- an analytic error bar would miss that; resampling does not.

VALIDATION. The one cell with ten real repeats must come out at its measured sd. If the
Monte Carlo says 0.30 pp there, it is trustworthy on the other 71. If it does not, the
noise model is wrong and the whole table should be thrown away.
"""
import csv, glob, json, os, random, statistics as st
import followup_baseline as FB

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = 400.0
CLOCKS = [300, 510, 705, 900, 1200, 1410]
# The first version drew every measurement independently and returned 0.82 pp where ten
# real repeats give 0.30 -- 2.7x too wide. Decomposing the repeat table shows why: 89% of
# the POWER variance is common-mode, a single session-wide factor (thermal state, ambient)
# that moves every reading in a run together. The reported quantity is a RATIO of
# energies, so that factor very nearly cancels; only the independent residual survives.
# Latency is the other way round -- 23% common, mostly independent -- but it enters only
# through the latency constraint, not the ratio.
CV_P_COMMON, CV_P_INDEP = 0.0050, 0.0023     # measured, data/repeat/*
CV_T_COMMON, CV_T_INDEP = 0.0028, 0.0052
DRAWS = 400
random.seed(20260828)


def load():
    rows = []
    for f in sorted(glob.glob(os.path.join(HERE, "data", "commbound_*mib.csv"))):
        mib = int(os.path.basename(f).split("_")[1].replace("mib.csv", ""))
        for r in csv.DictReader(open(f)):
            r["_mib"] = mib
            rows.append(r)
    return rows


def cell(Xc, Gc, Cc, grid, jitter):
    """Baseline pick + matched-latency two-clock min, on possibly perturbed inputs."""
    if jitter:
        kp, kt = random.gauss(0, CV_P_COMMON), random.gauss(0, CV_T_COMMON)
        jp = lambda v: v * (1 + kp + random.gauss(0, CV_P_INDEP))
        jt = lambda v: v * (1 + kt + random.gauss(0, CV_T_INDEP))
    else:
        jp = jt = lambda v: v
    meas = []
    for f in CLOCKS:
        if f not in Xc or f not in Gc or f not in Cc:
            continue
        P, T = jp(Xc[f][0]), jt(Xc[f][1])
        if P > CAP:
            continue
        meas.append((P * T, T))
    if not meas:
        return None
    Bm = FB.pick(meas, "T")
    two = []
    for fg in Gc:
        g = dict(power_per_gpu_w=jp(Gc[fg][0]), iter_ms=jt(Gc[fg][1]))
        for fc in Cc:
            c = dict(power_per_gpu_w=jp(Cc[fc][0]), iter_ms=jt(Cc[fc][1]),
                     ctas=Cc[fc][2])
            ti, E, pk = FB.compose(g, c, fg, fc, grid)
            if E / ti > CAP or ti > Bm[1] * 1.0001:
                continue
            two.append(E)
    if not two:
        return None
    return (min(two) - Bm[0]) / Bm[0] * 100


def main():
    rows = load()
    held = lambda r: r["clock_held"] == "True"
    G = {(int(r["clock"]), int(r["m"]), int(r["n"])):
         (float(r["power_per_gpu_w"]), float(r["iter_ms"]))
         for r in rows if r["mode"] == "gemm_only" and held(r)}
    C = {(r["_mib"], int(r["clock"]), int(r["ctas"])):
         (float(r["power_per_gpu_w"]), float(r["iter_ms"]), int(r["ctas"]))
         for r in rows if r["mode"] == "comm_only" and held(r)}
    X = {(r["_mib"], int(r["clock"]), int(r["ctas"]), int(r["m"]), int(r["n"])):
         (float(r["power_per_gpu_w"]), float(r["iter_ms"]))
         for r in rows if r["mode"] == "concurrent"}
    grids = {(int(r["m"]), int(r["n"])): int(r["grid"])
             for r in rows if r["mode"] == "concurrent"}
    cells = sorted({(k[0], k[3], k[4], k[2]) for k in X})

    out = []
    for (mib, m, n, c) in cells:
        Xc = {f: X[(mib, f, c, m, n)] for f in CLOCKS if (mib, f, c, m, n) in X}
        Gc = {f: G[(f, m, n)] for f in CLOCKS if (f, m, n) in G}
        Cc = {f: C[(mib, f, c)] for f in CLOCKS if (mib, f, c) in C}
        pt = cell(Xc, Gc, Cc, grids[(m, n)], False)
        if pt is None:
            continue
        d = [v for v in (cell(Xc, Gc, Cc, grids[(m, n)], True) for _ in range(DRAWS))
             if v is not None]
        out.append(dict(mib=mib, m=m, nk=n, ctas=c, dE_pct=round(pt, 2),
                        sd_pp=round(st.stdev(d), 3),
                        p05=round(sorted(d)[int(.05 * len(d))], 2),
                        p95=round(sorted(d)[int(.95 * len(d))], 2), draws=len(d)))

    p = os.path.join(HERE, "data", "fig7_err.csv")
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys())); w.writeheader()
        w.writerows(out)

    v = [r for r in out if (r["mib"], r["m"], r["nk"], r["ctas"]) == (128, 4096, 4096, 32)][0]
    reps = json.load(open(os.path.join(HERE, "data", "repeat_savings.json"))) \
        if os.path.exists(os.path.join(HERE, "data", "repeat_savings.json")) else None
    print("VALIDATION on the one cell measured 10 times "
          "(128 MiB, 4096x4096, 32 CTA)")
    print(f"  Monte Carlo : {v['dE_pct']:+.2f}%  sd {v['sd_pp']:.2f} pp")
    print(f"  10 real runs: -16.43%  sd 0.30 pp   (range -17.00 .. -15.96)")
    sds = [r["sd_pp"] for r in out]
    print(f"\n{len(out)} cells   sd: median {st.median(sds):.2f} pp   "
          f"p90 {sorted(sds)[int(.9*len(sds))]:.2f}   max {max(sds):.2f} pp")
    print(f"  wrote {p}")


if __name__ == "__main__":
    main()
