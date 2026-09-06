#!/usr/bin/env python3
"""Two-clock improvement computed from the MEASURED sweep, not from the fitted model.

WHY REDO THIS. The earlier two-domain study ran entirely on fitted coefficients, so
every number inherited the model's error (3.3% median on power, 5.4% on latency) on top
of whatever the counterfactual itself assumes. But data/overlap_432.csv already contains
the two kernels measured separately at all six clocks. Composing those measurements
directly removes the model from everything except the one step that genuinely cannot be
measured -- running the two kernels at different clocks at the same time.

WHAT IS MEASURED AND WHAT IS NOT

    t_gemm(f), P_gemm(f)      measured, `gemm_only` rows, 108 SMs
    t_comm(f, c), P_comm(f,c) measured, `comm_only` rows
    P_static(f)               measured, from the GEMM calibration campaign
    the composition rule      NOT measured -- but it IS validated below against the 432
                              measured `concurrent` rows, at f_gemm == f_comm, which is
                              the only setting the hardware can actually produce

THE VALIDATION IS THE POINT. If composing two measured solo kernels reproduces the
measured concurrent row whenever the two clocks are equal, then the same composition at
f_gemm != f_comm is credible to about the same tolerance. That is a far stronger footing
than "the model says so", and it is checked first, before any counterfactual is quoted.

THROTTLED ROWS ARE DROPPED FROM THE CANDIDATE SET. 39 concurrent and 11 gemm_only rows
ran below their requested clock because the 400 W cap pulled them down. A measurement
taken at 1185 MHz is not the value at 1410 MHz, so those operating points are not
offered to the optimiser. The iteration-mean cap is applied to composed candidates too.

  python3 twoclock_measured.py
"""
import argparse
import csv
import os
import statistics as st

import table_432 as t          # only for P_static(f); every other quantity is measured

HERE = os.path.dirname(os.path.abspath(__file__))
TOTAL_SM = 108
CAP_W = 400.0
CLOCKS = [300, 510, 705, 900, 1200, 1410]


def load():
    rows = list(csv.DictReader(open(os.path.join(HERE, "data", "overlap_432.csv"))))
    held = lambda r: r["clock_held"] == "True"
    G = {(int(r["clock"]), int(r["m"]), int(r["n"])): r
         for r in rows if r["mode"] == "gemm_only" and held(r)}
    C = {(int(r["clock"]), int(r["ctas"])): r
         for r in rows if r["mode"] == "comm_only" and held(r)}
    X = {(int(r["clock"]), int(r["ctas"]), int(r["m"]), int(r["n"])): r
         for r in rows if r["mode"] == "concurrent"}
    return rows, G, C, X


def compose(g, c, fg, fc, grid):
    """Compose two measured solo kernels into one overlapped iteration.

    Same soft-partition form validated at 3.3% median on this dataset: the GEMM is
    re-waved onto the SMs the collective is not holding, charged only for the fraction
    of the GEMM's duration the collective is actually resident. Dynamic powers are the
    measured solo powers with the static floor removed, so the floor is counted once.
    """
    tg, tc = float(g["iter_ms"]), float(c["iter_ms"])
    ncta = int(c["ctas"])
    w = -(-grid // TOTAL_SM)
    w2 = -(-grid // max(TOTAL_SM - ncta, 1))
    t_iter = max(tg * (1 + (w2 / w - 1) * min(1, tc / tg)), tc)
    pg = float(g["power_per_gpu_w"]) - t.PS[fg]        # dynamic only
    pc = float(c["power_per_gpu_w"]) - t.PS[fc]
    # two voltage domains: chip-wide floor plus each domain's leakage by SM share
    floor = t.PS[min(t.PS)]
    dg, dc = t.PS[fg] - floor, t.PS[fc] - floor
    ps = floor + ((TOTAL_SM - ncta) * dg + ncta * dc) / TOTAL_SM
    E = ps * t_iter + pg * tg + pc * tc
    return t_iter, E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-cap", action="store_true", help="ignore the 400 W mean cap")
    a = ap.parse_args()
    rows, G, C, X = load()
    shapes = sorted({(int(r["m"]), int(r["n"]), int(r["k"]), int(r["grid"]))
                     for r in rows if r["mode"] == "concurrent"})
    ctas = sorted({int(r["ctas"]) for r in rows if r["mode"] == "concurrent"})

    # ---- 1. does composing measured solo kernels reproduce the measured overlap? ----
    et, ee = [], []
    for (m, n, k, grid) in shapes:
        for c in ctas:
            for f in CLOCKS:
                g, cm, x = G.get((f, m, n)), C.get((f, c)), X.get((f, c, m, n))
                if not (g and cm and x) or x["clock_held"] != "True":
                    continue
                ti, E = compose(g, cm, f, f, grid)
                tm, Em = float(x["iter_ms"]), float(x["power_per_gpu_w"]) * float(x["iter_ms"])
                et.append(abs(ti - tm) / tm); ee.append(abs(E - Em) / Em)
    et.sort(); ee.sort()
    print(f"1. VALIDATION -- compose two measured solo kernels, compare with the measured")
    print(f"   concurrent row, at f_gemm == f_comm.  n = {len(et)}\n")
    print(f"     latency: median {st.median(et)*100:5.2f}%   p90 {et[int(.9*len(et))]*100:5.2f}%"
          f"   max {max(et)*100:5.2f}%")
    print(f"     energy : median {st.median(ee)*100:5.2f}%   p90 {ee[int(.9*len(ee))]*100:5.2f}%"
          f"   max {max(ee)*100:5.2f}%")
    print(f"\n   The two-clock numbers below inherit roughly this tolerance, and nothing")
    print(f"   more -- every input to them is a measurement.\n")

    # ---- 2. the optimisation, both arms through the same composition ----------------
    def cands(m, n, grid, c, two):
        out = []
        for fg in CLOCKS:
            g = G.get((fg, m, n))
            if not g:
                continue
            for fc in (CLOCKS if two else [fg]):
                cm = C.get((fc, c))
                if not cm:
                    continue
                ti, E = compose(g, cm, fg, fc, grid)
                if not a.no_cap and E / ti > CAP_W:
                    continue
                out.append((E, ti, fg, fc))
        return out

    def pick(cs, obj):
        if obj == "EDP":
            return min(cs, key=lambda z: z[0] * z[1])
        p, s = (0, 1) if obj == "E" else (1, 0)
        b = min(z[p] for z in cs)
        return min([z for z in cs if z[p] <= b * (1 + 1e-9)], key=lambda z: z[s])

    # CTA count is FIXED per row in the sweep, so the fixed-c view below includes
    # configurations nobody would deploy -- 1 CTA against a 64 MiB message makes the
    # collective 25x the GEMM, which inflates what a second clock appears to buy. The
    # deployment question is "choosing c freely, what does a second clock buy", so that
    # is computed first and reported as the headline.
    free = []
    for (m, n, k, grid) in shapes:
        one = [z for c in ctas for z in cands(m, n, grid, c, False)]
        two = [z for c in ctas for z in cands(m, n, grid, c, True)]
        if not one or not two:
            continue
        row = dict(m=m, nk=n, k=k, grid=grid, ar_mib=64)
        for obj in ("E", "T", "EDP"):
            B, D = pick(one, obj), pick(two, obj)
            row.update({f"{obj}_1clk_mJ": round(B[0], 2), f"{obj}_1clk_ms": round(B[1], 4),
                        f"{obj}_2clk_mJ": round(D[0], 2), f"{obj}_2clk_ms": round(D[1], 4),
                        f"{obj}_dE_pct": round((D[0] - B[0]) / B[0] * 100, 2),
                        f"{obj}_dT_pct": round((D[1] - B[1]) / B[1] * 100, 2)})
        free.append(row)
    pf = os.path.join(HERE, "data", "twoclock_measured_ctafree.csv")
    with open(pf, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(free[0].keys()))
        wr.writeheader(); wr.writerows(free)
    print(f"2. CTA CHOSEN FREELY -- the deployment question. {len(free)} GEMM shapes, "
          f"64 MiB\n")
    print(f"   {'objective':>12}{'median dE':>12}{'best dE':>10}{'median dT':>12}")
    for obj, name in (("E", "min energy"), ("T", "min latency"), ("EDP", "min EDP")):
        print(f"   {name:>12}{st.median(x[f'{obj}_dE_pct'] for x in free):>+11.2f}%"
              f"{min(x[f'{obj}_dE_pct'] for x in free):>+9.2f}%"
              f"{st.median(x[f'{obj}_dT_pct'] for x in free):>+11.2f}%")
    print(f"   wrote {pf}\n")

    out = []
    for (m, n, k, grid) in shapes:
        for c in ctas:
            one, two = cands(m, n, grid, c, False), cands(m, n, grid, c, True)
            if not one or not two:
                continue
            g900, c900 = G.get((900, m, n)), C.get((900, c))
            ratio = float(c900["iter_ms"]) / float(g900["iter_ms"]) if g900 and c900 else None
            r = dict(m=m, nk=n, k=k, grid=grid, ctas=c, ar_mib=64,
                     tc_over_tg=round(ratio, 4) if ratio else "")
            for obj in ("E", "T", "EDP"):
                B, D = pick(one, obj), pick(two, obj)
                r.update({f"{obj}_1clk_mJ": round(B[0], 2), f"{obj}_1clk_ms": round(B[1], 4),
                          f"{obj}_1clk_f": B[2],
                          f"{obj}_2clk_mJ": round(D[0], 2), f"{obj}_2clk_ms": round(D[1], 4),
                          f"{obj}_2clk_fg": D[2], f"{obj}_2clk_fc": D[3],
                          f"{obj}_dE_pct": round((D[0] - B[0]) / B[0] * 100, 2),
                          f"{obj}_dT_pct": round((D[1] - B[1]) / B[1] * 100, 2)})
            out.append(r)

    p = os.path.join(HERE, "data", "twoclock_measured.csv")
    with open(p, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        wr.writeheader(); wr.writerows(out)

    print(f"3. CTA HELD FIXED -- {len(out)} measured (shape, CTA) pairs, 64 MiB.")
    print(f"   Answers a narrower question: given you are ALREADY at this CTA count,")
    print(f"   what does a second clock buy. Larger numbers, because the low-CTA rows")
    print(f"   are configurations a planner would never have picked.\n")
    bins = [(0, 0.5, "0 - 0.5"), (0.5, 1.0, "0.5 - 1"), (1.0, 2.0, "1 - 2"),
            (2.0, 1e9, "> 2")]
    for obj, name in (("E", "min energy"), ("T", "min latency"), ("EDP", "min EDP")):
        print(f"   {name}")
        print(f"     {'t_c/t_g':>10}{'n':>4}{'median dE':>12}{'best dE':>10}{'median dT':>12}")
        for lo, hi, lab in bins:
            v = [x for x in out if x["tc_over_tg"] != "" and lo <= x["tc_over_tg"] < hi]
            if not v:
                continue
            print(f"     {lab:>10}{len(v):>4}"
                  f"{st.median(x[f'{obj}_dE_pct'] for x in v):>+11.2f}%"
                  f"{min(x[f'{obj}_dE_pct'] for x in v):>+9.2f}%"
                  f"{st.median(x[f'{obj}_dT_pct'] for x in v):>+11.2f}%")
        v = out
        print(f"     {'ALL':>10}{len(v):>4}"
              f"{st.median(x[f'{obj}_dE_pct'] for x in v):>+11.2f}%"
              f"{min(x[f'{obj}_dE_pct'] for x in v):>+9.2f}%"
              f"{st.median(x[f'{obj}_dT_pct'] for x in v):>+11.2f}%\n")
    print(f"   wrote {p}  ({len(out)} rows)")


if __name__ == "__main__":
    main()
