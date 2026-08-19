#!/usr/bin/env python3
"""Score the 432 concurrent overlap cases against the composed GEMM+collective model.

EVERY COEFFICIENT IS REFITTED HERE FROM THE TWO SOURCE CAMPAIGNS. Nothing is
hardcoded. The earlier verify script pinned numbers at 1200 MHz because that was the
only clock the overlap runs covered; this sweep covers all six, and b/alpha/p0/gamma
were never written down away from 1200. Refitting also means a correction to either
campaign propagates instead of silently going stale.

    ../gemm_dcfs_char/data/gemm_dynamic_energy.csv   1080 rows -> a(f), b(f, N*K), P_static(f)
    ../comm_dvfs_char/data/comm_trtllm_ar_ctas.csv    288 rows -> B(c,f), alpha(c,f), p0(f), gamma(f)

THE MODEL, per GPU:

    P(t) = P_static(f)                                   board floor, counted ONCE
         + a(f)*s_gemm(t) + b(f, N*K)                     GEMM, a staircase in time
         + p0(f) + gamma(f)*B(c,f)                        collective, a rectangle

    s_gemm(t) = S for the first W-1 waves, grid-(W-1)*S for the last;  W = ceil(grid/S)

NVML AVERAGES OVER A WINDOW FAR LONGER THAN ONE ITERATION, so the prediction compared
against it is the model's time-average over the MEASURED iteration length, including
the sync gap where neither kernel runs and only P_static is drawn. Comparing an
overlap-interval mean against a loop average over-predicted by 142 W once.

WHAT THIS CAN AND CANNOT SETTLE. It tests the time-average. The instantaneous peak --
the thing that decides whether a real overlap throttles -- is not observable with NVML
at 50 Hz against 0.3-70 ms kernels, and ncu serialises concurrent kernels so it cannot
see an overlap at all. Peaks below are model output, not measurement.

  python3 score_432.py
"""

import collections
import csv
import math
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
TOTAL_SM = 108


def fit(xs, ys):
    """least squares y = m*x + c"""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    m = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else 0.0
    return m, my - m * mx


# ---------------------------------------------------------------- GEMM coefficients
def gemm_coeffs():
    """P_dynamic = a(f)*S_eff + b(f, N*K). Fit a per clock across SM counts, then b per
    (clock, footprint) as the residual. The squatter sweep varied sm_avail, which is
    exactly the lever a needs."""
    rows = [r for r in csv.DictReader(
        open(os.path.join(HERE, "..", "gemm_dcfs_char", "data",
                          "gemm_dynamic_energy.csv")))
        if r["clock_held"] == "True" and r["status"] == "ok"]
    A, B, PS = {}, collections.defaultdict(dict), {}
    by_f = collections.defaultdict(list)
    for r in rows:
        by_f[int(r["freq_req_mhz"])].append(r)
    for f, rs in by_f.items():
        PS[f] = st.median(float(r["p_static_w"]) for r in rs)
        pts = []
        for r in rs:
            g = int(r["m"]) * int(r["n"]) // 32768
            sm = int(r["sm_avail"])
            w = -(-g // sm)
            pts.append((g / w, float(r["p_dynamic_w"])))
        A[f], _ = fit([p[0] for p in pts], [p[1] for p in pts])
        # b is the per-footprint intercept once a is fixed
        by_fp = collections.defaultdict(list)
        for r in rs:
            g = int(r["m"]) * int(r["n"]) // 32768
            sm = int(r["sm_avail"])
            w = -(-g // sm)
            mb = int(r["n"]) * int(r["k"]) * 2 / 2 ** 20
            by_fp[mb].append(float(r["p_dynamic_w"]) - A[f] * (g / w))
        for mb, v in by_fp.items():
            B[f][mb] = st.median(v)
    return A, dict(B), PS


def b_lookup(B, f, n, k):
    mb = n * k * 2 / 2 ** 20
    tab = sorted(B[f].items())
    if mb <= tab[0][0]:
        return tab[0][1]
    if mb >= tab[-1][0]:
        return tab[-1][1]
    for (x0, y0), (x1, y1) in zip(tab, tab[1:]):
        if x0 <= mb <= x1:
            return y0 + (y1 - y0) * (mb - x0) / (x1 - x0)


# ---------------------------------------------------------- collective coefficients
def comm_coeffs():
    """t = alpha(c,f) + S/B(c,f) per cell; then P_comm - P_static = p0(f) + gamma(f)*B."""
    rows = [r for r in csv.DictReader(
        open(os.path.join(HERE, "..", "comm_dvfs_char", "data",
                          "comm_trtllm_ar_ctas.csv")))
        if r["throttled"] == "False"]
    cell = collections.defaultdict(list)
    for r in rows:
        cell[(int(r["freq_mhz"]), int(r["nccl_ctas"]))].append(
            (float(r["msg_bytes"]) / 1e9, float(r["latency_ms"]) / 1e3,
             float(r["power_w"])))
    Bw, AL, PW = collections.defaultdict(dict), collections.defaultdict(dict), \
        collections.defaultdict(dict)
    for (f, c), v in cell.items():
        slope, icept = fit([x[0] for x in v], [x[1] for x in v])
        Bw[f][c] = 1 / slope
        AL[f][c] = icept * 1e6                      # microseconds
        PW[f][c] = st.median(x[2] for x in v)
    return dict(Bw), dict(AL), dict(PW)


def comm_power_fit(Bw, PW, PS):
    """p0(f), gamma(f) from P_comm(c) - P_static(f) regressed on B(c,f)."""
    P0, G = {}, {}
    for f in Bw:
        cs = sorted(Bw[f])
        g, p0 = fit([Bw[f][c] for c in cs], [PW[f][c] - PS[f] for c in cs])
        P0[f], G[f] = p0, g
    return P0, G


def main():
    A, B, PS = gemm_coeffs()
    Bw, AL, PW = comm_coeffs()
    P0, G = comm_power_fit(Bw, PW, PS)

    print("REFITTED COEFFICIENTS")
    print(f"  {'f':>6}{'a(f)':>9}{'P_static':>10}{'p0(f)':>8}{'gamma(f)':>10}"
          f"{'b range':>16}")
    for f in sorted(A):
        bs = sorted(B[f].values())
        print(f"  {f:>6}{A[f]:>9.4f}{PS[f]:>10.2f}{P0.get(f, 0):>8.1f}"
              f"{G.get(f, 0):>10.4f}   {bs[0]:>5.1f} .. {bs[-1]:>5.1f} W")

    rows = list(csv.DictReader(open(os.path.join(HERE, "data", "overlap_432.csv"))))
    by = {(r["mode"], int(r["ctas"]), int(r["clock"]), int(r["m"]), int(r["n"])): r
          for r in rows}
    gemm_only = {(int(r["clock"]), int(r["m"]), int(r["n"])): r
                 for r in rows if r["mode"] == "gemm_only"}
    comm_only = {(int(r["clock"]), int(r["ctas"])): r
                 for r in rows if r["mode"] == "comm_only"}

    conc = [r for r in rows if r["mode"] == "concurrent"]
    print(f"\n\nCONCURRENT CASES: {len(conc)}   "
          f"clock held on {sum(r['clock_held'] == 'True' for r in conc)}")

    recs = []
    for r in conc:
        f, c = int(r["clock"]), int(r["ctas"])
        m, n, k = int(r["m"]), int(r["n"]), int(r["k"])
        g, w = int(r["grid"]), int(r["waves"])
        s_eff = g / w
        go = gemm_only.get((f, m, n))
        co = comm_only.get((f, c))
        if not go or not co:
            continue
        tg, tc = float(go["iter_ms"]), float(co["iter_ms"])
        it = float(r["iter_ms"])
        p_g = A[f] * s_eff + b_lookup(B, f, n, k)
        p_c = P0[f] + G[f] * Bw[f][c]
        # energy per iteration / iteration length; P_static covers the whole period
        model = PS[f] + (p_g * tg + p_c * tc) / it
        meas = float(r["power_per_gpu_w"])
        ser = by.get(("serial", c, f, m, n))
        recs.append(dict(f=f, c=c, m=m, n=n, k=k, s_eff=s_eff, waves=w,
                         occ=s_eff / TOTAL_SM, tg=tg, tc=tc, it=it,
                         model=model, meas=meas, err=(model - meas) / meas,
                         speedup=float(ser["iter_ms"]) / it if ser else None,
                         ceil=(tg + tc) / max(tg, tc),
                         peak=PS[f] + A[f] * min(g, TOTAL_SM) + b_lookup(B, f, n, k) + p_c))

    e = [abs(x["err"]) for x in recs]
    e.sort()
    print(f"  power model, all {len(recs)}: median |err| {st.median(e) * 100:.2f}%   "
          f"p90 {e[int(.9 * len(e))] * 100:.2f}%   max {e[-1] * 100:.2f}%")

    def group(key, label, fmt="{}"):
        print(f"\n  by {label}:")
        d = collections.defaultdict(list)
        for x in recs:
            d[key(x)].append(abs(x["err"]))
        print(f"    {label:>10}{'n':>5}{'median':>9}{'max':>8}")
        for kk in sorted(d):
            print(f"    {fmt.format(kk):>10}{len(d[kk]):>5}"
                  f"{st.median(d[kk]) * 100:>8.2f}%{max(d[kk]) * 100:>7.2f}%")

    group(lambda x: x["f"], "clock")
    group(lambda x: x["c"], "CTA")
    group(lambda x: f"{x['occ'] * 100:.0f}%", "occupancy")
    group(lambda x: x["waves"], "waves")

    print("\n\n  worst 10 by |err|:")
    print(f"    {'f':>5}{'CTA':>4}{'M':>6}{'N=K':>7}{'occ':>6}{'t_g':>8}{'t_c':>8}"
          f"{'meas':>8}{'model':>8}{'err':>8}")
    for x in sorted(recs, key=lambda z: -abs(z["err"]))[:10]:
        print(f"    {x['f']:>5}{x['c']:>4}{x['m']:>6}{x['n']:>7}"
              f"{x['occ'] * 100:>5.0f}%{x['tg']:>8.3f}{x['tc']:>8.3f}"
              f"{x['meas']:>8.1f}{x['model']:>8.1f}{x['err'] * 100:>7.1f}%")

    print("\n\nOVERLAP PAYOFF (speedup = t_serial / t_concurrent)")
    sp = [x for x in recs if x["speedup"]]
    print(f"  median {st.median(x['speedup'] for x in sp):.3f}   "
          f"best {max(x['speedup'] for x in sp):.3f}   "
          f"worst {min(x['speedup'] for x in sp):.3f}")
    print(f"  fraction of the achievable ceiling: median "
          f"{st.median((x['speedup'] - 1) / max(x['ceil'] - 1, 1e-9) for x in sp) * 100:.0f}%")
    print(f"\n  top 10 by speedup:")
    print(f"    {'f':>5}{'CTA':>4}{'M':>6}{'N=K':>7}{'t_g':>8}{'t_c':>8}"
          f"{'serial':>8}{'conc':>8}{'speedup':>9}{'ceiling':>9}")
    for x in sorted(sp, key=lambda z: -z["speedup"])[:10]:
        print(f"    {x['f']:>5}{x['c']:>4}{x['m']:>6}{x['n']:>7}{x['tg']:>8.3f}"
              f"{x['tc']:>8.3f}{x['speedup'] * x['it']:>8.3f}{x['it']:>8.3f}"
              f"{x['speedup']:>9.3f}{x['ceil']:>9.3f}")

    print("\n\nMODEL-PREDICTED INSTANTANEOUS PEAK (not measurable -- model output)")
    print("  P_static + a*min(grid,108) + b + p_comm, i.e. a full GEMM wave with the")
    print("  collective resident. The 400 W board cap is what would throttle.")
    over = [x for x in recs if x["peak"] > 400]
    print(f"  max predicted peak {max(x['peak'] for x in recs):.0f} W   "
          f"cases above 400 W: {len(over)}/{len(recs)}")
    if over:
        print(f"    {'f':>5}{'CTA':>4}{'M':>6}{'N=K':>7}{'peak':>8}{'mean meas':>11}"
              f"{'clk held':>10}")
        for x in sorted(over, key=lambda z: -z["peak"])[:12]:
            r = by[("concurrent", x["c"], x["f"], x["m"], x["n"])]
            print(f"    {x['f']:>5}{x['c']:>4}{x['m']:>6}{x['n']:>7}{x['peak']:>8.0f}"
                  f"{x['meas']:>11.1f}{r['clock_held']:>10}")


if __name__ == "__main__":
    main()
