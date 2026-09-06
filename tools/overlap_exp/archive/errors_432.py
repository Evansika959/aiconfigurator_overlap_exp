#!/usr/bin/env python3
"""Per-case error for all 432 concurrent cases -> data/errors_432.csv

WHY THIS EXISTS SEPARATELY FROM score_432.py.  That script reports the power error
using MEASURED t_gemm, t_comm and t_concurrent as inputs -- all three read out of the
same CSV it is scoring.  That is a fair test of the power model in isolation, but it is
not a prediction: nothing about the timing was forecast.  A model that has to be handed
the answer's timing before it can state a wattage cannot be used to plan a deployment.

So three error columns are computed here, in increasing order of what the model has to
supply for itself:

  err_p_meas   power, given measured t_gemm / t_comm / t_iter.       <- what was reported
  err_t        latency, t_concurrent predicted from the two solo kernel times.
  err_p_pred   power, given PREDICTED timings.  Nothing measured enters except the
               shape, the clock and the CTA count.  This is the number that matters
               for planning, and it is the one that carries both models' errors.

THE LATENCY MODEL FOR THE OVERLAP.  Two kernels sharing 108 SMs with the collective on
a priority -3 stream.  The naive bound is max(t_gemm, t_comm); it is optimistic because
the collective's CTAs are taken away from the GEMM.  The form used here charges the
GEMM for the SMs it loses:

    t_gemm' = t_gemm * W'/W,  W' = ceil(grid / (108 - c))     the GEMM re-waved onto
    t_conc  = max(t_gemm', t_comm)                            the SMs left over

  python3 errors_432.py
"""

import collections
import csv
import math
import os
import statistics as st

import score_432 as s

HERE = os.path.dirname(os.path.abspath(__file__))
TOTAL_SM = 108


def pct(v):
    v = sorted(v)
    return (st.median(v) * 100, v[int(.9 * len(v))] * 100, max(v) * 100)


def main():
    A, B, PS = s.gemm_coeffs()
    Bw, AL, PW = s.comm_coeffs()
    P0, G = s.comm_power_fit(Bw, PW, PS)

    rows = list(csv.DictReader(open(os.path.join(HERE, "data", "overlap_432.csv"))))
    go = {(int(r["clock"]), int(r["m"]), int(r["n"])): r
          for r in rows if r["mode"] == "gemm_only"}
    co = {(int(r["clock"]), int(r["ctas"])): r for r in rows if r["mode"] == "comm_only"}
    se = {(int(r["clock"]), int(r["ctas"]), int(r["m"]), int(r["n"])): r
          for r in rows if r["mode"] == "serial"}

    out = []
    for r in [x for x in rows if x["mode"] == "concurrent"]:
        f, c = int(r["clock"]), int(r["ctas"])
        m, n, k = int(r["m"]), int(r["n"]), int(r["k"])
        g, w = int(r["grid"]), int(r["waves"])
        tg, tc = float(go[(f, m, n)]["iter_ms"]), float(co[(f, c)]["iter_ms"])
        it = float(r["iter_ms"])
        meas = float(r["power_per_gpu_w"])

        p_g = A[f] * (g / w) + s.b_lookup(B, f, n, k)
        p_c = P0[f] + G[f] * Bw[f][c]

        # -- latency: re-wave the GEMM onto the SMs the collective is not holding
        sm_left = max(TOTAL_SM - c, 1)
        w2 = -(-g // sm_left)
        tg2 = tg * w2 / w
        t_pred = max(tg2, tc)

        # -- power with measured timings, and with predicted timings
        p_meas = PS[f] + (p_g * tg + p_c * tc) / it
        p_pred = PS[f] + (p_g * tg + p_c * tc) / t_pred

        out.append(dict(
            clock=f, ctas=c, m=m, n=n, k=k, grid=g, waves=w,
            s_eff=round(g / w, 2), occ_pct=round(g / w / TOTAL_SM * 100, 1),
            clock_min=int(r["clock_min"]), throttled=r["clock_held"] != "True",
            t_gemm_ms=round(tg, 4), t_comm_ms=round(tc, 4),
            t_serial_ms=round(float(se[(f, c, m, n)]["iter_ms"]), 4),
            t_conc_meas_ms=round(it, 4), t_conc_pred_ms=round(t_pred, 4),
            err_t_pct=round((t_pred - it) / it * 100, 2),
            p_meas_w=round(meas, 2),
            p_model_measT_w=round(p_meas, 2),
            err_p_measT_pct=round((p_meas - meas) / meas * 100, 2),
            p_model_predT_w=round(p_pred, 2),
            err_p_predT_pct=round((p_pred - meas) / meas * 100, 2),
            speedup=round(float(se[(f, c, m, n)]["iter_ms"]) / it, 4)))

    path = os.path.join(HERE, "data", "errors_432.csv")
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        wr.writeheader(); wr.writerows(out)
    print(f"wrote {path}  ({len(out)} rows)\n")

    held = [x for x in out if not x["throttled"]]
    print(f"ERROR SUMMARY   all {len(out)} concurrent cases / "
          f"{len(held)} non-throttled\n")
    print(f"  {'quantity':>34}{'set':>16}{'median':>9}{'p90':>8}{'max':>8}")
    for lab, key in (("power, measured timings in", "err_p_measT_pct"),
                     ("latency, predicted", "err_t_pct"),
                     ("power, predicted timings in", "err_p_predT_pct")):
        for tag, rs in (("all 432", out), ("non-throttled", held)):
            a, b, c_ = pct([abs(x[key]) / 100 for x in rs])
            print(f"  {lab:>34}{tag:>16}{a:>8.2f}%{b:>7.2f}%{c_:>7.2f}%")
        print()

    # signed, to separate bias from scatter
    print(f"  {'quantity':>34}{'set':>16}{'mean signed':>13}  (bias)")
    for lab, key in (("power, measured timings in", "err_p_measT_pct"),
                     ("latency, predicted", "err_t_pct"),
                     ("power, predicted timings in", "err_p_predT_pct")):
        v = [x[key] for x in held]
        print(f"  {lab:>34}{'non-throttled':>16}{st.mean(v):>12.2f}%")

    print("\n\nLATENCY MODEL, by how much of the iteration the two kernels share")
    print("  ratio = min(t_gemm,t_comm)/max(t_gemm,t_comm); 1.0 = perfectly matched\n")
    bins = collections.defaultdict(list)
    for x in held:
        rr = min(x["t_gemm_ms"], x["t_comm_ms"]) / max(x["t_gemm_ms"], x["t_comm_ms"])
        bins[min(int(rr * 5), 4)].append(x)
    print(f"  {'ratio':>12}{'n':>5}{'|err_t| med':>13}{'|err_p| med':>13}"
          f"{'speedup med':>13}")
    for b_ in sorted(bins):
        v = bins[b_]
        print(f"  {f'{b_ * .2:.1f}-{b_ * .2 + .2:.1f}':>12}{len(v):>5}"
              f"{st.median(abs(x['err_t_pct']) for x in v):>12.2f}%"
              f"{st.median(abs(x['err_p_predT_pct']) for x in v):>12.2f}%"
              f"{st.median(x['speedup'] for x in v):>13.3f}")

    print("\n\nWORST 15 BY |err_p_predT| (non-throttled)")
    print(f"  {'f':>5}{'CTA':>4}{'M':>6}{'N=K':>7}{'occ':>6}{'t_g':>8}{'t_c':>8}"
          f"{'t_meas':>8}{'t_pred':>8}{'err_t':>8}{'P_meas':>8}{'P_pred':>8}{'err_P':>8}")
    for x in sorted(held, key=lambda z: -abs(z["err_p_predT_pct"]))[:15]:
        print(f"  {x['clock']:>5}{x['ctas']:>4}{x['m']:>6}{x['n']:>7}"
              f"{x['occ_pct']:>5.0f}%{x['t_gemm_ms']:>8.3f}{x['t_comm_ms']:>8.3f}"
              f"{x['t_conc_meas_ms']:>8.3f}{x['t_conc_pred_ms']:>8.3f}"
              f"{x['err_t_pct']:>7.1f}%{x['p_meas_w']:>8.1f}{x['p_model_predT_w']:>8.1f}"
              f"{x['err_p_predT_pct']:>7.1f}%")


if __name__ == "__main__":
    main()
