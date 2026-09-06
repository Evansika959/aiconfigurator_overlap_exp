#!/usr/bin/env python3
"""Write everything that currently exists only inside code out to CSV.

WHY. Three classes of result were living in Python and nowhere else:

  the fitted coefficients   refitted at import time by score_432 / table_432, so the
                            numbers quoted in every writeup had no file behind them
  the throttle matrix       printed by score_432.py and then lost
  the figure source data    computed inside make_figures.py, so a figure could not be
                            checked without rerunning the plotting code

That is fragile in a way this project has already been bitten by: grid12.json lived in a
temp directory and was deleted by a cleanup, taking the ncu-measured launch grids with
it. Anything a claim rests on belongs in data/.

Writes to data/:
  coefficients.csv     every fitted coefficient, long format, with its source campaign
  throttle_matrix.csv  per-case throttle prediction, mean vs peak criterion
  frontier.csv         the energy-latency frontier behind fig4
  paired.csv           the 48-workload paired comparison behind fig5

  python3 export_data.py
"""
import csv
import json
import os

import numpy as np

import score_432 as s
import table_432 as t
import optimise_overlap as O
import objectives as J

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data")
GEMM_SRC = "gemm_dcfs_char/data/gemm_dynamic_energy.csv (1080 rows)"
COMM_SRC = "comm_dvfs_char/data/comm_trtllm_ar_ctas.csv (288 rows)"


def w(name, rows):
    p = os.path.join(D, name)
    with open(p, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)
    print(f"  {name:<22} {len(rows):>5} rows")


def coefficients():
    r = []
    for f in sorted(t.A):
        r.append(dict(coefficient="a", clock_mhz=f, index="", index_kind="",
                      value=round(t.A[f], 5), unit="W per busy SM",
                      note="P_gemm = a*S_eff + b", source=GEMM_SRC))
        r.append(dict(coefficient="kappa", clock_mhz=f, index="", index_kind="",
                      value=round(t.A[f] / f, 8), unit="W per SM per MHz",
                      note="a/f = C*V^2, the voltage curve; flat below ~1020 MHz",
                      source=GEMM_SRC))
        r.append(dict(coefficient="P_static", clock_mhz=f, index="", index_kind="",
                      value=round(t.PS[f], 3), unit="W",
                      note="board floor, counted over the whole iteration",
                      source=GEMM_SRC))
        for mb, v in sorted(t.B[f].items()):
            r.append(dict(coefficient="b", clock_mhz=f, index=int(mb),
                          index_kind="N*K footprint MiB", value=round(v, 3), unit="W",
                          note="residual intercept once a is fixed; collinear with a",
                          source=GEMM_SRC))
        tau, t0 = t.LAT[f]
        extra = " EXTRAPOLATED: 33 of 36 rows throttled at 1410" if f == 1410 else ""
        r.append(dict(coefficient="tau", clock_mhz=f, index="", index_kind="",
                      value=round(tau, 12), unit="ms per wave per unit K",
                      note="t_gemm = W*(K/splitK)*tau + t0." + extra, source=GEMM_SRC))
        r.append(dict(coefficient="t0", clock_mhz=f, index="", index_kind="",
                      value=round(t0, 8), unit="ms",
                      note="fixed per-launch cost." + extra, source=GEMM_SRC))
        r.append(dict(coefficient="p0", clock_mhz=f, index="", index_kind="",
                      value=round(t.P0[f], 3), unit="W",
                      note="P_comm = p0 + gamma*B", source=COMM_SRC))
        r.append(dict(coefficient="gamma", clock_mhz=f, index="", index_kind="",
                      value=round(t.G[f], 5), unit="W per GB/s",
                      note="P_comm = p0 + gamma*B", source=COMM_SRC))
        for c in sorted(t.Bw[f]):
            r.append(dict(coefficient="B", clock_mhz=f, index=c, index_kind="NCCL CTAs",
                          value=round(t.Bw[f][c], 4), unit="GB/s algorithmic",
                          note="world=4; 1 CTA = 1 SM (ncu-verified); unmeasured above 32",
                          source=COMM_SRC))
            r.append(dict(coefficient="alpha", clock_mhz=f, index=c,
                          index_kind="NCCL CTAs", value=round(t.AL[f][c], 3),
                          unit="microseconds", note="t_comm = alpha + S/B",
                          source=COMM_SRC))
    r.append(dict(coefficient="eta_k", clock_mhz="", index="", index_kind="",
                  value=O.ETA_K, unit="dimensionless",
                  note="eta(S) = 1 + eta_k*(1 - S/108). Corrects the GEMM running slower "
                       "than perfect wave scaling at low occupancy; fitted out of sample "
                       "on the 14/27/54/81 SM rows, cuts latency error 6.62% -> 1.75%",
                  source=GEMM_SRC))
    r.append(dict(coefficient="power_cap", clock_mhz="", index="", index_kind="",
                  value=400.0, unit="W",
                  note="A100-SXM4-40GB board limit, applied to the ITERATION MEAN",
                  source="nvidia-smi"))
    w("coefficients.csv", r)


def throttle_matrix():
    rows = list(csv.DictReader(open(os.path.join(D, "overlap_432.csv"))))
    go = {(int(x["clock"]), int(x["m"]), int(x["n"])): x for x in rows
          if x["mode"] == "gemm_only"}
    co = {(int(x["clock"]), int(x["ctas"])): x for x in rows if x["mode"] == "comm_only"}
    out = []
    for r in [x for x in rows if x["mode"] == "concurrent"]:
        f, c = int(r["clock"]), int(r["ctas"])
        m, n, k = int(r["m"]), int(r["n"]), int(r["k"])
        g, w_ = int(r["grid"]), int(r["waves"])
        tg, tc = float(go[(f, m, n)]["iter_ms"]), float(co[(f, c)]["iter_ms"])
        it = float(r["iter_ms"])
        p_g = t.A[f] * (g / w_) + s.b_lookup(t.B, f, n, k)
        p_c = t.P0[f] + t.G[f] * t.Bw[f][c]
        mean = t.PS[f] + (p_g * tg + p_c * tc) / it
        peak = t.PS[f] + t.A[f] * min(g, 108) + s.b_lookup(t.B, f, n, k) + p_c
        thr = r["clock_held"] != "True"
        out.append(dict(clock=f, ctas=c, m=m, nk=n, grid=g,
                        measured_w=float(r["power_per_gpu_w"]),
                        clock_achieved=int(r["clock_min"]), throttled=thr,
                        model_mean_w=round(mean, 1), model_peak_w=round(peak, 1),
                        mean_predicts=mean > 400, peak_predicts=peak > 400,
                        mean_correct=(mean > 400) == thr,
                        peak_correct=(peak > 400) == thr))
    w("throttle_matrix.csv", out)
    for k_ in ("mean", "peak"):
        tp = sum(1 for x in out if x[f"{k_}_predicts"] and x["throttled"])
        fp = sum(1 for x in out if x[f"{k_}_predicts"] and not x["throttled"])
        fn = sum(1 for x in out if not x[f"{k_}_predicts"] and x["throttled"])
        print(f"      {k_:>4} criterion: TP {tp} FP {fp} FN {fn}  "
              f"precision {tp/max(tp+fp,1)*100:.1f}%  recall {tp/max(tp+fn,1)*100:.1f}%")


def frontier():
    G = json.load(open(os.path.join(D, "grid12.json")))
    grid, nk, sk, gb = G["4096_4096_4096"], 4096, 1, 256 * 2 ** 20 / 1e9
    out = []
    for arm, tag in (("hard1", "1 clock"), ("hard2", "2 clocks")):
        c = sorted(J.feasible(J.candidates(grid, nk, sk, gb, arm)), key=lambda z: z[1])
        best = float("inf")
        for E, T, cta, fg, fc, sg, *_ in c:
            if E < best - 1e-9:
                best = E
                out.append(dict(arm=tag, m=4096, nk=4096, mib=256,
                                latency_budget_ms=round(T, 5), energy_mJ=round(E, 2),
                                f_gemm=fg, f_comm=fc, sm_gemm=sg, ctas=cta))
    w("frontier.csv", out)


def paired():
    G = json.load(open(os.path.join(D, "grid12.json")))
    out = []
    for nk in (4096, 8192, 16384):
        for m in (1024, 2048, 4096, 8192):
            grid = G[f"{m}_{nk}_{nk}"]; sk = max(1, grid // (m * nk // 32768))
            for mib in (8, 32, 128, 256):
                gb = mib * 2 ** 20 / 1e9
                cb = J.candidates(grid, nk, sk, gb, "hard1")
                cc = J.candidates(grid, nk, sk, gb, "hard2")
                r = dict(m=m, nk=nk, k=nk, mib=mib, grid=grid, splitk=sk,
                         tc_over_tg=round(O.t_comm(900, 16, gb)
                                          / O.t_gemm(900, 108, grid, nk, sk), 4))
                for o in ("E", "T", "EDP"):
                    B, C = J.pick(cb, o), J.pick(cc, o)
                    r.update({f"{o}_1clk_mJ": round(B[0], 2), f"{o}_1clk_ms": round(B[1], 5),
                              f"{o}_1clk_f": B[3], f"{o}_1clk_ctas": B[2],
                              f"{o}_2clk_mJ": round(C[0], 2), f"{o}_2clk_ms": round(C[1], 5),
                              f"{o}_2clk_fg": C[3], f"{o}_2clk_fc": C[4],
                              f"{o}_2clk_ctas": C[2],
                              f"{o}_E_ratio": round(C[0] / B[0], 5),
                              f"{o}_T_ratio": round(C[1] / B[1], 5)})
                out.append(r)
    w("paired.csv", out)


if __name__ == "__main__":
    print("writing derived data to data/")
    coefficients(); throttle_matrix(); frontier(); paired()
