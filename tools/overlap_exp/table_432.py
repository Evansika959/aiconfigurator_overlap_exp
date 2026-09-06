#!/usr/bin/env python3
"""The 432 concurrent cases as one table: configuration in, P(t) out, measured alongside.

NOTHING MEASURED ENTERS THE MODEL. Earlier scoring handed it t_gemm, t_comm and the
iteration length read out of the very CSV being scored, which tests the power model but
is not a prediction -- you cannot plan a deployment with numbers you could only get by
running it. Every column marked "model" below is computed from (M, N, K, clock, CTA,
message size) and the two calibration campaigns, full stop.

    P(t) = P_gemm(t) + P_comm(t) + P_const

    P_const  = P_static(f)                              board floor, counted once
    P_gemm   = a(f)*S_eff + b(f, N*K)   while resident, S_eff = grid/ceil(grid/108)
    P_comm   = p0(f) + gamma(f)*B(c,f)  while resident

reported as the time-average over one iteration, because NVML averages over a window
far longer than an iteration and the sync gap draws P_const alone.

THE THREE LATENCIES, all predicted:

    t_gemm = W*(K/splitK)*tau(f) + t0(f)      W = ceil(grid/108), splitK = grid /
                                    (M*N/32768); both fitted on the 1080-row sweep
    t_comm = alpha(c,f) + S/B(c,f)  from the 288-row collective campaign
    t_iter = max(t_gemm*(1 + (W'/W - 1)*min(1, t_comm/t_gemm)), t_comm)
                                    W' = ceil(grid/(108-c)): the GEMM re-waved onto the
                                    SMs the collective is not holding, charged only for
                                    the fraction of the GEMM the collective overlaps.
                                    Charging it for the whole GEMM over-predicts by 37%
                                    when a 32-CTA collective meets a 17 ms GEMM.

grid is ncu-MEASURED per shape (grid12.json), not M*N/32768 -- (1024,8192,8192) splits
K two ways and assuming otherwise cost 21% once. It is a property of the shape, not of
the run, so using it keeps the model config-only.

  python3 table_432.py            # writes data/table_432.csv + table_432.md
"""

import collections
import csv
import json
import os
import statistics as st

import score_432 as s

HERE = os.path.dirname(os.path.abspath(__file__))
TOTAL_SM = 108
COLS = ["clock", "clock_achieved", "throttled", "m", "nk", "ar_mib", "ctas",
        "t_gemm_ms", "t_comm_ms", "iter_ms_model", "iter_ms_meas",
        "meas_w", "model_w", "err_pct", "pred_w_throttled",
        "err_throttled_pct", "speedup", "dE_pct"]


GRIDS = json.load(open(os.path.join(HERE, "data", "grid12.json")))


def gemm_latency_fit():
    """t_gemm = W * (K/splitK) * tau(f) + t0(f), fitted per clock on the full-SM rows.

    TWO CORRECTIONS over the obvious W*K*tau, both found by looking at residuals:

      split-K.  A block that splits K s ways only walks K/s of the reduction, so it
      finishes in 1/s the time. Charging full K to (1024,8192,8192) -- the one shape
      here that splits -- over-predicted its latency by 76%. splitK is recoverable from
      the ncu grid: grid / (M*N/32768).

      t0.  A fixed per-launch cost. Without it the fit is dragged by the 17 ms shapes
      and under-predicts the 0.27 ms ones by 15%, which matters because short GEMMs are
      exactly where the collective dominates the iteration.
    """
    rows = [x for x in csv.DictReader(
        open(os.path.join(HERE, "..", "gemm_dcfs_char", "data",
                          "gemm_dynamic_energy.csv")))
        if x["clock_held"] == "True" and x["status"] == "ok"
        and int(x["sm_avail"]) == TOTAL_SM]
    d = collections.defaultdict(list)
    for x in rows:
        m, n, k = int(x["m"]), int(x["n"]), int(x["k"])
        g = GRIDS.get(f"{m}_{n}_{k}")
        if g is None:
            continue
        sk = max(1, g // (m * n // 32768))
        d[int(x["freq_req_mhz"])].append((-(-g // TOTAL_SM) * k / sk,
                                          float(x["latency_ms"])))
    out = {}
    for f, v in d.items():
        if len({p[0] for p in v}) >= 3:          # need distinct x to have a slope
            out[f] = s.fit([p[0] for p in v], [p[1] for p in v])
    # 1410 MHz is UNIDENTIFIABLE from this campaign: 33 of its 36 full-SM rows throttled,
    # leaving one distinct shape (three reps of 1024x4096x4096). Fitting two parameters
    # to one x gives tau=0 and t0 absorbing the whole latency -- every shape then gets the
    # same 0.206 ms. Extrapolate instead, on a law the other five clocks establish very
    # firmly: tau*f is constant to 0.03% CV across 300-1200 MHz, i.e. the GEMM is
    # purely clock-linear. t0*f rises linearly and is extrapolated the same way.
    # Against the one surviving 1410 shape this lands +7.9%, so 1410 GEMM latency is an
    # EXTRAPOLATION and is reported separately from the calibrated 300-1200 range.
    if 1410 not in out:
        fs = sorted(out)
        tau_f = st.mean(out[f][0] * f for f in fs)          # constant to 0.03%
        m, c = s.fit(fs, [out[f][1] * f for f in fs])       # t0*f rises linearly
        out[1410] = (tau_f / 1410, (m * 1410 + c) / 1410)
    return out


A, B, PS = s.gemm_coeffs()
Bw, AL, PW = s.comm_coeffs()
P0, G = s.comm_power_fit(Bw, PW, PS)
LAT = gemm_latency_fit()
FS = sorted(A)


def lin(d, f):
    """{clock: value} interpolated linearly in f, clamped outside the calibrated range"""
    ks = sorted(d)
    if f <= ks[0]:
        return d[ks[0]]
    if f >= ks[-1]:
        return d[ks[-1]]
    for a_, b_ in zip(ks, ks[1:]):
        if a_ <= f <= b_:
            return d[a_] + (d[b_] - d[a_]) * (f - a_) / (b_ - a_)


def predict(f, c, m, n, k, g, mib):
    """The whole model at an arbitrary f, which need not be a calibrated clock.

    f is a free argument rather than one of the six swept clocks because the throttled
    rows have to be evaluated at the frequency the hardware actually settled at, and
    that is whatever the 400 W cap dictated -- 1185, 1215, 1305 MHz. Every coefficient
    is interpolated there, including the per-CTA B(c,.) and alpha(c,.) tables.
    """
    w = -(-g // TOTAL_SM)
    sk = max(1, g // (m * n // 32768))
    tau, t0 = lin({q: LAT[q][0] for q in FS}, f), lin({q: LAT[q][1] for q in FS}, f)
    bw, al = lin({q: Bw[q][c] for q in FS}, f), lin({q: AL[q][c] for q in FS}, f)

    t_gemm = w * (k / sk) * tau + t0                               # ms
    t_comm = al / 1e3 + (mib * 2 ** 20 / 1e9) / bw * 1e3
    w2 = -(-g // max(TOTAL_SM - c, 1))
    t_iter = max(t_gemm * (1 + (w2 / w - 1) * min(1, t_comm / t_gemm)), t_comm)

    p_gemm = lin(A, f) * (g / w) + lin({q: s.b_lookup(B, q, n, k) for q in FS}, f)
    p_comm = lin(P0, f) + lin(G, f) * bw
    return dict(t_gemm=t_gemm, t_comm=t_comm, t_iter=t_iter,
                p=lin(PS, f) + (p_gemm * t_gemm + p_comm * t_comm) / t_iter)


def main():
    rows = list(csv.DictReader(open(os.path.join(HERE, "data", "overlap_432.csv"))))
    se = {(int(r["clock"]), int(r["ctas"]), int(r["m"]), int(r["n"])): r
          for r in rows if r["mode"] == "serial"}

    out = []
    for r in [x for x in rows if x["mode"] == "concurrent"]:
        f, c = int(r["clock"]), int(r["ctas"])
        m, n, k = int(r["m"]), int(r["n"]), int(r["k"])
        g, mib = int(r["grid"]), int(r["ar_mib"])
        fa = int(r["clock_min"])
        thr = r["clock_held"] != "True"

        q = predict(f, c, m, n, k, g, mib)
        t_gemm, t_comm, t_iter, model = q["t_gemm"], q["t_comm"], q["t_iter"], q["p"]
        # For a throttled row the model above answers a question the hardware was not
        # asked: what it would draw at 1410 MHz, when the cap held it to ~1200. Re-run
        # at the clock actually achieved so the row can be checked on its own terms.
        # Blank on rows that held their clock -- there `model_w` already is that number.
        pred_thr = round(predict(fa, c, m, n, k, g, mib)["p"], 1) if thr else ""

        it = float(r["iter_ms"])
        meas = float(r["power_per_gpu_w"])
        sr = se[(f, c, m, n)]
        e_ser = float(sr["power_per_gpu_w"]) * float(sr["iter_ms"])
        e_con = meas * it
        out.append(dict(clock=f,
                        # minimum SM clock seen during the window. Below `clock` means
                        # the 400 W board cap pulled it down -- the lock held everywhere
                        # else, so a shortfall here is power throttling, not a failed
                        # nvidia-smi -lgc. The model has no throttling term, so these
                        # rows are excluded from the accuracy figures.
                        clock_achieved=fa, throttled=thr,
                        m=m, nk=n, ar_mib=mib, ctas=c,
                        t_gemm_ms=round(t_gemm, 3), t_comm_ms=round(t_comm, 3),
                        iter_ms_model=round(t_iter, 3), iter_ms_meas=round(it, 3),
                        meas_w=round(meas, 1), model_w=round(model, 1),
                        err_pct=round((model - meas) / meas * 100, 1),
                        pred_w_throttled=pred_thr,
                        err_throttled_pct=(round((pred_thr - meas) / meas * 100, 1)
                                           if thr else ""),
                        speedup=round(float(sr["iter_ms"]) / it, 3),
                        dE_pct=round((e_ser - e_con) / e_ser * 100, 1),
                        _thr=thr))

    cp = os.path.join(HERE, "data", "table_432.csv")
    with open(cp, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
        wr.writeheader(); wr.writerows(out)

    hdr = ("| clock | achieved | M | N=K | AR MiB | CTA | t_gemm | t_comm | "
           "iter model | iter meas | meas W | model W | err | pred W @achieved | "
           "err @achieved | speedup | dE |")
    sep = "|" + "---|" * 17
    lines = ["# 432 concurrent overlap cases -- config-only model vs measurement", "",
             "`P(t) = P_gemm(t) + P_comm(t) + P_const`, time-averaged over one "
             "iteration. Every model column is computed from the configuration alone; "
             "no measured timing enters. Rows marked * throttled (clock fell below the "
             "locked value).", "", hdr, sep]
    for x in sorted(out, key=lambda z: (z["clock"], z["m"], z["nk"], z["ctas"])):
        lines.append(
            f"| {x['clock']} | {x['clock_achieved']}"
            f"{'*' if x['_thr'] else ''} | {x['m']} | {x['nk']} | "
            f"{x['ar_mib']} | {x['ctas']} | {x['t_gemm_ms']:.3f} | "
            f"{x['t_comm_ms']:.3f} | {x['iter_ms_model']:.3f} | "
            f"{x['iter_ms_meas']:.3f} | {x['meas_w']:.1f} | {x['model_w']:.1f} | "
            f"{x['err_pct']:+.1f}% | "
            f"{x['pred_w_throttled'] if x['_thr'] else ''} | "
            f"{f'{x["err_throttled_pct"]:+.1f}%' if x['_thr'] else ''} | "
            f"{x['speedup']:.3f} | {x['dE_pct']:+.1f}% |")
    mp = os.path.join(HERE, "table_432.md")
    open(mp, "w").write("\n".join(lines) + "\n")
    print(f"wrote {cp}\nwrote {mp}   ({len(out)} rows)\n")

    held = [x for x in out if not x["_thr"]]
    e = sorted(abs(x["err_pct"]) for x in held)
    ea = sorted(abs(x["err_pct"]) for x in out)
    print(f"CONFIG-ONLY MODEL, power vs measured concurrent")
    print(f"  non-throttled n={len(held)}: median {st.median(e):.2f}%  "
          f"p90 {e[int(.9 * len(e))]:.2f}%  max {max(e):.2f}%  "
          f"bias {st.mean(x['err_pct'] for x in held):+.2f}%")
    print(f"  all           n={len(out)}: median {st.median(ea):.2f}%  "
          f"p90 {ea[int(.9 * len(ea))]:.2f}%  max {max(ea):.2f}%")
    t = sorted(abs(x["iter_ms_model"] - x["iter_ms_meas"]) / x["iter_ms_meas"] * 100
               for x in held)
    print(f"  latency       n={len(held)}: median {st.median(t):.2f}%  "
          f"p90 {t[int(.9 * len(t))]:.2f}%  max {max(t):.2f}%")
    print(f"\n  by clock:")
    d = collections.defaultdict(list)
    for x in held:
        d[x["clock"]].append(abs(x["err_pct"]))
    for f in sorted(d):
        print(f"    {f:>5} MHz  n={len(d[f]):>3}  median {st.median(d[f]):>5.2f}%  "
              f"max {max(d[f]):>5.2f}%")
    print(f"\n  energy: overlap saves median {st.median(x['dE_pct'] for x in out):+.1f}% "
          f"per iteration  (best {max(x['dE_pct'] for x in out):+.1f}%, "
          f"worst {min(x['dE_pct'] for x in out):+.1f}%)")


if __name__ == "__main__":
    main()
