#!/usr/bin/env python3
"""Analyse the comm-bound sweep: the region every two-clock claim depends on.

WHAT WAS MISSING. All previous two-clock numbers rested on 64 MiB, where a sensibly
chosen CTA count leaves 11 of 12 shapes GEMM-bound. Only 11 of 48 workloads reached
t_comm/t_gemm > 1, and they got there mostly through 1-2 CTA configurations no planner
would pick. This sweep goes at the region directly: 128 / 256 / 512 MiB with CTA counts
a planner would actually use.

THREE THINGS IT SETTLES

  1. how much a second clock is worth where it is supposed to be worth something,
     computed from measured solo kernels rather than from the fitted model
  2. whether B(c,f), fitted on 8-256 MiB, still holds at 512 MiB -- the model has been
     extrapolating past its calibration ceiling every time a large message was quoted
  3. whether the composition rule survives in this regime, validated the same way as
     before: against the measured `concurrent` rows at equal clocks

  python3 analyse_commbound.py
"""
import csv
import glob
import os
import statistics as st

import table_432 as t
import followup_baseline as FB

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = 400.0
CLOCKS = [300, 510, 705, 900, 1200, 1410]


def load():
    rows = []
    for f in sorted(glob.glob(os.path.join(HERE, "data", "commbound_*mib.csv"))):
        mib = int(os.path.basename(f).split("_")[1].replace("mib.csv", ""))
        for r in csv.DictReader(open(f)):
            r["_mib"] = mib
            rows.append(r)
    return rows


def main():
    rows = load()
    if not rows:
        print("no comm-bound data yet"); return
    held = lambda r: r["clock_held"] == "True"
    sizes = sorted({r["_mib"] for r in rows})
    print(f"{len(rows)} rows over {len(sizes)} message sizes {sizes}\n")

    G = {(int(r["clock"]), int(r["m"]), int(r["n"])): r
         for r in rows if r["mode"] == "gemm_only" and held(r)}
    C = {(r["_mib"], int(r["clock"]), int(r["ctas"])): r
         for r in rows if r["mode"] == "comm_only" and held(r)}
    X = {(r["_mib"], int(r["clock"]), int(r["ctas"]), int(r["m"]), int(r["n"])): r
         for r in rows if r["mode"] == "concurrent"}
    grids = {(int(r["m"]), int(r["n"])): int(r["grid"])
             for r in rows if r["mode"] == "concurrent"}

    # ---- 1. does B(c,f) still hold at 512 MiB? ------------------------------------
    print("1. COLLECTIVE CALIBRATION vs measurement, including past its 256 MiB ceiling\n")
    print(f"  {'MiB':>6}{'CTA':>5}{'f':>6}{'measured ms':>13}{'model ms':>10}{'err':>8}")
    err_in, err_out = [], []
    for mib in sizes:
        for c in (4, 32):
            for f in (900, 1410):
                x = C.get((mib, f, c))
                if not x:
                    continue
                meas = float(x["iter_ms"])
                pred = t.AL[f][c] / 1e3 + (mib * 2 ** 20 / 1e9) / t.Bw[f][c] * 1e3
                e = (pred - meas) / meas
                (err_out if mib > 256 else err_in).append(abs(e))
                print(f"  {mib:>6}{c:>5}{f:>6}{meas:>13.3f}{pred:>10.3f}{e * 100:>7.1f}%"
                      + ("   <- beyond calibration" if mib > 256 else ""))
    if err_in:
        print(f"\n  within calibration (<=256 MiB): median |err| {st.median(err_in)*100:.2f}%")
    if err_out:
        print(f"  extrapolated    (512 MiB): median |err| {st.median(err_out)*100:.2f}%")

    # ---- 2. how comm-bound did we actually get? ------------------------------------
    print("\n\n2. t_comm / t_gemm REACHED, at 900 MHz\n")
    ratios = []
    for (mib, f, c, m, n), x in X.items():
        if f != 900:
            continue
        g, cm = G.get((900, m, n)), C.get((mib, 900, c))
        if g and cm:
            ratios.append((float(cm["iter_ms"]) / float(g["iter_ms"]), mib, c, m, n))
    if ratios:
        rs = [z[0] for z in ratios]
        print(f"  {len(rs)} configs   median {st.median(rs):.2f}   "
              f"range {min(rs):.2f} .. {max(rs):.1f}")
        print(f"  fraction with t_c/t_g > 1: {sum(1 for x in rs if x > 1)/len(rs)*100:.0f}%"
              f"   (was 23% at 64 MiB)")

    # ---- 3. composition validation, then the two-clock gain ------------------------
    print("\n\n3. COMPOSITION CHECK then TWO-CLOCK GAIN, all from measurement\n")
    ev = []
    for (mib, f, c, m, n), x in X.items():
        g, cm = G.get((f, m, n)), C.get((mib, f, c))
        if not (g and cm) or not held(x):
            continue
        ti, E, pk = FB.compose(g, cm, f, f, int(x["grid"]))
        Em = float(x["power_per_gpu_w"]) * float(x["iter_ms"])
        ev.append(abs(E - Em) / Em)
    if ev:
        ev.sort()
        print(f"  composition vs measured concurrent, equal clocks, n={len(ev)}: "
              f"median {st.median(ev)*100:.2f}%  p90 {ev[int(.9*len(ev))]*100:.2f}%")

    out = []
    shapes = sorted({(int(r["m"]), int(r["n"]), int(r["grid"]))
                     for r in rows if r["mode"] == "concurrent"})
    ctas = sorted({int(r["ctas"]) for r in rows if r["mode"] == "concurrent"})
    for mib in sizes:
        for (m, n, grid) in shapes:
            for c in ctas:
                one, two = [], []
                for fg in CLOCKS:
                    g = G.get((fg, m, n))
                    if not g:
                        continue
                    for fc in CLOCKS:
                        cm = C.get((mib, fc, c))
                        if not cm:
                            continue
                        ti, E, pk = FB.compose(g, cm, fg, fc, grid)
                        # MEAN ONLY. An earlier version also required the instantaneous
                        # peak under the cap. That is not the validated criterion (mean:
                        # 88.6% precision, peak: 31.5%), and in this dataset it does
                        # active harm: 27 of 372 measured concurrent rows have a composed
                        # peak above 400 W AND held their clock. Vetoing a configuration
                        # the GPU demonstrably ran, on the strength of a modelled peak,
                        # discards ground truth. It also inflated the reported gain by
                        # 2.2 pp by removing strong baselines.
                        if E / ti > CAP:
                            continue
                        two.append((E, ti))
                        if fg == fc:
                            one.append((E, ti))
                if not one or not two:
                    continue
                g9, c9 = G.get((900, m, n)), C.get((mib, 900, c))
                r = dict(mib=mib, m=m, nk=n, ctas=c,
                         tc_over_tg=round(float(c9["iter_ms"]) / float(g9["iter_ms"]), 3)
                         if (g9 and c9) else "")
                for obj in ("E", "T", "EDP"):
                    B, D = FB.pick(one, obj), FB.pick(two, obj)
                    r[f"{obj}_dE_pct"] = round((D[0] - B[0]) / B[0] * 100, 2)
                    r[f"{obj}_dT_pct"] = round((D[1] - B[1]) / B[1] * 100, 2)
                # MATCHED-LATENCY gain. Optimising each arm independently under a
                # lexicographic min-T objective lets the two-clock arm pick a strictly
                # FASTER point that costs more energy, which shows up as a positive
                # "saving" -- six cells in the matrix read +10 to +20 for that reason
                # alone. Holding it to the one-clock arm's own latency asks the only
                # well-posed question: at the same speed, how much energy does the
                # second domain save?
                B = FB.pick(one, "T")
                feas = [z for z in two if z[1] <= B[1] * 1.0001]
                r["matched_dE_pct"] = round(
                    (min(z[0] for z in feas) - B[0]) / B[0] * 100, 2) if feas else 0.0
                # THE TRUE BASELINE: the overlap the GPU actually ran, not a composed
                # stand-in for it. The composed arm is only 0.35% cheaper here, so the
                # two agree -- but the question "what is this measured against" deserves
                # the measured answer, and it happens to give a LARGER gain, so the
                # composed version was the conservative one.
                # The filter must match the other arms. The instantaneous peak is not
                # measurable, so it is taken from the composition at that same clock,
                # which is exactly what filters the composed candidates.
                meas = []
                for f in CLOCKS:
                    x = X.get((mib, f, c, m, n))
                    gg, cc2 = G.get((f, m, n)), C.get((mib, f, c))
                    if not (x and gg and cc2):
                        continue
                    T, P = float(x["iter_ms"]), float(x["power_per_gpu_w"])
                    if P > CAP:
                        continue
                    meas.append((P * T, T))
                if meas:
                    Bm = FB.pick(meas, "T")
                    fm = [z for z in two if z[1] <= Bm[1] * 1.0001]
                    r["vs_measured_dE_pct"] = round(
                        (min(z[0] for z in fm) - Bm[0]) / Bm[0] * 100, 2) if fm else 0.0
                    r["compose_bias_pct"] = round((B[0] - Bm[0]) / Bm[0] * 100, 2)
                else:
                    r["vs_measured_dE_pct"] = ""; r["compose_bias_pct"] = ""
                out.append(r)
    if not out:
        return
    p = os.path.join(HERE, "data", "commbound_analysis.csv")
    with open(p, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        wr.writeheader(); wr.writerows(out)
    print(f"\n  two-clock gain over {len(out)} configs, by message size:")
    print(f"  {'MiB':>6}{'n':>5}{'median t_c/t_g':>16}"
          + "".join(f"{o:>14}" for o in ("min energy", "min latency", "min EDP")))
    for mib in sizes:
        v = [x for x in out if x["mib"] == mib]
        rr = [x["tc_over_tg"] for x in v if x["tc_over_tg"] != ""]
        print(f"  {mib:>6}{len(v):>5}{st.median(rr) if rr else 0:>16.2f}"
              + "".join(f"{st.median(x[f'{o}_dE_pct'] for x in v):>+13.2f}%"
                        for o in ("E", "T", "EDP")))
    print(f"\n  wrote {p}")


if __name__ == "__main__":
    main()
