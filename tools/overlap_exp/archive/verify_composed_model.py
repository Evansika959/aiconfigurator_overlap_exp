#!/usr/bin/env python3
"""Check the composed GEMM+collective power model against the measured overlap runs.

The model being tested, per GPU:

    P(t) = P_static(f)                                        board floor, counted ONCE
         + a(f)*s_gemm(t) + b(N*K)                            GEMM, a staircase
         + [p0(f) + gamma(f)*B(c,f)]  while the collective is resident   a rectangle

    s_gemm(t) = S for the first W-1 waves, grid-(W-1)*S for the last,  W = ceil(grid/S)
    t_comm    = alpha(c,f) + S_bytes / B(c,f)

Coefficients come from two separate campaigns, neither of which saw an overlap run:
    a, b, P_static   ../gemm_dcfs_char  (1080 rows, 12 shapes x 6 clocks x 5 SM counts)
    alpha, B, p0, gamma  ../comm_dvfs_char (288 rows, 8 sizes x 6 clocks x 6 CTA counts)

TWO THINGS THAT DO NOT TRANSFER FOR FREE, and are tested rather than assumed:

  WORLD SIZE. The comm campaign ran world=4, the overlap runs world=2. A ring
  all-reduce moves 2(N-1)/N * S bytes per rank -- 1.5*S at N=4, 1.0*S at N=2 -- so the
  fitted B is a property of a different data volume. The naive correction is to scale
  by 1.5; whether that is right is measured below.

  WHAT NVML SEES. The model produces P(t); NVML reports an average over a window far
  longer than one iteration. So the comparison is against the model's time-average over
  the measured iteration, INCLUDING the sync gap where neither kernel runs and only
  P_static is drawn. Getting that gap wrong was a real error earlier: comparing the
  overlap-interval mean against a loop average over-predicted by 142 W.

  python3 verify_composed_model.py
"""

import collections
import csv
import math
import os
import statistics as st

# ---- from ../gemm_dcfs_char/MODEL.md ------------------------------------------------
A_GEMM = {300: 0.4992, 510: 0.8405, 705: 1.1747, 900: 1.4940, 1200: 2.6658, 1410: 4.4212}
P_STATIC = {300: 56.37, 510: 57.60, 705: 59.38, 900: 61.36, 1200: 69.66, 1410: 85.37}
B_FOOTPRINT = {1200: [(32, 0.2), (128, 12.1), (512, 16.3)]}

# ---- from ../comm_dvfs_char, fitted on 288 rows (world=4) ---------------------------
# B[f][c] algorithmic GB/s, ALPHA[f][c] microseconds
B_W4 = {1200: {1: 9.6, 2: 19.4, 4: 37.1, 8: 68.7, 16: 119.7, 32: 139.1}}
ALPHA_W4 = {1200: {1: 48.1, 2: 46.4, 4: 44.1, 8: 58.3, 16: 77.0, 32: 71.8}}
P0 = {1200: 24.6}
GAMMA = {1200: 0.258}

TILE_AREA = 32768          # 128x256 or 256x128; the area is what matters
TOTAL_SM = 108
F = 1200


def b_gemm(n, k, f=F):
    mb = n * k * 2 / 2 ** 20
    tab = B_FOOTPRINT[f]
    if mb <= tab[0][0]:
        return tab[0][1]
    if mb >= tab[-1][0]:
        return tab[-1][1]
    for (x0, y0), (x1, y1) in zip(tab, tab[1:]):
        if x0 <= mb <= x1:
            return y0 + (y1 - y0) * (mb - x0) / (x1 - x0)


def gemm_staircase(m, n, k, sm, f=F):
    """-> full-wave power, tail-wave power, kernel-average power (dynamic, per GPU)."""
    grid = m * n // TILE_AREA          # split-K = 1 for these shapes
    w = -(-grid // sm)
    tail = grid - (w - 1) * sm
    a, b = A_GEMM[f], b_gemm(n, k, f)
    return dict(grid=grid, waves=w, tail=tail,
                p_full=a * min(sm, grid) + b, p_tail=a * tail + b,
                p_mean=a * (grid / w) + b)


def comm(mib, ctas, world, f=F):
    """Predicted collective latency and power, per GPU. The comm fit is world=4; a ring
    all-reduce moves 2(N-1)/N*S per rank, so the transferred bytes scale by that ratio."""
    vol = 2 * (world - 1) / world          # 1.5 at N=4, 1.0 at N=2
    scale = vol / (2 * 3 / 4)
    gb = mib * 2 ** 20 / 1e9
    b = B_W4[f][ctas]
    t_ms = ALPHA_W4[f][ctas] / 1e3 + gb * scale / b * 1e3
    return dict(t_ms=t_ms, p=P0[f] + GAMMA[f] * b * scale, b_eff=b / scale)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    SH = {"low_occ": (1024, 4096, 16384), "high_occ": (8192, 4096, 4096)}
    WORLD = 2

    # ---- 1. does the collective's latency transfer from world=4 to world=2? --------
    print("1. Collective latency: world=4 fit applied to the world=2 overlap runs")
    print(f"   volume ratio 2(N-1)/N : {2 * 3 / 4:.2f} at N=4 -> {2 * 1 / 2:.2f} at N=2, "
          f"so predicted speedup {1.5 / 1.0:.2f}x\n")
    rows = [r for r in csv.DictReader(open(os.path.join(here, "data",
                                                        "phase1_partial.csv")))
            if r["mode"] == "serial"]
    print(f"   {'MiB':>5}{'CTA':>5}{'measured':>10}{'predicted':>11}{'err':>8}"
          f"{'implied ratio':>15}")
    errs, ratios = [], []
    seen = set()
    for r in sorted(rows, key=lambda x: (int(x["ar_mib"]), int(x["ctas"]))):
        key = (int(r["ar_mib"]), int(r["ctas"]))
        if key in seen:
            continue
        seen.add(key)
        mib, c = key
        meas = float(r["t_ar_ms"])
        p = comm(mib, c, WORLD)
        e = (p["t_ms"] - meas) / meas
        # what volume ratio would the measurement imply?
        gb = mib * 2 ** 20 / 1e9
        implied = (meas - ALPHA_W4[F][c] / 1e3) / (gb / B_W4[F][c] * 1e3) * 1.5
        errs.append(abs(e)); ratios.append(implied)
        print(f"   {mib:>5}{c:>5}{meas:>10.3f}{p['t_ms']:>11.3f}{e * 100:>7.1f}%"
              f"{implied:>15.2f}")
    print(f"\n   median |error| {st.median(errs) * 100:.1f}%   "
          f"implied volume ratio {st.median(ratios):.2f} against the ring model's 1.00")

    # ---- 2. the composed power model ----------------------------------------------
    print("\n\n2. Composed power, against the measured concurrent runs")
    print("   Model averaged over the MEASURED iteration, so the sync gap (P_static only)")
    print("   is included exactly as the hardware saw it.\n")
    all_rows = list(csv.DictReader(open(os.path.join(here, "data",
                                                     "phase1_partial.csv"))))
    print(f"   {'shape':>9}{'MiB':>5}{'CTA':>4}{'mode':>12}{'iter ms':>9}"
          f"{'meas W/GPU':>12}{'model W':>9}{'err':>8}")
    E = collections.defaultdict(list)
    for r in sorted(all_rows, key=lambda x: (x["shape"], int(x["ar_mib"]),
                                             int(x["ctas"]), x["mode"])):
        if r["mode"] == "gemm_first_p0":
            continue
        m, n, k = SH[r["shape"]]
        g = gemm_staircase(m, n, k, TOTAL_SM)
        cm = comm(int(r["ar_mib"]), int(r["ctas"]), WORLD)
        tg, tc = float(r["t_gemm_ms"]), float(r["t_ar_ms"])
        it = float(r["iter_ms"])
        # energy per iteration / iteration length; P_static covers the whole period
        e_dyn = g["p_mean"] * tg + cm["p"] * tc
        model = P_STATIC[F] + e_dyn / it
        meas = float(r["power_node_w"]) / WORLD
        err = (model - meas) / meas
        E[r["mode"]].append(abs(err))
        print(f"   {r['shape']:>9}{int(r['ar_mib']):>5}{int(r['ctas']):>4}"
              f"{r['mode']:>12}{it:>9.3f}{meas:>12.1f}{model:>9.1f}{err * 100:>7.1f}%")
    print()
    for mode, v in E.items():
        print(f"   {mode:>12}: median |err| {st.median(v) * 100:5.1f}%   "
              f"max {max(v) * 100:5.1f}%   n={len(v)}")
    allv = [x for v in E.values() for x in v]
    print(f"   {'overall':>12}: median |err| {st.median(allv) * 100:5.1f}%   n={len(allv)}")

    # ---- 3. the instantaneous trace, which NVML cannot check ----------------------
    print("\n\n3. What the model says about the instantaneous trace (NOT verifiable here)")
    for tag, (m, n, k) in SH.items():
        g = gemm_staircase(m, n, k, TOTAL_SM)
        cm = comm(64, 8, WORLD)
        print(f"   {tag}: grid {g['grid']}, {g['waves']} waves, tail {g['tail']}")
        print(f"      GEMM alone   full wave {P_STATIC[F] + g['p_full']:6.0f} W   "
              f"tail {P_STATIC[F] + g['p_tail']:6.0f} W   mean {P_STATIC[F] + g['p_mean']:6.0f} W")
        print(f"      + 64MiB/8CTA rectangle (+{cm['p']:.0f} W): "
              f"peak {P_STATIC[F] + g['p_full'] + cm['p']:6.0f} W")


if __name__ == "__main__":
    main()
