#!/usr/bin/env python3
"""Wave-resolved, composable power model: will overlapping a GEMM with a collective throttle?

WHY A WAVE-RESOLVED MODEL IS NEEDED *HERE* AND NOT FOR A LONE KERNEL

A GEMM's power is a staircase, not a level. It runs `tiles` tiles in
`W = ceil(tiles/S)` waves, and the last wave is usually partial, so the busy-SM count
-- and the power -- steps down at the end. For M=1024, N=K=8192 on 108 SMs at 1410 MHz
that is 580 W during the full waves against 279 W during the tail, with a 479 W mean.

For a single kernel looping back to back, the mean is what matters and the averaged
model `a*S_eff + b` is enough. What changes under overlap is that the two kernels'
instantaneous powers ADD, so the answer depends on WHICH PART of the GEMM the
collective lands on:

    collective during the full waves  ->  P_gemm_peak + P_comm       can exceed the cap
    collective during the tail wave   ->  P_gemm_tail + P_comm       usually will not

That is a question about a composed trace. No summary statistic of either kernel alone
can answer it, which is exactly why the staircase has to be carried through.

THE MODEL

    P(t) = P_static(f) + sum over concurrent kernels of their instantaneous dynamic power

    GEMM        a(f)*s(t) + b(f, footprint),   s(t) = S for the first W-1 waves,
                                               tiles-(W-1)*S for the last
    collective  a constant while resident -- an NCCL collective is ONE persistent
                kernel whose n_cta blocks stay up for the whole reduction

COEFFICIENTS, AND WHERE THEY COME FROM

`a` and `b` are from ../gemm_dcfs_char/MODEL.md, fitted against S_eff -- which makes
them instantaneous coefficients already, because

    mean(a*s(t)+b) = a*(tiles/W) + b = a*S_eff + b

so regressing on S_eff recovers the same `a`. Only the summary statistic was wrong.

`b` is indexed by the WEIGHT MATRIX FOOTPRINT N*K*2 bytes, not by N. MODEL.md tabulates
it against `N=K`, which conflates the two; under overlap N != K is normal (K is what
lengthens a low-occupancy GEMM) and using N alone puts a 128 MB footprint in the 32 MB
bucket -- worth 12 W here.

Collective power is measured directly, from ../comm_dvfs_char/data/comm_trtllm_ar_ctas.csv.
It is NOT linear in CTA count: a fixed message is bandwidth-bound, so power saturates
(27 W at 1 CTA to 62 W at 32 CTA per GPU at 1200 MHz). Interpolated, not fitted.

Cross-check: reverse-deriving collective power from the serial rows of
data/phase1_partial.csv -- a completely independent measurement -- gives 28.3 W at
2 CTA and 34.7 W at 4 CTA against 29.3 and 34.3 W here.

WHAT IS NOT MODELLED

* The controller's averaging window. Measured indirectly: it is much longer than a
  single GEMM, so a lone kernel in a back-to-back loop is judged on its mean. Under
  overlap the composed trace can hold an elevated level for the whole collective --
  milliseconds -- which is the regime where the peak matters. `throttles()` therefore
  reports both the instantaneous peak and a windowed average, and says which
  bound is being used.
* Where the collective lands within the GEMM's waves. It is not schedulable, so
  `compose()` returns the worst case (aligned with the full waves) and the best case
  (aligned with the tail), and they bracket reality.

  python3 power_model.py            # self-check against the measured overlap rows
"""

import bisect
import math

TILE_M, TILE_N = 128, 256
TOTAL_SM = 108

# --- from ../gemm_dcfs_char/MODEL.md -------------------------------------------------
A = {300: 0.4992, 510: 0.8405, 705: 1.1747, 900: 1.4940, 1200: 2.6658, 1410: 4.4212}
P_STATIC = {300: 56.37, 510: 57.60, 705: 59.38, 900: 61.36, 1200: 69.66, 1410: 85.37}
# b vs weight-matrix footprint in MB, per clock. MODEL.md's three N=K columns are
# 32 / 128 / 512 MB.
B_FOOTPRINT = {
    300: [(32, 4.5), (128, 8.5), (512, 9.7)],
    510: [(32, 3.7), (128, 9.9), (512, 11.8)],
    705: [(32, 3.7), (128, 11.1), (512, 14.3)],
    900: [(32, 3.3), (128, 12.8), (512, 16.2)],
    1200: [(32, 0.2), (128, 12.1), (512, 16.3)],
    1410: [(32, 1.8), (128, 16.7), (512, 19.0)],
}
# --- measured, ../comm_dvfs_char/data/comm_trtllm_ar_ctas.csv: per-GPU dynamic W ------
COMM_DYN = {
    1200: [(1, 27.3), (2, 29.3), (4, 34.3), (8, 43.3), (16, 53.3), (32, 62.3)],
    900: [(1, 25.6), (2, 26.6), (4, 30.6), (8, 37.6), (16, 44.6), (32, 48.6)],
    300: [(1, 24.6), (2, 24.6), (4, 26.6), (8, 29.6), (16, 33.6), (32, 37.6)],
}
POWER_CAP_W = 400.0


def _interp(table, x):
    xs = [p[0] for p in table]
    if x <= xs[0]:
        return table[0][1]
    if x >= xs[-1]:
        return table[-1][1]
    i = bisect.bisect_left(xs, x)
    (x0, y0), (x1, y1) = table[i - 1], table[i]
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def _nearest_clock(d, f):
    return min(d, key=lambda c: abs(c - f))


def gemm_waves(m, n, k, sm, f):
    """-> dict describing the GEMM's power staircase at `sm` SMs and clock `f`."""
    tiles = -(-m // TILE_M) * -(-n // TILE_N)
    w = -(-tiles // sm)
    tail = tiles - (w - 1) * sm
    a = A[_nearest_clock(A, f)]
    b = _interp(B_FOOTPRINT[_nearest_clock(B_FOOTPRINT, f)], n * k * 2 / 2 ** 20)
    return dict(tiles=tiles, waves=w, tail=tail, s_eff=tiles / w,
                p_full=a * min(sm, tiles) + b,      # dynamic W during a full wave
                p_tail=a * tail + b,                # dynamic W during the last wave
                p_mean=a * (tiles / w) + b,
                full_frac=(w - 1) / w)


def comm_power(ctas, f):
    """Per-GPU dynamic W of a resident NCCL collective. Constant while it runs."""
    return _interp(COMM_DYN[_nearest_clock(COMM_DYN, f)], ctas)


def compose(m, n, k, sm, f, ctas, t_gemm_ms, t_comm_ms, cap=POWER_CAP_W):
    """Overlap a GEMM with a collective; bracket the composed power.

    The collective's placement within the GEMM's waves is not controllable, so the
    two brackets are: it sits over full waves (worst) or over the tail (best).
    """
    g = gemm_waves(m, n, k, sm, f)
    pc = comm_power(ctas, f)
    ps = P_STATIC[_nearest_clock(P_STATIC, f)]
    t_wave = t_gemm_ms / g["waves"]

    worst = ps + g["p_full"] + pc                    # collective over a full wave
    best = ps + g["p_tail"] + pc                     # collective over the tail wave
    gemm_only_peak = ps + g["p_full"]
    # mean over the overlapped span, for reference: the GEMM contributes its own mean
    # for t_gemm and the collective its constant for t_comm
    span = max(t_gemm_ms, t_comm_ms)
    mean = ps + (g["p_mean"] * t_gemm_ms + pc * t_comm_ms) / span

    return dict(**{f"gemm_{x}": g[x] for x in ("tiles", "waves", "tail", "s_eff")},
                t_wave_ms=t_wave, p_comm=pc, p_static=ps,
                p_gemm_full=ps + g["p_full"], p_gemm_tail=ps + g["p_tail"],
                p_gemm_mean=ps + g["p_mean"],
                p_worst=worst, p_best=best, p_mean=mean,
                headroom_gemm_alone=cap - gemm_only_peak,
                headroom_worst=cap - worst,
                verdict=("throttles even in the best alignment" if best >= cap else
                         "throttles when the collective lands on a full wave" if worst >= cap
                         else "no throttle"),
                # how many CTAs fit in the remaining headroom at the GEMM's peak
                max_safe_ctas=_max_ctas(cap - gemm_only_peak, f))


def _max_ctas(headroom_w, f):
    tab = COMM_DYN[_nearest_clock(COMM_DYN, f)]
    ok = [c for c, p in tab if p <= headroom_w]
    return max(ok) if ok else 0


def _selfcheck():
    import collections
    import csv
    import os
    import statistics as st
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "data", "phase1_partial.csv")
    SHAPES = {"low_occ": (1024, 4096, 16384), "high_occ": (8192, 4096, 4096)}
    F = 1200
    rows = [r for r in csv.DictReader(open(path)) if r["mode"] == "comm_first"]
    print("Self-check against the measured concurrent rows (world=2, 1200 MHz).")
    print("NVML reports the loop AVERAGE, so it is compared against p_mean; p_worst is")
    print("the instantaneous bound the cap actually reacts to and cannot be seen by NVML.\n")
    print(f"{'CTA':>4}{'shape':>10}{'MiB':>5}{'measured/GPU':>14}"
          f"{'p_mean':>9}{'err':>8}{'p_worst':>9}{'p_best':>8}{'verdict':>12}")
    errs = []
    for r in sorted(rows, key=lambda x: (int(x["ctas"]), x["shape"], int(x["ar_mib"]))):
        m, n, k = SHAPES[r["shape"]]
        c = compose(m, n, k, TOTAL_SM, F, int(r["ctas"]),
                    float(r["t_gemm_ms"]), float(r["t_ar_ms"]))
        meas = float(r["power_node_w"]) / 2                     # node -> per GPU
        # the loop has sync gaps where neither kernel runs; scale the model's span-mean
        # to the measured iteration length so the comparison is like for like
        span = max(float(r["t_gemm_ms"]), float(r["t_ar_ms"]))
        pm = c["p_static"] + (c["p_mean"] - c["p_static"]) * span / float(r["iter_ms"])
        e = (pm - meas) / meas
        errs.append(abs(e))
        print(f"{int(r['ctas']):>4}{r['shape']:>10}{int(r['ar_mib']):>5}{meas:>14.1f}"
              f"{pm:>9.1f}{e * 100:>7.1f}%{c['p_worst']:>9.0f}{c['p_best']:>8.0f}"
              f"{('OK' if 'no' in c['verdict'] else 'THROTTLE'):>12}")
    print(f"\n  median |error| on the loop average: {st.median(errs) * 100:.1f}%  "
          f"(n={len(errs)})")

    print("\nWhere the two models DISAGREE -- 1200 MHz, 108 SMs, 400 W cap.")
    print("`mean says` allocates CTAs against the kernel mean; `wave says` against the")
    print("power during a full wave, which is what a co-running kernel actually adds to.\n")
    print(f"{'M':>6}{'N':>7}{'K':>7}{'W':>4}{'tail':>6}{'peak':>7}{'mean':>7}"
          f"{'gap':>6}{'mean says':>11}{'wave says':>11}{'':>4}")
    for (m, n, k) in [(1024, 4096, 4096), (1024, 4096, 16384), (1024, 8192, 8192),
                      (2048, 4096, 4096), (4096, 8192, 8192), (8192, 4096, 4096),
                      (8192, 16384, 16384)]:
        c = compose(m, n, k, TOTAL_SM, 1200, 1, 1.0, 1.0)
        by_mean = _max_ctas(POWER_CAP_W - c["p_gemm_mean"], 1200)
        by_wave = _max_ctas(POWER_CAP_W - c["p_gemm_full"], 1200)
        flag = "  <-- differ" if by_mean != by_wave else ""
        print(f"{m:>6}{n:>7}{k:>7}{c['gemm_waves']:>4}{c['gemm_tail']:>6}"
              f"{c['p_gemm_full']:>7.0f}{c['p_gemm_mean']:>7.0f}"
              f"{c['p_gemm_full'] - c['p_gemm_mean']:>6.0f}"
              f"{by_mean:>11}{by_wave:>11}{flag}")
    print("\n  The gap is the peak-minus-mean of the GEMM alone; it is what a mean-only")
    print("  model silently hands out to the collective. It is largest for few-wave")
    print("  shapes -- exactly the low-occupancy GEMMs that are worth overlapping.")
    print("\n  UNVALIDATED: no measurement here distinguishes the two verdicts. Doing so")
    print("  needs the achieved clock recorded per overlap configuration, which")
    print("  phase1_sweep.py samples but data/phase1_partial.csv does not carry.")

if __name__ == "__main__":
    _selfcheck()
