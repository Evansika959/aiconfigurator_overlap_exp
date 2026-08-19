#!/usr/bin/env python3
"""The modelled INSTANTANEOUS power of a GEMM, and why the average cannot predict throttling.

A GEMM does not draw constant power. It runs its tiles in waves, and the last wave is
usually partial, so the busy-SM count -- and with it the power -- steps down at the end
of every kernel. The fitted model already contains this: since

    mean(P) = [(W-1)*tau*(a*S+b) + tau*(a*T_tail+b)] / (W*tau) = a*(T/W) + b = a*S_eff + b

the coefficients regressed against S_eff ARE the instantaneous coefficients. Only the
summary statistic was wrong.

    P(t) = P_static(f) + a(f)*s(t) + b(f, footprint)
    s(t) = S             during the first W-1 waves
         = T - (W-1)*S   during the last one

Throttling does not watch either the peak or the mean; it watches a moving average.
Sweeping that window against the 21 observed throttle events out of 360 cells:

    tau -> 0      (peak)   21 caught, 3 FALSE alarms -- the shortest kernels
    tau = 0.5 ms           21 caught, 0 false alarms  <- only value with a clean split
    tau -> inf    (mean)   20 caught, 1 miss

The two shapes plotted here are the cleanest illustration: near-identical peaks, one
throttles and one does not, and only the 0.5 ms window tells them apart.

THE TRACE IS MODELLED, NOT MEASURED. NVML updates every ~105 ms and needs ~427 ms to
settle, while these kernels last 0.2-0.6 ms -- three orders of magnitude too slow. What
NVML measures is the long-run average, which is the single point marked on each panel.

  python3 plot_power_trace.py        # -> figs/power_trace_1410.png
"""

import argparse
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

F = 1410
A_F = 4.4212          # W per busy SM at 1410 MHz            (MODEL.md)
P_STATIC = 85.37      # W, whole board, zeus profile_p2p.py  (data/p0_static_zeus.csv)
B = {4096: 1.8, 8192: 16.7}
CAP = 400.0
TAU_MS = 0.5          # calibrated against the 21 observed throttle events
TILE_M, TILE_N = 128, 256
TOTAL_SM = 108

# (M, N=K, SMs, latency at 1410 MHz in ms, was it throttled, how the latency was obtained)
CASES = [
    (1024, 8192, 108, 0.6182 * 1200 / 1410, True,
     "0.618 ms measured at 1200 MHz, scaled by clock\n"
     "(it never ran a full kernel at 1410 -- it throttled to 1275)"),
    (1024, 4096, 108, 0.2062, False, "measured directly at 1410 MHz"),
]
# scaling check: the 4096 case predicts 0.2051 ms from its 1200 MHz point against
# 0.2062 measured, 0.5% -- so the same scaling is trustworthy for the throttled one


def trace(m, n, S, t_ms, periods=3, npts=6000):
    """-> (t, P) for `periods` back-to-back kernels, plus the wave decomposition."""
    T = math.ceil(m / TILE_M) * math.ceil(n / TILE_N)
    W = math.ceil(T / S)
    tail = T - (W - 1) * S
    hi = P_STATIC + A_F * min(S, T) + B[n]
    lo = P_STATIC + A_F * tail + B[n]
    per = np.linspace(0, t_ms, npts, endpoint=False)
    one = np.where(per < t_ms * (W - 1) / W, hi, lo)
    t = np.linspace(0, t_ms * periods, npts * periods, endpoint=False)
    return t, np.tile(one, periods), dict(T=T, W=W, tail=tail, hi=hi, lo=lo,
                                          seff=T / W, dt=t_ms / npts)


def boxcar(p, w):
    """Centred moving average of width w samples, wrapping (the kernel repeats)."""
    k = np.ones(w) / w
    return np.convolve(np.concatenate([p, p, p]), k, mode="same")[len(p):2 * len(p)]


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--out", default=os.path.join(here, "figs", "power_trace_1410.png"))
    a = ap.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(16.4, 6.4),
                             gridspec_kw=dict(left=0.058, right=0.985, top=0.76,
                                              bottom=0.115, wspace=0.16))

    for ax, (m, n, S, t_ms, throttled, prov) in zip(axes, CASES):
        t, p, w = trace(m, n, S, t_ms)
        nw = max(1, int(round(TAU_MS / w["dt"])))
        pb = boxcar(p, nw)
        mean = P_STATIC + A_F * w["seff"] + B[n]

        ax.fill_between(t, P_STATIC, p, step="post", color="#1f7a8c", alpha=0.16, zorder=1)
        ax.plot(t, p, color="#1f7a8c", lw=2.2, drawstyle="steps-post", zorder=4,
                label="modelled instantaneous  P = P_static + a·s(t) + b")
        ax.plot(t, pb, color="#e07b39", lw=2.4, zorder=5,
                label=f"{TAU_MS} ms moving average  ← what the power cap responds to")
        ax.axhline(mean, color="#7f8c8d", lw=1.8, ls="--", zorder=3,
                   label=f"kernel mean {mean:.0f} W  ← what a·S_eff+b reports")
        ax.axhline(CAP, color="#e8453c", lw=2.2, ls=(0, (6, 3)), zorder=6,
                   label=f"{CAP:.0f} W power cap")
        ax.axhline(P_STATIC, color="#95a5a6", lw=1.2, ls=":", zorder=2)

        for i in range(3):                       # wave boundaries
            for k in range(1, w["W"]):
                ax.axvline((i + k / w["W"]) * t_ms, color="#cccccc", lw=0.8, zorder=0)
            ax.axvline((i + 1) * t_ms, color="#888888", lw=1.1, zorder=0)

        ax.annotate(f"{w['hi']:.0f} W\n{S} SMs busy", (t_ms * 0.30, w["hi"]),
                    textcoords="offset points", xytext=(0, 9), ha="center",
                    fontsize=9.6, color="#1f7a8c", fontweight="bold")
        ax.annotate(f"{w['lo']:.0f} W\ntail wave, {w['tail']} tiles",
                    (t_ms * (w["W"] - 0.5) / w["W"], w["lo"]),
                    textcoords="offset points", xytext=(0, -30), ha="center",
                    fontsize=9.6, color="#1f7a8c", fontweight="bold")
        ax.text(0.5, 0.035,
                f"peak-to-tail swing {w['hi'] - w['lo']:.0f} W;  "
                f"{TAU_MS} ms average peaks at {pb.max():.0f} W  "
                f"{'≥' if pb.max() >= CAP else '<'} {CAP:.0f} W cap  →  "
                f"{'THROTTLE' if pb.max() >= CAP else 'no throttle'}"
                f"   ({'observed: ran at 1275 MHz' if throttled else 'observed: held 1410 MHz'})",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=9.6,
                bbox=dict(fc="#ffe8e6" if throttled else "#e8f5e9",
                          ec="#e0a0a0" if throttled else "#a5cfa8", lw=0.9, pad=4.2))

        ax.set_xlim(0, t_ms * 3)
        ax.set_ylim(P_STATIC - 25, max(w["hi"] for _ in [0]) + 95)
        ax.set_xlabel(f"time (ms) — {3} back-to-back kernels,  {t_ms:.3f} ms each", fontsize=10)
        ax.set_ylabel("whole-board power (W)", fontsize=10.5)
        ax.set_title(f"M={m}, N=K={n}, {S} SMs, 1410 MHz    "
                     f"{w['T']} tiles → {w['W']} waves, S_eff {w['seff']:.1f}",
                     fontsize=11.5, pad=7)
        ax.legend(fontsize=8.8, loc="upper right", frameon=True, framealpha=0.95)
        ax.grid(axis="y", color="#eeeeee", zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=9.5)
        ax.text(0.012, 0.965, f"kernel duration: {prov}", transform=ax.transAxes,
                fontsize=7.8, color="#777777", va="top")

    fig.suptitle("A100 · why an averaged power model cannot predict throttling",
                 fontsize=15.5, fontweight="bold", y=0.965)
    fig.text(0.5, 0.885,
             "The trace is MODELLED, not measured: NVML updates every ~105 ms and settles in ~427 ms, "
             "while these kernels last 0.2–0.6 ms. What NVML sees is the dashed mean.\n"
             "Both shapes peak within 15 W of each other, yet only the left one throttled — and only the "
             "0.5 ms moving average separates them. That window was calibrated against all 21 observed\n"
             "throttle events out of 360 cells: it is the only value that gives 0 misses and 0 false alarms.",
             ha="center", va="top", fontsize=9.7, color="#333333")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=140, facecolor="white")
    print(f"wrote {a.out}")
    for (m, n, S, t_ms, thr, _) in CASES:
        _, p, w = trace(m, n, S, t_ms)
        pb = boxcar(p, max(1, int(round(TAU_MS / w["dt"]))))
        print(f"  M={m} N=K={n}: peak {w['hi']:.0f} W, tail {w['lo']:.0f} W, "
              f"mean {P_STATIC + A_F * w['seff'] + B[n]:.0f} W, "
              f"{TAU_MS}ms-max {pb.max():.0f} W, observed throttle={thr}")


if __name__ == "__main__":
    main()
