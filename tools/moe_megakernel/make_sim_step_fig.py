#!/usr/bin/env python3
"""fig28 -- the first 400 ms after a load step: what the power limiter does, and whether
the simulator's limiter does the same.

NVML refreshes about every 100 ms, so in fig25 a load step is one or two readings. But
`governor_trace.py` timed every graph replay -- 4 layers, 4-16 ms -- and a layer's time
follows the clock, so the replays are a clock witness ten times finer than NVML.

  top     time per MoE layer, one point per graph replay. Measured and simulated, replay
          k against replay k.
  bottom  SM clock. Simulated: every set-point. Measured: what NVML reported.

Four steps of the `trace2` plan, which the limiter's numbers were NOT set against
(`sim.validate_trace --tag trace2` writes sim/out/trace_trace2_steps.csv):

  (a) a heavier batch replaces a lighter one: the clock falls within 60 ms
  (b) a lighter batch replaces a heavier one: it rises as fast
  (c) a heavy batch after a short idle gap, the clock still at the top
  (d) a heavy batch after a long one, the clock parked at 1095 MHz: nothing for 0.2-0.35 s,
      then one jump

Time per layer is in ms on each panel's own axis. Nothing is normalised, so where the
simulator's steady state is off -- 2.4% fast on the 512-token batch in (b) -- it shows.
"""
import argparse, csv, os, sys
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs

HERE = os.path.dirname(os.path.abspath(__file__))
MEAS, SIM = fs.INK, fs.GEMM
# (segment, title): what was running before, and what arrives
PANELS = {"trace2": [(6, "(a) 1024 to 3072 tokens"), (2, "(b) prefill to 512 tokens"),
                     (4, "(c) idle 0.5 s, then 8192"), (12, "(d) idle 3.5 s, then 6144")]}
X_MAX = 400.0


def load(tag, run):
    steps = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(HERE, "sim", "out", f"trace_{tag}_steps.csv"))):
        if int(r["run"]) == run:
            steps[int(r["seg"])].append(r)
    S = defaultdict(lambda: ([], []))
    for r in csv.DictReader(open(os.path.join(HERE, "sim", "out", f"trace_{tag}_run{run}.csv"))):
        if r["series"] in ("meas_clock_mhz", "sim_clock_exact_mhz"):
            S[r["series"]][0].append(float(r["t_ms"])); S[r["series"]][1].append(float(r["value"]))
    return steps, {k: (np.array(v[0]), np.array(v[1])) for k, v in S.items()}


def window(t, v, t0, lo=-40.0, hi=X_MAX):
    """A sample-and-hold series cut to [t0 + lo, t0 + hi], time measured from t0."""
    i = max(int(np.searchsorted(t, t0 + lo, side="right")) - 1, 0)
    j = int(np.searchsorted(t, t0 + hi, side="right"))
    tt, vv = t[i:j] - t0, v[i:j]
    return np.append(np.maximum(tt, lo), hi), np.append(vv, vv[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="trace2")
    ap.add_argument("--run", type=int, default=0)
    a = ap.parse_args()
    steps, S = load(a.tag, a.run)
    panels = PANELS[a.tag]
    fs.use()
    W, H = fs.TEXT_W, 2.62
    fig = plt.figure(figsize=(W, H))
    n = len(panels)
    left, gap, right = 0.47, 0.40, 0.05
    pw = (W - left - right - gap * (n - 1)) / n
    for i, (seg, title) in enumerate(panels):
        x0 = left + i * (pw + gap)
        at = fig.add_axes([x0 / W, 1.37 / H, pw / W, 0.90 / H])
        ac = fig.add_axes([x0 / W, 0.36 / H, pw / W, 0.84 / H])
        rows = steps[seg]
        tm = np.array([float(r["t_ms"]) for r in rows]); mm = np.array([float(r["meas_ms"]) for r in rows])
        ts = np.array([float(r["sim_t_ms"]) for r in rows]); ms = np.array([float(r["sim_ms"]) for r in rows])
        at.plot(ts, ms, color=SIM, lw=1.0, zorder=3)
        at.plot(tm, mm, ls="", marker="o", ms=1.5, mfc=MEAS, mec="none", zorder=4)
        lo, hi = min(mm[tm <= X_MAX].min(), ms[ts <= X_MAX].min()), max(mm[tm <= X_MAX].max(), ms[ts <= X_MAX].max())
        pad = 0.12 * (hi - lo)
        at.set_ylim(lo - pad, hi + pad)
        at.set_title(title, loc="left", fontsize=7, pad=3)
        at.tick_params(labelbottom=False)
        t, v = window(*S["meas_clock_mhz"], float(rows[0]["t0_ms"]))
        ac.step(t, v, where="post", color=MEAS, lw=0.9, zorder=3)
        t, v = window(*S["sim_clock_exact_mhz"], float(rows[0]["sim_t0_ms"]))
        ac.step(t, v, where="post", color=SIM, lw=1.0, zorder=4)
        if seg == panels[-1][0]:
            ac.set_ylim(1060, 1440); ac.set_yticks([1100, 1200, 1300, 1400])
        else:
            ac.set_ylim(1290, 1425); ac.set_yticks([1300, 1350, 1400])
        for ax in (at, ac):
            ax.set_xlim(0, X_MAX); ax.set_xticks([0, 100, 200, 300, 400])
            fs.despine(ax)
        if i == 0:
            at.set_ylabel("ms per layer"); ac.set_ylabel("SM clock (MHz)")
    fig.text((left + (W - left - right) / 2) / W, 0.035 / H, "time since the load step (ms)",
             ha="center", va="bottom", fontsize=8)
    hs = [plt.Line2D([], [], color=MEAS, ls="", marker="o", ms=2.6, mec="none"),
          plt.Line2D([], [], color=MEAS, lw=0.9), plt.Line2D([], [], color=SIM, lw=1.0)]
    fig.legend(hs, ["measured, every graph replay", "measured, NVML", "simulated"],
               loc="upper right", ncol=3, bbox_to_anchor=(1.0, 1.0), columnspacing=1.4)
    out = fs.save(fig, "fig28_sim_load_step")
    print("wrote", *out)


if __name__ == "__main__":
    main()
