#!/usr/bin/env python3
"""fig25 -- the simulator against the A100 on a load that moves.

`governor_trace.py --plan trace` ran fourteen segments of mixed batch size, with idle
gaps, under the stock governor. `python3 -m sim.validate_trace` replayed the same script
through the event-driven simulator and wrote both to sim/out/trace_trace_run0.csv.

  top     SM clock. Measured: NVML, which refreshes about every 100 ms.
  middle  board power. Measured: NVML. Simulated: the 100 ms mean NVML would report.
  bottom  time per MoE layer. Measured: every graph replay (4 layers). This is the
          finest-grained observation of the three and the one a clock error shows in.

Nothing was fitted to this trace. Where the two disagree it is left visible.

THE HELD-OUT TRACES HAVE TWO FIGURES. `heldout1` and `heldout2` were replayed once by a
frozen simulator, and that replay is the registered result (HELDOUT.md). Its series are
kept in data/heldout_first_shot and `--first-shot` draws them as fig25_sim_<tag>. A
replay by the working tree is a different thing and never overwrites it: without the
flag those tags are written as fig25_sim_<tag>_second_shot.
"""
import argparse, csv, json, os, sys
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs

HERE = os.path.dirname(os.path.abspath(__file__))
MEAS, SIM = fs.INK, fs.GEMM
OUT = os.path.join(HERE, "sim", "out")                        # the working tree's replays
FIRST_SHOT = os.path.join(HERE, "data", "heldout_first_shot")  # the frozen simulator's


def load(tag, run, src=OUT):
    S = defaultdict(lambda: ([], []))
    for r in csv.DictReader(open(os.path.join(src, f"trace_{tag}_run{run}.csv"))):
        S[r["series"]][0].append(float(r["t_ms"]) / 1e3)
        S[r["series"]][1].append(float(r["value"]))
    segs = [dict(load=r["load"], tokens=int(r["tokens"]), t0=float(r["t_start_ms"]) / 1e3,
                 t1=float(r["t_end_ms"]) / 1e3)
            for r in csv.DictReader(open(os.path.join(HERE, "data", f"governor_trace_{tag}_segments_qwen.csv")))
            if int(r["run"]) == run]
    return {k: (np.array(v[0]), np.array(v[1])) for k, v in S.items()}, segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="trace")
    ap.add_argument("--run", type=int, default=0)
    ap.add_argument("--first-shot", action="store_true",
                    help="a held-out trace as the frozen simulator replayed it, the registered result")
    a = ap.parse_args()
    held_out = os.path.exists(os.path.join(FIRST_SHOT, f"trace_{a.tag}.json"))
    if a.first_shot and not held_out:
        raise SystemExit(f"{a.tag} has no first-shot replay in {FIRST_SHOT}")
    src = FIRST_SHOT if a.first_shot else OUT
    stem = f"fig25_sim_{a.tag}" + ("_second_shot" if held_out and not a.first_shot else "")
    S, segs = load(a.tag, a.run, src)
    rep = json.load(open(os.path.join(src, f"trace_{a.tag}.json")))
    fs.use()
    W, H = fs.TEXT_W, 3.55
    fig = plt.figure(figsize=(W, H))
    x0, wid = 0.50, W - 0.50 - 0.08
    axs = [fig.add_axes([x0 / W, y / H, wid / W, h / H])
           for y, h in ((2.42, 0.78), (1.42, 0.86), (0.36, 0.92))]
    t_end = segs[-1]["t1"]

    ac, ap_, al = axs
    t, v = S["meas_clock_mhz"]; ac.step(t, v, where="post", color=MEAS, lw=0.9, zorder=3)
    t, v = S["sim_clock_exact_mhz"]; ac.step(t, v, where="post", color=SIM, lw=0.9, zorder=4)
    ac.set_ylim(1050, 1440); ac.set_yticks([1100, 1200, 1300, 1400])
    ac.set_ylabel("SM clock (MHz)")

    t, v = S["meas_power_w"]; ap_.step(t, v, where="post", color=MEAS, lw=0.9, zorder=3)
    t, v = S["sim_power_w"]; ap_.step(t, v, where="pre", color=SIM, lw=0.9, zorder=4)
    ap_.axhline(400, color=fs.INK, lw=0.5, ls=(0, (3, 2)), zorder=2)
    ap_.text(t_end * 0.995, 408, "TDP", ha="right", va="bottom", fontsize=7, color=fs.INK)
    ap_.set_ylim(0, 470); ap_.set_yticks([0, 100, 200, 300, 400])
    ap_.set_ylabel("board power (W)")

    t, v = S["meas_layer_ms"]; al.plot(t, v, ls="", marker=".", ms=1.2, color=MEAS, zorder=3,
                                      rasterized=True)
    t, v = S["sim_layer_ms"]; al.plot(t, v, ls="", marker=".", ms=1.0, color=SIM, zorder=4,
                                     rasterized=True)
    al.set_yscale("log")
    al.set_yticks([1, 2, 4]); al.set_yticklabels(["1", "2", "4"])
    al.minorticks_off()
    al.set_ylim(0.85, 5.2)
    al.set_ylabel("ms per layer")
    al.set_xlabel("time (s)")

    for ax in axs:
        ax.set_xlim(0, t_end)
        fs.despine(ax)
        for s in segs:                                  # idle gaps, shaded
            if s["load"] == "idle":
                ax.axvspan(s["t0"], s["t1"], color=fs.GRID, lw=0, zorder=1)
    for ax in axs[:2]:
        ax.tick_params(labelbottom=False)
    # what is running, along the top. A label that would run into its neighbour moves up
    # a row; one that fits in neither row is left out (a segment shorter than its label).
    s_per_char = 0.047 / wid * t_end                    # 7 pt type, in seconds of x-axis
    right = [-1e9, -1e9]                                # where the last label of each row ends
    for s in segs:
        lab = "idle" if s["load"] == "idle" else (str(s["tokens"]) + "*" if s["load"].startswith("p")
                                                  else str(s["tokens"]))
        mid, half = (s["t0"] + s["t1"]) / 2, 0.5 * len(lab) * s_per_char
        row = next((k for k in (0, 1) if mid - half >= right[k] + s_per_char), None)
        if row is None or (s["load"] == "idle" and s["t1"] - s["t0"] < 2 * half):
            continue
        right[row] = mid + half
        ac.text(mid, 1452 + 52 * row, lab, ha="center", va="bottom", fontsize=7, color=fs.INK)
    ac.text(0, 1.30, "tokens per batch (* prefill):", transform=ac.transAxes, fontsize=7,
            va="bottom", ha="left", color=fs.MUTED)
    hs = [plt.Line2D([], [], color=MEAS, lw=1.0), plt.Line2D([], [], color=SIM, lw=1.0)]
    fig.legend(hs, ["measured on the A100, stock governor", "simulated"], loc="upper right",
               ncol=2, bbox_to_anchor=(1.0, 1.005), columnspacing=1.4)
    out = fs.save(fig, stem)
    s = rep["summary"]
    print(f"  {s['segments']} segments over {s['runs']} runs: latency rms {s['latency_rms_pct']:.2f}%, "
          f"power rms {s['power_rms_pct']:.2f}%, clock rms {s['clock_rms_mhz']:.0f} MHz, energy "
          + ", ".join(f"{x:+.1f}%" for x in s["energy_err_pct"]))
    print("wrote", *out)


if __name__ == "__main__":
    main()
