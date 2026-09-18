#!/usr/bin/env python3
"""fig21 -- cold and hot expert workloads differ in LEVEL, not in SHAPE.

The two-domain scheme needs the cold partition to be most efficient at a lower clock
than the hot one -- a lower knee in E(f). The measurement separates the two properties
that are easy to conflate:

  LEVEL  -- how much energy a token costs. The partitions differ here by 1.77x, from
            padding and weaker operand reuse in the cold partition.
  SHAPE  -- how that cost varies with clock, and where it turns up. Identical.

(a) plots absolute energy on a log axis, so a pure level difference is a vertical
    offset and a shape difference would be a change of slope. The curves are parallel:
    no normalisation is needed to see it.
(b) locates the knee objectively, as the lowest clock at which a curve exceeds its own
    minimum by a given margin. All eight curves cross every threshold within one step
    of the measurement grid.
"""
import csv, glob, os, sys
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs

HERE = os.path.dirname(os.path.abspath(__file__))
COLD, HOT = fs.COMM, fs.GEMM
THRESH = [2, 5, 10, 20]


def load():
    pk = []
    for f in ([os.path.join(HERE, "data", "partition_kappa_qwen.csv")]
              + sorted(glob.glob(os.path.join(HERE, "data", "partition_kappa_L*.csv")))):
        L = 16 if f.endswith("qwen.csv") else int(f.split("_L")[1].split(".")[0])
        for x in csv.DictReader(open(f)):
            if x["held"] == "True" and int(x["block_m"]) == 128:
                x["L"] = L; pk.append(x)
    g = defaultdict(list)
    for x in pk:
        g[(x["L"], x["part"], int(x["clock"]))].append(x)
    real = {(x["L"], x["part"]): int(x["real_rows"]) for x in pk}
    cls = np.array(sorted({int(x["clock"]) for x in pk}))
    splits = [L for L in sorted({x["L"] for x in pk}) if L >= 16]
    E = {(L, p): np.array([np.mean([float(x["e_dyn_mj"]) for x in g[(L, p, c)]])
                           / real[(L, p)] * 1000 for c in cls])
         for L in splits for p in ("light", "heavy")}
    return cls, splits, E


def main():
    cls, splits, E = load()
    fs.use()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(fs.TEXT_W, 2.55),
                                 gridspec_kw=dict(width_ratios=[1.15, 1]))

    # (a) absolute, log y -- parallel curves mean identical shape ------------
    for L in splits:
        for p, col, mk in (("light", COLD, "s"), ("heavy", HOT, "o")):
            a1.plot(cls, E[(L, p)], "-" + mk, color=col, ms=2.2, lw=0.9, zorder=3)
    a1.set_yscale("log")
    a1.yaxis.set_major_locator(FixedLocator([10, 12, 15, 20, 25]))
    a1.yaxis.set_minor_locator(FixedLocator([]))
    a1.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
    a1.set_xlabel("SM clock (MHz)")
    a1.set_ylabel("energy per token ($\\mu$J, log)")
    a1.set_title("(a) the curves are parallel", loc="left", pad=3)
    a1.plot([], [], "-s", color=COLD, ms=2.6, label="cold partition")
    a1.plot([], [], "-o", color=HOT, ms=2.6, label="hot partition")
    a1.legend(loc="upper left", fontsize=7, handlelength=1.4)
    fs.despine(a1)
    # the 1.77x level gap is stated in the caption rather than drawn: eight curves
    # leave no empty space in this panel, and the point of (a) is the parallelism
    # (b) knee location, measured as a threshold crossing --------------------
    rng = np.random.default_rng(0)
    for L in splits:
        for p, col, mk in (("light", COLD, "s"), ("heavy", HOT, "o")):
            e = E[(L, p)]; mn = e.min(); imin = int(np.argmin(e))
            for j, t in enumerate(THRESH):
                idx = [i for i in np.where(e > mn * (1 + t / 100))[0] if i > imin]
                if idx:
                    a2.plot(cls[idx[0]], j + rng.uniform(-0.22, 0.22), mk, color=col,
                            ms=3.2, alpha=0.8, zorder=3)
    a2.set_yticks(range(len(THRESH)))
    a2.set_yticklabels([f"+{t}%" for t in THRESH])
    a2.set_ylim(-0.6, len(THRESH) - 0.4)
    a2.invert_yaxis()
    a2.set_xlabel("SM clock at the crossing (MHz)")
    a2.set_ylabel("margin above minimum")
    a2.set_title("(b) every curve turns up at the same clock", loc="left", pad=3)
    fs.despine(a2, grid_axis="x")

    fig.text(0.005, 0.99,
             "A100-SXM4-40GB  |  Qwen3-30B-A3B MoE layer  |  each partition run alone "
             "on 54 SMs, BLOCK_SIZE_M=128, 3 passes  |  8 curves: cold and hot, at "
             "splits of\n16, 32, 64 and 96 cold experts (2.9% to 57.8% of the tokens).",
             fontsize=7, color=fs.INK, va="top", ha="left", linespacing=1.35)
    fig.text(0.005, 0.012,
             "Cold and hot differ in LEVEL by 1.77x, and not at all in SHAPE. On log "
             "axes (a) a pure level difference is a vertical offset; a different knee "
             "would be a\ndifferent slope. In (b) all eight curves cross +2% at 1110 MHz "
             "and +20% at 1245 MHz, with zero spread; +5% and +10% vary by one grid step.",
             fontsize=7, color=fs.INK, va="bottom", ha="left", linespacing=1.35)
    fig.tight_layout(pad=0.3, w_pad=1.7, rect=(0, 0.100, 1, 0.855))
    print(fs.save(fig, "fig21_same_knee"))


if __name__ == "__main__":
    main()
