#!/usr/bin/env python3
"""fig19 -- why giving the lightly-routed experts a smaller tile makes things worse.

The obvious lever for a custom MoE kernel: at BLOCK_SIZE_M=128 the 16 least-loaded
experts of this layer compute 3200 padded rows for 1821 real tokens, 76% waste. Shrink
the tile and the waste goes away. It does, and the layer gets 2.8x slower.

A FIRST VERSION OF THIS FIGURE GAVE THE WRONG REASON, and the wrong reason is the
intuitive one. It counted the bytes a smaller tile *asks* for -- every row-block sweeps
its expert's whole 9 MiB weight panel, so eight times the row-blocks is eight times the
weight requests -- and presented that as memory traffic. Measured with ncu, DRAM traffic
is FLAT: 2000 MiB at BLOCK_M=16 against 2099 at 128. L2 absorbs every one of those
re-reads. So does the arithmetic: tensor-core instructions barely move, and are in fact
12% HIGHER at 128 because of the padding.

What actually changes by 2.6x is on-chip data movement. A 16-row tile pulls the same B
fragment out of L2 and through the register file to multiply it against one eighth as
many rows of A, so the machine spends its time shuffling operands rather than issuing
MMAs: SM throughput falls from 79% of peak to 24%.

The padding argument is not wrong about the padding. It is wrong about what a MoE expert
GEMM is short of, which is not DRAM bandwidth and not flops.
"""
import csv, os, sys
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    n = {int(r["block_m"]): r for r in
         csv.DictReader(open(os.path.join(HERE, "data", "tile_ncu_qwen.csv")))}
    BMS = sorted(n)
    l1 = np.array([float(n[b]["l1_bank_reads"]) for b in BMS]) / 1e9
    dram = np.array([float(n[b]["dram_mib"]) for b in BMS]) / 1024
    tens = np.array([float(n[b]["tensor_inst"]) for b in BMS]) / 1e6

    rows = [x for x in csv.DictReader(
        open(os.path.join(HERE, "data", "partition_kappa_qwen.csv")))
        if x["held"] == "True" and x["part"] == "light"]
    g = defaultdict(list)
    for x in rows:
        g[(int(x["block_m"]), int(x["clock"]))].append(x)
    real = int(rows[0]["real_rows"])
    uj, sd = [], []
    for b in BMS:
        per = {c: np.array([float(x["e_dyn_mj"]) for x in v]) / real * 1000
               for (bb, c), v in g.items() if bb == b}
        best = min(per, key=lambda c: per[c].mean())
        uj.append(per[best].mean()); sd.append(per[best].std(ddof=1))

    fs.use()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(fs.TEXT_W, 2.6),
                                 gridspec_kw=dict(width_ratios=[1.3, 1]))
    x = np.arange(len(BMS))

    a1.bar(x, l1, color=fs.GEMM, hatch=fs.HATCH["gemm"], edgecolor="white",
           width=0.66, zorder=3)
    for i, v in enumerate(l1):
        a1.text(i, v + 0.35, f"{v:.1f}", ha="center", va="bottom", fontsize=7,
                color=fs.INK)
    a1.set_xticks(x); a1.set_xticklabels([str(b) for b in BMS])
    a1.set_xlabel("BLOCK_SIZE_M (token rows per tile)")
    a1.set_ylabel("on-chip data movement\n(L1 bank reads, billions)")
    a1.set_ylim(0, l1.max() * 1.42)
    a1.set_title("(a) what a smaller tile actually costs", loc="left", pad=4)
    fs.despine(a1)
    a1.text(0.5, l1.max() * 1.31,
            "the two things that DON'T change:\n"
            "  DRAM traffic  %.2f$-$%.2f GiB\n"
            "  tensor-core work  %.0f$-$%.0f M instructions"
            % (dram.min(), dram.max(), tens.min(), tens.max()),
            fontsize=7, color=fs.MUTED, ha="left", va="top")

    a2.errorbar(x, uj, yerr=sd, fmt="-o", color=fs.COMM, ms=3.4, lw=1.2,
                elinewidth=0.7, capsize=2, zorder=3)
    a2.plot([x[int(np.argmin(uj))]], [min(uj)], "o", color=fs.COMM, ms=8, mfc="none",
            mew=1.0, zorder=4)
    a2.annotate("best", (x[int(np.argmin(uj))], min(uj)), textcoords="offset points",
                xytext=(-3, 13), fontsize=7, color=fs.INK, ha="center")
    a2.text(0.07, uj[0], f"+{uj[0]/min(uj)-1:.0%}", fontsize=7, color=fs.INK,
            ha="left", va="center")
    a2.set_xticks(x); a2.set_xticklabels([str(b) for b in BMS])
    a2.set_xlabel("BLOCK_SIZE_M")
    a2.set_ylabel("energy per real token ($\\mu$J)")
    a2.set_title("(b) and what it costs in energy", loc="left", pad=4)
    fs.despine(a2)

    fig.text(0.005, 0.99,
             "A100-SXM4-40GB  |  Qwen3-30B-A3B MoE layer, 7744 prefill tokens, real "
             "ShareGPT routing.  (a) ncu, whole layer, both grouped GEMMs.  (b) the 16\n"
             "least-loaded experts (1821 real tokens, 3$-$217 each) run alone on 54 SMs, "
             "each tile size at its own energy-optimal clock, 3 passes, mean $\\pm$ sd.\n"
             "A 16-row tile reuses each B fragment one eighth as often as a 128-row "
             "tile, so SM throughput falls from 79% of peak to 24% while DRAM traffic\n"
             "and tensor-core work stay put. Padding is real -- 76% of the light "
             "partition's rows at BLOCK_SIZE_M=128 -- and it is not what this kernel is "
             "short of.",
             fontsize=7, color=fs.INK, va="top", ha="left", linespacing=1.35)
    fig.tight_layout(pad=0.3, w_pad=1.6, rect=(0, 0, 1, 0.775))
    print(fs.save(fig, "fig19_tile_size"))


if __name__ == "__main__":
    main()
