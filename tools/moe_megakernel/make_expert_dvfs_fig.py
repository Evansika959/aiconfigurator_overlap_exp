#!/usr/bin/env python3
"""fig20 -- cold/hot experts on separate SM voltage domains, on our persistent kernel.

Kept deliberately sparse: one split shown, one label per series, the argument in the
caption. The 4-split overlay that establishes the curves collapse is in fig20_appendix.
"""
import csv, glob, os, sys
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs

HERE = os.path.dirname(os.path.abspath(__file__))
NCOLD = 16
COLD, HOT = fs.COMM, fs.GEMM


def load_partitions():
    pk = []
    for f in ([os.path.join(HERE, "data", "partition_kappa_qwen.csv")]
              + sorted(glob.glob(os.path.join(HERE, "data", "partition_kappa_L*.csv")))):
        L = NCOLD if f.endswith("qwen.csv") else int(f.split("_L")[1].split(".")[0])
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
    SD = {(L, p): np.array([np.std([float(x["e_dyn_mj"]) for x in g[(L, p, c)]], ddof=1)
                            / real[(L, p)] * 1000 for c in cls])
          for L in splits for p in ("light", "heavy")}
    return cls, splits, E, SD


def agg(rows, case, key="ms"):
    v = np.array([float(r[key]) for r in rows if r["case"] == case])
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)


def main(appendix=False):
    counts = np.load(os.path.join(HERE, "data",
                                  "routing_qwen.npz"))["prefill_b64"][0, 0].astype(int)
    srt = np.sort(counts)
    cls, splits, E, SD = load_partitions()
    pq = list(csv.DictReader(open(os.path.join(HERE, "data",
                                               "pin_vs_queue_qwen.csv"))))
    bestpin = min((c for c in {r["case"] for r in pq} if c.startswith("pinned")),
                  key=lambda c: agg(pq, c)[0])

    fs.use()
    fig = plt.figure(figsize=(fs.TEXT_W, 3.9))
    a1 = fig.add_axes([0.095, 0.690, 0.885, 0.200])
    a2 = fig.add_axes([0.095, 0.180, 0.470, 0.370])
    a3 = fig.add_axes([0.715, 0.180, 0.265, 0.370])

    # (a) the split ---------------------------------------------------------
    x = np.arange(len(srt))
    a1.bar(x[:NCOLD], srt[:NCOLD], color=COLD, width=1.0, zorder=3,
           label=f"cold: {NCOLD} experts, slow domain")
    a1.bar(x[NCOLD:], srt[NCOLD:], color=HOT, width=1.0, zorder=3,
           label=f"hot: {len(srt)-NCOLD} experts, fast domain")
    a1.set_xlim(-1, len(srt))
    a1.set_xlabel("experts, sorted by tokens routed to them", labelpad=2)
    a1.set_ylabel("tokens", labelpad=2)
    a1.set_title("(a) the split", loc="left", pad=3)
    a1.legend(loc="upper left", fontsize=7, handlelength=1.0)
    fs.despine(a1)

    # (b) energy vs clock, each partition alone -------------------------------
    show = splits if appendix else [NCOLD]
    for L in show:
        for p, col, mk, lab in (("light", COLD, "s", "cold partition"),
                                ("heavy", HOT, "o", "hot partition")):
            a2.errorbar(cls, E[(L, p)], yerr=SD[(L, p)], fmt="-" + mk, color=col,
                        ms=2.8, lw=1.1, elinewidth=0.6, capsize=1.4, zorder=3,
                        label=lab if L == show[0] else None, alpha=0.9)
    a2.axvspan(cls.min(), 1065, color=fs.GRID, alpha=0.5, zorder=0)
    lo = min(v.min() for v in E.values()); hi = max(v.max() for v in E.values())
    a2.set_ylim(lo - 1.5, hi + 1.0)
    a2.text(1065 - 15, hi + 0.6, "flat within 1%", ha="right", va="top", fontsize=7,
            color=fs.MUTED)
    a2.set_xlabel("SM clock (MHz)")
    a2.set_ylabel("dynamic energy per token ($\\mu$J)")
    a2.set_title("(b) each partition run alone, swept over clock", loc="left", pad=3)
    a2.legend(loc="upper left" if appendix else "center left", fontsize=7,
              handlelength=1.4)
    fs.despine(a2)

    # (c) cost ---------------------------------------------------------------
    bars = [("vLLM", "queue", fs.BASE, ""),
            ("persistent", "persistent_2blk_per_sm", COLD, fs.HATCH["comm"]),
            ("2 domains", bestpin, HOT, fs.HATCH["gemm"])]
    v = [agg(pq, c) for _, c, _, _ in bars]
    q = v[0][0]
    xb = np.arange(len(bars))
    a3.bar(xb, [t[0] for t in v], yerr=[t[1] for t in v], capsize=2,
           error_kw=dict(elinewidth=0.7, ecolor=fs.INK),
           color=[b[2] for b in bars], hatch=[b[3] for b in bars],
           edgecolor="white", width=0.66, zorder=3)
    for i, t in enumerate(v):
        a3.text(i, t[0] + t[1] + 0.08, "" if i == 0 else f"{(t[0]/q-1)*100:+.1f}%",
                ha="center", va="bottom", fontsize=7, color=fs.INK)
    a3.set_xticks(xb); a3.set_xticklabels([b[0] for b in bars], fontsize=7)
    a3.set_ylabel("layer latency (ms)")
    a3.set_ylim(0, max(t[0] for t in v) * 1.22)
    a3.set_title("(c) kernel cost", loc="left", pad=3)
    fs.despine(a3)

    fig.text(0.005, 0.99,
             "A100-SXM4-40GB  |  Qwen3-30B-A3B MoE layer, 7744 prefill tokens, real "
             "ShareGPT routing  |  persistent kernel, one block per SM, output "
             "identical to vLLM.",
             fontsize=7, color=fs.INK, va="top", ha="left")
    fig.text(0.005, 0.012,
             "For the scheme to save energy the cold partition must prefer a lower "
             "clock. It has no preference: both curves are flat within 1% below 1065 "
             "MHz and rise\ntogether above it. The same holds at 32, 64 and 96 cold "
             "experts (appendix). Meanwhile a partitionable kernel costs +5.6% latency "
             "before any DVFS.",
             fontsize=7, color=fs.INK, va="bottom", ha="left", linespacing=1.35)
    print(fs.save(fig, "fig20_expert_routed_dvfs" + ("_appendix" if appendix else "")))


if __name__ == "__main__":
    main(appendix=False)
    main(appendix=True)
