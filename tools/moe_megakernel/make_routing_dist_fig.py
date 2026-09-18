#!/usr/bin/env python3
"""fig11 -- the real per-expert GEMM shapes that MoE routing produces.

Everything the prediction rests on comes from this: OLMoE-1B-7B's own router, hooked at
the gate, on wikitext-103. Each expert e receives n_e tokens and its GEMM is
(n_e x 2048) @ (2048 x 1024), so n_e IS the M dimension. Nothing about the shape is
assumed -- only the tile size (128 x 128) is a choice.

(a) how skewed the loads are, and how that changes between prefill and decode
(b) the same thing as GEMM shapes: where M lands relative to the 128-row tile boundary,
    which is what decides how much of the last tile-row is wasted
(c) how concentrated the work is, which is what a partitioning scheme has to exploit
"""
import json, os
import numpy as np
import matplotlib.pyplot as plt
import figstyle as F

HERE = os.path.dirname(os.path.abspath(__file__))
Z = np.load(os.path.join(HERE, "data", "routing_olmoe.npz"))
meta = json.loads(str(Z["meta"]))
REG = [("prefill_b16", "prefill B=16", F.GEMM),
       ("prefill_b128", "prefill B=128", "#E09A4B"),
       ("decode_b64", "decode B=64", F.COMM)]

F.use()
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(F.TEXT_W, 2.3),
                                 gridspec_kw=dict(wspace=0.46))
fig.subplots_adjust(bottom=0.24)

# ---- (a) sorted load profile, median and spread across all layer-samples -------------
F.despine(a1)
for key, lab, col in REG:
    C = np.sort(Z[key].reshape(-1, 64).astype(float), axis=1)[:, ::-1]
    med = np.median(C, 0)
    lo, hi = np.percentile(C, 10, axis=0), np.percentile(C, 90, axis=0)
    x = np.arange(1, 65)
    a1.fill_between(x, np.maximum(lo, 0.5), np.maximum(hi, 0.5), color=col, alpha=0.20,
                    lw=0, zorder=2)
    a1.plot(x, np.maximum(med, 0.5), "-", lw=1.4, color=col, label=lab, zorder=4)
a1.axhline(128, color=F.INK, lw=0.8, ls=(0, (3, 2)), zorder=3)
a1.text(63, 150, "128 = one tile-row", fontsize=5.8, ha="right", color=F.INK)
a1.set_yscale("log")
a1.set_xlabel("expert rank")
a1.set_ylabel("tokens routed, $n_e$  ( = the GEMM's $M$ )")
a1.set_title("(a) real routed load per expert\n", fontsize=7.2, pad=3)
a1.legend(fontsize=5.8, loc="lower left", handlelength=1.4)
a1.set_xlim(1, 64)

# ---- (b) where M lands against the tile boundary -------------------------------------
F.despine(a2)
for key, lab, col in REG:
    C = Z[key].reshape(-1, 64).astype(float)
    C = C[C > 0]
    a2.hist(C % 128, bins=32, range=(0, 128), histtype="step", lw=1.3, color=col,
            density=True, zorder=4)
a2.set_xlabel("$M$ mod 128")
a2.set_ylabel("density")
a2.set_title("(b) 0 would mean a full tile-row;\nit almost never happens",
             fontsize=7.2, pad=3)
a2.set_xticks([0, 32, 64, 96, 128])
a2.text(0.5, 0.93, "flat $\\Rightarrow$ the last tile-row is\nhalf empty on average",
        transform=a2.transAxes, ha="center", va="top", fontsize=5.8, color=F.INK)

# ---- (c) concentration of the work ---------------------------------------------------
F.despine(a3)
for key, lab, col in REG:
    T = np.ceil(Z[key].reshape(-1, 64).astype(float) / 128) * 8
    cs = np.sort(T, 1)[:, ::-1].cumsum(1) / T.sum(1)[:, None]
    a3.plot(np.arange(1, 65), np.median(cs, 0) * 100, "-", lw=1.4, color=col, zorder=4)
a3.plot([1, 64], [100 / 64, 100], "--", lw=0.8, color=F.MUTED, zorder=2)
a3.text(40, 62, "perfectly balanced", fontsize=5.8, color=F.MUTED, rotation=27)
a3.set_xlabel("top-$k$ experts")
a3.set_ylabel("share of the layer's tiles (%)")
a3.set_title("(c) the top 16 experts hold 42%\nof the work", fontsize=7.2, pad=3)
a3.set_xlim(1, 64); a3.set_ylim(0, 102)

fig.text(0.5, -0.02, f"{meta['model']}, {meta['layers']} layers, {meta['experts']} "
         f"experts, top-{meta['top_k']}, on wikitext-103. Router hooked at the gate and "
         "top-k recomputed exactly as the block does it, so\nthese are the model's own "
         "routing decisions, not a synthetic skew. Bands in (a) are the 10th-90th "
         "percentile across all layer-samples. An expert's GEMM is "
         "$(n_e \\times 2048) @ (2048 \\times 1024)$,\nso $n_e$ is $M$; tiles "
         "$= \\lceil n_e/128 \\rceil \\times 8$. Decode at B=64 gives 97% of experts a "
         "single tile-row -- their GEMMs are too small to fill even one wave.",
         ha="center", va="top", fontsize=5.5, color=F.INK, linespacing=1.5)
print(F.save(fig, "fig11_routing_shapes"))
