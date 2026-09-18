#!/usr/bin/env python3
"""fig16 -- routing load profiles across six MoE families, same prompts.

Per-expert token count divided by the mean, experts sorted by load. Plotted against the
expert's RANK as a percentage so models with 8 and 128 experts share an x axis.

The dashed curve is the null for that model: the same token count thrown uniformly at
random over the same number of experts. It is not a "perfectly balanced" reference line --
it is what a uniform router would actually measure, sampling noise included, and the gap
between a solid curve and its own dashed partner is the skew the router is responsible for.
"""
import json, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import figstyle as F

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(0)
MODELS = [("routing_granite.npz", "Granite-3b-a800m", 40),
          ("routing_olmoe_sharegpt.npz", "OLMoE-1B-7B", 64),
          ("routing_qwen.npz", "Qwen3-30B-A3B", 128),
          ("routing_qwen3next.npz", "Qwen3-Next-80B-A3B", 512)]
# ordinal by expert count, so the ramp carries the variable under study
RAMP = LinearSegmentedColormap.from_list("e", ["#9AA6DF", F.COMM, "#B8860B", F.GEMM,
                                               "#7A3F02"])


def gini(x):
    x = np.sort(np.asarray(x, float), -1)
    n = x.shape[-1]; i = np.arange(1, n + 1); s = x.sum(-1)
    return np.where(s > 0, (2 * (x * i).sum(-1) - (n + 1) * s) / (n * np.maximum(s, 1)), 0)


F.use()
fig, ax = plt.subplots(figsize=(F.TEXT_W * 0.80, 2.7))
fig.subplots_adjust(bottom=0.26)
F.despine(ax)
info = []
for i, (fn, name, E) in enumerate(MODELS):
    p = os.path.join(HERE, "data", fn)
    if not os.path.exists(p):
        continue
    Z = np.load(p)
    m = json.loads(str(Z["meta"]))
    A = Z[next(k for k in Z.files if k.startswith("prefill_b"))].astype(float)
    if A.ndim == 2:
        A = A[None]
    C = A.reshape(-1, m["experts"])        # all repeats x layers pooled for the profile
    n = int(np.median(C.sum(1)))
    null = rng.multinomial(n, [1 / m["experts"]] * m["experts"], size=300).astype(float)
    col = RAMP(i / (len(MODELS) - 1))
    x = np.linspace(0, 100, m["experts"])
    # share of the layer's dispatched tokens, in per cent
    sh = np.sort(C / C.sum(1, keepdims=True) * 100, 1)[:, ::-1]
    obs = np.median(sh, 0)
    ax.plot(x, np.maximum(obs, 1e-4), "-", lw=1.6, color=col, zorder=5,
            label=f"{name}  ({m['experts']}e, top-{m['top_k']})")
    ax.fill_between(x, np.maximum(np.percentile(sh, 10, axis=0), 1e-4),
                    np.percentile(sh, 90, axis=0), color=col, alpha=0.13, lw=0, zorder=2)
    info.append((name, m["experts"], float(np.median(gini(C)) - np.median(gini(null)))))

ax.set_yscale("log")
ax.set_xlabel("experts, sorted by load (rank, %)")
ax.set_ylabel("share of the layer's dispatched tokens (%)")
ax.set_title("Routing load profile across MoE families\n"
             "24 independent ShareGPT prompt batches per model",
             fontsize=7.4, pad=4)
ax.legend(fontsize=5.8, loc="lower left", handlelength=1.6, labelspacing=0.3)
ax.set_xlim(0, 100)
ax.text(0.985, 0.96, "band = 10th-90th pct over 24 batches x all layers",
        transform=ax.transAxes, ha="right", va="top", fontsize=5.8, color=F.MUTED)

cap = ("Median over 24 independent ShareGPT prompt batches x all layers. Each expert's "
       "share of the tokens its layer\ndispatched, experts sorted by load, rank on a "
       "percentage axis so 40 and 512 experts share a scale.\nA uniform router would put "
       "every model on its own flat line at 100/E %: 2.5% at 40 experts, 0.20% at 512.\n"
       "The excess Gini below is null-corrected for sampling noise even though the null "
       "is not drawn here.\n\nexcess Gini:  "
       + "   ".join(f"{g:+.2f} ({e}e)" for _, e, g in info))
fig.text(0.5, -0.02, cap, ha="center", va="top", fontsize=5.5, color=F.INK,
         linespacing=1.5)
print(F.save(fig, "fig16_skew_across_models"))
