#!/usr/bin/env python3
"""fig15 -- what "excess Gini" means, drawn instead of asserted.

Gini alone cannot be read: a PERFECTLY uniform router still produces a large Gini when
there are few tokens per expert, purely from multinomial noise (0.51 at 1 token/expert,
0.02 at 1000). So every measurement is quoted against a null -- the same number of tokens
thrown uniformly at random over the same experts -- and the EXCESS is what the router did
beyond chance.

(a) The load profile itself: per-expert token count over the mean, sorted. The null is the
    band a random router would occupy.
(b) The Lorenz curve, which is what Gini actually measures: cumulative share of tokens
    against cumulative share of experts. The diagonal is perfect balance; Gini is twice
    the area between a curve and that diagonal. Excess Gini is twice the area between the
    OBSERVED curve and the NULL curve -- the shaded region.
"""
import json, os
import numpy as np
import matplotlib.pyplot as plt
import figstyle as F

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(0)
CASES = [("routing_olmoe_sharegpt.npz", "prefill_b64", 64, "OLMoE, 64 experts", F.COMM),
         ("routing_qwen.npz", "prefill_b64", 128, "Qwen3, 128 experts", F.GEMM)]


def gini(x):
    x = np.sort(np.asarray(x, float), -1)
    n = x.shape[-1]
    i = np.arange(1, n + 1)
    s = x.sum(-1)
    return np.where(s > 0, (2 * (x * i).sum(-1) - (n + 1) * s) / (n * np.maximum(s, 1)), 0)


def lorenz(x):
    x = np.sort(np.asarray(x, float))
    c = np.concatenate([[0], np.cumsum(x) / max(x.sum(), 1)])
    return np.linspace(0, 1, len(c)), c


F.use()
fig, (a1, a2) = plt.subplots(1, 2, figsize=(F.TEXT_W, 2.5),
                             gridspec_kw=dict(wspace=0.34))
fig.subplots_adjust(bottom=0.24)

for fn, key, E, lab, col in CASES:
    Z = np.load(os.path.join(HERE, "data", fn))
    C = Z[key].reshape(-1, E).astype(float)
    n = int(np.median(C.sum(1)))
    null = rng.multinomial(n, [1 / E] * E, size=400).astype(float)

    # ---- (a) sorted profile, normalised to the mean -------------------------------
    obs = np.sort(C / np.maximum(C.mean(1, keepdims=True), 1e-9), 1)[:, ::-1]
    nul = np.sort(null / null.mean(1, keepdims=True), 1)[:, ::-1]
    x = np.linspace(0, 100, E)
    a1.plot(x, np.median(obs, 0), "-", lw=1.6, color=col, zorder=5, label=lab)
    a1.fill_between(x, np.percentile(nul, 5, axis=0), np.percentile(nul, 95, axis=0),
                    color=col, alpha=0.18, lw=0, zorder=2)
    a1.plot(x, np.median(nul, 0), "--", lw=0.9, color=col, alpha=0.8, zorder=3)

    # ---- (b) Lorenz ----------------------------------------------------------------
    lo, co = lorenz(np.median(C, 0))
    ln, cn = lorenz(np.median(null, 0))
    a2.fill_between(lo, co, np.interp(lo, ln, cn), color=col, alpha=0.25, lw=0, zorder=3)
    a2.plot(lo, co, "-", lw=1.6, color=col, zorder=5, label=lab)
    a2.plot(ln, cn, "--", lw=0.9, color=col, alpha=0.8, zorder=4,
            label="_null" if key else None)
    g = float(np.median(gini(C)))
    g0 = float(np.median(gini(null)))
    print(f"  {lab:<22} n={n:<6} gini {g:.2f}  null {g0:.2f}  excess {g-g0:+.2f}")

F.despine(a1)
a1.axhline(1.0, color=F.INK, lw=0.7, ls=(0, (3, 2)), zorder=1)
a1.text(97, 1.12, "perfectly balanced", fontsize=6, ha="right", color=F.INK)
a1.set_yscale("log")
a1.set_xlabel("experts, sorted by load (%)")
a1.set_ylabel("tokens received / mean")
a1.set_title("(a) the load profile", fontsize=7.4, pad=3)
a1.legend(fontsize=6, loc="lower left", handlelength=1.5)
a1.text(4, 0.28, "dashed + band =\na uniform router\n(sampling noise only)",
        fontsize=5.8, color=F.MUTED, va="top")

F.despine(a2, grid_axis=None)
a2.plot([0, 1], [0, 1], "-", lw=0.8, color=F.INK, zorder=2)
a2.text(0.80, 0.86, "perfect balance", fontsize=6, color=F.INK, rotation=38,
        ha="center", va="center")
a2.set_xlabel("cumulative share of experts")
a2.set_ylabel("cumulative share of tokens")
a2.set_title("(b) shaded area $\\times$ 2 = excess Gini", fontsize=7.4, pad=3)
a2.text(0.03, 0.97, "solid = measured\ndashed = uniform router", transform=a2.transAxes,
        fontsize=5.8, color=F.MUTED, va="top")
a2.set_xlim(0, 1); a2.set_ylim(0, 1)
a2.annotate("excess = what the router\ndid beyond chance", xy=(0.55, 0.34),
            xytext=(0.62, 0.14), fontsize=5.8, color=F.INK, ha="center",
            arrowprops=dict(arrowstyle="->", lw=0.6, color=F.INK))

fig.text(0.5, -0.02,
    "Gini alone is unreadable: a uniform router scores 0.51 at 1 token per expert and "
    "0.02 at 1000, purely from multinomial noise. So each\nmeasurement is compared with "
    "that null -- the same token count thrown uniformly over the same experts -- and the "
    "EXCESS is the part the\nrouter is responsible for. Both panels are prefill B=64 on "
    "ShareGPT, the one setting where the two models are directly comparable. OLMoE: "
    "gini\n0.24, null 0.02, excess +0.22. Qwen3: gini 0.46, null 0.03, excess +0.44 -- "
    "twice the imbalance, on identical prompts.",
    ha="center", va="top", fontsize=5.6, color=F.INK, linespacing=1.5)
print(F.save(fig, "fig15_excess_gini"))
