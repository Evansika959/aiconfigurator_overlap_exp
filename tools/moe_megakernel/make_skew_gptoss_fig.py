#!/usr/bin/env python3
"""fig16c -- routing load profiles across MoE families, with GPT-OSS-20B added.

fig16 (make_skew_fig.py) with one family added on 2026-10-01: GPT-OSS-20B (32 experts,
top-4). Same quantity, same axes: each expert's share of the tokens its layer dispatched,
experts sorted by load, rank on a percentage axis so 32 and 512 experts share a scale.
Median over 24 independent ShareGPT prompt batches x all layers; the band is the
10th-90th percentile.

  python3 make_skew_gptoss_fig.py                 # -> figs/fig16c_skew_with_gptoss.png
  python3 make_skew_gptoss_fig.py --with-llama4   # -> figs/fig16b_skew_six_models.png

Llama 4 Scout was traced too and is NOT in the comparison (analyse_skew.ROLE says why);
`--with-llama4` redraws the six-model version that was made before that was decided.

COLOUR FOLLOWS THE MODEL. The four families of fig16 keep the colours they have there, so
the figures can be read side by side; an added family gets a new hue and a broken line,
so it can be told apart without colour. Checked pairwise under protanopia and
deuteranopia (Machado 2009, severity 1.0), in OKLab dE x 100 against a target of 8: the
closest pair is 13.5 without Llama 4 and 9.3 with it.

The excess Gini in the legend is null-corrected: the model's Gini minus that of a uniform
router throwing the same number of tokens at the same number of experts
(analyse_skew.py, data/skew_across_models_v2.csv).
"""
import argparse
import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np

import figstyle as F

HERE = os.path.dirname(os.path.abspath(__file__))
# (trace, name, tag in skew_across_models_v2.csv, colour, line style); by expert count
LLAMA4 = ("routing_llama4.npz", "Llama-4-Scout", "llama4", "#117555", (0, (4.0, 1.6)))
MODELS = [
    ("routing_gptoss.npz", "GPT-OSS-20B", "gptoss", "#AF61F8", (0, (5.0, 1.4, 1.2, 1.4))),
    ("routing_granite.npz", "Granite-3b-a800m", "granite", "#9AA6DF", "-"),
    ("routing_olmoe_sharegpt.npz", "OLMoE-1B-7B", "olmoe", "#7E7F96", "-"),
    ("routing_qwen.npz", "Qwen3-30B-A3B", "qwen3", "#C17106", "-"),
    ("routing_qwen3next.npz", "Qwen3-Next-80B-A3B", "qwen3next", "#7A3F02", "-"),
]
FLOOR = 1e-4          # % share drawn for an expert that got no tokens (log axis)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-llama4", action="store_true",
                    help="the six-model version, fig16b, with Llama 4 Scout")
    ap.add_argument("--outdir", default="figs")
    a = ap.parse_args()
    models = [LLAMA4, *MODELS] if a.with_llama4 else MODELS
    stem = "fig16b_skew_six_models" if a.with_llama4 else "fig16c_skew_with_gptoss"
    with open(os.path.join(HERE, "data", "skew_across_models_v2.csv")) as fh:
        skew = {r["tag"]: float(r["excess_gini"]) for r in csv.DictReader(fh)}
    F.use()
    fig, ax = plt.subplots(figsize=(F.TEXT_W, 3.0))
    F.despine(ax)
    for fn, name, tag, col, ls in models:
        Z = np.load(os.path.join(HERE, "data", fn))
        m = json.loads(str(Z["meta"]))
        A = Z[next(k for k in Z.files if k.startswith("prefill_b"))].astype(float)
        C = A.reshape(-1, m["experts"])                 # every batch x layer
        sh = np.sort(C / C.sum(1, keepdims=True) * 100, 1)[:, ::-1]
        x = np.linspace(0, 100, m["experts"])
        ax.fill_between(x, np.maximum(np.percentile(sh, 10, axis=0), FLOOR),
                        np.maximum(np.percentile(sh, 90, axis=0), FLOOR), color=col, alpha=0.13,
                        lw=0, zorder=2)
        ax.plot(x, np.maximum(np.median(sh, 0), FLOOR), ls=ls, lw=1.5, color=col, zorder=5,
                label=f"{name}  ({m['experts']}e, top-{m['top_k']})  {skew[tag]:+.2f}")
    ax.set_yscale("log")
    ax.set_xlim(0, 100)
    ax.set_xlabel("experts, sorted by load (rank, %)")
    ax.set_ylabel("share of the layer's dispatched tokens (%)")
    ax.set_title("Routing load profile across MoE families: "
                 "24 independent ShareGPT prompt batches per model", loc="left", pad=5)
    ax.legend(loc="lower left", handlelength=2.6, labelspacing=0.3, borderaxespad=0.3,
              title="model  (experts, top-k)  excess Gini", title_fontsize=7, alignment="left")
    ax.text(0.99, 0.97, "band: 10th-90th percentile over 24 batches x all layers",
            transform=ax.transAxes, ha="right", va="top", fontsize=7, color=F.MUTED)
    print(*F.save(fig, stem, a.outdir))


if __name__ == "__main__":
    main()
