#!/usr/bin/env python3
"""Routing skew across MoE families, on identical ShareGPT prompts.

Gini alone is unreadable because a perfectly uniform router still scores high when there
are few tokens per expert (0.51 at 1 token/expert, 0.02 at 1000). Every measurement is
therefore quoted against a NULL -- the same token count thrown uniformly at random over
the same experts -- and the EXCESS is the part the router is responsible for.

All models see the same prompts (first human turn of each ShareGPT conversation, through
each model's own chat template), the same batch (64) and the same 512-token cap. Padded
positions are dropped by the tracer before counting.

  python3 analyse_skew.py
"""
import csv, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(0)
MODELS = [
    ("granite",   "Granite-3.0-3b-a800m", "routing_granite.npz"),
    ("olmoe",     "OLMoE-1B-7B",         "routing_olmoe_sharegpt.npz"),
    ("olmoe_it",  "OLMoE-1B-7B-Instruct", "routing_olmoe_it.npz"),
    ("qwen3",     "Qwen3-30B-A3B",       "routing_qwen.npz"),
    ("qwen3next", "Qwen3-Next-80B-A3B",  "routing_qwen3next.npz"),
]


def gini(x):
    x = np.sort(np.asarray(x, float), -1)
    n = x.shape[-1]
    i = np.arange(1, n + 1)
    s = x.sum(-1)
    return np.where(s > 0, (2 * (x * i).sum(-1) - (n + 1) * s) / (n * np.maximum(s, 1)), 0)


def load(fn, key=None):
    """Returns (repeats, layers, E). Each repeat is an INDEPENDENT draw of 64 ShareGPT
    conversations, which is the only source of uncertainty in this measurement -- the
    layer-samples inside one repeat all see the same prompts."""
    Z = np.load(os.path.join(HERE, "data", fn))
    m = json.loads(str(Z["meta"]))
    E = m["experts"]
    if key is None:
        # batch size is not uniform across models: Qwen3-Next-80B OOMs at 64 because its
        # gated-delta-rule fallback materialises float32 q/k/v, so it ran at 32.
        key = next(k for k in Z.files if k.startswith("prefill_b"))
    m["batch"] = int(key.split("_b")[1])
    A = Z[key].astype(float)
    if A.ndim == 2:                      # a single-batch trace from before --repeats
        A = A[None]
    return A.reshape(A.shape[0], -1, E), m


def main():
    rows = []
    print("Routing skew on ShareGPT, prefill, independent prompt batches\n")
    print(f"  {'model':<24}{'experts':>8}{'top-k':>7}{'B':>4}{'reps':>6}{'convs':>7}"
          f"{'tok/exp':>9}{'excess gini':>15}{'max/mean':>13}")
    for tag, name, fn in MODELS:
        p = os.path.join(HERE, "data", fn)
        if not os.path.exists(p):
            print(f"  {name:<24} MISSING {fn}")
            continue
        A, m = load(fn)
        E = m["experts"]
        R = A.shape[0]
        # one excess-Gini per independent prompt batch, so the spread across batches is
        # the error bar the single-batch version did not have
        ex, mms, tpe = [], [], []
        for r in range(R):
            C = A[r]
            n = int(np.median(C.sum(1)))
            null = rng.multinomial(n, [1 / E] * E, size=200).astype(float)
            ex.append(float(np.median(gini(C)) - np.median(gini(null))))
            mms.append(float(np.median(C.max(1) / np.maximum(C.mean(1), 1e-9))))
            tpe.append(n / E)
        ex, mms = np.array(ex), np.array(mms)
        B = m["batch"]
        print(f"  {name:<24}{E:>8}{m['top_k']:>7}{B:>4}{R:>6}{R*B:>7}{np.mean(tpe):>9.0f}"
              f"{f'{ex.mean():+.3f} ± {ex.std():.3f}':>15}"
              f"{f'{mms.mean():.2f} ± {mms.std():.2f}':>13}")
        rows.append(dict(tag=tag, model=m["model"], experts=E, top_k=m["top_k"],
                         layers=m["layers"], repeats=R, conversations=R * 64,
                         tokens_per_expert=round(float(np.mean(tpe)), 1),
                         excess_gini=round(float(ex.mean()), 4),
                         excess_gini_sd=round(float(ex.std()), 4),
                         excess_gini_min=round(float(ex.min()), 4),
                         excess_gini_max=round(float(ex.max()), 4),
                         max_over_mean=round(float(mms.mean()), 3),
                         dataset=m.get("dataset", "")))
    p = os.path.join(HERE, "data", "skew_across_models.csv")
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n  wrote {p}")
    ex = [r["excess_gini"] for r in rows]
    ne = [r["experts"] for r in rows]
    if len(ex) > 2:
        print(f"  correlation(expert count, excess gini) = "
              f"{np.corrcoef(ne, ex)[0,1]:+.2f}")


if __name__ == "__main__":
    main()
