#!/usr/bin/env python3
"""Phase 0a analysis -- is the routing skew big enough to be worth a hardware argument?

THE METRIC THAT MATTERS HERE IS NOT GINI. For this project the question is how much SM
time a spatial partition would waste, so the quantity of interest is the **makespan
ratio**: if every expert were given an equal share of SMs, the layer finishes when the
BUSIEST expert finishes, while the average expert has been idle since much earlier.

    slack fraction = 1 - mean_load / max_load

That is the ceiling on what any spatial scheme -- DVFS on the idle partitions, or handing
their SMs back -- could possibly harvest. Gini and the active-expert count are reported
alongside because they say *why* the ratio is what it is.

Decode at small batch is a special case worth separating: with top-8 of 64 experts and B
tokens, only 8B expert slots are filled, so at B=1 at most 8 of 64 experts do ANY work.
Those 56 experts are not "slack", they are absent -- the imbalance among the experts that
actually run is the number that matters, and both are reported.
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
Z = np.load(os.path.join(HERE, "data", "routing_olmoe.npz"))
meta = json.loads(str(Z["meta"]))
E, K, L = meta["experts"], meta["top_k"], meta["layers"]


def gini(x):
    x = np.sort(x.astype(float), axis=-1)
    n = x.shape[-1]
    idx = np.arange(1, n + 1)
    s = x.sum(-1)
    out = np.where(s > 0, (2 * (x * idx).sum(-1) - (n + 1) * s) / (n * np.maximum(s, 1)), 0.0)
    return out


def stats(c):
    """c: (..., E) integer token counts. Returns per-sample metrics."""
    c = c.astype(float)
    mx = c.max(-1)
    mean = c.mean(-1)
    act = (c > 0).sum(-1)
    mean_act = np.where(act > 0, c.sum(-1) / np.maximum(act, 1), 0.0)
    return dict(max=mx, mean=mean, n_active=act,
                slack_all=1 - np.where(mx > 0, mean / np.maximum(mx, 1e-9), 1.0),
                slack_act=1 - np.where(mx > 0, mean_act / np.maximum(mx, 1e-9), 1.0),
                maxmean=np.where(mean > 0, mx / np.maximum(mean, 1e-9), 1.0),
                gini=gini(c))


def row(tag, c):
    s = stats(c)
    f = lambda k: np.median(s[k])
    p90 = lambda k: np.percentile(s[k], 90)
    return (f"  {tag:<18}{f('n_active'):>7.0f}/{E}{f('max'):>9.0f}{f('mean'):>9.1f}"
            f"{f('maxmean'):>9.2f}{f('gini'):>8.2f}"
            f"{f('slack_all')*100:>10.1f}%{f('slack_act')*100:>10.1f}%"
            f"{p90('slack_act')*100:>9.1f}%")


print(f"{meta['model']}\n{L} layers, {E} experts, top-{K}, "
      f"prefill {meta['prefill_len']} tok, {meta['decode_steps']} decode steps\n")
print(f"  {'regime':<18}{'active':>10}{'max':>9}{'mean':>9}{'max/mean':>9}{'gini':>8}"
      f"{'slack(all)':>11}{'slack(act)':>10}{'p90':>9}")
print("  " + "-" * 84)
for reg in ("prefill", "decode"):
    for k in sorted([k for k in Z.files if k.startswith(reg + "_b")],
                    key=lambda s: int(s.split("_b")[1])):
        B = int(k.split("_b")[1])
        c = Z[k]                      # prefill (L,E) | decode (steps,L,E)
        c = c.reshape(-1, E)
        print(row(f"{reg} B={B}", c))
    print()

# --- the layer axis: is skew a property of the model or of particular layers? ----------
print("  per-layer slack(active), decode B=64  (median over 128 steps)\n")
c = Z["decode_b64"]                                   # (steps, L, E)
s = stats(c)["slack_act"]                             # (steps, L)
med = np.median(s, axis=0) * 100
print("   " + " ".join(f"L{i:<2d}" for i in range(L)))
print("   " + " ".join(f"{v:3.0f}" for v in med))
print(f"\n   spread across layers: {med.min():.0f}% .. {med.max():.0f}%  "
      f"(median {np.median(med):.0f}%)")

# --- how fast does the load pattern change? -------------------------------------------
# This decides whether ANY adaptive scheme could track it. Correlation of the per-expert
# load vector between consecutive decode steps.
c = Z["decode_b64"].astype(float)                     # (steps, L, E)
cs = c - c.mean(-1, keepdims=True)
num = (cs[:-1] * cs[1:]).sum(-1)
den = np.sqrt((cs[:-1] ** 2).sum(-1) * (cs[1:] ** 2).sum(-1))
r = np.where(den > 0, num / np.maximum(den, 1e-9), 0.0)
print(f"\n  step-to-step correlation of the expert-load vector (decode B=64): "
      f"median r = {np.median(r):.3f}")
lagr = []
for lag in (1, 2, 4, 8, 16, 32, 64):
    a_, b_ = cs[:-lag], cs[lag:]
    nu = (a_ * b_).sum(-1); de = np.sqrt((a_ ** 2).sum(-1) * (b_ ** 2).sum(-1))
    lagr.append((lag, np.median(np.where(de > 0, nu / np.maximum(de, 1e-9), 0.0))))
print("   lag:  " + "  ".join(f"{l:>2d}" for l, _ in lagr))
print("   r  :  " + "  ".join(f"{v:.2f}" for _, v in lagr))

# --- the number that is actually actionable -------------------------------------------
# `slack` above assumes every expert gets an EQUAL share of SMs, which no sane runtime
# would do. Allocating SMs in proportion to load is free, in software, and removes most of
# that slack. So the honest question is what survives AFTER the free fix:
#
#   ideal makespan    = total_tokens / S          (perfect divisibility)
#   achievable        = max_i ceil-allocated n_i / s_i
#   residual slack    = 1 - ideal / achievable
#
# What survives is QUANTISATION: with S=108 SMs and 64 experts most experts rate a
# fractional SM, and you cannot hand out 1.37 SMs. That is the same wave-quantisation
# story `overlap_exp` measured, arriving from a different direction.
print("\n\n  AFTER the free fix: SMs allocated in proportion to load, S = 108\n")
S = 108
print(f"  {'regime':<18}{'equal-share':>13}{'proportional':>14}{'what DVFS':>12}")
print(f"  {'':<18}{'slack':>13}{'residual slack':>14}{'could sell':>12}")
print("  " + "-" * 57)


def prop_slack(c, S=108):
    c = c.astype(float)
    out = []
    for row_ in c:
        tot = row_.sum()
        if tot == 0:
            continue
        act = row_ > 0
        n = row_[act]
        # largest-remainder allocation, every active expert guaranteed at least one SM
        if act.sum() > S:                      # more active experts than SMs: batch them
            out.append(0.0); continue
        share = n / n.sum() * S
        s = np.floor(share); s[s < 1] = 1
        left = int(S - s.sum())
        if left > 0:
            order = np.argsort(-(share - np.floor(share)))
            s[order[:left]] += 1
        out.append(1 - (n.sum() / S) / (n / s).max())
    return np.array(out)


for reg in ("prefill", "decode"):
    for k in sorted([x for x in Z.files if x.startswith(reg + "_b")],
                    key=lambda s_: int(s_.split("_b")[1])):
        B = int(k.split("_b")[1])
        c = Z[k].reshape(-1, E)
        eq = np.median(stats(c)["slack_all"]) * 100
        pr = prop_slack(c)
        pr = np.median(pr) * 100 if len(pr) else float("nan")
        print(f"  {reg+' B='+str(B):<18}{eq:>12.1f}%{pr:>13.1f}%{eq-pr:>11.1f} pp")
    print()
