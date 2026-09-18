# Phase 0a — how imbalanced is OLMoE's expert routing?

`allenai/OLMoE-1B-7B-0924`, 16 layers, 64 experts, top-8, fp16 on one A100.
Real text (wikitext-103 test), prefill 512 tokens, 128 decode steps, batch 1-128.
Router hooked at the `gate` linear, top-k recomputed exactly as the block does it.
Raw counts in `data/routing_olmoe.npz`; full output in `data/phase0a_results.txt`.

**HF loops over all 64 experts in Python, so no timing here is a baseline.** The router
is exact; the dispatch is not representative. Energy baselines wait for Phase 1.

## The skew is real and stable

| regime | active | max/mean | gini | slack under equal SM shares |
|---|---|---|---|---|
| prefill B=1 | 64/64 | 3.58 | 0.38 | 72.0% |
| prefill B=128 | 64/64 | 2.81 | 0.30 | 64.4% |
| decode B=1 | **8/64** | 8.00 | 0.88 | 87.5% |
| decode B=16 | 52/64 | 3.50 | 0.44 | 71.4% |
| decode B=128 | 63/64 | 3.00 | 0.33 | 66.7% |

`max/mean` sits at **2.8-3.6** almost everywhere. Consistent with the low end of the
2x-12x range the serving literature reports; those numbers are per-GPU under expert
parallelism, which aggregates several experts and is not the same quantity.

Per layer (decode B=64) the slack runs **57% to 75%**, median 67% — skew is a property of
the model, not of one pathological layer.

## But the headline number is measured against a strawman

"Slack under equal SM shares" assumes every expert gets 108/64 SMs regardless of load. No
runtime would do that. Allocating SMs **in proportion to load** is free, in software, and
it is the honest baseline. What survives:

| regime | equal shares | proportional | free fix is worth |
|---|---|---|---|
| prefill B=128 | 64.4% | **41.4%** | 23 pp |
| decode B=16 | 71.4% | **40.7%** | 31 pp |
| decode B=64 | 69.2% | **40.7%** | 28 pp |
| decode B=1 | 87.5% | **3.7%** | 84 pp |

**~41% residual slack survives the free fix**, and it is a *granularity* effect: with 108
SMs and 64 active experts most experts rate a fractional SM and you cannot hand out 1.37
of one. This is `overlap_exp`'s wave-quantisation story arriving from a new direction.

## The tension, now quantified

| design | slack | cost |
|---|---|---|
| equal partition | 65-72% | none, but nobody does this |
| **proportional partition** | **~41%** | free, software |
| work-conserving megakernel | ~0 (tail only) | free — **and nothing left for DVFS to sell** |

This is the project's central trade, with numbers: a megakernel removes the slack by
construction. Pinning partitions creates 41% of harvestable slack but lengthens the
makespan. **Is it better to run work-conserving at one low clock, or pinned at
heterogeneous clocks?** Phase 2/3 answers that; it could not even be posed before Phase 0a.

## An inconvenient finding for the "spatial only" premise

The expert-load vector is **strongly autocorrelated across decode steps**: median
r = 0.83 at lag 1, still **0.74 at lag 64**. The load pattern is not the fast-changing
noise the plan assumed — it persists for tens of steps.

If a decode step is ~1 ms, 64 steps is ~64 ms, which is **longer than the 37.7-38.5 ms
DVFS transition** `overlap_exp` measured. That would mean a temporal scheme is not dead
after all, contradicting section 1 of `PLAN.md`.

Not claimed yet: the step time here comes from HF's Python expert loop and is meaningless.
**Phase 1 must measure real decode step latency before this is either acted on or
dismissed.** Recorded now so it is not quietly forgotten.

## Decode at batch 1 is a different problem

Only **8 of 64** experts run, and each gets exactly one token, so the load among active
experts is perfectly uniform — `slack(active) = 0`. There is no load imbalance to exploit
at B=1; the imbalance is in *which* experts, not *how much*. That regime is bound by
weight loading, not compute, and DVFS on idle partitions has nothing to sell there.

---

# B0 — the reference path, and how bad a baseline it is

`baseline_b0.py` -> `data/baseline_b0.csv`. Clocks unlocked (the governor's own choice is
the only honest baseline, per `overlap_exp`). Sustained 3 s loops, NVML at 20 ms.

B0 is HuggingFace's `OlmoeSparseMoeBlock`: a **Python loop over all 64 experts**, each a
pair of `nn.Linear` calls into cuBLAS. Its router is exact — Phase 0a's traces stand — but
its dispatch is not representative of anything.

## It is launch-bound, not compute-bound

| regime | B | wall ms | device ms | **GPU busy** | math % of device | **math % of wall** | clock |
|---|---|---|---|---|---|---|---|
| decode | 1 | 322.6 | 25.8 | **8.0%** | 18.8% | **1.5%** | 1095 |
| decode | 4 | 360.2 | 44.6 | 12.4% | 30.2% | 3.7% | 1095 |
| decode | 16 | 419.8 | 61.5 | 14.7% | 39.6% | 5.8% | 1095 |
| decode | 64 | 472.1 | 86.5 | 18.3% | 37.7% | 6.9% | 1095 |
| prefill | 1 | 506.7 | 103.4 | 20.4% | 38.9% | 7.9% | 1095 |
| prefill | 64 | 1069.3 | 884.2 | 82.7% | 39.2% | 32.4% | 1410 |

**In decode at batch 1 the GPU is idle 92% of the wall clock, and only 1.5% of the wall
clock is arithmetic.** The `torch.where` inside the expert loop does a device-to-host
memcpy per (layer, expert) — 1024 synchronising copies per step — so the GPU spends the
step waiting for Python.

## Three reasons B0 cannot be a baseline, now with numbers

1. **Latency**: 322 ms per decode step at batch 1. A fused implementation is one to two
   orders of magnitude faster. Any speedup quoted against this is meaningless.
2. **Energy**: 21.6 **J** per token in decode at B=1. Off by roughly three orders of
   magnitude from what this model size should cost.
3. **It never reaches the regime we study.** Because utilisation is so low, the governor
   settles at **1095 MHz** in every decode case and 74 W — barely above the ~60 W static
   floor. B0 does not exercise the voltage knee, does not throttle, and does not spend
   meaningful dynamic power. There is no DVFS story to tell about it at all.

Only `prefill B=64` gets the GPU genuinely loaded (82.7% busy, 1410 MHz, 291 W). That one
row is the only part of B0 that resembles a real workload.

## What B0 delivered

- `data/routing_olmoe.npz` — the routing traces, valid and used.
- `data/baseline_b0.csv` — the strawman, sized. The instruction "do not compare against
  B0" is now backed by "B0 is 92% idle and burns 21.6 J per token", which is the version
  people remember.
- The decode step time question is **still open**: 322 ms here says nothing about a real
  system, so whether the load autocorrelation (r = 0.74 at lag 64) outlives the 38 ms DVFS
  transition remains B1's to settle.
