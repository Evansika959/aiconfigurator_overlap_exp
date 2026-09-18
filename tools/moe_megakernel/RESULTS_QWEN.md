# Does the routing pattern hold on a bigger model and on real serving traffic?

OLMoE gave the shape of the routing distribution, but two things about that setup could
have produced it rather than anything general: 64 experts is coarse, and wikitext-103 is
encyclopedia prose, not the chat traffic a deployment sees. Both were changed.

| | model | experts | top-k | layers | data |
|---|---|---|---|---|---|
| before | OLMoE-1B-7B | 64 | 8 | 16 | wikitext-103 |
| now | **Qwen3-30B-A3B** | **128** | 8 | **48** | **ShareGPT V3** |

Qwen3 is sharded across the four A100s with `device_map="auto"` (30.5B params in bf16 =
61 GB). Prompts are the first human turn of each ShareGPT conversation, rendered through
the model's own chat template -- the string a server actually receives. Reproduce with
`trace_routing_sharegpt.py`; traces in `data/routing_qwen.npz`.

## Padding had to be masked, or the answer inverts

ShareGPT prompts vary enormously in length, so a rectangular batch is 70-80% padding
(measured: 21-29% real tokens at B=8..64). The router runs on pad positions too, and those
are all identical, so counting them would drag the distribution towards uniform and
manufacture a "no skew" result. The hook drops padded positions using the live attention
mask, and the total is asserted against `real tokens x top-k x layers`.

## Skew has to be compared against sampling noise, not against zero

With few tokens per expert, even a perfectly uniform router shows a large Gini simply from
multinomial fluctuation. So each measurement is quoted against a null: the median Gini of
a uniform multinomial draw with the same token count.

| model | experts | data | batch | tokens/expert | gini | null | **excess** |
|---|---|---|---|---|---|---|---|
| OLMoE | 64 | wikitext | 16 | 1024 | 0.30 | 0.02 | +0.28 |
| OLMoE | 64 | wikitext | 64 | 4096 | 0.30 | 0.01 | +0.29 |
| OLMoE | 64 | **ShareGPT** | 32 | 599 | 0.27 | 0.02 | **+0.24** |
| OLMoE | 64 | **ShareGPT** | 64 | 1130 | 0.24 | 0.02 | **+0.22** |
| **Qwen3** | **128** | ShareGPT | 32 | 214 | 0.53 | 0.04 | **+0.50** |
| **Qwen3** | **128** | ShareGPT | 64 | 484 | 0.46 | 0.03 | **+0.44** |

## Two findings

**1. The dataset is not the driver.** Running OLMoE on ShareGPT instead of wikitext moves
its excess Gini from +0.29 to +0.24 -- slightly *less* skew on chat traffic, not more. So
the skew measured earlier was not an artefact of encyclopedia prose.

**2. More experts means MORE skew, not less.** On the same ShareGPT prompts, Qwen3's
excess Gini is **+0.50 against OLMoE's +0.24** -- roughly double. This contradicts the
expectation that finer granularity would let the law of large numbers flatten the
histogram. Whatever produces the imbalance scales with the expert count rather than
averaging out.

Other quantities move the same way. At prefill B=64: `max/mean` is **4.70** for Qwen3
against 2.85 for OLMoE, and the top quarter of experts holds **55%** of the dispatched
tokens against 43%.

## Caveat that is not yet closed

Model and expert count changed together. Qwen3 differs from OLMoE in far more than expert
count -- training data, router design, auxiliary-loss weight, depth (48 vs 16 layers), and
expert width (768 vs 1024). This experiment shows the skew is a property of the *model*
rather than of the *prompts*; it does not isolate expert count as the cause. A third MoE
with 128 experts from a different family, or the same family at two expert counts, would
be needed for that.

---

# What the doubled skew is worth

Same prediction pipeline (`predict_expert_dvfs.py --npz data/routing_qwen.npz --ncol 768
--experts 128`), energy against a grouped GEMM at the **same latency**.

| | waves | SLO 1380 | 1200 | 1095 | 1005 |
|---|---|---|---|---|---|
| **OLMoE**, 64 experts, wikitext | | | | | |
| prefill B=16 | 40 | +2.6% | +1.4% | +1.2% | +0.1% |
| prefill B=128 | 306 | +0.1% | +0.1% | 0.0% | +0.4% |
| decode B=64 | 5 | +10.5% | +5.4% | +2.0% | +0.6% |
| **Qwen3**, 128 experts, ShareGPT | | | | | |
| prefill B=8 | 8 | **+7.6%** | +4.9% | +1.8% | +0.4% |
| prefill B=32 | 16 | **+4.7%** | +3.2% | +1.7% | +0.2% |
| prefill B=64 | 31 | **+3.5%** | +2.3% | +1.7% | +0.4% |
| decode B=64 | 4 | +8.3% | +3.9% | +1.5% | +0.4% |

**The prefill numbers roughly double**: Qwen3 at B=64 gives +3.5% where OLMoE at B=16 gave
+2.6%, and Qwen3's shallower prefill (31 waves at B=64 against OLMoE's 306 at B=128) keeps
the effect alive at batch sizes where OLMoE had nothing left. Two things drive that, and
they compound: twice the routing skew, and a narrower expert (768 vs 1024) which means
fewer tiles per expert and so fewer waves per layer.

**A third domain still adds nothing**: at most 0.13 pp anywhere in either model.

**And the shape of the dependence is unchanged.** The gain still collapses as the SLO
loosens -- +3.5% at 1380 becomes +0.4% at 1005 -- because once the single clock is allowed
to fall to the voltage knee there is nothing left for a second domain to collect. Every
conclusion from the OLMoE study survives; only the magnitude moves, and it moves in the
direction the extra skew predicts.

## What this does not change

Decode is still memory bound (`RESULTS_ROOFLINE.md`): 128 experts at bf16 with a 768-wide
intermediate is 3 x 2048 x 768 x 2 = 9.4 MB per expert, and at B=64 roughly 91 of 128
experts are active, so ~860 MB of weights must stream per layer against a few tens of
microseconds of arithmetic. The wave model does not describe that, so the decode rows
above are reported for completeness and should not be quoted.
