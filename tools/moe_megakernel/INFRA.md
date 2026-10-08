# How the routing traces are produced

Everything needed to reproduce `data/routing_*.npz`, `data/skew_across_models.csv` (five
families; `data/skew_across_models_v2.csv` with GPT-OSS-20B, added on 2026-10-01),
`figs/fig16_skew_across_models.png` and `figs/fig16c_skew_with_gptoss.png`.

## Hardware

One GCP node, **4 x NVIDIA A100-SXM4-40GB** (160 GB total), 108 SMs each, driver exposing
81 graphics clocks at 15 MHz steps (210-1410 MHz). Tracing needs no clock control -- only
the energy campaigns do.

## Software, and why it is arranged this way

The system python is **PEP 668 externally-managed** and `python3-venv` is not installed
(getting it needs `sudo apt`). So the extra packages live in a folder-local
`pip --target` directory that goes on `PYTHONPATH`:

```bash
source tools/moe_megakernel/env.sh    # puts ./pylibs on PYTHONPATH
```

Nothing outside `tools/moe_megakernel/` is modified, and the system `torch 2.9.1+cu129`
and `triton 3.5.1` are reused rather than duplicated.

| package | version | why pinned |
|---|---|---|
| transformers | **4.57.1** | 5.x requires torch >= 2.10 |
| tokenizers | **0.22.2** | 4.57.1 wants >=0.22.0,<=0.23.0 |
| huggingface_hub | **0.35.3** | 4.57.1 wants >=0.34.0,<1.0 |
| datasets | **3.6.0** | 4.x wants hub >= 1.0, which transformers forbids |
| accelerate | 1.10.1 | `device_map="auto"` sharding |

**Two traps, both hit and both documented in `env.sh`:**

1. Letting pip resolve transformers' dependencies freely pulled in **torch 2.14**, which
   shadowed the system torch and broke torchvision's operator registry. Everything large
   is installed `--no-deps` with explicit pins.
2. `pip install --target` **leaves the old `.dist-info` behind** when upgrading in place,
   so `importlib.metadata` keeps reporting the stale version and transformers' version
   check fails on a package that is actually correct. Delete the stale directory by hand.

## Models

Sharded across the four GPUs with `device_map="auto"`, bf16.

| tag | model | experts | top-k | layers | params |
|---|---|---|---|---|---|
| `granite` | ibm-granite/granite-3.0-3b-a800m-instruct | 40 | 8 | 32 | 3.4B |
| `olmoe` | allenai/OLMoE-1B-7B-0924 | 64 | 8 | 16 | 6.9B |
| `olmoe_it` | allenai/OLMoE-1B-7B-0924-Instruct | 64 | 8 | 16 | 6.9B |
| `qwen3` | Qwen/Qwen3-30B-A3B | 128 | 8 | 48 | 30.5B |
| `qwen3next` | Qwen/Qwen3-Next-80B-A3B-Instruct | **512** | 10 | 48 | **80B** |
| `gptoss` | openai/gpt-oss-20b | 32 | 4 | 24 | 20.9B |

Mixtral-8x7B (8 experts) and Qwen1.5-MoE-A2.7B (60) were traced and then dropped from the
study; their `.npz` files are kept in `data/retired/`.

### Qwen3-Next-80B needs two accommodations

**CPU offload.** 80B in bf16 is **160 GB**, and the four GPUs total exactly 160 GB -- the
weights alone fill them, leaving nothing for CUDA context or activations. `device_map`
spills the remainder to CPU RAM (334 GB on this box). Quantising instead was rejected: the
router is a Linear, and perturbing its logits could change the top-k decisions, which are
the thing being measured.

**Batch 32 instead of 64.** Without `flash-linear-attention` installed, Qwen3-Next falls
back to a torch gated-delta-rule that materialises **float32** copies of q/k/v, which OOMs
at B=64. Run with `--gpu-mem 32GiB` to reserve headroom. It therefore sees 768
conversations against the others' 1536, and only **78 tokens per expert** against
508-1708 -- which is exactly why the null correction matters here.

204 GB of weights in `~/.cache/huggingface`. `fetch_models.sh` downloads them
sequentially -- parallel downloads saturate the link and the later ones time out.

### GPT-OSS-20B (added 2026-10-01)

Same prompts, same 24 batches of 64 conversations, same 512-token cap, same analysis.
What had to change is listed here, with what each change costs. A trace of OLMoE with the
changed tracer is identical to the stored one, count for count, prefill and decode.

**The router is not a bias-free `nn.Linear`.** It is a module of its own with a weight
and a bias, returning `(scores, indices)`. It is found by the shape of its weight,
(experts, hidden), and the indices it returns are what is counted -- the block's own
top-4. The exactly-one-candidate guard is unchanged.

**A 64-row batch does not fit in one forward.** transformers computes EVERY expert on
EVERY position (the input is repeated once per expert), and at 64 x 512 positions the
temporaries of one layer run a 40 GB card out of memory. `--rows 16` forwards the same
padded batch 16 rows at a time and adds the counts. A row is the tensor it would have been in the full batch, so
the routing is the same up to floating-point ties: on OLMoE, where both ways fit, 0.12%
of the prefill dispatches move.

**It is released in MXFP4.** The MoE weights in the checkpoint are 4-bit; without the
`kernels` package transformers dequantises them to bf16 at load. That loses nothing: an
MXFP4 value is one of sixteen two-bit mantissas times a power of two, which bf16 holds
exactly. This is the model as OpenAI ships it, not a quantisation chosen here.

**Its chat template is long.** It prepends a 67-token system preamble to every prompt:
39% of all tokens counted in a batch, against 0-9% for the other families, and identical
in all 64 conversations. The trace with the template is the string a server receives and
is the one in the table and the figure; `routing_gptoss_raw.npz` is the control with the
raw turn (`--no-template`). The preamble accounts for 0.03 of the excess Gini: +0.464
with it, +0.435 without.

### Llama 4 Scout: traced, not in the comparison

`data/routing_llama4.npz` exists and is left out of the table and the figures. The model
is 107.8B parameters (16 experts, top-1, 48 layers; about 17B of them active per token),
217 GB in bf16, and its trace is not like for like with the others in four ways:

- **Prefill only.** 26 of its 48 layers fit on the four GPUs; the other 22 stay in RAM
  and accelerate moves each to GPU 0 for its forward and back. A 64-row prefill takes
  35 s that way. A decode step would move the same 22 layers again for one token per
  row, so none were run.
- **A modified MoE forward** (`--sparse-moe`): each expert is computed only on the
  positions routed to it, where transformers computes all 16 on every position. The 15
  a position was not routed to receive an input multiplied by a score of exactly zero
  and return exactly zero, so the arithmetic on the rows that matter is the same, but it
  is not bit-identical: on 4 rows, 0.77% of the routing decisions differ from the dense
  path. That is the model's own noise floor with top-1 routing through 48 bf16 layers --
  the dense path forwarded 2 rows at a time instead of 4 differs from itself by 0.74%,
  and twice the same way by 0 -- but it is a different code path from every other model
  here.
- **Top-1 beside a shared expert.** Every token goes through one routed expert of 16
  and through a shared expert. Only the routed one can be counted, so its "share of
  dispatched tokens" is a share of half the MoE work.
- **Weights from a mirror.** `meta-llama/Llama-4-Scout-17B-16E-Instruct` is gated and
  this box has no token. The 50 weight files and `tokenizer.json` of
  `unsloth/Llama-4-Scout-17B-16E-Instruct` have the same size and sha256 as Meta's (the
  hashes are public metadata of the gated repository); `config.json`,
  `tokenizer_config.json` and `chat_template.jinja` are the mirror's.

For the record, its excess Gini is +0.389 ± 0.017 and its busiest expert takes 18.0% of
a layer's tokens where a uniform router would give 6.25%
(`data/skew_across_models_v2.csv`, role "not compared").
`python3 make_skew_gptoss_fig.py --with-llama4` redraws the figure with it
(`figs/fig16b_skew_six_models.png`).

## Inference framework -- and why it is the reference implementation, not a server

**HuggingFace `transformers` 4.57.1 on PyTorch 2.9.1+cu129.** Deliberately NOT vLLM,
TensorRT-LLM, SGLang or FlashInfer -- none of them is installed here.

That choice is right for THIS measurement and wrong for any other:

* **Routing is exact.** The router is an `nn.Linear` and top-k is recomputed the way the
  block itself does it, so the dispatch decisions are the model's own. A serving framework
  would give identical routing.
* **Timing and energy are NOT representative.** HF's MoE forward loops over every expert
  in Python. Measured on OLMoE (`RESULTS_*`): one decode step at batch 1 launches **8277
  CUDA kernels**, the GPU is **idle 92% of the wall clock**, and it burns **21.6 J per
  token**. Nothing about latency or power from this path may be quoted; that is what the
  fused-kernel work (B1) exists to fix.

### B1 closes that gap: the kernel is vLLM's, without installing vLLM

`vendor/fused_moe_triton.py` carries vLLM v0.11.0's Triton `fused_moe_kernel` **verbatim**
(see `vendor/README.md` for line numbers and the Apache-2.0 notice). The align step and the
driver are rewritten here, because vLLM's align is a compiled CUDA op in `csrc/`.

vLLM itself is *not* installed, and that is deliberate rather than lazy: vLLM ships its own
compiled `_C` extension pinned to a particular torch build, so `pip install vllm` into
`pylibs/` would pull a second torch onto `PYTHONPATH` -- the exact failure this folder
already hit once and documented in `env.sh`. The Triton kernel has no vLLM imports, so it
lifts out cleanly and runs on the system torch 2.9.1 / Triton 3.5.1.

What this means for which numbers are quotable:

| measurement | path | quotable as |
|---|---|---|
| expert routing counts | HF `transformers` | the model's own dispatch decisions |
| MoE layer latency and energy | `vendor/fused_moe_triton.py` | a real serving kernel's cost |
| end-to-end tokens/s or J/token | *neither* | **nothing** -- there is still no full model on the fused path |

One caveat that must travel with every B1 number: vLLM ships pre-tuned Triton configs per
`(experts, intermediate size, GPU)`, and **has none for any A100 at Qwen3-30B's
`E=128, N=768`** (3 A100-SXM4-40GB files exist out of 230, all 8/16-expert Mixtral-era
shapes). Left alone it falls back to a six-line heuristic that costs 37% latency. The
config used here was tuned on this box by `b1_bench.py --tune`, the same procedure as
vLLM's own `benchmark_moe.py --tune`, and lives in `data/b1_config_qwen.json`.

## Prompts and how much of the dataset is actually seen

**ShareGPT V3** — https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered
file `ShareGPT_V3_unfiltered_cleaned_split.json`, 94145 conversations, 673 MB. Fetched
with `huggingface_hub.hf_hub_download(..., repo_type="dataset")`; the resolved path is
cached in `data/_sharegpt_path.txt`. The **first
human turn** of each conversation, rendered through **that model's own chat template** --
the string a server actually receives. Base models without a template fall back to the raw
turn.

Chosen over wikitext-103 because it is what vLLM and TensorRT-LLM benchmark on. The
control run (`routing_olmoe.npz` on wikitext vs `routing_olmoe_sharegpt.npz` on ShareGPT)
shows the dataset is **not** what drives the skew.

**The dataset is sampled, not exhausted.** Each model sees **24 independent batches of 64
conversations = 1536 conversations, 1.6% of ShareGPT**, each batch drawn with its own seed
after a shuffle. That sampling is the ONLY source of uncertainty in these numbers -- the
16-48 layer-samples inside one batch all see the same prompts -- so the error bar is the
spread ACROSS batches, not across layers.

A first version of this study used a single batch of 64 (0.068% of the dataset) and quoted
no error bar at all. It was wrong by about one standard deviation: OLMoE's excess Gini read
+0.22 there against +0.251 +- 0.035 over 24 batches, and an apparent base-vs-Instruct gap
of +0.04 turned out to be +0.016.

## Three things that must be right or the answer inverts

**1. The router is found structurally, not by name.** Every family calls it something
different (`mlp.gate`, `block_sparse_moe.gate`, `block_sparse_moe.router.layer`). The
tracer looks inside each decoder layer for the bias-free `nn.Linear` whose `out_features`
equals the expert count, excluding anything whose module path contains `expert`, `attn` or
`attention`, **and raises if that does not leave exactly one.**

That guard earned its keep twice on Qwen3-Next, where 512 collides with two other widths:
`moe_intermediate_size` is also 512, so all 512 experts' `gate_proj`/`up_proj` matched
(1027 candidates); and 4 KV heads x 128 head_dim is also 512, so `self_attn.k_proj` and
`v_proj` matched. Picking the first candidate instead of raising would have hooked an
expert's projection and produced routing counts that look entirely plausible and are
entirely wrong.

**2. Padding is masked.** ShareGPT prompts vary enormously in length, so a rectangular
batch is **72-78% padding** (measured: 22-28% real tokens at B=64). The router runs on pad
positions too and they are all identical, so counting them drags the histogram towards
uniform and manufactures a "no skew" result. The hook drops padded positions using the
live attention mask, and the total is asserted against `real tokens x top-k x layers`.

**3. Skew is compared against a null, not against zero.** With few tokens per expert even
a perfectly uniform router shows a large Gini from multinomial noise:

| tokens/expert | Gini of a *uniform* router |
|---|---|
| 1 | 0.51 |
| 10 | 0.18 |
| 100 | 0.056 |
| 1000 | 0.018 |

So each measurement is quoted against the median Gini of a uniform multinomial draw with
the same token count, and the **excess** is the part the router caused.

## Reproduce

```bash
cd tools/moe_megakernel
./fetch_models.sh                 # ~204 GB, sequential
./run_all_models.sh               # B=64, 512-token cap, 8 decode steps, per model
python3 analyse_skew.py           # -> data/skew_across_models_v2.csv
python3 make_skew_fig.py          # -> figs/fig16_skew_across_models.png
python3 make_skew_gptoss_fig.py   # -> figs/fig16c_skew_with_gptoss.png
```

Per-model tracing runs in 3-8 s once the weights are cached; the whole set is ~2 minutes
of GPU time. `run_all_models.sh` skips a model whose `.npz` already exists.

## Result

24 batches x 64 conversations per model; +- is the standard deviation across batches.

| model | experts | top-k | tokens/expert | **excess Gini** | max/mean |
|---|---|---|---|---|---|
| Mixtral-8x7B | 8 | 2 | 2360 | **+0.054 ± 0.007** | 1.18 ± 0.02 |
| Granite-3b-a800m | 40 | 8 | 1708 | **+0.329 ± 0.009** | 3.08 ± 0.06 |
| Qwen1.5-MoE-A2.7B | 60 | 4 | 571 | **+0.116 ± 0.018** | 1.88 ± 0.15 |
| OLMoE-1B-7B | 64 | 8 | 1175 | **+0.251 ± 0.035** | 3.00 ± 0.43 |
| OLMoE-1B-7B-Instruct | 64 | 8 | 1054 | **+0.266 ± 0.039** | 2.86 ± 0.44 |
| Qwen3-30B-A3B | 128 | 8 | 508 | **+0.427 ± 0.028** | 4.52 ± 0.42 |

Correlation between expert count and excess Gini is **+0.90** across these five, and the
512-expert model is the most skewed of all. But it is still **not monotonic** -- Granite's
40 experts (+0.329 ± 0.009) are decisively more skewed than OLMoE's 64 (+0.250 ± 0.035),
a gap far larger than either error bar. Expert count is also confounded with top-k,
family, depth and training recipe, so this is a correlation across five models, not a
demonstration that expert count causes skew.

The one clean controlled pair is **OLMoE base vs Instruct**: identical architecture,
expert count, and the *same 24 prompt batches* (same seeds), so the comparison can be
**paired** rather than compared as two means.

- unpaired: +0.251 vs +0.266, a gap of +0.016 against a standard deviation of 0.035 --
  **not separable**
- paired: **+0.0158 ± 0.0163 per batch, positive in 20 of 24 batches, t = 4.66 on 23 df**

So instruction tuning does raise the skew, but by about **+0.016**, not the +0.04 the
single-batch version suggested. The effect is only visible because the pairing removes the
prompt-draw variance, which is an order of magnitude larger than the effect itself.


### With GPT-OSS-20B (2026-10-01)

`figs/fig16c_skew_with_gptoss.png`, `data/skew_across_models_v2.csv`. The Qwen3-Next row
was in the data and missing from the table above.

| model | experts | top-k | tokens/expert | **excess Gini** | max/mean |
|---|---|---|---|---|---|
| Qwen3-Next-80B-A3B (batch 32) | 512 | 10 | 78 | **+0.568 ± 0.036** | 10.96 ± 2.42 |
| GPT-OSS-20B | 32 | 4 | 1406 | **+0.464 ± 0.026** | 4.33 ± 0.20 |
| GPT-OSS-20B, raw prompt (control) | 32 | 4 | 1146 | +0.435 ± 0.037 | 3.77 ± 0.38 |

**GPT-OSS breaks the pattern.** It has 32 experts, fewer than every family above, and is
as skewed as Qwen3-30B with 128 (+0.464 against +0.427) and more skewed than Granite with
40 and OLMoE with 64 (+0.329 and +0.250). Its busiest expert takes 13.5% of a layer's
tokens where a uniform router would give 3.1%, and its skew grows with depth, from a Gini
of 0.21 at the first layer to 0.71 at the last. The correlation between expert count and
excess Gini, +0.90 over the five families above, is +0.73 with GPT-OSS (+0.61 against
log2 of the expert count). Expert count alone does not predict the skew.

The checked-in `skew_across_models.csv` says Qwen3-Next saw 1536 conversations; it saw
768 (24 batches of 32). The v2 file has batches x batch size and a `role` column;
nothing else in its first five rows differs.
