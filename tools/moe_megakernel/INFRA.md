# How the routing traces are produced

Everything needed to reproduce `data/routing_*.npz`, `data/skew_across_models.csv` and
`figs/fig16_skew_across_models.png`.

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
python3 analyse_skew.py           # -> data/skew_across_models.csv
python3 make_skew_fig.py          # -> figs/fig16_skew_across_models.png
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
