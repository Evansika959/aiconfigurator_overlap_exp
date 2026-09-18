# B1 — the fused grouped-GEMM baseline

**Status: the kernel runs and is verified correct. Energy sweep in progress.**

B1 is the rung of the ladder in `PLAN.md` §8 that makes every later energy claim
meaningful. B0 (the HuggingFace Python loop over experts) has an exact router but
is 92% idle and issues 8277 kernels per step, so no latency or energy number may
be quoted against it. B1 is one launch per GEMM, tokens sorted by expert, and is
what a real serving stack actually runs.

---

## 1. What the kernel is, and what is ours

`vendor/fused_moe_triton.py`. See `vendor/README.md` for line-level provenance.

| part | origin |
|---|---|
| `fused_moe_kernel`, `write_zeros_to_output` | **verbatim** from vLLM v0.11.0 |
| `moe_align_block_size` | ours — vLLM's is a CUDA op in `csrc/`, reimplemented in PyTorch |
| `_invoke`, `fused_experts`, `PreparedMoE` | ours — vLLM's driver minus quantisation, chunking, expert parallelism, bias |

vLLM itself is **not** installed: it pins its own compiled `_C` extension against a
specific torch and would drag a second torch into `pylibs/` — the exact failure
documented in `env.sh`. The Triton kernel has no vLLM imports, so vendoring it is
clean.

Reduced scope: bf16/fp16 only (no fp8/int8/int4), no expert-parallel `expert_map`,
no chunking, no bias. The quantisation `tl.constexpr` flags are kept so the copied
kernel text is unmodified; the driver always passes `False`.

## 2. Correctness

`test_fused_moe.py` checks against a naive per-expert loop in fp32.

| M | E | K | I | topk | max rel. err |
|---|---|---|---|---|---|
| 8 | 128 | 2048 | 768 | 8 | 5.5e-3 |
| 64 | 128 | 2048 | 768 | 8 | 5.7e-3 |
| 512 | 128 | 2048 | 768 | 8 | 5.2e-3 |
| 4096 | 128 | 2048 | 768 | 8 | 5.4e-3 |
| 1024 | 64 | 2048 | 1024 | 8 | 5.5e-3 |
| 333 | 40 | 1536 | 512 | 8 | 4.6e-3 |

~5e-3 is bf16 accumulation order, not a bug: the reference accumulates in fp32 in a
different order. M=8 exercises the `M <= E` small-batch config path and the
`EM = min(...)` block-skipping branch; M=333 exercises a non-multiple-of-block M.

`PreparedMoE` (allocation-free, CUDA-graph-capturable) is bit-identical to
`fused_experts`.

## 3. The baseline had to be tuned first, or it would have been a strawman

vLLM does not search the Triton tile parameters at runtime. It ships a lookup table
of pre-tuned JSON configs, one file per `(E, N, device, dtype)`, where `E` is the
expert count and `N` the per-expert intermediate size (`E, _, N = w2_shape`).

**Qwen3-30B-A3B is `E=128, N=768`. That file exists for H20, H200, B200, GB200 and
MI300X — and for no A100.** `NVIDIA_A100-SXM4-40GB` has 3 files out of 230 in the
whole directory (`E=16,N=1344`, `E=8,N=1792`, `E=8,N=3584`), all Mixtral-era 8/16
expert shapes. So `get_moe_configs` returns `None`, vLLM logs *"Using default MoE
config. Performance might be sub-optimal!"*, and falls back to a six-line if/else.

Running `b1_bench.py --model qwen --tune` (576 configs, 707 s) is the same procedure
as vLLM's own `benchmark_moe.py --tune`, for a combination upstream never covered:

At a locked 1335 MHz, CUDA graph of 4 layers, GEMMs only:

| config | layer time | TFLOP/s | % bf16 peak |
|---|---|---|---|
| vLLM `get_default_config` fallback `64/64/32, GROUP 8` | 5.050 ms | 115.8 | 37.1% |
| **tuned `128/128/64, GROUP 1, 4 warps, 3 stages`** | **3.686 ms** | **158.6** | **50.8%** |

**+37.0%.** Quoting a DVFS saving against the untuned config would have smuggled
that 37% into the result. Repeating the comparison with clocks *unlocked* (default
governor, which settles at 1320 MHz and 400 W for both configs) gives +37.3% — the
gain is not an artefact of locking. Tuned config is at `data/b1_config_qwen.json`,
same format as vLLM's.

Two measurement notes, both of which bit us first:

* **Time the GEMMs, not our align.** The tuning run's printed absolute times
  (6.850 / 5.280 ms) included our pure-PyTorch `moe_align_block_size` on every call
  — argsort + bincount + scatter, 0.76 ms. vLLM's align is a fused CUDA op, so
  charging the baseline for ours would understate it. The *ranking* was unaffected
  (a constant added to every config does not move the argmin), so the tuned config
  itself stands. `b1_bench.py` now times `PreparedMoE.run()` and reports align
  separately.
* **Use CUDA graphs.** Eager launch of this driver is CPU-bound: 4 kernel launches
  per layer through Triton's Python path cost enough that the 3.7 ms tuned config
  measures 4.42 ms eagerly while the 5.05 ms default config is unaffected (its GPU
  time hides the CPU time). All numbers here are CUDA-graph replays.

## 4. The tuner chose to eat the imbalance penalty — this matters for B2

`BLOCK_SIZE_M` sets how many token-rows one tile covers, and each expert's run is
padded up to a multiple of it. Rows the kernel actually computes, versus rows that
carry real tokens, for this routing draw (rep 0, layer 0):

| BLOCK_SIZE_M | rows computed | wasted |
|---|---|---|
| 16 | 62 816 | +1.4% |
| 32 | 63 904 | +3.2% |
| 64 | 65 984 | +6.5% |
| **128 (chosen)** | **70 272** | **+13.4%** |
| 256 | 78 336 | +26.4% |

**The tuner picked the tile with the most padding waste.** A wide tile amortises the
weight load and keeps the tensor cores fed, and that is worth more than 12 points of
wasted rows. This is the same trade B2 faces from the other side: pinning blocks to
SM partitions to harvest slack costs makespan, and the kernel is already telling us
it would rather waste work than lose tile efficiency.

The imbalance is real and large: this layer routes **min 3, mean 484, max 1349**
tokens per expert (max/mean = 2.79). At `BLOCK_SIZE_M=128`, the expert with 3 tokens
still occupies a full 128-row tile — 97.7% padding for that expert.

## 5. How far the fused MoE is from a dense GEMM of the same FLOPs

Upper bound: the same 584.7 GFLOP as two plain cuBLAS GEMMs — no expert grouping, no
padding, weights read once.

| | time | TFLOP/s | % peak |
|---|---|---|---|
| dense `[61952x2048]@[2048x1536]` + `[61952x768]@[768x2048]` | 2.761 ms | 211.8 | 67.9% |
| tuned fused MoE, same FLOPs | 3.724 ms | 157.0 | 50.3% |

Both under the default governor, which settles at ~1320 MHz against the 400 W cap for
either workload. **1.35x.** 13.4 points of that is padding. The rest is weight traffic — the dense
case reads one 6.3 MB B matrix, the MoE case reads all 128 experts' 1.12 GiB of
weights — plus Triton-vs-cuBLAS. Half of peak on a grouped GEMM with a 768-wide
expert is a fair baseline, not a crippled one.

## 6. Reproduce

```bash
source tools/moe_megakernel/env.sh
cd tools/moe_megakernel
CUDA_VISIBLE_DEVICES=0 python3 test_fused_moe.py            # correctness
CUDA_VISIBLE_DEVICES=0 python3 b1_bench.py --model qwen --tune   # 707 s
CUDA_VISIBLE_DEVICES=0 python3 b1_energy.py --model qwen --repeats 6   # ~25 min,
                                                            # locks clocks on GPU 0
CUDA_VISIBLE_DEVICES=0 python3 make_b1_fig.py               # -> figs/fig17_b1_energy.pdf
```

`--model` also accepts `qwen3next`, `olmoe`, `granite`; shapes and the routing file
each one reads are in `SHAPES` at the top of `b1_bench.py`.

## 7. Energy — the first honest MoE energy measurement in this project

`b1_energy.py`: **6 independent passes** over a 38-point clock grid from 510 to 1410 MHz,
the grid **reshuffled each pass** so slow thermal drift cannot alias onto clock, and one
**unlocked default-governor** measurement at the head of every pass. NVML median board
power over a ~2.5 s window of CUDA-graph replays; idle measured and subtracted at every
clock. The grid steps 15 MHz — the A100's real step — from 915 all the way to 1290, in one
campaign. An earlier version stepped 45 MHz above 1155 and had the six missing clocks
filled in later as a *separate* campaign; that is not comparable at the few-mJ level,
because the reshuffle exists to stop drift aliasing onto clock and it does not reach
across campaigns. Re-collecting everything at once measured the offset that had been
introduced: **-2.96 mJ**. 234 rows in `data/b1_energy_qwen.csv`; figure
`figs/fig17_b1_energy.pdf`.

| SM clock (requested -> measured) | layer time | energy/layer | |
|---|---|---|---|
| governor (unlocked) -> 1322 | 3.759 +/- 0.019 ms | 1503.6 +/- 7.4 mJ | **the deployed point** |
| 1410 -> 1320 | 3.727 +/- 0.002 ms | 1495.6 +/- 8.2 mJ | 400 W cap, clock not held |
| 1380 -> 1320 | 3.722 +/- 0.003 ms | 1490.2 +/- 11.5 mJ | 400 W cap, clock not held |
| 1335 -> 1320 | 3.727 +/- 0.003 ms | 1494.2 +/- 8.3 mJ | 400 W cap, clock not held |
| 1200 -> 1200 | 4.047 +/- 0.000 ms | 1298.1 +/- 5.9 mJ | |
| 1155 -> 1155 | 4.195 +/- 0.003 ms | 1234.7 +/- 5.2 mJ | |
| **1035 -> 1035** | **4.627 +/- 0.001 ms** | **1141.8 +/- 1.2 mJ** | **energy minimum** |
| 1050 -> 1050 | 4.571 +/- 0.004 ms | 1144.5 +/- 8.7 mJ | tied with 1035 |
| 510 -> 510 | 9.021 +/- 0.000 ms | 1408.2 +/- 1.9 mJ | static power dominates |

**Against the deployed governor point, 1050 MHz is -23.9% energy for +21.6% latency**
— and 1050, not 1035, is the one to deploy: they tie on energy (1144.5 +/- 8.7 vs
1141.8 +/- 1.2 mJ, t = 0.74) and 1050 is 1.2% faster. 1035 is -24.1% / +23.1%, which is
the same energy for 1.5 pp more latency. `figs/fig17_b1_energy.pdf` labels 1050 for that
reason.
Essentially a 1:1 trade — which is why the earlier draft of the figure, which drew that
as a near-vertical arrow, was misleading and was removed.

### Four things this measurement settles that no composed number could

1. **The entire top of the clock range is one operating state.** 1335, 1380 and 1410 MHz
   all *measure* 1320 MHz and all land at 3.722-3.727 ms — within 0.13% of each other,
   against standard deviations of 0.002-0.003 ms. The layer is against the 400 W board cap at all
   three, so asking for a higher clock changes nothing except the number NVML reports back.
   The unlocked governor sits in the same place (1322 MHz). Whatever the default policy
   thinks it is buying above ~1320 MHz, it is buying nothing.
2. **Most of the U-shape is idle power, not the kernel.** Over 510-1050 MHz **dynamic**
   energy (idle subtracted) is flat within **1.7%**, while **total** board energy varies
   **23%**. The left-hand rise of the total-energy curve is the 60-79 W idle draw amortised
   over a layer that takes twice as long — not the silicon becoming less efficient. This
   matters for the recommendation: the optimum is conditional on charging one layer the
   full idle draw of an otherwise-empty GPU, and on a saturated or shared GPU it moves.
   (Dynamic energy does still minimise at 1035 MHz, so the optimum is not *purely* an
   artefact — just a very shallow one.)
3. **The optimum is a basin, not a point, and it is metric-dependent.** 1035 MHz
   (1141.8 +/- 1.2 mJ) and 1050 MHz (1144.5 +/- 8.7 mJ) are statistically tied; the
   energy-**delay** product instead minimises at **1125 MHz**. Quoting a single
   "energy-optimal clock" without saying which objective it optimises is not defensible.
4. **The curve is shallow on the low side and steep on the high side**, so the marginal
   rate of exchange collapses as you descend: 1290 MHz returns 2.5 units of energy per
   unit of latency, 1200 MHz returns 1.7, and by 1035 MHz it is 1.0. If latency has any
   value at all, the interesting region is 1200-1290 MHz, not the minimum.

### What this does NOT say

This is the **work-conserving** baseline: one grouped GEMM, one clock, the queue free to
put any tile on any SM. It is the number B2 must beat. It says nothing yet about whether
pinning blocks to SM partitions — which is what a second V/f domain requires — costs more
makespan than the second domain saves. That is the next experiment, and it needs no DVFS
hardware: same kernel, same clock, work-conserving vs pinned.
