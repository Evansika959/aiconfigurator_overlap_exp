# MoE megakernel DVFS — plan

Successor to `../overlap_exp`. Same box (4x A100-SXM4-40GB), same measurement stack, a
different source of heterogeneity: **expert routing imbalance inside one persistent
kernel**, instead of two concurrent kernels with different appetites.

---

## 1. The premise has a timescale problem — face it before designing anything

`overlap_exp` measured it: **an NVML clock change costs 37.7-38.5 ms one way** (10 ms
blocking call + 28 ms settle; the `nvidia-smi` CLI adds 65-74 ms of process spawn on top).
A MoE decode layer is tens to hundreds of **microseconds**. The routing draw changes every
batch.

So "observe this batch's imbalance, then lower the clock on the slack experts" is
**physically impossible on this hardware**, by a factor of ~1000 in time. Any plan that
does not say how it escapes this is dead on arrival. There are exactly three escapes:

| escape | what it means | status |
|---|---|---|
| **spatial** | different SMs run at different V/f *simultaneously*; no switching in time | a hardware proposal. `overlap_exp` already built the machinery to cost one (compose from measured pieces, validate on the diagonal) |
| **statistical** | one clock per *serving regime* (batch size / SLO tier), chosen to be good in expectation over the routing distribution, not per batch | free today, changes at deployment cadence, not per token |
| **assumed hardware** | posit per-SM fast DVFS (sub-microsecond), as a research premise | legitimate, but must be declared in the title, not buried |

**This is decision #1 and it determines what we measure.**

## 2. A megakernel deliberately destroys the slack we want to harvest

A persistent kernel launches ~one block per SM and each block pulls work items from a
queue until the queue is empty. That design is **work-conserving on purpose**: no expert
"finishes early and idles", because a block that finishes expert 3's tile immediately
takes expert 7's tile. Load imbalance does not become idle time; it becomes *tail* time.

So the slack we want to sell only exists if we **pin blocks to experts** — reintroducing
the spatial partition that the megakernel was written to avoid. That costs latency.

**The trade is the research question**: how much energy does a partitioned megakernel buy,
and how much latency does it give back, versus a work-conserving one at a lower clock?

## 3. Where the slack actually is, in decreasing size

1. **SLO slack.** The layer finishes before its deadline; today's governor races to idle
   at 1410 MHz, the most expensive clock available. `overlap_exp`: pacing to the deadline
   at the energy-optimal clock is worth **-25% median, winning 216/216 configs**, free.
   This is almost certainly the biggest term here too and it needs no MoE-specific idea.
2. **Tail / wave slack.** The last wave of a grouped GEMM has fewer live blocks than SMs.
   Imbalance makes the tail longer and emptier. Those SMs are the natural target of a
   second domain.
3. **Intensity heterogeneity across experts.** A thin expert (few tokens, small M) is
   wave-quantised and memory-bound; a fat expert is compute-bound. `overlap_exp` measured
   that the energy-optimal clock moves with operational intensity, and that kappa = P_dyn/f
   is flat to 1035 MHz then climbs 33% by 1230. **Different experts in the same kernel
   want different clocks.** This is the direct analogue of GEMM-vs-collective, and it is
   the strongest argument for a spatial multi-domain design.

## 4. The hypothesis worth testing

> Routing imbalance manufactures **more** intra-kernel heterogeneity than GEMM+collective
> overlap does, so a spatial multi-V/f design is worth **more** in MoE inference than the
> **2.2%** (at a 30% SM split) we measured in `overlap_exp`.

Null hypothesis, and it is a real possibility: a work-conserving queue smooths the
imbalance away, the whole layer looks like one homogeneous compute-bound blob, and the
second domain is worth ~0 — with all of the gain sitting in item 1 above, which is free.

Either answer is publishable. The failure mode to avoid is measuring only the composed
optimum and reporting a large number that is really item 1 wearing a costume.

## 5. What transfers from `overlap_exp` — do not rebuild

| asset | why it still applies |
|---|---|
| kappa = P_dyn/f curve, knee at ~1035 MHz | property of the silicon, not the workload |
| mean-power throttle criterion (88.6% vs 31.5% precision for peak) | same |
| `P_static(f)`, the 400 W cap, NVML sampling at 20 ms | same |
| the wave model `W = ceil(grid/S)`, `S_eff = grid/W`, eta(S) low-occupancy correction | grouped GEMM is still a GEMM; imbalance is exactly a wave-quantisation story |
| ncu-measured grids, never computed (split-K bit us for 21%) | same |
| compose-then-validate-on-the-diagonal methodology | the only way to cost hardware that does not exist |
| the 10-repeat protocol; power CV 0.53% of which 89% is common-mode | same instrument |
| matched-latency comparison; unlocked-governor baseline | same failure modes |
| `figstyle.py`, the validated palette, PNG-only at 600 dpi | same |

## 6. Phases

**Phase 0 — how imbalanced is routing, really?** Without this number the project has no
premise. Per-layer, per-batch token counts per expert; the distribution of
max/mean and of the Gini coefficient; how it varies with batch size and with
prefill vs decode. Cheap, mostly CPU.

**Phase 1 — a persistent-kernel testbed.** Grouped GEMM as a megakernel with (a) a
controllable block-to-work mapping (work-conserving queue <-> pinned partition, and points
in between) and (b) **per-block start/end timestamps and SM id** (`clock64()`, `%smid`), so
we can see the slack map rather than infer it. This is the real build cost of the project.

**Phase 2 — energy characterisation.** Sweep clock x imbalance level x assignment policy x
batch size. Same harness discipline as `../overlap_exp/sweep_overlap_432.py`: one parameterised script,
env overrides, resume, held-clock filter.

**Phase 3 — the optimiser.** Given a routing draw and an SLO, choose (assignment, clock or
per-domain clocks) minimising energy subject to the deadline. Three arms, exactly as
before: today's default; best single clock (free); N spatial domains (silicon).

**Phase 4 — the honesty pass.** Repeats for the headline cells; composition bias quoted;
the statistical-vs-per-batch distinction enforced everywhere; what is measured vs composed
labelled on every figure.

## 7. Decisions taken

| | decision | consequence |
|---|---|---|
| premise | **spatial only** | no switching in time, so the 38 ms transition never enters. Different SM groups at different V/f *simultaneously*. A silicon proposal, costed the way `overlap_exp` costed one: compose from measured pieces, validate on the diagonal. Any per-batch-adaptive claim is out of scope by construction. |
| kernel | **custom kernel is the goal**, real workloads first as the baseline | Phase 1 builds our own persistent kernel with per-block timing. But nothing is compared against a strawman: the baseline is a real MoE model actually running. |
| workload | **real MoE model end to end** | routing traces and the baseline both come from a model that exists, not from a synthetic imbalance knob. Imbalance becomes an observed quantity first, a swept one second. |

## 8. The three-rung ladder — B0, B1, B2

Profiling the current path settled what Phase 1 has to build. One decode step of OLMoE at
batch 1 through HuggingFace launches **8277 CUDA kernels** and spends 25.45 ms on device,
of which only 16% is arithmetic (cuBLAS `gemvx`); the rest is `cub` compaction, indexing,
and **1024 device-to-host memcpys per step** — one per (layer, expert) from the
`torch.where` inside the Python expert loop. 1024 = 16 layers x 64 experts.

Nothing can be compared against that. So three rungs, and every later claim must say which
rung it is measured against:

| rung | what it is | what it is FOR | what it must never be used for |
|---|---|---|---|
| **B0** | HuggingFace `OlmoeSparseMoeBlock`: a Python loop over all 64 experts | routing traces (the router is exact) and a *quantified* strawman | any energy or latency baseline |
| **B1** | fused grouped GEMM — one launch for all experts, tokens sorted by expert, segmented | **the honest energy and latency baseline**, and the real decode step time | — |
| **B2** | our persistent megakernel: blocks pinned to SM partitions, `clock64()` / `%smid` per-block timing | the vehicle for spatial V/f partitioning | — |

B1 comes first because without it there is no honest baseline, and B2 reuses B1's
grouped-GEMM arithmetic with a different scheduler — so B1 is not a detour, it is B2's
first half.

**B1 also settles an open question.** The autocorrelation in `RESULTS_PHASE0.md` (r = 0.74
at lag 64) only threatens the spatial-only premise if a decode step is long enough that 38
steps exceed 38 ms. B0's 25 ms/step is an artefact of the Python loop and says nothing.
B1's step time is the number that decides it.

### B1 status: BUILT — see `RESULTS_B1.md`

vLLM v0.11.0's Triton `fused_moe_kernel` vendored into `vendor/fused_moe_triton.py`
(align step and driver rewritten here), verified against a naive per-expert loop, and
tuned for Qwen3-30B-A3B on A100 — a shape vLLM ships no config for. Prefill layer at
1335 MHz: **3.686 ms, 158.6 TFLOP/s, 50.8% of bf16 peak**, 1.35x a dense GEMM of the
same FLOPs; tuning was worth **+37%** over vLLM's fallback heuristic. Energy sweep, 6
passes over a reshuffled clock grid plus a governor baseline each pass: **1035 MHz is
-23.6% energy for +23.4% latency** against the deployed governor point. Two findings
that only a measurement could give: 1335/1380/1410 MHz all *measure* 1320 MHz and are
one operating state against the 400 W cap; and **dynamic** energy is flat within 1.0%
from 510-1050 MHz while **total** varies 23%, so most of the U-shape is idle draw
amortised over a longer layer, not the kernel.

Still open on this rung: the decode step time (only prefill has been measured).

### B2 status: BUILT, AND THE ANSWER IS NO — see `RESULTS_B2.md`

`persistent_moe.py` is a working partitioned persistent grouped-GEMM MoE kernel,
bit-identical to the vLLM path whole and partitioned. It was built to test the
hypothesis in §4 and it falsifies it.

The experiment that could have killed B2 early — pinning vs work-conserving at one
clock — **cleared it**: pinning costs only +1.0%. What kills it is two other things.

1. **Persistence costs +4.6%**, and the reason is occupancy, not scheduling: the
   non-persistent kernel keeps several blocks resident per SM and overlaps one tile's
   epilogue with the next tile's loads. One block per SM — the crisp ncu-verified
   "1 CTA = 1 SM" partition — forfeits that and costs +24.4%. Two blocks/SM recovers it,
   at the price of weakening the disjointness guarantee to occupancy accounting.
2. **There is no voltage left to harvest.** `RESULTS_B1.md` §7: dynamic energy is flat
   within 1.0% from 510 to 1050 MHz. kappa is the entire prize of a low-voltage domain,
   and below the knee there is none; above the knee the saving is equally available to a
   single uniform clock, which pays no makespan tax to get it.

Composed at matched latency over every (f_hi, f_lo, split), with both sides scored on the
same clock set under the same deadline rule: **given a free persistent kernel the second
domain is worth -1.7% at the A100's real 15 MHz clock granularity** — and that is
granularity, not voltage. The win scales as **(clock step)^0.73 +/- 0.05** and goes to
zero in the continuous limit; across 8316 refits the exponent never exceeded 0.85. **With
the measured build cost (+5.6% latency, +3.0% dynamic energy) it wins 0 of 31 reachable
deadlines, is +7.1% worse at best, and loses by 41.9 mJ even with no deadline at all.**

The null hypothesis stated in §4 — "the second domain is worth ~0, with all of the gain
sitting in item 1 of §3, which is free" — is the one the data supports. §3 item 1 is
indeed where the money is: pacing to the deadline at the energy-optimal clock is worth
-24% on the measured layer, and needs no MoE-specific idea at all.

### What is still open

* **Decode.** Memory-bound by 12-82x (`RESULTS_ROOFLINE.md`), so neither the wave model
  nor the compute-bound composition above describes it. Untested.
* **Per-partition kappa.** The composition charges each partition's work at the whole
  layer's kappa. A partition of only lightly-routed experts is more wave-quantised and
  may differ. Measuring it needs each partition run alone at each clock.
* **Disjointness at 2 blocks/SM** is not ncu-verified.
* **Other silicon.** The verdict is a statement about this voltage curve, not about the
  idea. The method — measure kappa, then compose — transfers; the number does not.

### What the review of these two figures cost, and what it bought

Seven rounds, and the substantive findings were never about drawing. Each was a claim
that was true of a number but false of what the number measured: a baseline scored by
continuous interpolation against a discrete search; an idle reference taken at a clock
the kernel never ran at; a clock grid with six holes in the region carrying the result;
a second measurement campaign that no reshuffle could reach across; and three successive
uncertainty components each quoted as though it were the whole. Two of those inverted the
headline. The record of them is in `RESULTS_B2.md` §3-4, because the transferable part is
the pattern, not the numbers.

## 9. Environment

PEP 668 blocks `pip --user` here and `python3-venv` is absent (installing it needs sudo).
Extra packages therefore live in a folder-local `pylibs/` on `PYTHONPATH` — `source
env.sh`. Nothing outside this folder is modified; the system torch 2.9.1 and Triton 3.5.1
are reused. `env.sh` documents the pins and the two traps (`--no-deps`, or pip drags in
torch 2.14 and shadows the system one; stale `.dist-info` after a `--target` upgrade).

Installed: `transformers 4.57.1`, `accelerate 1.10.1`, `tokenizers 0.22.2`,
`huggingface_hub 0.35.3`, `safetensors 0.8.0`. Verified: Olmoe / Qwen3Moe / Qwen2Moe /
Mixtral architectures all import against the system torch.

## 10. One thing to be careful about in Phase 0

HuggingFace's MoE forward **loops over experts in Python**. That is fine for harvesting
routing traces — the router is exact — but it is **not a realistic energy baseline**, and
quoting it as one would inflate every later result. So Phase 0 splits:

- **0a routing traces** — HF is the right tool. Hook the router, dump per-layer,
  per-batch token counts per expert.
- **0b a real baseline** — needs a *fused* grouped-GEMM MoE. Building it is Phase 1's job
  anyway, so 0b is the first milestone of the custom kernel rather than a separate
  dependency on a serving stack.
