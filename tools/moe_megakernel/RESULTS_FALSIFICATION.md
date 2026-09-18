# The hypothesis is falsified. Two experiments, ~1 hour, no kernel written.

`PLAN.md` §4 proposed: *routing imbalance manufactures more intra-kernel heterogeneity
than GEMM+collective overlap does, so spatial multi-V/f is worth more in MoE inference.*

It is not — though not for the reason I first gave. Heterogeneity **does** exist inside a
MoE layer. It is simply concentrated in a stage worth 8% of the energy, so a perfect
per-stage V/f assignment is worth **0.20%**.

## Experiment 1 — do different expert loads want different clocks?

`expert_clock_sweep.py` -> `data/expert_clock.csv`. Nine `n_e` values taken from the
measured load distribution (72 to 30754 rows), both expert GEMMs, eight locked clocks,
2.5 s sustained windows, NVML median, held-clock filter.

| energy-optimal clock | shapes |
|---|---|
| **1050 MHz** | **17 / 18** |
| 705 MHz | 1 / 18 (the smallest `down` GEMM, n_e = 72) |

Occupancy varies enormously across those shapes — `S_eff` runs from **8 to 107 SMs**, 1
wave to 36 waves — and the optimum does not move. The low-occupancy correction that
mattered in `overlap_exp` changes *how much* energy is spent, not *at what clock*.

## Experiment 2 — do different pipeline stages want different clocks?

`stage_clock_sweep.py` -> `data/stage_clock.csv`. The four stages of a prefill layer at
their real shapes, since a megakernel fuses compute and data movement into one kernel.

| stage | kind | best f | penalty at 1050 |
|---|---|---|---|
| router GEMM | compute, tiny N | 1050 | 0.0% |
| permute (gather) | pure data movement | 705 | **2.7%** |
| expert GEMM | compute, the 99.7% | 1050 | 0.0% |
| combine (scatter) | pure data movement | 1050 | 0.0% |

The only stage that prefers a different clock gives up **2.7%** by running at 1050 anyway.
That is not a hardware argument.

## Why — and the first version of this section was WRONG

I first wrote "nothing in a MoE layer is insensitive to SM frequency". Plotting it
disproved that. `T x f` is constant if and only if time scales as 1/f, i.e. the work is
compute bound. Measured spread across 300-1410 MHz:

| | T x f spread | verdict |
|---|---|---|
| expert GEMM (99.7% of FLOPs) | **1.02x** | perfectly compute bound |
| combine (scatter) | 1.62x | partly bandwidth bound |
| NCCL all-reduce (`overlap_exp`) | 2.26x | link bound |
| **permute (gather)** | **3.19x** | **more bandwidth bound than the collective** |

**Heterogeneity is not absent — the permute is further from 1/f than the NCCL collective
that made a second domain pay in `overlap_exp`.** It genuinely wants 705 MHz, not 1050.

The real blockage is the energy budget:

| stage | share of layer energy at 1050 | wants | cost of running it at 1050 |
|---|---|---|---|
| expert GEMM | 43.0% | 1050 | 0 |
| combine | 48.4% | 1050 | 0 |
| **permute** | **7.8%** | **705** | 2.7% of itself = **0.20% of the layer** |
| router GEMM | 0.8% | 1050 | 0 |

**A perfect per-stage V/f assignment saves 0.20% of a prefill layer.** The 91% that
dominates the budget is compute bound, so its optimum is pinned to the silicon's kappa
knee (~1035 MHz, measured in `overlap_exp` at 15 MHz resolution) and nothing about the
workload can move it: for compute-bound work

    E(f) = (W/k) [ kappa(f) + P_static / f ]

and the workload appears only in the prefactor `W/k`.

**So: heterogeneity exists, but it is in the wrong place to pay for a voltage domain.**

### The one thing that could change this

`combine` is 48% of the budget here, measured as torch's `index_add_` with atomics. A
fused kernel would make it far cheaper, which would *raise* the permute's share. Even if
combine collapsed to a tenth of its cost, the permute would rise to ~13% of the budget and
the gain to ~0.35%. Still not a hardware argument — but B1 is what would settle it.

## What survived, and it is the useful half

The free win is large and independently confirmed on a completely different workload:

| | |
|---|---|
| expert GEMM at 1050 vs 1410 | **-32% to -39% energy** |
| optimal clock, this workload | 1050 MHz |
| optimal clock, `overlap_exp` | 1050 MHz (202/216 configs) |

Two unrelated workloads, same answer, because it is the silicon's knee. And at the larger
expert shapes **1410 MHz did not even hold** — the GPU throttled off it in 12 of 18 cases,
so the governor's preferred clock is not merely expensive, it is partly fictional.

## The cost of finding out

Two sweeps, ~1 hour of GPU time, zero lines of kernel code. Had we built B1 and B2 first —
a fused grouped GEMM and a persistent megakernel with per-block timing — we would have
spent weeks arriving at the same 2.7%.
