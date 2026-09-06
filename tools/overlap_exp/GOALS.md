# Project: energy modelling and DVFS design for overlapped GEMM + collective

Fork of NVIDIA `aiconfigurator`. Hardware: one 4x A100-SXM4-40GB GCP node.
Written 2026-08-28. This file states what we are trying to find out and why; `README.md`
states how the code is arranged, `OVERLAP_MODEL.md` states what the model is.

---

## 1. The question

Distributed LLM training and inference spend a large fraction of every iteration with a
**GEMM and a NCCL collective resident on the same GPU at the same time**. The two have
opposite appetites: the GEMM is compute-bound and wants a high clock; the collective is
latency- and link-bound and mostly wants SMs, not MHz. Today one clock serves both,
chosen by a governor whose objective is *finish fast*, not *finish cheap*.

**Central question — given an arbitrary GEMM and an arbitrary collective that overlap,
what is the optimal setting of (#SM given to each, frequency of each), and how much
energy does getting it right save?**

Two sub-questions follow, and they turn out to have very different answers:

1. **Is a second voltage/frequency domain worth the silicon?** If the GEMM's SMs and the
   collective's SMs could run at different clocks, how much does that buy over the best
   single clock?
2. **What is already free?** How much of the opportunity needs no new hardware at all —
   just picking a better clock, or a better SM split, than the default?

## 2. Why it needs measurement rather than a formula

Every plausible shortcut turned out to be wrong on this hardware, which is why the
project is measurement-first:

- Occupancy is a **staircase**, not a ratio: `W = ceil(grid/S)`, so latency does not
  scale as `108/S`. `1/S` models are wrong by tens of percent at low occupancy.
- cuBLAS **split-K** silently changes the work per block. Assuming otherwise cost 21% on
  one shape, so grids are ncu-measured, never computed.
- The **voltage curve is flat then knees**. `kappa = P_dyn/f = C*V^2` is flat to 0.5% from
  510 to 1035 MHz and then climbs 33% by 1230. Any model linear in `f` misses this, and
  the knee is exactly where the optimum sits.
- **Throttling is governed by mean power, not peak** (88.6% vs 31.5% precision over 432
  cases). A peak-based cap vetoes configurations the GPU demonstrably runs.
- A GEMM inside an overlap runs **2.41x** slower than solo, where the wave model predicts
  1.40x. Contention is not captured by occupancy arithmetic alone.

## 3. Scope

**In scope**

- A100, world = 4, NCCL all-reduce, TensorRT-LLM GEMM shapes (the user's standing
  constraint: TRT-LLM variants only).
- SM splits from 2 to 64 NCCL CTAs (1 CTA == 1 SM exactly, ncu-verified), clocks from 300
  to 1410 MHz, all-reduce messages from 64 to 512 MiB.
- Soft partitioning as measured (the hardware block scheduler shares SMs); hard
  partitioning and two clock domains as *composed* counterfactuals.

**Out of scope / known limits**

- **No hardware with two V/f domains exists here.** Every two-domain number is composed
  from measured solo kernels, validated on the diagonal (`f_gemm == f_comm`) against 1404
  real overlapped runs: energy bias **-1.29%**, median absolute error 1.60%. The
  composition is mildly optimistic and that bias is quoted with every result.
- One GPU generation, one collective, one world size. Nothing here is claimed to
  generalise to H100/NVL, to all-gather / reduce-scatter, or to world != 4.
- Temperature is not modelled. Power caps are never changed (a box-wide side effect).
- Per-phase runtime DVFS is **ruled out, not ignored**: an NVML clock change costs
  37.7-38.5 ms one way against a 2.65% ceiling on what per-phase switching could win.
  Phases would have to be ~2.9 s long to pay for it. The knobs of interest are therefore
  *design-time*, not per-kernel.

## 4. What we are building

| layer | artefact |
|---|---|
| measurement | one parameterised harness (`sweep_overlap_432.py` + `SWEEP_*`) driving every campaign, so all data shares a schema and every trap (NCCL env caching, rank-identical iteration counts, persistence mode, the GCP gIB shim) is handled once |
| model | a **bottom-up** energy model: fit the GEMM and the collective separately from solo runs, then *compose* them into a prediction for the overlap. `P = P_static(f) + a(f)*S_eff + b(f, footprint)` for the GEMM, `P = p0(f) + gamma(f)*B(c,f)` for the collective |
| analysis | the joint (SM split, f_gemm, f_comm) optimiser under three objectives — energy, latency, EDP — with a mean-power cap and matched-latency comparison |
| deliverable | NeurIPS-class figures, each regenerated from data by a named script, plus the CSVs behind them |

## 5. What has been established

| finding | number |
|---|---|
| The governor's choice is the most expensive one | picks 1410 MHz on 155/216 configs; the energy-optimal locked clock is 1050 MHz on 202/216 |
| Locking to the best single clock is free and large | **-25.0% median energy**, wins 216/216 configs |
| The second V/f domain is worth much less, and shrinks as the split widens | -20.4% at 2 CTA, -3.5% at 22 CTA, **-2.2% at 32 CTA**, -0.5% at 64 CTA |
| The SM split is what actually matters | committing to the wrong split costs up to +172%; the split-cost curve is U-shaped with a flat bottom at 22-32 CTA |
| The collective saturates | past ~32 CTAs it gets no faster while the overlap gets slower |
| Runtime DVFS is not needed once the split is fixed | one burned-in (f_gemm, f_comm) pair costs only **0.25-1.19 pp** against retuning per workload |
| Overlap alone never causes throttling | 0/492 cases; adding the collective *lowers* mean power by a median 178 W |

**The headline**: the opportunity is real but it is mostly *not* an argument for new
silicon. Roughly 25 percentage points of the ~27-30% total are available today from a
better single clock and a sensible SM split; the second voltage domain adds a few points
on top, and only when the split is narrow enough that the two kernels genuinely want
different frequencies.

## 6. Open problems

- A **clock-correlated residual** (+0.93) on throttled rows that two hypotheses failed to
  explain.
- Two `P_static` campaigns disagree by 3-4 W (61.4/69.7 vs 64.3/74.1 W at 900/1200 MHz).
- `P_static` at 1050 and 1305 MHz is interpolated, not measured (both bracketed).
- `B(c,f)` was fitted on CTA <= 32; the 48/64 CTA rows are measured but not refitted.
- **Error bars understate cross-session drift.** Within one session the answer for a cell
  is reproducible to +-0.30 pp; across sessions one of three checked cells moved 1.35 pp.
  The Monte-Carlo bars in fig14 model the former only.
- The composition's -1.3% optimism is characterised but not corrected for.

## 7. What "done" looks like

1. A calibrated model that predicts the energy and latency of an arbitrary overlapped
   (GEMM, collective) pair at an arbitrary (split, f_gemm, f_comm), with a stated error
   bar, from solo-kernel measurements alone.
2. A design recommendation with its cost of being wrong: what SM split to fix in silicon,
   what clocks to burn in, and what the second V/f domain is worth.
3. Figures and CSVs that regenerate end to end from `data/` with one command each.
