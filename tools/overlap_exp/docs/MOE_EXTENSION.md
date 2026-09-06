# Proposed extension: energy-efficient execution under MoE imbalance

Research synthesis, 2026-09-06. This is a proposal following repository review,
not a report of new GPU experiments. The existing GEMM/all-reduce study remains
the measured foundation; its coefficients and conclusions do not automatically
transfer to MoE.

## Expanded question

Given routed inference work with uneven expert loads and communication
dependencies, how should we schedule useful work, allocate GPU execution
resources, and choose clocks to minimize total GPU energy within a latency
budget? How much comes from existing configuration controls, how much from
custom persistent scheduling, and how much would require new hardware?

The current experiment is a special case: two independent operations, identical
work on four ranks, and a fixed communication footprint. The extension introduces
unequal work across experts and ranks, dependencies, and changing work readiness.
The output should still be a measured cost model and an actionable configuration
recommendation, with the cost of choosing incorrectly.

## What the existing work establishes

- `../GOALS.md` captures the latest research position. `EXPERIMENTS.md`,
  `RESULTS_OVERNIGHT.md`, and `OPPORTUNITY.md` document earlier stages; their
  numerical conclusions use different candidate sets, objectives, and baselines.
- `../../gemm_dcfs_char/` measures SM availability, wave quantization, static
  power, and clock dependence. `../../comm_dvfs_char/` measures the collective
  bandwidth/CTA/clock relationship and checks TRT-LLM/NCCL equivalence.
- `../sweep_overlap_432.py` measures solo, serial, and concurrent execution.
  `../table_432.py` predicts from fitted coefficients; `../followup_baseline.py`
  composes measured solo kernels; `../fixed_design.py` searches fixed splits.
  These are different levels of modeling evidence.
- The voltage knee, communication saturation, and large split-selection penalty
  motivate software resource control. Independent per-SM clocks remain a hardware
  counterfactual. Equal-clock validation does not validate unequal-clock behavior.
- The measured NVML transition cost rules out per-kernel clock switching for the
  tested regime. Device-side scheduling is a different control mechanism whose
  cost must be measured separately.

One baseline distinction needs to remain explicit. A read-only calculation over
`commbound_*mib.csv` and `ext_*mib.csv`, selecting the lowest-energy held-clock
**measured concurrent** row per `(ar_mib, ctas, m, n, k)` and matching the unlocked
rows, gives 216 pairs: median energy change **-22.98%**, median latency change
**+18.91%**, and energy improvements in all 216. The chosen clock is 1050 MHz in
203 cases. There are no duplicate concurrent keys in that input set.

The diagonal of the existing **composed** `fixed_design.csv` gives **-25.045%**
and 202 selections of 1050 MHz. This explains the ~25% headline in GOALS; it is
not the direct measured-only comparison. Neither energy optimum is automatically
a saving at unchanged latency. These checks reselect from existing data and are
not independent repeat measurements of the selected configurations.

The analysis policy also differs by script: `fixed_design.py` uses a mean-power
cap, whereas `followup_baseline.py:main` and the default in `objectives.py` also
cap modeled peak power. Record the policy with each result before comparing it.

## What the repository already provides for MoE

Paths below are relative to the repository root.

| Existing component | Reuse and limitation |
|---|---|
| `collector/helper.py` | Balanced and power-law routing, expert replication, and placement helpers. Existing power-law paths remap the most heavily loaded rank to rank 0; preserve the actual mapping and all ranks for the new study. |
| `collector/trtllm/collect_moe.py` | TRT-LLM MoE invocation, shape/quantization handling, and synthetic routing. Multiple power-law samples are reduced to one latency; retain individual seeds and distributions for imbalance analysis. |
| `collector/wideep/trtllm/collect_moe_compute.py` | Explicitly a single-GPU compute simulator, including EPLB slot assignments. Useful for local compute calibration, not proof of distributed completion time or total GPU energy. |
| `collector/network/slurm/collect_trtllm_alltoall.py` | Separate dispatch/combine measurement boundaries. Current default cases are balanced and NVFP4; this is not an already-validated BF16/A100 imbalance harness. |
| `collector/sglang/dsv4_megamoe/README.md` | Useful precedent for cross-rank routing, source placement, and explicit timing boundaries. Its Blackwell MegaMoE setup is not the A100 baseline. |
| `aic-core/src/aiconfigurator_core/sdk/operations/moe.py` | Distribution-aware MoE queries and communication operations already exist. The new contribution should add the missing measured execution behavior, not merely another distribution label. |

Keep TRT-LLM as the production reference and use its model geometries. Start with
BF16 and EP=4, ETP=1 on the existing node. Confirm the installed runtime's actual
kernel and communication path before selecting the adapter; repository collector
imports do not by themselves establish compatibility with the installed package.

## Separate three mechanisms

| Mechanism | Controlled experiment | Candidate intervention |
|---|---|---|
| Uneven work within a GPU | Uneven expert loads with equal aggregate work across ranks | Tile scheduling across local experts; persistent worker count and granularity |
| Uneven work across GPUs | Concentrate hot experts on one rank while holding logical routing fixed | Expert placement/replication; later, static per-rank clock choices |
| Work waiting on dependencies | Hold expert counts fixed while changing source placement and arrival order | Dispatch/compute/combine chunking and scheduling only ready work |

A local queue cannot move work to a GPU that does not hold the required expert
weights. Cross-rank balancing must account for placement, replication memory,
traffic, and any migration overhead. Likewise, more expert skew does not imply
slower grouped GEMM: concentrating tokens can improve tile utilization while
making rank imbalance worse. Measure both effects.

Persistent execution alone is not a novelty claim. CUTLASS already documents
persistent grouped GEMM and tile scheduling across problems. First profile the
TRT-LLM baseline. The hypothesis is that a controllable schedule, resource budget,
or readiness policy improves the measured energy/latency tradeoff beyond that
baseline. [CUTLASS grouped scheduler documentation](https://docs.nvidia.com/cutlass/4.5.2/media/docs/cpp/grouped_scheduler.html).

## First custom kernel and ablations

Start with a grouped expert GEMM using an A100-compatible implementation and
fixed worker count. Preserve arithmetic, layout, tile shape, and epilogue while
comparing a static tile assignment with dynamic local work assignment. Use the
same routed inputs and weights in every arm. Change worker count separately.
Instrument completed tiles and the final worker tail; do not infer useful SM
activity from the persistent launch grid. A resident block can be waiting.

Compare against both the existing TRT-LLM kernel and the same custom kernel with
static scheduling. This distinguishes implementation changes from scheduling
changes. Validate numerical outputs, routing weights, token conservation, empty
experts, partial tiles, and zero-token ranks before performance measurements.

Only after this comparison should the experiment add overlap with communication.
Use separate streams and explicit chunk dependencies first. Within each chunk,
dispatch must precede expert computation, which must precede combine. Any overlap
must come from independent chunks or work with satisfied dependencies. A fused
persistent communication/compute kernel is a later experiment, requiring its own
progress and synchronization design. Stream priority alone is not a guarantee
that communication can make progress alongside resident compute blocks.

## Measurement and modeling changes

Use total energy of the four GPU boards over a common completed-work interval:

```
E_gpu_total = sum_r integral_[0,T_complete] P_r(t) dt
minimize E_gpu_total subject to T_complete <= latency_budget
```

This includes early-finishing ranks waiting for the others. It is GPU energy,
not whole-machine energy. Record per-rank achieved clocks, power samples,
temperature, phase durations, routing counts, source/destination traffic, expert
placement, scheduler settings, padding, and actual kernel identities.

The existing harness polls all boards but stores their averaged median power and
rank-0 loop timing. Preserve per-rank observations and timestamped samples for
the new harness. Use sufficiently long repeated windows for energy and separate
GPU timing/tracing for phase behavior; NVML samples do not resolve individual
short kernels. Changing persistent execution also requires revisiting the current
per-iteration device-wide synchronization.

The model should retain calibrated compute/communication costs but compose them
over a dependency graph with per-rank work and readiness. Do not replace this by
`max(total_compute, total_comm)` or by a single skew multiplier. Queue overhead,
waiting activity, padding, locality, and contention are measured terms or explicit
residuals. Refit for the new kernel; dense-GEMM clock linearity and conserved solo
dynamic energy are hypotheses here, not inherited laws.

Report energy/latency frontiers and savings at the same latency budget. Keep
unlocked TRT-LLM and tuned shared-clock TRT-LLM baselines. Use routing seeds and
separate measurement sessions as sampling units, retaining them in the raw data.
Select settings on calibration runs and confirm them on held-out seeds, shapes,
and sessions. Do not interpret tiny gains below selection noise or session drift.

## Order of experiments and decisions

1. **Establish the baseline timeline.** One TRT-LLM-derived MoE layer, small decode
   and larger prefill token counts, balanced/local-skew/rank-skew routing. Measure
   all four ranks. Synthetic routed-layer execution is the first scope; inference
   serving claims require later validation in a real model and request workload.
2. **Test local scheduling.** Existing TRT-LLM, custom static, custom dynamic,
   matched arithmetic and inputs. If dynamic scheduling does not reduce exposed
   worker tails or energy at the latency budget, stop expanding that mechanism.
3. **Test dependency-aware overlap.** Same placement and routing, varying chunk
   size and compute/communication resource budgets. Reject apparent gains from
   omitted dispatch, routing, combine, or synchronization work.
4. **Test placement separately.** Replay the same logical routing with fixed and
   balanced expert placements, then optional replicas under an explicit memory
   budget. Measure changed traffic and setup cost separately from steady state.
5. **Reintroduce clocks.** Sweep shared clocks on the measured Pareto candidates.
   Persistent skew also makes static per-GPU clocks a testable hypothesis on this
   node, unlike symmetric TP. Rotate the hot rank and test changing skew before
   claiming a policy; retain the measured per-kernel DVFS limitation.
6. **Integrate the predictive result.** Once validated, expose kernel/schedule,
   workload distribution, and execution scope through the existing AIC operation
   data/query machinery, preserving source attribution and interpolation identity.

The immediate decision is whether useful work is stranded locally, remotely, or
behind communication dependencies. That determines which kernel and scheduling
control is worth building. No new performance gain is assumed by this proposal.
