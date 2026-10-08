# From one GPU to a node: extending the simulator to multi-GPU training

2026-10-08. What changes when the workload is a training step on several GPUs instead
of one MoE layer on one, what of that is built and measured (`sim/node.py`,
`RESULTS_NODE.md`), and what is still a plan.

## 1. Why the question changes sign

Inside one GPU the answer to "is a second V/f domain worth it" was no, and the reason
was structural: expert work can be moved between SMs freely, every SM has the same
convex cost curve, so the best plan re-splits the SMs until one clock is optimal
(`RESULTS_SIM.md`; Jensen). A second clock only repaired a split that was wrong.

Across GPUs none of the three premises holds.

| Premise inside a GPU | Across GPUs |
|---|---|
| Work moves freely between SMs | A rank's work is fixed by the parallelisation: its shard of the batch, its pipeline stage, its experts. It cannot be re-split per step. |
| One clock, so a second domain is hypothetical | Every GPU already has its own clock, voltage and 400 W limiter. The domain exists and `nvidia-smi -lgc` sets it. |
| Partitions wait for each other at a barrier inside one kernel | Ranks wait for each other at a collective. A rank that arrives early idles, at the governor's top clock, until the slowest one arrives. |

So imbalance between GPUs is real, it persists, and the knob to exploit it is on the
shelf. The research question for a node is therefore not "would a new domain help" but
"how much of the slack do per-GPU clocks recover, and what does the collective cost".

## 2. What a training step needs from the model

**Time is a critical path, not a layer.** A step is a graph: kernels in stream order on
each GPU, collectives tying GPUs together. Step time is the longest path; everything
off it has slack.

**Kernels overlap by design.** DDP issues a gradient bucket's all-reduce as soon as the
bucket is ready, beside the rest of the backward pass; FSDP does the same with
reduce-scatter and all-gather. That is scenario one (GEMM beside a collective) with
the launch order reversed: the collective arrives behind compute that is already
running, which the measurements say behaves differently (section 4).

**More kinds of kernel.** Besides the GEMM and the all-reduce: backward GEMMs whose
inner dimension is the batch, elementwise passes (activation, optimiser update:
memory-bound, the MoE study's "epilogue" profile), attention, reduce-scatter,
all-gather, send/recv for pipeline stages, all-to-all for expert parallelism.

**Power is per GPU, and may also be capped per node.** Each board has its own limiter.
A rack or PSU budget couples them: watts not drawn by a waiting GPU could be spent on
the critical one, if a node-level controller exists.

**Time scales.** Kernels: 0.1-10 ms. A step: 10 ms to seconds. Limiter: 20 ms. A clock
set through NVML: 38 ms. The governor: wakes in 0.25 s, parks after 2 s. So a clock per
GPU per step is always possible, a clock per phase only when phases are much longer
than 38 ms, and a clock per kernel needs an actuator that does not exist yet.

## 3. The model, in layers

| Layer | Content | Status |
|---|---|---|
| Chip | idle power, voltage curve, energy per MAC and per byte, limiter. Fitted once (`sim/calib/a100_qwen.json`), unchanged. Plus three numbers per board: four A100s under one load differ by up to 7%. | built |
| Kernel profile | occupancy (SMs held, SMs denied to other shared-memory classes), work, rate at a clock alone and beside another kernel, activity. `sim/kernels.py`: cuBLAS GEMM, NCCL all-reduce. | 2 of ~8 kinds |
| Co-run rules | per pair of kernels and per launch order: SMs lost, rate factor. Measured for GEMM x all-reduce (section 4). | built for one pair |
| Workload | per GPU, a host program: kernels issued on CUDA streams, with dependencies; collectives couple the GPUs. The same JSON runs on the GPUs (`overlap_exp/sweep_spans.py`, `mode: program`) and in the simulator (`node.from_ops`). | built |
| Engine | events: a host is free, a kernel reaches its stream, a wave of tiles ends, a collective ends a phase, a clock arrives, a limiter looks. One limiter per GPU. `sim/node.py`. | built |
| Controllers | fixed clock per GPU (built); `clock` op inside a program, with actuation delay (built); an offline search of two clock groups with the engine as predictor (`node_policy.py`, built and checked by measurement); online slack reclamation, node power budget, per-phase clocks (planned, section 6) | partly |

Two properties are worth keeping as the model grows.

*One program, two executors.* The workload description that the simulator replays is
the file the harness runs. A model error cannot hide in a difference between what was
simulated and what was measured.

*Trace-driven for real models.* Nobody will write a host program by hand for a
transformer. The route is to record one step of the real job (kernel name, launch
grid, stream, start and end, from CUPTI or the PyTorch profiler), map each kernel to a
profile, and replay it at other clocks and under other controllers. Kernels with no
profile get a two-term stand-in (a part that scales with the clock and a part that
does not), fitted from two clocks of the same trace.

## 4. What the measurements of 2026-10-08 add

(Tables: `RESULTS_NODE.md`.)

**Launch order decides what a collective costs.** A probe carrying the cuBLAS block's
shared memory was run while an all-reduce was resident:

- Collective launched on an idle GPU: a GEMM loses exactly 2 SMs per CTA (c = 1..48),
  the pairs with the lowest SM ids. The collective runs at full speed.
- Collective launched behind a running GEMM, as in training: the GEMM loses exactly 1
  SM per CTA, and the collective is 1.1-1.5 times slower for its whole length.

ncu says why: NCCL's block runs under a 100 KB shared-memory carveout, cuBLAS's under
164 KB, and the carveout is a property of an SM pair for as long as a block is
resident. Whichever kernel arrives first sets it. This is the mechanism behind the
"GEMM is 2.4x slower than the wave model says" anomaly of the overlap study, and it is
a property of the kernels' shared-memory classes, so it will recur for every pair of
kernels in a training step.

**A collective runs at its slowest participant's rate** (to within 10% of the way to
the mean): slowing three GPUs of four slows the all-reduce for all four.

**A waiting collective holds SMs but draws little.** A rank that issues its all-reduce
early keeps the kernel resident, with its `2c` SMs, until every rank has joined; while
it waits it draws a tenth of its CTAs' power.

**A GPU never quite idles inside a step.** Between collectives a few ms apart a board
draws 16-20 W above its idle power, at any clock; between GEMMs it does not.

**Boards differ**, by 5-7% under the same GEMM at the same clock.

**The payoff, measured.** An unbalanced DDP step (GPU 0 four times the others' batch,
gradient all-reduce beside the backward pass, 16 CTAs), against the same step with no
clock lock, where the stock governor leaves the three waiting GPUs at 1410 MHz:

| Clocks, GPU 0 / GPUs 1-3 | Step time | Node energy per step |
|---|---|---|
| 1410 / 900 MHz | +0.4% | -19.7% |
| 1200 / 900 MHz | +3.4% | -20.3% |
| 1200 / 1200 MHz (one clock for all) | +3.1% | -13.0% |

The engine, not fitted on these steps, predicts their time to 0.6% (median) and each
GPU's power to 1.5%. Asked to choose the clocks itself (`sim/node_policy.py`: least
energy within 1% of the unlocked step time), it picked 1410 / 900 MHz for this step and
1410 / 975 MHz for a 2:1 imbalance, predicted -20.4% and -25.7% of node energy, and
the measurement gave -19.7% and -25.5% for +0.9% and +0.5% of step time. One locked
clock for all four saves 10.5% and 13.6% on the same steps; on a balanced step neither
policy saves anything within that budget.

**The registered blind test of the multi-GPU half passed** (`OVERLAP_HELDOUT.md`): a
clock per GPU, an unbalanced step and four-layer training steps at shapes, message
sizes, CTA counts and clocks never measured before: time 0.7-2.0% (median), power
1.4-1.8%.

## 5. Where the per-GPU clock pays, and where it does not

It pays when a GPU is off the critical path for a time long compared with 38 ms, or
for the whole step:

- **Unbalanced shards.** Variable-length batches, expert-parallel MoE with skewed
  routing, a straggler. The light GPUs' slack is whole-step, so a static per-GPU clock
  recovers it. Measured and simulated in RESULTS_NODE.md.
- **Pipeline stages** of unequal cost: same structure.
- **Communication-bound phases.** Above ~16 CTAs the all-reduce's bandwidth moves
  little with the clock while the board's power moves a lot. A phase that is all
  collective can run at a lower clock at small cost, if it lasts long enough to switch.

It does not pay, on today's hardware, for imbalance that changes faster than the
actuator. Expert-parallel MoE is the example, and the routing traces of the MoE study
say how much is at stake. Experts sharded in order over the GPUs, prefill batches of
64 conversations (`data/routing_*.npz`):

| Model (experts) | GPUs | Heaviest GPU / mean, per layer: median, p90 | Time a GPU waits per layer, mean | Same GPU heaviest in |
|---|---|---|---|---|
| Qwen3-30B-A3B (128) | 4 | 1.14, 1.24 | 13% | 34% of layers |
| | 8 | 1.32, 1.52 | 25% | 21% |
| gpt-oss-20b (32) | 4 | 1.33, 1.59 | 25% | 32% |
| | 8 | 1.75, 2.22 | 43% | 18% |
| Granite 3.0 3B (40) | 4 | 1.21, 1.38 | 17% | 43% |
| OLMoE (64) | 4 | 1.13, 1.23 | 12% | 33% |
| Qwen3-Next (512) | 4 | 1.12, 1.21 | 11% | 37% |

Two things follow. The slack is there (a GPU waits 11-25% of every layer on four GPUs,
up to 43% on eight), but the heavy GPU changes from layer to layer, about as often as
chance: a clock per GPU per step recovers almost none of it, and a layer is 1-5 ms
against NVML's 38 ms. And most of it is a placement problem before it is a clock
problem: assigning each layer's experts to GPUs by their mean load brings the
heaviest-to-mean ratio down to 1.03-1.07. What remains for DVFS is the batch-to-batch
part, on an actuator that does not exist yet. The simulator can price both;
`node.clock(mhz, delay_ms)` takes the delay as a parameter.

## 6. What is not built yet

1. **Profiles.** Elementwise and optimiser kernels (reuse the epilogue fit),
   attention, reduce-scatter and all-gather (the all-reduce's tables with a different
   byte count per GPU: to be measured, the world-of-2 factor was 1.24, not the 1.5 a
   byte count predicts), send/recv, the MoE megakernel as a profile so both case
   studies run in one engine.
2. **Co-run rules for the other pairs.** The carveout rule predicts them from each
   kernel's shared memory per block; ncu reads that. To be checked pair by pair.
3. **Trace capture and replay** of a real DDP or FSDP step.
4. **The governor in the node engine.** For work with no idle gaps it asks for the top
   clock, which is what `locked=False` does today. Park and wake (2 s, 0.25 s) matter
   for bursty loads and are in `sim/control.py` for one GPU only.
5. **Controllers**: slack reclamation that uses the engine as its own predictor (the
   multi-GPU version of `tdp_bisection`), a node power budget, per-phase clocks.
6. **Thermal coupling** between neighbours, and a thermal state at all: the simulator
   charges a warm die's leakage from the first millisecond of a load.
7. **Collectives across nodes** (InfiniBand or Ethernet instead of NVLink).
