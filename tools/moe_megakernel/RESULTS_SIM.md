# The event-driven simulator: how far it is from the A100, and what it says a second V/f domain is worth

**Verdict. The simulator runs end to end and reproduces the one-domain A100 within 2-4%
on a single partition and within 1-3% on whole layers and on loads that change batch
size every few seconds; the energy of a run is within 1.1%. In a test registered before
its data existed, four of five criteria passed on both traces and the clock missed on
one. With every ingredient of a two-domain design added one at a time and every group
given its best configuration, a second rail changes energy at equal latency by between
-2.3 and +0.7 points for an efficient pinned kernel, and latency at the 400 W wall by
between -0.2% and +0.0%. That is inside the simulator's own error: it cannot be told
from zero. Boosting the domain that is still running when the other one finishes -- the
move an event loop exists for -- adds 0.0-0.1% at each policy's best split. vLLM's kernel
at one well-chosen clock is within one point of the best two-domain configuration in
every case tried.**

That is the same conclusion `RESULTS_B2.md` reached by static composition, now from a
model with the shared memory system, the power limiter and the governor in it, and with
its error against 5000 measured times stated.

Usage: `sim/README.md`. Figures: `figs/fig23d_sim_event_loop.png`,
`figs/fig24_sim_validation.png`, `figs/fig25_sim_trace2.png`,
`figs/fig25_sim_heldout1.png`, `figs/fig26_sim_ladder.png`,
`figs/fig27_sim_layer_events.png`, `figs/fig28_sim_load_step.png`. The same material as a
paper section: `paper/sec_simulator.md`.

## What an independent review changed (2026-10-01)

A reviewer with no part in building the simulator read the code, the data and this
document. It found nothing fatal and several things wrong. They are changed here:

| finding | what was done | where |
|---|---|---|
| The rate fit had not converged: windows next to a saturated DRAM were stopped early, the co-run fixed point had no single answer, two parameters had never moved | solver and fit repaired; nine parameters; the form of the model chosen on data it was not fitted to | section 2 |
| B3 was searched only on B4's whole-GPC grid, which credited the second rail with what a finer SM split gives for free | B3 may use any SM count (12-96 in steps of 4); B4's domains stay whole GPCs, and are also reported one size at a time | section 6 |
| The gap between B3* and B4* is largest at tight latency targets, which were not reported | targets of 1.00, 1.02, 1.05, 1.10 and 1.25 x B0's latency | section 6 |
| A 20 us offset between the two streams' launches had no support | removed: the pinned layers fit better without it (1.38% rms against 1.55%) | section 6 |
| B0 ran at 400.3-400.5 W while every other group was held to 400.0 W | the wall is the power B0 itself settles at | section 6 |
| B3* and B4* were called a bound | they are a what-if: nothing validates a pinned kernel with vLLM's costs | sections 5, 6 |
| The four traces called hold-outs had been looked at while the limiter was built | a test registered before its data existed | section 4, `HELDOUT.md` |
| Fitted rows were not marked; power errors were given only against board power, 55-77 W of which is idle; 113 short records were left out without a score | every row says whether it was fitted; power also against power above idle; the 113 scored | section 3, appendix A |
| The pinned-layer table showed five of six cases, and the check is a composition of pairs the training sweeps had run | all six cases, and the records split by whether the pair had been swept | section 3 |
| Which clock the per-SM delivery cost, the DRAM-L2 coupling, the loaded-die power and the epilogue follow was assumed and not swept; "never exceeds 4%" contradicted this document's own table | four more assumptions and the form of the model in the sensitivity table; the sentence withdrawn | section 8 |

Two of the reviewer's statements did not hold up. The calibration then in use did
regenerate, bit for bit; the reviewer's copy of the code carried a patch. And the DRAM
wait cost is not unidentified over a factor of eight: its error profile has a clear
minimum, and its standard deviation in the repaired fit is 7.6%.

---

## 1. What was collected (2026-10-01)

Sweeps on GPU 0, clocks locked per sweep and reset afterwards; no power cap changed.
Persistence mode is left on for GPU 0: a clock lock does not hold without it. The ncu
counters ran on GPU 1: instruction and byte counts do not depend on the clock.

| what | file | rows | used for |
|---|---|---|---|
| each partition alone on 16-108 SMs and beside its partner on six SM splits; whole layer on both kernels; 930-1410 MHz | `data/sm_corun_scan_qwen.csv` | 3816 | fit |
| the same for 8 other cases (other batch sizes, tile sizes, cold-expert counts), 990-1380 MHz | `data/sm_corun_holdout_qwen.csv` | 2130 | scored as hold-out, then fit |
| the first 8 cases again at 510-870 MHz | `data/sm_corun_lowclk_qwen.csv` | 1696 | scored as hold-out, then fit |
| the same for MoE layers 6, 24, 36 and 47 | `data/sm_corun_holdout_L24_qwen.csv` and `_L6_`, `_L36_`, `_L47_` | 4 x 642 | never fitted |
| the same for three other models, two clocks | `data/sm_corun_xmodel_olmoe.csv`, `data/sm_corun_xmodel_granite.csv`, `data/sm_corun_xmodel_qwen3next.csv` | 3 x 214 | never fitted |
| idle board power, 210-1410 MHz, die temperature recorded | `data/idle_sweep_qwen.csv` | 246 | fit (static power) |
| ncu counters for 100 kernels | `data/ncu_counters_qwen.csv` | 100 | checks the demand decomposition |
| the pinned layer end to end, 6 cases x 6 SM splits x 3 clocks | `data/pinned_layer_qwen.csv` | 216 | never fitted |
| the unpartitioned persistent kernel at 1-3 resident blocks per SM | `data/occupancy_qwen.csv` | 210 | never fitted; section 5 |
| a scripted load under the unlocked governor, five plans | `data/governor_trace_dynamics_segments_qwen.csv` and its `_replays`, `_nvml`; likewise `trace`, `trace2`, `forward`, `forward2` | 14 runs | governor and limiter built against them: seen, not a test |
| the `trace2` plan again with the clock locked at 1050 and 1290 MHz | `data/governor_trace_lock1050_segments_qwen.csv`, `data/governor_trace_lock1290_segments_qwen.csv` and theirs | 4 runs | never fitted: a policy other than the governor, measured |
| two more plans, drawn from a random seed and registered before they were run | `data/governor_trace_heldout1_segments_qwen.csv`, `data/governor_trace_heldout2_segments_qwen.csv` and theirs; `data/heldout_plan.json` | 6 runs | the registered test (`HELDOUT.md`) |

## 2. What the fits found

Nine numbers describe how fast the persistent kernel runs on any SM count, at any clock
from 510 to 1410 MHz, alone or beside another partition (2964 measured times, rms
3.10%); four energies and one voltage curve describe board power (2261 records, rms 2.28%
of board power). Four things in them are findings, not conveniences:

**One mechanism gives both saturation and interference.** A partition's SMs wait on two
resources every SM shares: DRAM (1555 GB/s) and L2 delivery (14.8 TB/s at 1290 MHz).
Its own traffic fills them as it is given more SMs; a neighbour's traffic fills them the
same way. Fitted on partitions running alone it under-predicts how much two partitions
slow each other; one more term, how hard a busy DRAM leans on L2 delivery, is what a
neighbour adds, and with it solo and co-run errors are the same size (3.20% and 2.98%).

**Congestion has to be computed per window of tiles, not per pass.** A whole layer on 108
SMs alternates between stretches of cold experts, where every SM pulls a distinct weight
panel from DRAM, and stretches inside one hot expert that touch DRAM not at all. One
average utilisation per pass under-predicted those layers by 6-13%.

**DRAM traffic costs the same energy at every clock; everything else follows V^2.**
e_dram is 77.6 mJ/GB; refitted clock by clock with the rest held it stays within
74.8-81.4 mJ/GB from 510 to 1410 MHz with no trend, while the SM-side energies rise 1.83x.
The voltage curve g(f) came out flat within 3% from 510 to 990 MHz (0.97-1.00) and then
1.043 at 1050, 1.424 at 1290, 1.833 at 1410: the voltage floor and its knee, found a
second way.

**Padding is cheap in energy and not in time.** A row-block is BLOCK_M rows whether or
not the expert has that many tokens; the missing rows are zeros. They take as long as
real rows, and a MAC on them costs 35% of a real one (0.598 against 1.695 J/TMAC). With
one price per MAC the model over-predicted a 1024-token batch on 128-row tiles (56%
padding) by 9-13%. So time follows padded rows and energy follows real pairs.

| parameter | value | meaning |
|---|---|---|
| kP at 1290 MHz | 22.10 us SM / MB | weight-panel delivery per SM; 7% of it does not scale with the clock |
| kR at 1290 MHz | 0.903 ms SM / GMAC | tensor work per SM: 2.21 TFLOP/s per SM |
| L0 at 1290 MHz | 1.07 us SM / MB | DRAM wait per SM, DRAM idle |
| BW_D | 1555 GB/s | DRAM bandwidth; pinned at the A100-40GB's nominal |
| BW_L at 1290 MHz | 14.8 TB/s | L2 delivery capacity; grows as clock^0.44 |
| xi at 1290 MHz | 0.595 | how hard a busy DRAM leans on L2 delivery; grows as clock^1.49 |
| q | 1.72 | ... and how sharply that grows as DRAM fills: as (DRAM utilisation)^q |
| e_mac | 1.695 J/TMAC | tensor work on real pairs, at the voltage floor |
| e_pad | 0.598 J/TMAC | tensor work on padding rows: 35% of real |
| e_panel | 19.5 mJ/GB | weight panels delivered to SMs, at the floor |
| e_dram | 77.6 mJ/GB | DRAM traffic; the same at every clock |
| g(1050), g(1290), g(1410) | 1.043, 1.424, 1.833 | supply voltage squared, relative to the floor |
| P_idle(510), (1050), (1290), (1410) | 56.7, 60.2, 69.3, 76.9 W | board, nothing running |
| P_shared | 54.4 W | the part of idle power no clock changes |
| epilogue | 45 us + pairs x (4.88 + 7703/f) ns | activation and top-k sum |

One standard deviation of the 9 fitted rate parameters: aP 1.2%, bP 18.7%, aR 0.2%, aL 7.6%, bw_l 3.1%, gamma_l 0.07, xi 1.9%, gamma_x 0.08, q 3.3%.

vLLM's queue kernel shares the chip's capacities and has its own per-SM costs (4 numbers
and a window, 197 end-to-end layer times from 64 to 7744 tokens and 510 to 1410 MHz,
rms 1.85%).

### The fit had not converged, and the form of the model had not been tested

The review found both. Until 2026-10-01 the window solve stopped after 14 Newton steps.
Next to a saturated DRAM a window needs 30-40, so those windows were up to a percent off
and a co-running partner's time was a jagged function of them; the fixed point between
the two partitions then had no single answer (it moved by up to 0.5% from one evaluation
to the next), and the least-squares fit stopped wherever the noise let it. Two of its
parameters had never moved from their starting values, because the finite-difference
step was a thousandth of the parameter's logarithm and theirs was zero. The calibration
then in use regenerated bit for bit, but it was not an optimum.

Repaired: every window is iterated until it has converged, the co-run fixed point is
solved by Gauss-Seidel sweeps from a cold start, and the step no longer vanishes. The
residual is now a deterministic function of the parameters, and four different starting
points reach the same optimum.

With a fit that converges, which parameters to fit became a question with an answer.
Three forms were fitted on the first sweep alone and scored on the two collected after
it:

| form | pinned | `holdout`: n | rms | bias | worst | `lowclk`: n | rms | bias | worst |
|---|---|---|---|---|---|---|---|---|---|
| shipped (in use) | bR = 0, bL = 0, bw_d = 1555 | 994 | 3.79% | +0.72% | +15.8% | 800 | 2.68% | -0.67% | -7.7% |
| proportional | bR = 0, bL = 0, bw_d = 1555, gamma_l = 1, gamma_x = 1 | 994 | 3.82% | +0.79% | +15.5% | 800 | 3.00% | -1.18% | -9.7% |
| linear | bR = 0, bL = 0, q = 1, gamma_l = 1, gamma_x = 1 | 994 | 4.18% | +1.00% | +17.9% | 800 | 3.10% | -1.25% | -8.7% |

The form in use frees how the coupling grows with DRAM utilisation (q = 1.72) and how the
two shared capacities move with the clock (L2 delivery as clock^0.44, the coupling as
clock^1.49), and pins DRAM bandwidth at the A100-40GB's nominal 1555 GB/s. With q and the
bandwidth both free the two trade against each other and the bandwidth runs off to
25-34 TB/s. Checked a second way, on data none of the three was fitted to: the form in
use has the lowest time error of the three on 22 of the 34 never-fitted rows of section
3 (the others on 7 and 8, ties counted for both), and on all five moving-load traces of
section 4 (the held-out pair took no part in this).

What moved, against the numbers this document carried before the repair:

| | before | now |
|---|---|---|
| rate fit, 2964 times | 3.30% | 3.10% |
| second sweep, predicted from the first | 4.20% | 3.79% |
| third sweep (420 MHz below the first's range), same | 3.05% | 2.68% |
| queue kernel, 197 whole layers | 1.59% | 1.85% |
| pinned layer end to end | 1.55% | 1.38% |
| moving load, latency per segment, five traces | 0.91-2.42% | 0.96-2.73% |
| ... energy per run | -0.1% to -1.4% | -1.1% to +0.9% |
| registered test, latency (`HELDOUT.md`) | 1.58%, 0.90% (first shot) | 1.56%, 0.68% (second shot) |

The persistent kernel is predicted better on its own sweeps, on the two later sweeps, on
the four other MoE layers and on the pinned layer; the earlier campaigns and the other
models moved by up to 0.7 points either way. The queue kernel fits its own sweeps
slightly worse under the new shared capacities, and the three mixed-batch traces
are 0.3-0.7 points worse in latency: small batches at high clocks, which were 2-4% fast
in the simulator, are now 4% fast (section 10).

## 3. How far it is from every measurement

`python3 -m sim.validate`; `figs/fig24_sim_validation.png`. Time errors are ln(simulated /
measured). Power errors are given twice: as a share of board power, and as a share of
board power above idle at that clock. Idle power is a measured table, 55-77 W of every
reading, that the model cannot get wrong; the second number is the error of what the
model predicts. `fit` says whether the calibration used the records. Every row, with
bias and worst case, is in appendix A.

| rung | what | fit | time: n | rms | power: n | rms | rms above idle |
|---|---|---|---|---|---|---|---|
| 0 | demand decomposition vs ncu, 90 kernels | | | tiles and tensor instructions exact; DRAM bytes median -0.11%, worst -17.27% | | | |
| 1 | three training sweeps, 510-1410 MHz, solo | yes | 1558 | 2.71-3.92% | 1558 | 1.54-2.86% | 3.35-4.75% |
| 1 | ... co-running, six SM splits | yes | 1406 | 2.12-3.28% | 703 | 2.30-3.36% | 3.41-4.62% |
| 1 | second sweep, scored by a fit on the first alone | no | 994 | 3.79% | | | |
| 1 | third sweep (420 MHz below the first's range), same | no | 800 | 2.68% | | | |
| 2 | earlier campaigns, 9 groups | no | 537 | 0.48-6.10% | 489 | 1.41-5.51% | 2.49-10.39% |
| 2 | one earlier sweep that shaped the voltage curve and set the launch gap | partly | 32 | 0.83% | 32 | 2.95% | 4.36% |
| 3 | four other MoE layers (6, 24, 36, 47), solo | no | 616 | 3.41-4.37% | 616 | 2.70-4.99% | 4.43-9.35% |
| 3 | four other MoE layers, co-run | no | 496 | 3.52-5.76% | 248 | 2.40-3.30% | 3.25-4.68% |
| 3 | **the pinned layer end to end**, 6 cases x 6 splits x 3 clocks | no | 108 | **1.38%** | 108 | 2.32% | 3.48% |
| 3b | OLMoE (64 experts, I = 1024): solo / co-run / whole layers | no | 214 | 2.74 / 2.86 / 1.92% | 166 | 2.61 / 2.96 / 2.70% | 4.34 / 3.92 / 3.61% |
| 3b | Qwen3-Next (512 experts, top-10, I = 512): the same | no | 214 | 2.72 / 3.43 / 2.36% | 166 | 2.64 / 4.60 / 3.34% | 4.31 / 6.44 / 4.56% |
| 3b | Granite (40 experts, K = 1536): the same | no | 178 | 8.08 / 5.86 / 5.10% | 142 | 6.12 / 6.11 / 7.37% | 9.86 / 8.47 / 10.14% |
| 3b | Granite, a partition whose weights fit in L2 | no | 36 | 4.43% | 24 | 19.23% | 36.64% |
| 4 | whole layers of the training sweeps, both kernels | yes | 174 | 2.00-2.18% | 174 | 1.84-2.16% | 2.66-2.98% |
| 4 | queue kernel, 64-7744 tokens, 510-1410 MHz | yes | 141 | 1.79% | 141 | 1.68% | 2.24% |
| 4 | both kernels on the four other MoE layers | no | 65 | 1.17-1.24% | 65 | 1.92-2.56% | 2.51-3.55% |
| 4 | both kernels, tile sizes and batches of the occupancy sweep | no | 40 | 2.44-2.47% | 40 | 2.23-2.70% | 3.00-3.50% |
| 4 | queue kernel, 512-12288 tokens, 1290 MHz | no | 8 | 2.10% | 8 | 2.04% | 2.61% |

The three other models were never fitted and run with Qwen's kernel configuration. OLMoE
and Qwen3-Next, which share its hidden size (K = 2048), come out at the model's own
error. Granite does not. It is slower than simulated at every SM count from 16 to 108
(by 4-10% on average) and at both clocks (8.1% at 1050 MHz, 7.3% at 1290 MHz), so what
is missing is a per-SM cost and not congestion. Its energy per pass is right: with the
measured time in place of the simulated one, board power is within 2.00% rms, so the
6-7% power error is the time error seen through the activity rates.

One explanation was tested. A share of the tensor cost may go per output element instead
of per MAC. Within one model the two are proportional, so Qwen's sweeps cannot separate
them; across models they are not -- Granite has 0.70 of Qwen's MACs per output element.
A 20% share takes the three models' solo times from 5.02% to 3.00% rms together (Granite
8.08% to 3.56%, OLMoE 2.74% to 2.11%) and Qwen3-Next the wrong way, 2.72% to 3.20%. It
is not in the calibration: one number set on three models is a fit, not a finding.

The last Granite row is something inference never does. Of Granite's 8 coldest experts
the 6 that receive tokens hold 28.3 MB of weights, less than the 40 MB L2, and a sweep
repeats its pass: after the first one nothing comes from DRAM. The simulator charges
DRAM on every pass, as one pass per layer would, and is 18% over in power there.

**The pinned layer** is the configuration a two-domain design executes -- both partitions
on their own SM sets, barriers between the GEMMs, activation and sum -- and the engine
was not fitted to it. It is mostly a test of composition. For 94 of its 108 records the
training sweeps had run the same two partitions side by side on the same SM split at the
same clock; what is new there is the barriers, the epilogue and the event loop that put
a layer together, not the partitions. Those 94 are within 1.46% rms; the 14 whose pair
was never swept, 0.70%. Across SM splits the layer's time varies fourfold; at 1290 MHz,
measured / simulated ms, every case:

| batch, tile, cold experts | 16+92 | 32+76 | 48+60 | 54+54 | 64+44 | 80+28 | rms, 3 clocks |
|---|---|---|---|---|---|---|---|
| prefill, BM128, 16 | 5.236 / 5.260 | 6.149 / 6.165 | 7.589 / 7.553 | 8.296 / 8.285 | 9.999 / 9.950 | 15.257 / 15.082 | 0.57% |
| prefill, BM128, 64 | 9.507 / 9.333 | 5.152 / 5.149 | 5.625 / 5.618 | 6.118 / 6.135 | 7.344 / 7.309 | 11.038 / 10.940 | 0.87% |
| decode 1024, BM64, 64 | 2.347 / 2.337 | 1.423 / 1.409 | 1.568 / 1.571 | 1.706 / 1.695 | 2.008 / 1.975 | 2.887 / 2.850 | 0.98% |
| decode 2048, BM16, 64 | 3.789 / 3.802 | 4.229 / 4.401 | 5.351 / 5.312 | 5.605 / 5.792 | 6.824 / 6.880 | 10.089 / 10.234 | 2.38% |
| decode 2048, BM64, 64 | 2.597 / 2.594 | 2.139 / 2.153 | 2.595 / 2.587 | 2.823 / 2.819 | 3.379 / 3.333 | 5.060 / 4.929 | 1.27% |
| decode 4096, BM64, 64 | 3.544 / 3.554 | 3.812 / 3.842 | 4.705 / 4.665 | 5.122 / 5.097 | 6.188 / 6.084 | 9.382 / 9.124 | 1.46% |

The worst case is the one on 16-row tiles (bias +1.67%), where the model, linear in rows
per tile, is at the edge of what it was fitted on.

## 4. A load that moves

`governor_trace.py` ran scripted sequences of batches under the unlocked governor, timing
every graph replay and logging every NVML reading; `python3 -m sim.validate_trace` replays
the same script through the simulator (`figs/fig25_sim_trace2.png`). The rate and power
models never saw a moving load. The governor and limiter have eight numbers, set by hand
against the `dynamics` plan.

**These five traces are not a test.** They were called hold-outs because the eight
numbers were set against `dynamics` only. That is true of the numbers and not of the
model: the limiter was rebuilt as a 20 ms loop after the replays of `trace2` were looked
at, and all five were replayed many times while the engine was written. The test is the
registered one below.

Per segment (mean latency, power and clock) and per run (energy). The clock is scored
twice: as registered, with the measured side averaged from 150 ms into each segment
(NVML lags) and the simulated side from its start, and over the same window on both
sides:

| trace | segments x runs | latency rms | power rms | clock rms, as registered | clock rms, same window | energy of each run |
|---|---|---|---|---|---|---|
| `dynamics` | 7 x 3 | 2.73% | 0.91% | 9.4 MHz | 6.6 MHz | -0.7%, -0.9%, -1.1% |
| `trace` | 11 x 3 | 2.22% | 1.28% | 12.0 MHz | 7.1 MHz | -0.5%, -0.3%, -0.4% |
| `trace2` | 11 x 2 | 2.72% | 1.67% | 14.8 MHz | 11.6 MHz | -0.4%, -0.7% |
| `forward` | 7 x 3 | 1.01% | 0.67% | 11.1 MHz | 5.9 MHz | +0.1%, -0.2%, -0.2% |
| `forward2` | 8 x 3 | 0.96% | 1.54% | 8.2 MHz | 8.8 MHz | +0.9%, +0.6%, +0.4% |
| `trace2` plan, clock locked at 1050 MHz | 11 x 2 | 1.23% | 3.04% |  |  | +2.3%, +2.0% |
| `trace2` plan, clock locked at 1290 MHz | 11 x 2 | 2.59% | 3.39% |  |  | +1.7%, +0.9% |

Per graph replay, over the first 400 ms after every load step, replay k of the
simulation against replay k of the measurement with the segment's steady-state error
taken out (`figs/fig28_sim_load_step.png`). A replay is 4 layers, 4-16 ms, and a layer's
time follows the clock, so this sees the clock move where NVML's 100 ms readings cannot.
The score subtracts each segment's own steady error, so it says how well the SHAPE of a
transient is followed, not its level:

| trace | another batch size arrives | a batch arrives after a short idle gap | ... after a long one (clock parked) | highest 100 ms reading, simulated / measured |
|---|---|---|---|---|
| `dynamics` | 0.63% |  | 2.40% | 432 / 470 W |
| `trace` | 0.76% | 0.63% | 4.64% | 434 / 448 W |
| `trace2` | 0.56% | 0.70% | 6.55% | 421 / 412 W |
| `forward` | 0.66% | 0.08% | 5.04% | 422 / 422 W |
| `forward2` | 1.04% | 1.46% | 6.33% | 433 / 416 W |
| clock locked at 1050 MHz | 0.08% | 0.16% | 0.31% | 258 / 258 W |
| clock locked at 1290 MHz | 0.09% | 0.19% | 0.28% | 378 / 378 W |

The locked rows are the floor of this score: nothing moves, 0.1-0.3%. The wake from the
parked clock is the worst because the chip itself is not repeatable there: over 16 wakes
the clock jumped between 0.13 and 0.35 s after the load arrived (median 0.25 s), and the
simulator always wakes at 0.25 s.

### The registered test

`HELDOUT.md`. The simulator and its parameters were frozen and archived, two 33-second
plans were drawn from a random seed -- batch sizes, prompt sets, a layer and decode
routing that nothing had used -- and pass criteria were written down, all before either
plan was run. Each was then replayed once through the frozen simulator
(`figs/fig25_sim_heldout1.png`, `figs/fig25_sim_heldout2.png`).

| quantity | pass if | `heldout1` (layer 17) | `heldout2` (48 layers) |
|---|---|---|---|
| latency per layer, rms over segments | <= 3% | 1.58% -- pass | 0.90% -- pass |
| board power, rms over segments | <= 2% | 1.60% -- pass | 0.61% -- pass |
| SM clock, rms over segments | <= 20 MHz | **21.6 MHz -- miss** | 9.0 MHz -- pass |
| energy of each whole run | within +-2% | -0.6%, -0.8%, -1.7% -- pass | -0.7%, -1.0%, -0.5% -- pass |
| first 400 ms after a batch-size change, per replay | <= 1.5% | 0.94% -- pass | 0.86% |

Four of five pass on both; the clock misses on one. The miss is the six segments that
begin with a wake from the parked clock, where the chip woke 0.11-0.33 s after the load.

The calibration in this document is not the frozen one: it was repaired afterwards
(section 2). Replayed again with it -- a second shot, not a registered test -- the two
traces give 1.56% and 0.68% in latency, 1.76% and 0.53% in power, 19.4 and 8.5 MHz in
clock, and run energies of +0.0%, -0.3%, -1.1% and +0.0%, -0.3%, +0.2%. Both columns are
in `HELDOUT.md`; the first shot is the result.

### Whole forward passes, and a policy change

`forward` and `forward2` are whole forward passes: every replay runs all 48 MoE layers of
the model, each with its own traced routing, 47 of them layers the simulator was never
calibrated on. `forward2` uses five other prompt sets, 5427 to 10779 tokens, and decode
batches of 768, 1536 and 3072 tokens: sizes no sweep ran.

The locked-clock rows are the one policy change that CAN be made on the real chip.
Against the governor, on the same load, a fixed 1050 MHz clock was measured at +15.9%
busy time and -26.0% busy energy; the simulator says +17.3% and -23.9%. A fixed 1290 MHz
clock: measured +1.7% and -9.5%, simulated +1.8% and -7.9%.

What the governor and the limiter do, read off the replays (36 load steps of 40 MHz or
more, and 29 wakes) and reproduced by the simulator:

- **Parked.** An idle chip sits at 1095 MHz. About 0.25 s into a load the clock jumps
  once, to within 45 MHz of where it then settles (28 of 29 wakes; one landed 60 MHz
  short). It does not visit the top clock on the way.
- **A heavier batch** (16 steps). It runs at the old clock for 9-26 ms (median 15). By
  43-153 ms (median 58) the clock is at its new level, 45-117 MHz lower, having
  undershot it by up to 33 MHz. For those tens of ms the board is far above TDP -- the
  power model says 500 W for a prefill batch at 1410 MHz -- and NVML, averaging over
  100 ms, shows at most 412-470 W.
- **A lighter batch** (15 steps). The clock starts back up after 9-33 ms (median 23) and
  is at its new level, 43-103 MHz higher, by 57-116 ms (median 77).
- **After a short idle gap** (5 steps) the limiter starts later: the heavy batch runs at
  the top clock for 43-77 ms (median 45). The simulator does not make that distinction
  and starts after 10-30 ms as for any other step.
- **Held.** After that it walks 15 MHz at a time around the wall, 4-8 W either side of
  400 W. 2 s after the load stops it parks again.

The limiter's time constant is therefore about 20 ms, not NVML's 100. The simulator
reads the board every 20 ms, moves the clock ceiling by 0.45 MHz per watt of excess and
reaches the chip 10 ms later: three numbers, read off the `dynamics` replays.

Where it settles, for a held batch (never fitted):

| batch | measured MHz | W | ms | simulated MHz | W | ms | ms, simulated vs measured |
|---|---|---|---|---|---|---|---|
| 512 | 1410 | 362.8 | 1.0060 | 1410 | 373.5 | 0.9662 | -4.0% |
| 1024 | 1410 | 397.7 | 1.1565 | 1400 | 400.1 | 1.1073 | -4.3% |
| 2048 | 1340 | 396.8 | 1.4703 | 1352 | 399.7 | 1.4795 | +0.6% |
| 4096 | 1320 | 399.5 | 2.2409 | 1327 | 400.3 | 2.2633 | +1.0% |
| 6144 | 1320 | 399.8 | 3.1073 | 1322 | 399.7 | 3.1308 | +0.8% |
| 8192 | 1320 | 399.1 | 3.9623 | 1318 | 399.8 | 3.9658 | +0.1% |
| 10240 | 1320 | 399.2 | 4.8396 | 1318 | 400.2 | 4.8363 | -0.1% |
| 12288 | 1317 | 397.2 | 5.6860 | 1315 | 399.9 | 5.6628 | -0.4% |

The two smallest batches are 4% fast in the simulator; from 2048 tokens up it is within
1%.

## 5. What the persistent kernel costs is occupancy

The simulator's persistent kernel is the one that was measured with SM sets it could
verify: one resident block per SM. `decode_wall.py occupancy` put the unpartitioned
kernel beside vLLM's at 1-3 resident blocks per SM (`data/occupancy_qwen.csv`, 1290 MHz,
whole layer end to end, time against the queue kernel):

| tokens | queue ms | BM64, 1 block | BM64, 2 blocks | BM128, 1 block | BM128, 2 blocks |
|---|---|---|---|---|---|
| 512 | 1.023 | +6.0% | -2.1% | +11.2% | +1.2% |
| 1024 | 1.177 | +18.0% | +1.1% | +12.7% | +2.1% |
| 2048 | 1.502 | +31.0% | +3.0% | +20.3% | +3.3% |
| 4096 | 2.278 | +44.4% | +11.1% | +24.8% | +5.2% |
| 7744 (prefill) | 3.817 | +50.7% | +15.5% | +24.4% | +4.7% |

With two blocks per SM and 128-row tiles the persistent kernel is within 1-5% of the
queue kernel at every batch size; with one it is 11-25% behind. One block per SM is the
form ncu verified as one block on one SM; whether two concurrent launches of 2n blocks
land on disjoint SM sets was never checked, so the partitioned form was measured, and is
simulated, at one. The rows marked * below replace its per-SM costs with vLLM's: a pinned
kernel as efficient as the queue kernel. Nobody has built it and nothing validates it:
those rows are a what-if, not a bound.

## 6. The control groups

`python3 -m sim.study --cells N`; `figs/fig26_sim_ladder.png`. Each row adds one thing,
and each group gets the best configuration it is allowed. Which experts are cold is
software, for B3 and B4 alike: 8 to 120 of 128, in steps of 8. How many SMs the cold set
gets is not alike. B3 pins partitions on today's chip and may use any SM count (12 to 96
in steps of 4). B4's cold set is a V/f domain, which is silicon: whole GPCs (16 to 80).
The wall is 400 W or the power B0 itself settles at, whichever is higher (400.0-400.1 W),
so that no group is held to a stricter limit than the baseline. The two streams of a
pinned layer start together: the 20 us offset an earlier version gave the second one is
gone, because the pinned layers measured end to end fit better without it (1.38% rms
against 1.55%). Two V/f domains given the same clock reproduce the one-domain pinned
kernel to rounding (`sim/test_sim.py`).

Time and energy per layer against B0, the stock governor:

| objective | | decode1024 | decode2048 | decode4096 | prefill |
|---|---|---|---|---|---|
| fastest under TDP (time) | B1 queue kernel, one clock | +0.1% | +0.1% | +0.3% | +0.8% |
| fastest under TDP (time) | B2 persistent kernel, one clock | +16.9% | +19.3% | +19.9% | +20.3% |
| fastest under TDP (time) | B3 pinned, one clock | +16.8% | +19.3% | +19.3% | +20.8% |
| fastest under TDP (time) | B4 pinned, a clock per domain | +17.2% | +22.2% | +19.5% | +19.8% |
| fastest under TDP (time) | B3* pinned, one clock (vLLM per-SM costs) | +0.1% | +0.4% | +0.3% | +0.5% |
| fastest under TDP (time) | B4* pinned, a clock per domain (vLLM per-SM costs) | +0.2% | +0.3% | +0.3% | +0.3% |
| least energy, <= 1.00 x B0 latency (energy) | B1 queue kernel, one clock | cannot meet it | cannot meet it | cannot meet it | cannot meet it |
| least energy, <= 1.00 x B0 latency (energy) | B2 persistent kernel, one clock | cannot meet it | cannot meet it | cannot meet it | cannot meet it |
| least energy, <= 1.00 x B0 latency (energy) | B3 pinned, one clock | cannot meet it | cannot meet it | cannot meet it | cannot meet it |
| least energy, <= 1.00 x B0 latency (energy) | B4 pinned, a clock per domain | cannot meet it | cannot meet it | cannot meet it | cannot meet it |
| least energy, <= 1.00 x B0 latency (energy) | B3* pinned, one clock (vLLM per-SM costs) | cannot meet it | cannot meet it | cannot meet it | cannot meet it |
| least energy, <= 1.00 x B0 latency (energy) | B4* pinned, a clock per domain (vLLM per-SM costs) | cannot meet it | cannot meet it | cannot meet it | cannot meet it |
| least energy, <= 1.02 x B0 latency (energy) | B1 queue kernel, one clock | -11.9% | -7.0% | -3.0% | -3.7% |
| least energy, <= 1.02 x B0 latency (energy) | B2 persistent kernel, one clock | cannot meet it | cannot meet it | cannot meet it | cannot meet it |
| least energy, <= 1.02 x B0 latency (energy) | B3 pinned, one clock | cannot meet it | cannot meet it | cannot meet it | cannot meet it |
| least energy, <= 1.02 x B0 latency (energy) | B4 pinned, a clock per domain | cannot meet it | cannot meet it | cannot meet it | cannot meet it |
| least energy, <= 1.02 x B0 latency (energy) | B3* pinned, one clock (vLLM per-SM costs) | -11.9% | -5.3% | -3.0% | -1.7% |
| least energy, <= 1.02 x B0 latency (energy) | B4* pinned, a clock per domain (vLLM per-SM costs) | -12.4% | -6.6% | -3.9% | -3.9% |
| least energy, <= 1.05 x B0 latency (energy) | B1 queue kernel, one clock | -20.6% | -13.1% | -9.7% | -8.9% |
| least energy, <= 1.05 x B0 latency (energy) | B2 persistent kernel, one clock | cannot meet it | cannot meet it | cannot meet it | cannot meet it |
| least energy, <= 1.05 x B0 latency (energy) | B3 pinned, one clock | cannot meet it | cannot meet it | cannot meet it | cannot meet it |
| least energy, <= 1.05 x B0 latency (energy) | B4 pinned, a clock per domain | cannot meet it | cannot meet it | cannot meet it | cannot meet it |
| least energy, <= 1.05 x B0 latency (energy) | B3* pinned, one clock (vLLM per-SM costs) | -20.5% | -12.0% | -9.7% | -7.7% |
| least energy, <= 1.05 x B0 latency (energy) | B4* pinned, a clock per domain (vLLM per-SM costs) | -20.0% | -13.2% | -9.7% | -9.0% |
| least energy, <= 1.10 x B0 latency (energy) | B1 queue kernel, one clock | -26.7% | -19.0% | -16.4% | -14.7% |
| least energy, <= 1.10 x B0 latency (energy) | B2 persistent kernel, one clock | cannot meet it | cannot meet it | cannot meet it | cannot meet it |
| least energy, <= 1.10 x B0 latency (energy) | B3 pinned, one clock | cannot meet it | cannot meet it | cannot meet it | cannot meet it |
| least energy, <= 1.10 x B0 latency (energy) | B4 pinned, a clock per domain | cannot meet it | cannot meet it | cannot meet it | cannot meet it |
| least energy, <= 1.10 x B0 latency (energy) | B3* pinned, one clock (vLLM per-SM costs) | -26.7% | -18.9% | -15.1% | -14.6% |
| least energy, <= 1.10 x B0 latency (energy) | B4* pinned, a clock per domain (vLLM per-SM costs) | -26.6% | -19.7% | -16.2% | -15.2% |
| least energy, <= 1.25 x B0 latency (energy) | B1 queue kernel, one clock | -29.4% | -25.7% | -23.5% | -23.2% |
| least energy, <= 1.25 x B0 latency (energy) | B2 persistent kernel, one clock | -15.7% | -0.7% | +3.9% | +5.3% |
| least energy, <= 1.25 x B0 latency (energy) | B3 pinned, one clock | -15.7% | -0.7% | +3.8% | +8.4% |
| least energy, <= 1.25 x B0 latency (energy) | B4 pinned, a clock per domain | -15.8% | -2.5% | +3.8% | +5.5% |
| least energy, <= 1.25 x B0 latency (energy) | B3* pinned, one clock (vLLM per-SM costs) | -29.4% | -25.6% | -23.5% | -23.1% |
| least energy, <= 1.25 x B0 latency (energy) | B4* pinned, a clock per domain (vLLM per-SM costs) | -29.4% | -25.6% | -23.5% | -23.1% |

Nothing meets B0's own latency. B0 sits at the wall between two 15 MHz clock steps, and
every other group has to stay under it.

Reading it, one ingredient at a time:

- **A better clock than the governor's (B1 - B0)** is the large term: 3-12% of energy for
  2% of latency, 9-21% for 5%, 15-27% for 10%, 23-29% for 25% -- with no new kernel and no
  new hardware. Nothing below beats it by as much as one point.
- **The kernel (B2 - B1)** costs 17-20% of latency at one block per SM (section 5), which
  is why B2, B3 and B4 cannot meet any target tighter than 1.25 x.
- **Pinning (B3 - B2)** changes latency by between -0.6% and +0.5%. With the SM split free
  it costs nothing.
- **The second rail (B4 - B3, B4* - B3*)** is the table below.
- **Against the queue kernel (B4* - B1).** The best two-domain configuration, with a
  pinned kernel as efficient as vLLM's, is within 1.0 point of vLLM's kernel at one clock
  in energy at every target (-1.0 to +0.6) and within 0.5% in latency at the wall.
  Neither is consistently ahead.

### The second rail alone

B4 against B3, and B4* against B3*, in points of the one-clock group's own time or
energy:

| objective | | decode1024 | decode2048 | decode4096 | prefill |
|---|---|---|---|---|---|
| fastest under TDP (time) | B4 - B3 | +0.34% | +2.42% | +0.18% | -0.85% |
| fastest under TDP (time) | B4* - B3* | +0.04% | -0.08% | -0.07% | -0.21% |
| least energy, <= 1.00 x B0 latency (energy) | B4 - B3 | - | - | - | - |
| least energy, <= 1.00 x B0 latency (energy) | B4* - B3* | - | - | - | - |
| least energy, <= 1.02 x B0 latency (energy) | B4 - B3 | - | - | - | - |
| least energy, <= 1.02 x B0 latency (energy) | B4* - B3* | -0.52% | -1.35% | -0.99% | -2.30% |
| least energy, <= 1.05 x B0 latency (energy) | B4 - B3 | - | - | - | - |
| least energy, <= 1.05 x B0 latency (energy) | B4* - B3* | +0.70% | -1.33% | +0.00% | -1.40% |
| least energy, <= 1.10 x B0 latency (energy) | B4 - B3 | - | - | - | - |
| least energy, <= 1.10 x B0 latency (energy) | B4* - B3* | +0.04% | -0.88% | -1.28% | -0.73% |
| least energy, <= 1.25 x B0 latency (energy) | B4 - B3 | -0.07% | -1.81% | +0.04% | -2.66% |
| least energy, <= 1.25 x B0 latency (energy) | B4* - B3* | +0.03% | -0.00% | +0.00% | -0.02% |

With vLLM's per-SM costs the second rail is worth between -2.3 and +0.7 points of energy
at equal latency (median -0.6 over the sixteen cases) and between -0.2% and +0.0% of
latency at the wall. It is largest at the tightest target and gone at 1.25 x. With the
kernel as measured it is worth 0.0 to 2.7 points at 1.25 x, the only target that kernel
can meet, and at the wall the two-domain chip is SLOWER in three workloads of four (by
0.2% to 2.4%): its cold domain must be a whole number of GPCs and B3's SM split need not
be.

Every one of these numbers is smaller than the simulator's error on a single
configuration (2-4%). B3 and B4 share partitions and models, so their difference is
better determined than either alone, but an interval that straddles zero inside the
model's error is not a small gain. It is no detectable effect.

### One domain size for every workload

B4 above picks its domain size per workload. Silicon cannot. With the cold domain fixed
at one size for all four workloads, energy against B1 in points of B0's energy (B4* with
vLLM's per-SM costs; every row of every group is in appendix B):

| cold domain | <= 1.02 x: decode 1024 / 2048 / 4096 / prefill | <= 1.10 x: the same |
|---|---|---|
| 16 SMs | +11.1 / +1.3 / -1.0 / +4.1 | +1.4 / -0.4 / +0.2 / +0.1 |
| 32 SMs | +4.3 / +0.5 / +1.1 / +3.9 | +1.4 / -0.6 / +1.3 / +1.1 |
| 48 SMs | -0.4 / +7.6 / +3.1 / -0.2 | +0.0 / +1.9 / +2.2 / -0.1 |
| 64 SMs | +1.5 / cannot meet it / +0.3 / +1.2 | +0.6 / +1.6 / +0.6 / -0.5 |
| 80 SMs | cannot meet it / +3.3 / cannot meet it / +4.0 | +1.5 / +0.4 / +7.9 / +1.7 |
| chosen per workload (B4*) | -0.4 / +0.5 / -1.0 / -0.2 | +0.0 / -0.6 / +0.2 / -0.5 |

No size is best for all four. At the tightest target every fixed size is at least 3
points behind the queue kernel on one workload or cannot meet the target at all; at
1.10 x the 16-SM domain is within 1.4 points everywhere.

### The same on a load that moves

`python3 -m sim.policies --tag trace2`: the 30-second `trace2` load of section 4 -- 11720
MoE layers in 11 segments of eight batch sizes, with idle gaps -- under eleven policies.
Row 1 is the configuration that was measured (simulated energy 0.4-0.7% low), and row 3
was measured too (section 4). Domains are fixed at 32 + 76 SMs, as silicon would be.
Rows marked * use vLLM's per-SM costs.

| policy | busy s | vs row 1 | busy J | vs row 1 | highest 100 ms reading |
|---|---|---|---|---|---|
| queue kernel, stock governor (measured) | 19.82 |  | 7789 |  | 421 W |
| queue kernel, tdp_bisection | 19.84 | +0.1% | 7717 | -0.9% | 415 W |
| queue kernel, fixed 1050 MHz | 23.24 | +17.3% | 5924 | -23.9% | 258 W |
| pinned*, 64 cold experts, one clock | 22.37 | +12.8% | 8314 | +6.7% | 426 W |
| pinned*, 64 cold experts, two domains | 22.31 | +12.6% | 7546 | -3.1% | 411 W |
| pinned*, experts rebalanced per batch, one clock | 20.16 | +1.7% | 7842 | +0.7% | 416 W |
| pinned*, rebalanced, two domains, governor | 20.16 | +1.7% | 7912 | +1.6% | 432 W |
| pinned*, rebalanced, two domains, tdp_bisection | 20.09 | +1.4% | 7798 | +0.1% | 422 W |
| pinned*, rebalanced, two domains, 0.05 ms actuation | 20.08 | +1.3% | 7832 | +0.6% | 399 W |
| pinned*, rebalanced, two domains, closed_loop | 20.14 | +1.6% | 7735 | -0.7% | 431 W |
| pinned, one block per SM, rebalanced, two domains | 23.84 | +20.2% | 8855 | +13.7% | 402 W |

Moving experts across the boundary per batch -- software -- takes the pinned kernel from
12.8% behind to 1.7% behind. A second rail on top of that takes 0.3% off the time and
0.6% off the energy. On a placement that is NOT rebalanced it takes 9.2% off the energy,
which is the slack a bad split leaves, not something the rail created. A regulator that
switches in 0.05 ms instead of 38 ms changes nothing at this granularity. No row beats
the queue kernel in time. Four use less energy: `tdp_bisection` on the same kernel
(-0.9%, for +0.1% of time), the closed loop on two domains (-0.7% for +1.6%), and two
that pay for it in time -- the fixed 1050 MHz clock (-23.9% for +17.3%) and the
unbalanced two-domain row (-3.1% for +12.6%).

## 7. Boosting the domain that is still running

`python3 -m sim.boost`. A layer's two partitions rarely finish a GEMM at the same
instant, and the layer waits at a barrier for the slower one. The instant the faster one
finishes is an event; from then until the barrier the board draws less, and the domain
still running could be raised. This is the move an event loop exists for, and
`boost_on_finish` (`sim/control.py`) makes it: fixed clocks per domain, and a survivor's
domain goes to a boost clock until the barrier releases. A boost that could not land
before the survivor is done is not issued.

The comparison that matters is against fixed PER-DOMAIN clocks, not against one clock:
per-domain clocks already turn the early finisher's slack into a lower clock for it.
Each policy at its best clocks (the cold domain may go down to 510 MHz, the lowest clock
the rate model was fitted at), 7 cold-expert counts x 3 domain sizes, no actuation delay,
and two readings of the wall: the layer's MEAN power under 400 W, which is what the
A100's limiter enforces (it reads 20 ms means and a layer lasts 1.5-5 ms), and every
PHASE of the layer under 400 W, a limiter faster than a layer.

| workload | per-SM costs | wall | one clock, ms | a clock per domain | + boost the survivor | largest gain of the boost in any configuration | that configuration against the best |
|---|---|---|---|---|---|---|---|
| decode1024 | queue | mean | 1.1093 | 1.1093 (+0.00%) | 1.1093 (-0.00%) | -0.00% (64 cold experts on 48 SMs) | +0.0% |
| decode1024 | queue | instant | 1.1122 | 1.1122 (+0.00%) | 1.1122 (-0.00%) | -0.10% (16 cold experts on 16 SMs) | +1.9% |
| decode1024 | persistent | mean | 1.3644 | 1.3603 (-0.30%) | 1.3603 (+0.00%) | +0.00% (16 cold experts on 16 SMs) | +2.0% |
| decode1024 | persistent | instant | 1.3644 | 1.3603 (-0.30%) | 1.3603 (+0.00%) | +0.00% (16 cold experts on 16 SMs) | +2.0% |
| decode2048 | queue | mean | 1.5189 | 1.4838 (-2.31%) | 1.4838 (-0.00%) | -0.33% (48 cold experts on 32 SMs) | +4.2% |
| decode2048 | queue | instant | 1.5381 | 1.4975 (-2.64%) | 1.4972 (-0.02%) | -0.70% (32 cold experts on 32 SMs) | +9.8% |
| decode2048 | persistent | mean | 1.8895 | 1.8895 (+0.00%) | 1.8895 (+0.00%) | +0.00% (16 cold experts on 16 SMs) | +7.9% |
| decode2048 | persistent | instant | 1.8895 | 1.8895 (+0.00%) | 1.8895 (+0.00%) | +0.00% (16 cold experts on 16 SMs) | +7.9% |
| decode4096 | queue | mean | 2.2723 | 2.2707 (-0.07%) | 2.2685 (-0.10%) | -0.69% (48 cold experts on 32 SMs) | +11.2% |
| decode4096 | queue | instant | 2.3111 | 2.3095 (-0.07%) | 2.3075 (-0.09%) | -0.88% (16 cold experts on 32 SMs) | +18.9% |
| decode4096 | persistent | mean | 3.1747 | 3.1747 (+0.00%) | 3.1747 (+0.00%) | +0.00% (16 cold experts on 16 SMs) | +9.7% |
| decode4096 | persistent | instant | 3.1747 | 3.1747 (+0.00%) | 3.1747 (+0.00%) | +0.00% (16 cold experts on 16 SMs) | +9.7% |
| prefill | queue | mean | 3.8006 | 3.8006 (+0.00%) | 3.7966 (-0.11%) | -0.66% (32 cold experts on 32 SMs) | +12.8% |
| prefill | queue | instant | 3.9110 | 3.8413 (-1.78%) | 3.8413 (-0.00%) | -1.34% (16 cold experts on 32 SMs) | +19.8% |
| prefill | persistent | mean | 4.5758 | 4.5200 (-1.22%) | 4.5194 (-0.01%) | -0.01% (80 cold experts on 48 SMs) | +0.0% |
| prefill | persistent | instant | 4.5758 | 4.5389 (-0.81%) | 4.5387 (-0.00%) | -0.37% (32 cold experts on 16 SMs) | +2.1% |

At the configuration each policy would choose, the boost adds 0.00-0.11% to what
per-domain clocks already give. The reason is in the last two columns. Which experts are
cold is software, and at the best split the two domains finish together: one of them
runs alone for 0-4% of the GEMM time, so there is no window to boost into. The boost
gains most -- up to 1.3% -- in splits where one domain runs alone for a quarter to four
fifths of the GEMM time, and every split where it gains 0.3% or more is 2-20% slower
than the best one to begin with. With the kernel as measured, decode batches are not
power-limited: the domain left running is already at the top clock, and there is nothing
to boost to.

With the actuation delay put back, at the configuration where the boost gains most under
the instantaneous wall (latency against the best fixed per-domain clocks of the same
configuration; the same clocks at every delay):

| workload | per-SM costs | actuation delay: 0 ms | 0.01 ms | 0.03 ms | 0.1 ms | 0.3 ms | 1 ms | 38 ms |
|---|---|---|---|---|---|---|---|---|
| decode1024 | queue | -0.10% | -0.10% | -0.09% | -0.06% | -0.00% | -0.00% | -0.00% |
| decode2048 | queue | -0.70% | -0.69% | -0.66% | -0.58% | -0.33% | +0.00% | +0.00% |
| decode4096 | queue | -0.88% | -0.87% | -0.84% | -0.76% | -0.58% | -0.03% | +1.24% |
| prefill | queue | -1.34% | -1.33% | -1.30% | -1.21% | -0.96% | -0.83% | +2.51% |
| prefill | persistent | -0.37% | -0.37% | -0.36% | -0.33% | -0.24% | +0.00% | +0.00% |

A GEMM stage lasts about 1-3 ms. At 0.1 ms the boost keeps 83-90% of its gain in the
four cases where that gain is 0.3% or more. At 0.3 ms it keeps half to three quarters,
and in the two decode cases the restoring set-point arrives late enough that the next
stage starts at the boosted clock and overshoots the wall (424 and 432 W for that long).
Through NVML on the A100, 38 ms, no boost can be issued, and a plan that was counting on
one is up to 2.5% slower than the best fixed clocks.

## 8. Under every assumption nobody can measure

`python3 -m sim.sensitivity`. Eight things about a two-domain chip are parameters of the
hardware description and not facts: which clock the shared logic follows, how much idle
power it holds, what a clock-domain crossing costs, how the lower rail is regulated, how
much of an SM's panel-delivery cost follows the shared logic's clock, which clock the
DRAM-L2 coupling follows, whether a loaded die's extra power follows each domain's clock
or the faster one, and which clock the epilogue runs at. One more is a choice of ours:
the form of the rate model (section 2). For one fixed partition (64 cold experts on a
32-SM domain), B4 against B3, with the same kernel on both sides:

*decode 2048 tokens, the measured kernel*

| assumption | energy at B3's 1290 MHz latency: B4 - B3 | clocks | energy at B3's 1350 MHz latency: B4 - B3 | clocks | fastest under TDP: B4 - B3 | clocks |
|---|---|---|---|---|---|---|
| as shipped | -3.30% | 930/1305 | -6.10% | 930/1350 | -0.17% | 840/1410 |
| uncore follows the SLOWER domain | -2.80% | 930/1320 | -3.74% | 930/1380 | +0.00% | 1410/1410 |
| no idle power in shared logic | -3.60% | 930/1305 | -6.42% | 930/1350 | -0.17% | 840/1410 |
| half the idle power in shared logic | -3.00% | 930/1305 | -5.77% | 930/1350 | -0.17% | 840/1410 |
| domain crossing costs 5% | -3.09% | 1140/1290 | -6.11% | 930/1350 | -0.18% | 870/1410 |
| domain crossing costs 10% | -4.67% | 930/1290 | -6.12% | 930/1350 | -0.22% | 885/1410 |
| lower rail through an LDO | -1.12% | 1215/1290 | -4.27% | 930/1350 | -0.17% | 840/1410 |
| half of panel delivery on the uncore clock | -3.27% | 930/1305 | -4.38% | 930/1365 | -0.02% | 1200/1410 |
| all of panel delivery on the uncore clock | -4.02% | 1410/1170 | -5.24% | 1410/1275 | +0.00% | 1410/1410 |
| DRAM-L2 coupling on the uncore clock | -4.67% | 930/1290 | -6.12% | 930/1350 | -0.23% | 900/1410 |
| loaded-die power at the faster clock | -2.68% | 930/1305 | -5.17% | 930/1350 | -0.17% | 840/1410 |
| epilogue at the slower clock | -1.52% | 945/1320 | -2.11% | 1005/1380 | +0.00% | 1410/1410 |
| rate model: proportional form | -3.30% | 930/1305 | -6.09% | 930/1350 | -0.13% | 840/1410 |
| rate model: linear form | -3.25% | 930/1305 | -4.35% | 930/1365 | +0.00% | 1410/1410 |

*decode 4096 tokens, vLLM per-SM costs (a what-if)*

| assumption | energy at B3's 1290 MHz latency: B4 - B3 | clocks | energy at B3's 1350 MHz latency: B4 - B3 | clocks | fastest under TDP: B4 - B3 | clocks |
|---|---|---|---|---|---|---|
| as shipped | -3.10% | 930/1305 | -4.29% | 930/1365 | -1.47% | 1200/1380 |
| uncore follows the SLOWER domain | -3.84% | 930/1305 | -5.14% | 930/1365 | -1.59% | 945/1395 |
| no idle power in shared logic | -3.34% | 930/1305 | -4.57% | 930/1365 | -1.48% | 1215/1380 |
| half the idle power in shared logic | -2.86% | 930/1305 | -4.01% | 930/1365 | -1.47% | 1200/1380 |
| domain crossing costs 5% | -3.11% | 930/1305 | -4.29% | 930/1365 | -1.48% | 1200/1380 |
| domain crossing costs 10% | -3.11% | 930/1305 | -4.29% | 930/1365 | -1.49% | 1200/1380 |
| lower rail through an LDO | +0.00% | 1290/1290 | -2.03% | 990/1365 | -1.34% | 1065/1380 |
| half of panel delivery on the uncore clock | -3.09% | 930/1305 | -4.28% | 930/1365 | -1.45% | 1200/1380 |
| all of panel delivery on the uncore clock | -3.08% | 930/1305 | -4.27% | 930/1365 | -1.43% | 1200/1380 |
| DRAM-L2 coupling on the uncore clock | -3.11% | 930/1305 | -4.29% | 930/1365 | -1.48% | 1200/1380 |
| loaded-die power at the faster clock | -2.61% | 930/1305 | -2.57% | 1080/1365 | -1.42% | 1155/1380 |
| epilogue at the slower clock | -0.42% | 1065/1320 | -1.42% | 1215/1365 | -0.84% | 1230/1380 |
| rate model: proportional form | -3.10% | 930/1305 | -3.22% | 1110/1365 | -1.49% | 1185/1380 |
| rate model: linear form | -3.13% | 930/1305 | -3.94% | 1020/1365 | -1.67% | 1080/1380 |

Against a FIXED split the second rail is worth 1.1-4.7% of energy at the latency of a
1290 MHz clock with the kernel as measured (0.0-3.8% with vLLM's per-SM costs), 2.1-6.4%
at the latency of a 1350 MHz clock (1.4-5.1%), and 0.0-0.2% of latency at the wall
(0.8-1.7%). It never turns into a loss, because here B4 contains B3. The two settings
that shrink it most are an LDO on the lower rail, which removes the energy saving
entirely in one case, and an epilogue that runs at the slower domain's clock. The form
of the rate model moves the result by at most 1.8 points. An earlier version of this
section said the effect "never exceeds 4%"; it reaches 6.4%. Section 6 is smaller
because there B3 may move experts across the boundary and choose its SM split, which
costs nothing and removes most of the slack a slower cold domain would have used.

## 9. What changed in the conclusions

Before the review this document said the second rail was worth "0-2 points of energy at
equal latency and 0.0-0.4% of latency". With B3 searched fairly, tighter targets
reported and the calibration repaired, the range is -2.3 to +0.7 points and -0.2% to
+0.0% (vLLM's per-SM costs), and the reading is different: an interval that straddles
zero and sits inside the model's error is not a small gain, it is no detectable effect.
What did not change: a better single clock is the large term, the persistent kernel's
occupancy is what it costs, and vLLM's kernel at one clock is as good as anything with
two domains.

## 10. What the simulator cannot say

- **Anything below about 2%.** B3 and B4 share partitions and models, so their difference
  is better determined than either alone, but the co-run interference term carries a
  worst-case error of 12%, and the second-rail effect is of the size of the rms error.
- **Two domains were never measured.** Section 8 varies what can be varied; the rest is
  the statement that per-SM costs follow the domain's clock and shared capacities do not
  depend on which domain is asking. One experiment could test part of it on today's chip
  and has not been run: throttling the cold partition in software while the hot one
  runs at full speed, which gives the two partitions different rates under one clock.
- **A pinned kernel as efficient as vLLM's.** The rows marked * are a what-if. Nobody has
  built that kernel and no measurement stands behind its per-SM costs when pinned.
- **Tile size.** The model is linear in rows per tile. BLOCK_M 32 is 4-7% faster than
  that, and the pinned layer on 16-row tiles is the worst of its six cases (2.38%).
- **Small batches at high clocks.** The queue kernel at 512-1024 tokens above 1350 MHz is
  4% slower than simulated.
- **Single passes shorter than 1 ms.** 113 of the 490 replayed single passes of the
  earlier campaigns are that short, mostly launch gap, and are left out of the ladder.
  Scored, they are 7.80% rms (bias +5.78%); the 377 kept are 3.33%, all 490 together
  4.75%.
- **SM counts that divide evenly into tiles.** A 48+60 split interferes less than 54+54
  by several percent; the model is smooth in SM count and does not know why.
- **Thermal transients.** After an idle gap the board draws 10-20 W less for 2-5 s while
  the die warms. The simulator charges the warm value at once. A single time constant
  was tried and made the traces worse; the die warms and cools with two.
- **The Sep 17 `partition_kappa_qwen.csv` sweep** disagrees with the simulator's power by
  -6% to +9% depending on tile size, where three later campaigns agree within 3%. Cause
  not found.
- **One calibration model, MoE layers only.** Fitted on Qwen3-30B-A3B. It carries over to
  OLMoE and Qwen3-Next, which share its hidden size, and not to Granite (K = 1536: 5-8%
  slow in time, section 3): a model with another hidden size needs its own sweep.
  Attention and request-level scheduling are outside it. Within Qwen, layer 47, where
  10 experts get no tokens and a 16-expert cold set holds 25 pairs, is the worst case
  (co-run rms 5.76%).
- **The governor and limiter** are eight hand-set numbers. The limiter's loop was read
  off replay latencies at 9-15 ms resolution; what the board draws inside one loop needs
  an external meter. The wake from the parked clock comes 0.11-0.35 s after the load on
  the chip and always at 0.25 s in the simulator. And right after an idle gap the real
  board draws less than the model says, so its clock lands 15-45 MHz above the
  simulated one for the first few hundred ms (fig28c).

## 11. Reproduce

```bash
bash sim/reproduce.sh          # calibrate, validate, control groups, boost, figures; ~45 min on 16 cores, no GPU
python3 -m sim.report          # every table above as markdown
python3 check_consistency.py   # the numbers quoted in the .md files against the data
```

## Appendix A. Every row of the validation ladder

`fit`: yes, the calibration used these records; partly, it used a by-product of them; no,
it never saw them. Time errors are ln(simulated / measured). Power errors are a share of
board power; the last column is their rms as a share of board power above idle.

| rung | what | fit | time: n | rms | bias | worst | power: n | rms | bias | worst | rms above idle |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 fitted | scan, 930-1410 MHz: solo | yes | 622 | 2.81% | +0.20% | -12.8% | 622 | 2.16% | -0.09% | +7.6% | 3.65% |
| 1 fitted | scan, 930-1410 MHz: co-run | yes | 548 | 3.17% | -0.66% | -8.6% | 274 | 2.49% | -0.71% | +9.0% | 3.41% |
| 1 fitted | holdout, 990-1380 MHz: solo | yes | 520 | 3.92% | +0.95% | -15.2% | 520 | 2.86% | -0.31% | +7.8% | 4.75% |
| 1 fitted | holdout, 990-1380 MHz: co-run | yes | 474 | 3.28% | +0.12% | +11.7% | 237 | 3.36% | -0.27% | +14.0% | 4.62% |
| 1 fitted | lowclk, 510-870 MHz: solo | yes | 416 | 2.71% | -0.23% | -9.0% | 416 | 1.54% | +0.23% | +5.7% | 3.35% |
| 1 fitted | lowclk, 510-870 MHz: co-run | yes | 384 | 2.12% | -0.24% | -6.9% | 192 | 2.30% | -0.71% | +7.9% | 3.75% |
| 2 other days | prefill, BM16-128, 54 SMs, 600-1290 MHz | no | 55 | 6.10% | +3.75% | +12.4% | 55 | 5.51% | +1.04% | +11.1% | 10.39% |
| 2 other days | decode 1024, BM16-128, 600-1290 MHz | no | 94 | 3.39% | +2.93% | +6.7% | 94 | 2.16% | -1.63% | -5.1% | 3.71% |
| 2 other days | decode 2048, BM16-128, 600-1290 MHz | no | 106 | 3.03% | +2.56% | +6.4% | 106 | 2.69% | -2.41% | -5.0% | 4.76% |
| 2 other days | prefill, 8 cold experts | no | 13 | 0.48% | +0.19% | +1.0% | 13 | 1.41% | -0.41% | +3.0% | 2.49% |
| 2 other days | prefill, 32 cold experts | no | 25 | 1.13% | +0.65% | +2.5% | 25 | 2.07% | -1.08% | -4.1% | 3.79% |
| 2 other days | prefill, 64 cold experts | no | 26 | 0.62% | +0.19% | +1.5% | 26 | 1.70% | -0.47% | -3.1% | 3.09% |
| 2 other days | prefill, 96 cold experts | no | 26 | 0.64% | -0.00% | +1.3% | 26 | 1.81% | -0.00% | +3.7% | 3.18% |
| 2 other days | prefill, 54 SMs, 1050-1410 MHz | partly | 32 | 0.83% | +0.78% | +1.3% | 32 | 2.95% | +2.17% | +5.8% | 4.36% |
| 2 other days | first co-run campaign: solo | no | 96 | 3.11% | +0.93% | -7.5% | 96 | 1.67% | -0.69% | -4.5% | 2.86% |
| 2 other days | first co-run campaign: 54+54, 32+76 | no | 96 | 2.87% | -0.13% | +7.1% | 48 | 2.19% | -0.74% | -5.1% | 3.11% |
| 3 other workloads | MoE layer 6: solo | no | 154 | 3.41% | +1.27% | -10.6% | 154 | 2.70% | +0.03% | +6.9% | 4.43% |
| 3 other workloads | MoE layer 6: co-run | no | 122 | 3.52% | +0.32% | -10.0% | 61 | 2.40% | -0.94% | +6.4% | 3.25% |
| 3 other workloads | MoE layer 24: solo | no | 154 | 3.80% | +1.40% | +11.5% | 154 | 3.08% | +0.02% | -8.5% | 4.94% |
| 3 other workloads | MoE layer 24: co-run | no | 124 | 4.12% | +0.84% | -11.6% | 62 | 2.46% | -0.75% | -5.2% | 3.31% |
| 3 other workloads | MoE layer 36: solo | no | 154 | 3.49% | +1.19% | -11.2% | 154 | 2.90% | -0.13% | -8.9% | 4.70% |
| 3 other workloads | MoE layer 36: co-run | no | 124 | 3.60% | +0.14% | -10.5% | 62 | 2.51% | -0.73% | -5.5% | 3.35% |
| 3 other workloads | MoE layer 47: solo | no | 154 | 4.37% | +1.73% | +17.2% | 154 | 4.99% | +1.48% | +27.3% | 9.35% |
| 3 other workloads | MoE layer 47: co-run | no | 126 | 5.76% | +2.12% | +19.7% | 63 | 3.30% | +0.10% | +13.8% | 4.68% |
| 3 other workloads | pinned layer END TO END, six SM splits | no | 108 | 1.38% | -0.20% | +4.2% | 108 | 2.32% | +1.71% | +5.5% | 3.48% |
| 3 other workloads | &nbsp;&nbsp;the pair was also swept side by side | no | 94 | 1.46% | -0.19% | +4.2% | 94 | 1.94% | +1.41% | +4.8% | 3.00% |
| 3 other workloads | &nbsp;&nbsp;the pair was not | no | 14 | 0.70% | -0.29% | -1.9% | 14 | 4.01% | +3.76% | +5.5% | 5.75% |
| 3 other workloads | &nbsp;&nbsp;prefill, BM128, 16 cold experts | no | 18 | 0.57% | -0.26% | -1.1% | 18 | 3.54% | +3.31% | +5.5% | 5.26% |
| 3 other workloads | &nbsp;&nbsp;prefill, BM128, 64 cold experts | no | 18 | 0.87% | -0.57% | -1.9% | 18 | 2.66% | +2.35% | +5.5% | 3.91% |
| 3 other workloads | &nbsp;&nbsp;1024 tokens, BM64, 64 cold experts | no | 18 | 0.98% | -0.57% | -1.9% | 18 | 1.35% | +1.01% | +2.9% | 1.99% |
| 3 other workloads | &nbsp;&nbsp;2048 tokens, BM16, 64 cold experts | no | 18 | 2.38% | +1.67% | +4.2% | 18 | 1.81% | +0.78% | +4.0% | 2.90% |
| 3 other workloads | &nbsp;&nbsp;2048 tokens, BM64, 64 cold experts | no | 18 | 1.27% | -0.62% | -3.0% | 18 | 1.64% | +1.05% | +4.1% | 2.47% |
| 3 other workloads | &nbsp;&nbsp;4096 tokens, BM64, 64 cold experts | no | 18 | 1.46% | -0.87% | -3.0% | 18 | 2.19% | +1.78% | +4.8% | 3.34% |
| 3b other models | OLMoE, 64 experts, I=1024: solo | no | 104 | 2.74% | +1.97% | +7.2% | 104 | 2.61% | +1.27% | -5.7% | 4.34% |
| 3b other models | OLMoE, 64 experts, I=1024: co-run | no | 96 | 2.86% | +2.04% | +6.8% | 48 | 2.96% | -2.10% | -6.5% | 3.92% |
| 3b other models | OLMoE, 64 experts, I=1024: whole layers, both kernels | no | 14 | 1.92% | +1.34% | +4.0% | 14 | 2.70% | +2.18% | +4.1% | 3.61% |
| 3b other models | Granite, 40 experts, K=1536: solo | no | 92 | 8.08% | -7.73% | -15.1% | 92 | 6.12% | +5.42% | +17.1% | 9.86% |
| 3b other models | Granite, 40 experts, K=1536: co-run | no | 72 | 5.86% | -5.28% | -9.6% | 36 | 6.11% | +5.59% | +11.3% | 8.47% |
| 3b other models | Granite, 40 experts, K=1536: a partition that fits in L2 | no | 36 | 4.43% | -3.72% | -7.6% | 24 | 19.23% | +17.68% | +33.3% | 36.64% |
| 3b other models | Granite, 40 experts, K=1536: whole layers, both kernels | no | 14 | 5.10% | -4.40% | -7.0% | 14 | 7.37% | +7.00% | +10.1% | 10.14% |
| 3b other models | Qwen3-Next, 512 experts, top-10: solo | no | 104 | 2.72% | -1.03% | -6.2% | 104 | 2.64% | -0.62% | +9.6% | 4.31% |
| 3b other models | Qwen3-Next, 512 experts, top-10: co-run | no | 96 | 3.43% | -1.72% | -6.8% | 48 | 4.60% | +3.17% | +11.7% | 6.44% |
| 3b other models | Qwen3-Next, 512 experts, top-10: whole layers, both kernels | no | 14 | 2.36% | -1.20% | -5.1% | 14 | 3.34% | +2.96% | +6.1% | 4.56% |
| 4 whole layers | persistent kernel, other layers (never fitted) | no | 40 | 1.17% | +0.75% | +2.2% | 40 | 2.56% | +2.37% | +4.5% | 3.55% |
| 4 whole layers | persistent kernel, training sweeps | yes | 118 | 2.18% | +1.11% | +6.5% | 118 | 1.84% | +0.94% | +7.2% | 2.66% |
| 4 whole layers | queue kernel, other layers (never fitted) | no | 25 | 1.24% | +0.51% | -2.7% | 25 | 1.92% | +1.58% | +3.2% | 2.51% |
| 4 whole layers | queue kernel, training sweeps | yes | 56 | 2.00% | -0.09% | -4.7% | 56 | 2.16% | +1.43% | +4.8% | 2.98% |
| 4 whole layers | queue kernel, 64-7744 tokens, 510-1410 MHz | yes | 141 | 1.79% | -0.03% | -5.3% | 141 | 1.68% | -0.66% | +4.0% | 2.24% |
| 4 whole layers | queue kernel, 512-12288 tokens, 1290 MHz | no | 8 | 2.10% | -0.55% | -4.0% | 8 | 2.04% | +0.65% | +4.2% | 2.61% |
| 4 whole layers | persistent, BM64 and BM128, 512-7744 tokens (never fitted) | no | 28 | 2.44% | +0.94% | -5.0% | 28 | 2.23% | +0.19% | +6.7% | 3.00% |
| 4 whole layers | queue kernel, 512-7744 tokens (never fitted) | no | 12 | 2.47% | -0.82% | -4.3% | 12 | 2.70% | +1.84% | +4.9% | 3.50% |
| 4 whole layers | prefill at 1200 MHz, both kernels (Sep 17) | no | 2 | 0.93% | +0.89% | +1.1% | 2 | 1.82% | -1.72% | -2.3% | 2.30% |

## Appendix B. B4 on one domain size

The control groups with the cold V/f domain fixed at 16 to 80 SMs for every workload,
against B0. B4 and B4* of section 6 are, for each workload and target, the best of these
rows.

| objective | | decode1024 | decode2048 | decode4096 | prefill |
|---|---|---|---|---|---|
| fastest under TDP (time) | B4@16 | +20.1% | +22.3% | +19.5% | +21.4% |
| fastest under TDP (time) | B4@32 | +19.1% | +22.2% | +23.7% | +22.2% |
| fastest under TDP (time) | B4@48 | +17.2% | +24.1% | +25.7% | +19.9% |
| fastest under TDP (time) | B4@64 | +19.1% | +24.6% | +20.9% | +19.8% |
| fastest under TDP (time) | B4@80 | +20.2% | +23.5% | +27.0% | +22.3% |
| fastest under TDP (time) | B4*@16 | +1.9% | +0.3% | +0.3% | +1.7% |
| fastest under TDP (time) | B4*@32 | +1.4% | +0.4% | +1.3% | +1.5% |
| fastest under TDP (time) | B4*@48 | +0.2% | +1.7% | +1.5% | +0.8% |
| fastest under TDP (time) | B4*@64 | +1.3% | +2.2% | +0.5% | +0.3% |
| fastest under TDP (time) | B4*@80 | +2.0% | +1.9% | +3.8% | +1.5% |
| least energy, <= 1.00 x B0 latency (energy) | B4@16 | cannot meet it | cannot meet it | cannot meet it | cannot meet it |
| least energy, <= 1.00 x B0 latency (energy) | B4@32 | cannot meet it | cannot meet it | cannot meet it | cannot meet it |
| least energy, <= 1.00 x B0 latency (energy) | B4@48 | cannot meet it | cannot meet it | cannot meet it | cannot meet it |
| least energy, <= 1.00 x B0 latency (energy) | B4@64 | cannot meet it | cannot meet it | cannot meet it | cannot meet it |
| least energy, <= 1.00 x B0 latency (energy) | B4@80 | cannot meet it | cannot meet it | cannot meet it | cannot meet it |
| least energy, <= 1.00 x B0 latency (energy) | B4*@16 | cannot meet it | cannot meet it | cannot meet it | cannot meet it |
| least energy, <= 1.00 x B0 latency (energy) | B4*@32 | cannot meet it | cannot meet it | cannot meet it | cannot meet it |
| least energy, <= 1.00 x B0 latency (energy) | B4*@48 | cannot meet it | cannot meet it | cannot meet it | cannot meet it |
| least energy, <= 1.00 x B0 latency (energy) | B4*@64 | cannot meet it | cannot meet it | cannot meet it | cannot meet it |
| least energy, <= 1.00 x B0 latency (energy) | B4*@80 | cannot meet it | cannot meet it | cannot meet it | cannot meet it |
| least energy, <= 1.02 x B0 latency (energy) | B4@16 | cannot meet it | cannot meet it | cannot meet it | cannot meet it |
| least energy, <= 1.02 x B0 latency (energy) | B4@32 | cannot meet it | cannot meet it | cannot meet it | cannot meet it |
| least energy, <= 1.02 x B0 latency (energy) | B4@48 | cannot meet it | cannot meet it | cannot meet it | cannot meet it |
| least energy, <= 1.02 x B0 latency (energy) | B4@64 | cannot meet it | cannot meet it | cannot meet it | cannot meet it |
| least energy, <= 1.02 x B0 latency (energy) | B4@80 | cannot meet it | cannot meet it | cannot meet it | cannot meet it |
| least energy, <= 1.02 x B0 latency (energy) | B4*@16 | -0.8% | -5.8% | -3.9% | +0.4% |
| least energy, <= 1.02 x B0 latency (energy) | B4*@32 | -7.6% | -6.6% | -1.9% | +0.2% |
| least energy, <= 1.02 x B0 latency (energy) | B4*@48 | -12.4% | +0.6% | +0.1% | -3.9% |
| least energy, <= 1.02 x B0 latency (energy) | B4*@64 | -10.4% | cannot meet it | -2.6% | -2.5% |
| least energy, <= 1.02 x B0 latency (energy) | B4*@80 | cannot meet it | -3.7% | cannot meet it | +0.2% |
| least energy, <= 1.05 x B0 latency (energy) | B4@16 | cannot meet it | cannot meet it | cannot meet it | cannot meet it |
| least energy, <= 1.05 x B0 latency (energy) | B4@32 | cannot meet it | cannot meet it | cannot meet it | cannot meet it |
| least energy, <= 1.05 x B0 latency (energy) | B4@48 | cannot meet it | cannot meet it | cannot meet it | cannot meet it |
| least energy, <= 1.05 x B0 latency (energy) | B4@64 | cannot meet it | cannot meet it | cannot meet it | cannot meet it |
| least energy, <= 1.05 x B0 latency (energy) | B4@80 | cannot meet it | cannot meet it | cannot meet it | cannot meet it |
| least energy, <= 1.05 x B0 latency (energy) | B4*@16 | -15.9% | -12.1% | -9.7% | -6.2% |
| least energy, <= 1.05 x B0 latency (energy) | B4*@32 | -18.2% | -13.2% | -8.5% | -6.1% |
| least energy, <= 1.05 x B0 latency (energy) | B4*@48 | -19.8% | -9.2% | -7.5% | -9.0% |
| least energy, <= 1.05 x B0 latency (energy) | B4*@64 | -20.0% | -9.2% | -8.9% | -8.2% |
| least energy, <= 1.05 x B0 latency (energy) | B4*@80 | -17.5% | -11.3% | +1.3% | -6.1% |
| least energy, <= 1.10 x B0 latency (energy) | B4@16 | cannot meet it | cannot meet it | cannot meet it | cannot meet it |
| least energy, <= 1.10 x B0 latency (energy) | B4@32 | cannot meet it | cannot meet it | cannot meet it | cannot meet it |
| least energy, <= 1.10 x B0 latency (energy) | B4@48 | cannot meet it | cannot meet it | cannot meet it | cannot meet it |
| least energy, <= 1.10 x B0 latency (energy) | B4@64 | cannot meet it | cannot meet it | cannot meet it | cannot meet it |
| least energy, <= 1.10 x B0 latency (energy) | B4@80 | cannot meet it | cannot meet it | cannot meet it | cannot meet it |
| least energy, <= 1.10 x B0 latency (energy) | B4*@16 | -25.2% | -19.5% | -16.2% | -14.6% |
| least energy, <= 1.10 x B0 latency (energy) | B4*@32 | -25.3% | -19.7% | -15.2% | -13.6% |
| least energy, <= 1.10 x B0 latency (energy) | B4*@48 | -26.6% | -17.1% | -14.3% | -14.8% |
| least energy, <= 1.10 x B0 latency (energy) | B4*@64 | -26.1% | -17.4% | -15.8% | -15.2% |
| least energy, <= 1.10 x B0 latency (energy) | B4*@80 | -25.2% | -18.6% | -8.6% | -12.9% |
| least energy, <= 1.25 x B0 latency (energy) | B4@16 | -14.5% | -1.1% | +3.8% | +10.5% |
| least energy, <= 1.25 x B0 latency (energy) | B4@32 | -14.3% | -2.5% | +11.0% | +9.9% |
| least energy, <= 1.25 x B0 latency (energy) | B4@48 | -15.7% | +3.3% | cannot meet it | +5.5% |
| least energy, <= 1.25 x B0 latency (energy) | B4@64 | -15.8% | +2.9% | +5.5% | +6.2% |
| least energy, <= 1.25 x B0 latency (energy) | B4@80 | -15.6% | -0.1% | cannot meet it | +9.5% |
| least energy, <= 1.25 x B0 latency (energy) | B4*@16 | -29.0% | -25.6% | -23.5% | -22.7% |
| least energy, <= 1.25 x B0 latency (energy) | B4*@32 | -29.2% | -25.5% | -22.9% | -22.5% |
| least energy, <= 1.25 x B0 latency (energy) | B4*@48 | -29.4% | -24.5% | -22.4% | -23.1% |
| least energy, <= 1.25 x B0 latency (energy) | B4*@64 | -29.3% | -24.8% | -23.1% | -23.1% |
| least energy, <= 1.25 x B0 latency (energy) | B4*@80 | -29.0% | -25.1% | -20.5% | -22.0% |
