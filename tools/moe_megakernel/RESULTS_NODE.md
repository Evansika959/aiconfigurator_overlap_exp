# The node engine: GEMM + all-reduce overlap, and multi-GPU steps

2026-10-08, 4x A100-SXM4-40GB. Scenario one of the project (a GEMM overlapped with an
NCCL all-reduce) measured completely enough to model, the simulator extended to it and
to several GPUs, and a registered blind test. Every "simulated" number is from
`sim/node.py` with the profiles of `sim/calib/a100_kernels.json`; the chip's energies,
voltage curve and limiter are the MoE calibration, unchanged.

Design and plan for multi-GPU training: `MULTIGPU.md`. Registered test:
`OVERLAP_HELDOUT.md`. How to rerun: section 9.

## 1. Summary

1. **What a collective costs a GEMM depends on which was launched first, and it is
   exactly an integer.** Launched on an idle GPU, an all-reduce with c CTAs leaves a
   cuBLAS GEMM `108 - 2c` SMs. Launched behind a running GEMM it leaves `108 - c`, and
   is itself 1.1 to 1.5 times slower for its whole length. Probe: 16 CTA counts, both
   orders, 1,218 valid repetitions, no exception up to 48 CTAs. ncu gives the cause:
   NCCL's block needs 89 KB of shared memory and runs under a 100 KB carveout, cuBLAS's
   needs 144 KB under 164 KB, and the carveout belongs to an SM pair.
2. **One event engine now covers both case studies' kernels.** With the profiles
   alone it reproduces the 1,975 earlier overlapped runs to 0.75% median iteration
   time (p90 3.4%, max 11.2%) and 1.9% median power; the overlap study's closed form
   scored 4.1% median, p90 17.9%, max 31.7% on its own runs.
3. **It carries to several GPUs without being fitted on them.** A different clock on
   each GPU: iteration time 1.0% median. One heavy GPU and three light: 1.0%. A
   training step written as kernels on streams (never fitted on): step time 0.6%
   median, max 2.5%; the end of each of 1,176 kernels within 0.12% of the step
   (median); power of each GPU 1.5% median.
4. **Per-GPU clocks pay, measured.** An unbalanced DDP step (GPU 0 four times the
   others' batch): with no clock lock the three light GPUs wait at 1410 MHz. Locking
   only those three at 900 MHz costs +0.4% step time and saves 19.7% of the node's
   energy (16 CTAs; +2.3% and 18.7% at 4 CTAs). Locking all four at 1200 MHz saves
   13.0% for +3.1%.
5. **Three things the single-GPU model did not have**, each measured tonight: a GPU
   draws 16-20 W above idle between collectives a few ms apart and returns to idle
   within 0.5 s of the last one; a collective that waits for a participant draws a
   tenth of its CTAs' power; four boards under one GEMM differ by up to 7%.
6. **Registered blind test: 19 of 19 criteria pass** (section 6). 864 cells new on
   every axis, predictions on disk before the first was run: overlapped iteration time
   1.07% median, 3.8% at p90, 8.7% at worst; power of each GPU 1.8%; training steps
   1.3%. Worst: the all-reduce's own span when the GEMM outlasts it (9% slow on
   average, up to 54%), which leaves the iteration time alone.
7. **The engine chose clocks and the measurement agreed** (section 3.7): predicted
   -20.4% and -25.7% of node energy for under 1% of step time on two unbalanced steps,
   measured -19.7% and -25.5%.
8. **What-if for scenario one** (section 7): a V/f domain for the collective's SMs is
   worth 5 to 9 points of energy at equal time against NCCL's default of 24 CTAs, once
   the single clock is also allowed to choose the CTA count. That is not zero, unlike
   MoE: the two kernels are of different kinds.

## 2. What was measured

All under `tools/overlap_exp/data/sim_overlap/`; harness `sweep_spans.py` unless noted.
A "cell" is one configuration; spans, loop time and per-GPU power are recorded on every
rank. Clocks were locked per run and restored; power caps were never touched.

| Campaign | What | Cells |
|---|---|---|
| `anchor_*` (old harness) | 132 cells of the earlier campaigns repeated after the driver change (580.173.02 to 580.178.04): time within 2.7%, median 0.0 to 0.4%; power median -1.6 to +1.4% | 132 |
| `cal` | per-kernel spans inside the overlap, both launch orders, 6 GEMM shapes x 2 message sizes x 5 CTA counts x 4 clocks | 544 |
| `commsize` | the all-reduce alone, 4 to 512 MiB, 1 to 64 CTAs, 6 clocks | 300 |
| `nccl_occupancy` (`probe_nccl_occupancy.py`) | SMs a block of a given shared memory can reach while c CTAs are resident: 16 CTA counts x 2 launch orders x 7 block sizes x 6 repetitions | 1,344 |
| `ncu_gemm` (`profile_kernels.py`) | launch grid, shared memory, occupancy, DRAM and L2 bytes of 47 GEMM shapes | 47 |
| `ncu_nccl_*` (`profile_nccl.py`) | launch attributes of NCCL's all-reduce kernel | 2 |
| `world2`, `world3` | the all-reduce and the overlap on 2 and 3 GPUs | 108 |
| `hetero` | a different clock on each GPU, 11 patterns | 198 |
| `unbalanced` | GPU 0: 8192x8192x8192, others 2048x8192x8192; light GPUs slowed | 60 |
| `endphase` | message stepped 16 to 256 MiB beside a long GEMM | 80 |
| `train` | training steps as programs: DDP overlap, DDP serial, tensor parallel; balanced and unbalanced; locked, per-GPU and unlocked clocks | 52 |
| `waiting`, `waiting_swap` | three GPUs wait for a fourth, with the collective resident or the host asleep | 32 |
| `gaps` | a GEMM and/or a small all-reduce, then 6-12 ms of nothing | 20 |
| `link_idle` (`probe_link_idle.py`) | board power every 20 ms before, during and 8 s after a burst of all-reduces, 4 clocks | trace |
| `blind`, `blind_auto`, `blind_multi`, `blind_train` | the registered test | 864 |

## 3. What the measurements say, before any simulator

### 3.1 Launch order and the carveout (fig29)

A probe kernel whose blocks carry a chosen amount of dynamic shared memory was run while
a 2 GiB all-reduce was resident; each block records the SM it ran on. With the cuBLAS
block's 147,456 B:

| NCCL CTAs | 1 | 2 | 3 | 4 | 6 | 8 | 12 | 16 | 22 | 27 | 32 | 40 | 48 | 54 | 64 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SMs lost, all-reduce launched first | 2 | 4 | 6 | 8 | 12 | 16 | 24 | 32 | 44 | 54 | 64 | 80 | 96 | 102-104 | 104-106 |
| SMs lost, launched behind a GEMM | 1 | 2 | 3 | 4 | 6 | 8 | 12 | 16 | 22 | 27 | 32 | 40 | 48 | - | - |
| All-reduce time, behind / first | 1.16 | 1.18 | 1.22 | 1.25 | 1.26 | 1.31 | 1.31 | 1.29 | 1.31 | 1.31 | 1.31 | 1.32 | 1.26 | 1.14 | 1.02 |

Launched first, the CTAs take the SM pairs with the lowest ids (4 CTAs: SMs 0-7 are
lost). A probe with no shared memory loses one SM per CTA in both orders; one with
32-64 KB loses none behind a GEMM (it shares the SM with NCCL's block).

ncu, launch attributes:

| Kernel | Threads per block | Shared memory per block | Carveout it runs under |
|---|---|---|---|
| `ampere_bf16_s16816gemm_bf16_{256x128,128x256}` (cuBLAS) | 256 | 98,304 dynamic + 49,152 static = 147,456 B | 167,936 B |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` | 544 | 82,240 + 7,104 = 89,344 B | 102,400 B |

So: an SM pair holds one carveout while a block is resident. NCCL first: its pair is
in the 100 KB class and no 144 KB block fits on either SM. GEMM first: every pair is in
the 164 KB class, NCCL's block runs there beside nothing else of that size, and an SM
at 164 KB of shared memory has 28 KB of L1 left where NCCL normally has 92. That the
missing L1 is what slows it is an inference; the carveouts and the SM counts are
measured. This is the "GEMM runs 2.41x slower, the wave model says 1.40x" problem of
`overlap_exp/GOALS.md`: its `compose` and `predict` assume `108 - c`.

### 3.2 The collective beside a GEMM

Inside an overlapped iteration, while GEMM tiles are in flight the collective's rate is
`1 / (1 + s)`, `s = 0.21 (f / 1200 MHz)^0.45 (2 / c)^0.65`: 17% slower at 2 CTAs and
1200 MHz, 8% at 8, 3% at 32 (130 spans where the GEMM ends first: 0.95% rms of the
span). The GEMM pays 21 us once.

Not explained: a GEMM with unsplit K = 8192 that is still running when the collective
would end sometimes costs it 0.3 to 1.3 ms more. The same cell (8192x8192x8192 beside
64 MiB, 8 CTAs, 1200 MHz) gave +0.70 ms in `cal` and +0.09 ms in `endphase`; within
`endphase` it appears at some message sizes and not their neighbours. It moves the
iteration time only when the collective ends last.

### 3.3 A GPU with nothing to run

| State (locked clock, 35 C) | 1410 MHz | 1200 | 900 | 510 |
|---|---|---|---|---|
| Idle table of the power model (GPU 0, Sept) | 76.9 | 65.2 | 58.9 | 56.7 |
| Context alive, no collective for 8 s, mean of 4 boards | 78.2 | 65.5 | 59.2 | 56.9 |
| One 16 MiB all-reduce every 12 ms, nothing else | 97.9 | 83.5 | 75.4 | 73.4 |
| Waiting for a slower GPU, host asleep (16 CTAs, 6 boards) | 97.4 | 83.3 | 76.9 | 75.3 |
| Waiting for a slower GPU, collective resident (same) | 103.7 | 86.8 | 78.6 | 76.7 |
| 0.5 to 1 s after the last collective | 78.6 | 65.8 | 59.3 | 56.9 |

(W per GPU; in rows 3-5 the GPU moves bytes for 2-9% of the time.) Between GEMMs a few
ms apart the board does fall back to the idle table (`gaps`: a 4096x4096x4096 GEMM
every 6.3 ms at 1200 MHz draws 91.8 W, its duty cycle says 92.8). Between collectives
it does not: 16 to 20 W above idle at every clock, gone half a second after the last
one. Read as the NVLink leaving its low-power state; net of the bytes moved the engine
has it as `link_w` = 14 W. A resident collective that only waits adds 2 to 6 W more
(`spin` = 0.10 of its CTAs' power) and none of the busy-chip term.

### 3.4 Several GPUs

- **A collective runs at its slowest participant's pace, nearly.** With one GPU of
  four at 900 MHz and three at 1410 the all-reduce is 4% faster than with all four at
  900; `mix` = 0.10 of the mean per-byte time and 0.90 of the slowest's fits 42 cells
  to 2.4% rms.
- **A waiting collective is not free in SMs.** A GPU that reaches its all-reduce early
  keeps the kernel resident, and its `2c` SMs, until the last GPU joins. With clocks
  1410/900/1410/900 and a long GEMM on each, the fast GPUs' own next GEMM then runs on
  `108 - 2c` SMs for milliseconds: at 32 CTAs their GEMM took 6.4 ms against about 4 alone.
- **Boards differ.** Under one GEMM at 1200 MHz GPUs 1 and 3 read 5-7% less than GPUs
  0 and 2. Two gains per board (rail, everything else: 0.92-0.98 and 0.98-1.01); with
  nothing running all four are within 3 W of the idle table.
- **World size.** Per byte of message a world of 2 is 1.24x faster than a world of 4
  and a world of 3 is 1.06x; a ring's byte count would say 1.50 and 1.125.

### 3.5 The cuBLAS kernel (ncu, 47 shapes)

One block per 256x128 output tile, 256 threads; occupancy `sm__cycles_active /
elapsed` equals `grid / (waves * 108)` within 2% for every shape, the blind ones
included. cuBLAS splits K when there are few tiles and K is long (x2 for
1024x8192x8192, 2048x2048x8192, 1024x4096x8192, 2048x4096x8192; x3 for 512x4096x4096
and 1024x2048x4096; x5 for 1024x4096x16384). DRAM traffic is 3 to 5 times the three
matrices once (8192x8192x8192: 1,300 MB read, 127 MB written, 15.5 GB through L2).

### 3.6 An unbalanced training step, against the stock governor

Synthetic step, three layers, hidden 8192: forward GEMMs, weight-gradient GEMMs whose
128 MiB all-reduce is issued as soon as each is (DDP), input-gradient GEMMs. GPU 0 has
batch 8192, GPUs 1-3 batch 2048. Measured, mean power, 16 CTAs:

| Clocks (GPU 0 / 1-3), MHz | Step, ms | Node power, W | Energy per step, J |
|---|---|---|---|
| not locked (GPU 0 settles at 1245, the others stay at 1410) | 36.02 | 1003 | 36.11 |
| 1410 / 1410 (asked) | 36.00 (-0.0%) | 1016 (+1.3%) | 36.57 (+1.3%) |
| 1410 / 900 | 36.15 (+0.4%) | 802 (-20.0%) | 29.01 (-19.7%) |
| 1200 / 1200 | 37.14 (+3.1%) | 846 (-15.6%) | 31.41 (-13.0%) |
| 1200 / 900 | 37.25 (+3.4%) | 772 (-23.0%) | 28.77 (-20.3%) |
| 1200 / 705 | 37.71 (+4.7%) | 755 (-24.6%) | 28.49 (-21.1%) |
| 1200 / 510 | 38.18 (+6.0%) | 747 (-25.4%) | 28.54 (-21.0%) |

At 4 CTAs: 1410 / 900 is +2.3% and -18.7%. With the all-reduces after the backward
pass instead of beside it (16 CTAs): +1.7% and -18.3%. The balanced step (batch 4096
everywhere) has no such free lunch: one clock for all, 1200 MHz, is +4.8% and -11.2%;
900 MHz is +38.5% and -20.5%.

### 3.7 Clocks chosen by the engine, then measured

`python3 -m sim.node_policy`: for a step, the least-energy clocks whose predicted step
time is within a budget of A's (no clock lock), searched over a 13-clock grid for two
groups of GPUs, the one on the critical path and the rest. The chosen settings were
then run (`policy`, 10 cells, 16 CTAs). Against A:

| Step (batch per GPU) | Policy | Clocks, MHz | Predicted: time, energy | Measured: time, energy |
|---|---|---|---|---|
| 8192 + 3 x 2048 | one clock, +1% | 1260 for all | -0.9%, -11.7% | +0.3%, -10.5% |
| | a clock per GPU, +1% | 1410 / 900 | +0.8%, -20.4% | +0.9%, -19.7% |
| 8192 + 3 x 4096 | one clock, +1% | 1260 for all | -0.8%, -14.6% | +0.3%, -13.6% |
| | a clock per GPU, +1% | 1410 / 975 | +0.9%, -25.7% | +0.5%, -25.5% |
| | a clock per GPU, +3% | 1260 / 705 | +2.1%, -26.1% | +1.9%, -25.8% |
| 4 x 4096 (balanced) | one clock, +1% | 1305 for all | 0.0%, -0.5% | 0.0%, -0.2% |
| | one clock, +3% | 1260 for all | +1.2%, -4.3% | +0.8%, -3.2% |

A's own step time and energy were predicted to 0.1-1.5% and 0.2-1.6%. One locked clock
already saves 10-14% on the unbalanced steps, because under the stock governor the
waiting GPUs sit at 1410 MHz; a clock per GPU adds 9 to 12 points; on the balanced step
neither does anything, as it should.

## 4. The model

`sim/kernels.py` (profiles), `sim/fit_kernels.py` (their fit), `sim/node.py` (engine).

- **cuBLAS GEMM**: `grid` tiles (ncu), one per SM, in waves. A wave takes
  `(1004 * GMAC per tile + 8.3) / f` ms, 3.2% more if K is split; launch `3.1 / f` ms;
  32 us of host time before the kernel is queued. Six numbers, 0.39% rms on 69 solo
  cells (12 shapes x 8 clocks). Energy per tile from the MoE calibration's energy per
  MAC and per byte: no number fitted.
- **NCCL all-reduce**: latency + bytes / bandwidth, both tables over CTAs x clock (72
  cells, 0.85% rms within a cell); between CTA counts through a knee law (leave-one-out
  1.6% rms). Power of its CTAs: a table.
- **Co-run rules**: section 3.1 and 3.2, as measured.
- **Several GPUs**: sections 3.3 and 3.4: `link_w`, `spin`, `mix`, two gains per board.
- **Workload**: per GPU a host program (issue a kernel on a stream, wait for the device,
  sleep, ask for a clock); the same JSON `sweep_spans.py` runs.
- **Engine**: events (host free, kernel queued, wave ends, collective ends a phase,
  set-point, limiter), one limiter per GPU with the MoE engine's rules.

## 5. Validation on data that existed at registration

`python3 -m sim.validate_overlap`. Error = simulated / measured - 1.

Earlier campaigns (one clock for all GPUs, node-mean power, loop time only):

| Group | Fitted on these rows? | n | Time: median, p90, max | Power: median, p90 |
|---|---|---|---|---|
| GEMM alone | yes | 341 | 0.25%, 0.71%, 2.4% | 3.0%, 5.9% |
| All-reduce alone | yes | 318 | 0.48%, 1.6%, 3.6% | 2.0%, 3.7% |
| Overlap, clock held | no | 1,975 | 0.75%, 3.4%, 11.2% | 1.9%, 5.8% |
| GEMM then all-reduce, one stream | no | 757 | 0.36%, 1.1%, 2.9% | 1.6%, 3.7% |
| Overlap, throttled below the clock asked | no | 113 | 2.7%, 4.0%, 5.5% | 0.7%, 1.8% |
| Overlap, no clock lock | no | 288 | 1.7%, 3.9%, 6.9% | 3.2%, 7.0% |

The simulator throttles in 110 of the 113 throttled runs and settles 65 MHz above the
measured lowest clock (a minimum over a window that wanders +-15 to 30 MHz); with no
lock it throttles in 65 of the 74 runs that did and in 1 of the 214 that did not. The
earlier solo-GEMM power rows read 3% above tonight's on the same cells.

Tonight's spans (times: rank 0; power: every GPU, mean of the samples):

| Campaign | Fitted on? | Cells | Iteration time | GEMM span | All-reduce span | Power, each GPU |
|---|---|---|---|---|---|---|
| `cal`, all-reduce first | co-run rules, boards | 225 | 0.74%, p90 3.2% | 0.61% | 1.1%, p90 6.7% | 1.7%, p90 5.3% |
| `cal`, GEMM first | the behind-a-GEMM table | 222 | 1.5%, p90 5.7% | 0.65% | 2.0%, p90 12% | 2.4%, p90 6.3% |
| `world2`, overlap | 2 numbers | 36 | 0.73% | 0.32% | 1.4% | 2.0% |
| `world3`, overlap | 2 numbers | 36 | 0.54% | 0.41% | 2.3% | 1.8% |
| `hetero`, all-reduce alone | `mix` | 66 | 0.89%, p90 3.5% | - | 0.86% | 1.7% |
| `hetero`, overlap | no | 117 | 1.0%, p90 3.2% | 0.47% | 1.6% | 2.1% |
| `unbalanced`, overlap | no | 21 | 0.99%, max 4.3% | 0.38% | 5.5% | 2.2% |
| `unbalanced`, one stream | no | 22 | 1.3%, max 4.8% | 0.35% | 1.3% | 2.0% |
| `train`, clock held | no | 29 | 0.57%, max 2.5% (each GPU) | end of each kernel: 0.12% of the step, max 2.7% | | 1.5%, max 8.2% |
| `train`, at the wall or not locked | no | 23 | 0.79%, max 3.0% | 0.25%, max 3.1% | | 1.4%, max 8.0% |

(Medians unless marked.)

## 6. Registered blind test

Predictions were written before the runs existed (`OVERLAP_HELDOUT.md`; stamp
2026-10-08 09:59 UTC, first blind cell measured 10:33). 864 cells, each new on all four
axes: 8 GEMM shapes, 3 message sizes, 5 CTA counts and 5 clocks no earlier run used.
`python3 -m sim.heldout_overlap score`: **19 of 19 registered criteria pass.**

| Plan | Cells | Quantity | Limit | Result |
|---|---|---|---|---|
| `blind`, clock held | 34 | GEMM alone, time: median, p90 | 2%, 5% | 0.33%, 0.83% |
| | 75 | all-reduce alone, time: median, p90 | 3%, 6% | 1.78%, 4.74% |
| | 565 | overlap, iteration time: median, p90, max | 2.5%, 7%, 15% | 1.07%, 3.83%, 8.73% |
| | 565 | GEMM span inside the iteration: median | 2.5% | 0.57% |
| | 565 | all-reduce span inside the iteration: median | 3.5% | 2.55% |
| | 2,260 | power, each GPU: median, p90 | 4%, 8% | 1.81%, 4.08% |
| `blind_auto`, no clock lock | 48 | overlap, iteration time; power: median | 3%, 5% | 2.36%, 1.52% |
| `blind_multi`, a clock per GPU | 12 | all-reduce alone, time each GPU: median | 4% | 1.20% |
| `blind_multi`, per-GPU clocks and unbalanced | 21 | time each GPU: median, all-reduce first; one stream | 4%, 4% | 0.70%, 1.99% |
| | 14 | power, each GPU: median | 5% | 1.78% |
| `blind_train`, clock held | 13 | step time, each GPU; power, each GPU: median | 3%, 5% | 1.31%, 1.41% |

Not among the criteria, same run:

- Cells the board throttled (asked for 1350 MHz): the simulator throttled in all 41 of
  `blind` and all 14 of `blind_multi`; iteration time 0.78% median (max 2.4%), power
  0.52%, its median clock 15 MHz above the measured one.
- `blind_train` at the wall or with no lock (25 cells): step time 0.92% median, max
  4.9%; power 1.17%; the end of each of 1,444 kernels 0.22% of the step.
- Training steps with the clock held: the end of each of 668 kernels within 0.17% of
  the step (median), 1.5% at p90, 3.5% at worst.

Where it was worst:

- **The all-reduce's own span when the GEMM outlasts it** (254 of the 565 overlap
  cells): the collective is 9.4% slower than simulated on average, 28% at p90, 54% at
  worst. This is the effect of section 3.2 that the rules do not explain, and in the
  blind set it is not rare. Looked at afterwards, over all 352 such cells of `cal`,
  `endphase` and `blind`: the median collective is slowed about twice as much as the
  rule says at 6 to 16 CTAs (the rule was fitted where the GEMM covers only its
  start) and as much as it says from 24 CTAs up; on top of that, in a tenth to a third
  of the cells, it loses another 2 to 5 GEMM tile times, most often beside the GEMMs
  with the most waves and at 3 and 10 CTAs. The iteration time is hardly touched
  (0.55% median in those cells) because the GEMM ends last. Where the collective ends last (311 cells) its span
  is within 1.7% (median), 4.7% (p90), 9.2% (max).
- **Small messages.** At 32 MiB the all-reduce's span is 12% short on average beside a
  GEMM, and alone its worst cell is 16.6% off: a third of a 32 MiB all-reduce is
  latency, and the latency table is the noisiest thing in the profile.
- **Sub-half-millisecond cells with no clock lock**: a 0.19 ms GEMM alone was 16% longer
  than predicted, a 0.5 ms all-reduce alone 30% (2 of 14 solo cells; the same cells
  with a locked clock were within 2%).

The working tree after registration (formatting, the `comm_clock` what-if and the
`clock` op added) reproduces the registered predictions: see
`data/overlap_heldout/pred_second_shot_*.csv`.

## 7. What-if: a V/f domain for the collective's SMs

`python3 -m sim.study_overlap`. Not a measurement: the A100 has one clock per GPU. The
overlapped iteration of scenario one (collective first, c CTAs, the GEMM on the other
`108 - 2c` SMs), 6 GEMM shapes x 4 message sizes x 7 CTA counts = 168 workloads, three
policies: A, no clock lock; the best single clock; and the best pair of clocks, one for
the collective's SM pairs and one for the GEMM's SMs. "Best" is least energy with the
iteration no more than a budget longer than A's. Assumptions that a one-clock chip
cannot check are listed at the top of `sim/study_overlap.py` (the shared logic follows
the faster clock; the domain is exactly the collective's SM pairs).

With the CTA count held at A's:

| Budget | One clock, energy vs A: median (best) | Two clocks | Two minus one, points |
|---|---|---|---|
| +0% | 0.0% (-12.4%) | 0.0% (-16.7%) | 0.0 (best -13.8) |
| +1% | 0.0% (-13.6%) | -21.8% (-30.5%) | -17.2 (best -30.5) |
| +3% | -4.2% (-19.0%) | -23.7% (-35.4%) | -11.7 (best -35.4) |

At +1% the gain is where the collective ends last (112 workloads: -24.8 points) and
not where the GEMM does (56: -2.0); by CTA count, -28 points at 2 CTAs falling to -6 at
48. The mechanism is the one the MoE study lacked: two kernels of different kinds. The
GEMM finishes early and wants a low clock; the collective sets the iteration and its
bandwidth needs a high one; one clock cannot give both.

But the CTA count is a knob too, and it exists today. More CTAs make the collective
fast enough that the whole chip can drop its clock. With the CTA count free for every
policy (A at a fixed default; 24 workloads):

| A's CTA count | Budget | One clock (and any CTA count) | Two clocks (and any CTA count) | Two minus one, points: median (best) |
|---|---|---|---|---|
| 8 | +1% | -41.4% | -42.3% | -1.9 (-7.2) |
| 16 | +1% | -23.2% | -27.9% | -3.1 (-10.1) |
| 22 | +0% | -13.2% | -21.0% | -6.6 (-12.5) |
| 22 | +1% | -13.7% | -21.8% | -8.5 (-17.3) |
| 22 | +3% | -16.5% | -23.0% | -5.1 (-13.8) |
| 32 | +1% | -11.8% | -22.1% | -6.9 (-14.2) |
| 32 | +3% | -15.7% | -24.0% | -6.5 (-12.8) |

(Both policies then choose about 27-32 CTAs. NCCL's own choice on this node, read by
ncu with no CTA variable set, is 24 CTAs at 64 and at 512 MiB; 22 is the nearest count
in the study's grid.) The one-clock column can be checked
against the earlier campaigns, which measured the unlocked run and locked runs at ten
CTA counts and eight clocks. The best MEASURED locked (clock, CTA count) within +1% of
the unlocked run's iteration time, median over the same 24 workloads: -43.1% against A
at 8 CTAs (simulated -41.4%), -25.2% at 16 (-23.2%), -14.2% at 32 (-11.8%). That part
is real and available today.

So in scenario one a second V/f domain
is worth something, unlike in MoE: 5 to 9 points of energy at equal time in the median
workload against NCCL's default, up to 17, once the baseline is allowed the knob it
already has. Against a
baseline that keeps a low CTA count, most of what looks like a two-domain gain is a
CTA-count gain. These numbers are inside what the engine can resolve only barely (its
power error is 2%, 6% at p90), and none of the two-clock half can be measured here.

## 8. What the engine gets wrong, or does not have

- The collective's own span when a GEMM outlasts it: 9% slow on average in the blind
  test, 54% at worst (3.2, section 6). Two modes on the same cell; cause unknown.
- The all-reduce's latency at small messages and unseen CTA counts (16.6% at worst,
  alone, at 32 MiB).
- The collective launched behind a GEMM: its factor is a measured table over 2-48 CTAs
  and 510-1410 MHz from 64 and 256 MiB messages. At 2 GiB the probe saw x1.29-1.32
  where the table says x1.3-1.4.
- At the wall the simulated clock settles higher than the measured minimum; the
  iteration is 1-3% short.
- `link_ms`, how long the +14 W lasts after a collective, is only bracketed (15 ms to
  0.5 s).
- No thermal state, as in the MoE engine.
- The stock governor is "top clock while there is work". Park and wake are not in the
  node engine; nothing measured here had an idle gap longer than 12 ms.
- Kernels other than the cuBLAS GEMM and the NCCL all-reduce have no profile.

## 9. Reproduce

    cd tools/overlap_exp
    python3 plan_spans.py                                   # every plan
    python3 sweep_spans.py --plan data/sim_overlap/plan_cal.json --out data/sim_overlap/cal.csv
    python3 probe_nccl_occupancy.py --out data/sim_overlap/nccl_occupancy.csv
    bash data/sim_overlap/run_night.sh                      # the calibration queue, in order
    cd ../moe_megakernel && source env.sh
    python3 -m sim.fit_kernels                              # -> sim/calib/a100_kernels.json, ~7 min
    python3 -m sim.test_node                                # 12 invariants
    python3 -m sim.validate_overlap                         # every table of section 5, ~15 min
    python3 -m sim.heldout_overlap score                    # section 6
    python3 -m sim.node_policy                              # section 3.7: the clocks, and the plan to measure
    python3 -m sim.study_overlap --default-ctas 22          # section 7, ~5 min on 12 cores
    python3 make_node_figs.py                               # fig29-32
