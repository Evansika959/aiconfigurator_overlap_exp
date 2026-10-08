# Registered test of the node engine (GEMM + all-reduce overlap, multi-GPU steps)

Written on 2026-10-08, **before** any of the runs below existed. The predictions are in
`data/overlap_heldout/pred_first_shot_<plan>.csv`, with a timestamp and SHA-256 of every
input in `data/overlap_heldout/stamp_first_shot.json` and the simulator's source as it
was in `data/overlap_heldout/frozen_sim.tar.gz`. `python3 -m sim.heldout_overlap score`
reads those files; it never recomputes a prediction. A later simulator may be scored
with `--second-shot`, and is then reported beside this one, never instead of it.

## What the simulator had seen

| Source | What it fitted |
|---|---|
| MoE calibration (`sim/calib/a100_qwen.json`, GPU 0, Sept 2026) | idle power, voltage curve, energy per MAC and per byte, the limiter's rules: used as they are |
| Earlier overlap campaigns (`overlap_432`, `commbound_*`, `ext_*`; solo rows only) | GEMM cost per MAC, per tile and per split, launch cost (6 numbers); the all-reduce's latency, bandwidth and power tables |
| `sim_overlap/cal` (per-kernel spans on already-measured cells) | host-side launch costs; how a co-running GEMM slows the collective (3 numbers); the table of factors when the collective is launched behind a GEMM; two gains per board |
| `sim_overlap/commsize` | the all-reduce alone, 8 to 512 MiB, 1 to 64 CTAs |
| `sim_overlap/nccl_occupancy` (probe) | SMs a GEMM loses per CTA: 2 when the collective launches first, 1 when it launches behind a GEMM |
| `sim_overlap/hetero` (all-reduce alone, a clock per GPU) | `mix`: how far a collective is from its slowest participant's pace (1 number) |
| `sim_overlap/gaps`, `waiting`, `waiting_swap`, `link_idle` | `link_w`, what a GPU draws between collectives a few ms apart, and `spin`, what a waiting collective draws (2 numbers) |
| `sim_overlap/world2`, `world3` | factors for a world of 2 and 3 GPUs (not used by this test: every plan is 4 GPUs) |
| `sim_overlap/grids.json` (ncu) | launch grids of every GEMM shape, the blind ones included: a launch attribute, not a timing |

Scored but **not fitted on**: the overlapped rows of the earlier campaigns (2,845 with
the serial and throttled ones), the overlap cells of `hetero`, and all of `unbalanced`
and `train`, the calibration-side twins of `blind_multi` and `blind_train`. Looking at
them did change the code twice before this registration, and both changes are in the
frozen simulator: power is compared with the mean of the samples (the harness's median
was wrong on a 20-60 ms step), and a waiting GPU no longer draws the busy-chip term.

## What is unseen

| Axis | Seen in any earlier run | Blind |
|---|---|---|
| GEMM shape (M x N x K) | 12 shapes of `grid12.json` (tiles 128 to 4096) | 768x4096x4096, 3072x4096x4096, 6144x4096x4096, 1536x8192x8192, 3072x8192x8192, 6144x8192x8192, 4096x12288x4096, 2048x2048x8192 |
| Message | 4, 8, 16, 64, 128, 256, 512 MiB | 32, 96, 384 MiB |
| NCCL CTAs | 1, 2, 4, 8, 12, 16, 22, 32, 48, 64 | 3, 6, 10, 24, 40 |
| Clock | 300, 510, 705, 900, 1050, 1200, 1305, 1410 MHz | 600, 810, 975, 1125, 1350 MHz |

No blind cell shares a value with an earlier run on ANY of the four axes.

## Plans

| Plan | Cells | What it asks |
|---|---|---|
| `blind` | 715 | one clock for all four GPUs, the same GEMM on each: solo GEMM, solo all-reduce, the overlap with the collective launched first |
| `blind_auto` | 62 | the same overlap cells with no clock lock |
| `blind_multi` | 49 | a different clock on each GPU (the collective alone, and the overlap); an unbalanced step (GPU 0: 6144x8192x8192, others 1536x8192x8192) with the light GPUs slowed |
| `blind_train` | 38 | training steps as programs: 4 layers, batch 3072 on every GPU or 6144 + 3 x 1536, gradient all-reduce beside the backward pass (DDP), after it, or tensor-parallel; one clock, per-GPU clocks, no lock |

## Criteria

Error is `simulated / measured - 1`. "Held" means every GPU's lowest sampled clock was
within 20 MHz of the clock asked for. Times are from CUDA events on rank 0 unless the
row says each GPU; power is each GPU's own, through the board correction. The limits
were set from `python3 -m sim.validate_overlap` on the day of registration (the same
quantity on cells already measured is given in brackets).

| # | Plan | Cells | Quantity | Limit [same quantity on cells already measured] |
|---|---|---|---|---|
| 1 | blind | solo GEMM, held | time: median, p90 | 2%, 5% [0.3%, 1.1%] |
| 2 | blind | solo all-reduce, held | time: median, p90 | 3%, 6% [0.6%, 1.9%; leave-one-CTA-out bandwidth 1.6% rms] |
| 3 | blind | overlap, held | iteration time: median, p90, max | 2.5%, 7%, 15% [0.75%, 3.4%, 11.2% on 1,975 earlier runs] |
| 4 | blind | overlap, held | GEMM span inside the iteration: median | 2.5% [0.6%] |
| 5 | blind | overlap, held | all-reduce span inside the iteration: median | 3.5% [1.1%] |
| 6 | blind | overlap, held | power, each GPU: median, p90 | 4%, 8% [1.7%, 5.3%] |
| 7 | blind_auto | overlap, not locked | iteration time, power: median | 3%, 5% [1.7%, 3.2%] |
| 8 | blind_multi | all-reduce alone, per-GPU clocks, held | time, each GPU: median | 4% [0.9%] |
| 9 | blind_multi | overlap and unbalanced step, held | time, each GPU: median, both launch patterns; power, each GPU: median | 4%, 4%, 5% [1.0%, 1.3%, 2.2%] |
| 10 | blind_train | programs, held | step time, each GPU: median; power, each GPU: median | 3%, 5% [0.6%, 1.6%] |

The limits are several times the bracketed numbers on purpose: a blind cell is new on
four axes at once, and every table it needs is interpolated.

Known before the test, and expected to show up in it:

- A GEMM with unsplit K = 8192 that is still running when the collective would end
  costs the collective 0.3 to 1.3 ms the rules do not explain. Three blind shapes have
  K = 8192. It lengthens the all-reduce span (criterion 5) and matters to the iteration
  time only when the collective ends last.
- The factor for a collective launched behind a GEMM is a table over 2 to 48 CTAs and
  510 to 1410 MHz from 64 and 256 MiB messages; `blind_train` uses it at 6 and 24 CTAs.
- That 0.3 to 1.3 ms is not reproducible: the same cell measured in two campaigns gave
  +0.09 and +0.70 ms.
- Cells throttled at 1350 MHz: on earlier throttled runs the simulator's clock settles
  65 MHz above the measured minimum and its iteration is 2.7% short.

## Outcome (2026-10-08, 11:17 UTC)

All 864 cells were measured between 10:33 and 11:17 UTC, after the predictions were on
disk. **19 of 19 criteria pass.** Numbers, and where the simulator was worst, are in
`RESULTS_NODE.md` section 6; `python3 -m sim.heldout_overlap score` prints them from
the registered files.

| # | Result |
|---|---|
| 1 | 0.33%, 0.83% |
| 2 | 1.78%, 4.74% |
| 3 | 1.07%, 3.83%, 8.73% |
| 4 | 0.57% |
| 5 | 2.55% (p90 17%, max 54%: the known effect, in 254 cells where the GEMM ends last) |
| 6 | 1.81%, 4.08% |
| 7 | 2.36%, 1.52% |
| 8 | 1.20% |
| 9 | 0.70%, 1.99%, 1.78% |
| 10 | 1.31%, 1.41% |
