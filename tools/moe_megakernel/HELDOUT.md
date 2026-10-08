# A held-out test of the simulator, registered before the data existed

**Why.** Every governor trace in `RESULTS_SIM.md` section 4 was called a hold-out because
the governor's and limiter's numbers were set against the `dynamics` plan only. That is
true of the numbers and not of the model: the limiter was rebuilt as a 20 ms loop after
the per-replay latencies of `trace2` were looked at, and `trace`, `forward` and `forward2`
were replayed many times while the engine was being written. None of them is a clean test
any more. This one is.

## Registered on 2026-10-01 at 15:21 UTC, before any of it was measured

**Frozen.** The simulator and its calibration as of this moment, archived in
`data/heldout_frozen_sim.tar.gz` (sha256 `7593cce444df3fb627e8343eb9310981363927b24f7cf817e5241e83ac39d4d8`);
`data/heldout_frozen.sha256` lists every file. It is the repository's `sim/` with one
change, three lines in `validate_trace.spec_of` that read a load name such as `d20@160`
(20 decode cells beginning at traced cell 160) -- an input format, not a model change.
The evaluation runs from that archive, not from the working tree.

**The plans** come from `heldout_plan.py` (seed 20261001), not from anybody's head, and
are stored in `data/heldout_plan.json`:

| tag | layers per graph replay | plan |
|---|---|---|
| `heldout1` | MoE layer 17, four times | `idle:3,p20:0.8,d40@150:1.1,p3:1.8,p10:3,p14:1.7,d40@150:2.8,d20@160:1.2,d56@132:1.2,d112@80:1.3,idle:4.15,d40@150:2.2,idle:0.42,d80@112:0.7,d10@182:2.4,d20@160:1.4,d10@182:1.6,idle:1.76` |
| `heldout2` | all 48 MoE layers, once | `idle:3,p10:3.3,p17:3.5,idle:1.29,d112@80:2.9,d20@160:3.1,d28@130:2.7,d56@132:3.1,idle:1.61,d40@150:3.6,idle:1.03,p9:3.2,idle:1` |

What is new in them against every sweep and trace so far:

- **Batch sizes.** 640, 1280, 1792, 2560, 3584, 5120 and 7168 decode tokens. None was
  ever run.
- **Routing.** Decode batches are pooled from traced cells 96-191, which no fit used (the
  7168-token batch needs 112 cells and takes 16 of them from the fitted range). Prefill
  uses prompt sets 3, 9, 10, 14, 17 and 20; only 0, 2, 4, 7, 11 and 13 were ever used.
- **Layer.** `heldout1` runs MoE layer 17. No sweep ran it on its own.
- **Order.** Segment order, lengths (0.7-3.6 s) and idle gaps (0.4-4.2 s, on both sides
  of the 2 s after which the governor parks the clock) are random.

**Procedure.** Each plan is run three times on GPU 0 under the stock governor (no clock
lock, no power cap) with `governor_trace.py`, then replayed ONCE through the frozen
simulator with `python3 -m sim.validate_trace --tag heldout1` (and `heldout2`). Whatever
that prints is the result. Nothing in the simulator is changed between the measurement
and the replay. If the simulator is changed later, the numbers below stay as the
first-shot result and any later number is reported beside them, not instead of them.

**What will be reported, and what counts as a pass.** The thresholds are looser than what
the earlier traces gave, because these batch sizes, prompts and this layer are new:

| quantity | earlier traces (seen while building) | pass if |
|---|---|---|
| latency per layer, rms over segments | 0.91-2.07% | <= 3% |
| board power, rms over segments | 0.68-1.35% | <= 2% |
| SM clock, rms over segments | 7-15 MHz | <= 20 MHz |
| energy of each whole run | -0.1% to -1.2% | within +-2% |
| first 400 ms after a batch-size change, per replay (`heldout1`) | 0.55-0.98% | <= 1.5% |

The wake from the parked clock is reported and not judged: the chip itself wakes
anywhere between 0.13 and 0.35 s after the load arrives.

## Result (measured 15:21-15:26 UTC, replayed once at 15:26 UTC)

Four of the five registered quantities pass on both traces. One misses on one trace: the
clock on `heldout1`, 21.6 MHz against a limit of 20.

| quantity | pass if | `heldout1` (layer 17, 14 segments x 3 runs) | `heldout2` (48 layers, 8 segments x 3 runs) |
|---|---|---|---|
| latency per layer, rms over segments | <= 3% | 1.58% (worst +3.7%) -- pass | 0.90% (worst -1.6%) -- pass |
| board power, rms over segments | <= 2% | 1.60% (worst -5.5%) -- pass | 0.61% (worst +1.4%) -- pass |
| SM clock, rms over segments | <= 20 MHz | **21.6 MHz (worst -89) -- MISS** | 9.0 MHz (worst -23) -- pass |
| energy of each whole run | within +-2% | -0.6%, -0.8%, -1.7% -- pass | -0.7%, -1.0%, -0.5% -- pass |
| first 400 ms after a batch-size change, per replay | <= 1.5% | 0.94% (1834 replays) -- pass | 0.86% (48 replays) |

Reported and not judged: the first 400 ms after a wake from the parked clock, 6.95% and
4.67%; after a short idle gap, 0.85% and 1.04%; the highest 100 ms power reading, 418 W
simulated against 427 W measured and 417 W against 444 W.

Figures: `figs/fig25_sim_heldout1.png`, `figs/fig25_sim_heldout2.png`
(`make_sim_trace_fig.py --tag heldout1 --first-shot`). Raw output of the replay, kept
apart from `sim/out` because that directory is rewritten by every later run:
`data/heldout_first_shot/trace_heldout1.txt`, `trace_heldout2.txt` and the series behind
the figures.

### Looked at afterwards (not part of the registered test)

- **Where the clock miss comes from.** The six `heldout1` segments that begin with a wake
  from the parked clock are 49.3 MHz rms off; the other 36 are 11.7 MHz. Two causes. The
  chip's wake is not repeatable: it came 0.11-0.33 s after the load in these six wakes,
  and the simulator always wakes at 0.25 s. And the score treats the two sides
  differently: the measured clock is averaged from 150 ms into each segment (NVML lags)
  and the simulated one from the segment's start, which in a 0.8 s segment that begins
  parked is a third of the window. Scored over the same window on both sides the clock
  error is 14.8 MHz on `heldout1` and 6.8 MHz on `heldout2`. That is a flaw in the score,
  found after the fact; the miss stands as registered.
- **The simulated clock is low after a wake.** After the wake at 3 s the chip runs
  18-25 MHz above the simulator through the next segment and 6-15 MHz above it for the
  ten seconds after that, at the same power. A die that has just idled draws less than
  the power model says; the simulator has no thermal state.
- **Small batches are 2-3% fast in the simulator.** Batches of 640 and 1280 tokens: bias
  -2.6% (12 segments). Larger ones: bias +0.3%, rms 0.9% (30 segments). The same bias the
  earlier traces showed at 512 and 1024 tokens, on sizes that were never run.
- **Energy is low in every run**, by 0.5-1.7% here and 0.1-1.4% on the earlier traces: a
  bias of about one percent, not noise.

## Second shot: the same measurements, the calibration repaired (17:50 UTC)

Not a registered test. After the first shot a review found that the rate model's window
solve stopped before it had converged, in the fit and in the engine alike
(`RESULTS_SIM.md` section 2). The solver was repaired and the calibration refitted; the
governor's and limiter's eight numbers were not touched. The two held-out traces took no
part in the refit or in choosing the form of the rate model. Replayed through the working
tree, they give the second column. The first-shot result above stands as the registered
one.

| quantity | pass if | `heldout1`: first shot | second shot | `heldout2`: first shot | second shot |
|---|---|---|---|---|---|
| latency per layer, rms over segments | <= 3% | 1.58% (worst +3.7%) | 1.56% (worst +4.4%) | 0.90% (worst -1.6%) | 0.68% (worst +1.5%) |
| board power, rms over segments | <= 2% | 1.60% (worst -5.5%) | 1.76% (worst +6.2%) | 0.61% (worst +1.4%) | 0.53% (worst +1.4%) |
| SM clock, rms over segments | <= 20 MHz | **21.6 MHz (worst -89): miss** | 19.4 MHz (worst -88) | 9.0 MHz (worst -23) | 8.5 MHz (worst -20) |
| energy of each whole run | within +-2% | -0.6%, -0.8%, -1.7% | +0.0%, -0.3%, -1.1% | -0.7%, -1.0%, -0.5% | +0.0%, -0.3%, +0.2% |
| first 400 ms after a batch-size change, per replay | <= 1.5% | 0.94% | 0.79% | 0.86% | 0.94% |
| ... after a wake from the parked clock (not judged) | | 6.95% | 6.81% | 4.67% | 4.31% |
| ... after a short idle gap (not judged) | | 0.85% | 0.79% | 1.04% | 0.89% |
| highest 100 ms reading, simulated / measured (not judged) | | 418 / 427 W | 414 / 427 W | 417 / 444 W | 416 / 444 W |
| SM clock, the same window on both sides (not registered) | | 14.8 MHz | 12.6 MHz (worst -55) | 6.8 MHz | 6.5 MHz (worst -12) |
| board power, the same window on both sides (not registered) | | | 2.54% | | 0.58% |

The clock on `heldout1` is now under the registered limit, by 0.6 MHz. That is not a
pass: it is a second attempt, and the margin is far inside the chip's own spread (the
three runs score 16.2, 14.4 and 25.8 MHz). What the second shot does show is that
repairing the calibration did not cost accuracy on data it never saw, and that the energy
bias of about one percent is gone (+0.0% to -1.1% in six runs, against -0.5% to -1.7%).

Figures of the second shot: `figs/fig25_sim_heldout1_second_shot.png`,
`figs/fig25_sim_heldout2_second_shot.png`; its raw output is `sim/out/trace_heldout1.txt`
and `trace_heldout2.txt`.

`sim.validate_trace` now prints both scores, the registered one and the one over the same
window on both sides. The like-for-like score is better for the clock and worse for
`heldout1`'s power, 2.54% against 1.76%. Nearly all of that difference is one segment,
the 0.8 s prefill batch that follows the wake at 3 s. The chip's own mean power over it
was 340, 352 and 384 W in the three runs, because it woke at a different moment each
time; the simulator always wakes 0.25 s in and says 380 W over the same window (+11.8%,
+8.0%, -1.1%). Scored whole, the simulated segment includes its own 0.25 s at the parked
clock and comes to 361 W, which happens to sit between the three.
