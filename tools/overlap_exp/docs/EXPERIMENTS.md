# Where the energy opportunities are, and what to measure next

Written after the 432-case overlap sweep. Everything in section 1 is measured, not
assumed; sections 2-3 are the designs that follow from it.

---

## 1. What is already settled

### The two kernels want opposite clocks

Energy per iteration, `E = P * t`, minimised over the six swept clocks:

| kernel | energy-optimal clock | n |
|---|---|---|
| GEMM alone | **900 MHz** | 12 of 12 shapes, unanimous |
| all-reduce alone | **1410 MHz** (4 CTA counts), 1200 MHz (2) | 6 of 6 |
| the two overlapped | **900 MHz** (58), 1200 (13), 1410 (1) | 72 (shape, CTA) pairs |

The GEMM is power-dense and wants to sit at the floor voltage. The collective is
cheap in power and mostly pays static energy, so it wants to finish fast. Overlapping
forces one clock, so the conflict is real.

### But splitting the clock is not the opportunity

Four ways to run one GEMM plus one all-reduce, over all 72 (shape, CTA) pairs:

| | energy vs C | latency vs C |
|---|---|---|
| A serial, per-phase DVFS (GEMM at its best f, comm at its best f) | **−1.1%** median | **+5.8%** median |
| B serial, one clock | worse than A by 2.65% median | — |
| C concurrent, one clock (best f) | baseline | baseline |
| D concurrent at 1410 (naive "run fast") | +18.9% median, +31.2% max | −27% |

Holding execution mode fixed so overlap is not a confound, the **pure** split-clock
gain is **+2.65% median, +8.68% max**, and it clears 2% on only 45 of 72 pairs. It is
small because the collective is a median 28.4% of pair energy and its own clock
sensitivity is modest, so the product lands around 2-3%.

**Overlap beats it.** Compressing the iteration amortises `P_static` -- a median 47% of
board power -- over less time, and that saves more than per-phase voltage tuning does.

### Three hard constraints close the "different frequency domains" idea

| constraint | measurement |
|---|---|
| one SM clock domain per GPU | two concurrent kernels cannot have different `f`, by construction |
| one memory clock | `nvidia-smi -q -d SUPPORTED_CLOCKS`: **1215 MHz, single value**. No second domain |
| DVFS transition is slow | **38.5 ms one-way** (10 ms blocking NVML call + 28 ms settle), 77 ms round trip |

The transition cost is decisive. For the 2.65% ceiling to pay for a 77 ms round trip,
each phase would have to run **2.9 seconds**. Kernels here are 0.2-30 ms; a whole
training step is 100-500 ms. Two orders of magnitude short, and the gap is not
closeable by engineering -- 28 ms of it is the hardware settling, not software.

Measured through the CLI the cost is worse still (~65 ms per `nvidia-smi` call, process
spawn), but that part is avoidable and is not the binding constraint.

### So the ranking of levers is

| lever | size | cost to exploit |
|---|---|---|
| **1. pick the right single clock** | 18.9% median, 31.2% max energy | free |
| **2. pick the right CTA count** | 8.6x on collective energy (1->32 CTA at 1200 MHz) | free |
| **3. overlap at all** | 8.6% median energy, 1.18x median speedup | one stream + priority |
| 4. per-phase DVFS | 2.65% median | needs 2.9 s phases. dead |

Levers 1 and 2 are configuration choices a planner already gets to make. That is where
the remaining measurement effort belongs.

---

## 2. The gap that blocks lever 1

`kappa = a/f` is `C*V^2`, the voltage curve:

| f (MHz) | 300 | 510 | 705 | 900 | 1200 | 1410 |
|---|---|---|---|---|---|---|
| `kappa` (x10^-3) | 1.7069 | 1.6890 | 1.6940 | **1.6933** | 2.2455 | 3.1670 |

Flat to 0.5% CV from 300 to 900, then +33% and +41%. The GPU sits at floor voltage up to
somewhere past 900 and ramps after -- and **there are zero calibration points inside
900-1200**.

That single gap is load-bearing three times:

* 58 of 72 pairs pick 900 MHz as energy-optimal, but 900 is the **edge** of the grid.
  An optimum at the edge of a sampling grid is what an unresolved optimum looks like.
  The real one is wherever the voltage starts climbing, which is inside the gap.
* all 39 throttled rows settled between 1170 and 1395 MHz, so every coefficient used to
  re-score them was interpolated across this gap from two endpoints. The residual there
  correlates +0.82 with clock: −7.0% at the bottom, +11.7% at the top.
* the closed-loop solver `f* : P(f*) = cap` searches inside it.

The hardware exposes **15 MHz DVFS steps** (81 supported graphics clocks, 855..1245 in
the region of interest), so the curve can simply be measured.

---

## 3. Designed experiments, in value-per-hour order

### E1 -- voltage knee at 15 MHz resolution   [DONE, 12 min]

`sweep_voltage_knee.py`, `data/voltage_knee.csv` (90 rows). Single GPU, GEMM only,
108 SMs, clocks 510/705/780 then 855..1245 in 15 MHz steps, three shapes.

**The kill criterion did not fire -- the optimum moved, so E2 and E6 are warranted.**

**The floor-voltage region extends to ~1020 MHz, not 900.** `kappa_eff` normalised to
its own 855-900 average holds at 1.00 +/- 0.04 through 1020, then climbs monotonically:
1.03 at 1035, 1.07 at 1050, 1.09 at 1065, 1.11 at 1080, 1.14 at 1095. All three shapes
agree on the location.

**The energy optimum is 945-1065 MHz, and 900 was a grid artifact:**

| shape | best f | E there | E@900 | E@1200 | within 2% of best |
|---|---|---|---|---|---|
| 1024x4096 | 1020 | 51 mJ | +3.5% | +15.0% | 990-1050 |
| 4096x8192 | 1065 (990 smoothed) | 679 mJ | +3.7% | +17.9% | 870-1065 |
| 8192x16384 | 945 | 5258 mJ | +4.7% | +20.4% | 870-975 |

So the coarse grid's 900 MHz recommendation gives up **3.5-4.7%**, on top of the 18.9%
it already saves over 1410. The bottom is broad -- any clock in roughly 950-1050 is
within 2% for all three shapes -- so the practical answer is a band, not a point.

**Two by-products worth keeping.**

*The GEMM is clock-linear to 1245 MHz.* `t*f` is constant to 0.1% over the whole range
(294.2 -> 296.0, 2776.8 -> 2779.3, 20756 -> 20765). No memory-bound saturation anywhere
in the DVFS range, which confirms the `tau*f = const` law at 15 MHz resolution rather
than at the 6 coarse points it was fitted on.

*Two-point interpolation of `P_static` across 900-1200 was wrong by up to 6%*, and
biased low at both ends -- measured 64.3 / 74.1 W at 900 / 1200 against 61.4 / 69.7 W
interpolated. The curve is convex, not linear. This is a direct contribution to the
+0.82 clock-correlated residual in the throttled rescore.

Not resolved here: `a` and `b` separately -- that needs the SM-count axis, which is E2.
`kappa_eff` above folds `b` in, so its shape is the voltage curve but its level is not
`a/f`.

### E2 -- re-fit `a(f)`, `b(f)` through the knee   [~40 min]

**Warranted: E1 moved the optimum.** Reuse the `../gemm_dcfs_char` squatter rig, which varies
SM count and is what makes `a` identifiable at all: clocks {900, 990, 1035, 1080, 1140, 1200} -- straddling the knee E1
put at ~1035 -- x 12 shapes x 5 SM counts (108/96/72/48/24) = 360 rows.

Resolves: `a(f)` where every throttled row lands. E1 already showed `P_static` was off
by up to 6% there and convex where the model assumed linear, so part of the +0.82 clock
correlation is explained and part is presumably `a(f)` doing the same thing.

Note this rig also fixes the 1410 MHz identifiability problem for free -- fewer active
SMs keeps power under the cap, so the clock holds where it did not at 108 SM (33 of 36
rows throttled there, leaving one distinct shape and a degenerate latency fit).

### E3 -- fill the throttling shape gap   [~10 min]

The open question from earlier: is there a case where GEMM and comm each hold clock
alone but throttle when overlapped? The answer was 0 of 432, **but that is a coverage
artifact**. At 1410 MHz the twelve shapes are bimodal -- one at 321 W, eleven clustered
in a 6 W band at 397-404 W, nothing in between. There was no shape in the 340-395 W band
where a +40 W overlap bump would decide anything.

Design: walk `S_eff` through the gap at N=K=4096, where `grid = M/8` so M steps of 256
move occupancy smoothly.

| M | grid | W | S_eff | occ | predicted P at 1410 |
|---|---|---|---|---|---|
| 1024 | 128 | 2 | 64.0 | 59% | 321 W (measured) |
| 1280 | 160 | 2 | 80.0 | 74% | ~393 W |
| 1536 | 192 | 2 | 96.0 | 89% | ~465 W |
| 1792 | 224 | 3 | 74.7 | 69% | ~369 W |
| 2048 | 256 | 3 | 85.3 | 79% | 399 W (measured) |

M = 1280 and 1792 land squarely in the empty band. 5 shapes x 6 CTAs at 1410 only,
gemm_only + comm_only + serial + concurrent.

Verify grids with ncu first -- these M values are new and split-K is not predictable.

### E4 -- the CTA saturation knee   [~30 min]

`B(c,f)` saturates on the product `c*f`, not on either alone: efficiency `B/(c*f)` is
flat at ~8.9e-3 in the unsaturated corner and collapses to 3.2e-3 at 32 CTA / 1410 MHz,
knee around `c*f ~ 15000`. At 900 MHz that is `c ~ 17` -- directly between the 16 and 32
we sampled.

Since CTA count is the **largest single lever** on collective energy (8.6x vs the
clock's 1.5-3.0x), its knee being unresolved is the biggest remaining hole.

Design: collective only, no GEMM. `c` in {12, 16, 20, 24, 28, 32, 40, 48, 64} x 3 clocks
{705, 900, 1200} x 8 message sizes = 216 rows. One process launch per CTA value (NCCL
caches the env var).

Check first whether NCCL honours `c > 32` on this build -- if it caps, the top of the
range is wasted and the design shrinks to {12..32}.

### E5 -- overlap-ratio sweep, to fix the latency blind spot   [~40 min]

The latency model's error is a clean function of how well-matched the two kernels are:

| `min/max` duration ratio | 0.0-0.2 | 0.2-0.4 | 0.4-0.6 | 0.6-0.8 | 0.8-1.0 |
|---|---|---|---|---|---|
| n | 135 | 104 | 42 | 84 | **28** |
| median abs err | 3.84% | 4.46% | 5.92% | 7.21% | **16.27%** |

The worst bin is also the thinnest -- 28 points, all of them accidents of the grid. That
is not enough to fit a contention term.

Design: solve the fitted model for the message size that puts `t_comm/t_gemm` at each of
{0.6, 0.8, 0.9, 1.0, 1.1, 1.25} -- `t_comm` is affine in message size so this is closed
form. 6 ratios x 4 shapes x 4 CTAs x 3 clocks = 288 concurrent cases, deliberately
placed in the blind spot instead of stumbled into.

This is the one place the earlier duration-matched shape design was pointing in the
right direction: matched durations are worth measuring *because* the model is worst
there, not because they are the best operating point.

### E6 -- the planner's recommendation table   [~45 min]

Last, once E1-E2 have fixed the clock grid. Re-run the concurrent sweep at six clocks
chosen around the true optimum rather than the arbitrary original six -- E1 says
{900, 960, 1020, 1080, 1140, 1200} rather than {300..1410}. 6 x 12 x 6 = 432, same
harness as `sweep_overlap_432.py`.

Note E1 measured the GEMM alone. The collective wants a *higher* clock, so the
concurrent optimum should sit above the GEMM's 945-1065 -- by how much is exactly what
this measures.

Output: for each (shape, latency budget), the (f, c) that minimises energy. That is the
artifact the config tool actually consumes.

### E7 -- hard SM partitioning via green contexts   [~half day + 1 h]

Speculative, listed for completeness. Since one GPU has one clock domain, the only way
to give two kernels genuinely different resource allocations is to partition SMs. Today
that is soft: CTA count sets the collective's block count and stream priority biases the
scheduler, but the GEMM's blocks can still occupy SMs the collective is not using at
that instant -- which is exactly why the re-waving term needed a duty-cycle correction.

CUDA 13.0 is installed, so green contexts are available. Hard partitioning would make
the re-waving term exact and is a direct test of it. Whether it helps or costs
throughput is genuinely unknown.

---

## 4. What is NOT worth running

**Per-phase DVFS.** Closed on the numbers in section 1: 2.65% ceiling against a 77 ms
round trip needing 2.9 s phases.

**Memory-clock DVFS.** One supported memory clock. Nothing to sweep.

**Heterogeneous clocks across GPUs.** In tensor parallel all four ranks run the same
GEMM and the ring all-reduce couples their timing, so the slowest rank sets the pace.
Clocking any GPU down slows all four. This could pay under pipeline parallelism, where
bubbles mean some stages genuinely idle -- but that is not testable on a 4-GPU TP box.

**Power cap as an energy knob.** The data says a lower clock is more energy-efficient,
and a cap is a hardware way to enforce that adaptively -- the closed-loop solver already
predicts the resulting clock to −2 MHz median. But changing power caps affects the whole
box and has not been authorised. Flagged, not proposed.
