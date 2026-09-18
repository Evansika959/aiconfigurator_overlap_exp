# B2 — a second spatial V/f domain buys latency granularity, not energy

**Verdict: on this hardware, for MoE prefill, a second spatial V/f domain is worth about
1% of layer energy at the A100's real clock granularity if the partitioned kernel were
free — and it is not free. With the measured cost of building it, it loses to simply
lowering the single clock at every latency it can reach.**

The 1% is not voltage harvesting either: it is the two-domain design landing exactly on a
deadline that a quantised clock grid overshoots, and it shrinks by 10x as the grid
refines. See §3.

This is the negative result the project was set up to be able to reach. It is reached
with a working partitioned persistent kernel, not by argument: the kernel exists, is
bit-identical to vLLM's path, and the partitioning works. What kills the idea is the
silicon, not the software.

Figure: `figs/fig18_two_domain_verdict.pdf`.

---

## 1. The fear was wrong: pinning is nearly free

`PLAN.md` identified the experiment that could kill B2 before any DVFS code existed. A
second voltage domain needs blocks **pinned** to a set of SMs, because a work-conserving
queue has no slack to sell — a block that finishes a lightly-routed expert's tile
immediately takes another expert's tile, so imbalance becomes tail time rather than idle
SMs. Pinning creates the idle SMs we want to run slow, and charges makespan for them.

Measured at a locked 1200 MHz (`pin_vs_queue.py`, `data/pin_vs_queue_qwen.csv`), on the
real ShareGPT routing histogram:

6 independent passes, cases reshuffled each pass (`--repeats 6`):

| | layer time | vs vLLM |
|---|---|---|
| vLLM non-persistent grouped GEMM | 4.051 +/- 0.001 ms | baseline |
| persistent, 1 block/SM | 5.043 +/- 0.003 ms | +24.5% |
| persistent, 2 blocks/SM | 4.240 +/- 0.002 ms | +4.7% |
| **best pinned 2-partition split** | **4.277 +/- 0.001 ms** | **+5.6%** |

**Pinning costs +0.8% over the persistent kernel.** The split was swept, not assumed: an
initial single-pass sweep covered 6 choices of how many experts go in the light partition
x 5 SM allocations either side of the work-proportional split, all re-measured over 6
passes (29 distinct pinned configurations, 192 rows). The work-proportional allocation
wins for every partition size — L8→sm2, L16→sm5, L32→sm13, L48→sm22, L64→sm35,
L96→sm64 — and `pinned_L16_sm5` (4.27687 ms) is the overall best. The margin is thin:
the runner-up is `pinned_L8_sm2` at 4.28179 ms, **0.115% behind**, which is still ~11
standard errors given sds of ~0.001 ms. Quoting the minimum of one-shot measurements, as
a first version of this did, is a winner's-curse estimate biased low; the best split is
now chosen by its mean.

So the thing we were afraid of is not the problem. **Persistence is.**

## 2. Two wrong turns on the way to a fast persistent kernel, both informative

**Occupancy, not scheduling, is what a persistent kernel gives up.** At one resident
block per SM — the crisp "1 CTA = 1 SM" partition `overlap_exp` verified with ncu — the
kernel is 24.5% slower than vLLM's. The non-persistent kernel launches ~15 000 blocks and
the hardware keeps several resident per SM, overlapping one tile's epilogue with the
next tile's loads. One block per SM forfeits that. Raising to 2 blocks/SM recovers almost
all of it (+4.7%), 6 blocks/SM reaches +3.1%.

**The cost: the partition is now enforced by occupancy accounting, not by the ncu-verified
1-CTA-1-SM identity.** Two concurrent launches of 2*S_A and 2*S_B blocks fill the machine
exactly, but nothing guarantees disjointness the way one-block-per-SM does. This is a real
weakening of the claim and it is not yet re-verified with ncu.

**Inter-block locality beats intra-block locality, by a lot.** My first attempt to close
the gap was to give each block a *contiguous* run of tiles, so it would reuse one expert's
weight panel out of L2 instead of refetching. Every tile costs the same, so the split is
still perfectly balanced. It measured **7.883 ms — 56% worse** than the grid stride.

The reason is that **L2 is shared across all 108 SMs**. Under a grid stride every block
is working near the same tile index at any instant, so only a handful of expert weight
panels are live in L2 at once. Under contiguous chunks, 108 different experts are live
simultaneously and L2 thrashes across 1.12 GiB of weights. The kernel keeps the grid
stride, and the comment in `persistent_moe.py` records why so nobody "fixes" it again.

## 3. Scored honestly, two domains win — and the win is the clock grid, not the voltage

**First, an error I made and had to retract.** The first version of `predict_two_domain.py`
scored the single-clock baseline by linear *interpolation* over the clock grid, and the
two-domain design by a *discrete* search with deadline semantics. That is not a
comparison. The two-domain candidate set contains every single clock (the `f_hi == f_lo`
cases), so scoring the superset discretely against a continuous relaxation of itself can
only flatter the baseline — here by a mean of 18.0 mJ and up to 73.0 mJ, against a panel
spanning about 71 mJ. It produced "two domains never win, 0/42", which was an artefact of
the construction and not a finding. Both sides now get the same clock set and the same
deadline rule.

Fixed that way, two domains **do** win — and the size of the win is set by how finely the
single clock is allowed to be set:

**Second, the grid the comparison rests on had holes in it.** The original sweep stepped
45 MHz above 1155 MHz, so 1170, 1185, 1215, 1230, 1260 and 1275 — all real A100 clocks —
were never measured, and anything calling itself "the A100's real 15 MHz grid" was
interpolating exactly where most of its win lay. Those six have now been measured, 6
passes each. Over the plotted latency range (1035–1320 MHz) **every clock on the 15 MHz
grid is measured**; only the 5 MHz row is interpolated, and it is labelled hypothetical.

| single clock restricted to | two domains win | best gain (idle at fast rail / SM-weighted mix) |
|---|---|---|
| 45 MHz — the original sweep's spacing, by subsampling the measured data | 41/42 | −3.40% / −3.94% |
| **15 MHz — the A100's real grid, fully measured** | **41/42** | **−1.45% / −1.67%** |
| 5 MHz — hypothetical, shows the limit | 40/42 | −0.46% / −0.60% |

**The win scales as (clock step)^0.73 ± 0.05.** Fitted on the measured grid thinned to
15/30/45/60/75/90 MHz — no interpolation anywhere, since the plotted deadline range spans
1035–1290 MHz which is entirely inside the region measured at 15 MHz:

| clock step | 15 | 30 | 45 | 60 | 75 | 90 |
|---|---|---|---|---|---|---|
| best gain | 1.67% | 2.56% | 3.50% | 4.42% | 5.28% | 6.12% |

Sublinear, monotone, and it never approaches 1: over **all 462 distinct resamples of the
six passes × all 18 thinning anchors — 8316 refits — the exponent spans 0.516–0.850 and
never reaches 1**, with no distributional assumption anywhere. That, not three curves a
reader eyeballs, is what licenses "it goes to zero in the continuous limit".

A 3σ statement was written here first and withdrawn: 462 distinct resamples resolve no
finer than 1/462 = 2.2e-3, so a p ≈ 1.3e-3 claim sits *below the resolution of the
resampling that produced the σ*. The distribution is also asymmetric (0.516 low tail
against 0.850 high), so a symmetric ± is the wrong shape for a tail argument. The
empirical maximum needs neither.

### The uncertainty had three components and the first two were each quoted as the whole

This went wrong twice in a row, in the same shape, and it is worth recording because the
error is not arithmetic — each number was correct, and each was presented as if it were
the entire uncertainty.

| component | size | when it was found |
|---|---|---|
| within-fit CI of a single thinning anchor | ±0.027 | quoted first, by both analyses |
| spread across the 18 thinning anchors | ±0.028 | quoted second |
| measurement noise (bootstrap over the 6 passes) | **±0.039** | the largest, found last |
| anchor ⊕ noise, in quadrature | **±0.048** | what the figure quotes |

### A fourth instance of the same pattern, and the sharpest

Enumerating all 462 distinct resamples instead of sampling them raised the noise term
from ±0.036 to ±0.046, and it was recorded here as "sampling had understated the largest
component". **That was backwards.**

A bootstrap draws 6^6 = 46 656 equiprobable *ordered* resamples. Collapsed to the 462
distinct multisets, those are emphatically not equiprobable: the all-distinct multiset
has p = 0.01543 and the degenerate one p = 2.14e-05, a ratio of **720x**. Weighting the
462 equally gives the extreme resamples **101x** more weight than they are due, and it
inflates the sd. Weighting each by its multinomial probability gives **±0.0386** — which
is what plain sampling with replacement had been reporting all along (two independent
runs at 0.0370 and 0.0369), because sampling with replacement *is* that weighted
distribution.

So the enumeration was overstating by 20%, not catching an understatement.

**The pattern, four times now: every number was computed correctly, and the error was in
what the label claimed the number was.** The within-fit CI was a real CI. The anchor
spread was a real spread. The equal-weighted enumeration is a real enumeration. Each was
labelled "the bootstrap uncertainty on the exponent", and none of them was.

The one place equal weighting is the right object is the claim it is actually used for:
enumerating all 462 with equal weight is a *wider* net than the bootstrap precisely
because it over-samples the tails, so "the exponent never exceeded 0.85 across 8316
refits" holds a fortiori. `data/two_domain_uncertainty.csv` carries both, under names
that say which is which.

**The thinning has a phase.** Fitting the exponent requires choosing which clock the
thinning starts from, and there are 18 such anchors in 1035–1290 MHz. Two independent
analyses of this same data, each fitting from one anchor, produced **0.684** (anchor
1035) and **0.774** (anchor 1290) — *non-overlapping* 95% intervals, both arithmetically
correct. Neither should have been quoted, because the within-fit CI of one anchor is
narrower than the spread across anchors.

Then the spread across anchors was quoted, and it too is only a part: bootstrapping the
six measurement passes gives ±0.039, larger than the anchor spread. The figure now
quotes the two combined.

A separate 45 MHz "phase check" was removed rather than fixed: its construction had a
free choice — whether the boundary clock 1155 stays in every phase — that moved the
numbers by 0.5 pp and made them irreproducible from a stated rule. The 18-anchor
ensemble *is* the phase check, done uniformly at every step size, and it is the quantity
the uncertainty is built from rather than a side note. It is not voltage harvesting. It is **latency granularity**: with a work split the
two-domain design can land exactly on a deadline that a quantised clock grid overshoots.

Note what the first row is: the *same measured data* subsampled back to the sweep's
original spacing. The ~−4% this project reported two revisions ago was not a property of
the hardware, it was a property of six missing measurements.

**A second protocol defect, and how it was closed.** The six clocks were first filled in
as a *separate* campaign — unshuffled, with no control clocks, run after the original. A
reshuffle cannot reach across campaigns, so a batch offset was unbounded, and propagating
a plausible one moved the 15 MHz headline by +/-0.65 pp — a third of its own value. The
whole grid was therefore re-collected as one reshuffled campaign, which both removes the
problem and measures what the offset had been: **-2.96 mJ**.

`overlap_exp` was bitten by exactly this once before — a 45 MHz sweep grid manufactured a
4.82% win that the real 15 MHz grid turned into a 0.66% loss. Same error class. It is also
why the six clocks were measured rather than interpolated and argued about: a reviewer
predicted, from the convexity of E(f), that linear interpolation would overstate energy by
a mean of +4.84 mJ and up to +20.64 mJ. Measured, the bias is a mean of **−1.99 mJ** with
the sign varying by clock (−13.1 to +10.9). E(f) is not uniformly convex up there — the
400 W cap flattens it — so the reasoning was directionally wrong. Reasoning about a bias
is not a substitute for measuring it.

**Why there is nothing else to win.** From `RESULTS_B1.md` §7: dynamic energy is flat
within 1.0% from 510 to 1050 MHz while total board energy varies 23%. kappa is the entire
prize of a low-voltage domain, and below the knee there is none. Above the knee the saving
is real but equally available to one uniform clock, which pays no makespan tax to get it.

## 4. With the cost of actually building it, it never wins

The handicap is measured, and it is charged to **both** time and energy, because the
measurement shows both: **+5.58% latency and +3.01% dynamic energy** (6 passes, at a
locked 1200 MHz). A first version charged it only to time; a single-pass estimate also
put the energy handicap at +5.55%, which the 6-pass mean corrects to +3.01%.

| | |
|---|---|
| wins | **0 / 31 reachable deadlines**, on all three clock grids |
| best case | **+7.1% worse** than one uniform clock |
| unreachable | the fastest 11 of 42 deadlines cannot be hit at all |
| with **no deadline at all** | still loses, by **41.9 mJ** |

That last row answers the obvious objection to the restricted latency range. The range
does cost the handicapped design something — measured at +3.88%, since its own optimum is
pushed past the range end — but comparing each side's global best, unbounded by any
deadline, it still loses by 41.9 mJ. There is no deadline at which it wins.

That last row matters: a design carrying a +5.7% latency handicap simply cannot reach the
fastest quarter of the latency range, whatever clock it runs.

So the null hypothesis in `PLAN.md` §4 is the one the data supports, with one refinement:
the second domain is worth ~1% at the hardware's real clock granularity if the kernel were
free, and negative once it is not.
## 4b. Does a second domain at least let the GPU dodge the power wall?

A fair objection, and a different objective from §3-4: that analysis optimised *energy at
matched latency*, and never had the 400 W cap as a binding constraint. This one asks
whether a split buys *throughput* under the cap — the cap is real, and the hardware's
response to it is to throttle the whole GPU uniformly to 1320 MHz.

**It does not, and the reason is the same Jensen argument moved to the power side.**

Dynamic power per SM, g(f) = P_dyn(f)/108, is **convex** over the measured 915-1290 MHz
range (quadratic curvature +8.05e-06 W/SM/MHz^2; 14 of 24 second differences positive,
mean +0.0039). For a convex g, Jensen gives

    sum_d x_d g(f_d)  >=  g( sum_d x_d f_d )

so at a fixed power budget the uniform assignment maximises the SM-weighted mean
frequency, and any split gives mean frequency away. Solved directly rather than argued:

| | makespan | power |
|---|---|---|
| fastest uniform clock under 400 W | 3.797 ms | 380.1 W |
| best **strict** split (f_lo < f_hi) under 400 W | 3.798 ms | 379.9 W |
| the hardware's own throttle response | **3.727 ms** | 400.9 W |

A genuine split is **+0.02% slower** for the same power budget. And the hardware beats
both model points, because throttling lands it at an effective operating point *between*
grid clocks that draws exactly the cap — its uniform response is already close to optimal
for a convex g.

Two further checks:

* **The cap never bound the §3-4 analysis.** Of the 42 energy-optimal two-domain configs
  chosen there, **0 would exceed 400 W** (worst 380.1 W). The composition was not
  flattering two domains by ignoring the cap.
* **Where the objection WOULD hold.** A split beats uniform under a power cap when the
  partitions have *different* kappa — different energy per operation at the same clock.
  That is exactly the GEMM-plus-collective case `overlap_exp` measured, where the two
  kernels have genuinely different appetites. Inside one MoE layer every partition runs
  the same grouped-GEMM kernel on the same shapes and differs only in M, so the kappa
  heterogeneity that would make this work is not there. This is the same assumption
  flagged in §6 as the weakest one, seen from the power side.

## 4c. Could a CUSTOM kernel create the heterogeneity a grouped GEMM lacks?

Everything above partitions one grouped GEMM. The objection is fair: the whole reason to
write a custom megakernel is to make the two partitions *different*, and charging both at
the whole layer's kappa assumes that difference away. So it was measured
(`partition_kappa.py`, `data/partition_kappa_qwen.csv`): each partition run **alone** on
54 SMs, one resident block per SM so the SM count is exact, 13 locked clocks, 3 passes.

**Finding 1 — with the same tile, there is no heterogeneity to harvest.** kappa per SM,
light partition over heavy, at the tuned BLOCK_M=128: **1.025** (range 0.978-1.045). The
two partitions differ in how much work they hold and in nothing else. That is the
objection, confirmed: a uniform grouped GEMM cannot feed a second voltage domain.

**Finding 2 — the obvious custom-kernel lever is decisively wrong.** At BLOCK_M=128 the
16 least-loaded experts occupy 3200 padded rows for 1821 real tokens, 76% waste; at
BLOCK_M=16 they need 1936 rows, 6% waste. That looks like a large, DVFS-free saving. It
is not:

| light partition, energy per real token | BM=16 | BM=32 | BM=64 | **BM=128** | BM=256 |
|---|---|---|---|---|---|
| uJ/token at its own best clock | 28.2 | 22.5 | 19.1 | **18.8** | see below |

**A smaller tile costs 50% more energy, not less.** Padding is not the dominant term —
re-reading the expert weight panel is. Cutting BLOCK_M from 128 to 16 takes the light
partition from 700 tiles to 3388, and each tile re-reads the same B panel. The padding
saving of 1264 rows is bought with 4.8x the weight traffic.

BLOCK_M=128 is a genuine interior optimum, not an artefact of the tuner's grid. A first
BLOCK_M=256 measurement showed a 20x collapse, which was a broken configuration rather
than a finding: 256x128 with `num_warps=4` puts ~147 KB in shared memory against the
A100's 164 KB and gives four warps an 8192-output tile, so it spills. Retuned
(`num_warps=8`, `num_stages=3`) it costs **+15.7%** against BLOCK_M=128 — real, modest,
and on the other side of the optimum.

**Finding 3 — both partitions want the same clock anyway.** Energy-optimal clock at
BLOCK_M=128: **1065 MHz light, 1035 MHz heavy**. One 15 MHz step apart, inside the basin
where energy is flat. Even a free split has nowhere different to put the light partition.

### What this leaves

Partitioning **by expert load** gives a custom kernel no lever: not in kappa, not in tile
size, not in optimal clock. The heterogeneity `overlap_exp` measured came from two
genuinely different kernels (GEMM and a collective) with different appetites. Inside one
MoE layer every partition runs the same arithmetic on the same shapes.

If the idea is to be rescued, the split has to be along an axis where the work really
does differ — producer/consumer rather than expert/expert, i.e. SMs doing weight movement
against SMs doing MACs. That is warp specialisation, it is a different partitioning
entirely from the one this project modelled, and on A100 the two halves live in the same
block rather than on separate SMs. Untested here, and it is a Hopper-and-later design.

## 5. Reproduce

```bash
source tools/moe_megakernel/env.sh
cd tools/moe_megakernel
CUDA_VISIBLE_DEVICES=0 python3 test_persistent.py                    # bit-exactness
CUDA_VISIBLE_DEVICES=0 python3 pin_vs_queue.py --clock 1200 --blocks-per-sm 2
python3 predict_two_domain.py
python3 make_b2_fig.py                        # -> figs/fig18_two_domain_verdict.pdf
```

`persistent_moe.py` holds the kernel; `test_persistent.py` checks it against
`vendor/fused_moe_triton.py` whole and partitioned (relative error exactly 0).
