# Optimal (SM split, frequency) for an overlapped GEMM + all-reduce

Answers: **given an arbitrary GEMM and an arbitrary collective, what are the optimal
`#SM` and `f` for each when they overlap** -- under the counterfactual that the two
kernels sit in separate DVFS domains.

Solver: `optimise_overlap.py`. Results: `data/opportunity_2domain.csv` (72 workloads,
12 GEMM shapes x 6 message sizes).

---

## 1. The problem

```
given    GEMM (M, N, K)  and  all-reduce (bytes, world=4)
choose   S_g,  S_c = 108 - S_g,  f_g,  f_c
minimise E  =  P_static(f_g, f_c, S_g) * T
             + P_gemm(f_g, S_g) * t_g
             + P_comm(f_c, S_c) * t_c            T = max(t_g, t_c)
```

`t_g == t_c` is deliberately **not** imposed. If duration-matching is optimal the search
should discover it; assuming it would beg the question.

SMs and frequency are substitutes for hitting a finish time, and the two kernels trade
them at different rates. More SMs to the GEMM lets it clear its work at a lower clock,
which is cheap while `kappa = a/f` is flat below ~1020 MHz -- but the collective then has
fewer SMs and needs a higher clock, which is expensive because `gamma(f)` climbs. The
optimum is where those marginal costs meet.

Three arms, the last two strictly nested so the comparison is clean:

| arm | partition | clocks | what it is |
|---|---|---|---|
| **A** | soft | 1 | what the hardware does today; validated at 3.3% median on 432 measured cases |
| **B** | hard | 1 | all 108 SMs split, both partitions busy, one shared clock |
| **C** | hard | 2 | the counterfactual. **B's search space is contained in C's**, so C can never be worse |

`C` vs `B` is the pure second-DVFS-domain effect. `C` vs `A` also changes soft to hard
partitioning and so mixes two things -- reported, but it is not the clean number.

---

## 2. The answer

**The collective always wants a higher clock than the GEMM.** Across all 72 workloads
the optimum has `f_c > f_g`, by +60 to +690 MHz. What varies is how much that is worth,
and it is predicted by one number: **`t_c / t_g`**, the ratio of the two kernels' solo
durations.

| `t_c/t_g` | n | energy, C vs B | best | EDP, C vs B | median `f_c - f_g` | C vs A |
|---|---|---|---|---|---|---|
| 0 - 0.5 | 45 | −0.16% | −1.3% | +0.00% | +300 | +1.58% |
| 0.5 - 1.0 | 12 | −0.29% | −1.8% | −0.06% | +300 | +2.16% |
| 1.0 - 1.5 | 3 | −0.04% | −3.0% | −0.00% | +120 | +1.60% |
| **1.5 - 3.0** | 6 | **−4.33%** | −8.1% | **−8.98%** | +300 | −2.58% |
| **> 3.0** | 6 | **−8.56%** | −9.9% | **−11.02%** | +690 | −7.88% |
| overall | 72 | −0.34% | **−9.90%** | | | +1.35% |

**The second domain pays only when the collective is on the critical path.** When the
GEMM dominates -- 60 of these 72 workloads, and the regime a 64 MiB all-reduce against a
large GEMM sits in -- it is worth essentially nothing.

### Why, mechanically

When the collective is the bottleneck the GEMM has **slack**. Two domains let you spend
that slack by dropping the GEMM's voltage while the collective runs flat out. One domain
forces both to the collective's clock, which is far above the GEMM's optimum. Concretely,
at `1024x4096` with a 256 MiB all-reduce:

| | f | c | S_g | energy |
|---|---|---|---|---|
| B, one clock | 1200 both | 32 | 76 | 306.9 mJ |
| C, two clocks | **f_g 510, f_c 1200** | 32 | 76 | **280.8 mJ** (−8.5%) |

Same SM split, same collective settings. The entire saving is the GEMM dropping from
1200 to 510 MHz using time it had spare anyway.

### Three objectives, and why the tie-break decides the answer

Minimising energy alone is not what a planner wants -- it gives up a median 37% latency
to save 19%. Minimising latency alone ignores the power bill. EDP is a compromise whose
weighting has no physical meaning. So all three are reported (`objectives.py`,
`data/objectives.csv`).

**Each is optimised lexicographically, and both E and T are reported under every
objective.** This matters more than it sounds. Under a pure latency objective every arm
reaches the SAME minimum T -- just set all clocks to maximum. Comparing arms on T alone
therefore shows 0.00% and hides the entire result, which is that *at that same T* the
arms differ enormously in energy. An earlier version of this analysis made exactly that
mistake and concluded the second domain was useless under a latency objective. It is the
opposite: that is where it is worth the most.

`C vs B`, energy, 48 workloads (12 shapes x 4 message sizes):

| `t_c/t_g` | n | objective E | objective T | objective EDP |
|---|---|---|---|---|
| 0 - 0.5 | 29 | −0.13% | +0.00% | +0.00% |
| 0.5 - 1.5 | 9 | −0.20% | −0.10% | −0.04% |
| **1.5 - 4** | 7 | **−5.50%** | **−23.92%** | **−10.25%** |
| **> 4** | 3 | **−9.71%** | **−21.61%** | **−8.52%** |
| best single workload | | −9.90% | **−26.36%** | −13.41% |

Latency is unchanged in every cell (`dT` is 0.00% throughout, and −19.65% in one bin
where C also happens to finish sooner). So under a latency objective the second domain
buys **21-26% energy at identical latency** in comm-bound workloads.

### Why: it is the energy share of the NON-critical kernel

Under a latency objective the critical-path kernel must run at maximum clock. With one
domain the *other* kernel is dragged to that same clock even though it has slack. With
two domains it drops to whatever just meets the deadline. The saving is the energy the
non-critical kernel wastes by running faster than it needs to -- so it scales with how
much of the total energy that kernel holds.

| critical path | non-critical kernel | its energy share | C vs B |
|---|---|---|---|
| comm | GEMM | 33-87% | −9% to −26% |
| GEMM | collective | 1-13% | 0% to −0.3% |

Correlation between non-critical energy share and the win: **−0.89** over 24 workloads.

That also corrects an earlier reading of the same data. The relevant slack is not
*budget* slack -- how much longer than `T_min` you allow -- but slack in the
**non-critical kernel within a fixed makespan**, which exists at `T_min` already. The
solver's refusal to duration-match (median match ratio 0.50) is consistent with this:
it is not trying to fill idle SMs, it is trying to run the slack kernel as slowly as the
deadline permits. An idle SM draws leakage only; a slack kernel running at full clock
burns dynamic power for nothing, and that is the larger waste by an order of magnitude.

## 3. What is measured and what is not

Every coefficient comes from the two calibration campaigns. Two get used outside their
fitting range and are corrected:

**`eta(S) = 1 + 0.118*(1 - S/108)`.** `tau` and `t0` were fitted on 108-SM rows only, but
the solver's whole job is choosing `S_g`, so `t_gemm(f, S)` at `S < 108` is load-bearing.
Tested out of sample against the squatter campaign's 14/27/54/81 SM rows: the uncorrected
wave model under-predicts latency by 6-9% below 54 SMs, an occupancy effect (it tracks
SM count, not clock or wave count). The correction takes median error from **6.62% to
1.75%**. Power needed no correction -- it generalises at 1.6-4.3% median at every SM count.

**`c <= 32`.** `B(c,f)` is measured at `c` in {1,2,4,8,16,32} and `B/c` is already falling
at 32, so interpolating between samples is sound but extrapolating past it is not. **8 of
72 optima sit at the cap**, so those are bounds, not optima -- the true optimum may want
more collective SMs.

**Static power with two voltage domains is the one unmeasurable quantity.** Split it as
`P_floor + D(f)` with `P_floor` the 300 MHz value (uncore, HBM, PCIe -- chip-wide) and
`D(f)` the voltage-dependent leakage, apportioned by SM share. At `f_g == f_c` this
collapses exactly to the measured single-domain `P_static(f)`, so arms A and B are
unaffected and only C carries the assumption. Tested against two alternatives:

| static model | C vs B median | best |
|---|---|---|
| apportion by SM share | −2.15% | −9.71% |
| charge the higher domain chip-wide | −0.49% | −5.36% |
| charge both in full (hard upper bound) | −0.61% | −9.58% |

The magnitude moves; the pattern does not. The win stays concentrated in the comm-bound
workloads under every assumption.

---

## 4. The gap, and the experiment that closes it

**The opportunity lives exactly where the measurements are thinnest.** The 432-case
sweep fixed the message size at 64 MiB, where all but the smallest shapes are
GEMM-dominated -- `t_c/t_g` below 0.5 for most of it. The comm-bound corner where the
second domain is worth 4-10% was never measured.

**Arm B is measurable, and it is the whole validation.** Hard partitioning needs no new
hardware: build the GEMM so `grid = k*(108-c)` -- a whole number of waves on exactly the
SMs the collective is not holding, no partial wave -- and the chip is split with nothing
idle. Arm C is then arm B plus one modelled step (change `f_c`), so validating B carries
almost all the confidence C needs.

Proposed run:

* **Comm-bound region.** Small GEMMs (M = 1024, 2048 at N=K=4096) x message sizes
  {64, 128, 256} MiB, giving `t_c/t_g` from 1.6 to 8.4. This is where the model claims
  4-10% and has never been checked.
* **Hard partition.** `grid = k*(108-c)` for `c` in {8, 16, 32}, verified with ncu before
  running -- the last hand-built shape set failed grid verification 3 of 5, and an
  assumed grid has already cost 21% once.
* **Sweep `f` at the resolved grid.** Use the clocks the voltage-knee sweep found, not
  the original six.
* Arms measured: `gemm_only`, `comm_only`, `serial`, `concurrent`, per (shape, c, size, f).

Estimated 250-300 measurement windows, ~30 minutes.

What it settles: whether arm B's energy matches prediction in the comm-bound regime.
What it cannot settle: arm C, on one clock domain -- that stays a model result.
