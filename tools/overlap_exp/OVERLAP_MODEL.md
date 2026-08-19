# Concurrent GEMM + all-reduce: latency and power model

Predicts the wall-clock latency and the mean board power of a GEMM overlapped with an
NCCL all-reduce on one A100-SXM4-40GB, from configuration alone.

**Input** `(M, N, K, clock f, NCCL CTA count c, message size S, world size 4)`
**Output** `t_iter` in ms and `P` in W per GPU.

No measured timing enters. Validated against 432 measured concurrent cases
(`data/overlap_432.csv`), none of which were used to fit any coefficient.

---

## 1. The model

### Shape constants

```
grid    = ncu-measured launch grid for (M,N,K)          -> data/grid12.json
splitK  = grid / (M*N/32768)                            tile area is always 32768
W       = ceil(grid / 108)                              waves on a free GPU
W'      = ceil(grid / (108 - c))                        waves with c SMs held by NCCL
S_eff   = grid / W                                      time-averaged busy SM count
```

`grid` must be measured, not computed. `M*N/32768` is right for 11 of the 12 shapes
here but wrong for `(1024, 8192, 8192)`, which splits K two ways. Assuming split-K=1
cost a 21% latency error and, through it, a 51% power error. It is a property of the
shape and the cuBLAS heuristic, not of the run, so a lookup table keeps the model
config-only.

### Latency

```
t_gemm = W * (K / splitK) * tau(f)  +  t0(f)
t_comm = alpha(c,f)  +  S / B(c,f)
t_iter = max( t_gemm * [1 + (W'/W - 1) * min(1, t_comm/t_gemm)],  t_comm )
```

The `t_iter` form charges the GEMM for the SMs the collective takes away, but only for
the fraction of the GEMM the collective is actually resident. Charging it for the whole
GEMM over-predicts by 37% when a 32-CTA collective (0.6 ms) meets a 17 ms GEMM.

### Power

```
P = P_static(f)  +  [ P_gemm * t_gemm  +  P_comm * t_comm ] / t_iter

P_gemm = a(f) * S_eff  +  b(f, N*K)
P_comm = p0(f) + gamma(f) * B(c,f)
```

`P_static` is the P0 idle floor and is counted once over the whole iteration, including
the sync gap where neither kernel runs. `P_gemm` and `P_comm` are charged only while
their kernel is resident. The result is a time-average because NVML integrates over a
window far longer than one iteration; comparing an overlap-interval mean against a loop
average over-predicted by 142 W once.

---

## 2. Coefficients

### GEMM power — `../gemm_dcfs_char`, 1080 rows

| f (MHz) | `a(f)` W/SM | `P_static(f)` W | `b` @32 MiB | @128 MiB | @512 MiB |
|---|---|---|---|---|---|
| 300 | 0.5121 | 56.37 | 5.0 | 7.0 | 8.1 |
| 510 | 0.8614 | 57.60 | 4.4 | 7.9 | 9.6 |
| 705 | 1.1943 | 59.38 | 4.4 | 9.1 | 11.5 |
| 900 | 1.5240 | 61.36 | 4.3 | 10.7 | 12.8 |
| 1200 | 2.6947 | 69.66 | −0.1 | 8.1 | 13.1 |
| 1410 | 4.4655 | 85.37 | 3.2 | 14.9 | 19.4 |

`b` is indexed by the **N·K weight footprint in MiB** (`N*K*2/2^20`), log-interpolated
between the three anchors. It is the residual intercept after `a` is fixed, so `a` and
`b` are collinear and only meaningful as a pair — a 1% shift in `a` moves `b` by ~3 W.
The −0.1 W cell is that collinearity pushing an intercept slightly negative; the
footprint term is weakly identified at small footprints.

### GEMM latency — same campaign, full-SM rows

| f (MHz) | `tau(f)` (ms per wave per unit K) | `t0(f)` (µs) |
|---|---|---|
| 300 | 110.837e−6 | 110.6 |
| 510 | 65.213e−6 | 69.1 |
| 705 | 47.182e−6 | 52.3 |
| 900 | 36.964e−6 | 41.5 |
| 1200 | 27.733e−6 | 32.7 |
| 1410 | 23.592e−6 **(extrapolated)** | 28.9 **(extrapolated)** |

`tau * f` is constant to **0.03% CV** across 300–1200 MHz — the GEMM is purely
clock-linear. That law is what the 1410 MHz row is extrapolated on.

**1410 MHz GEMM latency is not calibrated.** 33 of its 36 full-SM rows throttled,
leaving one distinct shape, so a two-parameter fit is unidentifiable: it returns
`tau = 0` with `t0` absorbing the whole latency, giving every shape the same 0.206 ms.
The extrapolation lands +7.9% against that one surviving shape. Treat 1410 as
out-of-range.

### Collective — `../comm_dvfs_char`, 288 rows, world=4

`B(c,f)` algorithmic GB/s. Median relative standard error 0.35%, per-cell R² ≥ 0.99926.

| f (MHz) | 1 CTA | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| 300 | 2.68 | 5.37 | 10.76 | 21.45 | 43.02 | 74.63 |
| 510 | 4.43 | 8.87 | 17.62 | 34.03 | 63.51 | 77.98 |
| 705 | 5.91 | 11.86 | 23.29 | 43.72 | 78.28 | 89.49 |
| 900 | 7.40 | 14.85 | 28.86 | 54.11 | 96.50 | 107.67 |
| 1200 | 9.63 | 19.36 | 37.12 | 68.66 | 119.66 | 139.13 |
| 1410 | 11.15 | 22.37 | 42.60 | 77.69 | 128.87 | 144.72 |

`alpha(c,f)` fixed cost, µs:

| f (MHz) | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| 300 | 108.7 | 131.8 | 163.5 | 210.1 | 265.9 | 286.9 |
| 510 | 83.2 | 94.8 | 108.0 | 137.0 | 172.2 | 147.0 |
| 705 | 62.9 | 74.7 | 82.9 | 98.1 | 133.4 | 110.7 |
| 900 | 49.6 | 57.3 | 58.8 | 79.5 | 102.9 | 86.0 |
| 1200 | 48.1 | 46.4 | 44.1 | 58.3 | 77.0 | 71.8 |
| 1410 | 35.5 | 39.8 | 34.4 | 46.0 | 64.4 | 53.9 |

Collective power, `P_comm - P_static = p0(f) + gamma(f) * B(c,f)`:

| f (MHz) | 300 | 510 | 705 | 900 | 1200 | 1410 |
|---|---|---|---|---|---|---|
| `p0` W | 24.7 | 24.1 | 23.7 | 24.1 | 24.6 | 24.9 |
| `gamma` W per GB/s | 0.1850 | 0.2210 | 0.2189 | 0.2254 | 0.2583 | 0.3192 |

One CTA is one SM exactly (ncu: grid = c, 544 threads/block). `B` saturates on `c*f`,
not on either alone — efficiency `B/(c*f)` is flat at ~8.9e−3 in the unsaturated corner
and collapses to 3.2e−3 at 32 CTA / 1410 MHz, knee around `c*f ~ 15000`.

**World size 4 only.** Rescaling to world=2 by the ring volume `2(N-1)/N` failed by 18%
(implied ratio 1.22 against the predicted 1.00).

---

## 3. Accuracy

432 measured concurrent cases: 6 clocks × 12 shapes × 6 CTA counts, 64 MiB all-reduce.
393 held their clock; 39 throttled against the 400 W board cap and are excluded from
the accuracy figures (the model has no throttling term).

| quantity | median | p90 | max | bias |
|---|---|---|---|---|
| power, config-only | **3.30%** | 10.30% | 35.60% | +4.09% |
| latency, config-only | **5.37%** | 15.22% | 32.49% | −1.0% |
| power, given measured timings | 2.96% | 6.36% | 12.06% | −0.02% |

The third row isolates the power model: it is nearly unbiased and the gap to the first
row is entirely the latency model leaking in, since `t_iter` sits in the denominator of
the energy average — a −20% timing error becomes a +25% wattage error.

Power error by clock:

| f (MHz) | 300 | 510 | 705 | 900 | 1200 | 1410 |
|---|---|---|---|---|---|---|
| n | 72 | 72 | 72 | 72 | 72 | 33 |
| median | 2.90% | 3.00% | 2.85% | 2.80% | 5.85% | 9.00% |
| max | 11.70% | 12.50% | 16.60% | 18.10% | 29.30% | 35.60% |

By occupancy (the wave term's own test — this run is the first to span it):

| `S_eff/108` | 59% | 79% | 95% | 100% |
|---|---|---|---|---|
| n | 36 | 36 | 252 | 108 |
| median | 1.79% | 2.48% | 3.70% | 3.00% |

### Throttle prediction

The model's use case is deciding whether an overlap will throttle. Ground truth is
"the SM clock fell below the locked value during the window".

| predictor | TP | FP | FN | TN | precision | recall | accuracy |
|---|---|---|---|---|---|---|---|
| **mean P > 400 W** | 39 | 5 | 0 | 388 | **88.6%** | **100%** | **98.8%** |
| peak P > 400 W | 39 | 85 | 0 | 308 | 31.5% | 100% | 80.3% |

**Use the mean, not the instantaneous peak.** The peak model over-flags by 3×. The
power controller integrates over ≥12 ms while a GEMM wave is 0.1–1 ms, so a full-wave
peak never survives its window. The wave staircase is real and it sets `S_eff`, but the
quantity the cap acts on is the time-average.

The classes separate with a gap: no throttled case has a model mean below 418 W, no
held case above 400 W, and all five false positives sit in the 400–423 W band.

### Overlap payoff

Median speedup **1.179×** (best 1.702×), a median **65%** of the achievable
`(t_gemm+t_comm)/max(t_gemm,t_comm)` ceiling. Energy per iteration falls a median
**8.6%** (best 29.6%, worst −4.8%).

---

## 4. Known limitations

**Matched durations are the blind spot.** When `t_gemm ≈ t_comm` the `max()` says the
two finish together and the hardware disagrees. Latency error by duration ratio
`min/max`:

| ratio | 0.0–0.2 | 0.2–0.4 | 0.4–0.6 | 0.6–0.8 | 0.8–1.0 |
|---|---|---|---|---|---|
| n | 135 | 104 | 42 | 84 | 28 |
| median \|err\| | 3.84% | 4.46% | 5.92% | 7.21% | **16.27%** |

There is contention here that no SM-accounting form captures — an SM-area bound and a
duty-weighted re-wave were both tried and neither fixes this bin.

**1410 MHz GEMM latency is extrapolated**, see above.

**No throttling term.** The model predicts unconstrained power; when that exceeds the
cap the hardware clocks down and the model is simply wrong by however much it
over-predicted. Told the achieved clock instead of the requested one, error on the
throttled set drops from 14.08% to 6.15% median. A closed-loop version would need to
solve for the fixed point `f` such that `P(f) = cap`.

**One rep per cell.** No run-to-run error bar. A separate repeatability file on the
collective campaign gives median 0.11% / max 2.43% on latency.

**Single message size** in the validation set (64 MiB). Message size does not enter
`P_comm`, only the duty cycle, so this exercises no uncalibrated coefficient — but it
also means the duty-cycle arithmetic was tested over the range the 12 shapes happened
to produce, not a designed one.

**`b` extrapolates flat** outside 32–512 MiB and is weakly identified at the small end.

---

## 5. Reproducing

```
python3 sweep_overlap_432.py     # ~45 min, 972 measurement windows -> data/overlap_432.csv
python3 score_432.py             # power model given measured timings + throttle matrix
python3 errors_432.py            # per-case three-way error -> data/errors_432.csv
python3 table_432.py             # config-only model -> data/table_432.csv, table_432.md
```

All four refit every coefficient from `../gemm_dcfs_char/data/gemm_dynamic_energy.csv`
and `../comm_dvfs_char/data/comm_trtllm_ar_ctas.csv` at import; nothing is hardcoded.
