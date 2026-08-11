# A100 bf16 GEMM energy model — fitted coefficients

Fitted on `data/gemm_dynamic_energy.csv` (1080 rows = 12 shapes × 6 clocks × 5 SM
counts × 3 reps), using only rows whose **achieved** SM clock matched the locked one.
SM count is set by an `__nanosleep` squatter, verified per configuration by a
147456 B occupancy probe. Static power is subtracted using zeus's `profile_p2p.py`,
run unmodified at each locked clock.

## Structure

```
P_board = P_static(f)  +  P_dyn(shape, S, f)
E_dyn   = P_dyn × latency          <- the number to record; latency is measured, not modelled
```

`P_static` is drawn whether or not a GEMM runs, so it is **not** attributable to the
kernel. It is a median 47 % of board power across this grid (18 % at the hot corner,
84 % at 300 MHz / 14 SM).

## Dynamic power

```
P_dyn = a(f) · S_eff  +  b(f, footprint)

S_eff = tiles / ceil(tiles / S)                      time-averaged busy SMs
tiles = ceil(M / tile_m) · ceil(N / tile_n)          K does NOT enter the grid
```

`S_eff`, not `S`, is the regressor. The last wave of a GEMM is usually partial and the
SMs it does not fill draw no dynamic power. Substituting `S_eff` for `S` cuts the
spread of `a` across the 12 shapes from 43 % to 24 % and raises the median from 2.51
to 2.71 W/SM at 1200 MHz — most of the apparent shape dependence of `a` was occupancy,
not silicon.

### Both halves of S_eff are confirmed by hardware counters, not inferred

`ncu` (see `profile_occupancy.py`), on four shapes:

| shape (M, N) | tiles predicted | `launch__grid_size` | S_eff/S predicted | `sm__cycles_active.avg / sm__cycles_elapsed.max` |
|---|---|---|---|---|
| 1024, 4096 | 128 | **128** | 0.5926 | **0.5855** |
| 2048, 4096 | 256 | **256** | 0.7901 | **0.7753** |
| 4096, 8192 | 1024 | **1024** | 0.9481 | **0.9348** |
| 8192, 16384 | 4096 | **4096** | 0.9981 | **0.9845** |

Grid size matched exactly every time, confirming the 256×128 tile read off the kernel
name. Occupancy matched within 2 % across a 0.59–0.98 range, consistently ~1.4 % low —
the idealised wave model has no launch ramp-up or drain.

Note `torch 2.10/cu130` selects `..._128x256` where `torch 2.9/cu129` selects
`..._256x128`. The tile COUNT is unchanged (`M·N/32768` either way when both divide),
so `S_eff` transfers; measured latency differs by ~4 %.

### a(f) — per-SM dynamic power

| f (MHz) | 300 | 510 | 705 | 900 | 1200 | 1410 |
|---|---|---|---|---|---|---|
| **a (W/SM)** | 0.4992 | 0.8405 | 1.1747 | 1.4940 | 2.6658 | 4.4212 |
| standard error | ±0.0066 | ±0.0093 | ±0.0132 | ±0.0129 | ±0.0303 | ±0.0707 |
| κ = a/f (nW/SM/Hz) | 1.664 | 1.648 | 1.666 | 1.660 | **2.222** | **3.136** |
| implied V / V(900) | 1.00 | 1.00 | 1.00 | 1.00 | **1.16** | **1.37** |

κ is `C·V²` in `P = C·V²·f`. It is **flat to within 1 % from 300 to 900 MHz** — one
voltage domain — then rises, which is the DVFS voltage curve read straight out of the
power data. This conclusion is robust to the model form: a quadratic fit (below) gives
κ = 1.906/1.917/1.910/1.935 over the same four clocks.

1410 MHz is fitted from the 54/27/14 SM points only: at 108 and 81 SM the board hit the
400 W SwPowerCap and ran at 1245–1380 MHz instead.

### b(f) — the part that does not scale with SM count

L2 on this part is 40 MB. `b` orders by weight-matrix footprint, so it is HBM traffic
power.

| f (MHz) | N=K=4096 (32 MB) | N=K=8192 (128 MB) | N=K=16384 (512 MB) |
|---|---|---|---|
| 300 | 4.5 ± 0.5 | 8.5 ± 0.4 | 9.7 ± 0.5 |
| 510 | 3.7 ± 0.5 | 9.9 ± 0.4 | 11.8 ± 0.6 |
| 705 | 3.7 ± 0.9 | 11.1 ± 0.7 | 14.3 ± 0.9 |
| 900 | 3.3 ± 0.8 | 12.8 ± 0.8 | 16.2 ± 0.9 |
| 1200 | 0.2 ± 2.2 | 12.1 ± 1.6 | 16.3 ± 1.3 |
| 1410 | 1.8 ± 2.7 | 16.7 ± 1.7 | 19.0 ± 2.6 |

`b` is obtained by extrapolating to `S_eff = 0`, so its standard error is large; entries
within ~2σ of zero should be read as zero. At 108 SM `b` is ~8 % of dynamic power, at
14 SM about a third of it — which is why dynamic energy is not perfectly flat along the
SM axis, and why the model is least accurate at low SM counts.

`b` also carries the squatter's own power (0.9 W at 300 MHz to 7.2 W at 1410), because
the subtraction uses zeus's bare-board number end to end rather than a
squatter-matched baseline. That choice was deliberate — one method throughout — and it
cannot move `a`, only the intercept. See "Choice of static baseline" below.

### Optional quadratic term

On the largest shape (M=8192, N=K=16384, 4096 tiles, where quantisation loss is ≤0.8 %
at every SM point so `S_eff ≈ S`), the linear fit leaves a residual that is negative at
both ends of the SM axis and positive in the middle. Adding `c·S_eff²`:

| f (MHz) | a (W/SM) | c (W/SM²) | b (W) | R² | residual RMS |
|---|---|---|---|---|---|
| 300 | 0.5717 ± 0.0352 | −0.00073 ± 0.00029 | 8.47 | 0.99718 | 0.88 W |
| 900 | 1.7415 ± 0.0265 | −0.00233 ± 0.00021 | 10.93 | **0.99982** | **0.67 W** |
| 1200 | 2.7535 ± 0.0862 | −0.00116 ± 0.00070 | 15.36 | 0.99942 | 2.16 W |

`dP/dS = a + 2cS`: at 900 MHz the 108th SM contributes 1.238 W against the first SM's
1.742 W, a 29 % fall. The concavity is strongest on the shapes that stream the most
from HBM (c = −0.00233 and −0.00280 at N=K=16384 versus −0.00075 to +0.00030 at
N=K=8192), consistent with shared-bandwidth saturation. Note `a` means different
things in the two fits — average slope over the range, versus the first SM's marginal
power — so do not mix the tables.

## Static power

zeus `profile_p2p.py` run unmodified at each locked clock (`run_p2p_static.py` →
`data/p0_static_zeus.csv`), 3 reps, spread 0.13–0.33 W.

| f (MHz) | 300 | 510 | 705 | 900 | 1200 | 1410 |
|---|---|---|---|---|---|---|
| **P_static (W)** | 56.37 | 57.60 | 59.38 | 61.36 | 69.66 | 85.37 |

Whole GPU module — all 108 SMs, HBM, L2, memory controllers, interconnect, regulator
losses. NVML reports one number per board and per-SM static power is not separable
from it; it does not need to be, since this is subtracted from the board power measured
during the GEMM.

`+5.0 W` from 300 to 900 MHz and `+24.0 W` from 900 to 1410 — the same voltage step
that shows up in κ.

### Choice of static baseline

Two methods were measured. zeus's (a rank parked in a blocking NCCL recv) reads a
near-constant **3.0–3.4 W below** the `__nanosleep` squatter reading at every clock
(`data/p0_static.csv`, 6 clocks × 8 occupancies × 3 reps, 144/144 held their clock).

Because that offset is flat along the SM axis it lands entirely in `b` and cannot move
`a`. Measured: switching baselines moves `b` by +3.5 to +5.0 W and `a` by −1.6 %, and
that 1.6 % is itself an artefact of the squatter table's step between n=0 (no resident
kernel) and n≥1 — dropping the n=0 point makes the two baselines agree on `a` to 0.00 %
at four of five clocks. The zeus baseline is therefore the cleaner one for the slope.

A squatter-matched alternative is carried per row as `e_dyn_squat_mj`, so the cost of
the choice is visible in the data: 0.8 % of E_dyn at 108 SM, 11.2 % at 14 SM.

### Temperature — measured, then deliberately left out

`P_static(f, T) = P_static(f, T_ref) + γ(f)·(T − T_ref)`, with measured
**γ = 0.260 W/°C at 300 MHz and 0.345 W/°C at 1200 MHz**
(`profile_static_temp.py`, `data/p0_static_vs_temp.csv`, fitted over 34–44 °C,
R² 0.99+). The table above was taken at 33–40 °C; the sweep spanned 39–68 °C.

**Not applied, and it does not need to be.** Propagating each row's own ΔT moves
`E_dynamic` by a median of **2.8 %** (p90 4.7 %, max 18.5 %; 8 of 1080 rows exceed
10 %). The two effects that could have made this matter anti-correlate: the hot rows
are the high-clock, many-SM ones where dynamic power is large, while the rows where
static dominates barely heat the die at all.

Note for anyone re-measuring γ: the die falls 62 → 38 °C in **under 5 seconds** once the
GEMM stops, so the informative range passes almost immediately. Sampling must start
~1.2 s after the load stops (NVML power needs ~427 ms to settle, which bounds it from
below) and run at ≥20 Hz. A 5 s settle window discards the whole range.

## Measured accuracy

R² of the per-(shape, clock) fits is 0.9983 median, 0.9811 worst — that is evidence the
model FORM is right, **not** a prediction error. Three levels, each more honest than the
last, on the 900 rows with a held clock:

| | | n | median | p90 | max |
|---|---|---|---|---|---|
| 1 | per-(shape, clock) fit, **in sample** | 900 | 1.59 % | 6.20 % | 26.8 % |
| 2 | predict every row from the aggregate `a(f)`, `b(f,N=K)` tables | 900 | 2.65 % | 9.76 % | 39.1 % |
| 3 | **leave-one-shape-out** — fit on 11 shapes, predict the 12th | 900 | **3.29 %** | **10.21 %** | 39.7 % |

**Quote level 3.** Level 1 flatters the model: each (shape, clock) gets its own two
parameters for its own five points.

Where the error concentrates, both explainable:

* **By shape.** `M=1024, N=K=8192` (256 tiles) is the outlier at 13.3 % median / 29.6 %
  p90; the other eleven are 1.3–5.6 %. `b` is grouped by `N=K`, but its three
  group-mates have 512/1024/2048 tiles — so `b` depends on tile count too, and grouping
  by footprint alone is a crude proxy.
* **By SM count.** 14 SM is 7.00 % median / 15.4 % p90 against 2.1–3.6 % elsewhere,
  because `b` is a third of dynamic power there and `b` is the badly-determined term.

Usable at 2–4 % for mainstream operating points (≥27 SM, ≥512 tiles). The single
highest-value improvement is replacing "`b` grouped by N=K" with a continuous function
of tile count or actual HBM traffic; `a` is not the bottleneck.

Coverage caveat: these numbers are over the 900 rows with `clock_held=True`. The 1410 MHz
plane's 108 and 81 SM cells hit the 400 W cap and ran at 1245–1380 MHz, so they are
excluded.

## What this replaces

"Running on 70 % of the SMs burns 70 % of the dynamic power" is the `b = 0`,
`S_eff = S` version of the above. It scores R² 0.979 where this model scores 0.999, and
it fails hardest at low SM counts — where `b` dominates — and on shapes with few tiles,
where the last wave is mostly empty.
