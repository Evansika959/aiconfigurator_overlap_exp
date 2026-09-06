# data

Schema of every sweep file is the harness's own: `ctas, clock, m, n, k, mode, grid,
waves, s_eff, ar_mib, iter_ms, iters, clock_min, clock_held, power_node_w,
power_per_gpu_w, n_samples, world`. `mode` is one of `gemm_only`, `comm_only`,
`serial`, `concurrent`. **`clock_held == False` means the GPU throttled off the
requested clock; every analysis drops those rows.** `clock == 0` means the clock was
not locked, i.e. the row records what the governor chose.

## Measured — the four campaigns

| file | rows | what it is |
|---|---|---|
| `overlap_432.csv` | 972 | the original sweep: 6 clocks x 12 shapes x 6 CTAs, 64 MiB |
| `commbound_{128,256,512}mib.csv` | 384/348/348 | comm-bound campaign: 6 clocks x 6 shapes x 4 CTAs, three message sizes |
| `ext_{64,128,256,512}mib.csv` | 552/384/384/384 | the extension: splits 2..64 CTA, clocks incl. 1050 and 1305, plus 64 MiB (GEMM-led) |
| `auto_*mib.csv`, `ext_auto_*mib.csv` | 24 / 54 each | the same configs run **unlocked** — the only honest baseline |
| `repeat/`, `repeat_cta4/`, `repeat_cta8/` | 10 x 2 files each | one cell measured ten times, BOTH sides (`spans_*.json` = the real overlapped run with per-kernel spans; `solo_*.csv` = the solo kernels the composition reads) |

## Measured — supporting

| file | rows | what it is |
|---|---|---|
| `voltage_knee.csv` | 90 | kappa = P_dyn/f at 15 MHz resolution; the knee is at ~1035 MHz |
| `throttle_matrix.csv` | 432 | mean-vs-peak throttle criterion (mean wins: 88.6% precision vs 31.5%) |
| `throttle_gap.csv` | 142 | the 340-395 W window, ~6 SMs wide |
| `dvfs_transition.csv` | 7 | NVML clock-change round trip, 37.7-38.5 ms one way |
| `grid12.json` | 15 | **ncu-measured** `launch__grid_size` per shape. Not a formula — `(1024,8192,8192)` split-Ks 2-way and assuming otherwise cost 21%. |
| `coefficients.csv` | 134 | the fitted model coefficients |

## Derived — regenerate, do not hand-edit

| file | rows | produced by |
|---|---|---|
| `fixed_design.csv` | 11232 | `fixed_design.py` — every (split, workload, f_gemm, f_comm) |
| `fixed_design_{gain,penalty}.json` | 9 | same; feeds the fig10 panel titles |
| `fig7_err.csv` | 72 | `matrix_errorbars.py` — per-cell Monte-Carlo sd |
| `ladder.csv`, `commbound_analysis.csv` | 72 | `analyse_commbound.py` |
| `followup_baseline.csv` | 72 | `followup_baseline.py` |
| `objectives.csv`, `frontier.csv`, `paired.csv` | 48/61/48 | `export_data.py`, `objectives.py` |
| `table_432.csv`, `errors_432.csv` | 432 | `table_432.py` (tracked in git) |
| `fig9_case.json` | — | the fig9 workload, kept because fig9 is the only figure that reads it |

## Other

`logs/` raw sweep stdout, kept for provenance only — nothing reads it.
`cases/` one-off span/case intermediates from superseded scripts in `../archive/`.
`archive/` superseded tables.
`overlap_w4.csv`, `peak_vs_mean.csv`, `phase1_partial.csv` are phase-0/1 files tracked in
git; their scripts now live in `../archive/`.
