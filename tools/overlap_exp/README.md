# overlap_exp — a bottom-up energy model for overlapped GEMM + NCCL all-reduce

4x A100-SXM4-40GB, world = 4, TensorRT-LLM shapes. Everything here answers one question:
**when a GEMM and a collective share a GPU, what SM split and what clock(s) minimise
energy — and how much would a second voltage/frequency domain be worth?**

Every figure is matplotlib at NeurIPS camera-ready spec, emitted as PDF (the
deliverable for LaTeX) plus a 600 dpi PNG companion, regenerated from the
data by the script named beside it. Nothing here is hand-drawn.

## Layout

`GOALS.md` states what the project is for and where it stands; this file is the map.

| | |
|---|---|
| `*.py`, `run_*.sh` | the live pipeline — measurement, model, analysis, figures |
| `data/` | measurements and derived tables (`data/README.md` maps them) |
| `figs/` | generated PDFs and PNGs, one pair per `make_*` entry below |
| `docs/` | narrative write-ups and the rendered 432-case table |
| `archive/` | superseded scripts, kept for provenance. Nothing live imports them. |

## Pipeline

**1. Measure** (touches the GPU; locks clocks, so do not run two at once)

| script | what it produces |
|---|---|
| `sweep_overlap_432.py` | THE harness. Every campaign is this file plus `SWEEP_*` env overrides — clocks, CTAs, shapes, message size, modes. `SWEEP_CLOCKS=0` means do not lock, i.e. record what the governor picks. |
| `measure_case_spans.py` | one config with per-stream CUDA events on a common timeline: each kernel's span *inside* the overlapped iteration |
| `sweep_voltage_knee.py` | 15 MHz-resolution sweep for kappa = P_dyn/f |
| `probe_dvfs_latency.py` | NVML clock-change round-trip |
| `demo_overlap.py` | the reference overlap pattern: two fresh streams, collective at priority -3 issued first, one common gating event |
| `run_commbound.sh` `run_auto.sh` `run_ext_campaign.sh` | the three sweep campaigns |
| `run_repeat_case.sh` `run_repeat_cta.sh` | 10x repeats of one cell, both sides re-measured |

**2. Model**

| script | what it is |
|---|---|
| `table_432.py` | the fitted GEMM + collective model: coefficient tables, `lin()` interpolation, `predict()`. Imports `score_432.py`. |
| `followup_baseline.py` | `compose(g, c, f_gemm, f_comm, grid)` — build a two-domain candidate from measured solo kernels — and `pick()`. Every analysis below goes through it. |

**3. Analyse**

| script | question |
|---|---|
| `fixed_design.py` | SM split welded shut: what frequencies, what does the 2nd domain buy, what does committing to one split cost |
| `analyse_commbound.py` | validates `B(c,f)` past its calibration, then the two-clock gain on the comm-bound set |
| `matrix_errorbars.py` | Monte-Carlo propagation of measured single-window noise through the whole selection |
| `optimise_overlap.py` + `objectives.py` | the joint (S, f) solver, three objectives (energy / latency / EDP) |
| `analyse_knee.py` | locates the voltage knee |
| `export_data.py` | dumps the model's intermediate tables |

**4. Figures** — `figstyle.py` holds the style, `validate_palette.py` checks the palette
(the six dataviz checks, ported to Python because this box has no Node).

| script | figures |
|---|---|
| `make_figures.py` | fig1 mechanism, fig2 objectives, fig3 rule, fig4 frontier, fig5 paired, fig6 knee, fig7 comm-bound matrix, fig9 case |
| `make_fixed_fig.py` | fig10 — frequency map at each of 9 fixed SM splits |
| `make_split_fig.py` | fig11 — split-commit cost, one line per message size |
| `make_summary_fig.py` | fig12 — the three findings of the extension campaign |
| `make_err_figs.py` | fig14 — the matrix with per-cell uncertainty; fig15 — one cell as bars |
| `build_cell_fig.py --ctas N` | fig16 — any cell as power x time, from its 10 repeats. Only `--ctas 4` and `--ctas 8` regenerate: `data/repeat_cta32/` was not kept, so the committed `fig16_cell_4096x4096_32cta.png` predates the current data layout and has no PDF. |

## Reproduce a result

    python3 fixed_design.py && python3 make_fixed_fig.py          # the fixed-split study
    python3 matrix_errorbars.py && python3 make_err_figs.py       # matrix + error bars
    python3 build_cell_fig.py --ctas 32 --dir data/repeat         # one cell, measured 10x

## Standing constraints of this box

4 GPUs max. Clock locking needs persistence mode. Do not change power caps. Long sweeps
run in the background. `NCCL_{MIN,MAX}_CTAS` is cached on first read, so one process
launch per CTA value; the GCP gIB shim must come off `LD_LIBRARY_PATH` or it rejects
those variables outright.
