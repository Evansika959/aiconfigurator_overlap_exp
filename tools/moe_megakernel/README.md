# `moe_megakernel` — spatial DVFS for MoE inference

Successor to `../overlap_exp`. Same box (4x A100-SXM4-40GB), same measurement stack, a
different source of heterogeneity: **expert routing imbalance inside one kernel**, rather
than two concurrent kernels with different appetites.

**Every measurement here ran on GPU 0 alone.** A Qwen3-30B-A3B MoE layer holds 1.12 GiB
of expert weights against 40 GB, so nothing forces a split, and adding expert parallelism
would put NCCL traffic and a second synchronisation point inside every energy number.

---

## The result

**The hypothesis is false, and the reason is measurable.** Routing imbalance manufactures
imbalance in *how much work* a partition holds, not in *what a unit of that work costs*.
Every partition of a MoE layer is bound by the same resource — on-chip operand movement,
which the SM clock drives — so the cold and hot partitions have the same energy-vs-clock
curve and a second voltage domain has nothing to harvest.

| | measured |
|---|---|
| pinning blocks to SM sets | **+0.9%** latency — nearly free |
| making the kernel persistent (what pinning requires) | **+4.7%** — an occupancy effect |
| cold vs hot partition, kappa per SM at the same tile | **1.025** — no heterogeneity |
| dynamic energy, 510-1050 MHz | **flat within 1%** — no voltage to harvest |
| best two-domain config, free kernel, real clock grid | **-1.53%** |
| with the measured build cost | **0 wins in 31 deadlines**, +7.1% at best |

What does work is boring and large: **one fixed clock at 1050 MHz, -24% energy against
the default governor**, no new kernel, one number for every batch size from 64 to 4096
tokens (max regret 0.12%).

Full reasoning in `RESULTS_B2.md`; the baseline it rests on in `RESULTS_B1.md`.

---

## Start here

| if you want | read |
|---|---|
| the plan, the three-rung ladder, what is still open | `PLAN.md` |
| the honest fused baseline and the first real MoE energy measurement | `RESULTS_B1.md` |
| why the two-domain design fails, with the mechanism | `RESULTS_B2.md` |
| how the routing traces are produced, and what may be quoted from them | `INFRA.md` |
| routing skew across five MoE models | `RESULTS_QWEN.md` |

Earlier phases, superseded but kept because their measurements still stand:
`RESULTS_PHASE0.md`, `RESULTS_VOLTAGE_FLOOR.md`, `RESULTS_ROOFLINE.md`,
`RESULTS_FALSIFICATION.md`.

---

## Kernels

| file | what it is |
|---|---|
| `vendor/fused_moe_triton.py` | vLLM v0.11.0's Triton `fused_moe_kernel`, **verbatim**; align step and driver rewritten here. See `vendor/README.md` for line-level provenance and the Apache-2.0 notice. |
| `persistent_moe.py` | **ours.** One resident block per SM, each handed an explicit tile list, so SM sets are chosen by us rather than by the hardware scheduler. Output **identical** to the vLLM path (`relmax = 0`), whole and partitioned. |
| `test_fused_moe.py` | vendored kernel vs a naive per-expert fp32 loop, 6 shapes |
| `test_persistent.py` | persistent kernel vs the vLLM path, whole and partitioned |

vLLM itself is not installed: it ships a compiled `_C` pinned to its own torch, which
would drag a second torch onto `PYTHONPATH` — the failure `env.sh` documents. The Triton
kernel has no vLLM imports and lifts out cleanly.

---

## Measurement scripts

| script | produces | what it measures |
|---|---|---|
| `trace_routing_sharegpt.py` | `data/routing_*.npz` | per-(rep, layer, expert) token counts, 5 model families |
| `b1_bench.py` | `data/b1_config_*.json` | autotunes the Triton config; vLLM ships none for A100 at `E=128, N=768` |
| `b1_energy.py` | `data/b1_energy_*.csv` | locked-clock energy sweep, 38 clocks + governor, reshuffled each pass |
| `pin_vs_queue.py` | `data/pin_vs_queue_*.csv` | work-conserving vs persistent vs pinned, one clock |
| `partition_kappa.py` | `data/partition_kappa*.csv` | **each partition alone**, swept over clock and tile size |
| `batch_clock_sweep.py` | `data/batch_clock_*.csv` | energy-optimal clock as a function of batch size |
| `predict_two_domain.py` | `data/two_domain_*.csv` | composes every (f_hot, f_cold, split) at matched latency |
| `analyse_skew.py` | `data/skew_across_models.csv` | excess Gini, null-corrected, across models |

Earlier-phase collectors, still the source of their sections:
`baseline_b0.py`, `kappa_sweep.py`, `idle_power.py`, `roofline_and_tiles.py`,
`slo_tightness.py`, `wave_sawtooth.py`, `expert_clock_sweep.py`, `stage_clock_sweep.py`,
`search_split.py`, `predict_expert_dvfs.py`, `trace_routing.py`, `analyse_routing.py`.

---

## Figures

Every figure is produced by one script and ships as PDF (the deliverable) plus a 600 dpi
PNG. `figstyle.py` holds the shared NeurIPS camera-ready style — import it, do not
re-derive.

| figure | script | shows |
|---|---|---|
| **fig17** `b1_energy` | `make_b1_fig.py` | the layer's measured energy and latency vs clock; the U-shape is idle amortisation |
| **fig18** `two_domain_verdict` | `make_b2_fig.py` | the two-domain gain scales with clock step and vanishes |
| **fig19** `tile_size` | `make_tile_fig.py` | a smaller tile costs 50% more energy, and why |
| **fig20** `expert_routed_dvfs` | `make_expert_dvfs_fig.py` | the scheme, the curves that kill it, the kernel's price (+ `_appendix` with all four splits) |
| **fig21** `same_knee` | `make_knee_fig.py` | cold and hot differ in level (1.77x), not in shape or knee |
| **fig22** `sweep_heatmap` | `make_sweep_3d.py` | every enumerated (f_hot, f_cold, split) as a 3D surface: height = energy vs the deployed governor, colour = the latency it costs |
| fig16 `skew_across_models` | `make_skew_fig.py` | routing skew across five MoE models |
| fig15 `excess_gini`, fig11 `routing_shapes` | `make_gini_fig.py`, `make_routing_dist_fig.py` | the skew distributions behind it |
| fig4, fig5, fig14 | `make_kappa_fig.py`, `make_kappa_sweep_fig.py`, `make_knee_full_fig.py` | the voltage floor: kappa is flat below the knee |
| fig1, fig2, fig3 | `make_concept_fig.py`, `make_prefill_fig.py`, `make_why_fig.py` | the premise and the timescale problem |
| fig6-fig9 | `make_slo_fig.py`, `make_jensen_fig.py`, `make_sawtooth_fig.py`, `make_waves_fig.py` | SLO slack, Jensen, wave quantisation |
| fig10, fig12, fig13 | `make_pred_fig.py`, `make_pareto_fig.py`, `make_pareto30_fig.py` | **superseded** early predictions, composed before B1 existed. Kept for provenance; do not quote — `RESULTS_B2.md` has the measured versions. |

---

## Reproduce

```bash
source tools/moe_megakernel/env.sh          # PYTHONPATH -> ./pylibs; see the two pip traps
cd tools/moe_megakernel

CUDA_VISIBLE_DEVICES=0 python3 test_fused_moe.py         # correctness
CUDA_VISIBLE_DEVICES=0 python3 test_persistent.py        # bit-exactness, incl. partitioned

CUDA_VISIBLE_DEVICES=0 python3 b1_bench.py --model qwen --tune          # ~12 min
CUDA_VISIBLE_DEVICES=0 python3 b1_energy.py --model qwen --repeats 6    # ~35 min, locks clocks
CUDA_VISIBLE_DEVICES=0 python3 pin_vs_queue.py --clock 1200 --blocks-per-sm 2 --repeats 6
CUDA_VISIBLE_DEVICES=0 python3 partition_kappa.py --light-experts 16 --repeats 3
python3 predict_two_domain.py               # ~4 min, exhaustive bootstrap

for f in make_*.py; do python3 "$f"; done   # all figures
```

Scripts that lock clocks use `sudo nvidia-smi -i 0` and always reset in a `finally`.

---

## Conventions this folder keeps

- **Measure, don't reason, about bias.** Three findings in `RESULTS_B2.md` reversed when
  the thing being argued about was measured instead: an interpolation bias predicted from
  convexity had the opposite sign; an "exhaustive bootstrap" over-weighted its tails; a
  padding argument turned out to be about on-chip movement, not DRAM.
- **Never compare a discrete search against a continuous relaxation of itself.** Both
  sides get the same clock set and the same deadline rule.
- **A coarse measurement grid manufactures results.** `overlap_exp` lost a 4.82% "win"
  this way; this folder lost a 3.60% one. Report the sensitivity, don't pick a grid.
- **Quote uncertainty in full.** The scaling exponent had three components and the first
  two were each quoted as the whole before the third was found.
