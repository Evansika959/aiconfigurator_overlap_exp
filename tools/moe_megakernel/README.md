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
| the same question in the event-driven simulator (`sim/`, 2-4% from 5000 measured times) | second rail: **-2.3 to +0.7 points** of energy at equal latency, **-0.2% to +0.0%** of latency at 400 W: inside the simulator's error, not distinguishable from zero |
| boosting the domain still running when the other finishes (`sim.boost`) | **0.0-0.1%** of latency at the best split; needs an actuator faster than 0.3 ms |

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
| the event-driven simulator: how far it is from every measurement, and what it says a second V/f domain is worth | `RESULTS_SIM.md`, `sim/README.md` |
| the simulator's test that was registered before its data existed | `HELDOUT.md` |
| the node engine: GEMM + all-reduce overlap and multi-GPU steps, what was measured for it (launch order and the shared-memory carveout, link power, per-GPU clocks) and how far it is from every measurement | `RESULTS_NODE.md` |
| its registered blind test | `OVERLAP_HELDOUT.md` |
| extending the model to multi-GPU training: what changes, what is built, what is not | `MULTIGPU.md` |
| the simulator written up as a paper section | `paper/sec_simulator.md` |
| routing skew across five MoE models | `RESULTS_QWEN.md` |
| the same with GPT-OSS-20B added, and what it needed | `INFRA.md` (last section), `figs/fig16c_skew_with_gptoss.png` |

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
| `trace_routing_sharegpt.py` | `data/routing_*.npz` | per-(rep, layer, expert) token counts, 6 model families (`routing_gptoss.npz` and its control `routing_gptoss_raw.npz` added 2026-10-01, with `--rows` and `--no-template`). `routing_llama4.npz` was traced too (`--sparse-moe`) and is not part of the comparison: `INFRA.md` |
| `b1_bench.py` | `data/b1_config_*.json` | autotunes the Triton config; vLLM ships none for A100 at `E=128, N=768` |
| `b1_energy.py` | `data/b1_energy_*.csv` | locked-clock energy sweep, 38 clocks + governor, reshuffled each pass |
| `pin_vs_queue.py` | `data/pin_vs_queue_*.csv` | work-conserving vs persistent vs pinned, one clock |
| `partition_kappa.py` | `data/partition_kappa*.csv` | **each partition alone**, swept over clock and tile size |
| `batch_clock_sweep.py` | `data/batch_clock_*.csv` | energy-optimal clock as a function of batch size |
| `predict_two_domain.py` | `data/two_domain_*.csv` | composes every (f_hot, f_cold, split) at matched latency |
| `decode_wall.py` | `data/decode_governor_*.csv`, `decode_corun_*.csv`, `limiter_step_*.csv` | decode under the real governor (where 400 W binds), per-partition clock elasticity when cold and hot run TOGETHER, and the power limiter's step response. Read with `analyse_decode_wall.py` |
| `decode_wall.py scan` | `data/sm_corun_scan_*.csv`, `sm_corun_holdout*_*.csv`, `sm_corun_lowclk_*.csv`, `sm_corun_xmodel_*.csv` (`--config-from qwen`: three other models, never fitted) | **the simulator's calibration sweeps**: each partition alone on 16-108 SMs and beside its partner on six GPC-aligned splits, 510-1410 MHz, plus the whole layer on both kernels |
| `decode_wall.py idle` | `data/idle_sweep_*.csv` | board power with nothing running at every clock 210-1410 MHz, one method, die temperature recorded |
| `decode_wall.py pinned` | `data/pinned_layer_*.csv` | the pinned layer END TO END (both partitions, activation, sum), one block per SM: what the simulator's event engine is checked against |
| `decode_wall.py occupancy` | `data/occupancy_*.csv` | the unpartitioned persistent kernel at 1-3 resident blocks per SM beside the queue kernel |
| `ncu_counters.py` | `data/ncu_counters_*.csv` | hardware counters (tensor instructions, L1/L2 traffic, DRAM bytes) for 100 kernels: what the simulator's demand decomposition is checked against. Needs `sudo ncu` |
| `governor_trace.py` | `data/governor_trace_*_{replays,nvml,segments}_*.csv` | a scripted load that MOVES, under the unlocked governor: latency of every graph replay and every NVML reading on one time axis |
| `heldout_plan.py` | `data/heldout_plan.json` | the two load scripts of the registered test, drawn from a fixed seed before anything was measured (`HELDOUT.md`). The frozen simulator that replayed them is `data/heldout_frozen_sim.tar.gz`; what it printed is in `data/heldout_first_shot/` |
| `analyse_hiclk.py` | (reads `partition_kappa_qwen_hiclk54.csv`) | the full-width curve above 1290 MHz rebuilt from a 54-SM sweep, checked against the governor |
| `predict_tdp.py` | `data/tdp_two_domain.csv` | the same design under the 400 W limit: binary-searches the highest f_hot the cold domain's headroom buys, up to 1410 MHz (RESULTS_B2 §4b-ii) |
| `analyse_skew.py` | `data/skew_across_models_v2.csv` | excess Gini, null-corrected, across models (`skew_across_models.csv` is the earlier five-family table, left as checked in) |

Earlier-phase collectors, still the source of their sections:
`baseline_b0.py`, `kappa_sweep.py`, `idle_power.py`, `roofline_and_tiles.py`,
`slo_tightness.py`, `wave_sawtooth.py`, `expert_clock_sweep.py`, `stage_clock_sweep.py`,
`search_split.py`, `predict_expert_dvfs.py`, `trace_routing.py`, `analyse_routing.py`.

---

## Figures

Every figure is produced by one script and ships as a 600 dpi PNG (no PDF since
2026-10-01). `figstyle.py` holds the shared NeurIPS camera-ready style — import it, do
not re-derive.

| figure | script | shows |
|---|---|---|
| **fig17** `b1_energy` | `make_b1_fig.py` | the layer's measured energy and latency vs clock; the U-shape is idle amortisation |
| **fig18** `two_domain_verdict` | `make_b2_fig.py` | the two-domain gain scales with clock step and vanishes |
| **fig19** `tile_size` | `make_tile_fig.py` | a smaller tile costs 50% more energy, and why |
| **fig20** `expert_routed_dvfs` | `make_expert_dvfs_fig.py` | the scheme, the curves that kill it, the kernel's price (+ `_appendix` with all four splits) |
| **fig21** `same_knee` | `make_knee_fig.py` | cold and hot differ in level (1.77x), not in shape or knee |
| **fig22** `sweep_heatmap` | `make_sweep_3d.py` | every enumerated (f_hot, f_cold, split) as 3D pillars: height = makespan vs the deployed baseline (what the design costs), colour = energy vs that baseline (what it buys) |
| **fig22b** `sweep_cold_gpc` | `make_sweep_3d.py --cold-sms 16` | the same sweep with the cold V/f domain FIXED at one GPC (16 SMs) and each side's experts pinned: the side that finishes first idles, makespan is the slower side |
| **fig22c** `sweep_gpc_round` | `make_sweep_3d.py --round-gpc 16` | the split follows the work but is rounded to whole GPCs (16 SMs, 16-96 for the cold side); work stays pinned, so the rounding shows up as idle time |
| **fig22c_row** `sweep_gpc_round_row` | `make_sweep_3d.py --round-gpc 16 --layout row` | fig22c as one row of four panels, 7.0 in (two-column figure*) |
| **fig23** `sim_architecture` | `make_sim_arch_fig.py` | the dynamic model as designed: a discrete-event fluid simulator, its inputs by abstraction level and outputs by time scale. Drawn before `sim/` existed; the implementation follows it. 7.0 in (two-column figure*) |
| **fig23b** `sim_simple` | `make_sim_simple_fig.py` | fig23 simplified: the five models (demand, rate, bandwidth, power, controller), inputs, outputs, calibrate and validate. Superseded as the overview figure by fig23c/fig23d |
| **fig23c** `sim_flow_slide` | `make_sim_flow_slide.py` | the simulator's event loop, steps 1-5, in SLIDE format (13.33 x 7.5 in, sans-serif), after five rounds of review. For talks only; its paper version is fig23d |
| **fig23d** `sim_event_loop` | `make_sim_flow_fig.py` | fig23c in paper format (5.5 in, serif, 7-8 pt): Figure 1 of `paper/sec_simulator.md`. Same blocks and connections, plus the power limiter as a fourth handler, which it became in `sim/engine.py` after fig23c was drawn |
| **fig24** `sim_validation` | `make_sim_validation_fig.py` | the simulator against every measurement: parity for time and power, and the share of points within a given error, by rung of the validation ladder |
| **fig25** `sim_trace2`, `sim_trace`, `sim_dynamics`, `sim_forward`, `sim_forward2` | `make_sim_trace_fig.py --tag trace2` | the simulator against the A100 on a load that moves: clock, board power and time per layer over 30-47 s under the stock governor, one figure per scripted plan (`forward`: whole passes over all 48 MoE layers) |
| **fig25** `sim_heldout1`, `sim_heldout2` | `make_sim_trace_fig.py --tag heldout1 --first-shot` | the same for the two plans registered before they were run, as the FROZEN simulator replayed them (`HELDOUT.md`; series in `data/heldout_first_shot/`). `..._second_shot`: the working tree's replay, without the flag |
| **fig26** `sim_ladder` | `make_sim_ladder_fig.py` | the control groups B0-B4 as Pareto fronts in (time, energy): what the kernel, the pinning and the second rail are each worth |
| **fig27** `sim_layer_events` | `make_sim_layer_fig.py` | one layer inside the event-driven simulator, with one clock and with the cold domain slowed: who runs when, and the board power between events |
| **fig28** `sim_load_step` | `make_sim_step_fig.py` | the first 400 ms after a load step, one point per graph replay: what the power limiter does inside NVML's 100 ms, measured and simulated |
| **fig29** `nccl_occupancy` | `make_node_figs.py --only 29` | SMs a cuBLAS GEMM loses per NCCL CTA, by which kernel was launched first (probe), and what launching behind a GEMM costs the all-reduce |
| **fig30** `node_validation` | `make_node_figs.py --only 30` | the node engine against the earlier overlapped runs: iteration time, board power, and the share of runs within an error |
| **fig31** `overlap_spans` | `make_node_figs.py --only 31` | one overlapped iteration, measured and simulated spans, both launch orders: a cell the rules explain and the one they do not |
| **fig32** `unbalanced_step` | `make_node_figs.py --only 32` | an unbalanced training step on four GPUs as the three light ones are slowed: step time, node power, energy, measured and simulated, against no clock lock |
| **slide1-5** `sim_why`, `sim_how`, `sim_validation`, `sim_results`, `sim_boost` | `make_sim_deck.py` | the simulator and what it found as SLIDES in fig23c's format (13.33 x 7.5 in, sans-serif, 200 dpi): why it was built, one layer event by event, the blind test and the errors against the A100, what a second V/f domain is worth, and one layer at the power cap with the hot domain boosted when the cold one finishes (a split unbalanced on purpose). Every number is read from `sim/calib`, `sim/out` and `data/heldout_first_shot`, or simulated when the slide is drawn. For talks only |
| fig16 `skew_across_models` | `make_skew_fig.py` | routing skew across five MoE models |
| fig16c `skew_with_gptoss` | `make_skew_gptoss_fig.py` | fig16 with GPT-OSS-20B (32 experts) added; excess Gini in the legend |
| fig16b `skew_six_models` | `make_skew_gptoss_fig.py --with-llama4` | the same with Llama 4 Scout too. Not used: Llama 4 is outside the comparison |
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

for f in make_*.py; do python3 "$f"; done   # all figures (fig24-28 need sim/out, below)
```

The simulator (`sim/`, no GPU):

```bash
bash sim/reproduce.sh                       # calibrate, validate, control groups, boost, fig24-28; ~45 min
python3 -m sim.run sim/configs/decode2048_b4.yaml
python3 cross_study_check.py --models       # what of the calibration carries over to cuBLAS, NCCL and
                                            # other MoE models, and ../overlap_exp replayed as events; ~10 min

# the node engine (GEMM + all-reduce overlap, several GPUs); measurements: ../overlap_exp/sweep_spans.py
python3 -m sim.fit_kernels                  # kernel profiles -> sim/calib/a100_kernels.json; ~7 min
python3 -m sim.test_node                    # 12 invariants
python3 -m sim.validate_overlap             # every table of RESULTS_NODE.md section 5; ~15 min
python3 -m sim.heldout_overlap score        # the registered blind test (OVERLAP_HELDOUT.md)
python3 make_node_figs.py                   # fig29-32
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
