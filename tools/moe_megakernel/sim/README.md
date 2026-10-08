# `sim/` — an event-driven power and performance simulator for MoE inference on a GPU with V/f domains

One question: **what does a MoE layer cost, in time and in joules, on a GPU whose SMs are
split into V/f domains — when no such GPU exists?** The A100 has one domain. Everything
that can be measured on it is measured (`../data`), fitted (`calibrate.py`) and checked
(`validate.py`); the two-domain numbers are what the fitted model says when its clocks
are allowed to differ, with every assumption that step needs written down as a parameter.

Nothing in this directory touches a GPU.

## Run it

```bash
source ../env.sh            # from tools/moe_megakernel
python3 -m sim.calibrate --compare                    # ../data -> sim/calib/a100_qwen.json  (~2 min)
python3 -m sim.validate                               # every measurement replayed, error per rung
python3 -m sim.validate_trace --tag trace2            # a moving load, simulated vs measured
python3 -m sim.run sim/configs/decode2048_b4.yaml     # one scenario -> four tables
python3 -m sim.study --cells 32                       # the control groups B0-B4
python3 -m sim.sensitivity --cells 32                 # ... under every unmeasurable assumption
python3 -m sim.boost --cells 64                       # boosting the domain that is left running
python3 -m sim.policies --tag trace2                  # one measured load under eleven policies
python3 -m sim.test_sim                               # invariants, no data needed
bash sim/reproduce.sh                                 # all of the above and fig24-28
```

## What goes in

One YAML file, four parts (`configs/`):

| part | what it says | defined in |
|---|---|---|
| `hardware` | V/f domains and their SM counts, TDP, the power limiter, how long a clock set-point takes to arrive, what the shared logic's clock follows | `hardware.py` |
| `kernel` | queue (vLLM) or persistent (ours) per-SM costs, tile size, how many experts are cold (`n_cold: auto` rebalances per layer) and how many SMs they get | `engine.py` |
| `controller` | who sets the clocks: `governor`, `fixed`, `boost_on_finish`, `tdp_bisection`, `closed_loop`, `min_energy_slo`, `offline_optimal` | `control.py` |
| `workload` | batches: prefill or decode, how many tokens, which MoE layers, how often, with what gaps. Routing is always a real traced one | `workload.py` |

```yaml
hardware:
  domains: [{name: cold, sms: 32}, {name: hot, sms: 76}]
  tdp_w: 400
  actuation_ms: 38
kernel: {type: persistent, block_m: 64, n_cold: 64}
controller: {type: tdp_bisection}
workload:
  - {phase: decode, cells: 32, layers: [0], repeat: 2000}
```

`--set controller.type=governor` overrides any field from the command line.

## What comes out

Four tables at four time scales, and a summary (`out/<name>/`):

| file | one row per | columns |
|---|---|---|
| `events.csv` | event | time, kind (`arrive`, `start`, `finish`, `setpoint`, `layer_end`, ...), batch, layer, stage, who |
| `trace.csv` | interval between two events | power, each domain's clock, what was running |
| `layers.csv` | MoE layer | latency, energy, mean power, clocks, when each partition finished each GEMM |
| `windows.csv` | reported reading (100 ms) | window power, sensor power, clock ceiling, clocks, requests |
| `summary.json` | run | time, energy, peak reading, time above the limiter's peak, mean clocks, steady-state latency / power / energy per layer |

## How it works

**Demand** (`demand.py`). A partition of a layer is four counts, all from the routing
histogram: live experts, real pairs, row-blocks, padded rows. From them: weight-panel
bytes the SMs load, MACs (real and padding apart), bytes that must come from DRAM.

**Rate** (`rate.py`). On n SMs a partition's tile list is walked in windows of n tiles.
A window takes as long as the per-SM time it asks for:

```
n * t  =  kP(f) * panel / (1 - rho_L)  +  kR(f) * MACs  +  L0(f) * wait / (1 - rho_D)
rho_D  =  all DRAM traffic / BW_D
rho_L  =  all L2 traffic / BW_L(f_uncore)  +  xi(f) * rho_D ** q
```

`kP, kR, L0` are per-SM costs of the partition's own clock. DRAM and L2 delivery are
shared by every SM whatever its domain, so two partitions side by side slow each other
through `rho_D` and `rho_L`; `solve_rates` is that fixed point. Every window is solved to
convergence, which matters: stopped early, the fit has no single optimum.

Nine fitted numbers: four for the per-SM costs, `BW_L` and its exponent in the clock,
`xi` and its exponent, and `q`. `BW_D` is pinned at the A100-40GB's 1555 GB/s, because
left free together with `q` it runs away. Which parameters are free is a FORM of the
model (`calibrate.FORMS`); `sim.calibrate --compare` fits each form on the first sweep
alone and scores it on the two collected later, and the form in use is the one that
predicts those best.

**Power** (`power.py`). Board power is a sum over activity, not a table of
configurations:

```
P = P_idle(clocks) + act(clocks)
  + sum over running jobs:  g(f) * (e_panel * panel/s + e_mac * real MACs/s + e_pad * padding MACs/s)
                          + e_dram * DRAM bytes/s
```

`g(f)` is the supply voltage squared relative to its floor, one fitted number per clock.

**Engine** (`engine.py`). Between two events every rate and the power are constant, so
time jumps from event to event and energy is integrated exactly. An event is a batch
arriving, a kernel launch reaching the GPU, a partition finishing, a clock set-point
taking effect, the limiter's loop, or a reported power reading. Exactly one handler runs
per event, and after it the rates of everything still running are solved again
(`../figs/fig23d_sim_event_loop.png`). A partition finishing is the event a V/f-domain
policy can act on: `boost_on_finish` raises the domain that is still running until the
stage's barrier releases, and `sim.boost` says what that is worth.

**Limiter and controllers** (`hardware.py`, `control.py`). The limiter is part of the
hardware and applies to every configuration alike: every 20 ms it reads the board's
mean power; above 1.05 x TDP the clock ceiling drops by 0.45 MHz per watt of excess and
reaches the chip 10 ms later, so a load step is absorbed in about 60 ms; inside the band
the excess is accumulated as energy and every 4 J moves the ceiling one 15 MHz step.
That is what the A100 was measured doing, with the latency of every graph replay as the
clock's witness. Controllers only request clocks, and see a reading every 100 ms.
Model-based ones call `Engine.predict` — the plant's own model on one layer — so a
policy cannot look good through a mismatch between the model it plans with and the one
it is scored on.

## Extending it

| to add | where | what to write |
|---|---|---|
| a GPU | a scenario's `hardware:` block, or `hardware.py` | SM count, clock range, TDP, limiter, domains: every field of `Hardware`. Its fitted numbers are a separate file (`calibration:` in the scenario), produced by `calibrate.py` from that GPU's sweeps |
| a model | `SHAPES` in `demand.py` | its layer dimensions and its routing trace in `../data`. `decode_wall.py scan --model X --config-from qwen` collects a sweep; `sim.validate` scores it (rung 3b) |
| a kernel implementation | `kernels` in the calibration file | a `KernelCal`: per-SM costs and tiles in flight per SM. `kernel: {type: <its name>}` uses it; the engine does not change |
| a controller | `control.py`, or any module of yours | subclass `Controller`, give it a `name`, decorate with `@register`. The hooks and the three calls it may make are in `control.py`'s docstring |
| something a one-domain chip cannot tell us | a field of `Hardware`, and a `Variant` in `sensitivity.py` | the field with its default, where the model reads it, and one row that sets it the other way |
| a form of the rate model | `FORMS` in `calibrate.py` | which parameters it pins; `--compare` cross-validates it against the others |
| a policy to compare | `POLICIES` in `policies.py` | one `Policy(name, kernel, controller, ...)` |
| a validation check | `validate.py` | a function that calls `Ladder.row`, called from `run` |
| a measurement | `meas.py` | a loader that returns records; list its file in `TRAIN` to have it fitted |

`test_sim.py` exercises the controller and kernel extension points.

## Files

| file | what |
|---|---|
| `demand.py` | model shapes, traced routing, and `Partition`: routing histogram -> windows of tiles with their bytes and MACs; checked against ncu |
| `rate.py` | windowed service-time model, shared DRAM and L2, `solve_rates` |
| `power.py` | activity-based board power, static power with domains |
| `hardware.py` | domains, limiter, actuation delay, the unmeasurable parameters |
| `engine.py` | the event loop, the layer's four stages, the predictor |
| `control.py` | the seven controllers |
| `workload.py` | scenario batches -> layers with real routing |
| `meas.py` | loaders for every measured CSV; `TRAIN` names what is fitted |
| `calibrate.py` | the five fits -> `calib/a100_qwen.json` |
| `validate.py`, `validate_trace.py` | the validation ladder; the moving-load trace |
| `study.py`, `sensitivity.py`, `policies.py` | control groups B0-B4; the unmeasurable assumptions; one moving load under many policies |
| `boost.py` | what boosting the surviving domain is worth, against fixed per-domain clocks |
| `calib.py`, `files.py` | the calibration file's container; CSV and JSON helpers |
| `run.py` | command line |
| `report.py` | the tables of `../RESULTS_SIM.md`, generated |
| `reproduce.sh` | everything, in order |
| `test_sim.py` | invariants |
| `lsq.py` | Levenberg-Marquardt in numpy (this box has no scipy) |

Results, validation numbers and limits: `../RESULTS_SIM.md`.

## The node engine: other kernels, several GPUs

Everything above is one MoE layer on one GPU. A GEMM beside an NCCL all-reduce, a
different clock on each GPU, or a training step are run by a second engine that keeps
the chip (`hardware.py`, `calib.Calib.power`, the limiter's rules) and replaces the
workload: per GPU a host program that issues kernels on CUDA streams, and collectives
that tie the GPUs together.

```bash
python3 -m sim.fit_kernels          # kernel profiles -> calib/a100_kernels.json (~7 min)
python3 -m sim.test_node            # 12 invariants
python3 -m sim.validate_overlap     # against every measured overlap, span and training step
python3 -m sim.heldout_overlap score
```

```python
from sim import node
from sim.calib import Calib
from sim.kernels import Kernels

k, cal = Kernels.load(), Calib.load()
# one cell of the measurement harness: GEMM beside a 256 MiB all-reduce, 16 CTAs, 1200 MHz
c = node.cell(k, cal, "concurrent", shape="4096x4096x4096", mib=256, ctas=16, clock=1200)
# a training step as ops on streams, a clock per GPU (the JSON ../../overlap_exp/sweep_spans.py runs)
ops = [dict(op="gemm", tag="g", stream="main", shape="8192x8192x4096"),
       dict(op="allreduce", tag="ar", stream="comm", mib=128, wait=["g"]),
       dict(op="gemm", tag="x", stream="main", shape="4096x8192x8192")]
s = node.program_cell(k, cal, ops, ctas=16, clock=[1410, 900, 900, 900])
```

| file | what |
|---|---|
| `kernels.py` | kernel profiles: the cuBLAS GEMM (tiles in waves) and the NCCL all-reduce (latency + bytes / bandwidth, power, what it costs a GEMM beside it); add a kernel here |
| `fit_kernels.py` | their fit from `../../overlap_exp/data` |
| `node.py` | the engine: host programs, streams, collectives, one limiter per GPU; `cell`, `program_cell`; the what-if `comm_clock` |
| `validate_overlap.py`, `heldout_overlap.py` | every measured run against the profiles; the registered blind test |
| `node_policy.py`, `study_overlap.py` | clocks per GPU for a training step, chosen by the engine; what a V/f domain for the collective's SMs would be worth |
| `test_node.py` | invariants |

Results: `../RESULTS_NODE.md`. Design for multi-GPU training: `../MULTIGPU.md`.
