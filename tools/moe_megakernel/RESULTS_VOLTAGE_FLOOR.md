# kappa is the voltage rail, not the workload

**Question.** The whole "slow the slack expert" idea dies if `kappa = P_dyn/f = C*V^2` is
flat below a knee, because then energy per unit of work does not depend on frequency down
there and there is no voltage left to give up. That flatness had only ever been measured
on large bf16 GEMMs. Does it carry to the MoE expert shapes, whose N and K are several
times smaller?

**Prediction, if kappa really is the rail.** Changing the workload changes `C`, the
effective switched capacitance, and therefore the LEVEL of kappa. Only the rail's `V(f)`
can change its SHAPE. So different workloads must give curves of different height and
identical shape. Falsifiable either way.

## Method — `kappa_sweep.py` -> `data/kappa_sweep.csv`

33 clocks from 300 to 1380 MHz, dense (15 MHz) through 915-1140 where the knee is.
GPU 0 only, locked, persistence on. 99 rows, 2 dropped for not holding their clock.

| workload | shape, bf16 |
|---|---|
| `gemm_large` | (4096 x 8192) @ (8192 x 8192) — the control, matching the original campaign |
| `moe_expert_n7652` | (7652 x 2048) @ (2048 x 1024) — real expert GEMM, prefill B=128 median |
| `moe_expert_n955` | (955 x 2048) @ (2048 x 1024) — prefill B=16 median |

Two corrections relative to my earlier 8-clock attempt, both of which had bitten:

- **CUDA graphs.** A tight `torch.mm` loop carries a 5-10 us launch gap, and the small
  expert GEMM runs in ~17 us — so most of what I measured before was launch overhead. It
  faked non-compute-boundness and produced a spurious "2-3% saving". Every measurement
  here replays a graph of 64 back-to-back GEMMs.
- **`P_idle` measured at every clock in the same pass**, so `P_dyn = P - P_idle(f)` uses a
  floor from the same thermal state rather than a table from another day.

A third bug was caught mid-run: `g.replay()` is asynchronous, so a `while time < secs:
replay()` loop queues hundreds of replays and the trailing `synchronize()` drains the
backlog — the timing came out right but each clock took minutes. The loop now calibrates
one replay and issues only as many as fill the window.

## Result — the prediction holds

| workload | level at 870 MHz | flat over 510-1035 | first >+5% | kappa(1380)/kappa(870) |
|---|---|---|---|---|
| `gemm_large` | 179.0 mW/MHz | **±1.7%** | 1095 MHz | (1380 throttled) |
| `moe_expert_n7652` | 136.2 mW/MHz | **±2.0%** | 1080 MHz | **1.71** |
| `moe_expert_n955` | 82.9 mW/MHz | **±2.9%** | 1095 MHz | **1.78** |

**Levels span 2.2x. Knees land within 15 MHz of each other. Floors are flat to 1.7-2.9%
and the rise above the knee agrees to 4%.** Normalise and the three curves lie on top of
one another (fig5).

## What this settles

1. **The voltage floor is a property of the silicon**, confirmed on three workloads whose
   effective capacitance differs by 2.2x, including the actual MoE expert shapes. It is
   not an artefact of the large-GEMM characterisation.
2. **"Give the slack expert a slower clock" saves nothing on A100.** Between 510 and 1035
   MHz the energy per operation moves by 2-3%, which is the measurement's own noise. The
   expert does the same number of operations either way, so it burns the same joules,
   just spread over more time.
3. **All of the opportunity is in not exceeding ~1080 MHz.** kappa(1380)/kappa(870) = 1.7-1.8,
   so the clock the governor picks costs 71-78% more energy per operation than the floor.
   That is a single global clock setting and needs no partitioning at all.

## What it does not settle

- One GPU, one architecture. The cross-architecture claim rests on the literature's
  piecewise model (power linear in f below a transition frequency, which is exactly
  kappa = const) and on their reported f_t ~ 1005 MHz, not on our own measurement.
- Whole-GPU kappa, not per-SM. Sufficient here because only the shape was under test, and
  the earlier campaign showed whole-GPU and per-SM fits agree on shape.
- The floor is measured down to 300 MHz but only densely above 510.
