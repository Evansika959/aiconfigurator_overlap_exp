# GEMM ∥ all-reduce overlap on 2 × A100

Can a GEMM and a collective run at the same time on disjoint SMs, and does it pay in
time and energy? Uses the same measurement machinery as `../gemm_dcfs_char`, and
subtracts the same zeus-measured static-power floor.

```
demo_overlap.py        one worked example, start to finish
run_demo_overlap.sh    ^ with clock locking, gIB workaround and cleanup      <- start here
phase0_probe.py        feasibility probes: does concurrency happen at all?
phase1_sweep.py        the sweep: shape x message size x CTA count x launch order
data/phase1_partial.csv   30 of 80 configurations (CTA 2 and 4 only) -- see below
data/logs_phase1.txt      the raw run log those rows were parsed from
```

## Quick start

```bash
./run_demo_overlap.sh          # lock 1200 MHz, run, restore clocks
./run_demo_overlap.sh none     # don't touch clocks (no sudo needed)
```

Needs 2 GPUs and a stock PyTorch with CUDA. **Nothing else to install** — the
collective goes through `torch.distributed`, and `torch.profiler` is built in.

## The result

`M=8192, N=4096, K=4096` bf16 GEMM ∥ 64 MiB all-reduce, world=2, 4 NCCL CTAs, 1200 MHz:

```
                GEMM [start, end] ms      all-reduce [start, end] ms
serial              [0.006, 1.256]                  [1.261, 2.707]     back to back
concurrent          [0.148, 1.512]                  [0.028, 1.694]     GEMM nested inside
                    intersection 1.364 ms = 100% of the shorter kernel
```

| | serial | concurrent | change |
|---|---|---|---|
| wall per iteration | 2.7 ms | 1.7 ms | **−36.9 %** |
| node power (2 GPUs) | 412.4 W | 557.3 W | +35.1 % |
| **energy per iteration** | 1122.8 mJ | **957.0 mJ** | **−14.8 %** |

Speedup **1.62×**. Higher instantaneous power bought a shorter wall time; the saving is
almost entirely static energy, which tracks time exactly, while dynamic energy is
roughly conserved (same MACs, same bytes).

## Four things the overlap depends on

A GEMM submits ~1000 blocks that fill all 108 SMs within microseconds, and **a resident
block runs to completion** — A100 does not preempt at block granularity. An NCCL
collective is **one persistent kernel** whose `n_cta` blocks stay resident for the whole
reduction. Everything below follows from that.

1. **Two fresh, non-default streams.** One stream is ordered by the hardware; the
   default stream also synchronises against the others.
2. **The collective's stream at priority −3** (A100's range is 0..−3). *Not optional* —
   at equal priority the all-reduce ran **10.6× slower** and the pair finished **slower
   than serial** (0.96×).
3. **Issue the collective first**, so its CTAs land before the GEMM floods the SMs. This
   is what "reserve N SMs for the collective" actually means here.
4. **A common reference event** — record `base` once and have each stream
   `wait_event(base)`. Without it the second stream starts later by whatever the first
   launch cost on the host, and part of the measured overlap is an artefact of issue
   order.

`NCCL_MIN_CTAS = NCCL_MAX_CTAS = n` sets the collective's footprint and **must be set
before `init_process_group`** — NCCL reads them when the communicator is built.

## What is proven, and what is not

| evidence | supports |
|---|---|
| CUDA events on a common timeline | the two kernels were **resident** simultaneously |
| speedup 1.62× | they made **progress** simultaneously — time-slicing one set of SMs cannot produce a speedup |
| `torch.profiler` trace: separate stream rows, overlapping bars | the same, visually |
| NVML on both boards | the energy, node-wide |

**Not proven: which SMs each kernel occupied.** `ncu` can read per-SM counters but
serialises kernels to do it, which destroys the concurrency under study. In the example
above the GEMM did not slow down at all (1.357 → 1.169 ms), so no SM concession can even
be inferred from timing — with 4 CTAs the collective fits in the tail waves this GEMM
already leaves idle.

## Phase 1 status: partial

`data/phase1_partial.csv` holds **30 of 80** configurations — CTA 2 and 4 across both
shapes, both message sizes, all four launch orders. CTA 8/16/32 were not run. To finish:

```bash
for C in 8 16 32; do
  mpirun -n 2 --allow-run-as-root -x LD_LIBRARY_PATH \
      python3 phase1_sweep.py --ctas $C
done
```

What the 30 rows already show, for `comm_first` (best in every configuration):

| CTA | shape | MiB | balance t_g/t_c | speedup | total energy |
|---|---|---|---|---|---|
| 4 | high_occ | 64 | 0.83 | **1.455** | **−15.0 %** |
| 4 | low_occ | 64 | 0.45 | 1.273 | −9.9 % |
| 2 | high_occ | 64 | 0.44 | 1.271 | −9.1 % |
| 2 | high_occ | 256 | 0.11 | 1.036 | −1.5 % |

**Balance — the ratio of the two kernels' solo durations — very nearly determines the
payoff.** The point of tuning the CTA count is not "give the collective more SMs", it is
to bring the collective's duration close to the GEMM's. The naive
`gemm_first_p0` ordering was *worse than serial* in every configuration (0.930–0.966×).

## Traps already hit here

* **A collective inside a loop needs an identical iteration count on every rank.**
  Deriving it from a locally measured duration puts the ranks one apart, one rank issues
  a collective nobody joins, and the job hangs with one GPU at 100 % and the other idle.
  Both scripts now broadcast the count from rank 0. This is what stalled the first sweep
  at 30/80.
* **`dist.all_reduce` returns before the collective runs** — it only enqueues. Timing it
  without a stream sync measures ~122 µs of launch overhead instead of ~1.5 ms of work.
* **A fixed `MASTER_PORT` is a trap**: a previous hung run leaves it bound, and the next
  run dies with `EADDRINUSE` on rank 0 and a misleading NCCL "remote process exited" on
  rank 1. `demo_overlap.py` picks a free port.
* **`nvidia-smi -lgc` silently does not hold without persistence mode.** After a reboot
  cleared it, a requested 1200 MHz ran at 1410 while `-lgc` still reported success.
  `run_demo_overlap.sh` enables persistence and restores the prior setting on exit.
* **GCP's gIB NCCL shim** breaks NCCL on A100 and rejects `NCCL_{MIN,MAX}_CTAS`.
  Unsetting `NCCL_NET` is not enough — `/usr/local/gib` must come off
  `LD_LIBRARY_PATH`.
* **The first collective in a profiled region absorbs all inter-rank skew** — measured
  20.3 ms against 1.4 ms for the following four. Always discard it; the profiler's
  `schedule(wait=1, warmup=1, active=N)` does this for you.
* **`torch.profiler` with CUDA activities inflates kernel times ~3.5×** (CUPTI
  serialises and replays). Use it for structure, CUDA events for numbers.
