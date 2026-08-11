# GEMM DVFS × SM-count characterization (A100-SXM4-40GB)

How a bf16 GEMM's **latency, board power and dynamic energy** vary with **SM clock** and
**the number of SMs it can use**, and a fitted model for the dynamic part.
Fitted coefficients and their measured accuracy: **[`MODEL.md`](MODEL.md)**.

Supersedes `tools/gemm_dvfs_char_unused` (the MPS-based first attempt).

## Layout

```
MODEL.md                     the model, its coefficients, and its accuracy
squatter.py                  the mechanism: occupy n SMs with an __nanosleep kernel
measure.py                   NVML power / achieved clock / throttle sampler + measure()
gen_config_table.py          the sweep grid -- single source of truth
sweep_driver.py              the 1080-row sweep
profile_static.py            P_static(clock, occupancy) via the squatter
run_p2p_static.py            P_static(clock) via zeus's profile_p2p.py, unmodified   <- used
profile_p2p.py               zeus's script, vendored verbatim for reference
profile_p2p_local.py         the same method reimplemented without the zeus dependency
profile_static_temp.py       leakage vs die temperature
profile_occupancy.py         ncu: real grid size and real SM occupancy
validate.py                  proves the squatter mechanism (with a negative control)
make_dynamic_db.py           joins P_static onto the sweep -> the deliverable
plot_sweep.py                operating maps (latency / power / energy / dynamic)
plot_decomposition.py        static vs dynamic split, and P_dyn = a*S_eff + b
plot_static.py               the static-power characterization
fit_models.py plot_tiling.py model fitting and tiling/wave figures
```

`data/gemm_dynamic_energy.csv` is **the deliverable**: 1080 rows of latency + dynamic
energy. `data/gemm_squat_sweep.csv` is the raw sweep it is built from.

## Reproduce

```bash
sudo nvidia-smi -pm 1                        # WITHOUT THIS THE CLOCK LOCK DOES NOT HOLD
python3 validate.py                          # squatter mechanism, exits 0 if sound
python3 sweep_driver.py --reps 3             # 1080 rows, ~41 min
python3 run_p2p_static.py --reps 3           # static floor, 6 clocks, ~26 min
python3 make_dynamic_db.py                   # -> data/gemm_dynamic_energy.csv
python3 plot_sweep.py --csv data/gemm_dynamic_energy.csv
python3 plot_decomposition.py && python3 plot_static.py
```

`run_p2p_static.py` needs TensorRT-LLM's venv only because zeus's script imports it;
everything else runs on the system Python.

## How the two axes are controlled

**SM count — an `__nanosleep` squatter, not MPS.** `n` blocks each claim 160 KB of the
SM's 164 KB shared memory, so one block fits per SM and `n` blocks hold `n` SMs; the
GEMM gets `108 − n`. Every configuration re-runs a probe with the cuBLAS kernel's real
147456 B footprint and asserts the reachable set is exactly `108 − n` — 1080/1080
passed. `__nanosleep` counts nanoseconds, not cycles, so residency is unaffected by
DVFS.

**Clock — `nvidia-smi -lgc`, verified by read-back, and the ACHIEVED clock is recorded.**
A lock is a request. At 1410 MHz every shape at 108 and 81 SM hit the 400 W SwPowerCap
and ran at 1245–1380 MHz instead — and the clamp eases as SMs are removed, so inside
that column the SM effect and the clock effect are **not separable**. Those 21 of 360
cells carry `clock_held=False` and are excluded from every fit. Clocks ≤1200 MHz held
exactly, 900/900.

## Traps already hit here

* **Persistence mode off ⇒ the clock lock silently does not hold.** `-lgc` still prints
  "All done." while the run executes at the boost clock.
* **A device-wide sync deadlocks under a resident squatter** — it waits out the
  backstop. `torch.cuda.synchronize()`, `empty_cache()`, `cudaFreeHost` and CUDA graphs
  all qualify. Only stream-scoped syncs inside the window; warm everything outside it.
* **The handshake cannot go through a kernel.** Polling a device tensor launches a
  kernel that cannot run until the squatter finishes. Flags live in mapped pinned host
  memory, written with `__threadfence_system()`.
* **A squatter can expire mid-window** and leave the row looking clean while holding
  108-SM numbers. Per-block heartbeats must cover the *whole* window, not just tick once.
* **Shared-memory carveout class, not byte count, is what confines a kernel.** A
  squatter in a lower class also poisons its TPC partner: at 100 KB the 144 KB cuBLAS
  kernel could not be placed anywhere and deadlocked at n≥54. 160 KB is safe against any
  co-runner.
* **NVML power needs ~427 ms to reach 95 % of steady state.** Sub-second windows swing
  ±43 %; the same configuration read 64.0 / 482.8 / 82.9 W on three 130 ms reps.
* **P-state is not a clock indicator.** This board reports P0 while idling the SM clock
  down to 210 MHz.
* **Row-normalised heatmaps invert the reading** — 0.207 ms once rendered as "worst" and
  1.96 ms as "best". Every figure here is absolute, with the value printed in the cell.
