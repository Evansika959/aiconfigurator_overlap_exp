# DVFS × #SM GEMM characterization (A100)

Preliminary kernel-characterization sweep for a bottom-up GPU performance/energy
model. Measures how a bf16 GEMM's **latency, power, and energy** vary across four
knobs — **GEMM size, SM clock (V/f), and number of active SMs** — on a real
A100-SXM4-40GB (SM80).

The kernel under test is the **vendor cuBLAS** tensor-core GEMM
(`ampere_bf16_s16816gemm_*`), reached via `torch.nn.functional.linear` on bf16
inputs — identical to TensorRT-LLM's bf16 `Linear` (`torch_flow`) on this arch.
Measurement uses **this repo's own collector primitives**
(`collector/helper.py`: `benchmark_with_power`, `log_perf`, `get_sm_version`) —
CUDA-graph capture + CUDA-event timing + NVML power sampling.

## Sweep space (360 configs)

| knob | values | mechanism |
|------|--------|-----------|
| dtype | bf16 | (A100 has no FP8/NVFP4 tensor cores) |
| M | 1024, 2048, 4096, 8192 | activation rows |
| (N, K) | 4096², 8192², 16384² | weight shape |
| SM clock | 300, 510, 705, 900, 1200, 1410 MHz | `sudo nvidia-smi -lgc f,f` (DRAM fixed 1215) |
| active SMs | 108, 81, 54, 27, 14 | MPS `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` = 100/75/50/25/12.5 % |

## Files

```
gen_config_table.py     generate gemm_config_table.csv (edit knob lists to widen)
gemm_config_table.csv   the 360-row sweep definition
sweep_worker.py         measures one (freq, sm) group in-process (uses ../../collector)
sweep_launcher.py       locks clock per group, sets MPS %, drives the worker, --resume
plot_heatmap.py         renders the multi-dimensional operating-point heatmaps
data/gemm_char_merged.csv   merged latency+power+energy table (the deliverable)
results/                raw collector passes (.txt/.parquet) + logs  [git-ignored]
figures/                gemm_{energy,latency,power}_opmap.png
```

`data/gemm_char_merged.csv` columns:
`m,n,k,freq_mhz,sm_count,gflop,latency_ms,power_w,energy_mj,gflops,throttled`
(`energy_mj = power_w × latency_ms / 1000`).

## Reproduce

Requires: A100 (SM80), a torch build with CUDA, passwordless `sudo` (for
`nvidia-smi` clock locking), and the MPS daemon. The collector's runtime deps
(`nvidia-ml-py`, `pandas`, `pyarrow`, …) must be importable.

```bash
cd tools/gemm_dvfs_char
python3 gen_config_table.py                      # -> gemm_config_table.csv (360 rows)

nvidia-cuda-mps-control -d                        # start MPS (enables the #SM cap)
python3 sweep_launcher.py --out results/gemm_lat_perf.txt                  # latency (~4 min)
python3 sweep_launcher.py --out results/gemm_pow_perf.txt --measure-power  # power   (~11 min)
echo quit | nvidia-cuda-mps-control               # stop MPS; launcher resets clocks itself

# merge the two passes into data/gemm_char_merged.csv (see results/*.txt), then:
python3 plot_heatmap.py                           # -> figures/*.png
```

`--resume` skips `(freq,sm,m,n,k)` rows already present in `--out`, so an
interrupted pass can be continued. Clocks are always reset via `nvidia-smi -rgc`
in the launcher's `finally`.

## Figures

`plot_heatmap.py` emits one 4×3 facet grid per metric (facet row = M, facet
col = N=K; each panel is a **SM-clock × active-SMs** heatmap):

- `gemm_energy_opmap` — color = energy normalized within each shape (0 = best, 1 = worst),
  so every panel uses the full color range; the energy-optimal (clock, #SM) is the darkest
  cell (the DVFS "knee", ~900–1200 MHz at high SM count).
- `gemm_latency_opmap` — color = latency normalized within each shape (0 = fastest).
- `gemm_power_opmap` — color = absolute Watts on one global scale.

**Red-hatched** cells were thermal/power-throttled (the 400 W cap briefly
overrode the clock lock at 1410 MHz / 108 SM / 16384²); their power is a slight
under-read (2 of 360).
