# all_reduce DVFS × SM-footprint characterization (4 × A100)

Measures how a bf16 all_reduce's **latency, power, energy and bus bandwidth** vary
with **SM clock** and **the collective's SM footprint** on 4 × A100-SXM4-40GB over
all-to-all NVLink (NV12).

The kernel under test is the one a real deployment runs: TensorRT-LLM's
`AllReduce` with `AllReduceStrategy.NCCL`, driven through **this repo's own
collector** (`collector/network/collect_all_reduce.py`) under `mpirun -n 4`.

```
sweep_launcher_trtllm.py   locks the clock per group, pins NCCL CTAs, drives the collector
merge_trtllm_ar.py         per-group perf files -> data/comm_trtllm_ar_ctas.csv
plot_cta_opmap.py          the operating maps
verify_nccl_equivalence.py is TRT-LLM's AllReduce(strategy=NCCL) the same thing as a
                           plain NCCL all-reduce?  Both launch exactly one kernel per
                           call, the same one (ncclDevKernel_AllReduce_Sum_bf16_RING_LL),
                           within 4% on latency -- so these rows characterise raw NCCL.
data/comm_trtllm_ar_ctas.csv     288 rows (the deliverable)
data/repeatability_check.csv     144 points measured twice -> the 2.4% noise floor
figures/comm_cta_{energy,latency,power,busbw}_opmap.png
results/                   raw collector perf files  [git-ignored]
_superseded/               earlier sweeps and figures, kept for reference only
```

## Sweep space (288 configs)

| knob | values | mechanism |
|---|---|---|
| dtype | bf16 | matches the GEMM characterization |
| message size | 8, 16, 25, 32, 64, 100, 128, 256 MiB | `--sizes` (ELEMENTS; bf16 bytes = 2×) |
| SM clock | 1410, 1200, 900, 705, 510, 300 MHz | `sudo nvidia-smi -i 0,1,2,3 -lgc f,f`, read-back verified |
| **SM footprint** | **1, 2, 4, 8, 16, 32 CTAs** | **`NCCL_MIN_CTAS = NCCL_MAX_CTAS`** |
| world size | 4 | `mpirun -n 4` |
| strategy | NCCL | `--strategy` |

Message sizes are the real multi-GPU-training sizes: 25 MiB is PyTorch DDP's
default `bucket_cap_mb`, 100 MiB a large-bucket setting, 32 MiB a typical
tensor-parallel activation all_reduce, and FSDP per-layer shards land at 12–48 MiB.

## Why CTAs, and not MPS or NCCL channels

A **CTA** (cooperative thread array) is a thread block; one is resident per SM, so
`NCCL_MIN_CTAS = NCCL_MAX_CTAS = N` is a direct request for N SMs of collective
footprint. Verified on this box: `N=1` → NCCL logs `1 p2p channels`, `N=32` → `32`,
with near-linear bandwidth scaling between.

- **Not `NCCL_MAX_NCHANNELS`** — NCCL supersedes it with `NCCL_MAX_CTAS`.
- **Not MPS** — no daemon, no `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`, nothing to start
  or leave running. MPS also caps a *percentage of the whole GPU per client*, which
  is a blunter instrument than an exact CTA count.
- **Note `NCCL_CTX_MIN` / `NCCL_CTX_MAX` do not exist** in NCCL 2.27.5
  (`strings libnccl.so.2` finds nothing). The env vars are `NCCL_MIN_CTAS` /
  `NCCL_MAX_CTAS`.

The knob governs the **NCCL path only**. TensorRT-LLM's own MIN_LATENCY kernels are
its own CUDA code in `libtensorrt_llm.so` and ignore it, so there is no SM axis for
that strategy.

## Reproduce

Prerequisites: OpenMPI, and TensorRT-LLM 1.3.0rc10 in an isolated venv. TRT-LLM will
not resolve unless torch comes from the cu130 index **first**:

```bash
sudo apt-get install -y libopenmpi-dev openmpi-bin
uv venv ~/trtllm_venv --python 3.12
uv pip install --python ~/trtllm_venv/bin/python torch==2.10.0 torchvision \
    --index-url https://download.pytorch.org/whl/cu130
uv pip install --python ~/trtllm_venv/bin/python tensorrt_llm==1.3.0rc10 mpi4py nvidia-ml-py \
    --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match
```

```bash
cd tools/comm_dvfs_char
SZ=$(python3 -c "print(','.join(str(int(m*1048576/2)) for m in [8,16,25,32,64,100,128,256]))")
for C in 1 2 4 8 16 32; do
  python3 sweep_launcher_trtllm.py --out results/cta_sweep --measure-power --power-duration 2.0 \
      --resume --sizes "$SZ" --strategies NCCL --ctas $C
done                                                    # 288 configs, ~18 min
python3 merge_trtllm_ar.py --indir results/cta_sweep --out data/comm_trtllm_ar_ctas.csv
python3 plot_cta_opmap.py
```

Clocks are restored via `nvidia-smi -rgc` on **every** exit path (`finally`, `atexit`,
SIGINT, SIGTERM) — verified by locking to 1410, killing the run, and confirming the
return to 210 MHz idle.

## The one gotcha that will stop you cold

GCP's gIB plugin breaks all NCCL on this node, in two independent ways. The image
sets `NCCL_NET=gIB` (an A3-Ultra/A4 RDMA plugin) **and** puts `/usr/local/gib/lib64`
on the global `LD_LIBRARY_PATH`. On A100 the plugin fails to load; worse, the shim on
the library path enforces a config policy on the guest and **rejects
`NCCL_{MIN,MAX}_CTAS` and `NCCL_{MIN,MAX}_NCHANNELS`** — i.e. it forbids this sweep's
entire SM-footprint axis. Unsetting the env vars is *not* sufficient; the gib
directory must come off `LD_LIBRARY_PATH`. The launcher does both internally.

## Measurement notes

**Achieved clock is recorded, not assumed.** A locked clock is a request; the board
can still clamp below it. `PowerMonitor` samples `nvmlDeviceGetClockInfo(NVML_CLOCK_SM)`
in the same loop as power, and every row carries `clock_sm_mean / clock_sm_min` plus a
`throttled` flag (`clock_sm_min` more than 20 MHz under the request). On this dataset:
**0 / 288 throttled**; peak power 169.7 W on the rank-0 GPU against a 400 W cap.

**Power and energy are the rank-0 GPU only.** `collect_all_reduce.py` samples
`PowerMonitor(local_rank)` and only rank 0 logs. Node total ≈ 4×.
`energy_mj = power_w × latency_ms` (W·ms == mJ, the SDK's `PerformanceResult`
convention).

**Repeatability.** All 144 (strategy, clock, size) points of the preceding sweep were
collected twice: median 0.105 %, p90 0.360 %, max 2.434 % (nearest-rank order
statistic; numpy's default linear interpolation gives p99 1.721 % rather than
2.037 %). `data/repeatability_check.csv` ships those pairs. Differences below ~2.4 %
are not signal.

## Figure conventions

Colour is an **absolute log scale shared by every facet** — no normalization of any
kind, so one colour means one value everywhere and the ~3000× span across message
size and CTA count is visible rather than hidden. Cell values are absolute, printed at
2 significant figures (3 would be false precision against the 2.4 % floor). Any cell
whose achieved clock fell below the request is outlined in red.

An earlier version normalized within each row; that inverted the data, rendering a
0.207 ms cell as the panel's "worst" while a 1.96 ms cell (9.5× slower) rendered as
"best". Do not reintroduce it.
