#!/usr/bin/env python3
"""GEMM sweep WORKER — runs one (freq, sm) group's shapes IN-PROCESS using this repo's
collector measurement functions (benchmark_with_power, log_perf, get_sm_version) and a
torch bf16 cuBLAS matmul as the kernel_func.  Launched by sweep_launcher.py once per
(freq, sm) group, with CUDA_MPS_ACTIVE_THREAD_PERCENTAGE already set in the env (so the
SM cap is active).

Set COLLECTOR_MEASURE_POWER=1 for the power pass (auto-detected by benchmark_with_power).
"""
import argparse, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))   # tools/dvfs_char -> repo root
sys.path.insert(0, os.path.join(REPO, "collector"))                # use THIS fork's collector

import torch
from helper import benchmark_with_power, log_perf, get_sm_version  # <-- aiconfigurator collector

DTYPE = torch.bfloat16


def run_shape(M, N, K, freq, sm_count, mps_pct, perf_file, device="cuda:0"):
    dev = torch.device(device); torch.cuda.set_device(dev)
    # Linear-layer GEMM (mirrors collector's Linear(k, n)):  y[M,N] = x[M,K] @ W[N,K]^T
    x = torch.randn((M, K), dtype=DTYPE, device=dev)
    W = torch.randn((N, K), dtype=DTYPE, device=dev)
    kernel_func = lambda: torch.nn.functional.linear(x, W)         # cuBLAS ampere_bf16_s16816gemm
    with benchmark_with_power(device=dev, kernel_func=kernel_func, repeat_n=1,
                              allow_graph_fail=True) as r:
        pass
    pw = (r["power_stats"] or {}).get("power")
    log_perf(
        item_list=[{"gemm_dtype": "bfloat16", "m": M, "n": N, "k": K,
                    "freq_mhz": freq, "sm_count": sm_count, "mps_pct": mps_pct,
                    "latency": r["latency_ms"], "throttled": r["throttled"]}],
        framework="torch", version=torch.__version__,
        device_name=torch.cuda.get_device_name(dev),
        op_name="gemm", kernel_source="torch_cublas",
        perf_filename=perf_file, power_stats=r["power_stats"],
    )
    del x, W; torch.cuda.empty_cache()
    return r["latency_ms"], pw, r["throttled"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freq", type=int, required=True)
    ap.add_argument("--sm-count", type=int, required=True)
    ap.add_argument("--mps-pct", type=float, required=True)
    ap.add_argument("--shapes", required=True, help="JSON list of [M,N,K]")
    ap.add_argument("--perf-file", required=True)
    args = ap.parse_args()
    shapes = json.loads(args.shapes)
    print(f"[worker] freq={args.freq} sm~{args.sm_count} (mps%={args.mps_pct}) "
          f"MPS_env={os.environ.get('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE', '<none>')} "
          f"measure_power={os.environ.get('COLLECTOR_MEASURE_POWER', '0')} shapes={len(shapes)}", flush=True)
    for (M, N, K) in shapes:
        lat, pw, th = run_shape(M, N, K, args.freq, args.sm_count, args.mps_pct, args.perf_file)
        print(f"  M={M:>5} N={N:>5} K={K:>5}  lat={lat:.3f}ms  power={pw if pw else '-'}"
              f"{'  THROTTLED' if th else ''}", flush=True)


if __name__ == "__main__":
    main()
