# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collect NCCL collective latency with nccl-tests binaries.

This collector wraps the standard nccl-tests command-line tools for all-gather,
all-to-all, reduce-scatter, and all-reduce sweeps. It translates the benchmark
output into AIC perf rows and optionally samples representative GPU power while
each collective size is measured.
"""

import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helper import PowerMonitor, log_perf


def _parse_latency_ms(result, cmd_args) -> float:
    """Pull the collective time (us -> ms) out of nccl-tests stdout.

    Raises instead of returning a bogus number when the binary did not produce a
    result row -- a failed collective must surface as a classified error, never
    as a silently wrong data point.
    """
    print_lines = result.stdout.split("\n")
    for index_line in range(len(print_lines)):
        if "time" in print_lines[index_line]:
            break
    try:
        return float(print_lines[index_line + 2].split()[5]) * 1e-3  # us to ms
    except (IndexError, ValueError) as e:
        raise RuntimeError(
            f"nccl-tests produced no result row for {' '.join(cmd_args)} "
            f"(rc={result.returncode}).\nstdout tail:\n{result.stdout[-800:]}\n"
            f"stderr tail:\n{result.stderr[-800:]}"
        ) from e


def nccl_benchmark(
    dtype: str,
    nccl_op: str = "all_gather",
    test_range: str = "10,10000000,1000",
    num_gpus: int = 8,
    measure_power: bool = False,
    power_test_duration_sec: float = 1.0,
    perf_filename: str = "nccl_perf.txt",
    sizes: list[int] | None = None,
):
    nccl_test_bin = ""
    if nccl_op == "all_gather":
        nccl_test_bin = "all_gather_perf"
    elif nccl_op == "alltoall":
        nccl_test_bin = "alltoall_perf"
    elif nccl_op == "reduce_scatter":
        nccl_test_bin = "reduce_scatter_perf"
    elif nccl_op == "all_reduce":
        nccl_test_bin = "all_reduce_perf"
    assert nccl_test_bin != ""

    if sizes:
        size_list = list(sizes)
    else:
        min_size, max_size, ratio = [int(i) for i in test_range.split(",")]
        size_list = []
        size = min_size
        while size < max_size:
            size_list.append(size)
            size *= ratio

    major, minor, patch = torch.cuda.nccl.version()
    nccl_version = f"{major}.{minor}.{patch}"

    bytes_per_element = {"half": 2, "bfloat16": 2, "int8": 1}[dtype]

    # Initialize power monitoring if enabled.
    # A collective is a NODE-level event: every participating GPU burns power, so
    # sampling only GPU 0 under-reports it by ~num_gpus. Monitor every rank and
    # report the node total.
    power_monitors = []
    if measure_power:
        for dev in range(num_gpus):
            mon = PowerMonitor(device_id=dev)
            if mon._init_handle():
                power_monitors.append(mon)
        if not power_monitors:
            print("Warning: Failed to initialize power monitoring, continuing without power measurement")

    def build_cmd(iters, warmup):
        return [
            nccl_test_bin,
            "-b",
            str(size),
            "-e",
            str(size),
            "-t",
            str(num_gpus),
            "-d",
            dtype,
            "-w",
            str(warmup),
            "-a",
            "1",
            "-n",
            str(iters),
            "-c",
            "0",
        ]

    for size in size_list:
        inner_loop = 100 if size <= 16777216 else 60

        # Power pass: NVML samples at 100ms, so the timed loop must be long enough
        # to contain a meaningful number of samples. Calibrate the iteration count
        # from a cheap probe run so the timed loop lasts >= power_test_duration_sec.
        if power_monitors:
            probe = subprocess.run(build_cmd(5, 5), capture_output=True, text=True)
            probe_latency_s = _parse_latency_ms(probe, build_cmd(5, 5)) / 1e3
            if probe_latency_s > 0:
                inner_loop = max(inner_loop, min(int(power_test_duration_sec / probe_latency_s) + 1, 100000))

        cmd_args = build_cmd(inner_loop, 40)

        # Start power monitoring before benchmark
        power_stats = None
        for mon in power_monitors:
            mon.start_sampling()

        result = subprocess.run(cmd_args, capture_output=True, text=True)
        latency = _parse_latency_ms(result, cmd_args)

        # Stop power monitoring after benchmark. The timed loop is the FINAL
        # inner_loop * latency seconds of the subprocess -- everything before it is
        # NCCL bootstrap and warmup, which sits near idle and would otherwise
        # dominate the average (measured: ~1.3% duty cycle without this window).
        if power_monitors:
            window_s = inner_loop * latency / 1e3
            per_gpu = [mon.stop_sampling(window_s=window_s) for mon in power_monitors]
            per_gpu = [s for s in per_gpu if s]
            if per_gpu:
                power_stats = {
                    "power": sum(s["power"] for s in per_gpu),
                    "power_limit": sum(s["power_limit"] for s in per_gpu if s["power_limit"]),
                }

        print(nccl_test_bin, f"{size=}, {latency=}")
        if power_stats:
            print(
                f"  Node power ({len(per_gpu)} GPU): {power_stats['power']:.2f}W "
                f"(limit: {power_stats['power_limit']:.2f}W) over {window_s:.2f}s x {inner_loop} iters"
            )

        log_perf(
            item_list=[
                {
                    "nccl_dtype": dtype,
                    "num_gpus": num_gpus,
                    "message_size": size // bytes_per_element,
                    "latency": latency,
                }
            ],
            framework="TRTLLM",
            version=nccl_version,
            device_name=torch.cuda.get_device_name(),
            op_name=nccl_op,
            kernel_source="NCCL",
            perf_filename=perf_filename,
            power_stats=power_stats,
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--nccl_op",
        "-NCCL",
        default="all_gather",
        choices=["all_gather", "alltoall", "reduce_scatter", "all_reduce"],
        help="NCCL OP: all_gather, alltoall, reduce_scatter, all_reduce",
    )
    parser.add_argument(
        "--dtype",
        "-t",
        default="half",
        choices=["half", "bfloat16", "int8"],
        help="NCCL OP data type. bfloat16 matches the GEMM characterization sweep; "
        "on NVLink it is bandwidth-equivalent to half (same 2 bytes/element).",
    )
    parser.add_argument(
        "--range",
        "-r",
        default="512,536870913,2",  # 512B to 512MB
        help="min_size,max_size,multiplicative_ratio",
    )
    parser.add_argument("--num_gpus", "-n", default=8, type=int)
    parser.add_argument(
        "--measure_power",
        action="store_true",
        help="Enable power monitoring during NCCL benchmark execution (samples at 100ms intervals)",
    )
    parser.add_argument(
        "--power_test_duration_sec",
        type=float,
        default=1.0,
        help="Minimum duration of the TIMED collective loop when power measurement is enabled "
        "(default: 1.0s). The iteration count is calibrated from a probe run to reach it, and "
        "power is averaged over that trailing window only.",
    )
    parser.add_argument(
        "--perf-filename",
        default="nccl_perf.txt",
        help="Output perf file (default: nccl_perf.txt). Matches collect_all_reduce.py.",
    )
    parser.add_argument(
        "--sizes",
        default=None,
        help="Explicit comma-separated message sizes in BYTES, e.g. '2097152,26214400'. "
        "Takes precedence over --range. Lets one process cover a non-geometric size set "
        "(DDP bucket caps are not powers of two) instead of one process per size.",
    )
    args = parser.parse_args()

    nccl_benchmark(
        args.dtype,
        args.nccl_op,
        args.range,
        args.num_gpus,
        args.measure_power,
        args.power_test_duration_sec,
        args.perf_filename,
        [int(s) for s in args.sizes.split(",")] if args.sizes else None,
    )
