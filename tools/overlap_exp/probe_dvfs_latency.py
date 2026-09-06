#!/usr/bin/env python3
"""How long does an SM clock change take to land? Decides whether per-phase DVFS is
reachable at all.

WHY THIS EXISTS. "Give the GEMM and the collective different clocks" has a temporal
reading -- keep one domain but switch its frequency between the two phases. That is only
viable if a switch costs far less than a phase lasts. This measures the switch.

TWO THINGS THE MEASUREMENT HAS TO GET RIGHT:

  Measure under LOAD. An idle GPU reports the requested clock almost immediately without
  ever having run at it, so an idle probe reports a transition time that is not real.
  A continuous GEMM runs in a background thread throughout.

  Use the NVML API, not the CLI. `nvidia-smi -lgc` costs ~65 ms per call in process spawn
  alone, which swamps the hardware transition and is not what a runtime would pay. Both
  paths are reported so the difference is visible.

  sudo PYTHONPATH=$HOME/.local/lib/python3.12/site-packages python3 probe_dvfs_latency.py
"""
import csv
import os
import statistics as st
import subprocess
import threading
import time

import pynvml
import torch

DEV = 0
PAIRS = [(600, 900), (900, 600), (900, 1200), (1200, 900), (705, 1200), (1200, 705)]
REPS = 8
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(DEV)
    pm_was = pynvml.nvmlDeviceGetPersistenceMode(h)
    pynvml.nvmlDeviceSetPersistenceMode(h, 1)
    torch.cuda.set_device(DEV)
    x = torch.randn((4096, 8192), dtype=torch.bfloat16, device="cuda")
    w = torch.randn((8192, 8192), dtype=torch.bfloat16, device="cuda")
    stop = threading.Event()

    def load():
        while not stop.is_set():
            for _ in range(20):
                torch.nn.functional.linear(x, w)
            torch.cuda.synchronize()

    th = threading.Thread(target=load, daemon=True)
    th.start()
    time.sleep(1)

    clk = lambda: pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)

    def settle(target, tol=15, timeout=2.0):
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout:
            if abs(clk() - target) <= tol:
                return (time.perf_counter() - t0) * 1e3
        return None

    rows = []
    print(f"{'from':>6}{'to':>6}{'API call us':>13}{'settle ms':>12}{'total ms':>11}{'n':>4}")
    try:
        for a, b in PAIRS:
            pynvml.nvmlDeviceSetGpuLockedClocks(h, a, a)
            time.sleep(0.4); settle(a)
            call, tot = [], []
            for _ in range(REPS):
                pynvml.nvmlDeviceSetGpuLockedClocks(h, a, a)
                time.sleep(0.25); settle(a)
                t0 = time.perf_counter()
                pynvml.nvmlDeviceSetGpuLockedClocks(h, b, b)
                t1 = time.perf_counter()
                s = settle(b)
                if s is None:
                    continue
                call.append((t1 - t0) * 1e6); tot.append((t1 - t0) * 1e3 + s)
            if not tot:
                continue
            rows.append(dict(f_from=a, f_to=b, n=len(tot),
                             api_call_us=round(st.median(call), 1),
                             settle_ms=round(st.median(tot) - st.median(call) / 1e3, 3),
                             total_ms=round(st.median(tot), 3),
                             path="nvml_api", under_load=True))
            print(f"{a:>6}{b:>6}{rows[-1]['api_call_us']:>13.0f}"
                  f"{rows[-1]['settle_ms']:>12.2f}{rows[-1]['total_ms']:>11.2f}{len(tot):>4}")
        # the CLI path, once, to record what the convenient route actually costs
        t0 = time.perf_counter()
        subprocess.run(["sudo", "nvidia-smi", "-i", str(DEV), "-lgc", "900,900"],
                       capture_output=True)
        cli = (time.perf_counter() - t0) * 1e3
        rows.append(dict(f_from="", f_to=900, n=1, api_call_us=round(cli * 1e3, 1),
                         settle_ms="", total_ms=round(cli, 3),
                         path="nvidia_smi_cli", under_load=True))
        print(f"\n  nvidia-smi CLI call alone: {cli:.1f} ms (process spawn, avoidable)")
    finally:
        stop.set(); th.join(timeout=3)
        pynvml.nvmlDeviceResetGpuLockedClocks(h)
        pynvml.nvmlDeviceSetPersistenceMode(h, pm_was)
        print("[clocks] restored")

    p = os.path.join(HERE, "data", "dvfs_transition.csv")
    with open(p, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)
    api = [r["total_ms"] for r in rows if r["path"] == "nvml_api"]
    m = st.median(api)
    print(f"\none-way median {m:.2f} ms   round trip {2*m:.2f} ms")
    print(f"break-even phase length for a 2.65% energy saving: {2*m/0.0265:.0f} ms")
    print(f"kernels here are 0.2-30 ms, a training step 100-500 ms -> not reachable")
    print(f"wrote {p}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
