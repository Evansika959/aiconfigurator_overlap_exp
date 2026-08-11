#!/usr/bin/env python3
"""How much does P_static move with die temperature?  Leakage rises with heat.

WHY THIS MATTERS. p0_static.csv was taken on a cool board (37-40 C) but the sweep ran
at 38-63 C, so subtracting the cool number under-subtracts and inflates E_dynamic.
The sweep's own in-situ baseline has the right temperature but only a 2 s window
taken right after a GEMM, so it catches that GEMM's power tail instead: at
300 MHz / 14 SM the two disagree by 6 W (59.8 vs 65.8), which is 65% of E_dynamic
there because static is 87% of the board power at that corner. One of them is
wrong and this measures which.

METHOD. Hold the clock, park all 108 SMs under the __nanosleep squatter so the GPU
stays at the locked clock with no dynamic activity, then heat the die with a
full-SM GEMM and record (temperature, power) once a second for several minutes as
it cools back down. The two effects separate cleanly IN TIME: a kernel's power tail
settles in well under a second (NVML needs ~427 ms to reach 95%), while the die
cools over minutes. So the first seconds are discarded and everything after is
leakage-vs-temperature.

  python3 profile_static_temp.py            # ~12 min, 2 clocks
"""

import argparse
import atexit
import csv
import os
import signal
import subprocess
import sys
import time

import pynvml
import torch

from measure import _nvml, _throttle_reasons
from squatter import Squatter

DEV = 0
GPUS = "0"
HEAT_SHAPE = (8192, 8192, 8192)
_locked = False


def sh(c):
    return subprocess.run(c, capture_output=True, text=True)


def reset_clocks():
    global _locked
    if _locked:
        sh(["sudo", "nvidia-smi", "-i", GPUS, "-rgc"])
        _locked = False
        print("[clocks] reset", flush=True)


def lock_clock(f):
    global _locked
    sh(["sudo", "nvidia-smi", "-i", GPUS, "-lgc", f"{f},{f}"])
    _locked = True
    time.sleep(0.4)


def _sig(s, _f):
    reset_clocks()
    sys.exit(128 + s)


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--out", default=os.path.join(here, "data", "p0_static_vs_temp.csv"))
    ap.add_argument("--clocks", type=int, nargs="+", default=[1200, 300])
    ap.add_argument("--heat-s", type=float, default=420.0, help="max seconds of heating")
    ap.add_argument("--heat-to-c", type=float, default=62.0,
                    help="stop heating once this die temperature is reached. The sweep ran "
                         "up to 64 C, so a cooldown that only spans 34-44 C would force a "
                         "20 C extrapolation of a superlinear leakage curve.")
    ap.add_argument("--cool-s", type=float, default=240.0)
    ap.add_argument("--settle-s", type=float, default=1.2,
                    help="discarded after the GEMM stops. NVML power needs ~427 ms to reach "
                         "95%% of steady state, so this cannot go much below 1 s -- but it "
                         "cannot go much above it either: measured, the DIE falls 62->38 C "
                         "in under 5 s once the GEMM stops (small die thermal mass; it is the "
                         "heatsink that is slow). A 5 s settle threw away the entire "
                         "informative range and left a 4 C window to fit 30 C of extrapolation on.")
    ap.add_argument("--fast-s", type=float, default=40.0,
                    help="sample at FAST_HZ for this long, to catch the steep part of the "
                         "cooldown before the die settles")
    ap.add_argument("--fast-hz", type=float, default=20.0)
    a = ap.parse_args()

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)
    atexit.register(reset_clocks)
    torch.cuda.set_device(DEV)
    h = _nvml(DEV)
    st = torch.cuda.current_stream()

    m, n, k = HEAT_SHAPE
    x = torch.randn((m, k), dtype=torch.bfloat16, device=f"cuda:{DEV}")
    w = torch.randn((n, k), dtype=torch.bfloat16, device=f"cuda:{DEV}")
    gemm = lambda: torch.nn.functional.linear(x, w)
    for _ in range(6):                      # warm cuBLAS OUTSIDE any squatter
        gemm()
    st.synchronize()

    out = os.path.abspath(a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fh = open(out, "w", newline="")
    wr = csv.DictWriter(fh, fieldnames=["clock_req_mhz", "t_since_heat_s", "temp_c",
                                        "power_w", "clock_sm", "throttle"])
    wr.writeheader()
    fh.flush()          # an unflushed header is lost if the run is killed mid-heat

    try:
        for f in a.clocks:
            # Heat with the clock UNLOCKED so the board reaches the temperatures the
            # sweep actually saw; a 90 s soak at 1200 MHz only got to 44 C. The
            # measurement clock is locked afterwards, for the cooldown.
            reset_clocks()
            print(f"\n=== {f} MHz ===", flush=True)
            print(f"  heating with GEMM{HEAT_SHAPE} on all 108 SMs until "
                  f"{a.heat_to_c:.0f} C (max {a.heat_s:.0f}s) ...", flush=True)
            t_end = time.time() + a.heat_s
            while time.time() < t_end:
                for _ in range(20):
                    gemm()
                st.synchronize()
                if pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU) >= a.heat_to_c:
                    break
            hot = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
            lock_clock(f)
            print(f"  reached {hot} C; parking all 108 SMs and logging the cooldown "
                  f"for {a.cool_s:.0f}s", flush=True)

            sq = Squatter(108, device=f"cuda:{DEV}",
                          max_seconds=a.cool_s + 120.0).__enter__()
            beats = sq.beats()
            try:
                t0 = time.time()
                time.sleep(a.settle_s)      # let the GEMM's power tail die out
                while time.time() - t0 < a.cool_s:
                    dt = time.time() - t0
                    wr.writerow(dict(
                        clock_req_mhz=f, t_since_heat_s=round(dt, 2),
                        temp_c=pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU),
                        power_w=round(pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0, 3),
                        clock_sm=pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM),
                        throttle=_throttle_reasons(h)))
                    fh.flush()
                    time.sleep(1.0 / a.fast_hz if dt < a.fast_s else 1.0)
                sq.assert_alive(beats, a.cool_s)
            finally:
                sq.__exit__()
            cold = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
            print(f"  cooled {hot} -> {cold} C", flush=True)
    finally:
        fh.close()
        reset_clocks()
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
