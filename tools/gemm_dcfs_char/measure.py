#!/usr/bin/env python3
"""Measurement primitive for the squatter sweep: latency + power + ACHIEVED clock.

Two review findings shape every choice here.

CLOCK IS NOT WHAT YOU ASKED FOR. `nvidia-smi -lgc f,f` is a request. Under a
400 W-capped GEMM at full SM count the board clamps below it -- measured 1275 MHz
while 1410 was locked, and the clamp is correlated with BOTH the SM count and the
shape, because fewer SMs draw less power and therefore hold the requested clock.
An earlier analysis read that clamp as "sublinear scaling from more L2 per SM";
it was DVFS. Every sample therefore records the achieved SM clock and the NVML
throttle reasons, and the achieved clock -- never the requested one -- is the
regressor.

NVML POWER IS SLOW. On this A100 the reading updates about every 105 ms and needs
~427 ms to reach 95% of steady state. Sub-second windows swing +-43%: the same
configuration read 64.0 / 482.8 / 82.9 W on three 130 ms reps. So a window is at
least MIN_WINDOW_S long and the first DISCARD_S is thrown away before any
statistic is taken.

DEVICE-WIDE SYNC IS FORBIDDEN while a squatter is resident -- it waits for the
squatter's backstop. Measured under a 6 s backstop: torch.cuda.synchronize() 6.29 s,
torch.cuda.empty_cache() 6.29 s, the process's FIRST cuBLAS call 6.44 s. Anything
used inside the window must be warmed outside it; only stream-scoped sync is used
below.
"""

import statistics
import threading
import time

import pynvml
import torch

SAMPLE_INTERVAL_S = 0.05
MIN_WINDOW_S = 1.5      # >= 1 s of steady state after the discard
DISCARD_S = 0.5         # NVML needs ~427 ms to reach 95% of steady state
CLOCK_TOL_MHZ = 20

_THROTTLE_BITS = {
    0x0001: "GpuIdle",
    0x0002: "AppClocksSetting",
    0x0004: "SwPowerCap",
    0x0008: "HwSlowdown",
    0x0010: "SyncBoost",
    0x0020: "SwThermal",
    0x0040: "HwThermal",
    0x0080: "HwPowerBrake",
    0x0100: "DisplayClock",
}
_nvml_ready = False


def _nvml(dev=0):
    global _nvml_ready
    if not _nvml_ready:
        pynvml.nvmlInit()
        _nvml_ready = True
    return pynvml.nvmlDeviceGetHandleByIndex(dev)


def _throttle_reasons(h):
    for name in ("nvmlDeviceGetCurrentClocksEventReasons",
                 "nvmlDeviceGetCurrentClocksThrottleReasons"):
        fn = getattr(pynvml, name, None)
        if fn:
            try:
                return int(fn(h))
            except Exception:
                pass
    return 0


class Sampler:
    """Background NVML sampler: power, achieved SM clock, throttle reasons, temp."""

    def __init__(self, dev=0):
        self.h = _nvml(dev)
        self._stop = threading.Event()
        self._rows = []
        self._t = None

    def _loop(self):
        while not self._stop.is_set():
            try:
                self._rows.append((
                    time.time(),
                    pynvml.nvmlDeviceGetPowerUsage(self.h) / 1000.0,
                    pynvml.nvmlDeviceGetClockInfo(self.h, pynvml.NVML_CLOCK_SM),
                    _throttle_reasons(self.h),
                    pynvml.nvmlDeviceGetTemperature(self.h, pynvml.NVML_TEMPERATURE_GPU),
                    pynvml.nvmlDeviceGetPerformanceState(self.h),
                ))
            except Exception:
                pass
            self._stop.wait(SAMPLE_INTERVAL_S)

    def start(self):
        self._rows.clear()
        self._stop.clear()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def stop(self, discard_s=DISCARD_S):
        self._stop.set()
        self._t.join(timeout=2.0)
        rows = list(self._rows)
        if not rows:
            return None
        t0 = rows[0][0]
        kept = [r for r in rows if r[0] - t0 >= discard_s] or rows
        pw = [r[1] for r in kept]
        clk = [r[2] for r in kept]
        thr = 0
        for r in kept:
            thr |= r[3]
        return dict(
            power_w=round(statistics.median(pw), 3),
            power_min_w=round(min(pw), 3),
            power_max_w=round(max(pw), 3),
            clock_sm_med=int(statistics.median(clk)),
            clock_sm_min=int(min(clk)),
            clock_sm_max=int(max(clk)),
            temp_c=int(statistics.median(r[4] for r in kept)),
            throttle_bits=thr,
            throttle=",".join(n for b, n in _THROTTLE_BITS.items() if thr & b) or "none",
            # P-state is NOT a usable "is the clock up?" signal on A100: the board
            # reports P0 while idling the SM clock down to 210 MHz. Recorded only to
            # document that; clock_sm_med is the regressor that actually matters.
            pstate=int(statistics.median(r[5] for r in kept)),
            n_samples=len(kept),
        )


def measure(fn, squatter=None, requested_clock=None, dev=0,
            min_window_s=MIN_WINDOW_S, warmup=5):
    """Time `fn` and sample the GPU over the same window.

    `fn` must already have been warmed OUTSIDE any squatter context: the first cuBLAS
    call in a process device-synchronises and would wait out the squatter's backstop.
    """
    st = torch.cuda.current_stream()
    for _ in range(warmup):
        fn()
    st.synchronize()

    beats_before = squatter.beats() if squatter is not None and squatter.n > 0 else None

    # size the loop so the window covers the discard plus >=1 s of steady state
    t = time.perf_counter()
    fn()
    st.synchronize()
    one = time.perf_counter() - t
    iters = max(3, int((min_window_s + DISCARD_S) / max(one, 1e-6)))

    s = Sampler(dev)
    s.start()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    st.synchronize()
    wall = time.perf_counter() - t0
    stats = s.stop()

    if beats_before is not None:
        squatter.assert_alive(beats_before, wall)   # raises unless residency covered the WHOLE window

    out = dict(latency_ms=wall / iters * 1e3, iters=iters, window_s=round(wall, 3))
    out.update(stats or {})
    if requested_clock is not None:
        out["clock_req_mhz"] = requested_clock
        out["clock_held"] = bool(stats and stats["clock_sm_min"] >= requested_clock - CLOCK_TOL_MHZ)
    return out
