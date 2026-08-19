#!/usr/bin/env python3
"""Measure the sawtooth that the wave staircase predicts in mean power.

THE POINT. P(t) for a GEMM is a staircase -- S busy SMs for W-1 waves, then a partial
tail wave. No instrument here can see it directly: NVML's power reading updates at
~10 Hz, its buffered samples at 50 Hz, and its energy counter at 10 Hz, against kernels
of 0.2-3 ms. But the staircase leaves a signature in a quantity a slow instrument CAN
measure -- the mean.

Sweeping M at fixed N, K and SM count walks the tile count across wave boundaries:

    S_eff = tiles / ceil(tiles / S)

climbs linearly to S within a wave and DROPS the instant a new wave opens. Mean power
a*S_eff + b therefore traces a sawtooth with a ~95 W peak-to-trough at 1200 MHz. A model
without waves -- constant power per SM, or power proportional to total work -- predicts a
flat line. The two are not close.

THREE INSTRUMENTS, because they fail differently:

  nvmlDeviceGetPowerUsage     ~10 Hz sampler. What every earlier script here used.
  nvmlDeviceGetSamples        50 Hz, buffered by the driver with microsecond stamps.
                              5x finer, and it was available all along.
  nvmlDeviceGetTotalEnergyConsumption
                              10 Hz, but an INTEGRATOR: the difference of two readings
                              is the exact energy in between, with no sampling bias.
                              Energy/time is therefore an unbiased mean power, which the
                              samplers are not.

Agreement between the three is the check that the mean itself is trustworthy before
anything is concluded from its shape.

  python3 sweep_sawtooth.py                 # ~12 min
"""

import argparse
import atexit
import csv
import math
import os
import signal
import statistics as st
import subprocess
import sys
import threading
import time

import pynvml
import torch

DEV, GPUS = 0, "0"
CLOCK = 1200
N, K = 4096, 8192
SM = 108
# M in steps of 256 (one tile row) walks tiles across five wave boundaries
M_LIST = list(range(256, 4097, 256))
REPS = 3
WINDOW_S = 3.0
DISCARD_S = 0.8
TILE_M, TILE_N = 256, 128        # system torch 2.9/cu129 -> ampere_bf16..._256x128
_locked = _pm_was = None


def sh(c):
    return subprocess.run(c, capture_output=True, text=True)


def restore():
    global _locked
    if _locked:
        sh(["sudo", "nvidia-smi", "-i", GPUS, "-rgc"])
        if _pm_was == "Disabled":
            sh(["sudo", "nvidia-smi", "-i", GPUS, "-pm", "0"])
        _locked = None
        print("[clocks] restored", flush=True)


def _sig(s, _f):
    restore(); sys.exit(128 + s)


class TripleMeter:
    """Poll the 10 Hz sampler while the run proceeds; read the energy counter and drain
    the driver's 50 Hz sample buffer at the boundaries."""

    def __init__(self, h):
        self.h = h
        self._stop, self._p = threading.Event(), []

    def _loop(self):
        while not self._stop.is_set():
            try:
                self._p.append((time.time(),
                                pynvml.nvmlDeviceGetPowerUsage(self.h) / 1000))
            except Exception:
                pass
            self._stop.wait(0.02)

    def __enter__(self):
        self._p.clear(); self._stop.clear()
        try:                                     # drain stale buffered samples
            pynvml.nvmlDeviceGetSamples(self.h, pynvml.NVML_TOTAL_POWER_SAMPLES, 0)
        except Exception:
            pass
        self.e0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(self.h)
        self.t0 = time.time()
        self._t = threading.Thread(target=self._loop, daemon=True); self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set(); self._t.join(timeout=2)
        self.e1 = pynvml.nvmlDeviceGetTotalEnergyConsumption(self.h)
        self.t1 = time.time()
        try:
            _, s = pynvml.nvmlDeviceGetSamples(self.h, pynvml.NVML_TOTAL_POWER_SAMPLES,
                                               int(self.t0 * 1e6))
            self.buf = [x.sampleValue.uiVal / 1000 for x in s]
        except Exception:
            self.buf = []

    def results(self):
        kept = [p for t, p in self._p if t - self.t0 >= DISCARD_S] or \
               [p for _, p in self._p]
        nb = int(len(self.buf) * DISCARD_S / max(self.t1 - self.t0, 1e-9))
        buf = self.buf[nb:] or self.buf
        return dict(
            p_poll=round(st.median(kept), 2) if kept else "",
            # energy/time is an unbiased mean over the WHOLE window, discard included --
            # the counter cannot be sliced, which is the price of it being exact
            p_energy=round((self.e1 - self.e0) / (self.t1 - self.t0) / 1000, 2),
            p_buffered=round(st.median(buf), 2) if buf else "",
            n_poll=len(kept), n_buf=len(buf))


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--out", default=os.path.join(here, "data", "wave_sawtooth.csv"))
    a = ap.parse_args()
    global _locked, _pm_was
    signal.signal(signal.SIGINT, _sig); signal.signal(signal.SIGTERM, _sig)
    atexit.register(restore)

    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(DEV)
    _pm_was = sh(["nvidia-smi", "-i", GPUS, "--query-gpu=persistence_mode",
                  "--format=csv,noheader"]).stdout.strip()
    if _pm_was != "Enabled":
        sh(["sudo", "nvidia-smi", "-i", GPUS, "-pm", "1"])
    sh(["sudo", "nvidia-smi", "-i", GPUS, "-lgc", f"{CLOCK},{CLOCK}"])
    _locked = True
    time.sleep(1)
    print(f"[clocks] {CLOCK} MHz locked (persistence was {_pm_was})\n", flush=True)

    torch.cuda.set_device(DEV)
    rows = []
    print(f"{'M':>6}{'tiles':>7}{'W':>4}{'S_eff':>8}{'pred':>7} | "
          f"{'poll':>7}{'buffered':>10}{'energy':>8} | {'lat ms':>8}{'err':>7}")
    for rep in range(1, REPS + 1):
        for m in M_LIST:
            x = torch.randn((m, K), dtype=torch.bfloat16, device=f"cuda:{DEV}")
            w = torch.randn((N, K), dtype=torch.bfloat16, device=f"cuda:{DEV}")
            fn = lambda: torch.nn.functional.linear(x, w)
            for _ in range(10):
                fn()
            torch.cuda.synchronize()

            t = time.perf_counter()
            for _ in range(20):
                fn()
            torch.cuda.synchronize()
            lat = (time.perf_counter() - t) / 20 * 1e3
            iters = max(20, int(WINDOW_S / (lat / 1e3)))

            with TripleMeter(h) as meter:
                for _ in range(iters):
                    fn()
                torch.cuda.synchronize()
            r = meter.results()

            tiles = -(-m // TILE_M) * -(-N // TILE_N)
            wv = -(-tiles // SM)
            se = tiles / wv
            pred = 69.66 + 2.6658 * se + 12.1
            err = (pred - r["p_energy"]) / r["p_energy"] * 100
            rows.append(dict(rep=rep, m=m, n=N, k=K, sm=SM, tiles=tiles, waves=wv,
                             s_eff=round(se, 2), pred_w=round(pred, 1),
                             latency_ms=round(lat, 5), iters=iters, **r))
            if rep == 1:
                print(f"{m:>6}{tiles:>7}{wv:>4}{se:>8.1f}{pred:>7.0f} | "
                      f"{r['p_poll']:>7}{r['p_buffered']:>10}{r['p_energy']:>8} | "
                      f"{lat:>8.3f}{err:>6.1f}%", flush=True)
            del x, w
            torch.cuda.empty_cache()
    restore()

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)
    print(f"\nwrote {a.out}  ({len(rows)} rows)")

    # --- do the three instruments agree? -------------------------------------
    d_pe = [abs(r["p_poll"] - r["p_energy"]) for r in rows if r["p_poll"] != ""]
    d_be = [abs(r["p_buffered"] - r["p_energy"]) for r in rows if r["p_buffered"] != ""]
    print(f"\ninstrument agreement (vs the energy counter, W): "
          f"poll median {st.median(d_pe):.2f}, buffered median "
          f"{st.median(d_be):.2f}" if d_be else "")

    # --- sawtooth vs flat -----------------------------------------------------
    by = {}
    for r in rows:
        by.setdefault(r["m"], []).append(r["p_energy"])
    ms = sorted(by)
    meas = [st.median(by[m]) for m in ms]
    wave, flat = [], []
    for m in ms:
        tiles = -(-m // TILE_M) * -(-N // TILE_N)
        wv = -(-tiles // SM)
        wave.append(69.66 + 2.6658 * (tiles / wv) + 12.1)
        flat.append(69.66 + 2.6658 * min(SM, tiles) + 12.1)   # no waves: always S busy
    ew = [abs(p - q) / q for p, q in zip(wave, meas)]
    ef = [abs(p - q) / q for p, q in zip(flat, meas)]
    print(f"\n{'M':>6}{'tiles':>7}{'W':>4}{'measured':>10}{'wave model':>12}"
          f"{'flat model':>12}")
    for m, q, pw, pf in zip(ms, meas, wave, flat):
        tiles = -(-m // TILE_M) * -(-N // TILE_N)
        print(f"{m:>6}{tiles:>7}{-(-tiles // SM):>4}{q:>10.1f}{pw:>12.1f}{pf:>12.1f}")
    print(f"\n  wave model  median error {st.median(ew) * 100:5.2f}%")
    print(f"  flat model  median error {st.median(ef) * 100:5.2f}%")
    drops = [i for i in range(1, len(ms)) if wave[i] < wave[i - 1]]
    print(f"\n  the model predicts drops at M = {[ms[i] for i in drops]}")
    print(f"  measured change there:     "
          f"{[f'{meas[i] - meas[i - 1]:+.0f} W' for i in drops]}")


if __name__ == "__main__":
    main()
