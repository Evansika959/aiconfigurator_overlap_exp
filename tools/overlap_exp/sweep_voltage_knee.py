#!/usr/bin/env python3
"""Locate the voltage step, and with it the true energy-optimal clock.

WHY. kappa = a/f is C*V^2 -- the voltage curve read out of the power data. Measured on
the six-clock grid it is flat to 0.5% CV from 300 to 900 MHz, then jumps +33% by 1200
and +41% more by 1410. So the GPU sits at its floor voltage up to somewhere past 900 and
ramps after. We have ZERO calibration points inside 900-1200.

That gap matters three times over:
  * 58 of 72 concurrent (shape, CTA) pairs pick 900 MHz as energy-optimal -- but 900 is
    the EDGE of the grid, which is what an unresolved optimum looks like. The real one
    is wherever the voltage starts climbing, and that is inside the gap.
  * all 39 throttled rows settled between 1170 and 1395 MHz, so every coefficient used
    to re-score them was interpolated across this gap on two endpoints.
  * the closed-loop solver f* : P(f*) = cap searches inside it.

The hardware exposes 15 MHz DVFS steps, so the curve can simply be measured.

WHAT IS AND IS NOT MEASURED HERE. This is single-GPU, GEMM-only, at full 108 SMs. It
gives P_total(f), P_idle(f), latency(f) and hence energy(f) at 15 MHz resolution --
enough to locate the knee and the energy optimum. It does NOT re-fit a(f) and b
separately; that needs the squatter rig varying SM count, which is the follow-up if the
knee turns out to sit where it changes the answer.

  python3 sweep_voltage_knee.py            # ~10 min
"""
import argparse, atexit, csv, os, signal, statistics as st, subprocess, sys, threading, time
import pynvml, torch

DEV, GPUS = 0, "0"
CLOCKS = [510, 705, 780] + list(range(855, 1246, 15))
SHAPES = [(1024, 4096, 4096), (4096, 8192, 8192), (8192, 16384, 16384)]
# The original run stopped at 1245 because the largest shape hit the 400 W cap there and
# throttled. Smaller shapes still had headroom (262 W at 1245), so the ceiling is the
# WORKLOAD's, not the sweep's -- these overrides let the range be pushed per shape.
if os.environ.get("KNEE_CLOCKS"):
    CLOCKS = [int(x) for x in os.environ["KNEE_CLOCKS"].split(",")]
if os.environ.get("KNEE_SHAPES"):
    SHAPES = [tuple(int(v) for v in t.split("x"))
              for t in os.environ["KNEE_SHAPES"].split(",")]
SM = 108
WINDOW_S, DISCARD_S = 2.0, 0.5
HERE = os.path.dirname(os.path.abspath(__file__))
_locked = _pm = None


def sh(c): return subprocess.run(c, capture_output=True, text=True)


def restore():
    global _locked
    if _locked:
        sh(["sudo", "nvidia-smi", "-i", GPUS, "-rgc"])
        if _pm != "Enabled":
            sh(["sudo", "nvidia-smi", "-i", GPUS, "-pm", "0"])
        _locked = None
        print("[clocks] restored", flush=True)


signal.signal(signal.SIGINT, lambda *a: (restore(), sys.exit(130)))
signal.signal(signal.SIGTERM, lambda *a: (restore(), sys.exit(143)))
atexit.register(restore)


class Meter:
    """Poll at 50 Hz and bracket with the energy counter. The counter is an INTEGRATOR --
    the difference of two readings is exact energy over the interval, with none of the
    sampling bias a poller has. Where they disagree, trust the counter."""

    def __init__(self, h):
        self.h = h; self._stop = threading.Event(); self._p = []

    def _loop(self):
        while not self._stop.is_set():
            try:
                self._p.append((time.time(),
                                pynvml.nvmlDeviceGetPowerUsage(self.h) / 1000,
                                pynvml.nvmlDeviceGetClockInfo(self.h, pynvml.NVML_CLOCK_SM)))
            except Exception:
                pass
            self._stop.wait(0.02)

    def __enter__(self):
        self._p.clear(); self._stop.clear()
        self.e0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(self.h); self.t0 = time.time()
        self._t = threading.Thread(target=self._loop, daemon=True); self._t.start(); return self

    def __exit__(self, *a):
        self._stop.set(); self._t.join(timeout=2)
        self.e1 = pynvml.nvmlDeviceGetTotalEnergyConsumption(self.h); self.t1 = time.time()

    def out(self):
        k = [x for x in self._p if x[0] - self.t0 >= DISCARD_S] or self._p
        return dict(p_poll=round(st.median(x[1] for x in k), 2),
                    p_energy=round((self.e1 - self.e0) / (self.t1 - self.t0) / 1000, 2),
                    clock_min=min(x[2] for x in k), clock_med=int(st.median(x[2] for x in k)),
                    n_samples=len(k))


def main():
    global _locked, _pm
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "data", "voltage_knee.csv"))
    a = ap.parse_args()
    pynvml.nvmlInit(); h = pynvml.nvmlDeviceGetHandleByIndex(DEV)
    _pm = sh(["nvidia-smi", "-i", GPUS, "--query-gpu=persistence_mode",
              "--format=csv,noheader"]).stdout.strip()
    if _pm != "Enabled":
        sh(["sudo", "nvidia-smi", "-i", GPUS, "-pm", "1"])
    torch.cuda.set_device(DEV)
    rows = []
    print(f"{len(CLOCKS)} clocks x {len(SHAPES)} shapes, 15 MHz steps through the knee\n")
    print(f"{'f':>6}{'idle W':>9}" + "".join(f"{f'{m//1024}k x{nk//1024}k':>22}" for m, nk, _ in SHAPES))
    print(f"{'':>6}{'':>9}" + "".join(f"{'W':>7}{'ms':>8}{'mJ':>7}" for _ in SHAPES))
    for f in CLOCKS:
        sh(["sudo", "nvidia-smi", "-i", GPUS, "-lgc", f"{f},{f}"]); _locked = True
        time.sleep(1.0)
        with Meter(h) as mt:          # idle floor at this clock
            time.sleep(1.5)
        idle = mt.out()
        line = f"{f:>6}{idle['p_energy']:>9.1f}"
        for (m, n, k) in SHAPES:
            x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
            w = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
            fn = lambda: torch.nn.functional.linear(x, w)
            for _ in range(8):
                fn()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(10):
                fn()
            torch.cuda.synchronize()
            lat = (time.perf_counter() - t0) / 10
            it = max(10, int((WINDOW_S + DISCARD_S) / lat))
            with Meter(h) as mt:
                for _ in range(it):
                    fn()
                torch.cuda.synchronize()
            r = mt.out()
            lat_ms = lat * 1e3
            e_mj = r["p_energy"] * lat_ms
            rows.append(dict(clock=f, m=m, n=n, k=k, sm=SM,
                             idle_w=idle["p_energy"], latency_ms=round(lat_ms, 5),
                             energy_mj=round(e_mj, 2), iters=it,
                             clock_held=r["clock_min"] >= f - 20, **r))
            line += f"{r['p_energy']:>7.0f}{lat_ms:>8.3f}{e_mj:>7.0f}" + ("*" if r["clock_min"] < f - 20 else "")
            del x, w
            torch.cuda.empty_cache()
        print(line, flush=True)
    restore()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)
    print(f"\nwrote {a.out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
