#!/usr/bin/env python3
"""GEMM DVFS x SM-count sweep, with the SM count set by an __nanosleep squatter.

Grid comes from gen_config_table.py so there is a single source of truth:
  12 shapes x 6 clocks x 5 SM points = 360 configurations, x --reps.

Per configuration this records latency, power, achieved SM clock, NVML throttle
reasons, temperature, an in-situ power baseline, and TWO independent guards that the
SM axis actually held. Everything here is shaped by what has already been measured to
go silently wrong:

* DEVICE-WIDE SYNC DEADLOCKS under a resident squatter -- it waits out the backstop.
  torch.cuda.synchronize(), torch.cuda.empty_cache(), cudaHostAlloc/cudaFreeHost and
  torch.cuda.graph() all qualify. Only stream-scoped sync is used inside a squatter,
  and probe_warmup() plus every GEMM warmup happens OUTSIDE one.

* THE SM AXIS CAN BE SILENTLY WRONG. A squatter whose shared memory sits in a lower
  carveout class also excludes its TPC partner, giving the GEMM 108-2n SMs while
  every "n distinct smids" check still passes. So each configuration re-runs the
  147456 B probe and asserts the reachable set is exactly 108-n and disjoint from the
  squatted set. ~60 ms, and it is the only direct evidence.

* A SQUATTER CAN EXPIRE MID-WINDOW. `ready` still reads n/n afterwards, so the row
  looks clean while holding 108-SM numbers. The heartbeat must cover the whole window
  (measure() asserts it) and the backstop is sized well past the slowest config.

* THE CLOCK IS A REQUEST, NOT A FACT. At 1410 MHz a large GEMM at n=0 clamps to
  ~1185-1260 MHz on SwPowerCap while the same shape at n>=54 holds 1410 -- the clock
  rises WITH n, so inside that plane the SM effect is not identifiable. Those rows are
  still collected (1410 is a real operating point) but carry clock_held=False; SM
  scaling must be read off clocks <= 1200, which hold everywhere measured.

* ORDER ALIASES TEMPERATURE ONTO THE AXES. Freq-outer ordering runs a whole plane
  coldest-first (38 -> 63 C observed). All three loops are shuffled per rep.

  python3 sweep_driver.py --out data/gemm_squat_sweep.csv --reps 3
  python3 sweep_driver.py --out ... --limit-shapes 2 --limit-freqs 1 --reps 1   # smoke
  python3 sweep_driver.py --out ... --resume
"""

import argparse
import atexit
import csv
import os
import random
import signal
import subprocess
import sys
import time
import traceback

import torch

from gen_config_table import FREQS, M_LIST, NK_LIST, SQUAT, TOTAL_SM
from measure import Sampler, measure
from squatter import Squatter, probe_available_sms, probe_warmup

DEV = 0
GPUS = "0"
DTYPE = torch.bfloat16
# well past the slowest configuration (~0.6 s/iter at 300 MHz / 14 SM, window ~2 s)
SQUAT_BACKSTOP_S = 300.0

FIELDS = [
    "rep", "m", "n", "k", "freq_req_mhz", "n_squat", "sm_avail",
    "latency_ms", "power_w", "energy_mj", "gflops",
    "clock_sm_med", "clock_sm_min", "clock_held", "throttle", "temp_c",
    "p_squat_w", "probe_sm_count", "probe_ok",
    "iters", "window_s", "n_samples", "power_min_w", "power_max_w", "status",
]
_locked = False


def sh(c):
    return subprocess.run(c, capture_output=True, text=True)


def reset_clocks():
    global _locked
    if _locked:
        r = sh(["sudo", "nvidia-smi", "-i", GPUS, "-rgc"])
        _locked = False
        print(f"[clocks] reset -> {'ok' if r.returncode == 0 else r.stderr.strip()}", flush=True)


def lock_clock(f, tol=20):
    """Lock and verify by read-back. A failed lock silently mislabels a whole plane."""
    global _locked
    r = sh(["sudo", "nvidia-smi", "-i", GPUS, "-lgc", f"{f},{f}"])
    if r.returncode != 0:
        print(f"[clocks] LOCK FAILED {f}: {r.stderr.strip()}", flush=True)
        return False
    _locked = True
    time.sleep(0.4)
    q = sh(["nvidia-smi", "-i", GPUS, "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"])
    try:
        got = int(q.stdout.strip().splitlines()[0])
    except Exception:
        return False
    if abs(got - f) > tol:
        print(f"[clocks] read-back {got} != {f}", flush=True)
        return False
    return True


def _sig(s, _f):
    reset_clocks()
    sys.exit(128 + s)


def build_shapes(limit_shapes=0):
    shapes = [(m, n, k) for (n, k) in NK_LIST for m in M_LIST]
    if limit_shapes:
        shapes = shapes[:limit_shapes]
    return shapes


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--out", default=os.path.join(here, "data", "gemm_squat_sweep.csv"))
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--limit-shapes", type=int, default=0)
    ap.add_argument("--limit-freqs", type=int, default=0)
    ap.add_argument("--limit-squat", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=20260806)
    a = ap.parse_args()

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)
    atexit.register(reset_clocks)
    torch.cuda.set_device(DEV)

    freqs = FREQS[: a.limit_freqs] if a.limit_freqs else list(FREQS)
    squat = SQUAT[: a.limit_squat] if a.limit_squat else list(SQUAT)
    shapes = build_shapes(a.limit_shapes)
    total = a.reps * len(freqs) * len(squat) * len(shapes)
    print(f"device {torch.cuda.get_device_name(DEV)}  SM={TOTAL_SM}")
    print(f"grid: {len(shapes)} shapes x {len(freqs)} clocks x {len(squat)} SM points "
          f"x {a.reps} reps = {total} rows -> {a.out}", flush=True)

    # --- everything that device-syncs must happen before any squatter exists -------
    print("warmup (outside any squatter): probe module + all shapes ...", flush=True)
    probe_warmup(TOTAL_SM)
    st = torch.cuda.current_stream()
    fns = {}
    for (m, n, k) in shapes:
        x = torch.randn((m, k), dtype=DTYPE, device=f"cuda:{DEV}")
        w = torch.randn((n, k), dtype=DTYPE, device=f"cuda:{DEV}")
        fns[(m, n, k)] = lambda x=x, w=w: torch.nn.functional.linear(x, w)
        for _ in range(6):
            fns[(m, n, k)]()
    st.synchronize()
    print("warmup done", flush=True)

    out = os.path.abspath(a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = set()
    if a.resume and os.path.exists(out):
        for row in csv.DictReader(open(out)):
            try:
                done.add((int(row["rep"]), int(row["m"]), int(row["n"]), int(row["k"]),
                          int(row["freq_req_mhz"]), int(row["n_squat"])))
            except (KeyError, ValueError):
                pass
        print(f"resume: {len(done)} rows already present", flush=True)
    new_file = not (a.resume and os.path.exists(out))
    fh = open(out, "w" if new_file else "a", newline="")
    wr = csv.DictWriter(fh, fieldnames=FIELDS)
    if new_file:
        wr.writeheader()
        fh.flush()

    rng = random.Random(a.seed)
    t0 = time.time()
    n_ok = n_skip = n_fail = 0
    try:
        for rep in range(1, a.reps + 1):
            fo = freqs[:]; rng.shuffle(fo)                 # break thermal aliasing
            for freq in fo:
                if not lock_clock(freq):
                    print(f"  skipping clock {freq}", flush=True)
                    continue
                so = squat[:]; rng.shuffle(so)
                for (nsq, sm_avail) in so:
                    todo = [s for s in shapes
                            if (rep, s[0], s[1], s[2], freq, nsq) not in done]
                    if not todo:
                        continue
                    rng.shuffle(todo)
                    sq = None
                    try:
                        if nsq:
                            sq = Squatter(nsq, device=f"cuda:{DEV}",
                                          max_seconds=SQUAT_BACKSTOP_S).__enter__()
                        # (1) in-situ power baseline, same treatment at every n
                        s = Sampler(DEV); s.start(); time.sleep(2.0)
                        p_squat = (s.stop() or {}).get("power_w", "")
                        # (2) direct proof the SM axis held for THIS configuration
                        avail = probe_available_sms()
                        probe_ok = len(avail) == sm_avail and not (
                            avail & {v - 1 for v in sq.ready.tolist()} if sq else set())
                        if not probe_ok:
                            print(f"  PROBE FAIL freq={freq} n={nsq}: reachable "
                                  f"{len(avail)} != {sm_avail}", flush=True)
                        for (m, n, k) in todo:
                            row = dict.fromkeys(FIELDS, "")
                            row.update(rep=rep, m=m, n=n, k=k, freq_req_mhz=freq,
                                       n_squat=nsq, sm_avail=sm_avail,
                                       p_squat_w=p_squat, probe_sm_count=len(avail),
                                       probe_ok=probe_ok)
                            try:
                                r = measure(fns[(m, n, k)], squatter=sq,
                                            requested_clock=freq, dev=DEV)
                                gf = 2 * m * n * k / 1e9
                                row.update(
                                    latency_ms=round(r["latency_ms"], 6),
                                    power_w=r.get("power_w", ""),
                                    energy_mj=(round(r["power_w"] * r["latency_ms"], 4)
                                               if r.get("power_w") else ""),
                                    gflops=round(gf / (r["latency_ms"] / 1e3), 1),
                                    clock_sm_med=r.get("clock_sm_med", ""),
                                    clock_sm_min=r.get("clock_sm_min", ""),
                                    clock_held=r.get("clock_held", ""),
                                    throttle=r.get("throttle", ""),
                                    temp_c=r.get("temp_c", ""),
                                    iters=r["iters"], window_s=r["window_s"],
                                    n_samples=r.get("n_samples", ""),
                                    power_min_w=r.get("power_min_w", ""),
                                    power_max_w=r.get("power_max_w", ""),
                                    status="ok" if probe_ok else "probe_fail")
                                n_ok += 1
                            except Exception as e:
                                # a heartbeat failure means the SM count was not held
                                row["status"] = f"{type(e).__name__}: {str(e)[:90]}"
                                n_fail += 1
                                print(f"  DROP {(m, n, k)} f={freq} n={nsq}: "
                                      f"{row['status']}", flush=True)
                            wr.writerow(row); fh.flush()
                    except Exception:
                        n_skip += len(todo)
                        print(f"  GROUP FAILED freq={freq} n={nsq}\n"
                              f"{traceback.format_exc()}", flush=True)
                    finally:
                        if sq is not None:
                            try:
                                sq.__exit__()
                            except Exception:
                                pass
                    el = time.time() - t0
                    print(f"  rep{rep} f={freq:>4} n_sq={nsq:>3} -> {sm_avail:>3} SM  "
                          f"ok={n_ok} fail={n_fail}  {el / 60:.1f} min", flush=True)
    finally:
        fh.close()
        reset_clocks()
    print(f"DONE in {(time.time() - t0) / 60:.1f} min   ok={n_ok} fail={n_fail} "
          f"skip={n_skip}  -> {out}", flush=True)


if __name__ == "__main__":
    main()
