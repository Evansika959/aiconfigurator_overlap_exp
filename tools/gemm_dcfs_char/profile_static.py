#!/usr/bin/env python3
"""P0 static power of the GPU vs clock and vs number of occupied SMs.

WHY. The sweep records BOARD power, which is static + dynamic. Only the dynamic
part is attributable to the GEMM, so we need P_static(f) to subtract. Same idea as
zeus's examples/pipeline_frequency_optimizer/profile_p2p.py, which parks rank 0 in a
blocking dist.recv() (a resident, spinning NCCL kernel doing no useful work) while
rank 1 sleeps 60 s, and calls the power drawn over that window the P0 static power.

THE TRAP THIS SCRIPT EXISTS TO CATCH. `nvmlDeviceGetPerformanceState` is useless as
a "is the clock up?" signal on A100 -- measured on this box, the GPU reports P0 while
idling the SM clock down to 210 MHz at 58.9 W. So "P0 static power" is not one number
per board; it is a number per ACHIEVED CLOCK, and an idle GPU is not at the clock you
locked. Every row here therefore carries the achieved SM clock alongside the power,
and the headline question is whether `nvidia-smi -lgc f,f` holds f with NOTHING
resident. If it does not, the n=0 baseline already in gemm_squat_sweep.csv was taken
at an unknown lower clock and understates P_static -- which would inflate every
dynamic-energy number derived from it.

WHY THE SQUATTER IS THE RIGHT PROBE. zeus keeps the GPU busy with a SPINNING kernel,
which burns real dynamic power in its busy-wait loop. __nanosleep parks the SM with
almost no switching activity, so P(squat) is a closer approximation to true static
power at the same clock. Sweeping n also separates the two candidate readings of
"static": the whole board's floor (n=0) vs the floor with n SMs held (n>0).

  python3 profile_static.py --out data/p0_static.csv           # ~19 min
  python3 profile_static.py --reps 1 --window 4                # quick look
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

import torch

from measure import Sampler
from squatter import Squatter

DEV = 0
GPUS = "0"
FREQS = [1410, 1200, 900, 705, 510, 300]
# 0 = nothing resident at all.  The rest occupy n SMs with __nanosleep blocks;
# 1 isolates the fixed cost of having ANY resident kernel, 108 fills the board.
SQUAT_N = [0, 1, 14, 27, 54, 81, 94, 108]
FIELDS = [
    "rep", "freq_req_mhz", "n_squat", "sm_idle",
    "power_w", "power_min_w", "power_max_w",
    "clock_sm_med", "clock_sm_min", "clock_sm_max", "clock_held",
    "pstate", "temp_c", "throttle", "n_samples", "window_s", "status",
]
CLOCK_TOL_MHZ = 20
_locked = False


def sh(c):
    return subprocess.run(c, capture_output=True, text=True)


def reset_clocks():
    global _locked
    if _locked:
        r = sh(["sudo", "nvidia-smi", "-i", GPUS, "-rgc"])
        _locked = False
        print(f"[clocks] reset -> {'ok' if r.returncode == 0 else r.stderr.strip()}", flush=True)


def lock_clock(f):
    global _locked
    r = sh(["sudo", "nvidia-smi", "-i", GPUS, "-lgc", f"{f},{f}"])
    if r.returncode != 0:
        print(f"[clocks] LOCK FAILED {f}: {r.stderr.strip()}", flush=True)
        return False
    _locked = True
    time.sleep(0.4)
    return True


def _sig(s, _f):
    reset_clocks()
    sys.exit(128 + s)


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--out", default=os.path.join(here, "data", "p0_static.csv"))
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--window", type=float, default=6.0, help="sampled seconds per point")
    ap.add_argument("--discard", type=float, default=2.0,
                    help="leading seconds dropped; NVML needs ~427 ms to reach 95%% "
                         "of steady state and the clock change needs longer")
    ap.add_argument("--seed", type=int, default=20260806)
    a = ap.parse_args()

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)
    atexit.register(reset_clocks)
    torch.cuda.set_device(DEV)
    torch.zeros(1, device=f"cuda:{DEV}")          # materialise the CUDA context first

    out = os.path.abspath(a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fh = open(out, "w", newline="")
    wr = csv.DictWriter(fh, fieldnames=FIELDS)
    wr.writeheader()
    fh.flush()

    total = a.reps * len(FREQS) * len(SQUAT_N)
    per = a.window + a.discard + 2.0
    print(f"device {torch.cuda.get_device_name(DEV)}")
    print(f"{len(FREQS)} clocks x {len(SQUAT_N)} occupancies x {a.reps} reps = {total} points"
          f"  (~{total * per / 60:.0f} min)  -> {out}", flush=True)

    rng = random.Random(a.seed)
    t0 = time.time()
    n_ok = n_fail = 0
    try:
        for rep in range(1, a.reps + 1):
            fo = FREQS[:]; rng.shuffle(fo)         # break thermal aliasing onto the clock axis
            for freq in fo:
                if not lock_clock(freq):
                    continue
                so = SQUAT_N[:]; rng.shuffle(so)
                for nsq in so:
                    row = dict.fromkeys(FIELDS, "")
                    row.update(rep=rep, freq_req_mhz=freq, n_squat=nsq, sm_idle=108 - nsq)
                    sq = None
                    try:
                        if nsq:
                            sq = Squatter(nsq, device=f"cuda:{DEV}",
                                          max_seconds=a.window + a.discard + 60.0).__enter__()
                            beats = sq.beats()
                        s = Sampler(DEV)
                        s.start()
                        time.sleep(a.window + a.discard)
                        st = s.stop(discard_s=a.discard)
                        if sq is not None:
                            # a squatter that expired mid-window would leave the board
                            # idling and the row would read as "static at n SMs"
                            sq.assert_alive(beats, a.window + a.discard)
                        row.update(
                            power_w=st["power_w"], power_min_w=st["power_min_w"],
                            power_max_w=st["power_max_w"],
                            clock_sm_med=st["clock_sm_med"], clock_sm_min=st["clock_sm_min"],
                            clock_sm_max=st["clock_sm_max"],
                            clock_held=st["clock_sm_min"] >= freq - CLOCK_TOL_MHZ,
                            pstate=st["pstate"], temp_c=st["temp_c"],
                            throttle=st["throttle"], n_samples=st["n_samples"],
                            window_s=round(a.window, 2), status="ok")
                        n_ok += 1
                    except Exception as e:
                        row["status"] = f"{type(e).__name__}: {str(e)[:90]}"
                        n_fail += 1
                        print(f"  DROP f={freq} n={nsq}: {row['status']}", flush=True)
                    finally:
                        if sq is not None:
                            try:
                                sq.__exit__()
                            except Exception:
                                pass
                    wr.writerow(row); fh.flush()
                    print(f"  rep{rep} f={freq:>4} n={nsq:>3} -> "
                          f"P={row['power_w'] or '--':>7} W  clk={row['clock_sm_med'] or '--':>5}"
                          f"  held={row['clock_held']}  P{row['pstate']}  "
                          f"{row['temp_c']}C   [{(time.time() - t0) / 60:.1f} min]", flush=True)
    finally:
        fh.close()
        reset_clocks()
    print(f"DONE in {(time.time() - t0) / 60:.1f} min  ok={n_ok} fail={n_fail} -> {out}",
          flush=True)


if __name__ == "__main__":
    main()
