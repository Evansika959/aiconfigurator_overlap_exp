#!/usr/bin/env python3
"""Run zeus's profile_p2p.py UNMODIFIED at each locked clock, to build P_static(f).

profile_p2p.py is the reference implementation from
ml-energy/zeus/examples/pipeline_frequency_optimizer/. It parks rank 0 in a blocking
dist.recv() while rank 1 sleeps 60 s, and reports the power drawn over that window --
the GPU is in its performance state with a resident kernel doing no useful work, so
that power is the static power.

This runner does not touch the script. It only:
  * locks the SM clock before each invocation (the script does not control clocks, and
    P_static is a curve in clock, not a number -- on A100 the board reports P0 while
    idling the SM clock down to 210 MHz),
  * drops /usr/local/gib from LD_LIBRARY_PATH, because the gIB NCCL shim on this node
    rejects several NCCL settings and unsetting the env vars alone is not enough,
  * points PYTHONPATH at the isolated zeus install,
  * records the achieved clock and temperature alongside, and resets the clock on
    every exit path.

  python3 run_p2p_static.py                 # ~10 min for 6 clocks
"""

import argparse
import atexit
import csv
import os
import re
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "profile_p2p.py")
ZEUS_PKGS = "/home/xinting/.local/zeus-pkgs"
GPUS = "0"
FREQS = [1410, 1200, 900, 705, 510, 300]
_locked = False


def sh(c, **kw):
    return subprocess.run(c, capture_output=True, text=True, **kw)


def reset_clocks():
    global _locked
    if _locked:
        sh(["sudo", "nvidia-smi", "-i", GPUS, "-rgc"])
        _locked = False
        print("[clocks] reset", flush=True)


def lock_clock(f, tol=20):
    global _locked
    r = sh(["sudo", "nvidia-smi", "-i", GPUS, "-lgc", f"{f},{f}"])
    if r.returncode != 0:
        print(f"[clocks] LOCK FAILED {f}: {r.stderr.strip()}", flush=True)
        return False
    _locked = True
    time.sleep(0.4)
    q = sh(["nvidia-smi", "-i", GPUS, "--query-gpu=clocks.sm",
            "--format=csv,noheader,nounits"])
    try:
        return abs(int(q.stdout.strip().splitlines()[0]) - f) <= tol
    except Exception:
        return False


def _sig(s, _f):
    reset_clocks()
    sys.exit(128 + s)


def gpu_state():
    q = sh(["nvidia-smi", "-i", GPUS,
            "--query-gpu=clocks.sm,temperature.gpu,power.draw,pstate",
            "--format=csv,noheader,nounits"])
    return [x.strip() for x in q.stdout.strip().split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "data", "p0_static_zeus.csv"))
    ap.add_argument("--clocks", type=int, nargs="+", default=FREQS)
    ap.add_argument("--reps", type=int, default=1)
    a = ap.parse_args()

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)
    atexit.register(reset_clocks)

    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = ":".join(
        p for p in env.get("LD_LIBRARY_PATH", "").split(":") if p and "gib" not in p)
    env.pop("NCCL_NET", None)
    env["PYTHONPATH"] = ZEUS_PKGS + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    rows = []
    try:
        for rep in range(1, a.reps + 1):
            for f in a.clocks:
                if not lock_clock(f):
                    print(f"  skipping {f} (lock not held)", flush=True)
                    continue
                t0 = time.time()
                r = subprocess.run([sys.executable, SCRIPT], capture_output=True,
                                   text=True, env=env, cwd=HERE, timeout=600)
                clk, temp, pw, ps = gpu_state()
                out = r.stdout
                m_t = re.search(r"Time \(s\):\s*([\d.eE+-]+)", out)
                m_e = re.search(r"Energy \(J\):\s*([\d.eE+-]+)", out)
                m_p = re.search(r"Power \(W\):\s*([\d.eE+-]+)", out)
                if not (m_t and m_e and m_p):
                    print(f"  {f} MHz FAILED (rc={r.returncode})\n"
                          f"    stdout: {out.strip()[-400:]}\n"
                          f"    stderr: {r.stderr.strip()[-600:]}", flush=True)
                    continue
                row = dict(rep=rep, freq_req_mhz=f,
                           time_s=round(float(m_t.group(1)), 3),
                           energy_j=round(float(m_e.group(1)), 3),
                           power_w=round(float(m_p.group(1)), 3),
                           clock_sm_after=clk, temp_c_after=temp,
                           pstate_after=ps, wall_s=round(time.time() - t0, 1))
                rows.append(row)
                print(f"  rep{rep} {f:>4} MHz -> P={row['power_w']:7.2f} W  "
                      f"E={row['energy_j']:8.1f} J over {row['time_s']:.1f} s   "
                      f"(after: {clk} MHz, {temp} C, P{ps})", flush=True)
    finally:
        reset_clocks()

    if rows:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {a.out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
