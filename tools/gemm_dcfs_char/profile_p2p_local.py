#!/usr/bin/env python3
"""Cross-check P_static against zeus's P2P-blocking method.

zeus's examples/pipeline_frequency_optimizer/profile_p2p.py parks rank 0 in a blocking
dist.recv() while rank 1 sleeps, and calls the power drawn over that window the P0
static power. This reproduces that measurement -- same idea, same NCCL recv, no zeus
dependency -- so it can be compared directly against data/p0_static.csv, which was
taken with an __nanosleep squatter instead.

THE QUESTION. Both methods keep a kernel resident so the GPU does not fall out of its
high-performance state. But NCCL's recv kernel BUSY-WAITS: it polls a flag in a loop,
which is real switching activity and therefore real dynamic power. __nanosleep parks
the SM with essentially none -- measured, going from 1 occupied SM to 108 changes
board power by 0.0 W at <=900 MHz. If the two methods agree, the squatter table is
confirmed; if the NCCL number is higher, the spin is being counted as "static" and
subtracting it would under-count the GEMM's own dynamic energy.

Both are measured at the SAME LOCKED CLOCK, which zeus does not control -- P_static is
a curve in clock, not a number, and on A100 the board reports P0 while idling the SM
clock to 210 MHz.

  python3 profile_p2p_local.py                  # ~4 min
"""

import argparse
import atexit
import csv
import multiprocessing as mp
import os
import signal
import subprocess
import sys
import time

GPUS = "0"
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


def worker(rank, freq, window_s, q):
    # the gIB shim on this node rejects several NCCL env vars; drop it from the loader
    # path before torch.distributed initialises NCCL
    os.environ["LD_LIBRARY_PATH"] = ":".join(
        p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if "gib" not in p)
    os.environ.pop("NCCL_NET", None)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29577"

    import torch
    import torch.distributed as dist
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from measure import Sampler

    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", init_method="env://", world_size=2, rank=rank)
    t = torch.rand(4096, 4096, device=f"cuda:{rank}")

    if rank == 0:
        for _ in range(5):                       # communication warmup, as in zeus
            dist.recv(t, src=1)
            dist.send(t, dst=1)
        torch.cuda.synchronize()
        s = Sampler(0)
        s.start()
        t0 = time.perf_counter()
        dist.recv(t, src=1)
        # dist.recv only ENQUEUES the NCCL kernel; the CPU returns immediately and a
        # window closed here measures 0.0 s with one sample. The wait happens on the
        # CUDA stream, so the sync is what actually covers the P2P block. zeus gets
        # this via ZeusMonitor.end_window(sync_execution=True).
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        st = s.stop(discard_s=1.0)
        q.put(dict(freq=freq, wall=round(wall, 2), **st))
    else:
        for _ in range(5):
            dist.send(t, dst=0)
            dist.recv(t, src=0)
        torch.cuda.synchronize()
        time.sleep(window_s)
        dist.send(t, dst=0)
        torch.cuda.synchronize()
    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--out", default=os.path.join(here, "data", "p0_static_p2p.csv"))
    ap.add_argument("--clocks", type=int, nargs="+", default=[1410, 1200, 900, 300])
    ap.add_argument("--window", type=float, default=20.0)
    a = ap.parse_args()

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)
    atexit.register(reset_clocks)
    mp.set_start_method("spawn", force=True)

    rows = []
    try:
        for f in a.clocks:
            lock_clock(f)
            q = mp.Queue()
            ps = [mp.Process(target=worker, args=(r, f, a.window, q)) for r in (0, 1)]
            for p in ps:
                p.start()
            res = q.get(timeout=a.window + 180)
            for p in ps:
                p.join(timeout=60)
                if p.is_alive():
                    p.terminate()
            rows.append(res)
            print(f"  {f:>4} MHz: P={res['power_w']:6.2f} W  clk={res['clock_sm_med']}  "
                  f"P{res['pstate']}  {res['temp_c']}C  window={res['wall']}s  "
                  f"n={res['n_samples']}", flush=True)
    finally:
        reset_clocks()

    if rows:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
