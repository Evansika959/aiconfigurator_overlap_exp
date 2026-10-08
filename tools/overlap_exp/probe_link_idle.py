#!/usr/bin/env python3
"""What does a GPU draw between collectives, and for how long after the last one?

A GPU that waits for a slower one inside a training step draws 16-20 W more than the
idle table of the power model says, at every clock and at 35 C (`waiting`,
`waiting_swap`). The table was measured on one GPU that never used NVLink. This traces
the board power of all four GPUs, every 20 ms, through three phases at a locked clock:

  before   a CUDA context and an NCCL process group exist, GEMMs have run, but no
           collective has been issued yet
  burst    2 s of back-to-back all-reduces
  after    nothing is issued for 8 s

If the extra power is the NVLink's (links leave their low-power state when traffic
starts and return to it some time after it stops), `before` sits on the idle table,
and `after` steps down to it once, late. The step's height and delay are what the node
engine needs (`NcclAllReduce.link_w`, `link_ms`).

  python3 probe_link_idle.py --out data/sim_overlap/link_idle.csv

Clocks are locked for the run and restored on every exit path; power caps are untouched.
"""
import argparse
import csv
import os
import socket
import subprocess
import sys
import threading
import time

os.environ.setdefault("LD_LIBRARY_PATH", "")
os.environ["LD_LIBRARY_PATH"] = ":".join(
    p for p in os.environ["LD_LIBRARY_PATH"].split(":") if p and "gib" not in p)
os.environ.pop("NCCL_NET", None)

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD, GPUS, CTAS, MIB = 4, "0,1,2,3", 16, 128
CLOCKS = [1410, 1200, 900, 510]
BEFORE_S, BURST_S, AFTER_S = 3.0, 2.0, 8.0


def sh(c):
    return subprocess.run(c, capture_output=True, text=True)


def worker(rank, port, out_path, q):
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(WORLD), MASTER_ADDR="127.0.0.1", MASTER_PORT=port,
                      NCCL_MIN_CTAS=str(CTAS), NCCL_MAX_CTAS=str(CTAS))
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=WORLD)
    buf = torch.ones(MIB * 1024 * 1024 // 2, dtype=torch.bfloat16, device="cuda")
    x = torch.randn((2048, 4096), dtype=torch.bfloat16, device="cuda")
    w = torch.randn((4096, 4096), dtype=torch.bfloat16, device="cuda")
    rows, stop, phase = [], threading.Event(), ["init"]
    if rank == 0:
        import pynvml
        pynvml.nvmlInit()
        hs = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(WORLD)]

        def poll():
            while not stop.is_set():
                rows.append((time.time(), phase[0], [pynvml.nvmlDeviceGetPowerUsage(h) / 1000 for h in hs],
                             [pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM) for h in hs]))
                stop.wait(0.02)

        th = threading.Thread(target=poll, daemon=True)
        th.start()
    for clock in CLOCKS:
        if rank == 0:
            sh(["sudo", "-n", "nvidia-smi", "-i", GPUS, "-lgc", f"{clock},{clock}"])
        dist.barrier()                      # (a collective: the links are in use from here on,
        time.sleep(AFTER_S)                 #  so wait them out before `before`)
        phase[0] = f"{clock}:before"
        t0 = time.time()
        while time.time() - t0 < 0.3:       # a little compute: the context is awake, the links are not
            torch.nn.functional.linear(x, w)
        torch.cuda.synchronize()
        time.sleep(BEFORE_S)
        phase[0] = f"{clock}:burst"
        n = torch.tensor([0], dtype=torch.int64, device="cuda")
        if rank == 0:
            t0 = time.perf_counter()
            for _ in range(20):
                dist.all_reduce(buf)
            torch.cuda.synchronize()
            n[0] = max(20, int(BURST_S / ((time.perf_counter() - t0) / 20)))
        else:
            for _ in range(20):
                dist.all_reduce(buf)
        dist.broadcast(n, src=0)
        for _ in range(int(n.item())):
            dist.all_reduce(buf)
        torch.cuda.synchronize()
        phase[0] = f"{clock}:after"
        time.sleep(AFTER_S)
    phase[0] = "end"
    if rank == 0:
        stop.set()
        th.join(timeout=2)
        with open(out_path, "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["t", "phase"] + [f"w{i}" for i in range(WORLD)] + [f"clk{i}" for i in range(WORLD)])
            for t, ph, p, c in rows:
                wr.writerow([round(t - rows[0][0], 3), ph] + p + c)
    dist.destroy_process_group()
    if rank == 0:
        q.put(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        sys.exit(f"{a.out} exists; choose another name")
    pm = [x.strip() for x in sh(["nvidia-smi", "-i", GPUS, "--query-gpu=persistence_mode",
                                 "--format=csv,noheader"]).stdout.split("\n") if x.strip()]
    for g, s in enumerate(pm):
        if s != "Enabled":
            sh(["sudo", "-n", "nvidia-smi", "-i", str(g), "-pm", "1"])
    ps = []
    try:
        mp.set_start_method("spawn", force=True)
        with socket.socket() as sk:
            sk.bind(("127.0.0.1", 0)); port = str(sk.getsockname()[1])
        q = mp.Queue()
        ps = [mp.Process(target=worker, args=(r, port, a.out, q)) for r in range(WORLD)]
        for p in ps:
            p.start()
        q.get(timeout=600)
        for p in ps:
            p.join(timeout=60)
    finally:
        for p in ps:
            if p.is_alive():
                p.terminate()
        sh(["sudo", "-n", "nvidia-smi", "-i", GPUS, "-rgc"])
        for g, s in enumerate(pm):
            if s != "Enabled":
                sh(["sudo", "-n", "nvidia-smi", "-i", str(g), "-pm", "0"])
        print("[clocks] restored")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
