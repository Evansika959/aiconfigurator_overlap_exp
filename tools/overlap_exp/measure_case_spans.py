#!/usr/bin/env python3
"""Measure ONE overlap configuration with per-kernel spans, so a figure can be drawn
entirely from measurement.

WHAT WAS MISSING. The 432 and comm-bound sweeps launch a genuine overlap -- two fresh
non-default streams, the collective at priority -3 and issued first, both gated on one
event, and the hardware block scheduler decides how the SMs get shared. But they record
only the iteration's total time and mean power. Drawing a power-vs-time figure needs the
two kernels' spans INSIDE that iteration, and those were borrowed from solo runs, which
is a different thing: a GEMM sharing the chip with a resident collective is slower than
the same GEMM alone.

This closes that gap using demo_overlap.py's method: per-stream CUDA events on a common
timeline give each kernel's start and end within the overlapped iteration.

TRAPS, all of them already paid for elsewhere in this directory:
  * NCCL caches NCCL_{MIN,MAX}_CTAS on first read -- one process, one CTA value.
  * The iteration count must be identical on every rank or a collective goes unanswered.
  * A clock lock does not hold without persistence mode; -lgc still reports success.
  * GCP's gIB shim rejects NCCL_{MIN,MAX}_CTAS; its directory must leave LD_LIBRARY_PATH.

  sudo -v && python3 measure_case_spans.py
"""
import argparse
import json
import os
import socket
import statistics as st
import subprocess
import threading
import time

os.environ.setdefault("LD_LIBRARY_PATH", "")
os.environ["LD_LIBRARY_PATH"] = ":".join(
    p for p in os.environ["LD_LIBRARY_PATH"].split(":") if p and "gib" not in p)
os.environ.pop("NCCL_NET", None)

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD, GPUS = 4, "0,1,2,3"
HERE = os.path.dirname(os.path.abspath(__file__))
REPS, WINDOW_S, DISCARD_S = 60, 2.0, 0.6


def sh(c):
    return subprocess.run(c, capture_output=True, text=True)


def worker(rank, port, cfg, q):
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(WORLD), MASTER_ADDR="127.0.0.1",
                      MASTER_PORT=port, NCCL_MIN_CTAS=str(cfg["ctas"]),
                      NCCL_MAX_CTAS=str(cfg["ctas"]))
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=WORLD)
    _, hi = torch.cuda.Stream.priority_range()
    s_gemm, s_comm = torch.cuda.Stream(priority=0), torch.cuda.Stream(priority=hi)
    cur = torch.cuda.current_stream()
    x = torch.randn((cfg["m"], cfg["k"]), dtype=torch.bfloat16, device="cuda")
    w = torch.randn((cfg["n"], cfg["k"]), dtype=torch.bfloat16, device="cuda")
    buf = torch.ones(cfg["mib"] * 1024 * 1024 // 2, dtype=torch.bfloat16, device="cuda")
    gemm = lambda: torch.nn.functional.linear(x, w)
    ar = lambda: dist.all_reduce(buf)

    for _ in range(8):
        gemm(); ar()
    torch.cuda.synchronize()

    # --- per-kernel spans on a common timeline -------------------------------------
    gs, cs, tot = [], [], []
    for _ in range(REPS):
        torch.cuda.synchronize()
        base = torch.cuda.Event(enable_timing=True)
        eg = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        ec = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        base.record(cur)
        with torch.cuda.stream(s_comm):          # collective first
            s_comm.wait_event(base); ec[0].record(s_comm); ar(); ec[1].record(s_comm)
        with torch.cuda.stream(s_gemm):
            s_gemm.wait_event(base); eg[0].record(s_gemm); gemm(); eg[1].record(s_gemm)
        torch.cuda.synchronize()
        gs.append((base.elapsed_time(eg[0]), base.elapsed_time(eg[1])))
        cs.append((base.elapsed_time(ec[0]), base.elapsed_time(ec[1])))
        tot.append(max(base.elapsed_time(eg[1]), base.elapsed_time(ec[1])))

    # --- mean power over a window, same launch pattern -------------------------------
    def once():
        ev = torch.cuda.Event(); ev.record(cur)
        for (s_, f_) in ((s_comm, ar), (s_gemm, gemm)):
            with torch.cuda.stream(s_):
                s_.wait_event(ev); f_()

    it = st.median(tot) / 1e3
    nt = torch.tensor([max(4, int((WINDOW_S + DISCARD_S) / it))], dtype=torch.int64,
                      device="cuda")
    dist.broadcast(nt, src=0)                    # identical on every rank
    iters = int(nt.item())
    samples = []
    stop = threading.Event()
    if rank == 0:
        import pynvml
        pynvml.nvmlInit()
        hs = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(WORLD)]

        def poll():
            while not stop.is_set():
                try:
                    cl = [pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
                                  for h in hs]
                    samples.append((time.time(),
                                    [pynvml.nvmlDeviceGetPowerUsage(h) / 1000 for h in hs],
                                    min(cl), st.median(cl)))
                except Exception:
                    pass
                stop.wait(0.02)
        th = threading.Thread(target=poll, daemon=True); th.start()
    t0 = time.perf_counter()
    for _ in range(iters):
        once(); torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) / iters * 1e3
    if rank == 0:
        stop.set(); th.join(timeout=2)
        k = [s for s in samples if s[0] - samples[0][0] >= DISCARD_S] or samples
        per = [st.median(s[1][i] for s in k) for i in range(WORLD)]
        q.put(dict(
            gemm_start=st.median(a for a, _ in gs), gemm_end=st.median(b for _, b in gs),
            comm_start=st.median(a for a, _ in cs), comm_end=st.median(b for _, b in cs),
            T_events=st.median(tot), T_loop=wall, iters=iters,
            power_per_gpu_w=round(sum(per) / WORLD, 2),
            clock_min=min(s[2] for s in k),
            clock_med=round(st.median(s[3] for s in k)),
            clock_max=max(s[3] for s in k), n_samples=len(k)))
    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=1024)
    ap.add_argument("--n", type=int, default=8192)
    ap.add_argument("--k", type=int, default=8192)
    ap.add_argument("--mib", type=int, default=512)
    ap.add_argument("--ctas", type=int, default=32)
    ap.add_argument("--clock", type=int, default=1200,
                    help="0 = do not lock; let the GPU's own governor choose")
    ap.add_argument("--out", default=os.path.join(HERE, "data", "case_spans.json"))
    a = ap.parse_args()
    cfg = dict(m=a.m, n=a.n, k=a.k, mib=a.mib, ctas=a.ctas, clock=a.clock)
    pm = sh(["nvidia-smi", "-i", GPUS, "--query-gpu=persistence_mode",
             "--format=csv,noheader"]).stdout.split("\n")[0].strip()
    if pm != "Enabled":
        sh(["sudo", "nvidia-smi", "-i", GPUS, "-pm", "1"])
    if a.clock:
        sh(["sudo", "nvidia-smi", "-i", GPUS, "-lgc", f"{a.clock},{a.clock}"])
    else:
        # --clock 0: no lock at all. This is what a deployment actually gets -- the
        # GPU's own DVFS governor boosts toward 1410 and backs off against the power
        # and thermal limits on its own. Every locked measurement in this directory
        # exists to attribute effects to frequency; this one exists to say what
        # frequency the hardware would have picked if nobody had asked.
        sh(["sudo", "nvidia-smi", "-i", GPUS, "-rgc"])
    time.sleep(1)
    try:
        mp.set_start_method("spawn", force=True)
        with socket.socket() as sk:
            sk.bind(("127.0.0.1", 0)); port = str(sk.getsockname()[1])
        q = mp.Queue()
        ps = [mp.Process(target=worker, args=(r, port, cfg, q)) for r in range(WORLD)]
        for p in ps:
            p.start()
        res = q.get(timeout=900)
        for p in ps:
            p.join(timeout=120)
            if p.is_alive():
                p.terminate()
    finally:
        sh(["sudo", "nvidia-smi", "-i", GPUS, "-rgc"])
        if pm != "Enabled":
            sh(["sudo", "nvidia-smi", "-i", GPUS, "-pm", "0"])
        print("[clocks] restored")
    res.update(cfg)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"\nGEMM {a.m}x{a.n}x{a.k} + {a.mib} MiB, {a.ctas} CTAs, {a.clock} MHz, world=4")
    print(f"  measured INSIDE the overlapped iteration:")
    print(f"    collective  {res['comm_start']:6.3f} -> {res['comm_end']:6.3f} ms "
          f"({res['comm_end'] - res['comm_start']:.3f} ms)")
    print(f"    GEMM        {res['gemm_start']:6.3f} -> {res['gemm_end']:6.3f} ms "
          f"({res['gemm_end'] - res['gemm_start']:.3f} ms)")
    print(f"    iteration   {res['T_events']:.3f} ms (events) / {res['T_loop']:.3f} ms (loop)")
    print(f"    power {res['power_per_gpu_w']:.1f} W/GPU   clock "
          f"min {res['clock_min']} / median {res['clock_med']} / max {res['clock_max']} MHz")
    print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
