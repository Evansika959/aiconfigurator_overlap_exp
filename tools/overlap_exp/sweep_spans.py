#!/usr/bin/env python3
"""Per-kernel spans and per-GPU power for a PLAN of GEMM / all-reduce cells.

WHY A THIRD HARNESS. sweep_overlap_432.py records one number per cell: the mean
iteration time. measure_case_spans.py records what happens INSIDE the iteration (when
the collective and the GEMM start and end) but for one cell per launch, on rank 0, at
one clock for every GPU. Checking an event simulator needs the spans on a grid, on
every rank, and -- for the multi-GPU model -- with a different clock or a different
GEMM on each GPU. This harness runs a list of cells from a JSON plan:

  {"world": 4, "ctas": [4, 32],
   "runs": [{"mode": "concurrent", "clock": 1200, "mib": 128, "shape": "4096x4096x4096"},
            {"mode": "concurrent", "clock": [1410, 900, 900, 900], "mib": 128,
             "shape": ["8192x8192x8192", "2048x4096x4096", "2048x4096x4096", "2048x4096x4096"]},
            {"mode": "gemm_only", "clock": 0, "shape": "4096x4096x4096", "once": true}]}

  mode     comm_only      the all-reduce alone
           gemm_only      the GEMM alone
           concurrent     both, the collective issued first (as in sweep_overlap_432.py)
           concurrent_gf  both, the GEMM issued first
           serial         the GEMM, then the all-reduce, on one stream
           program        a list of kernels on named streams with dependencies (below)
  clock    MHz for every GPU, or one per rank; 0 = not locked (the stock governor)
  shape    MxNxK for every rank, or one per rank (an unbalanced step)
  once     run only under the first CTA count (a solo GEMM does not depend on it)

PROGRAMS. A training step is not two kernels. `mode: program` runs, on every rank, a
list of ops -- one list for all ranks, or one per rank with the same collectives in the
same order:

  (an op may also be {"op": "sleep", "ms": 12}: the host waits and issues nothing)

  {"mode": "program", "name": "ddp_overlap", "clock": 1200,
   "ops": [{"op": "gemm", "tag": "f1", "stream": "main", "shape": "4096x8192x8192"},
           {"op": "gemm", "tag": "gw1", "stream": "main", "shape": "8192x8192x4096"},
           {"op": "allreduce", "tag": "ar1", "stream": "comm", "mib": 128, "wait": ["gw1"]},
           {"op": "gemm", "tag": "gx1", "stream": "main", "shape": "4096x8192x8192"}]}

Kernels on one stream run in order; `wait` holds a kernel back until the tagged kernels
have ended (a CUDA event), which is how a gradient all-reduce follows the GEMM that
produced the gradient while later GEMMs run beside it. The same JSON is what
../moe_megakernel/sim/node.py replays. Per-op spans go to <out>.ops.csv.

One CSV row per (cell, rank). Times are medians over REPS repetitions of CUDA events on
a common timeline, in ms from the launch gate; power and clock are that GPU's, sampled
by rank 0 every 20 ms over a WINDOW_S window of the same loop.

TRAPS, all inherited from sweep_overlap_432.py: NCCL caches NCCL_{MIN,MAX}_CTAS (one
process group per CTA count); every rank must issue the same number of collectives; a
clock lock does not hold without persistence mode; GCP's gIB shim rejects the CTA
variables and has to come off LD_LIBRARY_PATH.

  sudo -v && python3 sweep_spans.py --plan data/sim_overlap/plan_cal.json \
      --out data/sim_overlap/cal.csv [--resume]

CLOCKS are restored (`nvidia-smi -rgc`) on every exit path, persistence mode is put
back as it was found, and power caps are never touched.
"""
import argparse
import csv
import json
import os
import signal
import socket
import statistics as st
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

REPS, REPS_S = 40, 3.0          # span repetitions: at most REPS, at most ~REPS_S seconds
WINDOW_S, DISCARD_S = 2.0, 0.6
MODES = ("comm_only", "gemm_only", "concurrent", "concurrent_gf", "serial", "program")
OP_FIELDS = ["ctas", "run", "name", "clock", "rank", "i", "tag", "op", "stream", "what", "start", "end",
             "dur_p10", "dur_p90", "reps"]
FIELDS = ["ts", "world", "ctas", "run", "mode", "clock", "mib", "shape", "rank",
          "clock_req", "m", "n", "k",
          "gemm_start", "gemm_end", "comm_start", "comm_end", "t_events",
          "gemm_p10", "gemm_p90", "comm_p10", "comm_p90", "t_p10", "t_p90", "reps",
          "t_loop", "iters", "power_w", "power_mean_w", "clock_min", "clock_med",
          "temp_c", "n_samples"]


def sh(c):
    return subprocess.run(c, capture_output=True, text=True)


def per_rank(v, world):
    return list(v) if isinstance(v, (list, tuple)) else [v] * world


def spec(v):
    return "/".join(str(x) for x in v) if isinstance(v, (list, tuple)) else str(v)


def set_clocks(clocks):
    """clocks: one MHz per GPU of the world; 0 = that GPU is not locked."""
    for g, f in enumerate(clocks):
        if f:
            sh(["sudo", "-n", "nvidia-smi", "-i", str(g), "-lgc", f"{f},{f}"])
        else:
            sh(["sudo", "-n", "nvidia-smi", "-i", str(g), "-rgc"])


class Boards:
    """Power, SM clock and temperature of the world's GPUs every 20 ms."""

    def __init__(self, n):
        import pynvml
        self.nv = pynvml
        pynvml.nvmlInit()
        self.h = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(n)]
        self._stop, self._rows = threading.Event(), []

    def _loop(self):
        nv = self.nv
        while not self._stop.is_set():
            try:
                self._rows.append((
                    time.time(),
                    [nv.nvmlDeviceGetPowerUsage(x) / 1000 for x in self.h],
                    [nv.nvmlDeviceGetClockInfo(x, nv.NVML_CLOCK_SM) for x in self.h]))
            except Exception:
                pass
            self._stop.wait(0.02)

    def __enter__(self):
        self._rows.clear(); self._stop.clear()
        self._t = threading.Thread(target=self._loop, daemon=True); self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set(); self._t.join(timeout=2)

    def summary(self):
        rows = self._rows
        if not rows:
            return None
        kept = [r for r in rows if r[0] - rows[0][0] >= DISCARD_S] or rows
        out = []
        for i, h in enumerate(self.h):
            p = [r[1][i] for r in kept]
            c = [r[2][i] for r in kept]
            out.append(dict(power_w=round(st.median(p), 2), power_mean_w=round(st.fmean(p), 2),
                            clock_min=min(c), clock_med=int(st.median(c)),
                            temp_c=self.nv.nvmlDeviceGetTemperature(h, 0), n_samples=len(kept)))
        return out


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))]


def worker(rank, port, world, ctas, first, runs, done, out_path, q):
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world), MASTER_ADDR="127.0.0.1",
                      MASTER_PORT=port, NCCL_MIN_CTAS=str(ctas), NCCL_MAX_CTAS=str(ctas))
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    _, hi = torch.cuda.Stream.priority_range()
    s_gemm, s_comm = torch.cuda.Stream(priority=0), torch.cuda.Stream(priority=hi)
    cur = torch.cuda.current_stream()
    fh = wr = boards = None
    if rank == 0:
        new = not os.path.exists(out_path)
        fh = open(out_path, "a", newline="")
        wr = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            wr.writeheader(); fh.flush()
        boards = Boards(world)

    state = dict(clock=None, mib=None, shape=None)
    buf = x = w = None
    ar = lambda: dist.all_reduce(buf)
    gemm = lambda: torch.nn.functional.linear(x, w)

    def timed(mode):
        """One repetition with events; (gemm_start, gemm_end, comm_start, comm_end)."""
        ev = lambda: torch.cuda.Event(enable_timing=True)
        base, g0, g1, c0, c1 = ev(), ev(), ev(), ev(), ev()
        torch.cuda.synchronize()
        base.record(cur)

        def comm():
            with torch.cuda.stream(s_comm):
                s_comm.wait_event(base); c0.record(s_comm); ar(); c1.record(s_comm)

        def mult(s=s_gemm):
            with torch.cuda.stream(s):
                s.wait_event(base); g0.record(s); gemm(); g1.record(s)

        if mode == "comm_only":
            comm()
        elif mode == "gemm_only":
            mult()
        elif mode == "concurrent":
            comm(); mult()
        elif mode == "concurrent_gf":
            mult(); comm()
        else:                                    # serial: one stream, GEMM then all-reduce
            with torch.cuda.stream(s_gemm):
                s_gemm.wait_event(base); g0.record(s_gemm); gemm(); g1.record(s_gemm)
                c0.record(s_gemm); ar(); c1.record(s_gemm)
        torch.cuda.synchronize()
        el = base.elapsed_time
        g = (el(g0), el(g1)) if mode != "comm_only" else (0.0, 0.0)
        c = (el(c0), el(c1)) if mode != "gemm_only" else (0.0, 0.0)
        return g + c

    def plain(mode):
        """The same launch pattern without timing events, for the power window."""
        if mode == "comm_only":
            with torch.cuda.stream(s_comm):
                ar()
        elif mode == "gemm_only":
            with torch.cuda.stream(s_gemm):
                gemm()
        elif mode == "serial":
            with torch.cuda.stream(s_gemm):
                gemm(); ar()
        else:
            ev = torch.cuda.Event(); ev.record(cur)
            order = ((s_comm, ar), (s_gemm, gemm))
            for (s_, f_) in (order if mode == "concurrent" else order[::-1]):
                with torch.cuda.stream(s_):
                    s_.wait_event(ev); f_()
        torch.cuda.synchronize()

    streams = {"main": s_gemm, "comm": s_comm}
    bufs, mats = {}, {}
    ofh = owr = None
    if rank == 0 and any(r["mode"] == "program" for r in runs):
        opath = out_path + ".ops.csv"
        onew = not os.path.exists(opath)
        ofh = open(opath, "a", newline="")
        owr = csv.DictWriter(ofh, fieldnames=OP_FIELDS)
        if onew:
            owr.writeheader(); ofh.flush()

    def stream_of(name):
        if name not in streams:
            streams[name] = torch.cuda.Stream(priority=0)
        return streams[name]

    def prepare(ops):
        """Tensors for a program: one pair per GEMM shape, one buffer per message size."""
        for o in ops:
            if o["op"] == "gemm" and o["shape"] not in mats:
                m_, n_, k_ = (int(v) for v in o["shape"].split("x"))
                mats[o["shape"]] = (torch.randn((m_, k_), dtype=torch.bfloat16, device="cuda"),
                                    torch.randn((n_, k_), dtype=torch.bfloat16, device="cuda"))
            if o["op"] == "allreduce" and o["mib"] not in bufs:
                bufs[o["mib"]] = torch.ones(o["mib"] * 1024 * 1024 // 2, dtype=torch.bfloat16, device="cuda")

    def run_program(ops, timing):
        """Issue every op, then wait for the device. With timing: [(start, end)] per op,
        ms after the gate."""
        mk = (lambda: torch.cuda.Event(enable_timing=True)) if timing else torch.cuda.Event
        if timing:
            torch.cuda.synchronize()
        base = mk(); base.record(cur)
        ends, marks = {}, []
        for o in ops:
            if o["op"] == "sleep":               # the host waits; nothing is issued
                time.sleep(o["ms"] / 1e3)
                marks.append(None)
                continue
            st = stream_of(o.get("stream", "main"))
            with torch.cuda.stream(st):
                st.wait_event(base)
                for wtag in o.get("wait", ()):
                    st.wait_event(ends[wtag])
                e0 = mk() if timing else None
                if timing:
                    e0.record(st)
                if o["op"] == "gemm":
                    a_, b_ = mats[o["shape"]]
                    torch.nn.functional.linear(a_, b_)
                else:
                    dist.all_reduce(bufs[o["mib"]])
                e1 = mk(); e1.record(st)
            ends[o["tag"]] = e1
            marks.append((e0, e1))
        torch.cuda.synchronize()
        if not timing:
            return None
        return [(0.0, 0.0) if mk_ is None else (base.elapsed_time(mk_[0]), base.elapsed_time(mk_[1])) for mk_ in marks]

    for i, r in enumerate(runs):
        mode = r["mode"]
        if r.get("once") and not first:
            continue
        if mode == "program":
            if (ctas, i) in done:
                continue
            ops = r["ops"][rank] if isinstance(r["ops"][0], list) else r["ops"]
            clocks = per_rank(r.get("clock", 0), world)
            if clocks != state["clock"]:
                if rank == 0:
                    set_clocks(clocks)
                    time.sleep(0.6)
                state["clock"] = clocks
            dist.barrier()
            if sum(v[0].numel() + v[1].numel() for v in mats.values()) > 3e9:
                mats.clear(); torch.cuda.empty_cache()
            prepare(ops)
            for _ in range(4):
                run_program(ops, False)
            t0 = time.perf_counter()
            for _ in range(4):
                run_program(ops, False)
            it = (time.perf_counter() - t0) / 4
            nt = torch.tensor([max(8, min(REPS, int(REPS_S / max(it, 1e-6))))], dtype=torch.int64, device="cuda")
            dist.broadcast(nt, src=0)
            reps = int(nt.item())
            sp = [run_program(ops, True) for _ in range(reps)]
            nt = torch.tensor([max(4, int((WINDOW_S + DISCARD_S) / max(it, 1e-6)))], dtype=torch.int64, device="cuda")
            dist.broadcast(nt, src=0)
            iters = int(nt.item())
            dist.barrier()
            if rank == 0:
                boards.__enter__()
            t0 = time.perf_counter()
            for _ in range(iters):
                run_program(ops, False)
            t_loop = (time.perf_counter() - t0) / iters * 1e3
            if rank == 0:
                boards.__exit__()
            tt = [max(b_ for _, b_ in rep) for rep in sp]
            flat = [st.median(tt), pct(tt, 0.1), pct(tt, 0.9), t_loop]
            for j in range(len(ops)):
                d_ = [rep[j][1] - rep[j][0] for rep in sp]
                flat += [st.median(rep[j][0] for rep in sp), st.median(rep[j][1] for rep in sp),
                         pct(d_, 0.1), pct(d_, 0.9)]
            n_ops = max(len(x) for x in r["ops"]) if isinstance(r["ops"][0], list) else len(r["ops"])
            flat += [0.0] * (4 + 4 * n_ops - len(flat))   # ranks may have programs of different lengths
            mine = torch.tensor(flat, dtype=torch.float64, device="cuda")
            allr = [torch.zeros_like(mine) for _ in range(world)]
            dist.all_gather(allr, mine)
            if rank == 0:
                sm = boards.summary()
                for rr in range(world):
                    v = [round(float(z), 5) for z in allr[rr].tolist()]
                    wr.writerow(dict(ts=round(time.time(), 1), world=world, ctas=ctas, run=i, mode=mode,
                                     clock=spec(r.get("clock", 0)), mib=0, shape=r.get("name", "program"),
                                     rank=rr, clock_req=clocks[rr], m=0, n=0, k=0, gemm_start=0, gemm_end=0,
                                     comm_start=0, comm_end=0, t_events=v[0], gemm_p10=0, gemm_p90=0,
                                     comm_p10=0, comm_p90=0, t_p10=v[1], t_p90=v[2], reps=reps, t_loop=v[3],
                                     iters=iters, **sm[rr]))
                    rops = r["ops"][rr] if isinstance(r["ops"][0], list) else r["ops"]
                    for j, o in enumerate(rops):
                        owr.writerow(dict(ctas=ctas, run=i, name=r.get("name", "program"), clock=spec(r.get("clock", 0)),
                                          rank=rr, i=j, tag=o["tag"], op=o["op"], stream=o.get("stream", "main"),
                                          what=o.get("shape", o.get("mib", o.get("ms"))), start=v[4 + 4 * j], end=v[5 + 4 * j],
                                          dur_p10=v[6 + 4 * j], dur_p90=v[7 + 4 * j], reps=reps))
                fh.flush(); ofh.flush()
                print(f"  {ctas:>2}CTA {spec(r.get('clock', 0)):>19} {r.get('name', 'program'):>26} | "
                      f"T {v_fmt(allr, 0)} ms  loop {float(allr[0][3]):8.3f}  "
                      f"{'/'.join(str(int(z['power_w'])) for z in sm)} W  "
                      f"clk {'/'.join(str(z['clock_min']) for z in sm)}", flush=True)
            continue
        if (ctas, i) in done:
            continue
        clocks = per_rank(r.get("clock", 0), world)
        shapes = per_rank(r.get("shape", "0x0x0"), world)
        mib = r.get("mib", 0)
        if clocks != state["clock"]:
            if rank == 0:
                set_clocks(clocks)
                time.sleep(0.6)
            state["clock"] = clocks
        dist.barrier()
        if mode != "gemm_only" and mib != state["mib"]:
            buf = None
            torch.cuda.empty_cache()
            buf = torch.ones(mib * 1024 * 1024 // 2, dtype=torch.bfloat16, device="cuda")
            state["mib"] = mib
        m, n, k = (int(v) for v in shapes[rank].split("x"))
        if mode != "comm_only" and shapes[rank] != state["shape"]:
            x = w = None
            torch.cuda.empty_cache()
            x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
            w = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
            state["shape"] = shapes[rank]

        for _ in range(5):
            plain(mode)
        # Counts must be identical on every rank, or a collective goes unanswered and the
        # job hangs. The first iterations after an allocation are slow, so the span count
        # comes from a second timing and the window's count from the span loop itself:
        # a count taken from the warm-up made windows of 50 ms that read idle power.
        t0 = time.perf_counter()
        for _ in range(5):
            plain(mode)
        it = (time.perf_counter() - t0) / 5
        nt = torch.tensor([max(8, min(REPS, int(REPS_S / max(it, 1e-6))))],
                          dtype=torch.int64, device="cuda")
        dist.broadcast(nt, src=0)
        reps = int(nt.item())
        t0 = time.perf_counter()
        sp = [timed(mode) for _ in range(reps)]
        it = min(it, (time.perf_counter() - t0) / reps)
        nt = torch.tensor([max(4, int((WINDOW_S + DISCARD_S) / max(it, 1e-6)))],
                          dtype=torch.int64, device="cuda")
        if mode != "gemm_only":         # no collective inside: each rank fills the window
            dist.broadcast(nt, src=0)   # with its own GEMM, however long that takes
        iters = int(nt.item())
        dist.barrier()
        if rank == 0:
            boards.__enter__()
        t0 = time.perf_counter()
        for _ in range(iters):
            plain(mode)
        t_loop = (time.perf_counter() - t0) / iters * 1e3
        if rank == 0:
            boards.__exit__()
        dg = [b - a for a, b, _, _ in sp]
        dc = [d - c for _, _, c, d in sp]
        tt = [max(b, d) for _, b, _, d in sp]
        mine = torch.tensor([st.median(a for a, _, _, _ in sp), st.median(b for _, b, _, _ in sp),
                             st.median(c for _, _, c, _ in sp), st.median(d for _, _, _, d in sp),
                             st.median(tt), pct(dg, 0.1), pct(dg, 0.9), pct(dc, 0.1), pct(dc, 0.9),
                             pct(tt, 0.1), pct(tt, 0.9), t_loop],
                            dtype=torch.float64, device="cuda")
        allr = [torch.zeros_like(mine) for _ in range(world)]
        dist.all_gather(allr, mine)
        if rank == 0:
            s = boards.summary()
            for rr in range(world):
                v = [round(float(z), 5) for z in allr[rr].tolist()]
                mm, nn, kk = (int(z) for z in shapes[rr].split("x"))
                row = dict(ts=round(time.time(), 1), world=world, ctas=ctas, run=i, mode=mode,
                           clock=spec(r.get("clock", 0)), mib=mib, shape=spec(r.get("shape", "0x0x0")),
                           rank=rr, clock_req=clocks[rr], m=mm, n=nn, k=kk,
                           gemm_start=v[0], gemm_end=v[1], comm_start=v[2], comm_end=v[3],
                           t_events=v[4], gemm_p10=v[5], gemm_p90=v[6], comm_p10=v[7],
                           comm_p90=v[8], t_p10=v[9], t_p90=v[10], reps=reps, t_loop=v[11],
                           iters=iters, **s[rr])
                wr.writerow(row)
            fh.flush()
            print(f"  {ctas:>2}CTA {spec(r.get('clock', 0)):>19} {mib:>4}MiB "
                  f"{spec(r.get('shape', '-')):>16.16} {mode:>13} | T {v_fmt(allr, 4)} ms  "
                  f"gemm {allr[0][1] - allr[0][0]:7.3f}  comm {allr[0][3] - allr[0][2]:7.3f}  "
                  f"{'/'.join(str(int(z['power_w'])) for z in s)} W  "
                  f"clk {'/'.join(str(z['clock_min']) for z in s)}", flush=True)

    if rank == 0:
        fh.close()
        if ofh:
            ofh.close()
    dist.destroy_process_group()
    if rank == 0:
        q.put(True)


def v_fmt(allr, j):
    v = [float(a[j]) for a in allr]
    return f"{v[0]:8.3f}" if max(v) - min(v) < 0.02 * max(v) else "/".join(f"{z:.3f}" for z in v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    plan = json.load(open(a.plan))
    world, runs = int(plan.get("world", 4)), plan["runs"]
    for r in runs:
        assert r["mode"] in MODES, r
    assert torch.cuda.device_count() >= world
    gpus = ",".join(str(g) for g in range(world))

    done = set()
    if a.resume and os.path.exists(a.out):
        for r in csv.DictReader(open(a.out)):
            done.add((int(r["ctas"]), int(r["run"])))
        print(f"resume: {len(done)} cells present", flush=True)
    elif os.path.exists(a.out):
        sys.exit(f"{a.out} exists; pass --resume or choose another name")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

    pm = [x.strip() for x in sh(["nvidia-smi", "-i", gpus, "--query-gpu=persistence_mode",
                                 "--format=csv,noheader"]).stdout.split("\n") if x.strip()]
    for g, s in enumerate(pm):
        if s != "Enabled":
            sh(["sudo", "-n", "nvidia-smi", "-i", str(g), "-pm", "1"])

    def restore(*_):
        sh(["sudo", "-n", "nvidia-smi", "-i", gpus, "-rgc"])
        for g, s in enumerate(pm):
            if s != "Enabled":
                sh(["sudo", "-n", "nvidia-smi", "-i", str(g), "-pm", "0"])
        print("[clocks] restored", flush=True)

    def on_signal(sig, _frm):
        raise SystemExit(128 + sig)

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    print(f"world={world}  {len(runs)} runs x {len(plan['ctas'])} CTA counts  "
          f"(persistence was {'/'.join(pm)})", flush=True)
    t0 = time.time()
    ps = []
    try:
        mp.set_start_method("spawn", force=True)
        for i, c in enumerate(plan["ctas"]):
            print(f"=== {c} CTA  [{(time.time() - t0) / 60:.0f} min elapsed] ===", flush=True)
            for attempt in range(4):        # a free port can be taken between probe and bind
                with socket.socket() as sk:
                    sk.bind(("127.0.0.1", 0)); port = str(sk.getsockname()[1])
                q = mp.Queue()
                ps = [mp.Process(target=worker, args=(r, port, world, c, i == 0, runs, done, a.out, q))
                      for r in range(world)]
                for p in ps:
                    p.start()
                time.sleep(3)
                if all(p.is_alive() for p in ps):
                    break
                for p in ps:
                    if p.is_alive():
                        p.terminate()
                    p.join(timeout=20)
                print(f"  !! rank died at startup (attempt {attempt + 1}), new port", flush=True)
                time.sleep(5)
            while True:                     # poll: a dead rank is noticed in seconds
                try:
                    q.get(timeout=20); break
                except Exception:
                    if not all(p.is_alive() for p in ps):
                        print(f"!! a rank exited early at {c} CTA "
                              f"({[p.exitcode for p in ps]}) -- moving on", flush=True)
                        break
            for p in ps:
                p.join(timeout=180)
                if p.is_alive():
                    p.terminate()
    finally:
        for p in ps:
            if p.is_alive():
                p.terminate()
        restore()
    n = sum(1 for _ in csv.DictReader(open(a.out))) if os.path.exists(a.out) else 0
    print(f"\nDONE in {(time.time() - t0) / 60:.1f} min -> {a.out}  ({n} rows)", flush=True)


if __name__ == "__main__":
    main()
