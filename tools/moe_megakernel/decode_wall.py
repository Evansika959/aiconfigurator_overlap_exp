#!/usr/bin/env python3
"""Three measurements that decide whether a second V/f domain has anything to work with
in DECODE. Each is a subcommand; each resets the clocks on the way out.

  governor  Does a decode batch ever hit the 400 W wall, and where? The whole analysis
            so far locked the clock. Deployed, the governor picks it, and nobody has
            measured what it picks in decode. Unlocked, locked 1290, and requested
            1410, for batches of 512 to 12288 tokens.

  corun     Does the cold/hot difference in clock elasticity survive when the two
            partitions run TOGETHER? Measured alone on 54 SMs the cold side has
            elasticity 0.55-0.68 above the knee and the hot side 0.80-0.96. But the
            whole layer, where they share HBM, has 0.31 at batch 1024. If contention
            flattens the hot side too, there is no heterogeneity for a second domain
            to exploit and the solo numbers were an artefact of measuring alone.

  limiter   How long may the board exceed 400 W, and by how much, before the power
            controller pulls the clock down? A step from idle to a full-width prefill
            layer, sampled as fast as NVML answers.

HOW corun TIMES EACH PARTITION SEPARATELY. A CUDA graph cannot time one stream, so the
partitions are launched eagerly and each stream is bracketed with its own CUDA events.
A single decode layer is 1-2 ms and a Python kernel launch is 0.1-0.3 ms, which would
stagger the streams and idle the SMs between launches. So each partition's tile list is
REPEATED R times into one long list: one launch runs ~60 ms and the launch cost
vanishes. Repeats do not help each other through the cache -- one pass streams 576 MiB
of weights through a 40 MB L2. The two streams must stay busy over the same window, so
at every clock a pilot run rebalances R until they finish within 5% of each other; the
overlap actually achieved is recorded per row.

  scan      The calibration sweep for the simulator (sim/). corun fixed the SM split at
            54+54 and 32+76; a V/f domain is a whole number of GPCs, so the rate model
            needs each partition on 16, 32, 48, ... SMs, alone and beside its partner.
            Six GPC-aligned splits, decode and prefill, 930-1410 MHz. Per case it also
            measures the whole layer on a persistent kernel (launches only, and end to
            end with the activation) and on vLLM's queue kernel, so every baseline the
            simulator reports was measured in the same campaign at the same clocks.
            Rows are flushed after every clock; one failing case is logged and skipped.

  idle      Board power with nothing running, at every locked clock, by ONE method and
            with the die temperature recorded -- cold, and again right after a burst of
            load. Two earlier campaigns disagreed by 3-4 W and neither recorded
            temperature.

  python3 decode_wall.py governor --repeats 3
  python3 decode_wall.py corun --repeats 3
  python3 decode_wall.py limiter
  python3 decode_wall.py scan --repeats 3          # ~80 min
  python3 decode_wall.py idle --repeats 3          # ~15 min
  python3 decode_wall.py pinned --repeats 2        # ~8 min: the pinned layer end to end
  python3 decode_wall.py occupancy --repeats 2     # ~6 min: 1-3 resident blocks per SM
"""
import argparse, csv, json, os, random, subprocess, sys, threading, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vendor.fused_moe_triton import PreparedMoE
from persistent_moe import PersistentMoE, _launch
from b1_bench import SHAPES, load_counts, make_inputs
from batch_clock_sweep import decode_batches
from baseline_b0 import Power

HERE = os.path.dirname(os.path.abspath(__file__))
TOL = 7
smi = lambda *a: subprocess.run(["sudo", "nvidia-smi", "-i", "0", *a], capture_output=True)


def lock(f):
    """f < 0 hands the clock back to the governor."""
    smi("-rgc") if f < 0 else smi("-lgc", f"{f},{f}")
    time.sleep(1.0)


def decode_inputs(spec, layer, cells, start=0):
    counts = decode_batches(spec, layer, cells, start)
    counts[np.argmax(counts)] -= int(counts.sum()) % spec["topk"]
    return counts, make_inputs(spec, counts)


def replay_for(g, secs, cap=4000):
    """Replay a CUDA graph for about `secs`, a FIXED number of times.

    graph.replay() is asynchronous: it returns at once. `while elapsed < secs: replay()`
    therefore queues work as fast as the CPU can issue it -- tens of thousands of
    replays for a 0.5 ms decode layer -- and the first version of this script
    segfaulted on exactly that. Time one replay, then run a computed count."""
    g.replay(); torch.cuda.synchronize()
    t0 = time.time(); g.replay(); torch.cuda.synchronize()
    n = max(2, min(cap, int(secs / max(time.time() - t0, 1e-5))))
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(n):
        g.replay()
    torch.cuda.synchronize()
    return time.time() - t0, n


def gpu_temp(_h=[None]):
    import pynvml
    if _h[0] is None:
        pynvml.nvmlInit(); _h[0] = pynvml.nvmlDeviceGetHandleByIndex(0)
    return pynvml.nvmlDeviceGetTemperature(_h[0], pynvml.NVML_TEMPERATURE_GPU)


def graph_of(fn, reps=4):
    """Warm `fn` on a side stream, then capture `reps` calls of it as one CUDA graph."""
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    return g


class RowSink:
    """Append rows to a CSV as they are produced. An overnight sweep must not lose an
    hour of measurements to a crash in its last minute."""

    def __init__(self, path, fields):
        self.path, self.fields, self.n = path, fields, 0
        self.fh = open(path, "w", newline="")
        self.w = csv.DictWriter(self.fh, fieldnames=fields)
        self.w.writeheader(); self.fh.flush()

    def add(self, **row):
        self.w.writerow({k: row.get(k, "") for k in self.fields}); self.n += 1

    def flush(self):
        self.fh.flush(); os.fsync(self.fh.fileno())

    def close(self):
        self.fh.close(); print(f"wrote {self.path} ({self.n} rows)")


def write(rows, out):
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")


# ---------------------------------------------------------------- governor ----
def governor(a):
    spec = SHAPES[a.model]
    cfg = json.load(open(os.path.join(HERE, "data", f"b1_config_{a.model}.json")))
    jobs = []
    for cells in [int(v) for v in a.cells.split(",")]:
        counts, (x, w1, w2, tw, ti, M) = decode_inputs(spec, a.layer, cells)
        p = PreparedMoE(x, w1, w2, tw, ti, cfg)
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                p.run()
        torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(4):
                p.run()
        live = int((counts > 0).sum())
        # keep `p`: the graph replays against its tensors, and a job that held only the
        # graph let them be freed when the loop moved on -> illegal memory access
        jobs.append(dict(tokens=M, per_expert=round(counts.sum() / max(live, 1), 1), g=g,
                         p=p))
        print(f"  batch {M:6d}: {jobs[-1]['per_expert']:7.1f} pairs per expert",
              flush=True)

    smi("-pm", "1")
    rows, rng = [], random.Random(5)
    modes = [int(v) for v in a.modes.split(",")]
    try:
        for pas in range(a.repeats):
            order = modes[:]; rng.shuffle(order)
            for f in order:
                lock(f)
                with Power(discard=0.5) as pw:
                    time.sleep(1.5)
                idle = pw.summary()
                for j in jobs:
                    replay_for(j["g"], a.settle)          # let the governor settle
                    with Power(discard=0.4) as pw:
                        wall, n = replay_for(j["g"], a.secs)
                    sm = pw.summary()
                    cl = [r[2] for r in pw.rows]
                    rows.append(dict(
                        pas=pas, mode="governor" if f < 0 else f"lock{f}",
                        tokens=j["tokens"], per_expert=j["per_expert"],
                        clock_req=f, clock_med=sm["clock_med"], clock_min=min(cl),
                        clock_max=max(cl),
                        held=f < 0 or abs(sm["clock_med"] - f) <= TOL,
                        ms_per_layer=round(wall / (n * 4) * 1e3, 5),
                        power_w=sm["power_w"], idle_w=idle["power_w"], replays=n))
                    r = rows[-1]
                    print(f"  p{pas} {r['mode']:9s} batch {r['tokens']:6d}: "
                          f"{r['clock_med']} MHz  {r['power_w']:6.1f} W  "
                          f"{r['ms_per_layer']:.3f} ms", flush=True)
    finally:
        smi("-rgc"); print("  [clocks reset]")
    write(rows, a.out or os.path.join(HERE, "data", f"decode_governor_{a.model}.csv"))


# ------------------------------------------------------------------- corun ----
def fire(p, cfg, parts):
    """parts: [(stream, g1 tiles, g2 tiles, blocks)]. One long launch pair per stream,
    bracketed by that stream's own events. GEMM 1 goes out on every stream before
    GEMM 2 on any, so the streams enter each phase together."""
    ev = []
    for st, *_ in parts:
        s = torch.cuda.Event(enable_timing=True); s.record(st); ev.append([s, None])
    for st, t1, _, nb in parts:
        _launch(p.x, p.w1, p.c1, p.tw, p.sorted_ids, p.expert_ids, t1, nb, False,
                p.topk, cfg, p.ct, stream=st)
    for st, _, t2, nb in parts:
        _launch(p.c2, p.w2, p.c3, p.tw, p.sorted_ids, p.expert_ids, t2, nb, True, 1,
                cfg, p.ct, stream=st)
    for i, (st, *_) in enumerate(parts):
        e = torch.cuda.Event(enable_timing=True); e.record(st); ev[i][1] = e
    for st, *_ in parts:
        st.synchronize()
    return [s.elapsed_time(e) for s, e in ev]            # ms per stream


def corun(a):
    spec = SHAPES[a.model]
    base = json.load(open(os.path.join(HERE, "data", f"b1_config_{a.model}.json")))
    splits = [tuple(int(v) for v in s.split("+")) for s in a.splits.split(",")]
    sA, sB = torch.cuda.Stream(), torch.cuda.Stream()
    cases = []
    for cells in [int(v) for v in a.cells.split(",")]:
        counts, (x, w1, w2, tw, ti, M) = decode_inputs(spec, a.layer, cells)
        light_e = np.argsort(counts, kind="stable")[:a.light_experts]
        for bm in [int(v) for v in a.block_m.split(",")]:
            cfg = dict(base); cfg["BLOCK_SIZE_M"] = bm
            p = PersistentMoE(x, w1, w2, tw, ti, cfg)
            p.run(); torch.cuda.synchronize()            # fills c2, GEMM 2's input
            mask = torch.from_numpy(
                np.isin(p.rowblock_expert.cpu().numpy(), light_e)).to(p.live_m.device)
            T = {}
            for name, rows in (("light", p.live_m[mask]), ("heavy", p.live_m[~mask])):
                T[name] = (p.tiles("g1", rows, key=("c", name)),
                           p.tiles("g2", rows, key=("c", name)))
            cases.append(dict(tokens=M, bm=bm, p=p, cfg=cfg, T=T,
                              real=dict(light=int(counts[light_e].sum()),
                                        heavy=int(counts.sum() - counts[light_e].sum()))))
            print(f"  batch {M} BM{bm}: light {cases[-1]['real']['light']} pairs, "
                  f"heavy {cases[-1]['real']['heavy']} pairs", flush=True)

    def rep(c, name, R):
        t1, t2 = c["T"][name]
        return t1.repeat(R), t2.repeat(R)

    def measure(c, parts, secs):
        """Run `parts` back to back for `secs` under the power sampler."""
        fire(c["p"], c["cfg"], parts)                              # warm
        out = []
        with Power(discard=0.3) as pw:
            t0 = time.time()
            while time.time() - t0 < secs:
                out.append(fire(c["p"], c["cfg"], parts))
        return np.array(out), pw.summary()

    smi("-pm", "1")
    rows, rng = [], random.Random(17)
    clocks = [int(v) for v in a.clocks.split(",")]
    try:
        for pas in range(a.repeats):
            grid = clocks[:]; rng.shuffle(grid)
            for f in grid:
                lock(f)
                with Power(discard=0.5) as pw:
                    time.sleep(1.2)
                idle = pw.summary()
                for c in cases:
                    for nl, nh in splits:
                        # per-pass time, solo, to size R for ~TARGET ms per launch
                        R = {}
                        for name, nb, st in (("light", nl, sA), ("heavy", nh, sB)):
                            t = fire(c["p"], c["cfg"], [(st, *rep(c, name, 4), nb)])[0] / 4
                            R[name] = max(2, int(round(a.target_ms / t)))
                        solo = {}
                        for name, nb, st in (("light", nl, sA), ("heavy", nh, sB)):
                            ts, sm = measure(c, [(st, *rep(c, name, R[name]), nb)], a.secs)
                            solo[name] = (np.median(ts[:, 0]) / R[name], sm)
                        # rebalance so both streams finish together under contention
                        for _ in range(3):
                            tl_, th_ = fire(c["p"], c["cfg"],
                                            [(sA, *rep(c, "light", R["light"]), nl),
                                             (sB, *rep(c, "heavy", R["heavy"]), nh)])
                            if abs(tl_ / th_ - 1) < 0.05:
                                break
                            R["heavy"] = max(2, int(round(R["heavy"] * tl_ / th_)))
                        ts, smc = measure(c, [(sA, *rep(c, "light", R["light"]), nl),
                                              (sB, *rep(c, "heavy", R["heavy"]), nh)],
                                          a.secs)
                        med = np.median(ts, axis=0)
                        co = dict(light=med[0] / R["light"], heavy=med[1] / R["heavy"])
                        overlap = float(min(med) / max(med))
                        for name, nb in (("light", nl), ("heavy", nh)):
                            for mode, ms, sm in (("solo", solo[name][0], solo[name][1]),
                                                 ("corun", co[name], smc)):
                                rows.append(dict(
                                    pas=pas, tokens=c["tokens"], block_m=c["bm"],
                                    sm_light=nl, sm_heavy=nh, part=name, mode=mode,
                                    sms=nb, real_pairs=c["real"][name], clock=f,
                                    clock_med=sm["clock_med"],
                                    held=abs(sm["clock_med"] - f) <= TOL,
                                    ms_per_pass=round(float(ms), 5),
                                    passes=R[name],
                                    overlap=round(overlap, 3) if mode == "corun" else "",
                                    power_w=sm["power_w"], idle_w=idle["power_w"]))
                        print(f"  p{pas} {f} MHz batch {c['tokens']} BM{c['bm']} "
                              f"{nl}+{nh}: light x{co['light'] / solo['light'][0]:.2f} "
                              f"heavy x{co['heavy'] / solo['heavy'][0]:.2f} "
                              f"overlap {overlap:.2f}  {smc['power_w']:.0f} W", flush=True)
    finally:
        smi("-rgc"); print("  [clocks reset]")
    write(rows, a.out or os.path.join(HERE, "data", f"decode_corun_{a.model}.csv"))


# ----------------------------------------------------------------- limiter ----
def limiter(a):
    import pynvml
    spec = SHAPES[a.model]
    cfg = json.load(open(os.path.join(HERE, "data", f"b1_config_{a.model}.json")))
    counts = load_counts(spec, 0, a.layer)                # the prefill layer: hits 400 W
    x, w1, w2, tw, ti, M = make_inputs(spec, counts)
    p = PreparedMoE(x, w1, w2, tw, ti, cfg)
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            p.run()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(4):
            p.run()

    pynvml.nvmlInit(); h = pynvml.nvmlDeviceGetHandleByIndex(0)
    smi("-pm", "1"); smi("-rgc")
    g.replay(); torch.cuda.synchronize()
    t0 = time.time(); g.replay(); torch.cuda.synchronize()
    n_load = max(2, int(a.load / (time.time() - t0)))      # sized once, see replay_for
    samples, stop, phase = [], threading.Event(), ["idle"]

    def poll():
        # NVML answers these two queries in under a microsecond -- from a value the
        # driver refreshes on its own schedule -- so polling flat out returned 1.4
        # million identical samples a second. Poll at 2 kHz and keep a sample only when
        # the reading CHANGES (plus a 50 ms heartbeat): the change times are the
        # sensor's real update cadence, which bounds what this experiment can resolve.
        last, last_t = None, 0.0
        while not stop.is_set():
            t = time.time()
            cur = (pynvml.nvmlDeviceGetPowerUsage(h) / 1000,
                   pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM))
            if cur != last or t - last_t > 0.05 or phase[0] != (samples[-1][1] if samples else None):
                samples.append((t, phase[0], *cur)); last, last_t = cur, t
            time.sleep(0.0005)

    th = threading.Thread(target=poll, daemon=True); th.start()
    rows = []
    try:
        for trial in range(a.trials):
            phase[0] = "idle"; time.sleep(a.idle)
            torch.cuda.synchronize()
            phase[0] = "load"; t0 = time.time()
            for _ in range(n_load):
                g.replay()
            torch.cuda.synchronize()
            phase[0] = "idle"; time.sleep(0.5)
            for t, ph, pw_, ck in samples:
                rows.append(dict(trial=trial, t_ms=round((t - t0) * 1e3, 3), phase=ph,
                                 power_w=pw_, clock_mhz=ck))
            n = sum(1 for r in samples if r[1] == "load")
            print(f"  trial {trial}: {n} readings kept under load "
                  f"({n / a.load:.0f}/s)", flush=True)
            samples.clear()
    finally:
        stop.set(); th.join(timeout=2); smi("-rgc"); print("  [clocks reset]")
    write(rows, a.out or os.path.join(HERE, "data", f"limiter_step_{a.model}.csv"))


# -------------------------------------------------------------------- scan ----
SCAN_FIELDS = ["pas", "cells", "tokens", "block_m", "light_experts", "kernel", "part",
               "mode", "sms", "sm_light", "sm_heavy", "experts", "real_pairs",
               "padded_rows", "clock", "clock_med", "held", "ms_per_pass", "passes",
               "overlap", "power_w", "idle_w", "temp_c"]


def scan(a):
    spec = SHAPES[a.model]
    base = json.load(open(os.path.join(HERE, "data",
                                       f"b1_config_{a.config_from or a.model}.json")))
    splits = [tuple(int(v) for v in s.split("+")) for s in a.splits.split(",")]
    assert all(l + h == a.all_sms for l, h in splits), "every split must use all SMs"
    specs = [tuple(int(v) for v in c.split(":")) for c in a.cases.split(",")]
    sA, sB = torch.cuda.Stream(), torch.cuda.Stream()
    inputs, queues, cases = {}, {}, []
    for cells, bm, nlight in specs:
        if cells not in inputs:
            if cells == 0:                                # the prefill batch
                counts = load_counts(spec, 0, a.layer)
                inputs[cells] = (counts, make_inputs(spec, counts))
            else:
                inputs[cells] = decode_inputs(spec, a.layer, cells)
            counts, (x, w1, w2, tw, ti, M) = inputs[cells]
            q = PreparedMoE(x, w1, w2, tw, ti, base)
            queues[cells] = dict(p=q, g=graph_of(q.run), tokens=M,
                                 pairs=int(counts.sum()),
                                 experts=int((counts > 0).sum()))
        counts, (x, w1, w2, tw, ti, M) = inputs[cells]
        cfg = dict(base); cfg["BLOCK_SIZE_M"] = bm
        p = PersistentMoE(x, w1, w2, tw, ti, cfg)
        p.run(); torch.cuda.synchronize()                 # fills c2, GEMM 2's input
        light_e = np.argsort(counts, kind="stable")[:nlight]
        re_ = p.rowblock_expert.cpu().numpy()
        mnp = np.isin(re_, light_e)
        mask = torch.from_numpy(mnp).to(p.live_m.device)
        T, meta = {}, {}
        for name, rows, ex in (("light", p.live_m[mask], np.unique(re_[mnp])),
                               ("heavy", p.live_m[~mask], np.unique(re_[~mnp])),
                               ("all", p.live_m, np.unique(re_))):
            T[name] = (p.tiles("g1", rows, key=("s", name)),
                       p.tiles("g2", rows, key=("s", name)))
            meta[name] = dict(experts=int(len(ex)), real_pairs=int(counts[ex].sum()),
                              padded_rows=int(rows.numel() * bm))
        cases.append(dict(cells=cells, tokens=M, bm=bm, nlight=nlight, p=p, cfg=cfg,
                          T=T, meta=meta, gfull=graph_of(p.run)))
        print(f"  case cells={cells} tokens={M} BM{bm} light={nlight}: "
              + "  ".join(f"{k}: {v['experts']} experts, {v['real_pairs']} pairs, "
                          f"{v['padded_rows']} rows" for k, v in meta.items()),
              flush=True)

    rep = lambda c, name, R: tuple(t.repeat(R) for t in c["T"][name])

    def measure(c, parts, secs):
        fire(c["p"], c["cfg"], parts)                              # warm
        out = []
        with Power(discard=0.3) as pw:
            t0 = time.time()
            while time.time() - t0 < secs:
                out.append(fire(c["p"], c["cfg"], parts))
        return np.array(out), pw.summary()

    out = a.out or os.path.join(HERE, "data", f"sm_corun_scan_{a.model}.csv")
    sink = RowSink(out, SCAN_FIELDS)
    smi("-pm", "1")
    rng, clocks = random.Random(23), [int(v) for v in a.clocks.split(",")]
    t_start = time.time()
    try:
        for pas in range(a.repeats):
            grid = clocks[:]; rng.shuffle(grid)
            for f in grid:
                lock(f)
                with Power(discard=0.5) as pw:
                    time.sleep(1.2)
                idle = pw.summary()

                def emit(c, part, mode, sms, ms, R, sm, kernel="persistent", nl="",
                         nh="", overlap=""):
                    m = c["meta"][part]
                    sink.add(pas=pas, cells=c["cells"], tokens=c["tokens"],
                             block_m=c["bm"], light_experts=c["nlight"], kernel=kernel,
                             part=part, mode=mode, sms=sms, sm_light=nl, sm_heavy=nh,
                             experts=m["experts"], real_pairs=m["real_pairs"],
                             padded_rows=m["padded_rows"], clock=f,
                             clock_med=sm["clock_med"],
                             held=abs(sm["clock_med"] - f) <= TOL,
                             ms_per_pass=round(float(ms), 5), passes=R, overlap=overlap,
                             power_w=sm["power_w"], idle_w=idle["power_w"],
                             temp_c=gpu_temp())

                for c in cases:
                    tag = f"cells={c['cells']} BM{c['bm']} L{c['nlight']}"
                    try:
                        # the whole layer on every SM: launches only, then end to end
                        t = fire(c["p"], c["cfg"], [(sA, *rep(c, "all", 4), a.all_sms)])[0] / 4
                        R = max(2, int(round(a.target_ms / t)))
                        ts, sm = measure(c, [(sA, *rep(c, "all", R), a.all_sms)], a.secs)
                        emit(c, "all", "solo", a.all_sms, np.median(ts[:, 0]) / R, R, sm)
                        replay_for(c["gfull"], 0.2)
                        with Power(discard=0.3) as pw:
                            wall, n = replay_for(c["gfull"], a.secs)
                        emit(c, "all", "solo_full", a.all_sms, wall / (n * 4) * 1e3, n,
                             pw.summary())
                        for nl, nh in splits:
                            R, solo = {}, {}
                            for name, nb, st in (("light", nl, sA), ("heavy", nh, sB)):
                                t = fire(c["p"], c["cfg"], [(st, *rep(c, name, 4), nb)])[0] / 4
                                R[name] = max(2, int(round(a.target_ms / t)))
                                ts, sm = measure(c, [(st, *rep(c, name, R[name]), nb)],
                                                 a.secs)
                                solo[name] = np.median(ts[:, 0]) / R[name]
                                emit(c, name, "solo", nb, solo[name], R[name], sm, nl=nl,
                                     nh=nh)
                            for _ in range(4):             # finish together
                                tl_, th_ = fire(c["p"], c["cfg"],
                                                [(sA, *rep(c, "light", R["light"]), nl),
                                                 (sB, *rep(c, "heavy", R["heavy"]), nh)])
                                if abs(tl_ / th_ - 1) < 0.04:
                                    break
                                # retune the stream with MORE passes: a stream that
                                # repeats 3 times can only be moved in 33% steps
                                if R["light"] >= R["heavy"]:
                                    R["light"] = max(2, int(round(R["light"] * th_ / tl_)))
                                else:
                                    R["heavy"] = max(2, int(round(R["heavy"] * tl_ / th_)))
                            ts, smc = measure(c, [(sA, *rep(c, "light", R["light"]), nl),
                                                  (sB, *rep(c, "heavy", R["heavy"]), nh)],
                                              a.secs)
                            med = np.median(ts, axis=0)
                            ov = round(float(min(med) / max(med)), 3)
                            for k, (name, nb) in enumerate((("light", nl), ("heavy", nh))):
                                emit(c, name, "corun", nb, med[k] / R[name], R[name], smc,
                                     nl=nl, nh=nh, overlap=ov)
                    except Exception as e:                 # log it, keep the sweep alive
                        print(f"  !! p{pas} {f} MHz {tag}: {type(e).__name__}: {e}",
                              flush=True)
                        torch.cuda.synchronize()
                for cells, q in queues.items():            # vLLM's queue kernel
                    try:
                        replay_for(q["g"], 0.2)
                        with Power(discard=0.3) as pw:
                            wall, n = replay_for(q["g"], a.secs)
                        sm = pw.summary()
                        sink.add(pas=pas, cells=cells, tokens=q["tokens"],
                                 block_m=base["BLOCK_SIZE_M"], kernel="queue", part="all",
                                 mode="solo_full", sms=a.all_sms, experts=q["experts"],
                                 real_pairs=q["pairs"], clock=f,
                                 clock_med=sm["clock_med"],
                                 held=abs(sm["clock_med"] - f) <= TOL,
                                 ms_per_pass=round(wall / (n * 4) * 1e3, 5), passes=n,
                                 power_w=sm["power_w"], idle_w=idle["power_w"],
                                 temp_c=gpu_temp())
                    except Exception as e:
                        print(f"  !! p{pas} {f} MHz queue cells={cells}: {e}", flush=True)
                sink.flush()
                print(f"  p{pas} {f} MHz done: {sink.n} rows, "
                      f"{(time.time() - t_start) / 60:.1f} min, {gpu_temp()} C",
                      flush=True)
    finally:
        smi("-rgc"); print("  [clocks reset]", flush=True)
        sink.close()


# ------------------------------------------------------------------ pinned ----
def pinned(a):
    """The pinned layer END TO END: both partitions launched side by side on their own
    SM sets, the activation between the two GEMMs, the top-k sum after them -- one block
    per SM. scan timed the partitions as two free-running streams; this is the layer a
    two-domain design would actually execute, barriers included, and it is what the
    simulator's event engine is checked against."""
    spec = SHAPES[a.model]
    base = json.load(open(os.path.join(HERE, "data", f"b1_config_{a.model}.json")))
    splits = [tuple(int(v) for v in s.split("+")) for s in a.splits.split(",")]
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    jobs = []
    for cells, bm, nlight in [tuple(int(v) for v in c.split(":")) for c in a.cases.split(",")]:
        if cells == 0:
            counts = load_counts(spec, 0, a.layer)
            x, w1, w2, tw, ti, M = make_inputs(spec, counts)
        else:
            counts, (x, w1, w2, tw, ti, M) = decode_inputs(spec, a.layer, cells)
        cfg = dict(base); cfg["BLOCK_SIZE_M"] = bm
        light_e = np.argsort(counts, kind="stable")[:nlight]
        for nl, nh in splits:
            # one PersistentMoE per split: its tile cache is keyed by partition index
            p = PersistentMoE(x, w1, w2, tw, ti, cfg)
            p.run(); torch.cuda.synchronize()
            mask = torch.from_numpy(
                np.isin(p.rowblock_expert.cpu().numpy(), light_e)).to(p.live_m.device)
            part = [(p.live_m[mask], nl), (p.live_m[~mask], nh)]     # cold first
            g = graph_of(lambda p=p, part=part: p.run(partition=part, streams=streams))
            jobs.append(dict(cells=cells, tokens=M, bm=bm, nlight=nlight, nl=nl, nh=nh,
                             p=p, g=g))
        print(f"  case cells={cells} tokens={M} BM{bm} light={nlight}: "
              f"{len(splits)} splits captured", flush=True)
    out = a.out or os.path.join(HERE, "data", f"pinned_layer_{a.model}.csv")
    sink = RowSink(out, ["pas", "cells", "tokens", "block_m", "light_experts", "sm_light",
                         "sm_heavy", "clock", "clock_med", "held", "ms_per_layer", "replays",
                         "power_w", "idle_w", "temp_c"])
    smi("-pm", "1")
    rng, clocks = random.Random(41), [int(v) for v in a.clocks.split(",")]
    try:
        for pas in range(a.repeats):
            grid = clocks[:]; rng.shuffle(grid)
            for f in grid:
                lock(f)
                with Power(discard=0.5) as pw:
                    time.sleep(1.2)
                idle_ = pw.summary()
                for j in jobs:
                    replay_for(j["g"], 0.2)
                    with Power(discard=0.3) as pw:
                        wall, n = replay_for(j["g"], a.secs)
                    sm = pw.summary()
                    sink.add(pas=pas, cells=j["cells"], tokens=j["tokens"], block_m=j["bm"],
                             light_experts=j["nlight"], sm_light=j["nl"], sm_heavy=j["nh"],
                             clock=f, clock_med=sm["clock_med"],
                             held=abs(sm["clock_med"] - f) <= TOL,
                             ms_per_layer=round(wall / (n * 4) * 1e3, 5), replays=n,
                             power_w=sm["power_w"], idle_w=idle_["power_w"],
                             temp_c=gpu_temp())
                sink.flush()
                print(f"  p{pas} {f} MHz done: {sink.n} rows", flush=True)
    finally:
        smi("-rgc"); print("  [clocks reset]", flush=True)
        sink.close()


# --------------------------------------------------------------- occupancy ----
def occupancy(a):
    """How much of the persistent kernel's handicap is OCCUPANCY. One resident block per
    SM is what makes an SM set a real partition, and it gives up the latency hiding
    vLLM's kernel gets from keeping several blocks on an SM. pin_vs_queue measured that
    at one point (prefill, 1200 MHz: +24% at one block per SM, +5% at two). This is the
    whole layer, unpartitioned, end to end, at 1-3 blocks per SM, for decode batches
    too, beside the queue kernel at the same clock."""
    spec = SHAPES[a.model]
    base = json.load(open(os.path.join(HERE, "data", f"b1_config_{a.model}.json")))
    jobs = []
    for cells in [int(v) for v in a.cells.split(",")]:
        if cells == 0:
            counts = load_counts(spec, 0, a.layer)
            x, w1, w2, tw, ti, M = make_inputs(spec, counts)
        else:
            counts, (x, w1, w2, tw, ti, M) = decode_inputs(spec, a.layer, cells)
        q = PreparedMoE(x, w1, w2, tw, ti, base)
        jobs.append(dict(cells=cells, tokens=M, kernel="queue", bm=base["BLOCK_SIZE_M"],
                         blocks=0, keep=q, g=graph_of(q.run)))
        for bm in [int(v) for v in a.block_m.split(",")]:
            cfg = dict(base); cfg["BLOCK_SIZE_M"] = bm
            p = PersistentMoE(x, w1, w2, tw, ti, cfg)
            for b in [int(v) for v in a.blocks.split(",")]:
                jobs.append(dict(cells=cells, tokens=M, kernel="persistent", bm=bm, blocks=b,
                                 keep=p, g=graph_of(lambda p=p, b=b: p.run(blocks_per_sm=b))))
        print(f"  batch {M}: captured", flush=True)
    out = a.out or os.path.join(HERE, "data", f"occupancy_{a.model}.csv")
    sink = RowSink(out, ["pas", "cells", "tokens", "kernel", "block_m", "blocks_per_sm",
                         "clock", "clock_med", "held", "ms_per_layer", "replays", "power_w",
                         "idle_w", "temp_c"])
    smi("-pm", "1")
    rng, clocks = random.Random(53), [int(v) for v in a.clocks.split(",")]
    try:
        for pas in range(a.repeats):
            grid = clocks[:]; rng.shuffle(grid)
            for f in grid:
                lock(f)
                with Power(discard=0.5) as pw:
                    time.sleep(1.2)
                idle_ = pw.summary()
                for j in jobs:
                    replay_for(j["g"], 0.2)
                    with Power(discard=0.3) as pw:
                        wall, n = replay_for(j["g"], a.secs)
                    sm = pw.summary()
                    sink.add(pas=pas, cells=j["cells"], tokens=j["tokens"], kernel=j["kernel"],
                             block_m=j["bm"], blocks_per_sm=j["blocks"], clock=f,
                             clock_med=sm["clock_med"], held=abs(sm["clock_med"] - f) <= TOL,
                             ms_per_layer=round(wall / (n * 4) * 1e3, 5), replays=n,
                             power_w=sm["power_w"], idle_w=idle_["power_w"],
                             temp_c=gpu_temp())
                sink.flush()
                print(f"  p{pas} {f} MHz done: {sink.n} rows", flush=True)
    finally:
        smi("-rgc"); print("  [clocks reset]", flush=True)
        sink.close()


# -------------------------------------------------------------------- idle ----
def idle(a):
    spec = SHAPES[a.model]
    cfg = json.load(open(os.path.join(HERE, "data", f"b1_config_{a.model}.json")))
    counts = load_counts(spec, 0, a.layer)
    x, w1, w2, tw, ti, M = make_inputs(spec, counts)       # a live context with memory
    p = PreparedMoE(x, w1, w2, tw, ti, cfg)                # held, as in every campaign
    g = graph_of(p.run)
    out = a.out or os.path.join(HERE, "data", f"idle_sweep_{a.model}.csv")
    sink = RowSink(out, ["pas", "clock", "state", "clock_med", "power_w", "temp_start_c",
                         "temp_end_c", "n_samples"])
    smi("-pm", "1")
    grid, rng = list(range(a.lo, a.hi + 1, a.step)), random.Random(31)
    try:
        for pas in range(a.repeats):
            order = grid[:]; rng.shuffle(order)
            for f in order:
                lock(f)
                t0 = gpu_temp()
                with Power(discard=0.5) as pw:
                    time.sleep(a.secs)
                sm = pw.summary()
                sink.add(pas=pas, clock=f, state="cold", clock_med=sm["clock_med"],
                         power_w=sm["power_w"], temp_start_c=t0, temp_end_c=gpu_temp(),
                         n_samples=sm["n_samples"])
                # the same clock with the die still hot: NVML power needs ~0.45 s to
                # settle after a load stops, and the die cools within seconds
                replay_for(g, a.load)
                time.sleep(0.5)
                t0 = gpu_temp()
                with Power(discard=0.0) as pw:
                    time.sleep(1.0)
                sm = pw.summary()
                sink.add(pas=pas, clock=f, state="warm", clock_med=sm["clock_med"],
                         power_w=sm["power_w"], temp_start_c=t0, temp_end_c=gpu_temp(),
                         n_samples=sm["n_samples"])
            sink.flush()
            print(f"  pass {pas} done: {sink.n} rows", flush=True)
    finally:
        smi("-rgc"); print("  [clocks reset]", flush=True)
        sink.close()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("governor", "corun", "limiter", "scan", "idle", "pinned", "occupancy"):
        s = sub.add_parser(name)
        s.add_argument("--model", default="qwen")
        s.add_argument("--layer", type=int, default=0)
        s.add_argument("--out", default=None)
        if name == "governor":
            s.add_argument("--cells", default="8,16,32,64,96,128,160,192",
                           help="decode cells pooled per batch; 1 cell = 64 tokens")
            s.add_argument("--modes", default="-1,1290,1410",
                           help="clock requests; -1 = unlocked governor")
            s.add_argument("--repeats", type=int, default=3)
            s.add_argument("--settle", type=float, default=1.5)
            s.add_argument("--secs", type=float, default=2.0)
        if name == "corun":
            s.add_argument("--cells", default="16,32,64")
            s.add_argument("--block-m", default="16,64")
            s.add_argument("--light-experts", type=int, default=64)
            s.add_argument("--splits", default="54+54,32+76",
                           help="light+heavy SM counts; must sum to 108")
            s.add_argument("--clocks", default="930,1050,1170,1290")
            s.add_argument("--repeats", type=int, default=3)
            s.add_argument("--target-ms", type=float, default=60.0)
            s.add_argument("--secs", type=float, default=1.2)
        if name == "scan":
            s.add_argument("--cases", default="16:16:64,16:64:64,32:16:64,32:64:64,"
                           "64:16:64,64:64:64,0:128:64,0:128:16",
                           help="cells:BLOCK_M:light_experts; cells=0 is the prefill batch")
            s.add_argument("--splits", default="16+92,32+76,48+60,54+54,64+44,80+28",
                           help="light+heavy SM counts, whole GPCs where possible")
            s.add_argument("--clocks", default="930,1050,1170,1290,1350,1410")
            s.add_argument("--all-sms", type=int, default=108)
            s.add_argument("--config-from", default=None,
                           help="use this model's tuned kernel config (for a model that "
                                "has none of its own)")
            s.add_argument("--repeats", type=int, default=3)
            s.add_argument("--target-ms", type=float, default=80.0)
            s.add_argument("--secs", type=float, default=1.2)
        if name == "idle":
            s.add_argument("--lo", type=int, default=210)
            s.add_argument("--hi", type=int, default=1410)
            s.add_argument("--step", type=int, default=30)
            s.add_argument("--repeats", type=int, default=3)
            s.add_argument("--secs", type=float, default=2.0)
            s.add_argument("--load", type=float, default=2.0)
        if name == "pinned":
            s.add_argument("--cases", default="16:64:64,32:64:64,64:64:64,32:16:64,"
                           "0:128:64,0:128:16",
                           help="cells:BLOCK_M:light_experts; cells=0 is the prefill batch")
            s.add_argument("--splits", default="16+92,32+76,48+60,54+54,64+44,80+28")
            s.add_argument("--clocks", default="1050,1290,1410")
            s.add_argument("--repeats", type=int, default=2)
            s.add_argument("--secs", type=float, default=1.0)
        if name == "occupancy":
            s.add_argument("--cells", default="8,16,32,64,0",
                           help="decode cells per batch; 0 is the prefill batch")
            s.add_argument("--block-m", default="64,128")
            s.add_argument("--blocks", default="1,2,3", help="resident blocks per SM")
            s.add_argument("--clocks", default="1050,1290,1410")
            s.add_argument("--repeats", type=int, default=2)
            s.add_argument("--secs", type=float, default=1.0)
        if name == "limiter":
            s.add_argument("--trials", type=int, default=5)
            s.add_argument("--idle", type=float, default=3.0)
            s.add_argument("--load", type=float, default=4.0)
    a = ap.parse_args()
    {"governor": governor, "corun": corun, "limiter": limiter, "scan": scan,
     "idle": idle, "pinned": pinned, "occupancy": occupancy}[a.cmd](a)


if __name__ == "__main__":
    main()
