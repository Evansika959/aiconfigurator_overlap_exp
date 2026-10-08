#!/usr/bin/env python3
"""What the stock governor does while the LOAD CHANGES, layer by layer.

Every sweep so far held one batch at one clock until it settled. A simulator that claims
to be dynamic has to be checked against something that moves: a scripted sequence of
batches under the unlocked governor, with the latency of every graph replay and every
NVML power and clock reading on one time axis.

  plan      "d16:4,d64:4,idle:3,p0:5"   run decode batches of 16 and 64 cells (64 tokens
            a cell) for 4 s each, sit idle for 3 s, run the prefill batch for 5 s.
            "d20@160" pools 20 cells beginning at traced cell 160 instead of cell 0, and
            "p7" is prompt set 7: routing no sweep has used.

Two plans are built in:

  dynamics  alternates a batch the wall does not bind (1024 tokens, governor at 1410 MHz)
            with ones it does (4096 tokens and prefill, 1320 MHz): the governor's step
            response in both directions, and its wake-up from idle
  trace     fourteen segments of mixed size and length with idle gaps: the end-to-end
            validation trace the simulator is replayed against

THE CLOCK IS NEVER LOCKED and the power cap is never touched: this is the box as it
runs by default. Latency is the clock proxy -- NVML refreshes the clock about every
100 ms, a replay finishes every 4-20 ms.

HOW A REPLAY IS TIMED WITHOUT DISTURBING IT. A torch.cuda.synchronize() after every
replay would leave the GPU idle while the host wakes up, and the governor would see a
lighter load than the one being studied. Instead each replay is followed by a CUDA event
recorded on the same stream; events cost nothing and are read afterwards. The host stays
a few replays ahead and blocks only on an event from several replays back, so the queue
never empties and never grows without bound.

WHOLE FORWARD PASSES. `--layers all` makes every replay one pass over all 48 MoE
layers of the model, each with its own traced routing (weights and buffers are shared
between layers; routing is what differs, and it is what time and power depend on). A
prefill pass is then ~180 ms and a 2048-token decode step ~70 ms: the simulator is
checked on the routing of layers it was never calibrated on, in the order a model runs
them.

  python3 governor_trace.py --plan dynamics --repeats 3      # ~2.5 min
  python3 governor_trace.py --plan trace --repeats 3         # ~2 min
  python3 governor_trace.py --plan forward --layers all --tag forward --repeats 3
"""
import argparse, csv, json, os, subprocess, sys, threading, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch.nn.functional as F
from vendor.fused_moe_triton import PreparedMoE, _invoke, moe_align_block_size
from b1_bench import SHAPES, load_counts, make_inputs
from batch_clock_sweep import decode_batches
from decode_wall import decode_inputs, graph_of

HERE = os.path.dirname(os.path.abspath(__file__))
REPS = 4                                    # layers per graph replay (one layer, repeated)
PLANS = {
    "forward": "idle:2,d32:4,p0:4,idle:1.5,d16:3,d64:4,p0:3,idle:2.5,d8:3,d32:3,idle:1",
    "dynamics": "idle:4,d16:5,d64:5,d16:5,p0:5,d16:5,idle:4,p0:6,idle:3,d64:5,idle:2",
    "trace": "idle:2,d8:2,d32:3,p0:3,idle:1.5,d16:2,d64:3,d96:3,idle:2.5,p0:2,d32:2,"
             "d16:3,d128:3,idle:1,d64:2,idle:1",
}


def decode_of(name):
    """'d20' -> (20 cells, from cell 0); 'd20@160' -> (20 cells, from cell 160)."""
    cells, _, start = name[1:].partition("@")
    return int(cells), int(start or 0)


class Stack:
    """One forward pass over several MoE layers: one set of weights and buffers, one
    routing per layer. `run()` is what PreparedMoE.run does, per layer."""

    def __init__(self, spec, name, layers, cfg):
        E, topk = spec["E"], spec["topk"]
        g = torch.Generator(device="cpu").manual_seed(0)
        self.per = []
        for i, L in enumerate(layers):
            if name.startswith("p"):
                counts = load_counts(spec, int(name[1:]), L)
            else:
                counts = decode_batches(spec, L, *decode_of(name))
                counts[np.argmax(counts)] -= int(counts.sum()) % topk
            if i == 0:
                x, w1, w2, tw, ti, M = make_inputs(spec, counts)
                self.p = PreparedMoE(x, w1, w2, tw, ti, cfg)       # the shared buffers
                self.tokens = M
            total = int(counts.sum())
            assert total == self.tokens * topk, "every layer must hold the same tokens"
            flat = torch.repeat_interleave(torch.arange(E, dtype=torch.int32),
                                           torch.from_numpy(counts))
            ti = flat[torch.randperm(total, generator=g)].view(self.tokens, topk).contiguous().cuda()
            self.per.append(moe_align_block_size(ti, cfg["BLOCK_SIZE_M"], E))
        self.n = len(layers)

    def run(self):
        p = self.p
        for sorted_ids, expert_ids, n_pad in self.per:
            _invoke(p.x, p.w1, p.c1, p.tw, sorted_ids, expert_ids, n_pad, False, p.topk,
                    p.cfg, p.ct)
            g = p.c1.view(-1, 2 * p.I)
            torch.mul(F.silu(g[:, :p.I]), g[:, p.I:], out=p.c2)
            _invoke(p.c2, p.w2, p.c3, p.tw, sorted_ids, expert_ids, n_pad, True, 1, p.cfg,
                    p.ct)
            torch.sum(p.c3, dim=1, out=p.out)
        return p.out


class Poller:
    """NVML power and clock at 200 Hz, keeping a sample only when a reading changes
    (plus a heartbeat): NVML answers from a value it refreshes about every 100 ms."""

    def __init__(self, dev=0):
        import pynvml
        self.nv = pynvml
        pynvml.nvmlInit()
        self.h = pynvml.nvmlDeviceGetHandleByIndex(dev)
        self.rows, self._stop = [], threading.Event()

    def _loop(self):
        last, last_t = None, 0.0
        while not self._stop.is_set():
            t = time.time()
            cur = (self.nv.nvmlDeviceGetPowerUsage(self.h) / 1000,
                   self.nv.nvmlDeviceGetClockInfo(self.h, self.nv.NVML_CLOCK_SM),
                   self.nv.nvmlDeviceGetTemperature(self.h, self.nv.NVML_TEMPERATURE_GPU))
            if cur != last or t - last_t > 0.05:
                self.rows.append((t, *cur)); last, last_t = cur, t
            time.sleep(0.005)

    def __enter__(self):
        self._t = threading.Thread(target=self._loop, daemon=True); self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set(); self._t.join(timeout=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--plan", default="dynamics", help="a built-in plan name or a plan string")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--ahead-ms", type=float, default=60.0,
                    help="how far the host may run ahead of the GPU")
    ap.add_argument("--layers", default=None,
                    help="'all', or a comma list of MoE layers: one replay = one pass "
                         "over them. Default: --layer alone, four times per replay")
    ap.add_argument("--lock", type=int, default=None,
                    help="lock the SM clock at this many MHz for the run (reset afterwards): "
                         "the same script under a fixed clock instead of the governor")
    ap.add_argument("--nvml-index", type=int, default=0,
                    help="physical GPU the process runs on (with CUDA_VISIBLE_DEVICES set, "
                         "torch's device 0 is not NVML's device 0)")
    a = ap.parse_args()
    tag = a.tag or (a.plan if a.plan in PLANS else "custom")
    plan = [(s.split(":")[0], float(s.split(":")[1]))
            for s in PLANS.get(a.plan, a.plan).split(",")]

    spec = SHAPES[a.model]
    cfg = json.load(open(os.path.join(HERE, "data", f"b1_config_{a.model}.json")))
    layers = None
    if a.layers:
        layers = list(range(spec["layers"])) if a.layers == "all" \
            else [int(v) for v in a.layers.split(",")]
    per_replay = len(layers) if layers else REPS       # layers one replay runs
    loads = {}
    for name in sorted({n for n, _ in plan if n != "idle"}):
        if layers:
            p = Stack(spec, name, layers, cfg)
            M, g = p.tokens, graph_of(p.run, 1)
        else:
            if name.startswith("p"):
                counts = load_counts(spec, int(name[1:]), a.layer)
                x, w1, w2, tw, ti, M = make_inputs(spec, counts)
            else:
                counts, (x, w1, w2, tw, ti, M) = decode_inputs(spec, a.layer, *decode_of(name))
            p = PreparedMoE(x, w1, w2, tw, ti, cfg)
            g = graph_of(p.run, REPS)
        g.replay(); torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(5):
            g.replay()
        torch.cuda.synchronize()
        loads[name] = dict(p=p, g=g, tokens=M, est=(time.time() - t0) / 5 * 1e3)
        print(f"  {name}: {M} tokens, {loads[name]['est'] / per_replay:.3f} ms per layer, "
              f"{per_replay} layers per replay", flush=True)

    smi = lambda *x: subprocess.run(["sudo", "nvidia-smi", "-i", str(a.nvml_index), *x],
                                    capture_output=True)
    smi("-rgc")
    if a.lock:
        smi("-lgc", f"{a.lock},{a.lock}")
    rep_rows, nv_rows, seg_rows = [], [], []
    try:
        _runs(a, plan, loads, per_replay, rep_rows, nv_rows, seg_rows)
    finally:
        if a.lock:
            smi("-rgc"); print("  [clocks reset]", flush=True)
    for name, rows in (("replays", rep_rows), ("nvml", nv_rows), ("segments", seg_rows)):
        out = os.path.join(HERE, "data", f"governor_trace_{tag}_{name}_{a.model}.csv")
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"wrote {out} ({len(rows)} rows)")


def _runs(a, plan, loads, per_replay, rep_rows, nv_rows, seg_rows):
    for run in range(a.repeats):
        torch.cuda.synchronize()
        time.sleep(3.0)                                 # start every run from a parked GPU
        with Poller(a.nvml_index) as nv:
            e0 = torch.cuda.Event(enable_timing=True); e0.record(); e0.synchronize()
            t_host = time.time()
            marks = []                                  # (seg, name, k, event)
            for si, (name, secs) in enumerate(plan):
                s = torch.cuda.Event(enable_timing=True); s.record()
                marks.append((si, name, -1, s))
                if name == "idle":
                    torch.cuda.synchronize(); time.sleep(secs)
                    continue
                L = loads[name]
                n = max(4, int(secs * 1e3 / L["est"]))
                ahead = max(2, int(a.ahead_ms / L["est"]))
                evs = []
                for k in range(n):
                    L["g"].replay()
                    e = torch.cuda.Event(enable_timing=True); e.record()
                    evs.append(e); marks.append((si, name, k, e))
                    if k >= ahead:
                        evs[k - ahead].synchronize()
            end = torch.cuda.Event(enable_timing=True); end.record()
            torch.cuda.synchronize()
            time.sleep(1.0)
        t_seg, prev = {}, None
        for si, name, k, e in marks:
            t = e0.elapsed_time(e)
            if k < 0:
                t_seg[si] = t; prev = t
                continue
            rep_rows.append(dict(run=run, seg=si, load=name, tokens=loads[name]["tokens"],
                                 k=k, t_end_ms=round(t, 4),
                                 ms_per_layer=round((t - prev) / per_replay, 5)))
            prev = t
        t_end = e0.elapsed_time(end)
        for si, (name, secs) in enumerate(plan):
            t1 = t_seg[si + 1] if si + 1 in t_seg else t_end
            n = sum(1 for m in marks if m[0] == si and m[2] >= 0)
            seg_rows.append(dict(run=run, seg=si, load=name,
                                 tokens=loads[name]["tokens"] if name != "idle" else 0,
                                 t_start_ms=round(t_seg[si], 3), t_end_ms=round(t1, 3),
                                 replays=n, layers=n * per_replay,
                                 layer_list=a.layers or str(a.layer)))
        for t, pw, ck, temp in nv.rows:
            nv_rows.append(dict(run=run, t_ms=round((t - t_host) * 1e3, 2), power_w=pw,
                                clock_mhz=ck, temp_c=temp))
        busy = [r for r in rep_rows if r["run"] == run]
        print(f"  run {run}: {len(busy)} replays, {len(nv.rows)} NVML readings, "
              f"{t_end / 1e3:.1f} s", flush=True)


if __name__ == "__main__":
    main()
