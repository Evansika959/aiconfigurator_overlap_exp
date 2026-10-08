"""The measured CSVs in ../data, as records the fit and the validation both read.

One record = one (configuration, clock), averaged over the sweep's repeat passes, with
the partitions it ran already built. Rows where the clock was not held are dropped.

A record is a dict. The kinds every consumer knows:

  kind 'solo'   one partition alone on `n` SMs                    part, n, f, ms, power
       'corun'  light on n_l SMs beside heavy on n_h              light{part,n,ms}, heavy{...}, f, power
       'full'   the whole layer end to end, persistent kernel     part, n, f, ms, power, cells, bm
       'queue'  the whole layer end to end, vLLM's kernel         the same

To add a measurement: write a loader here that returns such records, and either list
its file in TRAIN (it is then fitted) or score it from a rung of validate.py.
"""

import os
from collections import defaultdict

import numpy as np

from .demand import DATA, SHAPES, Partition, counts_for, split
from .files import read_csv

# The sweeps the calibration is fitted on: (file stem, MoE layer). Everything else in
# ../data is hold-out. The second one was collected AS a hold-out and scored as one
# (calibrate.py prints that cross-validation) before it was added here.
TRAIN = [
    ("sm_corun_scan", 0),  # 8 cases, 6 SM splits, 930-1410 MHz
    ("sm_corun_holdout", 0),  # 8 other cases, 990-1380 MHz
    ("sm_corun_lowclk", 0),
]  # the first 8 cases again, 510-870 MHz

_PART = {}


def partition(model, cells, block_m, n_cold, part, layer=0, rep=0):
    """The 'light' / 'heavy' / 'all' partition of a measured batch (cached)."""
    key = (model, cells, block_m, n_cold, part, layer, rep)
    if key not in _PART:
        shape = SHAPES[model]
        c = counts_for(shape, cells, layer, rep)
        cold = split(c, n_cold)
        ex = None if part == "all" else (cold if part == "light" else ~cold)
        _PART[key] = Partition(shape, c, block_m, ex)
    return _PART[key]


# -------------------------------------------------------------------- helpers ----
def _groups(name, key, keep=None):
    """Rows of ../data/<name> that pass `keep`, grouped by `key(row)`, sorted by key:
    one group is the repeat passes of one configuration."""
    g = defaultdict(list)
    for r in read_csv(os.path.join(DATA, name)):
        if keep is None or keep(r):
            g[key(r)].append(r)
    return sorted(g.items())


def _held(r):
    return r["held"] == "True"


def _mean(v, col):
    return float(np.mean([float(x[col]) for x in v]))


# ------------------------------------------------------------- the calibration ----
def train(model="qwen", only=None):
    """Records of every training sweep that exists (or of the stems in `only`)."""
    out = []
    for stem, layer in TRAIN:
        path = os.path.join(DATA, f"{stem}_{model}.csv")
        if os.path.exists(path) and (only is None or stem in only):
            for r in scan(model, f"{stem}_{model}.csv", layer=layer):
                r["src"] = stem
                out.append(r)
    return out


def scan(model="qwen", name=None, held_only=True, layer=0):
    """decode_wall.py scan -> 'solo', 'corun', 'full' and 'queue' records."""
    key = lambda r: (
        int(r["cells"]),
        int(r["block_m"]),
        int(r["light_experts"] or 0),
        r["kernel"],
        r["part"],
        r["mode"],
        int(r["sms"]),
        int(r["sm_light"] or 0),
        int(r["clock"]),
    )
    out, co = [], {}
    for (cells, bm, nl, kern, part, mode, sms, sml, f), v in _groups(
        name or f"sm_corun_scan_{model}.csv", key, _held if held_only else None
    ):
        rec = dict(
            cells=cells,
            bm=bm,
            n_cold=nl,
            f=f,
            ms=_mean(v, "ms_per_pass"),
            power=_mean(v, "power_w"),
            idle=_mean(v, "idle_w"),
            temp=_mean(v, "temp_c"),
            reps=len(v),
            tokens=int(v[0]["tokens"]),
            pairs=int(v[0]["real_pairs"]),
        )
        if kern == "queue":
            out.append(dict(rec, kind="queue", n=sms, part=partition(model, cells, bm, 0, "all", layer)))
        elif mode == "solo_full":
            out.append(dict(rec, kind="full", n=sms, part=partition(model, cells, bm, nl, "all", layer)))
        elif mode == "solo":
            p = partition(model, cells, bm, nl, part, layer)
            assert int(p.rows) == int(v[0]["padded_rows"]), (cells, bm, nl, part)
            out.append(dict(rec, kind="solo", n=sms, name=part, part=p, split=sml))
        else:  # corun: join the two streams into one record
            c = co.setdefault((cells, bm, nl, sml, f), dict(rec, kind="corun", split=sml, overlap=_mean(v, "overlap")))
            c[part] = dict(n=sms, ms=rec["ms"], part=partition(model, cells, bm, nl, part, layer))
    return out + [c for c in co.values() if "light" in c and "heavy" in c]


def queue_sweeps(model="qwen"):
    """Earlier end-to-end sweeps of vLLM's queue kernel at locked clocks -> 'queue'
    records: batch_clock (decode batches 64-4096 tokens, 15 clocks to 1290 MHz) and
    b1_energy (the prefill batch, 37 clocks to 1410 MHz)."""
    out = []
    for (tok, f), v in _groups(f"batch_clock_{model}.csv", lambda r: (int(r["tokens"]), int(r["clock"])), _held):
        out.append(
            dict(
                kind="queue",
                cells=tok // 64,
                bm=128,
                n_cold=0,
                f=f,
                n=108,
                tokens=tok,
                pairs=int(v[0]["pairs"]),
                ms=_mean(v, "ms_per_layer"),
                power=_mean(v, "power_w"),
                idle=_mean(v, "idle_w"),
                reps=len(v),
                part=partition(model, tok // 64, 128, 0, "all"),
                src="batch_clock",
            )
        )
    keep = lambda r: _held(r) and r["locked"] == "True" and int(r["rep"]) == 0 and int(r["layer"]) == 0
    for f, v in _groups(f"b1_energy_{model}.csv", lambda r: int(r["clock"]), keep):
        p = partition(model, 0, 128, 0, "all")
        out.append(
            dict(
                kind="queue",
                cells=0,
                bm=128,
                n_cold=0,
                f=f,
                n=108,
                tokens=int(v[0]["m_tokens"]),
                pairs=int(p.pairs),
                ms=_mean(v, "ms_per_layer"),
                power=_mean(v, "power_w"),
                idle=_mean(v, "idle_w"),
                reps=len(v),
                part=p,
                src="b1_energy",
            )
        )
    return out


def idle_sweep(model="qwen", state="cold"):
    """decode_wall.py idle -> (clocks, W, die temperature). One method, every clock."""
    keep = lambda r: r["state"] == state and abs(int(r["clock_med"]) - int(r["clock"])) <= 7
    g = _groups(f"idle_sweep_{model}.csv", lambda r: int(r["clock"]), keep)
    return (
        [f for f, _ in g],
        [_mean(v, "power_w") for _, v in g],
        [0.5 * (_mean(v, "temp_start_c") + _mean(v, "temp_end_c")) for _, v in g],
    )


# ---------------------------------------------------------------- the hold-outs ----
def kappa(model="qwen", name="partition_kappa_qwen_hiclk54.csv", cells=0, n_cold=16):
    """partition_kappa.py sweeps -> 'solo' records. Earlier campaigns than the scan, on
    other days and other clocks: used to densify the voltage curve and as hold-out.

    Those sweeps replayed a CUDA graph of single passes, so each pass carries the gap
    between two kernel launches (`replay=True`); the scan repeated the tile list inside
    one long launch and has none."""
    out = []
    for (bm, part, sms, f), v in _groups(
        name, lambda r: (int(r["block_m"]), r["part"], int(r["sms"]), int(r["clock"])), _held
    ):
        p = partition(model, cells, bm, n_cold, part)
        if int(p.rows) != int(v[0]["padded_rows"]):
            raise ValueError(
                f"{name}: {part} BM{bm} has {v[0]['padded_rows']} rows, the decomposition says {int(p.rows)}"
            )
        out.append(
            dict(
                kind="solo",
                cells=cells,
                bm=bm,
                n_cold=n_cold,
                f=f,
                n=sms,
                name=part,
                part=p,
                ms=_mean(v, "ms"),
                power=_mean(v, "power_w"),
                idle=_mean(v, "idle_w"),
                reps=len(v),
                src=name,
                replay=True,
            )
        )
    return out


def corun_first(model="qwen", n_cold=64):
    """decode_wall.py corun -> 'solo' and 'corun' records: the first co-run campaign
    (54+54 and 32+76, 930-1290 MHz), run before the scan and never fitted."""
    key = lambda r: (
        int(r["tokens"]) // 64,
        int(r["block_m"]),
        int(r["sm_light"]),
        r["part"],
        r["mode"],
        int(r["sms"]),
        int(r["clock"]),
    )
    out, co = [], {}
    for (cells, bm, sml, part, mode, sms, f), v in _groups(f"decode_corun_{model}.csv", key, _held):
        rec = dict(
            cells=cells,
            bm=bm,
            n_cold=n_cold,
            f=f,
            ms=_mean(v, "ms_per_pass"),
            power=_mean(v, "power_w"),
            idle=_mean(v, "idle_w"),
            reps=len(v),
            src="decode_corun",
        )
        p = partition(model, cells, bm, n_cold, part)
        if mode == "solo":
            out.append(dict(rec, kind="solo", n=sms, name=part, part=p, split=sml))
        else:
            c = co.setdefault((cells, bm, sml, f), dict(rec, kind="corun", split=sml))
            c[part] = dict(n=sms, ms=rec["ms"], part=p)
    return out + [c for c in co.values() if "light" in c and "heavy" in c]


def governor(model="qwen"):
    """decode_wall.py governor -> one row per (mode, batch): clock, power, latency."""
    g = _groups(f"decode_governor_{model}.csv", lambda r: (r["mode"], int(r["tokens"])))
    return [
        dict(
            mode=m,
            tokens=t,
            cells=t // 64,
            clock=_mean(v, "clock_med"),
            power=_mean(v, "power_w"),
            ms=_mean(v, "ms_per_layer"),
            held=all(_held(x) for x in v),
        )
        for (m, t), v in g
    ]


def pin_vs_queue(model="qwen"):
    """pin_vs_queue.py, prefill at 1200 MHz, end to end: {case: (ms, W)}."""
    return {
        k: (_mean(v, "ms"), _mean(v, "power_w")) for k, v in _groups(f"pin_vs_queue_{model}.csv", lambda r: r["case"])
    }


def pinned(model="qwen"):
    """decode_wall.py pinned -> the pinned layer end to end: one record per
    (case, SM split, clock)."""
    key = lambda r: (int(r["cells"]), int(r["block_m"]), int(r["light_experts"]), int(r["sm_light"]), int(r["clock"]))
    return [
        dict(
            cells=c,
            bm=bm,
            n_cold=nc,
            sm_light=sl,
            f=f,
            ms=_mean(v, "ms_per_layer"),
            power=_mean(v, "power_w"),
            reps=len(v),
        )
        for (c, bm, nc, sl, f), v in _groups(f"pinned_layer_{model}.csv", key, _held)
    ]


def occupancy(model="qwen"):
    """decode_wall.py occupancy -> the whole layer end to end, unpartitioned: the queue
    kernel and the persistent kernel at 1-3 resident blocks per SM."""
    key = lambda r: (int(r["cells"]), r["kernel"], int(r["block_m"]), int(r["blocks_per_sm"]), int(r["clock"]))
    return [
        dict(
            cells=c,
            kernel=k,
            bm=bm,
            blocks=b,
            f=f,
            tokens=int(v[0]["tokens"]),
            ms=_mean(v, "ms_per_layer"),
            power=_mean(v, "power_w"),
            reps=len(v),
        )
        for (c, k, bm, b, f), v in _groups(f"occupancy_{model}.csv", key, _held)
    ]
