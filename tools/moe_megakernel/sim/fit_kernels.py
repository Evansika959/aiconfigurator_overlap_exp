"""Fit the kernel profiles of `kernels.py` from measurements and write
calib/a100_kernels.json.

  python3 -m sim.fit_kernels [--no-new]

SOURCES, all under ../overlap_exp/data:
  earlier campaigns   overlap_432, commbound_*, ext_*: solo GEMM and solo all-reduce loop
                      times and node-mean power, 12 shapes x 8 clocks and 4 sizes x 10
                      CTA counts x 8 clocks
  sim_overlap/cal     per-kernel spans on the same cells (2026-10-08): the host's share
                      of a launch, what a co-running kernel costs, per-GPU power
  sim_overlap/commsize  the all-reduce alone from 4 to 512 MiB, 1 to 64 CTAs
  sim_overlap/grids.json  launch grids read by ncu

Nothing from a `blind*` file is read: those are the registered test
(../OVERLAP_HELDOUT.md). `--no-new` fits on the earlier campaigns alone.

WHAT IS FITTED AND WHAT IS NOT. The chip's energies (per MAC, per byte, idle, the
voltage curve) are `calib.Calib.power`, fitted on the MoE kernel, and are used as they
are. Fitted here: the GEMM's cost per MAC and launch cost (3 numbers), the collective's
latency, bandwidth and power tables (measured cells, not a law), how a co-running GEMM
slows the collective, and three numbers per board.
"""

import argparse
import collections
import csv
import json
import os

import numpy as np

from .calib import Calib
from .kernels import DEFAULT, TILE_M, TILE_N, CublasGemm, Kernels, NcclAllReduce
from .lsq import lm

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "overlap_exp", "data")
NEW = os.path.join(DATA, "sim_overlap")
OLD = [
    "overlap_432.csv",
    "commbound_128mib.csv",
    "commbound_256mib.csv",
    "commbound_512mib.csv",
    "ext_64mib.csv",
    "ext_128mib.csv",
    "ext_256mib.csv",
    "ext_512mib.csv",
]
OLD_AUTO = [
    "auto_128mib.csv",
    "auto_256mib.csv",
    "auto_512mib.csv",
    "ext_auto_64mib.csv",
    "ext_auto_128mib.csv",
    "ext_auto_256mib.csv",
    "ext_auto_512mib.csv",
]


def rms(e):
    return float(np.sqrt(np.mean(np.square(e)))) if len(e) else float("nan")


# ------------------------------------------------------------------- loaders ----
def old_rows(files=OLD):
    """The earlier campaigns: one dict per row, times in ms, power the mean of 4 GPUs."""
    out = []
    for name in files:
        with open(os.path.join(DATA, name)) as fh:
            for r in csv.DictReader(fh):
                out.append(
                    dict(
                        src=name,
                        mode=r["mode"],
                        ctas=int(r["ctas"]),
                        f=int(r["clock"]),
                        m=int(r["m"]),
                        n=int(r["n"]),
                        k=int(r["k"]),
                        mib=int(r["ar_mib"]),
                        ms=float(r["iter_ms"]),
                        held=r["clock_held"] == "True",
                        clk=int(r["clock_min"]),
                        w=float(r["power_per_gpu_w"]),
                        shape=f"{r['m']}x{r['n']}x{r['k']}",
                    )
                )
    return out


def finished(name):
    """Has that campaign's sweep written its last line? A file still being written must
    not be fitted on: its cells are a biased sample of the plan."""
    log = os.path.join(NEW, f"{name}.log")
    if not os.path.exists(log):
        return False
    with open(log) as fh:
        return any(ln.startswith("DONE") for ln in fh)


def span_rows(name, need_finished=False):
    """A sweep_spans.py file: one dict per (cell, rank); [] if it does not exist."""
    path = os.path.join(NEW, f"{name}.csv")
    if not os.path.exists(path) or (need_finished and not finished(name)):
        return []
    out = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            d = dict(r)
            for key in (
                "world",
                "ctas",
                "run",
                "mib",
                "rank",
                "clock_req",
                "m",
                "n",
                "k",
                "reps",
                "iters",
                "clock_min",
                "clock_med",
                "temp_c",
                "n_samples",
            ):
                d[key] = int(r[key])
            for key in (
                "gemm_start",
                "gemm_end",
                "comm_start",
                "comm_end",
                "t_events",
                "t_loop",
                "power_w",
                "power_mean_w",
            ):
                d[key] = float(r[key])
            # The simulator predicts energy over time, so the MEAN of the power samples is
            # what it is compared with. The harness prints the median (the earlier
            # campaigns' convention): the two agree within 1 W on millisecond iterations
            # and differ by up to 50 W on a 20-60 ms training step with idle stretches.
            d["power_median_w"], d["power_w"] = d["power_w"], d["power_mean_w"]
            d["held"] = d["clock_req"] == 0 or d["clock_min"] >= d["clock_req"] - 20
            d["window_ok"] = d["n_samples"] >= 60
            out.append(d)
    return out


def grids():
    """Launch grids by 'MxNxK': the 15 shapes of grid12.json and tonight's ncu table."""
    with open(os.path.join(DATA, "grid12.json")) as fh:
        g = {k.replace("_", "x"): v for k, v in json.load(fh).items()}
    path = os.path.join(NEW, "grids.json")
    if os.path.exists(path):
        with open(path) as fh:
            g.update(json.load(fh))
    return g


# ---------------------------------------------------------------------- GEMM ----
def fit_gemm(old, cal, grid):
    """span = host_pre + u / f + waves * (tile_gmac * a + c_tile) * (1 + split_cost) / f;
    an old loop time is a span plus that harness's own constant. Six numbers on 69 loop
    cells (12 shapes x 8 clocks) and the solo spans of `cal`: 0.45% rms, against 1.8%
    without c_tile and split_cost."""
    cells = collections.defaultdict(list)
    for r in old:
        if r["mode"] == "gemm_only" and r["held"]:
            cells[(r["shape"], r["f"])].append(r["ms"])
    pts = [dict(shape=s, f=f, ms=float(np.median(v)), kind="loop") for (s, f), v in cells.items()]
    for r in cal:
        if r["mode"] == "gemm_only" and r["held"] and r["rank"] == 0:
            pts.append(dict(shape=r["shape"], f=r["clock_req"], ms=r["gemm_end"] - r["gemm_start"], kind="span"))
    have_span = any(p["kind"] == "span" for p in pts)

    def model(q):
        a, u, pre, c_old, c_tile, split_cost = np.exp(q)
        out = []
        for p in pts:
            m, n, k = (int(v) for v in p["shape"].split("x"))
            tiles = -(-m // TILE_M) * -(-n // TILE_N)
            g = grid.get(p["shape"]) or tiles
            split = max(1, round(g / tiles))
            per = (TILE_M * TILE_N * k / split / 1e9 * a + c_tile) * (1 + (split_cost if split > 1 else 0.0))
            t = -(-g // 108) * per / p["f"] + u / p["f"]
            out.append(t + (pre if p["kind"] == "span" else pre + c_old))
        return np.array(out)

    ms = np.array([p["ms"] for p in pts])
    q, _ = lm(lambda q: np.log(model(q) / ms), np.log([1000.0, 20.0, 0.02, 0.01, 1.0, 0.05]), iters=400)
    e = np.log(model(q) / ms) * 100
    a, u, pre, c_old, c_tile, split_cost = (float(v) for v in np.exp(q))
    if not have_span:  # the two constants cannot be told apart: give the span the usual share
        pre, c_old = 0.02, pre + c_old - 0.02
    kinds = np.array([p["kind"] for p in pts])
    meta = dict(
        n_loop=int((kinds == "loop").sum()),
        n_span=int((kinds == "span").sum()),
        rms_pct=rms(e),
        rms_loop_pct=rms(e[kinds == "loop"]),
        rms_span_pct=rms(e[kinds == "span"]) if have_span else None,
        worst_pct=float(e[np.abs(e).argmax()]),
        loop_const_ms=c_old,
    )
    return dict(a=a, u=u, host_pre=pre, c_tile=c_tile, split_cost=split_cost), meta


# ---------------------------------------------------------------- all-reduce ----
# ms an iteration of the EARLIER harness's comm-only loop took beyond the collective's
# span: none that the spans can see (with 0.029, tonight's own loop overhead, the old rows
# sit 1.6% under tonight's spans of the same cells; with 0, 0.15%).
LOOP_COMM = 0.0


def comm_cells(old, extra):
    """(CTAs, clock) -> [(GB, span ms, W, loop ms)] of the all-reduce alone, clock held.
    The earlier campaigns recorded the loop time only."""
    cells = collections.defaultdict(list)
    for r in old:
        if r["mode"] == "comm_only" and r["held"]:
            cells[(r["ctas"], r["f"])].append((r["mib"] * 2**20 / 1e9, r["ms"] - LOOP_COMM, r["w"], r["ms"] + 0.02))
    by = collections.defaultdict(list)
    for x in extra:
        if x["window_ok"]:
            by[(x["ctas"], x["run"], x["shape"], x["mib"], x["clock"])].append(x["power_w"])
    for r in extra:
        if r["mode"] == "comm_only" and r["held"] and r["rank"] == 0 and r["world"] == 4:
            w = float(np.mean(by.get((r["ctas"], r["run"], r["shape"], r["mib"], r["clock"])) or [np.nan]))
            cells[(r["ctas"], r["clock_req"])].append(
                (r["mib"] * 2**20 / 1e9, r["comm_end"] - r["comm_start"], w, r["t_loop"])
            )
    return cells


MIN_GB = 8 * 2**20 / 1e9  # the smallest message the latency + bytes / bandwidth line is fitted on


def fit_allreduce(old, extra, pw, min_sizes=3, host_pre=0.05, link_w=0.0, board=1.0):
    """Tables of the all-reduce alone. `board`: the mean gain of the four boards on
    non-rail power. The measured power is the mean of four boards, the power model's
    watts are the reference board's (GPU 0, MoE calibration): the table is stored in
    the reference board's watts, so a board's gain applies to it as to everything else."""
    # 8 MiB and up: below that the time no longer falls on the line (4 MiB takes 0.03-0.05
    # ms longer than the line through 8-512 MiB says)
    cells = {k: [x for x in v if x[0] >= MIN_GB] for k, v in comm_cells(old, extra).items()}
    cells = {k: v for k, v in cells.items() if len({round(x[0], 4) for x in v}) >= min_sizes}
    # the table is a full grid: the clocks at which every CTA count of 2 or more was
    # measured, and then every CTA count (1 included) measured at all of those clocks
    cs, fs = sorted({c for c, _ in cells if c >= 2}), sorted({f for _, f in cells})
    full = [f for f in fs if all((c, f) in cells for c in cs)]
    cs = [c for c in sorted({c for c, _ in cells}) if all((c, f) in cells for f in full)]
    alpha, bw, dyn, res = [], [], [], []
    for f in full:
        ra, rb, rw = [], [], []
        for c in cs:
            v = cells[(c, f)]
            gb, ms = np.array([x[0] for x in v]), np.array([x[1] for x in v])
            A, w = np.vstack([np.ones_like(gb), gb]).T, 1.0 / ms  # relative errors: every size counts alike
            co, *_ = np.linalg.lstsq(A * w[:, None], ms * w, rcond=None)
            res += list((A @ co / ms - 1) * 100)
            ra.append(float(co[0]))
            rb.append(float(1e3 / co[1]))
            # W while the kernel is resident: the loop's mean power, less the part of
            # each iteration (host_pre, and the loop's own overhead) the GPU sat idle
            resident = []
            for _, span, w_loop, loop in v:
                if np.isfinite(w_loop):
                    d = min(1.0, max(0.1, (span - host_pre) / loop))
                    # between two collectives the board is at idle + link_w, not idle
                    gap = pw.idle_at(f) + link_w
                    resident.append((w_loop / board - gap * (1 - d)) / d - pw.idle_at(f) - pw.act_at(f) - link_w)
            rw.append(float(np.median(resident)))
        alpha.append(ra)
        bw.append(rb)
        dyn.append(rw)
    # the knee law, per clock, with one sharpness for all
    c_arr = np.array(cs, float)

    def knee(q, i):
        return NcclAllReduce.knee(c_arr, np.exp(q[0]), np.exp(q[1]), p_all)

    best = None
    for p_try in (2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0):
        p_all, tot, laws = p_try, 0.0, []
        for i in range(len(full)):
            b = np.array(bw[i])
            q, cost = lm(
                lambda q, i=i, b=b: np.log(knee(q, i) / b), [np.log(c_arr[0] / b[0]), np.log(b.max())], iters=200
            )
            tot += cost
            laws.append([float(np.exp(q[0])), float(np.exp(q[1])), p_try])
        if best is None or tot < best[0]:
            best = (tot, laws)
    meta = dict(
        cells=len(cs) * len(full),
        sizes_gb=sorted({round(x[0], 4) for v in cells.values() for x in v}),
        linear_rms_pct=rms(res),
        linear_worst_pct=float(np.abs(res).max()),
        knee_rms_pct=float(np.sqrt(best[0] / (len(cs) * len(full))) * 100),
        knee_p=best[1][0][2],
    )
    return dict(ctas=cs, clocks=full, alpha=alpha, bw=bw, dyn_w=dyn, law=best[1]), meta


# -------------------------------------------------------------------- co-run ----
def fit_corun(cal):
    """How a GEMM changes the collective beside it, from the spans of `cal` (rank 0).

    comm first   the collective loses s / (1 + s) of every ms it shares with GEMM tiles.
                 Fitted where the GEMM ends first; where the GEMM outlasts the collective
                 two shapes (K = 8192, unsplit) cost it 0.3-1.3 ms more, which this does
                 not explain and is reported as it is.
    GEMM first   the collective's span over its solo span, by CTA count (the median over
                 clocks and sizes of cells where it outlasts the GEMM by a margin).
    """
    rows = [r for r in cal if r["rank"] == 0 and r["held"]]
    solo = {
        (r["ctas"], r["clock_req"], r["mib"]): r["comm_end"] - r["comm_start"] for r in rows if r["mode"] == "comm_only"
    }
    solo_g = {(r["clock_req"], r["shape"]): r["gemm_end"] - r["gemm_start"] for r in rows if r["mode"] == "gemm_only"}
    pts, gf, dg = [], collections.defaultdict(list), []
    for r in rows:
        key = (r["ctas"], r["clock_req"], r["mib"])
        if key not in solo:
            continue
        span = r["comm_end"] - r["comm_start"]
        if r["mode"] == "concurrent":
            o = min(r["comm_end"], r["gemm_end"]) - max(r["comm_start"], r["gemm_start"])
            if r["gemm_end"] < r["comm_end"] - 0.05 and o > 0.2:
                pts.append((r["ctas"], r["clock_req"], o, span - solo[key], span))
            g = solo_g.get((r["clock_req"], r["shape"]))
            # the GEMM's own extra when no wave is lost: the launch beside a resident kernel
            if g and r["ctas"] == 2 and r["shape"] != "8192x8192x8192":
                dg.append(r["gemm_end"] - r["gemm_start"] - g)
        elif r["mode"] == "concurrent_gf" and r["comm_end"] > r["gemm_end"] + 0.3 * span:
            gf[(r["ctas"], r["clock_req"])].append(span / solo[key])
    if not pts:
        return {}, 0.0, {}
    c, f, o, extra, span = (np.array(v, float) for v in zip(*pts, strict=True))

    def model(q):
        s = np.exp(q[0]) * (f / 1200.0) ** q[1] * (2.0 / c) ** q[2]
        return o * s / (1 + s)

    q, _ = lm(lambda q: (model(q) - extra) / span * 100, [np.log(0.25), 1.0, 0.8], iters=200)
    e = (model(q) - extra) / span * 100
    cs, fs = sorted({k[0] for k in gf}), sorted({k[1] for k in gf})
    rate = []
    for f_ in fs:  # a cell with no usable span takes the nearest clock's value at that CTA count
        row = []
        for c_ in cs:
            have = [x for x in fs if (c_, x) in gf]
            near = min(have, key=lambda x: abs(x - f_))
            row.append(float(1.0 / np.median(gf[(c_, near)])))
        rate.append(row)
    corun = dict(
        kappa=float(np.exp(q[0])),
        x=float(q[1]),
        y=float(q[2]),
        behind=dict(sms_per_cta=1.0, ctas=cs, clocks=fs, rate=rate),
    )
    meta = dict(
        n=len(pts),
        span_rms_pct=rms(e),
        span_worst_pct=float(np.abs(e).max()),
        gemm_first_cells={f"{k[0]}@{k[1]}": len(v) for k, v in sorted(gf.items())},
    )
    return corun, float(np.median(dg)) if dg else 0.0, meta


# ----------------------------------------------------------------- world size ----
def fit_world(allreduce, rows):
    """Factors on the world-of-4 bandwidth and latency for a world of n GPUs, from the
    all-reduce alone on n GPUs: two sizes per (CTAs, clock) give both, and the median
    over cells is kept. A ring moves 2 (n - 1) / n bytes per GPU for every byte of
    message, which would make the bandwidth factor 1.5 at n = 2 and 1.125 at n = 3."""
    cells = collections.defaultdict(list)
    for r in rows:
        if r["mode"] == "comm_only" and r["rank"] == 0 and r["held"]:
            cells[(r["world"], r["ctas"], r["clock_req"])].append(
                (r["mib"] * 2**20 / 1e9, r["comm_end"] - r["comm_start"])
            )
    out, meta = {}, {}
    for n in sorted({k[0] for k in cells}):
        fb, fa = [], []
        for (w, c, f), v in cells.items():
            if w != n or len(v) < 2:
                continue
            (g0, t0), (g1, t1) = sorted(v)[0], sorted(v)[-1]
            bw = (g1 - g0) / (t1 - t0) * 1e3
            fb.append(bw / allreduce.bandwidth(c, f))
            fa.append((t0 - g0 / bw * 1e3) / allreduce.latency_ms(c, f))
        if fb:
            out[str(n)] = [float(np.median(fb)), float(np.median(fa))]
            meta[str(n)] = dict(
                cells=len(fb), bw_min=float(min(fb)), bw_max=float(max(fb)), ring=n / (2.0 * (n - 1)) / (4 / 6.0)
            )
    return out, meta


# ------------------------------------------------------- per-GPU clocks (hetero) ----
def fit_hetero(k, cal_obj, rows):
    """`mix`: how far a collective among GPUs at different clocks is from running at its
    slowest participant's pace. One number, scored on the all-reduce's span in the
    comm_only cells of `hetero` that have more than one clock."""
    from . import node

    cells = collections.defaultdict(dict)
    for r in rows:
        if r["mode"] == "comm_only":
            cells[(r["ctas"], r["run"])][r["rank"]] = r
    cells = [
        v
        for v in cells.values()
        if len(v) == 4 and all(x["held"] for x in v.values()) and len({x["clock_req"] for x in v.values()}) > 1
    ]
    if not cells:
        return {}

    def score():
        e = []
        for v in cells:
            c = node.cell(
                k,
                cal_obj,
                "comm_only",
                mib=v[0]["mib"],
                ctas=v[0]["ctas"],
                clock=[v[i]["clock_req"] for i in range(4)],
                limiter=False,
            )
            e.append(np.log((c["comm"][1] - c["comm"][0]) / (v[0]["comm_end"] - v[0]["comm_start"])))
        return rms(e) * 100

    best = None
    for x in np.arange(0.0, 0.51, 0.05):
        k.allreduce.mix = float(x)
        sc = score()
        if best is None or sc < best[0]:
            best = (sc, float(x))
    k.allreduce.mix = 0.0
    out = dict(mix=dict(value=best[1], rms_pct=best[0], rms_without_pct=score(), cells=len(cells)))
    k.allreduce.mix = best[1]
    return out


# ------------------------------------------------------------- waiting, and gaps ----
def program_cells(name):
    """Cells of a `mode: program` campaign: (ctas, clock, ops, {rank: row})."""
    rows = span_rows(name, True)
    path = os.path.join(NEW, f"plan_{name}.json")
    if not rows or not os.path.exists(path):
        return []
    with open(path) as fh:
        runs = json.load(fh)["runs"]
    cells = collections.defaultdict(dict)
    for r in rows:
        if r["mode"] == "program":
            cells[(r["ctas"], r["run"])][r["rank"]] = r
    return [
        (c, runs[i].get("clock", 0), runs[i]["ops"], v)
        for (c, i), v in sorted(cells.items())
        if all(x["held"] and x["window_ok"] for x in v.values())
    ]


def fit_idle_states(k, cal_obj, rebuild):
    """Two numbers about a GPU that is in a job but has nothing to run.

    link_w   W it draws above idle between collectives a few ms apart. Scanned on `gaps`
             (a GEMM or a small all-reduce, then 6-12 ms of nothing) and `waiting` (three
             GPUs waiting for a fourth). `rebuild(link_w)` re-derives the all-reduce's own
             power table net of it.
    spin     share of its CTAs' power a resident collective draws while it waits for a
             participant. Scanned on `waiting`, where the same wait is spent with the
             kernel resident and with the host asleep.

    Scored on the mean power of every GPU; the programs are replayed by the engine."""
    from . import node

    gaps, wait = program_cells("gaps"), program_cells("waiting") + program_cells("waiting_swap")
    if not gaps and not wait:
        return {}

    def score(cells):
        """Over the GPUs whose program has no GEMM: the others' error is the GEMM's."""
        e = []
        for ctas, clock, ops, ranks in cells:
            c = node.program_cell(k, cal_obj, ops, ctas=ctas, clock=clock, limiter=False)
            per = ops if isinstance(ops[0], list) else [ops] * 4
            e += [c["power"][i] - ranks[i]["power_w"] for i in range(4) if all(o["op"] != "gemm" for o in per[i])]
        return rms(e) if e else 0.0

    out = {}
    for name, cells, grid in (
        ("link_w", gaps + wait, np.arange(0.0, 26.1, 1.0)),
        ("spin", wait, np.arange(0.0, 0.41, 0.05)),
    ):
        if not cells:
            continue
        first = None
        best = None
        for x in grid:
            if name == "link_w":
                rebuild(float(x))
            setattr(k.allreduce, name, float(x))
            sc = score(cells)
            first = sc if first is None else first
            if best is None or sc < best[0]:
                best = (sc, float(x))
        if name == "link_w":
            rebuild(best[1])
        setattr(k.allreduce, name, best[1])
        out[name] = dict(value=best[1], rms_w=best[0], rms_at_zero_w=first, cells=len(cells))
    return out


# -------------------------------------------------------------------- boards ----
def fit_boards(k, cal_obj, cal):
    """Per GPU: measured W = k_rail * rail + k_other * other, over every cell of
    `cal` where all four GPUs do the same thing at a held clock. `rail` and `other` are
    the simulator's own split of its prediction for that cell (GPU 0's calibration)."""
    from . import node

    cells = collections.defaultdict(dict)
    for r in cal:
        cells[(r["ctas"], r["run"])][r["rank"]] = r
    X, Y = [], [[] for _ in range(4)]
    for (ctas, _), ranks in cells.items():
        r0 = ranks[0]
        if len(ranks) != 4 or not all(x["held"] and x["window_ok"] for x in ranks.values()) or ctas < 2:
            continue
        if r0["mode"] == "concurrent_gf":  # scored, not fitted on: its rules are the least certain
            continue
        c = node.cell(
            k,
            cal_obj,
            r0["mode"],
            shape=f"{r0['m']}x{r0['n']}x{r0['k']}",
            mib=r0["mib"],
            ctas=ctas,
            clock=r0["clock_req"],
            limiter=False,
        )
        rail = c["rail"][0]
        X.append([rail, c["power"][0] - rail])
        for i in range(4):
            Y[i].append(ranks[i]["power_w"])
    X = np.array(X)
    boards, meta = [], []
    for i in range(4):
        # no offset: with nothing running all four boards read the idle table within 3 W
        # (`link_idle`, phase `before`), and an offset fitted on loaded cells moved a
        # waiting board by up to 11 W
        co, *_ = np.linalg.lstsq(X, np.array(Y[i]), rcond=None)
        res = X @ co - np.array(Y[i])
        boards.append([0.0] + [float(v) for v in co])
        meta.append(dict(n=len(X), rms_w=rms(res), rms_identity_w=rms(X[:, 0] + X[:, 1] - np.array(Y[i]))))
    return boards, meta


# ---------------------------------------------------------------------- main ----
def build(use_new=True, verbose=True):
    pw = Calib.load().power
    old = old_rows()
    cal = span_rows("cal", True) if use_new else []
    size = span_rows("commsize", True) if use_new else []
    grid = grids()
    g, gmeta = fit_gemm(old, cal, grid)
    host_g = 0.0915 if cal else 0.09
    corun, corun_ms, cmeta = fit_corun(cal)
    gemm = CublasGemm(
        a=g["a"],
        u=g["u"],
        host_pre=g["host_pre"],
        host=host_g,
        c_tile=g["c_tile"],
        split_cost=g["split_cost"],
        corun_ms=corun_ms,
        grids=grid,
    )
    ar, ameta = fit_allreduce(old, cal + size, pw)
    allreduce = NcclAllReduce(**ar, corun=corun, spin=0.1)
    k = Kernels(
        gemm=gemm,
        allreduce=allreduce,
        meta=dict(gemm=gmeta, allreduce=ameta, corun=cmeta, sources=dict(old=OLD, cal=len(cal), commsize=len(size))),
    )
    worlds = (span_rows("world2", True) + span_rows("world3", True)) if use_new else []
    if worlds:
        allreduce.world_scale, k.meta["world"] = fit_world(allreduce, worlds)
    if cal:
        k.boards, k.meta["boards"] = fit_boards(k, Calib.load(), cal)
    if use_new:

        def rebuild(link_w):
            """The all-reduce's own power, net of `link_w`, in the reference board's watts."""
            board = float(np.mean([b[2] for b in k.boards]))
            t, _ = fit_allreduce(old, cal + size, pw, link_w=link_w, board=board)
            allreduce.dyn_w = t["dyn_w"]
            allreduce.__post_init__()

        # The link power, the table net of it and the boards' gains depend on one
        # another: three rounds settle them (the gains move by under 0.2% in the third).
        for _ in range(3):
            k.meta["idle_states"] = fit_idle_states(k, Calib.load(), rebuild)
            if cal:
                k.boards, k.meta["boards"] = fit_boards(k, Calib.load(), cal)
        rebuild(allreduce.link_w)
    het = span_rows("hetero", True) if use_new else []
    if het:
        k.meta["hetero"] = fit_hetero(k, Calib.load(), het)
    if verbose:
        print(
            f"GEMM: a = {g['a']:.1f} SM*ms*MHz/GMAC + {g['c_tile']:.2f} ms*MHz per tile, "
            f"split-K +{g['split_cost'] * 100:.1f}%, "
            f"launch {g['u']:.2f}/f ms, host share {g['host_pre'] * 1e3:.1f} us; "
            f"{gmeta['n_loop']} loop cells rms {gmeta['rms_loop_pct']:.2f}%"
            + (f", {gmeta['n_span']} spans rms {gmeta['rms_span_pct']:.2f}%" if gmeta["n_span"] else "")
            + f", worst {gmeta['worst_pct']:+.1f}%"
        )
        print(
            f"all-reduce: {ameta['cells']} cells (CTAs {ar['ctas']}, clocks {ar['clocks']}); latency + bytes/bandwidth "
            f"rms {ameta['linear_rms_pct']:.2f}% worst {ameta['linear_worst_pct']:.2f}%; "
            f"knee law p = {ameta['knee_p']}, "
            f"rms {ameta['knee_rms_pct']:.2f}%"
        )
        if corun:
            pk = corun["behind"]
            print(
                f"co-run: collective beside GEMM tiles runs at 1 / (1 + s), "
                f"s = {corun['kappa']:.3f} * (f/1200)^{corun['x']:.2f}"
                f" * (2/c)^{corun['y']:.2f} ({cmeta['n']} spans, rms {cmeta['span_rms_pct']:.2f}% of the span); "
                f"GEMM launch beside a resident kernel +{corun_ms * 1e3:.0f} us"
            )
            print(
                f"        launched behind a GEMM it costs the GEMM {pk['sms_per_cta']:.0f} SM per CTA "
                f"and runs at x rate, "
                f"rows = MHz, columns = CTAs {pk['ctas']}"
            )
            for f_, row in zip(pk["clocks"], pk["rate"], strict=True):
                print(f"          {f_:5d}  " + "  ".join(f"{r:.3f}" for r in row))
        for n, (fb, fa) in allreduce.world_scale.items():
            m = k.meta["world"][n]
            print(
                f"world of {n}: bandwidth x {fb:.3f} (cells {m['bw_min']:.3f}-{m['bw_max']:.3f}; "
                f"a ring's byte count says "
                f"{m['ring']:.3f}), latency x {fa:.3f}   ({m['cells']} cells)"
            )
        for i, (b, m) in enumerate(zip(k.boards, k.meta.get("boards", []), strict=False)):
            print(
                f"board {i}: W = {b[0]:+.1f} + {b[1]:.3f} x rail + {b[2]:.3f} x other   "
                f"({m['n']} cells, rms {m['rms_w']:.1f} W; "
                f"with no correction {m['rms_identity_w']:.1f} W)"
            )
        for name, m in k.meta.get("idle_states", {}).items():
            print(
                f"in a job with nothing to run, {name} = {m['value']:.2f}: "
                f"rms {m['rms_w']:.1f} W on {m['cells']} cells "
                f"(at 0: {m['rms_at_zero_w']:.1f} W)"
            )
        for name, m in k.meta.get("hetero", {}).items():
            print(
                f"per-GPU clocks, {name} = {m['value']:.2f}: rms {m['rms_pct']:.2f}% on {m['cells']} cells "
                f"(without it {m['rms_without_pct']:.2f}%)"
            )
    return k


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-new", action="store_true", help="fit on the earlier campaigns alone")
    ap.add_argument("--out", default=DEFAULT)
    a = ap.parse_args()
    k = build(use_new=not a.no_new)
    print("wrote", os.path.relpath(k.save(a.out)))


if __name__ == "__main__":
    main()
