"""Score the node engine (`node.py`) on the GEMM + all-reduce overlap: every measured
run that exists, each compared with a prediction made from the kernel profiles alone.

  python3 -m sim.validate_overlap                    # the earlier campaigns + tonight's
  python3 -m sim.validate_overlap --only spans       # one group
  python3 -m sim.validate_overlap --kernels x.json   # another calibration

GROUPS
  solo      a GEMM alone and an all-reduce alone: time and power              (profiles)
  overlap   the overlapped runs of ../overlap_exp, clock held: iteration time and
            mean power, one clock for all four GPUs                           (co-run rules)
  serial    the same cells, GEMM then all-reduce on one stream                (composition)
  wall      overlapped runs the board throttled at 1410 or 1305 MHz: the clock it
            settled at, and the time                                          (limiter)
  auto      overlapped runs with no clock lock: clock, time, power            (governor)
  spans     tonight's per-kernel spans: when the GEMM and the collective each start
            and end inside the iteration, both launch orders, per-GPU power

`fit` says whether a number was fitted on the rows it is scored on. Every table is the
simulator against a measurement; none is the simulator against itself.
"""

import argparse
import collections
import copy
import os

import numpy as np

from . import node
from .calib import Calib
from .fit_kernels import OLD_AUTO, old_rows, span_rows
from .kernels import DEFAULT, Kernels


def pct(a, b):
    return (a / b - 1.0) * 100.0


def summary(e):
    e = np.asarray(e, float)
    if not len(e):
        return dict(n=0)
    a = np.abs(e)
    return dict(
        n=len(e),
        median=float(np.median(a)),
        p90=float(np.percentile(a, 90)),
        max=float(a.max()),
        bias=float(e.mean()),
        rms=float(np.sqrt((e**2).mean())),
    )


def line(label, e, unit="%"):
    s = summary(e)
    if not s["n"]:
        return f"  {label:44s} (no rows)"
    return (
        f"  {label:44s} n={s['n']:5d}  median {s['median']:5.2f}{unit}  p90 {s['p90']:5.2f}{unit}  "
        f"max {s['max']:6.2f}{unit}  bias {s['bias']:+6.2f}{unit}  rms {s['rms']:5.2f}{unit}"
    )


class Scorer:
    def __init__(self, kernels, cal):
        self.k, self.cal, self.cache = kernels, cal, {}
        self.old_loop = {}  # ms the EARLIER harness's loop took beyond the events, by mode
        self.uniform = copy.deepcopy(kernels)  # the earlier campaigns report the node's mean power
        b = np.mean(np.asarray(kernels.boards, float), axis=0)
        self.uniform.boards = [list(b)] * 4

    def cell(self, mode, per_gpu=False, **kw):
        key = (mode, per_gpu, tuple(sorted((a, str(b)) for a, b in kw.items())))
        if key not in self.cache:
            self.cache[key] = node.cell(self.k if per_gpu else self.uniform, self.cal, mode, **kw)
        return self.cache[key]

    def loop_ms(self, old, mode, c):
        """An earlier campaign recorded one number per cell, its loop's time per
        iteration: the simulated events plus what that harness's own loop cost the host.
        That cost is not chip physics and was never measured, so it is one constant per
        mode, the median of (measured - simulated events) over that mode's held rows."""
        if mode not in self.old_loop:
            d = []
            for r in old:
                if r["mode"] == mode and r["held"] and r["ctas"] >= 2:
                    kw = (
                        dict(shape=r["shape"], clock=r["f"])
                        if mode == "gemm_only"
                        else dict(mib=r["mib"], ctas=r["ctas"], clock=r["f"])
                        if mode == "comm_only"
                        else dict(shape=r["shape"], mib=r["mib"], ctas=r["ctas"], clock=r["f"])
                    )
                    d.append(r["ms"] - (self.cell(mode, **kw)["t_events"] - node.Host().gate))
            self.old_loop[mode] = float(np.median(d)) if d else 0.0
        return c["t_events"] - node.Host().gate + self.old_loop[mode]

    # ------------------------------------------------------------------ solo ----
    def solo(self, old):
        print("\nSOLO: one kernel alone, clock held (fit: yes -- these rows fitted the profiles)")
        et, ep = collections.defaultdict(list), collections.defaultdict(list)
        for r in old:
            if not r["held"] or r["mode"] not in ("gemm_only", "comm_only"):
                continue
            if r["mode"] == "comm_only" and r["ctas"] < 2:
                continue
            kw = (
                dict(shape=r["shape"], clock=r["f"])
                if r["mode"] == "gemm_only"
                else dict(mib=r["mib"], ctas=r["ctas"], clock=r["f"])
            )
            c = self.cell(r["mode"], **kw)
            et[r["mode"]].append(pct(self.loop_ms(old, r["mode"], c), r["ms"]))
            ep[r["mode"]].append(pct(np.mean(c["power"]), r["w"]))
        for m in ("gemm_only", "comm_only"):
            print(line(f"{m} time", et[m]))
            print(line(f"{m} power (node mean)", ep[m]))
        print(
            "  the earlier harness's loop, ms beyond the events (one constant per mode): "
            + ", ".join(f"{m} {v:+.3f}" for m, v in self.old_loop.items())
        )
        return et, ep

    # --------------------------------------------------------------- overlap ----
    def overlap(self, old, mode="concurrent", held=True, label=None):
        rows = [r for r in old if r["mode"] == mode and r["held"] == held and r["ctas"] >= 2]
        out = []
        for r in rows:
            c = self.cell(mode, shape=r["shape"], mib=r["mib"], ctas=r["ctas"], clock=r["f"])
            t = self.loop_ms(old, mode, c)
            out.append(
                dict(
                    r,
                    t_sim=t,
                    w_sim=float(np.mean(c["power"])),
                    clk_sim=min(c["clock_min"]),
                    clk_mean_sim=float(np.mean(c["clock_mean"])),
                    limited=c["limited"],
                    et=pct(t, r["ms"]),
                    ep=pct(float(np.mean(c["power"])), r["w"]),
                )
            )
        return out

    def print_overlap(self, out, title, by=("ctas", "f", "mib")):
        print(f"\n{title}")
        if out:
            print(f"  (the earlier harness's loop: {self.old_loop.get(out[0]['mode'], 0.0):+.3f} ms beyond the events)")
        print(line("iteration time", [o["et"] for o in out]))
        print(line("mean power (node)", [o["ep"] for o in out]))
        for key in by:
            vals = sorted({o[key] for o in out})
            print(
                f"  by {key:5s} median |time error| %: "
                + "  ".join(f"{v}: {np.median(np.abs([o['et'] for o in out if o[key] == v])):.1f}" for v in vals)
            )
            print(
                f"  {'':8s} bias of time %:          "
                + "  ".join(f"{v}: {np.mean([o['et'] for o in out if o[key] == v]):+.1f}" for v in vals)
            )
            print(
                f"  {'':8s} median |power error| %:  "
                + "  ".join(f"{v}: {np.median(np.abs([o['ep'] for o in out if o[key] == v])):.1f}" for v in vals)
            )

    def wall(self, old):
        out = self.overlap(old, held=False)
        print("\nWALL: overlapped runs the board throttled below the clock asked for (fit: no)")
        if not out:
            print("  (no rows)")
            return out
        print(line("iteration time", [o["et"] for o in out]))
        print(line("mean power (node)", [o["ep"] for o in out]))
        print(line("lowest clock, simulated - measured", [o["clk_sim"] - o["clk"] for o in out], " MHz"))
        print(f"  simulator also throttled: {sum(o['limited'] for o in out)} of {len(out)}")
        return out

    def auto(self):
        rows = [r for r in old_rows(OLD_AUTO) if r["mode"] == "concurrent" and r["ctas"] >= 2]
        out = []
        for r in rows:
            c = self.cell("concurrent", shape=r["shape"], mib=r["mib"], ctas=r["ctas"], clock=0)
            t = c["t_events"] - node.Host().gate + self.old_loop.get("concurrent", 0.021)
            out.append(
                dict(
                    r,
                    et=pct(t, r["ms"]),
                    ep=pct(float(np.mean(c["power"])), r["w"]),
                    dclk=min(c["clock_min"]) - r["clk"],
                    limited=c["limited"],
                )
            )
        print("\nAUTO: overlapped runs with no clock lock, the stock governor (fit: no)")
        print(line("iteration time", [o["et"] for o in out]))
        print(line("mean power (node)", [o["ep"] for o in out]))
        print(line("lowest clock, simulated - measured", [o["dclk"] for o in out], " MHz"))
        thr = [o for o in out if o["clk"] < 1400]
        print(
            f"  measured below 1400 MHz: {len(thr)} of {len(out)}; "
            f"simulator throttled {sum(o['limited'] for o in thr)} "
            f"of those and {sum(o['limited'] for o in out if o['clk'] >= 1400)} of the others"
        )
        return out

    # ----------------------------------------------------------------- spans ----
    def spans(self, name, modes=("concurrent", "concurrent_gf", "serial", "gemm_only", "comm_only"), quiet=False):
        rows = [r for r in span_rows(name) if r["ctas"] >= 2]
        cells = collections.defaultdict(dict)
        for r in rows:
            cells[(r["ctas"], r["run"])][r["rank"]] = r
        out = []
        for (ctas, _), ranks in sorted(cells.items()):
            r0 = ranks[0]
            if r0["mode"] not in modes or not all(x["held"] for x in ranks.values()):
                continue
            world = r0["world"]
            clock = [ranks[i]["clock_req"] for i in range(world)]
            shape = [f"{ranks[i]['m']}x{ranks[i]['n']}x{ranks[i]['k']}" for i in range(world)]
            uni = len(set(clock)) == 1 and len(set(shape)) == 1
            c = self.cell(
                r0["mode"],
                per_gpu=True,
                shape=shape[0] if uni else shape,
                mib=r0["mib"],
                ctas=ctas,
                clock=clock[0] if uni else clock,
                world=world,
            )
            for i in range(world):
                m, s = ranks[i], c["spans"][i]
                d = dict(
                    name=name,
                    mode=m["mode"],
                    ctas=ctas,
                    f=m["clock_req"],
                    mib=m["mib"],
                    shape=shape[i],
                    rank=i,
                    uniform=uni,
                    world=world,
                    clock=r0["clock"],
                    cell_shape=r0["shape"],
                )
                d["t"] = (m["t_events"], s["t"])
                # On a stream that is busy (serial), the measured start is when the stream
                # reached the kernel and the simulated one when the host issued it: only
                # the ends compare, so a serial "span" runs from the gate.
                z = 0.0 if m["mode"] == "serial" else 1.0
                d["gemm"] = (m["gemm_end"] - z * m["gemm_start"], s["gemm"][1] - z * s["gemm"][0])
                d["comm"] = (m["comm_end"] - z * m["comm_start"], s["comm"][1] - z * s["comm"][0])
                d["gemm_end"] = (m["gemm_end"], s["gemm"][1])
                d["comm_end"] = (m["comm_end"], s["comm"][1])
                d["power"] = (m["power_w"], c["power"][i]) if m["window_ok"] else None
                d["loop"] = (m["t_loop"], c["t_loop"])
                out.append(d)
        if not quiet:
            self.print_spans(out, name)
        return out

    @staticmethod
    def print_spans(out, name):
        print(f"\nSPANS ({name}): inside the iteration, rank 0 for times, every GPU for power")
        for mode in sorted({o["mode"] for o in out}):
            v = [o for o in out if o["mode"] == mode]
            v0 = [o for o in v if o["rank"] == 0]
            print(f" {mode}")
            print(line("iteration time (events)", [pct(o["t"][1], o["t"][0]) for o in v0]))
            if mode != "comm_only":
                print(line("GEMM span", [pct(o["gemm"][1], o["gemm"][0]) for o in v0]))
            if mode != "gemm_only":
                print(line("all-reduce span", [pct(o["comm"][1], o["comm"][0]) for o in v0]))
            print(line("power, each GPU", [pct(o["power"][1], o["power"][0]) for o in v if o["power"]]))


def program_rows(name):
    """A `mode: program` campaign: one dict per cell with its ops, its rows by rank and
    its measured per-kernel spans by rank."""
    import csv
    import json

    from .fit_kernels import NEW

    plan_path, ops_path = os.path.join(NEW, f"plan_{name}.json"), os.path.join(NEW, f"{name}.csv.ops.csv")
    rows = span_rows(name)
    if not rows or not os.path.exists(ops_path):
        return []
    with open(plan_path) as fh:
        runs = json.load(fh)["runs"]
    with open(ops_path) as fh:
        op_rows = list(csv.DictReader(fh))
    out = []
    for key in sorted({(r["ctas"], r["run"]) for r in rows if r["mode"] == "program"}):
        main = {r["rank"]: r for r in rows if (r["ctas"], r["run"]) == key}
        ops = collections.defaultdict(dict)
        for o in op_rows:
            if (int(o["ctas"]), int(o["run"])) == key:
                ops[int(o["rank"])][o["tag"]] = (float(o["start"]), float(o["end"]))
        run = runs[key[1]]
        out.append(
            dict(
                ctas=key[0],
                run=key[1],
                name=run.get("name", "program"),
                clock=run.get("clock", 0),
                ops=run["ops"],
                main=main,
                spans=dict(ops),
            )
        )
    return out


def score_programs(scorer, name, quiet=False):
    """Training-step programs: step time, when every kernel ends, per-GPU power."""
    out = []
    for c in program_rows(name):
        world = len(c["main"])
        held = all(r["held"] for r in c["main"].values())
        sim = node.program_cell(scorer.k, scorer.cal, c["ops"], ctas=c["ctas"], clock=c["clock"], world=world)
        for i in range(world):
            m = c["main"][i]
            ends = [(c["spans"][i][t][1], sim["ends"][i][t]) for t in c["spans"][i]]
            out.append(
                dict(
                    name=c["name"],
                    ctas=c["ctas"],
                    clock=c["clock"],
                    rank=i,
                    held=held,
                    t=(m["t_events"], sim["t_events"][i]),
                    loop=(m["t_loop"], sim["t_loop"]),
                    power=(m["power_w"], sim["power"][i]) if m["window_ok"] else None,
                    clk=(m["clock_min"], sim["clock_min"][i]),
                    ends=ends,
                    limited=sim["limited"],
                )
            )
    if not quiet and out:
        print(f"\nPROGRAMS ({name}): a training step as kernels on streams")
        for label, sel in (
            ("clock held", [o for o in out if o["held"] and o["clock"] != 0]),
            ("at the wall or not locked", [o for o in out if not o["held"] or o["clock"] == 0]),
        ):
            if not sel:
                continue
            print(f" {label}")
            print(line("step time (events), each GPU", [pct(o["t"][1], o["t"][0]) for o in sel]))
            print(line("step time (loop), rank 0", [pct(o["loop"][1], o["loop"][0]) for o in sel if o["rank"] == 0]))
            print(
                line(
                    "end of each kernel, % of the step", [(b - a) / o["t"][0] * 100 for o in sel for a, b in o["ends"]]
                )
            )
            print(line("power, each GPU", [pct(o["power"][1], o["power"][0]) for o in sel if o["power"]]))
            print(line("lowest clock, simulated - measured", [o["clk"][1] - o["clk"][0] for o in sel], " MHz"))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kernels", default=DEFAULT)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    s = Scorer(Kernels.load(a.kernels), Calib.load())
    want = lambda g: not a.only or g in a.only.split(",")
    old = old_rows()
    print(f"kernel profiles: {os.path.relpath(a.kernels)}")
    if want("solo"):
        s.solo(old)
    if want("overlap"):
        s.print_overlap(
            s.overlap(old),
            "OVERLAP: GEMM beside the all-reduce, clock held (fit: the co-run rules, on tonight's spans)",
        )
    if want("serial"):
        s.print_overlap(s.overlap(old, mode="serial"), "SERIAL: GEMM, then all-reduce, one stream (fit: no)", by=())
    if want("wall"):
        s.wall(old)
    if want("auto"):
        s.auto()
    if want("spans"):
        for name in ("cal", "world2", "world3", "hetero", "unbalanced"):
            if span_rows(name):
                s.spans(name)
    if want("programs"):
        score_programs(s, "train")


if __name__ == "__main__":
    main()
