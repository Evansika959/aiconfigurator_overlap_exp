"""The registered test of the node engine: predictions written before the runs exist.

  python3 -m sim.heldout_overlap predict      # freeze: predictions + an archive of sim/
  python3 -m sim.heldout_overlap score        # after ../overlap_exp has run the plans
  python3 -m sim.heldout_overlap score --second-shot   # with today's sim/, beside the first

`predict` refuses to run once any blind measurement file exists, and `score` reads the
predictions from disk: it never recomputes them. A later simulator can be scored with
--second-shot, which prints its numbers BESIDE the registered ones, not instead of them.
The plans, what is unseen in them and the pass criteria are in ../OVERLAP_HELDOUT.md.

Plans (../overlap_exp/data/sim_overlap/plan_<name>.json):
  blind        one clock for all four GPUs, the same GEMM on each
  blind_auto   the same cells, clock not locked
  blind_multi  a different clock per GPU; an unbalanced step
  blind_train  training steps as programs
"""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import time

import numpy as np

from . import node
from .calib import Calib
from .fit_kernels import NEW, span_rows
from .kernels import DEFAULT, Kernels
from .validate_overlap import line, pct, program_rows

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "overlap_heldout")
PLANS = ("blind", "blind_auto", "blind_multi", "blind_train")
FIELDS = [
    "plan",
    "ctas",
    "run",
    "mode",
    "name",
    "clock",
    "mib",
    "shape",
    "rank",
    "t_events",
    "t_loop",
    "gemm_start",
    "gemm_end",
    "comm_start",
    "comm_end",
    "power_w",
    "clock_min",
    "clock_mean",
    "limited",
    "ends",
]


def spec(v):
    return "/".join(str(x) for x in v) if isinstance(v, (list, tuple)) else str(v)


def sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def predict_plan(k, cal, name):
    """One dict per (cell, rank) of a plan, as the simulator sees it."""
    with open(os.path.join(NEW, f"plan_{name}.json")) as fh:
        plan = json.load(fh)
    world, rows = plan["world"], []
    for ci, ctas in enumerate(plan["ctas"]):
        for i, r in enumerate(plan["runs"]):
            if r.get("once") and ci:
                continue
            base = dict(
                plan=name,
                ctas=ctas,
                run=i,
                mode=r["mode"],
                name=r.get("name", ""),
                clock=spec(r.get("clock", 0)),
                mib=r.get("mib", 0),
                shape=spec(r.get("shape", "0x0x0")),
            )
            if r["mode"] == "program":
                c = node.program_cell(k, cal, r["ops"], ctas=ctas, clock=r.get("clock", 0), world=world)
                for g in range(world):
                    rows.append(
                        dict(
                            base,
                            rank=g,
                            t_events=c["t_events"][g],
                            t_loop=c["t_loop"],
                            power_w=c["power"][g],
                            clock_min=c["clock_min"][g],
                            clock_mean=c["clock_mean"][g],
                            limited=c["limited"],
                            ends=json.dumps({t: round(v, 5) for t, v in c["ends"][g].items()}),
                        )
                    )
                continue
            c = node.cell(
                k,
                cal,
                r["mode"],
                shape=r.get("shape", "0x0x0"),
                mib=r.get("mib", 0),
                ctas=ctas,
                clock=r.get("clock", 0),
                world=world,
            )
            for g in range(world):
                s = c["spans"][g]
                rows.append(
                    dict(
                        base,
                        rank=g,
                        t_events=s["t"],
                        t_loop=c["t_loop"],
                        gemm_start=s["gemm"][0],
                        gemm_end=s["gemm"][1],
                        comm_start=s["comm"][0],
                        comm_end=s["comm"][1],
                        power_w=c["power"][g],
                        clock_min=c["clock_min"][g],
                        clock_mean=c["clock_mean"][g],
                        limited=c["limited"],
                    )
                )
    return rows


def write(rows, path):
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=FIELDS)
        wr.writeheader()
        for r in rows:
            wr.writerow({a: (round(b, 5) if isinstance(b, float) else b) for a, b in r.items()})


def predict(kernels_path, tag="first_shot", force=False):
    os.makedirs(OUT, exist_ok=True)
    measured = [n for n in PLANS if os.path.exists(os.path.join(NEW, f"{n}.csv"))]
    if tag == "first_shot" and measured and not force:
        raise SystemExit(f"measurements exist for {measured}: a first shot can no longer be registered")
    k, cal = Kernels.load(kernels_path), Calib.load()
    stamp = dict(
        written_utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        tag=tag,
        kernels=os.path.relpath(kernels_path, HERE),
        kernels_sha256=sha(kernels_path),
        calib_sha256=sha(os.path.join(HERE, "calib", "a100_qwen.json")),
        measured_already=measured,
        plans={},
    )
    for name in PLANS:
        t0 = time.time()
        rows = predict_plan(k, cal, name)
        path = os.path.join(OUT, f"pred_{tag}_{name}.csv")
        write(rows, path)
        stamp["plans"][name] = dict(
            rows=len(rows), sha256=sha(path), plan_sha256=sha(os.path.join(NEW, f"plan_{name}.json"))
        )
        print(f"{name:12s} {len(rows):5d} rows  {time.time() - t0:6.1f} s  -> {os.path.relpath(path)}")
    if tag == "first_shot":
        tar = os.path.join(OUT, "frozen_sim.tar.gz")
        subprocess.run(
            ["tar", "czf", tar, "-C", os.path.join(HERE, ".."), "--exclude=__pycache__", "--exclude=sim/out", "sim"],
            check=True,
        )
        stamp["frozen_sim_sha256"] = sha(tar)
    with open(os.path.join(OUT, f"stamp_{tag}.json"), "w") as fh:
        json.dump(stamp, fh, indent=1)
    print(f"stamp -> {os.path.relpath(os.path.join(OUT, f'stamp_{tag}.json'))}")


# --------------------------------------------------------------------- score ----
def load_pred(tag, name):
    path = os.path.join(OUT, f"pred_{tag}_{name}.csv")
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        return {(int(r["ctas"]), int(r["run"]), int(r["rank"])): r for r in csv.DictReader(fh)}


def score_spans(tag, name, out):
    pred, rows = load_pred(tag, name), span_rows(name)
    if not pred or not rows:
        return
    cells = {}
    for r in rows:
        cells.setdefault((r["ctas"], r["run"]), {})[r["rank"]] = r
    recs = []
    for (ctas, run), ranks in cells.items():
        held = all(x["held"] for x in ranks.values())
        for g, m in ranks.items():
            p = pred.get((ctas, run, g))
            if p is None or m["mode"] == "program":
                continue
            f = lambda key: float(p[key] or 0)
            z = 0.0 if m["mode"] == "serial" else 1.0
            recs.append(
                dict(
                    mode=m["mode"],
                    held=held,
                    rank=g,
                    locked=m["clock_req"] > 0,
                    t=(m["t_events"], f("t_events")),
                    # serial: ends only (see validate_overlap.Scorer.spans)
                    gemm=(m["gemm_end"] - z * m["gemm_start"], f("gemm_end") - z * f("gemm_start")),
                    comm=(m["comm_end"] - z * m["comm_start"], f("comm_end") - z * f("comm_start")),
                    power=(m["power_w"], f("power_w")) if m["window_ok"] else None,
                    clk=(m["clock_med"], f("clock_mean")),
                    limited=p["limited"] == "True",
                    below=m["clock_req"] > 0 and m["clock_min"] < m["clock_req"] - 20,
                )
            )
    print(f"\n{name} [{tag}]: {len(cells)} cells measured")
    for label, sel in (
        ("clock held", [r for r in recs if r["held"] and r["locked"]]),
        ("throttled below the clock asked for", [r for r in recs if not r["held"]]),
        ("clock not locked", [r for r in recs if not r["locked"]]),
    ):
        if not sel:
            continue
        print(f" {label}")
        for mode in ("gemm_only", "comm_only", "concurrent", "serial"):
            v = [r for r in sel if r["mode"] == mode]
            v0 = [r for r in v if r["rank"] == 0] if name != "blind_multi" else v
            if not v:
                continue
            e_t = [pct(r["t"][1], r["t"][0]) for r in v0]
            print(line(f"{mode}: time", e_t))
            out[(name, label, mode, "time")] = e_t
            if mode in ("concurrent", "serial"):
                e_g = [pct(r["gemm"][1], r["gemm"][0]) for r in v0]
                e_c = [pct(r["comm"][1], r["comm"][0]) for r in v0]
                print(line(f"{mode}: GEMM span", e_g))
                print(line(f"{mode}: all-reduce span", e_c))
                out[(name, label, mode, "gemm")], out[(name, label, mode, "comm")] = e_g, e_c
            e_p = [pct(r["power"][1], r["power"][0]) for r in v if r["power"]]
            print(line(f"{mode}: power, each GPU", e_p))
            out[(name, label, mode, "power")] = e_p
        if label != "clock held":
            d = [r["clk"][1] - r["clk"][0] for r in sel]
            print(line("median clock, simulated - measured", d, " MHz"))
            out[(name, label, "all", "clock")] = d
            thr = [r for r in sel if r["rank"] == 0]
            print(f"  simulator throttled in {sum(r['limited'] for r in thr)} of {len(thr)} cells")


def score_programs(tag, name, out):
    pred = load_pred(tag, name)
    cells = program_rows(name)
    if not pred or not cells:
        return
    recs = []
    for c in cells:
        held = all(r["held"] for r in c["main"].values())
        for g, m in c["main"].items():
            p = pred.get((c["ctas"], c["run"], g))
            if p is None:
                continue
            ends = json.loads(p["ends"])
            recs.append(
                dict(
                    held=held,
                    locked=m["clock_req"] > 0,
                    rank=g,
                    t=(m["t_events"], float(p["t_events"])),
                    power=(m["power_w"], float(p["power_w"])) if m["window_ok"] else None,
                    ends=[(c["spans"][g][t][1], ends[t]) for t in c["spans"][g]],
                    name=c["name"],
                )
            )
    print(f"\n{name} [{tag}]: {len(cells)} cells measured")
    for label, sel in (
        ("clock held", [r for r in recs if r["held"] and r["locked"]]),
        ("at the wall or not locked", [r for r in recs if not (r["held"] and r["locked"])]),
    ):
        if not sel:
            continue
        e_t = [pct(r["t"][1], r["t"][0]) for r in sel]
        e_e = [(b - a) / r["t"][0] * 100 for r in sel for a, b in r["ends"]]
        e_p = [pct(r["power"][1], r["power"][0]) for r in sel if r["power"]]
        print(f" {label}")
        print(line("step time, each GPU", e_t))
        print(line("end of each kernel, % of the step", e_e))
        print(line("power, each GPU", e_p))
        out[(name, label, "program", "time")], out[(name, label, "program", "power")] = e_t, e_p
        out[(name, label, "program", "ends")] = e_e


CRITERIA = [  # (plan, selection, mode, quantity, statistic, limit %)
    ("blind", "clock held", "gemm_only", "time", "median", 2.0),
    ("blind", "clock held", "gemm_only", "time", "p90", 5.0),
    ("blind", "clock held", "comm_only", "time", "median", 3.0),
    ("blind", "clock held", "comm_only", "time", "p90", 6.0),
    ("blind", "clock held", "concurrent", "time", "median", 2.5),
    ("blind", "clock held", "concurrent", "time", "p90", 7.0),
    ("blind", "clock held", "concurrent", "time", "max", 15.0),
    ("blind", "clock held", "concurrent", "gemm", "median", 2.5),
    ("blind", "clock held", "concurrent", "comm", "median", 3.5),
    ("blind", "clock held", "concurrent", "power", "median", 4.0),
    ("blind", "clock held", "concurrent", "power", "p90", 8.0),
    ("blind_auto", "clock not locked", "concurrent", "time", "median", 3.0),
    ("blind_auto", "clock not locked", "concurrent", "power", "median", 5.0),
    ("blind_multi", "clock held", "comm_only", "time", "median", 4.0),
    ("blind_multi", "clock held", "concurrent", "time", "median", 4.0),
    ("blind_multi", "clock held", "serial", "time", "median", 4.0),
    ("blind_multi", "clock held", "concurrent", "power", "median", 5.0),
    ("blind_train", "clock held", "program", "time", "median", 3.0),
    ("blind_train", "clock held", "program", "power", "median", 5.0),
]


def score(tag):
    out = {}
    for name in ("blind", "blind_auto", "blind_multi"):
        score_spans(tag, name, out)
    score_programs(tag, "blind_train", out)
    print(f"\nREGISTERED CRITERIA [{tag}]")
    n_pass = n_scored = 0
    for plan, sel, mode, what, stat, limit in CRITERIA:
        e = out.get((plan, sel, mode, what))
        if not e:
            print(f"  {plan:12s} {sel:18s} {mode:10s} {what:6s} {stat:6s} <= {limit:5.1f}%   not measured yet")
            continue
        a = np.abs(e)
        v = dict(median=np.median(a), p90=np.percentile(a, 90), max=a.max())[stat]
        n_scored += 1
        n_pass += v <= limit
        print(
            f"  {plan:12s} {sel:18s} {mode:10s} {what:6s} {stat:6s} <= {limit:5.1f}%   {v:6.2f}%   "
            f"{'pass' if v <= limit else 'FAIL'}   (n={len(e)})"
        )
    print(f"  {n_pass} of {n_scored} pass")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("predict", "score"))
    ap.add_argument("--kernels", default=DEFAULT)
    ap.add_argument("--second-shot", action="store_true")
    a = ap.parse_args()
    if a.what == "predict":
        predict(a.kernels, "second_shot" if a.second_shot else "first_shot")
    else:
        score("first_shot")
        if a.second_shot:
            score("second_shot")


if __name__ == "__main__":
    main()
