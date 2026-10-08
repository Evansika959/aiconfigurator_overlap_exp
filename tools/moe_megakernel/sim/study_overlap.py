"""What a V/f domain for the collective's SMs would be worth in the GEMM + all-reduce
overlap. A WHAT-IF: the A100 has one clock per GPU.

  python3 -m sim.study_overlap [--jobs 40]        # -> sim/out/study_overlap.csv, and a table

The overlapped iteration of ../overlap_exp (collective launched first, c CTAs, the GEMM
on the other 108 - 2c SMs), for six GEMM shapes, four message sizes and seven CTA
counts, under three policies:

  A      no clock lock: the stock governor (top clock, the board's limiter)
  one    the best single clock: least energy with the iteration at most (1 + budget)
         times A's
  two    the same with two clocks: the collective's SM pairs at one, the GEMM's SMs at
         the other

`two - one` is what the second domain adds; `one` alone is what a clock lock already
gives. What the chip is assumed to be, none of which a one-clock A100 can confirm:

  - the domain is exactly the SM pairs the collective holds (a real one would be whole
    GPCs of 16 SMs);
  - the shared logic (L2, memory, links) follows the faster of the two clocks, so the
    collective's bandwidth ceiling does too, while its per-CTA cost follows its own SMs
    (`NcclAllReduce.bandwidth_split`);
  - idle power is split by SM count, a quarter of its clock-dependent part belonging to
    the shared logic (`PowerParams.static`); link power does not change;
  - the limiter moves the GEMM's clock only.
"""

import argparse
import csv
import itertools
import os
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from . import node
from .calib import Calib
from .kernels import Kernels

HERE = os.path.dirname(os.path.abspath(__file__))
SHAPES = ["1024x4096x4096", "2048x4096x4096", "4096x4096x4096", "1024x8192x8192", "2048x8192x8192", "8192x8192x8192"]
MIBS = [64, 128, 256, 512]
CTAS = [2, 4, 8, 16, 22, 32, 48]
GRID = [510, 705, 900, 975, 1050, 1125, 1200, 1260, 1305, 1350, 1410]
BUDGETS = [0.0, 0.01, 0.03]


def one(args):
    shape, mib, ctas = args
    warnings.simplefilter("ignore")
    k, cal = Kernels.load(), Calib.load()
    k.boards = [list(np.mean(np.asarray(k.boards, float), axis=0))] * 4  # the node's mean board
    run = lambda f, fc=None: node.cell(k, cal, "concurrent", shape=shape, mib=mib, ctas=ctas, clock=f, comm_clock=fc)
    a = run(0)
    t_a, w_a = a["t_loop"], a["power"][0]
    grid = {}
    for fg, fc in itertools.product(GRID, GRID):
        c = run(fg, None if fg == fc else fc)
        grid[(fg, fc)] = (c["t_loop"], c["power"][0])
    row = dict(
        shape=shape,
        mib=mib,
        ctas=ctas,
        t_a=t_a,
        w_a=w_a,
        j_a=t_a * w_a / 1e3,
        clock_a=a["clock_mean"][0],
        gemm_last=a["gemm"][1] > a["comm"][1],
        grid=";".join(f"{fg}/{fc}/{t:.5f}/{w:.3f}" for (fg, fc), (t, w) in grid.items()),
    )
    for b in BUDGETS:
        ok = {key: v for key, v in grid.items() if v[0] <= t_a * (1 + b)}
        for name, sel in (("one", {key: v for key, v in ok.items() if key[0] == key[1]}), ("two", ok)):
            tag = f"{name}_{int(b * 100)}"
            if not sel:
                row.update({f"{tag}_fg": "", f"{tag}_fc": "", f"{tag}_t": "", f"{tag}_j": ""})
                continue
            (fg, fc), (t, w) = min(sel.items(), key=lambda kv: kv[1][0] * kv[1][1])
            row.update({f"{tag}_fg": fg, f"{tag}_fc": fc, f"{tag}_t": t, f"{tag}_j": t * w / 1e3})
    return row


def report(rows):
    pct = lambda r, tag: (float(r[f"{tag}_j"]) / float(r["j_a"]) - 1) * 100 if r[f"{tag}_j"] != "" else np.nan
    print(
        f"{len(rows)} workloads (6 GEMM shapes x 4 message sizes x 7 CTA counts). "
        "Energy per iteration against A (no clock lock), %"
    )
    for b in BUDGETS:
        t1, t2 = f"one_{int(b * 100)}", f"two_{int(b * 100)}"
        e1, e2 = np.array([pct(r, t1) for r in rows]), np.array([pct(r, t2) for r in rows])
        d = e2 - e1
        print(f"\n iteration at most {b * 100:.0f}% longer than A's")
        print(
            f"   one clock:  median {np.nanmedian(e1):+6.1f}  best {np.nanmin(e1):+6.1f}  worst {np.nanmax(e1):+6.1f}"
        )
        print(
            f"   two clocks: median {np.nanmedian(e2):+6.1f}  best {np.nanmin(e2):+6.1f}  worst {np.nanmax(e2):+6.1f}"
        )
        print(
            f"   two - one, points: median {np.nanmedian(d):+6.1f}  best {np.nanmin(d):+6.1f}  "
            "within 1 point of zero in "
            f"{int(np.sum(np.abs(d) < 1.0))} of {int(np.sum(np.isfinite(d)))}"
        )
        for label, sel in (
            ("collective ends last", [not _b(r["gemm_last"]) for r in rows]),
            ("GEMM ends last", [_b(r["gemm_last"]) for r in rows]),
        ):
            m = np.array(sel)
            if m.any():
                print(
                    f"     {label:21s} n={int(m.sum()):3d}  one {np.nanmedian(e1[m]):+6.1f}  "
                    f"two {np.nanmedian(e2[m]):+6.1f}  "
                    f"two - one {np.nanmedian(d[m]):+6.1f} (best {np.nanmin(d[m]):+6.1f})"
                )
        print(
            "     two - one by CTA count, median: "
            + "  ".join(f"{c}: {np.nanmedian(d[[int(r['ctas']) == c for r in rows]]):+.1f}" for c in CTAS)
        )
        print(
            "     two - one by message, median:   "
            + "  ".join(f"{m} MiB: {np.nanmedian(d[[int(r['mib']) == m for r in rows]]):+.1f}" for m in MIBS)
        )


def _b(v):
    return v is True or v == "True"


def report_free(rows, default_ctas):
    """The same question with the CTA count free: A is the stock governor at NCCL's
    default CTA count; `one` may pick any clock AND any CTA count, `two` any two clocks
    and any CTA count. More CTAs at a lower clock is a knob that exists today, and it
    competes with the second domain for the same slack."""
    by = {}
    for r in rows:
        g = {}
        for item in r["grid"].split(";"):
            fg, fc, t, w = item.split("/")
            g[(int(fg), int(fc))] = (float(t), float(w))
        by.setdefault((r["shape"], int(r["mib"])), {})[int(r["ctas"])] = (float(r["t_a"]), float(r["j_a"]), g)
    print(f"\nCTA count free; A = no clock lock at NCCL's default of {default_ctas} CTAs. Energy against A, %")
    for b in BUDGETS:
        e1, e2, picks = [], [], []
        for wl, per_c in by.items():
            t_a, j_a, _ = per_c[default_ctas]
            best = {}
            for name in ("one", "two"):
                cand = [
                    (t * w / 1e3, c, key)
                    for c, (_, _, g) in per_c.items()
                    for key, (t, w) in g.items()
                    if t <= t_a * (1 + b) and (name == "two" or key[0] == key[1])
                ]
                best[name] = min(cand) if cand else (j_a, default_ctas, None)
            e1.append((best["one"][0] / j_a - 1) * 100)
            e2.append((best["two"][0] / j_a - 1) * 100)
            picks.append((best["one"][1], best["two"][1]))
        e1, e2 = np.array(e1), np.array(e2)
        d = e2 - e1
        print(
            f" at most {b * 100:.0f}% longer:  one clock median {np.median(e1):+6.1f} (best {e1.min():+6.1f})   "
            "two clocks median "
            f"{np.median(e2):+6.1f} (best {e2.min():+6.1f})   two - one median {np.median(d):+6.1f} points, "
            f"best {d.min():+6.1f}, "
            f"within 1 point of zero in {int(np.sum(np.abs(d) < 1))} of {len(d)};  CTAs picked (one / two), median "
            f"{int(np.median([p[0] for p in picks]))} / {int(np.median([p[1] for p in picks]))}"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=40)
    ap.add_argument("--out", default=os.path.join(HERE, "out", "study_overlap.csv"))
    ap.add_argument("--report", action="store_true", help="print the table from an existing file")
    ap.add_argument("--default-ctas", type=int, default=0, help="NCCL's own CTA count, for the CTA-free table")
    a = ap.parse_args()
    csv.field_size_limit(10**9)
    if a.report:
        with open(a.out) as fh:
            rows = list(csv.DictReader(fh))
        report(rows)
        if a.default_ctas:
            report_free(rows, a.default_ctas)
        return None
    tasks = list(itertools.product(SHAPES, MIBS, CTAS))
    with ProcessPoolExecutor(a.jobs) as ex:
        rows = list(ex.map(one, tasks, chunksize=1))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    report(rows)
    if a.default_ctas:
        report_free(rows, a.default_ctas)
    print("wrote", os.path.relpath(a.out))


if __name__ == "__main__":
    main()
