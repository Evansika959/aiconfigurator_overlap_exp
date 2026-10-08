#!/usr/bin/env python3
"""Is the simulator's hardware layer the chip's, or only the MoE kernel's?

The simulator was calibrated on one kernel (the persistent MoE GEMM) and one model
(Qwen3-30B). Five checks, on measurements that already exist, of what carries over to
other kernels and other models, and of what the event view adds to the GEMM + all-reduce
overlap. Nothing is measured and nothing is written.

  1  cuBLAS GEMM, board power   the calibration's energies (per MAC, per byte, idle,
                                voltage curve) applied to ../gemm_dcfs_char as they are:
                                no number is fitted to the cuBLAS runs
  2  cuBLAS GEMM, time          the wave rule and one tensor cost, fitted on 108 SMs and
                                scored on 14-81 SMs
  3  NCCL all-reduce            its bandwidth table B(CTAs, clock) from ../comm_dvfs_char,
                                written in the simulator's kind of rate law
  4  other MoE models           the chip's capacities held at the Qwen fit: how many
                                per-model numbers Granite, OLMoE and Qwen3-Next need
                                (--models; about 10 min)
  5  GEMM beside the all-reduce the overlapped runs of ../overlap_exp replayed as events
                                from the measured solo times: how many SMs each NCCL CTA
                                takes from the GEMM

  source env.sh && python3 cross_study_check.py [--models]
"""

import argparse
import contextlib
import copy
import csv
import glob
import heapq
import io
import json
import os
import sys

import numpy as np
from sim import meas
from sim.calib import Calib
from sim.calibrate import fit_rate
from sim.lsq import lm
from sim.rate import RateParams
from sim.validate import corun, fits_l2, solo

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(TOOLS, "overlap_exp"))
import score_432  # the overlap study's own coefficient fits

TILE = 256 * 128  # outputs per cuBLAS bf16 tile on this part (gemm_dcfs_char/MODEL.md)
SMS = 108


def rms(e):
    return float(np.sqrt(np.mean(np.square(e))))


def worst(e):
    return float(np.ravel(e)[np.argmax(np.abs(e))])


# ------------------------------------------------------------- cuBLAS GEMM ----
def gemm_rows():
    """Solo cuBLAS GEMMs that held their clock: one dict per measured row."""
    with open(os.path.join(TOOLS, "overlap_exp", "data", "grid12.json")) as fh:
        grids = json.load(fh)  # ncu-measured: one shape splits K
    with open(os.path.join(TOOLS, "gemm_dcfs_char", "data", "gemm_dynamic_energy.csv")) as fh:
        held = [r for r in csv.DictReader(fh) if r["clock_held"] == "True" and r["status"] == "ok"]
    out = []
    for r in held:
        m, n, k = int(r["m"]), int(r["n"]), int(r["k"])
        row = dict(m=m, n=n, k=k, f=int(r["freq_req_mhz"]), sms=int(r["sm_avail"]), grid=grids[f"{m}_{n}_{k}"])
        out.append(dict(row, ms=float(r["latency_ms"]), w=float(r["power_w"])))
    return out


def gemm_power(r, pw, tile_m):
    """(board W, W of the MAC term alone) of a GEMM repeated back to back, from what it
    does per pass:

    MACs        M * N * K
    L2 bytes    the weight tiles its CTAs read. The simulator's e_panel is per byte of
                WEIGHT panel, so this depends on which side of the 256x128 tile runs
                along M: tile_m = 256 or 128. The kernel name does not say.
    DRAM bytes  each of the three matrices once
    """
    gmac = r["m"] * r["n"] * r["k"] / 1e9
    weights = gmac * 1e3 * 2 / tile_m  # MB
    dram = (r["m"] * r["k"] + r["k"] * r["n"] + r["m"] * r["n"]) * 2 / 1e6  # MB
    g = pw.g_at(r["f"])
    mj = g * (pw.e_mac * gmac + pw.e_panel * weights) + pw.e_dram * dram
    return pw.idle_at(r["f"]) + pw.act_at(r["f"]) + mj / r["ms"], g * pw.e_mac * gmac / r["ms"]


def check_gemm_power(rows, pw):
    print("1  cuBLAS GEMM, board power from the MoE calibration (no refit; measured latency)")
    full = [r for r in rows if r["sms"] == SMS]  # below 108 SMs a squatter kernel draws power too
    w = np.array([r["w"] for r in full])
    for tile_m in (256, 128):
        e = (np.array([gemm_power(r, pw, tile_m)[0] for r in full]) / w - 1) * 100
        a = np.abs(e)
        print(
            f"   weight tiles at tile_m = {tile_m}: n={len(full)}  median {np.median(a):.2f}%  bias {e.mean():+.2f}%"
            f"  p90 {np.percentile(a, 90):.2f}%  max {a.max():.2f}%"
        )
    p, mac = np.array([gemm_power(r, pw, 256) for r in full]).T
    idle = np.array([pw.idle_at(r["f"]) for r in full])
    print("   by clock (tile_m = 256):  MHz, rows, median error, MAC term as a share of measured power above idle")
    for f in sorted({r["f"] for r in full}):
        i = np.array([r["f"] == f for r in full])
        share = np.median(mac[i] / (w[i] - idle[i])) * 100
        print(f"     {f:5d}  {i.sum():3d}  {np.median(np.abs(p[i] / w[i] - 1)) * 100:5.2f}%  {share:5.1f}%")


def gemm_time(q, rows):
    """ms: full waves of CTAs, each CTA one tile, plus a launch cost.
    q = log of (SM*ms*MHz per GMAC, launch ms*MHz, launch ms)."""
    a, u, v = np.exp(q)
    out = []
    for r in rows:
        waves = -(-r["grid"] // r["sms"])
        split_k = max(1, r["grid"] // (r["m"] * r["n"] // TILE))
        out.append(waves * TILE * r["k"] / split_k / 1e9 * a / r["f"] + u / r["f"] + v)
    return np.array(out)


def check_gemm_time(rows, pw):
    print("\n2  cuBLAS GEMM, time: waves = ceil(tiles / SMs), one tensor cost, a launch cost")
    full = [r for r in rows if r["sms"] == SMS]
    err = lambda q, rs: np.log(gemm_time(q, rs) / np.array([r["ms"] for r in rs])) * 100
    q, _ = lm(lambda q: err(q, full) / 100, np.log([1000.0, 30.0, 0.005]), iters=200)
    a = float(np.exp(q[0]))
    print(f"   fitted on {len(full)} rows at 108 SMs: rms {rms(err(q, full)):.2f}%")
    print("   scored where a squatter kernel holds the other SMs:  SMs, rows, rms, bias, worst")
    for s in sorted({r["sms"] for r in rows} - {SMS}, reverse=True):
        e = err(q, [r for r in rows if r["sms"] == s])
        print(f"     {s:3d}  {len(e):3d}  {rms(e):5.2f}%  {e.mean():+5.2f}%  {worst(e):+6.1f}%")
    kernels = Calib.load().kernels
    moe = ",  ".join(f"MoE {k} {1200 / kernels[k].rate.aR:.3f}" for k in ("persistent", "queue"))
    print(f"   tensor throughput per SM at 1200 MHz, TMAC/s:  cuBLAS {1200 / a:.3f},  {moe}")
    # per-SM dynamic power over per-SM throughput, against the calibration's e_mac * g(f)
    slope = score_432.gemm_coeffs()[0]
    print("   J per TMAC:  MHz, cuBLAS (its W per busy SM / its throughput), simulator (e_mac * g)")
    for f in sorted(slope):
        cu, si = slope[f] * a / f, pw.e_mac * pw.g_at(f)
        print(f"     {f:5d}  {cu:.3f}  {si:.3f}  {(cu / si - 1) * 100:+5.1f}%")
    u, v = np.exp(q[1:])
    return lambda f: u / f + v  # the launch cost, ms


# ------------------------------------------------------------ NCCL all-reduce ----
def check_collective():
    """c SMs, each with a per-byte cost k(f) = a / f + b, sharing a capacity
    BW(f) = BW0 * (f / 1200) ** gamma.

    queue    1 / B = k / c + 1 / BW                       the simulator's 1 / (1 - rho), solved for B
    p-norm   B = ((c / k) ** -p + BW ** -p) ** (-1 / p)   a sharper knee
    """
    print("\n3  NCCL all-reduce (world 4): algorithmic GB/s over CTAs x clock, in the simulator's kind of rate law")
    table = score_432.comm_coeffs()[0]
    fs, cs = sorted(table), sorted(table[min(table)])
    clock, ctas = np.meshgrid(fs, cs, indexing="ij")
    bw = np.array([[table[f][c] for c in cs] for f in fs])

    def model(q, form):
        k = np.exp(q[0]) / clock + np.exp(q[1])
        cap = np.exp(q[2]) * (clock / 1200.0) ** q[3]
        if form == "queue":
            return 1.0 / (k / ctas + 1.0 / cap)
        p = np.exp(q[4])
        return ((ctas / k) ** -p + cap**-p) ** (-1.0 / p)

    for form in ("queue", "p-norm"):
        best = None
        for b0 in (1e-6, 1e-3, 1e-2):  # the fit has local minima: several starts
            for g0 in (0.2, 0.6, 1.0):
                x0 = [np.log(120), np.log(b0), np.log(150), g0] + ([np.log(3.0)] if form == "p-norm" else [])
                q, cost = lm(lambda q, form=form: np.log(model(q, form) / bw).ravel(), x0, iters=400)
                if best is None or cost < best[1]:
                    best = (q, cost)
        q = best[0]
        e = np.log(model(q, form) / bw) * 100
        knee = f", p = {np.exp(q[4]):.1f}" if form == "p-norm" else ""
        print(
            f"   {form:6s} {len(q)} numbers on {bw.size} cells: rms {rms(e):.2f}%  worst {worst(e):+.1f}%"
            f"   (capacity {np.exp(q[2]):.0f} GB/s at 1200 MHz, ~ f ** {q[3]:.2f}{knee})"
        )
        print("     error %, rows = MHz, columns = CTAs " + " ".join(f"{c:5d}" for c in cs))
        for i, f in enumerate(fs):
            print(f"     {f:5d}" + " " * 29 + " ".join(f"{v:+5.1f}" for v in e[i]))


# ------------------------------------------------------------ other MoE models ----
def check_models():
    """Per model, on its two-clock sweep (4 cases): transfer the Qwen calibration as it
    is, refit ONE scale on the three per-SM costs, or refit the three costs. The refits
    are scored on a case, or a clock, they did not see."""
    print("\n4  other MoE models, the chip's capacities held at the Qwen fit")
    cal = Calib.load()
    q = cal.rate("persistent")
    caps = dict(bR=0.0, bL=0.0, bw_d=q.bw_d, bw_l=q.bw_l, gamma_l=q.gamma_l, xi=q.xi, gamma_x=q.gamma_x, q=q.q)

    def errors(rate, recs):
        c = copy.deepcopy(cal)
        c.kernels["persistent"].rate = rate
        e = []
        for r in recs:
            if r["kind"] == "solo":
                e.append(np.log(solo(c, r)[0] / r["ms"]) * 100)
            else:
                (tl, th), _ = corun(c, r)
                e += [np.log(tl / r["light"]["ms"]) * 100, np.log(th / r["heavy"]["ms"]) * 100]
        return np.array(e)

    def scaled(s):
        d = dict(q.__dict__)
        d.update(aP=q.aP * s, bP=q.bP * s, aR=q.aR * s, aL=q.aL * s)
        return RateParams(**d)

    def fit_scale(train):
        """Golden-section search of the one scale."""
        lo, hi, g = 0.8, 1.4, (5**0.5 - 1) / 2
        f = lambda s: rms(errors(scaled(s), train))
        a, b = hi - g * (hi - lo), lo + g * (hi - lo)
        fa, fb = f(a), f(b)
        for _ in range(16):
            if fa < fb:
                hi, b, fb = b, a, fa
                a = hi - g * (hi - lo)
                fa = f(a)
            else:
                lo, a, fa = a, b, fb
                b = lo + g * (hi - lo)
                fb = f(b)
        return 0.5 * (lo + hi)

    def fit_costs(train):
        with contextlib.redirect_stdout(io.StringIO()):
            return fit_rate(train, fixed=dict(caps, bP=q.bP))[0]

    def leave_out(recs, key):
        """(train, test) with each value of key(record) held out in turn."""
        return [([r for r in recs if key(r) != v], [r for r in recs if key(r) == v]) for v in {key(r) for r in recs}]

    for m in ("granite", "olmoe", "qwen3next"):
        recs = [r for r in meas.scan(m, f"sm_corun_xmodel_{m}.csv") if r["kind"] in ("solo", "corun")]
        recs = [r for r in recs if not fits_l2(r)]
        folds = {
            "case": leave_out(recs, lambda r: (r["cells"], r["bm"], r["n_cold"])),
            "clock": leave_out(recs, lambda r: r["f"]),
        }
        e = errors(q, recs)
        print(f"   {m}: {len(e)} times.  Qwen calibration as it is: rms {rms(e):.2f}%  bias {e.mean():+.2f}%")
        for name, fs in folds.items():
            scales = [fit_scale(tr) for tr, _ in fs]
            e1 = np.concatenate([errors(scaled(s), ts) for s, (_, ts) in zip(scales, fs, strict=True)])
            e3 = np.concatenate([errors(fit_costs(tr), ts) for tr, ts in fs])
            print(
                f"     held-out {name:5s}: one scale ({min(scales):.3f}-{max(scales):.3f})"
                f" rms {rms(e1):.2f}%  bias {e1.mean():+.2f}%"
                f"   |   three costs rms {rms(e3):.2f}%  bias {e3.mean():+.2f}%"
            )


# -------------------------------------------------- GEMM beside the all-reduce ----
def overlap_cases():
    """Overlapped runs of ../overlap_exp that held their clock (three campaigns), each
    with the measured solo time of its GEMM and of its collective at the same clock."""
    names = (
        ["overlap_432"] + [f"ext_{m}mib" for m in (64, 128, 256, 512)] + [f"commbound_{m}mib" for m in (128, 256, 512)]
    )
    out = []
    for name in names:
        with open(os.path.join(TOOLS, "overlap_exp", "data", name + ".csv")) as fh:
            rows = [r for r in csv.DictReader(fh) if r["clock_held"] == "True" and int(r["clock"]) > 0]
        gemm_key = lambda r: (r["clock"], r["m"], r["n"], r["k"])
        comm_key = lambda r: (r["clock"], r["ctas"], r["ar_mib"])
        gemm = {gemm_key(r): float(r["iter_ms"]) for r in rows if r["mode"] == "gemm_only"}
        comm = {comm_key(r): float(r["iter_ms"]) for r in rows if r["mode"] == "comm_only"}
        for r in rows:
            if r["mode"] == "concurrent" and gemm_key(r) in gemm and comm_key(r) in comm:
                case = dict(f=int(r["clock"]), ctas=int(r["ctas"]), grid=int(r["grid"]), ms=float(r["iter_ms"]))
                out.append(dict(case, gemm=gemm[gemm_key(r)], comm=comm[comm_key(r)]))
    return out


def launch_offset():
    """ms from the collective's start to the GEMM's inside one overlapped iteration, on
    the overlap study's per-kernel timeline (30 runs of one cell at 1200 MHz)."""
    gaps = []
    for path in glob.glob(os.path.join(TOOLS, "overlap_exp", "data", "repeat*", "spans_*.json")):
        with open(path) as fh:
            s = json.load(fh)
        gaps.append(s["gemm_start"] - s["comm_start"])
    return float(np.median(gaps))


def replay(r, launch, lost, offset):
    """ms per iteration, as events. The collective keeps `lost` SMs per CTA from the GEMM
    until it ends; the GEMM starts `offset` ms after it, and each of its tiles goes to
    whichever SM frees first. The collective itself takes its solo time."""
    start = offset + launch(r["f"])
    tile = (r["gemm"] - launch(r["f"])) / -(-r["grid"] // SMS)  # one wave of the solo GEMM
    held = min(SMS, round(lost * r["ctas"]))
    free = [r["comm"]] * held + [start] * (SMS - held)
    heapq.heapify(free)
    end = 0.0
    for _ in range(r["grid"]):
        t = max(heapq.heappop(free), start) + tile
        end = max(end, t)
        heapq.heappush(free, t)
    return max(end, r["comm"])


def closed_form(r):
    """The overlap study's rule: the GEMM re-waved onto 108 - CTAs SMs for the share of
    it the collective overlaps."""
    w, w2 = -(-r["grid"] // SMS), -(-r["grid"] // max(SMS - r["ctas"], 1))
    return max(r["gemm"] * (1 + (w2 / w - 1) * min(1, r["comm"] / r["gemm"])), r["comm"])


def check_overlap(launch):
    cases, offset = overlap_cases(), launch_offset()
    ms = np.array([r["ms"] for r in cases])
    err = lambda fn: (np.array([fn(r) for r in cases]) / ms - 1) * 100
    line = lambda e: (
        f"median {np.median(np.abs(e)):5.2f}%  p90 {np.percentile(np.abs(e), 90):5.2f}%"
        f"  max {np.abs(e).max():5.2f}%  bias {e.mean():+5.2f}%"
    )
    print(
        f"\n5  GEMM beside the all-reduce: iteration time of {len(cases)} overlapped runs from the measured solo times"
    )
    print(f"   the overlap study's closed form (1 SM per NCCL CTA):  {line(err(closed_form))}")
    print(
        f"   replayed as events, the GEMM issued {offset:.3f} ms after the collective; SMs the GEMM loses per NCCL CTA:"
    )
    by_lost = {}
    for lost in (1.0, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0):
        by_lost[lost] = err(lambda r, lost=lost: replay(r, launch, lost, offset))
        print(f"     {lost:4.2f}  {line(by_lost[lost])}  rms {rms(by_lost[lost]):5.2f}%")
    print("   by CTAs:  CTAs, runs, median error (bias) at 1 SM per CTA, at 2 SMs per CTA")
    ctas = np.array([r["ctas"] for r in cases])
    for c in sorted(set(ctas)):
        e1, e2 = by_lost[1.0][ctas == c], by_lost[2.0][ctas == c]
        print(
            f"     {c:3d}  {len(e1):4d}  {np.median(np.abs(e1)):5.2f}% ({e1.mean():+6.2f})"
            f"  {np.median(np.abs(e2)):5.2f}% ({e2.mean():+6.2f})"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", action="store_true", help="also check 4 (about 10 min)")
    args = ap.parse_args()
    pw = Calib.load().power
    rows = gemm_rows()
    check_gemm_power(rows, pw)
    launch = check_gemm_time(rows, pw)
    check_collective()
    if args.models:
        check_models()
    check_overlap(launch)


if __name__ == "__main__":
    main()
