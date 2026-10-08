#!/usr/bin/env python3
"""Fit the simulator's hardware parameters to the measured CSVs in ../data.

  python3 -m sim.calibrate            # -> sim/calib/a100_qwen.json, and a residual report

Five fits, in the order they depend on each other. Each prints what it was fitted on and
how far the fitted model still is from its own data, so a number in the calibration file
is never quoted without its error.

  1 rate       persistent kernel, every solo and co-run time of the        7 parameters
               training sweeps (meas.TRAIN)
  2 epilogue   (whole layer end to end) - (its two GEMM launches)          3
  3 queue      vLLM's kernel end to end, three campaigns                   4 + window
  4 idle       board power with nothing running, one method, every clock   a table
  5 power      energy per unit activity, V^2 per clock                     3 + 2 per clock

Hold-out data never enters here; `sim.validate` reports it.
"""

import argparse
import os
import time
from collections import defaultdict

import numpy as np

from . import meas
from .calib import DEFAULT, Calib, Epilogue, KernelCal
from .lsq import lm
from .power import PowerParams
from .rate import RateParams, window_times

HERE = os.path.dirname(os.path.abspath(__file__))
F_REF = 1290.0
NAMES = ["aP", "bP", "aR", "bR", "aL", "bL", "bw_d", "bw_l", "gamma_l", "xi", "gamma_x", "q"]
_LOG = dict(aP=1, bP=1, aR=1, bR=1, aL=1, bL=1, bw_d=1, bw_l=1, gamma_l=0, xi=1, gamma_x=0, q=1)
# start near the per-clock fits: kP = 22.7e-3 and L0 = 0.9e-3 SM*ms/MB, kR = 0.86 at 1290
_X0 = dict(
    aP=22.0,
    bP=5e-3,
    aR=1110.0,
    bR=1e-3,
    aL=1.0,
    bL=1e-4,
    bw_d=1340.0,
    bw_l=21000.0,
    gamma_l=1.0,
    xi=0.4,
    gamma_x=1.0,
    q=1.0,
)
# MODEL FORMS: which parameters are pinned, and to what. A form is chosen by how well a
# fit on the first sweep alone predicts the two sweeps collected after it
# (`python3 -m sim.calibrate --compare`), not by its fit to its own data.
#
#   bR = bL = 0     tensor work and DRAM waits are pure cycles: freed, both go to zero
#                   and nothing else moves
#   bw_d = 1555     the A100-40GB's nominal memory bandwidth (GB/s, NVIDIA datasheet).
#                   Freed together with q it runs off to 25-34 TB/s with no gain on the
#                   later sweeps: the data cannot tell a DRAM queue from a steeper
#                   coupling, so the queue keeps the capacity the hardware has
#   q, gammas       FREE in the shipped form. 'linear' pins q = 1 and both gammas at 1,
#                   the model as first written; the later sweeps reject it (4.2% against
#                   3.8% on the second sweep). 'proportional' frees q alone.
FORMS = {
    "shipped": dict(bR=0.0, bL=0.0, bw_d=1555.0),
    "proportional": dict(bR=0.0, bL=0.0, bw_d=1555.0, gamma_l=1.0, gamma_x=1.0),
    "linear": dict(bR=0.0, bL=0.0, q=1.0, gamma_l=1.0, gamma_x=1.0),
}
PINNED = FORMS["shipped"]


def coefs(p, f):
    """Per-unit coefficient arrays for clocks f; p maps name -> scalar or per-unit array."""
    return dict(
        kP=p["aP"] / f + p["bP"],
        kR=p["aR"] / f + p["bR"],
        L0=p["aL"] / f + p["bL"],
        bw_d=p["bw_d"],
        bw_l=p["bw_l"] * (f / F_REF) ** p["gamma_l"],
        xi=p["xi"] * (f / F_REF) ** p["gamma_x"],
        q=p["q"],
    )


class Units:
    """Many stages as one flat array of windows, so one vectorised solve times all of
    them. A unit is one GEMM of one partition at one clock; two units can be partners,
    meaning they ran side by side and each sees the other's streams."""

    def __init__(self):
        self.cols = defaultdict(list)
        self.f, self.partner, self.tot = [], [], []

    def add(self, part, n, f, window=None):
        ids = []
        for g in ("g1", "g2"):
            st = part.stage(g, n, window)
            u = len(self.f)
            for k in ("cnt", "panel", "gmac", "dram", "wait"):
                self.cols[k].append(getattr(st, k))
            self.cols["n"].append(np.full(len(st.cnt), float(n)))
            self.cols["u"].append(np.full(len(st.cnt), u))
            self.f.append(float(f))
            self.partner.append(-1)
            self.tot.append((st.tot_dram, st.tot_panel))
            ids.append(u)
        return ids

    def pair(self, a, b):
        for x, y in zip(a, b, strict=True):  # g1 beside g1, g2 beside g2
            self.partner[x], self.partner[y] = y, x

    def freeze(self):
        self.w = {k: np.concatenate(v) for k, v in self.cols.items()}
        self.w["u"] = u = self.w["u"].astype(int)
        self.f, self.partner = np.array(self.f), np.array(self.partner)
        self.tot_d = np.array([t[0] for t in self.tot])
        self.tot_l = np.array([t[1] for t in self.tot])
        self.co, self.nu = self.partner >= 0, len(self.f)
        # the windows of each half of every pair: one half is solved against the other's
        # newest times, then the other against that
        first = self.co & (np.arange(self.nu) < self.partner)
        self.halves = [np.where(first[u])[0], np.where((self.co & ~first)[u])[0]]
        self.sweeps = 0
        return self

    def _solve(self, cw, ix, ext_d, ext_l):
        """ms per unit for the windows ix (0 for units with none), given what the rest
        of the chip streams at each unit."""
        w, u = self.w, self.w["u"][ix]
        col = lambda v: v[ix] if np.ndim(v) else v
        t = window_times(
            w["panel"][ix], w["gmac"][ix], w["dram"][ix], w["wait"][ix], w["n"][ix],
            col(cw["kP"]), col(cw["kR"]), col(cw["L0"]), col(cw["bw_d"]), col(cw["bw_l"]),
            col(cw["xi"]), col(cw["q"]), ext_d[u], ext_l[u],
        )  # fmt: skip
        return np.bincount(u, weights=w["cnt"][ix] * t, minlength=self.nu)

    def times(self, p, max_sweeps=120, tol=1e-11):
        """ms per unit under parameters p (see coefs).

        Units that ran alone are one solve. Partners are a fixed point: each one's time
        depends on what the other streams, which depends on the other's time. It is
        solved by Gauss-Seidel sweeps from the same start every time -- both alone --
        down to `tol`, so the answer is a smooth function of p and nothing else. (A
        warm start from the previous call, cut off after a fixed number of steps, made
        two evaluations of the same p differ by up to half a percent, and the
        optimiser's finite differences were differences of that noise.)"""
        c = coefs(p, self.f)
        u = self.w["u"]
        cw = {k: (v[u] if np.ndim(v) else v) for k, v in c.items()}
        zero = np.zeros(self.nu)
        T = self._solve(cw, np.arange(len(u)), zero, zero)
        if not self.co.any():
            return T
        stream = lambda tot: np.where(self.co, (tot / T)[self.partner], 0.0)
        for sweep in range(1, max_sweeps + 1):
            before = T.copy()
            for ix in self.halves:
                Tn = self._solve(cw, ix, stream(self.tot_d), stream(self.tot_l))
                T = np.where(Tn > 0, Tn, T)
            if np.abs(T / before - 1).max() < tol:
                break
        self.sweeps = sweep
        return T


def _stats(e):
    e = np.asarray(e, float)
    return dict(n=len(e), rms=float(np.sqrt(np.mean(e**2))), bias=float(e.mean()), worst=float(e[np.argmax(np.abs(e))]))


def _line(label, s):
    return f"      {label:22s} n={s['n']:4d}  rms {s['rms']:5.2f}%  bias {s['bias']:+5.2f}%  worst {s['worst']:+6.1f}%"


def _fit(res, names, start, fixed, verbose):
    free = [k for k in names if k not in fixed]
    x0 = np.array([np.log(start[k]) if _LOG[k] else start[k] for k in free])

    def unpack(x):
        p = dict(start)
        p.update(fixed)
        for k, v in zip(free, x, strict=True):
            p[k] = float(np.exp(v)) if _LOG[k] else float(v)
        return p

    # one LM run can stall where the surface is flat: restart from where it stopped
    # until a run no longer improves the cost
    cost = None
    for _ in range(6):
        x, c = lm(lambda x: res(unpack(x)), x0, iters=60, rel_step=1e-3, verbose=verbose)
        if cost is not None and c > cost * (1 - 1e-7):
            break
        x0, cost = x, c
    return unpack(x0), len(free)


def _sigma(res, p, names, fixed):
    """One-sigma of each free parameter from the Gauss-Newton covariance at the optimum:
    in log units for the log-fitted ones (0.03 is 3%), absolute for the gammas. It
    treats the residuals as independent noise, which model error is not: read it as how
    sharply the data pin a parameter, not as a confidence interval."""
    free = [k for k in names if k not in fixed]
    r0 = res(p)
    J = np.empty((len(r0), len(free)))
    for j, k in enumerate(free):
        q = dict(p)
        q[k] = p[k] * np.exp(1e-3) if _LOG[k] else p[k] + 1e-3
        J[:, j] = (res(q) - r0) / 1e-3
    cov = np.linalg.inv(J.T @ J) * (r0 @ r0) / (len(r0) - len(free))
    return {k: float(v) for k, v in zip(free, np.sqrt(np.diag(cov)), strict=True)}


# ------------------------------------------------------------------- 1 rate ----
def fit_rate(recs, fixed=PINNED, verbose=False):
    """Persistent kernel: every solo and co-run time of the scan, all clocks at once."""
    U, targets = Units(), []
    for r in recs:
        if r["kind"] == "solo":
            targets.append((r["ms"], U.add(r["part"], r["n"], r["f"]), "solo", r))
        elif r["kind"] == "corun":
            a = U.add(r["light"]["part"], r["light"]["n"], r["f"])
            b = U.add(r["heavy"]["part"], r["heavy"]["n"], r["f"])
            U.pair(a, b)
            targets.append((r["light"]["ms"], a, "corun", r))
            targets.append((r["heavy"]["ms"], b, "corun", r))
    U.freeze()
    y = np.array([t[0] for t in targets])
    ids = [t[1] for t in targets]

    def predict(p):
        T = U.times(p)
        return np.array([T[i].sum() for i in ids])

    t0 = time.time()
    res = lambda p: np.log(predict(p) / y)
    p, nfree = _fit(res, NAMES, _X0, fixed, verbose)
    e = res(p) * 100
    kind = np.array([t[2] for t in targets])
    rep = dict(
        all=_stats(e),
        solo=_stats(e[kind == "solo"]),
        corun=_stats(e[kind == "corun"]),
        free=nfree,
        pinned=dict(fixed),
        sigma=_sigma(res, p, NAMES, fixed),
        clocks=sorted({int(t[3]["f"]) for t in targets}),
    )
    print(f"  1 rate (persistent): {len(y)} times, {nfree} parameters, {time.time() - t0:.0f} s")
    for k in ("all", "solo", "corun"):
        print(_line(k, rep[k]))
    print("      " + "  ".join(f"{k}={p[k]:.4g}" for k in NAMES if k not in fixed))
    print("      1-sigma: " + "  ".join(f"{k} {v:.3f}" for k, v in rep["sigma"].items()))
    return RateParams(**p, f_ref=F_REF), rep, (targets, e)


# --------------------------------------------------------------- 2 epilogue ----
def fit_epilogue(recs):
    """T_epi = (end-to-end layer) - (its launches alone), same campaign, same clock."""
    solo = {(r["cells"], r["bm"], r["n_cold"], r["f"]): r for r in recs if r["kind"] == "solo" and r["name"] == "all"}
    rows = []
    for r in recs:
        s = solo.get((r["cells"], r["bm"], r["n_cold"], r["f"])) if r["kind"] == "full" else None
        if s is not None:
            rows.append((r["pairs"], r["f"], r["ms"] - s["ms"], r, s))
    P = np.array([x[0] for x in rows], float)
    f = np.array([x[1] for x in rows], float)
    t = np.array([x[2] for x in rows])
    X = np.stack([np.ones_like(P), P, P / f], 1)
    b, *_ = np.linalg.lstsq(X, t, rcond=None)
    e = (X @ b - t) / t * 100
    rep = _stats(e)
    print(
        f"  2 epilogue: {len(t)} (case, clock) pairs; ms = {b[0] * 1e3:.1f} us + pairs x "
        f"({b[1] * 1e6:.2f} + {b[2] * 1e6:.0f} / f[MHz]) ns"
    )
    print(_line("epilogue time", rep))
    return dict(t0=float(b[0]), c0=float(b[1]), c1=float(b[2])), rep, rows


# ------------------------------------------------------------------ 3 queue ----
def fit_queue(recs, base: RateParams, epi, windows=(1, 2, 3, 4), verbose=False):
    """vLLM's queue kernel, end to end. It shares the chip -- DRAM, L2, their coupling --
    with the persistent kernel, so those are inherited; only its per-SM costs differ
    (several blocks resident on an SM hide each other's latency) and how many tiles it
    keeps in flight."""
    recs = [r for r in recs if r["kind"] == "queue"]
    y = np.array([r["ms"] for r in recs])
    t_epi = np.array([epi["t0"] + r["pairs"] * (epi["c0"] + epi["c1"] / r["f"]) for r in recs])
    shared = {k: getattr(base, k) for k in ("bw_d", "bw_l", "gamma_l", "xi", "gamma_x", "q")}
    fixed = dict(shared, bR=0.0, bL=0.0)
    start = dict(_X0, aP=base.aP, bP=base.bP, aR=base.aR, aL=base.aL, **fixed)
    best = None
    for wm in windows:
        U = Units()
        ids = [U.add(r["part"], r["n"], r["f"], window=wm * r["n"]) for r in recs]
        U.freeze()

        def predict(p):
            T = U.times(p)
            return np.array([T[i].sum() for i in ids]) + t_epi

        p, nfree = _fit(lambda p: np.log(predict(p) / y), NAMES, start, fixed, verbose)
        e = np.log(predict(p) / y) * 100
        s = _stats(e)
        print(f"      window = {wm} tiles per SM: rms {s['rms']:.2f}%  worst {s['worst']:+.1f}%")
        if best is None or s["rms"] < best[2]["rms"]:
            best = (wm, p, s, e)
    wm, p, s, e = best
    rep = dict(all=s, window=wm)
    src = np.array(
        [r.get("src", "scan") if r.get("src", "").startswith(("batch", "b1")) else "training sweeps" for r in recs]
    )
    print(f"  3 queue kernel: {len(y)} end-to-end times, {nfree} parameters + window {wm}")
    for k in sorted(set(src)):
        rep[k] = _stats(e[src == k])
        print(_line(k, rep[k]))
    print("      " + "  ".join(f"{k}={p[k]:.4g}" for k in ("aP", "bP", "aR", "aL")))
    return RateParams(**p, f_ref=F_REF), float(wm), rep, (recs, e)


# ------------------------------------------------------------------- 4 idle ----
def fit_idle(model):
    f, w, temp = meas.idle_sweep(model, "cold")
    fw, ww, tw = meas.idle_sweep(model, "warm")
    lo = [i for i, x in enumerate(f) if x <= 1050]
    slope, shared = np.polyfit(np.array(f)[lo], np.array(w)[lo], 1)
    print(
        f"  4 idle: {len(f)} clocks {f[0]}-{f[-1]} MHz, die {np.mean(temp):.0f} C; "
        f"{w[0]:.1f} W at {f[0]}, {w[-1]:.1f} W at {f[-1]}; below 1050 MHz "
        f"{slope * 1e3:.2f} mW/MHz on {shared:.1f} W"
    )
    print(
        f"      right after a 2 s load burst (die +{np.mean(tw) - np.mean(temp):.1f} C): "
        f"+{ww[0] - w[0]:.2f} W at {f[0]}, +{ww[-1] - w[-1]:.2f} W at {f[-1]}"
    )
    return dict(clocks=f, idle=w, p_shared=float(shared), slope=float(slope), temp=float(np.mean(temp)))


# ------------------------------------------------------------------ 5 power ----
def streams(r):
    """(panel MB/ms, real GMAC/ms, padding GMAC/ms, DRAM MB/ms) of a solo or co-run
    record, from its MEASURED times -- so the power fit does not inherit the rate
    model's error."""

    def one(part, n, ms):
        s1, s2 = part.stage("g1", n), part.stage("g2", n)
        return (
            np.array(
                [
                    s1.tot_panel + s2.tot_panel,
                    s1.gmac_real + s2.gmac_real,
                    s1.gmac_pad + s2.gmac_pad,
                    s1.tot_dram + s2.tot_dram,
                ]
            )
            / ms
        )

    if r["kind"] == "solo":
        return one(r["part"], r["n"], r["ms"])
    return sum(one(r[k]["part"], r[k]["n"], r[k]["ms"]) for k in ("light", "heavy"))


def _alternate(S, y, k_of, n_clocks, sweeps):
    """Solve y = act[k] + g[k] * (S[:, :3] @ e[:3]) + e[3] * S[:, 3] by alternating:
    given the four energies e, each clock k is a straight line (act, g); given the
    lines, the energies are a 4-column regression. g is 1 at the lowest clock."""
    e = np.array([0.022, 1.5, 0.5, 0.059])
    act, g = np.zeros(n_clocks), np.ones(n_clocks)
    for _ in range(sweeps):
        x, rest = S[:, :3] @ e[:3], y - e[3] * S[:, 3]
        for k in range(n_clocks):
            m = k_of == k
            act[k], g[k] = np.linalg.lstsq(np.stack([np.ones(m.sum()), x[m]], 1), rest[m], rcond=None)[0]
        gr = g[k_of]
        th = np.linalg.lstsq(np.concatenate([gr[:, None] * S[:, :3], S[:, 3:]], 1), y - act[k_of], rcond=None)[0]
        e = th * np.array([g[0], g[0], g[0], 1.0])
        g = g / g[0]
    return e, act, g


def _densify(table, clocks, act, e, extra, idle_at, rep):
    """A second campaign (`extra`: solo records at clocks the scan does not have) fills
    in g between the scan's clocks, and checks it at the shared ones. It gives the
    SHAPE between two scan clocks; it is scaled to agree with the scan at both."""
    Sx = np.array([streams(r) for r in extra])
    fx = np.array([r["f"] for r in extra], float)
    yx = np.array([r["power"] for r in extra]) - idle_at(fx)
    x = Sx[:, :3] @ e[:3]
    c = np.interp(fx, clocks, act) + e[3] * Sx[:, 3]
    gx = {}
    for fk in sorted(set(fx.tolist())):
        m = fx == fk
        gx[fk] = float(x[m] @ (yx[m] - c[m]) / (x[m] @ x[m]))
    both = [fk for fk in gx if fk in clocks]
    d = [gx[fk] / table[fk] - 1 for fk in both]
    rep["g_second_campaign"] = dict(clocks=[int(v) for v in both], ratio=[float(1 + v) for v in d])
    print(
        "      second campaign, g at the shared clocks: "
        + "  ".join(f"{int(fk)}:{v * 100:+.1f}%" for fk, v in zip(both, d, strict=True))
    )
    for fk, v in gx.items():
        if fk not in table:
            lo = max([b for b in both if b < fk], default=None)
            hi = min([b for b in both if b > fk], default=None)
            if lo is not None and hi is not None:
                w = (fk - lo) / (hi - lo)
                k = (1 - w) * table[lo] / gx[lo] + w * table[hi] / gx[hi]
            else:
                near = lo if lo is not None else hi
                k = table[near] / gx[near] if near is not None else 1.0
            table[fk] = v * k


def fit_power(recs, extra, idle, sweeps=400):
    """P - P_idle(f) = act(f) + g(f) (e_panel S_L + e_mac S_real + e_pad S_pad) + e_dram S_D.

    One (act, g) per clock and four energies shared by every clock, fitted to the solo
    and co-run records of the training sweeps; `extra` then fills in g at the clocks
    between (see _densify)."""
    idle_at = lambda f: np.interp(f, idle["clocks"], idle["idle"])
    recs = [r for r in recs if r["kind"] in ("solo", "corun")]
    S = np.array([streams(r) for r in recs])
    f = np.array([r["f"] for r in recs], float)
    y = np.array([r["power"] for r in recs]) - idle_at(f)
    board = np.array([r["power"] for r in recs])
    clocks = sorted(set(f.tolist()))
    k_of = np.array([clocks.index(x) for x in f])
    e, act, g = _alternate(S, y, k_of, len(clocks), sweeps)
    pred = act[k_of] + g[k_of] * (S[:, :3] @ e[:3]) + e[3] * S[:, 3]
    err = (pred - y) / board * 100
    kind = np.array([r["kind"] for r in recs])
    rep = dict(all=_stats(err), solo=_stats(err[kind == "solo"]), corun=_stats(err[kind == "corun"]))
    print(f"  5 power: {len(y)} records, {len(clocks)} clocks; error as % of board power")
    for k in ("all", "solo", "corun"):
        print(_line(k, rep[k]))
    print(
        f"      e_panel {e[0] * 1e3:.2f} mJ/GB   e_mac {e[1]:.3f} J/TMAC (real pairs)   "
        f"e_pad {e[2]:.3f} J/TMAC (padding)   e_dram {e[3] * 1e3:.2f} mJ/GB"
    )
    print("      g:   " + "  ".join(f"{int(c)}:{v:.3f}" for c, v in zip(clocks, g, strict=True)))
    print("      act: " + "  ".join(f"{int(c)}:{v:.1f}" for c, v in zip(clocks, act, strict=True)))
    # is DRAM energy per byte really the same at every clock? Refit it clock by clock
    # with everything else held: a rail that followed the SM voltage would rise 1.8x.
    rest = y - act[k_of] - g[k_of] * (S[:, :3] @ e[:3])
    per = [float(S[k_of == k, 3] @ rest[k_of == k] / (S[k_of == k, 3] @ S[k_of == k, 3])) for k in range(len(clocks))]
    rep["e_dram_per_clock_mj_gb"] = [round(v * 1e3, 1) for v in per]
    print(
        f"      e_dram refitted clock by clock: {min(per) * 1e3:.1f}-{max(per) * 1e3:.1f} "
        f"mJ/GB over {int(clocks[0])}-{int(clocks[-1])} MHz, no trend"
    )
    act = _isotonic(act)
    print("      act, made monotone: " + "  ".join(f"{v:.1f}" for v in act))
    table = dict(zip(clocks, g, strict=True))
    if extra:
        _densify(table, clocks, act, e, extra, idle_at, rep)
    fs = sorted(table)
    return (
        dict(
            clocks=fs,
            g=[float(table[k]) for k in fs],
            e_panel=float(e[0]),
            e_mac=float(e[1]),
            e_pad=float(e[2]),
            e_dram=float(e[3]),
            act_clocks=clocks,
            act=act.tolist(),
        ),
        rep,
        (recs, err),
    )


def _isotonic(y):
    """Non-decreasing least-squares fit (pool adjacent violators). The per-clock
    intercept is a few watts with +-2 W of noise from which cases a clock happened to
    see; nothing physical makes a busy chip draw LESS extra power at a higher clock."""
    blocks = [[v, 1] for v in y]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0] + 1e-12:
            a, b = blocks[i], blocks[i + 1]
            blocks[i : i + 2] = [[(a[0] * a[1] + b[0] * b[1]) / (a[1] + b[1]), a[1] + b[1]]]
            i = max(i - 1, 0)
        else:
            i += 1
    return np.array([v for v, n in blocks for _ in range(n)])


def fit_epilogue_energy(rows, pw, idle):
    """Dynamic energy of the epilogue per pair: j_dram + g(f) j_sm. The difference of
    two layer energies is noisy, so this is two numbers fitted to all of it."""
    idle_at = lambda f: np.interp(f, idle["clocks"], idle["idle"])
    P, G, E = [], [], []
    for pairs, f, t_epi, full, solo in rows:
        e = (full["power"] - idle_at(f)) * full["ms"] - (solo["power"] - idle_at(f)) * solo["ms"]
        P.append(pairs)
        G.append(np.interp(f, pw["clocks"], pw["g"]))
        E.append(e)
    P, G, E = np.array(P, float), np.array(G), np.array(E)
    b, *_ = np.linalg.lstsq(np.stack([P, P * G], 1), E, rcond=None)
    b = np.maximum(b, 0.0)
    err = np.stack([P, P * G], 1) @ b - E
    print(
        f"  epilogue energy: {b[0] * 1e3:.2f} + g x {b[1] * 1e3:.2f} uJ per pair; "
        f"rms {np.sqrt(np.mean(err**2)):.1f} mJ on layers of {np.median(E):.0f} mJ median"
    )
    return float(b[0]), float(b[1])


# ------------------------------------------------------------------- driver ----
def _units(recs):
    """The solo and co-run records as Units, with each measured time and its unit ids."""
    U, targets = Units(), []
    for r in recs:
        if r["kind"] == "solo":
            targets.append((r["ms"], U.add(r["part"], r["n"], r["f"])))
        elif r["kind"] == "corun":
            x = U.add(r["light"]["part"], r["light"]["n"], r["f"])
            y = U.add(r["heavy"]["part"], r["heavy"]["n"], r["f"])
            U.pair(x, y)
            targets += [(r["light"]["ms"], x), (r["heavy"]["ms"], y)]
    return U.freeze(), targets


def cross_validate(recs, forms=("shipped",)):
    """Fit on the first sweep alone, score each later one: what the error is on
    workloads and clocks the fit has never seen, measured before they were allowed in.
    Returns {form: {sweep: stats}}; this is how a form is chosen."""
    first = meas.TRAIN[0][0]
    a = [r for r in recs if r["src"] == first]
    others = [stem for stem, _ in meas.TRAIN[1:] if any(r["src"] == stem for r in recs)]
    if not a or not others:
        return None
    later = {stem: _units([r for r in recs if r["src"] == stem]) for stem in others}
    out = {}
    for form in forms:
        print(f"  cross-validation, form '{form}': fitted on {first} alone ...")
        rp, _, _ = fit_rate(a, fixed=FORMS[form])
        out[form] = {}
        for stem, (U, targets) in later.items():
            T = U.times({k: getattr(rp, k) for k in NAMES})
            out[form][stem] = _stats(np.array([np.log(T[i].sum() / m) * 100 for m, i in targets]))
            print(_line(f"... scored on {stem[9:]}", out[form][stem]))
    return out


def run(model="qwen", out=DEFAULT, verbose=False, form="shipped", compare=False):
    """Fit everything and write the calibration file. `form` picks the rate model's
    pinned parameters (FORMS); `compare` cross-validates every form, not only that one."""
    recs = meas.train(model)
    kinds = defaultdict(int)
    for r in recs:
        kinds[r["kind"]] += 1
    print(f"training sweeps: {sorted({r['src'] for r in recs})}: {dict(kinds)}")
    cv = cross_validate(recs, tuple(FORMS) if compare else (form,))
    persistent, rep_rate, _ = fit_rate(recs, fixed=FORMS[form], verbose=verbose)
    epi, rep_epi, epi_rows = fit_epilogue(recs)
    queue, qwin, rep_q, _ = fit_queue(
        [r for r in recs if r["kind"] == "queue"] + meas.queue_sweeps(model), persistent, epi, verbose=verbose
    )
    idle = fit_idle(model)
    # the second campaign timed graph replays of single passes: ~20 us of launch gap per
    # kernel, 10% of a 0.4 ms pass and invisible in a 7 ms one. Keep the long ones.
    extra = [r for r in meas.kappa(model) if r["ms"] > 3.0]
    pw, rep_pw, _ = fit_power(recs, extra, idle)
    j_dram, j_sm = fit_epilogue_energy(epi_rows, pw, idle)

    # tables on one grid: the idle sweep's clocks; g is 1 below the lowest fitted clock
    grid = idle["clocks"]
    g = np.interp(grid, pw["clocks"], pw["g"], left=1.0)
    act_f, act = pw["act_clocks"], pw["act"]
    power = PowerParams(
        clocks=[float(x) for x in grid],
        g=[float(x) for x in g],
        idle=[float(x) for x in idle["idle"]],
        e_panel=pw["e_panel"],
        e_mac=pw["e_mac"],
        e_pad=pw["e_pad"],
        e_dram=pw["e_dram"],
        act=[float(x) for x in np.interp(grid, act_f, act)],
        p_shared=idle["p_shared"],
    )
    cal = Calib(
        kernels=dict(persistent=KernelCal(persistent), queue=KernelCal(queue, qwin)),
        epilogue=Epilogue(j_dram=j_dram, j_sm=j_sm, **epi),
        power=power,
        meta=dict(
            model=model,
            fitted=time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
            data=sorted({r["src"] for r in recs})
            + ["partition_kappa_hiclk54", "batch_clock", "b1_energy", "idle_sweep"],
            form=form,
            cross_validation=cv,
            rate=rep_rate,
            epilogue=rep_epi,
            queue=rep_q,
            power=rep_pw,
            idle=dict(p_shared=idle["p_shared"], slope_w_per_mhz=idle["slope"], die_c=idle["temp"]),
        ),
    )
    path = cal.save(out)
    print(f"wrote {path}")
    return cal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--out", default=DEFAULT)
    ap.add_argument("--form", default="shipped", choices=list(FORMS), help="which parameters are pinned")
    ap.add_argument("--compare", action="store_true", help="cross-validate every form")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    run(a.model, a.out, a.verbose, a.form, a.compare)


if __name__ == "__main__":
    main()
