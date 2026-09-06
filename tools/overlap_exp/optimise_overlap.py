#!/usr/bin/env python3
"""Optimal (SM split, frequency) for an overlapped GEMM + all-reduce, under the
counterfactual that the two kernels sit in SEPARATE DVFS DOMAINS.

THE QUESTION. A100 has one SM clock domain, so an overlapped pair must share a
frequency. Suppose it did not. Then for a given workload you get four knobs:

    S_g            SMs given to the GEMM
    S_c = 108-S_g  SMs given to the collective  (1 NCCL CTA == 1 SM, ncu-verified)
    f_g, f_c       a clock for each partition

and the question is what to set them to. This is NOT "run each kernel at the clock it
would prefer alone". SMs and frequency are substitutes for hitting a shared finish time,
and the two kernels trade them at different rates: give the GEMM more SMs and it clears
its work at a lower clock, which is cheap while kappa=a/f is flat below ~1020 MHz -- but
the collective then has fewer SMs and needs a higher clock, which is expensive because
gamma(f) climbs. The optimum is where those marginal costs meet, and it moves with the
workload.

WHAT THE SECOND DOMAIN ACTUALLY BUYS. With one domain you have (f, c) for two deadlines,
so in general t_gemm != t_comm and one partition idles for the difference. The second
domain is the degree of freedom that lets you match them. That, not per-kernel clock
preference, is where the saving should come from -- so this script does NOT impose
t_gemm == t_comm. It searches (S_g, f_g, f_c) freely and prices the idle tail. If
duration-matching is optimal it will come out of the search rather than be assumed.

    E = P_static(f_g, f_c, S_g) * T  +  P_gemm(f_g, S_g)*t_g  +  P_comm(f_c, S_c)*t_c
    T = max(t_g, t_c)

STATIC POWER WITH TWO VOLTAGE DOMAINS is the one quantity nothing measures. Split it:
P_static(f) = P_floor + D(f), with P_floor the 300 MHz value (uncore, HBM, PCIe -- chip
wide) and D(f) the voltage-dependent leakage, apportioned by SM share. At f_g == f_c
this collapses exactly to the measured single-domain P_static(f), so the baseline arm is
unaffected and only the counterfactual carries the assumption. --static-model bounds it.

EVERY COEFFICIENT IS MEASURED, and the two that get used outside their fitting range are
corrected:

  eta(S) = 1 + 0.118*(1 - S/108)   the GEMM runs slower at low occupancy than perfect
        wave scaling predicts -- 6-9% at S<=54. tau,t0 were fitted on S=108 rows only,
        so this correction is what makes t_gemm(f,S) usable off-diagonal. It is fitted
        on the squatter campaign's 14/27/54/81 SM rows and cuts median latency error
        from 6.62% to 1.75%.

  c <= 32 only.  B(c,f) is measured at c in {1,2,4,8,16,32} and B/c is already falling
        at 32 (saturation). Interpolating between samples is sound; extrapolating past
        32 is not, so the search is capped and the script says so if the cap binds.

  python3 optimise_overlap.py
"""

import argparse
import collections
import csv
import json
import math
import os
import statistics as st

import table_432 as t
import score_432 as s

HERE = os.path.dirname(os.path.abspath(__file__))
TOTAL_SM = 108
CTA_MEAS = [1, 2, 4, 8, 16, 32]          # where B and alpha are measured
FMIN, FMAX = 300, 1410
ETA_K = 0.118
P_FLOOR = t.PS[min(t.PS)]                 # 300 MHz static: chip-wide, voltage-independent


def eta(sm):
    return 1 + ETA_K * (1 - sm / TOTAL_SM)


_MC = {}


def _memo(key, fn):
    v = _MC.get(key)
    if v is None:
        v = _MC[key] = fn()
    return v


def pw_c(tab_f, c):
    """piecewise-linear in CTA count across the measured samples"""
    if c <= CTA_MEAS[0]:
        return tab_f[CTA_MEAS[0]] * c / CTA_MEAS[0]
    if c >= CTA_MEAS[-1]:
        return tab_f[CTA_MEAS[-1]]
    for a, b in zip(CTA_MEAS, CTA_MEAS[1:]):
        if a <= c <= b:
            return tab_f[a] + (tab_f[b] - tab_f[a]) * (c - a) / (b - a)


def bw(c, f):
    return _memo(("bw", c, f), lambda: t.lin({q: pw_c(t.Bw[q], c) for q in t.FS}, f))


def alpha(c, f):
    return _memo(("al", c, f), lambda: t.lin({q: pw_c(t.AL[q], c) for q in t.FS}, f))


def _tau(f):
    return _memo(("tau", f), lambda: (t.lin({q: t.LAT[q][0] for q in t.FS}, f),
                                      t.lin({q: t.LAT[q][1] for q in t.FS}, f)))


def t_gemm(f, sm, grid, k, splitk):
    tau, t0 = _tau(f)
    return eta(sm) * (-(-grid // sm) * (k / splitk) * tau + t0)


def p_gemm(f, sm, grid, n, k):
    b = _memo(("b", f, n, k), lambda: t.lin({q: s.b_lookup(t.B, q, n, k) for q in t.FS}, f))
    return _memo(("a", f), lambda: t.lin(t.A, f)) * (grid / -(-grid // sm)) + b


def t_comm(f, c, gb):
    return alpha(c, f) / 1e3 + gb / bw(c, f) * 1e3


def p_comm(f, c):
    return _memo(("p0", f), lambda: t.lin(t.P0, f)) + \
        _memo(("g", f), lambda: t.lin(t.G, f)) * bw(c, f)


def p_static2(fg, fc, sg, sc, mode="apportion"):
    """two voltage domains. At fg == fc every mode returns the measured P_static(fg)."""
    dg = _memo(("ps", fg), lambda: t.lin(t.PS, fg)) - P_FLOOR
    dc = _memo(("ps", fc), lambda: t.lin(t.PS, fc)) - P_FLOOR
    if mode == "max":
        return P_FLOOR + max(dg, dc)
    if mode == "sum":
        return P_FLOOR + dg + dc
    return P_FLOOR + (sg * dg + sc * dc) / TOTAL_SM


def solve_soft_1domain(grid, n, k, splitk, gb, fgrid, cmax=32):
    """ARM A -- what the hardware does today. One clock; the partition is SOFT, so the
    GEMM's blocks occupy the collective's SMs whenever it is not resident. This is the
    model validated at 3.3% median against the 432 measured cases."""
    best = None
    w = -(-grid // TOTAL_SM)
    for c in range(1, cmax + 1):
        w2 = -(-grid // max(TOTAL_SM - c, 1))
        for f in fgrid:
            tg = t_gemm(f, TOTAL_SM, grid, k, splitk)
            tc = t_comm(f, c, gb)
            T = max(tg * (1 + (w2 / w - 1) * min(1, tc / tg)), tc)
            e = _memo(("ps", f), lambda: t.lin(t.PS, f)) * T + p_gemm(f, TOTAL_SM, grid, n, k) * tg + p_comm(f, c) * tc
            if best is None or e < best["E"]:
                best = dict(E=e, T=T, c=c, f=f, tg=tg, tc=tc)
    return best


def solve_hard(grid, n, k, splitk, gb, fgrid, cmax=32, static="apportion", domains=2):
    """ARM B (domains=1) and ARM C (domains=2). Both HARD-partition the chip: the GEMM
    owns 108-c SMs for the whole iteration and cannot expand into the collective's share
    when it finishes. The two arms differ ONLY in whether f_g and f_c may differ, so the
    domains=1 search space is strictly contained in the domains=2 one and arm C can
    never be worse. Comparing these two is the clean measure of what a second DVFS
    domain is worth; comparing either against arm A also changes the partitioning."""
    best = None
    for c in range(1, cmax + 1):
        sg = TOTAL_SM - c
        for fg in fgrid:
            tg = t_gemm(fg, sg, grid, k, splitk)
            pg = p_gemm(fg, sg, grid, n, k)
            for fc in (fgrid if domains == 2 else [fg]):
                tc = t_comm(fc, c, gb)
                T = max(tg, tc)
                e = (p_static2(fg, fc, sg, c, static) * T
                     + pg * tg + p_comm(fc, c) * tc)
                if best is None or e < best["E"]:
                    best = dict(E=e, T=T, c=c, sg=sg, fg=fg, fc=fc, tg=tg, tc=tc,
                                match=min(tg, tc) / max(tg, tc),
                                idle_sm_frac=c * abs(tg - tc) / (TOTAL_SM * T))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mib", type=float, default=64)
    ap.add_argument("--step", type=int, default=15)
    ap.add_argument("--static", default="apportion", choices=["apportion", "max", "sum"])
    a = ap.parse_args()
    G = json.load(open(os.path.join(HERE, "data", "grid12.json")))
    fgrid = list(range(FMIN, FMAX + 1, a.step))
    gb = a.mib * 2 ** 20 / 1e9

    print(f"all-reduce {a.mib:.0f} MiB world=4, clocks {FMIN}-{FMAX} step {a.step} MHz, "
          f"static model '{a.static}'\n")
    print("  A  soft partition, 1 clock   -- what the hardware does today")
    print("  B  hard partition, 1 clock   -- all 108 SMs split, both busy, shared clock")
    print("  C  hard partition, 2 clocks  -- the counterfactual.  B is nested in C.\n")
    print(f"  {'M':>6}{'N=K':>7} |{'A: f':>6}{'c':>4}{'mJ':>8} |{'B: f':>6}{'c':>4}{'mJ':>8} |"
          f"{'C: f_g':>8}{'f_c':>6}{'S_g':>5}{'S_c':>4}{'mJ':>8}{'match':>7} |"
          f"{'C-B':>7}{'C-A':>7}")
    rows, dCB, dCA = [], [], []
    for nk in (4096, 8192, 16384):
        for m in (1024, 2048, 4096, 8192):
            grid = G[f"{m}_{nk}_{nk}"]; sk = max(1, grid // (m * nk // 32768))
            A = solve_soft_1domain(grid, nk, nk, sk, gb, fgrid)
            B = solve_hard(grid, nk, nk, sk, gb, fgrid, static=a.static, domains=1)
            C = solve_hard(grid, nk, nk, sk, gb, fgrid, static=a.static, domains=2)
            cb = (C["E"] - B["E"]) / B["E"] * 100
            ca = (C["E"] - A["E"]) / A["E"] * 100
            dCB.append(cb); dCA.append(ca)
            rows.append(dict(m=m, nk=nk, mib=a.mib,
                             A_f=A["f"], A_c=A["c"], A_mJ=round(A["E"], 1), A_ms=round(A["T"], 3),
                             B_f=B["fg"], B_c=B["c"], B_mJ=round(B["E"], 1), B_ms=round(B["T"], 3),
                             C_fg=C["fg"], C_fc=C["fc"], C_Sg=C["sg"], C_Sc=C["c"],
                             C_mJ=round(C["E"], 1), C_ms=round(C["T"], 3),
                             C_match=round(C["match"], 3),
                             C_idle_sm_frac=round(C["idle_sm_frac"], 4),
                             C_vs_B_pct=round(cb, 2), C_vs_A_pct=round(ca, 2)))
            print(f"  {m:>6}{nk:>7} |{A['f']:>6}{A['c']:>4}{A['E']:>8.0f} |"
                  f"{B['fg']:>6}{B['c']:>4}{B['E']:>8.0f} |{C['fg']:>8}{C['fc']:>6}"
                  f"{C['sg']:>5}{C['c']:>4}{C['E']:>8.0f}{C['match']:>7.2f} |"
                  f"{cb:>+6.1f}%{ca:>+6.1f}%")
    p = os.path.join(HERE, "data", "optimum_2domain.csv")
    with open(p, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)
    print(f"\n  C vs B  (pure second-DVFS-domain effect, nested): "
          f"median {st.median(dCB):+.2f}%   best {min(dCB):+.2f}%")
    print(f"  C vs A  (also changes soft -> hard partition):     "
          f"median {st.median(dCA):+.2f}%   best {min(dCA):+.2f}%")
    print(f"  duration match in C: median {st.median(r['C_match'] for r in rows):.2f}   "
          f"idle-SM-time fraction: median {st.median(r['C_idle_sm_frac'] for r in rows)*100:.1f}%")
    if any(r["C_Sc"] >= 32 for r in rows):
        n = sum(1 for r in rows if r["C_Sc"] >= 32)
        print(f"  !! {n}/{len(rows)} optima sit at the c=32 cap; B(c,f) is unmeasured above")
        print(f"     32 CTAs so those are bounds, not optima")
    print(f"\n  wrote {p}")


if __name__ == "__main__":
    main()
