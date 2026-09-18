#!/usr/bin/env python3
"""Predicted saving of expert-to-frequency-domain routing, against a grouped GEMM.

THE SCHEME. Sort the experts by how many tokens the router sent them. Give the light
experts to a low-frequency region of the chip and the heavy ones to a high-frequency
region (optionally a third region in between). Every region must finish by the same
deadline, so this is a pure energy play: the light experts have slack, and slack spent as
a lower clock is cheaper than slack spent idling.

BASELINE. A standard grouped GEMM: one kernel, all 108 SMs, one clock, work-conserving
across every expert's tiles. Its deadline is ceil(N_total/108) waves at that clock, and
for a fair comparison it is allowed the CHEAPEST clock that still meets whatever deadline
the multi-domain design is being held to.

MODEL, every coefficient measured on this GPU:
    tiles of expert e     ceil(n_e/128) * ceil(1024/128)     (M x 2048) @ (2048 x 1024)
    time of a domain      ceil(N_d / s_d) / f_d              waves over frequency
    dynamic energy        N_d * kappa(f_d)                   per tile; the f in P=kappa*f
                                                             cancels against the time
    leakage               s_d * P_leak(f_d) * T              every SM powered until the end
    kappa   from data/kappa_sweep.csv  (moe_expert_n7652, whole-GPU, per SM)
    P_leak  from data/idle_power.csv
    clocks  the A100's real 15 MHz grid -- a coarse grid manufactures fake gains

NO SEARCH OVER FREQUENCIES IS NEEDED. Given a deadline T and an assignment, each domain's
cheapest feasible clock is forced: the lowest grid clock at or above waves_d / T. So the
search is only over how many experts and how many SMs each domain gets.
"""
import argparse, csv, itertools, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SM, BM, BN = 108, 128, 128
NCOL = 1024                 # expert intermediate width; per model, see --ncol
GRID = np.arange(300.0, 1381.0, 15.0)


def _curves():
    d = {}
    for r in csv.DictReader(open(os.path.join(HERE, "data", "kappa_sweep.csv"))):
        if r["workload"] == "moe_expert_n7652" and r["held"] == "True":
            d[int(r["clock"])] = float(r["kappa_mw_per_mhz"]) / SM
    f = np.array(sorted(d), float)
    k = np.array([d[int(x)] for x in f])
    p = {int(r["clock"]): float(r["idle_w"]) / SM
         for r in csv.DictReader(open(os.path.join(HERE, "data", "idle_power.csv")))}
    pf = np.array(sorted(p), float)
    return f, k, pf, np.array([p[int(x)] for x in pf])


FK, KK, PF, PV = _curves()
KAP = np.interp(GRID, FK, KK) / 1e3          # W per SM per MHz, on the clock grid
LEAK = np.interp(GRID, PF, PV)               # W per SM


def snap_up(need):
    """Cheapest feasible clock: the lowest grid point at or above `need`."""
    i = np.searchsorted(GRID, need - 1e-9)
    return i if i < len(GRID) else -1


def tiles_of(n_e, ncol=None):
    return (np.ceil(n_e / BM) * np.ceil((ncol or NCOL) / BN)).astype(int)


def baseline(N, f_idx, T=None):
    """Grouped GEMM: 108 SMs, one clock, work-conserving. Returns (T, energy)."""
    w = np.ceil(N / SM)
    t = w / GRID[f_idx]
    if T is None:
        T = t
    return T, N * KAP[f_idx] + SM * LEAK[f_idx] * T


def _cost_vec(N, T):
    """Cost of putting N tiles on s SMs, for every s = 1..108, at deadline T.
    Vectorised: the clock is forced (lowest grid point >= waves/T), so there is nothing
    to search -- which is what makes the whole thing cheap enough to sweep."""
    s = np.arange(1, SM + 1, dtype=float)
    waves = np.ceil(N / s)
    j = np.searchsorted(GRID, waves / T - 1e-9)
    bad = j >= len(GRID)
    j = np.clip(j, 0, len(GRID) - 1)
    c = N * KAP[j] + s * LEAK[j] * T
    c[bad] = np.inf
    return c, GRID[j]


def best_multi(tiles_sorted, T, ndom, sm_step=1):
    """Cheapest contiguous grouping of load-sorted experts into `ndom` domains.

    For a fixed grouping the SM allocation is a small resource-distribution problem, and
    cost_d(s) is a vector, so it is solved by min-plus over vectors instead of a nested
    loop. That turns an O(cuts x SM^2) python search into O(cuts x SM^2) numpy, which is
    the difference between minutes and hours.
    """
    E = len(tiles_sorted)
    cum = np.concatenate([[0], np.cumsum(tiles_sorted)])
    best = None
    cuts = ([(c,) for c in range(1, E)] if ndom == 2
            else list(itertools.combinations(range(1, E), 2)))
    for c in cuts:
        bounds = (0,) + tuple(c) + (E,)
        Nd = [cum[bounds[i + 1]] - cum[bounds[i]] for i in range(ndom)]
        if any(x == 0 for x in Nd):
            continue
        cv = [_cost_vec(n, T) for n in Nd]
        if ndom == 2:
            c0, c1 = cv[0][0], cv[1][0]
            tot = c0[:SM - 1] + c1[SM - 2::-1]          # s1 = 1..107, s2 = 108 - s1
            i = int(np.argmin(tot))
            if not np.isfinite(tot[i]):
                continue
            e = float(tot[i]); split = (i + 1, SM - i - 1)
            fs = (cv[0][1][i], cv[1][1][SM - i - 2])
        else:
            c0, c1, c2 = cv[0][0], cv[1][0], cv[2][0]
            # tail[k] = cheapest way to give k SMs to domains 2 and 3 together
            best_t = np.full(SM + 1, np.inf); arg_t = np.zeros(SM + 1, int)
            for s2 in range(1, SM):
                k = np.arange(s2 + 1, SM + 1)
                v = c1[s2 - 1] + c2[k - s2 - 1]
                m = v < best_t[k]
                best_t[k[m]] = v[m]; arg_t[k[m]] = s2
            s1r = np.arange(1, SM - 1)
            tot = c0[s1r - 1] + best_t[SM - s1r]
            i = int(np.argmin(tot))
            if not np.isfinite(tot[i]):
                continue
            s1 = int(s1r[i]); s2 = int(arg_t[SM - s1]); s3 = SM - s1 - s2
            e = float(tot[i]); split = (s1, s2, s3)
            fs = (cv[0][1][s1 - 1], cv[1][1][s2 - 1], cv[2][1][s3 - 1])
        if best is None or e < best[0]:
            best = (e, bounds, split, fs)
    return best


_CACHE = {}


def _sm_splits(ndom, step):
    key = (ndom, step)
    if key not in _CACHE:
        if ndom == 2:
            v = [(a, SM - a) for a in range(step, SM, step)]
        else:
            v = [(a, b, SM - a - b) for a in range(step, SM, step)
                 for b in range(step, SM - a, step) if SM - a - b >= step]
        _CACHE[key] = v
    return _CACHE[key]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sm-step", type=int, default=1)
    ap.add_argument("--npz", default=os.path.join(HERE, "data", "routing_olmoe.npz"))
    ap.add_argument("--ncol", type=int, default=1024,
                    help="expert intermediate width: OLMoE 1024, Qwen3-30B-A3B 768")
    ap.add_argument("--experts", type=int, default=64)
    ap.add_argument("--out", default=os.path.join(HERE, "data", "expert_dvfs_pred.csv"))
    a = ap.parse_args()
    Z = np.load(a.npz)
    rows = []
    # iterate over whatever the trace actually contains: batch sizes differ per model
    keys = sorted((k for k in Z.files if k.startswith(("prefill_b", "decode_b"))),
                  key=lambda k: (k.split("_b")[0], int(k.split("_b")[1])))
    for key in keys:
        if int(key.split("_b")[1]) < 8:      # B<8 gives too few tokens to be meaningful
            continue
        C = Z[key].reshape(-1, a.experts)
        print(f"\n{key}   ({C.shape[0]} layer-samples)")
        print(f"  {'baseline clock':>15}{'waves':>7}{'2 domains':>12}{'3 domains':>12}"
              f"   example 3-domain design")
        for fb in (1380, 1290, 1200, 1095, 1005, 900, 795, 690):
            j = int(np.searchsorted(GRID, fb))
            g2 = g3 = 0.0
            n = 0
            ex = ""
            for row in C[:4]:
                t = tiles_of(row.astype(float), a.ncol)
                t = np.sort(t[t > 0])
                N = int(t.sum())
                T, Eb = baseline(N, j)
                b2 = best_multi(t, T, 2, a.sm_step)
                b3 = best_multi(t, T, 3)
                if b2 is None or b3 is None:
                    continue
                g2 += (b2[0] - Eb) / Eb * 100
                g3 += (b3[0] - Eb) / Eb * 100
                n += 1
                if not ex:
                    bd, sp, fs = b3[1], b3[2], b3[3]
                    ex = (f"{bd[1]}/{bd[2]-bd[1]}/{len(t)-bd[2]} experts, "
                          f"{sp[0]}/{sp[1]}/{sp[2]} SMs, "
                          f"{fs[0]:.0f}/{fs[1]:.0f}/{fs[2]:.0f} MHz")
            if not n:
                continue
            w = int(np.ceil(tiles_of(C[0].astype(float), a.ncol)[C[0] > 0].sum() / SM))
            print(f"  {fb:>15}{w:>7}{g2/n:>+11.2f}%{g3/n:>+11.2f}%   {ex}")
            rows.append(dict(workload=key, baseline_clock=fb, waves=w,
                             gain2_pct=round(g2 / n, 3), gain3_pct=round(g3 / n, 3),
                             example3=ex, n_samples=n))
    with open(a.out, "w", newline="") as fh:
        w_ = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w_.writeheader(); w_.writerows(rows)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
