#!/usr/bin/env python3
"""Brute-force search: is there ANY two-domain split that beats one clock?

I argued from Jensen that uniform frequency is optimal, but that argument assumes work is
freely divisible and throughput is linear in s*f. Real GEMMs violate the second: a tile is
indivisible and an expert on s SMs takes ceil(tiles/s) waves, a staircase. Wave
quantisation is exactly the kind of thing that breaks a smooth convexity argument, so the
claim deserves a search rather than a proof sketch.

MODEL (all pieces measured):
    a tile costs a fixed number of cycles, so
        busy time of domain d = ceil(n_d / s_d) * tau / f_d          (waves / frequency)
        dynamic energy        = n_d * kappa(f_d) * tau               (per tile, f cancels)
        leakage               = s_d * p_leak(f_d) * T                (powered until the end)
    T = max_d busy_d, and every domain stays powered until T.

kappa(f) and p_leak(f) are the measured curves (kappa_sweep.csv, idle_power.csv).
Tiles come from the real OLMoE routing: tiles_e = ceil(n_e/128) * ceil(1024/128).

SEARCH: every SM split s1 in 1..107, every pair of clocks on the measured grid, and for
each, the tile assignment that balances the two domains' finish times (the best case for
the split -- an unbalanced assignment can only be worse). Compared against one domain of
108 SMs at whichever single clock meets the same deadline.
"""
import csv, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SM = 108


def curves():
    d = {}
    for r in csv.DictReader(open(os.path.join(HERE, "data", "kappa_sweep.csv"))):
        if r["workload"] == "moe_expert_n7652" and r["held"] == "True":
            d[int(r["clock"])] = float(r["kappa_mw_per_mhz"]) / SM      # per SM
    f = np.array(sorted(d), float)
    k = np.array([d[int(x)] for x in f])
    p = {int(r["clock"]): float(r["idle_w"]) / SM
         for r in csv.DictReader(open(os.path.join(HERE, "data", "idle_power.csv")))}
    pf = np.array(sorted(p), float)
    pv = np.array([p[int(x)] for x in pf])
    return f, k, pf, pv


FK, KK, PF, PV = curves()
kap = lambda f: np.interp(f, FK, KK) / 1e3          # W per SM per MHz -> W/(SM*MHz)
leak = lambda f: np.interp(f, PF, PV)               # W per SM
# THE A100's REAL CLOCK GRID, 15 MHz steps -- not my sweep's grid, which was 45 MHz
# apart above 1155. A coarse grid hands the split a free win: it lets a mixture of two
# clocks reach an average the uniform arm cannot express, and the first version of this
# search "found" a 4.8% gain that was entirely that. On the real grid the same split
# LOSES 0.66%.
GRID = np.arange(300.0, 1381.0, 15.0)
TAU = 1.0                                            # cycles per tile; cancels everywhere


def energy_two(n_tot, s1, f1, f2, split_frac):
    """n_tot tiles, s1 SMs at f1 and 108-s1 at f2, split_frac of the tiles to domain 1."""
    s2 = SM - s1
    n1 = int(round(n_tot * split_frac)); n2 = n_tot - n1
    if n1 < 0 or n2 < 0:
        return None
    t1 = np.ceil(n1 / s1) * TAU / f1 if n1 else 0.0
    t2 = np.ceil(n2 / s2) * TAU / f2 if n2 else 0.0
    T = max(t1, t2)
    e_dyn = (n1 * kap(f1) + n2 * kap(f2)) * TAU
    e_leak = (s1 * leak(f1) + s2 * leak(f2)) * T
    return T, e_dyn + e_leak


_W = None


def best_uniform(n_tot, T):
    """Cheapest single clock over all 108 SMs that still finishes by T. Energy falls
    monotonically with f (both kappa and leakage do), so the cheapest feasible clock is
    simply the LOWEST one fast enough: f >= waves/T."""
    global _W
    if _W is None:
        _W = np.ceil(n_tot / SM) * TAU
    need = _W / (T * (1 + 1e-9))
    i = np.searchsorted(GRID, need)
    if i >= len(GRID):
        return None
    f = GRID[i]
    return (n_tot * kap(f) * TAU + SM * leak(f) * T, f, _W / f)


def main():
    Z = np.load(os.path.join(HERE, "data", "routing_olmoe.npz"))
    for key in ("prefill_b16", "prefill_b128", "decode_b64"):
        C = Z[key].reshape(-1, 64).astype(float)
        n_e = C[0]
        tiles = int((np.ceil(n_e / 128) * np.ceil(1024 / 128))[n_e > 0].sum())
        print(f"\n{key}: {int(n_e.sum())} rows -> {tiles} tiles on {SM} SMs "
              f"({tiles/SM:.1f} waves)")
        global _W
        _W = None
        wins, best_win, checked = 0, None, 0
        for s1 in range(1, SM):
            for f1 in GRID:
                for f2 in GRID:
                    # balance the finish times: give domain 1 the share its speed earns
                    fr = (s1 * f1) / (s1 * f1 + (SM - s1) * f2)
                    for frac in (fr, min(1.0, fr * 1.05), max(0.0, fr * 0.95)):
                        r = energy_two(tiles, s1, f1, f2, frac)
                        if r is None:
                            continue
                        T, E = r
                        u = best_uniform(tiles, T)
                        if u is None:
                            continue
                        checked += 1
                        gain = (E - u[0]) / u[0] * 100
                        if gain < -0.5:
                            wins += 1
                            if best_win is None or gain < best_win[0]:
                                best_win = (gain, s1, f1, f2, frac, u[1], T)
        print(f"  searched {checked} feasible (split, f1, f2, assignment) points")
        if best_win:
            g_, s1, f1, f2, fr, fu, T = best_win
            print(f"  BEST SPLIT BEATS UNIFORM by {-g_:.2f}%: "
                  f"{s1} SMs @ {f1:.0f} + {SM-s1} SMs @ {f2:.0f}, "
                  f"{fr*100:.0f}% of tiles to domain 1; uniform would be {fu:.0f} MHz")
            print(f"  splits beating uniform by >0.5%: {wins}")
        else:
            print(f"  no split beats one clock by more than 0.5% anywhere in the search")


if __name__ == "__main__":
    main()
