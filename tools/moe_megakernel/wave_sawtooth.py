#!/usr/bin/env python3
"""WHERE a two-domain split beats one clock, and why. The user was right that a case exists.

MECHANISM. A uniform design over S SMs must run ceil(N/S) waves for N tiles. When N is
just above a multiple of S it pays a WHOLE extra wave for a handful of tiles, and to meet
a deadline it must raise the clock for all of them. A split can carve off a small domain
that absorbs the remainder, leaving the big domain an exact number of waves -- so the big
domain, holding almost all the work, runs far slower and still finishes on time.

Example found by search, 325 tiles on 108 SMs at a deadline of 4 waves / 1380 MHz:
    uniform   ceil(325/108) = 4 waves  -> every tile at 1380 MHz
    split     4 tiles on 1 SM @ 1380 (4 waves) + 321 tiles on 107 SMs @ 1050 (exactly 3
              waves) -> 99% of the tiles run at 1050, and 1050 is below the knee
    -35.7% energy at the same deadline.

So the gain is a SAWTOOTH in the tile count, peaking just above each multiple of 108 and
vanishing at the multiples themselves. It is largest when the wave count is small, because
one wasted wave out of four matters and one out of three hundred does not.

This is a scheduling-granularity argument, not a voltage argument: kappa being flat below
the knee is what makes the big domain's slower clock free, but the OPPORTUNITY comes from
wave quantisation.
"""
import os
import numpy as np
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from search_split import kap, leak, SM, GRID

HERE = os.path.dirname(os.path.abspath(__file__))
FS = GRID[GRID >= 600]                      # candidate domain clocks


def best_split(N):
    """Cheapest two-domain design and the cheapest uniform meeting the SAME deadline."""
    best = (0.0, None)
    W = np.ceil(N / SM)
    F1, F2 = np.meshgrid(FS, FS, indexing="ij")
    for s1 in range(1, SM):
        s2 = SM - s1
        fr = (s1 * F1) / (s1 * F1 + s2 * F2)
        for m in (1.0, 1.05, 0.95, 1.12, 0.88):
            n1 = np.round(N * np.clip(fr * m, 0, 1)).astype(int)
            n2 = N - n1
            ok = (n1 >= 0) & (n2 >= 0)
            w1 = np.where(n1 > 0, np.ceil(np.maximum(n1, 1) / s1), 0.0)
            w2 = np.where(n2 > 0, np.ceil(np.maximum(n2, 1) / s2), 0.0)
            T = np.maximum(w1 / F1, w2 / F2)
            need = W / (T * (1 + 1e-9))
            i = np.searchsorted(GRID, need)
            ok &= (T > 0) & (i < len(GRID))
            i = np.clip(i, 0, len(GRID) - 1)
            fu = GRID[i]
            Eu = N * kap(fu) + SM * leak(fu) * T
            Es = n1 * kap(F1) + n2 * kap(F2) + (s1 * leak(F1) + s2 * leak(F2)) * T
            g = np.where(ok, (Es - Eu) / Eu * 100, 0.0)
            j = np.unravel_index(np.argmin(g), g.shape)
            if g[j] < best[0]:
                best = (float(g[j]), (s1, float(F1[j]), float(F2[j]),
                                      int(n1[j]), float(fu[j]), float(T[j])))
    return best


if __name__ == "__main__":
    import csv
    Ns = list(range(110, 660, 5))
    rows = []
    for N in Ns:
        g, d = best_split(N)
        rows.append(dict(tiles=N, waves=int(np.ceil(N / SM)),
                         rem=(N - 1) % SM + 1, gain_pct=round(g, 3),
                         s1=d[0] if d else "", f1=d[1] if d else "",
                         f2=d[2] if d else "", n1=d[3] if d else "",
                         f_uniform=d[4] if d else ""))
        print(f"  N={N:4d}  waves {rows[-1]['waves']}  gain {g:+7.2f}%"
              + (f"   {d[0]}@{d[1]:.0f} + {SM-d[0]}@{d[2]:.0f}, "
                 f"{d[3]} tiles to d1, uniform {d[4]:.0f}" if d else ""), flush=True)
    with open(os.path.join(HERE, "data", "wave_sawtooth.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("\nwrote data/wave_sawtooth.csv")
