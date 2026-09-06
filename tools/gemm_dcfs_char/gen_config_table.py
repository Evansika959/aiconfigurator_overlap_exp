#!/usr/bin/env python3
"""Generate the GEMM DVFS x SM-count config table (A100 / SM80).

SM count is set by an SM-squatter kernel (see squatter.py), not MPS: n sleeping
blocks each hold one SM, so the GEMM sees 108 - n. The n values below are chosen so
the GEMM's available-SM counts are exactly 108/81/54/27/14 -- the same axis the
earlier MPS-based sweep used, so the two datasets overlay directly.

Rows are emitted in SWEEP order (freq outer -> n_squat -> shape) so the run walks
top-to-bottom while minimising clock changes, which are the expensive state change.

  python3 gen_config_table.py                 # -> ./gemm_config_table.csv
"""

import argparse
import csv
import os

TOTAL_SM = 108
DTYPE = "bfloat16"
M_LIST = [1024, 2048, 4096, 8192]
NK_LIST = [(4096, 4096), (8192, 8192), (16384, 16384)]
FREQS = [1410, 1200, 900, 705, 510, 300]
# The 900-1200 gap is where every throttled overlap row lands, where the energy optimum
# sits, and where P_static's two-point interpolation was measured to be 6% off and
# CONVEX rather than linear. kappa = a/f is flat to 0.5% below 900 and +33% by 1200, so
# the voltage knee is inside a span with no calibration points at all. GEMM_FREQS fills
# it without disturbing the original grid, which stays the default so existing data
# regenerates identically.
if os.environ.get("GEMM_FREQS"):
    FREQS = [int(x) for x in os.environ["GEMM_FREQS"].split(",")]
# (squatter blocks, SMs left to the GEMM)
SQUAT = [(0, 108), (27, 81), (54, 54), (81, 27), (94, 14)]


def build_rows():
    rows, cid = [], 0
    for f in FREQS:
        for n_sq, sm_avail in SQUAT:
            for (n, k) in NK_LIST:
                for m in M_LIST:
                    rows.append(dict(
                        config_id=cid, dtype=DTYPE, M=m, N=n, K=k,
                        freq_mhz=f, dram_mhz=1215,
                        n_squat=n_sq, sm_avail=sm_avail,
                        gflop=round(2 * m * n * k / 1e9, 3),
                    ))
                    cid += 1
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "gemm_config_table.csv"))
    a = ap.parse_args()
    rows = build_rows()
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"shapes={len(M_LIST) * len(NK_LIST)}  freq={len(FREQS)}  sm_points={len(SQUAT)}")
    print(f"TOTAL configs = {len(rows)}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
