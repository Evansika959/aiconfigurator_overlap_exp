#!/usr/bin/env python3
"""Generate the GEMM characterization config table (A100 / SM80).

Knobs: data format x GEMM size (M,N,K) x GPU SM clock x #SMs.
Rows are emitted in SWEEP order (freq outer -> sm -> dtype -> shape) so the sweep
runs top-to-bottom while minimizing expensive clock/MPS state changes.

This is the REDUCED bf16 sweep that was actually run for the A100 DVFS
characterization (360 configs = 1 dtype x 4 M x 3 (N,K) x 6 freq x 5 SM).
Edit the knob lists below to widen it (e.g. add dtypes/shapes back).

  python3 gen_config_table.py                      # -> ./gemm_config_table.csv
  python3 gen_config_table.py --out other.csv
"""
import argparse, csv, os

DTYPES = ["bf16"]                                       # A100: NO fp8/nvfp4; reduced sweep = bf16 only
DBYTES = {"fp32": 4, "tf32": 4, "fp16": 2, "bf16": 2, "int8": 1}
M_LIST  = [1024, 2048, 4096, 8192]
NK_LIST = [(4096, 4096), (8192, 8192), (16384, 16384)]  # (N, K)
FREQS   = [1410, 1200, 900, 705, 510, 300]              # SM clock (MHz); DRAM fixed 1215 on A100
SMS     = [(108, 100.0), (81, 75.0), (54, 50.0), (27, 25.0), (14, 12.5)]  # (sm_count, mps_pct)


def build_rows():
    rows = []
    cid = 0
    for f in FREQS:                                     # outer: lock SM clock once per group
        for sm_count, sm_pct in SMS:                    # mid: one MPS %/process per group
            for dt in DTYPES:                           # inner: many GEMMs share one process
                for (N, K) in NK_LIST:
                    for M in M_LIST:
                        flop = 2 * M * N * K
                        byts = (M * K + K * N + M * N) * DBYTES[dt]
                        rows.append(dict(config_id=cid, dtype=dt, M=M, N=N, K=K,
                                         freq_mhz=f, dram_mhz=1215, sm_count=sm_count, mps_pct=sm_pct,
                                         gflop=round(flop / 1e9, 3), mbytes=round(byts / 1e6, 2),
                                         arith_intensity=round(flop / byts, 1)))
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
    print(f"knob sizes: dtype={len(DTYPES)}  shapes={len(M_LIST) * len(NK_LIST)}  "
          f"freq={len(FREQS)}  sm={len(SMS)}")
    print(f"TOTAL configs = {len(rows)}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
