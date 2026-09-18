#!/usr/bin/env python3
"""Does the wave/tile model apply, and would a smaller tile help?

WHY THIS EXISTS. Every prediction in predict_expert_dvfs.py rests on a model in which a
layer's cost is `ceil(tiles/SMs)` waves of compute. That model is only meaningful if the
work is COMPUTE bound. This script checks that per regime, and the answer splits the
project in two.

Then, for the regimes where the model does apply, it asks what changing the tile's M
dimension would do -- since the routing distribution shows the last tile-row is half empty
on average, a smaller tile is the obvious software-only alternative to a DVFS scheme.

  python3 roofline_and_tiles.py
"""
import csv, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
H, I = 2048, 1024                  # OLMoE: hidden, expert intermediate
PEAK, BW = 312e12, 1555e9          # A100 40GB: bf16 tensor core FLOP/s, HBM byte/s
RIDGE = PEAK / BW
SM = 108


def main():
    Z = np.load(os.path.join(HERE, "data", "routing_olmoe.npz"))
    rows = []
    print(f"1. ROOFLINE -- is the expert GEMM compute or memory bound?")
    print(f"   A100 ridge point = {RIDGE:.0f} FLOP/byte\n")
    print(f"  {'regime':<16}{'median n_e':>12}{'intensity':>12}{'verdict':>16}")
    for key in ("prefill_b128", "prefill_b16", "decode_b128", "decode_b64", "decode_b16"):
        if key not in Z.files:
            continue
        C = Z[key].reshape(-1, 64).astype(float)
        ne = float(np.median(C[C > 0]))
        flop = 2 * ne * H * I
        byts = H * I * 2 + ne * (H + I) * 2
        ai = flop / byts
        print(f"  {key:<16}{ne:>12.0f}{ai:>12.1f}"
              f"{'COMPUTE bound' if ai > RIDGE else 'MEMORY bound':>16}")
        rows.append(dict(kind="roofline", regime=key, median_ne=round(ne, 1),
                         intensity=round(ai, 2), ridge=round(RIDGE, 1),
                         verdict="compute" if ai > RIDGE else "memory"))

    print("\n2. THE WHOLE LAYER IN DECODE -- weights dominate\n")
    for key in ("decode_b16", "decode_b64", "decode_b128"):
        if key not in Z.files:
            continue
        C = Z[key].reshape(-1, 64).astype(float)
        act = float((C > 0).sum(1).mean())
        wt_b = act * 3 * H * I * 2                       # bytes of weights per layer
        fl = float(np.median(C.sum(1))) * 3 * 2 * H * I  # FLOP per layer
        t_mem, t_math = wt_b / BW * 1e6, fl / PEAK * 1e6
        print(f"  {key:<14}{act:>5.0f} active experts   weights {wt_b/1e6:>6.0f} MB "
              f"-> {t_mem:>6.0f} us of HBM   math {t_math:>5.1f} us   "
              f"ratio {t_mem/t_math:>5.1f}x")
        rows.append(dict(kind="layer_decode", regime=key, active_experts=round(act, 1),
                         weight_mb=round(wt_b / 1e6, 1), t_mem_us=round(t_mem, 1),
                         t_math_us=round(t_math, 2), mem_over_math=round(t_mem / t_math, 1)))

    print("\n3. TILE SIZE -- only meaningful where the model applies, i.e. prefill\n")
    print("   Counting TILES across different tile sizes is meaningless: a 64-row tile")
    print("   does half the work of a 128-row one. The comparable quantity is SM-time,")
    print("   proportional to waves x BM. And a smaller BM re-reads the whole weight")
    print("   matrix more often. NOTE the weight column is an UPPER BOUND: it assumes")
    print("   no reuse at all, whereas a 4.2 MB expert weight matrix fits easily in the")
    print("   A100's 40 MB L2, so a real kernel re-reads L2, not HBM. The intensity")
    print("   column inherits that pessimism -- treat it as a floor, not a verdict.\n")
    print(f"  {'regime':<15}{'tile M':>7}{'waves':>7}{'SM-time':>9}{'wasted rows':>12}"
          f"{'weight traffic':>15}{'intensity':>11}")
    for key in ("prefill_b16", "prefill_b128"):
        C = Z[key].reshape(-1, 64).astype(float)
        base = None
        for BM in (128, 64, 32, 16):
            t = np.ceil(C / BM) * np.ceil(I / 128)
            waves = np.ceil(t.sum(1) / SM)
            smtime = float((waves * BM).mean())          # proportional to real SM-time
            wr = float((((np.ceil(C / BM) * BM - C) * (C > 0)).sum(1)
                        / C.sum(1) * 100).mean())
            # every row-block of an expert re-reads that expert's whole weight matrix
            wt = float((np.ceil(C / BM) * (H * I * 2) * (C > 0)).sum(1).mean())
            flop = float((2 * C * H * I).sum(1).mean())
            acts = float(((C * (H + I) * 2) * (C > 0)).sum(1).mean())
            ai = flop / (wt + acts)
            if base is None:
                base = smtime
            print(f"  {key:<15}{BM:>7}{waves.mean():>7.0f}{(smtime-base)/base*100:>+8.1f}%"
                  f"{wr:>11.1f}%{wt/1e6:>13.0f} MB{ai:>11.0f}")
            rows.append(dict(kind="tile", regime=key, tile_m=BM,
                             waves=round(float(waves.mean())),
                             smtime_vs_128_pct=round((smtime - base) / base * 100, 2),
                             wasted_rows_pct=round(wr, 2),
                             weight_mb_upper=round(wt / 1e6, 1),
                             intensity_lower=round(ai, 1)))
    p = os.path.join(HERE, "data", "roofline_tiles.csv")
    ks = sorted({k for r in rows for k in r})
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=ks)
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
