#!/usr/bin/env python3
"""What does throughput cost in power ABOVE 1290 MHz? Measured, not extrapolated.

WHY. predict_tdp.py decides whether a split can beat the uniform clock under the 400 W
limit, and the answer turns on one number: the marginal power cost of throughput above
the operating point, m in W per (layer/s). At full width nothing above 1290 MHz can be
measured -- locked at 1335/1380/1410 the kernel hits 400 W and settles near 1320. Two
estimates of m disagreed: the last full-width segment (1290 -> the governor point) gave
7.37 with per-pass values from 5.5 to 17.6, and a GEMM-derived voltage curve gave ~2.6.

THE MEASUREMENT. partition_kappa.py --sms 54 --all-sms 54: the whole layer, and its
light and heavy partitions, each alone on 54 SMs, locked from 1050 to 1410 MHz. Half
the SMs keep the board under the limit, so every clock holds.

THREE CHECKS, each against data that did not build it
  1. Is the 54-SM curve's SHAPE the full-width shape? Both are measured over 1050-1290;
     only if they agree may the 54-SM shape carry the full-width curve above 1290.
  2. Was the GEMM-voltage extrapolation right? E_dyn = k1 g(f) + k2, fitted on the
     full-width floor and 1200 MHz, against the 54-SM measurement above 1290.
  3. Does the reconstructed full-width curve land on the governor? At ~1320 MHz it
     should read ~400 W and the governor's throughput. The governor point is not used
     to build it.
Only then is m read off, 1322 -> 1410.
"""
import csv, os
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GEMM_KAPPA = {900: 1.660, 1200: 2.222, 1410: 3.136}   # gemm_dcfs_char/MODEL.md, a(f)/f


def load(fn, key_part=None):
    by = defaultdict(list)
    for r in csv.DictReader(open(fn)):
        if key_part and r["part"] != key_part:
            continue
        if r.get("held", "True") != "True":
            continue
        by[int(r["clock"])].append(r)
    return by


def full_width_curve(ref=1290):
    """The full-width layer from 510 to 1410 MHz: measured up to `ref`, and above it
    rebuilt from the 54-SM sweep's shape -- the construction check 3 validates against
    the governor. Returns (clocks, T ms, E_dyn mJ, idle W, rebuilt flag), in the units
    predict_two_domain.load_sweep() uses, so a consumer can swap one for the other."""
    full = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(HERE, "data", "b1_energy_qwen.csv"))):
        full[int(r["clock"])].append(r)
    mf = lambda c, k: float(np.mean([float(x[k]) for x in full[c]]))
    fh = [c for c in sorted(full) if c > 0 and abs(c - mf(c, "clock_med")) <= 7]
    hi = load(os.path.join(HERE, "data", "partition_kappa_qwen_hiclk54.csv"), "all")
    m54 = lambda c, k: float(np.mean([float(x[k]) for x in hi[c]]))
    off = m54(ref, "idle_w") - mf(ref, "idle_w")         # cross-campaign idle offset
    rows = [(c, mf(c, "ms_per_layer"), mf(c, "e_dyn_mj_per_layer"), mf(c, "idle_w"),
             False) for c in fh]
    for c in sorted(hi):
        if c > ref:
            rows.append((c, mf(ref, "ms_per_layer") * m54(c, "ms") / m54(ref, "ms"),
                         mf(ref, "e_dyn_mj_per_layer") * m54(c, "e_dyn_mj")
                         / m54(ref, "e_dyn_mj"),
                         m54(c, "idle_w") - off, True))
    return tuple(np.array(v) for v in zip(*rows))


def main():
    hi = load(os.path.join(HERE, "data", "partition_kappa_qwen_hiclk54.csv"), "all")
    raw = list(csv.DictReader(open(os.path.join(HERE, "data",
                                                 "partition_kappa_qwen_hiclk54.csv"))))
    nh = [(r["part"], r["clock"], r["clock_med"]) for r in raw if r["held"] != "True"]
    print(f"54-SM sweep: {len(raw)} rows, {len(nh)} not held" +
          (f": {nh[:6]}" if nh else ""))
    m54 = lambda c, k: float(np.mean([float(x[k]) for x in hi[c]]))
    s54 = lambda c, k: float(np.std([float(x[k]) for x in hi[c]], ddof=1))
    cl = sorted(hi)

    full = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(HERE, "data", "b1_energy_qwen.csv"))):
        full[int(r["clock"])].append(r)
    mf = lambda c, k: float(np.mean([float(x[k]) for x in full[c]]))
    fheld = [c for c in sorted(full) if c > 0 and abs(c - mf(c, "clock_med")) <= 7]

    # ---- per-clock table ---------------------------------------------------------
    print(f"\n{'MHz':>5} {'ms':>8} {'board W':>8} {'idle W':>7} {'P_dyn W':>8} "
          f"{'E_dyn mJ':>9} {'sd':>5}")
    for c in cl:
        print(f"{c:>5} {m54(c, 'ms'):>8.4f} {m54(c, 'power_w'):>8.1f} "
              f"{m54(c, 'idle_w'):>7.1f} {m54(c, 'p_dyn_w'):>8.1f} "
              f"{m54(c, 'e_dyn_mj'):>9.3f} {s54(c, 'e_dyn_mj') / m54(c, 'e_dyn_mj'):>5.1%}")

    # ---- check 1: shape, 54 SM vs full width, where both are measured -------------
    ref = 1290
    print(f"\ncheck 1 -- shape relative to {ref} MHz, where both widths are measured")
    print(f"{'MHz':>5} {'T 54SM':>7} {'T full':>7} {'Edyn 54SM':>9} {'Edyn full':>9}")
    worst_t = worst_e = 0.0
    for c in cl:
        if c > ref or c not in fheld:
            continue
        t54 = m54(c, "ms") / m54(ref, "ms"); tf = mf(c, "ms_per_layer") / mf(ref, "ms_per_layer")
        e54 = m54(c, "e_dyn_mj") / m54(ref, "e_dyn_mj")
        ef = mf(c, "e_dyn_mj_per_layer") / mf(ref, "e_dyn_mj_per_layer")
        worst_t, worst_e = max(worst_t, abs(t54 / tf - 1)), max(worst_e, abs(e54 / ef - 1))
        print(f"{c:>5} {t54:>7.4f} {tf:>7.4f} {e54:>9.4f} {ef:>9.4f}")
    print(f"  worst disagreement: time shape {worst_t:.2%}, dynamic-energy shape {worst_e:.2%}")

    # ---- check 2: the GEMM-voltage extrapolation ----------------------------------
    g = lambda f: 1.0 if f <= 900 else float(np.interp(
        f, [900, 1200, 1410], [1.0, GEMM_KAPPA[1200] / GEMM_KAPPA[900],
                               GEMM_KAPPA[1410] / GEMM_KAPPA[900]]))
    Ef = lambda c: mf(c, "e_dyn_mj_per_layer")
    Elo = np.mean([Ef(c) for c in fheld if c <= 900])
    k1 = (Ef(1200) - Elo) / (g(1200) - 1.0); k2 = Elo - k1
    print(f"\ncheck 2 -- GEMM-voltage model E_dyn = {k1:.0f} g(f) + {k2:.0f}, "
          f"as a ratio to {ref} MHz, against the 54-SM measurement")
    for c in cl:
        if c <= ref:
            continue
        pred = (k1 * g(c) + k2) / (k1 * g(ref) + k2)
        meas = m54(c, "e_dyn_mj") / m54(ref, "e_dyn_mj")
        print(f"  {c:>5}: predicted x{pred:.4f}  measured x{meas:.4f}  ({pred / meas - 1:+.2%})")

    # ---- check 3 and m: reconstruct full width above 1290 from the 54-SM shape --
    Xf0, Ef0 = 1000 / mf(ref, "ms_per_layer"), mf(ref, "e_dyn_mj_per_layer")
    idle_ref = m54(ref, "idle_w") - mf(ref, "idle_w")      # cross-campaign idle offset

    def full_at(c):
        X = Xf0 * m54(ref, "ms") / m54(c, "ms")
        E = Ef0 * m54(c, "e_dyn_mj") / m54(ref, "e_dyn_mj")
        idle = m54(c, "idle_w") - idle_ref
        return X, idle + E * X / 1000

    Xg, Pg, Fg = 1000 / mf(-1, "ms_per_layer"), mf(-1, "power_w"), mf(-1, "clock_med")
    c3 = min(cl, key=lambda c: abs(c - Fg))
    X3, P3 = full_at(c3)
    print(f"\ncheck 3 -- reconstructed full width at {c3} MHz: X={X3:.1f}/s, P={P3:.1f} W;"
          f"  governor measured at {Fg:.0f} MHz: X={Xg:.1f}/s, P={Pg:.1f} W"
          f"  ({X3 / Xg - 1:+.1%} throughput, {P3 - Pg:+.1f} W)")

    print(f"\nmarginal cost m of the reconstructed full-width curve, W per (layer/s):")
    prev = None
    for c in cl:
        if c < ref:
            continue
        X, P = full_at(c)
        if prev:
            print(f"  {prev[0]:>5} -> {c:>5}: {(P - prev[2]) / (X - prev[1]):6.2f}")
        prev = (c, X, P)
    X14, P14 = full_at(max(cl))
    XA, PA = full_at(c3)
    print(f"  chord {c3} -> {max(cl)}: {(P14 - PA) / (X14 - XA):.2f}   "
          f"(break-even in predict_tdp: 4.67-7.24; GEMM-voltage estimate ~2.6)")


if __name__ == "__main__":
    main()
