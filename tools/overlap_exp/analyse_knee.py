#!/usr/bin/env python3
"""Refit a(f) and P_static(f) with the 900-1200 MHz gap filled in.

WHY THIS MATTERS MORE THAN IT LOOKS. kappa = a/f is C*V^2, the voltage curve. On the
original six-clock grid it is flat to 0.5% CV from 300 to 900 MHz and then +33% by 1200,
so the knee is somewhere inside a span with NO calibration points. That one gap is
load-bearing three times over:

  * every one of the 39 throttled overlap rows settled between 1170 and 1395 MHz, so
    each was re-scored on coefficients interpolated across the gap from its two endpoints
  * the closed-loop solver f* : P(f*) = cap searches inside it
  * the 15 MHz sweep found the energy optimum at 945-1065 MHz -- inside it

And the interpolation was already known to be wrong there: measured P_static at 900 and
1200 is 64.3 and 74.1 W against 61.4 and 69.7 W interpolated, and the curve is CONVEX,
not linear. A straight line through a convex curve runs cold in the middle and hot at
the ends, which is exactly the +0.82 clock-correlated residual the throttled rescore
showed.

This adds 960 / 1020 / 1080 / 1140 MHz on the same squatter rig, so a is identifiable
(it needs the SM axis) rather than merely bounded.

  python3 analyse_knee.py
"""
import csv
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "gemm_dcfs_char", "data")
TOTAL_SM = 108


def fit(xs, ys):
    n = len(xs); mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    m = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else 0.0
    return m, my - m * mx


def load(path, need_dyn=True):
    if not os.path.exists(path):
        return []
    out = []
    for x in csv.DictReader(open(path)):
        if x.get("clock_held") != "True" or x.get("status") not in (None, "", "ok"):
            continue
        try:
            r = dict(f=int(x["freq_req_mhz"]), m=int(x["m"]), n=int(x["n"]),
                     k=int(x["k"]), sm=int(x["sm_avail"]),
                     lat=float(x["latency_ms"]), p=float(x["power_w"]))
            if "p_dynamic_w" in x and x["p_dynamic_w"]:
                r["pdyn"] = float(x["p_dynamic_w"])
            if "p_static_w" in x and x["p_static_w"]:
                r["ps"] = float(x["p_static_w"])
        except (ValueError, KeyError):
            continue
        out.append(r)
    return out


def main():
    old = load(os.path.join(SRC, "gemm_dynamic_energy.csv"))
    new = load(os.path.join(SRC, "gemm_squat_knee.csv"))
    print(f"original campaign: {len(old)} usable rows, clocks "
          f"{sorted({r['f'] for r in old}, reverse=True)}")
    print(f"knee campaign    : {len(new)} usable rows, clocks "
          f"{sorted({r['f'] for r in new}, reverse=True)}\n")
    if not new:
        print("knee data not present yet"); return

    # a(f) needs the SM axis; fit P_dyn = a*S_eff + b per clock across SM counts
    import json
    GR = json.load(open(os.path.join(HERE, "data", "grid12.json")))

    def a_of(rows):
        by = {}
        for r in rows:
            g = GR.get(f"{r['m']}_{r['n']}_{r['k']}")
            if g is None:
                continue
            se = g / -(-g // r["sm"])
            by.setdefault(r["f"], []).append((se, r.get("pdyn", r["p"])))
        return {f: fit([p[0] for p in v], [p[1] for p in v])[0]
                for f, v in by.items() if len({p[0] for p in v}) >= 3}

    A = {**a_of(old), **a_of(new)}
    print("kappa = a/f across the filled grid (x10^-3):\n")
    print(f"  {'f':>6}{'a':>9}{'kappa':>9}{'source':>10}{'':>4}vs the 300-900 floor")
    base = st.mean(A[f] / f for f in sorted(A) if f <= 900)
    knee = None
    for f in sorted(A):
        k = A[f] / f
        src = "new" if f in {r["f"] for r in new} else "orig"
        rel = k / base
        bar = "#" * int(max(0, (rel - 1)) * 60)
        if knee is None and f > 900 and rel > 1.05:
            knee = f
        print(f"  {f:>6}{A[f]:>9.4f}{k * 1e3:>9.4f}{src:>10}{'':>4}{rel:>5.3f} {bar}")
    print(f"\n  floor value 300-900 MHz: {base * 1e3:.4f}e-3")
    if knee:
        print(f"  first clock more than 5% above the floor: {knee} MHz")
        print(f"  -> the voltage knee is between {knee - 60} and {knee} MHz, not at 900")

    # P_static convexity
    ps = {}
    for r in old + new:
        if "ps" in r:
            ps.setdefault(r["f"], []).append(r["ps"])
    ps = {f: st.median(v) for f, v in ps.items()}
    if len(ps) > 6:
        print(f"\n\nP_static(f), with the gap filled:\n")
        print(f"  {'f':>6}{'measured':>10}{'2-pt interp':>13}{'err':>8}")
        lo, hi = 900, 1200
        for f in sorted(ps):
            if lo <= f <= hi:
                lin = ps[lo] + (ps[hi] - ps[lo]) * (f - lo) / (hi - lo)
                print(f"  {f:>6}{ps[f]:>10.2f}{lin:>13.2f}{(lin - ps[f]) / ps[f] * 100:>+7.2f}%")


if __name__ == "__main__":
    main()
