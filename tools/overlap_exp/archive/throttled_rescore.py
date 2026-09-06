#!/usr/bin/env python3
"""Re-score the 39 throttled rows at the clock they ACTUALLY ran at.

The model has no throttling term: it predicts what the GPU would draw if the clock it
was told to hold were the clock it held. On the 39 rows where the 400 W board cap pulled
the clock down, it therefore over-predicts by ~47%. That is expected, and uninformative
about whether the model is right.

The informative question is the one this script asks: told the true frequency, does the
model reproduce the measured watts? If it does, the 47% is entirely the missing
throttling term and the physics underneath is sound. If it does not, something else is
wrong at 1410 MHz as well.

Every coefficient is linearly interpolated in f between the six calibrated clocks:
a, P_static, b, tau, t0, p0, gamma, and the per-CTA tables B(c,.) and alpha(c,.).

THEN THE CLOSED LOOP. A model that only works when handed the achieved clock is not
useful for planning -- you do not know the achieved clock until you run it. So the
script also solves for the fixed point

    f* such that P_model(f*) = 400 W

by bisection, and compares f* against the measured clock. That is the form a deployment
planner would actually call.

TWO CAVEATS ON THE GROUND TRUTH, both of which bias this test in a known direction:

  clock_achieved is the MINIMUM clock seen in the window, not the mean. The board
  oscillates around the cap, so the true time-average clock is somewhere above it.
  Modelling at the minimum therefore UNDER-predicts power, and a negative bias here is
  the instrument, not the model.

  measured power is the median over the window and pins near 400 W by construction --
  the cap is what produced these rows. Agreement on watts alone is weak evidence; the
  clock comparison in the closed-loop section is the stronger test.

  python3 throttled_rescore.py
"""

import csv
import os
import statistics as st

import table_432 as t

HERE = os.path.dirname(os.path.abspath(__file__))
TOTAL_SM = 108
CAP = 400.0

# the model itself lives in table_432.predict(f, ...), which already takes f as a free
# argument and interpolates every coefficient there. Importing it rather than copying it
# means a correction to the model cannot silently fail to reach this analysis.
predict = t.predict


def solve_clock(c, m, n, k, g, mib, f_req, cap=CAP):
    """largest f <= f_req with P_model(f) <= cap -- what the hardware is doing"""
    if predict(f_req, c, m, n, k, g, mib)['p'] <= cap:
        return f_req
    lo, hi = 150.0, float(f_req)
    for _ in range(40):
        mid = (lo + hi) / 2
        if predict(mid, c, m, n, k, g, mib)['p'] > cap:
            hi = mid
        else:
            lo = mid
    return lo


def main():
    rows = [x for x in csv.DictReader(
        open(os.path.join(HERE, "data", "overlap_432.csv")))
        if x["mode"] == "concurrent" and x["clock_held"] != "True"]

    print(f"{len(rows)} throttled rows, re-scored at the achieved clock\n")
    print(f"  {'req':>5}{'achv':>6}{'CTA':>4}{'M':>6}{'N=K':>7}{'meas W':>8}"
          f"{'@req':>8}{'err':>7}{'@achv':>8}{'err':>7}{'|':>3}{'f*':>6}{'df':>6}")
    e_req, e_ach, d_f = [], [], []
    out = []
    for r in rows:
        f, c = int(r["clock"]), int(r["ctas"])
        m, n, k = int(r["m"]), int(r["n"]), int(r["k"])
        g, mib = int(r["grid"]), int(r["ar_mib"])
        fa = int(r["clock_min"])
        meas = float(r["power_per_gpu_w"])

        p_req = predict(f, c, m, n, k, g, mib)['p']
        p_ach = predict(fa, c, m, n, k, g, mib)['p']
        fstar = solve_clock(c, m, n, k, g, mib, f)

        er, ea = (p_req - meas) / meas, (p_ach - meas) / meas
        e_req.append(abs(er)); e_ach.append(abs(ea)); d_f.append(fstar - fa)
        out.append(dict(clock=f, clock_achieved=fa, ctas=c, m=m, nk=n, k=k,
                        meas_w=round(meas, 1),
                        model_w_at_req=round(p_req, 1), err_at_req_pct=round(er * 100, 1),
                        model_w_at_achieved=round(p_ach, 1),
                        err_at_achieved_pct=round(ea * 100, 1),
                        clock_solved=round(fstar), clock_solved_minus_achieved=round(fstar - fa)))
        print(f"  {f:>5}{fa:>6}{c:>4}{m:>6}{n:>7}{meas:>8.1f}"
              f"{p_req:>8.0f}{er * 100:>6.0f}%{p_ach:>8.1f}{ea * 100:>6.1f}%"
              f"{'|':>3}{fstar:>6.0f}{fstar - fa:>+6.0f}")

    def q(v):
        v = sorted(v)
        return st.median(v) * 100, v[int(.9 * len(v))] * 100, max(v) * 100
    print()
    a1, b1, c1 = q(e_req); a2, b2, c2 = q(e_ach)
    print(f"  power at the REQUESTED clock: median {a1:6.2f}%  p90 {b1:6.2f}%  max {c1:6.2f}%")
    print(f"  power at the ACHIEVED  clock: median {a2:6.2f}%  p90 {b2:6.2f}%  max {c2:6.2f}%")
    sb = st.mean((x["err_at_achieved_pct"]) for x in out)
    print(f"  bias at the achieved clock: {sb:+.2f}%   "
          f"(expected negative: clock_achieved is a minimum, not a mean)")
    print(f"\n  closed loop, solving P_model(f*) = {CAP:.0f} W:")
    print(f"    f* - clock_achieved: median {st.median(d_f):+.0f} MHz   "
          f"mean {st.mean(d_f):+.0f}   range {min(d_f):+.0f} .. {max(d_f):+.0f}")
    within = sum(1 for x in d_f if abs(x) <= 100)
    print(f"    within +/-100 MHz of the measured floor: {within}/{len(d_f)}")
    print("\n  per-row numbers live in data/table_432.csv, columns "
          "pred_w_throttled / err_throttled_pct")


if __name__ == "__main__":
    main()
