#!/usr/bin/env python3
"""Two V/f domains under the board power limit: can the cold side's power headroom
buy the hot side a clock high enough to beat the uniform clock on LATENCY?

WHY THIS SCRIPT EXISTS. predict_two_domain.py composes the two domains from the locked
sweep and never imposes TDP. It caps both clocks at 1290 MHz, the fastest the
full-width kernel holds, which quietly denies the design the one operating point it is
built for: cold SMs slowed down, the power they free spent on hot SMs clocked above
what the whole chip could sustain. The deployed baseline runs at the power limit
(measured 400.0 W at ~1322 MHz), so a model without the limit cannot say whether that
trade wins.

THE SEARCH is the dynamic procedure, not a static enumeration:
  1. fix the cold domain's clock f_c
  2. ask whether the pair (f_c, f_h) exceeds TDP
  3. if it does not, f_h can go higher
  4. binary-search the highest f_h that fits (up to the A100's 1410 MHz max)
then keep the f_c whose best f_h gives the shortest makespan. The single-domain
baseline gets the SAME procedure with one knob (f_c = f_h), which is what the GPU's own
boost loop does, so both sides are scored on one curve, one limit, one rule. The binary
search is checked against an exhaustive 1 MHz scan, because it is only valid if power
rises monotonically along f_h, and that is asserted rather than assumed.

THE MODEL. Both partitions end together, so with measured full-width layer time T(f)
and board power P(f):
      s        = phi T(f_c) / (phi T(f_c) + (1-phi) T(f_h))    SM share of the cold side
      makespan = phi T(f_c) + (1-phi) T(f_h)
      power    = s P(f_c) + (1-s) P(f_h)
The power mix is exact under per-SM additivity for the same reason as the static-power
mix in make_sweep_3d.py: the weights sum to 1, so the shared floor is counted once. It
uses the WHOLE LAYER's curve for both partitions, which partition_kappa.py measured to
be fair: cold over hot kappa is 1.025 at the same tile. phi is restricted to the tile
shares achievable at expert granularity; s is continuous, which favours the design
(real SM counts are integers, and real V/f domains would be coarser still).

ABOVE 1322 MHz NOTHING IS MEASURED, and cannot be at full width: locked at 1335, 1380
or 1410 the kernel runs at ~1320 MHz on the power limit. So the curve is extended to
1410 with throughput linear in clock (the last measured segment's slope) and power
rising at a CONSTANT marginal cost m, in W per (layer/s). A constant m is the most
generous extension available to the design: the measured marginal cost rises
steeply through the knee, and any further rise only makes the hot side dearer.
The script sweeps m and reports the BREAK-EVEN m below which the split would win,
so the conclusion is stated against the one number that is unmeasured.

HOW WELL THE TOP OF THE CURVE IS KNOWN, printed at run time. The 60 MHz segments at
1170-1290 are tight (per-pass sd ~0.1-0.2). The last segment, 1290 to the governor
point, is only ~30 MHz wide and its per-pass slope scatters by a factor of three. So
the break-even is compared against a range, not a point, and the honest reading is
two-sided: if the curve keeps steepening above 1322, as the rising voltage says it
must, the split gains nothing; if it stopped steepening at exactly the operating
point, the free-kernel split would gain a fraction of a percent. Either way it is
smaller than the measured cost of the partitioned kernel.

KERNEL COST. "free" is a partitioned kernel that costs nothing. "measured" charges
data/pin_vs_queue_qwen.csv's best pinned split against vLLM's work queue, in time AND
power (a slower kernel draws less, which buys clock under the limit, and the design is
given that credit). Those ratios are measured at 1200 MHz and assumed clock-independent.
"""
import csv, os, sys
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from predict_two_domain import load_pin

TDP = 400.0          # W, A100-SXM4-40GB board power limit
F_MAX = 1410.0       # MHz, A100 max SM clock
M_SWEEP = [0.91, 2.35, 3.49, 5.0, 7.37, 10.0]   # W per (layer/s); see docstring
WIN_MIN = 1e-4       # a latency win smaller than 0.01% is not a win


def load_curve():
    by = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(HERE, "data", "b1_energy_qwen.csv"))):
        by[int(r["clock"])].append(r)
    A = lambda c, k: float(np.mean([float(x[k]) for x in by[c]]))
    held = [c for c in sorted(by) if c > 0 and abs(c - A(c, "clock_med")) <= 7]
    f = [float(c) for c in held] + [A(-1, "clock_med")]
    T = [A(c, "ms_per_layer") for c in held] + [A(-1, "ms_per_layer")]
    P = [A(c, "power_w") for c in held] + [A(-1, "power_w")]
    return np.array(f), np.array(T), np.array(P)


class Curve:
    """T(f), P(f): measured up to the governor point, extended above it with slope m."""

    def __init__(self, f, T, P, m):
        self.f, self.T, self.P, self.m = f, T, P, m
        X = 1000.0 / T
        self.f0, self.X0, self.P0 = f[-1], X[-1], P[-1]
        self.dXdf = (X[-1] - X[-2]) / (f[-1] - f[-2])

    def X(self, f):
        f = np.asarray(f, float)
        inside = np.interp(f, self.f, 1000.0 / self.T)
        return np.where(f <= self.f0, inside, self.X0 + self.dXdf * (f - self.f0))

    def t(self, f):
        return 1000.0 / self.X(f)

    def p(self, f):
        f = np.asarray(f, float)
        inside = np.interp(f, self.f, self.P)
        return np.where(f <= self.f0, inside, self.P0 + self.m * (self.X(f) - self.X0))


def design(c, phi, fc, fh, kt=1.0, kp=1.0):
    tc, th = c.t(fc), c.t(fh)
    s = phi * tc / (phi * tc + (1 - phi) * th)
    return s, (phi * tc + (1 - phi) * th) * kt, (s * c.p(fc) + (1 - s) * c.p(fh)) * kp


def max_fh(c, phi, fc, kt, kp, tol=0.01):
    """Steps 2-4: the highest f_h >= f_c that keeps the pair under TDP, or None."""
    if design(c, phi, fc, fc, kt, kp)[2] > TDP:
        return None
    if design(c, phi, fc, F_MAX, kt, kp)[2] <= TDP:
        return F_MAX
    lo, hi = fc, F_MAX
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if design(c, phi, fc, mid, kt, kp)[2] <= TDP else (lo, mid)
    return lo


def best_split(c, phi, fcs, kt, kp):
    best = None
    for fc in fcs:
        fh = max_fh(c, phi, fc, kt, kp)
        if fh is None:
            continue
        s, t, p = design(c, phi, fc, fh, kt, kp)
        if best is None or t < best[3]:
            best = (fc, fh, s, t, p)
    return best


def uniform(c, tol=0.01):
    """The one-knob version of the same search: the governor's boost loop."""
    lo, hi = float(c.f[0]), F_MAX
    if c.p(hi) <= TDP:
        return hi
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if c.p(mid) <= TDP else (lo, mid)
    return lo


def scan_check(c, phi, kt, kp):
    """Exhaustive 1 MHz scan. Must agree with the binary search, and the feasible f_h
    must form ONE interval starting at f_c, or the binary search is not valid.
    That is the property the search needs -- not strict monotonicity. Power does dip
    by ~0.004 W along f_h at ~180 W, in the flat noisy region under the voltage floor,
    and a wiggle 220 W below the limit cannot create a second crossing of it."""
    grid = np.arange(c.f[0], F_MAX + 0.5, 1.0)
    best, nonmono = None, 0
    for fc in grid[::5]:
        fh = grid[grid >= fc]
        s, t, p = design(c, phi, fc, fh, kt, kp)
        ok = p <= TDP
        nonmono += int(np.any(np.diff(ok.astype(int)) > 0))   # infeasible, then feasible
        if ok.any():
            i = int(np.argmin(np.where(ok, t, np.inf)))
            if best is None or t[i] < best[1]:
                best = (fc, t[i])
    return best, nonmono


def main():
    f, T, P = load_curve()
    by, m, sd, best_pin, shares = load_pin()
    kt = m(best_pin, "ms") / m("queue", "ms")
    kp = (m(best_pin, "energy_mj") / m("queue", "energy_mj")) / kt
    phis = [x for x in shares if x > 0]
    fcs = np.arange(f[0], F_MAX + 0.5, 5.0)

    print(f"curve: {len(f) - 1} locked clocks {f[0]:.0f}-{f[-2]:.0f} MHz + governor "
          f"{f[-1]:.0f} MHz at {P[-1]:.1f} W;  TDP {TDP:.0f} W, f_max {F_MAX:.0f} MHz")
    X = 1000.0 / T
    mc = np.diff(P) / np.diff(X)
    print(f"measured marginal cost dP/dX: {mc[0]:.2f} at the bottom, "
          f"{mc[-2]:.2f} at {f[-3]:.0f}-{f[-2]:.0f}, {mc[-1]:.2f} for the last "
          f"segment to the governor")
    # how well is the top of the curve known? per-pass slopes, since the break-even
    # below has to be read against them
    by = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(HERE, "data", "b1_energy_qwen.csv"))):
        by[int(r["clock"])].append(r)
    for a, b in ((1170, 1230), (1230, 1290), (1290, -1)):
        ra, rb = ({r["pas"]: r for r in by[a]}, {r["pas"]: r for r in by[b]})
        sl = np.array([(float(rb[q]["power_w"]) - float(ra[q]["power_w"]))
                       / (1000 / float(rb[q]["ms_per_layer"])
                          - 1000 / float(ra[q]["ms_per_layer"]))
                       for q in sorted(set(ra) & set(rb))])
        print(f"  per-pass slope {a}->{'governor' if b < 0 else b}: {sl.mean():.2f} "
              f"+/- {sl.std(ddof=1):.2f} (range {sl.min():.2f}-{sl.max():.2f}, "
              f"n={len(sl)})")
    print(f"kernel cost ({best_pin} vs queue): time x{kt:.4f}, power x{kp:.4f}\n")

    rows = []
    for m_ in M_SWEEP:
        c = Curve(f, T, P, m_)
        fu = uniform(c)
        tu, pu = float(c.t(fu)), float(c.p(fu))
        print(f"m = {m_:5.2f} W/(layer/s): uniform under TDP -> {fu:.0f} MHz, "
              f"{tu:.3f} ms, {pu:.1f} W")
        for kname, KT, KP in (("free", 1.0, 1.0), ("measured", kt, kp)):
            for phi in phis:
                fc, fh, s, t, p = best_split(c, phi, fcs, KT, KP)
                (sc_fc, sc_t), nonmono = scan_check(c, phi, KT, KP)
                assert nonmono == 0, "TDP crossed more than once along f_h: binary search invalid"
                assert abs(sc_t - t) / t < 2e-3, (sc_t, t)
                rows.append(dict(
                    m_w_per_layer_s=m_, kernel=kname, phi=phi,
                    f_cold=round(fc, 1), f_hot=round(fh, 1), sm_share_cold=round(s, 4),
                    makespan_ms=round(t, 4), power_w=round(p, 2),
                    energy_mj=round(p * t, 2), uniform_mhz=round(fu, 1),
                    uniform_ms=round(tu, 4), uniform_energy_mj=round(pu * tu, 2),
                    latency_vs_uniform_pct=round((t / tu - 1) * 100, 3),
                    energy_vs_uniform_pct=round((p * t / (pu * tu) - 1) * 100, 3)))
                print(f"    {kname:8s} phi={phi:.3f}: f_c={fc:6.1f} f_h={fh:6.1f} "
                      f"s={s:.3f}  {t:.3f} ms ({(t / tu - 1) * 100:+.2f}%)  "
                      f"E {(p * t / (pu * tu) - 1) * 100:+.2f}%")
        print()

    # break-even: the constant marginal cost above the governor point below which the
    # free-kernel split starts to beat the uniform clock on latency. A "win" must be at
    # least WIN_MIN: without a floor the search counts 1e-5 % gains, which are the
    # bisection tolerance, not the model
    print("break-even m (free kernel): the split wins only if the power cost of "
          "throughput above ~1322 MHz is BELOW this")
    for phi in phis:
        def gain(m_):
            c = Curve(f, T, P, m_)
            return best_split(c, phi, fcs, 1.0, 1.0)[3] / float(c.t(uniform(c))) - 1
        lo, hi = 0.01, 20.0
        if gain(lo) >= -WIN_MIN:
            print(f"    phi={phi:.3f}: never, even at m -> 0")
            continue
        while hi - lo > 0.01:
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if gain(mid) < -WIN_MIN else (lo, mid)
        print(f"    phi={phi:.3f}: m < {lo:.2f}")
        rows.append(dict(m_w_per_layer_s=round(lo, 2), kernel="break_even", phi=phi))

    out = os.path.join(HERE, "data", "tdp_two_domain.csv")
    keys = list(rows[0].keys())
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print(f"\nwrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
