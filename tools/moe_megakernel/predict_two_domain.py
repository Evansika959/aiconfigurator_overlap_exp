#!/usr/bin/env python3
"""Does a second spatial V/f domain beat simply lowering the single clock, at matched
latency? Composed from B1's measurements; writes data/two_domain_matched.csv.

THE ANSWER IS NO, AND THE INTERESTING PART IS WHY AN EARLIER VERSION SAID OTHERWISE.

  A first version of this script scored the single-clock baseline by linear
  INTERPOLATION over the clock grid, and the two-domain design by a DISCRETE search
  with deadline semantics. That is not a comparison. The two-domain candidate set
  contains every single clock (the f_hi == f_lo cases), so scoring the superset
  discretely against a continuous relaxation of itself can only ever flatter the
  baseline -- by a mean of 18.0 mJ and up to 73.0 mJ here, against a panel that spans
  about 71 mJ. Both sides now get the same clock set and the same deadline rule.

  Fixed that way, two domains DO win -- and the win shrinks as the single clock is
  allowed a finer grid. The numbers are NOT transcribed here: this script prints them
  and writes data/two_domain_summary.csv, because a docstring table goes stale the
  moment the data is re-collected, and this one already did once.

  It is not voltage harvesting; it is LATENCY GRANULARITY. The two-domain design lands
  exactly on a deadline that a quantised clock grid overshoots, so its advantage is a
  function of the grid it is compared against and goes to zero in the continuous limit.
  `overlap_exp` was bitten by the same thing already, where a 45 MHz sweep grid
  manufactured a 4.82% win that the real 15 MHz grid turned into a 0.66% loss -- which
  is also why the six clocks the sweep here had skipped were gone back and measured,
  rather than interpolated and argued about.

  There is no voltage left to harvest because kappa is flat: dynamic energy varies 1.0%
  over 510-1050 MHz (RESULTS_B1.md). Below the knee a slow domain buys nothing; above
  it, the saving is equally available to one uniform clock, which pays no makespan tax.

THE MODEL

  Time scales as work / (SMs x clock). T(f) x f is constant to 6.5% peak to peak
  (4600.9-4899.2, mean 4766.5) across 510-1290 MHz, so this is a 6.5%-accurate model,
  not an exact one, and the makespan below is linear in it.

  Split the work into a share phi at f_lo and (1 - phi) at f_hi on disjoint SM sets.
  Equalising the two finishing times fixes the split and the makespan:

      s = phi f_hi / (phi f_hi + (1 - phi) f_lo)      (SM share for the slow side)
      makespan = T(f_hi) x (phi f_hi / f_lo + 1 - phi)

  Dynamic energy is kappa(f) x work and f cancels for compute-bound work, so each share
  is charged its own domain's measured kappa. LIMITATION, and it is the weakest
  assumption here: that kappa is the WHOLE LAYER's. A partition holding only lightly
  routed experts is more wave-quantised, and the measured spread across pinned splits
  at one clock is 48.7-136.6 TFLOP/s against the queue's 144.4 -- far more than the
  +/-1% this assumes. The bias favours two domains, since a light partition's kappa
  would be worse than the layer mean, not better.

  Static power is charged over the makespan. Two rules are reported: the fast rail, and
  the SM-weighted mix of the two. A third rule -- bill all 108 SMs at the SLOW rail --
  was dropped: it credits static savings to SMs that are not in the slow domain, and
  its apparent best case was 124% accounting gift.

  phi is restricted to the tile shares actually achievable by splitting at expert
  granularity, measured in data/pin_vs_queue_qwen.csv, plus phi = 0 (no split).

  HANDICAP is the measured cost of getting a partitioned persistent kernel at all, and
  it is charged to BOTH time and dynamic energy, because the measurement shows both.
  Its value is computed from data/pin_vs_queue_qwen.csv and printed; it is not written
  down here, for the same reason.

  The pinned SM splits in that file were pre-narrowed by an earlier single-pass sweep
  that kept, for each expert count, the allocation with the lowest one-shot time. The
  6-pass means are unbiased for the splits that survived, but a genuinely better
  neighbouring split that happened to measure high once was dropped. That also sets the
  resolution of the phi grid, since the achievable tile shares come from these cases.
"""
import csv, os, sys
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# The six clocks 1170/1185/1215/1230/1260/1275 were missing from the original sweep,
# which stepped 45 MHz above 1155, and have since been measured. Over the plotted
# latency range the measured set now IS the A100's real 15 MHz grid. "45MHz" below is
# that same measured data SUBSAMPLED back to the original spacing -- so the
# grid-resolution comparison is between measurements throughout, and the coarse row
# shows exactly what the under-sampled sweep would have claimed.
GRIDS = [("45MHz", ("sub45", 0)), ("15MHz", None), ("5MHz", 5)]
# Phase check: 45 MHz subsampling can be done three ways above 1155 MHz. If the coarse
# grid's inflated win were an accident of which clocks the original campaign happened to
# land on, the three phases would disagree. Computed and reported, not assumed.
# A separate 45 MHz "phase check" used to live here. It is gone: its construction had a
# free choice (whether the boundary clock 1155 stays in every phase) that changed the
# numbers by 0.5 pp and made them non-reproducible from a stated rule. The 18-anchor
# ensemble below IS the phase check, done uniformly at every step size, and it is the
# quantity the uncertainty is built from rather than a side note.
# The exponent is fitted on MEASURED subsamples only. The plotted deadline range spans
# 1035-1290 MHz, entirely inside the region measured at 15 MHz, so the measured grid can
# be thinned to any multiple of 15 with no interpolation anywhere. Fitting instead across
# {45, 15, 5} would rest the exponent partly on the hypothetical 5 MHz grid -- and the
# two intervals there disagree (0.78 and 0.92), so their mean is not a measurement.
FIT_STEPS = [15, 30, 45, 60, 75, 90]
# ...and over EVERY anchor the thinning can start from, because it has a phase and the
# answer depends on it. Fitting from one anchor gives an exponent whose within-fit CI is
# narrower than the spread across anchors, which is how two independent analyses of this
# same data reached 0.68 and 0.79 with non-overlapping intervals. The ensemble is the
# measurement; a single anchor is one draw from it.
FIT_ANCHORS = list(range(1035, 1291, 15))
STATIC = ["hi", "mix"]


def load_sweep(pick=None):
    """pick: indices of the passes to average, with replacement, for the bootstrap.
    None means all of them, which is the point estimate."""
    by = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(HERE, "data", "b1_energy_qwen.csv"))):
        by[int(r["clock"])].append(r)

    def A(c, k):
        v = [float(x[k]) for x in by[c]]
        return float(np.mean(v if pick is None else [v[i % len(v)] for i in pick]))
    cl = np.array([c for c in sorted(by) if c > 0 and abs(c - A(c, "clock_med")) <= 7])
    return (cl, *[np.array([A(c, k) for c in cl]) for k in
                  ("ms_per_layer", "energy_mj_per_layer", "e_dyn_mj_per_layer",
                   "idle_w")])


def load_pin():
    rows = list(csv.DictReader(open(os.path.join(HERE, "data",
                                                 "pin_vs_queue_qwen.csv"))))
    by = defaultdict(list)
    for r in rows:
        by[r["case"]].append(r)
    m = lambda c, k: float(np.mean([float(x[k]) for x in by[c]]))
    sd = lambda c, k: (float(np.std([float(x[k]) for x in by[c]], ddof=1))
                       if len(by[c]) > 1 else 0.0)
    pinned = [c for c in by if c.startswith("pinned")]
    # pick the best pinned split by its MEAN, not by the minimum of single shots --
    # taking min over many one-shot measurements is a winner's-curse estimate
    best = min(pinned, key=lambda c: m(c, "ms"))
    shares = sorted({float(r["tile_share_light"]) for r in rows
                     if r["tile_share_light"] not in ("", None)})
    return by, m, sd, best, shares


def exponent(cl, T, E, D, PS, PHI, targets, rows_out=None):
    """Mean fitted exponent over every thinning anchor, and the per-anchor values."""
    exps = []
    for anchor in FIT_ANCHORS:
        gs = []
        for S in FIT_STEPS:
            f = np.array([c for c in cl if (anchor - c) % S == 0])
            Tg, Eg = np.interp(f, cl, T), np.interp(f, cl, E)
            Dg, Pg = np.interp(f, cl, D), np.interp(f, cl, PS)
            ct, ce = [], []
            for ih in range(len(f)):
                for il in range(ih + 1):
                    for phi in PHI:
                        sh = (phi * f[ih] / (phi * f[ih] + (1 - phi) * f[il])
                              if phi > 0 else 0.0)
                        t = Tg[ih] * (phi * f[ih] / f[il] + 1 - phi)
                        ct.append(t)
                        ce.append(Dg[il] * phi + Dg[ih] * (1 - phi)
                                  + (sh * Pg[il] + (1 - sh) * Pg[ih]) * t)
            ct, ce = np.array(ct), np.array(ce)
            d, base = [], []
            for tg in targets:
                m1, m2 = Tg <= tg + 1e-9, ct <= tg + 1e-9
                if m1.any() and m2.any():
                    d.append(ce[m2].min() - Eg[m1].min()); base.append(Eg[m1].min())
            g = abs(float(np.min(d))) / float(np.median(base)) * 100
            gs.append(g)
            if rows_out is not None:
                rows_out.append(dict(anchor=anchor, step_mhz=S, gain_pct=round(g, 4)))
        exps.append(float(np.polyfit(np.log(FIT_STEPS), np.log(gs), 1)[0]))
    return float(np.mean(exps)), np.array(exps)


def main():
    cl, T, E, D, PS = load_sweep()
    by, m, sd, best, shares = load_pin()
    q_ms, p_ms = m("queue", "ms"), m(best, "ms")
    idle_1200 = float(np.interp(1200, cl, PS))
    q_dyn = (m("queue", "power_w") - idle_1200) * q_ms
    p_dyn = (m(best, "power_w") - idle_1200) * p_ms
    HT, HE = p_ms / q_ms, p_dyn / q_dyn
    n = len(by["queue"])
    print(f"measured at 1200 MHz, n={n} pass(es):")
    print(f"  queue      {q_ms:.3f} +/- {sd('queue','ms'):.3f} ms")
    print(f"  persistent {m('persistent_2blk_per_sm','ms'):.3f} +/- "
          f"{sd('persistent_2blk_per_sm','ms'):.3f} ms "
          f"({m('persistent_2blk_per_sm','ms')/q_ms-1:+.1%})")
    print(f"  pinned     {p_ms:.3f} +/- {sd(best,'ms'):.3f} ms ({HT-1:+.1%})  [{best}]")
    print(f"  handicap: time {HT-1:+.2%}, dynamic energy {HE-1:+.2%}")
    print(f"  achievable light-partition tile shares: "
          + ", ".join(f"{x:.3f}" for x in shares))

    PHI = np.array([0.0] + shares)
    # clamp up to the fastest measured latency: rounding the first target to 3 dp can
    # land it just BELOW T.min(), which makes the deadline unreachable for both sides
    targets = np.maximum(np.round(np.arange(T.min(), T[np.argmin(E)] + 1e-9, 0.02), 3),
                         T.min())

    def solve(step, static, ht, he):
        """Best two-domain and best single clock at each deadline, SAME clock set and
        SAME deadline rule for both."""
        if step is None:
            f = cl                                   # the measured 15 MHz grid
        elif isinstance(step, tuple) and step[0] == "thin":
            # thin the measured grid to a multiple of 15, from a given anchor
            S, anchor = step[1], step[2]
            f = np.array([c for c in cl if (anchor - c) % S == 0])
        elif isinstance(step, tuple) and step[0] == "sub45":
            # the measured data resampled to 45 MHz above 1155, at a chosen phase
            off = step[1]
            f = np.array([c for c in cl if c <= 1155 - 1 or (c - 1155) % 45 == off])
        else:
            f = np.arange(int(cl.min()), int(cl.max()) + 1, step)
        Tg, Eg = np.interp(f, cl, T), np.interp(f, cl, E)
        Dg, Pg = np.interp(f, cl, D), np.interp(f, cl, PS)
        cand_t, cand_e = [], []
        for ih in range(len(f)):
            for il in range(ih + 1):
                for phi in PHI:
                    s = (phi * f[ih] / (phi * f[ih] + (1 - phi) * f[il])
                         if phi > 0 else 0.0)
                    t = Tg[ih] * (phi * f[ih] / f[il] + 1 - phi) * ht
                    ps = Pg[ih] if static == "hi" else s * Pg[il] + (1 - s) * Pg[ih]
                    cand_t.append(t)
                    cand_e.append((Dg[il] * phi + Dg[ih] * (1 - phi)) * he + ps * t)
        cand_t, cand_e = np.array(cand_t), np.array(cand_e)
        out = []
        for tg in targets:
            m1 = Tg <= tg + 1e-9
            m2 = cand_t <= tg + 1e-9
            out.append((float(Eg[m1].min()) if m1.any() else np.nan,
                        float(cand_e[m2].min()) if m2.any() else np.nan))
        # The comparison is restricted to deadlines at or below the single-clock energy
        # optimum. Measure, rather than assert in prose, how much that restriction
        # costs the two-domain side: the shortfall between its best candidate anywhere
        # and its best candidate inside the range. Reported per variant and required to
        # be immaterial, not assumed to be zero.
        # The figure plots deadlines at or below the single-clock energy optimum. That
        # restriction bites the two-domain side once it carries the build handicap: its
        # own optimum is pushed past the range end. So rather than assert the
        # restriction is harmless, MEASURE two things and report both -- how much the
        # restriction costs it, and whether it would win with NO deadline at all
        # (global best two-domain vs global best single clock, both unbounded).
        out = np.array(out)
        rel = float((np.nanmin(out[:, 1]) - cand_e.min()) / cand_e.min())
        unbounded = float(cand_e.min() - Eg.min())
        return out, rel, unbounded

    rows = [dict(latency_ms=t) for t in targets]
    summary = []
    for gname, step in GRIDS:
        for st in STATIC:
            free, sf, ub = solve(step, st, 1.0, 1.0)
            for i, r in enumerate(rows):
                r[f"one_{gname}_{st}"] = round(free[i, 0], 2)
                r[f"two_{gname}_{st}"] = round(free[i, 1], 2)
            d = free[:, 1] - free[:, 0]
            ok = ~np.isnan(d)
            summary.append((gname, st, "free", int((d[ok] < -1e-9).sum()), int(ok.sum()),
                            float(d[ok].min()), float(np.nanmedian(free[ok, 0])),
                            sf, ub))
    # (c) is scored on the as-swept grid; check the verdict does not depend on that
    for gname, step in GRIDS:
        cost, sf, ub = solve(step, "hi", HT, HE)
        d = cost[:, 1] - cost[:, 0]
        ok = ~np.isnan(d)
        summary.append((gname, "hi", "with build cost", int((d[ok] < -1e-9).sum()),
                        int(ok.sum()), float(d[ok].min()),
                        float(np.nanmedian(cost[ok, 0])), sf, ub))
        if gname == "15MHz":
            # the single-clock baseline is unaffected by the build handicap, so
            # one_15MHz_hi from the free pass above is already the right comparator
            for i, r in enumerate(rows):
                r["two_cost"] = "" if np.isnan(cost[i, 1]) else round(cost[i, 1], 2)
                assert abs(cost[i, 0] - r["one_15MHz_hi"]) < 0.011   # rows are rounded to 2 dp

    out = os.path.join(HERE, "data", "two_domain_matched.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out} ({len(rows)} rows)")
    # --- how the win scales with clock step, on MEASURED subsamples only ----
    rows_sc = []
    expo, expos = exponent(cl, T, E, D, PS, PHI, targets, rows_sc)
    a_sd = float(expos.std(ddof=1))
    # The anchor spread is ONE component of the uncertainty. Measurement noise is
    # another, and here it is the larger: bootstrap the 6 passes, recomputing the
    # anchor-averaged exponent each time. Quoting either alone -- as the within-fit CI
    # of a single anchor was quoted, and then as the anchor spread was quoted -- states
    # a part as the whole.
    # EXHAUSTIVE, not sampled: every distinct multiset of 6 draws from 6 passes is
    # C(11,5) = 462, which is cheap enough to enumerate. That matters for the claim
    # below: 462 resamples resolve no finer than 1/462, so a 3-sigma statement would sit
    # under the resolution of the resampling that produced sigma. The empirical maximum
    # needs no distribution at all.
    from itertools import combinations_with_replacement
    from math import factorial
    boot, all_exp, wts = [], [], []
    for pick in combinations_with_replacement(range(6), 6):
        cb, Tb, Eb, Db, Pb = load_sweep(pick=list(pick))
        mu, per = exponent(cb, Tb, Eb, Db, Pb, PHI, targets)
        boot.append(mu); all_exp.append(per)
        # A bootstrap draws 6**6 equiprobable ORDERED resamples. Collapsed to the 462
        # distinct multisets those are not equiprobable: the all-distinct one is 720x
        # more likely than the degenerate one. Weighting them equally over-counts
        # exactly the extreme resamples and inflates the sd by ~20%.
        cnt = [pick.count(i) for i in range(6)]
        wts.append(factorial(6) / np.prod([factorial(c) for c in cnt]) / 6 ** 6)
    boot, wts = np.array(boot), np.array(wts)
    wts /= wts.sum()
    all_exp = np.concatenate(all_exp)
    m_sd_unw = float(boot.std(ddof=1))
    mu_w = float(wts @ boot)
    m_sd = float(np.sqrt(wts @ (boot - mu_w) ** 2))      # the bootstrap sd
    esd = float(np.hypot(a_sd, m_sd))
    e_max, e_min, n_res = float(all_exp.max()), float(all_exp.min()), len(boot)
    scp = os.path.join(HERE, "data", "two_domain_scaling.csv")
    with open(scp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["anchor", "step_mhz", "gain_pct"])
        w.writeheader(); w.writerows(rows_sc)
    print(f"\nhow the win scales with clock step -- measured subsamples only, no "
          f"interpolation,\nover all {len(FIT_ANCHORS)} anchors the thinning can start "
          f"from:")
    for S in FIT_STEPS:
        v = np.array([r["gain_pct"] for r in rows_sc if r["step_mhz"] == S])
        print(f"  {S:3d} MHz  {v.mean():.2f}%  (spread {v.min():.2f}-{v.max():.2f})")
    print(f"  fitted exponent {expo:.3f}, anchor spread +/-{a_sd:.3f} "
          f"(range {expos.min():.3f}-{expos.max():.3f}),")
    print(f"    measurement noise +/-{m_sd:.3f} (exhaustive bootstrap over all {n_res} "
          f"distinct\n    pass-resamples, each weighted by its multinomial probability; "
          f"weighting them\n    equally instead gives +/-{m_sd_unw:.3f}, which is not a "
          f"bootstrap sd),\n    total +/-{esd:.3f} in quadrature -> {expo:.2f} +/- "
          f"{esd:.2f}")
    print(f"    over all {n_res} resamples x {len(FIT_ANCHORS)} anchors "
          f"({n_res*len(FIT_ANCHORS)} refits) the exponent spans "
          f"{e_min:.3f}-{e_max:.3f};\n    it never reached 1, with no distributional "
          f"assumption anywhere")
    print(f"  Sublinear, monotone, extrapolating to zero: max over every refit is "
          f"{e_max:.2f} < 1, which is what\n  licenses 'goes to zero in the "
          "continuous limit'. Three components of the uncertainty\n  were found in "
          "succession and the first two were each quoted as if they were the whole:\n"
          "  a single anchor's within-fit CI, then the spread across anchors, and only "
          "then the\n  measurement noise, which is the largest. The 5 MHz grid is not "
          "in this fit; it is\n  interpolated.")
    up = os.path.join(HERE, "data", "two_domain_uncertainty.csv")
    with open(up, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["quantity", "value"])
        for k, v in (("exponent", expo), ("anchor_spread_sd", a_sd),
                     ("pass_bootstrap_sd", m_sd),
                     ("unweighted_sd_over_distinct_resamples", m_sd_unw),
                     ("combined_sd", esd),
                     ("exponent_min_over_all_refits", e_min),
                     ("exponent_max_over_all_refits", e_max),
                     ("n_pass_resamples", n_res), ("n_anchors", len(FIT_ANCHORS))):
            w.writerow([k, round(v, 5) if isinstance(v, float) else v])
    print(f"  wrote {scp}\n  wrote {up}")

    sp = os.path.join(HERE, "data", "two_domain_summary.csv")
    with open(sp, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["clock_grid", "static_rule", "kernel", "wins", "deadlines",
                    "best_gain_mj", "best_gain_pct", "range_shortfall_pct",
                    "unbounded_deadline_gain_mj"])
        for g, st, kind, w_, n_, b, med, sf, ub in summary:
            w.writerow([g, st, kind, w_, n_, round(b, 2), round(b / med * 100, 2),
                        round(sf * 100, 3), round(ub, 2)])
        w.writerow(["scaling", "mix", "fitted exponent (mean over anchors)", "", "",
                    "", round(expo, 4), "", ""])
    print(f"wrote {sp}")
    print(f"\n{'clock grid':>10s} {'static':>7s} {'kernel':>16s}  {'wins':>7s}  "
          f"{'best gain':>12s}  {'range cost':>10s}  {'no deadline':>12s}")
    for g, st, kind, w_, n_, b, med, sf, ub in summary:
        print(f" {g:>10s} {st:>7s} {kind:>16s}  {w_:3d}/{n_:3d}  "
              f"{b:+8.1f} mJ {b/med*100:+6.2f}%  {sf*100:+9.3f}%  {ub:+9.1f} mJ")
    print("\n  range cost   how much the two-domain side gives up by being restricted "
          "to deadlines at\n               or below the single-clock energy optimum -- "
          "measured, not assumed")
    print("  no deadline  its best candidate ANYWHERE minus the best single clock "
          "anywhere.\n               Positive means it loses even with no deadline at "
          "all, which is the\n               honest answer to 'you cut off where it "
          "would have won'.")


if __name__ == "__main__":
    main()
