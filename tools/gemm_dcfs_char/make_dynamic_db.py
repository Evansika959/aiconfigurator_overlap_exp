#!/usr/bin/env python3
"""Join the P0 static-power table onto the sweep and emit dynamic energy.

    P_dynamic = P_board(measured while the GEMM runs) - P_static(achieved clock, n_squat)
    E_dynamic = P_dynamic * latency

That subtraction is the point of the whole exercise: NVML reports BOARD power, and
the static part is drawn whether or not a GEMM is running, so it is not attributable
to the kernel. What is left is what the GEMM actually cost.

P_static comes from zeus's own profile_p2p.py, run UNMODIFIED at each locked clock
(`run_p2p_static.py` -> `data/p0_static_zeus.csv`): rank 0 parked in a blocking
dist.recv() for 60 s while rank 1 sleeps, board energy integrated by NVML.
3-rep spread 0.13-0.33 W.

Three things the join has to get right:

* SUBTRACT AT THE ACHIEVED CLOCK, NOT THE REQUESTED ONE. 21 of 360 cells hit the
  400 W SwPowerCap and ran at 1245-1380 MHz instead of 1410. P_static rises steeply
  with clock (56.4 W at 300 MHz -> 85.4 W at 1410 on this board), so using the
  requested clock there would over-subtract. P_static is interpolated linearly in
  clock.

* THE SQUATTER'S OWN POWER IS *NOT* REMOVED, BY CHOICE. zeus measures a bare board
  plus a resident NCCL kernel; the sweep's board additionally carries the
  __nanosleep squatter, whose fixed cost was measured separately at 0.9 W (300 MHz)
  to 7.2 W (1410 MHz), flat in the number of SMs held. Subtracting only the zeus
  number therefore charges that difference to the GEMM. This is deliberate -- one
  method, end to end -- and the size of the effect is carried in `e_dyn_squat_mj`
  (same quantity using the squatter-matched baseline from `p0_static.csv`) so it is
  visible in the data rather than hidden. It is ~0.4 % of E_dyn at 108 SM and ~11 %
  at 14 SM, where static dominates the board power.

* A CONSTANT SUBTRAHEND CANNOT MOVE A SLOPE. P_static enters as a per-clock constant,
  flat along the SM axis, so switching baselines shifts the intercept `b` of
  P_dyn = a·S_eff + b and leaves `a` -- and the κ = a/f voltage curve derived from it
  -- untouched. The zeus table sits a near-uniform 3.0-3.4 W below the squatter table
  at every clock, so that shift is ~3 W into `b` and nothing else.

  python3 make_dynamic_db.py                      # -> data/gemm_dynamic_energy.csv
  python3 make_dynamic_db.py --static-source squatter    # the other baseline
"""

import argparse
import collections
import csv
import os
import statistics

FIELDS_EXTRA = ["p_static_w", "p_static_squat_w", "p_dynamic_w",
                "energy_total_mj", "energy_dynamic_mj", "energy_static_mj",
                "static_frac", "p_dynamic_per_sm_w", "e_dyn_squat_mj"]


def load_zeus(path):
    """-> [(clock, power), ...] sorted.  zeus profile_p2p.py, one curve, no SM axis."""
    by = collections.defaultdict(list)
    for r in csv.DictReader(open(path)):
        by[int(r["freq_req_mhz"])].append(float(r["power_w"]))
    return sorted((c, statistics.median(v)) for c, v in by.items())


def load_squatter(path):
    """-> {n_squat: [(clock, power), ...] sorted}, the squatter-matched baseline."""
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in csv.DictReader(open(path)):
        if r["status"] != "ok" or r["clock_held"] != "True":
            continue
        by[int(r["n_squat"])][int(r["clock_sm_med"])].append(float(r["power_w"]))
    return {n: sorted((c, statistics.median(v)) for c, v in d.items()) for n, d in by.items()}


def interp(curve, x):
    """Linear interpolation in clock, clamped at both ends."""
    if x <= curve[0][0]:
        return curve[0][1]
    if x >= curve[-1][0]:
        return curve[-1][1]
    for (x0, y0), (x1, y1) in zip(curve, curve[1:]):
        if x0 <= x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return curve[-1][1]


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--sweep", default=os.path.join(here, "data", "gemm_squat_sweep.csv"))
    ap.add_argument("--zeus", default=os.path.join(here, "data", "p0_static_zeus.csv"))
    ap.add_argument("--squatter", default=os.path.join(here, "data", "p0_static.csv"))
    ap.add_argument("--static-source", choices=("zeus", "squatter"), default="zeus")
    ap.add_argument("--out", default=os.path.join(here, "data", "gemm_dynamic_energy.csv"))
    a = ap.parse_args()

    zeus = load_zeus(a.zeus)
    squat = load_squatter(a.squatter)
    print(f"P_static source = {a.static_source}")
    print("  zeus profile_p2p.py:  " + "  ".join(f"{c}MHz:{p:.2f}W" for c, p in zeus))
    print("  squatter, n>=1:       " + "  ".join(
        f"{c}MHz:{p:.2f}W" for c, p in squat[54]))
    print("  difference (zeus - squatter): " + "  ".join(
        f"{c}:{interp(zeus, c) - p:+.2f}W" for c, p in squat[54]))

    rows = list(csv.DictReader(open(a.sweep)))
    fields = list(rows[0].keys()) + FIELDS_EXTRA
    missing = sorted({int(r["n_squat"]) for r in rows} - set(squat))
    if missing:
        raise SystemExit(f"squatter table has no occupancy {missing}; rerun profile_static.py")

    neg = 0
    with open(a.out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        for r in rows:
            if r["status"] != "ok":
                wr.writerow(r)
                continue
            n = int(r["n_squat"])
            clk = int(r["clock_sm_med"])          # ACHIEVED, not requested
            lat = float(r["latency_ms"])
            p_tot = float(r["power_w"])
            p_zeus = interp(zeus, clk)
            p_squat = interp(squat[n], clk)
            p_st = p_zeus if a.static_source == "zeus" else p_squat
            p_dyn = p_tot - p_st
            if p_dyn <= 0:
                neg += 1
            sm = int(r["sm_avail"])
            out = dict(r)
            out.update(
                p_static_w=round(p_st, 3),
                p_static_squat_w=round(p_squat, 3),
                p_dynamic_w=round(p_dyn, 3),
                energy_total_mj=round(p_tot * lat, 4),
                energy_dynamic_mj=round(p_dyn * lat, 4),
                energy_static_mj=round(p_st * lat, 4),
                static_frac=round(p_st / p_tot, 4),
                p_dynamic_per_sm_w=round(p_dyn / sm, 4),
                # the same quantity computed against the squatter-matched baseline.
                # The gap is exactly what choosing one method end-to-end costs, and
                # it belongs in the data, not in a footnote.
                e_dyn_squat_mj=round((p_tot - p_squat) * lat, 4),
            )
            wr.writerow(out)
    print(f"\nwrote {a.out}   ({len(rows)} rows, {neg} with P_dynamic <= 0)")

    # --- summary -------------------------------------------------------------
    D = [r for r in csv.DictReader(open(a.out)) if r["status"] == "ok"]
    sf = [float(r["static_frac"]) for r in D]
    pd = [float(r["p_dynamic_w"]) for r in D]
    print(f"\n静态占整板功耗: 中位 {statistics.median(sf) * 100:.0f}%  "
          f"范围 {min(sf) * 100:.0f}%-{max(sf) * 100:.0f}%     "
          f"最小 P_dynamic {min(pd):.2f} W")
    gap = [abs(float(r["energy_dynamic_mj"]) - float(r["e_dyn_squat_mj"]))
           / float(r["energy_dynamic_mj"]) for r in D]
    print(f"选 zeus 而非占位基线的代价 (|ΔE_dyn|): 中位 {statistics.median(gap) * 100:.1f}%  "
          f"p90 {sorted(gap)[int(.9 * len(gap))] * 100:.1f}%  max {max(gap) * 100:.1f}%")
    by = collections.defaultdict(list)
    for r, e in zip(D, gap):
        by[int(r["sm_avail"])].append(e)
    print("  按 SM 数: " + "  ".join(
        f"{s}SM {statistics.median(by[s]) * 100:.1f}%" for s in sorted(by, reverse=True)))


if __name__ == "__main__":
    main()
