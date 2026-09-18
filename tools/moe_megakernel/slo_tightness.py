#!/usr/bin/env python3
"""What a second V/f domain is worth, as a function of how tight the SLO is.

I previously concluded it was worth ~0, but that answer silently assumed a LOOSE deadline:
if every expert is free to sit at the kappa knee, they all pick the same clock and there
is nothing for a second domain to do. The user pointed out the gap.

When the SLO forces the CRITICAL expert above the knee, the picture inverts. The busiest
expert has to race at, say, 1410; the light ones still meet the same deadline at a much
lower clock, and dropping them BELOW the knee is worth the full 42-47% per operation that
crossing it buys. The second domain pays exactly when the deadline is tight.

MODEL, all of it measured rather than assumed:
    expert e is compute bound (verified: T x f constant to 1.02x), so its time is
        T_e = W_e / (k * s_e * f_e),  W_e proportional to its token count n_e
    its dynamic energy is
        E_e = kappa(f_e) * W_e            <- work is fixed; only kappa depends on f
    so under a deadline T every expert picks the lowest feasible clock, and never goes
    below the knee because kappa stops falling there:
        f_e* = max( W_e / (k s_e T), f_knee )

kappa(f) is the measured moe_expert_n7652 curve from kappa_sweep.csv, interpolated.
n_e comes from the measured OLMoE routing traces. Two SM allocation policies, because
the whole effect lives in how much slack the allocation leaves:
    equal          every expert gets the same SM share  -> light experts have huge slack
    proportional   SM share tracks load (largest remainder, min 1 SM) -> almost no slack
"""
import csv, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SM, KNEE = 108, 1050
CL = np.array(sorted({int(r["clock"]) for r in
                      csv.DictReader(open(os.path.join(HERE, "data", "kappa_sweep.csv")))}))


def kappa_curve():
    d = {}
    for r in csv.DictReader(open(os.path.join(HERE, "data", "kappa_sweep.csv"))):
        if r["workload"] == "moe_expert_n7652" and r["held"] == "True":
            d[int(r["clock"])] = float(r["kappa_mw_per_mhz"])
    f = np.array(sorted(d), float)
    return f, np.array([d[int(x)] for x in f])


FK, KK = kappa_curve()
kap = lambda f: np.interp(f, FK, KK)
GRID = FK[FK >= 300]


def allocate(n, policy):
    """SMs per expert. Only experts with load participate."""
    act = n > 0
    s = np.zeros_like(n, dtype=float)
    if policy == "equal":
        s[act] = SM / act.sum()
    else:
        share = n[act] / n[act].sum() * SM
        q = np.floor(share); q[q < 1] = 1
        left = int(SM - q.sum())
        if left > 0:
            q[np.argsort(-(share - np.floor(share)))[:left]] += 1
        elif left < 0:                      # more active experts than SMs
            q[:] = SM / act.sum()
        s[act] = q
    return s


def solve(n, policy, f_crit):
    """Deadline is set by the critical expert running at f_crit. Return (E_one, E_multi)
    in arbitrary but consistent units, plus the clock each expert lands on."""
    s = allocate(n, policy)
    act = n > 0
    t = np.zeros_like(n, dtype=float)
    t[act] = n[act] / s[act]                       # time * f, i.e. work per SM
    T = t.max() / f_crit                           # the deadline
    need = np.where(act, t / max(T, 1e-12), 0.0)   # minimum feasible clock per expert
    f_e = np.clip(np.maximum(need, KNEE), None, FK.max())
    f_e = np.where(act, f_e, KNEE)
    # snap to the measured clock grid, upward so the deadline is still met
    f_snap = np.array([GRID[np.searchsorted(GRID, x)] if x <= GRID[-1] else GRID[-1]
                       for x in f_e])
    E_one = (kap(f_crit) * n).sum()
    E_multi = (kap(f_snap) * n).sum()
    return E_one, E_multi, f_snap[act]


def work_conserving(n, T):
    """One clock, no pinning: blocks pull from a shared queue, so the makespan is set by
    the TOTAL work over all 108 SMs, not by the busiest expert. This is the single-clock
    design a megakernel actually gives you, and it is the baseline the multi-domain arm
    has to beat -- comparing against a PINNED single clock stacks the deck, because
    pinning is what created the slack in the first place."""
    return n.sum() / (SM * max(T, 1e-12))


def main():
    Z = np.load(os.path.join(HERE, "data", "routing_olmoe.npz"))
    print("Three designs at the SAME deadline. The deadline is set by what a PINNED\n"
          "partition needs, parameterised by the clock its critical expert must run at.\n"
          "Both savings are against `one clock, pinned` -- which is the design that has\n"
          "to run at that critical clock.\n")
    for key in ("prefill_b16", "prefill_b128"):
        C = Z[key].reshape(-1, 64).astype(float)
        print(f"  {key}   ({C.shape[0]} layer-samples)")
        print(f"    {'SM policy':<14}{'critical f':>12}{'2 domains':>11}"
              f"{'work-conserv.':>15}{'its clock':>11}")
        for policy in ("equal", "proportional"):
            for f_crit in (1050, 1200, 1305, 1410):
                a = b = c = 0.0
                fw = []
                for row in C:
                    e1, e2, fs = solve(row, policy, f_crit)
                    s_ = allocate(row, policy)
                    T = (row / np.where(s_ > 0, s_, 1))[row > 0].max() / f_crit
                    f_wc = max(work_conserving(row, T), KNEE)
                    f_wc = GRID[np.searchsorted(GRID, min(f_wc, GRID[-1]))]
                    a += e1; b += e2; c += kap(f_wc) * row.sum(); fw.append(f_wc)
                print(f"    {policy:<14}{f_crit:>12}{(b-a)/a*100:>+10.1f}%"
                      f"{(c-a)/a*100:>+14.1f}%{f'{np.mean(fw):.0f} MHz':>11}")
            print()


if __name__ == "__main__":
    main()
