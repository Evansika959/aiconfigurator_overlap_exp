#!/usr/bin/env python3
"""FOLLOW-UP 2 -- how much of the gain survives when the V/f domains are quantized?

THE GAP THIS CLOSES. Every two-clock result so far lets the solver pick each domain's
frequency from a continuum and split the SMs at any boundary. No hardware works that
way. A real design fixes two things in advance:

    how many V/f LEVELS a domain can select      (a voltage regulator per level costs area)
    at what SM GRANULARITY the chip can split    (a domain boundary cannot fall mid-GPC)

Both make the counterfactual worse, and the question is by how much. If the gain
survives 3 levels at 8-SM granularity, that is a buildable proposal. If it needs a
continuum, it is a curiosity.

HOW THE LEVELS ARE CHOSEN. For n levels the script enumerates every subset of that size
from the six measured clocks and keeps the subset that maximises the MEDIAN gain across
all workloads -- one set of levels for the whole chip, as a hardware designer would have
to commit to, not a different set per workload. That makes the reported figure an upper
bound on what n levels can achieve, which is the right side to err on when the answer is
"this is not enough".

SM GRANULARITY g means the collective's CTA count must be a multiple of g, since one CTA
occupies one SM. The measured set is {1,2,4,8,16,32}, so g filters it.

THE REFERENCE IS FIXED, AND THAT MATTERS. A first version compared the two-clock arm
against the one-clock arm UNDER THE SAME QUANTIZATION. That inverts the answer: with only
two levels the one-clock baseline is crippled (it must run both kernels at 300 or both at
1410), so the relative gain balloons to -7.8% median and -73% best, making coarse hardware
look BETTER than fine hardware. It also produced positive "gains", because under a
lexicographic min-latency objective the two-clock arm can find a strictly faster point
that costs more energy.

So everything below is measured against ONE fixed reference: the best a single-domain
chip can do with the full six levels and no SM-granularity restriction. That is the thing
a second domain has to beat, and it does not move as the counterfactual hardware is
coarsened. AND THE LATENCY IS MATCHED. Reporting energy under a lexicographic min-latency objective
was still wrong: with six levels the two-domain chip can run 12.7% FASTER, so the
objective spends the freedom on speed and only 1.4% of energy, while with two levels it
cannot go faster and spends everything on energy, showing -13.1%. That reads as "coarser
hardware saves more", which is nonsense. So the two-domain arm is held to the reference's
own latency -- min E subject to T <= T_ref -- and the question becomes the only one that
is well posed: at the SAME speed as the best single-domain chip, how much energy does a
quantized two-domain chip save?

INFEASIBILITY IS A RESULT, NOT AN EXCLUSION. Coarse levels can leave a workload unable to
match the reference latency at all -- with two levels only 15 of 72 pairs can. Dropping
those and taking the median over the survivors is a selection bias that made two levels
look like the best design (-25.8%), because the survivors are exactly the pairs where the
gain is largest. They are counted as zero saving instead: if the two-domain chip cannot
hit the deadline you would deploy the single-domain configuration and gain nothing. Both
the feasible fraction and the median over feasible pairs are reported alongside, so the
two effects stay separable.

  python3 followup_quantized.py
"""
import csv
import itertools
import os
import statistics as st

import followup_baseline as FB

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = 400.0
ALL_CLOCKS = [300, 510, 705, 900, 1200, 1410]
ALL_CTAS = [1, 2, 4, 8, 16, 32]


def main():
    rows = list(csv.DictReader(open(os.path.join(HERE, "data", "overlap_432.csv"))))
    held = lambda r: r["clock_held"] == "True"
    G = {(int(r["clock"]), int(r["m"]), int(r["n"])): r
         for r in rows if r["mode"] == "gemm_only" and held(r)}
    C = {(int(r["clock"]), int(r["ctas"])): r
         for r in rows if r["mode"] == "comm_only" and held(r)}
    shapes = sorted({(int(r["m"]), int(r["n"]), int(r["grid"]))
                     for r in rows if r["mode"] == "concurrent"})

    def opt(clocks, ctas, obj, two_domain):
        """per (shape, CTA) optimum under this quantization -> {key: (E, T)}"""
        res = {}
        for (m, n, grid) in shapes:
            for c in ctas:
                cand = []
                for fg in clocks:
                    g = G.get((fg, m, n))
                    if not g:
                        continue
                    for fc in (clocks if two_domain else [fg]):
                        cm = C.get((fc, c))
                        if not cm:
                            continue
                        ti, E, pk = FB.compose(g, cm, fg, fc, grid)
                        if E / ti > CAP or pk > CAP:
                            continue
                        cand.append((E, ti))
                if cand:
                    res[(m, n, c)] = FB.pick(cand, obj)
        return res

    REF = opt(ALL_CLOCKS, ALL_CTAS, "T", False)      # best single-domain chip, unrestricted

    def gain(clocks, ctas, obj=None):
        """min E subject to T <= T_ref, against the fixed reference. Latency matched."""
        de = []
        for (m, n, grid) in shapes:
            for c in ctas:
                key = (m, n, c)
                if key not in REF:
                    continue
                Eref, Tref = REF[key]
                best = None
                for fg in clocks:
                    g = G.get((fg, m, n))
                    if not g:
                        continue
                    for fc in clocks:
                        cm = C.get((fc, c))
                        if not cm:
                            continue
                        ti, E, pk = FB.compose(g, cm, fg, fc, grid)
                        if E / ti > CAP or pk > CAP or ti > Tref * 1.0001:
                            continue
                        if best is None or E < best:
                            best = E
                # infeasible -> you fall back to the single-domain config, gain 0
                de.append((best - Eref) / Eref * 100 if best is not None else 0.0)
        # the MEAN, not the median. More than half of these workloads are GEMM-bound and
        # gain exactly zero, so the median saturates at the same value for every design
        # and stops discriminating -- it made all four level counts tie, and the "best
        # subset" search then returned whatever the tie-break happened to hit. The mean
        # responds to both how often a design helps and by how much, which is what
        # choosing between designs requires.
        feas = [x for x in de if x != 0.0]
        if not de:
            return (0.0, 0.0, 0.0, 0.0)
        return (st.mean(de), min(de), len(feas) / len(de) * 100,
                st.median(feas) if feas else 0.0)

    # ---- best level subset for each n, chosen once for the whole chip ---------------
    print("Which V/f levels would you build, if you could only have n of them?")
    print("(one subset for the whole chip, chosen to maximise the MEAN saving)\n")
    best_set = {}
    print(f"  {'n levels':>10}{'best subset (MHz)':>34}{'mean dE':>10}{'best dE':>10}"
          f"{'feasible':>10}{'med|feas':>10}")
    for n in (2, 3, 4, 6):
        cands = []
        for sub in itertools.combinations(ALL_CLOCKS, n):
            med, bst, dt, fm = gain(list(sub), ALL_CTAS)
            cands.append((med, bst, dt, fm, sub))
        med, bst, dt, fm, sub = min(cands)
        best_set[n] = list(sub)
        print(f"  {n:>10}{', '.join(str(x) for x in sub):>34}{med:>+9.2f}%{bst:>+9.2f}%"
              f"{dt:>9.0f}%{fm:>+9.2f}%")

    # ---- the 2-D sweep --------------------------------------------------------------
    print(f"\n\nMEAN ENERGY vs the FIXED reference (best single-domain chip, 6 levels,")
    print(f"granularity 1). Negative = the quantized two-domain chip is cheaper.")
    print(f"  rows: SM granularity (collective CTA count must be a multiple)")
    print(f"  cols: number of V/f levels the chip provides\n")
    grans = [1, 2, 4, 8, 16, 32]
    print(f"  each cell: mean dE over ALL 72 pairs (infeasible counted as zero), then")
    print(f"  the % of pairs that could meet")
    print(f"  the reference latency at all\n")
    print(f"  {'granularity':>12}{'CTAs left':>12}" + "".join(f"{f'{n} levels':>15}" for n in (2, 3, 4, 6)))
    tab = []
    for g in grans:
        cs = [c for c in ALL_CTAS if c % g == 0]
        line = f"  {g:>12}{','.join(str(x) for x in cs):>12}"
        row = dict(sm_granularity=g, ctas_available=" ".join(str(x) for x in cs))
        for n in (2, 3, 4, 6):
            med, bst, feas, fm = gain(best_set[n], cs)
            line += f"{med:>+9.2f}%{feas:>5.0f}%"
            row[f"n{n}_mean_dE_pct"] = round(med, 2)
            row[f"n{n}_best_dE_pct"] = round(bst, 2)
            row[f"n{n}_feasible_pct"] = round(feas, 1)
            row[f"n{n}_median_feasible_pct"] = round(fm, 2)
        print(line); tab.append(row)
    p = os.path.join(HERE, "data", "followup_quantized.csv")
    with open(p, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(tab[0].keys()))
        wr.writeheader(); wr.writerows(tab)
    print(f"\n  wrote {p}")


if __name__ == "__main__":
    main()
