#!/usr/bin/env python3
"""The overlap optimum under three objectives: energy, latency, energy-delay product.

WHY ALL THREE. Minimising energy alone is not what a deployment planner wants -- it
gives up a median 37% latency to save 19% energy, which no serving system accepts.
Minimising latency alone ignores the power bill entirely. EDP is a compromise but its
weighting is arbitrary and has no physical meaning. None of the three is the "right"
objective, so report all three and let the SLA choose.

TIE-BREAKING MATTERS, and getting it wrong is what made an earlier version of this
analysis wrong. Under a pure latency objective every arm reaches the SAME minimum T --
just set every clock to maximum. Comparing arms on T alone therefore shows a 0.00%
difference and hides the real result, which is that at that same T the arms differ
enormously in ENERGY. So each objective is optimised lexicographically: primary key
first, then the other quantity as the tie-break, and both are reported.

    objective E     min E, tie-break min T
    objective T     min T, tie-break min E     <- the tie-break is the whole story here
    objective EDP   min E*T

ARMS (see OPPORTUNITY.md):
    A  soft partition, one clock   -- today's hardware
    B  hard partition, one clock
    C  hard partition, two clocks  -- B's search space is nested inside C's

  python3 objectives.py
"""
import csv, json, os, statistics as st
import optimise_overlap as O

HERE = os.path.dirname(os.path.abspath(__file__))
FG = list(range(300, 1411, 15))
TOT = 108


def candidates(grid, nk, sk, gb, mode):
    """every (c, f_g, f_c) the arm can reach, with its (E, T)"""
    out = []
    w = -(-grid // TOT)
    for c in range(1, 33):
        sg = TOT - c
        w2 = -(-grid // max(sg, 1))
        for f1 in FG:
            if mode == "soft":
                tg = O.t_gemm(f1, TOT, grid, nk, sk)
                tc = O.t_comm(f1, c, gb)
                T = max(tg * (1 + (w2 / w - 1) * min(1, tc / tg)), tc)
                E = (O.t.lin(O.t.PS, f1) * T + O.p_gemm(f1, TOT, grid, nk, nk) * tg
                     + O.p_comm(f1, c) * tc)
                pk = O.t.lin(O.t.PS, f1) + O.p_gemm(f1, TOT, grid, nk, nk) + O.p_comm(f1, c)
                out.append((E, T, c, f1, f1, TOT, pk))
                continue
            tg = O.t_gemm(f1, sg, grid, nk, sk)
            pg = O.p_gemm(f1, sg, grid, nk, nk)
            for f2 in (FG if mode == "hard2" else [f1]):
                tc = O.t_comm(f2, c, gb)
                T = max(tg, tc)
                E = O.p_static2(f1, f2, sg, c) * T + pg * tg + O.p_comm(f2, c) * tc
                pk = O.p_static2(f1, f2, sg, c) + pg + O.p_comm(f2, c)
                out.append((E, T, c, f1, f2, sg, pk))
    return out


POWER_CAP = 400.0          # A100-SXM4-40GB board limit
PEAK_CAP = True            # also require the instantaneous peak to fit under it


def feasible(cands, cap=POWER_CAP, peak_cap=None):
    """Drop candidates the board cannot actually sustain.

    The solver searches frequency up to 1410 MHz with no power constraint, and for large
    GEMMs that lands on configurations drawing 600 W mean against a 400 W cap -- the
    hardware would clock down instead, so those points do not exist. 78 of 288 optima
    were infeasible before this filter, almost all of them under the latency objective,
    which is exactly the objective that pushes both clocks to maximum.

    The criterion is the ITERATION MEAN, not the instantaneous peak. That is not a
    modelling convenience: it was measured. Across the 432-case sweep the mean predicted
    throttling with 88.6% precision and 100% recall (98.8% accuracy) while the peak
    over-flagged by 3x (31.5% precision). The power controller integrates over >=12 ms
    and a GEMM wave is 0.1-1 ms, so a short high-power phase never survives its window --
    the figure's one-clock case peaks at 486 W for 0.73 ms yet averages 282 W and does
    not throttle.

    Filtering rather than modelling the throttled state is equivalent here: a throttled
    GPU runs the same SM allocation at a lower clock, which is already another candidate
    in the search.

    THE PEAK IS ALSO CAPPED, BY DEFAULT, AND THAT IS A DELIBERATE OVER-CONSTRAINT. The
    measured evidence says it should not be needed: the model's peak exceeded 400 W in
    124 of the 432 measured cases and only 39 of those throttled, so a short high-power
    phase demonstrably survives the controller's window. But 25% of the optima this
    search returns peak above the cap -- up to 498 W -- and a figure whose headline case
    visibly exceeds the board limit invites an objection it cannot answer on its own.

    The cost of being conservative is small and confined to one objective: min energy and
    min EDP are completely unchanged, and min latency drops from -22.0% to -15.9% median
    on the comm-bound workloads (best -26.4% to -21.3%). Only min latency is affected
    because it is the only objective that pushes a clock to maximum. Pass peak_cap=False
    to recover the unconstrained numbers.
    """
    if peak_cap is None:
        peak_cap = PEAK_CAP
    keep = [c for c in cands
            if c[0] / c[1] <= cap and (not peak_cap or len(c) < 7 or c[6] <= cap)]
    return keep or [min(cands, key=lambda z: z[0] / z[1])]


def pick(cands, obj, cap=POWER_CAP, peak_cap=None):
    """lexicographic: primary objective, then the other quantity as tie-break"""
    cands = feasible(cands, cap, peak_cap)
    if obj == "EDP":
        return min(cands, key=lambda z: z[0] * z[1])
    pri, sec = (0, 1) if obj == "E" else (1, 0)
    best = min(c[pri] for c in cands)
    near = [c for c in cands if c[pri] <= best * (1 + 1e-9)]
    return min(near, key=lambda z: z[sec])


def main():
    G = json.load(open(os.path.join(HERE, "data", "grid12.json")))
    rows = []
    for nk in (4096, 8192, 16384):
        for m in (1024, 2048, 4096, 8192):
            grid = G[f"{m}_{nk}_{nk}"]
            sk = max(1, grid // (m * nk // 32768))
            for mib in (8, 32, 128, 256):
                gb = mib * 2 ** 20 / 1e9
                cand = {a: candidates(grid, nk, sk, gb, a)
                        for a in ("soft", "hard1", "hard2")}
                ratio = O.t_comm(900, 16, gb) / O.t_gemm(900, TOT, grid, nk, sk)
                r = dict(m=m, nk=nk, mib=mib, tc_over_tg=round(ratio, 3))
                for obj in ("E", "T", "EDP"):
                    p = {a: pick(cand[a], obj) for a in cand}
                    # MATCHED-BUDGET comparison. The loop above optimises each arm
                    # independently, so the two can land at different T -- under min
                    # energy arm C typically lands SHORTER, because it can slow the GEMM
                    # without slowing the collective, while a single clock has to slow
                    # both and the collective is the critical path. Quoting the energy
                    # gap alone then understates the two-clock arm, which is better on
                    # both axes. This column re-asks the question at a common deadline:
                    # holding T at whatever the one-clock arm achieved, how much less
                    # energy does the two-clock arm need? Only the latency objective is
                    # already controlled -- there both arms are pinned to the same T.
                    tb = p["hard1"][1]
                    at_tb = [q for q in feasible(cand["hard2"]) if q[1] <= tb * 1.0001]
                    ec = min(at_tb, key=lambda z: z[0])[0] if at_tb else float("nan")
                    r[f"matched_{obj}_pct"] = round(
                        (ec - p["hard1"][0]) / p["hard1"][0] * 100, 2)
                    for a, tag in (("soft", "A"), ("hard1", "B"), ("hard2", "C")):
                        E, T, c, fg, fc, sg = p[a][:6]
                        r[f"{tag}_{obj}_mJ"] = round(E, 1)
                        r[f"{tag}_{obj}_ms"] = round(T, 4)
                        r[f"{tag}_{obj}_set"] = f"f{fg}/{fc} c{c}"
                    for tag, base in (("CvB", "hard1"), ("CvA", "soft")):
                        for q, i in (("mJ", 0), ("ms", 1)):
                            r[f"{tag}_{obj}_{q}_pct"] = round(
                                (p["hard2"][i] - p[base][i]) / p[base][i] * 100, 2)
                rows.append(r)
    path = os.path.join(HERE, "data", "objectives.csv")
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)

    print(f"48 workloads (12 GEMM shapes x 4 message sizes), world=4\n")
    print("C vs B -- the pure second-DVFS-domain effect, by objective.")
    print("Both E and T reported under every objective, because an arm that ties on the")
    print("objective can still differ enormously on the other quantity.\n")
    bins = [(0, 0.5, "0 - 0.5"), (0.5, 1.5, "0.5 - 1.5"),
            (1.5, 4, "1.5 - 4"), (4, 1e9, "> 4")]
    for obj in ("E", "T", "EDP"):
        print(f"  objective = {obj}")
        print(f"    {'t_c/t_g':>10}{'n':>4} |{'dE':>10}{'dT':>10} |{'dE @ matched T':>16}")
        for lo, hi, lab in bins:
            v = [x for x in rows if lo <= x["tc_over_tg"] < hi]
            if not v:
                continue
            print(f"    {lab:>10}{len(v):>4} |"
                  f"{st.median(x[f'CvB_{obj}_mJ_pct'] for x in v):>+9.2f}%"
                  f"{st.median(x[f'CvB_{obj}_ms_pct'] for x in v):>+9.2f}% |"
                  f"{st.median(x[f'matched_{obj}_pct'] for x in v):>+15.2f}%")
        print(f"    {'ALL':>10}{len(rows):>4} |"
              f"{st.median(x[f'CvB_{obj}_mJ_pct'] for x in rows):>+9.2f}%"
              f"{st.median(x[f'CvB_{obj}_ms_pct'] for x in rows):>+9.2f}% |"
              f"{st.median(x[f'matched_{obj}_pct'] for x in rows):>+15.2f}%")
        print(f"    {'best':>10}{'':>4} |"
              f"{min(x[f'CvB_{obj}_mJ_pct'] for x in rows):>+9.2f}%{'':>10} |"
              f"{min(x[f'matched_{obj}_pct'] for x in rows):>+15.2f}%\n")
    print(f"  wrote {path}")


if __name__ == "__main__":
    main()
