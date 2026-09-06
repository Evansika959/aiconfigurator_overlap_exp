#!/usr/bin/env python3
"""FOLLOW-UP 1 -- compare against the TRUE baseline: what the GPU actually ran.

THE GAP THIS CLOSES. Every two-clock number so far compares against a one-clock arm that
was itself COMPUTED -- either from the fitted model, or by composing two measured solo
kernels at equal clocks. Neither is what the hardware does. data/overlap_432.csv already
contains the real thing: 432 measured `concurrent` rows, the overlap actually executed on
four A100s at six clocks. That is the baseline a claim should be made against.

THREE BASELINES, SO THE COMPOSITION'S OWN ERROR IS VISIBLE

    B_measured   the best measured `concurrent` row over the six clocks. Reality, no
                 model. Includes throttling exactly as it happened.
    B_composed   the same arm reconstructed by composing measured solo kernels at
                 f_gemm == f_comm. Differs from B_measured only by composition error.
    C            two clocks, the counterfactual, same composition machinery.

    C vs B_measured   the number to quote -- but it carries the composition bias
    C vs B_composed   the clean, like-for-like comparison
    B_composed vs B_measured   IS the composition bias, reported so it can be discounted

WHY BOTH MATTER. If the composition systematically under-predicts the one-clock arm's
energy, then C vs B_measured flatters the counterfactual, and the size of that flattery
is exactly B_composed vs B_measured. Quoting one without the other hides it.

  python3 followup_baseline.py
"""
import csv
import os
import statistics as st

import table_432 as t

HERE = os.path.dirname(os.path.abspath(__file__))
TOTAL_SM, CAP = 108, 400.0
CLOCKS = [300, 510, 705, 900, 1200, 1410]


def ps_at(f):
    """Static floor at f. PS is calibrated at the six clocks of the original campaign; the
    extension campaign added 1050 and 1305, both BRACKETED by calibrated points (900-1200
    and 1200-1410), so this is interpolation and never extrapolation. Straight dict
    indexing was a KeyError on the new clocks -- and would have been the wrong fix even
    if it had not been, because it silently forbids any clock the fit never saw."""
    return t.lin(t.PS, f)


def compose(g, c, fg, fc, grid):
    tg, tc = float(g["iter_ms"]), float(c["iter_ms"])
    n = int(c["ctas"])
    w = -(-grid // TOTAL_SM)
    w2 = -(-grid // max(TOTAL_SM - n, 1))
    ti = max(tg * (1 + (w2 / w - 1) * min(1, tc / tg)), tc)
    pg = float(g["power_per_gpu_w"]) - ps_at(fg)
    pc = float(c["power_per_gpu_w"]) - ps_at(fc)
    floor = t.PS[min(t.PS)]
    ps = floor + ((TOTAL_SM - n) * (ps_at(fg) - floor) + n * (ps_at(fc) - floor)) / TOTAL_SM
    E = ps * ti + pg * tg + pc * tc
    peak = ps + pg + pc
    return ti, E, peak


def pick(cs, obj):
    """cs: list of (E, T, tag...). lexicographic with the other quantity as tie-break."""
    if not cs:
        return None
    if obj == "EDP":
        return min(cs, key=lambda z: z[0] * z[1])
    p, s = (0, 1) if obj == "E" else (1, 0)
    b = min(z[p] for z in cs)
    return min([z for z in cs if z[p] <= b * (1 + 1e-9)], key=lambda z: z[s])


def main():
    rows = list(csv.DictReader(open(os.path.join(HERE, "data", "overlap_432.csv"))))
    held = lambda r: r["clock_held"] == "True"
    G = {(int(r["clock"]), int(r["m"]), int(r["n"])): r
         for r in rows if r["mode"] == "gemm_only" and held(r)}
    C = {(int(r["clock"]), int(r["ctas"])): r
         for r in rows if r["mode"] == "comm_only" and held(r)}
    X = {(int(r["clock"]), int(r["ctas"]), int(r["m"]), int(r["n"])): r
         for r in rows if r["mode"] == "concurrent"}
    shapes = sorted({(int(r["m"]), int(r["n"]), int(r["grid"]))
                     for r in rows if r["mode"] == "concurrent"})
    ctas = sorted({int(r["ctas"]) for r in rows if r["mode"] == "concurrent"})

    out = []
    for (m, n, grid) in shapes:
        for c in ctas:
            # --- B_measured: what the hardware actually did, no model at all
            #
            # THE FILTER MUST MATCH THE OTHER TWO ARMS. A first version capped only the
            # measured MEAN here while capping mean AND peak on the composed arms, so
            # B_measured drew from a wider candidate set. Under min latency that alone
            # produced an apparent 21.7% "composition bias" -- B_measured was picking a
            # fast expensive row the composed arms were not allowed to consider. The
            # peak is not measurable, so it is taken from the composition at that same
            # clock, which is exactly what filters the other two arms.
            bm = []
            for f in CLOCKS:
                x = X.get((f, c, m, n))
                g, cm = G.get((f, m, n)), C.get((f, c))
                if not (x and g and cm):
                    continue
                T, P = float(x["iter_ms"]), float(x["power_per_gpu_w"])
                if P > CAP or compose(g, cm, f, f, grid)[2] > CAP:
                    continue
                bm.append((P * T, T, f))
            # --- B_composed and C, same machinery
            bc, cc = [], []
            for fg in CLOCKS:
                g = G.get((fg, m, n))
                if not g:
                    continue
                for fc in CLOCKS:
                    cm = C.get((fc, c))
                    if not cm:
                        continue
                    ti, E, pk = compose(g, cm, fg, fc, grid)
                    if E / ti > CAP or pk > CAP:
                        continue
                    (bc if fg == fc else cc).append((E, ti, fg, fc))
                    if fg != fc:
                        continue
            cc += bc                    # two-clock search contains the one-clock one
            g900, c900 = G.get((900, m, n)), C.get((900, c))
            r = dict(m=m, nk=n, ctas=c, ar_mib=64,
                     tc_over_tg=round(float(c900["iter_ms"]) / float(g900["iter_ms"]), 4)
                     if (g900 and c900) else "")
            ok = True
            for obj in ("E", "T", "EDP"):
                a, b, d = pick(bm, obj), pick(bc, obj), pick(cc, obj)
                if not (a and b and d):
                    ok = False
                    break
                r.update({
                    f"{obj}_Bmeas_mJ": round(a[0], 2), f"{obj}_Bmeas_ms": round(a[1], 4),
                    f"{obj}_Bmeas_f": a[2],
                    f"{obj}_Bcomp_mJ": round(b[0], 2), f"{obj}_Bcomp_ms": round(b[1], 4),
                    f"{obj}_C_mJ": round(d[0], 2), f"{obj}_C_ms": round(d[1], 4),
                    f"{obj}_C_fg": d[2], f"{obj}_C_fc": d[3],
                    f"{obj}_C_vs_Bmeas_pct": round((d[0] - a[0]) / a[0] * 100, 2),
                    f"{obj}_C_vs_Bcomp_pct": round((d[0] - b[0]) / b[0] * 100, 2),
                    f"{obj}_compbias_pct": round((b[0] - a[0]) / a[0] * 100, 2)})
            if ok:
                out.append(r)

    p = os.path.join(HERE, "data", "followup_baseline.csv")
    with open(p, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        wr.writeheader(); wr.writerows(out)

    print(f"{len(out)} (shape, CTA) pairs where all three baselines exist, 64 MiB\n")
    print(f"  {'objective':>11}{'t_c/t_g':>10}{'n':>4}"
          f"{'C vs measured':>16}{'C vs composed':>16}{'composition bias':>19}")
    bins = [(0, 1.0, "< 1"), (1.0, 1e9, ">= 1")]
    for obj, name in (("E", "min energy"), ("T", "min latency"), ("EDP", "min EDP")):
        for lo, hi, lab in bins:
            v = [x for x in out if x["tc_over_tg"] != "" and lo <= x["tc_over_tg"] < hi]
            if not v:
                continue
            print(f"  {name if lab == '< 1' else '':>11}{lab:>10}{len(v):>4}"
                  f"{st.median(x[f'{obj}_C_vs_Bmeas_pct'] for x in v):>+15.2f}%"
                  f"{st.median(x[f'{obj}_C_vs_Bcomp_pct'] for x in v):>+15.2f}%"
                  f"{st.median(x[f'{obj}_compbias_pct'] for x in v):>+18.2f}%")
    allb = [x[f"E_compbias_pct"] for x in out]
    print(f"\n  composition bias over all {len(out)}: median {st.median(allb):+.2f}%   "
          f"mean {st.mean(allb):+.2f}%   |median| {st.median(abs(x) for x in allb):.2f}%")
    print(f"  wrote {p}")


if __name__ == "__main__":
    main()
