#!/usr/bin/env python3
"""If the SM split is fixed at design time, what frequencies should the two domains run?

THE QUESTION. Everything so far let the solver choose the SM split per workload. No chip
works that way: a partitioned design commits to a split in silicon, and the only runtime
knob left is the two frequencies. So fix the split -- 1 NCCL CTA is 1 SM, so a fixed CTA
count IS a fixed split -- and ask two things:

    1. per fixed split, what frequency pair should each workload run, and what does it
       save against the same split at the clock the GPU picks on its own?
    2. how much does committing to ONE split cost, against choosing it per workload?

THREE DESIGN TIGHTNESSES, because a real chip may not even have per-workload DVFS:

    per-workload f    both domains retuned for every workload. The upper bound.
    one fixed pair    one (f_gemm, f_comm) burned in for all workloads. The floor.
    the gap between them says whether runtime DVFS is worth its control complexity.

BASELINE IS WITHIN THE SAME SPLIT: the unlocked run at that CTA count, i.e. the GPU's own
governor on the same hardware. That isolates the frequency question from the split
question, so a saving here cannot be a split effect in disguise.

  python3 fixed_design.py
"""
import collections
import csv
import glob
import json
import os
import statistics as st

import followup_baseline as FB

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = 400.0
# The axes are whatever was measured, not a hard-coded list: the extension campaign adds
# splits (2,12,22,48,64), clocks (1050,1305) and a 64 MiB regime incrementally, and a
# hard-coded axis would silently ignore every file that arrives after this line was
# written. `ext_*` and `commbound_*` are the same schema from the same harness.
SIZE_RE = "_(\\d+)mib\\.csv$"


def _mib(path):
    import re
    return int(re.search(SIZE_RE, os.path.basename(path)).group(1))


def load():
    rows = []
    for pat in ("commbound_*mib.csv", "ext_*mib.csv"):
        for f in sorted(glob.glob(os.path.join(HERE, "data", pat))):
            if os.path.basename(f).startswith("ext_auto"):
                continue
            for r in csv.DictReader(open(f)):
                r["_mib"] = _mib(f)
                rows.append(r)
    auto = {}
    for pat in ("auto_*mib.csv", "ext_auto_*mib.csv"):
        for f in sorted(glob.glob(os.path.join(HERE, "data", pat))):
            for r in csv.DictReader(open(f)):
                if r["mode"] == "concurrent":
                    auto[(_mib(f), int(r["ctas"]), int(r["m"]), int(r["n"]))] = r
    return rows, auto


def main():
    rows, auto = load()
    held = lambda r: r["clock_held"] == "True"
    global CLOCKS, SPLITS
    CLOCKS = sorted({int(r["clock"]) for r in rows if int(r["clock"]) > 0})
    SPLITS = sorted({int(r["ctas"]) for r in rows if r["mode"] == "comm_only"})
    G = {(int(r["clock"]), int(r["m"]), int(r["n"])): r
         for r in rows if r["mode"] == "gemm_only" and held(r)}
    C = {(r["_mib"], int(r["clock"]), int(r["ctas"])): r
         for r in rows if r["mode"] == "comm_only" and held(r)}
    shapes = sorted({(int(r["m"]), int(r["n"]), int(r["grid"]))
                     for r in rows if r["mode"] == "concurrent"})
    sizes = sorted({r["_mib"] for r in rows})

    # energy of every (workload, split, f_gemm, f_comm) the composition can reach
    tab = {}
    for c in SPLITS:
        for mib in sizes:
            for (m, n, grid) in shapes:
                for fg in CLOCKS:
                    g = G.get((fg, m, n))
                    if not g:
                        continue
                    for fc in CLOCKS:
                        cm = C.get((mib, fc, c))
                        if not cm:
                            continue
                        ti, E, pk = FB.compose(g, cm, fg, fc, grid)
                        if E / ti > CAP:
                            continue
                        tab[(c, mib, m, n, fg, fc)] = (E, ti)
    base = {}
    for c in SPLITS:
        for mib in sizes:
            for (m, n, grid) in shapes:
                a = auto.get((mib, c, m, n))
                if a:
                    base[(c, mib, m, n)] = (float(a["power_per_gpu_w"])
                                            * float(a["iter_ms"]), float(a["iter_ms"]),
                                            int(a["clock_min"]))
    wl = [(mib, m, n) for mib in sizes for (m, n, _) in shapes]

    print(f"{len(wl)} workloads ({len(shapes)} GEMM shapes x {len(sizes)} message "
          f"sizes {sizes}), {len(SPLITS)} splits {SPLITS}, {len(CLOCKS)} clocks "
          f"{CLOCKS}, world=4\n")
    print("1. PER FIXED SPLIT: retune both frequencies for every workload\n")
    print(f"  {'split':>24}{'n':>4}{'median':>10}{'best':>9}{'worst':>9}"
          f"{'most-chosen f_g / f_c':>24}")
    per_split = {}
    for c in SPLITS:
        d, picks = [], []
        for (mib, m, n) in wl:
            b = base.get((c, mib, m, n))
            cand = [(v[0], fg, fc) for (cc, mm, mmm, nn, fg, fc), v in tab.items()
                    if (cc, mm, mmm, nn) == (c, mib, m, n)]
            if not b or not cand:
                continue
            e, fg, fc = min(cand)
            d.append((e - b[0]) / b[0] * 100)
            picks.append((fg, fc))
        if not d:
            continue
        per_split[c] = d
        import collections
        top = collections.Counter(picks).most_common(1)[0]
        print(f"  {f'{c} CTA = {c/108*100:.1f}% comm':>24}{len(d):>4}"
              f"{st.median(d):>+9.2f}%{min(d):>+8.2f}%{max(d):>+8.2f}%"
              f"{f'{top[0][0]} / {top[0][1]}  ({top[1]}/{len(picks)})':>24}")

    print("\n2. ONE FIXED FREQUENCY PAIR for all workloads, per split\n")
    print(f"  {'split':>24}{'best fixed pair':>18}{'median':>10}{'worst':>9}"
          f"{'cost vs retuning':>19}")
    GLOBAL = {}
    for c in SPLITS:
        best = None
        for fg in CLOCKS:
            for fc in CLOCKS:
                d = []
                for (mib, m, n) in wl:
                    b = base.get((c, mib, m, n))
                    v = tab.get((c, mib, m, n, fg, fc))
                    if b and v:
                        d.append((v[0] - b[0]) / b[0] * 100)
                if len(d) == len(wl) and (best is None or st.median(d) < best[0]):
                    best = (st.median(d), max(d), fg, fc)
        if best:
            GLOBAL[c] = (best[2], best[3])
        if best and c in per_split:
            print(f"  {f'{c} CTA = {c/108*100:.1f}% comm':>24}"
                  f"{f'{best[2]} / {best[3]}':>18}{best[0]:>+9.2f}%{best[1]:>+8.2f}%"
                  f"{best[0]-st.median(per_split[c]):>+18.2f} pp")

    print("\n3. COMMITTING TO ONE SPLIT: cost against choosing it per workload\n")
    freew, fixedw = [], {c: [] for c in SPLITS}
    for (mib, m, n) in wl:
        per_c = {}
        for c in SPLITS:
            b = base.get((c, mib, m, n))
            cand = [v[0] for (cc, mm, mmm, nn, fg, fc), v in tab.items()
                    if (cc, mm, mmm, nn) == (c, mib, m, n)]
            if b and cand:
                per_c[c] = min(cand)
        if len(per_c) != len(SPLITS):
            continue
        best = min(per_c.values())
        freew.append(best)
        for c in SPLITS:
            fixedw[c].append((per_c[c] - best) / best * 100)
    print(f"  {'split':>24}{'median penalty':>16}{'worst':>9}")
    json.dump({str(c): st.median(v) for c, v in fixedw.items() if v},
              open(os.path.join(HERE, "data", "fixed_design_penalty.json"), "w"))
    for c in SPLITS:
        if fixedw[c]:
            print(f"  {f'{c} CTA = {c/108*100:.1f}% comm':>24}"
                  f"{st.median(fixedw[c]):>+15.2f}%{max(fixedw[c]):>+8.2f}%")

    # ---- 3b. THE NUMBER THE PANELS SHOULD CARRY -------------------------------------
    # With the split welded shut, the only question left is whether a SECOND V/f domain
    # is worth the silicon. So compare like with like at that split: the cheapest point
    # reachable with ONE domain (f_gemm == f_comm, which is what an A100 can do today)
    # against the cheapest point reachable with TWO. Section 3 answers a different
    # question -- how good this split is versus a better split -- and using it as the
    # panel's headline confused "you picked the wrong partition" with "the second domain
    # earns its keep", which are independent.
    print("\n3b. WHAT THE SECOND V/f DOMAIN BUYS, at each fixed split\n")
    print(f"  {'split':>24}{'best 1-domain f':>17}{'best 2-domain':>15}"
          f"{'median gain':>13}{'best':>9}")
    gain = {}
    for c in SPLITS:
        d, p1, p2 = [], [], []
        for (mib, m, n) in wl:
            one = [(v[0], fg) for (cc, mm, mmm, nn, fg, fc), v in tab.items()
                   if (cc, mm, mmm, nn) == (c, mib, m, n) and fg == fc]
            allp = [(v[0], fg, fc) for (cc, mm, mmm, nn, fg, fc), v in tab.items()
                    if (cc, mm, mmm, nn) == (c, mib, m, n)]
            if not one or not allp:
                continue
            e1, e2 = min(one), min(allp)
            d.append((e2[0] - e1[0]) / e1[0] * 100)
            p1.append(e1[1]); p2.append((e2[1], e2[2]))
        if not d:
            continue
        gain[c] = st.median(d)
        t1 = collections.Counter(p1).most_common(1)[0]
        t2 = collections.Counter(p2).most_common(1)[0]
        print(f"  {f'{c} CTA = {c/108*100:.1f}% comm':>24}"
              f"{f'{t1[0]} ({t1[1]}/{len(p1)})':>17}"
              f"{f'{t2[0][0]}/{t2[0][1]}':>15}{st.median(d):>+12.2f}%{min(d):>+8.2f}%")
    json.dump({str(k): v for k, v in gain.items()},
              open(os.path.join(HERE, "data", "fixed_design_gain.json"), "w"))

    # ---- 4. is the split conclusion a property of the comm-bound regime? ----
    # Every workload in the original set was comm-bound by construction, which is exactly
    # the regime that rewards a wide comm split. 64 MiB is GEMM-led. If the best split
    # moves with the message size, "commit to c" is a statement about the workload mix,
    # not about the hardware.
    print("\n4. SPLIT PENALTY BY REGIME (median %, per message size)\n")
    print(f"  {'split':>24}" + "".join(f"{str(m)+' MiB':>10}" for m in sizes))
    by = {c: {m: [] for m in sizes} for c in SPLITS}
    for (mib, m, n) in wl:
        per_c = {}
        for c in SPLITS:
            b = base.get((c, mib, m, n))
            cand = [v[0] for (cc, mm, mmm, nn, fg, fc), v in tab.items()
                    if (cc, mm, mmm, nn) == (c, mib, m, n)]
            if b and cand:
                per_c[c] = min(cand)
        if len(per_c) != len(SPLITS):
            continue
        bst = min(per_c.values())
        for c in SPLITS:
            by[c][mib].append((per_c[c] - bst) / bst * 100)
    for c in SPLITS:
        cells = "".join(f"{st.median(by[c][m]):>+9.1f}%" if by[c][m] else f"{'--':>10}"
                        for m in sizes)
        print(f"  {f'{c} CTA = {c/108*100:.1f}% comm':>24}{cells}")
    print("\n  best split per regime: " + ", ".join(
        f"{m} MiB -> {min(SPLITS, key=lambda c: st.median(by[c][m]) if by[c][m] else 1e9)} CTA"
        for m in sizes))

    # ---- every (split, workload, f_gemm, f_comm) the composition can reach ----
    out = os.path.join(HERE, "data", "fixed_design.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ctas", "comm_sm_pct", "ar_mib", "m", "nk", "f_gemm", "f_comm",
                    "iter_ms", "energy_mj", "base_energy_mj", "base_iter_ms",
                    "base_clock", "dE_vs_auto_pct", "is_best_pair_for_workload",
                    "is_global_fixed_pair"])
        bestpair = {}
        for c in SPLITS:
            for (mib, m, n) in wl:
                cand = [(v[0], fg, fc) for (cc, mm, mmm, nn, fg, fc), v in tab.items()
                        if (cc, mm, mmm, nn) == (c, mib, m, n)]
                if cand:
                    bestpair[(c, mib, m, n)] = min(cand)[1:]
        for (c, mib, m, n, fg, fc), (E, ti) in sorted(tab.items()):
            b = base.get((c, mib, m, n))
            if not b:
                continue
            w.writerow([c, f"{c/108*100:.1f}", mib, m, n, fg, fc,
                        f"{ti:.4f}", f"{E:.2f}", f"{b[0]:.2f}", f"{b[1]:.4f}", b[2],
                        f"{(E-b[0])/b[0]*100:+.2f}",
                        bestpair.get((c, mib, m, n)) == (fg, fc),
                        (fg, fc) == GLOBAL.get(c)])
    print(f"\n  wrote {out}  ({len(tab)} rows)")


if __name__ == "__main__":
    main()
