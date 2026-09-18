#!/usr/bin/env python3
"""Re-derive every load-bearing number from the CSVs and check the docs still agree.

Run this before committing. Numbers in prose go stale the moment data is re-collected,
and this folder has already shipped one docstring table that contradicted the script
printing it. Everything quoted in a .md should be reproducible from data/ by a few lines
of arithmetic; if it is not, it should not be quoted.

Exits non-zero if any check fails.
"""
import csv, glob, os, sys
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)


def mean_by(path, key, val):
    d = defaultdict(list)
    for r in csv.DictReader(open(path)):
        d[r[key]].append(float(r[val]))
    return {k: float(np.mean(v)) for k, v in d.items()}


def main():
    docs = "\n".join(open(f).read() for f in sorted(glob.glob("*.md")))
    checks = []

    # --- what a partitionable kernel costs --------------------------------
    m = mean_by("data/pin_vs_queue_qwen.csv", "case", "ms")
    q = m["queue"]
    pinned = min(v for k, v in m.items() if k.startswith("pinned"))
    checks += [
        ("queue latency (ms)", f"{q:.3f}"),
        ("persistent 1 blk/SM", f"{m['persistent_1blk_per_sm']/q-1:+.1%}"),
        ("persistent 2 blk/SM", f"{m['persistent_2blk_per_sm']/q-1:+.1%}"),
        ("best pinned split", f"{pinned/q-1:+.1%}"),
    ]

    # --- the clock sweep ---------------------------------------------------
    by = defaultdict(list)
    for r in csv.DictReader(open("data/b1_energy_qwen.csv")):
        by[int(r["clock"])].append(r)
    A = lambda c, k: float(np.mean([float(x[k]) for x in by[c]]))
    held = [c for c in sorted(by) if c > 0 and abs(c - A(c, "clock_med")) <= 7]
    best = min(held, key=lambda c: A(c, "energy_mj_per_layer"))
    checks += [
        ("energy-optimal clock (MHz)", str(best)),
        ("1050 MHz vs governor, energy", f"{A(1050,'energy_mj_per_layer')/A(-1,'energy_mj_per_layer')-1:+.1%}"),
        ("1050 MHz vs governor, latency", f"{A(1050,'ms_per_layer')/A(-1,'ms_per_layer')-1:+.1%}"),
    ]

    # --- the two-domain composition ---------------------------------------
    u = {r["quantity"]: float(r["value"])
         for r in csv.DictReader(open("data/two_domain_uncertainty.csv"))}
    checks += [
        ("scaling exponent", f"{u['exponent']:.2f}"),
        ("its combined sd", f"{u['combined_sd']:.2f}"),
        ("exponent max over all refits", f"{u['exponent_max_over_all_refits']:.2f}"),
    ]

    # --- cold vs hot, the quantity the whole design needed -----------------
    pk = [x for x in csv.DictReader(open("data/partition_kappa_qwen.csv"))
          if x["held"] == "True" and int(x["block_m"]) == 128]
    g = defaultdict(list)
    for x in pk:
        g[(x["part"], int(x["clock"]))].append(float(x["kappa_mw_per_mhz_per_sm"]))
    cls = sorted({int(x["clock"]) for x in pk})
    ratio = np.mean([np.mean(g[("light", c)]) / np.mean(g[("heavy", c)]) for c in cls])
    # 1.025 sits on a rounding boundary; quote three decimals so prose and
    # checker cannot disagree about which way it rounds
    checks += [("kappa ratio, cold over hot", f"{ratio:.3f}")]

    bad = 0
    print(f"{'quantity':>32s} {'from data':>12s}   quoted in a .md?")
    for name, got in checks:
        present = got in docs or got.lstrip("+") in docs
        if not present:
            bad += 1
        print(f"{name:>32s} {got:>12s}   {'yes' if present else 'NOT FOUND'}")

    # --- cross-references --------------------------------------------------
    figs = {os.path.basename(p)[:-4] for p in glob.glob("figs/*.pdf")}
    for f in sorted(glob.glob("*.md")):
        text = open(f).read()
        import re
        for fig in re.findall(r"figs/([A-Za-z0-9_]+)\.(?:pdf|png)", text):
            if fig not in figs:
                print(f"  BROKEN figure reference: {f} -> figs/{fig}"); bad += 1
        for s in re.findall(r"`([a-z0-9_]+\.py)`", text):
            if not os.path.exists(s) and not os.path.exists(f"vendor/{s}"):
                print(f"  BROKEN script reference: {f} -> {s}"); bad += 1
        # require a real extension: `data/routing_*.npz` is a glob, not a reference
        for d in re.findall(r"data/([A-Za-z0-9_.]+\.(?:csv|npz|json|txt))\b", text):
            if not os.path.exists(f"data/{d}"):
                print(f"  BROKEN data reference: {f} -> data/{d}"); bad += 1

    print("\nall consistent" if bad == 0 else f"\n{bad} problem(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
