#!/usr/bin/env python3
"""Do the cold and hot partitions of a DECODE batch have different P-X curves?

Prefill said no: every partition is compute-bound, the cold/hot kappa ratio is 1.025,
and both share one energy-optimal clock (fig21). Decode is the case that could differ,
because the whole layer's clock elasticity falls to ~0.6 at batches up to 1024 tokens
(batch_clock_sweep.py) and one batch holds experts on both sides of the crossing.

Reads data/partition_kappa_qwen_decode*.csv (partition_kappa.py --decode-cells) and
reports, per (batch, BLOCK_M, partition):

  eps       d ln(throughput) / d ln(clock), fitted separately below and above the
            voltage knee. 1 = compute-bound, 0 = the clock buys nothing.
  f*_dyn    the clock minimising DYNAMIC energy -- where the partition itself stops
            paying for speed.
  f*_tot    the clock minimising dynamic energy plus the board's static power over the
            partition's own runtime. Static is attributed in proportion to the SMs the
            partition occupies (sms/108), the same per-SM split the two-domain models use.
  GB/s      expert weight bytes over runtime at the top clock: a floor on the DRAM
            traffic, since activations and partial sums are not counted.

The heterogeneity a second V/f domain needs is a DIFFERENT f* or a different eps between
the light and heavy partitions of the same batch at the same tile. A different LEVEL of
energy is not enough -- that is just more or less work.
"""
import csv, glob, os
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
KNEE = 1050          # MHz, the voltage knee measured in RESULTS_B1.md
N_SM = 108
HBM = 1555.0         # GB/s, A100-SXM4-40GB


def fit_eps(f, ms):
    f, ms = np.asarray(f, float), np.asarray(ms, float)
    if len(f) < 3:
        return float("nan")
    return float(np.polyfit(np.log(f), np.log(1.0 / ms), 1)[0])


def main():
    files = sorted(glob.glob(os.path.join(HERE, "data", "partition_kappa_qwen_decode*.csv")))
    if not files:
        raise SystemExit("no decode partition data yet")
    out = []
    for fn in files:
        rows = [r for r in csv.DictReader(open(fn)) if r["held"] == "True"]
        g = defaultdict(list)
        for r in rows:
            g[(int(r["block_m"]), r["part"], int(r["clock"]))].append(r)
        tokens = int(rows[0]["tokens"])
        print(f"\n=== decode batch {tokens} tokens  ({os.path.basename(fn)}) ===")
        print(f"{'BM':>4} {'part':>6} {'exp':>4} {'tok':>6} {'SMs':>4} "
              f"{'eps<knee':>8} {'eps>knee':>8} {'f*_dyn':>7} {'f*_tot':>7} "
              f"{'ms@top':>7} {'GB/s':>6} {'%HBM':>5} {'Edyn@f*':>8}")
        for bm in sorted({k[0] for k in g}):
            for part in ("light", "heavy", "all"):
                cl = sorted(c for (b, p, c) in g if b == bm and p == part)
                if not cl:
                    continue
                m = lambda c, k: float(np.mean([float(x[k]) for x in g[(bm, part, c)]]))
                ms = np.array([m(c, "ms") for c in cl])
                edyn = np.array([m(c, "e_dyn_mj") for c in cl])
                idle = np.array([m(c, "idle_w") for c in cl])
                sms = int(g[(bm, part, cl[0])][0]["sms"])
                etot = edyn + idle * (sms / N_SM) * ms
                lo = [i for i, c in enumerate(cl) if c <= KNEE]
                hi = [i for i, c in enumerate(cl) if c >= KNEE]
                e_lo = fit_eps([cl[i] for i in lo], ms[lo])
                e_hi = fit_eps([cl[i] for i in hi], ms[hi])
                r0 = g[(bm, part, cl[0])][0]
                wmib, nexp = float(r0["weight_mib"]), int(r0["experts"])
                bw = wmib * 2**20 / (ms[-1] * 1e-3) / 1e9
                idyn, itot = int(np.argmin(edyn)), int(np.argmin(etot))
                print(f"{bm:>4} {part:>6} {nexp:>4} {int(r0['real_rows']):>6} {sms:>4} "
                      f"{e_lo:>8.2f} {e_hi:>8.2f} {cl[idyn]:>7} {cl[itot]:>7} "
                      f"{ms[-1]:>7.3f} {bw:>6.0f} {bw / HBM * 100:>5.0f} "
                      f"{edyn[idyn]:>8.3f}")
                out.append(dict(tokens=tokens, block_m=bm, part=part, experts=nexp,
                                real_rows=int(r0["real_rows"]), sms=sms,
                                eps_below_knee=round(e_lo, 3),
                                eps_above_knee=round(e_hi, 3),
                                fstar_dyn=cl[idyn], fstar_tot=cl[itot],
                                ms_top=round(ms[-1], 4), weight_gbps_top=round(bw, 1),
                                edyn_min_mj=round(edyn[idyn], 4),
                                etot_min_mj=round(etot[itot], 4),
                                clocks=len(cl)))
    p = os.path.join(HERE, "data", "decode_partition_summary.csv")
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)
    print(f"\nwrote {p} ({len(out)} rows)")


if __name__ == "__main__":
    main()
