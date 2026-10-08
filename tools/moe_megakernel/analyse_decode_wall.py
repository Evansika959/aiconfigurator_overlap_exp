#!/usr/bin/env python3
"""Read decode_wall.py's three outputs and answer the three go/no-go questions.

  1. governor   From which decode batch does the 400 W wall bind? (B_TDP)
  2. corun      Does the cold/hot gap in clock elasticity survive contention? The gap is
                what a second V/f domain would exploit; measured solo it was ~0.25.
  3. limiter    How far and how long may the board overshoot 400 W?

Elasticity = d ln(1/time) / d ln(clock), fitted on the clocks at or above the voltage
knee (1050 MHz), where a split would operate.

Also checked, because the simulator's power model assumes it: is co-run dynamic power
the SUM of the two solo dynamic powers?
"""
import csv, os
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
KNEE, TDP = 1050, 400.0
path = lambda n: os.path.join(HERE, "data", n)


def eps(f, ms):
    f, ms = np.asarray(f, float), np.asarray(ms, float)
    return float(np.polyfit(np.log(f), np.log(1 / ms), 1)[0]) if len(f) >= 2 else np.nan


def governor():
    fn = path("decode_governor_qwen.csv")
    if not os.path.exists(fn):
        return
    g = defaultdict(list)
    for r in csv.DictReader(open(fn)):
        g[(r["mode"], int(r["tokens"]))].append(r)
    m = lambda k, c: float(np.mean([float(x[c]) for x in g[k]]))
    toks = sorted({k[1] for k in g})
    print("=== 1. governor: where does the wall bind in decode? ===")
    print(f"{'batch':>6} {'tok/exp':>7} | {'governor MHz':>12} {'W':>6} {'ms':>7} | "
          f"{'lock1290 W':>10} | {'req 1410 ->':>11} {'W':>6}")
    b_tdp = None
    for t in toks:
        gv, l12, l14 = ("governor", t), ("lock1290", t), ("lock1410", t)
        binds = m(gv, "clock_med") < 1400 or m(gv, "power_w") >= TDP - 2
        if binds and b_tdp is None:
            b_tdp = t
        print(f"{t:>6} {float(g[gv][0]['per_expert']):>7.1f} | {m(gv, 'clock_med'):>12.0f} "
              f"{m(gv, 'power_w'):>6.1f} {m(gv, 'ms_per_layer'):>7.3f} | "
              f"{m(l12, 'power_w'):>10.1f} | {m(l14, 'clock_med'):>11.0f} "
              f"{m(l14, 'power_w'):>6.1f}" + ("   <- wall binds" if binds else ""))
    print(f"  B_TDP (first batch where the governor is power-limited): {b_tdp}")


def corun():
    fn = path("decode_corun_qwen.csv")
    if not os.path.exists(fn):
        return
    rows = list(csv.DictReader(open(fn)))
    nh = sum(r["held"] != "True" for r in rows)
    g = defaultdict(list)
    for r in rows:
        if r["held"] == "True":
            g[(int(r["tokens"]), int(r["block_m"]), int(r["sm_light"]), r["part"],
               r["mode"], int(r["clock"]))].append(r)
    m = lambda k, c: float(np.mean([float(x[c]) for x in g[k]]))
    keys = sorted({k[:3] for k in g})
    ov = [float(r["overlap"]) for r in rows if r["overlap"]]
    print(f"\n=== 2. co-run: does the elasticity gap survive contention? "
          f"({len(rows)} rows, {nh} not held; overlap min {min(ov):.2f}, "
          f"median {np.median(ov):.2f}) ===")
    print(f"{'batch':>6} {'BM':>3} {'split':>6} | {'eps cold':>16} {'eps hot':>16} | "
          f"{'gap solo':>8} {'gap corun':>9} | {'slowdown @1290':>15} | {'P add':>6}")
    print(f"{'':>17} | {'solo':>7} {'corun':>8} {'solo':>7} {'corun':>8} |"
          f"{'':>19} | {'cold':>7} {'hot':>7} |")
    out = []
    for tok, bm, nl in keys:
        e = {}
        for part in ("light", "heavy"):
            for mode in ("solo", "corun"):
                cl = sorted(k[5] for k in g if k[:5] == (tok, bm, nl, part, mode)
                            and k[5] >= KNEE)
                e[(part, mode)] = eps(cl, [m((tok, bm, nl, part, mode, c), "ms_per_pass")
                                           for c in cl])
        top = max(k[5] for k in g if k[:3] == (tok, bm, nl))
        sl = {p: m((tok, bm, nl, p, "corun", top), "ms_per_pass")
              / m((tok, bm, nl, p, "solo", top), "ms_per_pass") for p in ("light", "heavy")}
        idle = m((tok, bm, nl, "light", "solo", top), "idle_w")
        dyn = lambda p, mo: m((tok, bm, nl, p, mo, top), "power_w") - idle
        padd = dyn("light", "corun") / (dyn("light", "solo") + dyn("heavy", "solo"))
        gs = e[("heavy", "solo")] - e[("light", "solo")]
        gc = e[("heavy", "corun")] - e[("light", "corun")]
        print(f"{tok:>6} {bm:>3} {nl:>3}+{108 - nl:<3}| {e[('light', 'solo')]:>7.2f} "
              f"{e[('light', 'corun')]:>8.2f} {e[('heavy', 'solo')]:>7.2f} "
              f"{e[('heavy', 'corun')]:>8.2f} | {gs:>8.2f} {gc:>9.2f} | "
              f"{sl['light']:>7.2f} {sl['heavy']:>7.2f} | {padd:>6.2f}")
        out.append(dict(tokens=tok, block_m=bm, sm_light=nl, sm_heavy=108 - nl,
                        eps_cold_solo=round(e[("light", "solo")], 3),
                        eps_cold_corun=round(e[("light", "corun")], 3),
                        eps_hot_solo=round(e[("heavy", "solo")], 3),
                        eps_hot_corun=round(e[("heavy", "corun")], 3),
                        gap_solo=round(gs, 3), gap_corun=round(gc, 3),
                        slowdown_cold=round(sl["light"], 3),
                        slowdown_hot=round(sl["heavy"], 3),
                        corun_dyn_over_sum_of_solo=round(padd, 3)))
    gaps = np.array([o["gap_corun"] for o in out])
    print(f"  gap under contention: min {gaps.min():.2f}, median {np.median(gaps):.2f}, "
          f"max {gaps.max():.2f}; cases with gap >= 0.2: {(gaps >= 0.2).sum()}/{len(gaps)}")
    with open(path("decode_corun_summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)


def limiter():
    fn = path("limiter_step_qwen.csv")
    if not os.path.exists(fn):
        return
    g = defaultdict(list)
    for r in csv.DictReader(open(fn)):
        g[int(r["trial"])].append((float(r["t_ms"]), r["phase"], float(r["power_w"]),
                                   int(r["clock_mhz"])))
    print("\n=== 3. limiter: step from idle to a full-width prefill layer ===")
    print(f"{'trial':>5} {'rate/s':>7} {'peak MHz':>8} {'at ms':>6} {'final MHz':>9} "
          f"{'settled ms':>10} {'peak W':>7} {'ms above 400 W':>14} {'final W':>8}")
    for t in sorted(g):
        s = [x for x in g[t] if x[1] == "load"]
        tm = np.array([x[0] for x in s]); pw = np.array([x[2] for x in s])
        ck = np.array([x[3] for x in s])
        final = np.median(ck[tm > tm.max() - 1000]); fpw = np.median(pw[tm > tm.max() - 1000])
        ipk = int(np.argmax(ck))
        off = np.where(np.abs(ck - final) > 15)[0]
        settled = tm[off[-1] + 1] if len(off) and off[-1] + 1 < len(tm) else tm[0]
        above = pw > TDP + 2
        dt = np.diff(tm, append=tm[-1])
        print(f"{t:>5} {len(s) / (tm.max() - tm.min()) * 1e3:>7.0f} {ck.max():>8} "
              f"{tm[ipk]:>6.0f} {final:>9.0f} {settled:>10.0f} {pw.max():>7.1f} "
              f"{dt[above].sum():>14.0f} {fpw:>8.1f}")
    u = np.unique(np.round(np.diff(sorted({x[2] for t in g for x in g[t]})), 2))
    print(f"  distinct power readings step by >= {u.min() if len(u) else 0} W; "
          "a slow or coarse sensor bounds what this can resolve")


if __name__ == "__main__":
    governor(); corun(); limiter()
