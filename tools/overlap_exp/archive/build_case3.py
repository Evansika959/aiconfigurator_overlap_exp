#!/usr/bin/env python3
"""Assemble the three-panel ladder for one cell of the fig7 matrix.

The cell asked about: GEMM 4096x4096x4096 + 128 MiB all-reduce, 32 CTAs, world = 4,
which reads -17 in "energy saved by a second clock domain vs the MEASURED overlap".

Panels (a) and (b) are REAL overlapped runs (measure_case_spans.py): two fresh streams,
collective at priority -3 issued first, both gated on one common event, hardware block
scheduler deciding the SM share. Panel (c) cannot be run -- A100 has one clock domain.

WHAT IS MEASURED per panel: total energy, both kernels' spans inside the overlapped
iteration, the iteration time, the clock actually held, the static floor. NOT measured:
how the non-static energy divides between the two kernels -- NVML reports one number for
the board -- so it is apportioned by the two solo dynamic energies, making the rectangles
sum to the measured total exactly.

SPANS ARE SCALED ONTO THE LOOP TIMELINE by T_loop / T_events: spans come from single-shot
reps with a device sync, power and iteration time from a sustained loop, and unlocked the
two phases differ because the governor behaves differently under stop-start.
"""
import csv, json, os
import followup_baseline as FB
import table_432 as t

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = 400.0
CLOCKS = [300, 510, 705, 900, 1200, 1410]
M, N, CT, MIB = 4096, 4096, 32, 128

rows = list(csv.DictReader(open(os.path.join(HERE, "data", "commbound_128mib.csv"))))
held = lambda r: r["clock_held"] == "True"
G = {int(r["clock"]): r for r in rows if r["mode"] == "gemm_only" and held(r)
     and (int(r["m"]), int(r["n"])) == (M, N)}
C = {int(r["clock"]): r for r in rows if r["mode"] == "comm_only" and held(r)
     and int(r["ctas"]) == CT}
grid = int(next(r for r in rows if r["mode"] == "concurrent"
                and (int(r["m"]), int(r["n"])) == (M, N))["grid"])


def panel(js):
    """One measured panel: rectangles that sum to the measured energy exactly."""
    f = js["clock_med"]
    T, E = js["T_loop"], js["power_per_gpu_w"] * js["T_loop"]
    sc = js["T_loop"] / js["T_events"]
    tg = (js["gemm_end"] - js["gemm_start"]) * sc
    tc = (js["comm_end"] - js["comm_start"]) * sc
    ps = t.lin(t.PS, f)
    es = ps * T
    # apportion the non-static energy by the two SOLO dynamic energies. This is the one
    # modelled step in a measured panel, and it is a split of a measured total, so the
    # panel's headline number stays measurement no matter how the split lands.
    # The solo GEMM at 1410 THROTTLED off its requested clock, so there is no held solo
    # pair there -- which is also exactly why 1410 is absent from the fig7 candidate set
    # and why that cell's baseline is 1200, not the 1410 the governor actually runs. For
    # the apportionment only the RATIO dg:dc is used, so fall back to the nearest clock
    # that held and say so; the panel's total stays the measured number either way.
    fr = f if (f in G and f in C) else max(k for k in G if k in C and k <= f)
    g, c = G[fr], C[fr]
    dg = (float(g["power_per_gpu_w"]) - t.lin(t.PS, fr)) * float(g["iter_ms"])
    dc = (float(c["power_per_gpu_w"]) - t.lin(t.PS, fr)) * float(c["iter_ms"])
    eg, ec = (E - es) * dg / (dg + dc), (E - es) * dc / (dg + dc)
    return dict(fg=f, fc=f, tg=tg, tc=tc, T=T, ps=ps, pg=eg / tg, pc=ec / tc,
                E_meas=E, es=es, eg=eg, ec=ec, scale=sc, split_from=fr,
                slow=tg / float(g["iter_ms"]))


a = panel(json.load(open(os.path.join(HERE, "data", "case3_0.json"))))
b = panel(json.load(open(os.path.join(HERE, "data", "case3_1200.json"))))

# --- the two-clock arm, held to the measured baseline's own latency -------------------
# Same rule as the fig7 cell: the second domain may not buy its saving with time.
two = []
for fg in CLOCKS:
    if fg not in G:
        continue
    for fc in CLOCKS:
        if fc not in C:
            continue
        ti, E, pk = FB.compose(G[fg], C[fc], fg, fc, grid)
        if E / ti > CAP or ti > b["T"] * 1.0001:
            continue
        two.append((E, ti, fg, fc))
E, ti, fg, fc = min(two)
g, c = G[fg], C[fc]
tg, tc = float(g["iter_ms"]), float(c["iter_ms"])
ps = t.lin(t.PS, min(t.PS)) + ((108 - CT) * (t.lin(t.PS, fg) - t.lin(t.PS, min(t.PS)))
                               + CT * (t.lin(t.PS, fc) - t.lin(t.PS, min(t.PS)))) / 108
pg = float(g["power_per_gpu_w"]) - t.lin(t.PS, fg)
pc = float(c["power_per_gpu_w"]) - t.lin(t.PS, fc)
cc = dict(fg=fg, fc=fc, tg=tg, tc=tc, T=ti, ps=ps, pg=pg, pc=pc, E=E,
          es=ps * ti, eg=pg * tg, ec=pc * tc)

json.dump(dict(a=a, b=b, c=cc, meta=dict(M=M, N=N, MIB=MIB, CT=CT, grid=grid)),
          open(os.path.join(HERE, "data", "case3.json"), "w"), indent=1)
for k, p in (("a no-lock", a), ("b locked", b), ("c two-clock", cc)):
    e = p.get("E_meas", p.get("E"))
    print(f"  {k:14s} {p['fg']:>5}/{p['fc']:<5} T {p['T']:.3f} ms   E {e:6.1f} mJ")
print(f"\n  (c) vs (b), the fig7 cell : {(cc['E']-b['E_meas'])/b['E_meas']*100:+.2f}%")
print(f"  (b) vs (a), free, one clock: {(b['E_meas']-a['E_meas'])/a['E_meas']*100:+.2f}%")
print(f"  (c) vs (a), both together  : {(cc['E']-a['E_meas'])/a['E_meas']*100:+.2f}%")
