#!/usr/bin/env python3
"""Print the tables of ../RESULTS_SIM.md from what is in sim/calib and sim/out.

  python3 -m sim.report            # markdown on stdout

A table in a results document goes stale the moment the data behind it is refitted.
These are generated, pasted, and then guarded by ../check_consistency.py.
"""

import os

from .calib import Calib
from .files import read_csv, read_json

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")


STEMS = ("decode1024", "decode2048", "decode4096", "prefill")


def _st(s, key):
    x = s.get(key)
    return f"{x['n']} | {x['rms']:.2f}% | {x['bias']:+.2f}% | {x['worst']:+.1f}%" if x else "| | |"


def calibration(cal):
    m, r, p, e = cal.meta, cal.rate("persistent"), cal.power, cal.epilogue
    print("| fit | fitted on | free numbers | rms | worst |")
    print("|---|---|---|---|---|")
    a = m["rate"]
    print(
        f"| rate, persistent kernel | {a['all']['n']} solo and co-run times, "
        f"{a['clocks'][0]}-{a['clocks'][-1]} MHz | {a['free']} | {a['all']['rms']:.2f}% | "
        f"{a['all']['worst']:+.1f}% |"
    )
    q = m["queue"]
    print(
        f"| rate, queue kernel | {q['all']['n']} end-to-end layer times, 3 campaigns | 4 + window | "
        f"{q['all']['rms']:.2f}% | {q['all']['worst']:+.1f}% |"
    )
    print(
        f"| epilogue time | {m['epilogue']['n']} (whole layer) - (its launches) | 3 | "
        f"{m['epilogue']['rms']:.2f}% | {m['epilogue']['worst']:+.1f}% |"
    )
    w = m["power"]
    print(
        f"| board power | {w['all']['n']} solo and co-run records | 4 + 2 per clock | "
        f"{w['all']['rms']:.2f}% of board power | {w['all']['worst']:+.1f}% |"
    )
    print()
    print("| parameter | value | meaning |")
    print("|---|---|---|")
    f = 1290.0
    rows = [
        (
            "kP at 1290 MHz",
            f"{(r.aP / f + r.bP) * 1e3:.2f} us SM / MB",
            f"weight-panel delivery per SM; {r.bP / (r.aP / f + r.bP) * 100:.0f}% of it does not scale with the clock",
        ),
        ("kR at 1290 MHz", f"{r.aR / f:.3f} ms SM / GMAC", f"tensor work per SM: {2 / (r.aR / f):.2f} TFLOP/s per SM"),
        ("L0 at 1290 MHz", f"{r.aL / f * 1e3:.2f} us SM / MB", "DRAM wait per SM, DRAM idle"),
        (
            "BW_D",
            f"{r.bw_d:.0f} GB/s",
            "DRAM bandwidth" + ("; pinned at the A100-40GB's nominal" if "bw_d" in a.get("pinned", {}) else ""),
        ),
        ("BW_L at 1290 MHz", f"{r.bw_l / 1e3:.1f} TB/s", f"L2 delivery capacity; grows as clock^{r.gamma_l:.2f}"),
        (
            "xi at 1290 MHz",
            f"{r.xi:.3f}",
            f"how hard a busy DRAM leans on L2 delivery; grows as clock^{r.gamma_x:.2f}",
        ),
        ("q", f"{r.q:.2f}", "... and how sharply that grows as DRAM fills: as (DRAM utilisation)^q"),
        ("e_mac", f"{p.e_mac:.3f} J/TMAC", "tensor work on real pairs, at the voltage floor"),
        ("e_pad", f"{p.e_pad:.3f} J/TMAC", f"tensor work on padding rows: {p.e_pad / p.e_mac * 100:.0f}% of real"),
        ("e_panel", f"{p.e_panel * 1e3:.1f} mJ/GB", "weight panels delivered to SMs, at the floor"),
        ("e_dram", f"{p.e_dram * 1e3:.1f} mJ/GB", "DRAM traffic; the same at every clock"),
        (
            "g(1050), g(1290), g(1410)",
            ", ".join(f"{p.g_at(x):.3f}" for x in (1050, 1290, 1410)),
            "supply voltage squared, relative to the floor",
        ),
        (
            "P_idle(510), (1050), (1290), (1410)",
            ", ".join(f"{p.idle_at(x):.1f}" for x in (510, 1050, 1290, 1410)) + " W",
            "board, nothing running",
        ),
        ("P_shared", f"{p.p_shared:.1f} W", "the part of idle power no clock changes"),
        (
            "epilogue",
            f"{e.t0 * 1e3:.0f} us + pairs x ({e.c0 * 1e6:.2f} + {e.c1 * 1e6:.0f}/f) ns",
            "activation and top-k sum",
        ),
    ]
    for a_, b, c in rows:
        print(f"| {a_} | {b} | {c} |")
    sg = a.get("sigma") or {}
    if sg:
        exps = ("gamma_l", "gamma_x")  # exponents: absolute; everything else: relative
        txt = ", ".join(f"{k} {v:.2f}" if k in exps else f"{k} {v * 100:.1f}%" for k, v in sg.items())
        print(f"\nOne standard deviation of the {a['free']} fitted rate parameters: {txt}.")
    cv = m.get("cross_validation") or {}
    if cv:
        stems = list(next(iter(cv.values())))
        print("\nCross-validation: each form of the rate model fitted on the first sweep alone, then")
        print("scored on the sweeps collected after it.\n")
        print("| form | pinned | " + " | ".join(f"`{s[9:]}`: n | rms | bias | worst" for s in stems) + " |")
        print("|---|---|" + "---|---|---|---|" * len(stems))
        from .calibrate import FORMS

        for form, per in cv.items():
            pins = ", ".join(f"{k} = {v:g}" for k, v in FORMS[form].items())
            cells = " | ".join(
                f"{per[s]['n']} | {per[s]['rms']:.2f}% | {per[s]['bias']:+.2f}% | {per[s]['worst']:+.1f}%"
                for s in stems
            )
            print(f"| {form}{' (in use)' if form == m.get('form') else ''} | {pins} | {cells} |")


def ladder():
    v = read_json(os.path.join(OUT, "validation.json"))
    print("| rung | what | fit | time: n | rms | bias | worst | power: n | rms | bias | worst | rms above idle |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for x in v.values():
        if isinstance(x, dict) and "label" in x:
            dyn = f"{x['power_dyn']['rms']:.2f}%" if x.get("power_dyn") else ""
            label = x["label"].replace("  ", "&nbsp;&nbsp;") if x["label"].startswith("  ") else x["label"]
            print(
                f"| {x.get('rung', '')} | {label} | {x.get('fit', '')} | {_st(x, 'time')} | {_st(x, 'power')} | {dyn} |"
            )
    print()
    print("| batch | measured MHz | W | ms | simulated MHz | W | ms |")
    print("|---|---|---|---|---|---|---|")
    for g in v["governor"]:
        print(
            f"| {g['tokens']} | {g['meas_clock']:.0f} | {g['meas_w']:.1f} | {g['meas_ms']:.4f} | "
            f"{g['sim_clock']:.0f} | {g['sim_w']:.1f} | {g['sim_ms']:.4f} |"
        )
    print()
    print(
        "| trace | segments x runs | latency rms | power rms | clock rms | clock rms, same window | "
        "energy of each run | first 400 ms after a load step: load change / short idle / wake | "
        "highest 100 ms reading, simulated / measured |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for tag, s in v.get("traces", {}).items():
        tr = s.get("transient_rms_pct", {})
        t3 = " / ".join(
            f"{tr[k]:.2f}%" if tr.get(k) is not None else "-" for k in ("load change", "short idle", "wake")
        )
        pk = s.get("peak_reading_w", {})
        print(
            f"| `{tag}` | {s['segments'] // s['runs']} x {s['runs']} | {s['latency_rms_pct']:.2f}% | "
            f"{s['power_rms_pct']:.2f}% | {s['clock_rms_mhz']:.1f} MHz | "
            f"{s.get('clock_rms_mhz_same_window', float('nan')):.1f} MHz | "
            + ", ".join(f"{x:+.1f}%" for x in s["energy_err_pct"])
            + f" | {t3} | {pk.get('sim', 0):.0f} / {pk.get('meas', 0):.0f} W |"
        )


def pinned(cal, f=1290):
    """The pinned layer end to end at one clock: measured / simulated ms for every case
    and SM split, and the case's rms over all its clocks."""
    from . import meas
    from .engine import Shared
    from .validate import layer

    recs = meas.pinned(cal.meta.get("model", "qwen"))
    v = read_json(os.path.join(OUT, "validation.json"))
    splits = sorted({r["sm_light"] for r in recs})
    print("| batch, tile, cold experts | " + " | ".join(f"{s}+{108 - s}" for s in splits) + " | rms, 3 clocks |")
    print("|---|" + "---|" * (len(splits) + 1))
    sh = Shared()
    for cells, bm, nc in sorted({(r["cells"], r["bm"], r["n_cold"]) for r in recs}):
        cellsof = {r["sm_light"]: r for r in recs if (r["cells"], r["bm"], r["n_cold"], r["f"]) == (cells, bm, nc, f)}
        row = []
        for s in splits:
            r = cellsof[s]
            t, _ = layer(cal, cells, bm, f, "persistent", nc, s, shared=sh)
            row.append(f"{r['ms']:.3f} / {t:.3f}")
        what = f"decode {cells * 64}" if cells else "prefill"
        rms = v[f"pinned_layer_c{cells}_bm{bm}_n{nc}"]["time"]["rms"]
        print(f"| {what}, BM{bm}, {nc} | " + " | ".join(row) + f" | {rms:.2f}% |")
    print()


def study(stem):
    path = os.path.join(OUT, f"study_{stem}.csv")
    if not os.path.exists(path):
        return
    rows = read_csv(path)
    b0 = next(r for r in rows if r["name"] == "B0")
    print(f"**{stem}**\n")
    for obj in dict.fromkeys(r["objective"] for r in rows):
        print(f"*{obj}*\n")
        print("| | configuration | clock MHz | ms | W | mJ | ms vs B0 | mJ vs B0 | best split |")
        print("|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            if r["objective"] != obj:
                continue
            if not r["ms"]:
                print(f"| {r['name']} | {r['what']} | cannot meet it | | | | | | |")
                continue
            ms, mj = float(r["ms"]), float(r["mj"])
            print(
                f"| {r['name']} | {r['what']} | {r['f']} | {ms:.4f} | {float(r['w']):.1f} | {mj:.1f} | "
                f"{(ms / float(b0['ms']) - 1) * 100:+.1f}% | {(mj / float(b0['mj']) - 1) * 100:+.1f}% | "
                f"{r['config']} |"
            )
        print()


def _studies(stems):
    """{workload: {(objective, group): row}} of the control-group studies, or None."""
    T = {}
    for stem in stems:
        path = os.path.join(OUT, f"study_{stem}.csv")
        if not os.path.exists(path):
            return None
        T[stem] = {(r["objective"], r["name"]): r for r in read_csv(path)}
    return T


def compact(stems=STEMS, fixed_domain=False):
    """Every control group against B0, one column per workload: time for the first
    objective, energy for the others. `fixed_domain`: the B4@<SMs> rows instead -- B4 on
    one domain size for every workload, as silicon would be."""
    T = _studies(stems)
    if T is None:
        return
    first = T[stems[0]]
    objs = list(dict.fromkeys(k[0] for k in first))
    names = [n for n in dict.fromkeys(k[1] for k in first) if n != "B0" and ("@" in n) == fixed_domain]
    print("| objective | | " + " | ".join(stems) + " |")
    print("|---|---|" + "---|" * len(stems))
    for i, obj in enumerate(objs):
        col = "ms" if i == 0 else "mj"
        for n in names:
            cells = []
            for stem in stems:
                r, b0 = T[stem][(obj, n)], T[stem][(obj, "B0")]
                cells.append(f"{(float(r[col]) / float(b0[col]) - 1) * 100:+.1f}%" if r["ms"] else "cannot meet it")
            what = "" if fixed_domain else " " + first[(obj, n)]["what"]
            print(f"| {obj} ({'time' if i == 0 else 'energy'}) | {n}{what} | " + " | ".join(cells) + " |")
    print()


def second_rail(stems=STEMS):
    """B4 - B3 and B4* - B3*, in points of B3's own time or energy: what the second rail
    adds to the same kernel, every objective and workload."""
    T = _studies(stems)
    if T is None:
        return
    objs = list(dict.fromkeys(k[0] for k in T[stems[0]]))
    print("| objective | | " + " | ".join(stems) + " |")
    print("|---|---|" + "---|" * len(stems))
    for i, obj in enumerate(objs):
        col = "ms" if i == 0 else "mj"
        for a, b in (("B4", "B3"), ("B4*", "B3*")):
            cells = []
            for stem in stems:
                x, y = T[stem][(obj, a)], T[stem][(obj, b)]
                cells.append(f"{(float(x[col]) / float(y[col]) - 1) * 100:+.2f}%" if x["ms"] and y["ms"] else "-")
            print(f"| {obj} ({'time' if i == 0 else 'energy'}) | {a} - {b} | " + " | ".join(cells) + " |")
    print()


def boost(stems=STEMS, kernels=("queue", "persistent")):
    """sim.boost: each policy at its best configuration, and where the boost gains most."""
    print(
        "| workload | per-SM costs | wall | one clock, ms | a clock per domain | + boost the survivor | "
        "largest gain of the boost in any configuration | that configuration against the best |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for stem in stems:
        for kern in kernels:
            path = os.path.join(OUT, f"boost_{stem}_{kern}.csv")
            if not os.path.exists(path):
                continue
            rows = read_csv(path)
            for wall in dict.fromkeys(r["wall"] for r in rows):
                ms = lambda pol, c=None, w=wall: [
                    float(r["ms"])
                    for r in rows
                    if r["wall"] == w and r["policy"] == pol and (c is None or (r["n_cold"], r["cold_sms"]) == c)
                ]
                one, two, bst = min(ms("one clock")), min(ms("per domain")), min(ms("boost"))
                confs = dict.fromkeys((r["n_cold"], r["cold_sms"]) for r in rows)
                c = min(confs, key=lambda c: ms("boost", c)[0] / ms("per domain", c)[0])
                gain = (ms("boost", c)[0] / ms("per domain", c)[0] - 1) * 100
                print(
                    f"| {stem} | {kern} | {wall} | {one:.4f} | {two:.4f} ({(two / one - 1) * 100:+.2f}%) | "
                    f"{bst:.4f} ({(bst / two - 1) * 100:+.2f}%) | {gain:+.2f}% ({c[0]} cold experts on {c[1]} SMs) | "
                    f"{(ms('boost', c)[0] / bst - 1) * 100:+.1f}% |"
                )
    print()
    head = None
    for stem in stems:
        for kern in kernels:
            path = os.path.join(OUT, f"boost_{stem}_{kern}_delay.csv")
            if not os.path.exists(path):
                continue
            rows = read_csv(path)
            if head is None:
                head = [f"{float(r['delay_ms']):g} ms" for r in rows]
                print("| workload | per-SM costs | actuation delay: " + " | ".join(head) + " |")
                print("|---|---|" + "---|" * len(head))
            print(f"| {stem} | {kern} | " + " | ".join(f"{float(r['vs_per_domain_pct']):+.2f}%" for r in rows) + " |")
    print()


def policies(tag="trace2"):
    path = os.path.join(OUT, f"policies_{tag}.csv")
    if not os.path.exists(path):
        return
    rows = read_csv(path)
    ref = rows[0]
    print("| policy | busy s | vs row 1 | busy J | vs row 1 | highest 100 ms reading |")
    print("|---|---|---|---|---|---|")
    for i, r in enumerate(rows):
        dt = f"{(float(r['busy_s']) / float(ref['busy_s']) - 1) * 100:+.1f}%" if i else ""
        de = f"{(float(r['busy_j']) / float(ref['busy_j']) - 1) * 100:+.1f}%" if i else ""
        print(
            f"| {r['policy']} | {float(r['busy_s']):.2f} | {dt} | {float(r['busy_j']):.0f} | {de} | "
            f"{float(r['peak_w']):.0f} W |"
        )
    fx = os.path.join(OUT, f"policies_{tag}_fixed.csv")
    if os.path.exists(fx):
        print()
        for r in read_csv(fx):
            print(
                f"Fixed {r['clock']} MHz against the governor: measured {float(r['meas_time_pct']):+.1f}% time, "
                f"{float(r['meas_energy_pct']):+.1f}% energy; simulated {float(r['sim_time_pct']):+.1f}%, "
                f"{float(r['sim_energy_pct']):+.1f}%."
            )
    print()


def sensitivity(name):
    path = os.path.join(OUT, name)
    if not os.path.exists(path):
        return
    rows = read_csv(path)
    clocks = [k[len("energy_pct_") :] for k in rows[0] if k.startswith("energy_pct_")]
    print(
        "| assumption | "
        + " | ".join(f"energy at B3's {f} MHz latency: B4 - B3 | clocks" for f in clocks)
        + " | fastest under TDP: B4 - B3 | clocks |"
    )
    print("|---|" + "---|---|" * len(clocks) + "---|---|")
    for r in rows:
        cols = " | ".join(f"{float(r[f'energy_pct_{f}']):+.2f}% | {r[f'b4_f_{f}']}" for f in clocks)
        print(f"| {r['assumption']} | {cols} | {float(r['latency_pct']):+.2f}% | {r['b4_f_tdp']} |")


def main():
    cal = Calib.load()
    print("## calibration\n")
    calibration(cal)
    print("\n## validation\n")
    ladder()
    print("\n### the pinned layer at 1290 MHz, measured / simulated ms\n")
    pinned(cal)
    print("\n## control groups\n")
    compact()
    print("### the second rail alone\n")
    second_rail()
    print("### B4 on one domain size\n")
    compact(fixed_domain=True)
    for stem in STEMS:
        study(stem)
    print("## boosting the surviving domain\n")
    boost()
    print("## policies on a moving load\n")
    policies()
    print("## sensitivity\n")
    for name in ("sensitivity_cells32_persistent.csv", "sensitivity_cells64_queue.csv"):
        print(f"`{name}`\n")
        sensitivity(name)
        print()


if __name__ == "__main__":
    main()
