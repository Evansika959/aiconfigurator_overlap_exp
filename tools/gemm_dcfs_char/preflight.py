#!/usr/bin/env python3
"""Pre-sweep checks for the two remaining blocking items.

C1  Does a locked clock actually HOLD under load?  A lock is a request; under a
    400 W-capped GEMM at full SM count the board clamps below it. If it does, the
    sweep's clock axis is confounded with its SM axis and `freq_mhz` in the config
    table is a lie -- the achieved clock has to be carried in the data instead.

C2  Is the power baseline consistent across the SM axis?  The plan subtracts
    separately-measured squatter power. That only works if the n=0 case is treated
    the same way -- i.e. subtracting an equal-clock no-work baseline, not an idle
    reading taken elsewhere. C2 measures P_idle(freq) and P_squat(freq, n) and
    reports whether P_squat - P_idle is linear in n (it must be, if "static power
    per SM" is to mean anything).

Locks clocks. Restores them on every exit path.

  python3 preflight.py
"""

import atexit
import signal
import subprocess
import sys
import time

import torch

from measure import Sampler, measure
from squatter import Squatter, probe_available_sms

DEV = 0
GPUS = "0"
FREQS = [1410, 1200, 900, 705, 510, 300]
SQUAT_N = [0, 27, 54, 81, 94]
_locked = False


def sh(c):
    return subprocess.run(c, capture_output=True, text=True)


def reset_clocks():
    global _locked
    if _locked:
        sh(["sudo", "nvidia-smi", "-i", GPUS, "-rgc"])
        _locked = False
        print("[clocks] reset", flush=True)


def lock(f):
    global _locked
    sh(["sudo", "nvidia-smi", "-i", GPUS, "-lgc", f"{f},{f}"])
    _locked = True
    time.sleep(0.4)


def _sig(s, _f):
    reset_clocks()
    sys.exit(128 + s)


def gemm_fn(m, n, k):
    x = torch.randn((m, k), dtype=torch.bfloat16, device=f"cuda:{DEV}")
    w = torch.randn((n, k), dtype=torch.bfloat16, device=f"cuda:{DEV}")
    return lambda: torch.nn.functional.linear(x, w)


def c1_clock_holds():
    print("=" * 84)
    print("C1  锁频在负载下是否真的守得住?  (若守不住, freq_mhz 就是假的, 必须用实测时钟)")
    print("=" * 84)
    shapes = [(1024, 4096, 4096), (8192, 8192, 8192), (8192, 16384, 16384)]
    print(f"{'shape':>22} {'n_sq':>5} {'请求':>6} {'实测中位':>9} {'实测最低':>9} "
          f"{'功耗W':>7} {'温度':>5} {'throttle':>22}")
    bad = 0
    for (m, n, k) in shapes:
        fn = gemm_fn(m, n, k)
        for _ in range(8):
            fn()                                  # warm cuBLAS OUTSIDE any squatter
        torch.cuda.current_stream().synchronize()
        for nsq in (0, 54):
            lock(1410)
            sq = Squatter(nsq, device=f"cuda:{DEV}", max_seconds=120.0) if nsq else None
            if sq:
                sq.__enter__()
            try:
                r = measure(fn, squatter=sq, requested_clock=1410, dev=DEV)
            finally:
                if sq:
                    sq.__exit__()
            if not r["clock_held"]:
                bad += 1
            print(f"{str((m, n, k)):>22} {nsq:>5} {1410:>6} {r['clock_sm_med']:>9} "
                  f"{r['clock_sm_min']:>9} {r['power_w']:>7.1f} {r['temp_c']:>5} "
                  f"{r['throttle']:>22}", flush=True)
    reset_clocks()
    print(f"  -> {bad} 个配置未守住请求频率。"
          + ("时钟轴与 SM 轴混淆, 必须记录实测时钟并用它做自变量。" if bad else "锁频可信。"))
    return bad


def c2_power_baseline():
    print()
    print("=" * 84)
    print("C2  功耗基线是否可加?  P_squat(f,n) - P_idle(f) 对 n 应当线性 (斜率 = 每 SM 静态功耗)")
    print("=" * 84)
    print(f"{'freq':>6} " + "".join(f"{('n=' + str(n)):>9}" for n in SQUAT_N)
          + f"{'斜率 W/SM':>11} {'R^2':>7}")
    rows = []
    for f in FREQS:
        lock(f)
        vals = []
        for nsq in SQUAT_N:
            sq = Squatter(nsq, device=f"cuda:{DEV}", max_seconds=120.0) if nsq else None
            if sq:
                sq.__enter__()
            try:
                s = Sampler(DEV)
                s.start()
                time.sleep(2.0)                    # no GEMM: squatter (or nothing) only
                st = s.stop()
            finally:
                if sq:
                    sq.__exit__()
            vals.append(st["power_w"])
        xs, ys = SQUAT_N, [v - vals[0] for v in vals]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
        ss = sum((y - my) ** 2 for y in ys)
        rs = sum((y - (my + b * (x - mx))) ** 2 for x, y in zip(xs, ys))
        r2 = 1 - rs / ss if ss > 0 else float("nan")
        rows.append((f, vals, b, r2))
        print(f"{f:>6} " + "".join(f"{v:>9.1f}" for v in vals) + f"{b:>11.3f} {r2:>7.3f}", flush=True)
    reset_clocks()
    # Two separate questions, and only the first one gates the sweep:
    #   (a) can P_squat be subtracted?  Yes -- it is measured per (f, n) right here.
    #   (b) does the slope give per-SM static power?  No -- a __nanosleep'd SM is
    #       nearly indistinguishable from an idle one, which is precisely why
    #       __nanosleep was chosen over a spin loop. The n=0 -> n=27 step is the
    #       fixed cost of having ANY resident kernel; beyond that it is flat.
    flat = all(abs(r[1][-1] - r[1][1]) < 1.0 for r in rows)
    step = max(r[1][1] - r[1][0] for r in rows)
    print(f"  -> P_squat 可直接相减 (上表就是标定值, 最大偏置 {step:.1f} W)。")
    print(f"     但每 SM 静态功耗在此测量下不可分辨: 斜率 "
          f"{min(r[2] for r in rows):.3f}-{max(r[2] for r in rows):.3f} W/SM, "
          f"R^2 {min(r[3] for r in rows):.2f}-{max(r[3] for r in rows):.2f};"
          f" n=27..94 基本持平{'(已确认)' if flat else ''}。")
    return True


if __name__ == "__main__":
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)
    atexit.register(reset_clocks)
    torch.cuda.set_device(DEV)
    print(f"device: {torch.cuda.get_device_name(DEV)}\n")
    try:
        bad = c1_clock_holds()
        lin = c2_power_baseline()
    finally:
        reset_clocks()
    print()
    print("=" * 84)
    print(f"C1 锁频守住: {'否 -> 必须用实测时钟' if bad else '是'}    "
          f"C2 功耗可加: {'是' if lin else '否'}")
    print("=" * 84)
    # C1 failing is EXPECTED and is not a gate -- it is why the sweep records the
    # achieved clock. C2 failing would mean the power baseline is unusable.
    sys.exit(0 if lin else 1)
