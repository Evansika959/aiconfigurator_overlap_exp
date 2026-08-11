#!/usr/bin/env python3
"""Pre-sweep validation. Each check guards a DIFFERENT silent-failure mode -- a way
the experiment runs, produces plausible numbers, and measures nothing.

  V1  the GEMM actually reaches exactly 108-n SMs, and none of the squatted ones
  V2  a squatter that dies mid-window is caught, not silently recorded
  V3  torch.cuda.graph() under a squatter (expected FAIL -> eager measurement)

V1 replaces an earlier pair of checks that both PASSED on a configuration whose SM
axis was wrong by 2x: verifying that n squatters land on n distinct SMs says nothing
about where the GEMM can go, because a squatter in a lower shared-memory carveout
class also excludes its TPC partner. The probe therefore carries the cuBLAS kernel's
exact shared memory and is subject to the same placement rules.

  python3 validate.py           # exit 0 only if V1 and V2 pass
"""

import sys
import time

import torch

from measure import measure
from squatter import (CUBLAS_GEMM_SMEM_BYTES, SQUAT_SMEM_BYTES, Squatter,
                      probe_available_sms, probe_warmup)

DEV = "cuda:0"
NSM = torch.cuda.get_device_properties(0).multi_processor_count


def _gemm_fn(m, n, k):
    x = torch.randn((m, k), dtype=torch.bfloat16, device=DEV)
    w = torch.randn((n, k), dtype=torch.bfloat16, device=DEV)
    return lambda: torch.nn.functional.linear(x, w)


def v1_probe():
    print("=" * 78)
    print("V1  GEMM 实际可达的 SM 集合 (探针带 cuBLAS kernel 真实的 %d B shared memory)"
          % CUBLAS_GEMM_SMEM_BYTES)
    print("=" * 78)
    probe_warmup()                      # MUST be outside any squatter
    ok = True
    print(f"  {'占位smem':>9} {'n':>4} {'占到SM':>7} {'可达SM':>7} {'期望':>6} {'交集':>5} {'判定':>6}")
    # 128 KB is a NEGATIVE CONTROL: it passes the old distinct-smid test but confines
    # the GEMM to 108-2n (or nowhere at all for n>=54). It must FAIL here.
    for kb, label in ((SQUAT_SMEM_BYTES // 1024, ""), (128, "  <- 负对照, 必须 FAIL")):
        for n in (27, 54, 94):
            with Squatter(n, device=DEV, smem_bytes=kb * 1024, max_seconds=60.0) as sq:
                occ = set(v - 1 for v in sq.ready.tolist())
                avail = probe_available_sms()
                sq.assert_alive([0] * n, 0.0)      # squatter must still be alive
            good = len(avail) == NSM - n and not (avail & occ)
            if kb == SQUAT_SMEM_BYTES // 1024:
                ok &= good
            else:
                ok &= not good                      # the control must fail
            print(f"  {kb:>7}KB {n:>4} {len(occ):>7} {len(avail):>7} {NSM - n:>6} "
                  f"{len(avail & occ):>5} {'PASS' if good else 'FAIL':>6}{label}", flush=True)
    print(f"  -> V1 {'PASS' if ok else 'FAIL'}  ({SQUAT_SMEM_BYTES // 1024} KB 精确互补; "
          f"128 KB 负对照如期失败)")
    return ok


def v2_heartbeat_catches_expiry():
    print()
    print("=" * 78)
    print("V2  占位 kernel 中途过期能否被抓到? (否则会把 108-SM 的数据记成 n-SM)")
    print("=" * 78)
    fn = _gemm_fn(4096, 8192, 8192)
    for _ in range(8):
        fn()
    torch.cuda.current_stream().synchronize()

    # backstop deliberately shorter than the measurement window
    sq = Squatter(27, device=DEV, max_seconds=1.0)
    sq.__enter__()
    caught = False
    try:
        measure(fn, squatter=sq, dev=0, min_window_s=2.0)
    except RuntimeError as e:
        caught = "residency" in str(e) or "beat" in str(e)
        print(f"  短 backstop(1.0s) vs 窗口(2.5s): 抓到 ✓  \"{str(e)[:88]}\"")
    finally:
        try:
            sq.__exit__()
        except Exception:
            pass
    if not caught:
        print("  短 backstop 未被抓到 ✗ -- 心跳判据太松, 会放过错误 SM 数的数据")

    sq2 = Squatter(27, device=DEV, max_seconds=120.0)
    sq2.__enter__()
    clean = False
    try:
        r = measure(fn, squatter=sq2, dev=0, min_window_s=1.5)
        clean = True
        print(f"  长 backstop(120s): 正常通过 ✓  lat={r['latency_ms']:.3f} ms")
    except RuntimeError as e:
        print(f"  长 backstop 误报 ✗  {str(e)[:88]}")
    finally:
        sq2.__exit__()
    ok = caught and clean
    print(f"  -> V2 {'PASS' if ok else 'FAIL'}")
    return ok


def v3_cuda_graph():
    print()
    print("=" * 78)
    print("V3  占位期间 torch.cuda.graph() 能否捕获? (预期 FAIL -> 用 eager)")
    print("=" * 78)
    fn = _gemm_fn(2048, 4096, 4096)
    BACK = 12.0
    ok = True
    for label, nsq in (("无占位", 0), ("54 占位", 54)):
        ctx = Squatter(nsq, device=DEV, max_seconds=BACK) if nsq else None
        if ctx:
            ctx.__enter__()
        t0 = time.time()
        try:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                fn()
            g.replay()
            torch.cuda.current_stream().synchronize()
            dt = time.time() - t0
            blocked = dt > BACK * 0.5
            print(f"  {label:>8}: {'被阻塞 ✗' if blocked else '成功 ✓'}  ({dt:.2f}s)")
            ok &= not blocked
        except Exception as e:
            print(f"  {label:>8}: 异常 ✗  {type(e).__name__}: {str(e)[:80]}")
            ok = False
        finally:
            if ctx:
                try:
                    ctx.__exit__()
                except Exception:
                    pass
    if not ok:
        print("  -> V3 FAIL (预期): torch.cuda.graph() 内部设备级同步。sweep 用 eager +")
        print("     流级同步; benchmark_with_power 有 8 处设备级同步, 同样不可用。")
    return ok


if __name__ == "__main__":
    print(f"device: {torch.cuda.get_device_name(0)}  SM={NSM}")
    print(f"占位 shared memory = {SQUAT_SMEM_BYTES // 1024} KB\n")
    r1 = v1_probe()
    r2 = v2_heartbeat_catches_expiry()
    r3 = v3_cuda_graph()
    print()
    print("=" * 78)
    print(f"V1 SM集合精确: {'PASS' if r1 else 'FAIL'}   "
          f"V2 过期可捕获: {'PASS' if r2 else 'FAIL'}   "
          f"V3 graph捕获: {'PASS(意外)' if r3 else 'FAIL(预期)'}")
    print("=" * 78)
    sys.exit(0 if (r1 and r2) else 1)
