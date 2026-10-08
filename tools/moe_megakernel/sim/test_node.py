"""Invariants of the node engine (`node.py`). None of them depends on a fitted number
being right: they check that the engine does what its rules say.

  python3 -m sim.test_node
"""

import copy
import sys

import numpy as np

from . import node
from .calib import Calib
from .kernels import Kernels
from .node import GATE, SYNC, AllReduce, Gemm, Node, launch

K, CAL = Kernels.load(), Calib.load()
FLAT = copy.deepcopy(K)  # every board alike: one GPU can stand for four
FLAT.boards = [[0.0, 1.0, 1.0]] * 4
G, AR = "4096x4096x4096", 128


def close(a, b, tol=1e-6):
    return abs(a - b) <= tol * max(abs(a), abs(b), 1e-12)


def test_solo_gemm_is_waves():
    """A GEMM alone takes its host share, a launch and ceil(grid / 108) waves."""
    gm, f = K.gemm, 1200
    c = node.cell(FLAT, CAL, "gemm_only", shape=G, clock=f)
    want = gm.host_pre + gm.solo_ms(4096, 4096, 4096, f)
    assert close(c["gemm"][1] - c["gemm"][0], want, 1e-9), (c["gemm"], want)


def test_solo_allreduce_is_latency_plus_bytes():
    ar, f, c = K.allreduce, 1200, 16
    r = node.cell(FLAT, CAL, "comm_only", mib=AR, ctas=c, clock=f)
    assert close(r["comm"][1] - r["comm"][0], ar.solo_ms(AR, c, f), 1e-9)


def test_serial_is_the_sum():
    """On one stream the collective starts when the GEMM ends; only the host's share of
    its latency overlaps."""
    f, c = 900, 8
    g = node.cell(FLAT, CAL, "gemm_only", shape=G, clock=f)
    a = node.cell(FLAT, CAL, "comm_only", mib=AR, ctas=c, clock=f)
    s = node.cell(FLAT, CAL, "serial", shape=G, mib=AR, ctas=c, clock=f)
    want = g["gemm"][1] + (a["comm"][1] - a["comm"][0]) - K.allreduce.host_pre
    assert abs(s["comm"][1] - want) < 0.02, (s["comm"], want)


def test_one_gpu_stands_for_four():
    """Four GPUs doing the same thing at one clock stay in step."""
    progs = node.programs("concurrent", G, AR)
    a = Node(FLAT, CAL, ctas=8, clocks=1200, limiter=False).run(progs, iters=3)
    b = Node(FLAT, CAL, ctas=8, clocks=1200, limiter=False, gpus=1).run(progs[:1], iters=3)
    assert close(a.t_ms, b.t_ms, 1e-9)
    assert all(close(e, b.energy_mj[0], 1e-9) for e in a.energy_mj)


def test_energy_is_the_integral_of_power():
    r = Node(FLAT, CAL, ctas=8, clocks=1200, limiter=False).run(node.programs("concurrent", G, AR), iters=3)
    e = sum(p[0] * (b - a) for a, b, p, _, _ in r.trace)
    assert close(e, r.energy_mj[0], 1e-9)


def test_collective_costs_the_gemm_two_sms_per_cta():
    """Launched first, c CTAs leave the GEMM 108 - 2c SMs; the GEMM's span is then a
    whole number of waves of that many tiles (the collective outlasts it here)."""
    gm, f, c = K.gemm, 1200, 16
    r = node.cell(FLAT, CAL, "concurrent", shape=G, mib=2048, ctas=c, clock=f)
    waves = -(-gm.grid(4096, 4096, 4096) // (108 - 2 * c))
    want = gm.host_pre + gm.corun_ms + gm.launch_ms(f) + waves * gm.tile_ms(4096, 4096, 4096, f)
    assert close(r["gemm"][1] - r["gemm"][0], want, 1e-9), (r["gemm"], want)


def test_behind_a_gemm_one_sm_per_cta():
    """Launched behind a running GEMM the collective costs it one SM per CTA."""
    n = Node(FLAT, CAL, ctas=16, clocks=1200, limiter=False, gpus=1)
    seen = []
    orig = n.settle

    def spy():
        orig()
        seen.append(n.gpus[0].nccl_lost)

    n.settle = spy
    n.run(node.programs("concurrent_gf", "8192x8192x8192", 2048, 1), iters=1)
    assert max(seen) == 16, max(seen)


def test_a_collective_waits_for_its_slowest_gpu():
    """One GPU with a GEMM four times the others': every GPU's iteration ends together,
    and the light GPUs' collective cannot end before the heavy GPU has issued its own."""
    shapes = ["8192x8192x8192"] + ["2048x8192x8192"] * 3
    r = Node(K, CAL, ctas=16, clocks=1200, limiter=False).run(node.programs("serial", shapes, AR), iters=3)
    ends = [it[-1]["end"] + it[-1]["gate"] for it in r.iters]
    assert max(ends) - min(ends) < 0.05, ends
    heavy = r.iters[0][-1]["spans"]["gemm"][1] + r.iters[0][-1]["gate"]
    light_comm_end = r.iters[1][-1]["spans"]["comm"][1] + r.iters[1][-1]["gate"]
    assert light_comm_end > heavy


def test_slower_light_gpus_save_energy():
    """In that unbalanced step, slowing the light GPUs lowers the node's energy per step
    and lengthens the step by less than it lowers the light GPUs' clock."""
    shapes = ["8192x8192x8192"] + ["2048x8192x8192"] * 3
    progs = node.programs("concurrent", shapes, AR)
    out = []
    for fs in ([1200] * 4, [1200, 900, 900, 900]):
        r = Node(K, CAL, ctas=16, clocks=fs, limiter=False).run(progs, iters=4)
        t0 = r.iters[0][1]["gate"]
        out.append((r.t_loop(0), sum(r.power(t0)) * r.t_loop(0)))
    (t0_, e0), (t1, e1) = out
    assert e1 < e0 and t1 / t0_ < 1200 / 900, out


def test_limiter_holds_the_wall():
    """Asked for 1410 MHz under a GEMM that would draw well over 400 W, the board settles
    below 1410 with its mean power within 2% of the limit."""
    r = node.cell(FLAT, CAL, "gemm_only", shape="8192x8192x8192", clock=1410)
    assert r["limited"] and r["clock_mean"][0] < 1400
    assert abs(r["power"][0] / 400.0 - 1) < 0.02, r["power"]


def test_program_dependencies():
    """A kernel that waits on another never starts before it ends, and kernels on one
    stream never overlap."""
    ops = [
        launch("main", Gemm(4096, 8192, 8192, tag="a")),
        launch("main", Gemm(8192, 8192, 4096, tag="b")),
        launch("comm", AllReduce(128, tag="r"), wait=("b",)),
        launch("main", Gemm(4096, 8192, 8192, tag="c")),
    ]
    r = Node(FLAT, CAL, ctas=16, clocks=1200, limiter=False, gpus=1).run([[GATE] + ops + [SYNC]], iters=2)
    it = r.iters[0][-1]
    assert it["starts"]["b"] >= it["spans"]["a"][1] - 1e-9
    assert it["starts"]["c"] >= it["spans"]["b"][1] - 1e-9
    assert it["starts"]["r"] >= it["spans"]["b"][1] - 1e-9
    assert it["spans"]["r"][1] > it["starts"]["c"]  # the collective runs beside GEMM c


def test_interpolated_bandwidth_is_monotone_in_ctas():
    ar = K.allreduce
    for f in (600, 975, 1350):
        b = [ar.bandwidth(c, f) for c in (2, 3, 4, 6, 8, 10, 12, 16, 20, 24)]
        assert all(np.diff(b) > 0), (f, b)


def main():
    tests = [v for name, v in sorted(globals().items()) if name.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"ok    {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"{len(tests) - failed} of {len(tests)} pass")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
