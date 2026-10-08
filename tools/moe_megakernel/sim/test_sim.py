#!/usr/bin/env python3
"""Invariants of the simulator. None of these needs a GPU or a measurement: they hold
for any calibration, and a change that breaks one has broken the physics, not tuned it.

  python3 -m sim.test_sim
"""

import numpy as np

from .calib import Calib
from .control import ClosedLoop, Fixed, Governor, MinEnergySLO, OfflineOptimal, TdpBisection
from .demand import SHAPES, Partition, decode_counts, split
from .engine import Engine, Kernel, Shared
from .hardware import Hardware, two_domain
from .rate import Job, solve_rates
from .workload import build

CAL = Calib.load()


def layers(cells=32, n=3, **kw):
    return build([dict(phase="prefill" if cells == 0 else "decode", cells=cells, layers=[0], repeat=n, **kw)])


def last(hw, kernel, f, cells=32, **kw):
    e = Engine(hw, CAL, kernel, layers(cells), Fixed(f), record=False, limiter=False, **kw).run()
    return e.layers[-1]


def test_partition_conserves_work():
    """Cold + hot partitions hold exactly the layer's pairs, row-blocks and MACs."""
    shape = SHAPES["qwen"]
    c = decode_counts(shape, 32)
    for bm in (16, 64, 128):
        for nc in (16, 64, 96):
            m = split(c, nc)
            a, b, w = Partition(shape, c, bm, m), Partition(shape, c, bm, ~m), Partition(shape, c, bm)
            assert a.pairs + b.pairs == w.pairs == int(c.sum())
            assert a.mblocks + b.mblocks == w.mblocks and a.rows + b.rows == w.rows
            for g in ("g1", "g2"):
                ca, cb, cw = a.counters(g), b.counters(g), w.counters(g)
                for k in ("tiles", "tensor", "dram_bytes"):
                    assert abs(ca[k] + cb[k] - cw[k]) < 1e-6 * cw[k], (bm, nc, g, k)


def test_windows_conserve_work():
    """Cutting a partition into windows for any SM count loses nothing."""
    shape = SHAPES["qwen"]
    p = Partition(shape, decode_counts(shape, 16), 16)
    ref = None
    for n in (16, 28, 54, 92, 108):
        tot = [sum(getattr(p.stage(g, n), k) for g in ("g1", "g2")) for k in ("tot_panel", "tot_gmac", "tot_dram")]
        ref = ref or tot
        assert np.allclose(tot, ref, rtol=1e-9), (n, tot, ref)


def test_two_domains_at_one_clock_equal_one_domain():
    """The control-group identity: two V/f domains given the same clock are the pinned
    kernel on one domain -- time and energy, to rounding."""
    for cells, bm, nc, s in ((32, 64, 64, 32), (16, 16, 64, 48), (0, 128, 16, 16)):
        for f in (930, 1290, 1410):
            a = last(Hardware(), Kernel("persistent", bm, nc, s), f, cells)
            b = last(two_domain(s), Kernel("persistent", bm, nc), [f, f], cells)
            assert abs(a["ms"] / b["ms"] - 1) < 1e-12 and abs(a["mj"] / b["mj"] - 1) < 1e-12


def test_energy_is_the_integral_of_power():
    e = Engine(two_domain(32), CAL, Kernel("persistent", 64, 64), layers(32, 60), Fixed([1050, 1290])).run()
    integ = sum((t1 - t0) * p for t0, t1, p, *_ in e.trace)
    assert abs(integ / e.energy - 1) < 1e-9
    assert abs(sum(x["mj"] for x in e.layers) / e.energy - 1) < 1e-9
    assert abs(sum(x["ms"] for x in e.layers) / e.t - 1) < 1e-9


def test_a_faster_clock_is_never_slower_alone():
    """One partition alone: time falls and power rises with the clock."""
    prev = None
    for f in range(600, 1411, 90):
        L = last(Hardware(), Kernel("persistent", 64), f)
        if prev:
            assert L["ms"] < prev["ms"] and L["w"] > prev["w"], f
        prev = L


def test_a_neighbour_never_helps():
    """A partition beside another is never faster than the same partition alone."""
    shape = SHAPES["qwen"]
    c = decode_counts(shape, 32)
    m = split(c, 64)
    cold, hot = Partition(shape, c, 64, m), Partition(shape, c, 64, ~m)
    k = CAL.rate("persistent").at(1290)
    for g in ("g1", "g2"):
        a, b = Job(cold.stage(g, 32), k), Job(hot.stage(g, 76), k)
        ta, tb = a.time(), b.time()
        solve_rates([a, b])
        assert a.t >= ta * (1 - 1e-9) and b.t >= tb * (1 - 1e-9)


def test_slowing_the_idle_domain_saves_energy():
    """The mechanism a second rail offers: the domain that waits at the barrier can
    run slower at almost no cost in layer time."""
    hw, k = two_domain(32), Kernel("persistent", 64, 64)
    a, b = last(hw, k, [1290, 1290]), last(hw, k, [930, 1290])
    assert b["mj"] < a["mj"] and b["ms"] < a["ms"] * 1.01


def test_limiter_holds_the_wall():
    """Under the governor a load that would draw 460 W at the top clock is held: after
    the first second no sensor reading exceeds k_peak x TDP, and the long run sits at
    TDP."""
    hw = Hardware()
    e = Engine(hw, CAL, Kernel("queue", 128), layers(0, 1500), Governor()).run()
    tail = [x for x in e.ticks if x["t_ms"] > 1000]
    assert max(x["p_sensor_w"] for x in tail) <= hw.limiter.k_peak * hw.tdp_w + 1.0
    assert abs(np.mean([x["p_sensor_w"] for x in tail]) - hw.tdp_w) < 6.0


def test_limiter_absorbs_a_load_step():
    """A heavy batch arriving at the top clock, where it would draw 500 W: 100 ms later
    the clock is within two steps of where it is after a second, and no reported
    reading ever shows 1.1 x TDP."""
    hw = Hardware()
    e = Engine(hw, CAL, Kernel("queue", 128), layers(0, 400), Fixed(hw.f_max)).run()
    at = lambda t: next(f[0] for a, b, _, f, _ in e.trace if a <= t < b)
    assert at(0.0) == hw.f_max and abs(at(100.0) - at(1000.0)) <= 2 * hw.f_step, (at(100.0), at(1000.0))
    assert max(x["p_sensor_w"] for x in e.ticks) < 1.1 * hw.tdp_w


def test_bisection_meets_tdp_and_matches_exhaustive_search():
    """tdp_bisection lands on the wall, and no clock pair the exhaustive search finds
    is faster by more than one grid step's worth."""
    hw = two_domain(32)
    k = Kernel("queue", 128, 64)  # efficient enough to reach the wall
    sh = Shared()
    b = Engine(hw, CAL, k, layers(64, 400), TdpBisection(), shared=sh).run()
    o = Engine(hw, CAL, k, layers(64, 400), OfflineOptimal(step_mhz=15, f_lo=1200), shared=sh).run()
    lb, lo = b.layers[-1], o.layers[-1]
    assert lb["w"] <= hw.tdp_w + 0.5 and lo["w"] <= hw.tdp_w + 0.5
    assert lb["ms"] <= lo["ms"] * 1.01, (lb["ms"], lo["ms"])


def test_slo_controller_meets_its_target():
    hw, k = two_domain(32), Kernel("persistent", 64, 64)
    e = Engine(hw, CAL, k, layers(32, 300), MinEnergySLO(slo_ms=2.4)).run()
    L = e.layers[-1]
    flat = last(hw, k, [1410, 1410])
    assert L["ms"] <= 2.4 * 1.001 and L["mj"] < flat["mj"]


def test_closed_loop_stays_under_tdp():
    hw = two_domain(32)
    e = Engine(hw, CAL, Kernel("queue", 128, 64), layers(64, 1500), ClosedLoop()).run()
    tail = [x["p_sensor_w"] for x in e.ticks if x["t_ms"] > 1500]
    assert np.mean(tail) <= hw.tdp_w + 3.0, np.mean(tail)


def test_a_boost_is_issued_only_if_it_can_land():
    """The event the loop exists for: one domain finishes first and the other is raised
    until the barrier. With no actuation delay the layer gets shorter; with the A100's
    38 ms no boost is issued, because none could land inside a stage of a millisecond."""
    from .control import make

    k, L = Kernel("queue", 128, 32), layers(64, 4)
    for act, faster in ((0.0, True), (38.0, False)):
        hw = two_domain(32, actuation_ms=act)
        plain = Engine(hw, CAL, k, L, Fixed([1140, 1350]), limiter=False).run().layers[-1]["ms"]
        ctl = make({"type": "boost_on_finish", "f": [1140, 1350], "boost": 1395})
        boosted = Engine(hw, CAL, k, L, ctl, limiter=False).run().layers[-1]["ms"]
        if faster:
            assert ctl.boosts == 2 * len(L) and boosted < 0.99 * plain, (ctl.boosts, boosted / plain)
        else:
            assert ctl.boosts == 0 and boosted == plain, (ctl.boosts, boosted / plain)


def test_a_controller_can_be_added_from_outside():
    """Extension point: a controller defined anywhere, registered by name, is built
    from a scenario's spec and drives the engine."""
    from .control import REGISTRY, Controller, make, register

    @register
    class HalfSpeed(Controller):
        name = "test_half_speed"

        def attach(self, sim):
            self.sim = sim
            sim.req = [sim.hw.snap(sim.hw.f_max // 2)] * sim.D

    try:
        hw = Hardware()
        e = Engine(hw, CAL, Kernel("queue", 128), layers(32, 5), make({"type": "test_half_speed"}), limiter=False).run()
        assert e.f == [hw.snap(hw.f_max // 2)] and len(e.layers) == 5
    finally:
        REGISTRY.pop("test_half_speed")


def test_a_kernel_can_be_added_to_the_calibration():
    """Extension point: another kernel implementation is one more KernelCal, under a
    new name; a name the calibration does not have is an error, not a silent default."""
    import copy

    from .calib import KernelCal

    cal = copy.deepcopy(CAL)
    cal.kernels["persistent_2_blocks"] = KernelCal(cal.rate("persistent"), window=2.0)
    run = lambda name: (
        Engine(Hardware(), cal, Kernel(name, 128), layers(32), Fixed(1290), record=False, limiter=False)
        .run()
        .layers[-1]["ms"]
    )
    assert run("persistent_2_blocks") > 0 and run("persistent_2_blocks") != run("persistent")
    try:
        run("no_such_kernel")
    except KeyError:
        pass
    else:
        raise AssertionError("an unknown kernel name must raise")


def test_calibration_file_round_trips():
    """save -> load returns the same numbers, and a file written before kernels were
    named still loads."""
    import json
    import os
    import tempfile
    from dataclasses import asdict

    with tempfile.TemporaryDirectory() as d:
        new = Calib.load(CAL.save(os.path.join(d, "new.json")))
        for k in CAL.kernels:
            assert asdict(new.rate(k)) == asdict(CAL.rate(k))
            assert new.kernel(k).window == CAL.kernel(k).window
        assert asdict(new.power) == asdict(CAL.power) and asdict(new.epilogue) == asdict(CAL.epilogue)
        with open(os.path.join(d, "new.json")) as fh:
            j = json.load(fh)
        old = dict(
            persistent=j["kernels"]["persistent"]["rate"],
            queue=j["kernels"]["queue"]["rate"],
            queue_window=j["kernels"]["queue"]["window"],
            epilogue=j["epilogue"],
            power=j["power"],
            meta=j["meta"],
        )
        with open(os.path.join(d, "old.json"), "w") as fh:
            json.dump(old, fh)
        legacy = Calib.load(os.path.join(d, "old.json"))
        assert asdict(legacy.rate("queue")) == asdict(CAL.rate("queue"))
        assert legacy.kernel("queue").window == CAL.kernel("queue").window


def main():
    import time

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    bad = 0
    for t in tests:
        t0 = time.time()
        try:
            t()
            print(f"  ok    {t.__name__}  ({time.time() - t0:.1f} s)")
        except AssertionError as ex:
            bad += 1
            print(f"  FAIL  {t.__name__}: {ex}")
    print(f"{len(tests) - bad}/{len(tests)} passed")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
