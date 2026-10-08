"""A node of GPUs, each running streams of kernels: the event engine for work that is not
one MoE layer on one GPU.

`engine.py` simulates SM sets and V/f domains INSIDE one GPU for one kind of work. A
multi-GPU job is a different shape of problem: every GPU is already its own V/f domain,
work arrives as kernels on CUDA streams, and a collective ties the GPUs together. This
engine keeps the event loop and the chip (`hardware.Hardware`, `calib.Calib.power`, the
limiter's rules) and replaces the workload with something general:

  host program   per GPU, what its host thread does: issue a kernel on a stream, wait
                 for the device, sleep. Issuing takes host time, so two kernels issued
                 "together" start a tenth of a millisecond apart, as they do for real.
  stream         kernels on one stream run in order; kernels on different streams run
                 side by side and compete for SMs.
  kernel         a profile from `kernels.py`. A GEMM is `grid` tiles placed on free SMs,
                 a wave at a time; an all-reduce is resident on every GPU of its world
                 from its launch to its end and moves bytes only while all are present.

EVENTS    host    a host thread is free to issue its next call
          queued  an issued kernel reaches its stream
          wave    a cohort of GEMM tiles finishes; waiting tiles take the SMs
          phase   a collective finishes its latency, or its last byte
          setpoint, limit    as in engine.py, one limiter per GPU

Between two events nothing changes, so progress is linear and power is constant: the
loop is advance -> handle -> settle, where `settle` starts whatever can now start.

WHAT IS MEASURED AND WHAT IS A PARAMETER. One clock per GPU is the real machine and is
what `validate_overlap.py` scores. Rules a single-clock run cannot see are spelled out
where they are used: a collective among GPUs at different clocks runs at the slowest
participant's rate (`_Coll.rates`).
"""

import heapq
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .calib import Calib
from .hardware import Hardware
from .kernels import Kernels

EPS = 1e-9


# ------------------------------------------------------------------ workload ----
@dataclass
class Gemm:
    m: int
    n: int
    k: int
    tag: str = "gemm"


@dataclass
class AllReduce:
    mib: float
    tag: str = "comm"


def launch(stream, kernel, wait=()):
    """Issue `kernel` on `stream`; it may not start before the kernels tagged in `wait`
    (issued earlier in this iteration, on this GPU) have ended."""
    return ("launch", stream, kernel, tuple(wait))


GATE = ("gate",)  # the instant spans are measured from
SYNC = ("sync",)  # wait until every stream of this GPU is empty


def sleep(ms):
    return ("sleep", ms)


def clock(mhz, delay_ms=None):
    """Ask for a clock on this GPU; it takes effect `delay_ms` later (default: the
    hardware's actuation delay, 38 ms through NVML). The host does not wait for it."""
    return ("clock", mhz, delay_ms)


@dataclass
class Host:
    """Host-side costs of the measurement loop, ms. Not chip physics: they are what
    PyTorch and the driver take on this box, read off the spans of the `cal` campaign.
    (What issuing a kernel costs the host is in the kernel's profile.)"""

    gate: float = 0.024  # the timed loop's gate event -> its first call
    # device empty -> the next iteration's first call, in the untimed loop the power
    # window runs: by what the iteration issued
    loop: dict = field(
        default_factory=lambda: dict(
            gemm_only=0.021, comm_only=0.029, concurrent=0.045, concurrent_gf=0.046, serial=0.035
        )
    )


# ------------------------------------------------------------------- pieces ----
class _Inst:
    """One kernel on one GPU."""

    def __init__(self, gpu, stream, spec, wait, t):
        self.gpu, self.stream, self.spec, self.wait = gpu, stream, spec, wait
        self.t_issue, self.t_start, self.t_end = t, None, None
        self.state = "issued"  # issued -> queued -> launch -> run -> done
        self.coll = None

    @property
    def kind(self):
        return "gemm" if isinstance(self.spec, Gemm) else "comm"


class _Coll:
    """One all-reduce: an instance per GPU, one progress for all of them."""

    def __init__(self, mib):
        self.gb = mib * 2**20 / 1e9
        self.parts = []
        self.latency = 1.0  # share of the latency still to go
        self.left = self.gb
        self.done = False

    def resident(self, world):
        return len(self.parts) == world and all(p.state == "run" for p in self.parts)


class _Gpu:
    def __init__(self, i, hw, f, locked, f_comm=None):
        self.i, self.hw, self.locked = i, hw, locked
        self.f = self.req = self.target = hw.snap(f)
        # A WHAT-IF: the SMs the collective holds as a V/f domain of their own. None on
        # the real chip, where they follow the one clock.
        self.f_comm = None if f_comm is None else hw.snap(f_comm)
        self.cap = hw.f_max
        self.excess_j, self.sp_seq, self.sp_last = 0.0, 0, 0
        self.streams = {}  # name -> deque of queued instances
        self.active = {}  # name -> the instance running on that stream
        self.cohorts = []  # [instance, tiles, share of a tile still to go]
        self.nccl = None  # the resident collective's instance
        self.nccl_lost = 0  # SMs the resident collective denies to a GEMM block
        self.link_until = -1.0  # the links stay up until then
        self.pc, self.it, self.syncing, self.gate_t = 0, 0, False, 0.0
        self.tags = {}
        self.P, self.energy, self.parts = 0.0, 0.0, (0.0, 0.0)
        self.win = deque()
        self.f_min_seen = self.f

    @property
    def fc(self):
        """The clock of the collective's SMs."""
        return self.f if self.f_comm is None else self.f_comm

    def busy(self):
        return bool(self.active) or any(self.streams.values())

    def drained(self):
        return not self.active and not any(self.streams.values())


@dataclass
class Result:
    iters: list  # per GPU: list of dict(gate, spans={tag: (start, end)}, end)
    t_ms: float
    energy_mj: list
    trace: list = field(default_factory=list)  # (t0, t1, [W], [MHz], [rail W]) per GPU
    clock_min: list = field(default_factory=list)
    limited: bool = False

    def span(self, tag, gpu=0, skip=1):
        """Median (start, end) of a tagged kernel, ms after its iteration's gate."""
        v = [it["spans"][tag] for it in self.iters[gpu][skip:] if tag in it["spans"]]
        return (float(np.median([a for a, _ in v])), float(np.median([b for _, b in v]))) if v else (0.0, 0.0)

    def t_events(self, gpu=0, skip=1):
        v = [it["end"] - it["gate"] for it in self.iters[gpu][skip:] if it.get("end") is not None]
        return float(np.median(v)) if v else 0.0

    def t_loop(self, gpu=0, skip=1):
        g = [it["gate"] for it in self.iters[gpu][skip:]]
        return float(np.median(np.diff(g))) if len(g) > 1 else self.t_events(gpu, skip)

    def power(self, t0=0.0, t1=None, col=2):
        """Mean W per GPU from t0 to t1 (default: the end)."""
        e, tot = np.zeros(len(self.energy_mj)), 0.0
        for row in self.trace:
            a, b = row[0], row[1]
            if b <= t0 or (t1 is not None and a >= t1):
                continue
            a, b = max(a, t0), b if t1 is None else min(b, t1)
            e += np.asarray(row[col]) * (b - a)
            tot += b - a
        return list(e / tot) if tot > 0 else [0.0] * len(self.energy_mj)

    def rail(self, t0=0.0, t1=None):
        """Mean W per GPU of the part that scales with the SM rail's voltage, before the
        board's gain."""
        return self.power(t0, t1, col=4)

    def clocks(self, t0=0.0, t1=None):
        """(min, time-weighted mean) MHz per GPU from t0 to t1 (default: the end)."""
        lo = [1e9] * len(self.energy_mj)
        acc, tot = np.zeros(len(self.energy_mj)), 0.0
        for a, b, _, f, _ in self.trace:
            if b <= t0 or (t1 is not None and a >= t1):
                continue
            a, b = max(a, t0), b if t1 is None else min(b, t1)
            lo = [min(x, y) for x, y in zip(lo, f, strict=True)]
            acc += np.asarray(f, float) * (b - a)
            tot += b - a
        return lo, list(acc / tot) if tot > 0 else lo


# -------------------------------------------------------------------- engine ----
class Node:
    """n GPUs. `clocks`: one MHz for all or one per GPU; `locked` False leaves each GPU
    to the stock governor (which, for work with no idle gaps, asks for the top clock)."""

    def __init__(
        self,
        kernels=None,
        cal=None,
        *,
        world=4,
        ctas=8,
        clocks=1410,
        locked=True,
        hw=None,
        host=None,
        limiter=True,
        record=True,
        gpus=None,
        comm_clock=None,
    ):
        self.k = kernels or Kernels.load()
        self.pw = (cal or Calib.load()).power
        self.world, self.ctas = world, ctas
        self.hw = hw or Hardware()
        self.host = host or Host()
        self.use_limiter = limiter and self.hw.limiter.enabled
        self.record = record
        # `gpus`: how many of the world's GPUs are simulated. When every GPU runs the same
        # program at the same clock they stay in step, and one stands for all.
        self.n = gpus or world
        fs = list(clocks) if isinstance(clocks, (list, tuple)) else [clocks] * self.n
        # `comm_clock`: a what-if, see _Gpu.f_comm. The shared logic (L2, memory, links)
        # then follows the faster of a GPU's two clocks, as in hardware.Hardware.uncore.
        fc = list(comm_clock) if isinstance(comm_clock, (list, tuple)) else [comm_clock] * self.n
        self.gpus = [
            _Gpu(i, self.hw, f if (locked and f) else self.hw.f_max, locked and bool(f), fc[i])
            for i, f in enumerate(fs)
        ]
        self.t, self.heap, self.seq = 0.0, [], 0
        self.colls = []  # collectives in progress
        self.open = {}  # index of the collective each GPU's next all-reduce joins
        self.n_coll = [0] * self.n
        self.trace = []

    # ---------------------------------------------------------------- helpers ----
    def push(self, t, kind, data=None):
        self.seq += 1
        heapq.heappush(self.heap, (t, self.seq, kind, data))

    def blocked(self, g):
        """SMs a GEMM block cannot use because the collective is resident."""
        return g.nccl_lost

    def in_flight(self, g):
        return sum(n for _, n, _ in g.cohorts)

    # ------------------------------------------------------------------ power ----
    def power(self, g):
        """Board W of GPU g right now: what scales with the SM rail, and the rest.

        idle(f)                       always
        act(f)                        while tiles run or a collective moves bytes: the
                                      busy-chip term of the MoE calibration. A kernel
                                      that only waits does not draw it.
        link_w                        from the first collective until link_ms after the last
        dyn_w * (spin + (1-spin) rho) while a collective is resident; rho = how fast it
                                      moves bytes against what this GPU could do alone
        tiles                         per tile in flight, energy / tile time
        """
        pw, f, ar = self.pw, g.f, self.k.allreduce
        load = self.coll_load(g) if g.nccl is not None else 0.0
        if g.f_comm is None:
            other = pw.idle_at(f)
            if g.cohorts or load > 0:
                other += pw.act_at(f)
        else:  # the what-if: the collective's SM pairs at their own clock, static power by SM share
            hw2 = self.split_hw(g)
            other = pw.static(hw2, [g.fc, f])
            if g.cohorts or load > 0:
                other += pw.active(hw2, [g.fc, f])
        rail, v2 = 0.0, pw.g_at(f)
        for inst, n, _ in g.cohorts:
            s = inst.spec
            sm, mem = self.k.gemm.tile_mj(pw, v2, s.m, s.n, s.k)
            ms = self.k.gemm.tile_ms(s.m, s.n, s.k, f)
            rail += n * sm / ms
            other += n * mem / ms
        if g.nccl is not None:
            other += ar.power_w(self.ctas, g.fc) * (ar.spin + (1.0 - ar.spin) * load)
        if g.nccl is not None or self.t < g.link_until:
            other += ar.link_w
        off, k_rail, k_other = self.k.boards[g.i] if g.i < len(self.k.boards) else (0.0, 1.0, 1.0)
        g.parts = (rail, other)
        return off + k_rail * rail + k_other * other

    def split_hw(self, g):
        """The chip of the what-if: the SM pairs a collective launched first would hold
        (two per CTA) as one V/f domain, the rest as another."""
        if not hasattr(self, "_split"):
            from .hardware import Domain

            n = min(self.hw.sms - 1, round(self.k.allreduce.sms_per_cta * self.ctas))
            self._split = Hardware(
                domains=[Domain("comm", n), Domain("gemm", self.hw.sms - n)],
                **{
                    a: getattr(self.hw, a)
                    for a in ("sms", "gpc_sms", "f_min", "f_max", "f_step", "tdp_w", "uncore", "actuation_ms")
                },
            )
        return self._split

    def window_power(self, g, span):
        t0, e, tot = self.t - span, 0.0, 0.0
        for a, b, p in reversed(g.win):
            if b <= t0:
                break
            a = max(a, t0)
            e += p * (b - a)
            tot += b - a
        return e / tot if tot > 0 else g.P

    # ------------------------------------------------------------------ rates ----
    def coll_rates(self, c):
        """(share of the latency per ms, GB per ms). A ring passes every byte through
        every GPU, so among GPUs at different clocks the slowest participant sets the
        pace -- almost: the measured time per byte is (1 - mix) of the slowest GPU's and
        mix of the mean (`NcclAllReduce.mix`, 0.2)."""
        ar, lat, per_gb = self.k.allreduce, 0.0, []
        for p in c.parts:
            g = self.gpus[p.gpu]
            slow = (ar.slowdown(self.ctas, g.fc) if g.cohorts else 1.0) * p.pack
            lat = max(lat, max(ar.latency_ms(self.ctas, g.fc, self.world) - ar.host_pre, 1e-3))
            per_gb.append(1.0 / (ar.bandwidth_split(self.ctas, g.fc, max(g.f, g.fc), self.world) * slow))
        bw = 1.0 / ((1.0 - ar.mix) * max(per_gb) + ar.mix * float(np.mean(per_gb)))
        return 1.0 / lat, bw / 1e3

    def coll_load(self, g):
        """How hard GPU g's resident collective is working, 0-1: 0 while it waits for a
        participant, 1 while it moves bytes as fast as this GPU could alone."""
        c = g.nccl.coll
        if not c.resident(self.n):
            return 0.0
        if c.latency > EPS:
            return 1.0
        alone = self.k.allreduce.bandwidth_split(self.ctas, g.fc, max(g.f, g.fc), self.world)
        return min(1.0, self.coll_rates(c)[1] * 1e3 / alone)

    def next_internal(self):
        """ms until the first cohort, launch or collective phase ends."""
        dt = np.inf
        for g in self.gpus:
            for inst, _, left in g.cohorts:
                s = inst.spec
                dt = min(dt, left * self.k.gemm.tile_ms(s.m, s.n, s.k, g.f))
            for inst in g.active.values():
                if inst.state == "launch":
                    dt = min(dt, inst.launch_left / g.f + inst.launch_fixed)
                elif inst.state == "tail":
                    dt = min(dt, inst.tail)
        for c in self.colls:
            if c.resident(self.n):
                rl, rb = self.coll_rates(c)
                dt = min(dt, c.latency / rl if c.latency > EPS else c.left / rb)
        return dt

    def advance(self, t):
        dt = t - self.t
        if dt <= 0:
            return
        for c in self.colls:
            if c.resident(self.n):
                rl, rb = self.coll_rates(c)
                if c.latency > EPS:
                    c.latency -= dt * rl
                else:
                    c.left -= dt * rb
        for g in self.gpus:
            for co in g.cohorts:
                s = co[0].spec
                co[2] -= dt / self.k.gemm.tile_ms(s.m, s.n, s.k, g.f)
            for inst in g.active.values():
                if inst.state == "launch":
                    d = min(dt, max(inst.launch_fixed, 0.0))
                    inst.launch_fixed -= d
                    inst.launch_left -= (dt - d) * g.f
                elif inst.state == "tail":
                    inst.tail -= dt
            g.energy += g.P * dt
            g.win.append((self.t, t, g.P))
            keep = max(self.hw.limiter.loop_ms, 100.0)
            while len(g.win) > 1 and g.win[0][1] <= t - keep:
                g.win.popleft()
        if self.record:
            self.trace.append(
                (self.t, t, [g.P for g in self.gpus], [g.f for g in self.gpus], [g.parts[0] for g in self.gpus])
            )
        self.t = t

    # ------------------------------------------------------------------ settle ----
    def settle(self):
        """Everything that can happen in zero time, until nothing more can."""
        again = True
        while again:
            again = False
            for g in self.gpus:
                # cohorts and launches that have run out
                for co in [c for c in g.cohorts if c[2] <= EPS]:
                    g.cohorts.remove(co)
                    again = True
                for inst in list(g.active.values()):
                    if inst.state == "launch" and inst.launch_left <= EPS * g.f and inst.launch_fixed <= EPS:
                        inst.state, again = "run", True
                    if (
                        inst.kind == "gemm"
                        and inst.state == "run"
                        and inst.tiles == 0
                        and not any(c[0] is inst for c in g.cohorts)
                    ):
                        inst.state, again = "tail", True
                    if inst.state == "tail" and inst.tail <= EPS:
                        self.finish(g, inst)
                        again = True
                # stream heads
                for name, q in g.streams.items():
                    if name in g.active or not q or q[0].state != "queued":
                        continue
                    if any(g.tags[w].state != "done" for w in q[0].wait):
                        continue
                    inst = q.popleft()
                    g.active[name] = inst
                    inst.t_start, again = self.t, True
                    if inst.kind == "gemm":
                        s, gm = inst.spec, self.k.gemm
                        inst.tiles = gm.grid(s.m, s.n, s.k)
                        inst.launch_left = gm.u
                        inst.launch_fixed = gm.corun_ms if g.nccl is not None else 0.0
                        inst.tail = 0.0  # ms it still owes after its last tile
                        inst.state = "launch"
                    else:
                        inst.state = "place"
                        # behind a GEMM that is already running: kernels.NcclAllReduce.behind
                        inst.behind = any(x.kind == "gemm" and x is not inst for x in g.active.values())
                # the collective takes SMs before a GEMM's waiting tiles do
                for inst in g.active.values():
                    if inst.state == "place":
                        free = self.hw.sms - self.in_flight(g)
                        if free <= 0:
                            continue
                        ar = self.k.allreduce
                        per_cta, rate = ar.behind(self.ctas, g.f) if inst.behind else (ar.sms_per_cta, 1.0)
                        g.nccl_lost = min(ar.sms_lost_max, round(per_cta * self.ctas))
                        inst.pack = rate
                        g.nccl, inst.state, again = inst, "run", True
                        for x in g.active.values():  # a GEMM it lands beside pays as one launched beside it
                            if x.kind == "gemm" and x.state in ("launch", "run"):
                                x.tail += self.k.gemm.corun_ms
                for inst in g.active.values():
                    if inst.kind == "gemm" and inst.state == "run" and inst.tiles > 0:
                        free = self.hw.sms - self.blocked(g) - self.in_flight(g)
                        n = min(free, inst.tiles)
                        if n > 0:
                            inst.tiles -= n
                            g.cohorts.append([inst, n, 1.0])
                            again = True
            for c in [c for c in self.colls if c.resident(self.n) and c.latency <= EPS and c.left <= EPS * c.gb]:
                self.colls.remove(c)
                c.done = True
                for p in c.parts:
                    gp = self.gpus[p.gpu]
                    gp.nccl, gp.nccl_lost = None, 0
                    gp.link_until = self.t + self.k.allreduce.link_ms
                    self.push(gp.link_until, "link", gp.i)  # power steps down then: an event of its own
                    self.finish(gp, p)
                again = True
            for g in self.gpus:
                if g.syncing and g.drained():
                    g.syncing = False
                    if g.cur is not None:
                        g.cur["end"] = self.t
                    self.push(self.t, "host", g.i)
        for g in self.gpus:
            g.P = self.power(g)

    def finish(self, g, inst):
        inst.state, inst.t_end = "done", self.t
        del g.active[inst.stream]
        if g.cur is not None:
            g.cur["spans"][inst.spec.tag] = (inst.t_issue - g.gate_t, self.t - g.gate_t)
            g.cur["starts"][inst.spec.tag] = inst.t_start - g.gate_t

    # ------------------------------------------------------------------- host ----
    def host_step(self, g):
        """Run host ops of GPU g until one takes time."""
        prog = self.programs[g.i]
        while True:
            if g.pc >= len(prog):
                g.it += 1
                if g.it >= self.n_iters:
                    g.finished = True
                    return
                g.pc = 0
            op = prog[g.pc]
            g.pc += 1
            if op[0] == "gate":
                g.gate_t, g.tags = self.t - self.host.gate, {}
                g.cur = dict(gate=g.gate_t, spans={}, starts={}, end=None)
                self.iters[g.i].append(g.cur)
                continue
            if op[0] == "sleep":
                self.push(self.t + op[1], "host", g.i)
                return
            if op[0] == "clock":
                g.req = self.hw.snap(op[1])
                self.send(g, self.hw.actuation_ms if op[2] is None else op[2])
                continue
            if op[0] == "sync":
                g.syncing = True
                return
            _, stream, spec, wait = op
            prof = self.k.gemm if isinstance(spec, Gemm) else self.k.allreduce
            inst = _Inst(g.i, stream, spec, wait, self.t)
            g.tags[spec.tag] = inst
            g.streams.setdefault(stream, deque()).append(inst)
            if inst.kind == "comm":
                j = self.n_coll[g.i]
                self.n_coll[g.i] += 1
                if j not in self.open:
                    self.open[j] = _Coll(spec.mib)
                    self.colls.append(self.open[j])
                inst.coll, inst.pack = self.open[j], 1.0
                self.open[j].parts.append(inst)
            self.push(self.t + prof.host_pre, "queued", inst)
            self.push(self.t + prof.host, "host", g.i)
            return

    # ---------------------------------------------------------------- limiter ----
    def send(self, g, delay):
        f = min(g.req, g.cap)
        if f != g.target:
            g.target = f
            g.sp_seq += 1
            self.push(self.t + delay, "setpoint", (g.i, f, g.sp_seq))

    def limit(self, g):
        """hardware.Limiter, one per GPU: the rules of engine.Engine.limit()."""
        lim, hw = self.hw.limiter, self.hw
        if g.busy() or g.syncing:
            step, top = lim.step_mhz, g.target
            band = (lim.k_peak - 1.0) * hw.tdp_w
            err = self.window_power(g, lim.loop_ms) - hw.tdp_w
            g.excess_j += err * lim.loop_ms / 1e3
            cap = None
            if g.target != g.f:
                pass
            elif err > band:
                cap = top - step * int(np.ceil(lim.gain_mhz_per_w * err / step))
            elif err < -band and int(lim.gain_mhz_per_w * -err / step) > 0:
                cap = max(g.cap, top + step * int(lim.gain_mhz_per_w * -err / step))
            elif g.excess_j > lim.budget_j:
                cap = min(g.cap, top) - step
            elif g.excess_j < -lim.budget_j:
                cap = max(g.cap, top + step)
            if cap is not None:
                g.excess_j = 0.0
                g.cap = min(max(cap, hw.f_min), hw.f_max)
            if g.cap >= hw.f_max and g.excess_j < 0:
                g.excess_j = 0.0
            self.send(g, min(lim.latency_ms, hw.actuation_ms))
        self.push(self.t + lim.loop_ms, "limit", g.i)

    # -------------------------------------------------------------------- run ----
    def run(self, programs, iters=3, t_max=None):
        """programs: one list of host ops per GPU (or one for all), repeated `iters`
        times; with t_max, repeated until that many ms have passed."""
        if programs and isinstance(programs[0], tuple):
            programs = [programs] * self.n
        self.programs = programs
        self.n_iters = 10**9 if t_max else iters
        self.iters = [[] for _ in self.gpus]
        for g in self.gpus:
            g.finished, g.cur = False, None
            self.push(0.0, "host", g.i)
            if self.use_limiter:
                self.push(self.hw.limiter.loop_ms, "limit", g.i)
        self.settle()
        while True:
            if all(g.finished and g.drained() for g in self.gpus):
                break
            t_int = self.t + self.next_internal()
            t_ev = self.heap[0][0] if self.heap else np.inf
            if t_max is not None and min(t_int, t_ev) > t_max:
                self.advance(t_max)
                break
            if t_int <= t_ev:
                if not np.isfinite(t_int):
                    raise RuntimeError("nothing can progress: a collective is waiting for a GPU that never joins")
                self.advance(t_int)
            else:
                t, _, kind, data = heapq.heappop(self.heap)
                self.advance(t)
                if kind == "host":
                    g = self.gpus[data]
                    if not g.finished and not g.syncing:
                        self.host_step(g)
                elif kind == "queued":
                    data.state = "queued"
                elif kind == "setpoint":
                    i, f, seq = data
                    g = self.gpus[i]
                    if seq >= g.sp_last:
                        g.sp_last, g.f = seq, f
                        g.f_min_seen = min(g.f_min_seen, f)
                elif kind == "limit":
                    if all(x.finished and x.drained() for x in self.gpus):
                        continue
                    self.limit(self.gpus[data])
            self.settle()
        return Result(
            iters=self.iters,
            t_ms=self.t,
            energy_mj=[g.energy for g in self.gpus],
            trace=self.trace,
            clock_min=[g.f_min_seen for g in self.gpus],
        )


# ------------------------------------------------------- the harness, as a workload ----
def programs(mode, shapes, mib, world=4, host=None):
    """Host programs of one iteration of sweep_spans.py / sweep_overlap_432.py.
    shapes: one 'MxNxK' for every GPU or one per GPU."""
    shapes = list(shapes) if isinstance(shapes, (list, tuple)) else [shapes] * world
    out = []
    for s in shapes:
        ops = [GATE]
        gm = Gemm(*(int(v) for v in s.split("x"))) if mode != "comm_only" else None
        ar = AllReduce(mib) if mode != "gemm_only" else None
        if mode == "comm_only":
            ops.append(launch("comm", ar))
        elif mode == "gemm_only":
            ops.append(launch("gemm", gm))
        elif mode == "concurrent":
            ops += [launch("comm", ar), launch("gemm", gm)]
        elif mode == "concurrent_gf":
            ops += [launch("gemm", gm), launch("comm", ar)]
        elif mode == "serial":
            ops += [launch("gemm", gm), launch("gemm", ar)]
        else:
            raise ValueError(mode)
        out.append(ops + [SYNC, sleep((host or Host()).loop[mode])])
    return out


def cell(
    kernels,
    cal,
    mode,
    *,
    shape="0x0x0",
    mib=0,
    ctas=8,
    clock=1410,
    world=4,
    window_ms=1200.0,
    discard_ms=600.0,
    limiter=True,
    comm_clock=None,
):
    """Predict one cell of the harness: dict(t_loop, t_events, gemm=(start, end),
    comm=(start, end), power=[W per GPU], clock_min, clock_mean=[MHz per GPU], spans=[per
    GPU]), times as rank 0 measured them. `clock`: MHz for all, one per GPU, or 0 = not
    locked.

    A cell that stays under the power limit repeats exactly from the second iteration,
    so four are run. One that may reach it is run with the limiter until that settles.
    GPUs that do the same thing at the same clock are simulated once."""
    same = not isinstance(shape, (list, tuple)) and not isinstance(clock, (list, tuple))
    boards = kernels.boards
    same = same and all(list(b) == list(boards[0]) for b in boards[:world])
    n_sim = 1 if same else world
    progs = programs(mode, shape, mib, n_sim) if same else programs(mode, shape, mib, world)
    fs = clock if clock else Hardware().f_max
    mk = lambda lim: Node(
        kernels, cal, world=world, ctas=ctas, clocks=fs, limiter=lim, gpus=n_sim, comm_clock=comm_clock
    )
    r = mk(False).run(progs, iters=4 if same else 12)
    w, limited = _windows(r), False
    p = _per_gpu(r, w, 2)
    if limiter and max(p) > 0.93 * Hardware().tdp_w:
        r = mk(True).run(progs, t_max=window_ms)
        w, limited = _windows(r, discard_ms), True
        p = _per_gpu(r, w, 2)
    lo, mean = _clocks(r, w)
    skip = max(1, int(len(r.iters[0]) * (w[0][0] / max(r.t_ms, 1e-9)))) if limited else 1
    rep = (lambda v: list(v) * world) if same else list
    return dict(
        t_loop=r.t_loop(0, skip),
        t_events=r.t_events(0, skip),
        gemm=r.span("gemm", 0, skip),
        comm=r.span("comm", 0, skip),
        power=rep(p),
        rail=rep(_per_gpu(r, w, 4)),
        clock_min=rep(lo),
        clock_mean=rep(mean),
        limited=limited,
        spans=rep(
            [
                dict(gemm=r.span("gemm", i, skip), comm=r.span("comm", i, skip), t=r.t_events(i, skip))
                for i in range(n_sim)
            ]
        ),
    )


def _windows(r, discard_ms=0.0):
    """Per GPU, the stretch a mean is taken over: whole iterations of that GPU, from its
    second gate (or its first after `discard_ms`) to its last. Whole iterations, because
    a run that stops mid-step leaves the GPUs that finish early idle to its end, and GPUs
    that are not in step have different iterations."""
    out = []
    for its in r.iters:
        gates = [it["gate"] for it in its]
        first = next((g for g in gates[1:] if g >= discard_ms), gates[min(1, len(gates) - 1)])
        last = gates[-1] if gates[-1] > first else r.t_ms
        out.append((first, last))
    return out


def _per_gpu(r, windows, col):
    return [r.power(a, b, col)[i] for i, (a, b) in enumerate(windows)]


def _clocks(r, windows):
    lo, mean = [], []
    for i, (a, b) in enumerate(windows):
        mn, av = r.clocks(a, b)
        lo.append(mn[i])
        mean.append(av[i])
    return lo, mean


def from_ops(ops, host_gap=0.045):
    """Host program of one rank from the JSON ops sweep_spans.py runs (`mode: program`):
    [{"op": "gemm" | "allreduce", "tag", "stream", "shape" | "mib", "wait": [tags]}]."""
    out = [GATE]
    for o in ops:
        if o["op"] == "sleep":
            out.append(sleep(o["ms"]))
            continue
        if o["op"] == "gemm":
            k = Gemm(*(int(v) for v in o["shape"].split("x")), tag=o["tag"])
        else:
            k = AllReduce(o["mib"], tag=o["tag"])
        out.append(launch(o.get("stream", "main"), k, o.get("wait", ())))
    return out + [SYNC, sleep(host_gap)]


def program_cell(kernels, cal, ops, *, ctas=8, clock=1410, world=4, window_ms=1200.0, discard_ms=600.0, limiter=True):
    """Predict one `mode: program` cell. ops: one list for every rank or one per rank.
    Returns dict(t_events=[per GPU], t_loop, ends=[{tag: ms} per GPU], starts=[...],
    power=[W per GPU], clock_min, clock_mean, limited)."""
    per_rank = isinstance(ops[0], list)
    progs = [from_ops(o) for o in ops] if per_rank else [from_ops(ops)] * world
    fs = clock if clock else Hardware().f_max
    mk = lambda lim: Node(kernels, cal, world=world, ctas=ctas, clocks=fs, limiter=lim)
    r = mk(False).run(progs, iters=5 if per_rank or isinstance(clock, (list, tuple)) else 3)
    w, limited = _windows(r), False
    p = _per_gpu(r, w, 2)
    if limiter and max(p) > 0.93 * Hardware().tdp_w:
        r = mk(True).run(progs, t_max=max(window_ms, 6 * r.t_loop(0)))
        w, limited = _windows(r, min(discard_ms, 0.5 * r.t_ms)), True
        p = _per_gpu(r, w, 2)
    lo, mean = _clocks(r, w)
    skip = max(1, int(len(r.iters[0]) * (w[0][0] / max(r.t_ms, 1e-9)))) if limited else 1
    med = lambda i, key, tag, j=None: float(
        np.median(
            [(it[key][tag] if j is None else it[key][tag][j]) for it in r.iters[i][skip:] if tag in it[key]] or [0.0]
        )
    )
    tags = lambda i: [o for o in (ops[i] if per_rank else ops) if o["op"] != "sleep"]
    return dict(
        t_events=[r.t_events(i, skip) for i in range(world)],
        t_loop=r.t_loop(0, skip),
        ends=[{o["tag"]: med(i, "spans", o["tag"], 1) for o in tags(i)} for i in range(world)],
        starts=[{o["tag"]: med(i, "starts", o["tag"]) for o in tags(i)} for i in range(world)],
        power=p,
        rail=_per_gpu(r, w, 4),
        clock_min=lo,
        clock_mean=mean,
        limited=limited,
    )
