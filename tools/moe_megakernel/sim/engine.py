"""The event loop.

Between two events nothing changes: every running job progresses at a constant rate and
the board draws a constant power. So the simulator never steps through time -- it jumps
from one event to the next and integrates exactly:

    progress += dt / T_job          energy += P * dt

An event is whatever changes a rate or the power:

  arrive     a batch arrives
  start      a kernel launch reaches the GPU (launches on a second stream trail the first)
  finish     a partition finishes its GEMM, or an epilogue pass ends
  setpoint   a clock set-point takes effect, `actuation_ms` after it was requested
  limit      the power limiter looks at the last few ms and moves its clock ceiling
  tick       a power reading is reported; the controller may act on it
  call       a controller's own timer (the governor's wake-up and park delays)

Exactly one handler runs per event. It may add events. Then the rates of everything
still running are solved again (`rate.solve_rates`), because a partition that has lost
its neighbour runs faster, and one whose clock just changed runs at a different speed.

A MoE layer is four stages, with a barrier after each:

  g1    every partition's GEMM 1, side by side          <- partitions finish at
  act   activation                                         different times; the layer
  g2    every partition's GEMM 2, side by side             waits for the last
  sum   top-k sum

The same engine, restricted to one layer and a fixed clock, is the PREDICTOR that
model-based controllers call (`Engine.predict`): a policy and the plant it controls
share one model by construction, so a policy cannot look good through a mismatch.

WHERE TO EXTEND
  another kernel implementation   a `KernelCal` in the calibration file, named by
                                  `Kernel.type`: nothing here changes
  another controller              control.py; the hooks it gets are in `Controller`
  another way to place experts    `Engine.plan`
"""

from collections import deque
from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Optional

import numpy as np

from .demand import GEMMS, SHAPES, Partition, split
from .hardware import Hardware
from .rate import Job, solve_rates
from .workload import Batch, Layer

NEXT_STAGE = {"g1": "act", "act": "g2", "g2": "sum", "sum": None}
REBALANCE = -1  # Kernel.n_cold: choose the cold set per layer


@dataclass
class Kernel:
    """How a layer is launched."""

    # Whose per-SM costs, by name in the calibration file: 'persistent' (ours, one block
    # per SM, as measured) or 'queue' (vLLM's). 'queue' WITH n_cold > 0 is a kernel nobody
    # has built: pinned like ours, as efficient per SM as vLLM's -- a what-if for a better
    # megakernel.
    type: str = "persistent"
    block_m: int = 64
    # The n least-routed experts form the cold partition. 0: one partition over every SM.
    # -1 ('auto' in a scenario): chosen per layer so that the two partitions finish
    # together at equal clocks. Which experts are cold is software and free to move; how
    # many SMs a domain has is silicon.
    n_cold: int = 0
    # SMs of the cold partition. On two-domain hardware the domains decide and this is
    # ignored.
    cold_sms: int = 0
    # A launch on a second stream trails the first by this. The two-stream prototype
    # could trail by up to 20 us; its pinned layers, measured end to end, fit best with
    # none (1.38% rms against 1.55% at 20 us), and a single-launch kernel has none.
    launch_ms: float = 0.0
    candidates: tuple = (8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120)

    def __post_init__(self):
        if self.n_cold == "auto":
            self.n_cold = REBALANCE

    def sig(self):
        return (self.type, self.block_m, self.n_cold, self.cold_sms, self.launch_ms)

    def with_cold(self, n_cold):
        return Kernel(self.type, self.block_m, n_cold, self.cold_sms, self.launch_ms)


@dataclass
class Part:
    """One partition of one layer, placed."""

    name: str
    partition: Partition
    n: int  # SMs it runs on
    domain: Optional[int]  # index of its V/f domain; None = spans every domain
    key: tuple = ()  # identifies the partition's content, for caches


class Shared:
    """Caches that outlive one Engine: the predictor creates thousands of them. One
    Shared belongs to one calibration.

      parts    (layer, kernel, domain sizes)            -> placed partitions
      jobs     (partition, GEMM, SMs, kernel, clocks)   -> rate.Job, with its time table
      solved   (the jobs running side by side)          -> their solved times
      pred     (layer, kernel, clocks, hardware)        -> one layer's steady state
    """

    def __init__(self):
        self.parts, self.jobs, self.solved, self.pred = {}, {}, {}, {}


class _Run:
    """A job in flight: one GEMM of one partition, or one epilogue pass over `pairs`."""

    __slots__ = ("T", "domains", "frac", "g", "gen", "job", "key", "name", "pairs", "part")

    def __init__(self, name, g, domains, part=None, pairs=0.0):
        self.name, self.g, self.domains, self.part, self.pairs = name, g, domains, part, pairs
        self.frac, self.T, self.job, self.key, self.gen = 1.0, 0.0, None, None, 0


class Engine:
    def __init__(
        self,
        hw: Hardware,
        cal,
        kernel: Kernel,
        batches: list[Batch],
        controller,
        *,
        model="qwen",
        shared=None,
        record=True,
        limiter=True,
    ):
        self.hw, self.cal, self.kernel, self.batches = hw, cal, kernel, batches
        self.ctl, self.shape = controller, SHAPES[model]
        self.shared = shared or Shared()
        self.record, self.use_limiter = record, limiter and hw.limiter.enabled
        self.D = len(hw.domains)
        # ---- time, energy, events
        self.t = 0.0
        self.energy = 0.0  # mJ
        self.heap, self.seq = [], 0
        self.done = False
        # ---- clocks: what is in force, what was asked for, what is on its way
        f0 = [d.f_init or hw.f_max for d in hw.domains]
        self.f = list(f0)  # effective clocks
        self.req = list(f0)  # what the controller asked for
        self.target = list(f0)  # last set-point sent to the chip
        self.cap = hw.f_max  # the limiter's ceiling
        self.sp_seq, self.sp_last = 0, [0] * self.D  # set-points: issued, last applied
        self.excess_j = 0.0  # the limiter's energy account
        # ---- what is running
        self.running: list[_Run] = []
        self.waiting = 0  # launches not yet on the GPU
        self.bi = self.li = -1  # batch and layer in progress
        self.stage = None
        self.parts: list[Part] = []
        self.P = 0.0  # board power right now, W
        self.busy_since = None
        self.idle_since = None
        # ---- records
        lim = hw.limiter
        self.keep_ms = max(lim.window_ms, lim.period_ms, lim.loop_ms)
        self.win = deque()  # (t0, t1, W) of the last keep_ms
        self.events, self.layers, self.ticks, self.trace = [], [], [], []
        self._layer = None

    # ---------------------------------------------------------------- helpers ----
    def push(self, t, kind, data=None):
        self.seq += 1
        heappush(self.heap, (t, self.seq, kind, data))

    def log(self, kind, who="", detail=""):
        if self.record:
            self.events.append(
                dict(
                    t_ms=round(self.t, 6),
                    event=kind,
                    batch=self.bi,
                    layer=self.li,
                    stage=self.stage or "",
                    who=who,
                    detail=detail,
                )
            )

    # ----------------------------------------------------------------- clocks ----
    def request(self, d, f):
        """Controller API: ask for clock f on domain d (None = every domain)."""
        for i in range(self.D) if d is None else [d]:
            self.req[i] = self.hw.snap(f)
        self._send()

    def _send(self, delay=None):
        """Forward min(request, limiter ceiling) to the chip. It arrives `actuation_ms`
        later if a controller asked, or after `delay` if the limiter moved its ceiling.
        A later decision supersedes an earlier one that is still on its way."""
        delay = self.hw.actuation_ms if delay is None else delay
        for i in range(self.D):
            f = min(self.req[i], self.cap)
            if f != self.target[i]:
                self.target[i] = f
                self.sp_seq += 1
                self.push(self.t + delay, "setpoint", (i, f, self.sp_seq))

    # ------------------------------------------------------------------- plan ----
    def plan(self, layer: Layer) -> list[Part]:
        """The partitions of a layer and the SMs each runs on (cached)."""
        k, hw = self.kernel, self.hw
        key = (layer.key, k.sig(), tuple(d.sms for d in hw.domains))
        if key not in self.shared.parts:
            if k.n_cold < 0:  # REBALANCE
                parts = self._rebalanced(layer)
            elif k.n_cold == 0:
                parts = [
                    Part(
                        "all",
                        Partition(self.shape, layer.counts, k.block_m),
                        hw.sms,
                        None,
                        (layer.key, k.block_m, 0, "all"),
                    )
                ]
            else:
                parts = self._pinned(layer)
            self.shared.parts[key] = parts
        return self.shared.parts[key]

    def _pinned(self, layer):
        """The n_cold least-routed experts on the cold SMs, the rest on the others."""
        k, hw = self.kernel, self.hw
        if self.D == 2:
            ic = hw.index("cold") if "cold" in [d.name for d in hw.domains] else 0
            n_c, d_c, d_h = hw.domains[ic].sms, ic, 1 - ic
        elif self.D == 1:
            n_c, d_c, d_h = k.cold_sms, 0, 0
            if not 0 < n_c < hw.sms:
                raise ValueError("a partitioned kernel on one-domain hardware needs 0 < cold_sms < sms")
        else:
            raise ValueError("partitioned kernels are defined for one or two domains")
        cold = split(layer.counts, k.n_cold)
        key = (layer.key, k.block_m, k.n_cold)
        parts = [
            Part("cold", Partition(self.shape, layer.counts, k.block_m, cold), n_c, d_c, key + ("cold",)),
            Part("hot", Partition(self.shape, layer.counts, k.block_m, ~cold), hw.sms - n_c, d_h, key + ("hot",)),
        ]
        return [p for p in parts if p.partition]

    def _rebalanced(self, layer):
        """The cold-expert count, among `Kernel.candidates`, with which the layer is
        fastest at the top clock: the partitions then finish together."""
        k, hw = self.kernel, self.hw
        best = None
        for c in k.candidates:
            ms = self.predict(layer, [hw.f_max] * self.D, k.with_cold(c))["ms"]
            if best is None or ms < best[0]:
                best = (ms, c)
        return Engine(
            hw,
            self.cal,
            k.with_cold(best[1]),
            [],
            self.ctl,
            model=self.shape.name,
            shared=self.shared,
            record=False,
            limiter=False,
        ).plan(layer)

    def _job(self, part: Part, g):
        """(cache key, rate.Job) of this partition's GEMM at the clocks in force."""
        hw, k = self.hw, self.kernel
        fu = hw.uncore_clock(self.f)
        if part.domain is None:
            if len(set(self.f)) > 1:
                raise ValueError(f"an unpartitioned kernel spans every SM and cannot run with domain clocks {self.f}")
            fd = self.f[0]
        else:
            fd = self.f[part.domain]
        key = (part.key, g, part.n, k.type, fd, fu, hw.cdc_penalty, hw.delivery_uncore_share, hw.xi_clock)
        j = self.shared.jobs.get(key)
        if j is None:
            kc = self.cal.kernel(k.type)
            window = None if kc.window == 1 else int(kc.window * part.n)
            st = part.partition.stage(g, part.n, window)
            c = kc.rate.at(fd, fu, hw.delivery_uncore_share, hw.xi_clock == "uncore")
            if fd != fu and hw.cdc_penalty:
                c = type(c)(**{**c.__dict__, "kP": c.kP * (1 + hw.cdc_penalty)})
            j = self.shared.jobs[key] = Job(st, c)
        return key, j

    # ---------------------------------------------------------------- physics ----
    def refresh(self):
        """Clocks or the set of running jobs changed: solve rates, power, finish times."""
        gemm = [r for r in self.running if r.g in GEMMS]
        if gemm:
            src = [self._job(r.part, r.g) for r in gemm]
            key = tuple(k for k, _ in src)
            sol = self.shared.solved.get(key)
            if sol is None:  # this set of jobs, first time
                jobs = [j for _, j in src]
                solve_rates(jobs)
                sol = self.shared.solved[key] = [j.t for j in jobs]
            for r, (_, j), t in zip(gemm, src, sol, strict=True):
                r.job, r.T = j, t
        f_eff = self.hw.epilogue_clock_of(self.f)
        for r in self.running:
            if r.g not in GEMMS:
                r.T = 0.5 * self.cal.epilogue.ms(r.pairs, f_eff)
            r.gen += 1
            self.push(self.t + r.frac * r.T, "finish", (r, r.gen))
        self.P = self.power()

    def power(self):
        """Board W right now: static, plus what each running job does per second."""
        hw, pw, f = self.hw, self.cal.power, self.f
        p = pw.static(hw, f)
        for r in self.running:
            if r.g in GEMMS:
                d = r.part.domain
                g = pw.g_at(f[0]) if d is None else pw.rail(hw, f, d)
                p += pw.stage_mj(g, r.job.st) / r.T  # mJ / ms = W
            else:
                g = sum(dm.sms * pw.rail(hw, f, i) for i, dm in enumerate(hw.domains)) / hw.sms
                e = self.cal.epilogue
                p += 0.5 * r.pairs * (e.j_dram + g * e.j_sm) / r.T
        return p + pw.active(hw, f) if self.running or self.waiting else p

    def advance(self, t):
        """Move to time t: nothing changes on the way."""
        dt = t - self.t
        if dt <= 0:
            return
        for r in self.running:
            r.frac -= dt / r.T
        self.energy += self.P * dt
        self.win.append((self.t, t, self.P))
        while len(self.win) > 1 and self.win[0][1] <= t - self.keep_ms:
            self.win.popleft()
        if self.record:
            self.trace.append((self.t, t, self.P, tuple(self.f), "+".join(f"{r.name}:{r.g}" for r in self.running)))
        self.t = t

    def window_power(self, span):
        """Mean board power over the last `span` ms (less if the run is younger)."""
        t0 = self.t - span
        e = tot = 0.0
        for a, b, p in reversed(self.win):
            if b <= t0:
                break
            a = max(a, t0)
            e += p * (b - a)
            tot += b - a
        return e / tot if tot > 0 else self.P

    # ----------------------------------------------------------------- stages ----
    def next_batch(self):
        self.bi += 1
        if self.bi >= len(self.batches):
            self.done = True
            self.stage = None
            self.log("end")
            return
        b = self.batches[self.bi]
        t = max(self.t + b.gap_ms, b.at_ms if b.at_ms is not None else self.t)
        self.stage = None
        if t > self.t + 1e-12:
            self.busy_since = None
            self.idle_since = self.t
            self.ctl.on_idle()
        self.push(t, "arrive", self.bi)

    def start_layer(self):
        b = self.batches[self.bi]
        layer = b.layers[self.li]
        self.parts = self.plan(layer)
        self._layer = dict(
            batch=self.bi,
            name=b.name,
            phase=b.phase,
            tokens=b.tokens,
            layer=layer.index,
            t_start=self.t,
            e0=self.energy,
            f_start=tuple(self.f),
            finish={},
        )
        self.ctl.on_layer(b, layer)
        self.start_stage("g1")

    def start_stage(self, g):
        self.stage = g
        if g in GEMMS:
            for i, p in enumerate(self.parts):
                doms = list(range(self.D)) if p.domain is None else [p.domain]
                r = _Run(p.name, g, doms, part=p)
                if i == 0 or self.kernel.launch_ms <= 0:
                    self.running.append(r)
                    self.log("start", p.name)
                else:
                    self.waiting += 1
                    self.push(self.t + i * self.kernel.launch_ms, "start", r)
        else:
            # in units of the calibration model's pairs: the epilogue moves bytes
            pairs = self.batches[self.bi].layers[self.li].pairs * self.shape.epi_bytes / self.cal.epilogue.ref_bytes
            self.running.append(_Run("epilogue", g, list(range(self.D)), pairs=pairs))
            self.log("start", "epilogue")

    def end_stage(self):
        nxt = NEXT_STAGE[self.stage]
        if nxt is not None:
            self.start_stage(nxt)
            return
        L = self._layer
        L.update(t_end=self.t, ms=self.t - L["t_start"], mj=self.energy - L["e0"], f_end=tuple(self.f))
        L["w"] = L["mj"] / L["ms"] if L["ms"] > 0 else 0.0
        del L["e0"]
        self.layers.append(L)
        self.log("layer_end", detail=f"{L['ms']:.4f} ms")
        self.ctl.on_layer_end(L)
        b = self.batches[self.bi]
        self.li += 1
        if self.li < len(b.layers):
            self.start_layer()
        else:
            self.log("batch_end", b.name)
            self.next_batch()

    # ----------------------------------------------------------------- events ----
    def handle(self, kind, data):
        """Run the handler of one event. False: nothing changed, rates need no solve."""
        if kind == "arrive":
            self.li = 0
            b = self.batches[self.bi]
            if self.busy_since is None:
                self.busy_since = self.t
            self.log("arrive", b.name)
            self.ctl.on_batch(b)
            self.start_layer()
        elif kind == "start":
            self.waiting -= 1
            self.running.append(data)
            self.log("start", data.name)
        elif kind == "finish":
            r, gen = data
            self.running.remove(r)
            if self._layer is not None:
                self._layer["finish"][(r.name, r.g)] = self.t
            self.log("finish", r.name)
            if r.g in GEMMS:
                self.ctl.on_finish(r.name, r.g, [x.name for x in self.running])
            if not self.running and not self.waiting:
                self.end_stage()
        elif kind == "setpoint":
            d, f, seq = data
            if seq < self.sp_last[d]:  # a later decision got here first
                return False
            self.sp_last[d] = seq
            if self.f[d] == f:
                return False
            self.log("setpoint", self.hw.domains[d].name, f"{self.f[d]} -> {f} MHz")
            self.f[d] = f
        elif kind == "limit":
            self.limit()
        elif kind == "tick":
            self.tick()
        elif kind == "call":  # a controller's own timer
            data()
        return True

    def limit(self):
        """The hardware limiter's own loop (hardware.Limiter)."""
        lim, hw = self.hw.limiter, self.hw
        if self.running or self.waiting:  # an idle chip is never over TDP
            step, top = lim.step_mhz, max(self.target)  # top: the clock asked of the chip
            band = (lim.k_peak - 1.0) * hw.tdp_w
            err = self.window_power(lim.loop_ms) - hw.tdp_w
            self.excess_j += err * lim.loop_ms / 1e3
            cap = None
            if self.target != self.f:
                pass  # a set-point is on its way: the reading is not of the clock asked for
            elif err > band:
                cap = top - step * int(np.ceil(lim.gain_mhz_per_w * err / step))
            elif err < -band and int(lim.gain_mhz_per_w * -err / step) > 0:
                cap = max(self.cap, top + step * int(lim.gain_mhz_per_w * -err / step))
            elif self.excess_j > lim.budget_j:
                cap = min(self.cap, top) - step
            elif self.excess_j < -lim.budget_j:
                cap = max(self.cap, top + step)
            if cap is not None:
                self.excess_j = 0.0
                self.cap = min(max(cap, hw.f_min), hw.f_max)
            if self.cap >= hw.f_max and self.excess_j < 0:
                self.excess_j = 0.0  # no credit is saved at the top clock
            self._send(min(lim.latency_ms, hw.actuation_ms))
        if not self.done:
            self.push(self.t + lim.loop_ms, "limit")

    def tick(self):
        """A reported power reading: what a controller, and NVML, would see."""
        lim = self.hw.limiter
        p_win = self.window_power(lim.window_ms)
        p_now = self.window_power(lim.period_ms)
        self.ctl.on_tick(p_win, p_now)
        self._send()
        if self.record:
            self.ticks.append(
                dict(
                    t_ms=round(self.t, 3),
                    p_window_w=round(p_win, 2),
                    p_sensor_w=round(p_now, 2),
                    cap_mhz=self.cap,
                    **{f"f_{d.name}": f for d, f in zip(self.hw.domains, self.f, strict=True)},
                    **{f"req_{d.name}": f for d, f in zip(self.hw.domains, self.req, strict=True)},
                )
            )
        if not self.done:
            self.push(self.t + lim.period_ms, "tick")

    def run(self, t_max=None):
        self.ctl.attach(self)
        self.target = [min(r, self.cap) for r in self.req]
        self.f = list(self.target)
        self.P = self.power()
        self.push(self.hw.limiter.period_ms, "tick")
        if self.use_limiter:
            self.push(self.hw.limiter.loop_ms, "limit")
        self.next_batch()
        while self.heap and not self.done:
            t, _, kind, data = heappop(self.heap)
            if kind == "finish" and (data[1] != data[0].gen or data[0] not in self.running):
                continue  # superseded by a later solve
            if t_max is not None and t > t_max:
                self.advance(t_max)
                break
            self.advance(t)
            if self.handle(kind, data) and not self.done:
                self.refresh()
        return self

    # -------------------------------------------------------------- predictor ----
    def predict(self, layer: Layer, f, kernel=None):
        """What one layer costs at fixed clocks f (one per domain), in steady state:
        dict(ms, mj, w, finish). The plant's own model, run on one layer."""
        from .control import Fixed

        k = kernel or self.kernel
        f = tuple(self.hw.snap(x) for x in f)
        key = (layer.key, k.sig(), f, self.hw.sig())
        hit = self.shared.pred.get(key)
        if hit is None:
            e = Engine(
                self.hw,
                self.cal,
                k,
                [Batch("p", "x", 0, [layer])],
                Fixed(list(f)),
                model=self.shape.name,
                shared=self.shared,
                record=False,
                limiter=False,
            )
            e.run()
            L = e.layers[0]
            hit = self.shared.pred[key] = dict(ms=L["ms"], mj=L["mj"], w=L["w"], finish=L["finish"])
        return hit
