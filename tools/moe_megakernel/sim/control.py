"""Who sets the clocks.

Every controller runs UNDER the hardware's power limiter (hardware.Limiter), which the
engine applies to all of them alike. A controller only requests clocks; a request takes
`actuation_ms` to arrive and is clipped to the limiter's ceiling.

  fixed            one clock per domain, never changed
  boost_on_finish  fixed clocks, and the event-driven move: the moment one domain's
                   partition finishes a GEMM, the domain still running is raised until
                   the stage's barrier releases
  governor         the stock behaviour, as measured: park an idle chip, wake to the top
                   clock, and let the limiter pull it back to the power wall
  tdp_bisection    open loop, model-based: the fastest clocks whose PREDICTED power fits
                   under TDP, by nested bisection -- an outer search on the layer time,
                   an inner search per domain for the slowest clock that still meets it
  closed_loop      no model: raise the bottleneck domain while the measured window is
                   under TDP, lower the domain that finishes early
  min_energy_slo   the least energy that still meets a latency target, under TDP
  offline_optimal  exhaustive search of the clock grid with full knowledge: the bound
                   every online policy is compared against

The model-based ones call `Engine.predict`, the plant's own model. They are open loop:
if the model is wrong the limiter, not the controller, catches it.

WRITING A CONTROLLER. Subclass `Controller`, give it a `name`, decorate it with
`@register`; a scenario's `controller: {type: <name>, ...}` then builds it, the other
keys becoming constructor arguments. The engine calls the hooks below. A controller
acts through three calls on `self.sim` (the Engine):

  sim.request(d, f)        ask for clock f on domain d (None: every domain)
  sim.predict(layer, f)    what one layer would cost at clocks f: dict(ms, mj, w, finish)
  sim.push(t, "call", fn)  run fn at time t (a timer of its own)

and may read sim.t, sim.f (clocks in force), sim.req, sim.cap, sim.hw, sim.parts.
"""

import numpy as np

REGISTRY = {}


def register(cls):
    """Make a controller class available to scenario files under its `name`."""
    REGISTRY[cls.name] = cls
    return cls


def make(spec):
    """A controller from a scenario's spec: {type: name, **constructor arguments}."""
    spec = dict(spec)
    kind = spec.pop("type")
    if kind not in REGISTRY:
        raise ValueError(f"unknown controller {kind}; have {sorted(REGISTRY)}")
    return REGISTRY[kind](**spec)


class Controller:
    """The hooks. Every one is optional."""

    name = "controller"

    def attach(self, sim):
        self.sim = sim

    def on_batch(self, batch):
        pass  # a batch arrives

    def on_layer(self, batch, layer):
        pass  # a layer is about to start

    def on_layer_end(self, rec):
        pass  # ... has ended: its record

    def on_finish(self, name, g, still_running):
        pass  # a partition finished a GEMM

    def on_idle(self):
        pass  # nothing left to run for now

    def on_tick(self, p_window, p_now):
        pass  # a power reading is reported

    def describe(self):
        return {"type": self.name}

    # -- shared by the model-based controllers ---------------------------------
    def _proxy(self, batch, k=3):
        """Layers the model is evaluated on. A set-point takes tens of ms to arrive, so
        there is no point solving for each of 48 layers; a few stand for the batch."""
        L = batch.layers
        if len(L) <= k:
            return L
        return [L[i] for i in np.linspace(0, len(L) - 1, k).round().astype(int)]

    def _cost(self, layers, f):
        """(ms, mJ, mean W) of the layers at clocks f, from the plant's own model."""
        ms = mj = 0.0
        for layer in layers:
            r = self.sim.predict(layer, f)
            ms += r["ms"]
            mj += r["mj"]
        return ms, mj, mj / ms


@register
class Fixed(Controller):
    name = "fixed"

    def __init__(self, f):
        self.f0 = f

    def attach(self, sim):
        self.sim = sim
        f = self.f0 if isinstance(self.f0, (list, tuple)) else [self.f0] * sim.D
        if len(f) != sim.D:
            raise ValueError(f"fixed: {len(f)} clocks for {sim.D} domains")
        sim.req = [sim.hw.snap(x) for x in f]

    def describe(self):
        return {"type": self.name, "f": self.f0}


@register
class BoostOnFinish(Fixed):
    """Fixed clocks `f`, and a boost that only an event loop can time: when a partition
    finishes a GEMM while another is still running, the survivor's domain goes to
    `boost` (if that is higher) until the barrier releases and both start again.

    A boost that cannot land in time is not issued. The survivor's remaining time is
    known when its neighbour finishes -- at the rate it had beside that neighbour, so an
    upper bound -- and must exceed the actuation delay. `sim.boost` searches f and boost
    and reports what the move is worth."""

    name = "boost_on_finish"

    def __init__(self, f, boost):
        super().__init__(f)
        self.boost, self.boosts = boost, 0  # boosts: how many were issued

    def attach(self, sim):
        super().attach(sim)
        self.plan = list(sim.req)

    def on_finish(self, name, g, still_running):
        sim = self.sim
        if not still_running:  # the barrier releases: back to the plan
            for d, f in enumerate(self.plan):
                sim.request(d, f)
            return
        if max(r.frac * r.T for r in sim.running) <= sim.hw.actuation_ms:
            return
        for p in sim.parts:
            if p.name in still_running and p.domain is not None and self.boost > self.plan[p.domain]:
                self.boosts += 1
                sim.request(p.domain, self.boost)

    def describe(self):
        return {"type": self.name, "f": self.f0, "boost": self.boost}


@register
class Governor(Controller):
    """The A100's stock policy, as measured (data/governor_trace_dynamics_*.csv):

      parked    an idle chip sits at 1095 MHz
      wake      about 0.25 s into a load the clock jumps, once, to where the power read
                at 1095 MHz says the wall is: `wake_gain` MHz per watt of headroom.
                It does not visit the top clock on the way (the replays show none).
      idle      when the load stops the clock goes to the top within a reading or two
      park      2 s of idleness later it is back at 1095 MHz

    It has no opinion about domains: every domain gets the same request."""

    name = "governor"

    def __init__(self, wake_ms=210.0, park_ms=2000.0, f_idle=None, wake_gain=1.6):
        self.wake_ms, self.park_ms, self.f_idle = wake_ms, park_ms, f_idle
        self.wake_gain = wake_gain  # MHz per W, from the parked clock
        self.awake = self._pending = False

    def attach(self, sim):
        self.sim = sim
        self.f_idle = sim.hw.snap(self.f_idle or sim.hw.f_idle)
        sim.req = [self.f_idle] * sim.D
        sim.cap = self.f_idle  # a parked chip has no headroom yet

    def on_batch(self, batch):
        if not self.awake and not self._pending:
            self._pending = True
            self.sim.push(self.sim.t + self.wake_ms, "call", self._wake)

    def _wake(self):
        sim, lim = self.sim, self.sim.hw.limiter
        self._pending, self.awake = False, True
        if sim.use_limiter:  # one jump, from the parked clock
            p = sim.window_power(lim.period_ms)
            jump = lim.step_mhz * int(self.wake_gain * max(sim.hw.tdp_w - p, 0) / lim.step_mhz)
            sim.cap = min(sim.hw.f_max, max(sim.target) + jump)
        else:
            sim.cap = sim.hw.f_max
        sim.request(None, sim.hw.f_max)

    def on_idle(self):
        sim, t0 = self.sim, self.sim.t
        sim.push(t0 + sim.hw.limiter.period_ms, "call", lambda: self._release(t0))
        sim.push(t0 + self.park_ms, "call", lambda: self._park(t0))

    def _still_idle(self, t0):
        return self.sim.busy_since is None and self.sim.idle_since == t0

    def _release(self, t0):
        if self._still_idle(t0) and self.awake:
            self.sim.cap = self.sim.hw.f_max
            self.sim._send()

    def _park(self, t0):
        if self._still_idle(t0):
            self.awake = False
            self.sim.cap = self.f_idle
            self.sim.request(None, self.f_idle)

    def describe(self):
        return {
            "type": self.name,
            "wake_ms": self.wake_ms,
            "park_ms": self.park_ms,
            "f_idle": self.f_idle,
            "wake_gain": self.wake_gain,
        }


@register
class TdpBisection(Controller):
    """Open loop: for the coming batch, the clocks that finish soonest with predicted
    board power at or under TDP.

      outer   bisect the layer time L between "everything at the top clock" and slow
      inner   per domain, the slowest clock that still meets L (the other domain held),
              swept twice because the domains share DRAM and L2
      test    predicted power at those clocks <= TDP ?  shrink L : grow L

    With one domain this is a plain bisection on the clock. `reclaim=False` keeps a
    domain that is not the bottleneck at the bottleneck's clock instead of slowing it."""

    name = "tdp_bisection"

    def __init__(self, headroom_w=0.0, reclaim=True, when="batch"):
        self.headroom_w, self.reclaim, self.when = headroom_w, reclaim, when
        self.memo, self.decisions = {}, []

    def on_batch(self, batch):
        if self.when == "batch":
            self._apply(self._proxy(batch), batch.name)

    def on_layer(self, batch, layer):
        if self.when == "layer":
            self._apply([layer], f"{batch.name}/L{layer.index}")

    def _apply(self, layers, tag):
        key = tuple(x.key for x in layers)
        if key not in self.memo:
            self.memo[key] = self.solve(layers)
        f, ms, w = self.memo[key]
        self.decisions.append(dict(t_ms=self.sim.t, batch=tag, f=list(f), pred_ms=ms, pred_w=w))
        for d, x in enumerate(f):
            self.sim.request(d, x)

    def _slowest(self, layers, f, d, L):
        """Slowest clock of domain d that keeps the layers within L (others held)."""
        hw = self.sim.hw
        grid = hw.clocks()
        lo, hi = 0, grid.index(hw.snap(f[d]))  # hi is known to meet L
        while lo < hi:
            mid = (lo + hi) // 2
            g = list(f)
            g[d] = grid[mid]
            if self._cost(layers, g)[0] <= L * (1 + 1e-9):
                hi = mid
            else:
                lo = mid + 1
        return grid[hi]

    def _clocks_for(self, layers, L):
        hw, D = self.sim.hw, self.sim.D
        f = [hw.f_max] * D
        if self._cost(layers, f)[0] > L:
            return f  # L is unreachable: flat out
        for _ in range(2):
            for d in range(D):
                f[d] = self._slowest(layers, f, d, L)
        return f

    def solve(self, layers):
        hw, D = self.sim.hw, self.sim.D
        budget = hw.tdp_w - self.headroom_w
        top = [hw.f_max] * D
        ms_top, _, w_top = self._cost(layers, top)
        if D == 1:
            grid = hw.clocks()
            lo, hi = 0, len(grid) - 1  # highest clock with power <= budget
            if w_top > budget:
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    if self._cost(layers, [grid[mid]])[2] <= budget:
                        lo = mid
                    else:
                        hi = mid - 1
                f = [grid[lo]]
            else:
                f = top
        else:
            L_lo, L_hi = ms_top, self._cost(layers, [hw.f_min] * D)[0]
            f = self._clocks_for(layers, L_lo)
            if self._cost(layers, f)[2] > budget:
                for _ in range(24):
                    L = 0.5 * (L_lo + L_hi)
                    f = self._clocks_for(layers, L)
                    if self._cost(layers, f)[2] <= budget:
                        L_hi = L
                    else:
                        L_lo = L
                    if L_hi / L_lo - 1 < 2e-4:
                        break
                f = self._clocks_for(layers, L_hi)
            if not self.reclaim:
                f = [max(f)] * D
                while self._cost(layers, f)[2] > budget and f[0] > hw.f_min:
                    f = [x - hw.f_step for x in f]
        ms, _, w = self._cost(layers, f)
        return tuple(f), ms, w

    def describe(self):
        return {"type": self.name, "headroom_w": self.headroom_w, "reclaim": self.reclaim, "when": self.when}


@register
class ClosedLoop(Controller):
    """No model. Each tick: over TDP -> every domain one step down. Under TDP by more
    than `margin_w` -> the domain that finished LAST in the previous layer one step up.
    And a domain that finished early by more than `slack` of the layer one step down:
    its speed bought nothing."""

    name = "closed_loop"

    def __init__(self, margin_w=6.0, slack=0.06, f_start=None):
        self.margin_w, self.slack, self.f_start = margin_w, slack, f_start
        self.busy = None

    def attach(self, sim):
        self.sim = sim
        sim.req = [sim.hw.snap(self.f_start or sim.hw.f_max)] * sim.D

    def on_layer_end(self, rec):
        """How long each domain's partition was busy in the layer just finished."""
        sim = self.sim
        if sim.D < 2 or len(sim.parts) < 2:
            self.busy = None
            return
        fin = rec["finish"]
        busy = [0.0] * sim.D
        for g, t0 in (("g1", rec["t_start"]), ("g2", fin.get(("epilogue", "act"), rec["t_start"]))):
            for p in sim.parts:
                if (p.name, g) in fin:
                    busy[p.domain] += fin[(p.name, g)] - t0
        self.busy = busy

    def on_tick(self, p_window, p_now):
        sim, hw = self.sim, self.sim.hw
        step = hw.f_step
        req = list(sim.req)
        if p_window > hw.tdp_w:
            req = [x - step for x in req]
        else:
            if self.busy:
                slow = int(np.argmax(self.busy))
                for d in range(sim.D):
                    if d != slow and self.busy[d] < (1 - self.slack) * self.busy[slow]:
                        req[d] -= step
                if p_window < hw.tdp_w - self.margin_w:
                    req[slow] += step
            elif p_window < hw.tdp_w - self.margin_w:
                req = [x + step for x in req]
        for d, x in enumerate(req):
            if hw.snap(x) != sim.req[d]:
                sim.request(d, x)

    def describe(self):
        return {"type": self.name, "margin_w": self.margin_w, "slack": self.slack}


class _GridSearch(Controller):
    """Exhaustive search of the clock grid, shared by the two controllers below."""

    def __init__(self, step_mhz=30, f_lo=600):
        self.step_mhz, self.f_lo, self.memo, self.decisions = step_mhz, f_lo, {}, []

    def _grid(self):
        hw = self.sim.hw
        g = list(range(hw.f_max, self.f_lo - 1, -self.step_mhz))
        return [hw.snap(x) for x in g]

    def _points(self):
        g = self._grid()
        if self.sim.D == 1:
            return [[a] for a in g]
        return [[a, b] for a in g for b in g]

    def on_batch(self, batch):
        layers = self._proxy(batch)
        key = tuple(x.key for x in layers)
        if key not in self.memo:
            self.memo[key] = self.solve(layers, batch)
        f, ms, w = self.memo[key]
        self.decisions.append(dict(t_ms=self.sim.t, batch=batch.name, f=list(f), pred_ms=ms, pred_w=w))
        for d, x in enumerate(f):
            self.sim.request(d, x)

    def _best(self, layers, feasible, score):
        best = None
        for f in self._points():
            ms, mj, w = self._cost(layers, f)
            if feasible(ms, mj, w):
                s = score(ms, mj, w)
                if best is None or s < best[0]:
                    best = (s, tuple(f), ms, w)
        if best is None:  # nothing feasible: flat out
            f = [self.sim.hw.f_max] * self.sim.D
            ms, mj, w = self._cost(layers, f)
            return tuple(f), ms, w
        return best[1:]


@register
class OfflineOptimal(_GridSearch):
    """Full knowledge of the batch, every clock pair tried. objective 'latency': the
    fastest under TDP. objective 'energy': the least energy within the latency target."""

    name = "offline_optimal"

    def __init__(self, objective="latency", slo_ms=None, **kw):
        super().__init__(**kw)
        self.objective, self.slo_ms = objective, slo_ms

    def solve(self, layers, batch):
        tdp = self.sim.hw.tdp_w
        if self.objective == "latency":
            # ties in latency go to the lower energy
            return self._best(layers, lambda ms, mj, w: w <= tdp, lambda ms, mj, w: (round(ms, 6), mj))
        slo = (self.slo_ms or batch.slo_ms or float("inf")) * len(layers)
        return self._best(layers, lambda ms, mj, w: w <= tdp and ms <= slo, lambda ms, mj, w: mj)

    def describe(self):
        return {"type": self.name, "objective": self.objective, "slo_ms": self.slo_ms, "step_mhz": self.step_mhz}


@register
class MinEnergySLO(_GridSearch):
    """Least energy that meets the per-layer latency target, under TDP. With one domain
    this is the classic 'slowest clock that makes the deadline'; with two it also asks
    which domain to slow."""

    name = "min_energy_slo"

    def __init__(self, slo_ms=None, **kw):
        super().__init__(**kw)
        self.slo_ms = slo_ms

    def solve(self, layers, batch):
        tdp = self.sim.hw.tdp_w
        slo = self.slo_ms or batch.slo_ms
        if slo is None:
            raise ValueError("min_energy_slo needs slo_ms (controller or batch)")
        slo *= len(layers)
        return self._best(layers, lambda ms, mj, w: w <= tdp and ms <= slo, lambda ms, mj, w: mj)

    def describe(self):
        return {"type": self.name, "slo_ms": self.slo_ms, "step_mhz": self.step_mhz}
