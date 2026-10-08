"""How fast a partition runs: per-SM service time, slowed by what the SMs share.

THE MODEL

A partition on n SMs walks its tile list with a grid stride, so at any instant its SMs
cover a WINDOW of n consecutive tiles: n / tiles_n row-blocks (demand.Partition cuts the
tile list into those windows). For each window, every SM needs three things, and the
window takes as long as their sum:

  weight panels   one expert panel per row-block, delivered by L2      kP * panel bytes
  tensor work     the MACs themselves                                  kR * MACs
  DRAM waits      the first row-block of each expert finds its panel
                  missing; every SM working on that expert in this
                  window waits for it, not only the first              L0 * wait bytes

kP, kR and L0 are per-SM costs and scale (mostly) with 1/clock. Two things are shared
by every SM on the chip, whatever V/f domain it sits in, and both slow down as they
fill up:

  DRAM            rho_D = (all DRAM traffic) / BW_D          waits  x 1 / (1 - rho_D)
  L2 delivery     rho_L = (all L2 traffic) / BW_L            panels x 1 / (1 - rho_L)
                          + xi * rho_D ** q

The second term of rho_L is the coupling that co-running partitions exposed: a partition
that streams from DRAM slows its neighbour's L2 HITS, by more than the neighbour's own
DRAM share explains, and by more the closer DRAM is to full.

A window's own traffic depends on how long it takes, so each window is a one-dimensional
root find. This one mechanism gives both effects the measurements show: a partition
alone saturates as it is given more SMs (its own traffic fills DRAM), and two partitions
side by side slow each other (each sees the other's traffic). `solve_rates` is that
second case: a fixed point in which every partition sees the others' average streams.

WHY WINDOWS AND NOT ONE AVERAGE PER PASS. Congestion is convex. A whole layer on 108 SMs
alternates between windows that are all cold experts -- 108 SMs pulling distinct panels
from DRAM at once -- and windows inside one hot expert that touch DRAM not at all. One
average utilisation for the pass under-predicted those layers by 6-13%; per window the
bias is under 2%.

Units: time ms, sizes MB, work GMAC, rates MB/ms (= GB/s), clocks MHz.
"""

from bisect import bisect_right
from dataclasses import dataclass
from math import exp

import numpy as np

from .demand import Stage


@dataclass(frozen=True)
class Coef:
    """Per-SM costs at one clock, plus the shared capacities."""

    kP: float  # SM*ms per MB of weight panel delivered
    kR: float  # SM*ms per GMAC
    L0: float  # SM*ms per MB of DRAM wait, DRAM idle
    bw_d: float  # DRAM bandwidth, MB/ms
    bw_l: float  # L2 delivery capacity, MB/ms
    xi: float  # how hard a busy DRAM leans on L2 delivery
    q: float  # ... and how sharply that grows as DRAM fills


@dataclass
class RateParams:
    """Clock dependence of Coef. A cost k(f) = a / f + b: `a` is cycles, `b` is the part
    that does not speed up with the SM clock.

    The two shared capacities move with the clock as powers of it, and the exponents are
    fitted (calibrate.FORMS says which parameters each model form leaves free, and
    `sim.calibrate --compare` scores the forms on sweeps they were not fitted on):

      bw_l   ~ f ** gamma_l, gamma_l = 0.44: the L2 serves faster at a higher clock, but
             much less than in proportion.
      xi     ~ f ** gamma_x, gamma_x = 1.49. xi is written relative to a per-SM cost kP
             that shrinks as 1/f, so gamma_x = 1 would be a delay that no clock changes;
             what a busy DRAM adds to a delivery grows, in absolute time, about as
             sqrt(f).
      q      1.72: the coupling grows faster than DRAM utilisation.
      bw_d   no clock: HBM runs on its own. Pinned at the A100-40GB's 1555 GB/s; left
             free together with q it runs off to 25-34 TB/s, because the two trade.

    WHICH CLOCK, once there are two. On today's chip there is one and the question does
    not arise; the fit only says how each term moves with THE clock. With V/f domains:

      kP, kR, L0   the partition's own domain clock: they are per-SM costs. Whether
                   ALL of kP and L0 is the SM's own is not knowable from one domain --
                   a panel is handed over by the shared logic -- so `uncore_share` of
                   their clock-dependent part can be put on the uncore clock instead
                   (hardware.delivery_uncore_share).
      bw_l         the UNCORE clock (whatever the hardware description says the shared
                   logic follows): it is the L2's own service rate.
      xi           the partition's own domain clock, or with `xi_uncore` the uncore
                   clock (hardware.xi_clock): both ends of that coupling are shared
                   logic, and one clock cannot tell the two readings apart.
      bw_d         none.

    `sim.sensitivity` sweeps every one of these choices."""

    aP: float
    bP: float
    aR: float
    bR: float
    aL: float
    bL: float
    bw_d: float
    bw_l: float  # at f_ref
    gamma_l: float  # bw_l ~ (uncore clock / f_ref) ** gamma_l
    xi: float  # at f_ref
    gamma_x: float  # xi   ~ (domain clock / f_ref) ** gamma_x
    q: float
    f_ref: float = 1290.0

    def at(self, f, f_uncore=None, uncore_share=0.0, xi_uncore=False):
        fu = f if f_uncore is None else f_uncore
        if uncore_share and fu != f:
            inv = (1.0 - uncore_share) / f + uncore_share / fu  # 1 / the clock delivery follows
            kP, L0 = self.aP * inv + self.bP, self.aL * inv + self.bL
        else:
            kP, L0 = self.aP / f + self.bP, self.aL / f + self.bL
        return Coef(
            kP=kP,
            kR=self.aR / f + self.bR,
            L0=L0,
            bw_d=self.bw_d,
            bw_l=self.bw_l * (fu / self.f_ref) ** self.gamma_l,
            xi=self.xi * ((fu if xi_uncore else f) / self.f_ref) ** self.gamma_x,
            q=self.q,
        )


_CAP = 1.0 - 1e-6  # a shared resource is never quite full


def window_times(panel, gmac, dram, wait, n, kP, kR, L0, bw_d, bw_l, xi, q, ext_d=0.0, ext_l=0.0, iters=200):
    """ms per window. Every argument broadcasts, so one call can solve the windows of
    one stage (the engine) or of a thousand measurements at different clocks (the fit).

    need(t) is the SM time the window asks for if it takes t; it falls as t grows
    because a slower window streams less. The window takes the t where need(t) = t.
    Newton from below, kept inside a bracket: need(t) - t is convex and decreasing, so
    the iterates climb to the root without overshooting, and the bracket catches the
    few windows where the saturation clip breaks convexity.

    Every window is iterated until it has CONVERGED. Next to a saturated DRAM need(t) is
    nearly a pole and Newton creeps: such a window needs 30-40 iterations, where an
    unsaturated one needs five. Stopping all of them at a fixed small count left the
    saturated ones a percent off and, worse, made a co-running partner's time a jagged
    function of this one's -- the fixed point between them then had no single answer.
    Windows drop out as they converge, so the stragglers cost little."""
    args = np.broadcast_arrays(
        *(np.asarray(v, float) for v in (panel, gmac, dram, wait, n, kP, kR, L0, bw_d, bw_l, xi, q, ext_d, ext_l))
    )
    shape = args[0].shape
    panel, gmac, dram, wait, n, kP, kR, L0, bw_d, bw_l, xi, q, ext_d, ext_l = (v.ravel() for v in args)
    base_d = ext_d / bw_d
    base_l = ext_l / bw_l

    def h(t, i):
        """need(t) - t and its slope, for the windows i."""
        ud = dram[i] / (t * bw_d[i])  # the window's own utilisation
        ul = panel[i] / (t * bw_l[i])
        rd_raw = base_d[i] + ud
        rd = np.clip(rd_raw, 1e-12, _CAP)
        rl_raw = base_l[i] + ul + xi[i] * rd ** q[i]
        rl = np.minimum(rl_raw, _CAP)
        need = (kP[i] * panel[i] / (1 - rl) + kR[i] * gmac[i] + L0[i] * wait[i] / (1 - rd)) / n[i]
        drd = np.where(rd_raw < _CAP, -ud / t, 0.0)
        drl = np.where(rl_raw < _CAP, -ul / t + xi[i] * q[i] * rd ** (q[i] - 1) * drd, 0.0)
        dneed = (kP[i] * panel[i] * drl / (1 - rl) ** 2 + L0[i] * wait[i] * drd / (1 - rd) ** 2) / n[i]
        return need - t, dneed - 1.0

    every = np.arange(panel.size)
    rd0 = np.clip(base_d, 1e-12, _CAP)
    rl0 = np.minimum(base_l + xi * rd0**q, _CAP)
    # the window with nothing of its own in the shared resources: a lower bound
    lo = (kP * panel / (1 - rl0) + kR * gmac + L0 * wait / (1 - rd0)) / n + 1e-12
    hi = 2.0 * lo
    for _ in range(60):  # widen until need(hi) < hi
        over = h(hi, every)[0] > 0
        if not over.any():
            break
        lo = np.where(over, hi, lo)
        hi = np.where(over, 2.0 * hi, hi)
    t = lo.copy()
    live = every  # the windows still iterating
    for _ in range(iters):
        x = t[live]
        f, df = h(x, live)
        lo[live] = np.where(f > 0, x, lo[live])
        hi[live] = np.where(f > 0, hi[live], x)
        tn = x - f / df
        bad = ~np.isfinite(tn) | (tn <= lo[live]) | (tn >= hi[live])
        tn = np.where(bad, 0.5 * (lo[live] + hi[live]), tn)
        t[live] = tn
        live = live[np.abs(tn - x) > 1e-12 * tn]
        if not live.size:
            break
    return t.reshape(shape)


def stage_time(st: Stage, c: Coef, ext_d=0.0, ext_l=0.0):
    """ms for the stage on its own SMs while the rest of the chip streams ext_d MB/ms
    from DRAM and ext_l MB/ms through L2."""
    return float(
        st.cnt
        @ window_times(
            st.panel, st.gmac, st.dram, st.wait, st.n, c.kP, c.kR, c.L0, c.bw_d, c.bw_l, c.xi, c.q, ext_d, ext_l
        )
    )


# What the rest of the chip can be streaming, as utilisations of DRAM and of L2 delivery.
# Dense where congestion bends the curve.
_UD = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.78, 0.85, 0.9, 0.94, 0.97]
_UL = [0, 0.04, 0.08, 0.14, 0.22, 0.32]


class Job:
    """One GEMM stage of one partition on its SMs at one clock.

    `time(ext_d, ext_l)` is how long the whole stage takes while the rest of the chip
    streams ext_d MB/ms from DRAM and ext_l MB/ms through L2. The engine asks that at
    every event and controllers ask it thousands of times, so it is tabulated once per
    (stage, clock) -- one vectorised solve of all 78 grid points -- and interpolated in
    ln T. `exact=True` skips the table; `sim.validate` reports the difference."""

    def __init__(self, st: Stage, c: Coef, exact=False):
        self.st, self.c, self.exact = st, c, exact
        self.t = self.s_d = self.s_l = 0.0
        if not exact:
            ud, ul = np.meshgrid(np.array(_UD), np.array(_UL), indexing="ij")
            t = window_times(
                st.panel,
                st.gmac,
                st.dram,
                st.wait,
                st.n,
                c.kP,
                c.kR,
                c.L0,
                c.bw_d,
                c.bw_l,
                c.xi,
                c.q,
                (ud * c.bw_d)[..., None],
                (ul * c.bw_l)[..., None],
            )
            self._ln = np.log(t @ st.cnt).tolist()

    def time(self, ext_d=0.0, ext_l=0.0):
        if self.exact:
            return stage_time(self.st, self.c, ext_d, ext_l)
        i, a = _cell(_UD, ext_d / self.c.bw_d)
        j, b = _cell(_UL, ext_l / self.c.bw_l)
        L = self._ln
        v = (1 - a) * ((1 - b) * L[i][j] + b * L[i][j + 1]) + a * ((1 - b) * L[i + 1][j] + b * L[i + 1][j + 1])
        return exp(v)


def _cell(grid, u):
    """Index of the grid cell holding u, and where in it; clamped to the grid."""
    if u <= grid[0]:
        return 0, 0.0
    if u >= grid[-1]:
        return len(grid) - 2, 1.0
    i = bisect_right(grid, u) - 1
    return i, (u - grid[i]) / (grid[i + 1] - grid[i])


RELAX = 0.65  # each job slows as its neighbour speeds up: under-relax


def solve_rates(jobs, tol=1e-7, max_iter=80):
    """Fixed point: every job sees the others' average streams. Fills t (ms for its
    whole stage under the current conditions), s_d and s_l on each job. One job needs
    no iteration."""
    for j in jobs:
        j.t = j.time()
        j.s_d, j.s_l = j.st.tot_dram / j.t, j.st.tot_panel / j.t
    if len(jobs) < 2:
        return jobs
    for _ in range(max_iter):
        worst = 0.0
        tot_d, tot_l = sum(j.s_d for j in jobs), sum(j.s_l for j in jobs)
        new = []
        for j in jobs:
            t = j.time(tot_d - j.s_d, tot_l - j.s_l)
            worst = max(worst, abs(t / j.t - 1))
            new.append(j.t + RELAX * (t - j.t))
        for j, t in zip(jobs, new, strict=True):
            j.t = t
            j.s_d, j.s_l = j.st.tot_dram / t, j.st.tot_panel / t
        if worst < tol:
            break
    return jobs
