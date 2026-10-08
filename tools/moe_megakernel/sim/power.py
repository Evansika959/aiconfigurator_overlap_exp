"""Board power from ACTIVITY, not from configuration.

A static model stores "this configuration at this clock draws P watts". That cannot say
what two partitions draw side by side, or what a partition draws while its neighbour's
traffic slows it down, because those configurations were never measured -- and with two
V/f domains they cannot be. So the independent variable is changed: power is a sum over
what the hardware is DOING per second.

  P = static(clocks)
    + act(clock)                                    whenever anything runs
    + sum over running jobs of
        g(f_d) * (e_panel * panel bytes/s                       SM rail, scales with V^2
                  + e_mac * real MACs/s + e_pad * padding MACs/s)
      + e_dram * DRAM bytes/s                                   memory side, does not

Padding is priced apart from real work. A row-block is BLOCK_M rows whether or not the
expert has that many tokens; the missing rows are loaded as zeros and multiplied like
any others, so they take the same TIME -- but a multiplier fed zeros barely toggles.
With one price per MAC the model over-predicted a 1024-token batch on 128-row tiles
(56% padding) by 13% while matching the 7744-token batch (12% padding).

g(f) is the square of the supply voltage relative to its floor: 1 up to the knee near
1050 MHz, rising above it to 1.8 at 1410 MHz. It is fitted, not assumed, as one number
per measured clock, shared by every partition and both kernels.

The energies came out as the two-rail picture says they should: e_dram is the same at
every clock (78 pJ per byte from 510 to 1410 MHz, refitted clock by clock), while
e_panel, e_mac and e_pad rise with g.

STATIC POWER WITH DOMAINS. The idle board draws P_idle(f) at one clock. The part of it
that changes with f belongs to the clocked logic. With domains, each domain's SMs carry
their share at their own clock, and the shared logic carries `uncore_share` of it at the
uncore clock:

  static = P_shared + (1 - u) * sum_d (n_d / N) * (P_idle(f_d) - P_shared)
                    + u * (P_idle(f_uncore) - P_shared)

With one domain this is exactly the measured P_idle(f), whatever u is.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class PowerParams:
    clocks: list[float]  # MHz, ascending: the grid every table below is on
    g: list[float]  # V^2 relative to the floor
    idle: list[float]  # W, board, nothing running, die at ~38 C
    # W a busy chip draws beyond its activity: mostly a warmer die (the sweeps ran at
    # 50-66 C).
    act: list[float]
    e_panel: float  # J per GB of weight panel delivered, at the floor
    e_mac: float  # J per TMAC on real pairs, at the floor
    e_pad: float  # J per TMAC on padding rows, at the floor
    e_dram: float  # J per GB of DRAM traffic
    p_shared: float  # W of the idle power no clock changes
    uncore_share: float = 0.25  # share of the clock-dependent idle power that is not SMs

    def __post_init__(self):
        self._f = np.asarray(self.clocks, float)
        self._g = np.asarray(self.g, float)
        self._idle = np.asarray(self.idle, float)
        self._act = np.asarray(self.act, float)

    def g_at(self, f):
        return float(np.interp(f, self._f, self._g))

    def idle_at(self, f):
        return float(np.interp(f, self._f, self._idle))

    def act_at(self, f):
        return float(np.interp(f, self._f, self._act))

    def static(self, hw, f):
        """W with nothing running; f: one clock per domain."""
        ps, u = self.p_shared, self.uncore_share
        sm = sum(d.sms / hw.sms * (self.idle_at(fd) - ps) for d, fd in zip(hw.domains, f, strict=True))
        return ps + (1 - u) * sm + u * (self.idle_at(hw.uncore_clock(f)) - ps)

    def active(self, hw, f):
        """W a busy chip draws beyond its activity. Fitted as one number per clock that
        did NOT depend on how many SMs the running partition had, so with domains it is
        charged whole, each domain's share at its own clock (or, hardware.active_power =
        'max', all of it at the faster domain's clock: one domain cannot tell which).

        Most of it is leakage at the loaded die temperature: the sweeps ran at 50-66 C,
        the idle table at 38 C. The simulator has no thermal state, so it charges the
        full amount from the first millisecond of a load; the A100 takes 2-5 s to get
        there after an idle gap and draws 10-20 W less meanwhile
        (data/governor_trace_dynamics_*.csv). A single time constant was tried and made
        the traces worse: the die warms and cools with two."""
        if hw.active_power == "max":
            return self.act_at(max(f))
        return sum(d.sms / hw.sms * self.act_at(fd) for d, fd in zip(hw.domains, f, strict=True))

    def rail(self, hw, f, i):
        """g of domain i as the BOARD sees it. An LDO that derives the lower rail from
        the higher one burns the difference: the saving is linear in V, not quadratic."""
        gi = self.g_at(f[i])
        if hw.cold_regulator == "ldo" and len(f) > 1:
            return float(np.sqrt(gi * max(self.g_at(x) for x in f)))
        return gi

    def stage_mj(self, g, st):
        """Dynamic mJ to run one stage once on a rail at g."""
        return (
            g * (self.e_panel * st.tot_panel + self.e_mac * st.gmac_real + self.e_pad * st.gmac_pad)
            + self.e_dram * st.tot_dram
        )
