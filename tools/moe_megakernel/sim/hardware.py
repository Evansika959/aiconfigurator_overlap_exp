"""The machine being simulated: SMs, V/f domains, the power limit, and how fast a clock
set-point takes effect.

Two different things are called "partition" in this project and the hardware description
keeps them apart:

  SM set      where a group of experts RUNS. A persistent kernel can pin work to any
              set of SMs today, on the real A100.
  V/f domain  which SMs SHARE A CLOCK AND A VOLTAGE. The real A100 has one. Everything
              with two or more is a design being evaluated.

So "pinned partitions at one clock" is two SM sets in one domain, and "two V/f domains"
is two SM sets in two domains. Giving both domains the same clock must reproduce the
first exactly; `sim.validate` checks that it does.

What cannot be measured on a single-domain chip is a parameter here, not an assumption
buried in the model: which clock the shared logic (L2, crossbar) follows, what crossing
between clock domains costs, how the second rail is regulated, and how long a set-point
takes to arrive.
"""

import warnings
from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class Domain:
    name: str
    sms: int
    f_init: Optional[int] = None  # MHz; None = the hardware's default clock


@dataclass
class Limiter:
    """The board's own power protection. It is part of the hardware, so EVERY
    configuration runs under the same one -- a policy must not look good because its
    baseline was simulated without the headroom the limiter allows.

    Every `loop_ms` it reads the mean board power over that long and keeps a clock
    CEILING; a controller's request above the ceiling is clipped to it, and a change of
    the ceiling reaches the chip `latency_ms` later.

      reading above k_peak * tdp      the ceiling drops by gain * excess
      reading below tdp by as much    the ceiling rises by gain * headroom
      otherwise                       the excess (reading - tdp) is accumulated as
                                      energy; every `budget_j` of it, one step down,
                                      and every `budget_j` of deficit, one step up

    The gain is partial -- about half of what would cancel the excess at once at these
    clocks -- so a load step is absorbed over three loops, and the board is above
    k_peak for about 60 ms. Inside the band it may sit above TDP for budget / excess
    seconds: 4 J is 0.4 s at 410 W. Those two time constants are what make short bursts
    cheap, and they apply to a single clock exactly as they do to two.

    Set against the A100 with the latency of every graph replay as the clock's witness
    (data/governor_trace_dynamics_replays_qwen.csv, 9-15 ms per replay; NVML refreshes
    every 100 ms and cannot see this). When a 4096-token or a prefill batch replaced a
    1024-token one at 1410 MHz, the clock began to fall 10-30 ms later and had dropped
    65-135 MHz 45-55 ms after that, landing within 15 MHz of where it then stayed.
    After that it walks 15 MHz at a time around the wall every 0.3-0.7 s, 4-8 W either
    side of 400. `sim.validate_trace` scores it on traces it was not set against."""

    loop_ms: float = 20.0  # how often it looks
    latency_ms: float = 10.0  # its decision -> the clock changes
    k_peak: float = 1.05
    gain_mhz_per_w: float = 0.45
    budget_j: float = 4.0
    step_mhz: int = 15
    # How often a reading is REPORTED (NVML's cadence on the A100): what controllers and
    # `summary` see.
    period_ms: float = 100.0
    window_ms: float = 1000.0  # reporting only: the window `summary` quotes
    enabled: bool = True


@dataclass
class Hardware:
    name: str = "a100-sxm4-40gb"
    sms: int = 108
    gpc_sms: int = 16  # a V/f domain is a whole number of these
    f_min: int = 210
    f_max: int = 1410
    f_step: int = 15
    f_idle: int = 1095  # where the stock governor parks an idle GPU
    tdp_w: float = 400.0
    domains: list[Domain] = field(default_factory=lambda: [Domain("all", 108)])
    # ---- what a single-domain chip cannot tell us ---------------------------------
    # Which clock the shared logic follows: max | min | <domain name>.
    uncore: str = "max"
    # Extra L2-delivery cost for a domain whose clock differs from the uncore's
    # (a clock-domain crossing).
    cdc_penalty: float = 0.0
    # 'ldo': the lower rail is derived from the higher one, so its saving is linear in V,
    # not quadratic.
    cold_regulator: str = "buck"
    # Share of the per-SM panel-delivery and DRAM-wait cost (kP, L0) that follows the
    # shared logic's clock instead of the SM's own. The fit cannot say: with one clock
    # the two are the same. 0 is "an SM's cost follows its own clock".
    delivery_uncore_share: float = 0.0
    # Which clock the DRAM-to-L2 coupling (`RateParams.xi`) follows: 'own', the
    # partition's domain clock, or 'uncore', the shared logic's. Both ends of that
    # coupling are shared logic; the fit only saw one clock.
    xi_clock: str = "own"
    # The extra power of a loaded die beyond its activity (`PowerParams.act`): 'share'
    # charges each domain's SM share at its own clock, 'max' all of it at the faster
    # domain's clock.
    active_power: str = "share"
    # The clock the epilogue (activation, top-k sum) runs at, on every SM: 'mean', the
    # SM-weighted mean, as if work moved to the faster domain; 'min', the slower
    # domain's, as if every SM had the same share.
    epilogue_clock: str = "mean"
    # Set-point -> effect. 38 ms is NVML on the A100; an on-die regulator would be
    # 0.01-0.1 ms.
    actuation_ms: float = 38.0
    limiter: Limiter = field(default_factory=Limiter)

    def __post_init__(self):
        self.domains = [d if isinstance(d, Domain) else Domain(**d) for d in self.domains]
        if not isinstance(self.limiter, Limiter):
            self.limiter = Limiter(**self.limiter)
        tot = sum(d.sms for d in self.domains)
        if tot != self.sms:
            raise ValueError(f"domains hold {tot} SMs, the chip has {self.sms}")
        names = [d.name for d in self.domains]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate domain names: {names}")
        if self.uncore not in ("max", "min") and self.uncore not in names:
            raise ValueError(f"uncore must be max, min or a domain name, not {self.uncore}")
        if self.active_power not in ("share", "max"):
            raise ValueError(f"active_power must be share or max, not {self.active_power}")
        if self.epilogue_clock not in ("mean", "min"):
            raise ValueError(f"epilogue_clock must be mean or min, not {self.epilogue_clock}")
        if not 0.0 <= self.delivery_uncore_share <= 1.0:
            raise ValueError("delivery_uncore_share is a share: 0 to 1")
        if self.xi_clock not in ("own", "uncore"):
            raise ValueError(f"xi_clock must be own or uncore, not {self.xi_clock}")
        if len(self.domains) > 1 and not self.gpc_aligned():
            warnings.warn(
                f"domains of {[d.sms for d in self.domains]} SMs are not whole GPCs "
                f"of {self.gpc_sms}: fine as a what-if, not as a chip",
                stacklevel=2,
            )

    def clocks(self):
        return list(range(self.f_min, self.f_max + 1, self.f_step))

    def snap(self, f):
        """Nearest clock the hardware can actually run."""
        f = min(max(f, self.f_min), self.f_max)
        return int(self.f_min + round((f - self.f_min) / self.f_step) * self.f_step)

    def index(self, name):
        for i, d in enumerate(self.domains):
            if d.name == name:
                return i
        raise KeyError(name)

    def epilogue_clock_of(self, f):
        """The clock the epilogue runs at; f: one clock per domain."""
        if self.epilogue_clock == "min":
            return min(f)
        return sum(d.sms * x for d, x in zip(self.domains, f, strict=True)) / self.sms

    def uncore_clock(self, f):
        """f: one clock per domain."""
        if self.uncore == "max":
            return max(f)
        if self.uncore == "min":
            return min(f)
        return f[self.index(self.uncore)]

    def gpc_aligned(self):
        """Domain sizes the hardware could really have: whole GPCs, the last GPC taking
        whatever is left of the SM count."""
        return all(d.sms % self.gpc_sms == 0 for d in self.domains[:-1])

    def to_dict(self):
        return asdict(self)

    def sig(self):
        """Everything a steady-state prediction depends on, as a cache key. (Not id():
        a search creates and drops hundreds of Hardware objects and ids get reused.)"""
        return (
            tuple((d.name, d.sms) for d in self.domains),
            self.sms,
            self.uncore,
            self.cdc_penalty,
            self.cold_regulator,
            self.delivery_uncore_share,
            self.xi_clock,
            self.active_power,
            self.epilogue_clock,
            self.f_min,
            self.f_max,
            self.f_step,
        )


def single_domain(**kw):
    return Hardware(**kw)


def two_domain(cold_sms, **kw):
    """`cold_sms` SMs in a 'cold' domain, the rest in 'hot'."""
    sms = kw.get("sms", 108)
    return Hardware(domains=[Domain("cold", cold_sms), Domain("hot", sms - cold_sms)], **kw)
