"""Kernel profiles: what a kernel needs from the chip, written once per kernel.

The MoE engine (`engine.py`) knows one kind of work, a layer of expert GEMMs run by our
own persistent kernel. Everything else that runs on a GPU -- a cuBLAS GEMM, an NCCL
collective -- is described here by a PROFILE, and `node.py` schedules profiles on SMs
without knowing which kernel they belong to. A profile answers four questions:

  occupancy   how many SMs it holds, and how many it denies to a kernel of another
              shared-memory class
  work        how much there is to do: tiles of MACs, or bytes
  rate        how fast that work goes at a clock, alone and beside another kernel
  activity    what it does per second that the board pays for

The chip's side of the bargain -- idle power, the voltage curve, energy per MAC and per
byte, the limiter -- is NOT in a profile: it is `calib.Calib.power` and
`hardware.Hardware`, fitted once on the MoE kernel and used here as they are.

Two profiles exist, both fitted by `fit_kernels.py` and stored in
`calib/a100_kernels.json`:

  CublasGemm     bf16 GEMM as torch.nn.functional.linear issues it: one thread block per
                 256x128 output tile, 147456 B of shared memory, so one block per SM and
                 the grid runs in waves.
  NcclAllReduce  ring all-reduce over NVLink, world of 2-4 GPUs: c CTAs resident for the
                 whole call, latency + bytes / bandwidth.
"""

import json
import os
from dataclasses import asdict, dataclass, field

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = os.path.join(HERE, "calib", "a100_kernels.json")
TILE_M, TILE_N = 256, 128


@dataclass
class CublasGemm:
    """ms of one wave = (tile_gmac * a + c_tile) / f, times 1 + split_cost if cuBLAS split
    K; a launch costs u / f on the GPU.

    a is per GMAC of ONE tile on ONE SM, so SM count enters only through the number of
    waves. c_tile is what a tile costs whatever its K (the block's start and its panel
    set-up): without it the model ran 2% slow on 2-wave GEMMs and 2% fast on 10-wave
    ones. `grids` holds launch grids read by ncu (cuBLAS splits K for some shapes and
    nothing in the shape says when); a shape that is not in it gets
    ceil(M/256) * ceil(N/128)."""

    a: float  # SM * ms * MHz per GMAC
    u: float  # ms * MHz: launch and teardown on the GPU
    host_pre: float  # ms from the host issuing the call to the kernel being queued
    host: float  # ms the host thread is busy issuing it
    c_tile: float = 0.0  # ms * MHz per tile, whatever its K
    split_cost: float = 0.0  # share by which a split-K tile is slower per MAC
    corun_ms: float = 0.0  # extra launch ms when another kernel is resident
    smem: int = 147456  # B of shared memory per block (98304 dynamic + 49152 static): the 164 KB class
    tile_m: int = TILE_M  # which side of the tile runs along M (for weight bytes)
    grids: dict = field(default_factory=dict)

    def grid(self, m, n, k):
        g = self.grids.get(f"{m}x{n}x{k}")
        return int(g) if g else -(-m // TILE_M) * -(-n // TILE_N)

    def split(self, m, n, k):
        """How many thread blocks share one output tile (a split along K)."""
        return max(1, round(self.grid(m, n, k) / (-(-m // TILE_M) * -(-n // TILE_N))))

    def tile_gmac(self, m, n, k):
        """GMAC per thread block: a split along K divides it."""
        return TILE_M * TILE_N * k / self.split(m, n, k) / 1e9

    def tile_ms(self, m, n, k, f):
        per = self.tile_gmac(m, n, k) * self.a + self.c_tile
        return per * (1.0 + (self.split_cost if self.split(m, n, k) > 1 else 0.0)) / f

    def launch_ms(self, f):
        return self.u / f

    def solo_ms(self, m, n, k, f, sms=108):
        """Kernel time alone on `sms` SMs (without the host's share)."""
        return -(-self.grid(m, n, k) // sms) * self.tile_ms(m, n, k, f) + self.launch_ms(f)

    def tile_mj(self, pw, g, m, n, k):
        """Dynamic mJ of one tile on a rail at g, as (SM rail, memory side): MACs and
        the weight panel it reads through L2, which scale with the voltage, and its
        share of the three matrices passing through DRAM once, which does not."""
        gmac = self.tile_gmac(m, n, k)
        weights = gmac * 1e3 * 2 / self.tile_m  # MB
        dram = (m * k + k * n + m * n) * 2 / 1e6 / self.grid(m, n, k)  # MB
        return g * (pw.e_mac * gmac + pw.e_panel * weights), pw.e_dram * dram


@dataclass
class NcclAllReduce:
    """ms = alpha(c, f) + GB / B(c, f), both tables on a (CTAs x clock) grid.

    B between grid points: along CTAs through a knee law fitted per clock,
    B = ((c / k) ** -p + cap ** -p) ** (-1 / p), with the measured-to-law ratio
    interpolated in log c; along the clock linearly. Leave-one-out on the table: 2.2%
    rms along CTAs, 2.5% along the clock (plain log-log interpolation: 3.5%, 2.7%).

    `dyn_w` is board W of the CTAs at work, per (CTAs, clock): what a GPU moving bytes
    alone draws, less its idle power, the busy-chip term and `link_w`. It did not depend
    on message size (64 against 512 MiB: within 4 W).

    `sms_per_cta`: SMs a kernel of the 164 KB shared-memory class loses per CTA when
    the collective was launched on an idle GPU (see `behind` for the other order)."""

    ctas: list
    clocks: list
    alpha: list  # ms, [clock][cta]
    bw: list  # GB/s of message, [clock][cta]
    dyn_w: list  # W, [clock][cta]
    law: list  # per clock: [k, cap, p]
    world: int = 4
    host: float = 0.11  # ms the host thread is busy issuing the call
    host_pre: float = 0.05  # ms from the call to the kernel being queued
    sms_per_cta: float = 2.0
    sms_lost_max: int = 104  # with every SM pair taken (54 CTAs and more) the probe still reached 4-6 SMs
    corun: dict = field(default_factory=dict)  # beside a GEMM: see slowdown() and behind()
    # Among GPUs at different clocks: the collective's time per byte is (1 - mix) of the
    # slowest participant's and mix of the participants' mean. 0 = the slowest sets it.
    mix: float = 0.0
    # Share of its CTAs' power a resident kernel draws while no bytes move (waiting for a
    # participant): W = dyn_w * (spin + (1 - spin) * rate / its own rate at that clock).
    spin: float = 1.0
    # W a GPU draws, at any clock, from its first collective until `link_ms` after its
    # last one, whether or not a kernel is resident: between collectives 6-15 ms apart
    # the board never returns to its idle power, while between GEMMs it does, and 0.5 s
    # after the last collective it has (`gaps`, `waiting`, `link_idle`). Read as the
    # NVLink leaving its low-power state; `dyn_w` is net of it.
    link_w: float = 0.0
    link_ms: float = 100.0  # somewhere between 15 and 500 ms: the data do not say where
    world_scale: dict = field(default_factory=dict)  # str(world) -> factor on B and alpha

    def __post_init__(self):
        self._c, self._f = np.asarray(self.ctas, float), np.asarray(self.clocks, float)
        self._a, self._b, self._w = (np.asarray(x, float) for x in (self.alpha, self.bw, self.dyn_w))
        self._law = np.asarray(self.law, float)

    @staticmethod
    def knee(c, k, cap, p):
        return ((c / k) ** -p + cap**-p) ** (-1.0 / p)

    def _along_c(self, row, i, c, law):
        """Row i of a table at c CTAs; `law`: interpolate through the knee law (the
        bandwidth) instead of linearly in log c."""
        cs = self._c
        if c <= cs[0]:  # below the table: bandwidth in proportion to the CTAs, the rest as at its edge
            return row[0] * c / cs[0] if law else row[0]
        j = min(int(np.searchsorted(cs, c, side="right")) - 1, len(cs) - 2)
        t = np.log(c / cs[j]) / np.log(cs[j + 1] / cs[j])
        if law:
            k, cap, p = self._law[i]
            r0, r1 = (row[x] / self.knee(cs[x], k, cap, p) for x in (j, j + 1))
            return self.knee(c, k, cap, p) * np.exp(np.log(r0) * (1 - t) + np.log(r1) * t)
        return row[j] * (1 - t) + row[j + 1] * t

    def _at(self, tab, c, f, law=False):
        fs = self._f
        f = min(max(f, fs[0]), fs[-1])
        i = min(int(np.searchsorted(fs, f, side="right")) - 1, len(fs) - 2)
        t = (f - fs[i]) / (fs[i + 1] - fs[i])
        return float(self._along_c(tab[i], i, c, law) * (1 - t) + self._along_c(tab[i + 1], i + 1, c, law) * t)

    def _scale(self, world):
        return self.world_scale.get(str(world or self.world), [1.0, 1.0])

    def bandwidth(self, c, f, world=None):
        """GB/s of message, alone."""
        return self._at(self._b, c, f, law=True) * self._scale(world)[0]

    def bandwidth_split(self, c, f_sm, f_shared, world=None):
        """GB/s when the CTAs' SMs and the shared logic are at DIFFERENT clocks: a chip
        that does not exist. The knee law says which part belongs to whom: the per-CTA
        cost k follows the SMs' clock, the ceiling `cap` the shared logic's. With both
        clocks equal this is `bandwidth` exactly."""
        if f_sm == f_shared:
            return self.bandwidth(c, f_sm, world)
        fs = self._f
        k = float(np.interp(f_sm, fs, self._law[:, 0]))
        cap = float(np.interp(f_shared, fs, self._law[:, 1]))
        p = float(self._law[0, 2])
        k_sm, cap_sm = float(np.interp(f_sm, fs, self._law[:, 0])), float(np.interp(f_sm, fs, self._law[:, 1]))
        ratio = self._at(self._b, c, f_sm, law=True) / self.knee(
            c, k_sm, cap_sm, p
        )  # table over law, at the SMs' clock
        return self.knee(c, k, cap, p) * ratio * self._scale(world)[0]

    def latency_ms(self, c, f, world=None):
        return self._at(self._a, c, f) * self._scale(world)[1]

    def power_w(self, c, f):
        return self._at(self._w, c, f)

    def solo_ms(self, mib, c, f, world=None):
        return self.latency_ms(c, f, world) + mib * 2**20 / 1e9 / self.bandwidth(c, f, world) * 1e3

    def slowdown(self, c, f):
        """Factor on its rate while GEMM tiles are in flight on the same GPU:
        1 / (1 + s), s = kappa * (f / 1200) ** x * (2 / c) ** y. Fitted on spans where
        the GEMM ends before the collective (0.3% median of the span). Largest with few
        CTAs, where each CTA's copy loop sets the rate, and gone by 32."""
        q = self.corun
        if not q.get("kappa"):
            return 1.0
        return 1.0 / (1.0 + q["kappa"] * (f / 1200.0) ** q.get("x", 0.0) * (2.0 / c) ** q.get("y", 0.0))

    def behind(self, c, f):
        """(SMs a GEMM loses per CTA, factor on the collective's rate) when its kernel
        arrives on a GPU whose SMs are running GEMM tiles.

        The shared-memory carveout belongs to a pair of SMs and holds while a block is
        resident. NCCL's block needs 89 KB and runs under a 100 KB carveout; the cuBLAS
        block needs 144 KB under 164 KB (ncu, `data/sim_overlap/ncu_*.csv`).

          collective first   its pair is in the 100 KB class, a cuBLAS block fits on
                             neither SM: the GEMM loses 2 SMs per CTA
          GEMM first         every pair is already in the 164 KB class; the CTA takes
                             one SM and its partner stays usable: the GEMM loses 1 --
                             but the collective is slower for its WHOLE length, x1.1-1.2
                             at 2 CTAs, x1.3-1.5 at 16-32, whatever the GEMM's length

        (An SM at 164 KB of shared memory has 28 KB of L1 left where NCCL normally has
        92. That this is what slows it is an inference.) The factor is a measured table
        over CTAs x clock. Probe: data/sim_overlap/nccl_occupancy.csv: lost = 2c and c,
        exactly, for c = 1..48. Rate: spans of `cal`."""
        q = self.corun.get("behind")
        if not q:
            return self.sms_per_cta, 1.0
        cs, fs, v = np.log(np.asarray(q["ctas"], float)), np.asarray(q["clocks"], float), np.asarray(q["rate"], float)
        along_c = [float(np.interp(np.log(c), cs, row)) for row in v]  # one value per clock
        return float(q.get("sms_per_cta", 1.0)), float(np.interp(f, fs, along_c))


@dataclass
class Kernels:
    gemm: CublasGemm
    allreduce: NcclAllReduce
    # Per GPU: W = offset + gain_rail * (what scales with the SM rail's voltage)
    #                     + gain_other * (everything else).
    # The power model was fitted on GPU 0 running alone; four boards under one load
    # differ by up to 7%, more under a GEMM than under a collective.
    boards: list = field(default_factory=lambda: [[0.0, 1.0, 1.0]] * 4)
    meta: dict = field(default_factory=dict)

    def save(self, path=DEFAULT):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        d = dict(gemm=asdict(self.gemm), allreduce=asdict(self.allreduce), boards=self.boards, meta=self.meta)
        with open(path, "w") as fh:
            json.dump(d, fh, indent=1)
        return path

    @classmethod
    def load(cls, path=DEFAULT):
        with open(path) as fh:
            d = json.load(fh)
        return cls(
            gemm=CublasGemm(**d["gemm"]),
            allreduce=NcclAllReduce(**d["allreduce"]),
            boards=d.get("boards", [[0.0, 1.0, 1.0]] * 4),
            meta=d.get("meta", {}),
        )
