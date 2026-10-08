"""What a batch asks of the hardware, computed from the routing histogram alone.

A MoE layer is two grouped GEMMs. The align step sorts the (token, expert) pairs by
expert and pads each expert's run to a multiple of BLOCK_M, so the layer is a list of
ROW-BLOCKS, each BLOCK_M rows of one expert, and each row-block is cut into N-tiles of
BLOCK_N output columns. Everything the hardware does follows from four counts per
partition, and all four come from the histogram:

  experts   how many experts have at least one pair        -> weights that must leave DRAM
  pairs     real (token, expert) pairs                     -> activations read and written
  mblocks   sum over experts of ceil(pairs_e / BLOCK_M)    -> weight panels streamed to SMs
  rows      mblocks * BLOCK_M (padding included)           -> tensor-core work

GEMM 1 is (rows x K) @ (K x 2I), GEMM 2 is (rows x I) @ (I x K): GEMM 1 is exactly 2/3 of
the MACs and of the weight bytes for every partition.

This module has three parts:

  Shape, SHAPES                  a model's layer dimensions                 (add a model here)
  prefill_counts, decode_counts  traced routing histograms from ../data/routing_<model>.npz
  Partition, Stage               a set of experts as the kernel sees it: row-blocks in
                                 expert order, cut into the WINDOWS of tiles that n SMs
                                 work on at once. This is what rate.py times and power.py
                                 prices, and what `sim.validate` checks against the
                                 hardware's own counters (ncu_counters.py).

Units: sizes MB, work GMAC (1e9 multiply-accumulates).
"""

import os
from dataclasses import dataclass

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")
MACS_PER_MMA = 2048  # one tensor-core instruction, m16n8k16
TILES_N = 128  # BLOCK_SIZE_N of every kernel in this study
GEMMS = ("g1", "g2")


# ------------------------------------------------------------------ shapes ----
@dataclass(frozen=True)
class Shape:
    """One MoE layer's dimensions. bf16: two bytes per element."""

    name: str
    K: int  # hidden
    I: int  # noqa: E741    expert intermediate (the paper's I)
    E: int  # experts
    topk: int
    layers: int
    npz: str  # routing traces, in ../data
    bpe: int = 2

    @property
    def w_bytes(self):
        """Weight bytes of one expert, both GEMMs."""
        return (2 * self.I * self.K + self.K * self.I) * self.bpe

    @property
    def epi_bytes(self):
        """Bytes per (token, expert) pair the epilogue moves: activation reads 2I and
        writes I, the top-k sum reads K and writes K / topk."""
        return (3 * self.I + self.K + self.K / self.topk) * self.bpe

    def gemm(self, g):
        """(output columns N, inner dimension K, weight bytes per expert) of GEMM g."""
        if g == "g1":
            return 2 * self.I, self.K, 2 * self.I * self.K * self.bpe
        return self.K, self.I, self.K * self.I * self.bpe


SHAPES = {
    "qwen": Shape("qwen", K=2048, I=768, E=128, topk=8, layers=48, npz="routing_qwen.npz"),
    "qwen3next": Shape("qwen3next", K=2048, I=512, E=512, topk=10, layers=48, npz="routing_qwen3next.npz"),
    "olmoe": Shape("olmoe", K=2048, I=1024, E=64, topk=8, layers=16, npz="routing_olmoe_sharegpt.npz"),
    "granite": Shape("granite", K=1536, I=512, E=40, topk=8, layers=32, npz="routing_granite.npz"),
}


# ----------------------------------------------------------------- routing ----
_ARR = {}


def _array(shape, prefix):
    """A routing array from the model's npz. NpzFile decompresses on EVERY access, so
    the array is kept: a 3000-batch workload otherwise spends 30 s unzipping."""
    key = (shape.npz, prefix)
    if key not in _ARR:
        Z = np.load(os.path.join(DATA, shape.npz))
        _ARR[key] = Z[[k for k in Z.files if k.startswith(prefix)][0]]
    return _ARR[key]


def prefill_counts(shape, rep=0, layer=0):
    """Pairs per expert of prompt set `rep` at MoE layer `layer`."""
    A = _array(shape, "prefill")
    return (A[layer] if A.ndim == 2 else A[rep, layer]).astype(np.int64)


def decode_counts(shape, cells, layer=0, start=0):
    """Pool `cells` real decode (rep, step) routings, 64 tokens each, beginning at cell
    `start`. Decode tokens pick their experts independently, so pooling cells is a real
    batch, not a scaled histogram. Trimmed to a whole number of tokens, as
    decode_wall.decode_inputs does."""
    A = _array(shape, "decode")  # (reps, steps, layers, experts)
    flat = A[:, :, layer, :].reshape(-1, A.shape[-1])
    idx = (start + np.arange(cells)) % len(flat)
    c = flat[idx].sum(axis=0).astype(np.int64)
    c[np.argmax(c)] -= int(c.sum()) % shape.topk
    return c


def counts_for(shape, cells, layer=0, rep=0):
    """cells = 0 is the prefill batch of `rep`; otherwise a pooled decode batch."""
    return prefill_counts(shape, rep, layer) if cells == 0 else decode_counts(shape, cells, layer)


def split(counts, n_cold):
    """The partition rule: the n_cold least-routed experts are 'cold'. Stable sort,
    exactly as the measurement scripts do it, so ties break the same way."""
    order = np.argsort(np.asarray(counts), kind="stable")
    cold = np.zeros(len(counts), dtype=bool)
    cold[order[:n_cold]] = True
    return cold


# -------------------------------------------------------------- partitions ----
@dataclass
class Stage:
    """One GEMM of one partition on n SMs: its windows, grouped by type. Windows that
    ask for the same thing are stored once, with a count."""

    n: int
    cnt: np.ndarray  # windows of this type
    panel: np.ndarray  # MB of weight panels per window
    gmac: np.ndarray  # GMAC per window, padding included
    dram: np.ndarray  # MB that must come from DRAM
    wait: np.ndarray  # MB of DRAM wait, summed over the SMs that wait
    gmac_real: float = 0.0  # GMAC on real (token, expert) pairs; the rest is padding
    tot_panel: float = 0.0
    tot_dram: float = 0.0
    tot_gmac: float = 0.0

    def __post_init__(self):
        self.tot_panel = float(self.cnt @ self.panel)
        self.tot_dram = float(self.cnt @ self.dram)
        self.tot_gmac = float(self.cnt @ self.gmac)

    @property
    def gmac_pad(self):
        """Tensor work on padding rows: it takes as long as real work and, multiplying
        zeros, costs much less energy."""
        return self.tot_gmac - self.gmac_real


class Partition:
    """A set of experts of one layer, as the kernel sees it: row-blocks in expert
    order. `stage(g, n)` cuts it into windows for n SMs."""

    def __init__(self, shape: Shape, counts, block_m, experts=None, block_n=TILES_N):
        c = np.asarray(counts, dtype=np.int64)
        if experts is not None:
            c = c[np.asarray(experts)]
        c = c[c > 0]
        self.shape, self.block_m, self.block_n = shape, block_m, block_n
        self.pairs_e = c.astype(float)
        self.m = ((c + block_m - 1) // block_m).astype(float)  # row-blocks per expert
        self.mblocks = float(self.m.sum())
        self.pairs = float(c.sum())
        self.experts = int(c.size)
        self.rows = self.mblocks * block_m
        self._stages = {}

    def __bool__(self):
        return self.mblocks > 0

    @property
    def weight_mb(self):
        """MB of weights its live experts hold, both GEMMs."""
        return self.experts * self.shape.w_bytes / 1e6

    def stage(self, g, n, window=None):
        """`window` is how many tiles are in flight at once. For the persistent kernel
        that is one per SM (the default). vLLM's queue kernel keeps several blocks
        resident on each SM, so its window is wider than its SM count."""
        key = (g, n, window)
        if key not in self._stages:
            self._stages[key] = self._build(g, n, window or n)
        return self._stages[key]

    def counters(self, g):
        """What the hardware's own counters should read for GEMM g, for comparison
        with ncu: tiles launched, tensor-core instructions, bytes read from DRAM."""
        st = self.stage(g, 1)
        tn = -(-self.shape.gemm(g)[0] // self.block_n)
        return dict(tiles=self.mblocks * tn, tensor=st.tot_gmac * 1e9 / MACS_PER_MMA, dram_bytes=st.tot_dram * 1e6)

    def _build(self, g, n, window):
        N, K, wb = self.shape.gemm(g)
        wb /= 1e6
        tn = -(-N // self.block_n)
        W = max(1.0, window / tn)  # row-blocks per window
        nw = int(np.ceil(self.mblocks / W - 1e-9))
        s = np.cumsum(self.m) - self.m  # where each expert starts
        k = np.minimum((s / W).astype(int), nw - 1)
        starts = np.bincount(k, minlength=nw).astype(float)
        # every row-block of the expert inside the window waits for its panel
        mult = np.bincount(k, weights=np.minimum(self.m, W), minlength=nw)
        length = np.full(nw, W)
        length[-1] = self.mblocks - W * (nw - 1)
        act = self.pairs * K * self.shape.bpe / 1e6 / self.mblocks  # per row-block
        cols = np.stack(
            [length * wb, length * self.block_m * N * K / 1e9, starts * wb + length * act, mult * wb + length * act],
            axis=1,
        )
        u, cnt = np.unique(np.round(cols, 9), axis=0, return_counts=True)
        return Stage(
            n=n,
            cnt=cnt.astype(float),
            panel=u[:, 0],
            gmac=u[:, 1],
            dram=u[:, 2],
            wait=u[:, 3],
            gmac_real=self.pairs * N * K / 1e9,
        )
