"""The workload: batches of tokens, each a list of MoE layers with a routing histogram.

A scenario describes batches compactly; `build` expands them. The routing is always a
real traced one from ../data/routing_<model>.npz -- a prefill batch is one traced prompt
set, a decode batch is `cells` pooled decode steps of 64 tokens each.

    batches:
      - {phase: prefill, rep: 0, layers: [0]}            # one layer of prompt set 0
      - {phase: decode, cells: 32, layers: [0], repeat: 200}
      - {phase: decode, cells: 16, layers: all, repeat: 8, vary: true, gap_ms: 5}

  layers   a list of MoE layer indices, or `all`
  repeat   run the batch this many times back to back (decode steps)
  vary     each repeat pools the NEXT cells, so every step has its own routing
  at_ms    absolute arrival time of the first repeat; default: when the previous ends
  gap_ms   idle time before each repeat
  slo_ms   latency target per layer, read by SLO-aware controllers
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .demand import SHAPES, decode_counts, prefill_counts


@dataclass
class Layer:
    index: int
    counts: np.ndarray
    key: tuple  # identifies the routing, for caches

    @property
    def pairs(self):
        return int(self.counts.sum())


@dataclass
class Batch:
    name: str
    phase: str
    tokens: int
    layers: list[Layer]
    at_ms: Optional[float] = None
    gap_ms: float = 0.0
    slo_ms: Optional[float] = None


def build(spec, model="qwen"):
    shape = SHAPES[model]
    out, seen = [], {}
    for bi, b in enumerate(spec):
        phase = b.get("phase", "decode")
        layers = b.get("layers", [0])
        if layers == "all":
            layers = list(range(shape.layers))
        for k in range(int(b.get("repeat", 1))):
            L = []
            for li in layers:
                if phase == "prefill":
                    rep = int(b.get("rep", 0))
                    key = (model, "prefill", rep, li)
                    if key not in seen:
                        seen[key] = Layer(li, prefill_counts(shape, rep, li), key)
                else:
                    cells = int(b["cells"])
                    start = int(b.get("start", 0)) + (k * cells if b.get("vary") else 0)
                    key = (model, "decode", cells, start, li)
                    if key not in seen:
                        seen[key] = Layer(li, decode_counts(shape, cells, li, start), key)
                L.append(seen[key])
            out.append(
                Batch(
                    name=b.get("name", f"{phase}{bi}") + (f".{k}" if k else ""),
                    phase=phase,
                    tokens=int(L[0].counts.sum()) // shape.topk,
                    layers=L,
                    at_ms=b.get("at_ms") if k == 0 else None,
                    gap_ms=float(b.get("gap_ms", 0.0)),
                    slo_ms=b.get("slo_ms"),
                )
            )
    return out
