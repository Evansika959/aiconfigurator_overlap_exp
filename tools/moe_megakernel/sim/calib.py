"""The fitted hardware: everything `calibrate.py` learned from the measured CSVs, in one
JSON file that the engine loads. Nothing in here is specific to a workload.

  kernels    per-SM costs of each kernel implementation, by name     (add a kernel here)
  epilogue   activation and top-k sum
  power      energy per unit of activity, V^2 and idle power per clock
  meta       what each number was fitted on, and how far the fit is from its own data
"""

import json
import os
from dataclasses import asdict, dataclass, field

from .files import read_json
from .power import PowerParams
from .rate import RateParams

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = os.path.join(HERE, "calib", "a100_qwen.json")


@dataclass
class KernelCal:
    """What one kernel implementation costs. `Kernel.type` in a scenario names one.

    The chip's shared capacities (DRAM, L2 delivery, their coupling) are the same in
    every kernel's `rate`; what differs is the per-SM cost and how many tiles an SM
    keeps in flight."""

    rate: RateParams
    # Tiles in flight per SM. 1: one resident block per SM, tiles handed to it (our
    # persistent kernel). vLLM's queue kernel keeps several blocks resident on an SM.
    window: float = 1.0


@dataclass
class Epilogue:
    """Activation after GEMM 1 and the top-k sum after GEMM 2: two elementwise passes
    over every SM, memory-bound. ms = t0 + pairs * (c0 + c1 / f); half before GEMM 2,
    half after it. Energy above idle: pairs * (j_dram + g(f) * j_sm)."""

    t0: float
    c0: float
    c1: float
    j_dram: float  # mJ per pair that does not scale with voltage
    j_sm: float  # mJ per pair at the voltage floor
    # Bytes per pair of the model it was fitted on (Qwen3-30B); another model's pairs are
    # scaled by its own bytes per pair.
    ref_bytes: float = 9216.0

    def ms(self, pairs, f):
        return self.t0 + pairs * (self.c0 + self.c1 / f)


@dataclass
class Calib:
    kernels: dict[str, KernelCal]
    epilogue: Epilogue
    power: PowerParams
    meta: dict = field(default_factory=dict)

    def kernel(self, name) -> KernelCal:
        if name not in self.kernels:
            raise KeyError(f"no calibration for kernel '{name}'; have {sorted(self.kernels)}")
        return self.kernels[name]

    def rate(self, name) -> RateParams:
        return self.kernel(name).rate

    def save(self, path=DEFAULT):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        d = dict(
            kernels={k: dict(rate=asdict(v.rate), window=v.window) for k, v in self.kernels.items()},
            epilogue=asdict(self.epilogue),
            power=asdict(self.power),
            meta=self.meta,
        )
        with open(path, "w") as fh:
            json.dump(d, fh, indent=1)
        return path

    @classmethod
    def load(cls, path=DEFAULT):
        d = read_json(path)
        if "kernels" in d:
            kernels = {k: KernelCal(RateParams(**v["rate"]), v.get("window", 1.0)) for k, v in d["kernels"].items()}
        else:  # files written before kernels were named
            kernels = {
                "persistent": KernelCal(RateParams(**d["persistent"])),
                "queue": KernelCal(RateParams(**d["queue"]), d["queue_window"]),
            }
        return cls(
            kernels=kernels, epilogue=Epilogue(**d["epilogue"]), power=PowerParams(**d["power"]), meta=d.get("meta", {})
        )
