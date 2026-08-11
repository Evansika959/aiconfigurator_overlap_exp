#!/usr/bin/env python3
"""Direct hardware confirmation of the wave model, via ncu.

The sweep's power model regresses on S_eff -- the TIME-AVERAGED number of busy SMs --
rather than on S, the number allocated, because a GEMM's last wave is usually partial.
That was inferred from goodness of fit; this measures it.

RUN IT LIKE THIS (this script only issues the GEMMs; ncu collects the counters):

  sudo /usr/local/cuda/bin/ncu --profile-from-start off --target-processes all \
    --metrics launch__grid_size,launch__block_size,\
sm__cycles_active.avg,sm__cycles_elapsed.max,gpu__time_duration.sum \
    --csv python3 profile_occupancy.py

WHAT THE TWO COUNTERS MEAN, and why their ratio is exactly S_eff/S:

  sm__cycles_active   "# of cycles with at least one warp in flight", per SM;
                      .avg averages over all 108 SMs
  sm__cycles_elapsed  "# of cycles elapsed on SM"; .max is the kernel's duration T

      sm__cycles_active.avg     (1/108)·Σ_i busy_cycles(SM_i)     mean busy SMs
      ----------------------  = ----------------------------  =  -------------
      sm__cycles_elapsed.max                 T                        108

  Measured against the wave model's tiles/ceil(tiles/108)/108, on the four shapes
  below: 0.5855 vs 0.5926, 0.7753 vs 0.7901, 0.9348 vs 0.9481, 0.9845 vs 0.9981 --
  every one within 2 %, across a 0.59-0.98 range, and low by a consistent ~1.4 %
  (launch ramp-up and drain, which the idealised wave model does not have).
  launch__grid_size also matched ceil(M/256)·ceil(N/128) exactly for all four,
  confirming the 256x128 tile shape read off the kernel name.

CAVEAT ON THE SEMANTICS. "At least one warp in flight" counts an SM whose warps are
all stalled on memory as active. This measures OCCUPANCY IN TIME -- which is what the
wave model is about -- not compute utilisation. For the latter use
sm__throughput.avg.pct_of_peak_sustained_elapsed or the tensor-pipe counters.

PRACTICAL NOTES. ncu needs sudo unless the driver is loaded with
NVreg_RestrictProfilingToAdminUsers=0. It serialises and replays kernels, so absolute
timings here are NOT comparable to the sweep -- but the ratio above is a
dimensionless within-kernel quantity and is unaffected.
"""

import math

import torch

DEV = 0
SHAPES = [(1024, 4096, 4096), (2048, 4096, 4096), (4096, 8192, 8192), (8192, 16384, 16384)]
TOTAL_SM = 108


def main():
    torch.cuda.set_device(DEV)
    bufs = []
    for (m, n, k) in SHAPES:
        x = torch.randn((m, k), dtype=torch.bfloat16, device=f"cuda:{DEV}")
        w = torch.randn((n, k), dtype=torch.bfloat16, device=f"cuda:{DEV}")
        bufs.append((m, n, k, x, w))

    for (m, n, k, x, w) in bufs:            # warm up OUTSIDE the profiled window
        for _ in range(3):
            torch.nn.functional.linear(x, w)
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    for (m, n, k, x, w) in bufs:
        torch.nn.functional.linear(x, w)
        torch.cuda.synchronize()            # one kernel per profiled range
    torch.cuda.cudart().cudaProfilerStop()

    print("expected, for cross-checking the ncu output:")
    print(f"{'shape M,N=K':>16}{'tiles':>8}{'waves@108':>11}{'S_eff':>9}{'S_eff/108':>11}")
    for (m, n, k) in SHAPES:
        t = math.ceil(m / 256) * math.ceil(n / 128)
        w_ = math.ceil(t / TOTAL_SM)
        print(f"{str((m, n)):>16}{t:>8}{w_:>11}{t / w_:>9.1f}{t / w_ / TOTAL_SM:>11.4f}")


if __name__ == "__main__":
    main()
