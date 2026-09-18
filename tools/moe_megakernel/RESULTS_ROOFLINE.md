# Where the model applies, and what tile size can do

Two checks that together decide which of this project's predictions survive.
Reproduce with `python3 roofline_and_tiles.py` -> `data/roofline_tiles.csv`.

## 1. Decode is memory bound, so the wave model does not apply there

Every prediction in `predict_expert_dvfs.py` assumes a layer costs `ceil(tiles/108)` waves
of *compute*. That is only meaningful if the expert GEMM is compute bound. A100's ridge
point is **201 FLOP/byte** (312 TFLOPS bf16 / 1555 GB/s).

| regime | median $n_e$ | arithmetic intensity | verdict |
|---|---|---|---|
| prefill B=128 | 7652 | 627 | **compute bound** |
| prefill B=16 | 955 | 398 | **compute bound** |
| decode B=128 | 15 | 14.7 | memory bound |
| decode B=64 | 7 | 6.9 | memory bound |
| decode B=16 | 2 | 2.0 | memory bound |

And at the layer level decode is not close:

| regime | active experts | weights to stream | HBM time | math time | ratio |
|---|---|---|---|---|---|
| decode B=16 | 52 | 656 MB | 422 us | 5.2 us | **82x** |
| decode B=64 | 62 | 778 MB | 500 us | 20.6 us | **24x** |
| decode B=128 | 63 | 793 MB | 510 us | 41.3 us | **12x** |

Every *active* expert must stream its whole weight matrix regardless of how few tokens it
received, and at batch 64 that is 778 MB per layer against 20 us of arithmetic.

**Consequence: the -10.5% predicted for decode in `fig10` is retracted.** It came from a
wave-quantisation argument, and decode does not spend its time on waves. Whether a
frequency split helps a memory-bound layer is a different question with a different model,
and this project has not measured the memory side at all -- `kappa` was fitted on GEMMs.

What survives is prefill, where the model does apply: **-2.6% (B=16) and -0.1% (B=128)**
at the tightest SLO, falling to ~0 as the SLO loosens.

## 2. A smaller tile is not the lever either

The routing shapes show the last tile-row is half empty on average, so shrinking the
tile's M dimension is the obvious software-only alternative to a DVFS scheme. Comparing
tile counts across tile sizes is meaningless -- a 64-row tile does half the work of a
128-row one -- so the comparable quantity is SM-time, proportional to waves x BM.

| regime | tile M | wasted rows | SM-time vs 128 | weight traffic (upper bound) |
|---|---|---|---|---|
| prefill B=16 | 128 | 6.2% | +0.0% | 2281 MB |
| | 64 | 3.0% | **-3.9%** | 4424 MB |
| | 32 | 1.5% | -5.8% | 8719 MB |
| | 16 | 0.7% | **-6.6%** | 17301 MB |
| prefill B=128 | 128 | 0.8% | +0.0% | 17313 MB |
| | 16 | 0.1% | **-0.8%** | 137564 MB |

So a 16-row tile removes almost all the waste and buys **6.6% of SM-time at B=16, 0.8% at
B=128** -- while multiplying weight traffic 7.6x, because each row-block re-reads its
expert's weights.

The weight column is an **upper bound**: it assumes no reuse, whereas a 4.2 MB expert
weight matrix fits comfortably in the A100's 40 MB L2, so a real kernel re-reads L2 rather
than HBM. The direction is still adverse, and the gain still small.

## Where that leaves the scheme

| lever | prefill B=16 | prefill B=128 | decode |
|---|---|---|---|
| two frequency domains, tightest SLO | -2.6% | -0.1% | not modellable |
| a third domain | +0.1 pp | +0.0 pp | — |
| tile M 128 -> 16 | -6.6% (costs weight traffic) | -0.8% | — |
| **lower the single clock 1380 -> 1005** | **-30%** | **-30%** | — |

The one lever worth an order of magnitude more than the others needs no new hardware, no
new kernel and no partitioning: do not run above the voltage knee.
