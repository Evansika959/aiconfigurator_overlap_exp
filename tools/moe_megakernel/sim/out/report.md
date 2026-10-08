## calibration

| fit | fitted on | free numbers | rms | worst |
|---|---|---|---|---|
| rate, persistent kernel | 2964 solo and co-run times, 510-1410 MHz | 7 | 3.30% | +12.5% |
| rate, queue kernel | 197 end-to-end layer times, 3 campaigns | 4 + window | 1.59% | -4.8% |
| epilogue time | 118 (whole layer) - (its launches) | 3 | 8.05% | +35.2% |
| board power | 2261 solo and co-run records | 4 + 2 per clock | 2.28% of board power | +16.2% |

| parameter | value | meaning |
|---|---|---|
| kP at 1290 MHz | 22.22 us SM / MB | weight-panel delivery per SM; 14% of it does not scale with the clock |
| kR at 1290 MHz | 0.897 ms SM / GMAC | tensor work per SM: 2.23 TFLOP/s per SM |
| L0 at 1290 MHz | 0.78 us SM / MB | DRAM wait per SM, DRAM idle |
| BW_D | 1388 GB/s | DRAM bandwidth |
| BW_L at 1290 MHz | 20.2 TB/s | L2 delivery capacity; proportional to the clock |
| xi at 1290 MHz | 0.406 | how hard a busy DRAM leans on L2 delivery |
| e_mac | 1.695 J/TMAC | tensor work on real pairs, at the voltage floor |
| e_pad | 0.598 J/TMAC | tensor work on padding rows: 35% of real |
| e_panel | 19.5 mJ/GB | weight panels delivered to SMs, at the floor |
| e_dram | 77.6 mJ/GB | DRAM traffic; the same at every clock |
| g(1050), g(1290), g(1410) | 1.043, 1.424, 1.833 | supply voltage squared, relative to the floor |
| P_idle(510), (1050), (1290), (1410) | 56.7, 60.2, 69.3, 76.9 W | board, nothing running |
| P_shared | 54.4 W | the part of idle power no clock changes |
| epilogue | 45 us + pairs x (4.88 + 7703/f) ns | activation and top-k sum |

Cross-validation: fitted on the scan alone, the `holdout` sweep's 994 times came out at rms 4.20%, bias +1.03%, worst +18.0%.

Cross-validation: fitted on the scan alone, the `lowclk` sweep's 800 times came out at rms 3.05%, bias -1.05%, worst -8.6%.

## validation

| rung | what | time: n | rms | bias | worst | power: n | rms | bias | worst |
|---|---|---|---|---|---|---|---|---|---|
| 1 fitted | scan, 930-1410 MHz: solo | 622 | 2.75% | +0.34% | -10.3% | 622 | 2.33% | -0.17% | +8.4% |
| 1 fitted | scan, 930-1410 MHz: co-run | 548 | 3.61% | -0.76% | -11.2% | 274 | 2.74% | -0.60% | +9.4% |
| 1 fitted | holdout, 990-1380 MHz: solo | 520 | 3.94% | +1.21% | -12.5% | 520 | 2.99% | -0.45% | -7.3% |
| 1 fitted | holdout, 990-1380 MHz: co-run | 474 | 3.74% | +0.33% | +12.7% | 237 | 3.46% | -0.32% | +13.8% |
| 1 fitted | lowclk, 510-870 MHz: solo | 416 | 3.01% | -1.07% | -9.2% | 416 | 1.71% | +0.54% | +5.6% |
| 1 fitted | lowclk, 510-870 MHz: co-run | 384 | 2.28% | +0.01% | -7.1% | 192 | 2.39% | -0.83% | -7.2% |
| 2 other days | prefill, BM16-128, 54 SMs, 600-1290 MHz | 55 | 5.39% | +2.78% | +10.8% | 55 | 5.47% | +1.47% | +11.0% |
| 2 other days | decode 1024, BM16-128, 600-1290 MHz | 94 | 3.50% | +2.96% | +7.0% | 94 | 2.24% | -1.66% | -5.5% |
| 2 other days | decode 2048, BM16-128, 600-1290 MHz | 106 | 2.98% | +2.42% | +5.4% | 106 | 2.72% | -2.37% | -4.8% |
| 2 other days | prefill, 8 cold experts | 13 | 0.95% | -0.34% | -1.8% | 13 | 1.20% | -0.16% | +2.9% |
| 2 other days | prefill, 32 cold experts | 25 | 1.37% | +0.24% | +3.0% | 25 | 1.93% | -0.89% | -3.9% |
| 2 other days | prefill, 64 cold experts | 26 | 1.09% | -0.29% | -2.2% | 26 | 1.53% | -0.24% | +3.0% |
| 2 other days | prefill, 96 cold experts | 26 | 1.17% | -0.54% | -2.4% | 26 | 1.70% | +0.26% | +3.7% |
| 2 other days | prefill, 54 SMs, 1050-1410 MHz | 32 | 1.00% | +0.84% | +1.8% | 32 | 2.82% | +2.12% | +5.6% |
| 2 other days | first co-run campaign: solo | 96 | 2.98% | +1.21% | +6.7% | 96 | 1.93% | -0.85% | -4.2% |
| 2 other days | first co-run campaign: 54+54, 32+76 | 96 | 3.21% | +0.06% | +7.6% | 48 | 2.46% | -0.85% | -5.6% |
| 3 other workloads | MoE layer 6: solo | 154 | 3.42% | +1.38% | +9.0% | 154 | 2.95% | -0.01% | +8.2% |
| 3 other workloads | MoE layer 6: co-run | 122 | 4.05% | +0.38% | -10.8% | 61 | 2.48% | -0.90% | -6.2% |
| 3 other workloads | MoE layer 24: solo | 154 | 3.75% | +1.49% | +10.5% | 154 | 3.30% | -0.01% | +9.0% |
| 3 other workloads | MoE layer 24: co-run | 124 | 4.62% | +0.82% | -12.6% | 62 | 2.51% | -0.67% | -5.6% |
| 3 other workloads | MoE layer 36: solo | 154 | 3.57% | +1.30% | +9.2% | 154 | 3.11% | -0.18% | -8.6% |
| 3 other workloads | MoE layer 36: co-run | 124 | 4.04% | +0.07% | -11.5% | 62 | 2.60% | -0.63% | -5.2% |
| 3 other workloads | MoE layer 47: solo | 154 | 4.44% | +1.85% | +16.9% | 154 | 5.03% | +1.42% | +27.0% |
| 3 other workloads | MoE layer 47: co-run | 126 | 5.97% | +1.96% | +18.6% | 63 | 3.32% | +0.26% | +13.5% |
| 3 other workloads | pinned layer END TO END, six SM splits | 108 | 1.55% | +0.48% | +4.5% | 108 | 2.05% | +1.26% | +5.3% |
| 3b other models | OLMoE, 64 experts, I=1024: solo | 104 | 2.95% | +1.91% | +8.5% | 104 | 2.82% | +1.29% | -6.4% |
| 3b other models | OLMoE, 64 experts, I=1024: co-run | 96 | 3.09% | +2.14% | -7.3% | 48 | 3.09% | -2.14% | -6.5% |
| 3b other models | OLMoE, 64 experts, I=1024: whole layers, both kernels | 14 | 1.92% | +1.00% | +3.9% | 14 | 2.96% | +2.45% | +4.2% |
| 3b other models | Granite, 40 experts, K=1536: solo | 92 | 8.20% | -7.76% | -14.7% | 92 | 6.17% | +5.42% | +16.8% |
| 3b other models | Granite, 40 experts, K=1536: co-run | 72 | 5.68% | -4.98% | -9.6% | 36 | 5.97% | +5.39% | +11.4% |
| 3b other models | Granite, 40 experts, K=1536: a partition that fits in L2 | 36 | 4.14% | -3.26% | -7.7% | 24 | 18.88% | +17.32% | +32.9% |
| 3b other models | Granite, 40 experts, K=1536: whole layers, both kernels | 14 | 5.20% | -4.59% | -7.3% | 14 | 7.47% | +7.15% | +10.0% |
| 3b other models | Qwen3-Next, 512 experts, top-10: solo | 104 | 2.62% | -0.55% | -5.8% | 104 | 2.80% | -0.87% | +9.4% |
| 3b other models | Qwen3-Next, 512 experts, top-10: co-run | 96 | 4.13% | -2.00% | -9.5% | 48 | 4.67% | +3.40% | +11.5% |
| 3b other models | Qwen3-Next, 512 experts, top-10: whole layers, both kernels | 14 | 2.42% | -1.08% | -4.3% | 14 | 3.36% | +2.87% | +5.8% |
| 4 whole layers | persistent kernel, other layers (never fitted) | 40 | 1.26% | -0.11% | -2.0% | 40 | 3.35% | +2.97% | +6.6% |
| 4 whole layers | persistent kernel, training sweeps | 118 | 2.17% | +0.79% | +6.1% | 118 | 2.09% | +1.15% | +7.1% |
| 4 whole layers | queue kernel, other layers (never fitted) | 25 | 1.28% | -0.24% | -4.2% | 25 | 2.35% | +2.17% | +4.1% |
| 4 whole layers | queue kernel, training sweeps | 56 | 1.56% | -0.31% | -4.4% | 56 | 2.02% | +1.61% | +4.5% |
| 4 whole layers | queue kernel, 64-7744 tokens, 510-1410 MHz | 141 | 1.61% | +0.17% | -4.8% | 141 | 1.71% | -0.79% | +4.6% |
| 4 whole layers | queue kernel, 512-12288 tokens, 1290 MHz | 8 | 1.51% | -0.94% | -3.6% | 8 | 1.64% | +0.96% | +3.9% |
| 4 whole layers | persistent, BM64 and BM128, 512-7744 tokens (never fitted) | 28 | 2.74% | +0.63% | -5.5% | 28 | 2.44% | +0.42% | +6.6% |
| 4 whole layers | queue kernel, 512-7744 tokens (never fitted) | 12 | 1.81% | -0.83% | -4.0% | 12 | 2.30% | +1.85% | +4.2% |
| 4 whole layers | prefill at 1200 MHz, both kernels (Sep 17) | 2 | 0.98% | +0.79% | +1.4% | 2 | 1.68% | -1.64% | -2.0% |

| batch | measured MHz | W | ms | simulated MHz | W | ms |
|---|---|---|---|---|---|---|
| 512 | 1410 | 362.8 | 1.0060 | 1410 | 368.2 | 0.9850 |
| 1024 | 1410 | 397.7 | 1.1565 | 1401 | 400.0 | 1.1107 |
| 2048 | 1340 | 396.8 | 1.4703 | 1344 | 400.3 | 1.4517 |
| 4096 | 1320 | 399.5 | 2.2409 | 1322 | 400.6 | 2.2369 |
| 6144 | 1320 | 399.8 | 3.1073 | 1318 | 399.3 | 3.1110 |
| 8192 | 1320 | 399.1 | 3.9623 | 1315 | 399.6 | 3.9483 |
| 10240 | 1320 | 399.2 | 4.8396 | 1315 | 399.8 | 4.8196 |
| 12288 | 1317 | 397.2 | 5.6860 | 1312 | 399.4 | 5.6464 |

| trace | segments x runs | latency rms | power rms | clock rms | energy of each run | first 400 ms after a load step: load change / short idle / wake | highest 100 ms reading, simulated / measured |
|---|---|---|---|---|---|---|---|
| `dynamics` | 7 x 3 | 2.42% | 0.91% | 9 MHz | -1.0%, -1.2%, -1.4% | 0.72% / - / 2.26% | 434 / 470 W |
| `trace` | 11 x 3 | 1.84% | 1.07% | 12 MHz | -1.0%, -0.8%, -1.0% | 0.79% / 0.65% / 4.14% | 436 / 448 W |
| `trace2` | 11 x 2 | 2.07% | 1.35% | 15 MHz | -1.0%, -1.2% | 0.61% / 0.68% / 6.58% | 429 / 412 W |
| `forward` | 7 x 3 | 1.24% | 0.69% | 13 MHz | -0.6%, -0.8%, -0.9% | 0.55% / 0.18% / 5.06% | 425 / 422 W |
| `forward2` | 8 x 3 | 0.91% | 0.68% | 7 MHz | -0.1%, -0.3%, -0.6% | 0.98% / 1.26% / 2.47% | 435 / 416 W |
| `lock1050` | 11 x 2 | 0.89% | 3.07% | 0 MHz | +2.3%, +2.0% | 0.08% / 0.16% / 0.31% | 259 / 258 W |
| `lock1290` | 11 x 2 | 1.82% | 3.39% | 0 MHz | +1.7%, +0.9% | 0.09% / 0.19% / 0.28% | 380 / 378 W |

## control groups

| objective | | decode1024 | decode2048 | decode4096 | prefill |
|---|---|---|---|---|---|
| fastest under TDP (time) | B1 queue kernel, one clock | +0.1% | +0.4% | +0.1% | +0.6% |
| fastest under TDP (time) | B2 persistent kernel, one clock | +14.7% | +21.2% | +21.7% | +21.5% |
| fastest under TDP (time) | B3 pinned, one clock | +18.1% | +24.7% | +23.4% | +21.6% |
| fastest under TDP (time) | B4 pinned, a clock per domain | +18.1% | +24.7% | +23.0% | +21.2% |
| fastest under TDP (time) | B3* pinned, one clock (vLLM per-SM costs) | +1.5% | +1.9% | +1.8% | +1.2% |
| fastest under TDP (time) | B4* pinned, a clock per domain (vLLM per-SM costs) | +1.2% | +1.5% | +1.5% | +0.9% |
| least energy, <= 1.00 x B0 latency (energy) | B1 queue kernel, one clock | cannot meet it | cannot meet it | cannot meet it | cannot meet it |
| least energy, <= 1.00 x B0 latency (energy) | B2 persistent kernel, one clock | cannot meet it | cannot meet it | cannot meet it | cannot meet it |
| least energy, <= 1.00 x B0 latency (energy) | B3 pinned, one clock | cannot meet it | cannot meet it | cannot meet it | cannot meet it |
| least energy, <= 1.00 x B0 latency (energy) | B4 pinned, a clock per domain | cannot meet it | cannot meet it | cannot meet it | cannot meet it |
| least energy, <= 1.00 x B0 latency (energy) | B3* pinned, one clock (vLLM per-SM costs) | cannot meet it | cannot meet it | cannot meet it | cannot meet it |
| least energy, <= 1.00 x B0 latency (energy) | B4* pinned, a clock per domain (vLLM per-SM costs) | cannot meet it | cannot meet it | cannot meet it | cannot meet it |
| least energy, <= 1.10 x B0 latency (energy) | B1 queue kernel, one clock | -27.2% | -17.8% | -14.5% | -14.3% |
| least energy, <= 1.10 x B0 latency (energy) | B2 persistent kernel, one clock | cannot meet it | cannot meet it | cannot meet it | cannot meet it |
| least energy, <= 1.10 x B0 latency (energy) | B3 pinned, one clock | cannot meet it | cannot meet it | cannot meet it | cannot meet it |
| least energy, <= 1.10 x B0 latency (energy) | B4 pinned, a clock per domain | cannot meet it | cannot meet it | cannot meet it | cannot meet it |
| least energy, <= 1.10 x B0 latency (energy) | B3* pinned, one clock (vLLM per-SM costs) | -24.4% | -13.7% | -12.9% | -14.2% |
| least energy, <= 1.10 x B0 latency (energy) | B4* pinned, a clock per domain (vLLM per-SM costs) | -25.9% | -15.8% | -13.5% | -14.2% |
| least energy, <= 1.25 x B0 latency (energy) | B1 queue kernel, one clock | -29.7% | -24.4% | -22.4% | -22.8% |
| least energy, <= 1.25 x B0 latency (energy) | B2 persistent kernel, one clock | -17.2% | +6.2% | +10.8% | +9.0% |
| least energy, <= 1.25 x B0 latency (energy) | B3 pinned, one clock | -12.3% | +18.6% | +16.2% | +12.2% |
| least energy, <= 1.25 x B0 latency (energy) | B4 pinned, a clock per domain | -16.2% | +13.2% | +15.5% | +10.5% |
| least energy, <= 1.25 x B0 latency (energy) | B3* pinned, one clock (vLLM per-SM costs) | -29.1% | -22.8% | -21.7% | -22.2% |
| least energy, <= 1.25 x B0 latency (energy) | B4* pinned, a clock per domain (vLLM per-SM costs) | -29.3% | -23.6% | -21.8% | -22.4% |

**decode1024**

*fastest under TDP*

| | configuration | clock MHz | ms | W | mJ | ms vs B0 | mJ vs B0 | best split |
|---|---|---|---|---|---|---|---|---|
| B0 | queue kernel, stock governor | 1401 | 1.1106 | 400.4 | 444.6 | +0.0% | +0.0% |  |
| B1 | queue kernel, one clock | 1395 | 1.1119 | 395.1 | 439.3 | +0.1% | -1.2% |  |
| B2 | persistent kernel, one clock | 1410 | 1.2735 | 367.7 | 468.2 | +14.7% | +5.3% | BM128 |
| B3 | pinned, one clock | 1410 | 1.3112 | 359.9 | 471.9 | +18.1% | +6.1% | BM128, 64 cold experts on 48 SMs |
| B4 | pinned, a clock per domain | 1410/1410 | 1.3112 | 359.9 | 471.9 | +18.1% | +6.1% | BM128, 64 cold experts on 48 SMs |
| B3* | pinned, one clock (vLLM per-SM costs) | 1395 | 1.1275 | 391.0 | 440.8 | +1.5% | -0.9% | BM128, 112 cold experts on 80 SMs |
| B4* | pinned, a clock per domain (vLLM per-SM costs) | 1410/1380 | 1.1238 | 397.5 | 446.7 | +1.2% | +0.5% | BM128, 112 cold experts on 80 SMs |

*least energy, <= 1.00 x B0 latency*

| | configuration | clock MHz | ms | W | mJ | ms vs B0 | mJ vs B0 | best split |
|---|---|---|---|---|---|---|---|---|
| B0 | queue kernel, stock governor | 1401 | 1.1106 | 400.4 | 444.6 | +0.0% | +0.0% |  |
| B1 | queue kernel, one clock | cannot meet it | | | | | | |
| B2 | persistent kernel, one clock | cannot meet it | | | | | | |
| B3 | pinned, one clock | cannot meet it | | | | | | |
| B4 | pinned, a clock per domain | cannot meet it | | | | | | |
| B3* | pinned, one clock (vLLM per-SM costs) | cannot meet it | | | | | | |
| B4* | pinned, a clock per domain (vLLM per-SM costs) | cannot meet it | | | | | | |

*least energy, <= 1.10 x B0 latency*

| | configuration | clock MHz | ms | W | mJ | ms vs B0 | mJ vs B0 | best split |
|---|---|---|---|---|---|---|---|---|
| B0 | queue kernel, stock governor | 1401 | 1.1106 | 400.4 | 444.6 | +0.0% | +0.0% |  |
| B1 | queue kernel, one clock | 1095 | 1.2212 | 265.0 | 323.6 | +10.0% | -27.2% |  |
| B2 | persistent kernel, one clock | cannot meet it | | | | | | |
| B3 | pinned, one clock | cannot meet it | | | | | | |
| B4 | pinned, a clock per domain | cannot meet it | | | | | | |
| B3* | pinned, one clock (vLLM per-SM costs) | 1155 | 1.2180 | 276.1 | 336.3 | +9.7% | -24.4% | BM128, 112 cold experts on 80 SMs |
| B4* | pinned, a clock per domain (vLLM per-SM costs) | 1155/990 | 1.2214 | 269.6 | 329.3 | +10.0% | -25.9% | BM128, 112 cold experts on 80 SMs |

*least energy, <= 1.25 x B0 latency*

| | configuration | clock MHz | ms | W | mJ | ms vs B0 | mJ vs B0 | best split |
|---|---|---|---|---|---|---|---|---|
| B0 | queue kernel, stock governor | 1401 | 1.1106 | 400.4 | 444.6 | +0.0% | +0.0% |  |
| B1 | queue kernel, one clock | 960 | 1.3226 | 236.4 | 312.6 | +19.1% | -29.7% |  |
| B2 | persistent kernel, one clock | 1215 | 1.3863 | 265.6 | 368.3 | +24.8% | -17.2% | BM128 |
| B3 | pinned, one clock | 1275 | 1.3846 | 281.8 | 390.1 | +24.7% | -12.3% | BM128, 64 cold experts on 48 SMs |
| B4 | pinned, a clock per domain | 1200/1320 | 1.3863 | 268.9 | 372.8 | +24.8% | -16.2% | BM64, 112 cold experts on 80 SMs |
| B3* | pinned, one clock (vLLM per-SM costs) | 960 | 1.3624 | 231.3 | 315.1 | +22.7% | -29.1% | BM128, 64 cold experts on 48 SMs |
| B4* | pinned, a clock per domain (vLLM per-SM costs) | 990/930 | 1.3378 | 235.1 | 314.5 | +20.5% | -29.3% | BM128, 112 cold experts on 80 SMs |

**decode2048**

*fastest under TDP*

| | configuration | clock MHz | ms | W | mJ | ms vs B0 | mJ vs B0 | best split |
|---|---|---|---|---|---|---|---|---|
| B0 | queue kernel, stock governor | 1345 | 1.4515 | 400.5 | 581.3 | +0.0% | +0.0% |  |
| B1 | queue kernel, one clock | 1335 | 1.4580 | 392.3 | 572.0 | +0.4% | -1.6% |  |
| B2 | persistent kernel, one clock | 1410 | 1.7594 | 389.1 | 684.5 | +21.2% | +17.8% | BM128 |
| B3 | pinned, one clock | 1410 | 1.8095 | 381.0 | 689.4 | +24.7% | +18.6% | BM128, 64 cold experts on 32 SMs |
| B4 | pinned, a clock per domain | 1410/1410 | 1.8095 | 381.0 | 689.4 | +24.7% | +18.6% | BM128, 64 cold experts on 32 SMs |
| B3* | pinned, one clock (vLLM per-SM costs) | 1350 | 1.4789 | 398.4 | 589.1 | +1.9% | +1.3% | BM128, 64 cold experts on 32 SMs |
| B4* | pinned, a clock per domain (vLLM per-SM costs) | 1380/1335 | 1.4732 | 396.9 | 584.8 | +1.5% | +0.6% | BM128, 64 cold experts on 32 SMs |

*least energy, <= 1.00 x B0 latency*

| | configuration | clock MHz | ms | W | mJ | ms vs B0 | mJ vs B0 | best split |
|---|---|---|---|---|---|---|---|---|
| B0 | queue kernel, stock governor | 1345 | 1.4515 | 400.5 | 581.3 | +0.0% | +0.0% |  |
| B1 | queue kernel, one clock | cannot meet it | | | | | | |
| B2 | persistent kernel, one clock | cannot meet it | | | | | | |
| B3 | pinned, one clock | cannot meet it | | | | | | |
| B4 | pinned, a clock per domain | cannot meet it | | | | | | |
| B3* | pinned, one clock (vLLM per-SM costs) | cannot meet it | | | | | | |
| B4* | pinned, a clock per domain (vLLM per-SM costs) | cannot meet it | | | | | | |

*least energy, <= 1.10 x B0 latency*

| | configuration | clock MHz | ms | W | mJ | ms vs B0 | mJ vs B0 | best split |
|---|---|---|---|---|---|---|---|---|
| B0 | queue kernel, stock governor | 1345 | 1.4515 | 400.5 | 581.3 | +0.0% | +0.0% |  |
| B1 | queue kernel, one clock | 1170 | 1.5943 | 299.6 | 477.7 | +9.8% | -17.8% |  |
| B2 | persistent kernel, one clock | cannot meet it | | | | | | |
| B3 | pinned, one clock | cannot meet it | | | | | | |
| B4 | pinned, a clock per domain | cannot meet it | | | | | | |
| B3* | pinned, one clock (vLLM per-SM costs) | 1215 | 1.5913 | 315.3 | 501.7 | +9.6% | -13.7% | BM128, 64 cold experts on 32 SMs |
| B4* | pinned, a clock per domain (vLLM per-SM costs) | 1140/1290 | 1.5945 | 307.1 | 489.6 | +9.8% | -15.8% | BM128, 112 cold experts on 80 SMs |

*least energy, <= 1.25 x B0 latency*

| | configuration | clock MHz | ms | W | mJ | ms vs B0 | mJ vs B0 | best split |
|---|---|---|---|---|---|---|---|---|
| B0 | queue kernel, stock governor | 1345 | 1.4515 | 400.5 | 581.3 | +0.0% | +0.0% |  |
| B1 | queue kernel, one clock | 990 | 1.8139 | 242.2 | 439.3 | +25.0% | -24.4% |  |
| B2 | persistent kernel, one clock | 1350 | 1.8103 | 341.1 | 617.5 | +24.7% | +6.2% | BM128 |
| B3 | pinned, one clock | 1410 | 1.8095 | 381.0 | 689.4 | +24.7% | +18.6% | BM128, 64 cold experts on 32 SMs |
| B4 | pinned, a clock per domain | 1410/1380 | 1.8110 | 363.4 | 658.1 | +24.8% | +13.2% | BM128, 64 cold experts on 32 SMs |
| B3* | pinned, one clock (vLLM per-SM costs) | 1035 | 1.8031 | 248.8 | 448.7 | +24.2% | -22.8% | BM128, 32 cold experts on 16 SMs |
| B4* | pinned, a clock per domain (vLLM per-SM costs) | 1035/1005 | 1.8106 | 245.3 | 444.1 | +24.7% | -23.6% | BM128, 64 cold experts on 32 SMs |

**decode4096**

*fastest under TDP*

| | configuration | clock MHz | ms | W | mJ | ms vs B0 | mJ vs B0 | best split |
|---|---|---|---|---|---|---|---|---|
| B0 | queue kernel, stock governor | 1322 | 2.2374 | 400.3 | 895.6 | +0.0% | +0.0% |  |
| B1 | queue kernel, one clock | 1320 | 2.2400 | 398.4 | 892.3 | +0.1% | -0.4% |  |
| B2 | persistent kernel, one clock | 1395 | 2.7239 | 393.4 | 1071.7 | +21.7% | +19.7% | BM128 |
| B3 | pinned, one clock | 1395 | 2.7604 | 389.5 | 1075.1 | +23.4% | +20.0% | BM128, 48 cold experts on 16 SMs |
| B4 | pinned, a clock per domain | 1365/1410 | 2.7525 | 399.6 | 1099.8 | +23.0% | +22.8% | BM128, 48 cold experts on 16 SMs |
| B3* | pinned, one clock (vLLM per-SM costs) | 1320 | 2.2777 | 393.1 | 895.4 | +1.8% | -0.0% | BM128, 48 cold experts on 16 SMs |
| B4* | pinned, a clock per domain (vLLM per-SM costs) | 1410/1275 | 2.2712 | 398.8 | 905.7 | +1.5% | +1.1% | BM128, 80 cold experts on 32 SMs |

*least energy, <= 1.00 x B0 latency*

| | configuration | clock MHz | ms | W | mJ | ms vs B0 | mJ vs B0 | best split |
|---|---|---|---|---|---|---|---|---|
| B0 | queue kernel, stock governor | 1322 | 2.2374 | 400.3 | 895.6 | +0.0% | +0.0% |  |
| B1 | queue kernel, one clock | cannot meet it | | | | | | |
| B2 | persistent kernel, one clock | cannot meet it | | | | | | |
| B3 | pinned, one clock | cannot meet it | | | | | | |
| B4 | pinned, a clock per domain | cannot meet it | | | | | | |
| B3* | pinned, one clock (vLLM per-SM costs) | cannot meet it | | | | | | |
| B4* | pinned, a clock per domain (vLLM per-SM costs) | cannot meet it | | | | | | |

*least energy, <= 1.10 x B0 latency*

| | configuration | clock MHz | ms | W | mJ | ms vs B0 | mJ vs B0 | best split |
|---|---|---|---|---|---|---|---|---|
| B0 | queue kernel, stock governor | 1322 | 2.2374 | 400.3 | 895.6 | +0.0% | +0.0% |  |
| B1 | queue kernel, one clock | 1185 | 2.4450 | 313.3 | 766.1 | +9.3% | -14.5% |  |
| B2 | persistent kernel, one clock | cannot meet it | | | | | | |
| B3 | pinned, one clock | cannot meet it | | | | | | |
| B4 | pinned, a clock per domain | cannot meet it | | | | | | |
| B3* | pinned, one clock (vLLM per-SM costs) | 1200 | 2.4587 | 317.4 | 780.4 | +9.9% | -12.9% | BM128, 48 cold experts on 16 SMs |
| B4* | pinned, a clock per domain (vLLM per-SM costs) | 1275/1155 | 2.4605 | 314.9 | 774.9 | +10.0% | -13.5% | BM128, 80 cold experts on 32 SMs |

*least energy, <= 1.25 x B0 latency*

| | configuration | clock MHz | ms | W | mJ | ms vs B0 | mJ vs B0 | best split |
|---|---|---|---|---|---|---|---|---|
| B0 | queue kernel, stock governor | 1322 | 2.2374 | 400.3 | 895.6 | +0.0% | +0.0% |  |
| B1 | queue kernel, one clock | 1020 | 2.7766 | 250.1 | 694.6 | +24.1% | -22.4% |  |
| B2 | persistent kernel, one clock | 1350 | 2.7913 | 355.5 | 992.2 | +24.8% | +10.8% | BM128 |
| B3 | pinned, one clock | 1380 | 2.7825 | 373.9 | 1040.3 | +24.4% | +16.2% | BM128, 48 cold experts on 16 SMs |
| B4 | pinned, a clock per domain | 1350/1380 | 2.7837 | 371.7 | 1034.7 | +24.4% | +15.5% | BM128, 48 cold experts on 16 SMs |
| B3* | pinned, one clock (vLLM per-SM costs) | 1035 | 2.7821 | 252.1 | 701.4 | +24.3% | -21.7% | BM128, 48 cold experts on 16 SMs |
| B4* | pinned, a clock per domain (vLLM per-SM costs) | 1020/1035 | 2.7826 | 251.8 | 700.7 | +24.4% | -21.8% | BM128, 48 cold experts on 16 SMs |

**prefill**

*fastest under TDP*

| | configuration | clock MHz | ms | W | mJ | ms vs B0 | mJ vs B0 | best split |
|---|---|---|---|---|---|---|---|---|
| B0 | queue kernel, stock governor | 1315 | 3.7526 | 399.7 | 1499.7 | +0.0% | +0.0% |  |
| B1 | queue kernel, one clock | 1305 | 3.7766 | 391.9 | 1480.0 | +0.6% | -1.3% |  |
| B2 | persistent kernel, one clock | 1380 | 4.5606 | 385.7 | 1759.1 | +21.5% | +17.3% | BM128 |
| B3 | pinned, one clock | 1395 | 4.5623 | 399.8 | 1824.1 | +21.6% | +21.6% | BM128, 80 cold experts on 48 SMs |
| B4 | pinned, a clock per domain | 1410/1380 | 4.5474 | 399.4 | 1816.3 | +21.2% | +21.1% | BM128, 80 cold experts on 48 SMs |
| B3* | pinned, one clock (vLLM per-SM costs) | 1305 | 3.7980 | 390.1 | 1481.7 | +1.2% | -1.2% | BM128, 96 cold experts on 64 SMs |
| B4* | pinned, a clock per domain (vLLM per-SM costs) | 1320/1305 | 3.7861 | 395.0 | 1495.7 | +0.9% | -0.3% | BM128, 80 cold experts on 48 SMs |

*least energy, <= 1.00 x B0 latency*

| | configuration | clock MHz | ms | W | mJ | ms vs B0 | mJ vs B0 | best split |
|---|---|---|---|---|---|---|---|---|
| B0 | queue kernel, stock governor | 1315 | 3.7526 | 399.7 | 1499.7 | +0.0% | +0.0% |  |
| B1 | queue kernel, one clock | cannot meet it | | | | | | |
| B2 | persistent kernel, one clock | cannot meet it | | | | | | |
| B3 | pinned, one clock | cannot meet it | | | | | | |
| B4 | pinned, a clock per domain | cannot meet it | | | | | | |
| B3* | pinned, one clock (vLLM per-SM costs) | cannot meet it | | | | | | |
| B4* | pinned, a clock per domain (vLLM per-SM costs) | cannot meet it | | | | | | |

*least energy, <= 1.10 x B0 latency*

| | configuration | clock MHz | ms | W | mJ | ms vs B0 | mJ vs B0 | best split |
|---|---|---|---|---|---|---|---|---|
| B0 | queue kernel, stock governor | 1315 | 3.7526 | 399.7 | 1499.7 | +0.0% | +0.0% |  |
| B1 | queue kernel, one clock | 1185 | 4.1043 | 313.2 | 1285.3 | +9.4% | -14.3% |  |
| B2 | persistent kernel, one clock | cannot meet it | | | | | | |
| B3 | pinned, one clock | cannot meet it | | | | | | |
| B4 | pinned, a clock per domain | cannot meet it | | | | | | |
| B3* | pinned, one clock (vLLM per-SM costs) | 1185 | 4.1260 | 311.9 | 1286.9 | +10.0% | -14.2% | BM128, 96 cold experts on 64 SMs |
| B4* | pinned, a clock per domain (vLLM per-SM costs) | 1185/1185 | 4.1260 | 311.9 | 1286.9 | +10.0% | -14.2% | BM128, 96 cold experts on 64 SMs |

*least energy, <= 1.25 x B0 latency*

| | configuration | clock MHz | ms | W | mJ | ms vs B0 | mJ vs B0 | best split |
|---|---|---|---|---|---|---|---|---|
| B0 | queue kernel, stock governor | 1315 | 3.7526 | 399.7 | 1499.7 | +0.0% | +0.0% |  |
| B1 | queue kernel, one clock | 1020 | 4.6850 | 247.1 | 1157.9 | +24.8% | -22.8% |  |
| B2 | persistent kernel, one clock | 1335 | 4.6841 | 349.1 | 1635.2 | +24.8% | +9.0% | BM128 |
| B3 | pinned, one clock | 1350 | 4.6606 | 361.0 | 1682.3 | +24.2% | +12.2% | BM128, 96 cold experts on 64 SMs |
| B4 | pinned, a clock per domain | 1350/1335 | 4.6876 | 353.6 | 1657.5 | +24.9% | +10.5% | BM128, 80 cold experts on 48 SMs |
| B3* | pinned, one clock (vLLM per-SM costs) | 1035 | 4.6466 | 251.1 | 1166.9 | +23.8% | -22.2% | BM128, 96 cold experts on 64 SMs |
| B4* | pinned, a clock per domain (vLLM per-SM costs) | 1035/1020 | 4.6847 | 248.3 | 1163.2 | +24.8% | -22.4% | BM128, 80 cold experts on 48 SMs |

## policies on a moving load

| policy | busy s | vs row 1 | busy J | vs row 1 | highest 100 ms reading |
|---|---|---|---|---|---|
| queue kernel, stock governor (measured) | 19.72 |  | 7736 |  | 429 W |
| queue kernel, tdp_bisection | 19.74 | +0.1% | 7656 | -1.0% | 418 W |
| queue kernel, fixed 1050 MHz | 23.18 | +17.6% | 5920 | -23.5% | 259 W |
| pinned*, 64 cold experts, one clock | 22.27 | +13.0% | 8289 | +7.1% | 409 W |
| pinned*, 64 cold experts, two domains | 22.23 | +12.8% | 7497 | -3.1% | 413 W |
| pinned*, experts rebalanced per batch, one clock | 20.26 | +2.7% | 7854 | +1.5% | 410 W |
| pinned*, rebalanced, two domains, governor | 20.26 | +2.8% | 7903 | +2.2% | 415 W |
| pinned*, rebalanced, two domains, tdp_bisection | 20.14 | +2.2% | 7766 | +0.4% | 416 W |
| pinned*, rebalanced, two domains, 0.05 ms actuation | 20.13 | +2.1% | 7777 | +0.5% | 400 W |
| pinned*, rebalanced, two domains, closed_loop | 20.26 | +2.7% | 7760 | +0.3% | 430 W |
| pinned, one block per SM, rebalanced, two domains | 24.09 | +22.2% | 9061 | +17.1% | 401 W |

Fixed 1050 MHz against the governor: measured +15.9% time, -26.0% energy; simulated +17.6%, -23.5%.
Fixed 1290 MHz against the governor: measured +1.7% time, -9.5% energy; simulated +1.8%, -7.4%.

## sensitivity

`sensitivity_cells32_persistent.csv`

| assumption | energy at B3's 1290 MHz latency: B4 - B3 | clocks | fastest under TDP: B4 - B3 | clocks |
|---|---|---|---|---|
| as shipped | -3.24% | 930/1305 | +0.00% | 1410/1410 |
| uncore follows the SLOWER domain | -2.33% | 1005/1320 | +0.00% | 1410/1410 |
| uncore follows the cold domain | -2.33% | 1005/1320 | +0.00% | 1410/1410 |
| no idle power in shared logic | -3.55% | 930/1305 | +0.00% | 1410/1410 |
| half the idle power in shared logic | -2.94% | 930/1305 | +0.00% | 1410/1410 |
| domain crossing costs 5% | -3.25% | 930/1305 | -0.01% | 1395/1410 |
| domain crossing costs 10% | -3.25% | 930/1305 | -0.02% | 1395/1410 |
| lower rail through an LDO | -1.67% | 930/1305 | +0.00% | 1410/1410 |

`sensitivity_cells64_queue.csv`

| assumption | energy at B3's 1290 MHz latency: B4 - B3 | clocks | fastest under TDP: B4 - B3 | clocks |
|---|---|---|---|---|
| as shipped | -3.10% | 930/1305 | -1.60% | 1215/1380 |
| uncore follows the SLOWER domain | -3.85% | 930/1305 | -1.53% | 1245/1380 |
| uncore follows the cold domain | -3.85% | 930/1305 | -1.53% | 1245/1380 |
| no idle power in shared logic | -3.35% | 930/1305 | -1.60% | 1215/1380 |
| half the idle power in shared logic | -2.86% | 930/1305 | -1.58% | 1200/1380 |
| domain crossing costs 5% | -3.10% | 930/1305 | -1.60% | 1215/1380 |
| domain crossing costs 10% | -3.10% | 930/1305 | -1.61% | 1215/1380 |
| lower rail through an LDO | +0.00% | 1290/1290 | -1.47% | 1080/1380 |

