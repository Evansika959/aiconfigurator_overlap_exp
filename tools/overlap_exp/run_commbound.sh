#!/bin/bash
# COMM-BOUND SWEEP -- the region every two-clock claim depends on and none of the
# measurements cover. The 432 sweep fixed the message at 64 MiB, where a well-chosen CTA
# count leaves 11 of 12 shapes GEMM-bound, so t_comm/t_gemm > 1 was reached almost only
# by low-CTA configurations nobody would deploy.
#
# Three message sizes, run as three passes because the harness takes one size per run.
# 512 MiB also pushes the collective calibration past its current 256 MiB ceiling, so
# B(c,f) stops being an extrapolation up there.
#
# Shapes: five small-to-mid GEMMs that can plausibly go comm-bound, plus 8192x8192 as a
# GEMM-bound control. All grids are already ncu-verified in grid12.json.
set -u
cd "$(dirname "$0")"
SHAPES=1024x4096x4096,2048x4096x4096,4096x4096x4096,1024x8192x8192,2048x8192x8192,8192x8192x8192
for MIB in 128 256 512; do
  echo "=== ${MIB} MiB  $(date +%H:%M:%S) ==="
  SWEEP_CLOCKS=1410,1200,900,705,510,300 \
  SWEEP_CTAS=4,8,16,32 \
  SWEEP_SHAPES=$SHAPES \
  SWEEP_MIB=$MIB \
  python3 -u sweep_overlap_432.py --out "data/commbound_${MIB}mib.csv" || echo "!! ${MIB} MiB failed"
done
echo "=== all sizes done $(date +%H:%M:%S) ==="
