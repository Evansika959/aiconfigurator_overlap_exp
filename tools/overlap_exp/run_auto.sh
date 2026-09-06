#!/bin/bash
set -u
cd "$(dirname "$0")"
SHAPES=1024x4096x4096,2048x4096x4096,4096x4096x4096,1024x8192x8192,2048x8192x8192,8192x8192x8192
for MIB in 128 256 512; do
  echo "=== ${MIB} MiB unlocked  $(date +%H:%M:%S) ==="
  SWEEP_CLOCKS=0 SWEEP_CTAS=4,8,16,32 SWEEP_SHAPES=$SHAPES SWEEP_MIB=$MIB \
  SWEEP_MODES=concurrent \
  python3 -u sweep_overlap_432.py --out "data/auto_${MIB}mib.csv" || echo "!! ${MIB} failed"
done
echo "=== done $(date +%H:%M:%S) ==="
