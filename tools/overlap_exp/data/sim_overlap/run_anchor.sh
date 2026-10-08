#!/usr/bin/env bash
# Drift check after the 2026-10-08 driver change (580.173.02 -> 580.178.04): repeat cells
# of overlap_432.csv, ext_128mib.csv and auto_128mib.csv with the unchanged harness.
set -u
cd "$(dirname "$0")/../.."
D=data/sim_overlap
SHAPES=1024x4096x4096,1024x8192x8192,2048x4096x4096,2048x8192x8192,4096x4096x4096,8192x8192x8192
echo "### anchor locked 64 MiB $(date -u +%H:%M:%S)"
SWEEP_CLOCKS=1410,1200,900 SWEEP_CTAS=4,32 SWEEP_SHAPES=$SHAPES SWEEP_MIB=64 \
  SWEEP_MODES=comm_only,gemm_only,concurrent python3 -u sweep_overlap_432.py --out $D/anchor_64mib.csv
echo "### anchor locked 128 MiB $(date -u +%H:%M:%S)"
SWEEP_CLOCKS=1410,1200,900 SWEEP_CTAS=4,32 SWEEP_SHAPES=$SHAPES SWEEP_MIB=128 \
  SWEEP_MODES=comm_only,gemm_only,concurrent python3 -u sweep_overlap_432.py --out $D/anchor_128mib.csv
echo "### anchor unlocked 128 MiB $(date -u +%H:%M:%S)"
SWEEP_CLOCKS=0 SWEEP_CTAS=4,32 SWEEP_SHAPES=$SHAPES SWEEP_MIB=128 SWEEP_MODES=concurrent \
  python3 -u sweep_overlap_432.py --out $D/anchor_auto_128mib.csv
echo "### anchor done $(date -u +%H:%M:%S)"
