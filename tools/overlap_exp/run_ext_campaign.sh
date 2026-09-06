#!/usr/bin/env bash
# Extend the two fixed-split studies along the three axes each was thin on.
#
#   SPLIT   commbound measured 4 CTA values (4,8,16,32 = 4/7/15/30% of the SMs). The
#           user's own 20% example (22 CTA) fell in an unmeasured gap, and nothing above
#           32 was ever run, so "committing to 32 CTA is free" rested on 32 being the
#           right edge of the sampled range rather than on evidence. Adds 2, 12, 22,
#           48, 64 -> 9 split points from 1.9% to 59% of the SMs.
#   CLOCK   every optimum in fig10 landed on 1200 or 1410 with a gap from 900 to 1200
#           straddling the measured voltage knee (1020-1080 MHz). Adds 1050 and 1305 at
#           every split, so the optimum is resolved instead of snapped to a coarse grid.
# `serial` is not measured. The two studies compose their candidates from gemm_only and
# comm_only and score them against the unlocked concurrent run; nothing reads the serial
# row, and it is half the windows. Backfill it if a speedup column is ever needed again.
#
#   WORKLOAD  all 18 workloads were comm-bound by construction, which is exactly the
#           regime that favours a wide comm split -- the split conclusion is confounded
#           with the regime it was derived in. Adds 64 MiB, where the GEMM leads, as the
#           out-of-regime test.
#
# Ordered so a partial run is still useful: baselines first (everything is quoted
# against them), then the split axis, then the clock axis, then the new regime.
set -u
cd "$(dirname "$0")"
D=data
L=$D/logs_ext.txt
NEW_CTAS=2,12,22,48,64
ALL_CTAS=2,4,8,12,16,22,32,48,64
OLD_CLK=1410,1200,900,705,510,300
NEW_CLK=1305,1050
SHAPES=1024x4096x4096,1024x8192x8192,2048x4096x4096,2048x8192x8192,4096x4096x4096,8192x8192x8192

run () {  # run <label> <out> <env assignments...>
  local label=$1 out=$2; shift 2
  echo "### $label -> $out  $(date +%H:%M:%S)" | tee -a "$L"
  env "$@" SWEEP_SHAPES=$SHAPES python3 sweep_overlap_432.py --out "$out" --resume \
    >>"$L" 2>&1
  echo "### $label done $(date +%H:%M:%S)  rows=$(( $(wc -l <"$out") - 1 ))" | tee -a "$L"
}

# 1. unlocked baselines for every split, including the new ones. fig10 quotes each split
#    against ITS OWN governor-chosen run, so a new split without this row cannot be scored.
for M in 128 256 512 64; do
  run "auto $M MiB" $D/ext_auto_${M}mib.csv \
      SWEEP_CLOCKS=0 SWEEP_CTAS=$ALL_CTAS SWEEP_MIB=$M SWEEP_MODES=concurrent
done

# 2. split axis, at the six clocks already measured
for M in 128 256 512; do
  run "split $M MiB" $D/ext_${M}mib.csv \
      SWEEP_CLOCKS=$OLD_CLK SWEEP_CTAS=$NEW_CTAS SWEEP_MIB=$M \
      SWEEP_MODES=comm_only,gemm_only,concurrent
done

# 3. clock axis, at every split
for M in 128 256 512; do
  run "clock $M MiB" $D/ext_${M}mib.csv \
      SWEEP_CLOCKS=$NEW_CLK SWEEP_CTAS=$ALL_CTAS SWEEP_MIB=$M \
      SWEEP_MODES=comm_only,gemm_only,concurrent
done

# 4. the out-of-regime workload: 64 MiB is GEMM-led, so it tests the split conclusion
#    outside the comm-bound set it was derived on.
run "64 MiB full" $D/ext_64mib.csv \
    SWEEP_CLOCKS=$OLD_CLK,$NEW_CLK SWEEP_CTAS=$ALL_CTAS SWEEP_MIB=64 \
    SWEEP_MODES=comm_only,gemm_only,concurrent

echo "ALL DONE $(date +%H:%M:%S)" | tee -a "$L"
