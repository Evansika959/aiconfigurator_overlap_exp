#!/usr/bin/env bash
# Same 10-repeat protocol as run_repeat_case.sh, for the other CTA counts of the same
# GEMM row. Both cells' tightest-SLA baseline is 1200 MHz (checked, not assumed).
set -u
cd "$(dirname "$0")"
M=4096; NK=4096; MIB=128
for CT in 4 8; do
  mkdir -p data/repeat_cta$CT
  for i in $(seq 1 10); do
    echo "### cta $CT rep $i/10  $(date +%H:%M:%S)"
    python3 measure_case_spans.py --m $M --n $NK --k $NK --mib $MIB --ctas $CT \
        --clock 1200 --out data/repeat_cta$CT/spans_$i.json >/dev/null 2>&1
    SWEEP_MODES=gemm_only,comm_only SWEEP_CLOCKS=1410,1200,900,705,510,300 \
      SWEEP_CTAS=$CT SWEEP_SHAPES=${M}x${NK}x${NK} SWEEP_MIB=$MIB \
      python3 sweep_overlap_432.py --out data/repeat_cta$CT/solo_$i.csv >/dev/null 2>&1
  done
done
echo "ALL DONE $(date +%H:%M:%S)"
