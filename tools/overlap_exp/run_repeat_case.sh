#!/usr/bin/env bash
# Repeat the one cell that fig7 and fig13 disagree on, so both can quote the same median.
#
# The disagreement (-16.71 vs -13.66) was traced to run-to-run drift in the BASELINE's
# power reading: same config, same clock, T agreeing to 0.26% but P differing 3.3%. The
# repeatability of this instrument is 1.25% median / 4.55% max on power (31 configs
# measured 3x each), so a single window is simply not precise enough to separate a 3 pp
# difference. Ten repeats of everything that feeds the number fixes that.
#
# BOTH SIDES must be repeated, not just the baseline: the composed arm drifted 0.52% too.
#   baseline   -> measure_case_spans (gives power, iteration time AND the spans fig13 draws)
#   composed   -> solo gemm_only / comm_only over all six clocks, via the sweep harness
set -u
cd "$(dirname "$0")"
N=10
M=4096; NK=4096; CT=32; MIB=128
mkdir -p data/repeat
for i in $(seq 1 $N); do
  echo "### rep $i/$N  $(date +%H:%M:%S)"
  python3 measure_case_spans.py --m $M --n $NK --k $NK --mib $MIB --ctas $CT \
      --clock 1200 --out data/repeat/spans_$i.json >/dev/null 2>&1
  SWEEP_MODES=gemm_only,comm_only SWEEP_CLOCKS=1410,1200,900,705,510,300 \
    SWEEP_CTAS=$CT SWEEP_SHAPES=${M}x${NK}x${NK} SWEEP_MIB=$MIB \
    python3 sweep_overlap_432.py --out data/repeat/solo_$i.csv >/dev/null 2>&1
done
echo "ALL DONE $(date +%H:%M:%S)"
