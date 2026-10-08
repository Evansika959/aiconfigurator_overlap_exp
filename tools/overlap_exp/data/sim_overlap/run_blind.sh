#!/usr/bin/env bash
# The registered test (../../../moe_megakernel/OVERLAP_HELDOUT.md). Refuses to start
# unless the predictions are on disk: a blind cell is measured only after it was predicted.
set -u
cd "$(dirname "$0")/../.."
D=data/sim_overlap
STAMP=../moe_megakernel/data/overlap_heldout/stamp_first_shot.json
say () { echo "### $* $(date -u +%H:%M:%S)" | tee -a $D/night.log; }
if [ ! -f "$STAMP" ]; then echo "no registered predictions ($STAMP): not measuring"; exit 1; fi
say "blind test starts; predictions registered $(python3 -c "import json;print(json.load(open('$STAMP'))['written_utc'])") UTC"
for P in blind_multi blind_train blind_auto blind; do
  if [ ! -f $D/$P.csv ]; then
    say "$P"
    python3 -u sweep_spans.py --plan $D/plan_$P.json --out $D/$P.csv > $D/$P.log 2>&1
  fi
done
say "blind test done"
nvidia-smi --query-gpu=index,persistence_mode,clocks.sm,power.limit --format=csv,noheader | tee -a $D/night.log
