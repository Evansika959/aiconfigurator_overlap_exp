#!/usr/bin/env bash
# Follow-ups queued behind run_night.sh (which must not be edited while it runs).
set -u
cd "$(dirname "$0")/../.."
D=data/sim_overlap
say () { echo "### $* $(date -u +%H:%M:%S)" | tee -a $D/night.log; }
until grep -q "night queue done" $D/night.log 2>/dev/null; do sleep 20; done
for P in endphase; do
  if [ ! -f $D/$P.csv ]; then
    say "$P"
    python3 -u sweep_spans.py --plan $D/plan_$P.json --out $D/$P.csv > $D/$P.log 2>&1
  fi
done
say "night queue 2 done"
