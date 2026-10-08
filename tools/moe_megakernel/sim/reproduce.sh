#!/usr/bin/env bash
# Everything in RESULTS_SIM.md, from the CSVs in ../data. No GPU. ~45 min on 16 cores:
# the control groups take 30 of them (each of the four workloads searches 400 pinned
# configurations for three kernels).
#
#   bash sim/reproduce.sh              # from tools/moe_megakernel
#   bash sim/reproduce.sh --no-fit     # keep sim/calib/a100_qwen.json, redo the rest
set -euo pipefail
cd "$(dirname "$0")/.."
source env.sh
mkdir -p sim/out
q() { grep -v "pynvml package is deprecated\|import pynvml" || true; }

if [[ "${1:-}" != "--no-fit" ]]; then
  # the calibration in use, with every form of the rate model cross-validated; then the
  # two forms that were not picked, for sim.sensitivity
  python3 -m sim.calibrate --compare 2>&1 | q | tee sim/out/calibrate.txt
  for form in proportional linear; do
    python3 -m sim.calibrate --form $form --out sim/calib/a100_qwen_$form.json 2>&1 | q > sim/out/calibrate_$form.txt &
  done
  wait
fi
python3 -m sim.test_sim 2>&1 | q | tee sim/out/test.txt
for tag in dynamics trace trace2 forward forward2 heldout1 heldout2; do   # before validate: rung 6 reads them
  python3 -m sim.validate_trace --tag $tag 2>&1 | q | tee sim/out/trace_$tag.txt
done
for f in 1050 1290; do                                 # the same plan at a locked clock
  python3 -m sim.validate_trace --tag lock$f --fixed $f 2>&1 | q | tee sim/out/trace_lock$f.txt
done
python3 -m sim.validate 2>&1 | q | tee sim/out/validation.txt
# the control groups, four batch sizes in parallel
for c in 0 16 32 64; do
  python3 -m sim.study --cells $c 2>&1 | q > sim/out/study_cells$c.txt &
done
python3 -m sim.sensitivity --cells 32 2>&1 | q > sim/out/sensitivity_decode2048.txt &
python3 -m sim.sensitivity --cells 64 --kernel queue 2>&1 | q > sim/out/sensitivity_decode4096_ideal.txt &
for c in 0 16 32 64; do                                # what boosting the surviving domain is worth
  for k in queue persistent; do
    python3 -m sim.boost --cells $c --kernel $k 2>&1 | q > sim/out/boost_cells${c}_$k.txt &
  done
done
wait
cat sim/out/study_cells32.txt sim/out/sensitivity_decode2048.txt
for c in b0 b1 b2 b3 b4 b4_slo; do
  python3 -m sim.run sim/configs/decode2048_$c.yaml 2>&1 | q
done
python3 -m sim.policies --tag trace2 2>&1 | q | tee sim/out/policies_trace2.txt
python3 make_sim_validation_fig.py 2>&1 | q
for tag in trace2 trace dynamics forward forward2; do
  python3 make_sim_trace_fig.py --tag $tag 2>&1 | q
done
for tag in heldout1 heldout2; do      # the registered replay (frozen simulator), then this tree's
  python3 make_sim_trace_fig.py --tag $tag --first-shot 2>&1 | q
  python3 make_sim_trace_fig.py --tag $tag 2>&1 | q
done
python3 make_sim_ladder_fig.py 2>&1 | q
python3 make_sim_layer_fig.py 2>&1 | q
python3 make_sim_step_fig.py 2>&1 | q
