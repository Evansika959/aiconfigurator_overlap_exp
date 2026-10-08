#!/usr/bin/env bash
# Behind the blind test: (1) the clocks the engine chose for three training steps
# (../../../moe_megakernel/sim/node_policy.py wrote the plan and its predictions), measured;
# (2) how many CTAs NCCL uses when nobody sets NCCL_{MIN,MAX}_CTAS: ncu on rank 0.
set -u
cd "$(dirname "$0")/../.."
D=data/sim_overlap
say () { echo "### $* $(date -u +%H:%M:%S)" | tee -a $D/night.log; }
until grep -q "blind test done" $D/night.log 2>/dev/null; do sleep 20; done
if [ -f $D/plan_policy.json ] && [ ! -f $D/policy.csv ]; then
  say "policy"
  python3 -u sweep_spans.py --plan $D/plan_policy.json --out $D/policy.csv > $D/policy.log 2>&1
fi
say "NCCL's default CTA count (ncu, rank 0)"
for MIB in 64 512; do
  PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('127.0.0.1',0)); print(s.getsockname()[1])")
  export WORLD_SIZE=4 MASTER_ADDR=127.0.0.1 MASTER_PORT=$PORT
  unset NCCL_MIN_CTAS NCCL_MAX_CTAS
  for R in 1 2 3; do RANK=$R AR_MIB=$MIB python3 profile_nccl.py 2>>$D/ncu_nccl.err & done
  RANK=0 AR_MIB=$MIB timeout 240 sudo -n --preserve-env=RANK,WORLD_SIZE,MASTER_ADDR,MASTER_PORT,LD_LIBRARY_PATH,AR_MIB \
    /usr/local/cuda/bin/ncu --profile-from-start off --target-processes application-only \
    --metrics launch__grid_size,launch__block_size,launch__shared_mem_config_size \
    --csv python3 profile_nccl.py > $D/ncu_nccl_default_${MIB}mib.csv 2>>$D/ncu_nccl.err
  wait
done
say "after-blind queue done"
nvidia-smi --query-gpu=index,persistence_mode,clocks.sm,power.limit --format=csv,noheader | tee -a $D/night.log
