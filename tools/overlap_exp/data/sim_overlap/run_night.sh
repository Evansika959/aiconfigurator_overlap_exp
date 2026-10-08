#!/usr/bin/env bash
# The calibration half of the 2026-10-08 simulator campaign, after `cal`. Every step
# writes its own file and is skipped if that file exists. Nothing here is a blind cell.
set -u
cd "$(dirname "$0")/../.."
D=data/sim_overlap
say () { echo "### $* $(date -u +%H:%M:%S)" | tee -a $D/night.log; }
until grep -q "^DONE" $D/cal.log 2>/dev/null; do sleep 20; done
say "cal finished"

if [ ! -f $D/ncu_gemm.csv ]; then
  say "ncu: GEMM launch attributes"
  sudo -n /usr/local/cuda/bin/ncu --profile-from-start off --target-processes all \
    --metrics launch__grid_size,launch__block_size,launch__shared_mem_per_block_dynamic,launch__shared_mem_per_block_static,launch__shared_mem_config_size,launch__registers_per_thread,sm__cycles_active.avg,sm__cycles_elapsed.max,dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes.sum \
    --csv python3 profile_kernels.py > $D/ncu_gemm.csv 2> $D/ncu_gemm.err
  python3 profile_kernels.py --parse $D/ncu_gemm.csv > $D/ncu_gemm.txt 2>&1
fi
if [ ! -f $D/nccl_occupancy.csv ]; then
  say "occupancy probe under NCCL"
  python3 -u probe_nccl_occupancy.py --out $D/nccl_occupancy.csv > $D/nccl_occupancy.log 2>&1
fi
for P in commsize world2 world3 hetero unbalanced; do
  if [ ! -f $D/$P.csv ]; then
    say "$P"
    python3 -u sweep_spans.py --plan $D/plan_$P.json --out $D/$P.csv > $D/$P.log 2>&1
  fi
done
say "night queue done"
nvidia-smi --query-gpu=index,persistence_mode,clocks.sm,power.limit --format=csv,noheader | tee -a $D/night.log
