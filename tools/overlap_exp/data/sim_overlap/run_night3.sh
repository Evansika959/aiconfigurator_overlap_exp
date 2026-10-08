#!/usr/bin/env bash
# Third queue: ncu (the first attempt failed on a stale lock file in /tmp), then the
# synthetic training steps.
set -u
cd "$(dirname "$0")/../.."
D=data/sim_overlap
say () { echo "### $* $(date -u +%H:%M:%S)" | tee -a $D/night.log; }
until grep -q "night queue 2 done" $D/night.log 2>/dev/null; do sleep 20; done
say "ncu: GEMM launch attributes (second attempt)"
sudo -n /usr/local/cuda/bin/ncu --profile-from-start off --target-processes all \
  --metrics launch__grid_size,launch__block_size,launch__shared_mem_per_block_dynamic,launch__shared_mem_per_block_static,launch__shared_mem_config_size,launch__registers_per_thread,sm__cycles_active.avg,sm__cycles_elapsed.max,dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes.sum \
  --csv python3 profile_kernels.py > $D/ncu_gemm.csv 2> $D/ncu_gemm.err
python3 profile_kernels.py --parse $D/ncu_gemm.csv > $D/ncu_gemm.txt 2>&1
for P in train; do
  if [ ! -f $D/$P.csv ]; then
    say "$P"
    python3 -u sweep_spans.py --plan $D/plan_$P.json --out $D/$P.csv > $D/$P.log 2>&1
  fi
done
say "night queue 3 done"
nvidia-smi --query-gpu=index,persistence_mode,clocks.sm,power.limit --format=csv,noheader | tee -a $D/night.log
