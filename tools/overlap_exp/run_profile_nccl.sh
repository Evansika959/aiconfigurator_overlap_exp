#!/usr/bin/env bash
# ncu on rank 0 of a 4-rank all-reduce; see profile_nccl.py.
set -u
cd "$(dirname "$0")"
D=data/sim_overlap
for C in ${1:-4 16}; do
  PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('127.0.0.1',0)); print(s.getsockname()[1])")
  export WORLD_SIZE=4 MASTER_ADDR=127.0.0.1 MASTER_PORT=$PORT NCCL_MIN_CTAS=$C NCCL_MAX_CTAS=$C
  for R in 1 2 3; do RANK=$R python3 profile_nccl.py 2>>$D/ncu_nccl.err & done
  RANK=0 timeout 240 sudo -n --preserve-env=RANK,WORLD_SIZE,MASTER_ADDR,MASTER_PORT,NCCL_MIN_CTAS,NCCL_MAX_CTAS,LD_LIBRARY_PATH \
    /usr/local/cuda/bin/ncu --profile-from-start off --target-processes application-only \
    --metrics launch__grid_size,launch__block_size,launch__thread_count,launch__shared_mem_per_block_dynamic,launch__shared_mem_per_block_static,launch__shared_mem_config_size,launch__registers_per_thread,launch__waves_per_multiprocessor,launch__occupancy_limit_shared_mem \
    --csv python3 profile_nccl.py > $D/ncu_nccl_$C.csv 2>>$D/ncu_nccl.err
  echo "ctas $C: ncu exit $?"
  wait
done
