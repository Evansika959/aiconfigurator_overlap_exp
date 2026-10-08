#!/usr/bin/env bash
# Routing traces for every MoE family we can run, on the SAME ShareGPT prompts and the
# same batch, so the only thing varying is the model.
#
# B=64 prompts, 512-token cap, 8 decode steps. Decode is kept short on purpose: the
# comparison is about prefill routing, and decode at these batch sizes is memory bound
# anyway (see RESULTS_ROOFLINE.md).
set -u
cd "$(dirname "$0")"
source ./env.sh
run () {            # run <tag> <model> [further tracer arguments, which override the defaults]
  local tag=$1 model=$2
  [ -f "data/routing_${tag}.npz" ] && { echo "### $tag already done"; return; }
  echo "### $tag  $model  $(date +%H:%M:%S)"
  python3 trace_routing_sharegpt.py --model "$model" --batch 64 --max-len 512 \
      --decode-steps 8 --repeats 24 --out "data/routing_${tag}.npz" "${@:3}" 2>&1 | grep -v "it/s\]"
}
run mixtral   mistralai/Mixtral-8x7B-v0.1
run granite   ibm-granite/granite-3.0-3b-a800m-instruct
run qwen15moe Qwen/Qwen1.5-MoE-A2.7B
run olmoe_sharegpt allenai/OLMoE-1B-7B-0924
run olmoe_it  allenai/OLMoE-1B-7B-0924-Instruct
run qwen3     Qwen/Qwen3-30B-A3B
# Added 2026-10-01 (INFRA.md, "GPT-OSS-20B"). transformers computes every expert on every
# position for it, so a 64-row batch does not fit in one forward.
run gptoss     openai/gpt-oss-20b --rows 16
run gptoss_raw openai/gpt-oss-20b --rows 16 --no-template      # control: no chat template
# Llama 4 Scout was traced once and is NOT part of the comparison (INFRA.md). This is the
# command, for the record; it needs 217 GB of weights that fetch_models.sh does not get.
#   run llama4 unsloth/Llama-4-Scout-17B-16E-Instruct --sparse-moe --decode-steps 0 \
#              --gpu-mem 30GiB --gpu0-mem 30GiB
echo "ALL DONE $(date +%H:%M:%S)"
