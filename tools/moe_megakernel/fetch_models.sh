#!/usr/bin/env bash
# Extra MoE models for the routing-skew comparison. Sequential: parallel downloads
# saturate the link and the later ones time out.
set -u
cd "$(dirname "$0")"
for M in "ibm-granite/granite-3.0-3b-a800m-instruct" \
         "allenai/OLMoE-1B-7B-0924-Instruct" \
         "Qwen/Qwen1.5-MoE-A2.7B" \
         "mistralai/Mixtral-8x7B-v0.1"; do
  echo "### $M  $(date +%H:%M:%S)"
  python3 -c "
from huggingface_hub import snapshot_download
p = snapshot_download('$M', allow_patterns=['*.json','*.safetensors','*.txt','*.model'])
print('DONE', p)
"
done
echo "ALL DONE $(date +%H:%M:%S)"
