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
# Added 2026-10-01. GPT-OSS: the transformers checkpoint only (13 GB), not the `original/`
# and `metal/` copies.
python3 -c "
from huggingface_hub import snapshot_download
print('DONE', snapshot_download('openai/gpt-oss-20b', allow_patterns=['*.json', 'model-*.safetensors', '*.jinja', '*.txt'], ignore_patterns=['original/*', 'metal/*']))
"
# Llama 4 Scout is not part of the comparison (INFRA.md) and is 217 GB, so it is not
# fetched here. Meta's repository is gated; the mirror's 50 weight files have the same
# sha256 as Meta's:
#   snapshot_download('unsloth/Llama-4-Scout-17B-16E-Instruct',
#                     allow_patterns=['*.json', '*.safetensors', '*.jinja', '*.txt'])
echo "ALL DONE $(date +%H:%M:%S)"
