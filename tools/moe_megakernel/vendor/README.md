# Vendored third-party kernel

`fused_moe_triton.py` contains two Triton JIT functions copied **verbatim** from
vLLM, plus a from-scratch driver written here.

| what | origin |
|---|---|
| `write_zeros_to_output` | vLLM `v0.11.0`, `vllm/model_executor/layers/fused_moe/fused_moe.py` L48-59 |
| `fused_moe_kernel` | same file, L270-489 |
| `moe_align_block_size` | **ours** — vLLM's is a CUDA op in `csrc/`; reimplemented in pure PyTorch |
| `fused_experts` driver | **ours** — vLLM's `fused_experts_impl` minus quantisation, chunking, EP, modular-kernel dispatch |

vLLM is Apache-2.0. Upstream:
https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/model_executor/layers/fused_moe/fused_moe.py

Why vendor rather than `pip install vllm`: vLLM 0.11 pins torch 2.8/2.9 with its own
compiled `_C` extension and would drag a second torch into `pylibs/` (the exact trap
documented in `../env.sh`). We only need the Triton kernel, which has no vLLM imports.
