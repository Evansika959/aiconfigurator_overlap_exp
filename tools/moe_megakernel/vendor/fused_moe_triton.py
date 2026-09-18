"""Fused MoE grouped-GEMM (Triton) — vendored from vLLM v0.11.0.

See README.md in this directory for exactly which parts are copied and which
are ours. Copied parts are Apache-2.0, (c) the vLLM contributors.

Scope deliberately reduced to what B1 needs:
  * bf16/fp16 only (no fp8/int8/int4 paths -- the constexpr flags are kept so
    the copied kernel text is unmodified, but the driver always passes False)
  * no expert-parallel expert_map, no chunking, no bias
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# ----------------------------------------------------------------------------
# BEGIN verbatim copy: vLLM v0.11.0 fused_moe.py L48-59
# ----------------------------------------------------------------------------
@triton.jit
def write_zeros_to_output(c_ptr, stride_cm, stride_cn, pid_n, N, offs_token,
                          token_mask, BLOCK_SIZE_M, BLOCK_SIZE_N,
                          compute_type):
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[
        None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)




# ----------------------------------------------------------------------------
# BEGIN verbatim copy: vLLM v0.11.0 fused_moe.py L270-489
# ----------------------------------------------------------------------------
@triton.jit
def fused_moe_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    b_bias_ptr,
    a_scale_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_ask,
    stride_bse,
    stride_bsk,
    stride_bsn,
    stride_bbe,  # bias expert stride
    stride_bbn,  # bias N stride
    # Block size for block-wise quantization
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    use_fp8_w8a8: tl.constexpr,
    use_int8_w8a8: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    per_channel_quant: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the
        total number of tokens post padding, topk is the number of times
        each token is repeated, and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens,
        repeated topk times and arranged by the expert index they are
        assigned to.
    - expert_ids: A tensor containing the indices of the expert for each
        block. It determines which expert matrix from B should be used for
        each block in A.
    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`. The sorting of
    `sorted_token_ids` by expert index and padding ensures divisibility by
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
    multiplication across different blocks processed by the same expert.
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(
        tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        # -----------------------------------------------------------
        # Write back zeros to the output when the expert is not
        # in the current expert parallel rank.
        write_zeros_to_output(c_ptr, stride_cm, stride_cn, pid_n, N,
                              offs_token, token_mask, BLOCK_SIZE_M,
                              BLOCK_SIZE_N, compute_type)
        return

    offs_bn = (pid_n * BLOCK_SIZE_N +
               tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am +
                      offs_k[None, :] * stride_ak)

    b_ptrs = b_ptr + off_experts * stride_be + (offs_k[:, None] * stride_bk +
                                                offs_bn[None, :] * stride_bn)
    if use_int8_w8a16:
        b_scale_ptrs = b_scale_ptr + off_experts * stride_bse + offs_bn[
            None, :] * stride_bsn
        b_scale = tl.load(b_scale_ptrs)

    if use_fp8_w8a8 or use_int8_w8a8:
        # block-wise
        if group_k > 0 and group_n > 0:
            a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
            offs_bsn = offs_bn // group_n
            b_scale_ptrs = (b_scale_ptr + off_experts * stride_bse +
                            offs_bsn * stride_bsn)
        # channel-wise
        elif per_channel_quant:
            b_scale_ptrs = b_scale_ptr + off_experts * stride_bse + offs_bn[
                None, :] * stride_bsn
            b_scale = tl.load(b_scale_ptrs)
            # Load per-token scale for activations
            a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
            a_scale = tl.load(a_scale_ptrs, mask=token_mask, other=0.0)[:,
                                                                        None]
        # tensor-wise
        else:
            a_scale = tl.load(a_scale_ptr)
            b_scale = tl.load(b_scale_ptr + off_experts)
    if HAS_BIAS:
        # bias shape: [num_experts, N]
        bias_ptrs = b_bias_ptr + off_experts * stride_bbe + offs_bn * stride_bbn
        bias = tl.load(bias_ptrs, mask=(offs_bn < N), other=0.0)
    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.
        a = tl.load(a_ptrs,
                    mask=token_mask[:, None] &
                    (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                    other=0.0)
        b = tl.load(b_ptrs,
                    mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
                    other=0.0)
        # We accumulate along the K dimension.
        if use_int8_w8a16:
            accumulator = tl.dot(a, b.to(compute_type), acc=accumulator)
        elif use_fp8_w8a8 or use_int8_w8a8:
            if group_k > 0 and group_n > 0:
                k_start = k * BLOCK_SIZE_K
                offs_ks = k_start // group_k
                a_scale = tl.load(a_scale_ptrs + offs_ks * stride_ask,
                                  mask=token_mask,
                                  other=0.0)
                b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)

                accumulator += tl.dot(a, b) * a_scale[:,
                                                      None] * b_scale[None, :]
            else:
                if use_fp8_w8a8:
                    # acc used to enable fp8_fast_accum
                    accumulator = tl.dot(a, b, acc=accumulator)
                else:
                    accumulator += tl.dot(a, b)
        else:
            accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    if HAS_BIAS:
        accumulator = accumulator + bias[None, :]
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token,
                             mask=token_mask,
                             other=0)
        accumulator = accumulator * moe_weight[:, None]
    if use_int8_w8a16:
        accumulator = (accumulator * b_scale).to(compute_type)
    elif use_fp8_w8a8 or use_int8_w8a8:
        if group_k > 0 and group_n > 0:
            accumulator = accumulator.to(compute_type)
        else:
            accumulator = (accumulator * a_scale * b_scale).to(compute_type)
    else:
        accumulator = accumulator.to(compute_type)

    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[
        None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)



# ----------------------------------------------------------------------------
# END verbatim copy. Everything below is ours.
# ----------------------------------------------------------------------------


def moe_align_block_size(topk_ids: torch.Tensor, block_size: int,
                         num_experts: int):
    """Pure-PyTorch stand-in for vLLM's `moe_align_block_size` CUDA op.

    Sorts the M*topk (token, expert) pairs by expert and pads each expert's run
    up to a multiple of `block_size`, so every BLOCK_SIZE_M row-block of the
    grouped GEMM touches exactly one expert.

    Returns
      sorted_token_ids : int32 [max_em]  index into the flattened topk_ids;
                                         padding slots hold M*topk so the
                                         kernel's `< num_valid_tokens` mask
                                         drops them
      expert_ids       : int32 [max_em // block_size]  expert per row-block,
                                         -1 for blocks past the end (the kernel
                                         writes zeros for those)
      num_tokens_post_padded : int32 []  real length of sorted_token_ids

    Buffers are allocated at the worst case so nothing here forces a device
    sync -- the grid is sized from the buffer, and the kernel early-returns on
    `pid_m * BLOCK_SIZE_M >= num_tokens_post_padded`.
    """
    dev = topk_ids.device
    flat = topk_ids.reshape(-1)
    n = flat.numel()
    # worst case: every expert's run needs block_size-1 rows of padding
    max_em = n + num_experts * (block_size - 1)
    max_em = ((max_em + block_size - 1) // block_size) * block_size

    cnts = torch.bincount(flat, minlength=num_experts)
    padded = ((cnts + block_size - 1) // block_size) * block_size
    cum = torch.cumsum(padded, 0)
    starts = cum - padded

    order = torch.argsort(flat, stable=True)      # grouped by expert
    exp_of = flat[order]
    excl = torch.cumsum(cnts, 0) - cnts
    rank = torch.arange(n, device=dev) - excl[exp_of]
    dest = starts[exp_of] + rank

    sorted_token_ids = torch.full((max_em, ), n, dtype=torch.int32, device=dev)
    sorted_token_ids[dest] = order.to(torch.int32)

    nblk = max_em // block_size
    expert_ids = torch.full((nblk, ), -1, dtype=torch.int32, device=dev)
    blk_expert = torch.repeat_interleave(
        torch.arange(num_experts, device=dev, dtype=torch.int32),
        padded // block_size)
    expert_ids[:blk_expert.numel()] = blk_expert

    num_tokens_post_padded = cum[-1].to(torch.int32)
    return sorted_token_ids, expert_ids, num_tokens_post_padded


def _invoke(A, B, C, topk_weights, sorted_token_ids, expert_ids,
            num_tokens_post_padded, mul_routed_weight, top_k, config,
            compute_type):
    """Trimmed `invoke_fused_moe_kernel`: bf16/fp16 only, no quant, no bias."""
    EM = sorted_token_ids.size(0)
    if A.size(0) < config["BLOCK_SIZE_M"]:
        EM = min(EM, A.size(0) * top_k * config["BLOCK_SIZE_M"])

    def grid(META):
        return (triton.cdiv(EM, META["BLOCK_SIZE_M"]) *
                triton.cdiv(B.size(1), META["BLOCK_SIZE_N"]), )

    fused_moe_kernel[grid](
        A, B, C,
        None,          # b_bias
        None, None,    # a_scale, b_scale
        topk_weights,
        sorted_token_ids, expert_ids, num_tokens_post_padded,
        B.size(1), B.size(2), EM, A.size(0) * top_k,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(2), B.stride(1),
        C.stride(1), C.stride(2),
        0, 0,          # a_scale strides
        0, 0, 0,       # b_scale strides
        0, 0,          # b_bias strides
        group_n=0, group_k=0,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        use_fp8_w8a8=False, use_int8_w8a8=False, use_int8_w8a16=False,
        per_channel_quant=False, HAS_BIAS=False,
        **config,
    )


def default_config(M, E, N, K, topk):
    """vLLM `get_default_config`, bf16 branch only."""
    if M <= E:
        return dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=32, BLOCK_SIZE_K=64,
                    GROUP_SIZE_M=1, num_warps=4, num_stages=4)
    return dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32,
                GROUP_SIZE_M=8, num_warps=4, num_stages=4)


def fused_experts(hidden_states, w1, w2, topk_weights, topk_ids,
                  config=None, activation="silu"):
    """One MoE FFN over already-routed tokens.

      hidden_states : [M, K]        bf16/fp16
      w1            : [E, 2*I, K]   gate and up stacked on dim 1
      w2            : [E, K, I]
      topk_weights  : [M, topk]     float32
      topk_ids      : [M, topk]     int32/int64

    Returns [M, K].
    """
    M, K = hidden_states.shape
    E, two_I, _ = w1.shape
    I = two_I // 2
    topk = topk_ids.size(1)
    dev, dt = hidden_states.device, hidden_states.dtype
    compute_type = tl.bfloat16 if dt == torch.bfloat16 else tl.float16

    if config is None:
        config = default_config(M, E, I, K, topk)

    sorted_ids, expert_ids, n_pad = moe_align_block_size(
        topk_ids, config["BLOCK_SIZE_M"], E)

    c1 = torch.empty((M, topk, two_I), device=dev, dtype=dt)
    c2 = torch.empty((M * topk, I), device=dev, dtype=dt)
    c3 = torch.empty((M, topk, K), device=dev, dtype=dt)

    _invoke(hidden_states, w1, c1, topk_weights, sorted_ids, expert_ids, n_pad,
            False, topk, config, compute_type)

    g = c1.view(-1, two_I)
    if activation == "silu":
        torch.mul(F.silu(g[:, :I]), g[:, I:], out=c2)
    elif activation == "gelu":
        torch.mul(F.gelu(g[:, :I]), g[:, I:], out=c2)
    else:
        raise ValueError(activation)

    _invoke(c2, w2, c3, topk_weights, sorted_ids, expert_ids, n_pad,
            True, 1, config, compute_type)

    return c3.sum(dim=1)


class PreparedMoE:
    """fused_experts with the align step and the buffers hoisted out.

    B1 measures energy under a locked clock with CUDA graphs, which needs a
    replayable, allocation-free body. It also should not charge the GEMMs for
    our pure-PyTorch `moe_align_block_size` -- vLLM's is a small fused CUDA op,
    ours is argsort+bincount, so including it would make the baseline look
    weaker than the real thing. Measure `run()` for the grouped GEMM, and
    `align_ms()` separately if you want the dispatch cost too.
    """

    def __init__(self, hidden_states, w1, w2, topk_weights, topk_ids, config,
                 activation="silu"):
        self.x, self.w1, self.w2 = hidden_states, w1, w2
        self.tw, self.ti = topk_weights, topk_ids
        self.cfg, self.act = config, activation
        M, K = hidden_states.shape
        E, two_I, _ = w1.shape
        self.I = two_I // 2
        self.topk = topk_ids.size(1)
        dev, dt = hidden_states.device, hidden_states.dtype
        self.ct = tl.bfloat16 if dt == torch.bfloat16 else tl.float16
        self.sorted_ids, self.expert_ids, self.n_pad = moe_align_block_size(
            topk_ids, config["BLOCK_SIZE_M"], E)
        self.c1 = torch.empty((M, self.topk, two_I), device=dev, dtype=dt)
        self.c2 = torch.empty((M * self.topk, self.I), device=dev, dtype=dt)
        self.c3 = torch.empty((M, self.topk, K), device=dev, dtype=dt)
        self.out = torch.empty((M, K), device=dev, dtype=dt)

    def run(self):
        _invoke(self.x, self.w1, self.c1, self.tw, self.sorted_ids,
                self.expert_ids, self.n_pad, False, self.topk, self.cfg, self.ct)
        g = self.c1.view(-1, 2 * self.I)
        f = F.silu if self.act == "silu" else F.gelu
        torch.mul(f(g[:, :self.I]), g[:, self.I:], out=self.c2)
        _invoke(self.c2, self.w2, self.c3, self.tw, self.sorted_ids,
                self.expert_ids, self.n_pad, True, 1, self.cfg, self.ct)
        torch.sum(self.c3, dim=1, out=self.out)
        return self.out
