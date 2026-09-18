#!/usr/bin/env python3
"""A persistent grouped-GEMM MoE kernel, and the tile-partitioning it makes possible.

WHY THIS EXISTS

`vendor/fused_moe_triton.py` launches one block per output tile and lets the hardware
scheduler place them. That is work-conserving by construction: a block that finishes
immediately gets the next tile, so a lightly-routed expert never leaves an SM idle. It
is also, for the same reason, useless as a vehicle for spatial DVFS -- there is no
persistent association between an SM and the work it runs, so there is nothing to put
in a slow voltage domain.

A persistent kernel launches exactly GRID blocks -- one per SM in its partition -- and
each block pulls tiles from a list it was handed. Two such kernels launched on two
streams with GRID_A + GRID_B = 108 blocks occupy disjoint sets of SMs. `overlap_exp`
established on this box that 1 resident CTA == 1 SM (ncu-verified), which is what makes
that partition real rather than nominal.

WHAT IT BUYS AND WHAT IT COSTS

Buys: the "light" experts can be confined to a named set of SMs, which on hypothetical
per-partition-DVFS hardware could run at a lower voltage.

Costs: makespan. The two partitions no longer share a queue, so the layer finishes when
the SLOWER of the two finishes, and any misallocation shows up directly as latency.

That trade is measurable TODAY, at a single clock, with no DVFS hardware at all --
which is the point of `pin_vs_queue.py`. If pinning alone costs more than a second
voltage domain could ever save, B2 is dead and we found out cheaply.

The MAC loop below is the same arithmetic as the vendored vLLM kernel (bf16/fp16 only,
no quantisation paths); the scheduling around it is what differs.
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from vendor.fused_moe_triton import moe_align_block_size


@triton.jit
def persistent_moe_kernel(
        a_ptr, b_ptr, c_ptr, topk_weights_ptr,
        sorted_token_ids_ptr, expert_ids_ptr, tile_ids_ptr,
        N, K, num_valid_tokens, n_tiles, tiles_n,
        stride_am, stride_ak, stride_be, stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
        MUL_ROUTED_WEIGHT: tl.constexpr, top_k: tl.constexpr,
        compute_type: tl.constexpr):
    """One block per SM; each walks its slice of `tile_ids` with a grid-stride loop.

    `tile_ids` is an explicit list of flat tile indices (row-block * tiles_n + col-block)
    that THIS launch owns. Handing the list in rather than deriving it from program_id
    is what lets the caller split the layer's tiles between two concurrent launches on
    any rule it likes -- by expert, by expert load, by anything.
    """
    # GRID STRIDE, NOT CONTIGUOUS CHUNKS. Every tile costs the same, so both split the
    # work evenly; the difference is entirely locality, and it is not the locality one
    # would guess. Contiguous chunks give each block temporal reuse of one expert's B
    # panel -- and measured 7.88 ms against the grid stride's 5.04 ms, 56% worse. The
    # reason is that L2 is shared across all 108 SMs: with a grid stride every block is
    # working near the same tile index at any instant, so only a handful of expert
    # panels are live in L2 at once; with contiguous chunks 108 different experts are
    # live simultaneously and L2 thrashes across 1.12 GiB of weights. Inter-block
    # locality beats intra-block locality here.
    start = tl.program_id(0)
    stride = tl.num_programs(0)
    for idx in range(start, n_tiles, stride):
        t = tl.load(tile_ids_ptr + idx)
        pid_m = t // tiles_n
        pid_n = t % tiles_n

        offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
        offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
        token_mask = offs_token < num_valid_tokens
        off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

        if off_experts >= 0:
            offs_bn = (pid_n * BLOCK_SIZE_N +
                       tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
            offs_k = tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am +
                              offs_k[None, :] * stride_ak)
            b_ptrs = b_ptr + off_experts * stride_be + (
                offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

            acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
                a = tl.load(a_ptrs,
                            mask=token_mask[:, None] &
                            (offs_k[None, :] < K - k * BLOCK_SIZE_K), other=0.0)
                b = tl.load(b_ptrs,
                            mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
                acc = tl.dot(a, b, acc)
                a_ptrs += BLOCK_SIZE_K * stride_ak
                b_ptrs += BLOCK_SIZE_K * stride_bk

            if MUL_ROUTED_WEIGHT:
                w = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
                acc = acc * w[:, None]
            acc = acc.to(compute_type)
        else:
            acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)

        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
        tl.store(c_ptrs, acc, mask=token_mask[:, None] & (offs_cn[None, :] < N))


def _launch(A, B, C, tw, sorted_ids, expert_ids, tile_ids, grid, mul_w, top_k,
            cfg, ct, stream=None):
    n_tiles = tile_ids.numel()
    if n_tiles == 0:
        return
    tiles_n = triton.cdiv(B.size(1), cfg["BLOCK_SIZE_N"])
    with torch.cuda.stream(stream) if stream is not None else _null():
        persistent_moe_kernel[(grid, )](
            A, B, C, tw, sorted_ids, expert_ids, tile_ids,
            B.size(1), B.size(2), A.size(0) * top_k, n_tiles, tiles_n,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(2), B.stride(1),
            C.stride(1), C.stride(2),
            BLOCK_SIZE_M=cfg["BLOCK_SIZE_M"], BLOCK_SIZE_N=cfg["BLOCK_SIZE_N"],
            BLOCK_SIZE_K=cfg["BLOCK_SIZE_K"], GROUP_SIZE_M=cfg["GROUP_SIZE_M"],
            MUL_ROUTED_WEIGHT=mul_w, top_k=top_k, compute_type=ct,
            num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


class PersistentMoE:
    """Persistent-kernel MoE layer with a caller-chosen tile partition.

    `partition` is None (one launch over every tile, all SMs) or a list of
    (tile_ids, n_blocks) pairs, each run on its own stream. n_blocks is how many SMs
    that partition gets -- one resident block per SM.
    """

    def __init__(self, hidden_states, w1, w2, topk_weights, topk_ids, config,
                 activation="silu"):
        self.x, self.w1, self.w2 = hidden_states, w1, w2
        self.tw, self.ti, self.cfg, self.act = topk_weights, topk_ids, config, activation
        M, K = hidden_states.shape
        E, two_I, _ = w1.shape
        self.M, self.K, self.E, self.I = M, K, E, two_I // 2
        self.topk = topk_ids.size(1)
        dev, dt = hidden_states.device, hidden_states.dtype
        self.ct = tl.bfloat16 if dt == torch.bfloat16 else tl.float16
        self.sorted_ids, self.expert_ids, _ = moe_align_block_size(
            topk_ids, config["BLOCK_SIZE_M"], E)
        self.c1 = torch.empty((M, self.topk, two_I), device=dev, dtype=dt)
        self.c2 = torch.empty((M * self.topk, self.I), device=dev, dtype=dt)
        self.c3 = torch.empty((M, self.topk, K), device=dev, dtype=dt)
        self.out = torch.empty((M, K), device=dev, dtype=dt)
        # a row-block is live iff its expert id is >= 0; dead ones only write zeros,
        # which the second GEMM's own mask already handles, so they are dropped
        self.live_m = (self.expert_ids >= 0).nonzero(as_tuple=True)[0]
        self.rowblock_expert = self.expert_ids[self.live_m]
        self.n_sm = torch.cuda.get_device_properties(dev).multi_processor_count
        self._tile_cache = {}

    def tiles(self, which, m_blocks=None, key=None):
        """Flat tile ids for GEMM 1 ('g1') or GEMM 2 ('g2'), optionally restricted to a
        subset of row-blocks. Cached: building these must not be inside the timed path,
        and in a real kernel this list would be produced by the align step anyway."""
        ck = (which, key)
        if ck in self._tile_cache:
            return self._tile_cache[ck]
        n = 2 * self.I if which == "g1" else self.K
        tn = triton.cdiv(n, self.cfg["BLOCK_SIZE_N"])
        m = self.live_m if m_blocks is None else m_blocks
        t = (m[:, None] * tn +
             torch.arange(tn, device=m.device)[None, :]).reshape(-1).int().contiguous()
        self._tile_cache[ck] = t
        return t

    def run(self, partition=None, streams=None, blocks_per_sm=1):
        """blocks_per_sm > 1 launches several resident blocks per SM. The non-persistent
        kernel gets latency hiding for free -- the hardware keeps more than one block
        resident and overlaps one tile's epilogue with the next tile's loads. A
        persistent launch of exactly one block per SM gives that up, so this is the
        knob that tests whether the gap is occupancy rather than scheduling."""
        cfg, ct = self.cfg, self.ct
        for which, A, B, Cc, mul, tk in (
                ("g1", self.x, self.w1, self.c1, False, self.topk),
                ("g2", self.c2, self.w2, self.c3, True, 1)):
            if partition is None:
                _launch(A, B, Cc, self.tw, self.sorted_ids, self.expert_ids,
                        self.tiles(which), self.n_sm * blocks_per_sm, mul, tk, cfg, ct)
            else:
                cur = torch.cuda.current_stream()
                for st in streams:
                    st.wait_stream(cur)          # respect the previous GEMM / activation
                for i, ((mb, nblk), st) in enumerate(zip(partition, streams)):
                    _launch(A, B, Cc, self.tw, self.sorted_ids, self.expert_ids,
                            self.tiles(which, mb, key=i), nblk * blocks_per_sm, mul,
                            tk, cfg, ct, stream=st)
                for st in streams:
                    cur.wait_stream(st)
            if which == "g1":
                g = self.c1.view(-1, 2 * self.I)
                f = F.silu if self.act == "silu" else F.gelu
                torch.mul(f(g[:, :self.I]), g[:, self.I:], out=self.c2)
        torch.sum(self.c3, dim=1, out=self.out)
        return self.out
