"""Correctness check for vendor/fused_moe_triton.py against a naive loop."""
import sys, torch, torch.nn.functional as F
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from vendor.fused_moe_triton import fused_experts


def ref(x, w1, w2, tw, ti):
    M, K = x.shape
    out = torch.zeros(M, K, dtype=torch.float32, device=x.device)
    I = w1.size(1) // 2
    for e in range(w1.size(0)):
        m, k = (ti == e).nonzero(as_tuple=True)
        if m.numel() == 0:
            continue
        g = x[m].to(torch.float32) @ w1[e].to(torch.float32).T
        h = F.silu(g[:, :I]) * g[:, I:]
        y = h @ w2[e].to(torch.float32).T
        out.index_add_(0, m, y * tw[m, k][:, None].to(torch.float32))
    return out


def one(M, E, K, I, topk, seed=0):
    torch.manual_seed(seed)
    dev, dt = "cuda", torch.bfloat16
    x = torch.randn(M, K, device=dev, dtype=dt) / K**0.5
    w1 = torch.randn(E, 2 * I, K, device=dev, dtype=dt) / K**0.5
    w2 = torch.randn(E, K, I, device=dev, dtype=dt) / I**0.5
    logits = torch.randn(M, E, device=dev)
    tw, ti = torch.topk(torch.softmax(logits, -1), topk, dim=-1)
    tw = tw.float().contiguous()
    ti = ti.int().contiguous()

    got = fused_experts(x, w1, w2, tw, ti).float()
    exp = ref(x, w1, w2, tw, ti)
    err = (got - exp).abs().max().item()
    rel = err / exp.abs().max().item()
    ok = rel < 3e-2
    print(f"  M={M:<6d} E={E:<4d} K={K:<5d} I={I:<5d} topk={topk}  "
          f"relmax={rel:.2e}  {'ok' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    print("fused_experts vs naive per-expert loop (bf16)")
    allok = True
    for a in [(8, 128, 2048, 768, 8),        # M <= E, small-batch config path
              (64, 128, 2048, 768, 8),
              (512, 128, 2048, 768, 8),      # Qwen3-30B-A3B shape
              (4096, 128, 2048, 768, 8),
              (1024, 64, 2048, 1024, 8),     # OLMoE shape
              (333, 40, 1536, 512, 8)]:      # odd M, Granite-ish
        allok &= one(*a)
    print("ALL OK" if allok else "SOME FAILED")
    sys.exit(0 if allok else 1)
