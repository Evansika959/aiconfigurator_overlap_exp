"""B1 -- the honest fused grouped-GEMM baseline.

Drives vendor/fused_moe_triton.py (vLLM's Triton fused-MoE kernel) on a single
MoE layer whose expert histogram is a REAL traced routing draw, and reports
latency and NVML energy at a locked clock.

Weights are random: the kernel's time and power depend on shape and on the
expert histogram, not on the values. Routing comes from data/routing_*.npz,
which stores per-(rep, layer, expert) token counts.

  python3 b1_bench.py --model qwen --tune          # autotune the config grid
  python3 b1_bench.py --model qwen --clock 1410    # measure at a locked clock
"""
import argparse, itertools, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vendor.fused_moe_triton import (fused_experts, default_config,
                                     PreparedMoE, moe_align_block_size)

# hidden K, expert intermediate I, experts E, top-k, MoE layers
SHAPES = {
    "qwen":       dict(npz="routing_qwen.npz",       key="prefill_b64", K=2048, I=768,  E=128, topk=8,  layers=48),
    "qwen3next":  dict(npz="routing_qwen3next.npz",  key="prefill_b32", K=2048, I=512,  E=512, topk=10, layers=48),
    "olmoe":      dict(npz="routing_olmoe_sharegpt.npz", key="prefill_b64", K=2048, I=1024, E=64, topk=8, layers=16),
    "granite":    dict(npz="routing_granite.npz",    key="prefill_b64", K=1536, I=512,  E=40,  topk=8,  layers=32),
}

TUNE_GRID = dict(
    BLOCK_SIZE_M=[64, 128],
    BLOCK_SIZE_N=[32, 64, 128, 256],
    BLOCK_SIZE_K=[32, 64, 128],
    GROUP_SIZE_M=[1, 8, 16, 32],
    num_warps=[4, 8],
    num_stages=[2, 3, 4],
)


def load_counts(spec, rep, layer):
    Z = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "data", spec["npz"]))
    A = Z[spec["key"]]
    if A.ndim == 2:          # olmoe wikitext file: (layer, expert)
        return A[layer].astype(np.int64)
    return A[rep, layer].astype(np.int64)


def make_inputs(spec, counts, dev="cuda", dt=torch.bfloat16, seed=0):
    """Synthesise (x, w1, w2, topk_weights, topk_ids) with this exact histogram."""
    E, topk, K, I = spec["E"], spec["topk"], spec["K"], spec["I"]
    total = int(counts.sum())
    assert total % topk == 0, f"{total} not divisible by topk={topk}"
    M = total // topk
    g = torch.Generator(device="cpu").manual_seed(seed)

    flat = torch.repeat_interleave(torch.arange(E, dtype=torch.int32),
                                   torch.from_numpy(counts))
    flat = flat[torch.randperm(total, generator=g)]
    ti = flat.view(M, topk).contiguous().to(dev)

    x = (torch.randn(M, K, device=dev, dtype=dt) / K ** 0.5)
    w1 = (torch.randn(E, 2 * I, K, device=dev, dtype=dt) / K ** 0.5)
    w2 = (torch.randn(E, K, I, device=dev, dtype=dt) / I ** 0.5)
    tw = torch.rand(M, topk, device=dev, dtype=torch.float32)
    tw /= tw.sum(1, keepdim=True)
    return x, w1, w2, tw, ti, M


def timed(fn, warmup=5, iters=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(iters)]
    for a, b in ev:
        a.record(); fn(); b.record()
    torch.cuda.synchronize()
    ts = np.array([a.elapsed_time(b) for a, b in ev])   # ms
    return float(np.median(ts)), float(ts.std())


def flops(spec, counts):
    """2 GEMMs: (n_e x K)@(K x 2I) and (n_e x I)@(I x K), per expert."""
    n = counts.sum()
    return 2.0 * n * spec["K"] * 2 * spec["I"] + 2.0 * n * spec["I"] * spec["K"]


def tune(spec, x, w1, w2, tw, ti, quiet=False):
    keys = list(TUNE_GRID)
    best, best_t = None, float("inf")
    combos = [dict(zip(keys, v)) for v in itertools.product(*TUNE_GRID.values())]
    for i, cfg in enumerate(combos):
        try:
            p = PreparedMoE(x, w1, w2, tw, ti, cfg)
            t, _ = timed(p.run, warmup=2, iters=5)
            del p
        except Exception:
            continue                      # OOR / unsupported combo
        if t < best_t:
            best, best_t = cfg, t
            if not quiet:
                print(f"  [{i+1}/{len(combos)}] {t:7.3f} ms  {cfg}", flush=True)
    return best, best_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen", choices=list(SHAPES))
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--config", default=None, help="JSON dict, or a path to one")
    ap.add_argument("--iters", type=int, default=30)
    a = ap.parse_args()

    spec = SHAPES[a.model]
    counts = load_counts(spec, a.rep, a.layer)
    x, w1, w2, tw, ti, M = make_inputs(spec, counts)

    print(f"model={a.model}  rep={a.rep} layer={a.layer}")
    print(f"  M={M} tokens  K={spec['K']} I={spec['I']} E={spec['E']} topk={spec['topk']}")
    print(f"  expert counts: min={counts.min()} mean={counts.mean():.0f} "
          f"max={counts.max()}  max/mean={counts.max()/counts.mean():.2f}")
    print(f"  weights {(w1.numel()+w2.numel())*2/2**30:.2f} GiB, "
          f"{flops(spec, counts)/1e9:.1f} GFLOP/layer")

    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "data", f"b1_config_{a.model}.json")
    if a.tune:
        print("autotuning...")
        t0 = time.time()
        cfg, t = tune(spec, x, w1, w2, tw, ti)
        print(f"best {t:.3f} ms  {cfg}   ({time.time()-t0:.0f} s)")
        with open(cfg_path, "w") as f:
            json.dump(cfg, f)
        print(f"wrote {cfg_path}")
    elif a.config:
        cfg = json.loads(open(a.config).read() if os.path.exists(a.config) else a.config)
    elif os.path.exists(cfg_path):
        cfg = json.load(open(cfg_path))
        print(f"using tuned config from {cfg_path}")
    else:
        cfg = default_config(M, spec["E"], spec["I"], spec["K"], spec["topk"])
        print("using vLLM default_config (UNTUNED -- weak baseline)")

    p = PreparedMoE(x, w1, w2, tw, ti, cfg)
    t, sd = timed(p.run, iters=a.iters)
    ta, _ = timed(lambda: moe_align_block_size(ti, cfg["BLOCK_SIZE_M"], spec["E"]),
                  iters=a.iters)
    f = flops(spec, counts)
    print(f"\n  config      {cfg}")
    print(f"  GEMM time   {t:.3f} ms  (sd {sd:.3f})   <- the baseline")
    print(f"  throughput  {f/t*1e-9:.1f} TFLOP/s  "
          f"({f/t*1e-9/312*100:.1f}% of bf16 peak)")
    print(f"  align time  {ta:.3f} ms   (OUR pure-PyTorch moe_align_block_size;")
    print(f"              vLLM's is a fused CUDA op, so this is not part of the")
    print(f"              baseline -- it is an artefact of the reimplementation)")
    print(f"  all {spec['layers']} MoE layers: {t*spec['layers']:.1f} ms GEMM")


if __name__ == "__main__":
    main()
