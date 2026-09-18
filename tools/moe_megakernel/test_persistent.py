"""Correctness of the persistent kernel, whole and partitioned, against the vLLM path."""
import sys, os, json, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vendor.fused_moe_triton import PreparedMoE
from persistent_moe import PersistentMoE
from b1_bench import SHAPES, load_counts, make_inputs

spec = SHAPES["qwen"]
counts = load_counts(spec, 0, 0)
x, w1, w2, tw, ti, M = make_inputs(spec, counts)
cfg = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "data", "b1_config_qwen.json")))
ref = PreparedMoE(x, w1, w2, tw, ti, cfg).run().float()

p = PersistentMoE(x, w1, w2, tw, ti, cfg)
print(f"row-blocks live: {p.live_m.numel()}   SMs: {p.n_sm}")

got = p.run().float()
d = (got - ref).abs().max().item() / ref.abs().max().item()
print(f"  persistent, one launch over all tiles      relmax {d:.2e}  "
      f"{'ok' if d < 1e-6 else 'FAIL'}")
ok = d < 1e-6

# partition by expert load: heavy experts to the big partition, light to the small one
cnt = torch.from_numpy(counts).to(p.live_m.device)[p.rowblock_expert.long()]
order = torch.argsort(cnt, descending=True)
for frac in (0.5, 0.75):
    cut = int(len(order) * frac)
    a, b = p.live_m[order[:cut]], p.live_m[order[cut:]]
    n_a = max(1, min(p.n_sm - 1, round(p.n_sm * frac)))
    st = [torch.cuda.Stream(), torch.cuda.Stream()]
    p._tile_cache.clear()
    got = p.run(partition=[(a, n_a), (b, p.n_sm - n_a)], streams=st).float()
    d = (got - ref).abs().max().item() / ref.abs().max().item()
    print(f"  partitioned {n_a}+{p.n_sm-n_a} SMs, {cut}/{len(order)} row-blocks  "
          f"relmax {d:.2e}  {'ok' if d < 1e-6 else 'FAIL'}")
    ok &= d < 1e-6
print("ALL OK" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
