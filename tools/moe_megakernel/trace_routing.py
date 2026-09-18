#!/usr/bin/env python3
"""Phase 0a -- how imbalanced is MoE expert routing, actually?

The whole project rests on there being real, exploitable skew in tokens-per-expert. This
script measures it on a model that exists, running on real text, rather than assuming a
synthetic imbalance knob.

WHAT IS HOOKED. A forward hook on each layer's router `gate` (an nn.Linear), from which
top-k is recomputed exactly as the block does it. Hooking the gate rather than the block
keeps this independent of how HuggingFace happens to implement dispatch -- which matters,
because HF loops over all 64 experts in Python. That loop is fine here (the ROUTER is
exact) but it is NOT a realistic energy baseline and no timing from this script should
ever be quoted as one.

PREFILL AND DECODE ARE DIFFERENT REGIMES and are recorded separately:
  prefill  one forward over B x L tokens -- thousands of tokens per layer, so the law of
           large numbers has a chance to flatten the histogram
  decode   B tokens per step -- tiny samples, where skew is worst and where a megakernel
           actually has idle SMs to sell

Counts are stored raw, per (regime, layer, step, expert), so any imbalance metric can be
computed later without re-running the model.

  python3 trace_routing.py --batch 1,4,16,64 --prefill-len 512 --decode-steps 64
"""
import argparse, json, os, time
os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def load_text(tok, n_seq, seq_len, seed=0):
    """Real corpus text, not hand-written prompts: the routing distribution is the claim,
    so the token distribution driving it has to be representative."""
    from datasets import load_dataset
    d = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    rng = np.random.default_rng(seed)
    paras = [t for t in d["text"] if len(t.strip()) > 200]
    ids, tries = [], 0
    while len(ids) < n_seq and tries < 20 * n_seq:
        tries += 1
        j = int(rng.integers(0, len(paras)))
        chunk = " ".join(paras[j:j + 40])
        e = tok(chunk, return_tensors="pt").input_ids[0]
        if e.numel() >= seq_len:
            ids.append(e[:seq_len])
    if len(ids) < n_seq:
        raise RuntimeError(f"only {len(ids)}/{n_seq} sequences of {seq_len} tokens")
    return torch.stack(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    ap.add_argument("--batch", default="1,4,16,64")
    ap.add_argument("--prefill-len", type=int, default=512)
    ap.add_argument("--decode-steps", type=int, default=64)
    ap.add_argument("--out", default=os.path.join(HERE, "data", "routing_olmoe.npz"))
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16,
                                                 device_map={"": 0})
    model.eval()
    cfg = model.config
    E, K, L = cfg.num_experts, cfg.num_experts_per_tok, cfg.num_hidden_layers
    print(f"{a.model}: {L} layers, {E} experts, top-{K}\n")

    # one hook per router. `bucket` is swapped by the driver between phases so a hook does
    # not need to know which regime it is in.
    bucket = {}
    def mk(li):
        def hook(_m, _inp, out):
            sel = out.float().softmax(-1).topk(K, dim=-1).indices     # (tokens, K)
            c = torch.bincount(sel.reshape(-1), minlength=E)
            bucket.setdefault(li, []).append(c.to(torch.int32).cpu().numpy())
        return hook
    for li, layer in enumerate(model.model.layers):
        layer.mlp.gate.register_forward_hook(mk(li))

    out, meta = {}, dict(model=a.model, experts=E, top_k=K, layers=L,
                         prefill_len=a.prefill_len, decode_steps=a.decode_steps)
    for B in [int(x) for x in a.batch.split(",")]:
        ids = load_text(tok, B, a.prefill_len).cuda()
        # ---- prefill: one forward, B*L tokens routed per layer ----
        bucket.clear()
        t0 = time.time()
        with torch.no_grad():
            o = model(ids, use_cache=True)
        torch.cuda.synchronize()
        pre = np.stack([bucket[li][0] for li in range(L)])            # (L, E)
        out[f"prefill_b{B}"] = pre
        # every token picks exactly K experts in every layer -- a cheap total check that
        # the hook fired once per layer and nothing was double counted
        want = L * B * a.prefill_len * K
        assert pre.sum() == want, (pre.sum(), want)

        # ---- decode: B tokens routed per layer per step ----
        bucket.clear()
        past, nxt = o.past_key_values, o.logits[:, -1:].argmax(-1)
        dec = []
        with torch.no_grad():
            for s in range(a.decode_steps):
                bucket.clear()
                o = model(nxt, past_key_values=past, use_cache=True)
                past, nxt = o.past_key_values, o.logits[:, -1:].argmax(-1)
                dec.append(np.stack([bucket[li][0] for li in range(L)]))
        torch.cuda.synchronize()
        dec = np.stack(dec)                                            # (steps, L, E)
        out[f"decode_b{B}"] = dec
        want = a.decode_steps * L * B * K
        assert dec.sum() == want, (dec.sum(), want)
        print(f"  B={B:<3d} prefill {B*a.prefill_len:6d} tok/layer   "
              f"decode {a.decode_steps} steps x {B} tok   [{time.time()-t0:.1f}s]")
        del past, o
        torch.cuda.empty_cache()

    np.savez_compressed(a.out, meta=json.dumps(meta), **out)
    print(f"\nwrote {a.out}  ({len(out)} arrays)")


if __name__ == "__main__":
    main()
