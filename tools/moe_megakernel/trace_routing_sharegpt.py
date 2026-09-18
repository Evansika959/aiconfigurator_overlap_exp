#!/usr/bin/env python3
"""Routing traces on real serving prompts, for any MoE family transformers can load.

Counts how many tokens each expert receives, per layer, on ShareGPT prompts. Works across
families because the router is located structurally rather than by attribute name.

WHY SHAREGPT. wikitext-103 is encyclopedia prose; real serving traffic is chat --
instructions, code, short questions. ShareGPT is what vLLM and TensorRT-LLM benchmark on,
so the prompt distribution is the one a deployment actually sees.

WHY TRANSFORMERS AND NOT A SERVING FRAMEWORK. The router is an nn.Linear and top-k is
recomputed exactly as the block does it, so the routing decisions here are the model's own
and a server would reproduce them. Timing and energy from this path are NOT
representative -- HF loops over every expert in Python -- and must never be quoted.

PADDING MUST BE MASKED. Prompts have wildly different lengths, so a rectangular batch is
mostly padding, and the router runs on pad positions too. Counting those would flatten the
distribution towards uniform and quietly manufacture the "no skew" answer. The hook takes
the live attention mask and drops padded positions.

Model is sharded across the GPUs with device_map="auto" (30.5B params in bf16 is 61 GB).

  python3 trace_routing_sharegpt.py --batch 1,8,32 --decode-steps 64
"""
import argparse, json, os, random, time
os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
_MASK = {"m": None}          # live attention mask, read by the router hooks


def sharegpt_prompts(tok, n, max_len, seed=0):
    """First human turn of each conversation, rendered through the model's own chat
    template -- that is the string a server would actually receive."""
    p = open(os.path.join(HERE, "data", "_sharegpt_path.txt")).read().strip()
    d = json.load(open(p))
    rng = random.Random(seed)
    rng.shuffle(d)
    out = []
    for conv in d:
        turns = conv.get("conversations") or []
        first = next((t for t in turns if t.get("from") in ("human", "user")), None)
        if not first or len(first["value"]) < 40:
            continue
        # base models have no chat template; the point of the control is the CONTENT
        # distribution (chat vs encyclopedia), so fall back to the raw turn
        try:
            s = tok.apply_chat_template([{"role": "user", "content": first["value"]}],
                                        tokenize=False, add_generation_prompt=True)
        except Exception:
            s = first["value"]
        ids = tok(s, return_tensors="pt", truncation=True,
                  max_length=max_len).input_ids[0]
        if ids.numel() >= 16:
            out.append(ids)
        if len(out) >= n:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    ap.add_argument("--batch", default="1,8,32")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--decode-steps", type=int, default=64)
    ap.add_argument("--gpu-mem", default=None,
                    help="per-GPU cap for device_map, e.g. 33GiB. Needed for models whose "
                         "weights nearly fill the GPUs: device_map fills each card to the "
                         "brim and leaves nothing for activations. Qwen3-Next also falls "
                         "back to a torch gated-delta-rule that materialises float32 "
                         "copies of q/k/v, which is what actually OOMs.")
    ap.add_argument("--repeats", type=int, default=1,
                    help="independent prompt batches, each with its own seed. One batch "
                         "of 64 is 0.07% of ShareGPT and gives no error bar at all; the "
                         "prompt draw is the only source of uncertainty here.")
    ap.add_argument("--out", default=os.path.join(HERE, "data", "routing_qwen.npz"))
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"                       # so decode continues from real tokens
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = {}
    if a.gpu_mem:
        import torch as _t
        kw["max_memory"] = {i: a.gpu_mem for i in range(_t.cuda.device_count())}
        kw["max_memory"]["cpu"] = "280GiB"
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16,
                                                 device_map="auto", **kw)
    model.eval()
    cfg = model.config
    E = (getattr(cfg, "num_experts", None) or getattr(cfg, "num_local_experts", None)
         or getattr(cfg, "n_routed_experts", None))
    K, L = cfg.num_experts_per_tok, cfg.num_hidden_layers
    print(f"{a.model}: {L} layers, {E} experts, top-{K}")
    print(f"  sharded over {len({str(p.device) for p in model.parameters()})} devices\n")

    bucket = {}

    def mk(li):
        def hook(_m, _inp, out):
            sel = out.float().softmax(-1).topk(K, dim=-1).indices     # (tokens, K)
            m = _MASK["m"]
            if m is not None:
                keep = m.reshape(-1).to(sel.device).bool()
                if keep.numel() == sel.shape[0]:
                    sel = sel[keep]                # drop padded positions
            c = torch.bincount(sel.reshape(-1), minlength=E)
            bucket.setdefault(li, []).append(c.to(torch.int32).cpu().numpy())
        return hook

    import torch.nn as nn
    n_hooked = 0
    for li, layer in enumerate(model.model.layers):
        # out_features == E alone is not enough. Qwen3-Next collides twice:
        #   moe_intermediate_size == num_experts == 512, so every expert's gate_proj and
        #     up_proj matches (1027 candidates), and
        #   4 KV heads x 128 head_dim == 512, so self_attn.k_proj and v_proj match too.
        # The router is the only such Linear that is neither inside the expert stack nor
        # part of attention, so both are excluded by name. The assert below refuses to
        # guess if that still leaves more than one -- silently hooking an expert's
        # gate_proj would produce routing counts that look entirely plausible and are
        # entirely wrong.
        BAD = ("expert", "attn", "attention")
        cand = [m for nm, m in layer.named_modules()
                if isinstance(m, nn.Linear) and m.out_features == E and m.bias is None
                and not any(b in nm.lower() for b in BAD)]
        if len(cand) == 1:
            cand[0].register_forward_hook(mk(li)); n_hooked += 1
        elif len(cand) > 1:
            raise RuntimeError(f"layer {li}: {len(cand)} Linears with out_features={E}; "
                               "cannot tell which is the router")
    if not n_hooked:
        raise RuntimeError("no router found -- no Linear has out_features == num_experts")
    print(f"  hooked {n_hooked} routers\n")
    out, meta = {}, dict(model=a.model, experts=E, top_k=K, layers=L,
                         max_len=a.max_len, decode_steps=a.decode_steps,
                         dataset="ShareGPT_V3 (first human turn, chat template)")

    for B in [int(x) for x in a.batch.split(",")]:
      pre_all, dec_all = [], []
      for rep in range(a.repeats):
        seqs = sharegpt_prompts(tok, B, a.max_len, seed=rep)
        n_tok = sum(int(s.numel()) for s in seqs)
        pad = tok.pad_token_id
        Lmax = max(int(s.numel()) for s in seqs)
        ids = torch.full((B, Lmax), pad, dtype=torch.long)
        msk = torch.zeros((B, Lmax), dtype=torch.long)
        for i, s in enumerate(seqs):                 # left padding
            ids[i, Lmax - s.numel():] = s
            msk[i, Lmax - s.numel():] = 1
        dev = next(model.parameters()).device
        ids, msk = ids.to(dev), msk.to(dev)

        bucket.clear(); _MASK["m"] = msk
        t0 = time.time()
        with torch.no_grad():
            o = model(ids, attention_mask=msk, use_cache=True)
        pre = np.stack([bucket[li][0] for li in sorted(bucket)])
        pre_all.append(pre)
        got, want = int(pre.sum()), n_tok * K * len(bucket)
        assert got == want, (got, want)

        _MASK["m"] = None                            # decode: one real token per sequence
        past, nxt = o.past_key_values, o.logits[:, -1:].argmax(-1)
        am = msk
        dec = []
        with torch.no_grad():
            for _ in range(a.decode_steps):
                bucket.clear()
                am = torch.cat([am, torch.ones((B, 1), dtype=am.dtype, device=dev)], 1)
                o = model(nxt, attention_mask=am, past_key_values=past, use_cache=True)
                past, nxt = o.past_key_values, o.logits[:, -1:].argmax(-1)
                dec.append(np.stack([bucket[li][0] for li in sorted(bucket)]))
        dec = np.stack(dec)
        dec_all.append(dec)
        assert int(dec.sum()) == a.decode_steps * len(bucket) * B * K
        print(f"  B={B:<3d} rep {rep+1}/{a.repeats}  {n_tok:6d} real tokens "
              f"({n_tok/(B*Lmax)*100:.0f}% of the padded batch)"
              f"   [{time.time()-t0:.0f}s]", flush=True)
        del past, o
        torch.cuda.empty_cache()
      # (repeats, layers, E); the analysis reshapes to (-1, E) and can also split by rep
      out[f"prefill_b{B}"] = np.stack(pre_all)
      out[f"decode_b{B}"] = np.stack(dec_all)

    meta["moe_layers"] = len(bucket)
    meta["repeats"] = a.repeats
    np.savez_compressed(a.out, meta=json.dumps(meta), **out)
    print(f"\nwrote {a.out}  ({len(out)} arrays, {meta['moe_layers']} MoE layers)")


if __name__ == "__main__":
    main()
