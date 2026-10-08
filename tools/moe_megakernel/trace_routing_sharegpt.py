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

TWO FAMILIES NEED MORE THAN A BIAS-FREE LINEAR (added 2026-10-01; nothing changes for
the families traced before them):

  GPT-OSS   the router is its own module with a weight AND a bias, and it returns
            (scores, indices). It is found by the shape of its weight, (experts, hidden),
            and the indices it returns are counted: the block's own top-k.
  Llama 4   the router is a bias-free Linear, found as before, but it returns
            (scores, logits). The top-k of either is the same set, and that is asserted.

Both compute EVERY expert on EVERY position in transformers (inputs repeated once per
expert), so the temporaries of one layer at 64 x 512 positions run a 40 GB card out of
memory. `--rows N`
forwards the same padded batch N rows at a time and adds the counts. Every row is the
tensor it would have been in the full batch, so the routing is the same up to
floating-point ties: on OLMoE, 16 rows at a time moves 0.12% of the prefill dispatches
against the whole batch at once. (Decode differs more, 2.5%, because a tie that flips one
generated token changes every token after it; the decode counts are a different sample of
the same process, not the same sample.)
"""
import argparse, json, os, random, time
os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
_MASK = {"m": None}          # live attention mask, read by the router hooks
_KEEP = {"sel": None}        # a dict here makes the hooks keep every position's experts
_SPARSE = {"on": False}      # Llama 4: compute the routed experts only (llama4_sparse_moe)


def sharegpt_prompts(tok, n, max_len, seed=0, template=True):
    """First human turn of each conversation, rendered through the model's own chat
    template -- that is the string a server would actually receive.

    `template=False` sends the raw turn instead. A control for families whose template
    is long: GPT-OSS prepends a 67-token system preamble to every prompt, 39% of all the
    tokens counted in a batch against 6-9% for the other families, and those tokens are
    identical in all 64 conversations."""
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
            if not template:
                raise ValueError
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


BAD = ("expert", "attn", "attention")     # module paths a router is never under


def decoder_layers(model):
    """The stack of decoder layers, wherever the family keeps it (a multimodal model
    keeps its text decoder under `language_model`)."""
    for path in ("model.layers", "language_model.model.layers"):
        m = model
        try:
            for name in path.split("."):
                m = getattr(m, name)
            return m
        except AttributeError:
            continue
    raise RuntimeError("no decoder layer stack found")


def find_routers(layer, E, H):
    """Candidate routers of one decoder layer; the caller refuses to guess unless there
    is exactly one.

    out_features == E alone is not enough. Qwen3-Next collides twice:
      moe_intermediate_size == num_experts == 512, so every expert's gate_proj and
        up_proj matches (1027 candidates), and
      4 KV heads x 128 head_dim == 512, so self_attn.k_proj and v_proj match too.
    The router is the only such Linear that is neither inside the expert stack nor part
    of attention, so both are excluded by name. Silently hooking an expert's gate_proj
    would produce routing counts that look entirely plausible and are entirely wrong.

    A family whose router is not a bias-free Linear (GPT-OSS: a module of its own, with a
    bias) is found by the one thing every router has: a weight of shape (experts,
    hidden)."""
    import torch.nn as nn
    named = [(nm, m) for nm, m in layer.named_modules() if not any(b in nm.lower() for b in BAD)]
    cand = [m for _, m in named
            if isinstance(m, nn.Linear) and m.out_features == E and m.bias is None]
    if not cand:
        cand = [m for _, m in named if not isinstance(m, nn.Linear)
                and torch.is_tensor(getattr(m, "weight", None))
                and tuple(m.weight.shape) == (E, H)]
    return cand


def selected(out, K, E):
    """The experts each position was sent to, (positions, K), from whatever the router
    returns: its logits (most families); a tuple that carries the indices themselves
    (GPT-OSS); or scores beside logits (Llama 4), whose top-k must be the same set."""
    if torch.is_tensor(out):
        return out.float().softmax(-1).topk(K, dim=-1).indices
    own = [t for t in out if torch.is_tensor(t) and not t.is_floating_point()
           and t.shape[-1] == K]
    if own:
        return own[0]                                  # the block's own top-k
    sets = [t.float().topk(K, dim=-1).indices.sort(-1).values
            for t in out if torch.is_tensor(t) and t.shape[-1] == E]
    if not sets or any(not torch.equal(x, sets[0]) for x in sets[1:]):
        raise RuntimeError("cannot read the selected experts off the router's output")
    return sets[0]


def llama4_sparse_moe():
    """Teach Llama 4's MoE to compute an expert only on the positions routed to it.

    transformers computes all 16 experts on every position. The 15 a position was not
    routed to get an INPUT multiplied by a score of exactly zero and return exactly zero
    (there is no bias, and SiLU(0) * 0 = 0), so skipping them is the same arithmetic on
    the rows that matter and a sixteenth of the work. It is what lets a 64-row batch fit
    on a GPU that also holds seven layers of weights.

    The dense path stays available (`_SPARSE["on"] = False`); `check_sparse` runs both on
    the same rows and counts the routing decisions that differ. Must be called before the
    model is loaded: accelerate's device hooks capture each module's forward."""
    from transformers.models.llama4 import modeling_llama4 as M

    dense_experts, dense_moe = M.Llama4TextExperts.forward, M.Llama4TextMoe.forward

    def experts(self, hidden_states, router_scores=None):
        if router_scores is None:
            return dense_experts(self, hidden_states)
        out = torch.zeros_like(hidden_states)
        for e in range(self.num_experts):
            idx = router_scores[:, e].nonzero(as_tuple=True)[0]
            if idx.numel():
                x = hidden_states[idx] * router_scores[idx, e, None]
                gate, up = (x @ self.gate_up_proj[e]).chunk(2, dim=-1)
                out.index_add_(0, idx, (up * self.act_fn(gate)) @ self.down_proj[e])
        return out

    def moe(self, hidden_states):
        if not _SPARSE["on"]:
            return dense_moe(self, hidden_states)
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_scores, router_logits = self.router(hidden_states)
        out = self.shared_expert(hidden_states)
        out.add_(self.experts(hidden_states, router_scores))
        return out, router_logits

    M.Llama4TextExperts.forward, M.Llama4TextMoe.forward = experts, moe
    _SPARSE["on"] = True


def check_sparse(model, ids, msk):
    """The same rows through the dense and the sparse MoE: how many routing decisions
    (position, layer) differ. Returns (differing, total)."""
    got = {}
    for sparse in (False, True):
        _SPARSE["on"], _KEEP["sel"], _MASK["m"] = sparse, {}, msk
        with torch.no_grad():
            model(ids, attention_mask=msk, use_cache=False, logits_to_keep=1)
        got[sparse] = {li: v[0] for li, v in _KEEP["sel"].items()}
    _KEEP["sel"] = None
    real = msk.reshape(-1).bool().cpu()
    n = diff = 0
    for li, dense in got[False].items():
        x, y = dense[real].sort(-1).values, got[True][li][real].sort(-1).values
        n += x.shape[0]
        diff += int((x != y).any(-1).sum())
    return diff, n


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
    ap.add_argument("--rows", type=int, default=0,
                    help="forward the padded batch this many rows at a time and add the "
                         "counts (0 = the whole batch at once). For families whose "
                         "transformers MoE computes every expert on every position.")
    ap.add_argument("--gpu0-mem", default=None,
                    help="a lower cap for GPU 0 than --gpu-mem. Layers that do not fit on "
                         "the GPUs are kept in RAM and run on GPU 0, one at a time, so it "
                         "needs room for one layer's weights on top of its own share.")
    ap.add_argument("--no-template", action="store_true",
                    help="send the raw first turn, not the chat-templated string")
    ap.add_argument("--sparse-moe", action="store_true",
                    help="Llama 4 only: compute each expert on the positions routed to it "
                         "instead of on every position (see llama4_sparse_moe)")
    ap.add_argument("--check-rows", type=int, default=4,
                    help="with --sparse-moe: rows of the first batch run both ways first, "
                         "to count the routing decisions the sparse path changes")
    ap.add_argument("--out", default=os.path.join(HERE, "data", "routing_qwen.npz"))
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if a.sparse_moe:
        llama4_sparse_moe()
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"                       # so decode continues from real tokens
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = {}
    if a.gpu_mem:
        import torch as _t
        kw["max_memory"] = {i: a.gpu_mem for i in range(_t.cuda.device_count())}
        if a.gpu0_mem:
            kw["max_memory"][0] = a.gpu0_mem
        kw["max_memory"]["cpu"] = "280GiB"
    model, info = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16,
                                                       device_map="auto",
                                                       output_loading_info=True, **kw)
    if info["missing_keys"]:        # a weight left at its random initialisation
        raise RuntimeError(f"{len(info['missing_keys'])} weights were not in the "
                           f"checkpoint, e.g. {info['missing_keys'][:3]}")
    model.eval()
    cfg = getattr(model.config, "text_config", None) or model.config
    E = (getattr(cfg, "num_experts", None) or getattr(cfg, "num_local_experts", None)
         or getattr(cfg, "n_routed_experts", None))
    K, L, H = cfg.num_experts_per_tok, cfg.num_hidden_layers, cfg.hidden_size
    print(f"{a.model}: {L} layers, {E} experts, top-{K}")
    print(f"  sharded over {len({str(p.device) for p in model.parameters()})} devices")
    where = {}
    for name, d in (getattr(model, "hf_device_map", None) or {}).items():
        if ".layers." in name:
            where[d] = where.get(d, 0) + 1
    if "cpu" in where:      # kept in RAM, moved to GPU 0 for each forward and back
        print("  decoder layers per device: "
              + ", ".join(f"{d}: {n}" for d, n in sorted(where.items(), key=str)))
    print()

    # Only the last position's logits are ever read (the next token of a decode step).
    # Asking for all of them is 12 GB at 64 x 512 positions and a 202k vocabulary.
    import inspect
    last = ({"logits_to_keep": 1}
            if "logits_to_keep" in inspect.signature(model.forward).parameters else {})

    bucket = {}

    def mk(li):
        def hook(_m, _inp, out):
            sel = selected(out, K, E)                                 # (tokens, K)
            if _KEEP["sel"] is not None:
                _KEEP["sel"].setdefault(li, []).append(sel.cpu())
            m = _MASK["m"]
            if m is not None:
                keep = m.reshape(-1).to(sel.device).bool()
                if keep.numel() == sel.shape[0]:
                    sel = sel[keep]                # drop padded positions
            c = torch.bincount(sel.reshape(-1), minlength=E)
            bucket.setdefault(li, []).append(c.to(torch.int32).cpu().numpy())
        return hook

    n_hooked = 0
    for li, layer in enumerate(decoder_layers(model)):
        cand = find_routers(layer, E, H)
        if len(cand) == 1:
            cand[0].register_forward_hook(mk(li)); n_hooked += 1
        elif len(cand) > 1:
            raise RuntimeError(f"layer {li}: {len(cand)} candidate routers with {E} "
                               "outputs; cannot tell which is the router")
    if not n_hooked:
        raise RuntimeError("no router found -- nothing has a weight of num_experts rows")
    print(f"  hooked {n_hooked} routers\n")
    out, meta = {}, dict(model=a.model, experts=E, top_k=K, layers=L,
                         max_len=a.max_len, decode_steps=a.decode_steps,
                         dataset="ShareGPT_V3 (first human turn, "
                                 + ("raw)" if a.no_template else "chat template)"))
    if a.rows:
        meta["rows_per_forward"] = a.rows

    for B in [int(x) for x in a.batch.split(",")]:
      pre_all, dec_all = [], []
      for rep in range(a.repeats):
        seqs = sharegpt_prompts(tok, B, a.max_len, seed=rep, template=not a.no_template)
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
        if a.sparse_moe and a.check_rows and "sparse_check" not in meta:
            diff, n = check_sparse(model, ids[:a.check_rows], msk[:a.check_rows])
            meta["sparse_check"] = dict(rows=a.check_rows, decisions=n, differ=diff)
            print(f"  dense against sparse MoE on {a.check_rows} rows: {diff} of {n} routing "
                  f"decisions differ ({diff / n * 100:.4f}%)", flush=True)

        # The whole batch at once, or the SAME padded tensors a few rows at a time: a row
        # is then exactly what it was in the full batch, and the counts add.
        rows = a.rows or B
        pre, dec = 0, 0
        t0 = time.time()
        for r0 in range(0, B, rows):
            ids_r, msk_r = ids[r0:r0 + rows], msk[r0:r0 + rows]
            bucket.clear(); _MASK["m"] = msk_r
            with torch.no_grad():
                o = model(ids_r, attention_mask=msk_r, use_cache=True, **last)
            pre = pre + np.stack([bucket[li][0] for li in sorted(bucket)])

            _MASK["m"] = None                        # decode: one real token per sequence
            past, nxt = o.past_key_values, o.logits[:, -1:].argmax(-1)
            am = msk_r
            steps = []
            with torch.no_grad():
                for _ in range(a.decode_steps):
                    bucket.clear()
                    am = torch.cat([am, torch.ones((am.shape[0], 1), dtype=am.dtype,
                                                   device=dev)], 1)
                    o = model(nxt, attention_mask=am, past_key_values=past, use_cache=True,
                              **last)
                    past, nxt = o.past_key_values, o.logits[:, -1:].argmax(-1)
                    steps.append(np.stack([bucket[li][0] for li in sorted(bucket)]))
            if steps:
                dec = dec + np.stack(steps)
            del past, o
            torch.cuda.empty_cache()
        if not a.decode_steps:
            dec = np.zeros((0,) + pre.shape, dtype=pre.dtype)
        pre_all.append(pre)
        got, want = int(pre.sum()), n_tok * K * len(bucket)
        assert got == want, (got, want)
        dec_all.append(dec)
        assert int(dec.sum()) == a.decode_steps * len(bucket) * B * K
        print(f"  B={B:<3d} rep {rep+1}/{a.repeats}  {n_tok:6d} real tokens "
              f"({n_tok/(B*Lmax)*100:.0f}% of the padded batch)"
              f"   [{time.time()-t0:.0f}s]", flush=True)
      # (repeats, layers, E); the analysis reshapes to (-1, E) and can also split by rep
      out[f"prefill_b{B}"] = np.stack(pre_all)
      out[f"decode_b{B}"] = np.stack(dec_all)

    meta["moe_layers"] = len(bucket)
    meta["repeats"] = a.repeats
    np.savez_compressed(a.out, meta=json.dumps(meta), **out)
    print(f"\nwrote {a.out}  ({len(out)} arrays, {meta['moe_layers']} MoE layers)")


if __name__ == "__main__":
    main()
