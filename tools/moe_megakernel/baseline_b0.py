#!/usr/bin/env python3
"""B0 -- characterise the HuggingFace reference path, and quantify how bad a baseline it is.

B0 is a Python loop over all 64 experts. Its ROUTER is exact, which is why Phase 0a's
traces are valid; its DISPATCH is not remotely representative, which is why no number from
it may be quoted as an energy or latency baseline.

The point of measuring it anyway is to put a size on that gap. "Do not compare against B0"
is an instruction people forget; "B1 is 20x faster than B0, so any speedup quoted against
B0 is meaningless" is a fact they remember. This script produces that fact.

RECORDED PER (regime, batch): wall latency per step, NVML power and energy, the clock the
governor actually chose, CUDA kernel launches per step, device time, and the share of
device time that is arithmetic rather than dispatch overhead.

CLOCKS ARE NOT LOCKED. `overlap_exp` established that the unlocked governor is the only
baseline a saving can honestly be quoted against -- and it also established that the
governor picks the most expensive clock available, which is itself part of the finding.

  python3 baseline_b0.py --batch 1,4,16,64 --decode-steps 32
"""
import argparse, csv, json, os, statistics as st, threading, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
# arithmetic, as opposed to dispatch/indexing/copy overhead
MATH = ("gemm", "gemv", "cutlass", "sgemm", "hgemm", "dot", "trmm", "syrk", "ampere",
        "turing", "volta", "wgrad", "implicit")


class Power:
    """NVML sampler. 20 ms cadence and a discarded head, as in overlap_exp: the controller
    integrates over ~12 ms so anything faster is noise, and the first samples still carry
    the previous state."""

    def __init__(self, dev=0, period=0.02, discard=0.4):
        import pynvml
        self.nv, self.period, self.discard = pynvml, period, discard
        pynvml.nvmlInit()
        self.h = pynvml.nvmlDeviceGetHandleByIndex(dev)
        self._stop = threading.Event()

    def _loop(self):
        while not self._stop.is_set():
            try:
                self.rows.append((time.time(),
                                  self.nv.nvmlDeviceGetPowerUsage(self.h) / 1000,
                                  self.nv.nvmlDeviceGetClockInfo(
                                      self.h, self.nv.NVML_CLOCK_SM)))
            except Exception:
                pass
            self._stop.wait(self.period)

    def __enter__(self):
        self.rows = []
        self._stop.clear()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set(); self._t.join(timeout=2)

    def summary(self):
        if not self.rows:
            return None
        t0 = self.rows[0][0]
        keep = [r for r in self.rows if r[0] - t0 >= self.discard] or self.rows
        return dict(power_w=round(st.median(r[1] for r in keep), 2),
                    clock_med=int(st.median(r[2] for r in keep)),
                    clock_max=max(r[2] for r in keep), n_samples=len(keep))


def profile_once(fn):
    """Kernel launches, device time, and how much of it is actually arithmetic."""
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as pr:
        fn()
        torch.cuda.synchronize()
    ev = [e for e in pr.key_averages() if e.device_time_total > 0]
    tot = sum(e.device_time_total for e in ev)
    math = sum(e.device_time_total for e in ev
               if any(m in e.key.lower() for m in MATH))
    return dict(launches=sum(e.count for e in ev), device_ms=round(tot / 1e3, 3),
                math_pct=round(math / tot * 100, 1) if tot else 0.0,
                top=[(e.key[:44], e.count, round(e.device_time_total)) for e in
                     sorted(ev, key=lambda x: -x.device_time_total)[:5]])


def timed(fn, secs=3.0, warmup=5):
    """Sustained loop, not a single shot: power needs a window and the governor behaves
    differently under stop-start than under load (overlap_exp measured 11% between the
    two)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    n, t0 = 0, time.time()
    while time.time() - t0 < secs:
        fn(); n += 1
    torch.cuda.synchronize()
    return (time.time() - t0) / n * 1e3, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    ap.add_argument("--batch", default="1,4,16,64")
    ap.add_argument("--prefill-len", type=int, default=512)
    ap.add_argument("--decode-steps", type=int, default=32)
    ap.add_argument("--secs", type=float, default=3.0)
    ap.add_argument("--out", default=os.path.join(HERE, "data", "baseline_b0.csv"))
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trace_routing import load_text

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16,
                                                 device_map={"": 0}).eval()
    cfg = model.config
    print(f"B0 = {type(model.model.layers[0].mlp).__name__}, "
          f"{cfg.num_hidden_layers} layers x {cfg.num_experts} experts, "
          f"top-{cfg.num_experts_per_tok}\n")

    rows = []
    for B in [int(x) for x in a.batch.split(",")]:
        ids = load_text(tok, B, a.prefill_len).cuda()
        with torch.no_grad():
            o = model(ids, use_cache=True)
            past0 = o.past_key_values
            nxt0 = o.logits[:, -1:].argmax(-1)

        for regime in ("prefill", "decode"):
            if regime == "prefill":
                def step(_ids=ids):
                    with torch.no_grad():
                        model(_ids, use_cache=False)
                ntok = B * a.prefill_len
            else:
                # a fresh cache each call would dominate the timing, so reuse one and feed
                # it the same single token: the arithmetic and the routing are the same
                # shape as real decode, which is what we are characterising
                def step(_p=past0, _n=nxt0):
                    with torch.no_grad():
                        model(_n, past_key_values=_p, use_cache=True)
                ntok = B

            prof = profile_once(step)
            with Power() as p:
                ms, n = timed(step, a.secs)
            pw = p.summary()
            e = pw["power_w"] * ms                                    # mJ per step
            rows.append(dict(regime=regime, batch=B, tokens=ntok,
                             ms=round(ms, 3), power_w=pw["power_w"],
                             energy_mj=round(e, 1),
                             mj_per_token=round(e / ntok, 2),
                             clock_med=pw["clock_med"], clock_max=pw["clock_max"],
                             launches=prof["launches"], device_ms=prof["device_ms"],
                             math_pct=prof["math_pct"], reps=n))
            r = rows[-1]
            print(f"  {regime:<8} B={B:<4d} {r['ms']:8.2f} ms  {r['power_w']:6.1f} W  "
                  f"{r['energy_mj']:9.1f} mJ  {r['mj_per_token']:8.2f} mJ/tok  "
                  f"clk {r['clock_med']:4d}  {r['launches']:6d} launches  "
                  f"math {r['math_pct']:4.1f}%")
        del past0, o
        torch.cuda.empty_cache()

    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {a.out}  ({len(rows)} rows)")
    print("\nREMINDER: B0 is a Python loop over 64 experts. These numbers exist to size "
          "the strawman,\nnot to be a baseline. Nothing may be quoted as a speedup or "
          "saving against them.")


if __name__ == "__main__":
    main()
