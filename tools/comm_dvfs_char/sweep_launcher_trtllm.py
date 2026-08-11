#!/usr/bin/env python3
"""TRT-LLM custom AllReduce DVFS sweep LAUNCHER (4x A100, NVLink).

The TRT-LLM counterpart of sweep_launcher.py.  Where that one drives nccl-tests via
collect_nccl.py, this drives THIS REPO'S OWN custom-AllReduce collector --
`collector/network/collect_all_reduce.py --backend trtllm` -- under `mpirun -n <ws>`,
so the kernel measured is TRT-LLM's Lamport custom allreduce (the one real inference
uses), not NCCL.

Differences from the NCCL sweep, and why:

  * NO channel axis.  NCCL_{MIN,MAX}_NCHANNELS controls NCCL's own collective kernel;
    the Lamport custom kernels do not use NCCL channels, so the SM-footprint knob does
    not exist here.  This is a 1-D DVFS sweep: freq x message size.
  * --range is in ELEMENTS, not bytes (collect_all_reduce.py builds a tensor of `size`
    elements: input_shape = [size//4096, 4096]).  For bf16, bytes = 2 x elements.
    collect_nccl.py's --range is in BYTES.  Mixing them up silently compares different
    physical message sizes.
  * One mpirun per frequency (not per size): importing tensorrt_llm across 4 ranks
    costs ~30s, and the collector loops the whole size range inside one launch.

  python3 sweep_launcher_trtllm.py --out results/trtllm_ar --measure-power
  python3 sweep_launcher_trtllm.py --out results/trtllm_ar --resume

CLOCK SAFETY: clocks are restored via `nvidia-smi -rgc` on every exit path -- normal
completion, exception, `finally`, atexit, SIGINT and SIGTERM.
"""

import argparse
import atexit
import csv
import os
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
COLLECTOR = os.path.join(REPO, "collector", "network", "collect_all_reduce.py")

FREQS = [1410, 1200, 900, 705, 510, 300]
# ELEMENTS (bf16 -> bytes = 2x). The upper five EXACTLY match the byte sizes of the
# NCCL sweep (2/8/25/100/256 MiB) so the two datasets compare without interpolation;
# 25 and 100 MiB are not powers of two, which is why --sizes exists. The lower three
# (64/256 KiB, 1 MiB) cover the latency-bound regime where MIN_LATENCY beats NCCL.
SIZES_ELEMS = [32768, 131072, 524288, 1048576, 4194304, 13107200, 52428800, 134217728]
# AllReduce implementations worth separating on this topology. AUTO is the serving
# default (tries MNNVL, then picks between NCCL and MIN_LATENCY by heuristic); the
# rest pin one kernel so the heuristic's choice can be checked against it. UB is
# excluded -- it needs user-buffer setup and fails here. SYMM_MEM/LOWPRECISION/MNNVL
# are excluded because they silently fall back on this hardware, which would log
# duplicate rows under distinct names.
STRATEGIES = ["AUTO", "NCCL", "MIN_LATENCY", "ONESHOT", "TWOSHOT"]
DTYPE = "bfloat16"
WORLD = 4

GIB_ENV = [
    "NCCL_NET", "NCCL_TUNER_CONFIG_PATH", "NCCL_NET_GDR_LEVEL", "NCCL_IB_TC",
    "NCCL_IB_FIFO_TC", "NCCL_IB_QPS_PER_CONNECTION", "NCCL_IB_ADAPTIVE_ROUTING",
    "NCCL_CROSS_NIC", "NCCL_P2P_NET_CHUNKSIZE", "NCCL_NVLS_CHUNKSIZE",
]

_locked = False
_gpus = ",".join(str(i) for i in range(WORLD))


def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def reset_clocks():
    global _locked
    if not _locked:
        return
    r = sh(["sudo", "nvidia-smi", "-i", _gpus, "-rgc"])
    _locked = False
    print(f"[clocks] reset -> {'ok' if r.returncode == 0 else 'FAILED: ' + r.stderr.strip()}", flush=True)


def _sig(signum, _frame):
    print(f"\n[clocks] caught signal {signum}, restoring clocks", flush=True)
    reset_clocks()
    sys.exit(128 + signum)


def lock_clock(freq, tol=20):
    global _locked
    r = sh(["sudo", "nvidia-smi", "-i", _gpus, "-lgc", f"{freq},{freq}"])
    if r.returncode != 0:
        print(f"[clocks] LOCK FAILED at {freq}: {r.stderr.strip()}", flush=True)
        return False
    _locked = True
    time.sleep(0.5)
    q = sh(["nvidia-smi", "-i", _gpus, "--query-gpu=index,clocks.sm", "--format=csv,noheader,nounits"])
    got = []
    for line in q.stdout.strip().splitlines():
        try:
            got.append(int(line.split(",")[1]))
        except (IndexError, ValueError):
            pass
    if not got or [c for c in got if abs(c - freq) > tol]:
        print(f"[clocks] read-back MISMATCH at {freq}: {got}", flush=True)
        return False
    return True


def main():
    global WORLD, _gpus, SIZES_ELEMS
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output DIRECTORY (one perf file per freq)")
    ap.add_argument("--measure-power", action="store_true")
    ap.add_argument("--power-duration", type=float, default=2.0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--python", default=os.path.expanduser("~/trtllm_venv/bin/python"))
    ap.add_argument("--limit-freqs", type=int, default=0)
    ap.add_argument("--strategies", default=",".join(STRATEGIES),
                    help="comma-separated AllReduceStrategy names to sweep")
    ap.add_argument("--freqs", default=",".join(str(f) for f in FREQS),
                    help="comma-separated SM clocks in MHz (A100 accepts 15 MHz steps)")
    ap.add_argument("--sizes", default=",".join(str(s) for s in SIZES_ELEMS),
                    help="comma-separated message sizes in ELEMENTS (bf16 bytes = 2x)")
    ap.add_argument("--world", type=int, default=WORLD, help="ranks / GPUs (this node has 4)")
    ap.add_argument("--ctas", default=None,
                    help="pin NCCL_MIN_CTAS=NCCL_MAX_CTAS=N. A CTA is a thread block, one\n"
                    "resident per SM, so this is the collective's SM footprint. Supersedes\n"
                    "NCCL_MAX_NCHANNELS and needs no MPS. Affects the NCCL path only -- the\n"
                    "MIN_LATENCY kernels are TRT-LLM's own and ignore it.")
    ap.add_argument("--nccl-proto", default=None,
                    help="pin NCCL_PROTO (LL|LL128|Simple). NCCL picks per message size by "
                    "default, and those switches are a candidate explanation for the "
                    "non-monotonic MIN_LATENCY-vs-NCCL crossover.")
    a = ap.parse_args()

    outdir = os.path.abspath(a.out)
    os.makedirs(outdir, exist_ok=True)
    freqs = [int(f) for f in a.freqs.split(",") if f.strip()]
    if a.limit_freqs:
        freqs = freqs[: a.limit_freqs]
    strategies = [s.strip() for s in a.strategies.split(",") if s.strip()]
    WORLD = a.world
    _gpus = ",".join(str(i) for i in range(max(WORLD, 1)))
    SIZES_ELEMS = [int(s) for s in a.sizes.split(",") if s.strip()]

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)
    atexit.register(reset_clocks)

    sizes_arg = ",".join(str(s) for s in SIZES_ELEMS)
    print(f"freqs={len(freqs)} strategies={len(strategies)} sizes={len(SIZES_ELEMS)} "
          f"-> {len(freqs) * len(strategies) * len(SIZES_ELEMS)} configs  "
          f"(bf16 {SIZES_ELEMS[0] * 2 / 1024:.0f} KiB .. {SIZES_ELEMS[-1] * 2 / 1048576:.0f} MiB)  "
          f"world={WORLD}  power={a.measure_power}", flush=True)
    sh(["sudo", "nvidia-smi", "-i", _gpus, "-pm", "1"])

    # strip the GCP gIB shim: its guest_config_checker aborts NCCL init, and TRT-LLM
    # still uses NCCL for bootstrap/fallback paths.
    base_env = {k: v for k, v in os.environ.items() if k not in GIB_ENV}
    ld = [p for p in base_env.get("LD_LIBRARY_PATH", "").split(os.pathsep) if p and "/gib/" not in p + "/"]
    base_env["LD_LIBRARY_PATH"] = os.pathsep.join(ld)
    if a.nccl_proto:
        base_env["NCCL_PROTO"] = a.nccl_proto
        print(f"NCCL_PROTO pinned to {a.nccl_proto}", flush=True)
    if a.ctas:
        base_env["NCCL_MIN_CTAS"] = str(a.ctas)
        base_env["NCCL_MAX_CTAS"] = str(a.ctas)
        print(f"NCCL_{{MIN,MAX}}_CTAS pinned to {a.ctas} (SM footprint)", flush=True)

    t0 = time.time()
    ok = fail = 0
    try:
        n_groups = len(freqs) * len(strategies)
        gi = 0
        for freq in freqs:  # outer: locking the clock is the expensive state change
            locked = None
            for strat in strategies:
                gi += 1
                # One perf file per (strategy, freq). The collector's row schema has no
                # strategy/freq columns and adding them would be a producer+consumer
                # contract change, so the keys live in the filename instead.
                suffix = f"_cta{a.ctas}" if a.ctas else ""
                perf_file = os.path.join(outdir, f"trtllm_ar_{strat}_f{freq}{suffix}.txt")
                if a.resume and os.path.exists(perf_file):
                    n = sum(1 for _ in csv.DictReader(open(perf_file)))
                    if n >= len(SIZES_ELEMS):
                        print(f"[{gi}/{n_groups}] {strat} f={freq} -- {n} rows, skip", flush=True)
                        continue
                if locked is None:
                    locked = lock_clock(freq)
                if not locked:
                    print(f"[{gi}/{n_groups}] SKIP: could not lock {freq} MHz", flush=True)
                    fail += len(SIZES_ELEMS)
                    continue

                cmd = [
                    "mpirun", "-n", str(WORLD), "--oversubscribe",
                    a.python, COLLECTOR,
                    "--backend", "trtllm", "--dtype", DTYPE,
                    "--sizes", sizes_arg, "--strategy", strat, "-f", perf_file,
                ]
                if a.measure_power:
                    cmd += ["--measure_power", "--power_test_duration_sec", str(a.power_duration)]

                print(f"[{gi}/{n_groups}] {strat:<12} f={freq}MHz  elapsed={time.time() - t0:.0f}s", flush=True)
                r = subprocess.run(cmd, env=base_env)
                if r.returncode != 0:
                    print(f"  FAILED rc={r.returncode} ({strat} f={freq})", flush=True)
                    fail += len(SIZES_ELEMS)
                else:
                    ok += len(SIZES_ELEMS)
    finally:
        reset_clocks()

    print(f"DONE in {(time.time() - t0) / 60:.1f} min  ({ok} configs, {fail} failed)  -> {outdir}", flush=True)


if __name__ == "__main__":
    main()
