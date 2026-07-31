#!/usr/bin/env python3
"""GEMM sweep LAUNCHER — walks gemm_config_table.csv, grouping by (freq, sm).
Per group: lock SM clock (sudo nvidia-smi -lgc) and launch sweep_worker.py in a subprocess with
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE set (SM cap). The worker uses this repo's collector functions
to measure latency (+power) and append to the collector-schema perf CSV.

  python3 sweep_launcher.py --out results/gemm_lat_perf.txt                 # latency pass
  python3 sweep_launcher.py --out results/gemm_pow_perf.txt --measure-power # power pass
  python3 sweep_launcher.py ... --resume                                    # skip configs already in --out
  python3 sweep_launcher.py ... --limit-groups 2 --limit-shapes 2           # smoke
Requires: MPS daemon running (start with: nvidia-cuda-mps-control -d) and passwordless sudo.
"""
import argparse, csv, json, os, subprocess, sys, time
from collections import OrderedDict
HERE = os.path.dirname(os.path.abspath(__file__)); WORKER = f"{HERE}/sweep_worker.py"; GPU = "0"

def sh(cmd): return subprocess.run(cmd, capture_output=True, text=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default=f"{HERE}/gemm_config_table.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--measure-power", action="store_true")
    ap.add_argument("--resume", action="store_true", help="skip (freq,sm,m,n,k) already present in --out")
    ap.add_argument("--limit-groups", type=int, default=0)
    ap.add_argument("--limit-shapes", type=int, default=0)
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.table)))
    groups = OrderedDict()
    for r in rows:
        groups.setdefault((int(r["freq_mhz"]), int(r["sm_count"]), float(r["mps_pct"])), []).append(
            [int(r["M"]), int(r["N"]), int(r["K"])])
    gitems = list(groups.items())
    if a.limit_groups: gitems = gitems[:a.limit_groups]

    out = os.path.abspath(a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    # resume: read already-logged (freq,sm,m,n,k); else start fresh
    done = set()
    if a.resume and os.path.exists(out):
        for row in csv.DictReader(open(out)):
            try: done.add((int(row["freq_mhz"]), int(row["sm_count"]), int(row["m"]), int(row["n"]), int(row["k"])))
            except (KeyError, ValueError): pass
        print(f"resume: {len(done)} configs already in {out}", flush=True)
    else:
        for f in (out, out + ".lock"):
            if os.path.exists(f): os.remove(f)

    total_todo = 0
    for f in (os.path.exists(out + ".lock"),):  # cleanup a stale lock even on resume
        if f: os.remove(out + ".lock")
    print(f"groups={len(gitems)}  measure_power={a.measure_power}  resume={a.resume}  out={out}", flush=True)
    sh(["sudo", "nvidia-smi", "-i", GPU, "-pm", "1"])
    t0 = time.time()
    try:
        for i, ((freq, sm, pct), shapes) in enumerate(gitems):
            if a.limit_shapes: shapes = shapes[:a.limit_shapes]
            shapes = [s for s in shapes if (freq, sm, s[0], s[1], s[2]) not in done]  # resume filter
            if not shapes:
                print(f"[{i+1}/{len(gitems)}] freq={freq} sm={sm} -- all done, skip", flush=True); continue
            total_todo += len(shapes)
            sh(["sudo", "nvidia-smi", "-i", GPU, "-lgc", f"{freq},{freq}"]); time.sleep(0.5)
            env = dict(os.environ, CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=str(int(pct)),
                       COLLECTOR_MEASURE_POWER=("1" if a.measure_power else "0"),
                       COLLECTOR_POWER_MIN_DURATION="1.0")
            print(f"[{i+1}/{len(gitems)}] freq={freq} sm={sm}(mps%={int(pct)}) shapes={len(shapes)} "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)
            r = subprocess.run([sys.executable, WORKER, "--freq", str(freq), "--sm-count", str(sm),
                                "--mps-pct", str(pct), "--shapes", json.dumps(shapes), "--perf-file", out],
                               env=env)
            if r.returncode != 0:
                print(f"  WORKER FAILED rc={r.returncode} (freq={freq} sm={sm})", flush=True)
    finally:
        sh(["sudo", "nvidia-smi", "-i", GPU, "-rgc"])
    print(f"DONE in {(time.time()-t0)/60:.1f} min  (+{total_todo} configs this run)  -> {out}", flush=True)

if __name__ == "__main__":
    main()
