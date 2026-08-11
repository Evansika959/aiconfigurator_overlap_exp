#!/usr/bin/env python3
"""Merge the TRT-LLM AllReduce sweep into one analysis table.

sweep_launcher_trtllm.py writes one collector-schema perf file per
(strategy, freq) -- `trtllm_ar_<STRATEGY>_f<FREQ>.txt` -- because the collector's
row schema carries neither key. This recovers both from the filename and adds
derived bandwidth/energy columns.

  python3 merge_trtllm_ar.py --indir results/train_range --out data/comm_trtllm_ar_8-256MiB.csv

Output columns:
  op,dtype,strategy,world_size,freq_mhz,msg_bytes,msg_mib,latency_ms,
  power_w,energy_mj,algbw_gbps,busbw_gbps

UNITS / SCOPE -- read before using:
  * `power_w` and therefore `energy_mj` are the RANK-0 GPU only.
    collect_all_reduce.py samples PowerMonitor(local_rank) and only rank 0 logs.
    Node total is approximately 4x for world_size=4.
  * energy_mj = power_w * latency_ms  (W*ms == mJ, the SDK's PerformanceResult
    convention). tools/gemm_dvfs_char/data/gemm_char_merged.csv divides by 1000
    under the same column name, so ITS values are Joules -- rescale before
    comparing.
  * Energy is trustworthy at >= 1 MiB only. Below that the dynamic power is
    ~1-2 W above a ~58 W idle floor, inside NVML's instantaneous-reading jitter;
    7% of frequency steps there violate physical monotonicity. This table's
    8-256 MiB range is entirely in the trustworthy region (1/150 violations,
    magnitude 0.3%). Latency is clean everywhere (0/200).
"""

import argparse
import csv
import glob
import os
import re

FNAME_RE = re.compile(r"trtllm_ar_(?P<st>.+?)_f(?P<freq>\d+)(?:_cta(?P<cta>\d+))?\.txt$")
BYTES_PER_ELEM = {"bfloat16": 2, "half": 2, "float16": 2, "float32": 4, "int8": 1}


def busbw_factor(n):
    """nccl-tests convention for all_reduce, so this compares to the NCCL sweep."""
    return 2.0 * (n - 1) / n if n >= 2 else 1.0


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default=os.path.join(here, "results", "train_range"))
    ap.add_argument("--out", default=os.path.join(here, "data", "comm_trtllm_ar_8-256MiB.csv"))
    ap.add_argument("--min-mib", type=float, default=0.0, help="drop rows below this size")
    ap.add_argument("--max-mib", type=float, default=1e9)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.indir, "trtllm_ar_*_f*.txt")))
    if not files:
        raise SystemExit(f"no trtllm_ar_*_f*.txt under {a.indir}")

    rows = []
    for path in files:
        m = FNAME_RE.search(os.path.basename(path))
        if not m:
            print(f"skip (unparseable name): {path}")
            continue
        st, freq = m.group("st"), int(m.group("freq"))
        cta = int(m.group("cta")) if m.group("cta") else ""
        for r in csv.DictReader(open(path)):
            dtype = r["allreduce_dtype"]
            n = int(r["num_gpus"])
            msg_bytes = int(r["message_size"]) * BYTES_PER_ELEM[dtype]
            mib = msg_bytes / 1048576
            if not (a.min_mib <= mib <= a.max_mib):
                continue
            lat = float(r["latency"])
            pw = float(r["power"]) if r.get("power") else None
            algbw = msg_bytes / (lat / 1e3) / 1e9 if lat > 0 else 0.0
            rows.append(dict(
                op=r["op_name"], dtype=dtype, strategy=st, world_size=n,
                freq_mhz=freq, nccl_ctas=cta,
                clock_sm_mean=r.get("clock_sm_mean", ""),
                clock_sm_min=r.get("clock_sm_min", ""),
                throttled=(int(r["clock_sm_min"]) < freq - 20) if r.get("clock_sm_min") else "", msg_bytes=msg_bytes, msg_mib=round(mib, 4),
                latency_ms=lat,
                power_w=round(pw, 3) if pw else "",
                energy_mj=round(pw * lat, 4) if pw else "",
                algbw_gbps=round(algbw, 3),
                busbw_gbps=round(algbw * busbw_factor(n), 3),
            ))

    rows.sort(key=lambda r: (r["strategy"], r["nccl_ctas"] if r["nccl_ctas"] != "" else -1, r["msg_bytes"], r["freq_mhz"]))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    sizes = sorted({r["msg_mib"] for r in rows})
    print(f"merged {len(files)} files -> {len(rows)} rows -> {a.out}")
    print(f"  strategies: {sorted({r['strategy'] for r in rows})}")
    print(f"  freqs: {sorted({r['freq_mhz'] for r in rows})}")
    print(f"  sizes (MiB): {[f'{s:g}' for s in sizes]}")


if __name__ == "__main__":
    main()
