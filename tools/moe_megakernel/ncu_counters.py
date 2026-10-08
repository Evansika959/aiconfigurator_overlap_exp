#!/usr/bin/env python3
"""Hardware counters for every kernel the simulator has to model, via ncu.

WHY. The simulator's power model is activity-based: power = sum over hardware units of
(energy per unit of activity) x (activity rate). To fit the per-unit energies, and to
check the byte and FLOP counts the workload decomposer computes from routing alone, the
activity has to be MEASURED for a spread of kernels: tensor-core instructions, L1 and L2
traffic, DRAM bytes. `data/tile_ncu_qwen.csv` has four prefill rows. This collects
decode and prefill, three tile sizes, and the cold / hot / whole-layer partitions of
the persistent kernel, plus vLLM's queue kernel.

Counters are instruction and byte counts, so they do not depend on the clock or on which
of the four identical GPUs runs them. Run it on a GPU that is not doing the locked-clock
sweep:

  source env.sh
  sudo env CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$PYTHONPATH PYTHONDONTWRITEBYTECODE=1 \\
    /usr/local/cuda/bin/ncu --profile-from-start off --target-processes all \\
    --metrics $(python3 ncu_counters.py --metrics) --csv \\
    --log-file data/ncu_raw_qwen.csv \\
    python3 ncu_counters.py --run --manifest data/ncu_manifest_qwen.json
  sudo chown $USER data/ncu_raw_qwen.csv data/ncu_manifest_qwen.json
  python3 ncu_counters.py --join            # -> data/ncu_counters_qwen.csv

ncu needs sudo unless the driver was loaded with NVreg_RestrictProfilingToAdminUsers=0.
It serialises and replays each kernel, so the durations it reports are not comparable
to the sweep's; the counters are.
"""
import argparse, csv, json, os, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
METRICS = ["launch__grid_size", "gpu__time_duration.sum",
           "sm__cycles_active.avg", "sm__cycles_elapsed.max",
           "sm__inst_executed.sum", "sm__inst_executed_pipe_tensor.sum",
           "l1tex__data_bank_reads.sum", "l1tex__data_bank_writes.sum",
           "lts__t_bytes.sum", "lts__t_sector_hit_rate.pct",
           "dram__bytes_read.sum", "dram__bytes_write.sum",
           "sm__throughput.avg.pct_of_peak_sustained_elapsed"]
CELLS = [8, 16, 32, 64, 0]                 # 0 = the prefill batch
BMS = [16, 64, 128]
LIGHT = 64


def run(a):
    import numpy as np, torch
    sys.path.insert(0, HERE)
    from vendor.fused_moe_triton import PreparedMoE, _invoke
    from persistent_moe import PersistentMoE, _launch
    from b1_bench import SHAPES, load_counts, make_inputs
    from batch_clock_sweep import decode_batches

    spec = SHAPES[a.model]
    base = json.load(open(os.path.join(HERE, "data", f"b1_config_{a.model}.json")))
    plan, manifest = [], []
    for cells in CELLS:
        if cells == 0:
            counts = load_counts(spec, 0, a.layer)
        else:
            counts = decode_batches(spec, a.layer, cells)
            counts[np.argmax(counts)] -= int(counts.sum()) % spec["topk"]
        x, w1, w2, tw, ti, M = make_inputs(spec, counts)
        light_e = np.argsort(counts, kind="stable")[:LIGHT]
        q = PreparedMoE(x, w1, w2, tw, ti, base)
        q.run(); torch.cuda.synchronize()
        info = dict(cells=cells, tokens=M, experts=int((counts > 0).sum()),
                    real_pairs=int(counts.sum()))
        for gemm, args in (("g1", (q.x, q.w1, q.c1, q.tw, q.sorted_ids, q.expert_ids,
                                   q.n_pad, False, q.topk, q.cfg, q.ct)),
                           ("g2", (q.c2, q.w2, q.c3, q.tw, q.sorted_ids, q.expert_ids,
                                   q.n_pad, True, 1, q.cfg, q.ct))):
            plan.append((lambda args=args: _invoke(*args)))
            manifest.append(dict(info, kernel="queue", block_m=base["BLOCK_SIZE_M"],
                                 part="all", gemm=gemm, blocks=0))
        for bm in BMS:
            cfg = dict(base); cfg["BLOCK_SIZE_M"] = bm
            p = PersistentMoE(x, w1, w2, tw, ti, cfg)
            p.run(); torch.cuda.synchronize()
            re_ = p.rowblock_expert.cpu().numpy()
            mnp = np.isin(re_, light_e)
            mask = torch.from_numpy(mnp).to(p.live_m.device)
            for name, rows, ex, nb in (("light", p.live_m[mask], np.unique(re_[mnp]), 54),
                                       ("heavy", p.live_m[~mask], np.unique(re_[~mnp]), 54),
                                       ("all", p.live_m, np.unique(re_), 108)):
                t1 = p.tiles("g1", rows, key=("n", name))
                t2 = p.tiles("g2", rows, key=("n", name))
                meta = dict(cells=cells, tokens=M, experts=int(len(ex)),
                            real_pairs=int(counts[ex].sum()), kernel="persistent",
                            block_m=bm, part=name, blocks=nb,
                            padded_rows=int(rows.numel() * bm))
                plan.append((lambda p=p, t1=t1, nb=nb, cfg=cfg: _launch(
                    p.x, p.w1, p.c1, p.tw, p.sorted_ids, p.expert_ids, t1, nb, False,
                    p.topk, cfg, p.ct)))
                manifest.append(dict(meta, gemm="g1", tiles=int(t1.numel())))
                plan.append((lambda p=p, t2=t2, nb=nb, cfg=cfg: _launch(
                    p.c2, p.w2, p.c3, p.tw, p.sorted_ids, p.expert_ids, t2, nb, True, 1,
                    cfg, p.ct)))
                manifest.append(dict(meta, gemm="g2", tiles=int(t2.numel())))
    for fn in plan:                                   # warm, outside the profiled range
        fn()
    torch.cuda.synchronize()
    json.dump(manifest, open(a.manifest, "w"), indent=0)
    print(f"{len(plan)} kernels to profile", flush=True)
    torch.cuda.cudart().cudaProfilerStart()
    for fn in plan:
        fn(); torch.cuda.synchronize()                # one kernel per launch
    torch.cuda.cudart().cudaProfilerStop()
    print("done", flush=True)


def join(a):
    raw = a.raw or os.path.join(HERE, "data", f"ncu_raw_{a.model}.csv")
    man = json.load(open(a.manifest))
    lines = open(raw, newline="").read().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith('"ID"'))
    rows = list(csv.DictReader(lines[start:]))
    per = defaultdict(dict)
    for r in rows:
        per[int(r["ID"])]["name"] = r["Kernel Name"]
        v = r["Metric Value"].replace(",", "")
        try:
            per[int(r["ID"])][r["Metric Name"]] = float(v)
        except ValueError:
            per[int(r["ID"])][r["Metric Name"]] = v
    ids = sorted(per)
    if len(ids) != len(man):
        raise SystemExit(f"{len(ids)} profiled kernels but {len(man)} in the manifest")
    out = []
    for k, m in zip(ids, man):
        want = "fused_moe_kernel" if m["kernel"] == "queue" else "persistent_moe_kernel"
        if want not in per[k]["name"]:
            raise SystemExit(f"kernel {k} is {per[k]['name']}, manifest says {want}")
        row = dict(m)
        for name in METRICS:
            row[name.replace(".", "_")] = per[k].get(name, "")
        out.append(row)
    dst = os.path.join(HERE, "data", f"ncu_counters_{a.model}.csv")
    keys = []
    for r in out:
        keys += [k for k in r if k not in keys]
    with open(dst, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader(); w.writerows(out)
    print(f"wrote {dst} ({len(out)} kernels)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--manifest", default=os.path.join(HERE, "data", "ncu_manifest_qwen.json"))
    ap.add_argument("--raw", default=None)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--join", action="store_true")
    ap.add_argument("--metrics", action="store_true", help="print the metric list")
    a = ap.parse_args()
    if a.metrics:
        print(",".join(METRICS))
    elif a.run:
        run(a)
    elif a.join:
        join(a)
    else:
        ap.error("one of --run, --join, --metrics")


if __name__ == "__main__":
    main()
