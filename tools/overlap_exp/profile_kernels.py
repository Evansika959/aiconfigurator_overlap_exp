#!/usr/bin/env python3
"""What the cuBLAS GEMM kernel is launched with, shape by shape, read by ncu.

The event model schedules a GEMM as `grid` thread blocks, one 256x128 output tile each,
and needs the grid of every shape it predicts. For most shapes that is
ceil(M/256) * ceil(N/128), but cuBLAS splits K for some (1024x8192x8192 launches twice
that), and nothing in the shape says when. So the grid is measured, as data/grid12.json
was, for every shape the simulator campaign uses -- including the shapes of the
registered blind test, whose predictions need it before the test is run. A launch
attribute is not a timing: reading it does not un-blind anything.

This script only issues the GEMMs; ncu collects:

  sudo /usr/local/cuda/bin/ncu --profile-from-start off --target-processes all \
    --metrics launch__grid_size,launch__block_size,launch__shared_mem_per_block_dynamic,\
launch__shared_mem_per_block_static,launch__shared_mem_config_size,\
launch__registers_per_thread,sm__cycles_active.avg,sm__cycles_elapsed.max,\
dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes.sum \
    --csv python3 profile_kernels.py > data/sim_overlap/ncu_gemm.csv

  python3 profile_kernels.py --parse data/sim_overlap/ncu_gemm.csv   # -> grids.json
"""
import argparse
import csv
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import plan_spans

EXTRA = ["8192x4096x4096", "4096x8192x8192", "1024x16384x16384", "2048x16384x16384",
         "4096x16384x16384", "8192x16384x16384", "1280x4096x4096", "1536x4096x4096",
         "1792x4096x4096", "512x4096x4096", "1024x2048x4096", "1024x4096x8192",
         "2048x4096x8192", "1024x8192x4096", "2048x8192x4096", "4096x8192x4096",
         "1024x4096x2048", "4096x4096x2048", "4096x4096x16384", "1024x4096x16384",
         # a training step's weight-gradient GEMMs: output N x K, inner dimension the batch
         "8192x8192x2048", "8192x8192x4096", "4096x4096x8192", "4096x4096x1024",
         "8192x8192x1024", "8192x4096x8192", "4096x8192x2048", "2048x4096x8192",
         "8192x8192x3072", "8192x8192x6144", "8192x8192x1536", "4096x4096x3072", "4096x4096x6144"]
SHAPES = plan_spans.SEEN_SHAPES + plan_spans.BLIND_SHAPES + EXTRA


def run():
    import torch
    torch.cuda.set_device(0)
    order = []
    for s in SHAPES:
        m, n, k = (int(v) for v in s.split("x"))
        x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
        w = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
        for _ in range(3):                       # warm up OUTSIDE the profiled range
            torch.nn.functional.linear(x, w)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        torch.nn.functional.linear(x, w)
        torch.cuda.synchronize()                 # one kernel per profiled range
        torch.cuda.cudart().cudaProfilerStop()
        order.append(s)
        del x, w
        torch.cuda.empty_cache()
    print("SHAPE_ORDER " + " ".join(order), file=sys.stderr)


def parse(path):
    """ncu --csv prints one row per (kernel, metric). Kernels come in SHAPES order."""
    text = open(path).read()
    start = text.index('"ID"')
    rows = list(csv.DictReader(io.StringIO(text[start:])))
    by_id = {}
    for r in rows:
        d = by_id.setdefault(int(r["ID"]), dict(kernel=r["Kernel Name"]))
        d[r["Metric Name"]] = r["Metric Value"].replace(",", "")
    ids = sorted(by_id)
    if len(ids) != len(SHAPES):
        sys.exit(f"{len(ids)} profiled kernels for {len(SHAPES)} shapes: not one kernel per GEMM")
    out, grids = [], {}
    for s, i in zip(SHAPES, ids):
        d = by_id[i]
        m, n, k = (int(v) for v in s.split("x"))
        tiles = -(-m // 256) * -(-n // 128)
        g = int(d["launch__grid_size"])
        grids[s] = g
        out.append(dict(shape=s, tiles=tiles, grid=g, split=round(g / tiles, 3), **d))
    with open(os.path.join(HERE, "data", "sim_overlap", "grids.json"), "w") as fh:
        json.dump(grids, fh, indent=1)
    cols = ["shape", "tiles", "grid", "split", "launch__block_size",
            "launch__shared_mem_per_block_dynamic", "launch__shared_mem_per_block_static",
            "launch__shared_mem_config_size", "launch__registers_per_thread",
            "sm__cycles_active.avg", "sm__cycles_elapsed.max", "dram__bytes_read.sum",
            "dram__bytes_write.sum", "lts__t_bytes.sum", "kernel"]
    with open(os.path.join(HERE, "data", "sim_overlap", "ncu_gemm_table.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(out)
    for r in out:
        print(f"{r['shape']:>18}  tiles {r['tiles']:5d}  grid {r['grid']:5d}  x{r['split']:<5}  "
              f"block {r.get('launch__block_size', '?'):>4}  smem {r.get('launch__shared_mem_per_block_dynamic', '?'):>7}  "
              f"{r['kernel'][:60]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parse")
    a = ap.parse_args()
    parse(a.parse) if a.parse else run()
