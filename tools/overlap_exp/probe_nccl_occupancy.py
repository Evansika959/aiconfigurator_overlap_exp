#!/usr/bin/env python3
"""How many SMs can a cuBLAS GEMM reach while an NCCL all-reduce is resident?

The overlap model composes "a GEMM on 108 - c SMs" with "a collective on c CTAs". The
overlapped runs say otherwise: replayed as events, their iteration times fit a GEMM that
has 108 - 2c SMs, not 108 - c (../moe_megakernel/cross_study_check.py, check 5). That is
an inference from timings. This measures it.

METHOD. ../gemm_dcfs_char/squatter.py has a probe kernel whose blocks carry a chosen
amount of dynamic shared memory and each write the id of the SM they ran on. With the
cuBLAS GEMM's 147456 B the probe is placed under the same constraint as a GEMM block.
Here the probe is launched on its own stream WHILE a long all-reduce is resident, and a
repetition counts only if CUDA events show the probe started after the collective's
kernel and ended before it. The set of SM ids the probe reached is what a GEMM could
have used; its complement is what the collective cost.

The probe is also run with less shared memory (0 ... 132 KB). A kernel with a small
block can share an SM with NCCL's, so the loss depends on the co-running kernel's
shared-memory class and not only on the collective.

ORDER. `comm_first`: the collective is launched on an idle GPU, as in every overlap
campaign. `gemm_first`: a GEMM that fills the chip is launched first and the collective
arrives while it runs -- what happens in training, where a gradient all-reduce is issued
behind compute that is already queued. The probe then waits for the GEMM to end. The
spans say the collective is ~16% slower for its WHOLE duration in that order, whatever
the GEMM's length; if its blocks were packed onto fewer SMs as those came free, the
probe will reach more SMs here than in `comm_first`.

One process group per CTA count (NCCL caches NCCL_{MIN,MAX}_CTAS). Clocks are not
touched; nothing is locked.

  python3 probe_nccl_occupancy.py --out data/sim_overlap/nccl_occupancy.csv
"""
import argparse
import csv
import os
import socket
import sys
import time

os.environ.setdefault("LD_LIBRARY_PATH", "")
os.environ["LD_LIBRARY_PATH"] = ":".join(
    p for p in os.environ["LD_LIBRARY_PATH"].split(":") if p and "gib" not in p)
os.environ.pop("NCCL_NET", None)

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "gemm_dcfs_char"))

WORLD, TOTAL_SM = 4, 108
CTAS = [1, 2, 3, 4, 6, 8, 12, 16, 22, 27, 32, 40, 48, 54, 56, 64]
SMEM = [147456, 135168, 102400, 65536, 32768, 16384, 0]     # bytes of dynamic shared memory
# 2 GiB: at 146 GB/s the collective still lasts 15 ms, longer than the GEMM plus the probe
MIB, REPS, WAVES = 2048, 6, 3
ORDERS = ("comm_first", "gemm_first")
FIELDS = ["ctas", "order", "smem", "rep", "reached", "lost", "valid", "probe_start",
          "probe_end", "comm_start", "comm_end", "gemm_end", "blocked_ids"]


def worker(rank, port, ctas, out_path, q):
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(WORLD), MASTER_ADDR="127.0.0.1",
                      MASTER_PORT=port, NCCL_MIN_CTAS=str(ctas), NCCL_MAX_CTAS=str(ctas))
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=WORLD)
    buf = torch.ones(MIB * 1024 * 1024 // 2, dtype=torch.bfloat16, device="cuda")
    for _ in range(3):
        dist.all_reduce(buf)
    torch.cuda.synchronize()
    x = torch.randn((8192, 8192), dtype=torch.bfloat16, device="cuda")
    w = torch.randn((8192, 8192), dtype=torch.bfloat16, device="cuda")
    gemm = lambda: torch.nn.functional.linear(x, w)    # 2048 tiles: 19 waves, ~4 ms
    for _ in range(3):
        gemm()
    torch.cuda.synchronize()
    _, hi = torch.cuda.Stream.priority_range()
    s_comm, s_probe = torch.cuda.Stream(priority=hi), torch.cuda.Stream(priority=0)
    s_gemm = torch.cuda.Stream(priority=0)
    if rank != 0:                               # the same launches as rank 0, minus the probe
        for order in ORDERS:
            for _ in range(len(SMEM) * REPS):
                if order == "gemm_first":
                    with torch.cuda.stream(s_gemm):
                        gemm()
                with torch.cuda.stream(s_comm):
                    dist.all_reduce(buf)
                torch.cuda.synchronize()
        dist.destroy_process_group()
        return

    import squatter
    hit = squatter.probe_warmup(TOTAL_SM)       # module load and host alloc: before any overlap
    mod = squatter._probe_module()
    for s in SMEM:                              # every shared-memory size once, alone
        mod.launch_probe(hit.dev_ptr, TOTAL_SM, s)
    torch.cuda.synchronize()
    cur = torch.cuda.current_stream()
    rows = []
    for order, smem in [(o, s) for o in ORDERS for s in SMEM]:
        for rep in range(REPS):
            for i in range(TOTAL_SM):
                hit[i] = 0
            ev = lambda: torch.cuda.Event(enable_timing=True)
            base, c0, c1, p0, p1, g1 = ev(), ev(), ev(), ev(), ev(), ev()
            torch.cuda.synchronize()
            base.record(cur)
            if order == "gemm_first":
                with torch.cuda.stream(s_gemm):
                    gemm(); g1.record(s_gemm)
            else:
                g1.record(cur)
            with torch.cuda.stream(s_comm):
                c0.record(s_comm); dist.all_reduce(buf); c1.record(s_comm)
            # let the collective's kernel become resident, and the GEMM finish
            time.sleep(0.002 if order == "comm_first" else 0.008)
            with torch.cuda.stream(s_probe):
                p0.record(s_probe)
                mod.launch_probe(hit.dev_ptr, WAVES * TOTAL_SM, smem)
                p1.record(s_probe)
            torch.cuda.synchronize()
            el = base.elapsed_time
            got = [i for i, v in enumerate(hit.tolist()) if v]
            valid = el(p0) > max(el(c0), el(g1)) + 1.0 and el(p1) < el(c1)
            rows.append(dict(ctas=ctas, order=order, smem=smem, rep=rep, reached=len(got),
                             lost=TOTAL_SM - len(got), valid=valid,
                             probe_start=round(el(p0), 3), probe_end=round(el(p1), 3),
                             comm_start=round(el(c0), 3), comm_end=round(el(c1), 3),
                             gemm_end=round(el(g1), 3),
                             blocked_ids=" ".join(str(i) for i in range(TOTAL_SM) if i not in got)))
    new = not os.path.exists(out_path)
    with open(out_path, "a", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            wr.writeheader()
        wr.writerows(rows)
    for order, smem in [(o, s) for o in ORDERS for s in SMEM]:
        v = [r for r in rows if r["smem"] == smem and r["order"] == order]
        ok = [r["lost"] for r in v if r["valid"]]
        print(f"  {ctas:>2} CTA {order:10s} smem {smem:>6} B: lost {sorted(set(ok))} SMs  "
              f"({len(ok)}/{len(v)} valid; collective {v[0]['comm_end'] - v[0]['comm_start']:.1f} ms, "
              f"probe {v[0]['probe_start']:.2f}-{v[0]['probe_end']:.2f} ms)", flush=True)
    dist.destroy_process_group()
    q.put(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ctas", default=",".join(str(c) for c in CTAS))
    a = ap.parse_args()
    if os.path.exists(a.out):
        sys.exit(f"{a.out} exists; choose another name")
    mp.set_start_method("spawn", force=True)
    for c in [int(x) for x in a.ctas.split(",")]:
        with socket.socket() as sk:
            sk.bind(("127.0.0.1", 0)); port = str(sk.getsockname()[1])
        q = mp.Queue()
        ps = [mp.Process(target=worker, args=(r, port, c, a.out, q)) for r in range(WORLD)]
        for p in ps:
            p.start()
        try:
            q.get(timeout=600)
        except Exception:
            print(f"!! {c} CTA: no result ({[p.exitcode for p in ps]})", flush=True)
        for p in ps:
            p.join(timeout=60)
            if p.is_alive():
                p.terminate()
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
