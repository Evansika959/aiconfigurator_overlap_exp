#!/usr/bin/env python3
"""Write the JSON plans sweep_spans.py runs. One function per campaign; each says what
the campaign is for. Plans are small and deterministic, so they are regenerated rather
than edited by hand:

  python3 plan_spans.py            # writes data/sim_overlap/plan_<name>.json for all
  python3 plan_spans.py cal blind  # only these

SEEN vs UNSEEN. `cal`, `world`, `hetero` and `unbalanced` are calibration data: the
simulator's overlap rules may be fitted on them. `blind` and `blind_multi` are a
registered test: their cells use message sizes, CTA counts, clocks and GEMM shapes that
no earlier campaign measured, and the simulator's predictions for them are written to
disk before the first of them is run (see ../moe_megakernel/OVERLAP_HELDOUT.md).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "sim_overlap")

# the six shapes every campaign after the first one used (run_ext_campaign.sh)
SEEN_SHAPES = ["1024x4096x4096", "2048x4096x4096", "4096x4096x4096",
               "1024x8192x8192", "2048x8192x8192", "8192x8192x8192"]
# Never measured before. Chosen for tile counts between and beyond the seen ones
# (ceil(M/256) * ceil(N/128): 96 ... 3072), a rectangular weight and a long K.
BLIND_SHAPES = ["768x4096x4096", "3072x4096x4096", "6144x4096x4096", "1536x8192x8192",
                "3072x8192x8192", "6144x8192x8192", "4096x12288x4096", "2048x2048x8192"]
BLIND_CLOCKS = [1350, 1125, 975, 810, 600]      # seen: 1410 1305 1200 1050 900 705 510 300
BLIND_CTAS = [3, 6, 10, 24, 40]                 # seen: 1 2 4 8 12 16 22 32 48 64
BLIND_MIB = [32, 96, 384]                       # seen: 64 128 256 512


def grid(clocks, mibs, shapes, modes=("concurrent",), solo=True):
    runs = []
    for f in clocks:
        if solo:
            runs += [dict(mode="gemm_only", clock=f, shape=s, once=True) for s in shapes]
        for mib in mibs:
            if solo:
                runs.append(dict(mode="comm_only", clock=f, mib=mib))
            runs += [dict(mode=mo, clock=f, mib=mib, shape=s) for s in shapes for mo in modes]
    return runs


def smoke():
    """Every mode once, two shapes, two CTA counts: does the harness run."""
    runs = grid([1200], [64], SEEN_SHAPES[1:3], ("concurrent", "concurrent_gf", "serial"))
    runs.append(dict(mode="concurrent", clock=[1410, 900, 900, 900], mib=64,
                     shape=["4096x4096x4096", "1024x4096x4096", "1024x4096x4096", "1024x4096x4096"]))
    return dict(world=4, ctas=[4, 16], runs=runs)


def cal():
    """Spans on cells the earlier campaigns measured without them: what the GEMM and the
    collective each do inside an overlapped iteration, in both launch orders."""
    return dict(world=4, ctas=[2, 8, 16, 32, 48],
                runs=grid([1200, 1410, 900, 510], [64, 256], SEEN_SHAPES, ("concurrent", "concurrent_gf")))


def blind():
    """The registered test (single clock for every GPU, the same GEMM on every GPU)."""
    return dict(world=4, ctas=BLIND_CTAS, runs=grid(BLIND_CLOCKS, BLIND_MIB, BLIND_SHAPES))


def blind_auto():
    """The same unseen cells under the stock governor, clock not locked."""
    return dict(world=4, ctas=BLIND_CTAS[1::2], runs=grid([0], BLIND_MIB, BLIND_SHAPES))


def world(n):
    """The all-reduce and the overlap on 2 and 3 GPUs: how the collective's rate depends
    on the number of participants."""
    return dict(world=n, ctas=[2, 8, 32],
                runs=grid([1200, 900], [64, 256], SEEN_SHAPES[2:5:2] + SEEN_SHAPES[5:]))


def commsize():
    """The all-reduce alone over message size: the earlier campaigns fitted latency and
    bandwidth on 64-512 MiB only, and on one size at 1 CTA. 32, 96 and 384 MiB stay
    unmeasured: they are the blind test's."""
    runs = [dict(mode="comm_only", clock=f, mib=mib)
            for f in (1410, 1200, 1050, 900, 705, 510) for mib in (4, 8, 16, 128, 512)]
    return dict(world=4, ctas=[1, 2, 4, 8, 12, 16, 22, 32, 48, 64], runs=runs)


def endphase():
    """Why does a long GEMM cost the collective more when it is still running at the
    collective's end? One clock, two CTA counts, the two GEMMs that show it and one that
    does not, and the message stepped from 16 to 256 MiB so the collective's end sweeps
    across the GEMM's waves. (Not 32 or 96 MiB: those are the blind test's.)"""
    runs = []
    for mib in (16, 48, 64, 80, 112, 128, 160, 192, 224, 256):
        runs.append(dict(mode="comm_only", clock=1200, mib=mib))
        runs += [dict(mode="concurrent", clock=1200, mib=mib, shape=s)
                 for s in ("8192x8192x8192", "2048x8192x8192", "4096x4096x4096")]
    return dict(world=4, ctas=[8, 16], runs=runs)


HETERO = [  # one clock per GPU: how a collective behaves when its participants differ
    [1410, 1410, 1410, 1410], [900, 1410, 1410, 1410], [900, 900, 1410, 1410],
    [900, 900, 900, 1410], [900, 900, 900, 900], [1410, 900, 1410, 900],
    [510, 1410, 1410, 1410], [510, 510, 510, 1410], [510, 510, 510, 510],
    [1200, 1200, 1200, 1200], [1200, 900, 705, 510],
]


def hetero():
    runs = []
    for f in HETERO:
        runs += [dict(mode="comm_only", clock=f, mib=mib) for mib in (64, 256)]
        runs += [dict(mode="concurrent", clock=f, mib=mib, shape=s)
                 for mib in (64, 256) for s in ("4096x4096x4096", "8192x8192x8192")]
    return dict(world=4, ctas=[4, 16, 32], runs=runs)


BIG, SMALL = "8192x8192x8192", "2048x8192x8192"   # 2048 and 512 tiles


def unbalanced():
    """One GPU carries a GEMM four times the others' (a straggler, or the hot shard of an
    expert-parallel layer). Calibration half: every GPU at one clock, and the light GPUs
    slowed while the heavy one keeps its clock."""
    shapes = [BIG, SMALL, SMALL, SMALL]
    runs = [dict(mode="gemm_only", clock=f, shape=shapes, once=True) for f in (1410, 1200, 900, 705)]
    for mib in (64, 256):
        for f in ([1410] * 4, [1200] * 4, [900] * 4, [1200, 900, 900, 900], [1200, 705, 705, 705],
                  [1410, 900, 900, 900], [1200, 510, 510, 510]):
            runs.append(dict(mode="concurrent", clock=f, mib=mib, shape=shapes))
            runs.append(dict(mode="serial", clock=f, mib=mib, shape=shapes))
    return dict(world=4, ctas=[4, 16], runs=runs)


def step(kind, batch, hidden=8192, layers=3):
    """Ops of one synthetic training step on one rank. Per layer: a forward GEMM
    (batch x hidden x hidden); backward, the weight gradient (hidden x hidden, inner
    dimension the batch), whose all-reduce is the DDP bucket, and the input gradient.

      ddp_overlap   each gradient's all-reduce is issued as soon as its GEMM is, on the
                    collective's stream: it runs beside the GEMMs that follow
      ddp_serial    all gradients are reduced after the backward pass
      tp            tensor parallel: an activation all-reduce after every forward GEMM
    """
    f, gw = f"{batch}x{hidden}x{hidden}", f"{hidden}x{hidden}x{batch}"
    grad_mib, act_mib = hidden * hidden * 2 // 2**20, 64
    ops = []
    if kind == "tp":
        for i in range(1, layers + 1):
            ops.append(dict(op="gemm", tag=f"f{i}", stream="main", shape=f))
            ops.append(dict(op="allreduce", tag=f"a{i}", stream="main", mib=act_mib))
        return ops
    ops += [dict(op="gemm", tag=f"f{i}", stream="main", shape=f) for i in range(1, layers + 1)]
    for i in range(layers, 0, -1):
        ops.append(dict(op="gemm", tag=f"gw{i}", stream="main", shape=gw))
        if kind == "ddp_overlap":
            ops.append(dict(op="allreduce", tag=f"ar{i}", stream="comm", mib=grad_mib, wait=[f"gw{i}"]))
        if i > 1:
            ops.append(dict(op="gemm", tag=f"gx{i}", stream="main", shape=f))
    if kind == "ddp_serial":
        ops += [dict(op="allreduce", tag=f"ar{i}", stream="main", mib=grad_mib) for i in range(layers, 0, -1)]
    return ops


def train():
    """A training step as a program: balanced (every GPU the same batch) and unbalanced
    (GPU 0 four times the others'), at one clock and with the light GPUs slowed."""
    runs = []
    for f in (1200, 900, 1410, 0):
        for kind in ("ddp_overlap", "ddp_serial", "tp"):
            runs.append(dict(mode="program", name=f"{kind}_b4096", clock=f, ops=step(kind, 4096)))
    for f in ([1200] * 4, [1200, 900, 900, 900], [1200, 705, 705, 705], [1200, 510, 510, 510],
              [1410] * 4, [1410, 900, 900, 900], 0):
        for kind in ("ddp_overlap", "ddp_serial"):
            runs.append(dict(mode="program", name=f"{kind}_b8192+3x2048", clock=f,
                             ops=[step(kind, 8192)] + [step(kind, 2048)] * 3))
    return dict(world=4, ctas=[4, 16], runs=runs)


def waiting():
    """What a GPU draws while it waits for a slower one, two ways. GPU 0 runs three long
    GEMMs and then joins an all-reduce. The other three either issue their all-reduce at
    once, so its kernel sits resident and waiting (`in_collective`), or sleep on the host
    and issue it just before GPU 0 does (`idle`). The difference in their power is the
    waiting kernel's; `idle` alone is an idle board next to a busy one."""
    heavy = [dict(op="gemm", tag=f"g{i}", stream="main", shape="8192x8192x8192") for i in (1, 2, 3)]
    heavy.append(dict(op="allreduce", tag="ar", stream="main", mib=128))
    runs = []
    for f in (1410, 1200, 900, 510):
        for name, light in (("in_collective", [dict(op="allreduce", tag="ar", stream="main", mib=128)]),
                            ("idle", [dict(op="sleep", tag="z", ms=12.0), dict(op="allreduce", tag="ar", stream="main", mib=128)])):
            runs.append(dict(mode="program", name=f"wait_{name}", clock=[1200, f, f, f], ops=[heavy] + [light] * 3))
    return dict(world=4, ctas=[4, 16, 32], runs=runs)


def waiting_swap():
    """`waiting` with the heavy GEMMs on GPU 3 instead of GPU 0, so GPU 0 -- the board the
    power model was fitted on -- is one of those that wait."""
    p = waiting()
    runs = [dict(r, clock=r["clock"][1:] + r["clock"][:1], ops=r["ops"][1:] + r["ops"][:1]) for r in p["runs"]]
    return dict(world=4, ctas=[16], runs=runs)


def gaps():
    """A GEMM, then nothing for a few ms, over and over, with no collective anywhere: does
    a GPU fall back to its idle power between kernels a few ms apart? And the same with
    one small all-reduce per period, to see what the collective adds to the gap."""
    runs = []
    for f in (1410, 1200, 900, 510):
        for shape, ms in (("4096x4096x4096", 6.0), ("8192x8192x8192", 12.0)):
            g = dict(op="gemm", tag="g", stream="main", shape=shape)
            runs.append(dict(mode="program", name=f"gap_gemm_{shape.split('x')[0]}_{ms:g}ms", clock=f,
                             ops=[g, dict(op="sleep", tag="z", ms=ms)]))
            runs.append(dict(mode="program", name=f"gap_gemm+ar_{shape.split('x')[0]}_{ms:g}ms", clock=f,
                             ops=[g, dict(op="allreduce", tag="ar", stream="main", mib=16), dict(op="sleep", tag="z", ms=ms)]))
        runs.append(dict(mode="program", name="gap_ar_12ms", clock=f,
                         ops=[dict(op="allreduce", tag="ar", stream="main", mib=16), dict(op="sleep", tag="z", ms=12.0)]))
    return dict(world=4, ctas=[16], runs=runs)


BLIND_BIG, BLIND_SMALL = "6144x8192x8192", "1536x8192x8192"   # 1536 and 384 tiles


def blind_multi():
    """Registered test, multi-GPU half: a different clock on each GPU, and an
    unbalanced step, at shapes, sizes, CTA counts and clocks never measured."""
    runs = []
    for f in ([1350] * 4, [975, 1350, 1350, 1350], [975, 975, 975, 1350], [600, 1350, 1350, 1350],
              [1125, 975, 810, 600], [1125] * 4):
        runs.append(dict(mode="comm_only", clock=f, mib=96))
        runs.append(dict(mode="concurrent", clock=f, mib=96, shape="3072x8192x8192"))
    shapes = [BLIND_BIG] + [BLIND_SMALL] * 3
    for f in ([1350] * 4, [1125] * 4, [1350, 975, 975, 975], [1350, 810, 810, 810], [1125, 600, 600, 600]):
        runs.append(dict(mode="gemm_only", clock=f, shape=shapes, once=True))
        runs.append(dict(mode="concurrent", clock=f, mib=96, shape=shapes))
        runs.append(dict(mode="serial", clock=f, mib=96, shape=shapes))
    return dict(world=4, ctas=[6, 24], runs=runs)


def blind_train():
    """Registered test, training half: four layers instead of three, batches of 3072
    (balanced) and 6144 + 3 x 1536 (unbalanced), unseen clocks and CTA counts."""
    runs = []
    for f in (1350, 975, 0):
        for kind in ("ddp_overlap", "ddp_serial", "tp"):
            runs.append(dict(mode="program", name=f"{kind}_b3072_L4", clock=f, ops=step(kind, 3072, layers=4)))
    for f in ([1350] * 4, [1350, 975, 975, 975], [1350, 810, 810, 810], [1125, 600, 600, 600], 0):
        for kind in ("ddp_overlap", "ddp_serial"):
            runs.append(dict(mode="program", name=f"{kind}_b6144+3x1536_L4", clock=f,
                             ops=[step(kind, 6144, layers=4)] + [step(kind, 1536, layers=4)] * 3))
    return dict(world=4, ctas=[6, 24], runs=runs)


PLANS = dict(train=train, gaps=gaps, waiting=waiting, waiting_swap=waiting_swap, blind_multi=blind_multi, blind_train=blind_train, smoke=smoke, cal=cal, commsize=commsize, endphase=endphase, blind=blind, blind_auto=blind_auto, world2=lambda: world(2),
             world3=lambda: world(3), hetero=hetero, unbalanced=unbalanced)


def main():
    os.makedirs(OUT, exist_ok=True)
    for name in sys.argv[1:] or PLANS:
        plan = PLANS[name]()
        path = os.path.join(OUT, f"plan_{name}.json")
        with open(path, "w") as fh:
            json.dump(plan, fh, indent=0)
        n = sum(len(plan["ctas"]) if not r.get("once") else 1 for r in plan["runs"])
        print(f"{name:11s} world {plan['world']}  {len(plan['ctas'])} CTA counts  {n:5d} cells  "
              f"~{n * 4.6 / 60:5.1f} min  -> {os.path.relpath(path, HERE)}")


if __name__ == "__main__":
    main()
