#!/usr/bin/env python3
"""SM squatter: occupy n SMs with a sleeping kernel so a co-running GEMM gets 108-n.

Mechanism. A thread block holds its SM for as long as it is resident. This kernel
does nothing but sleep, so launching it with grid=n on its own stream removes n SMs
from the pool available to anything launched on another stream.

Four things make it work, and each is easy to get silently wrong:

1. ONE BLOCK PER SM is not automatic. An A100 SM hosts up to 32 blocks / 2048
   threads / 164 KB shared memory, so a small block would let several squatters --
   or the GEMM itself -- share an SM and the limit would do nothing. The block
   claims SQUAT_SMEM_BYTES (160 KB of 164 KB) of dynamic shared memory, so neither a
   second squatter nor a large-GEMM block can fit. Over 48 KB needs the cudaFuncSetAttribute opt-in.
   VERIFIED: n distinct %smid values for n in {1, 27, 54, 94}.

2. RESIDENCY IS NOT LAUNCH ORDER, so the host must confirm it before starting the
   GEMM -- otherwise the GEMM may claim every SM first and the squatter merely
   queues behind it.

3. THE HANDSHAKE MUST NOT GO THROUGH A KERNEL. An earlier version kept `ready` and
   `stop` in device tensors and polled with `(ready != 0).sum().item()`. That reads
   device memory by *launching another kernel*, and the launch cannot complete until
   the squatter does: measured, the first poll blocked for the squatter's entire
   6.0 s backstop before returning. Both flags therefore live in MAPPED PINNED HOST
   memory -- the device writes them over the bus and the host CPU reads them with a
   plain load, no kernel, no stream synchronisation. Device-side writes use
   __threadfence_system(), not __threadfence(), because plain threadfence orders
   writes for other *devices*, not for the host.

4. THE SLEEP MUST NOT BE CLOCK-SCALED. __nanosleep counts nanoseconds, not cycles,
   so residency is unaffected by DVFS -- unlike a clock64() deadline, which would
   expire ~4.7x sooner at 300 MHz than at 1410 MHz. Sleep is issued at the
   documented ~1 ms maximum: fewer wake-ups means less memory traffic and less power
   contributed by the squatter itself, which lands in the same NVML window as the
   GEMM's.
"""

import ctypes
import glob
import os

import torch
from torch.utils.cpp_extension import load_inline

# 160 KB of A100's 164 KB. The binding constraint is the shared-memory CARVEOUT
# CLASS, not the byte count:
#   > 82 KB   -> a second squatter cannot fit, so n blocks occupy n distinct SMs
#   >= 132 KB -> the SM's carveout is raised to the 164 KB class
# A squatter in a LOWER carveout class than the co-running kernel poisons its TPC
# PARTNER SM as well, so at n >= 54 (one squatter per TPC, A100 has 54 TPCs) the
# 144 KB cuBLAS GEMM kernel cannot be placed anywhere and waits out the whole
# backstop. Measured: deadlock at 64/82/100/120 KB, clean at 140/160 KB.
# An earlier comment here blamed "64 KB of leftover letting some GEMM blocks in";
# that is arithmetically impossible -- every sweep shape uses
# ampere_bf16_s16816gemm_* with 147456 B, which never fits in 64 KB.
# 160 KB is safe for ANY co-running kernel because 164 KB is the maximum carveout.
SQUAT_SMEM_BYTES = 160 * 1024
CUBLAS_GEMM_SMEM_BYTES = 147456   # ampere_bf16_s16816gemm_*: the probe must match this
NANOSLEEP_NS = 1_000_000           # documented maximum is ~1 ms
_CUDA_HOST_ALLOC_MAPPED = 0x02

_rt = None


def _cudart():
    global _rt
    if _rt is None:
        cands = glob.glob("/usr/local/lib/python3.12/dist-packages/nvidia/cuda_runtime/lib/libcudart.so*")
        cands += glob.glob("/usr/local/cuda/lib64/libcudart.so*")
        _rt = ctypes.CDLL(cands[0])
        _rt.cudaHostAlloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint]
        _rt.cudaHostGetDevicePointer.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint]
        _rt.cudaFreeHost.argtypes = [ctypes.c_void_p]
    return _rt


class _MappedI32:
    """n int32s in mapped pinned host memory: host reads/writes with plain loads,
    device reads/writes through its own pointer. No kernel, no stream sync."""

    def __init__(self, n):
        rt = _cudart()
        self.n = n
        h = ctypes.c_void_p()
        rc = rt.cudaHostAlloc(ctypes.byref(h), 4 * n, _CUDA_HOST_ALLOC_MAPPED)
        if rc != 0:
            raise RuntimeError(f"cudaHostAlloc failed rc={rc}")
        d = ctypes.c_void_p()
        rc = rt.cudaHostGetDevicePointer(ctypes.byref(d), h, 0)
        if rc != 0:
            raise RuntimeError(f"cudaHostGetDevicePointer failed rc={rc}")
        self._h, self.dev_ptr = h, d.value
        self._arr = (ctypes.c_int32 * n).from_address(h.value)
        for i in range(n):
            self._arr[i] = 0

    def __getitem__(self, i):
        return self._arr[i]

    def __setitem__(self, i, v):
        self._arr[i] = v

    def tolist(self):
        return list(self._arr)

    def free(self):
        if self._h:
            _cudart().cudaFreeHost(self._h)
            self._h = None


_CUDA = r"""
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

__global__ void squat_kernel(volatile int* ready, volatile const int* stop,
                             volatile int* beat, long long max_iters) {
    extern __shared__ char smem[];
    (void)smem;                       // claimed for occupancy only, never touched
    if (threadIdx.x == 0) {
        unsigned smid;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
        ready[blockIdx.x] = (int)smid + 1;   // +1 so 0 unambiguously means "not resident"
        __threadfence_system();              // _system: make it visible to the HOST
    }
    __syncthreads();
    // stop -> host-controlled exit, covers a measurement window of any length
    // max_iters -> backstop so a dead host cannot leave the GPU occupied forever
    // beat -> per-block heartbeat. Without it a squatter that hits its backstop
    //         mid-measurement is invisible: `ready` still reads n/n afterwards and
    //         the config is silently recorded at the wrong SM count.
    for (long long i = 0; i < max_iters; ++i) {
        if (*stop) break;
        if (threadIdx.x == 0 && (i & 0x3F) == 0) {
            beat[blockIdx.x] = (int)(i >> 6) + 1;
            __threadfence_system();
        }
        __nanosleep(NANOSLEEP_NS_TOKEN);
    }
}

void launch_squat(long long ready_ptr, long long stop_ptr, long long beat_ptr,
                  long long max_iters, long long n_blocks, long long smem_bytes) {
    // Must be re-set whenever smem_bytes changes; caching it in a static made any
    // later, larger request fail with cudaErrorInvalidValue.
    cudaFuncSetAttribute(squat_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)smem_bytes);       // >48 KB dynamic smem needs this opt-in
    auto stream = at::cuda::getCurrentCUDAStream();
    squat_kernel<<<(int)n_blocks, 32, (size_t)smem_bytes, stream>>>(
        (volatile int*)ready_ptr, (volatile const int*)stop_ptr,
        (volatile int*)beat_ptr, max_iters);
    C10_CUDA_CHECK(cudaGetLastError());
}
""".replace("NANOSLEEP_NS_TOKEN", str(NANOSLEEP_NS))

_CPP = "void launch_squat(long long, long long, long long, long long, long long, long long);"
_mod = None


def _module():
    global _mod
    if _mod is None:
        _mod = load_inline(
            name="sm_squatter_v4",
            cpp_sources=_CPP,
            cuda_sources=_CUDA,
            functions=["launch_squat"],
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_80,code=sm_80"],
            verbose=False,
        )
    return _mod


class Squatter:
    """Occupies `n` SMs until released. Use as a context manager.

    with Squatter(54):        # 54 SMs taken, confirmed resident before returning
        ...                   # anything launched here sees 108-54 = 54 SMs
    """

    def __init__(self, n, device="cuda:0", smem_bytes=SQUAT_SMEM_BYTES, max_seconds=900.0):
        self.n = int(n)
        self.device = torch.device(device)
        self.smem = smem_bytes
        self.max_iters = max(1, int(max_seconds * 1e9 / NANOSLEEP_NS))
        self.stream = self.ready = self.stop = self.beat = None

    def __enter__(self):
        if self.n <= 0:
            return self
        torch.cuda.set_device(self.device)
        self.ready = _MappedI32(self.n)
        self.stop = _MappedI32(1)
        self.beat = _MappedI32(self.n)
        self.stream = torch.cuda.Stream(device=self.device)
        with torch.cuda.stream(self.stream):
            _module().launch_squat(self.ready.dev_ptr, self.stop.dev_ptr, self.beat.dev_ptr,
                                   self.max_iters, self.n, self.smem)
        try:
            self.wait_resident()
        except BaseException:
            # Without this the kernel stays resident for the full backstop and the
            # mapped buffers leak, so the NEXT configuration silently measures with
            # n_prev + n_next SMs held. A raise out of __enter__ never runs __exit__.
            self.__exit__()
            raise
        return self

    def wait_resident(self, timeout_s=30.0):
        """Block until all n blocks have published an SM id. Pure host-side polling
        of mapped memory -- no kernel launch, so it observes the squatter while it
        is still running."""
        import time
        t0 = time.time()
        while True:
            got = sum(1 for v in self.ready.tolist() if v != 0)
            if got == self.n:
                return time.time() - t0
            if time.time() - t0 > timeout_s:
                raise RuntimeError(f"squatter: only {got}/{self.n} blocks resident after {timeout_s}s")
            time.sleep(0.0005)

    def beats(self):
        return self.beat.tolist()

    BEAT_PERIOD_S = 64 * NANOSLEEP_NS / 1e9      # one beat every 64 sleeps

    def assert_alive(self, before, window_s, slack=0.25):
        """Every block must have beaten for essentially the WHOLE window.

        Requiring only `b > a` is not enough: a beat advances every ~64 ms, so a
        squatter that expired after one tick of a 1.85 s window still passes. Measured
        -- a 27-block squatter with a 1.2 s backstop died inside a 1.85 s window and
        the check let a 108-SM latency through labelled as 81 SMs. Require the beat
        count to cover at least (1-slack) of the elapsed window."""
        now = self.beats()
        need = max(1, int((1.0 - slack) * window_s / self.BEAT_PERIOD_S))
        short = [(i, b - a) for i, (a, b) in enumerate(zip(before, now)) if (b - a) < need]
        if short:
            worst = min(d for _, d in short)
            raise RuntimeError(
                f"squatter: {len(short)}/{self.n} blocks beat only {worst}/{need} times "
                f"over a {window_s:.2f}s window -- residency was not held for the whole "
                f"measurement; discard this configuration")
        return now

    def smids(self):
        """SM ids actually taken. len(set(...)) == n proves one block per SM."""
        return sorted(v - 1 for v in self.ready.tolist())

    def __exit__(self, *exc):
        if self.n <= 0:
            return False
        self.stop[0] = 1              # plain host store into mapped memory
        self.stream.synchronize()
        self.ready.free()
        self.stop.free()
        self.beat.free()
        return False


# ---------------------------------------------------------------------------
# Direct SM-availability probe.
#
# Checking that n squatters occupy n distinct SMs is NOT sufficient: it passes on
# configurations where the GEMM still only reaches 108-2n SMs, because a squatter in
# a lower shared-memory carveout class also excludes its TPC partner. Demonstrated by
# review: at 128 KB the smid test passes for every n while the 144 KB probe shows the
# GEMM confined to 54 SMs at n=27 instead of 81.
#
# The probe therefore carries EXACTLY the shared memory of the kernel under test, so
# it is subject to the same placement constraints, and reports the set of SMs that
# kernel can actually reach.
# ---------------------------------------------------------------------------
_PROBE_CUDA = r"""
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

__global__ void probe_kernel(int* hit) {
    extern __shared__ char smem[];
    (void)smem;
    if (threadIdx.x == 0) {
        unsigned smid;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
        hit[smid] = 1;
    }
}

void launch_probe(long long hit_ptr, long long n_blocks, long long smem_bytes) {
    cudaFuncSetAttribute(probe_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)smem_bytes);
    auto stream = at::cuda::getCurrentCUDAStream();
    probe_kernel<<<(int)n_blocks, 32, (size_t)smem_bytes, stream>>>((int*)hit_ptr);
    C10_CUDA_CHECK(cudaGetLastError());
}
"""
_probe_mod = None


def _probe_module():
    global _probe_mod
    if _probe_mod is None:
        _probe_mod = load_inline(
            name="sm_probe_v1",
            cpp_sources="void launch_probe(long long, long long, long long);",
            cuda_sources=_PROBE_CUDA,
            functions=["launch_probe"],
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_80,code=sm_80"],
            verbose=False,
        )
    return _probe_mod


_probe_buf = None


def probe_warmup(total_sm=108):
    """MUST be called before any squatter is resident.

    Two things in the probe path are device-wide syncs and would otherwise run inside
    the squatter's lifetime, waiting out its whole backstop -- and, worse, returning a
    wrong answer rather than an error:
      * load_inline's lazy module load. Measured: the FIRST call in a process blocks in
        the launch and reports 108 reachable SMs with full overlap, i.e. "no confinement",
        regardless of the truth.
      * cudaHostAlloc / cudaFreeHost. Measured 26.2 s against a 25 s backstop -- the probe
        outlives and kills the squatter it is measuring.
    So the module is loaded and the buffer allocated once, up front, and reused."""
    global _probe_buf
    _probe_module()
    if _probe_buf is None:
        _probe_buf = _MappedI32(total_sm)
    _probe_module().launch_probe(_probe_buf.dev_ptr, 2 * total_sm, CUBLAS_GEMM_SMEM_BYTES)
    torch.cuda.current_stream().synchronize()
    return _probe_buf


def probe_available_sms(smem_bytes=CUBLAS_GEMM_SMEM_BYTES, total_sm=108, waves=2):
    """Set of SM ids a kernel with `smem_bytes` of shared memory can currently reach.

    Requires probe_warmup() to have been called outside any squatter context.
    Stream-scoped sync only; allocates nothing.
    """
    if _probe_buf is None:
        raise RuntimeError("probe_warmup() must be called before any squatter is resident")
    for i in range(total_sm):
        _probe_buf[i] = 0
    _probe_module().launch_probe(_probe_buf.dev_ptr, waves * total_sm, smem_bytes)
    torch.cuda.current_stream().synchronize()
    return {i for i, v in enumerate(_probe_buf.tolist()) if v}
