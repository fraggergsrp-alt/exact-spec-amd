"""S10: second versions of the two largest non-matrix kernels (measured with our probes in S8).

add_rmsnorm v2 (0.42 ms/token on 1.7B in v1, ~7.5 us per call): each thread loads its whole share of the row with ONE
16-byte (or 8-byte) vector load, keeps it in registers (no second pass over memory), and the block reduces with wave
shuffles + one barrier instead of an 8-step shared-memory tree. Same float16 rounding points as v1; only the order of
the float32 sum of squares changes (may flip a last bit occasionally).

split-KV attention v2 (0.76 ms/token in v1, ~27 us per layer): the separate combine kernel is removed - the last block
of each KV head to finish combines all chunks ("last block reduces", using an atomic counter that it resets to 0), and
the value loop is unrolled 4x so more memory loads are in flight. Prior art: FlashDecoding / split-K reductions.

Run this file: correctness vs float32 and vs v1, then isolated 28-layer chain timings v1 vs v2.
"""
import math
import os
import statistics
import sys
import threading
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fuse_small as fs  # noqa: E402
import kernel_decode_attn as kda  # noqa: E402

NORM_SRC = r"""
#pragma clang fp contract(off)
typedef _Float16 half8 __attribute__((ext_vector_type(8)));
typedef _Float16 half4 __attribute__((ext_vector_type(4)));

extern "C" __global__ void add_rmsnorm_v2_8(const _Float16* __restrict__ a, const _Float16* __restrict__ b,
    const _Float16* __restrict__ w, _Float16* __restrict__ hout, _Float16* __restrict__ y, int n_cols, double eps_d)
{
    const long long off = (long long)blockIdx.x * n_cols + threadIdx.x * 8;
    const half8 av = *(const half8*)(a + off), bv = *(const half8*)(b + off), wv = *(const half8*)(w + threadIdx.x * 8);
    half8 hv;
    float ss = 0.0f;
    #pragma unroll
    for (int e = 0; e < 8; ++e) { _Float16 s = (_Float16)((float)av[e] + (float)bv[e]); hv[e] = s; float f = (float)s; ss += f * f; }
    *(half8*)(hout + off) = hv;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor(ss, o, 32);
    __shared__ float red[8];
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
    __syncthreads();
    float tot = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) tot += red[i];
    const float inv = 1.0f / sqrtf(tot / (float)n_cols + (float)eps_d);
    half8 yv;
    #pragma unroll
    for (int e = 0; e < 8; ++e) { _Float16 normed = (_Float16)((float)hv[e] * inv); yv[e] = wv[e] * normed; }
    *(half8*)(y + off) = yv;
}

extern "C" __global__ void add_rmsnorm_v2_4(const _Float16* __restrict__ a, const _Float16* __restrict__ b,
    const _Float16* __restrict__ w, _Float16* __restrict__ hout, _Float16* __restrict__ y, int n_cols, double eps_d)
{
    const long long off = (long long)blockIdx.x * n_cols + threadIdx.x * 4;
    const half4 av = *(const half4*)(a + off), bv = *(const half4*)(b + off), wv = *(const half4*)(w + threadIdx.x * 4);
    half4 hv;
    float ss = 0.0f;
    #pragma unroll
    for (int e = 0; e < 4; ++e) { _Float16 s = (_Float16)((float)av[e] + (float)bv[e]); hv[e] = s; float f = (float)s; ss += f * f; }
    *(half4*)(hout + off) = hv;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor(ss, o, 32);
    __shared__ float red[8];
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
    __syncthreads();
    float tot = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) tot += red[i];
    const float inv = 1.0f / sqrtf(tot / (float)n_cols + (float)eps_d);
    half4 yv;
    #pragma unroll
    for (int e = 0; e < 4; ++e) { _Float16 normed = (_Float16)((float)hv[e] * inv); yv[e] = wv[e] * normed; }
    *(half4*)(y + off) = yv;
}
"""

ATTN_SRC = r"""
typedef _Float16 half4 __attribute__((ext_vector_type(4)));
#define D 128
#define G 2
#define CMAX 256
#define NEG_INF (-__builtin_inff())

extern "C" __global__ void attn_v2(const _Float16* __restrict__ q, const _Float16* __restrict__ kc,
    const _Float16* __restrict__ vc, const long long* __restrict__ len_ptr, float* __restrict__ o_part,
    float* __restrict__ ml_part, int* __restrict__ counter, _Float16* __restrict__ out, int M, int S, double scale_d)
{
    const float scale = (float)scale_d;
    const int kvh = blockIdx.x, s = blockIdx.y;
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int L = (int)(*len_ptr);
    const int C = (L + S - 1) / S;
    const int j0 = s * C;
    const int j1 = min(j0 + C, L);
    __shared__ float qs[G][D];
    __shared__ float sc[G][CMAX];
    __shared__ float red[G][2];
    __shared__ int last;
    for (int i = tid; i < G * D; i += 256) qs[i / D][i % D] = (float)q[(kvh * G + i / D) * D + i % D] * scale;
    __syncthreads();

    const _Float16* kb = kc + (long long)kvh * M * D;
    for (int j = j0 + wave; j < j1; j += 8) {
        half4 k4 = *(const half4*)(kb + (long long)j * D + lane * 4);
        float a0 = 0.f, a1 = 0.f;
        #pragma unroll
        for (int t = 0; t < 4; ++t) { float kv = (float)k4[t]; a0 += qs[0][lane * 4 + t] * kv; a1 += qs[1][lane * 4 + t] * kv; }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) { a0 += __shfl_xor(a0, off, 32); a1 += __shfl_xor(a1, off, 32); }
        if (lane == 0) { sc[0][j - j0] = a0; sc[1][j - j0] = a1; }
    }
    __syncthreads();
    if (wave < G) {
        const int g = wave, n = max(j1 - j0, 0);
        float m = NEG_INF;
        for (int j = lane; j < n; j += 32) m = fmaxf(m, sc[g][j]);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor(m, off, 32));
        float l = 0.f;
        for (int j = lane; j < n; j += 32) { float p = __expf(sc[g][j] - m); sc[g][j] = p; l += p; }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) l += __shfl_xor(l, off, 32);
        if (lane == 0) { red[g][0] = m; red[g][1] = l; }
    }
    __syncthreads();

    const int g = tid / D, d = tid % D;
    const _Float16* vb = vc + (long long)kvh * M * D + d;
    float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
    int j = j0;
    for (; j + 3 < j1; j += 4) {                     // 4 independent loads in flight
        const float v0 = (float)vb[(long long)j * D], v1 = (float)vb[(long long)(j + 1) * D];
        const float v2 = (float)vb[(long long)(j + 2) * D], v3 = (float)vb[(long long)(j + 3) * D];
        c0 += sc[g][j - j0] * v0; c1 += sc[g][j + 1 - j0] * v1; c2 += sc[g][j + 2 - j0] * v2; c3 += sc[g][j + 3 - j0] * v3;
    }
    for (; j < j1; ++j) c0 += sc[g][j - j0] * (float)vb[(long long)j * D];
    const int h = kvh * G + g;
    o_part[((long long)h * S + s) * D + d] = (c0 + c1) + (c2 + c3);
    if (d == 0) { ml_part[(h * S + s) * 2] = red[g][0]; ml_part[(h * S + s) * 2 + 1] = red[g][1]; }

    // last block of this KV head combines all chunks (no second kernel launch)
    __threadfence();
    __syncthreads();
    if (tid == 0) last = (atomicAdd(&counter[kvh], 1) == S - 1);
    __syncthreads();
    if (!last) return;
    __threadfence();
    const volatile float* mlv = ml_part;
    const volatile float* opv = o_part;
    float mx = NEG_INF;
    for (int t = 0; t < S; ++t) mx = fmaxf(mx, mlv[(h * S + t) * 2]);
    float acc = 0.f, l = 0.f;
    for (int t = 0; t < S; ++t) {
        const float m = mlv[(h * S + t) * 2];
        if (m == NEG_INF) continue;
        const float w = __expf(m - mx);
        acc += w * opv[((long long)h * S + t) * D + d];
        l += w * mlv[(h * S + t) * 2 + 1];
    }
    out[h * D + d] = (_Float16)(acc / l);
    if (tid == 0) counter[kvh] = 0;
}
"""

_k = {}
_counter = {}


def kernels():
    if not _k:
        _k["n8"] = torch.cuda._compile_kernel(NORM_SRC, "add_rmsnorm_v2_8")
        _k["n4"] = torch.cuda._compile_kernel(NORM_SRC, "add_rmsnorm_v2_4")
        _k["a"] = torch.cuda._compile_kernel(ATTN_SRC, "attn_v2")
    return _k


def add_rmsnorm(a, b, w, eps):
    """Drop-in for fuse_small.add_rmsnorm. Falls back to v1 for shapes/alignments v2 does not handle."""
    shape = a.shape
    n = shape[-1]
    a2, b2 = a.reshape(-1, n).contiguous(), b.reshape(-1, n).contiguous()
    E = n // 256
    if n % 256 or E not in (4, 8) or any(t.data_ptr() % 16 for t in (a2, b2, w)):
        return fs_add_rmsnorm_v1(a, b, w, eps)
    h, y = torch.empty_like(a2), torch.empty_like(a2)
    kernels()["n8" if E == 8 else "n4"](grid=(a2.shape[0], 1, 1), block=(256, 1, 1), args=[a2, b2, w, h, y, n, float(eps)])
    return h.view(shape), y.view(shape)


fs_add_rmsnorm_v1 = fs.add_rmsnorm


def decode_attn(q, kc, vc, len_t, scale, splits=None):
    """Drop-in for kernel_decode_attn.decode_attn (one launch instead of two)."""
    Hq, Hk, M = q.shape[1], kc.shape[1], kc.shape[2]
    assert q.shape[-1] == 128 and Hq == 2 * Hk and kc.is_contiguous() and vc.is_contiguous()
    S = max(splits or kda.SPLITS, -(-M // 256))
    dev = q.device
    if dev not in _counter:
        _counter[dev] = torch.zeros(64, dtype=torch.int32, device=dev)   # allocated once, before graph capture
    qf = q.reshape(Hq, 128).contiguous()
    o_part = torch.empty(Hq * S * 128, dtype=torch.float32, device=dev)
    ml = torch.empty(Hq * S * 2, dtype=torch.float32, device=dev)
    out = torch.empty(1, 1, Hq, 128, dtype=torch.float16, device=dev)
    kernels()["a"](grid=(Hk, S, 1), block=(256, 1, 1),
                   args=[qf, kc, vc, len_t, o_part, ml, _counter[dev], out, M, S, float(scale)])
    return out


def install():
    """Swap the v1 kernels for v2 in fuse_small / kernel_decode_attn. Returns undo()."""
    kernels()
    old_n, old_a = fs.add_rmsnorm, kda.decode_attn
    fs.add_rmsnorm, kda.decode_attn = add_rmsnorm, decode_attn

    def undo():
        fs.add_rmsnorm, kda.decode_attn = old_n, old_a
    return undo


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    wd = threading.Timer(180, lambda: (log("RESULT S10a: TIMEOUT"), os._exit(3)))
    wd.daemon = True
    wd.start()
    dev = torch.device("cuda")
    try:
        kernels(); fs.kernels(); kda.get_kernels()
    except Exception as ex:
        log(f"RESULT S10a: COMPILE FAILED -> {type(ex).__name__}: {str(ex)[:1500]}")
        sys.stdout.flush(); os._exit(1)
    log("compiled OK")
    g = torch.Generator(device=dev).manual_seed(9)
    ok_all = True
    for rows, cols in ((1, 1024), (1, 2048), (1024, 1024), (1024, 2048)):
        a = torch.randn(rows, cols, device=dev, generator=g).half()
        b = torch.randn(rows, cols, device=dev, generator=g).half()
        w = (torch.rand(cols, device=dev, generator=g) + 0.5).half()
        hr = (a.float() + b.float()).half()
        yr = w.float() * (hr.float() * torch.rsqrt(hr.float().pow(2).mean(-1, keepdim=True) + 1e-6))
        h1, y1 = fs_add_rmsnorm_v1(a, b, w, 1e-6)
        h2, y2 = add_rmsnorm(a, b, w, 1e-6)
        e1, e2 = float((y1.float() - yr).abs().max()), float((y2.float() - yr).abs().max())
        nd = int((y1 != y2).sum())
        ok = bool(torch.equal(h1, h2)) and e2 <= 2 * e1 + 1e-3
        ok_all &= ok
        log(f"add_rmsnorm v2 {rows}x{cols}: residual identical {torch.equal(h1, h2)} | max err vs fp32 v1 {e1:.2e} v2 "
            f"{e2:.2e} | outputs differing from v1: {nd} of {rows * cols} -> {'CORRECT' if ok else 'WRONG'}")
    Hq, Hk, M, D = 16, 8, 1096, 128
    scale = 1 / math.sqrt(D)
    for L in (1, 7, 300, 1025, 1096):
        q = torch.randn(1, 1, Hq, D, device=dev, generator=g).half().transpose(1, 2)
        kc = torch.randn(1, Hk, M, D, device=dev, generator=g).half()
        vc = torch.randn(1, Hk, M, D, device=dev, generator=g).half()
        len_t = torch.tensor(L, dtype=torch.int64, device=dev)
        kr = kc[:, :, :L].float().repeat_interleave(2, 1)
        vr = vc[:, :, :L].float().repeat_interleave(2, 1)
        ref = (torch.softmax((q.float() @ kr.transpose(-1, -2)) * scale, -1) @ vr).transpose(1, 2)
        o1 = kda.decode_attn(q, kc, vc, len_t, scale).float()
        errs = []
        for _ in range(3):                                   # repeat: the counter must reset itself every call
            errs.append(float((decode_attn(q, kc, vc, len_t, scale).float() - ref).abs().max()))
        e1 = float((o1 - ref).abs().max())
        ok = max(errs) <= max(2 * e1, 2e-3)
        ok_all &= ok
        log(f"attention v2 L={L:5d}: max err vs fp32 v1 {e1:.2e} | v2 x3 {max(errs):.2e} -> {'CORRECT' if ok else 'WRONG'}")
    if not ok_all:
        log("RESULT S10a: all correct = False")
        sys.stdout.flush(); os._exit(2)

    def tg(fn, reps=21):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            fn(); fn()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            fn()
        for _ in range(5):
            gr.replay()
        torch.cuda.synchronize()
        ts = []
        for _ in range(reps):
            torch.cuda.synchronize(); t = time.perf_counter(); gr.replay(); torch.cuda.synchronize()
            ts.append((time.perf_counter() - t) * 1000)
        return statistics.median(ts)

    res = {}
    for h in (1024, 2048):
        a = torch.randn(1, 1, h, device=dev, generator=g).half()
        b = torch.randn(1, 1, h, device=dev, generator=g).half()
        w = (torch.rand(h, device=dev, generator=g) + 0.5).half()
        res[f"norm{h}_v1"] = tg(lambda: [fs_add_rmsnorm_v1(a, b, w, 1e-6) for _ in range(56)])
        res[f"norm{h}_v2"] = tg(lambda: [add_rmsnorm(a, b, w, 1e-6) for _ in range(56)])
    caches = [(torch.randn(1, Hk, M, D, device=dev, generator=g).half(),
               torch.randn(1, Hk, M, D, device=dev, generator=g).half()) for _ in range(28)]
    q = torch.randn(1, 1, Hq, D, device=dev, generator=g).half().transpose(1, 2)
    len_t = torch.tensor(1025, dtype=torch.int64, device=dev)
    res["attn_v1"] = tg(lambda: [kda.decode_attn(q, k, v, len_t, scale) for k, v in caches])
    res["attn_v2"] = tg(lambda: [decode_attn(q, k, v, len_t, scale) for k, v in caches])
    log(f"chains (ms per token): 56 norms h=1024 v1 {res['norm1024_v1']:.3f} -> v2 {res['norm1024_v2']:.3f} | h=2048 "
        f"v1 {res['norm2048_v1']:.3f} -> v2 {res['norm2048_v2']:.3f} | 28 attention v1 {res['attn_v1']:.3f} -> v2 "
        f"{res['attn_v2']:.3f}")
    log("RESULT S10a: all correct = True")
    wd.cancel()
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
