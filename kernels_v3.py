"""S11: split-KV decode attention v3 - parallelise the value pass across waves.

Diagnosis of v1 (0.76 ms/token in-model, ~27 us per layer vs a ~7 us memory floor): in phase 1, after the softmax, all
256 threads (2 heads x 128 dims) walk through the chunk's ~32 keys ONE AFTER ANOTHER, each step waiting on a load of
one value row. v3 does the value pass the same way v1 already does the score pass: each of the 8 waves takes different
keys (one 8-byte load per lane covers a 256-byte value row), keeps partial outputs for both heads in registers, and the
8 waves are summed once in shared memory. Two launches as in v1 (S10 showed the single-launch "last block combines"
variant was slower). Prior art: FlashDecoding-style split-KV kernels.

Run this file: correctness vs float32 and v1 at 5 lengths, then isolated 28-layer chains v1 vs v3 for several splits.
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
import kernel_decode_attn as kda  # noqa: E402

SRC = r"""
typedef _Float16 half4 __attribute__((ext_vector_type(4)));
#define D 128
#define G 2
#define CMAX 256
#define NEG_INF (-__builtin_inff())

extern "C" __global__ void attn_partial_v3(const _Float16* __restrict__ q, const _Float16* __restrict__ kc,
    const _Float16* __restrict__ vc, const long long* __restrict__ len_ptr, float* __restrict__ o_part,
    float* __restrict__ ml_part, int M, int S, double scale_d)
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
    __shared__ float part[8][G][D];
    for (int i = tid; i < G * D; i += 256) qs[i / D][i % D] = (float)q[(kvh * G + i / D) * D + i % D] * scale;
    __syncthreads();

    const _Float16* kb = kc + (long long)kvh * M * D;
    for (int j = j0 + wave; j < j1; j += 8) {                         // scores: one key row per wave
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

    const _Float16* vb = vc + (long long)kvh * M * D;
    float c0[4] = {0.f, 0.f, 0.f, 0.f}, c1[4] = {0.f, 0.f, 0.f, 0.f};
    for (int j = j0 + wave; j < j1; j += 8) {                         // values: one value row per wave, in parallel
        half4 v4 = *(const half4*)(vb + (long long)j * D + lane * 4);
        const float p0 = sc[0][j - j0], p1 = sc[1][j - j0];
        #pragma unroll
        for (int t = 0; t < 4; ++t) { float vv = (float)v4[t]; c0[t] += p0 * vv; c1[t] += p1 * vv; }
    }
    #pragma unroll
    for (int t = 0; t < 4; ++t) { part[wave][0][lane * 4 + t] = c0[t]; part[wave][1][lane * 4 + t] = c1[t]; }
    __syncthreads();
    const int g = tid / D, d = tid % D;
    float o = 0.f;
    #pragma unroll
    for (int w = 0; w < 8; ++w) o += part[w][g][d];
    const int h = kvh * G + g;
    o_part[((long long)h * S + s) * D + d] = o;
    if (d == 0) { ml_part[(h * S + s) * 2] = red[g][0]; ml_part[(h * S + s) * 2 + 1] = red[g][1]; }
}

extern "C" __global__ void attn_reduce_v3(const float* __restrict__ o_part, const float* __restrict__ ml_part,
                                          _Float16* __restrict__ out, int S)
{
    const int h = blockIdx.x, d = threadIdx.x;
    float mx = NEG_INF;
    for (int s = 0; s < S; ++s) mx = fmaxf(mx, ml_part[(h * S + s) * 2]);
    float acc = 0.f, l = 0.f;
    for (int s = 0; s < S; ++s) {
        const float m = ml_part[(h * S + s) * 2];
        if (m == NEG_INF) continue;
        const float w = __expf(m - mx);
        acc += w * o_part[((long long)h * S + s) * D + d];
        l += w * ml_part[(h * S + s) * 2 + 1];
    }
    out[h * D + d] = (_Float16)(acc / l);
}
"""

_k = {}
SPLITS = int(os.environ.get("KDA3_SPLITS", "32"))


def kernels():
    if not _k:
        _k["p"] = torch.cuda._compile_kernel(SRC, "attn_partial_v3")
        _k["r"] = torch.cuda._compile_kernel(SRC, "attn_reduce_v3")
    return _k


def decode_attn(q, kc, vc, len_t, scale, splits=None):
    """Drop-in for kernel_decode_attn.decode_attn."""
    Hq, Hk, M = q.shape[1], kc.shape[1], kc.shape[2]
    assert q.shape[-1] == 128 and Hq == 2 * Hk and kc.is_contiguous() and vc.is_contiguous()
    S = max(splits or SPLITS, -(-M // 256))
    dev = q.device
    qf = q.reshape(Hq, 128).contiguous()
    o_part = torch.empty(Hq * S * 128, dtype=torch.float32, device=dev)
    ml = torch.empty(Hq * S * 2, dtype=torch.float32, device=dev)
    out = torch.empty(1, 1, Hq, 128, dtype=torch.float16, device=dev)
    k = kernels()
    k["p"](grid=(Hk, S, 1), block=(256, 1, 1), args=[qf, kc, vc, len_t, o_part, ml, M, S, float(scale)])
    k["r"](grid=(Hq, 1, 1), block=(128, 1, 1), args=[o_part, ml, out, S])
    return out


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    wd = threading.Timer(180, lambda: (log("RESULT S11a: TIMEOUT"), os._exit(3)))
    wd.daemon = True
    wd.start()
    dev = torch.device("cuda")
    try:
        kernels(); kda.get_kernels()
    except Exception as ex:
        log(f"RESULT S11a: COMPILE FAILED -> {type(ex).__name__}: {str(ex)[:1500]}")
        sys.stdout.flush(); os._exit(1)
    log("compiled OK")
    g = torch.Generator(device=dev).manual_seed(13)
    Hq, Hk, M, D = 16, 8, 1096, 128
    scale = 1 / math.sqrt(D)
    ok_all = True
    for L in (1, 7, 300, 1025, 1096):
        q = torch.randn(1, 1, Hq, D, device=dev, generator=g).half().transpose(1, 2)
        kc = torch.randn(1, Hk, M, D, device=dev, generator=g).half()
        vc = torch.randn(1, Hk, M, D, device=dev, generator=g).half()
        len_t = torch.tensor(L, dtype=torch.int64, device=dev)
        kr = kc[:, :, :L].float().repeat_interleave(2, 1)
        vr = vc[:, :, :L].float().repeat_interleave(2, 1)
        ref = (torch.softmax((q.float() @ kr.transpose(-1, -2)) * scale, -1) @ vr).transpose(1, 2)
        e1 = float((kda.decode_attn(q, kc, vc, len_t, scale).float() - ref).abs().max())
        e3 = max(float((decode_attn(q, kc, vc, len_t, scale, S).float() - ref).abs().max()) for S in (16, 32, 64))
        ok = e3 <= max(2 * e1, 2e-3)
        ok_all &= ok
        log(f"attention v3 L={L:5d}: max err vs fp32 v1 {e1:.2e} | v3 (S=16/32/64) {e3:.2e} -> {'CORRECT' if ok else 'WRONG'}")
    if not ok_all:
        log("RESULT S11a: all correct = False")
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

    caches = [(torch.randn(1, Hk, M, D, device=dev, generator=g).half(),
               torch.randn(1, Hk, M, D, device=dev, generator=g).half()) for _ in range(28)]
    q = torch.randn(1, 1, Hq, D, device=dev, generator=g).half().transpose(1, 2)
    len_t = torch.tensor(1025, dtype=torch.int64, device=dev)
    res = {"v1 S=32": tg(lambda: [kda.decode_attn(q, k, v, len_t, scale, 32) for k, v in caches])}
    for S in (8, 16, 32, 64):
        res[f"v3 S={S}"] = tg(lambda: [decode_attn(q, k, v, len_t, scale, S) for k, v in caches])
    kv_bytes = 28 * 2 * Hk * 1025 * D * 2
    log("28-layer chain, L=1025: " + " | ".join(f"{k} {t:.3f} ms ({kv_bytes / t / 1e6:.0f} GB/s)" for k, t in res.items()))
    best = min((k for k in res if k.startswith("v3")), key=res.get)
    log(f"RESULT S11a: all correct = True | v1 {res['v1 S=32']:.3f} ms -> {best} {res[best]:.3f} ms "
        f"({res['v1 S=32'] / res[best]:.2f}x) | floor {kv_bytes / 612e6:.3f} ms")
    wd.cancel()
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
