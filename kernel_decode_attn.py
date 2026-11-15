"""S4: split-KV decode attention kernel (one query token), fifth hand-written kernel.

Everyday picture: instead of one worker reading a 1,000-page notebook alone, 16 workers each read a chunk at the
same time and then combine their notes. Technical name: split-KV / "FlashDecoding" (Dao et al., 2023) - prior art.

Phase 1 (grid: kv heads x splits): each block takes one KV head (both of its query heads, GQA) and one chunk of the
cache: scores q.k, a local softmax (max m, sum l) and an unnormalised partial output sum(p * v).
Phase 2 (grid: query heads): combine the chunks with the usual log-sum-exp rescaling.
The valid cache length is read ON THE GPU from StaticCache's own counter tensor, so the kernel works inside a graph.

Run this file to check correctness against an exact float32 reference and to time it against PyTorch SDPA as
transformers calls it (masked, packed GQA), as a chain of 28 layers in one graph.
"""
import math
import os
import statistics
import sys
import threading
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
D, CMAX = 128, 256

SRC = r"""
typedef _Float16 half4 __attribute__((ext_vector_type(4)));
#define D 128
#define G 2
#define CMAX 256
#define NEG_INF (-__builtin_inff())

extern "C" __global__ void attn_partial(const _Float16* __restrict__ q, const _Float16* __restrict__ kc,
    const _Float16* __restrict__ vc, const long long* __restrict__ len_ptr, float* __restrict__ o_part,
    float* __restrict__ ml_part, int M, int S, double scale_d)
{
    const float scale = (float)scale_d;                                      // torch passes Python floats as double
    const int kvh = blockIdx.x, s = blockIdx.y;
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;          // 256 threads = 8 waves of 32
    const int L = (int)(*len_ptr);
    const int C = (L + S - 1) / S;
    const int j0 = s * C;
    const int j1 = min(j0 + C, L);
    __shared__ float qs[G][D];
    __shared__ float sc[G][CMAX];
    __shared__ float red[G][2];
    for (int i = tid; i < G * D; i += 256) qs[i / D][i % D] = (float)q[(kvh * G + i / D) * D + i % D] * scale;
    __syncthreads();

    const _Float16* kb = kc + (long long)kvh * M * D;
    for (int j = j0 + wave; j < j1; j += 8) {                                // one key row (256 B) per wave
        half4 k4 = *(const half4*)(kb + (long long)j * D + lane * 4);
        float a0 = 0.f, a1 = 0.f;
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            float kv = (float)k4[t];
            a0 += qs[0][lane * 4 + t] * kv;
            a1 += qs[1][lane * 4 + t] * kv;
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) { a0 += __shfl_xor(a0, off, 32); a1 += __shfl_xor(a1, off, 32); }
        if (lane == 0) { sc[0][j - j0] = a0; sc[1][j - j0] = a1; }
    }
    __syncthreads();

    if (wave < G) {                                                          // wave g: softmax stats of head g
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

    const int g = tid / D, d = tid % D;                                      // 256 threads = G x D outputs
    const _Float16* vb = vc + (long long)kvh * M * D + d;
    float acc = 0.f;
    for (int j = j0; j < j1; ++j) acc += sc[g][j - j0] * (float)vb[(long long)j * D];
    const int h = kvh * G + g;
    o_part[((long long)h * S + s) * D + d] = acc;
    if (d == 0) { ml_part[(h * S + s) * 2] = red[g][0]; ml_part[(h * S + s) * 2 + 1] = red[g][1]; }
}

extern "C" __global__ void attn_reduce(const float* __restrict__ o_part, const float* __restrict__ ml_part,
                                       _Float16* __restrict__ out, int S)
{
    const int h = blockIdx.x, d = threadIdx.x;                               // 128 threads
    float mx = NEG_INF;
    for (int s = 0; s < S; ++s) mx = fmaxf(mx, ml_part[(h * S + s) * 2]);
    float acc = 0.f, l = 0.f;
    for (int s = 0; s < S; ++s) {
        const float m = ml_part[(h * S + s) * 2];
        if (m == NEG_INF) continue;                                        // empty chunk
        const float w = __expf(m - mx);
        acc += w * o_part[((long long)h * S + s) * D + d];
        l += w * ml_part[(h * S + s) * 2 + 1];
    }
    out[h * D + d] = (_Float16)(acc / l);
}
"""

_k = {}


def get_kernels():
    if not _k:
        _k["p"] = torch.cuda._compile_kernel(SRC, "attn_partial")
        _k["r"] = torch.cuda._compile_kernel(SRC, "attn_reduce")
    return _k["p"], _k["r"]


SPLITS = int(os.environ.get("KDA_SPLITS", "32"))  # S4c: S=32 was best in S4a


def decode_attn(q, kc, vc, len_t, scale, splits=None):
    """q: (1, Hq, 1, 128) fp16 (any view); kc, vc: (1, Hk, M, 128) fp16 contiguous; len_t: int64 0-dim GPU tensor with
    the number of valid cache slots. Returns (1, 1, Hq, 128) fp16, the layout transformers' attention functions return."""
    Hq, Hk, M = q.shape[1], kc.shape[1], kc.shape[2]
    assert q.shape[-1] == D and Hq == 2 * Hk and q.dtype == kc.dtype == torch.float16
    assert kc.is_contiguous() and vc.is_contiguous() and len_t.dtype == torch.int64
    S = max(splits or SPLITS, -(-M // CMAX))
    kp, kr = get_kernels()
    qf = q.reshape(Hq, D).contiguous()
    o_part = torch.empty(Hq * S * D, dtype=torch.float32, device=q.device)
    ml = torch.empty(Hq * S * 2, dtype=torch.float32, device=q.device)
    out = torch.empty(1, 1, Hq, D, dtype=torch.float16, device=q.device)
    kp(grid=(Hk, S, 1), block=(256, 1, 1), args=[qf, kc, vc, len_t, o_part, ml, M, S, float(scale)])
    kr(grid=(Hq, 1, 1), block=(128, 1, 1), args=[o_part, ml, out, S])
    return out


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sdpa_like_hf(q, kc, vc, mask, scale):
    """What transformers 5.23 does for one decode token with a mask: pack GQA, masked SDPA."""
    Hk = kc.shape[1]
    qp = q.reshape(1, Hk, q.shape[1] // Hk, D)
    o = F.scaled_dot_product_attention(qp, kc, vc, attn_mask=mask, scale=scale)
    return o.reshape(1, q.shape[1], 1, D).transpose(1, 2)


def time_graph(fn, reps=15):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn(); fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize(); t = time.perf_counter(); g.replay(); torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1000)
    del g
    return statistics.median(ts)


def main():
    wd = threading.Timer(180, lambda: (log("RESULT S4a: TIMEOUT"), os._exit(3)))
    wd.daemon = True
    wd.start()
    dev = torch.device("cuda")
    try:
        get_kernels()
    except Exception as ex:
        log(f"RESULT S4a: COMPILE FAILED -> {type(ex).__name__}: {str(ex)[:1500]}")
        sys.stdout.flush(); os._exit(1)
    log("compiled OK")
    gen = torch.Generator(device=dev).manual_seed(5)
    Hq, Hk, M = 16, 8, 1096
    scale = 1 / math.sqrt(D)
    all_ok = True
    for L in (1, 7, 300, 1024, 1025, 1096):
        q = torch.randn(1, 1, Hq, D, device=dev, generator=gen).half().transpose(1, 2)  # same view as in the model
        kc = torch.randn(1, Hk, M, D, device=dev, generator=gen).half()
        vc = torch.randn(1, Hk, M, D, device=dev, generator=gen).half()
        len_t = torch.tensor(L, dtype=torch.int64, device=dev)
        kr = kc[:, :, :L].float().repeat_interleave(2, 1)
        vr = vc[:, :, :L].float().repeat_interleave(2, 1)
        p = torch.softmax((q.float() @ kr.transpose(-1, -2)) * scale, -1)
        ref = (p @ vr).transpose(1, 2)                                   # (1, 1, Hq, D) exact-ish fp32
        mask = (torch.arange(M, device=dev) < L).view(1, 1, 1, M)
        ours = decode_attn(q, kc, vc, len_t, scale).float()
        sd = sdpa_like_hf(q, kc, vc, mask, scale).float()
        e_ours, e_sd = float((ours - ref).abs().max()), float((sd - ref).abs().max())
        ok = e_ours <= max(2 * e_sd, 2e-3)
        all_ok &= ok
        log(f"L={L:5d}: max error vs fp32 | ours {e_ours:.2e} | SDPA {e_sd:.2e} -> {'CORRECT' if ok else 'WRONG'}")
    if not all_ok:
        log("RESULT S4a: all correct = False")
        sys.stdout.flush(); os._exit(2)

    # timing: 28 layers, distinct caches (like the model), L = 1025 (1024-token prompt + 1)
    L = 1025
    caches = [(torch.randn(1, Hk, M, D, device=dev, generator=gen).half(),
               torch.randn(1, Hk, M, D, device=dev, generator=gen).half()) for _ in range(28)]
    q = torch.randn(1, 1, Hq, D, device=dev, generator=gen).half().transpose(1, 2)
    len_t = torch.tensor(L, dtype=torch.int64, device=dev)
    mask = (torch.arange(M, device=dev) < L).view(1, 1, 1, M)
    t_sd = time_graph(lambda: [sdpa_like_hf(q, k, v, mask, scale) for k, v in caches])
    res = {}
    for S in (8, 16, 32, 64, 128):
        res[S] = time_graph(lambda: [decode_attn(q, k, v, len_t, scale, S) for k, v in caches])
    kv_bytes = 28 * 2 * Hk * L * D * 2
    log(f"28-layer chain, L={L}: SDPA {t_sd:.3f} ms ({kv_bytes / t_sd / 1e6:.0f} GB/s) | " + " | ".join(
        f"ours S={S} {t:.3f} ms ({kv_bytes / t / 1e6:.0f} GB/s)" for S, t in res.items()))
    best = min(res, key=res.get)
    log(f"RESULT S4a: all correct = True | SDPA {t_sd:.3f} ms -> ours (S={best}) {res[best]:.3f} ms per token "
        f"({t_sd / res[best]:.2f}x) | KV-read floor at 612 GB/s {kv_bytes / 612e6:.3f} ms")
    wd.cancel()
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
