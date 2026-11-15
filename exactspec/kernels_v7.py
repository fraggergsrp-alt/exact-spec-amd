"""S4B: split-KV decode / verify attention for ANY group size G = query heads per key/value head (Qwen3-4B: G = 4).

Same design as kernels_v3 (1 token) and kernels_v6 (n tokens), which hard-code G = 2: blocks (kv head, split, query),
8 waves; score pass one key row per wave, softmax per head, value pass one value row per wave, partial results summed
over waves, then the v6 reduce kernel combines the splits. ONE kernel serves 1 and n tokens, so verify attention is
bit-identical to the 1-token decode attention by construction (same code, same chunking per query length).
Run this file: correctness vs float32 for G = 2 and 4, n = 1..8, and bit-identity of n-token vs 1-token results.
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
import kernels_v6  # noqa: E402  (reduce kernel)

SRC = r"""
typedef _Float16 half4 __attribute__((ext_vector_type(4)));
#define D 128
#define CMAX 256
#define NEG_INF (-__builtin_inff())

extern "C" __global__ void attn_partial_g(const _Float16* __restrict__ q, const _Float16* __restrict__ kc,
    const _Float16* __restrict__ vc, const long long* __restrict__ len_ptr, float* __restrict__ o_part,
    float* __restrict__ ml_part, int M, int S, double scale_d)
{
    const float scale = (float)scale_d;
    const int kvh = blockIdx.x, s = blockIdx.y, jq = blockIdx.z, nq = gridDim.z;
    const int Hq = gridDim.x * G;
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int L = (int)(*len_ptr) - nq + jq + 1;
    const int C = (L + S - 1) / S;
    const int j0 = s * C;
    const int j1 = min(j0 + C, L);
    __shared__ float qs[G][D];
    __shared__ float sc[G][CMAX];
    __shared__ float red[G][2];
    __shared__ float part[8][G][D];
    const _Float16* qj = q + (long long)jq * Hq * D;
    for (int i = tid; i < G * D; i += 256) qs[i / D][i % D] = (float)qj[(kvh * G + i / D) * D + i % D] * scale;
    __syncthreads();
    const _Float16* kb = kc + (long long)kvh * M * D;
    for (int j = j0 + wave; j < j1; j += 8) {
        half4 k4 = *(const half4*)(kb + (long long)j * D + lane * 4);
        float a[G];
        #pragma unroll
        for (int g = 0; g < G; ++g) a[g] = 0.f;
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            const float kv = (float)k4[t];
            #pragma unroll
            for (int g = 0; g < G; ++g) a[g] += qs[g][lane * 4 + t] * kv;
        }
        #pragma unroll
        for (int g = 0; g < G; ++g) {
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) a[g] += __shfl_xor(a[g], off, 32);
        }
        if (lane == 0) {
            #pragma unroll
            for (int g = 0; g < G; ++g) sc[g][j - j0] = a[g];
        }
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
    float c[G][4];
    #pragma unroll
    for (int g = 0; g < G; ++g) { c[g][0] = 0.f; c[g][1] = 0.f; c[g][2] = 0.f; c[g][3] = 0.f; }
    for (int j = j0 + wave; j < j1; j += 8) {
        half4 v4 = *(const half4*)(vb + (long long)j * D + lane * 4);
        #pragma unroll
        for (int g = 0; g < G; ++g) {
            const float p = sc[g][j - j0];
            #pragma unroll
            for (int t = 0; t < 4; ++t) c[g][t] += p * (float)v4[t];
        }
    }
    #pragma unroll
    for (int g = 0; g < G; ++g) {
        #pragma unroll
        for (int t = 0; t < 4; ++t) part[wave][g][lane * 4 + t] = c[g][t];
    }
    __syncthreads();
    for (int idx = tid; idx < G * D; idx += 256) {
        const int g = idx / D, d = idx % D;
        float o = 0.f;
        #pragma unroll
        for (int w = 0; w < 8; ++w) o += part[w][g][d];
        const int h = kvh * G + g;
        const long long base = (long long)jq * Hq + h;
        o_part[(base * S + s) * D + d] = o;
        if (d == 0) { ml_part[(base * S + s) * 2] = red[g][0]; ml_part[(base * S + s) * 2 + 1] = red[g][1]; }
    }
}
"""

_k = {}
SPLITS = int(os.environ.get("KDA7_SPLITS", "16"))


def kernel(G):
    if G not in _k:
        _k[G] = torch.cuda._compile_kernel(f"#define G {G}\n" + SRC, "attn_partial_g")
    return _k[G]


def attn_mq(q, kc, vc, len_t, scale, splits=None):
    """q: [1, Hq, n, D] (any strides); cache counter len_t already includes the n new rows. Returns [1, n, Hq, D]."""
    Hq, n, Hk, M = q.shape[1], q.shape[2], kc.shape[1], kc.shape[2]
    assert q.shape[-1] == 128 and Hq % Hk == 0 and kc.is_contiguous() and vc.is_contiguous()
    G = Hq // Hk
    S = max(splits or SPLITS, -(-M // 256))
    dev = q.device
    qf = q[0].transpose(0, 1).contiguous()
    o_part = torch.empty(n * Hq * S * 128, dtype=torch.float32, device=dev)
    ml = torch.empty(n * Hq * S * 2, dtype=torch.float32, device=dev)
    out = torch.empty(1, n, Hq, 128, dtype=torch.float16, device=dev)
    kernel(G)(grid=(Hk, S, n), block=(256, 1, 1), args=[qf, kc, vc, len_t, o_part, ml, M, S, float(scale)])
    kernels_v6.kernels()["attn_reduce_mq"](grid=(Hq, n, 1), block=(128, 1, 1), args=[o_part, ml, out, S])
    return out


def decode_attn(q, kc, vc, len_t, scale, splits=None):
    """1-token drop-in (same signature as kernels_v3.decode_attn)."""
    return attn_mq(q, kc, vc, len_t, scale, splits)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    wd = threading.Timer(240, lambda: (log("RESULT S4Ba: TIMEOUT"), os._exit(3)))
    wd.daemon = True
    wd.start()
    dev = torch.device("cuda")
    try:
        kernel(2); kernel(4); kernels_v6.kernels()
    except Exception as ex:
        log(f"RESULT S4Ba: COMPILE FAILED -> {type(ex).__name__}: {str(ex)[:1500]}")
        sys.stdout.flush(); os._exit(1)
    log("compiled OK")
    g = torch.Generator(device=dev).manual_seed(31)
    ok_all, bit_all = True, True
    for Hq, Hk in ((16, 8), (32, 8)):
        M, D = 1144, 128
        scale = 1 / math.sqrt(D)
        for L_end, n in ((1, 1), (300, 1), (1025, 1), (1028, 4), (1032, 8), (700, 3), (1144, 8)):
            q = torch.randn(1, n, Hq, D, device=dev, generator=g).half().transpose(1, 2)
            kc = torch.randn(1, Hk, M, D, device=dev, generator=g).half()
            vc = torch.randn(1, Hk, M, D, device=dev, generator=g).half()
            out = attn_mq(q, kc, vc, torch.tensor(L_end, dtype=torch.int64, device=dev), scale)
            err, bit = 0.0, True
            for j in range(n):
                Lj = L_end - n + j + 1
                kr = kc[:, :, :Lj].float().repeat_interleave(Hq // Hk, 1)
                vr = vc[:, :, :Lj].float().repeat_interleave(Hq // Hk, 1)
                ref = (torch.softmax((q[:, :, j:j + 1].float() @ kr.transpose(-1, -2)) * scale, -1) @ vr).transpose(1, 2)
                err = max(err, float((out[:, j:j + 1].float() - ref).abs().max()))
                one = decode_attn(q[:, :, j:j + 1], kc, vc, torch.tensor(Lj, dtype=torch.int64, device=dev), scale)
                bit &= bool(torch.equal(out[:, j:j + 1], one))
            ok = err < 2e-3
            ok_all &= ok; bit_all &= bit
            log(f"attention G={Hq // Hk} L_end={L_end:5d} n={n}: max err vs fp32 {err:.2e} -> {'CORRECT' if ok else 'WRONG'}"
                f" | n-token result bit-identical to 1-token calls: {bit}")
    log(f"RESULT S4Ba: all correct = {ok_all} | bit-identical = {bit_all}")
    wd.cancel()
    sys.stdout.flush()
    os._exit(0 if ok_all and bit_all else 2)


if __name__ == "__main__":
    main()
