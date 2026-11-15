"""Q1: int4 weight-only quantization (group-32, symmetric round-to-nearest) + a one-token GEMV kernel.

Format: each weight -> 4-bit signed integer (-8..7, stored as an unsigned nibble with +8 offset), two per byte; one
fp16 scale per 32 consecutive weights of a row. 0.5625 bytes per weight vs 1.0625 for our int8-g32 and 2 for fp16.
This is essentially llama.cpp's Q4_0 layout idea (prior art; also GPTQ/AWQ, which choose the 4-bit values more cleverly -
this is the simple round-to-nearest baseline, so expect a larger quality cost than those methods).

Kernel: one wave per output row; each lane loads 16 bytes = exactly one group of 32 weights, unpacks the nibbles and
multiplies with the input vector held in shared memory; shuffle reduction. Prompt-length inputs use a dequantized copy.

Run this file: per-shape correctness vs an exact float32 result of the same quantized weights, quantization error, and
isolated speed vs our int8-g32 kernel and PyTorch fp16 (graph replay, rotating >= 256 MB of copies, as E13).
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
sys.path.insert(0, str(HERE))
import kernel_int8 as ki  # noqa: E402

G = 32
SRC = r"""
typedef unsigned char uchar16 __attribute__((ext_vector_type(16)));
typedef _Float16 half8 __attribute__((ext_vector_type(8)));

extern "C" __global__ void gemv_i4g(const unsigned char* __restrict__ Wq, const _Float16* __restrict__ scale,
                                    const _Float16* __restrict__ x, _Float16* __restrict__ y, int N, int K)
{
    __shared__ __attribute__((aligned(16))) _Float16 xs[XMAX];
    for (int i = threadIdx.x; i < K; i += blockDim.x) xs[i] = x[i];
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= N) return;
    const uchar16* wr = (const uchar16*)(Wq + (long long)row * (K >> 1));
    const _Float16* sr = scale + (long long)row * (K >> 5);
    const half8* xv = (const half8*)xs;
    float acc = 0.0f;
    for (int c = lane; c < (K >> 5); c += 32) {          // one group (32 weights, 16 bytes) per load
        const uchar16 w = wr[c];
        float p = 0.0f;
        #pragma unroll
        for (int b = 0; b < 4; ++b) {
            const half8 xa = xv[c * 4 + b];               // x[32c + 8b .. 32c + 8b + 7]
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                const unsigned int byte = w[b * 4 + e];   // weights 32c + 8b + 2e (low nibble) and +1 (high nibble)
                p += (float)((int)(byte & 15u) - 8) * (float)xa[2 * e];
                p += (float)((int)(byte >> 4) - 8) * (float)xa[2 * e + 1];
            }
        }
        acc += p * (float)sr[c];
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor(acc, off, 32);
    if (lane == 0) y[row] = (_Float16)acc;
}

"""
NRMAX, XMAXN = 8, 3072
SRC_N = r"""
typedef unsigned char uchar16 __attribute__((ext_vector_type(16)));
typedef _Float16 half8 __attribute__((ext_vector_type(8)));

// SD7: the same kernel for n input rows at once (n <= NRMAX, K <= XMAXN): the weights are read ONCE and multiplied by
// all n inputs (draft "ingest" of several accepted tokens). y: [n, N].
extern "C" __global__ void gemv_i4g_n(const unsigned char* __restrict__ Wq, const _Float16* __restrict__ scale,
                                      const _Float16* __restrict__ x, _Float16* __restrict__ y, int N, int K, int n)
{
    __shared__ __attribute__((aligned(16))) _Float16 xs[NRMAX * XMAXN];
    for (int i = threadIdx.x; i < n * K; i += blockDim.x) xs[(i / K) * XMAXN + (i % K)] = x[i];
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= N) return;
    const uchar16* wr = (const uchar16*)(Wq + (long long)row * (K >> 1));
    const _Float16* sr = scale + (long long)row * (K >> 5);
    float acc[NRMAX];
    #pragma unroll
    for (int r = 0; r < NRMAX; ++r) acc[r] = 0.0f;
    for (int c = lane; c < (K >> 5); c += 32) {
        const uchar16 w = wr[c];
        float wv[32];
        #pragma unroll
        for (int t = 0; t < 16; ++t) {
            const unsigned int byte = w[t];
            wv[2 * t] = (float)((int)(byte & 15u) - 8);
            wv[2 * t + 1] = (float)((int)(byte >> 4) - 8);
        }
        const float s = (float)sr[c];
        #pragma unroll
        for (int r = 0; r < NRMAX; ++r) {
            if (r < n) {
                const half8* xv = (const half8*)(xs + r * XMAXN);
                float p = 0.0f;
                #pragma unroll
                for (int b = 0; b < 4; ++b) {
                    const half8 xa = xv[c * 4 + b];
                    #pragma unroll
                    for (int e = 0; e < 8; ++e) p += wv[b * 8 + e] * (float)xa[e];
                }
                acc[r] += p * s;
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < NRMAX; ++r) {
        if (r < n) {
            float a = acc[r];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) a += __shfl_xor(a, off, 32);
            if (lane == 0) y[(long long)r * N + row] = (_Float16)a;
        }
    }
}
"""

_k = {}
_registry = {}          # id(fp16 weight) -> (packed uint8 [N, K/2], fp16 scale [N, K/32])
_prev_linear = None


def kernel(K):
    key = 4096 if K <= 4096 else 8192
    if key not in _k:
        _k[key] = torch.cuda._compile_kernel(SRC.replace("XMAX", str(key)), "gemv_i4g")
    return _k[key]


def kernel_n():
    if "n" not in _k:
        src = SRC_N.replace("NRMAX", str(NRMAX)).replace("XMAXN", str(XMAXN))
        _k["n"] = torch.cuda._compile_kernel(src, "gemv_i4g_n")
    return _k["n"]


def gemv_n(x, packed, scale):
    """x: [..., n, K] with n <= NRMAX rows, K <= XMAXN. Weights read once for all rows."""
    N, K = packed.shape[0], packed.shape[1] * 2
    n = x.numel() // K
    assert 1 <= n <= NRMAX and K <= XMAXN
    x1 = x.reshape(n, K).contiguous()
    y = torch.empty(n, N, dtype=torch.float16, device=x.device)
    kernel_n()(grid=((N + 7) // 8, 1, 1), block=(256, 1, 1), args=[packed, scale, x1, y, N, K, n])
    return y.view(*x.shape[:-1], N)


def quantize(W, search=False):
    """Symmetric int4, one fp16 scale per 32 weights. search=False: round-to-nearest with scale = amax/7 (Q1).
    search=True: per group, try 16 clip factors (1.00 -> 0.70) and keep the scale with the lowest squared error (Q2's
    variant B; same storage format). Returns (packed uint8 [N, K/2], scale [N, K/32])."""
    N, K = W.shape
    assert K % G == 0
    q_all = torch.empty(N, K, dtype=torch.int16, device=W.device)
    sc_all = torch.empty(N, K // G, dtype=torch.float16, device=W.device)
    for r in range(0, N, 4096):
        Wg = W[r:r + 4096].float().view(-1, K // G, G)
        amax = Wg.abs().amax(-1)
        best_q, best_s, best_e = None, None, None
        for f in ([1.0] if not search else [1.0 - 0.02 * i for i in range(16)]):
            sc = (amax * f / 7.0).clamp_min(1e-12).half().float()
            q = torch.round(Wg / sc[..., None]).clamp(-8, 7)
            e = (q * sc[..., None] - Wg).pow(2).sum(-1)
            if best_q is None:
                best_q, best_s, best_e = q, sc, e
            else:
                m = e < best_e
                best_q = torch.where(m[..., None], q, best_q)
                best_s = torch.where(m, sc, best_s)
                best_e = torch.minimum(e, best_e)
        q_all[r:r + 4096] = best_q.view(-1, K).to(torch.int16)
        sc_all[r:r + 4096] = best_s.to(torch.float16)
    q = q_all + 8
    packed = (q[:, 0::2] | (q[:, 1::2] << 4)).to(torch.uint8).contiguous()
    return packed, sc_all.contiguous()


def dequant(packed, scale):
    N, Kh = packed.shape
    lo = (packed & 15).to(torch.int16) - 8
    hi = (packed >> 4).to(torch.int16) - 8
    q = torch.stack([lo, hi], -1).view(N, Kh * 2)
    return (q.view(N, -1, G).to(torch.float16) * scale[..., None]).view(N, Kh * 2)


def gemv(x, packed, scale):
    N, K = packed.shape[0], packed.shape[1] * 2
    assert K <= 8192 and K % G == 0
    x1 = x.reshape(K).contiguous()
    y = torch.empty(N, dtype=torch.float16, device=x.device)
    kernel(K)(grid=((N + 7) // 8, 1, 1), block=(256, 1, 1), args=[packed, scale, x1, y, N, K])
    return y.view(*x.shape[:-1], N)


def quant_linear(input, weight, bias=None):
    entry = _registry.get(id(weight))
    if entry is None or bias is not None:
        return _prev_linear(input, weight, bias)
    packed, scale = entry
    K = packed.shape[1] * 2
    if input.numel() == K and input.dtype == torch.float16:
        return gemv(input, packed, scale)
    if input.dtype == torch.float16 and K <= XMAXN and K * 2 <= input.numel() <= K * NRMAX and input.numel() % K == 0:
        return gemv_n(input, packed, scale)                # SD7: several rows, weights read once
    return ki._orig_linear(input, dequant(packed, scale), None)


def install(weights, search=False):
    """Quantize the given fp16 weights to int4-g32 and route F.linear through quant_linear. Returns undo()."""
    global _prev_linear
    kernel(4096); kernel(8192)
    for w in weights:
        _registry[id(w)] = quantize(w, search=search)
    _prev_linear = F.linear
    F.linear = quant_linear

    def undo():
        F.linear = _prev_linear
        _registry.clear()
    return undo


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    wd = threading.Timer(240, lambda: (log("RESULT Q1a: TIMEOUT"), os._exit(3)))
    wd.daemon = True
    wd.start()
    dev = torch.device("cuda")
    try:
        kernel(4096); kernel(8192); ki.get_kernel_g(); ki.get_kernel_g_big()
    except Exception as ex:
        log(f"RESULT Q1a: COMPILE FAILED -> {type(ex).__name__}: {str(ex)[:1500]}")
        sys.stdout.flush(); os._exit(1)
    log("compiled OK")
    gen = torch.Generator(device=dev).manual_seed(4)
    shapes = [(4096, 1024), (1024, 2048), (6144, 1024), (1024, 3072), (4096, 2048), (2048, 2048), (12288, 2048),
              (2048, 6144), (151936, 1024), (151936, 2048)]          # (N out, K in) for both Qwen3 sizes
    all_ok, t = True, {"fp16": 0.0, "int8": 0.0, "int4": 0.0}
    for N, K in shapes:
        W = (torch.randn(N, K, device=dev, generator=gen) * 0.02).half()
        x = torch.randn(1, 1, K, device=dev, generator=gen).half()
        pk, sc = quantize(W)
        ref_q = dequant(pk, sc).float() @ x.float().reshape(K)
        ref_w = W.float() @ x.float().reshape(K)
        got = gemv(x, pk, sc).float().reshape(N)
        err = float((got - ref_q).abs().max())
        ok = err <= 4e-3 * float(ref_q.abs().max()) + 1e-3
        all_ok &= ok
        rel4 = float((ref_q - ref_w).norm() / ref_w.norm())
        q8, s8 = ki.quantize_group(W, 32)
        rel8 = float(((ki.dequant(q8, s8, 32).float() @ x.float().reshape(K)) - ref_w).norm() / ref_w.norm())
        copies = max(1, math.ceil(256 * 2**20 / (N * K * 2)))
        ws = [(torch.randn(N, K, device=dev, generator=gen) * 0.02).half() for _ in range(copies)]
        t16 = ki.time_rotating(lambda w: ki._orig_linear(x, w), ws)
        q8s = [ki.quantize_group(w, 32) for w in ws]
        t8 = ki.time_rotating(lambda q: ki.gemv_i8g(x, q[0], q[1], 32), q8s)
        del q8s
        q4s = [quantize(w) for w in ws]
        t4 = ki.time_rotating(lambda q: gemv(x, q[0], q[1]), q4s)
        del ws, q4s
        torch.cuda.empty_cache()
        b4 = N * K // 2 + N * (K // G) * 2
        log(f"N {N:6d} K {K:5d} | kernel err {err:.1e} -> {'CORRECT' if ok else 'WRONG'} | quant rel. error int8 "
            f"{100 * rel8:.2f}% int4 {100 * rel4:.2f}% | fp16 {t16 * 1e6:7.1f} us | int8 {t8 * 1e6:7.1f} us | int4 "
            f"{t4 * 1e6:7.1f} us ({b4 / t4 / 1e9:4.0f} GB/s) | int4 vs int8 {t8 / t4:4.2f}x")
    log(f"RESULT Q1a: all correct = {all_ok}")
    wd.cancel()
    sys.stdout.flush()
    os._exit(0 if all_ok else 2)


if __name__ == "__main__":
    main()
