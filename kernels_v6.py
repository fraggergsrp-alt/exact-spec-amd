"""SD6: multi-token (verify) versions of our decode kernels, so speculative verification stops falling back to PyTorch.

SD5 showed the verify step over 4-8 tokens costs 3.1 ms more than a 1-token step, and 98% of that is two fallbacks:
  * attention: PyTorch SDPA over the whole 1112-slot static cache with a mask, 114 us per layer (vs 18 us for our
    split-KV kernel with 1 token);
  * KV write: PyTorch arange + counter add + 2x index_copy, 12 us per layer.
This file adds:
  * attn_partial_mq / attn_reduce_mq: the split-KV attention v3 (kernels_v3.py) run for n queries at once. Query j
    attends to the first L_end - n + j + 1 cache rows (causal: it sees itself and earlier tokens, not later ones).
    Each query uses EXACTLY v3's arithmetic for its own length (same splits, same chunking, same summation order), so
    its output is bit-identical to calling v3 with one token at that length. The keys/values are read once per query
    from memory or the GPU's 64 MB last-level cache (4.5 MB per layer fits).
  * kv_write_n: writes n new key/value rows at the cache counter and advances it by n (one launch).
Prior art: multi-query / tree verification attention in speculative decoding systems (e.g. SpecInfer, Medusa, EAGLE
use custom verify-attention kernels); split-KV decode (FlashDecoding).

Run this file: correctness (fp32 reference, bitwise vs v3, kv write vs PyTorch), then a 28-layer timing chain.
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
import kernels_v3  # noqa: E402

SRC = r"""
typedef _Float16 half4 __attribute__((ext_vector_type(4)));
#define D 128
#define G 2
#define CMAX 256
#define NEG_INF (-__builtin_inff())

// grid (Hk, S, n). Identical to attn_partial_v3 except: query j = blockIdx.z, its length L_j, and output offsets.
extern "C" __global__ void attn_partial_mq(const _Float16* __restrict__ q, const _Float16* __restrict__ kc,
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
    for (int j = j0 + wave; j < j1; j += 8) {
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
    const long long base = (long long)jq * Hq + h;
    o_part[(base * S + s) * D + d] = o;
    if (d == 0) { ml_part[(base * S + s) * 2] = red[g][0]; ml_part[(base * S + s) * 2 + 1] = red[g][1]; }
}

// grid (Hq, n), block 128. Identical to attn_reduce_v3 per (query, head).
extern "C" __global__ void attn_reduce_mq(const float* __restrict__ o_part, const float* __restrict__ ml_part,
                                          _Float16* __restrict__ out, int S)
{
    const int h = blockIdx.x, jq = blockIdx.y, Hq = gridDim.x, d = threadIdx.x;
    const long long base = (long long)jq * Hq + h;
    float mx = NEG_INF;
    for (int s = 0; s < S; ++s) mx = fmaxf(mx, ml_part[(base * S + s) * 2]);
    float acc = 0.f, l = 0.f;
    for (int s = 0; s < S; ++s) {
        const float m = ml_part[(base * S + s) * 2];
        if (m == NEG_INF) continue;
        const float w = __expf(m - mx);
        acc += w * o_part[(base * S + s) * D + d];
        l += w * ml_part[(base * S + s) * 2 + 1];
    }
    out[base * D + d] = (_Float16)(acc / l);
}

// n new rows. k, v: contiguous [Hk, n, D]. Writes rows L..L+n-1 of each head and advances the counter by n.
extern "C" __global__ void kv_write_n(const _Float16* __restrict__ k, const _Float16* __restrict__ v,
                                      _Float16* __restrict__ kc, _Float16* __restrict__ vc,
                                      long long* __restrict__ len_ptr, int Hk, int n, int M)
{
    const long long L = *len_ptr;
    if (L + n > M) return;                                  // never write past the cache
    const int total = Hk * n * D;
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < total; i += blockDim.x * gridDim.x) {
        const int d = i % D, t = (i / D) % n, h = i / (D * n);
        const long long o = ((long long)h * M + L + t) * D + d;
        kc[o] = k[i];
        vc[o] = v[i];
    }
}
// v2: ONE launch. Reads k/v through their strides (no .contiguous() copies), one block writes all rows, then advances
// the counter itself (same pattern as fuse_small's 1-token kv_write). SD6a showed v1's 4 launches cost as much as PyTorch.
extern "C" __global__ void kv_write_n1(const _Float16* __restrict__ k, const _Float16* __restrict__ v,
                                       _Float16* __restrict__ kc, _Float16* __restrict__ vc,
                                       long long* __restrict__ len_ptr, int Hk, int n, int M,
                                       long long ksh, long long kst, long long vsh, long long vst)
{
    const long long L = *len_ptr;
    if (L + n > M) return;
    const int total = Hk * n * D;
    for (int i = threadIdx.x; i < total; i += blockDim.x) {
        const int d = i % D, t = (i / D) % n, h = i / (D * n);
        const long long o = ((long long)h * M + L + t) * D + d;
        kc[o] = k[h * ksh + t * kst + d];
        vc[o] = v[h * vsh + t * vst + d];
    }
    __syncthreads();
    if (threadIdx.x == 0) *len_ptr = L + n;
}
extern "C" __global__ void counter_add(long long* __restrict__ len_ptr, int n, int M)
{
    if (*len_ptr + n <= M) *len_ptr += n;
}
"""

_k = {}


def kernels():
    if not _k:
        for name in ("attn_partial_mq", "attn_reduce_mq", "kv_write_n", "kv_write_n1", "counter_add"):
            _k[name] = torch.cuda._compile_kernel(SRC, name)
    return _k


def attn_mq(q, kc, vc, len_t, scale, splits=None):
    """q: [1, Hq, n, D] (any strides). Cache counter len_t already includes the n new rows. Returns [1, n, Hq, D]."""
    Hq, n, Hk, M = q.shape[1], q.shape[2], kc.shape[1], kc.shape[2]
    assert q.shape[-1] == 128 and Hq == 2 * Hk and kc.is_contiguous() and vc.is_contiguous()
    S = max(splits or kernels_v3.SPLITS, -(-M // 256))        # same rule as v3 -> same chunking per query
    dev = q.device
    qf = q[0].transpose(0, 1).contiguous()                    # [n, Hq, D]
    o_part = torch.empty(n * Hq * S * 128, dtype=torch.float32, device=dev)
    ml = torch.empty(n * Hq * S * 2, dtype=torch.float32, device=dev)
    out = torch.empty(1, n, Hq, 128, dtype=torch.float16, device=dev)
    k = kernels()
    k["attn_partial_mq"](grid=(Hk, S, n), block=(256, 1, 1), args=[qf, kc, vc, len_t, o_part, ml, M, S, float(scale)])
    k["attn_reduce_mq"](grid=(Hq, n, 1), block=(128, 1, 1), args=[o_part, ml, out, S])
    return out


def kv_write_n(layer, key_states, value_states):
    """StaticLayer.update for n > 1 new tokens (cache already initialised). Same result as PyTorch's index_copy."""
    Hk, n, D = key_states.shape[1], key_states.shape[2], key_states.shape[3]
    k = key_states[0].contiguous()
    v = value_states[0].contiguous()
    kk = kernels()
    kk["kv_write_n"](grid=(max(1, min(64, Hk * n * D // 256)), 1, 1), block=(256, 1, 1),
                     args=[k, v, layer.keys, layer.values, layer.cumulative_length, Hk, n, layer.max_cache_len])
    kk["counter_add"](grid=(1, 1, 1), block=(1, 1, 1), args=[layer.cumulative_length, n, layer.max_cache_len])
    return layer.keys, layer.values


def kv_write_n1(layer, key_states, value_states):
    """v2 of kv_write_n: one launch, no copies. key_states/value_states: [1, Hk, n, D] views with unit stride on D."""
    Hk, n = key_states.shape[1], key_states.shape[2]
    k, v = key_states[0], value_states[0]
    if k.stride(-1) != 1:
        k = k.contiguous()
    if v.stride(-1) != 1:
        v = v.contiguous()
    kernels()["kv_write_n1"](grid=(1, 1, 1), block=(512, 1, 1),
                             args=[k, v, layer.keys, layer.values, layer.cumulative_length, Hk, n, layer.max_cache_len,
                                   k.stride(0), k.stride(1), v.stride(0), v.stride(1)])
    return layer.keys, layer.values


def install_kv_write(max_n=16):
    """Route StaticLayer.update with 2..max_n new tokens to kv_write_n (on top of fuse_small's 1-token kernel)."""
    from transformers import cache_utils
    SL = cache_utils.StaticLayer
    prev = SL.update

    def update(self, key_states, value_states, *args, **kwargs):
        n = key_states.shape[-2]
        if 1 < n <= max_n and self.is_initialized:
            return kv_write_n1(self, key_states, value_states)
        return prev(self, key_states, value_states, *args, **kwargs)
    SL.update = update
    return lambda: setattr(SL, "update", prev)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    wd = threading.Timer(240, lambda: (log("RESULT SD6a: TIMEOUT"), os._exit(3)))
    wd.daemon = True
    wd.start()
    dev = torch.device("cuda")
    try:
        kernels(); kernels_v3.kernels()
    except Exception as ex:
        log(f"RESULT SD6a: COMPILE FAILED -> {type(ex).__name__}: {str(ex)[:1500]}")
        sys.stdout.flush(); os._exit(1)
    log("compiled OK")
    kernels_v3.SPLITS = 16
    g = torch.Generator(device=dev).manual_seed(21)
    Hq, Hk, M, D = 16, 8, 1112, 128
    scale = 1 / math.sqrt(D)
    ok_all, bit_all = True, True
    for L_end, n in ((5, 4), (300, 4), (1028, 4), (1030, 6), (1032, 8), (1112, 8), (700, 2)):
        q = torch.randn(1, n, Hq, D, device=dev, generator=g).half().transpose(1, 2)    # same layout as the model
        kc = torch.randn(1, Hk, M, D, device=dev, generator=g).half()
        vc = torch.randn(1, Hk, M, D, device=dev, generator=g).half()
        len_t = torch.tensor(L_end, dtype=torch.int64, device=dev)
        out = attn_mq(q, kc, vc, len_t, scale)
        err, bit = 0.0, True
        for j in range(n):
            Lj = L_end - n + j + 1
            kr = kc[:, :, :Lj].float().repeat_interleave(2, 1)
            vr = vc[:, :, :Lj].float().repeat_interleave(2, 1)
            qj = q[:, :, j:j + 1].float()
            ref = (torch.softmax((qj @ kr.transpose(-1, -2)) * scale, -1) @ vr).transpose(1, 2)   # [1,1,Hq,D]
            err = max(err, float((out[:, j:j + 1].float() - ref).abs().max()))
            v3 = kernels_v3.decode_attn(q[:, :, j:j + 1], kc, vc, torch.tensor(Lj, dtype=torch.int64, device=dev), scale)
            bit &= bool(torch.equal(out[:, j:j + 1], v3))
        ok = err < 2e-3
        ok_all &= ok
        bit_all &= bit
        log(f"attention mq L_end={L_end:5d} n={n}: max err vs fp32 {err:.2e} -> {'CORRECT' if ok else 'WRONG'} | "
            f"bit-identical to v3 per query: {bit}")

    from types import SimpleNamespace
    for n in (2, 4, 8):
        keys = torch.randn(1, Hk, M, D, device=dev, generator=g).half()
        vals = torch.randn(1, Hk, M, D, device=dev, generator=g).half()
        lay = SimpleNamespace(keys=keys.clone(), values=vals.clone(), max_cache_len=M,
                              cumulative_length=torch.tensor(500, dtype=torch.int64, device=dev))
        ks = torch.randn(1, n, Hk, D, device=dev, generator=g).half().transpose(1, 2)
        vs = torch.randn(1, n, Hk, D, device=dev, generator=g).half().transpose(1, 2)
        lay2 = SimpleNamespace(keys=keys.clone(), values=vals.clone(), max_cache_len=M,
                               cumulative_length=torch.tensor(500, dtype=torch.int64, device=dev))
        kv_write_n(lay, ks, vs)
        kv_write_n1(lay2, ks, vs)
        pos = torch.arange(n, device=dev) + 500
        keys.index_copy_(2, pos, ks); vals.index_copy_(2, pos, vs)
        okw = torch.equal(lay.keys, keys) and torch.equal(lay.values, vals) and int(lay.cumulative_length) == 500 + n
        okw1 = torch.equal(lay2.keys, keys) and torch.equal(lay2.values, vals) and int(lay2.cumulative_length) == 500 + n
        ok_all &= okw and okw1
        log(f"kv_write_n n={n}: identical to PyTorch index_copy + counter: v1 {okw} | v2 (one launch, strided) {okw1}")
    if not ok_all:
        log("RESULT SD6a: all correct = False")
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
    res = {}
    q1 = torch.randn(1, 1, Hq, D, device=dev, generator=g).half().transpose(1, 2)
    len1 = torch.tensor(1025, dtype=torch.int64, device=dev)
    res["v3 n=1"] = tg(lambda: [kernels_v3.decode_attn(q1, k, v, len1, scale) for k, v in caches])
    for n in (1, 4, 6, 8):
        qn = torch.randn(1, n, Hq, D, device=dev, generator=g).half().transpose(1, 2)
        lenn = torch.tensor(1024 + n, dtype=torch.int64, device=dev)
        res[f"mq n={n}"] = tg(lambda: [attn_mq(qn, k, v, lenn, scale) for k, v in caches])
    log("28-layer attention chain, ~1025-1032 rows: " + " | ".join(f"{k} {t:.3f} ms" for k, t in res.items()))
    log(f"RESULT SD6a: all correct = True | bit-identical to v3 = {bit_all} | 28 layers: v3 n=1 {res['v3 n=1']:.3f} ms, "
        f"mq n=4 {res['mq n=4']:.3f}, n=8 {res['mq n=8']:.3f} ms (SDPA in-model SD5: ~3.2 ms for n=4-8)")
    wd.cancel()
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
