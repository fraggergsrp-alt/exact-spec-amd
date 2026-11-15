"""SM1: exact, reproducible sampling for speculative decoding (pre-registered: docs/PREREG_SM1.md).

Token choice = argmax over the vocabulary of  logit[v] / T + G(seed, pos, v), where G is Gumbel noise computed from a
hash of (seed, position, vocabulary index) - no random-number state. This is the Gumbel-max trick: for independent
Gumbel noise it gives an exact sample from softmax(logit / T). Because the noise depends only on (seed, position, v),
the token at a position is a fixed function of that position's logits. With bit-identical logits (kernel_bi.py),
plain decoding and speculative verification therefore pick the same token at every position, so speculative output
equals plain sampled output token for token. The draft model uses the SAME noise, so it tends to pick the same token
as the target (shared-randomness coupling).
Prior art: Gumbel-max trick (Gumbel 1954; Maddison et al. 2014); drafter-invariant speculative decoding with common
random numbers (Daliri, Musco, Suresh, ISIT 2025); Gumbel-max list sampling (Rowan, Phan, Khisti, NeurIPS 2025);
counter-based random numbers (Salmon et al. 2011). New here is only the combination with batch-invariant kernels so
the equality holds bit-for-bit in fp16 on a consumer GPU.

Run this file: distribution test (sampled frequencies vs softmax), determinism, row-count invariance.
"""
import math
import os
import sys
import time

import torch

SRC = r"""
#pragma clang fp contract(off)
__device__ __forceinline__ unsigned long long mix64(unsigned long long z) {   // splitmix64 finaliser
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

// one block (256 threads) per row; fixed reduction order -> result independent of how many rows are in the call
extern "C" __global__ void gumbel_pick(const _Float16* __restrict__ logits, const long long* __restrict__ pos,
                                       long long* __restrict__ out, int V, int row_stride, double invT_d,
                                       int seed)
{
    const float invT = (float)invT_d;
    const int row = blockIdx.x;
    const _Float16* lg = logits + (long long)row * row_stride;
    const unsigned long long key = mix64((unsigned long long)(unsigned int)seed * 0x9E3779B97F4A7C15ULL
                                         ^ mix64((unsigned long long)pos[row] + 0x632BE59BD9B4E019ULL));
    float best = -__builtin_inff();
    int bi = 0x7fffffff;
    for (int v = threadIdx.x; v < V; v += blockDim.x) {
        const unsigned long long h = mix64(key ^ ((unsigned long long)v * 0xD1B54A32D192ED03ULL));
        const float u = ((float)(unsigned int)(h >> 40) + 0.5f) * (1.0f / 16777216.0f);   // (0, 1), 24 bits
        const float g = -logf(-logf(u));
        const float s = (float)lg[v] * invT + g;
        if (s > best) { best = s; bi = v; }        // v increases: ties keep the smaller index
    }
    __shared__ float sb[256];
    __shared__ int si[256];
    sb[threadIdx.x] = best; si[threadIdx.x] = bi;
    __syncthreads();
    for (int off = 128; off > 0; off >>= 1) {
        if (threadIdx.x < off) {
            const float b2 = sb[threadIdx.x + off]; const int i2 = si[threadIdx.x + off];
            if (b2 > sb[threadIdx.x] || (b2 == sb[threadIdx.x] && i2 < si[threadIdx.x])) {
                sb[threadIdx.x] = b2; si[threadIdx.x] = i2;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) out[row] = si[0];
}
"""

TEMP = float(os.environ.get("SD_TEMP", "0"))
SEED = int(os.environ.get("SD_SEED", "42"))
_k = {}


def kernel():
    if "k" not in _k:
        _k["k"] = torch.cuda._compile_kernel(SRC, "gumbel_pick")
    return _k["k"]


def pick(lg2, pos, temp=None, seed=None, out=None):
    """lg2: [n, V] fp16 logits (rows may be strided), pos: [n] int64 positions on the GPU.
    temp 0 -> greedy argmax (unchanged engine behaviour). Returns [n] int64 tokens."""
    temp = TEMP if temp is None else temp
    if temp <= 0:
        return lg2.argmax(-1)
    seed = SEED if seed is None else seed
    if lg2.dtype == torch.float32:                  # logits upcast from fp16 convert back exactly
        lg2 = lg2.half()
    n, V = lg2.shape
    assert lg2.dtype == torch.float16 and lg2.stride(1) == 1 and 0 <= seed < 2**31 and pos.dtype == torch.long and pos.numel() == n
    out = torch.empty(n, dtype=torch.long, device=lg2.device) if out is None else out
    kernel()(grid=(n, 1, 1), block=(256, 1, 1), args=[lg2, pos, out, V, lg2.stride(0), 1.0 / temp, seed])
    return out


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    dev = torch.device("cuda")
    ok = True
    # 1) distribution: one logit row repeated over 400k positions; frequencies vs softmax(logit / T)
    for temp in (0.7, 1.0):
        V = 16
        base = torch.tensor([3.0, 2.5, 2.0, 1.0, 0.5, 0.0, -0.5, -1.0, 1.5, 2.2, -2.0, 0.2, 0.9, -3.0, 1.1, 2.4])
        N = 400000
        lg = base.half().to(dev).expand(N, V).contiguous()
        pos = torch.arange(N, device=dev, dtype=torch.long)
        got = pick(lg, pos, temp=temp, seed=7)
        freq = torch.bincount(got, minlength=V).double().cpu() / N
        p = torch.softmax(base.half().double() / temp, 0)
        chi2 = float(((freq - p) ** 2 / p).sum() * N)
        dmax = float((freq - p).abs().max())
        good = chi2 < 37.7                          # 99.9th percentile of chi-square with 15 degrees of freedom
        ok &= good
        log(f"distribution T={temp}: max |freq - softmax| {dmax:.4f} | chi-square {chi2:.1f} (pass < 37.7) -> "
            f"{'OK' if good else 'FAIL'}")
    # 2) real vocabulary size: determinism + row-count invariance + seed / position sensitivity
    V = 151936
    g = torch.Generator(device=dev).manual_seed(3)
    lg = (torch.randn(32, V, device=dev, generator=g) * 3).half()
    pos = torch.arange(1000, 1032, device=dev, dtype=torch.long)
    a = pick(lg, pos, temp=0.8, seed=1)
    b = pick(lg, pos, temp=0.8, seed=1)
    one = torch.cat([pick(lg[i:i + 1], pos[i:i + 1], temp=0.8, seed=1) for i in range(32)])
    s2 = pick(lg, pos, temp=0.8, seed=2)
    p2 = pick(lg, pos + 1, temp=0.8, seed=1)
    det = torch.equal(a, b) and torch.equal(a, one)
    ok &= det
    log(f"V={V}: deterministic and row-count invariant: {det} | other seed changes {int((a != s2).sum())}/32 tokens | "
        f"other positions change {int((a != p2).sum())}/32 tokens")
    log(f"RESULT SM1-sampler: {'CORRECT' if ok else 'WRONG'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
