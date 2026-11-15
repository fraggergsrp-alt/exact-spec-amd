"""BI1: batch-invariant fp16 matrix-vector kernel for 1..32 token rows (pre-registered: docs/PREREG_BI1.md).

Idea 5 ("bit-identical speculative decoding"): verifying 8 tokens at once must give, for every token, EXACTLY the bits
that decoding that token alone gives. Attention, norms, RoPE and the KV write in our engine are already per-row; the
only part that changes with the number of rows is the matrix multiply (PyTorch picks a single-row path for 1 token and
a different multi-row path for 8). Here ONE kernel serves every row count, and every output value is summed in exactly
the same order as kernel_f12.gemv_f16_ref: lane-strided chunks of 8 weights, fp32 accumulation, FMA contraction off,
then the same 5-step shuffle reduction. So the bits for row t can not depend on how many rows are in the call.
Prior art: batch-invariant kernels (Thinking Machines Lab, "Defeating Nondeterminism in LLM Inference", 2025; vLLM /
SGLang deterministic modes; LLM-42). Ours applies the idea at batch 1 to speculative verification on a consumer GPU.

Two variants with the same arithmetic: "s" keeps all input rows in shared memory (when n*K <= 24576 halves),
"g" reads them from global memory (L1/L2 cached). Weights are read once per group of up to 8 rows.

Run this file: bit-identity tests (n-row vs 1-row vs gemv_f16_ref) on the real shapes of Qwen3-1.7B and -4B, accuracy
vs an fp32 reference, and speed vs PyTorch for 1 and 8 rows. Writes results/BI1_kernel.json.
"""
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
NMAX = 32          # most rows per call handled here; bigger calls (prefill) fall back to PyTorch
LDS_MAX = 24576    # halves (48 KiB) of shared memory for the "s" variant

SRC = r"""
#pragma clang fp contract(off)
typedef _Float16 half8 __attribute__((ext_vector_type(8)));
#define NT 8

extern "C" __global__ void gemv_bi_s(const _Float16* __restrict__ W, const _Float16* __restrict__ x,
                                     _Float16* __restrict__ y, int N, int K, int n)
{
    __shared__ __attribute__((aligned(16))) _Float16 xs[XS];
    const int tot8 = (n * K) >> 3;
    for (int i = threadIdx.x; i < tot8; i += blockDim.x) ((half8*)xs)[i] = ((const half8*)x)[i];
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= N) return;
    const half8* wr = (const half8*)(W + (long long)row * K);
    const int K8 = K >> 3;
    for (int t0 = 0; t0 < n; t0 += NT) {
        const int nt = min(NT, n - t0);
        float acc[NT];
        #pragma unroll
        for (int t = 0; t < NT; ++t) acc[t] = 0.0f;
        for (int c = lane; c < K8; c += 32) {
            const half8 w = wr[c];
            #pragma unroll
            for (int t = 0; t < NT; ++t) {
                if (t < nt) {
                    const half8 v = ((const half8*)(xs + (t0 + t) * K))[c];
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) acc[t] += (float)w[j] * (float)v[j];
                }
            }
        }
        #pragma unroll
        for (int t = 0; t < NT; ++t) {
            float a = acc[t];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) a += __shfl_xor(a, off, 32);
            if (lane == 0 && t < nt) y[(long long)(t0 + t) * N + row] = (_Float16)a;
        }
    }
}

extern "C" __global__ void gemv_bi_g(const _Float16* __restrict__ W, const _Float16* __restrict__ x,
                                     _Float16* __restrict__ y, int N, int K, int n)
{
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= N) return;
    const half8* wr = (const half8*)(W + (long long)row * K);
    const int K8 = K >> 3;
    for (int t0 = 0; t0 < n; t0 += NT) {
        const int nt = min(NT, n - t0);
        float acc[NT];
        #pragma unroll
        for (int t = 0; t < NT; ++t) acc[t] = 0.0f;
        for (int c = lane; c < K8; c += 32) {
            const half8 w = wr[c];
            #pragma unroll
            for (int t = 0; t < NT; ++t) {
                if (t < nt) {
                    const half8 v = ((const half8*)(x + (long long)(t0 + t) * K))[c];
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) acc[t] += (float)w[j] * (float)v[j];
                }
            }
        }
        #pragma unroll
        for (int t = 0; t < NT; ++t) {
            float a = acc[t];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) a += __shfl_xor(a, off, 32);
            if (lane == 0 && t < nt) y[(long long)(t0 + t) * N + row] = (_Float16)a;
        }
    }
}
"""

XS_SIZES = (2048, 4096, 8192, 12288, 16384, 24576)
_k = {}
_registry = {}
_prev_linear = None
stats = {}         # (n, K, variant) -> calls seen (Python-side, counted at capture/eager time)


def _kern(variant, xs=0):
    key = (variant, xs)
    if key not in _k:
        src = SRC.replace("[XS]", f"[{xs if variant == 's' else 8}]")
        _k[key] = torch.cuda._compile_kernel(src, "gemv_bi_" + variant)
    return _k[key]


def gemv_bi1(x, W, force=None):
    """BI1 kernels ("s"/"g"). x: [..., n, K] fp16 with n <= NMAX rows; W: [N, K] fp16 contiguous. Returns [..., n, N]."""
    N, K = W.shape
    n = x.numel() // K
    assert 1 <= n <= NMAX and K % 8 == 0 and W.is_contiguous()
    x1 = x.reshape(n, K).contiguous()
    y = torch.empty(n, N, dtype=torch.float16, device=x.device)
    need = n * K
    variant = force or ("s" if need <= LDS_MAX else "g")
    xs = next((s for s in XS_SIZES if s >= need), 0) if variant == "s" else 0
    stats[(n, K, variant)] = stats.get((n, K, variant), 0) + 1
    _kern(variant, xs)(grid=((N + 7) // 8, 1, 1), block=(256, 1, 1), args=[W, x1, y, N, K, n])
    return y.view(*x.shape[:-1], N)


# ---- BI2 (default): same summation order, input rows converted to float once per K-tile in shared memory, each warp
# computes R output rows (R=2 reuses every input read). BI2a (in graph): 8 rows 6.79 ms vs BI1 9.24 (1.7B), 15.27 vs
# 29.90 (4B); PyTorch 6.12 / 13.89. Bit-identical to BI1 on every shape and row count.
import os  # noqa: E402
TILE = int(os.environ.get("BI_TILE", "512"))
RW = int(os.environ.get("BI_R", "2"))
SRC_T = r"""
#pragma clang fp contract(off)
typedef _Float16 half8 __attribute__((ext_vector_type(8)));
typedef float float4v __attribute__((ext_vector_type(4)));

extern "C" __global__ void gemv_bit(const _Float16* __restrict__ W, const _Float16* __restrict__ x,
                                    _Float16* __restrict__ y, int N, int K, int n)
{
    __shared__ __attribute__((aligned(16))) float xs[NTOK * TILE];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int row0 = (blockIdx.x * 8 + warp) * R;
    for (int t0 = 0; t0 < n; t0 += NTOK) {
        const int nt = min(NTOK, n - t0);
        float acc[R][NTOK];
        #pragma unroll
        for (int r = 0; r < R; ++r)
            #pragma unroll
            for (int t = 0; t < NTOK; ++t) acc[r][t] = 0.0f;
        for (int k0 = 0; k0 < K; k0 += TILE) {
            const int tl = min(TILE, K - k0);
            __syncthreads();
            for (int idx = threadIdx.x * 8; idx < nt * TILE; idx += 256 * 8) {
                const int t = idx / TILE, kk = idx - t * TILE;
                if (kk < tl) {
                    const half8 h = *(const half8*)(x + (long long)(t0 + t) * K + k0 + kk);
                    float4v a, b;
                    a[0] = (float)h[0]; a[1] = (float)h[1]; a[2] = (float)h[2]; a[3] = (float)h[3];
                    b[0] = (float)h[4]; b[1] = (float)h[5]; b[2] = (float)h[6]; b[3] = (float)h[7];
                    *(float4v*)(xs + idx) = a;
                    *(float4v*)(xs + idx + 4) = b;
                }
            }
            __syncthreads();
            const int cend = (k0 + tl) >> 3;
            for (int c = (k0 >> 3) + lane; c < cend; c += 32) {
                const int off = c * 8 - k0;
                half8 w[R];
                #pragma unroll
                for (int r = 0; r < R; ++r) {
                    if (row0 + r < N) w[r] = ((const half8*)(W + (long long)(row0 + r) * K))[c];
                    else w[r] = (half8)((_Float16)0.0f);
                }
                #pragma unroll
                for (int t = 0; t < NTOK; ++t) {
                    if (t < nt) {
                        const float4v a = *(const float4v*)(xs + t * TILE + off);
                        const float4v b = *(const float4v*)(xs + t * TILE + off + 4);
                        #pragma unroll
                        for (int r = 0; r < R; ++r) {
                            acc[r][t] += (float)w[r][0] * a[0];
                            acc[r][t] += (float)w[r][1] * a[1];
                            acc[r][t] += (float)w[r][2] * a[2];
                            acc[r][t] += (float)w[r][3] * a[3];
                            acc[r][t] += (float)w[r][4] * b[0];
                            acc[r][t] += (float)w[r][5] * b[1];
                            acc[r][t] += (float)w[r][6] * b[2];
                            acc[r][t] += (float)w[r][7] * b[3];
                        }
                    }
                }
            }
        }
        #pragma unroll
        for (int r = 0; r < R; ++r)
            #pragma unroll
            for (int t = 0; t < NTOK; ++t) {
                float a = acc[r][t];
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) a += __shfl_xor(a, off, 32);
                if (lane == 0 && t < nt && row0 + r < N) y[(long long)(t0 + t) * N + row0 + r] = (_Float16)a;
            }
    }
}
"""

_kt = {}


def kt(ntok, tile, r):
    key = (ntok, tile, r)
    if key not in _kt:
        src = SRC_T.replace("NTOK", str(ntok)).replace("TILE", str(tile)).replace("R]", f"{r}]")
        src = src.replace("* R;", f"* {r};").replace("r < R;", f"r < {r};")
        _kt[key] = torch.cuda._compile_kernel(src, "gemv_bit")
    return _kt[key]


def gemv_t(x, W, tile=None, r=None):
    tile, r = tile or TILE, r or RW
    N, K = W.shape
    n = x.numel() // K
    assert 1 <= n <= NMAX and K % 256 == 0 and W.is_contiguous()
    x1 = x.reshape(n, K).contiguous()
    y = torch.empty(n, N, dtype=torch.float16, device=x.device)
    stats[(n, K, f"T{tile}R{r}")] = stats.get((n, K, f"T{tile}R{r}"), 0) + 1
    kt(1 if n == 1 else 8, tile, r)(grid=((N + 8 * r - 1) // (8 * r), 1, 1), block=(256, 1, 1),
                                    args=[W, x1, y, N, K, n])
    return y.view(*x.shape[:-1], N)


def gemv_bi(x, W, force=None):
    """Default entry: BI2 tiled kernel; force="s"/"g" runs the BI1 kernels (same bits)."""
    if force in ("s", "g") or os.environ.get("BI_KERNEL") == "bi1":
        return gemv_bi1(x, W, force)
    return gemv_t(x, W)


def bi_linear(input, weight, bias=None):
    if bias is None and id(weight) in _registry and input.dtype == torch.float16:
        K = weight.shape[1]
        if input.numel() % K == 0 and 1 <= input.numel() // K <= NMAX:
            return gemv_bi(input, weight)
    return _prev_linear(input, weight, bias)


def install(weights):
    """Route F.linear for these fp16 weights (any call with 1..NMAX rows) through gemv_bi. Chains to the previous
    F.linear (e.g. kernel_int4.quant_linear) for everything else. Returns undo()."""
    global _prev_linear
    for w in weights:
        assert w.dtype == torch.float16 and w.is_contiguous() and w.shape[1] % 256 == 0
        _registry[id(w)] = True
    _prev_linear = F.linear
    F.linear = bi_linear

    def undo():
        F.linear = _prev_linear
        _registry.clear()
    return undo


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    import kernel_f12
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(11)
    # (N out, K in) of the target matmuls: 1.7B (hidden 2048, mlp 6144) and 4B (hidden 2560, mlp 9728), merged q/k/v
    shapes = {"1.7B qkv": (4096, 2048), "1.7B o": (2048, 2048), "1.7B gate+up": (12288, 2048),
              "1.7B down": (2048, 6144), "1.7B lm_head": (151936, 2048),
              "4B qkv": (6144, 2560), "4B o": (2560, 4096), "4B gate+up": (19456, 2560),
              "4B down": (2560, 9728), "4B lm_head": (151936, 2560)}
    out = {"shapes": {}}
    all_ident, worst_rel = True, 0.0
    for name, (N, K) in shapes.items():
        W = (torch.randn(N, K, device=dev, generator=gen) * 0.02).half()
        X = (torch.randn(NMAX, K, device=dev, generator=gen)).half()
        one = torch.cat([gemv_bi(X[t:t + 1], W) for t in range(NMAX)], 0)          # each row alone (BI2 kernel)
        one1 = torch.cat([gemv_bi1(X[t:t + 1], W) for t in range(NMAX)], 0)        # each row alone (BI1 kernel)
        ident_12 = torch.equal(one.view(torch.int16), one1.view(torch.int16))
        ident = bool(ident_12)
        if not ident_12:
            log(f"  NOT identical: {name} BI2 kernel vs BI1 kernel (1 row)")
        for n in (2, 3, 4, 6, 8, 9, 12, 16, 24, 32):
            for v in ("s", "g", None):
                if v == "s" and n * K > LDS_MAX:
                    continue
                got = gemv_bi(X[:n], W, force=v)
                if not torch.equal(got.view(torch.int16), one[:n].view(torch.int16)):
                    ident = False
                    log(f"  NOT identical: {name} n={n} variant {v}: "
                        f"{int((got.view(torch.int16) != one[:n].view(torch.int16)).sum())} values differ")
        g_one = torch.cat([gemv_bi(X[t:t + 1], W, force="g") for t in range(4)], 0)
        ident &= torch.equal(g_one.view(torch.int16), one[:4].view(torch.int16))
        ref_same = None
        if K <= 8192:                                       # kernel_f12's reference kernel keeps x in shared memory
            r = kernel_f12.gemv_ref(X[0:1].view(1, 1, K), W).reshape(1, N)
            ref_same = bool(torch.equal(r.view(torch.int16), one[0:1].view(torch.int16)))
            ident &= ref_same
        ref32 = X[:8].float() @ W.float().T
        rel = float((one[:8].float() - ref32).abs().max() / ref32.abs().max())
        pt = F.linear(X[:8], W)
        pt_rel = float((pt.float() - ref32).abs().max() / ref32.abs().max())
        worst_rel = max(worst_rel, rel)
        all_ident &= ident

        def tm(fn, reps=30):
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            ts = []
            for _ in range(reps):
                t = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append((time.perf_counter() - t) * 1e3)
            return statistics.median(ts)
        x1, x8 = X[:1].view(1, 1, K), X[:8].view(1, 8, K)
        t = {"pt1": tm(lambda: F.linear(x1, W)), "bi1": tm(lambda: gemv_bi(x1, W)),
             "pt8": tm(lambda: F.linear(x8, W)), "bi8": tm(lambda: gemv_bi(x8, W))}
        out["shapes"][name] = {"N": N, "K": K, "identical_all_n": ident, "same_as_gemv_f16_ref": ref_same,
                               "rel_err_vs_fp32": rel, "pytorch_rel_err_vs_fp32": pt_rel, "ms": t}
        log(f"{name:<14} {N:>6}x{K:<5} | n-row == 1-row bits: {ident} | == gemv_f16_ref: {ref_same} | rel err "
            f"{rel:.1e} (PyTorch {pt_rel:.1e}) | ms 1 row: PyTorch {t['pt1']:.3f} ours {t['bi1']:.3f} | "
            f"8 rows: PyTorch {t['pt8']:.3f} ours {t['bi8']:.3f}")
    for model in ("1.7B", "4B"):
        s = {k: sum(v["ms"][k] for n_, v in out["shapes"].items() if n_.startswith(model + " "))
             for k in ("pt1", "bi1", "pt8", "bi8")}
        log(f"{model} one layer's matmuls + lm_head (sum, ms): 1 row PyTorch {s['pt1']:.3f} ours {s['bi1']:.3f} | "
            f"8 rows PyTorch {s['pt8']:.3f} ours {s['bi8']:.3f}")
    ok = all_ident and worst_rel < 1e-2
    log(f"RESULT BI1-kernel: bit-identical across row counts on all shapes: {all_ident} | worst rel err {worst_rel:.1e}"
        f" -> {'CORRECT' if ok else 'WRONG'}")
    out["all_identical"] = all_ident
    (HERE / "results" / "BI1_kernel.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
