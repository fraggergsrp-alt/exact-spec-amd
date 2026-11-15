"""L2a (v2 kernel added after the first run): lossless 12-bit weights for the fp16 stack ("F12"), a replication of the NyoomFloat12 / ZipServ idea on AMD RDNA4.

Why it can be lossless (L1): these fp16 weights come from a bf16 checkpoint, so their 3 lowest mantissa bits are always
0, and their 5-bit exponent takes few values. F12 stores, per weight: 1 sign bit + the 7 meaningful mantissa bits
(one byte, "hi plane") + a 4-bit exponent code relative to a per-matrix window start e0 ("exponent plane", two codes per
byte) = exactly 12 bits. Weights that do not fit (exponent outside the 16-value window, or a nonzero low mantissa bit)
are stored in a per-row exception list (column, exact fp16 bits) and patched inside the kernel.

Lossless check, strongest form: the F12 kernel and a reference fp16 kernel use the SAME loop and the same order of
additions (both compiled with FMA contraction off), so their outputs must be BITWISE identical if decoding is exact.
Prior art: NyoomFloat12 (lossless fixed 12-bit format, NeurIPS 2026), ZipServ (ASPLOS 2026), DFloat11, ECF8.

Run this file: per-matrix coverage, bitwise round-trip on every weight of both models, bitwise kernel equality on all
real shapes, and speed over the 28 layers' real (distinct) matrices: PyTorch fp16, our fp16 reference kernel, F12.
"""
import os
import statistics
import sys
import threading
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent

SRC = r"""
#pragma clang fp contract(off)
typedef _Float16 half8 __attribute__((ext_vector_type(8)));
typedef unsigned char uchar8 __attribute__((ext_vector_type(8)));

extern "C" __global__ void gemv_f16_ref(const _Float16* __restrict__ W, const _Float16* __restrict__ x,
                                        _Float16* __restrict__ y, int N, int K)
{
    __shared__ __attribute__((aligned(16))) _Float16 xs[XMAX];
    for (int i = threadIdx.x; i < K; i += blockDim.x) xs[i] = x[i];
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= N) return;
    const half8* wr = (const half8*)(W + (long long)row * K);
    const half8* xv = (const half8*)xs;
    float acc = 0.0f;
    for (int c = lane; c < (K >> 3); c += 32) {
        const half8 w = wr[c];
        const half8 v = xv[c];
        #pragma unroll
        for (int j = 0; j < 8; ++j) acc += (float)w[j] * (float)v[j];
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor(acc, off, 32);
    if (lane == 0) y[row] = (_Float16)acc;
}

extern "C" __global__ void gemv_f12_v1(const unsigned char* __restrict__ hi, const unsigned char* __restrict__ ex,
                                    const int* __restrict__ eoff, const unsigned short* __restrict__ ecol,
                                    const unsigned short* __restrict__ ebits, const _Float16* __restrict__ x,
                                    _Float16* __restrict__ y, int N, int K, int e0)
{
    __shared__ __attribute__((aligned(16))) _Float16 xs[XMAX];
    for (int i = threadIdx.x; i < K; i += blockDim.x) xs[i] = x[i];
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= N) return;
    const uchar8* hr = (const uchar8*)(hi + (long long)row * K);
    const unsigned int* er = (const unsigned int*)(ex + (long long)row * (K >> 1));
    const int es = eoff[row], ee = eoff[row + 1];
    const half8* xv = (const half8*)xs;
    float acc = 0.0f;
    for (int c = lane; c < (K >> 3); c += 32) {
        const uchar8 h = hr[c];                    // 8 bytes: sign + 7 mantissa bits of 8 weights
        const unsigned int e = er[c];              // 4 bytes: eight 4-bit exponent codes
        const half8 v = xv[c];
        unsigned short bits[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const unsigned int b = h[j];
            const unsigned int code = (e >> (4 * j)) & 15u;
            bits[j] = (unsigned short)(((b & 0x80u) << 8) | (((unsigned int)e0 + code) << 10) | ((b & 0x7Fu) << 3));
        }
        for (int t = es; t < ee; ++t) {            // patch this row's rare exceptions that fall in this chunk
            const int col = ecol[t];
            if ((col >> 3) == c) bits[col & 7] = ebits[t];
        }
        #pragma unroll
        for (int j = 0; j < 8; ++j) acc += (float)__builtin_bit_cast(_Float16, bits[j]) * (float)v[j];
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor(acc, off, 32);
    if (lane == 0) y[row] = (_Float16)acc;
}

// v2: (1) each lane pre-filters ONLY its own exceptions once per row (an exception at column col belongs to chunk
// col/8, which lane (col/8) % 32 processes); (2) no runtime-indexed local arrays (they spill to slow scratch memory);
// (3) two weights decoded at once with 32-bit integer ops. Same summation order as gemv_f16_ref.
extern "C" __global__ void gemv_f12(const unsigned char* __restrict__ hi, const unsigned char* __restrict__ ex,
                                    const int* __restrict__ eoff, const unsigned short* __restrict__ ecol,
                                    const unsigned short* __restrict__ ebits, const _Float16* __restrict__ x,
                                    _Float16* __restrict__ y, int N, int K, int e0)
{
    __shared__ __attribute__((aligned(16))) _Float16 xs[XMAX];
    for (int i = threadIdx.x; i < K; i += blockDim.x) xs[i] = x[i];
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= N) return;
    const uint2* hr = (const uint2*)(hi + (long long)row * K);
    const unsigned int* er = (const unsigned int*)(ex + (long long)row * (K >> 1));
    const int es = eoff[row], ee = eoff[row + 1];
    // this lane's exceptions (expected ~0.2 per lane); keep up to 4 in registers, else fall back to scanning
    int n_mine = 0, c0 = -1, c1 = -1, c2 = -1, c3 = -1;
    unsigned int b0 = 0, b1 = 0, b2 = 0, b3 = 0;
    for (int t = es; t < ee; ++t) {
        const int col = ecol[t];
        if (((col >> 3) & 31) == lane) {
            if (n_mine == 0) { c0 = col; b0 = ebits[t]; } else if (n_mine == 1) { c1 = col; b1 = ebits[t]; }
            else if (n_mine == 2) { c2 = col; b2 = ebits[t]; } else if (n_mine == 3) { c3 = col; b3 = ebits[t]; }
            ++n_mine;
        }
    }
    const bool overflow = n_mine > 4;
    const unsigned int e0x2 = (unsigned int)e0 | ((unsigned int)e0 << 16);
    const half8* xv = (const half8*)xs;
    float acc = 0.0f;
    for (int c = lane; c < (K >> 3); c += 32) {
        const uint2 h = hr[c];                     // 8 bytes: sign + 7 mantissa bits of 8 weights
        const unsigned int e = er[c];              // eight 4-bit exponent codes
        const half8 v = xv[c];
        unsigned int w[4];                         // w[p] = fp16 bits of weights 2p (low half) and 2p+1 (high half)
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            const unsigned int hw = (p < 2 ? h.x : h.y) >> (16 * (p & 1));
            const unsigned int yb = (hw & 0xFFu) | ((hw & 0xFF00u) << 8);
            const unsigned int cb = (e >> (8 * p)) & 0xFFu;
            const unsigned int cc = (cb & 15u) | ((cb & 0xF0u) << 12);
            w[p] = ((yb & 0x00800080u) << 8) | ((yb & 0x007F007Fu) << 3) | ((cc + e0x2) << 10);
        }
        if (n_mine) {                              // rare: patch exceptions in this chunk (compile-time indices only)
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int col = q == 0 ? c0 : q == 1 ? c1 : q == 2 ? c2 : c3;
                const unsigned int bb = q == 0 ? b0 : q == 1 ? b1 : q == 2 ? b2 : b3;
                if (q < n_mine && (col >> 3) == c) {
                    const int j = col & 7;
                    #pragma unroll
                    for (int p = 0; p < 4; ++p) {
                        if (j == 2 * p) w[p] = (w[p] & 0xFFFF0000u) | bb;
                        if (j == 2 * p + 1) w[p] = (w[p] & 0x0000FFFFu) | (bb << 16);
                    }
                }
            }
            if (overflow) {
                for (int t = es; t < ee; ++t) {
                    const int col = ecol[t];
                    if ((col >> 3) == c) {
                        const int j = col & 7;
                        const unsigned int bb = ebits[t];
                        #pragma unroll
                        for (int p = 0; p < 4; ++p) {
                            if (j == 2 * p) w[p] = (w[p] & 0xFFFF0000u) | bb;
                            if (j == 2 * p + 1) w[p] = (w[p] & 0x0000FFFFu) | (bb << 16);
                        }
                    }
                }
            }
        }
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            acc += (float)__builtin_bit_cast(_Float16, (unsigned short)(w[p] & 0xFFFFu)) * (float)v[2 * p];
            acc += (float)__builtin_bit_cast(_Float16, (unsigned short)(w[p] >> 16)) * (float)v[2 * p + 1];
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor(acc, off, 32);
    if (lane == 0) y[row] = (_Float16)acc;
}

// v3 = v2, but the row's exception list is loaded COOPERATIVELY: lane i reads entries i, i+32, ... and drops each
// into its owner lane's slot in shared memory (v2: every lane read the whole list serially). (Original v2 note:)
// v2: (1) each lane pre-filters ONLY its own exceptions once per row (an exception at column col belongs to chunk
// col/8, which lane (col/8) % 32 processes); (2) no runtime-indexed local arrays (they spill to slow scratch memory);
// (3) two weights decoded at once with 32-bit integer ops. Same summation order as gemv_f16_ref.
extern "C" __global__ void gemv_f12_v3(const unsigned char* __restrict__ hi, const unsigned char* __restrict__ ex,
                                    const int* __restrict__ eoff, const unsigned short* __restrict__ ecol,
                                    const unsigned short* __restrict__ ebits, const _Float16* __restrict__ x,
                                    _Float16* __restrict__ y, int N, int K, int e0)
{
    __shared__ __attribute__((aligned(16))) _Float16 xs[XMAX];
    for (int i = threadIdx.x; i < K; i += blockDim.x) xs[i] = x[i];
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int wave = threadIdx.x >> 5;
    const bool valid = row < N;
    const int es = valid ? eoff[row] : 0, ee = valid ? eoff[row + 1] : 0;
    __shared__ int cnt[8][32];
    __shared__ unsigned short scol[8][32][4], sbit[8][32][4];
    cnt[wave][lane] = 0;
    __syncthreads();
    for (int t = es + lane; t < ee; t += 32) {            // cooperative, coalesced load of the row's exception list
        const int col = ecol[t];
        const int o = (col >> 3) & 31;
        const int k = atomicAdd(&cnt[wave][o], 1);
        if (k < 4) { scol[wave][o][k] = (unsigned short)col; sbit[wave][o][k] = ebits[t]; }
    }
    __syncthreads();
    if (!valid) return;
    const uint2* hr = (const uint2*)(hi + (long long)row * K);
    const unsigned int* er = (const unsigned int*)(ex + (long long)row * (K >> 1));
    const int n_mine = cnt[wave][lane];
    const int c0 = n_mine > 0 ? (int)scol[wave][lane][0] : -1, c1 = n_mine > 1 ? (int)scol[wave][lane][1] : -1;
    const int c2 = n_mine > 2 ? (int)scol[wave][lane][2] : -1, c3 = n_mine > 3 ? (int)scol[wave][lane][3] : -1;
    const unsigned int b0 = n_mine > 0 ? sbit[wave][lane][0] : 0u, b1 = n_mine > 1 ? sbit[wave][lane][1] : 0u;
    const unsigned int b2 = n_mine > 2 ? sbit[wave][lane][2] : 0u, b3 = n_mine > 3 ? sbit[wave][lane][3] : 0u;
    const bool overflow = n_mine > 4;
    const unsigned int e0x2 = (unsigned int)e0 | ((unsigned int)e0 << 16);
    const half8* xv = (const half8*)xs;
    float acc = 0.0f;
    for (int c = lane; c < (K >> 3); c += 32) {
        const uint2 h = hr[c];                     // 8 bytes: sign + 7 mantissa bits of 8 weights
        const unsigned int e = er[c];              // eight 4-bit exponent codes
        const half8 v = xv[c];
        unsigned int w[4];                         // w[p] = fp16 bits of weights 2p (low half) and 2p+1 (high half)
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            const unsigned int hw = (p < 2 ? h.x : h.y) >> (16 * (p & 1));
            const unsigned int yb = (hw & 0xFFu) | ((hw & 0xFF00u) << 8);
            const unsigned int cb = (e >> (8 * p)) & 0xFFu;
            const unsigned int cc = (cb & 15u) | ((cb & 0xF0u) << 12);
            w[p] = ((yb & 0x00800080u) << 8) | ((yb & 0x007F007Fu) << 3) | ((cc + e0x2) << 10);
        }
        if (n_mine) {                              // rare: patch exceptions in this chunk (compile-time indices only)
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int col = q == 0 ? c0 : q == 1 ? c1 : q == 2 ? c2 : c3;
                const unsigned int bb = q == 0 ? b0 : q == 1 ? b1 : q == 2 ? b2 : b3;
                if (q < n_mine && (col >> 3) == c) {
                    const int j = col & 7;
                    #pragma unroll
                    for (int p = 0; p < 4; ++p) {
                        if (j == 2 * p) w[p] = (w[p] & 0xFFFF0000u) | bb;
                        if (j == 2 * p + 1) w[p] = (w[p] & 0x0000FFFFu) | (bb << 16);
                    }
                }
            }
            if (overflow) {
                for (int t = es; t < ee; ++t) {
                    const int col = ecol[t];
                    if ((col >> 3) == c) {
                        const int j = col & 7;
                        const unsigned int bb = ebits[t];
                        #pragma unroll
                        for (int p = 0; p < 4; ++p) {
                            if (j == 2 * p) w[p] = (w[p] & 0xFFFF0000u) | bb;
                            if (j == 2 * p + 1) w[p] = (w[p] & 0x0000FFFFu) | (bb << 16);
                        }
                    }
                }
            }
        }
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            acc += (float)__builtin_bit_cast(_Float16, (unsigned short)(w[p] & 0xFFFFu)) * (float)v[2 * p];
            acc += (float)__builtin_bit_cast(_Float16, (unsigned short)(w[p] >> 16)) * (float)v[2 * p + 1];
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor(acc, off, 32);
    if (lane == 0) y[row] = (_Float16)acc;
}
"""

_k = {}


def kernels(K):
    key = 4096 if K <= 4096 else 8192
    if key not in _k:
        src = SRC.replace("XMAX", str(key))
        _k[key] = (torch.cuda._compile_kernel(src, "gemv_f16_ref"), torch.cuda._compile_kernel(src, "gemv_f12"),
                   torch.cuda._compile_kernel(src, "gemv_f12_v1"), torch.cuda._compile_kernel(src, "gemv_f12_v3"))
    return _k[key]


def encode(W, rows_per_chunk=8192):
    """fp16 [N, K] -> dict with planes + exceptions. Lossless by construction (checked by decode()). Works in row
    chunks so temporary memory stays small even for the 151936-row lm_head."""
    N, K = W.shape
    assert W.dtype == torch.float16 and K % 8 == 0
    dev = W.device

    def fields(Wc):
        u = Wc.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
        return u, u >> 15, (u >> 10) & 31, u & 1023

    hist = torch.zeros(32, dtype=torch.int64, device=dev)          # exponent histogram of weights with zero low bits
    for r in range(0, N, rows_per_chunk):
        _, _, E, m = fields(W[r:r + rows_per_chunk])
        hist += torch.bincount(E[(m & 7) == 0].reshape(-1).long(), minlength=32)
    cum = torch.cat([torch.zeros(1, dtype=torch.int64, device=dev), torch.cumsum(hist, 0)])
    e0 = int(torch.argmax(torch.stack([cum[e + 16] - cum[e] for e in range(17)])))   # best 16-value window
    hi = torch.empty(N, K, dtype=torch.uint8, device=dev)
    ex = torch.empty(N, K // 2, dtype=torch.uint8, device=dev)
    rows_l, cols_l, bits_l = [], [], []
    for r in range(0, N, rows_per_chunk):
        u, s, E, m = fields(W[r:r + rows_per_chunk])
        fit = (E >= e0) & (E <= e0 + 15) & ((m & 7) == 0)
        code = torch.where(fit, E - e0, torch.zeros_like(E))
        h = torch.where(fit, (s << 7) | (m >> 3), torch.zeros_like(E))
        hi[r:r + rows_per_chunk] = h.to(torch.uint8)
        ex[r:r + rows_per_chunk] = (code[:, 0::2] | (code[:, 1::2] << 4)).to(torch.uint8)
        rr, cc = torch.nonzero(~fit, as_tuple=True)
        rows_l.append(rr + r); cols_l.append(cc); bits_l.append(u[rr, cc])
    rows, cols, ub = torch.cat(rows_l), torch.cat(cols_l), torch.cat(bits_l)
    eoff = torch.zeros(N + 1, dtype=torch.int32, device=dev)
    if rows.numel():
        eoff[1:] = torch.cumsum(torch.bincount(rows, minlength=N), 0).to(torch.int32)
    ecol = cols.to(torch.int32).to(torch.int16).contiguous()       # K <= 32767 here
    ebits = ub.to(torch.int16).contiguous()                        # exact fp16 bit patterns
    n_exc = int(rows.numel())
    if n_exc == 0:                                                 # kernel needs valid pointers
        ecol = torch.zeros(1, dtype=torch.int16, device=dev)
        ebits = torch.zeros(1, dtype=torch.int16, device=dev)
    per_row = eoff[1:] - eoff[:-1]
    return {"hi": hi, "ex": ex, "eoff": eoff, "ecol": ecol, "ebits": ebits, "e0": e0, "N": N, "K": K,
            "n_exc": n_exc, "max_exc_row": int(per_row.max()) if N else 0}


def decode(f):
    """Pure-PyTorch decoder (reference for the kernel)."""
    N, K, e0 = f["N"], f["K"], f["e0"]
    h = f["hi"].to(torch.int32)
    code = torch.stack([f["ex"].to(torch.int32) & 15, f["ex"].to(torch.int32) >> 4], -1).view(N, K)
    bits = ((h & 0x80) << 8) | ((e0 + code) << 10) | ((h & 0x7F) << 3)
    if f["n_exc"]:
        rows = torch.repeat_interleave(torch.arange(N, device=h.device), (f["eoff"][1:] - f["eoff"][:-1]).long())
        bits[rows, f["ecol"].to(torch.int32).long() & 0xFFFF] = f["ebits"].to(torch.int32) & 0xFFFF
    return bits.to(torch.int16).view(torch.float16)


def decode_rows(f, r0, n):
    """Decode rows r0..r0+n only (keeps memory small for lm_head)."""
    sub = {"N": min(n, f["N"] - r0), "K": f["K"], "e0": f["e0"], "hi": f["hi"][r0:r0 + n], "ex": f["ex"][r0:r0 + n]}
    lo, hi_ = int(f["eoff"][r0]), int(f["eoff"][r0 + sub["N"]])
    sub["eoff"] = f["eoff"][r0:r0 + sub["N"] + 1] - lo
    sub["n_exc"] = hi_ - lo
    sub["ecol"], sub["ebits"] = f["ecol"][lo:hi_], f["ebits"][lo:hi_]
    return decode(sub)


def gemv_f12(x, f):
    K = f["K"]
    x1 = x.reshape(K).contiguous()
    y = torch.empty(f["N"], dtype=torch.float16, device=x.device)
    kernels(K)[1](grid=((f["N"] + 7) // 8, 1, 1), block=(256, 1, 1),
                  args=[f["hi"], f["ex"], f["eoff"], f["ecol"], f["ebits"], x1, y, f["N"], K, f["e0"]])
    return y.view(*x.shape[:-1], f["N"])


def gemv_f12_v1(x, f):
    K = f["K"]
    x1 = x.reshape(K).contiguous()
    y = torch.empty(f["N"], dtype=torch.float16, device=x.device)
    kernels(K)[2](grid=((f["N"] + 7) // 8, 1, 1), block=(256, 1, 1),
                  args=[f["hi"], f["ex"], f["eoff"], f["ecol"], f["ebits"], x1, y, f["N"], K, f["e0"]])
    return y.view(*x.shape[:-1], f["N"])


def gemv_f12_v3(x, f):
    K = f["K"]
    x1 = x.reshape(K).contiguous()
    y = torch.empty(f["N"], dtype=torch.float16, device=x.device)
    kernels(K)[3](grid=((f["N"] + 7) // 8, 1, 1), block=(256, 1, 1),
                  args=[f["hi"], f["ex"], f["eoff"], f["ecol"], f["ebits"], x1, y, f["N"], K, f["e0"]])
    return y.view(*x.shape[:-1], f["N"])


def gemv_ref(x, W):
    N, K = W.shape
    x1 = x.reshape(K).contiguous()
    y = torch.empty(N, dtype=torch.float16, device=x.device)
    kernels(K)[0](grid=((N + 7) // 8, 1, 1), block=(256, 1, 1), args=[W, x1, y, N, K])
    return y.view(*x.shape[:-1], N)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def tg(fn, reps=15):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn(); fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize(); t = time.perf_counter(); g.replay(); torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1000)
    del g
    return statistics.median(ts)


def main():
    wd = threading.Timer(600, lambda: (log("RESULT L2a: TIMEOUT"), os._exit(3)))
    wd.daemon = True
    wd.start()
    from transformers import AutoModelForCausalLM
    dev = torch.device("cuda")
    try:
        kernels(4096); kernels(8192)
    except Exception as ex:
        log(f"RESULT L2a: COMPILE FAILED -> {type(ex).__name__}: {str(ex)[:1500]}")
        sys.stdout.flush(); os._exit(1)
    log("compiled OK")
    gen = torch.Generator(device=dev).manual_seed(12)
    all_rt, all_k = True, True
    summary = []
    for mid in ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B"):
        name = mid.split("/")[-1]
        model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16).to(dev).eval()
        types = {"qkv": [], "o": [], "gate_up": [], "down": []}
        with torch.no_grad():
            for layer in model.model.layers:
                a, m = layer.self_attn, layer.mlp
                types["qkv"].append(torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight], 0).contiguous())
                types["o"].append(a.o_proj.weight.contiguous())
                types["gate_up"].append(torch.cat([m.gate_proj.weight, m.up_proj.weight], 0).contiguous())
                types["down"].append(m.down_proj.weight.contiguous())
            types["lm_head"] = [model.lm_head.weight.contiguous()]
            tot_ms = {"torch": 0.0, "ref": 0.0, "f12": 0.0, "v3": 0.0}
            for tname, Ws in types.items():
                enc = [encode(W) for W in Ws]
                rt = all(torch.equal(decode(f).view(torch.int16), W.view(torch.int16)) for f, W in zip(enc, Ws)
                         if W.shape[0] < 100000) and all(
                    torch.equal(decode_rows(f, r, 8192).view(torch.int16), W[r:r + 8192].view(torch.int16))
                    for f, W in zip(enc, Ws) if W.shape[0] >= 100000 for r in range(0, W.shape[0], 8192))
                all_rt &= rt
                n = sum(W.numel() for W in Ws)
                n_exc = sum(f["n_exc"] for f in enc)
                bits = (sum(f["hi"].numel() + f["ex"].numel() + 4 * f["eoff"].numel() + 4 * f["n_exc"] for f in enc)
                        * 8 / n)
                K = Ws[0].shape[1]
                x = torch.randn(1, 1, K, device=dev, generator=gen).half()
                kok = all(torch.equal(gemv_f12(x, f).view(torch.int16), gemv_ref(x, W).view(torch.int16)) and
                          torch.equal(gemv_f12_v1(x, f).view(torch.int16), gemv_ref(x, W).view(torch.int16)) and
                          torch.equal(gemv_f12_v3(x, f).view(torch.int16), gemv_ref(x, W).view(torch.int16))
                          for f, W in zip(enc, Ws))
                all_k &= kok
                t_torch = tg(lambda: [F.linear(x, W) for W in Ws])
                t_ref = tg(lambda: [gemv_ref(x, W) for W in Ws])
                t_f12 = tg(lambda: [gemv_f12(x, f) for f in enc])
                t_v1 = tg(lambda: [gemv_f12_v3(x, f) for f in enc])        # (variable kept: now times v3)
                for k_, v_ in (("torch", t_torch), ("ref", t_ref), ("f12", t_f12), ("v3", t_v1)):
                    tot_ms[k_] += v_
                e0s = sorted({f["e0"] for f in enc})
                log(f"{name} {tname:<8} x{len(Ws):2d} | exceptions {n_exc} of {n / 1e6:.1f}M ({100 * n_exc / n:.4f}%), max/row "
                    f"{max(f['max_exc_row'] for f in enc)}, e0 {e0s} | {bits:.3f} bits/weight | round-trip bitwise "
                    f"{rt} | kernel == fp16 ref bitwise {kok} | per token: torch {t_torch:.3f} ms, fp16 ref {t_ref:.3f} ms,"
                    f" F12 v2 {t_f12:.3f} ms ({t_ref / t_f12:.2f}x), F12 v3 {t_v1:.3f} ms ({t_ref / t_v1:.2f}x vs ref)")
                del enc
                torch.cuda.empty_cache()
            log(f"{name} ALL matrices per token: torch fp16 {tot_ms['torch']:.3f} ms | our fp16 ref {tot_ms['ref']:.3f} ms | "
                f"F12 v2 {tot_ms['f12']:.3f} ms ({tot_ms['ref'] / tot_ms['f12']:.2f}x vs ref) | F12 v3 {tot_ms['v3']:.3f} ms "
                f"({tot_ms['ref'] / tot_ms['v3']:.2f}x vs ref, {tot_ms['torch'] / tot_ms['v3']:.2f}x vs torch)")
            summary.append((name, tot_ms))
        del model, types
        torch.cuda.empty_cache()
    log(f"RESULT L2a: round-trip bitwise = {all_rt} | kernel bitwise == fp16 reference = {all_k} | " + " | ".join(
        f"{n}: torch {t['torch']:.2f} / ref {t['ref']:.2f} / F12 v2 {t['f12']:.2f} / v3 {t['v3']:.2f} ms" for n, t in summary))
    wd.cancel()
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
