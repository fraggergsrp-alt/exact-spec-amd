"""CAP1b: B12 - a lossless 12-bit storage format for bf16 weights, and a batch-invariant matrix-vector kernel for it.
Pre-registered: docs/PREREG_CAP1.md (step 2). Research code; the frozen engine is not touched.

FORMAT (per weight matrix W [N, K], bf16 = 1 sign + 8 exponent + 7 mantissa bits):
  hi  [N, K]   uint8 : sign << 7 | mantissa (7 bits)                         -> 8 bits
  ex  [N, K/2] uint8 : 4-bit exponent code per weight, two per byte           -> 4 bits
                       code 0..14 = exponent e0[row] + code; code 15 = "not in the dense part"
  e0  [N]      uint8 : per-row exponent window start (chosen to cover the most weights of that row)
  exceptions (per row, CSR): eoff [N+1] int32, ecol uint16, ebits uint16 (the exact bf16 bits)
                       = every nonzero weight whose exponent is outside the row's 15-value window (incl. -0.0,
                         subnormals, inf/nan). +0.0 weights are dropped (they add nothing to a sum).
  Lossless: decode() reproduces every bf16 bit pattern exactly (+0.0 stays +0.0). bf16 -> fp32 is exact, so the kernel
  multiplies with the ORIGINAL checkpoint weights (unlike the fp16 conversion, which changes 0.04% of Qwen3-8B weights).

KERNEL (same structure as kernel_bi's BI2 kernel, so the result for a token row does not depend on how many rows
are in the call): dense part - lane-strided 8-weight chunks, fp32 accumulation, FMA contraction off, input rows
converted to float once per K-tile in shared memory, R output rows per warp; exception part - per row, lane-strided
over that row's exception list, fp32; each part reduced with the same 5-step shuffle; y = dense + exceptions.
Every product is (exact weight) x (input), only the summation order is ours and fixed.

Prior art: lossless exponent coding of bf16 weights (DFloat11, ZipServ TCA-TBE, ZipNN, SplitZip for KV), our F12
replication of NyoomFloat12. Not claimed as new. Per-row window + dense/sparse split is an engineering choice.

Run: python kernel_b12.py  -> whole-model B12 statistics, lossless round trip, batch invariance, accuracy and in-graph
speed on real Qwen3-8B matrices. Writes results/CAP1b_kernel.json.
"""
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]

SRC = r"""
#pragma clang fp contract(off)
typedef float float4v __attribute__((ext_vector_type(4)));

__device__ __forceinline__ float bf(unsigned int b16) { return __builtin_bit_cast(float, b16 << 16); }

extern "C" __global__ void gemv_b12(const unsigned char* __restrict__ hi, const unsigned char* __restrict__ ex,
                                    const unsigned char* __restrict__ e0r, const int* __restrict__ eoff,
                                    const unsigned short* __restrict__ ecol, const unsigned short* __restrict__ ebits,
                                    const _Float16* __restrict__ x, _Float16* __restrict__ y, int N, int K, int n)
{
    __shared__ __attribute__((aligned(16))) float xs[NTOK * TILE];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int row0 = (blockIdx.x * 8 + warp) * RR;
    unsigned int e0v[RR];
    #pragma unroll
    for (int r = 0; r < RR; ++r) e0v[r] = (row0 + r < N) ? (unsigned int)e0r[row0 + r] : 0u;
    for (int t0 = 0; t0 < n; t0 += NTOK) {
        const int nt = min(NTOK, n - t0);
        float acc[RR][NTOK];
        #pragma unroll
        for (int r = 0; r < RR; ++r)
            #pragma unroll
            for (int t = 0; t < NTOK; ++t) acc[r][t] = 0.0f;
        for (int k0 = 0; k0 < K; k0 += TILE) {
            const int tl = min(TILE, K - k0);
            __syncthreads();
            for (int idx = threadIdx.x * 8; idx < nt * TILE; idx += 256 * 8) {
                const int t = idx / TILE, kk = idx - t * TILE;
                if (kk < tl) {
                    const _Float16* src = x + (long long)(t0 + t) * K + k0 + kk;
                    float4v a, b;
                    a[0] = (float)src[0]; a[1] = (float)src[1]; a[2] = (float)src[2]; a[3] = (float)src[3];
                    b[0] = (float)src[4]; b[1] = (float)src[5]; b[2] = (float)src[6]; b[3] = (float)src[7];
                    *(float4v*)(xs + idx) = a;
                    *(float4v*)(xs + idx + 4) = b;
                }
            }
            __syncthreads();
            const int cend = (k0 + tl) >> 3;
            for (int c = (k0 >> 3) + lane; c < cend; c += 32) {
                const int off = c * 8 - k0;
                float w[RR][8];
                #pragma unroll
                for (int r = 0; r < RR; ++r) {
                    uint2 h = make_uint2(0u, 0u);
                    unsigned int e = 0xFFFFFFFFu;
                    if (row0 + r < N) {
                        h = ((const uint2*)(hi + (long long)(row0 + r) * K))[c];
                        e = ((const unsigned int*)(ex + (long long)(row0 + r) * (K >> 1)))[c];
                    }
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const unsigned int b = ((j < 4 ? h.x : h.y) >> (8 * (j & 3))) & 0xFFu;
                        const unsigned int code = (e >> (4 * j)) & 15u;
                        const unsigned int bits = (code == 15u) ? 0u : (((b & 0x80u) << 8) | ((e0v[r] + code) << 7) | (b & 0x7Fu));
                        w[r][j] = bf(bits);
                    }
                }
                #pragma unroll
                for (int t = 0; t < NTOK; ++t) {
                    if (t < nt) {
                        const float4v a = *(const float4v*)(xs + t * TILE + off);
                        const float4v b = *(const float4v*)(xs + t * TILE + off + 4);
                        #pragma unroll
                        for (int r = 0; r < RR; ++r) {
                            acc[r][t] += w[r][0] * a[0];
                            acc[r][t] += w[r][1] * a[1];
                            acc[r][t] += w[r][2] * a[2];
                            acc[r][t] += w[r][3] * a[3];
                            acc[r][t] += w[r][4] * b[0];
                            acc[r][t] += w[r][5] * b[1];
                            acc[r][t] += w[r][6] * b[2];
                            acc[r][t] += w[r][7] * b[3];
                        }
                    }
                }
            }
        }
        #pragma unroll
        for (int r = 0; r < RR; ++r) {
            if (row0 + r >= N) continue;
            const int s = eoff[row0 + r], e = eoff[row0 + r + 1];
            float sp[NTOK];
            #pragma unroll
            for (int t = 0; t < NTOK; ++t) sp[t] = 0.0f;
            for (int q = s + lane; q < e; q += 32) {
                const int col = ecol[q];
                const float wv = bf(ebits[q]);
                #pragma unroll
                for (int t = 0; t < NTOK; ++t)
                    if (t < nt) sp[t] += wv * (float)x[(long long)(t0 + t) * K + col];
            }
            #pragma unroll
            for (int t = 0; t < NTOK; ++t) {
                float a = acc[r][t], b = sp[t];
                #pragma unroll
                for (int off2 = 16; off2 > 0; off2 >>= 1) a += __shfl_xor(a, off2, 32);
                #pragma unroll
                for (int off2 = 16; off2 > 0; off2 >>= 1) b += __shfl_xor(b, off2, 32);
                if (lane == 0 && t < nt) y[(long long)(t0 + t) * N + row0 + r] = (_Float16)(a + b);
            }
        }
    }
}
"""

TILE = int(os.environ.get("B12_TILE", "512"))
RR = int(os.environ.get("B12_R", "2"))
NMAX = 32
_k = {}


def kernel(ntok, tile=None, r=None):
    tile, r = tile or TILE, r or RR
    key = (ntok, tile, r)
    if key not in _k:
        src = SRC.replace("NTOK", str(ntok)).replace("TILE", str(tile)).replace("RR", str(r))
        _k[key] = torch.cuda._compile_kernel(src, "gemv_b12")
    return _k[key]


def encode(W):
    """bf16 [N, K] (on GPU) -> dict. Per-row 15-value exponent window; exceptions in CSR form."""
    assert W.dtype == torch.bfloat16 and W.dim() == 2 and W.shape[1] % 256 == 0
    N, K = W.shape
    dev = W.device
    out = {"N": N, "K": K}
    hi = torch.empty(N, K, dtype=torch.uint8, device=dev)
    ex = torch.empty(N, K // 2, dtype=torch.uint8, device=dev)
    e0 = torch.empty(N, dtype=torch.uint8, device=dev)
    rows_l, cols_l, bits_l = [], [], []
    CH = max(1, (1 << 25) // K)                                    # rows per chunk (~32M weights)
    for r0 in range(0, N, CH):
        Wc = W[r0:r0 + CH]
        n = Wc.shape[0]
        u = Wc.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
        s, e, m = u >> 15, (u >> 7) & 0xFF, u & 0x7F
        posz = u == 0                                               # +0.0: dropped
        cand = (~posz) & (e > 0) & (e < 255)
        idx = (torch.arange(n, device=dev)[:, None] * 256 + e)[cand]
        cnt = torch.bincount(idx, minlength=n * 256).view(n, 256)
        cum = torch.cat([torch.zeros(n, 1, dtype=cnt.dtype, device=dev), torch.cumsum(cnt, 1)], 1)
        win = cum[:, 15:15 + 241] - cum[:, 0:241]                   # window [e0, e0+14] for e0 = 0..240
        best = torch.argmax(win, 1)
        best = torch.clamp(best, min=1)
        code = e - best[:, None]
        inwin = cand & (code >= 0) & (code <= 14)
        cv = torch.where(inwin, code, torch.full_like(code, 15))
        hi[r0:r0 + n] = torch.where(inwin, (s << 7) | m, torch.zeros_like(u)).to(torch.uint8)
        ex[r0:r0 + n] = (cv[:, 0::2] | (cv[:, 1::2] << 4)).to(torch.uint8)
        e0[r0:r0 + n] = best.to(torch.uint8)
        exc = (~posz) & (~inwin)
        rr, cc = torch.nonzero(exc, as_tuple=True)
        rows_l.append(rr + r0); cols_l.append(cc); bits_l.append(u[rr, cc])
    rows, cols, bits = torch.cat(rows_l), torch.cat(cols_l), torch.cat(bits_l)
    eoff = torch.zeros(N + 1, dtype=torch.int32, device=dev)
    if rows.numel():
        eoff[1:] = torch.cumsum(torch.bincount(rows, minlength=N), 0).to(torch.int32)
    order = torch.argsort(rows * K + cols)                          # CSR order: by row, then column
    ecol = cols[order].to(torch.int32).to(torch.int16) if rows.numel() else torch.zeros(1, dtype=torch.int16, device=dev)
    ebits = bits[order].to(torch.int16) if rows.numel() else torch.zeros(1, dtype=torch.int16, device=dev)
    per_row = eoff[1:] - eoff[:-1]
    out.update(hi=hi, ex=ex, e0=e0, eoff=eoff, ecol=ecol.contiguous(), ebits=ebits.contiguous(), n_exc=int(rows.numel()),
               max_exc_row=int(per_row.max()) if N else 0)
    out["bytes"] = N * K + N * K // 2 + N + (N + 1) * 4 + out["n_exc"] * 4
    return out


def decode(f):
    """B12 dict -> bf16 [N, K] (pure PyTorch reference, for the lossless check)."""
    N, K = f["N"], f["K"]
    h = f["hi"].to(torch.int32)
    exb = f["ex"].to(torch.int32)
    code = torch.stack([exb & 15, exb >> 4], -1).view(N, K)
    bits = ((h & 0x80) << 8) | ((f["e0"].to(torch.int32)[:, None] + code) << 7) | (h & 0x7F)
    bits = torch.where(code == 15, torch.zeros_like(bits), bits)
    if f["n_exc"]:
        rows = torch.repeat_interleave(torch.arange(N, device=h.device), (f["eoff"][1:] - f["eoff"][:-1]).long())
        bits[rows, f["ecol"].to(torch.int32) & 0xFFFF] = f["ebits"].to(torch.int32) & 0xFFFF
    return bits.to(torch.int16).view(torch.bfloat16)


def decode_rows(f, r0, n):
    """Decode rows r0..r0+n only (keeps memory small for the 151936-row lm_head)."""
    n = min(n, f["N"] - r0)
    lo, hi_ = int(f["eoff"][r0]), int(f["eoff"][r0 + n])
    sub = {"N": n, "K": f["K"], "hi": f["hi"][r0:r0 + n], "ex": f["ex"][r0:r0 + n], "e0": f["e0"][r0:r0 + n],
           "eoff": f["eoff"][r0:r0 + n + 1] - lo, "n_exc": hi_ - lo, "ecol": f["ecol"][lo:hi_], "ebits": f["ebits"][lo:hi_]}
    return decode(sub)


def lossless(f, W, rows=4096):
    return all(torch.equal(decode_rows(f, r0, rows).view(torch.int16), W[r0:r0 + rows].view(torch.int16))
               for r0 in range(0, f["N"], rows))


def gemv(x, f, tile=None, r=None):
    """x: [..., n, K] fp16 (n <= NMAX). Returns [..., n, N] fp16."""
    tile, r = tile or TILE, r or RR
    N, K = f["N"], f["K"]
    n = x.numel() // K
    assert 1 <= n <= NMAX
    x1 = x.reshape(n, K).contiguous()
    y = torch.empty(n, N, dtype=torch.float16, device=x.device)
    kernel(1 if n == 1 else 8, tile, r)(grid=((N + 8 * r - 1) // (8 * r), 1, 1), block=(256, 1, 1),
                                         args=[f["hi"], f["ex"], f["e0"], f["eoff"], f["ecol"], f["ebits"], x1, y, N, K, n])
    return y.view(*x.shape[:-1], N)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    torch.set_grad_enabled(False)
    dev = torch.device("cuda")
    path = Path(snapshot_download("Qwen/Qwen3-8B", allow_patterns=["*.json", "*.safetensors"]))
    files = sorted(path.glob("*.safetensors"))
    index = {}
    for fpath in files:
        with safe_open(str(fpath), framework="pt") as sf:
            for name in sf.keys():
                index[name] = fpath

    def get(name):
        with safe_open(str(index[name]), framework="pt") as sf:
            return sf.get_tensor(name)

    out = {"tile": TILE, "r": RR}
    # ---- 1. whole-model statistics + lossless round trip on every matrix
    tot_w = tot_b = tot_exc = 0
    worst = (0, "")
    lossless_all = True
    for name in sorted(index):
        if not name.endswith(".weight") or "norm" in name or "embed_tokens" in name:
            continue
        W = get(name).to(dev)
        if W.dim() != 2:
            continue
        f = encode(W)
        same = lossless(f, W)
        lossless_all &= same
        tot_w += W.numel(); tot_b += f["bytes"]; tot_exc += f["n_exc"]
        if f["max_exc_row"] > worst[0]:
            worst = (f["max_exc_row"], name)
        if not same:
            log(f"  NOT LOSSLESS: {name}")
        del W, f
        torch.cuda.empty_cache()
    out.update(weights=tot_w, b12_bytes=tot_b, exceptions=tot_exc, worst_row=worst, lossless=lossless_all)
    log(f"CAP1b B12 whole model (all matmul weights, no embeddings): {tot_w / 1e9:.3f} B weights | {tot_b / 1e9:.2f} GB "
        f"({8 * tot_b / tot_w:.3f} bits/weight) | exceptions {tot_exc} ({100 * tot_exc / tot_w:.4f}%) | worst row {worst[0]} "
        f"({worst[1]}) | lossless round trip on every matrix: {lossless_all}")
    # ---- 2. batch invariance + accuracy on layer 0..3 matrices + lm_head
    names = []
    for li in range(4):
        p = f"model.layers.{li}."
        names += [p + "self_attn.q_proj.weight", p + "self_attn.k_proj.weight", p + "self_attn.v_proj.weight",
                  p + "self_attn.o_proj.weight", p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight", p + "mlp.down_proj.weight"]
    names.append("lm_head.weight")
    gen = torch.Generator(device=dev).manual_seed(7)
    enc, raw = {}, {}
    ident_all, worst_rel = True, 0.0
    for name in names:
        W = get(name).to(dev)
        f = encode(W)
        enc[name], raw[name] = f, W
        K = W.shape[1]
        X = (torch.randn(13, K, device=dev, generator=gen)).half()
        one = torch.cat([gemv(X[t:t + 1], f) for t in range(13)], 0)
        for n in (2, 5, 8, 13):
            got = gemv(X[:n], f)
            if not torch.equal(got.view(torch.int16), one[:n].view(torch.int16)):
                ident_all = False
                log(f"  NOT batch-invariant: {name} n={n}")
        ref = X[:8].float() @ W.float().T
        rel = float((one[:8].float() - ref).abs().max() / ref.abs().max())
        worst_rel = max(worst_rel, rel)
    out.update(batch_invariant=ident_all, worst_rel_err=worst_rel)
    log(f"CAP1b batch invariance (n = 2/5/8/13 vs 1 row, 29 real matrices): {ident_all} | worst rel. error vs fp32 "
        f"reference with the exact bf16 weights: {worst_rel:.2e}")
    # ---- 3. in-graph speed: one forward's matmuls (36 layers cycling the 4 real layers + lm_head)
    layer_names = [names[i * 7:(i + 1) * 7] for i in range(4)]
    raw16 = {k: v.half() for k, v in raw.items()}
    Ks = sorted({raw[k].shape[1] for k in names})
    Xs = {K: torch.randn(8, K, device=dev, generator=gen).half() for K in Ks}
    Xb = {K: v.to(torch.bfloat16) for K, v in Xs.items()}
    variants = {
        "pytorch_bf16": lambda nm, n: F.linear(Xb[raw[nm].shape[1]][:n].view(1, n, -1), raw[nm]),
        "pytorch_fp16": lambda nm, n: F.linear(Xs[raw[nm].shape[1]][:n].view(1, n, -1), raw16[nm]),
    }
    for tile, r in ((256, 2), (512, 2), (512, 4)):
        variants[f"b12_T{tile}R{r}"] = (lambda tl, rr: lambda nm, n: gemv(Xs[raw[nm].shape[1]][:n].view(1, n, -1), enc[nm], tl, rr))(tile, r)
    res = {}
    for vname, fn in variants.items():
        res[vname] = {}
        for n in (1, 8):
            def fwd():
                for li in range(36):
                    for nm in layer_names[li % 4]:
                        fn(nm, n)
                fn("lm_head.weight", n)
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                fwd(); fwd()
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                fwd()
            for _ in range(10):
                g.replay()
            torch.cuda.synchronize()
            reps = []
            for _ in range(5):
                t = time.perf_counter()
                for _ in range(10):
                    g.replay()
                torch.cuda.synchronize()
                reps.append((time.perf_counter() - t) * 100)
            res[vname][n] = statistics.median(reps)
            del g
    out["ms"] = res
    layer_bytes_bf16 = 36 * sum(raw[k].numel() * 2 for k in layer_names[0]) / 1 + raw["lm_head.weight"].numel() * 2
    for vname, v in res.items():
        log(f"CAP1b one forward's matmuls (B, in graph) {vname:<14}: 1 row {v[1]:6.2f} ms | 8 rows {v[8]:6.2f} ms")
    b1 = min((k for k in res if k.startswith("b12")), key=lambda k: res[k][1])
    log(f"RESULT CAP1b: lossless {lossless_all} | batch-invariant {ident_all} | B12 {8 * tot_b / tot_w:.2f} bits/weight | "
        f"best B12 1 row {res[b1][1]:.2f} ms ({b1}) vs PyTorch bf16 {res['pytorch_bf16'][1]:.2f} ms -> "
        f"{res['pytorch_bf16'][1] / res[b1][1]:.2f}x | bf16 bytes per forward {layer_bytes_bf16 / 1e9:.2f} GB")
    (ROOT / "results" / "CAP1b_kernel.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
