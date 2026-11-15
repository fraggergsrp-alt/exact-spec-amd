"""Phase E11a: second fused kernel - q/k RMSNorm + rotary position embedding (RoPE) in ONE GPU job.

In every attention layer, Hugging Face does (for q, and again for k):
    q_norm (RMSNorm over each 128-number head)  ->  rotate_half (slice, negate, concatenate)
    ->  q*cos  ->  rotate_half(q)*sin  ->  add
That is ~12 separate GPU jobs per layer even after our E9 RMSNorm kernel. E5 found most of the
~2,000 tiny jobs per token are exactly these multiplies, adds and copies.
This kernel does all of it, for q AND k, in a single launch: one thread block per (token, head),
128 threads = one per number in the head.

It copies Hugging Face's rounding steps exactly (float32 sums, float16 results after each multiply/add),
so results should be identical or differ by at most one float16 step.

Checks: correctness vs Hugging Face (real Qwen3-0.6B weights and real cos/sin), for normal tensors
and for the "merged q/k/v" layout (q and k are slices of one bigger tensor); then speed in a GPU graph vs
(a) original Hugging Face and (b) our previous best (E9 fused RMSNorm + Hugging Face RoPE).

Usage: .venv-gpu\\Scripts\\python.exe -u kernel_qknorm_rope.py
"""
from __future__ import annotations
import json
import os
import statistics
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

QK_SRC = r"""
// E11c: forbid fusing multiply+add into one FMA instruction. Hugging Face rounds q*cos and
// rot*sin to float16 BEFORE adding; with FMA the kernel skipped that rounding and matched HF
// only ~78% bit-for-bit. With contraction off it matches ~99.97-99.99%.
#pragma clang fp contract(off)
// One block per (token, head) row; blockDim.x == 128 == head_dim.
// Rows [0, rows_q) are query heads, rows [rows_q, ...) are key heads.
// q/k may be slices of a bigger tensor: tok_stride_* = distance (in elements) between tokens.
extern "C" __global__ void qk_norm_rope(
    const _Float16* __restrict__ q, const _Float16* __restrict__ k,
    const _Float16* __restrict__ wq, const _Float16* __restrict__ wk,
    const _Float16* __restrict__ cosb, const _Float16* __restrict__ sinb,
    _Float16* __restrict__ qo, _Float16* __restrict__ ko,
    int rows_q, int heads_q, int heads_k, int tok_stride_q, int tok_stride_k, double eps)
{
    const int D = 128;
    const int r = blockIdx.x;
    const int i = threadIdx.x;
    const _Float16* src; const _Float16* w; _Float16* dst; int token;
    if (r < rows_q) {
        token = r / heads_q; int head = r % heads_q;
        src = q + (long long)token * tok_stride_q + head * D; w = wq; dst = qo + (long long)r * D;
    } else {
        int rr = r - rows_q;
        token = rr / heads_k; int head = rr % heads_k;
        src = k + (long long)token * tok_stride_k + head * D; w = wk; dst = ko + (long long)rr * D;
    }

    // RMSNorm over the 128 numbers of this head (float32 sum, shared-memory tree reduction)
    __shared__ float red[128];
    __shared__ _Float16 nrm[128];
    const float v = (float)src[i];
    red[i] = v * v;
    __syncthreads();
    for (int s = 64; s > 0; s >>= 1) {
        if (i < s) red[i] += red[i + s];
        __syncthreads();
    }
    const float inv = 1.0f / sqrtf(red[0] / (float)D + (float)eps);
    const _Float16 n = w[i] * (_Float16)(v * inv);      // same rounding as Hugging Face
    nrm[i] = n;
    __syncthreads();

    // RoPE: out = n*cos + rotate_half(n)*sin, rotate_half = (-second half, first half)
    const _Float16 rot = (i < 64) ? (_Float16)(-(float)nrm[i + 64]) : nrm[i - 64];
    const _Float16 c = cosb[(long long)token * D + i];
    const _Float16 sn = sinb[(long long)token * D + i];
    const _Float16 p1 = (_Float16)((float)n * (float)c);
    const _Float16 p2 = (_Float16)((float)rot * (float)sn);
    dst[i] = (_Float16)((float)p1 + (float)p2);
}
"""

_kernel = None


def get_kernel():
    global _kernel
    if _kernel is None:
        _kernel = torch.cuda._compile_kernel(QK_SRC, "qk_norm_rope")
    return _kernel


def qk_norm_rope(q3, k3, wq, wk, cos, sin, heads_q, heads_k, eps):
    """q3: (B, S, heads_q*128), k3: (B, S, heads_k*128) - may be slices with a larger token stride.
    cos/sin: (B, S, 128). Returns q, k as (B, S, heads, 128) contiguous, already normed + rotated."""
    B, S, _ = q3.shape
    assert q3.dtype == torch.float16 and q3.stride(-1) == 1 and k3.stride(-1) == 1
    assert q3.stride(0) == S * q3.stride(1) or B == 1
    assert k3.stride(0) == S * k3.stride(1) or B == 1
    cos, sin = cos.contiguous(), sin.contiguous()
    qo = torch.empty((B, S, heads_q, 128), dtype=q3.dtype, device=q3.device)
    ko = torch.empty((B, S, heads_k, 128), dtype=q3.dtype, device=q3.device)
    rows_q = B * S * heads_q
    get_kernel()(grid=(rows_q + B * S * heads_k, 1, 1), block=(128, 1, 1),
                 args=[q3, k3, wq, wk, cos, sin, qo, ko, rows_q, heads_q, heads_k,
                       q3.stride(1), k3.stride(1), float(eps)])
    return qo, ko


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def graph_us(fn, n=200):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n):
            fn()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    t_all = []
    for _ in range(7):
        torch.cuda.synchronize(); t = time.perf_counter(); g.replay(); torch.cuda.synchronize()
        t_all.append((time.perf_counter() - t) * 1e6 / n)
    del g
    return statistics.median(t_all)


def main():
    wd = threading.Timer(180, lambda: (log("TIMEOUT after 180 s"), os._exit(3)))
    wd.daemon = True
    wd.start()
    from transformers import AutoModelForCausalLM
    from transformers.models.qwen3 import modeling_qwen3 as mq
    from kernel_rmsnorm import KERNEL_SRC, BLOCK
    dev = torch.device("cuda")
    log("compiling qk_norm_rope kernel...")
    get_kernel()
    log("compiled OK")
    rms = torch.cuda._compile_kernel(KERNEL_SRC, "rmsnorm_fused")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float16).to(dev).eval()
    att = model.model.layers[0].self_attn
    Hq, Hk, D = model.config.num_attention_heads, model.config.num_key_value_heads, 128
    eps = float(att.q_norm.variance_epsilon)
    gen = torch.Generator(device=dev).manual_seed(2026)
    rows, all_ok = [], True

    def rms_fused(x, w):
        x2 = x.reshape(-1, D).contiguous()
        out = torch.empty_like(x2)
        rms(grid=(x2.shape[0], 1, 1), block=(BLOCK, 1, 1), args=[x2, w, out, D, eps])
        return out.view(x.shape)

    with torch.no_grad():
        for S, layout in [(1, "separate"), (1, "merged"), (1024, "separate"), (1024, "merged")]:
            pos = torch.arange(5000, 5000 + S, device=dev)[None]
            x = torch.randn(1, S, 1024, device=dev, dtype=torch.float16, generator=gen)
            cos, sin = model.model.rotary_emb(x, pos)
            if layout == "merged":   # q, k are slices of one (1, S, 4096) tensor, like after merging q/k/v
                big = (torch.randn(1, S, (Hq + 2 * Hk) * D, device=dev, generator=gen) * 2).half()
                q3, k3 = big[..., :Hq * D], big[..., Hq * D:(Hq + Hk) * D]
            else:
                q3 = (torch.randn(1, S, Hq * D, device=dev, generator=gen) * 2).half()
                k3 = (torch.randn(1, S, Hk * D, device=dev, generator=gen) * 2).half()

            def hf():
                qn = att.q_norm(q3.view(1, S, Hq, D)).transpose(1, 2)
                kn = att.k_norm(k3.view(1, S, Hk, D)).transpose(1, 2)
                return mq.apply_rotary_pos_emb(qn, kn, cos, sin)

            def prev_best():  # E9 fused RMSNorm + Hugging Face RoPE
                qn = rms_fused(q3.view(1, S, Hq, D), att.q_norm.weight).transpose(1, 2)
                kn = rms_fused(k3.view(1, S, Hk, D), att.k_norm.weight).transpose(1, 2)
                return mq.apply_rotary_pos_emb(qn, kn, cos, sin)

            def fused():
                return qk_norm_rope(q3, k3, att.q_norm.weight, att.k_norm.weight, cos, sin, Hq, Hk, eps)

            rq, rk = hf()
            fq, fk = fused()
            fq, fk = fq.transpose(1, 2), fk.transpose(1, 2)
            torch.cuda.synchronize()
            dq, dk = (fq.float() - rq.float()).abs(), (fk.float() - rk.float()).abs()
            maxd = max(float(dq.max()), float(dk.max()))
            exact = float(torch.cat([(fq == rq).flatten(), (fk == rk).flatten()]).float().mean())
            ok = maxd <= 4e-3 * max(float(rq.abs().max()), float(rk.abs().max()))
            all_ok &= ok
            t_hf, t_prev, t_fu = graph_us(hf), graph_us(prev_best), graph_us(fused)
            name = f"S={S:<4} {layout:<8}"
            rows.append({"case": name, "max_abs_diff": maxd, "bit_identical_fraction": exact, "correct": ok,
                         "hf_us": round(t_hf, 2), "prev_best_us": round(t_prev, 2), "fused_us": round(t_fu, 2)})
            log(f"{name} | max diff {maxd:.2e}, {100 * exact:6.2f}% bit-identical -> {'CORRECT' if ok else 'WRONG'} | "
                f"HF {t_hf:6.2f} us | prev best {t_prev:6.2f} us | fused {t_fu:6.2f} us "
                f"({t_prev / t_fu:4.1f}x vs prev best)")

    dec = [r for r in rows if r["case"].startswith("S=1 ") and "merged" in r["case"]][0]
    est = 28 * (dec["prev_best_us"] - dec["fused_us"]) / 1000
    log(f"RESULT E11a: all correct = {all_ok} | ESTIMATE (A): ~{est:.2f} ms/token saved vs previous best (28 layers)")
    out = {"cases": rows, "all_correct": all_ok, "estimated_saving_ms": round(est, 2),
           "timestamp_utc": datetime.now(timezone.utc).isoformat()}
    (HERE / "results" / "E11a_qknorm_rope.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    wd.cancel()
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
