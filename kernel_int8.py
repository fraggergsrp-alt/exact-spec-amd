"""Phase E13a: int8 weight-only quantization + a hand-written int8 GEMV kernel.

Idea ("move less data"): every token must read all ~1.1 GB of fp16 weights. Storing each weight as an
8-bit integer plus ONE fp16 scale per output row halves the bytes read. The kernel reads int8 weights
(16 per 16-byte load), converts them to float in registers, multiplies by the token's vector, and applies
the row's scale at the end:   y[row] = scale[row] * sum_k  int8_w[row,k] * x[k]

Quantization (per output row, symmetric): scale = max|w_row| / 127,  q = round(w / scale) in [-127, 127].
This is the simplest standard scheme (prior art: LLM.int8, llama.cpp Q8_0 uses per-32-weight blocks).

Checks here: kernel correctness vs a float32 reference of the SAME quantized weights, quantization error
vs the original fp16 weights, and speed/bandwidth per shape vs PyTorch fp16 (rotating >= 256 MB of copies).

Usage: .venv-gpu\\Scripts\\python.exe -u kernel_int8.py
"""
from __future__ import annotations
import json
import math
import os
import statistics
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent

I8_SRC = r"""
typedef signed char char16 __attribute__((ext_vector_type(16)));
typedef _Float16 half8 __attribute__((ext_vector_type(8)));

// y[N] = scale[N] * (Wq[N,K] (int8, row-major) . x[K]); 256 threads = 8 waves of 32, one row per wave.
extern "C" __global__ void gemv_i8(const signed char* __restrict__ Wq, const _Float16* __restrict__ scale,
                                   const _Float16* __restrict__ x, _Float16* __restrict__ y, int N, int K)
{
    __shared__ __attribute__((aligned(16))) _Float16 xs[4096];
    for (int i = threadIdx.x; i < K; i += blockDim.x) xs[i] = x[i];
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= N) return;

    const char16* wr = (const char16*)(Wq + (long long)row * K);
    const half8* xv = (const half8*)xs;
    float acc = 0.0f;
    for (int c = lane; c < (K >> 4); c += 32) {   // 16 weights (16 bytes) per load
        char16 w = wr[c];
        half8 a = xv[2 * c];
        half8 b = xv[2 * c + 1];
        #pragma unroll
        for (int j = 0; j < 8; ++j) acc += (float)w[j] * (float)a[j];
        #pragma unroll
        for (int j = 0; j < 8; ++j) acc += (float)w[j + 8] * (float)b[j];
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor(acc, off, 32);
    if (lane == 0) y[row] = (_Float16)(acc * (float)scale[row]);
}

// E14: group-wise version - one fp16 scale per G consecutive weights of a row (G = 32, 64 or 128).
// scale layout: [N, K/G] row-major. Each 16-weight chunk lies inside one group.
extern "C" __global__ void gemv_i8g(const signed char* __restrict__ Wq, const _Float16* __restrict__ scale,
                                    const _Float16* __restrict__ x, _Float16* __restrict__ y, int N, int K, int G)
{
    __shared__ __attribute__((aligned(16))) _Float16 xs[4096];
    for (int i = threadIdx.x; i < K; i += blockDim.x) xs[i] = x[i];
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= N) return;

    const char16* wr = (const char16*)(Wq + (long long)row * K);
    const _Float16* sr = scale + (long long)row * (K / G);
    const half8* xv = (const half8*)xs;
    float acc = 0.0f;
    for (int c = lane; c < (K >> 4); c += 32) {
        char16 w = wr[c];
        half8 a = xv[2 * c];
        half8 b = xv[2 * c + 1];
        float p = 0.0f;
        #pragma unroll
        for (int j = 0; j < 8; ++j) p += (float)w[j] * (float)a[j];
        #pragma unroll
        for (int j = 0; j < 8; ++j) p += (float)w[j + 8] * (float)b[j];
        acc += p * (float)sr[(c << 4) / G];
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor(acc, off, 32);
    if (lane == 0) y[row] = (_Float16)acc;
}
"""

_kernel = None
_kernel_g = None
_kernel_big = None      # B2: same kernels with an 8192-entry input buffer, used only when 4096 < K <= 8192
_kernel_g_big = None
I8_SRC_BIG = I8_SRC.replace("xs[4096]", "xs[8192]")
_registry = {}          # id(fp16 weight tensor) -> (int8 weights, fp16 scales, group size or 0 = per row)
_orig_linear = F.linear


def get_kernel():
    global _kernel
    if _kernel is None:
        _kernel = torch.cuda._compile_kernel(I8_SRC, "gemv_i8")
    return _kernel


def get_kernel_big():
    global _kernel_big
    if _kernel_big is None:
        _kernel_big = torch.cuda._compile_kernel(I8_SRC_BIG, "gemv_i8")
    return _kernel_big


def get_kernel_g_big():
    global _kernel_g_big
    if _kernel_g_big is None:
        _kernel_g_big = torch.cuda._compile_kernel(I8_SRC_BIG, "gemv_i8g")
    return _kernel_g_big


def get_kernel_g():
    global _kernel_g
    if _kernel_g is None:
        _kernel_g = torch.cuda._compile_kernel(I8_SRC, "gemv_i8g")
    return _kernel_g


def quantize_group(W, G):
    """Group-wise symmetric int8: one fp16 scale per G consecutive weights in a row."""
    N, K = W.shape
    assert K % G == 0 and G % 16 == 0
    Wf = W.float().view(N, K // G, G)
    scale = (Wf.abs().amax(dim=2) / 127.0).clamp_min(1e-12)
    q = torch.round(Wf / scale[..., None]).clamp(-127, 127).to(torch.int8).view(N, K).contiguous()
    return q, scale.to(torch.float16).contiguous()


def dequant(q, scale, G):
    N, K = q.shape
    if G == 0:
        return q.to(torch.float16) * scale[:, None]
    return (q.view(N, K // G, G).to(torch.float16) * scale[..., None]).view(N, K)


def gemv_i8g(x, q, scale, G):
    N, K = q.shape
    x1 = x.reshape(K).contiguous()
    y = torch.empty(N, dtype=torch.float16, device=x.device)
    assert K <= 8192, "input buffer holds at most 8192 values"
    (get_kernel_g() if K <= 4096 else get_kernel_g_big())(
        grid=((N + 7) // 8, 1, 1), block=(256, 1, 1), args=[q, scale, x1, y, N, K, G])
    return y.view(*x.shape[:-1], N)


def quantize(W):
    """Per-output-row symmetric int8. Returns (int8 [N,K] contiguous, fp16 scale [N])."""
    Wf = W.float()
    scale = (Wf.abs().amax(dim=1) / 127.0).clamp_min(1e-12)
    q = torch.round(Wf / scale[:, None]).clamp(-127, 127).to(torch.int8).contiguous()
    return q, scale.to(torch.float16).contiguous()


def gemv_i8(x, q, scale):
    N, K = q.shape
    x1 = x.reshape(K).contiguous()
    y = torch.empty(N, dtype=torch.float16, device=x.device)
    assert K <= 8192, "input buffer holds at most 8192 values"
    (get_kernel() if K <= 4096 else get_kernel_big())(
        grid=((N + 7) // 8, 1, 1), block=(256, 1, 1), args=[q, scale, x1, y, N, K])
    return y.view(*x.shape[:-1], N)


def quant_linear(input, weight, bias=None):
    """Drop-in for F.linear: registered weights use their int8 copy (kernel for 1 token,
    on-the-fly dequantization for prompts); everything else is untouched."""
    entry = _registry.get(id(weight))
    if entry is None or bias is not None:
        return _orig_linear(input, weight, bias)
    q, scale, G = entry
    K = q.shape[1]
    if input.numel() == K and input.dtype == torch.float16 and K % 16 == 0 and K <= 8192:
        return gemv_i8(input, q, scale) if G == 0 else gemv_i8g(input, q, scale, G)
    return _orig_linear(input, dequant(q, scale, G), None)


def install(weights, group=0):
    """Quantize the given fp16 weights (group=0: per row; else one scale per `group` weights) and
    route F.linear through quant_linear. Returns undo()."""
    get_kernel(); get_kernel_g(); get_kernel_big(); get_kernel_g_big()
    for w in weights:
        q, sc = quantize(w) if group == 0 else quantize_group(w, group)
        _registry[id(w)] = (q, sc, group)
    F.linear = quant_linear

    def undo():
        F.linear = _orig_linear
        _registry.clear()
    return undo


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def time_rotating(fn, items):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for it in items:
            fn(it)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        outs = [fn(it) for it in items]
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    t_all = []
    for _ in range(7):
        torch.cuda.synchronize(); t = time.perf_counter(); g.replay(); torch.cuda.synchronize()
        t_all.append((time.perf_counter() - t) / len(items))
    del g, outs
    return statistics.median(t_all)


def main():
    wd = threading.Timer(180, lambda: (log("TIMEOUT after 180 s"), os._exit(3)))
    wd.daemon = True
    wd.start()
    dev = torch.device("cuda")
    log("compiling int8 gemv kernel...")
    try:
        get_kernel()
    except Exception as ex:
        log(f"RESULT E13a: COMPILE FAILED -> {type(ex).__name__}: {str(ex)[:1500]}")
        sys.stdout.flush(); os._exit(1)
    log("compiled OK")
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float16).to(dev).eval()
    L0 = model.model.layers[0]
    real = {
        "qkv merged 1024->4096": (torch.cat([L0.self_attn.q_proj.weight, L0.self_attn.k_proj.weight,
                                             L0.self_attn.v_proj.weight], 0).contiguous(), 28),
        "o_proj 2048->1024": (L0.self_attn.o_proj.weight, 28),
        "gate_up merged 1024->6144": (torch.cat([L0.mlp.gate_proj.weight, L0.mlp.up_proj.weight], 0).contiguous(), 28),
        "down_proj 3072->1024": (L0.mlp.down_proj.weight, 28),
        "lm_head 1024->151936": (model.lm_head.weight, 1),
    }
    gen = torch.Generator(device=dev).manual_seed(2026)
    rows, all_ok, t16, t8 = [], True, 0.0, 0.0
    with torch.no_grad():
        for name, (W, count) in real.items():
            N, K = W.shape
            q, sc = quantize(W)
            x = torch.randn(1, 1, K, device=dev, generator=gen).half()
            deq32 = q.float() * sc.float()[:, None]
            y_ref_q = (deq32 @ x.float().reshape(K)).reshape(1, 1, N)          # exact result for quantized weights
            y_ref_orig = (W.float() @ x.float().reshape(K)).reshape(1, 1, N)   # original fp16 weights, in fp32
            y8 = gemv_i8(x, q, sc)
            torch.cuda.synchronize()
            kern_err = float((y8.float() - y_ref_q).abs().max())
            quant_err = float((y_ref_q - y_ref_orig).abs().max())
            rel_quant = float((y_ref_q - y_ref_orig).norm() / y_ref_orig.norm())
            ok = kern_err <= 4e-3 * float(y_ref_q.abs().max()) + 1e-3
            all_ok &= ok
            copies = max(1, math.ceil(256 * 2**20 / (N * K * 2)))
            ws = [torch.randn(N, K, device=dev, generator=gen).half() * 0.02 for _ in range(copies)]
            qs = [quantize(w) for w in ws]
            ta = time_rotating(lambda w: _orig_linear(x, w), ws)
            tb = time_rotating(lambda qq: gemv_i8(x, qq[0], qq[1]), qs)
            del ws, qs
            torch.cuda.empty_cache()
            t16 += ta * count
            t8 += tb * count
            bytes8 = N * K + N * 2
            rows.append({"shape": name, "kernel_err_vs_exact": kern_err, "quant_err_max": quant_err,
                         "quant_rel_err": rel_quant, "correct": ok, "fp16_torch_us": round(ta * 1e6, 1),
                         "int8_mine_us": round(tb * 1e6, 1), "int8_GBps": round(bytes8 / tb / 1e9, 1)})
            log(f"{name:<26} | kernel err {kern_err:.1e} -> {'CORRECT' if ok else 'WRONG'} | quantization rel. error "
                f"{100 * rel_quant:.2f}% | fp16 torch {ta * 1e6:6.1f} us | int8 mine {tb * 1e6:6.1f} us "
                f"({bytes8 / tb / 1e9:4.0f} GB/s) | {ta / tb:4.2f}x")
    log(f"RESULT E13a: all correct = {all_ok} | matrix time per token: fp16 {t16 * 1000:.2f} ms -> int8 "
        f"{t8 * 1000:.2f} ms | ESTIMATE (A): ~{(t16 - t8) * 1000:.2f} ms/token saved")
    out = {"rows": rows, "all_correct": all_ok, "fp16_ms": round(t16 * 1000, 3), "int8_ms": round(t8 * 1000, 3),
           "timestamp_utc": datetime.now(timezone.utc).isoformat()}
    (HERE / "results" / "E13a_int8.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    wd.cancel()
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
