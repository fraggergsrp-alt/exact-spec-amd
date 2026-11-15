"""Phase E9: your first hand-written, fused GPU kernel - RMSNorm.

Why RMSNorm: one decode step of Qwen3-0.6B runs it 113 times (E5: 113 x mean, 113 x rsqrt).
The Hugging Face version is ~7 separate GPU jobs per call:
    to(float32) -> pow -> mean -> add eps -> rsqrt -> multiply -> to(float16) -> multiply by weight
Each job reads its input from VRAM and writes its result back. Our FUSED kernel does all of it
in ONE GPU program: read x once, keep everything in registers / on-chip shared memory, write once.

No installs: PyTorch's ROCm build ships AMD's runtime compiler (hiprtc) inside .venv-gpu, and
torch.cuda._compile_kernel compiles the kernel source below for your GPU (gfx1201) at run time.

Checks (in this order - correctness first, rule 8):
  1. compiles
  2. same answer as the Hugging Face RMSNorm (real weights from Qwen3-0.6B), several shapes
  3. speed of both, inside a GPU graph (so Python launch cost is excluded, like in decode)
Then an ESTIMATE (A) of what it could save per token if it replaced all 113 norms.

Usage: .venv-gpu\\Scripts\\python.exe -u kernel_rmsnorm.py
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

KERNEL_SRC = r"""
// One thread block per row. 256 threads cooperate on one row of n_cols numbers.
// _Float16 is the compiler's built-in 16-bit float type (same format as torch.float16).
extern "C" __global__ void rmsnorm_fused(const _Float16* __restrict__ x,
                                         const _Float16* __restrict__ w,
                                         _Float16* __restrict__ out,
                                         int n_cols, double eps)
{
    const int row = blockIdx.x;
    const _Float16* xr = x + (long long)row * n_cols;
    _Float16* orow = out + (long long)row * n_cols;

    // Step 1: each thread adds up x*x for its share of the row, in float32 (like Hugging Face).
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n_cols; i += blockDim.x) {
        float v = (float)xr[i];
        ss += v * v;
    }

    // Step 2: combine the 256 partial sums in fast on-chip shared memory (a "tree reduction").
    __shared__ float red[256];
    red[threadIdx.x] = ss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    const float inv = 1.0f / sqrtf(red[0] / (float)n_cols + (float)eps);

    // Step 3: normalize, round to float16 (as Hugging Face does), multiply by the weight, write once.
    for (int i = threadIdx.x; i < n_cols; i += blockDim.x) {
        _Float16 normed = (_Float16)((float)xr[i] * inv);
        orow[i] = w[i] * normed;
    }
}
"""

BLOCK = 256
NORMS_PER_TOKEN = 113   # E5
CALLS_IN_GRAPH = 200


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def graph_us_per_call(fn, n=CALLS_IN_GRAPH):
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
    for _ in range(3):  # warm (E7: GPU clocks ramp up under load)
        g.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(7):
        torch.cuda.synchronize(); t = time.perf_counter(); g.replay(); torch.cuda.synchronize()
        samples.append((time.perf_counter() - t) * 1e6 / n)
    del g
    return statistics.median(samples)


def main():
    wd = threading.Timer(180, lambda: (log("TIMEOUT after 180 s"), os._exit(3)))
    wd.daemon = True
    wd.start()
    results = {"torch": torch.__version__, "hip": getattr(torch.version, "hip", None),
               "device": torch.cuda.get_device_name(0)}
    dev = torch.device("cuda")

    # 1. compile
    log("compiling kernel with hiprtc...")
    try:
        t = time.perf_counter()
        kernel = torch.cuda._compile_kernel(KERNEL_SRC, "rmsnorm_fused")
        log(f"compiled OK in {time.perf_counter() - t:.2f} s")
        results["compiled"] = True
    except Exception as ex:
        results.update(compiled=False, error=f"{type(ex).__name__}: {str(ex)[:1500]}")
        log(f"RESULT E9: COMPILE FAILED -> {results['error']}")
        (HERE / "results" / "E9_rmsnorm.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        sys.stdout.flush(); os._exit(1)

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float16).to(dev).eval()
    hidden_norm = model.model.layers[0].input_layernorm   # weight size 1024
    head_norm = model.model.layers[0].self_attn.q_norm    # weight size 128
    eps = float(hidden_norm.variance_epsilon)

    def fused(x, norm):
        out = torch.empty_like(x)
        rows, cols = x.shape
        kernel(grid=(rows, 1, 1), block=(BLOCK, 1, 1), args=[x, norm.weight, out, cols, eps])
        return out

    cases = [("decode hidden (1 x 1024)", 1, hidden_norm), ("decode q heads (16 x 128)", 16, head_norm),
             ("decode k heads (8 x 128)", 8, head_norm), ("prefill hidden (1024 x 1024)", 1024, hidden_norm)]
    gen = torch.Generator(device=dev).manual_seed(2026)
    rows_out, all_ok = [], True
    with torch.no_grad():
        for name, rows, norm in cases:
            cols = norm.weight.numel()
            x = (torch.randn(rows, cols, device=dev, generator=gen) * 3.0).to(torch.float16).contiguous()
            ref = norm(x)
            got = fused(x, norm)
            torch.cuda.synchronize()
            diff = (got.float() - ref.float()).abs()
            maxd, exact = float(diff.max()), float((got == ref).float().mean())
            rel = float((diff / ref.float().abs().clamp_min(1e-3)).max())
            ok = maxd <= 4e-3 * float(ref.float().abs().max())   # within a few float16 steps of the largest value
            all_ok &= ok
            hf_us = graph_us_per_call(lambda: norm(x))
            fu_us = graph_us_per_call(lambda: fused(x, norm))
            rows_out.append({"case": name, "max_abs_diff": maxd, "exactly_equal_fraction": exact, "max_rel_diff": rel,
                             "correct": ok, "hf_us": round(hf_us, 2), "fused_us": round(fu_us, 2),
                             "speedup": round(hf_us / fu_us, 2)})
            log(f"{name:<30} | max diff {maxd:.2e}, {100 * exact:5.1f}% bit-identical -> "
                f"{'CORRECT' if ok else 'WRONG'} | HF {hf_us:6.2f} us vs fused {fu_us:6.2f} us ({hf_us / fu_us:4.1f}x)")

    results["cases"] = rows_out
    dec = [r for r in rows_out if r["case"].startswith("decode")]
    hidden, qh, kh = dec[0], dec[1], dec[2]
    # per token: 57 hidden norms (2 per layer + final) + 28 q-head + 28 k-head norms = 113
    saved_us = 57 * (hidden["hf_us"] - hidden["fused_us"]) + 28 * (qh["hf_us"] - qh["fused_us"]) \
        + 28 * (kh["hf_us"] - kh["fused_us"])
    results.update(all_correct=all_ok, estimated_saving_ms_per_token=round(saved_us / 1000, 2))
    log(f"RESULT E9: all correct = {all_ok} | ESTIMATE (A): replacing all {NORMS_PER_TOKEN} norms could save "
        f"~{saved_us / 1000:.2f} ms per token (graph decode is ~6.0-6.6 ms warm)")
    results["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    (HERE / "results" / "E9_rmsnorm.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    log("saved E9_rmsnorm.json")
    wd.cancel()
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
