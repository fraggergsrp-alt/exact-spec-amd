"""S5: fuse the small per-layer GPU jobs of a decode step (sixth group of hand-written kernels).

S3/S4 showed ~1.3 ms/token of "small ops": ~13 tiny kernels per layer, each mostly waiting. Three fusions:
  (a) kv_write    - StaticCache decode update (arange, add, counter add, 2x index_copy: ~5 jobs) -> 1 job. Reads and
                    advances the cache's own GPU counter, so graph replay still works. Refuses to write past the end.
  (b) add_rmsnorm - residual add + the RMSNorm that follows it -> 1 job (both residual adds of every layer; the one at
                    the end of a layer is fused into the NEXT layer's input norm, the last one into the final norm).
  (c) silu_mul    - SiLU(gate) * up -> 1 job instead of 2.
Prior art: standard inference-engine fusions (vLLM, TensorRT-LLM, llama.cpp all fuse residual+norm and SwiGLU).
Arithmetic follows the unfused PyTorch path step by step (same float16 rounding points), so outputs should be
bit-identical except possibly (c), where SiLU's exp() may differ from PyTorch's in the last bit.
"""
import types

import torch
import torch.nn.functional as F

SRC = r"""
#pragma clang fp contract(off)

extern "C" __global__ void kv_write(const _Float16* __restrict__ k, const _Float16* __restrict__ v,
                                    _Float16* __restrict__ kc, _Float16* __restrict__ vc,
                                    long long* __restrict__ len_ptr, int HkD, int M, int D)
{
    const long long L = *len_ptr;
    if (L >= M) return;                                     // never write past the cache (the E2 hang)
    for (int i = threadIdx.x; i < HkD; i += blockDim.x) {
        const int h = i / D, d = i % D;
        const long long o = ((long long)h * M + L) * D + d;
        kc[o] = k[i];
        vc[o] = v[i];
    }
    __syncthreads();
    if (threadIdx.x == 0) *len_ptr = L + 1;
}

// h = a + b (rounded to fp16, as PyTorch does), y = RMSNorm(h) * w. Same reduction as rmsnorm_fused.
extern "C" __global__ void add_rmsnorm(const _Float16* __restrict__ a, const _Float16* __restrict__ b,
                                       const _Float16* __restrict__ w, _Float16* __restrict__ hout,
                                       _Float16* __restrict__ y, int n_cols, double eps)
{
    const int row = blockIdx.x;
    const long long off = (long long)row * n_cols;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n_cols; i += blockDim.x) {
        const _Float16 s = (_Float16)((float)a[off + i] + (float)b[off + i]);
        hout[off + i] = s;
        const float f = (float)s;
        ss += f * f;
    }
    __shared__ float red[256];
    red[threadIdx.x] = ss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    const float inv = 1.0f / sqrtf(red[0] / (float)n_cols + (float)eps);
    for (int i = threadIdx.x; i < n_cols; i += blockDim.x) {
        const _Float16 normed = (_Float16)((float)hout[off + i] * inv);
        y[off + i] = w[i] * normed;
    }
}

// out = fp16(SiLU(g)) * u, with g = gu[:, :F], u = gu[:, F:2F] (merged gate/up output).
extern "C" __global__ void silu_mul(const _Float16* __restrict__ gu, _Float16* __restrict__ out, int F, int rows)
{
    const long long n = (long long)rows * F;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        const long long r = i / F, c = i % F;
        const float g = (float)gu[r * 2 * F + c];
        const _Float16 act = (_Float16)(g / (1.0f + expf(-g)));
        out[i] = (_Float16)((float)act * (float)gu[r * 2 * F + F + c]);
    }
}
"""

_k = {}


def kernels():
    if not _k:
        for name in ("kv_write", "add_rmsnorm", "silu_mul"):
            _k[name] = torch.cuda._compile_kernel(SRC, name)
    return _k


def add_rmsnorm(a, b, w, eps):
    """Returns (a + b, RMSNorm(a + b) * w), both fp16, same shapes as a."""
    shape = a.shape
    a2, b2 = a.reshape(-1, shape[-1]).contiguous(), b.reshape(-1, shape[-1]).contiguous()
    h, y = torch.empty_like(a2), torch.empty_like(a2)
    kernels()["add_rmsnorm"](grid=(a2.shape[0], 1, 1), block=(256, 1, 1),
                             args=[a2, b2, w, h, y, a2.shape[1], float(eps)])
    return h.view(shape), y.view(shape)


def silu_mul(gu):
    F2 = gu.shape[-1]
    g2 = gu.reshape(-1, F2).contiguous()
    rows, Fh = g2.shape[0], F2 // 2
    out = torch.empty(rows, Fh, dtype=gu.dtype, device=gu.device)
    n = rows * Fh
    kernels()["silu_mul"](grid=(min(4096, (n + 255) // 256), 1, 1), block=(256, 1, 1), args=[g2, out, Fh, rows])
    return out.view(*gu.shape[:-1], Fh)


def install(model, cache_write=True, res_norm=True, swiglu=True):
    """Patch a merged/fused Qwen3 model (after graph_decode's fuse_rmsnorms, merge_projections, fuse_attention).
    Returns undo()."""
    from transformers import cache_utils
    kernels()
    undo = []

    if cache_write:
        SL = cache_utils.StaticLayer
        orig_update = SL.update

        def update(self, key_states, value_states, *args, **kwargs):
            if key_states.shape[-2] != 1 or not self.is_initialized:
                return orig_update(self, key_states, value_states, *args, **kwargs)
            Hk, D = key_states.shape[1], key_states.shape[-1]
            k = key_states.reshape(Hk, D).contiguous()
            v = value_states.reshape(Hk, D).contiguous()
            kernels()["kv_write"](grid=(1, 1, 1), block=(256, 1, 1),
                                  args=[k, v, self.keys, self.values, self.cumulative_length, Hk * D,
                                        self.max_cache_len, D])
            return self.keys, self.values
        SL.update = update
        undo.append(lambda: setattr(SL, "update", orig_update))

    if swiglu:
        for layer in model.model.layers:
            mlp = layer.mlp
            assert hasattr(mlp, "_w_gu"), "run merge_projections first"

            def mlp_forward(self, x):
                return self.down_proj(silu_mul(F.linear(x, self._w_gu)))
            old = mlp.forward
            mlp.forward = types.MethodType(mlp_forward, mlp)
            undo.append(lambda m=mlp, f=old: setattr(m, "forward", f))

    if res_norm:
        holder = {"pending": None}
        layers = model.model.layers
        for layer in layers:
            def layer_forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_values=None,
                              use_cache=False, position_embeddings=None, **kwargs):
                pend = holder["pending"] if self.self_attn.layer_idx > 0 else None
                eps = self.input_layernorm.variance_epsilon
                if pend is None:
                    residual, x = hidden_states, self.input_layernorm(hidden_states)
                else:  # previous layer's "h + mlp_out" fused with this layer's input norm
                    residual, x = add_rmsnorm(hidden_states, pend, self.input_layernorm.weight, eps)
                a, _ = self.self_attn(hidden_states=x, attention_mask=attention_mask, position_ids=position_ids,
                                      past_key_values=past_key_values, use_cache=use_cache,
                                      position_embeddings=position_embeddings, **kwargs)
                residual, x = add_rmsnorm(residual, a, self.post_attention_layernorm.weight,
                                          self.post_attention_layernorm.variance_epsilon)
                holder["pending"] = self.mlp(x)
                return residual                                  # NOTE: mlp output is added by the next norm
            old = layer.forward
            layer.forward = types.MethodType(layer_forward, layer)
            undo.append(lambda l=layer, f=old: setattr(l, "forward", f))

        norm = model.model.norm
        old_norm = norm.forward

        def norm_forward(self, h):
            pend, holder["pending"] = holder["pending"], None
            return add_rmsnorm(h, pend, self.weight, self.variance_epsilon)[1]
        norm.forward = types.MethodType(norm_forward, norm)
        undo.append(lambda: setattr(norm, "forward", old_norm))

    def restore():
        for u in reversed(undo):
            u()
    return restore


def _test():
    """S5a: each fused kernel vs the unfused PyTorch ops it replaces (bit-identical expected for (a), (b))."""
    import sys
    dev = torch.device("cuda")
    kernels()
    print("compiled OK", flush=True)
    g = torch.Generator(device=dev).manual_seed(3)
    ok_all = True
    for rows, cols in ((1, 1024), (1, 2048), (1024, 1024)):
        a = torch.randn(rows, cols, device=dev, generator=g).half()
        b = torch.randn(rows, cols, device=dev, generator=g).half()
        w = (torch.rand(cols, device=dev, generator=g) + 0.5).half()
        h_ref = a + b
        var = h_ref.float().pow(2).mean(-1, keepdim=True)
        y_ref = w * (h_ref.float() * torch.rsqrt(var + 1e-6)).half()
        h, y = add_rmsnorm(a, b, w, 1e-6)
        dh = int((h != h_ref).sum())
        dy = float((y.float() - y_ref.float()).abs().max())
        ok = dh == 0 and dy <= 2e-3 * float(y_ref.float().abs().max())
        ok_all &= ok
        print(f"add_rmsnorm {rows}x{cols}: residual bits differ {dh} | norm max diff {dy:.2e} -> "
              f"{'CORRECT' if ok else 'WRONG'}", flush=True)
    for rows, Fh in ((1, 3072), (1, 6144), (1024, 3072)):
        gu = torch.randn(rows, 2 * Fh, device=dev, generator=g).half() * 3
        ref = F.silu(gu[:, :Fh]) * gu[:, Fh:]
        out = silu_mul(gu)
        nd = int((out != ref).sum())
        md = float((out.float() - ref.float()).abs().max())
        ok = md <= 1e-2 * float(ref.float().abs().max())
        ok_all &= ok
        print(f"silu_mul {rows}x{Fh}: elements differing {nd} of {rows * Fh} | max diff {md:.2e} -> "
              f"{'CORRECT' if ok else 'WRONG'}", flush=True)
    Hk, M, D = 8, 1096, 128
    kc, vc = torch.zeros(1, Hk, M, D, device=dev).half(), torch.zeros(1, Hk, M, D, device=dev).half()
    k = torch.randn(1, 1, Hk, D, device=dev, generator=g).half().transpose(1, 2)
    v = torch.randn(1, 1, Hk, D, device=dev, generator=g).half().transpose(1, 2)
    L = torch.tensor(1024, dtype=torch.int64, device=dev)
    kernels()["kv_write"](grid=(1, 1, 1), block=(256, 1, 1),
                          args=[k.reshape(Hk, D).contiguous(), v.reshape(Hk, D).contiguous(), kc, vc, L, Hk * D, M, D])
    torch.cuda.synchronize()
    ok = bool(torch.equal(kc[0, :, 1024], k[0, :, 0]) and torch.equal(vc[0, :, 1024], v[0, :, 0])
              and int(L) == 1025 and int(kc[0, :, 1025].abs().sum()) == 0)
    full = torch.tensor(M, dtype=torch.int64, device=dev)
    kernels()["kv_write"](grid=(1, 1, 1), block=(256, 1, 1),
                          args=[k.reshape(Hk, D).contiguous(), v.reshape(Hk, D).contiguous(), kc, vc, full, Hk * D, M, D])
    torch.cuda.synchronize()
    ok &= int(full) == M  # refuses to write past the end and does not advance
    ok_all &= ok
    print(f"kv_write: slot 1024 written, counter 1024 -> {int(L)}, full cache refused -> {'CORRECT' if ok else 'WRONG'}",
          flush=True)
    print(f"RESULT S5a: all correct = {ok_all}", flush=True)
    sys.exit(0 if ok_all else 2)


if __name__ == "__main__":
    _test()
