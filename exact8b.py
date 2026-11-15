"""exact8b: a small, reusable engine for EXACT Qwen3 generation on a 16 GB AMD GPU.

    from exact8b import ExactEngine
    eng = ExactEngine("Qwen/Qwen3-8B", temperature=0.0, seed=42)
    print(eng.generate("Explain what a hash table is.", max_new_tokens=200))

What it does (see README):
  - target weights stored on the GPU in B12, a lossless 12-bit format (bit-identical to the bf16 checkpoint);
    the embedding table stays in system RAM;
  - exact speculative decoding with a Qwen3-0.6B int4 draft (k = 7): the output is bit-for-bit what plain decoding with
    this engine produces (batch-invariant kernels); with temperature > 0 the sampling is seeded and position-keyed,
    so the same seed gives the same text with or without speculation.
Limits: batch size 1; temperature sampling only (no top-k / top-p); temperature and seed are fixed per engine (they
are baked into the captured GPU graphs); context up to `max_context` tokens. Tested end to end with Qwen3-8B on an
RX 9070 XT (Windows, ROCm PyTorch). Qwen3-1.7B / 4B load through the same path but were validated with sd7b.py, not here.
"""
import os
import sys
import time
import types
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "research" / "cap1"))

EOS_IDS = (151643, 151645)          # <|endoftext|>, <|im_end|>


def _cat_b12(fs):
    N, K = sum(f["N"] for f in fs), fs[0]["K"]
    eoffs, base = [fs[0]["eoff"][:1]], 0
    for f in fs:
        eoffs.append(f["eoff"][1:] + base)
        base += f["n_exc"]
    ecol = torch.cat([f["ecol"][:f["n_exc"]] for f in fs]) if base else fs[0]["ecol"][:1]
    ebits = torch.cat([f["ebits"][:f["n_exc"]] for f in fs]) if base else fs[0]["ebits"][:1]
    return {"N": N, "K": K, "hi": torch.cat([f["hi"] for f in fs]), "ex": torch.cat([f["ex"] for f in fs]),
            "e0": torch.cat([f["e0"] for f in fs]), "eoff": torch.cat(eoffs).contiguous(), "ecol": ecol.contiguous(),
            "ebits": ebits.contiguous(), "n_exc": base, "bytes": sum(f["bytes"] for f in fs)}


class ExactEngine:
    @torch.no_grad()
    def __init__(self, model="Qwen/Qwen3-8B", draft="Qwen/Qwen3-0.6B", temperature=0.0, seed=42, max_context=4096,
                 k=7, verbose=True):
        os.environ["SD_TEMP"] = str(float(temperature))       # read by sampler.py at import
        os.environ["SD_SEED"] = str(int(seed))
        torch.set_grad_enabled(False)
        from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
        from transformers.models.qwen3 import modeling_qwen3 as mq
        import graph_decode as gd
        import kernel_b12 as kb
        import fuse_small
        import kernels_v2
        import kernels_v3
        import kernels_v6
        import kernels_v7
        import kernel_int4
        import kernel_decode_attn as kda
        import sampler
        from kernel_qknorm_rope import qk_norm_rope, get_kernel as qkr_kernel
        from inference_benchmark import make_prompt_ids
        self.sampler, self.kb, self.k, self.verbose = sampler, kb, k, verbose
        self.temperature, self.seed, self.max_context = float(temperature), int(seed), int(max_context)
        dev = self.dev = torch.device("cuda")
        t0 = time.time()
        self.tok = AutoTokenizer.from_pretrained(model)

        D = AutoModelForCausalLM.from_pretrained(draft, dtype=torch.float16, attn_implementation="sdpa").to(dev).eval()
        D.requires_grad_(False)
        gd.fuse_rmsnorms(D); gd.merge_projections(D); gd.fuse_attention(D, merged=True)

        T = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16, attn_implementation="sdpa").eval()
        T.requires_grad_(False)
        Hq, Hk = T.config.num_attention_heads, T.config.num_key_value_heads
        emb16 = T.model.embed_tokens.weight.data.to(torch.float16).contiguous()
        try:
            emb16 = emb16.pin_memory()
        except Exception:
            pass
        self.emb16 = emb16
        T.model.embed_tokens = torch.nn.Embedding(1, T.config.hidden_size, dtype=torch.float16)

        REG, enc = {}, {}
        lm = kb.encode(T.lm_head.weight.data.to(dev))
        T.lm_head.weight.data = torch.empty(0, dtype=torch.bfloat16)
        for li, layer in enumerate(T.model.layers):
            a, m = layer.self_attn, layer.mlp
            sizes = [a.q_proj.out_features, a.k_proj.out_features, a.v_proj.out_features]
            qkv = _cat_b12([kb.encode(l.weight.data.to(dev)) for l in (a.q_proj, a.k_proj, a.v_proj)])
            gu = _cat_b12([kb.encode(l.weight.data.to(dev)) for l in (m.gate_proj, m.up_proj)])
            o = kb.encode(a.o_proj.weight.data.to(dev))
            dn = kb.encode(m.down_proj.weight.data.to(dev))
            for l in (a.q_proj, a.k_proj, a.v_proj, a.o_proj, m.gate_proj, m.up_proj, m.down_proj):
                l.weight.data = torch.empty(0, dtype=torch.bfloat16)
            enc[li] = (qkv, o, gu, dn, sizes)
            torch.cuda.empty_cache()
        self.b12_gb = (lm["bytes"] + sum(sum(x["bytes"] for x in v[:4]) for v in enc.values())) / 1e9
        for p in T.parameters():
            if p.dtype == torch.bfloat16:
                p.data = p.data.to(torch.float16)
        T.to(dev)
        gd.fuse_rmsnorms(T)
        qkr_kernel()

        def att_forward(self_, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs):
            input_shape = hidden_states.shape[:-1]
            q, k_, v = F.linear(hidden_states, self_._w_qkv).split(self_._qkv_sizes, dim=-1)
            cos, sin = position_embeddings
            qo, ko = qk_norm_rope(q, k_, self_.q_norm.weight, self_.k_norm.weight, cos, sin, Hq, Hk,
                                  self_.q_norm.variance_epsilon)
            qs, ks = qo.transpose(1, 2), ko.transpose(1, 2)
            vs = v.reshape(*input_shape, -1, self_.head_dim).transpose(1, 2)
            ks, vs = past_key_values.update(ks, vs, self_.layer_idx)
            ai = mq.ALL_ATTENTION_FUNCTIONS.get_interface(self_.config._attn_implementation, mq.eager_attention_forward)
            out, w = ai(self_, qs, ks, vs, attention_mask, dropout=0.0, scaling=self_.scaling,
                        sliding_window=self_.sliding_window, **kwargs)
            return self_.o_proj(out.reshape(*input_shape, -1).contiguous()), w

        for li, layer in enumerate(T.model.layers):
            qkv, o, gu, dn, sizes = enc[li]
            a, m = layer.self_attn, layer.mlp
            a._w_qkv = torch.empty(0, dtype=torch.float16, device=dev); REG[id(a._w_qkv)] = qkv
            a._qkv_sizes = sizes
            m._w_gu = torch.empty(0, dtype=torch.float16, device=dev); REG[id(m._w_gu)] = gu
            REG[id(a.o_proj.weight)] = o
            REG[id(m.down_proj.weight)] = dn
            a.forward = types.MethodType(att_forward, a)
        REG[id(T.lm_head.weight)] = lm
        self._enc, self._lm = enc, lm                          # keep the B12 tensors alive

        kda.get_kernels(); kernels_v3.kernels(); kernels_v2.kernels(); kernels_v6.kernels()
        kernels_v3.SPLITS = 16
        kda.decode_attn = kernels_v3.decode_attn
        sdpa_fn = mq.ALL_ATTENTION_FUNCTIONS.get_interface("sdpa", mq.eager_attention_forward)

        def our_att(module, q, k_, v, mask, dropout=0.0, scaling=None, sliding_window=None, **kw):
            g2 = q.shape[1] == 2 * k_.shape[1]
            if 1 < q.shape[2] <= 32:
                return (kernels_v6.attn_mq if g2 else kernels_v7.attn_mq)(q, k_, v, module._len, scaling), None
            if q.shape[2] != 1:
                return sdpa_fn(module, q, k_, v, mask, dropout=dropout, scaling=scaling, sliding_window=sliding_window, **kw)
            return (kda.decode_attn if g2 else kernels_v7.decode_attn)(q, k_, v, module._len, scaling), None
        mq.ALL_ATTENTION_FUNCTIONS.get_interface = lambda *a_, **k_: our_att
        fuse_small.install(T)
        fuse_small.install(D)
        fuse_small.add_rmsnorm = kernels_v2.add_rmsnorm
        kernels_v6.install_kv_write(max_n=32)

        dws = []
        for layer in D.model.layers:
            dws += [layer.self_attn._w_qkv, layer.self_attn.o_proj.weight, layer.mlp._w_gu, layer.mlp.down_proj.weight]
        kernel_int4.install(dws + [D.lm_head.weight], search=True)
        for layer in D.model.layers:
            for l in (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj,
                      layer.mlp.gate_proj, layer.mlp.up_proj):
                l.weight.data = torch.empty(0, dtype=l.weight.dtype, device=dev)
        for w in dws:
            w.data = torch.empty(0, dtype=w.dtype, device=dev)
        torch.cuda.empty_cache()

        prev_linear = F.linear

        def b12_linear(input, weight, bias=None):
            f = REG.get(id(weight))
            if f is None or bias is not None:
                return prev_linear(input, weight, bias)
            K, N = f["K"], f["N"]
            n = input.numel() // K
            x = input.reshape(n, K)
            y = kb.gemv(x, f) if n <= kb.NMAX else torch.cat([kb.gemv(x[i:i + kb.NMAX], f) for i in range(0, n, kb.NMAX)], 0)
            return y.view(*input.shape[:-1], N)
        F.linear = b12_linear

        L = self.max_context + k + 8
        self.T, self.D = T, D
        self.cT = StaticCache(config=T.config, max_cache_len=L)
        self.cD = StaticCache(config=D.config, max_cache_len=L)
        warm = make_prompt_ids(self.tok, 1024 if self.max_context > 1100 else 64, dev)[0].tolist()
        self._prefill_T(warm); self._prefill_D(warm)            # long prompt first: sets the attention length pointers
        P0 = len(warm)
        H = T.config.hidden_size
        z = lambda *s, dt=torch.long: torch.zeros(s, dtype=dt, device=dev)   # noqa: E731
        self.t_emb, self.t_sp, self.t_tok = z(1, 1, H, dt=torch.float16), z(1), z(1)
        self.v_emb, self.v_sp, self.base, self.preds = z(1, k + 1, H, dt=torch.float16), z(k + 1), torch.arange(k + 1, device=dev), z(k + 1)
        self.props, self.d_st, self.d_sp, self.d_idx = z(k), z(1, 1), z(1), z(1)
        self.di_ids, self.di_sp = z(1, 1), z(1)
        pick = sampler.pick

        def t_step():
            lg = T(inputs_embeds=self.t_emb, past_key_values=self.cT, cache_position=self.t_sp, use_cache=True).logits
            self.t_tok.copy_(pick(lg[:, -1], self.t_sp).view(1))

        def v_step():
            lg = T(inputs_embeds=self.v_emb, past_key_values=self.cT, cache_position=self.v_sp, use_cache=True).logits
            self.preds.copy_(pick(lg[0], self.v_sp))

        def d_step():
            lg = D(input_ids=self.d_st, past_key_values=self.cD, cache_position=self.d_sp, use_cache=True).logits
            nxt = pick(lg[:, -1], self.d_sp)
            self.props.index_copy_(0, self.d_idx, nxt)
            self.d_st.copy_(nxt.view(1, 1)); self.d_sp.add_(1); self.d_idx.add_(1)

        def ingest():
            D(input_ids=self.di_ids, past_key_values=self.cD, cache_position=self.di_sp, use_cache=True, logits_to_keep=1)

        self.G1 = self._capture(t_step, lambda: (self._setc(self.cT, P0), self.t_sp.fill_(P0)))
        self.GV = self._capture(v_step, lambda: (self._setc(self.cT, P0), torch.add(self.base, P0, out=self.v_sp)))
        self.GD = self._capture(lambda: [d_step() for _ in range(k)],
                                lambda: (self._setc(self.cD, P0), self.d_sp.fill_(P0), self.d_idx.zero_()))
        self.GI = self._capture(ingest, lambda: (self._setc(self.cD, P0), self.di_sp.fill_(P0)))
        if verbose:
            print(f"[exact8b] ready in {time.time() - t0:.0f} s | {model} B12 {self.b12_gb:.2f} GB | draft {draft} int4 | "
                  f"T={self.temperature} seed={self.seed} | GPU memory {torch.cuda.memory_allocated() / 2**30:.1f} GiB",
                  flush=True)

    # ---------------------------------------------------------------- internals
    @staticmethod
    def _setc(cache, n):
        for layer in cache.layers:
            layer.cumulative_length.fill_(n)

    @staticmethod
    def _capture(fn, before):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                before(); fn()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        before()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        return g

    def _embed(self, ids):
        return self.emb16[torch.tensor(ids, dtype=torch.long)].to(self.dev, non_blocking=True)

    def _prefill_T(self, ids):
        if all(getattr(l_, "is_initialized", False) for l_ in self.cT.layers):
            self._setc(self.cT, 0)
        P = len(ids)
        lg = self.T(inputs_embeds=self._embed(ids).unsqueeze(0), past_key_values=self.cT,
                    cache_position=torch.arange(P, device=self.dev), use_cache=True, logits_to_keep=1).logits
        for i, layer in enumerate(self.T.model.layers):
            layer.self_attn._len = self.cT.layers[i].cumulative_length
        return int(self.sampler.pick(lg[0, -1:], torch.tensor([P - 1], device=self.dev))[0])

    def _prefill_D(self, ids):
        if all(getattr(l_, "is_initialized", False) for l_ in self.cD.layers):
            self._setc(self.cD, 0)
        P = len(ids)
        self.D(input_ids=torch.tensor([ids], device=self.dev), past_key_values=self.cD,
               cache_position=torch.arange(P, device=self.dev), use_cache=True, logits_to_keep=1)
        for i, layer in enumerate(self.D.model.layers):
            layer.self_attn._len = self.cD.layers[i].cumulative_length

    # ---------------------------------------------------------------- public API
    def encode_chat(self, messages):
        """messages: [{"role": "user"|"assistant"|"system", "content": str}, ...] -> token ids (thinking off)."""
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return self.tok(text, add_special_tokens=False)["input_ids"]

    @torch.no_grad()
    def generate_ids(self, ids, max_new_tokens=256, speculative=True, stop_at_eos=True, on_tokens=None):
        """Returns (new token ids, stats). on_tokens(list_of_new_ids) is called as tokens are produced."""
        ids = list(ids)
        P = len(ids)
        if P + max_new_tokens > self.max_context:
            raise ValueError(f"prompt ({P}) + max_new_tokens ({max_new_tokens}) exceeds max_context ({self.max_context})")
        k = self.k
        first = self._prefill_T(ids)
        out = [first]
        if on_tokens:
            on_tokens([first])
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        rounds = 0
        done = stop_at_eos and first in EOS_IDS
        if speculative and not done:
            self._prefill_D(ids)
            Pt = Pd = P
            while len(out) < max_new_tokens and not done:
                pending = out[-1]
                self._setc(self.cD, Pd); self.d_st.fill_(pending); self.d_sp.fill_(Pd); self.d_idx.zero_()
                self.GD.replay()
                pr = self.props.tolist()
                self.v_emb.copy_(self._embed([pending] + pr).unsqueeze(0))
                torch.add(self.base, Pt, out=self.v_sp); self._setc(self.cT, Pt)
                self.GV.replay()
                pd = self.preds.tolist()
                n = 0
                while n < k and pr[n] == pd[n]:
                    n += 1
                new = pr[:n] + [pd[n]]
                Pt += 1 + n
                Pd += 1 + min(n, k - 1)
                if n == k:
                    self.di_ids.fill_(pr[k - 1]); self._setc(self.cD, Pd); self.di_sp.fill_(Pd)
                    self.GI.replay()
                    Pd += 1
                if stop_at_eos and any(t in EOS_IDS for t in new):
                    new = new[:next(i for i, t in enumerate(new) if t in EOS_IDS) + 1]
                    done = True
                new = new[:max_new_tokens - len(out)]
                out += new
                rounds += 1
                if on_tokens:
                    on_tokens(new)
        else:
            pos = P
            while len(out) < max_new_tokens and not done:
                self.t_emb.copy_(self.emb16[out[-1]].view(1, 1, -1), non_blocking=True)
                self._setc(self.cT, pos); self.t_sp.fill_(pos)
                self.G1.replay()
                t = int(self.t_tok)
                out.append(t)
                pos += 1
                rounds += 1
                done = stop_at_eos and t in EOS_IDS
                if on_tokens:
                    on_tokens([t])
        dt = time.perf_counter() - t0
        n_dec = max(1, len(out) - 1)
        return out, {"tokens": len(out), "seconds": dt, "tok_s": n_dec / dt if dt > 0 else 0.0,
                     "rounds": rounds, "speculative": speculative}

    def generate(self, prompt, max_new_tokens=256, chat=True, speculative=True, stream=False):
        """Text in, text out. chat=True wraps the prompt in Qwen3's chat template (thinking off)."""
        ids = self.encode_chat([{"role": "user", "content": prompt}]) if chat else self.tok(prompt)["input_ids"]
        printed = [0]
        buf = []

        def on_tokens(new):
            buf.extend(new)
            if stream:
                text = self.tok.decode([t for t in buf if t not in EOS_IDS], skip_special_tokens=True)
                print(text[printed[0]:], end="", flush=True)
                printed[0] = len(text)
        out, stats = self.generate_ids(ids, max_new_tokens, speculative, on_tokens=on_tokens)
        if stream:
            print(flush=True)
        self.last_stats = stats
        return self.tok.decode([t for t in out if t not in EOS_IDS], skip_special_tokens=True)
