"""CAP1c: exact Qwen3-8B on a 16 GB card - B12 weights (lossless, bit-identical to the bf16 checkpoint) + exact
speculative decoding with the 0.6B int4 draft. Pre-registered: docs/PREREG_CAP1.md (step 3). Research code: imports the
frozen engine modules, modifies none of them.

Memory plan (the 16.4 GB bf16 model never sits on the GPU):
  - target loaded in bf16 on the CPU; every matmul weight (q/k/v merged, o, gate/up merged, down, lm_head) is encoded to
    B12 on the GPU one matrix at a time and the CPU copy is freed; norms move to the GPU in fp16 (tiny).
  - the token-embedding table stays in system RAM (fp16 copy, 1.2 GB); each step copies only the needed rows to the GPU
    and the target is fed with inputs_embeds.
  - draft: Qwen3-0.6B exactly as in the validated engine (fp16 stack, int4-g32 weights, FREE_DUP).
Kernels: B12 batch-invariant GEMV for every target matmul (prefill in chunks of 32 rows - same per-row arithmetic),
generic-G split-KV attention (kernels_v7, G = 4), multi-token KV write (kernels_v6), fused norms / SwiGLU / RoPE.
Exactness check: speculative output vs plain decoding with the same engine, token for token.
External reference (information only): llama.cpp BF16 tokens from CAP1 step 1.
Writes results/CAP1c.json.
"""
import json
import os
import statistics
import sys
import threading
import time
import types
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import graph_decode as gd  # noqa: E402
import kernel_b12 as kb  # noqa: E402

NEWT = int(os.environ.get("SD_NEWT", "96"))
REPS = int(os.environ.get("SD_REPS", "2"))
WARM = os.environ.get("SD_WARM", "1") == "1"
PROMPTSET = os.environ.get("SD_PROMPTSET", "b6")      # b6 (11 prompts) or bi3 (50 prompts)
OUT = os.environ.get("SD_OUT", "CAP1c")
MAXL = 1024 + NEWT + 40


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cat_b12(fs):
    """Concatenate B12 matrices along rows (per-row format -> plain concatenation)."""
    N = sum(f["N"] for f in fs)
    K = fs[0]["K"]
    eoffs, base = [fs[0]["eoff"][:1]], 0
    for f in fs:
        eoffs.append(f["eoff"][1:] + base)
        base += f["n_exc"]
    ecol = torch.cat([f["ecol"][:f["n_exc"]] for f in fs]) if base else fs[0]["ecol"][:1]
    ebits = torch.cat([f["ebits"][:f["n_exc"]] for f in fs]) if base else fs[0]["ebits"][:1]
    return {"N": N, "K": K, "hi": torch.cat([f["hi"] for f in fs]), "ex": torch.cat([f["ex"] for f in fs]),
            "e0": torch.cat([f["e0"] for f in fs]), "eoff": torch.cat(eoffs).contiguous(), "ecol": ecol.contiguous(),
            "ebits": ebits.contiguous(), "n_exc": base, "bytes": sum(f["bytes"] for f in fs)}


@torch.no_grad()
def main():
    torch.set_grad_enabled(False)
    wd = threading.Timer(int(os.environ.get("STAGE_TIMEOUT_S", "3600")), lambda: (log("RESULT CAP1c: TIMEOUT"), os._exit(3)))
    wd.daemon = True
    wd.start()
    from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
    from transformers.models.qwen3 import modeling_qwen3 as mq
    import fuse_small
    import kernels_v2
    import kernels_v3
    import kernels_v6
    import kernels_v7
    import kernel_int4
    import kernel_decode_attn as kda
    from kernel_qknorm_rope import qk_norm_rope, get_kernel as qkr_kernel
    import b6_prompts
    import bi3_prompts
    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

    # ---------- draft first (its temporary fp16 copy needs memory before the target arrives)
    D = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float16, attn_implementation="sdpa").to(dev).eval()
    D.requires_grad_(False)
    gd.fuse_rmsnorms(D); gd.merge_projections(D); gd.fuse_attention(D, merged=True)

    # ---------- target on the CPU in bf16
    T = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype=torch.bfloat16, attn_implementation="sdpa").eval()
    T.requires_grad_(False)
    Hq, Hk = T.config.num_attention_heads, T.config.num_key_value_heads
    emb16 = T.model.embed_tokens.weight.data.to(torch.float16).contiguous()      # system RAM
    emb_changed = int((emb16.to(torch.bfloat16).view(torch.int16) != T.model.embed_tokens.weight.data.view(torch.int16)).sum())
    try:
        emb16 = emb16.pin_memory()
    except Exception:
        pass
    T.model.embed_tokens = torch.nn.Embedding(1, T.config.hidden_size, dtype=torch.float16)   # unused (inputs_embeds)
    rot_dtype = str(getattr(T.model.rotary_emb, "inv_freq", torch.zeros(1)).dtype)

    REG = {}
    b12 = {}
    t_enc = time.time()
    lm = kb.encode(T.lm_head.weight.data.to(dev))
    T.lm_head.weight.data = torch.empty(0, dtype=torch.bfloat16)
    for li, layer in enumerate(T.model.layers):
        a, m = layer.self_attn, layer.mlp
        sizes = [a.q_proj.out_features, a.k_proj.out_features, a.v_proj.out_features]
        qkv = cat_b12([kb.encode(l.weight.data.to(dev)) for l in (a.q_proj, a.k_proj, a.v_proj)])
        gu = cat_b12([kb.encode(l.weight.data.to(dev)) for l in (m.gate_proj, m.up_proj)])
        o = kb.encode(a.o_proj.weight.data.to(dev))
        dn = kb.encode(m.down_proj.weight.data.to(dev))
        for l in (a.q_proj, a.k_proj, a.v_proj, a.o_proj, m.gate_proj, m.up_proj, m.down_proj):
            l.weight.data = torch.empty(0, dtype=torch.bfloat16)
        b12[li] = (qkv, o, gu, dn, sizes)
        torch.cuda.empty_cache()
    b12_bytes = lm["bytes"] + sum(sum(x["bytes"] for x in v[:4]) for v in b12.values())
    log(f"CAP1c B12 encode of the target: {time.time() - t_enc:.0f} s | {b12_bytes / 1e9:.2f} GB on the GPU | embedding rows "
        f"changed by bf16->fp16: {emb_changed}")
    for p in T.parameters():                       # remaining small params (norms) -> fp16
        if p.dtype == torch.bfloat16:
            p.data = p.data.to(torch.float16)
    T.to(dev)
    gd.fuse_rmsnorms(T)
    qkr_kernel()

    def att_forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs):
        input_shape = hidden_states.shape[:-1]
        q, k, v = F.linear(hidden_states, self._w_qkv).split(self._qkv_sizes, dim=-1)
        cos, sin = position_embeddings
        qo, ko = qk_norm_rope(q, k, self.q_norm.weight, self.k_norm.weight, cos, sin, Hq, Hk, self.q_norm.variance_epsilon)
        query_states, key_states = qo.transpose(1, 2), ko.transpose(1, 2)
        value_states = v.reshape(*input_shape, -1, self.head_dim).transpose(1, 2)
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
        ai = mq.ALL_ATTENTION_FUNCTIONS.get_interface(self.config._attn_implementation, mq.eager_attention_forward)
        out, w = ai(self, query_states, key_states, value_states, attention_mask, dropout=0.0, scaling=self.scaling,
                    sliding_window=self.sliding_window, **kwargs)
        return self.o_proj(out.reshape(*input_shape, -1).contiguous()), w

    for li, layer in enumerate(T.model.layers):
        qkv, o, gu, dn, sizes = b12[li]
        a, m = layer.self_attn, layer.mlp
        a._w_qkv = torch.empty(0, dtype=torch.float16, device=dev); REG[id(a._w_qkv)] = qkv
        a._qkv_sizes = sizes
        m._w_gu = torch.empty(0, dtype=torch.float16, device=dev); REG[id(m._w_gu)] = gu
        REG[id(a.o_proj.weight)] = o
        REG[id(m.down_proj.weight)] = dn
        a.forward = types.MethodType(att_forward, a)
    REG[id(T.lm_head.weight)] = lm

    kda.get_kernels(); kernels_v3.kernels(); kernels_v2.kernels(); kernels_v6.kernels()
    kernels_v3.SPLITS = 16
    kda.decode_attn = kernels_v3.decode_attn
    sdpa_fn = mq.ALL_ATTENTION_FUNCTIONS.get_interface("sdpa", mq.eager_attention_forward)

    def our_att(module, q, k, v, mask, dropout=0.0, scaling=None, sliding_window=None, **kw):
        g2 = q.shape[1] == 2 * k.shape[1]
        if 1 < q.shape[2] <= 32:
            return (kernels_v6.attn_mq if g2 else kernels_v7.attn_mq)(q, k, v, module._len, scaling), None
        if q.shape[2] != 1:
            return sdpa_fn(module, q, k, v, mask, dropout=dropout, scaling=scaling, sliding_window=sliding_window, **kw)
        return (kda.decode_attn if g2 else kernels_v7.decode_attn)(q, k, v, module._len, scaling), None
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
        for l in (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj, layer.mlp.gate_proj, layer.mlp.up_proj):
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
        if n <= kb.NMAX:
            y = kb.gemv(x, f)
        else:
            y = torch.cat([kb.gemv(x[i:i + kb.NMAX], f) for i in range(0, n, kb.NMAX)], 0)
        return y.view(*input.shape[:-1], N)
    F.linear = b12_linear
    fr, tot = torch.cuda.mem_get_info()
    log(f"CAP1c loaded: torch allocated {torch.cuda.memory_allocated() / 2**30:.2f} GiB | rotary inv_freq dtype {rot_dtype}")

    cT = StaticCache(config=T.config, max_cache_len=MAXL)
    cD = StaticCache(config=D.config, max_cache_len=MAXL)

    def setc(cache, n):
        for layer in cache.layers:
            layer.cumulative_length.fill_(n)

    def embed(ids_list):
        return emb16[torch.tensor(ids_list, dtype=torch.long)].to(dev, non_blocking=True)

    def prefill_T(ids_list):
        if all(getattr(l_, "is_initialized", False) for l_ in cT.layers):
            setc(cT, 0)
        P = len(ids_list)
        lg = T(inputs_embeds=embed(ids_list).unsqueeze(0), past_key_values=cT, cache_position=torch.arange(P, device=dev),
               use_cache=True, logits_to_keep=1).logits
        for i, layer in enumerate(T.model.layers):
            layer.self_attn._len = cT.layers[i].cumulative_length
        return int(lg[0, -1].argmax())

    def prefill_D(ids_list):
        if all(getattr(l_, "is_initialized", False) for l_ in cD.layers):
            setc(cD, 0)
        P = len(ids_list)
        D(input_ids=torch.tensor([ids_list], device=dev), past_key_values=cD, cache_position=torch.arange(P, device=dev),
          use_cache=True, logits_to_keep=1)
        for i, layer in enumerate(D.model.layers):
            layer.self_attn._len = cD.layers[i].cumulative_length

    def capture(fn, before):
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

    prompts = (bi3_prompts if PROMPTSET == "bi3" else b6_prompts).prompts(tok)
    P0 = len(prompts[0][1])
    prefill_T(prompts[0][1]); prefill_D(prompts[0][1])           # long prompt first (sets _len)

    t_emb = torch.zeros((1, 1, T.config.hidden_size), dtype=torch.float16, device=dev)
    t_sp = torch.zeros((1,), dtype=torch.long, device=dev)
    t_tok = torch.zeros((1,), dtype=torch.long, device=dev)
    v_emb = torch.zeros((1, 8, T.config.hidden_size), dtype=torch.float16, device=dev)
    v_sp = torch.zeros((8,), dtype=torch.long, device=dev)
    base = torch.arange(8, device=dev)
    preds = torch.zeros((8,), dtype=torch.long, device=dev)
    props = torch.zeros((7,), dtype=torch.long, device=dev)
    d_st = torch.zeros((1, 1), dtype=torch.long, device=dev)
    d_sp = torch.zeros((1,), dtype=torch.long, device=dev)
    d_idx = torch.zeros((1,), dtype=torch.long, device=dev)
    di_ids = torch.zeros((1, 1), dtype=torch.long, device=dev)
    di_sp = torch.zeros((1,), dtype=torch.long, device=dev)

    def t_step():
        lg = T(inputs_embeds=t_emb, past_key_values=cT, cache_position=t_sp, use_cache=True).logits
        t_tok.copy_(lg[0, -1].argmax(-1).view(1))

    def v_step():
        lg = T(inputs_embeds=v_emb, past_key_values=cT, cache_position=v_sp, use_cache=True).logits
        preds.copy_(lg[0].argmax(-1))

    def d_step():
        lg = D(input_ids=d_st, past_key_values=cD, cache_position=d_sp, use_cache=True).logits
        nxt = lg[:, -1].argmax(-1)
        props.index_copy_(0, d_idx, nxt)
        d_st.copy_(nxt.view(1, 1)); d_sp.add_(1); d_idx.add_(1)

    def ingest():
        D(input_ids=di_ids, past_key_values=cD, cache_position=di_sp, use_cache=True, logits_to_keep=1)

    G1 = capture(t_step, lambda: (setc(cT, P0), t_sp.fill_(P0)))
    GV = capture(v_step, lambda: (setc(cT, P0), torch.add(base, P0, out=v_sp)))
    GD = capture(lambda: [d_step() for _ in range(7)], lambda: (setc(cD, P0), d_sp.fill_(P0), d_idx.zero_()))
    GI = capture(ingest, lambda: (setc(cD, P0), di_sp.fill_(P0)))
    fr, tot = torch.cuda.mem_get_info()
    log(f"CAP1c graphs captured | torch allocated {torch.cuda.memory_allocated() / 2**30:.2f} GiB, peak "
        f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")

    def plain(ids_list):
        P = len(ids_list)
        first = prefill_T(ids_list)
        toks = [first]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        pos = P
        for _ in range(NEWT - 1):
            t_emb.copy_(emb16[toks[-1]].view(1, 1, -1), non_blocking=True)
            setc(cT, pos); t_sp.fill_(pos)
            G1.replay()
            toks.append(int(t_tok))
            pos += 1
        dt = time.perf_counter() - t0
        return toks, dt * 1000 / (NEWT - 1)

    def spec(ids_list):
        P = len(ids_list)
        first = prefill_T(ids_list)
        prefill_D(ids_list)
        ctx = list(ids_list) + [first]
        Pt = Pd = P
        rounds = 0
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        while len(ctx) - P < NEWT:
            pending = ctx[-1]
            setc(cD, Pd); d_st.fill_(pending); d_sp.fill_(Pd); d_idx.zero_()
            GD.replay()
            pr = props.tolist()
            v_emb.copy_(embed([pending] + pr).unsqueeze(0))
            torch.add(base, Pt, out=v_sp); setc(cT, Pt)
            GV.replay()
            pd = preds.tolist()
            n = 0
            while n < 7 and pr[n] == pd[n]:
                n += 1
            new = pr[:n] + [pd[n]]
            Pt += 1 + n
            Pd += 1 + min(n, 6)
            if n == 7:
                di_ids.fill_(pr[6]); setc(cD, Pd); di_sp.fill_(Pd)
                GI.replay()
                Pd += 1
            ctx += new
            rounds += 1
        dt = time.perf_counter() - t0
        return ctx[P:P + NEWT], dt * 1000 / (NEWT - 1), rounds

    ref = {}
    rp = ROOT / "results" / "B6_llamacpp_lemonade_8b_ngl99.json"
    if rp.exists():
        for r in json.loads(rp.read_text(encoding="utf-8"))["rows"]:
            if r["config"] == "none":
                ref[r["prompt"]] = r["tokens"]
    rows = []
    for label, ids in prompts:
        if WARM or not rows:
            plain(ids); spec(ids)                                # warm-up
        pm, sm, last = [], [], None
        for _ in range(REPS):
            pt, pms = plain(ids)
            st, sms, rr = spec(ids)
            pm.append(pms); sm.append(sms); last = (pt, st, rr)
        pt, st, rr = last
        d = gd.first_diff(st, pt)
        rl = ref.get(label)
        dl = gd.first_diff(pt, rl[:NEWT]) if rl else "n/a"
        row = {"prompt": label, "plain_tok_s": 1000 / statistics.median(pm), "spec_tok_s": 1000 / statistics.median(sm),
               "rounds": rr, "spec_vs_plain_first_diff": d, "plain_vs_llamacpp_bf16_first_diff": dl,
               "tokens_plain": pt, "tokens_spec": st}
        rows.append(row)
        log(f"CAP1c {label[:24]:<24} | plain {row['plain_tok_s']:6.1f} | spec {row['spec_tok_s']:6.1f} tok/s "
            f"({row['spec_tok_s'] / row['plain_tok_s']:.2f}x) | {NEWT / rr:.2f} tok/round | spec==plain: {d is None} "
            f"| vs llama.cpp BF16 first diff: {dl}")
    mp = statistics.mean(r["plain_tok_s"] for r in rows)
    ms = statistics.mean(r["spec_tok_s"] for r in rows)
    ident = sum(r["spec_vs_plain_first_diff"] is None for r in rows)
    out = {"newt": NEWT, "reps": REPS, "b12_gb": b12_bytes / 1e9, "rows": rows,
           "timestamp_utc": datetime.now(timezone.utc).isoformat()}
    out["promptset"] = PROMPTSET
    (ROOT / "results" / f"{OUT}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    log(f"RESULT CAP1c [{OUT}, {PROMPTSET}, {len(rows)} prompts x {NEWT}]: Qwen3-8B B12 (lossless weights) on the GPU | plain {mp:.1f} tok/s | exact speculative {ms:.1f} tok/s"
        f" | spec==plain {ident}/{len(rows)} | llama.cpp BF16 best (step 1): 26.9 tok/s -> {ms / 26.94:.2f}x")
    wd.cancel()
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
