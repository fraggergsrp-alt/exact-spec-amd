"""Phase E2: how much of the ~45 ms/token decode overhead can GPU graph capture recover?

E0 showed GPU graph replay (HIP graphs via torch.cuda.CUDAGraph) works on this PC and is
5.7x cheaper per tiny GPU job than eager launches. torch.compile does NOT work here
(TritonMissing), and llama.cpp's GPU builds crash on this driver. So we test the known fix
that IS available: capture one decode step of the real model as a graph and replay it.

Same model, weights, precision (fp16), attention (sdpa), prompt (1024 filler tokens),
and 64 new tokens as D001-E1. Three ways to decode, all greedy, ONE PER PROCESS (--stage),
so a hang or crash in one method cannot hide the results of the others:
  A. hf_generate   - model.generate(), what we measured before (includes HF's Python loop)
  B. manual_eager  - our own minimal loop: one model forward per token, StaticCache,
                     no generate() bookkeeping. Splits "generate overhead" from "model overhead".
  C. manual_graph  - the same single-token forward captured ONCE as a GPU graph, then replayed.
Correctness: B and C must produce the same tokens as A (first difference is reported).

Usage: .venv-gpu\\Scripts\\python.exe -u graph_decode.py --stage A   (then B, then C)
"""
from __future__ import annotations
import argparse
import json
import os
import threading
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from inference_benchmark import make_prompt_ids  # same filler prompt as D001-E1

MODEL_ID = "Qwen/Qwen3-0.6B"
N_PROMPT = 1024
NEW = 64
REPEAT = 3
STAGE_TIMEOUT_S = int(os.environ.get("STAGE_TIMEOUT_S", "300"))   # Q1: long quality runs set 900


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sync():
    torch.cuda.synchronize()


def first_diff(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)


def merge_projections(model):
    """E6: merge q/k/v into ONE matrix and gate/up into ONE matrix per layer (same math, fewer,
    bigger matrix-vector calls - E5 showed bigger shapes reach more memory bandwidth).
    Patches the forward methods of this loaded model only; returns a function that undoes it."""
    import types
    import torch.nn.functional as F
    from transformers.models.qwen3 import modeling_qwen3 as mq
    undo = []
    for layer in model.model.layers:
        att, mlp = layer.self_attn, layer.mlp
        assert att.q_proj.bias is None and att.k_proj.bias is None, "this patch assumes no projection bias"
        att._w_qkv = torch.cat([att.q_proj.weight, att.k_proj.weight, att.v_proj.weight], 0).contiguous()
        att._qkv_sizes = [att.q_proj.out_features, att.k_proj.out_features, att.v_proj.out_features]
        mlp._w_gu = torch.cat([mlp.gate_proj.weight, mlp.up_proj.weight], 0).contiguous()

        def att_forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs):
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)
            q, k, v = F.linear(hidden_states, self._w_qkv).split(self._qkv_sizes, dim=-1)  # 1 call instead of 3
            query_states = self.q_norm(q.view(hidden_shape)).transpose(1, 2)
            key_states = self.k_norm(k.view(hidden_shape)).transpose(1, 2)
            value_states = v.view(hidden_shape).transpose(1, 2)
            cos, sin = position_embeddings
            query_states, key_states = mq.apply_rotary_pos_emb(query_states, key_states, cos, sin)
            if past_key_values is not None:
                key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
            attention_interface = mq.ALL_ATTENTION_FUNCTIONS.get_interface(
                self.config._attn_implementation, mq.eager_attention_forward)
            attn_output, attn_weights = attention_interface(
                self, query_states, key_states, value_states, attention_mask, dropout=0.0,
                scaling=self.scaling, sliding_window=self.sliding_window, **kwargs)
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            return self.o_proj(attn_output), attn_weights

        def mlp_forward(self, x):
            g, u = F.linear(x, self._w_gu).split([self.intermediate_size, self.intermediate_size], dim=-1)
            return self.down_proj(self.act_fn(g) * u)  # 1 call instead of 2

        undo.append((att, att.forward, mlp, mlp.forward))
        att.forward = types.MethodType(att_forward, att)
        mlp.forward = types.MethodType(mlp_forward, mlp)

    def restore():
        for att, af, mlp, mf in undo:
            att.forward, mlp.forward = af, mf
    return restore


def fuse_rmsnorms(model):
    """E10: replace every Qwen3RMSNorm forward with the hand-written fused kernel from
    kernel_rmsnorm.py (1 GPU job instead of ~7). Returns a function that undoes it."""
    import types
    from kernel_rmsnorm import KERNEL_SRC, BLOCK
    kernel = torch.cuda._compile_kernel(KERNEL_SRC, "rmsnorm_fused")
    undo = []
    for m in model.modules():
        if type(m).__name__ == "Qwen3RMSNorm":
            def fwd(self, x):
                assert x.dtype == torch.float16
                shape = x.shape
                x2 = x.reshape(-1, shape[-1]).contiguous()
                out = torch.empty_like(x2)
                kernel(grid=(x2.shape[0], 1, 1), block=(BLOCK, 1, 1),
                       args=[x2, self.weight, out, x2.shape[1], float(self.variance_epsilon)])
                return out.view(shape)
            undo.append((m, m.forward))
            m.forward = types.MethodType(fwd, m)

    def restore():
        for m, f in undo:
            m.forward = f
    return restore, len(undo)


def fuse_attention(model, merged=True):
    """E11: replace each attention layer's forward with one that (optionally) uses merged q/k/v
    and does q/k RMSNorm + RoPE in ONE fused kernel (kernel_qknorm_rope.py). Returns undo()."""
    import types
    import torch.nn.functional as F
    from transformers.models.qwen3 import modeling_qwen3 as mq
    from kernel_qknorm_rope import qk_norm_rope, get_kernel
    get_kernel()
    Hq, Hk = model.config.num_attention_heads, model.config.num_key_value_heads
    undo = []
    for layer in model.model.layers:
        att = layer.self_attn
        assert att.head_dim == 128 and att.q_proj.bias is None
        assert att.q_norm.variance_epsilon == att.k_norm.variance_epsilon
        att._merged = merged
        if merged:
            att._w_qkv = torch.cat([att.q_proj.weight, att.k_proj.weight, att.v_proj.weight], 0).contiguous()
            att._qkv_sizes = [att.q_proj.out_features, att.k_proj.out_features, att.v_proj.out_features]

        def att_forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs):
            input_shape = hidden_states.shape[:-1]
            if self._merged:
                q, k, v = F.linear(hidden_states, self._w_qkv).split(self._qkv_sizes, dim=-1)
            else:
                q, k, v = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)
            cos, sin = position_embeddings
            qo, ko = qk_norm_rope(q, k, self.q_norm.weight, self.k_norm.weight, cos, sin, Hq, Hk,
                                  self.q_norm.variance_epsilon)
            query_states, key_states = qo.transpose(1, 2), ko.transpose(1, 2)
            value_states = v.reshape(*input_shape, -1, self.head_dim).transpose(1, 2)
            if past_key_values is not None:
                key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
            attention_interface = mq.ALL_ATTENTION_FUNCTIONS.get_interface(
                self.config._attn_implementation, mq.eager_attention_forward)
            attn_output, attn_weights = attention_interface(
                self, query_states, key_states, value_states, attention_mask, dropout=0.0,
                scaling=self.scaling, sliding_window=self.sliding_window, **kwargs)
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            return self.o_proj(attn_output), attn_weights

        undo.append((att, att.forward))
        att.forward = types.MethodType(att_forward, att)

    def restore():
        for a, f in undo:
            a.forward = f
    return restore


def main() -> None:
    global MODEL_ID, N_PROMPT
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "T", "U", "V", "W", "X", "Y", "Z", "Q1", "Q2", "SD1", "SD2"], required=True)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--prompt-tokens", type=int, default=N_PROMPT)
    ap.add_argument("--skip-quality", action="store_true", help="stage N: speed only (B2)")
    args = ap.parse_args()
    stage = args.stage
    MODEL_ID, N_PROMPT = args.model, args.prompt_tokens

    def timeout():
        log(f"RESULT {stage}: TIMEOUT - stage still running after {STAGE_TIMEOUT_S} s, process stopped by watchdog")
        os._exit(3)
    watchdog = threading.Timer(STAGE_TIMEOUT_S, timeout)  # E2 v2 hung for 30+ min; never again
    watchdog.daemon = True
    watchdog.start()

    from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
    dev = torch.device("cuda")
    log(f"stage {stage}: model {MODEL_ID}, prompt {N_PROMPT} tokens")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, attn_implementation="sdpa").to(device=dev, dtype=torch.float16).eval()
    ids = make_prompt_ids(tok, N_PROMPT, dev)
    results = {"stage": stage, "model": MODEL_ID, "prompt_tokens": N_PROMPT, "new_tokens": NEW,
               "repeats": REPEAT, "torch": torch.__version__,
               "hip": getattr(torch.version, "hip", None), "device": torch.cuda.get_device_name(0)}

    def gen(n_new):
        return model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=n_new,
                              min_new_tokens=n_new, do_sample=False, pad_token_id=tok.eos_token_id)

    # E2 v2 hung on the SECOND use of one StaticCache (re-prefilling a filled cache).
    # Hypothesis: reuse is the problem. So every decode run now gets a brand-new cache.
    def new_cache():
        return StaticCache(config=model.config, max_cache_len=N_PROMPT + NEW + 8)

    cache = None

    def prefill():
        pos = torch.arange(N_PROMPT, device=dev)
        logits = model(input_ids=ids, past_key_values=cache, cache_position=pos, use_cache=True).logits
        return logits[:, -1].argmax(-1)  # shape (1,), stays on GPU

    # FOUND IN E2b: in this transformers version, StaticLayer.update() IGNORES the cache_position
    # we pass and writes at its own GPU counter `cumulative_length`, adding +1 per call.
    # Our 3 warm-up steps therefore moved the write position to N_PROMPT+3 while the model's
    # positions/mask said N_PROMPT -> the graph wrote and read the wrong slots (garbage output).
    # It also explains the E2 v2 hang: re-prefilling a used cache pushed writes past the end.
    # Fix: put every layer's counter back to the true position after warm-up.
    def reset_counters(n):
        for layer in cache.layers:
            layer.cumulative_length.fill_(n)

    with torch.no_grad():
        log("reference tokens (generate)...")
        ref = gen(NEW)[0, N_PROMPT:].tolist()
        gen(1)
        sync()
        log("reference done")

        if stage == "A":
            t_full, t_one = [], []
            for i in range(REPEAT):
                sync(); t = time.perf_counter(); gen(NEW); sync(); t_full.append(time.perf_counter() - t)
                sync(); t = time.perf_counter(); gen(1); sync(); t_one.append(time.perf_counter() - t)
                log(f"A repeat {i + 1}/{REPEAT} done")
            ms = (statistics.median(t_full) - statistics.median(t_one)) * 1000 / (NEW - 1)
            results.update(ms_per_step=round(ms, 2), first_diff_vs_ref=None)
            log(f"RESULT A. hf_generate : {ms:7.2f} ms/step ({1000 / ms:6.1f} tok/s)")

        elif stage == "B":
            def eager_decode():
                nonlocal cache
                cache = new_cache()
                t0 = prefill()
                sync()
                start = time.perf_counter()
                toks, cur = [t0], t0
                pos = torch.tensor([N_PROMPT], device=dev)
                for _ in range(NEW - 1):
                    logits = model(input_ids=cur.view(1, 1), past_key_values=cache,
                                   cache_position=pos, use_cache=True).logits
                    cur = logits[:, -1].argmax(-1)
                    toks.append(cur)
                    pos = pos + 1
                sync()
                return torch.cat(toks).tolist(), (time.perf_counter() - start) * 1000 / (NEW - 1)

            eager_decode()
            log("B warm-up done")
            runs = []
            for i in range(REPEAT):
                runs.append(eager_decode())
                log(f"B repeat {i + 1}/{REPEAT} done ({runs[-1][1]:.2f} ms/step)")
            toks, ms = runs[-1][0], statistics.median(r[1] for r in runs)
            results.update(ms_per_step=round(ms, 2), first_diff_vs_ref=first_diff(toks, ref))
            log(f"RESULT B. manual_eager: {ms:7.2f} ms/step ({1000 / ms:6.1f} tok/s) | "
                f"first token difference vs generate: {first_diff(toks, ref)} (None = all {NEW} identical)")

        elif stage == "J":
            # E10: put the fused RMSNorm kernel into the real model. Warm graph decode for
            # original / fused norms / fused norms + merged matrices, tokens checked, then quality vs fp32.
            keep = []

            def build_graph():
                nonlocal cache
                cache = new_cache()
                st = torch.zeros((1, 1), dtype=torch.long, device=dev)
                sp = torch.zeros((1,), dtype=torch.long, device=dev)
                idx = torch.zeros((1,), dtype=torch.long, device=dev)
                buf = torch.zeros((NEW,), dtype=torch.long, device=dev)
                t0 = prefill()

                def full():
                    lg = model(input_ids=st, past_key_values=cache, cache_position=sp, use_cache=True).logits
                    nxt = lg[:, -1].argmax(-1)
                    buf.index_copy_(0, idx, nxt)
                    st.copy_(nxt.view(1, 1)); sp.add_(1); idx.add_(1)
                    return lg

                def arm():
                    reset_counters(N_PROMPT)
                    st.copy_(t0.view(1, 1)); sp.fill_(N_PROMPT); idx.fill_(1); buf.zero_(); buf[0] = t0[0]

                arm()
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(3):
                        full()
                torch.cuda.current_stream().wait_stream(s)
                sync()
                arm()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    full()
                keep.append(g)

                def run():
                    arm()
                    sync(); t = time.perf_counter()
                    for _ in range(NEW - 1):
                        g.replay()
                    sync()
                    return (time.perf_counter() - t) * 1000 / (NEW - 1), buf.tolist()
                return run

            def warm(name, run):
                run(); run()
                ms_all, toks = [], None
                for _ in range(5):
                    ms, toks = run()
                    ms_all.append(ms)
                ms = statistics.median(ms_all)
                log(f"J {name:<24} warm {ms:6.2f} ms/step ({1000 / ms:6.1f} tok/s) | tokens first diff vs generate: "
                    f"{first_diff(toks, ref)}")
                return round(ms, 2), first_diff(toks, ref)

            table = {}
            table["original"] = warm("original", build_graph())
            undo_norm, n_norms = fuse_rmsnorms(model)
            log(f"fused kernel installed in {n_norms} RMSNorm modules")
            table["fused_norms"] = warm("fused norms", build_graph())
            undo_merge = merge_projections(model)
            table["fused_norms_merged"] = warm("fused norms + merged", build_graph())
            undo_merge()
            undo_norm()
            table["original_again"] = warm("original (again)", build_graph())
            results["table"] = table

            # quality: fused-norm model vs float32 reference, same text for all
            m32 = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, dtype=torch.float32, attn_implementation="sdpa").to(device=dev).eval()
            texts = ["The capital of France is", "def fibonacci(n):\n", "Photosynthesis is the process by which plants"]
            items = [(t, tok(t, return_tensors="pt").input_ids.to(dev)) for t in texts] + \
                    [("filler 1024 tokens", make_prompt_ids(tok, 1024, dev))]
            qrows = []
            for label, p_ids in items:
                P = p_ids.shape[1]
                seq = model.generate(input_ids=p_ids, attention_mask=torch.ones_like(p_ids), max_new_tokens=48,
                                     min_new_tokens=48, do_sample=False, pad_token_id=tok.eos_token_id)
                l32 = torch.log_softmax(m32(input_ids=seq, use_cache=False).logits[0, P - 1:-1].float(), -1)
                lo = torch.log_softmax(model(input_ids=seq, use_cache=False).logits[0, P - 1:-1].float(), -1)
                undo_norm, _ = fuse_rmsnorms(model)
                lf = torch.log_softmax(model(input_ids=seq, use_cache=False).logits[0, P - 1:-1].float(), -1)
                seq_f = model.generate(input_ids=p_ids, attention_mask=torch.ones_like(p_ids), max_new_tokens=48,
                                       min_new_tokens=48, do_sample=False, pad_token_id=tok.eos_token_id)
                undo_norm()
                kl = lambda a, b: float((a.exp() * (a - b)).sum(-1).mean())
                top = lambda a, b: float((a.argmax(-1) == b.argmax(-1)).float().mean())
                row = {"prompt": label[:30], "orig_KL_vs_fp32": kl(l32, lo), "fused_KL_vs_fp32": kl(l32, lf),
                       "orig_top1": top(l32, lo), "fused_top1": top(l32, lf),
                       "text_first_diff": first_diff(seq[0, P:].tolist(), seq_f[0, P:].tolist())}
                qrows.append(row)
                log(f"J quality {label[:22]!r:<26} KL vs fp32: orig {row['orig_KL_vs_fp32']:.2e} fused "
                    f"{row['fused_KL_vs_fp32']:.2e} | top-1: orig {row['orig_top1']:.3f} fused {row['fused_top1']:.3f} "
                    f"| text first diff {row['text_first_diff']}")
            results["quality"] = qrows
            base = (table["original"][0] + table["original_again"][0]) / 2
            log(f"RESULT J: original {table['original'][0]}/{table['original_again'][0]} | fused norms "
                f"{table['fused_norms'][0]} | fused norms + merged {table['fused_norms_merged'][0]} ms/step | "
                f"vs original: {base / table['fused_norms'][0]:.2f}x and {base / table['fused_norms_merged'][0]:.2f}x")

        elif stage == "K":
            # E11b: previous best (fused RMSNorm + merged) vs new (+ fused q/k-norm+RoPE kernel), order P-N-P,
            # with an untimed warm-up of the first method; tokens checked; then quality vs fp32.
            keep = []

            def build_graph():
                nonlocal cache
                cache = new_cache()
                st = torch.zeros((1, 1), dtype=torch.long, device=dev)
                sp = torch.zeros((1,), dtype=torch.long, device=dev)
                idx = torch.zeros((1,), dtype=torch.long, device=dev)
                buf = torch.zeros((NEW,), dtype=torch.long, device=dev)
                t0 = prefill()

                def full():
                    lg = model(input_ids=st, past_key_values=cache, cache_position=sp, use_cache=True).logits
                    nxt = lg[:, -1].argmax(-1)
                    buf.index_copy_(0, idx, nxt)
                    st.copy_(nxt.view(1, 1)); sp.add_(1); idx.add_(1)
                    return lg

                def arm():
                    reset_counters(N_PROMPT)
                    st.copy_(t0.view(1, 1)); sp.fill_(N_PROMPT); idx.fill_(1); buf.zero_(); buf[0] = t0[0]

                arm()
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(3):
                        full()
                torch.cuda.current_stream().wait_stream(s)
                sync()
                arm()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    full()
                keep.append(g)

                def run():
                    arm()
                    sync(); t = time.perf_counter()
                    for _ in range(NEW - 1):
                        g.replay()
                    sync()
                    return (time.perf_counter() - t) * 1000 / (NEW - 1), buf.tolist()
                return run

            def warm(name, run):
                run(); run()
                ms_all, toks = [], None
                for _ in range(5):
                    ms, toks = run()
                    ms_all.append(ms)
                ms = statistics.median(ms_all)
                log(f"K {name:<24} warm {ms:6.2f} ms/step ({1000 / ms:6.1f} tok/s) | tokens first diff vs generate: "
                    f"{first_diff(toks, ref)}")
                return round(ms, 2), first_diff(toks, ref)

            table = {}
            undo_norm, n_norms = fuse_rmsnorms(model)
            undo_merge = merge_projections(model)
            first = build_graph()
            for _ in range(5):
                first()   # untimed warm-up of the FIRST method too (E10 lesson)
            table["prev_best"] = warm("prev best (norm+merge)", first)
            undo_att = fuse_attention(model, merged=True)      # on top of merge: MLP stays merged
            table["new"] = warm("new (+ qk-norm-RoPE)", build_graph())
            undo_att()                                          # back to exactly the previous best
            table["prev_best_again"] = warm("prev best (again)", build_graph())
            undo_merge()
            undo_norm()
            results["table"] = table

            # quality: fused-norm model vs float32 reference, same text for all
            m32 = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, dtype=torch.float32, attn_implementation="sdpa").to(device=dev).eval()
            texts = ["The capital of France is", "def fibonacci(n):\n", "Photosynthesis is the process by which plants"]
            items = [(t, tok(t, return_tensors="pt").input_ids.to(dev)) for t in texts] + \
                    [("filler 1024 tokens", make_prompt_ids(tok, 1024, dev))]
            qrows = []
            for label, p_ids in items:
                P = p_ids.shape[1]
                seq = model.generate(input_ids=p_ids, attention_mask=torch.ones_like(p_ids), max_new_tokens=48,
                                     min_new_tokens=48, do_sample=False, pad_token_id=tok.eos_token_id)
                l32 = torch.log_softmax(m32(input_ids=seq, use_cache=False).logits[0, P - 1:-1].float(), -1)
                lo = torch.log_softmax(model(input_ids=seq, use_cache=False).logits[0, P - 1:-1].float(), -1)
                undo_norm, _ = fuse_rmsnorms(model)
                undo_merge = merge_projections(model)
                undo_att = fuse_attention(model, merged=True)   # the full "new" model
                lf = torch.log_softmax(model(input_ids=seq, use_cache=False).logits[0, P - 1:-1].float(), -1)
                seq_f = model.generate(input_ids=p_ids, attention_mask=torch.ones_like(p_ids), max_new_tokens=48,
                                       min_new_tokens=48, do_sample=False, pad_token_id=tok.eos_token_id)
                undo_att()
                undo_merge()
                undo_norm()
                kl = lambda a, b: float((a.exp() * (a - b)).sum(-1).mean())
                top = lambda a, b: float((a.argmax(-1) == b.argmax(-1)).float().mean())
                row = {"prompt": label[:30], "orig_KL_vs_fp32": kl(l32, lo), "fused_KL_vs_fp32": kl(l32, lf),
                       "orig_top1": top(l32, lo), "fused_top1": top(l32, lf),
                       "text_first_diff": first_diff(seq[0, P:].tolist(), seq_f[0, P:].tolist())}
                qrows.append(row)
                log(f"K quality {label[:22]!r:<26} KL vs fp32: orig {row['orig_KL_vs_fp32']:.2e} fused "
                    f"{row['fused_KL_vs_fp32']:.2e} | top-1: orig {row['orig_top1']:.3f} fused {row['fused_top1']:.3f} "
                    f"| text first diff {row['text_first_diff']}")
            results["quality"] = qrows
            base = (table["prev_best"][0] + table["prev_best_again"][0]) / 2
            log(f"RESULT K: prev best {table['prev_best'][0]}/{table['prev_best_again'][0]} -> new "
                f"{table['new'][0]} ms/step ({1000 / table['new'][0]:.0f} tok/s) | saved {base - table['new'][0]:.2f} ms "
                f"({base / table['new'][0]:.2f}x) | new tokens first diff vs generate: {table['new'][1]}")

        elif stage == "L":
            # E12b: previous best (fused norms + merged + fused q/k-norm-RoPE) vs new (+ one-token GEMV kernel
            # for every matrix), order P-N-P with warm-up; tokens checked; then quality vs fp32.
            import kernel_gemv
            keep = []

            def build_graph():
                nonlocal cache
                cache = new_cache()
                st = torch.zeros((1, 1), dtype=torch.long, device=dev)
                sp = torch.zeros((1,), dtype=torch.long, device=dev)
                idx = torch.zeros((1,), dtype=torch.long, device=dev)
                buf = torch.zeros((NEW,), dtype=torch.long, device=dev)
                t0 = prefill()

                def full():
                    lg = model(input_ids=st, past_key_values=cache, cache_position=sp, use_cache=True).logits
                    nxt = lg[:, -1].argmax(-1)
                    buf.index_copy_(0, idx, nxt)
                    st.copy_(nxt.view(1, 1)); sp.add_(1); idx.add_(1)
                    return lg

                def arm():
                    reset_counters(N_PROMPT)
                    st.copy_(t0.view(1, 1)); sp.fill_(N_PROMPT); idx.fill_(1); buf.zero_(); buf[0] = t0[0]

                arm()
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(3):
                        full()
                torch.cuda.current_stream().wait_stream(s)
                sync()
                arm()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    full()
                keep.append(g)

                def run():
                    arm()
                    sync(); t = time.perf_counter()
                    for _ in range(NEW - 1):
                        g.replay()
                    sync()
                    return (time.perf_counter() - t) * 1000 / (NEW - 1), buf.tolist()
                return run

            def warm(name, run):
                run(); run()
                ms_all, toks = [], None
                for _ in range(5):
                    ms, toks = run()
                    ms_all.append(ms)
                ms = statistics.median(ms_all)
                log(f"L {name:<24} warm {ms:6.2f} ms/step ({1000 / ms:6.1f} tok/s) | tokens first diff vs generate: "
                    f"{first_diff(toks, ref)}")
                return round(ms, 2), first_diff(toks, ref)

            table = {}
            undo_norm, n_norms = fuse_rmsnorms(model)
            undo_merge = merge_projections(model)
            undo_att = fuse_attention(model, merged=True)
            first = build_graph()
            for _ in range(5):
                first()
            table["prev_best"] = warm("prev best (3 opts)", first)
            undo_gemv = kernel_gemv.install()
            table["new"] = warm("new (+ GEMV kernel)", build_graph())
            undo_gemv()
            table["prev_best_again"] = warm("prev best (again)", build_graph())
            undo_att()
            undo_merge()
            undo_norm()
            results["table"] = table

            # quality: fused-norm model vs float32 reference, same text for all
            m32 = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, dtype=torch.float32, attn_implementation="sdpa").to(device=dev).eval()
            texts = ["The capital of France is", "def fibonacci(n):\n", "Photosynthesis is the process by which plants"]
            items = [(t, tok(t, return_tensors="pt").input_ids.to(dev)) for t in texts] + \
                    [("filler 1024 tokens", make_prompt_ids(tok, 1024, dev))]
            qrows = []
            for label, p_ids in items:
                P = p_ids.shape[1]
                seq = model.generate(input_ids=p_ids, attention_mask=torch.ones_like(p_ids), max_new_tokens=48,
                                     min_new_tokens=48, do_sample=False, pad_token_id=tok.eos_token_id)
                l32 = torch.log_softmax(m32(input_ids=seq, use_cache=False).logits[0, P - 1:-1].float(), -1)
                lo = torch.log_softmax(model(input_ids=seq, use_cache=False).logits[0, P - 1:-1].float(), -1)
                undo_norm, _ = fuse_rmsnorms(model)
                undo_merge = merge_projections(model)
                undo_att = fuse_attention(model, merged=True)
                undo_gemv = kernel_gemv.install()               # the full "new" model
                lf = torch.log_softmax(model(input_ids=seq, use_cache=False).logits[0, P - 1:-1].float(), -1)
                seq_f = model.generate(input_ids=p_ids, attention_mask=torch.ones_like(p_ids), max_new_tokens=48,
                                       min_new_tokens=48, do_sample=False, pad_token_id=tok.eos_token_id)
                undo_gemv()
                undo_att()
                undo_merge()
                undo_norm()
                kl = lambda a, b: float((a.exp() * (a - b)).sum(-1).mean())
                top = lambda a, b: float((a.argmax(-1) == b.argmax(-1)).float().mean())
                row = {"prompt": label[:30], "orig_KL_vs_fp32": kl(l32, lo), "fused_KL_vs_fp32": kl(l32, lf),
                       "orig_top1": top(l32, lo), "fused_top1": top(l32, lf),
                       "text_first_diff": first_diff(seq[0, P:].tolist(), seq_f[0, P:].tolist())}
                qrows.append(row)
                log(f"L quality {label[:22]!r:<26} KL vs fp32: orig {row['orig_KL_vs_fp32']:.2e} fused "
                    f"{row['fused_KL_vs_fp32']:.2e} | top-1: orig {row['orig_top1']:.3f} fused {row['fused_top1']:.3f} "
                    f"| text first diff {row['text_first_diff']}")
            results["quality"] = qrows
            base = (table["prev_best"][0] + table["prev_best_again"][0]) / 2
            log(f"RESULT L: prev best {table['prev_best'][0]}/{table['prev_best_again'][0]} -> new "
                f"{table['new'][0]} ms/step ({1000 / table['new'][0]:.0f} tok/s) | saved {base - table['new'][0]:.2f} ms "
                f"({base / table['new'][0]:.2f}x) | new tokens first diff vs generate: {table['new'][1]}")

        elif stage == "M":
            # E13b (pre-registered, docs/PREREG_E13.md): previous best (fused norms + merged + fused q/k-norm-RoPE)
            # vs + int8 weight-only quantization with the int8 GEMV kernel. P-N-P with warm-up; quality vs fp32.
            import kernel_int8

            def qweights():
                ws = []
                for layer in model.model.layers:
                    ws += [layer.self_attn._w_qkv, layer.self_attn.o_proj.weight,
                           layer.mlp._w_gu, layer.mlp.down_proj.weight]
                return ws + [model.lm_head.weight]
            keep = []

            def build_graph():
                nonlocal cache
                cache = new_cache()
                st = torch.zeros((1, 1), dtype=torch.long, device=dev)
                sp = torch.zeros((1,), dtype=torch.long, device=dev)
                idx = torch.zeros((1,), dtype=torch.long, device=dev)
                buf = torch.zeros((NEW,), dtype=torch.long, device=dev)
                t0 = prefill()

                def full():
                    lg = model(input_ids=st, past_key_values=cache, cache_position=sp, use_cache=True).logits
                    nxt = lg[:, -1].argmax(-1)
                    buf.index_copy_(0, idx, nxt)
                    st.copy_(nxt.view(1, 1)); sp.add_(1); idx.add_(1)
                    return lg

                def arm():
                    reset_counters(N_PROMPT)
                    st.copy_(t0.view(1, 1)); sp.fill_(N_PROMPT); idx.fill_(1); buf.zero_(); buf[0] = t0[0]

                arm()
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(3):
                        full()
                torch.cuda.current_stream().wait_stream(s)
                sync()
                arm()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    full()
                keep.append(g)

                def run():
                    arm()
                    sync(); t = time.perf_counter()
                    for _ in range(NEW - 1):
                        g.replay()
                    sync()
                    return (time.perf_counter() - t) * 1000 / (NEW - 1), buf.tolist()
                return run

            def warm(name, run):
                run(); run()
                ms_all, toks = [], None
                for _ in range(5):
                    ms, toks = run()
                    ms_all.append(ms)
                ms = statistics.median(ms_all)
                log(f"M {name:<24} warm {ms:6.2f} ms/step ({1000 / ms:6.1f} tok/s) | tokens first diff vs generate: "
                    f"{first_diff(toks, ref)}")
                return round(ms, 2), first_diff(toks, ref)

            table = {}
            undo_norm, n_norms = fuse_rmsnorms(model)
            undo_merge = merge_projections(model)
            undo_att = fuse_attention(model, merged=True)
            first = build_graph()
            for _ in range(5):
                first()
            table["prev_best"] = warm("prev best (fp16)", first)
            undo_gemv = kernel_int8.install(qweights())
            table["new"] = warm("new (int8 weights)", build_graph())
            undo_gemv()
            table["prev_best_again"] = warm("prev best (again)", build_graph())
            undo_att()
            undo_merge()
            undo_norm()
            results["table"] = table

            # quality: fused-norm model vs float32 reference, same text for all
            m32 = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, dtype=torch.float32, attn_implementation="sdpa").to(device=dev).eval()
            texts = ["The capital of France is", "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
                     "Three tips for learning a new programming language are",
                     "Question: What is 17 multiplied by 23?\nAnswer:", "Once upon a time, in a small village by the sea,",
                     "Photosynthesis is the process by which plants",
                     "List of the planets in our solar system, in order from the Sun:\n1.",
                     "The main difference between a CPU and a GPU is that",
                     "Translate to French: 'Good morning, how are you?'\nFrench:"]
            items = [(t, tok(t, return_tensors="pt").input_ids.to(dev)) for t in texts] + \
                    [("filler 1024 tokens", make_prompt_ids(tok, 1024, dev))]
            qrows = []
            for label, p_ids in items:
                P = p_ids.shape[1]
                seq = model.generate(input_ids=p_ids, attention_mask=torch.ones_like(p_ids), max_new_tokens=48,
                                     min_new_tokens=48, do_sample=False, pad_token_id=tok.eos_token_id)
                l32 = torch.log_softmax(m32(input_ids=seq, use_cache=False).logits[0, P - 1:-1].float(), -1)
                lo = torch.log_softmax(model(input_ids=seq, use_cache=False).logits[0, P - 1:-1].float(), -1)
                undo_norm, _ = fuse_rmsnorms(model)
                undo_merge = merge_projections(model)
                undo_att = fuse_attention(model, merged=True)
                undo_gemv = kernel_int8.install(qweights())               # the full "new" model
                lf = torch.log_softmax(model(input_ids=seq, use_cache=False).logits[0, P - 1:-1].float(), -1)
                seq_f = model.generate(input_ids=p_ids, attention_mask=torch.ones_like(p_ids), max_new_tokens=48,
                                       min_new_tokens=48, do_sample=False, pad_token_id=tok.eos_token_id)
                undo_gemv()
                undo_att()
                undo_merge()
                undo_norm()
                kl = lambda a, b: float((a.exp() * (a - b)).sum(-1).mean())
                top = lambda a, b: float((a.argmax(-1) == b.argmax(-1)).float().mean())
                row = {"prompt": label[:30], "orig_KL_vs_fp32": kl(l32, lo), "fused_KL_vs_fp32": kl(l32, lf),
                       "orig_top1": top(l32, lo), "fused_top1": top(l32, lf),
                       "text_first_diff": first_diff(seq[0, P:].tolist(), seq_f[0, P:].tolist())}
                qrows.append(row)
                log(f"M quality {label[:22]!r:<26} KL vs fp32: orig {row['orig_KL_vs_fp32']:.2e} fused "
                    f"{row['fused_KL_vs_fp32']:.2e} | top-1: orig {row['orig_top1']:.3f} fused {row['fused_top1']:.3f} "
                    f"| text first diff {row['text_first_diff']}")
            results["quality"] = qrows
            base = (table["prev_best"][0] + table["prev_best_again"][0]) / 2
            mean = lambda k: statistics.mean(r[k] for r in qrows)
            same = sum(r["text_first_diff"] is None for r in qrows)
            results.update(mean_orig_KL=mean("orig_KL_vs_fp32"), mean_int8_KL=mean("fused_KL_vs_fp32"),
                           mean_orig_top1=mean("orig_top1"), mean_int8_top1=mean("fused_top1"), same_text=same)
            log(f"RESULT M: fp16 {table['prev_best'][0]}/{table['prev_best_again'][0]} -> int8 {table['new'][0]} ms/step "
                f"({1000 / table['new'][0]:.0f} tok/s), saved {base - table['new'][0]:.2f} ms | quality vs fp32 over "
                f"{len(qrows)} prompts: mean KL fp16 {mean('orig_KL_vs_fp32'):.2e} / int8 {mean('fused_KL_vs_fp32'):.2e}, "
                f"mean top-1 fp16 {mean('orig_top1'):.3f} / int8 {mean('fused_top1'):.3f}, identical text {same}/{len(qrows)}")

        elif stage == "N":
            # E14 (pre-registered, docs/PREREG_E14.md): group-wise int8 (one scale per 32 weights) vs per-row int8.
            # Speed: fp16 best -> int8 per-row -> int8 g32 -> fp16 best (warm). Quality vs fp32 for both on 10 prompts.
            import kernel_int8

            def qweights():
                ws = []
                for layer in model.model.layers:
                    ws += [layer.self_attn._w_qkv, layer.self_attn.o_proj.weight,
                           layer.mlp._w_gu, layer.mlp.down_proj.weight]
                return ws + [model.lm_head.weight]
            keep = []

            def build_graph():
                nonlocal cache
                cache = new_cache()
                st = torch.zeros((1, 1), dtype=torch.long, device=dev)
                sp = torch.zeros((1,), dtype=torch.long, device=dev)
                idx = torch.zeros((1,), dtype=torch.long, device=dev)
                buf = torch.zeros((NEW,), dtype=torch.long, device=dev)
                t0 = prefill()

                def full():
                    lg = model(input_ids=st, past_key_values=cache, cache_position=sp, use_cache=True).logits
                    nxt = lg[:, -1].argmax(-1)
                    buf.index_copy_(0, idx, nxt)
                    st.copy_(nxt.view(1, 1)); sp.add_(1); idx.add_(1)
                    return lg

                def arm():
                    reset_counters(N_PROMPT)
                    st.copy_(t0.view(1, 1)); sp.fill_(N_PROMPT); idx.fill_(1); buf.zero_(); buf[0] = t0[0]

                arm()
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(3):
                        full()
                torch.cuda.current_stream().wait_stream(s)
                sync()
                arm()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    full()
                keep.append(g)

                def run():
                    arm()
                    sync(); t = time.perf_counter()
                    for _ in range(NEW - 1):
                        g.replay()
                    sync()
                    return (time.perf_counter() - t) * 1000 / (NEW - 1), buf.tolist()
                return run

            def warm(name, run):
                run(); run()
                ms_all, toks = [], None
                for _ in range(5):
                    ms, toks = run()
                    ms_all.append(ms)
                ms = statistics.median(ms_all)
                log(f"N {name:<24} warm {ms:6.2f} ms/step ({1000 / ms:6.1f} tok/s) | tokens first diff vs generate: "
                    f"{first_diff(toks, ref)}")
                return round(ms, 2), first_diff(toks, ref)

            G = 32
            # kernel self-check first (rule 8): grouped kernel vs exact float32 result of the same quantized weights
            gen_ = torch.Generator(device=dev).manual_seed(7)
            for W in [model.model.layers[0].self_attn.o_proj.weight, model.model.layers[0].mlp.down_proj.weight,
                      model.lm_head.weight]:
                q_, sc_ = kernel_int8.quantize_group(W, G)
                x_ = torch.randn(1, 1, W.shape[1], device=dev, generator=gen_).half()
                ref_ = kernel_int8.dequant(q_, sc_, G).float() @ x_.float().reshape(-1)
                got_ = kernel_int8.gemv_i8g(x_, q_, sc_, G).float().reshape(-1)
                err_ = float((got_ - ref_).abs().max())
                ok_ = err_ <= 4e-3 * float(ref_.abs().max()) + 1e-3
                log(f"N self-check {tuple(W.shape)}: max err {err_:.2e} -> {'CORRECT' if ok_ else 'WRONG'}")
                if not ok_:
                    log("RESULT N: grouped kernel WRONG - stopping")
                    sys.stdout.flush(); os._exit(4)

            table = {}
            undo_norm, n_norms = fuse_rmsnorms(model)
            undo_merge = merge_projections(model)
            undo_att = fuse_attention(model, merged=True)
            first = build_graph()
            for _ in range(5):
                first()
            table["fp16"] = warm("fp16 best", first)
            u = kernel_int8.install(qweights(), group=0)
            table["int8_row"] = warm("int8 per-row", build_graph())
            u()
            u = kernel_int8.install(qweights(), group=G)
            table["int8_g32"] = warm("int8 group-32", build_graph())
            u()
            table["fp16_again"] = warm("fp16 best (again)", build_graph())
            undo_att(); undo_merge(); undo_norm()
            results["table"] = table

            if args.skip_quality:
                log(f"RESULT N: fp16 {table['fp16'][0]}/{table['fp16_again'][0]} | int8 per-row {table['int8_row'][0]} | "
                    f"int8 g32 {table['int8_g32'][0]} ms/step || quality skipped (--skip-quality)")
            else:
                m32 = AutoModelForCausalLM.from_pretrained(
                    MODEL_ID, dtype=torch.float32, attn_implementation="sdpa").to(device=dev).eval()
                texts = ["The capital of France is", "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
                         "Three tips for learning a new programming language are",
                         "Question: What is 17 multiplied by 23?\nAnswer:", "Once upon a time, in a small village by the sea,",
                         "Photosynthesis is the process by which plants",
                         "List of the planets in our solar system, in order from the Sun:\n1.",
                         "The main difference between a CPU and a GPU is that",
                         "Translate to French: 'Good morning, how are you?'\nFrench:"]
                items = [(t, tok(t, return_tensors="pt").input_ids.to(dev)) for t in texts] + \
                        [("filler 1024 tokens", make_prompt_ids(tok, 1024, dev))]
                kl = lambda a, b: float((a.exp() * (a - b)).sum(-1).mean())
                top = lambda a, b: float((a.argmax(-1) == b.argmax(-1)).float().mean())
                qrows = []
                for label, p_ids in items:
                    P = p_ids.shape[1]
                    genk = dict(attention_mask=torch.ones_like(p_ids), max_new_tokens=48, min_new_tokens=48,
                                do_sample=False, pad_token_id=tok.eos_token_id)
                    seq = model.generate(input_ids=p_ids, **genk)
                    l32 = torch.log_softmax(m32(input_ids=seq, use_cache=False).logits[0, P - 1:-1].float(), -1)
                    row = {"prompt": label[:30]}
                    for tag, grp in (("row", 0), ("g32", G)):
                        un = fuse_rmsnorms(model)[0]; um = merge_projections(model); ua = fuse_attention(model, merged=True)
                        uq = kernel_int8.install(qweights(), group=grp)
                        lq = torch.log_softmax(model(input_ids=seq, use_cache=False).logits[0, P - 1:-1].float(), -1)
                        sq = model.generate(input_ids=p_ids, **genk)
                        uq(); ua(); um(); un()
                        row[f"{tag}_KL"] = kl(l32, lq)
                        row[f"{tag}_top1"] = top(l32, lq)
                        row[f"{tag}_text_first_diff"] = first_diff(seq[0, P:].tolist(), sq[0, P:].tolist())
                    qrows.append(row)
                    log(f"N quality {label[:22]!r:<26} | per-row: KL {row['row_KL']:.2e} top-1 {row['row_top1']:.3f} text "
                        f"diff {row['row_text_first_diff']} | group-32: KL {row['g32_KL']:.2e} top-1 {row['g32_top1']:.3f} "
                        f"text diff {row['g32_text_first_diff']}")
                results["quality"] = qrows
                mean = lambda k: statistics.mean(r[k] for r in qrows)
                same = lambda tag: sum(r[f"{tag}_text_first_diff"] is None for r in qrows)
                results.update(mean_row_KL=mean("row_KL"), mean_g32_KL=mean("g32_KL"), mean_row_top1=mean("row_top1"),
                               mean_g32_top1=mean("g32_top1"), same_text_row=same("row"), same_text_g32=same("g32"))
                log(f"RESULT N: fp16 {table['fp16'][0]}/{table['fp16_again'][0]} | int8 per-row {table['int8_row'][0]} | "
                    f"int8 g32 {table['int8_g32'][0]} ms/step || quality over {len(qrows)} prompts: per-row KL "
                    f"{mean('row_KL'):.2e}, top-1 {mean('row_top1'):.3f}, same text {same('row')}/{len(qrows)} | group-32 KL "
                    f"{mean('g32_KL'):.2e}, top-1 {mean('g32_top1'):.3f}, same text {same('g32')}/{len(qrows)}")

        elif stage == "O":
            # S3 (pre-registered, docs/PREREG_S3.md): ablation timing of one decode step, fp16 best stack.
            # Switch off attention (SDPA) and/or all matrix multiplies (F.linear returns an empty tensor of the right
            # shape) to split the step into matrices / attention / small ops. Output tokens are garbage when a part is
            # off; only timing matters here.
            import torch.nn.functional as F
            from transformers.models.qwen3 import modeling_qwen3 as mq
            keep = []

            def build_graph():
                nonlocal cache
                cache = new_cache()
                st = torch.zeros((1, 1), dtype=torch.long, device=dev)
                sp = torch.zeros((1,), dtype=torch.long, device=dev)
                idx = torch.zeros((1,), dtype=torch.long, device=dev)
                buf = torch.zeros((NEW,), dtype=torch.long, device=dev)
                t0 = prefill()

                def full():
                    lg = model(input_ids=st, past_key_values=cache, cache_position=sp, use_cache=True).logits
                    nxt = lg[:, -1].argmax(-1)
                    buf.index_copy_(0, idx, nxt)
                    st.copy_(nxt.view(1, 1)); sp.add_(1); idx.add_(1)
                    return lg

                def arm():
                    reset_counters(N_PROMPT)
                    st.copy_(t0.view(1, 1)); sp.fill_(N_PROMPT); idx.fill_(1); buf.zero_(); buf[0] = t0[0]

                arm()
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(3):
                        full()
                torch.cuda.current_stream().wait_stream(s)
                sync()
                arm()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    full()
                keep.append(g)

                def run():
                    arm()
                    sync(); t = time.perf_counter()
                    for _ in range(NEW - 1):
                        g.replay()
                    sync()
                    return (time.perf_counter() - t) * 1000 / (NEW - 1), buf.tolist()
                return run

            def timed(name, run):
                run(); run()
                ms = statistics.median([run()[0] for _ in range(5)])
                log(f"O {name:<28} {ms:6.2f} ms/step")
                return round(ms, 3)

            undo_norm, _ = fuse_rmsnorms(model)
            undo_merge = merge_projections(model)
            undo_att = fuse_attention(model, merged=True)
            orig_get = mq.ALL_ATTENTION_FUNCTIONS.get_interface
            orig_lin = F.linear

            def fake_att(module, q, k, v, mask, **kw):
                return q.transpose(1, 2), None  # same output shape as SDPA, no KV-cache read

            def fake_lin(inp, w, b=None):
                return inp.new_empty(*inp.shape[:-1], w.shape[0])

            def set_mode(att_off, mm_off):
                mq.ALL_ATTENTION_FUNCTIONS.get_interface = (lambda *a, **k: fake_att) if att_off else orig_get
                F.linear = fake_lin if mm_off else orig_lin

            t = {}
            for name, a_off, m_off in [("full", 0, 0), ("no attention", 1, 0), ("no matrices", 0, 1),
                                       ("no attention, no matrices", 1, 1), ("full (again)", 0, 0)]:
                set_mode(a_off, m_off)
                t[name] = timed(name, build_graph())
            set_mode(0, 0)
            undo_att(); undo_merge(); undo_norm()
            full = (t["full"] + t["full (again)"]) / 2
            parts = {"full_ms": round(full, 3), "attention_ms": round(full - t["no attention"], 3),
                     "matrices_ms": round(full - t["no matrices"], 3), "small_ops_ms": t["no attention, no matrices"]}
            parts["interaction_ms"] = round(full - parts["attention_ms"] - parts["matrices_ms"] - parts["small_ops_ms"], 3)
            results["ablation"] = t
            results["parts"] = parts
            log(f"RESULT O: full {full:.2f} = matrices {parts['matrices_ms']:.2f} + attention {parts['attention_ms']:.2f} + "
                f"small ops {parts['small_ops_ms']:.2f} + interaction {parts['interaction_ms']:+.2f} ms")

        elif stage == "P":
            # S4b (pre-registered, docs/PREREG_S4.md): split-KV decode attention kernel inside the real model.
            # Order: fp16 SDPA -> fp16 ours -> int8-g32 SDPA -> int8-g32 ours -> fp16 SDPA (again). Prompts (q_len > 1)
            # still use SDPA; only single-token decode uses the new kernel.
            import kernel_int8
            import kernel_decode_attn as kda
            from transformers.models.qwen3 import modeling_qwen3 as mq
            kda.get_kernels()
            keep = []

            def qweights():
                ws = []
                for layer in model.model.layers:
                    ws += [layer.self_attn._w_qkv, layer.self_attn.o_proj.weight,
                           layer.mlp._w_gu, layer.mlp.down_proj.weight]
                return ws + [model.lm_head.weight]

            def build_graph():
                nonlocal cache
                cache = new_cache()
                st = torch.zeros((1, 1), dtype=torch.long, device=dev)
                sp = torch.zeros((1,), dtype=torch.long, device=dev)
                idx = torch.zeros((1,), dtype=torch.long, device=dev)
                buf = torch.zeros((NEW,), dtype=torch.long, device=dev)
                t0 = prefill()
                for i_, layer_ in enumerate(model.model.layers):   # S4: kernel reads the cache's own length counter
                    layer_.self_attn._len = cache.layers[i_].cumulative_length

                def full():
                    lg = model(input_ids=st, past_key_values=cache, cache_position=sp, use_cache=True).logits
                    nxt = lg[:, -1].argmax(-1)
                    buf.index_copy_(0, idx, nxt)
                    st.copy_(nxt.view(1, 1)); sp.add_(1); idx.add_(1)
                    return lg

                def arm():
                    reset_counters(N_PROMPT)
                    st.copy_(t0.view(1, 1)); sp.fill_(N_PROMPT); idx.fill_(1); buf.zero_(); buf[0] = t0[0]

                arm()
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(3):
                        full()
                torch.cuda.current_stream().wait_stream(s)
                sync()
                arm()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    full()
                keep.append(g)

                def run():
                    arm()
                    sync(); t = time.perf_counter()
                    for _ in range(NEW - 1):
                        g.replay()
                    sync()
                    return (time.perf_counter() - t) * 1000 / (NEW - 1), buf.tolist()
                return run

            def warm(name, run):
                run(); run()
                ms_all, toks = [], None
                for _ in range(5):
                    ms, toks = run()
                    ms_all.append(ms)
                ms = statistics.median(ms_all)
                log(f"P {name:<30} warm {ms:6.2f} ms/step ({1000 / ms:6.1f} tok/s) | tokens first diff vs generate: "
                    f"{first_diff(toks, ref)}")
                return round(ms, 2), first_diff(toks, ref)

            orig_get = mq.ALL_ATTENTION_FUNCTIONS.get_interface
            sdpa_fn = orig_get("sdpa", mq.eager_attention_forward)

            def our_att(module, q, k, v, mask, dropout=0.0, scaling=None, sliding_window=None, **kw):
                if q.shape[2] != 1:
                    return sdpa_fn(module, q, k, v, mask, dropout=dropout, scaling=scaling,
                                   sliding_window=sliding_window, **kw)
                return kda.decode_attn(q, k, v, module._len, scaling), None

            def use_ours(on):
                mq.ALL_ATTENTION_FUNCTIONS.get_interface = (lambda *a, **k: our_att) if on else orig_get

            undo_norm, _ = fuse_rmsnorms(model)
            undo_merge = merge_projections(model)
            undo_att = fuse_attention(model, merged=True)
            table = {}
            first = build_graph()
            for _ in range(5):
                first()
            table["fp16_sdpa"] = warm("fp16 SDPA", first)
            use_ours(True)
            for S_ in (16, 32, 64, 128):        # S4c: split count chosen on fp16 only, then reused for int8
                kda.SPLITS = S_
                table[f"fp16_ours_S{S_}"] = warm(f"fp16 + split-KV S={S_}", build_graph())
            best_S = min((16, 32, 64, 128), key=lambda S_: table[f"fp16_ours_S{S_}"][0])
            kda.SPLITS = best_S
            table["fp16_ours"] = table[f"fp16_ours_S{best_S}"]
            results["best_splits"] = best_S
            use_ours(False)
            u = kernel_int8.install(qweights(), group=32)
            table["int8_sdpa"] = warm("int8-g32 SDPA", build_graph())
            use_ours(True)
            table["int8_ours"] = warm(f"int8-g32 + split-KV S={best_S}", build_graph())
            use_ours(False)
            u()
            table["fp16_sdpa_again"] = warm("fp16 SDPA (again)", build_graph())
            undo_att(); undo_merge(); undo_norm()
            results["table"] = table
            log(f"RESULT P: best splits {best_S} | fp16 {table['fp16_sdpa'][0]}/{table['fp16_sdpa_again'][0]} -> {table['fp16_ours'][0]} ms | "
                f"int8-g32 {table['int8_sdpa'][0]} -> {table['int8_ours'][0]} ms | token first diff vs generate: "
                f"fp16 ours {table['fp16_ours'][1]}, int8 ours {table['int8_ours'][1]}")

        elif stage == "Q":
            # S4b (pre-registered, docs/PREREG_S4.md): split-KV decode attention kernel inside the real model.
            # Order: fp16 SDPA -> fp16 ours -> int8-g32 SDPA -> int8-g32 ours -> fp16 SDPA (again). Prompts (q_len > 1)
            # still use SDPA; only single-token decode uses the new kernel.
            import kernel_int8
            import kernel_decode_attn as kda
            from transformers.models.qwen3 import modeling_qwen3 as mq
            kda.get_kernels()
            keep = []

            def qweights():
                ws = []
                for layer in model.model.layers:
                    ws += [layer.self_attn._w_qkv, layer.self_attn.o_proj.weight,
                           layer.mlp._w_gu, layer.mlp.down_proj.weight]
                return ws + [model.lm_head.weight]

            def build_graph():
                nonlocal cache
                cache = new_cache()
                st = torch.zeros((1, 1), dtype=torch.long, device=dev)
                sp = torch.zeros((1,), dtype=torch.long, device=dev)
                idx = torch.zeros((1,), dtype=torch.long, device=dev)
                buf = torch.zeros((NEW,), dtype=torch.long, device=dev)
                t0 = prefill()
                for i_, layer_ in enumerate(model.model.layers):   # S4: kernel reads the cache's own length counter
                    layer_.self_attn._len = cache.layers[i_].cumulative_length

                def full():
                    lg = model(input_ids=st, past_key_values=cache, cache_position=sp, use_cache=True).logits
                    nxt = lg[:, -1].argmax(-1)
                    buf.index_copy_(0, idx, nxt)
                    st.copy_(nxt.view(1, 1)); sp.add_(1); idx.add_(1)
                    return lg

                def arm():
                    reset_counters(N_PROMPT)
                    st.copy_(t0.view(1, 1)); sp.fill_(N_PROMPT); idx.fill_(1); buf.zero_(); buf[0] = t0[0]

                arm()
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(3):
                        full()
                torch.cuda.current_stream().wait_stream(s)
                sync()
                arm()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    full()
                keep.append(g)

                def run():
                    arm()
                    sync(); t = time.perf_counter()
                    for _ in range(NEW - 1):
                        g.replay()
                    sync()
                    return (time.perf_counter() - t) * 1000 / (NEW - 1), buf.tolist()
                return run

            def warm(name, run):
                run(); run()
                ms_all, toks = [], None
                for _ in range(5):
                    ms, toks = run()
                    ms_all.append(ms)
                ms = statistics.median(ms_all)
                log(f"Q {name:<34} warm {ms:6.2f} ms/step ({1000 / ms:6.1f} tok/s) | tokens first diff vs generate: "
                    f"{first_diff(toks, ref)}")
                return round(ms, 2), first_diff(toks, ref)

            orig_get = mq.ALL_ATTENTION_FUNCTIONS.get_interface
            sdpa_fn = orig_get("sdpa", mq.eager_attention_forward)

            def our_att(module, q, k, v, mask, dropout=0.0, scaling=None, sliding_window=None, **kw):
                if q.shape[2] != 1:
                    return sdpa_fn(module, q, k, v, mask, dropout=dropout, scaling=scaling,
                                   sliding_window=sliding_window, **kw)
                return kda.decode_attn(q, k, v, module._len, scaling), None

            def use_ours(on):
                mq.ALL_ATTENTION_FUNCTIONS.get_interface = (lambda *a, **k: our_att) if on else orig_get

            import fuse_small
            kda.SPLITS = 32
            undo_norm, _ = fuse_rmsnorms(model)
            undo_merge = merge_projections(model)
            undo_att = fuse_attention(model, merged=True)
            use_ours(True)                       # S4 split-KV attention is the baseline from here on
            table = {}
            first = build_graph()
            for _ in range(5):
                first()
            table["fp16_base"] = warm("fp16 best (S4)", first)
            for tag, kw in [("fp16_cache", dict(cache_write=True, res_norm=False, swiglu=False)),
                            ("fp16_cache_norm", dict(cache_write=True, res_norm=True, swiglu=False)),
                            ("fp16_all", dict(cache_write=True, res_norm=True, swiglu=True))]:
                uf = fuse_small.install(model, **kw)
                table[tag] = warm("fp16 + " + tag[5:].replace("_", "+"), build_graph())
                uf()
            u = kernel_int8.install(qweights(), group=32)
            table["int8_base"] = warm("int8-g32 best (S4)", build_graph())
            uf = fuse_small.install(model)
            table["int8_all"] = warm("int8-g32 + all fusions", build_graph())
            uf(); u()
            table["fp16_base_again"] = warm("fp16 best (S4, again)", build_graph())
            use_ours(False)
            undo_att(); undo_merge(); undo_norm()
            results["table"] = table
            b = (table["fp16_base"][0] + table["fp16_base_again"][0]) / 2
            log(f"RESULT Q: fp16 {b:.2f} -> cache {table['fp16_cache'][0]} -> +res-norm {table['fp16_cache_norm'][0]} -> "
                f"+swiglu {table['fp16_all'][0]} ms | int8-g32 {table['int8_base'][0]} -> {table['int8_all'][0]} ms | "
                f"token first diff: fp16 all {table['fp16_all'][1]}, int8 all {table['int8_all'][1]}")

        elif stage == "R":
            # S7 (pre-registered, docs/PREREG_S7.md): profiler trace of the current best stack (int8-g32 + split-KV
            # attention + small-op fusions), EAGER decode (no graph) so every kernel gets its own GPU timestamps.
            # Kernel durations do not depend on launch gaps, so their sum per token is comparable with the graph step.
            import collections
            import kernel_int8
            import kernel_decode_attn as kda
            import fuse_small
            from transformers.models.qwen3 import modeling_qwen3 as mq
            from torch.profiler import profile, ProfilerActivity
            from torch.autograd import DeviceType
            kda.get_kernels()
            kda.SPLITS = 32

            def qweights():
                ws = []
                for layer in model.model.layers:
                    ws += [layer.self_attn._w_qkv, layer.self_attn.o_proj.weight,
                           layer.mlp._w_gu, layer.mlp.down_proj.weight]
                return ws + [model.lm_head.weight]

            orig_get = mq.ALL_ATTENTION_FUNCTIONS.get_interface
            sdpa_fn = orig_get("sdpa", mq.eager_attention_forward)

            def our_att(module, q, k, v, mask, dropout=0.0, scaling=None, sliding_window=None, **kw):
                if q.shape[2] != 1:
                    return sdpa_fn(module, q, k, v, mask, dropout=dropout, scaling=scaling,
                                   sliding_window=sliding_window, **kw)
                return kda.decode_attn(q, k, v, module._len, scaling), None

            def use_ours(on):
                mq.ALL_ATTENTION_FUNCTIONS.get_interface = (lambda *a, **k: our_att) if on else orig_get

            undo_norm, _ = fuse_rmsnorms(model)
            undo_merge = merge_projections(model)
            undo_att = fuse_attention(model, merged=True)
            use_ours(True)
            uq = kernel_int8.install(qweights(), group=32)
            uf = fuse_small.install(model)
            cache = new_cache()
            t0 = prefill()
            for i_, layer_ in enumerate(model.model.layers):
                layer_.self_attn._len = cache.layers[i_].cumulative_length
            cur, p_ = t0.view(1, 1), N_PROMPT

            def one(cur, p_):
                lg = model(input_ids=cur, past_key_values=cache, cache_position=torch.tensor([p_], device=dev),
                           use_cache=True).logits
                return lg[:, -1].argmax(-1).view(1, 1)

            for _ in range(10):
                cur = one(cur, p_); p_ += 1
            sync()
            STEPS = 20
            t_wall = time.perf_counter()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                for _ in range(STEPS):
                    cur = one(cur, p_); p_ += 1
                sync()
            t_wall = (time.perf_counter() - t_wall) * 1000 / STEPS
            kern = [e for e in prof.events() if e.device_type == DeviceType.CUDA]
            log(f"R profiler: {len(prof.events())} events, {len(kern)} GPU kernel events over {STEPS} steps "
                f"| eager wall time {t_wall:.2f} ms/step (includes profiler overhead)")
            if not kern:
                log("RESULT R: profiler recorded NO GPU kernels on this platform (no per-kernel timestamps available)")
            else:
                by = collections.defaultdict(lambda: [0.0, 0])
                for e in kern:
                    by[e.name][0] += e.time_range.elapsed_us()
                    by[e.name][1] += 1
                total = sum(v[0] for v in by.values()) / STEPS / 1000
                cats = collections.defaultdict(float)
                rules = [("matrices (int8 gemv)", ("gemv_i8",)), ("attention (split-KV)", ("attn_partial", "attn_reduce")),
                         ("add+RMSNorm / RMSNorm", ("add_rmsnorm", "rmsnorm_fused")), ("q/k-norm+RoPE", ("qk_norm_rope",)),
                         ("KV-cache write", ("kv_write",)), ("SiLU x up", ("silu_mul",))]
                for name, (us, n) in by.items():
                    cat = next((c for c, keys in rules if any(k in name for k in keys)), "other (PyTorch ops)")
                    cats[cat] += us / STEPS / 1000
                log(f"R GPU kernel time per token: {total:.3f} ms ({len(kern) / STEPS:.0f} kernels/token)")
                for c, ms in sorted(cats.items(), key=lambda kv: -kv[1]):
                    log(f"R   {c:<26} {ms:6.3f} ms ({100 * ms / total:4.1f}%)")
                log("R top kernels (per token):")
                for name, (us, n) in sorted(by.items(), key=lambda kv: -kv[1][0])[:25]:
                    log(f"R   {us / STEPS:8.1f} us  x{n / STEPS:5.1f}  {name[:90]}")
                results["kernel_ms_per_token"] = total
                results["categories_ms"] = dict(cats)
                results["top"] = [(n[:120], v[0] / STEPS, v[1] / STEPS) for n, v in
                                  sorted(by.items(), key=lambda kv: -kv[1][0])[:60]]
                tag_ = MODEL_ID.split("/")[-1]
                prof.export_chrome_trace(str(HERE / "results" / f"S7_trace_{tag_}.json"))
                log(f"RESULT R: GPU kernel time {total:.2f} ms/token | " +
                    " | ".join(f"{c} {ms:.2f}" for c, ms in sorted(cats.items(), key=lambda kv: -kv[1])))
            uf(); uq(); use_ours(False)
            undo_att(); undo_merge(); undo_norm()

        elif stage == "T":
            # S8 (pre-registered, docs/PREREG_S8.md): GPU timestamp probes INSIDE the real graph (profiler substitute).
            import collections
            import torch.nn.functional as F
            import kernel_int8
            import kernel_decode_attn as kda
            import kernel_qknorm_rope as kqr
            import fuse_small
            from transformers import cache_utils
            from transformers.models.qwen3 import modeling_qwen3 as mq
            from probes import Probes
            kda.get_kernels()
            kda.SPLITS = 32
            PR = Probes(dev)
            log(f"T probe clock: {PR.src}" + (f" (wall_clock64 failed: {PR.err})" if PR.err else ""))
            RATE_MHZ = 100.0   # AMD constant wall clock; validated below against host timing

            def qweights():
                ws = []
                for layer in model.model.layers:
                    ws += [layer.self_attn._w_qkv, layer.self_attn.o_proj.weight,
                           layer.mlp._w_gu, layer.mlp.down_proj.weight]
                return ws + [model.lm_head.weight]

            orig_get = mq.ALL_ATTENTION_FUNCTIONS.get_interface
            sdpa_fn = orig_get("sdpa", mq.eager_attention_forward)

            def our_att(module, q, k, v, mask, dropout=0.0, scaling=None, sliding_window=None, **kw):
                if q.shape[2] != 1:
                    return sdpa_fn(module, q, k, v, mask, dropout=dropout, scaling=scaling,
                                   sliding_window=sliding_window, **kw)
                return kda.decode_attn(q, k, v, module._len, scaling), None

            def use_ours(on):
                mq.ALL_ATTENTION_FUNCTIONS.get_interface = (lambda *a, **k: our_att) if on else orig_get

            def capture(fn):
                st_ = torch.cuda.Stream()
                st_.wait_stream(torch.cuda.current_stream())
                PR.on = False
                with torch.cuda.stream(st_):
                    fn(); fn()
                torch.cuda.current_stream().wait_stream(st_)
                sync()
                PR.on = True
                PR.tags.clear()
                g_ = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g_):
                    fn()
                return g_, list(PR.tags)

            # 1) cost of one probe: 400 back-to-back stamps
            g0, tags0 = capture(lambda: [PR.probe("pre:x") for _ in range(400)])
            ov = []
            for _ in range(10):
                g0.replay(); sync()
                t_ = PR.buf[:len(tags0)].tolist()
                ov += [t_[j] - t_[j - 1] for j in range(1, len(t_))]
            ov_ticks = statistics.median(ov)
            log(f"T probe overhead: {ov_ticks / RATE_MHZ:.2f} us per probe (median of {len(ov)})")
            del g0
            report = {}
            orig_qk, orig_da = kqr.qk_norm_rope, kda.decode_attn
            orig_arn, orig_sm = fuse_small.add_rmsnorm, fuse_small.silu_mul
            for prec in ("fp16", "int8"):
                kqr.qk_norm_rope = PR.wrap(orig_qk, "qk_norm_rope")       # must precede fuse_attention
                undo_norm, _ = fuse_rmsnorms(model)
                undo_merge = merge_projections(model)
                undo_att = fuse_attention(model, merged=True)
                use_ours(True)
                kda.decode_attn = PR.wrap(orig_da, "attention")
                uq = kernel_int8.install(qweights(), group=32) if prec == "int8" else None
                uf = fuse_small.install(model)
                SL = cache_utils.StaticLayer
                SL.update = PR.wrap(SL.update, "kv_write")
                fuse_small.add_rmsnorm = PR.wrap(orig_arn, "add_rmsnorm")
                fuse_small.silu_mul = PR.wrap(orig_sm, "silu_mul")
                saved_lin = F.linear
                F.linear = PR.wrap(saved_lin, lambda inp, w, b=None: f"matmul {w.shape[0]}x{w.shape[1]}")

                PR.on = False
                cache = new_cache()
                t0 = prefill()
                for i_, layer_ in enumerate(model.model.layers):
                    layer_.self_attn._len = cache.layers[i_].cumulative_length
                st = torch.zeros((1, 1), dtype=torch.long, device=dev)
                sp = torch.zeros((1,), dtype=torch.long, device=dev)

                def full():
                    PR.probe("pre:STEP")
                    lg = model(input_ids=st, past_key_values=cache, cache_position=sp, use_cache=True).logits
                    nxt = lg[:, -1].argmax(-1)
                    st.copy_(nxt.view(1, 1)); sp.add_(1)
                    PR.probe("pre:END")

                def arm():
                    reset_counters(N_PROMPT)
                    st.copy_(t0.view(1, 1)); sp.fill_(N_PROMPT)

                arm()
                g, tags = capture(full)
                PR.on = False
                n = len(tags)
                arm(); sync()
                th = time.perf_counter()
                for _ in range(20):
                    g.replay()
                sync()
                host_ms = (time.perf_counter() - th) * 1000 / 20
                arm(); sync()
                per_label = collections.defaultdict(list)
                totals = []
                for _ in range(20):
                    g.replay(); sync()
                    t_ = PR.buf[:n].tolist()
                    totals.append(t_[-1] - t_[0])
                    acc = collections.defaultdict(lambda: [0, 0])
                    for lab, dt in PR.intervals(t_):
                        acc[lab][0] += dt
                        acc[lab][1] += 1
                    for lab, (dt, c) in acc.items():
                        per_label[lab].append((dt - c * ov_ticks) / RATE_MHZ / 1000)   # ms, probe cost removed
                counts = collections.Counter(lab for lab, _ in PR.intervals(PR.buf[:n].tolist()))
                med = {lab: statistics.median(v) for lab, v in per_label.items()}
                cats = collections.defaultdict(float)
                for lab, ms in med.items():
                    cat = "matrices" if lab.startswith("matmul") else ("gaps (unwrapped ops)" if lab.startswith("gap")
                                                                       else lab)
                    cats[cat] += ms
                tot_ms = statistics.median(totals) / RATE_MHZ / 1000
                corrected = tot_ms - (n - 1) * ov_ticks / RATE_MHZ / 1000
                log(f"T {prec}: {n} probes | stamped graph host {host_ms:.3f} ms/token vs probe clock {tot_ms:.3f} ms "
                    f"(implied clock {RATE_MHZ * tot_ms / host_ms:.1f} MHz) | probe-cost-corrected step {corrected:.3f} ms")
                for c, ms in sorted(cats.items(), key=lambda kv: -kv[1]):
                    log(f"T {prec}   {c:<26} {ms:7.3f} ms ({100 * ms / max(corrected, 1e-9):4.1f}%)")
                log(f"T {prec} largest intervals (ms per token, count):")
                for lab, ms in sorted(med.items(), key=lambda kv: -kv[1])[:16]:
                    log(f"T {prec}   {ms:7.3f}  x{counts[lab]:3d}  {lab}")
                report[prec] = {"host_ms": host_ms, "probe_clock_ms": tot_ms, "corrected_ms": corrected,
                                "categories": dict(cats), "intervals": med, "counts": dict(counts),
                                "probe_overhead_us": ov_ticks / RATE_MHZ}
                del g
                F.linear = saved_lin
                fuse_small.add_rmsnorm, fuse_small.silu_mul = orig_arn, orig_sm
                kda.decode_attn = orig_da
                uf()
                if uq:
                    uq()
                use_ours(False)
                undo_att(); undo_merge(); undo_norm()
                kqr.qk_norm_rope = orig_qk
            results["probes"] = report
            r16, r8 = report["fp16"], report["int8"]
            log(f"RESULT T: probe overhead {r16['probe_overhead_us']:.2f} us | corrected step fp16 {r16['corrected_ms']:.2f} "
                f"/ int8 {r8['corrected_ms']:.2f} ms | int8: " + " | ".join(
                    f"{c} {ms:.2f}" for c, ms in sorted(r8['categories'].items(), key=lambda kv: -kv[1])))

        elif stage == "U":
            # S4b (pre-registered, docs/PREREG_S4.md): split-KV decode attention kernel inside the real model.
            # Order: fp16 SDPA -> fp16 ours -> int8-g32 SDPA -> int8-g32 ours -> fp16 SDPA (again). Prompts (q_len > 1)
            # still use SDPA; only single-token decode uses the new kernel.
            import kernel_int8
            import kernel_decode_attn as kda
            from transformers.models.qwen3 import modeling_qwen3 as mq
            kda.get_kernels()
            keep = []
            TOK = 56                  # tokens timed per run (divisible by 1, 2, 4, 8)

            def qweights():
                ws = []
                for layer in model.model.layers:
                    ws += [layer.self_attn._w_qkv, layer.self_attn.o_proj.weight,
                           layer.mlp._w_gu, layer.mlp.down_proj.weight]
                return ws + [model.lm_head.weight]

            def build_graph(KK=1):
                nonlocal cache
                cache = new_cache()
                st = torch.zeros((1, 1), dtype=torch.long, device=dev)
                sp = torch.zeros((1,), dtype=torch.long, device=dev)
                idx = torch.zeros((1,), dtype=torch.long, device=dev)
                buf = torch.zeros((NEW,), dtype=torch.long, device=dev)
                t0 = prefill()
                for i_, layer_ in enumerate(model.model.layers):   # S4: kernel reads the cache's own length counter
                    layer_.self_attn._len = cache.layers[i_].cumulative_length

                def full():
                    lg = model(input_ids=st, past_key_values=cache, cache_position=sp, use_cache=True).logits
                    nxt = lg[:, -1].argmax(-1)
                    buf.index_copy_(0, idx, nxt)
                    st.copy_(nxt.view(1, 1)); sp.add_(1); idx.add_(1)
                    return lg

                def arm():
                    reset_counters(N_PROMPT)
                    st.copy_(t0.view(1, 1)); sp.fill_(N_PROMPT); idx.fill_(1); buf.zero_(); buf[0] = t0[0]

                arm()
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(3):
                        full()
                torch.cuda.current_stream().wait_stream(s)
                sync()
                arm()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    for _k in range(KK):          # S9: KK decode steps per graph replay
                        full()
                keep.append(g)

                def run():
                    arm()
                    sync(); t = time.perf_counter()
                    for _ in range(TOK // KK):
                        g.replay()
                    sync()
                    return (time.perf_counter() - t) * 1000 / TOK, buf.tolist()[:TOK + 1]
                return run

            def warm(name, run):
                run(); run()
                ms_all, toks = [], None
                for _ in range(5):
                    ms, toks = run()
                    ms_all.append(ms)
                ms = statistics.median(ms_all)
                log(f"U {name:<30} warm {ms:6.2f} ms/step ({1000 / ms:6.1f} tok/s) | tokens first diff vs generate: "
                    f"{first_diff(toks, ref[:TOK + 1])}")
                return round(ms, 2), first_diff(toks, ref[:TOK + 1])

            orig_get = mq.ALL_ATTENTION_FUNCTIONS.get_interface
            sdpa_fn = orig_get("sdpa", mq.eager_attention_forward)

            def our_att(module, q, k, v, mask, dropout=0.0, scaling=None, sliding_window=None, **kw):
                if q.shape[2] != 1:
                    return sdpa_fn(module, q, k, v, mask, dropout=dropout, scaling=scaling,
                                   sliding_window=sliding_window, **kw)
                return kda.decode_attn(q, k, v, module._len, scaling), None

            def use_ours(on):
                mq.ALL_ATTENTION_FUNCTIONS.get_interface = (lambda *a, **k: our_att) if on else orig_get

            import fuse_small
            kda.SPLITS = 32
            undo_norm, _ = fuse_rmsnorms(model)
            undo_merge = merge_projections(model)
            undo_att = fuse_attention(model, merged=True)
            use_ours(True)
            uf = fuse_small.install(model)
            table = {}
            first = build_graph(1)
            for _ in range(5):
                first()
            for prec in ("fp16", "int8"):
                u = kernel_int8.install(qweights(), group=32) if prec == "int8" else None
                for KK in ((1, 2, 4, 8, 1) if prec == "fp16" else (1, 8)):
                    key = f"{prec}_K{KK}" + ("_again" if f"{prec}_K{KK}" in table else "")
                    table[key] = warm(f"{prec} best, {KK} token(s)/graph", build_graph(KK))
                if u:
                    u()
            uf(); use_ours(False)
            undo_att(); undo_merge(); undo_norm()
            results["table"] = table
            k1 = (table["fp16_K1"][0] + table["fp16_K1_again"][0]) / 2
            log(f"RESULT U: fp16 K=1 {k1:.2f} | K=2 {table['fp16_K2'][0]} | K=4 {table['fp16_K4'][0]} | K=8 "
                f"{table['fp16_K8'][0]} ms/token | int8 K=1 {table['int8_K1'][0]} -> K=8 {table['int8_K8'][0]} ms/token | "
                f"token first diff: fp16 K8 {table['fp16_K8'][1]}, int8 K8 {table['int8_K8'][1]}")

        elif stage == "V":
            # S4b (pre-registered, docs/PREREG_S4.md): split-KV decode attention kernel inside the real model.
            # Order: fp16 SDPA -> fp16 ours -> int8-g32 SDPA -> int8-g32 ours -> fp16 SDPA (again). Prompts (q_len > 1)
            # still use SDPA; only single-token decode uses the new kernel.
            import kernel_int8
            import kernel_decode_attn as kda
            from transformers.models.qwen3 import modeling_qwen3 as mq
            kda.get_kernels()
            keep = []
            TOK = 56                  # tokens timed per run (divisible by 1, 2, 4, 8)

            def qweights():
                ws = []
                for layer in model.model.layers:
                    ws += [layer.self_attn._w_qkv, layer.self_attn.o_proj.weight,
                           layer.mlp._w_gu, layer.mlp.down_proj.weight]
                return ws + [model.lm_head.weight]

            def build_graph(KK=1):
                nonlocal cache
                cache = new_cache()
                st = torch.zeros((1, 1), dtype=torch.long, device=dev)
                sp = torch.zeros((1,), dtype=torch.long, device=dev)
                idx = torch.zeros((1,), dtype=torch.long, device=dev)
                buf = torch.zeros((NEW,), dtype=torch.long, device=dev)
                t0 = prefill()
                for i_, layer_ in enumerate(model.model.layers):   # S4: kernel reads the cache's own length counter
                    layer_.self_attn._len = cache.layers[i_].cumulative_length

                def full():
                    lg = model(input_ids=st, past_key_values=cache, cache_position=sp, use_cache=True).logits
                    nxt = lg[:, -1].argmax(-1)
                    buf.index_copy_(0, idx, nxt)
                    st.copy_(nxt.view(1, 1)); sp.add_(1); idx.add_(1)
                    return lg

                def arm():
                    reset_counters(N_PROMPT)
                    st.copy_(t0.view(1, 1)); sp.fill_(N_PROMPT); idx.fill_(1); buf.zero_(); buf[0] = t0[0]

                arm()
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(3):
                        full()
                torch.cuda.current_stream().wait_stream(s)
                sync()
                arm()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    for _k in range(KK):          # S9: KK decode steps per graph replay
                        full()
                keep.append(g)

                def run():
                    arm()
                    sync(); t = time.perf_counter()
                    for _ in range(TOK // KK):
                        g.replay()
                    sync()
                    return (time.perf_counter() - t) * 1000 / TOK, buf.tolist()[:TOK + 1]
                return run

            def warm(name, run):
                run(); run()
                ms_all, toks = [], None
                for _ in range(5):
                    ms, toks = run()
                    ms_all.append(ms)
                ms = statistics.median(ms_all)
                log(f"V {name:<30} warm {ms:6.2f} ms/step ({1000 / ms:6.1f} tok/s) | tokens first diff vs generate: "
                    f"{first_diff(toks, ref[:TOK + 1])}")
                return round(ms, 2), first_diff(toks, ref[:TOK + 1])

            orig_get = mq.ALL_ATTENTION_FUNCTIONS.get_interface
            sdpa_fn = orig_get("sdpa", mq.eager_attention_forward)

            def our_att(module, q, k, v, mask, dropout=0.0, scaling=None, sliding_window=None, **kw):
                if q.shape[2] != 1:
                    return sdpa_fn(module, q, k, v, mask, dropout=dropout, scaling=scaling,
                                   sliding_window=sliding_window, **kw)
                return kda.decode_attn(q, k, v, module._len, scaling), None

            def use_ours(on):
                mq.ALL_ATTENTION_FUNCTIONS.get_interface = (lambda *a, **k: our_att) if on else orig_get

            import fuse_small
            import kernels_v2
            kda.SPLITS = 32
            undo_norm, _ = fuse_rmsnorms(model)
            undo_merge = merge_projections(model)
            undo_att = fuse_attention(model, merged=True)
            use_ours(True)
            uf = fuse_small.install(model)
            kernels_v2.kernels()
            table = {}
            first = build_graph(8)
            for _ in range(5):
                first()
            v1n, v1a = fuse_small.add_rmsnorm, kda.decode_attn

            def setv(norm2, attn2):
                fuse_small.add_rmsnorm = kernels_v2.add_rmsnorm if norm2 else v1n
                kda.decode_attn = kernels_v2.decode_attn if attn2 else v1a

            for key, n2, a2 in [("fp16_base", 0, 0), ("fp16_norm2", 1, 0), ("fp16_attn2", 0, 1), ("fp16_both", 1, 1),
                                ("fp16_base_again", 0, 0)]:
                setv(n2, a2)
                table[key] = warm(key.replace("_", " "), first if key == "fp16_base" else build_graph(8))
            u = kernel_int8.install(qweights(), group=32)
            for key, n2, a2 in [("int8_base", 0, 0), ("int8_both", 1, 1)]:
                setv(n2, a2)
                table[key] = warm(key.replace("_", " "), build_graph(8))
            u()
            setv(0, 0)
            uf(); use_ours(False)
            undo_att(); undo_merge(); undo_norm()
            results["table"] = table
            b = (table["fp16_base"][0] + table["fp16_base_again"][0]) / 2
            log(f"RESULT V (8 tokens/graph): fp16 base {b:.2f} | +norm v2 {table['fp16_norm2'][0]} | +attn v2 "
                f"{table['fp16_attn2'][0]} | both {table['fp16_both'][0]} ms | int8 {table['int8_base'][0]} -> "
                f"{table['int8_both'][0]} ms | token first diff: fp16 both {table['fp16_both'][1]}, int8 both "
                f"{table['int8_both'][1]}")

        elif stage == "W":
            # S4b (pre-registered, docs/PREREG_S4.md): split-KV decode attention kernel inside the real model.
            # Order: fp16 SDPA -> fp16 ours -> int8-g32 SDPA -> int8-g32 ours -> fp16 SDPA (again). Prompts (q_len > 1)
            # still use SDPA; only single-token decode uses the new kernel.
            import kernel_int8
            import kernel_decode_attn as kda
            from transformers.models.qwen3 import modeling_qwen3 as mq
            kda.get_kernels()
            keep = []
            TOK = 56                  # tokens timed per run (divisible by 1, 2, 4, 8)

            def qweights():
                ws = []
                for layer in model.model.layers:
                    ws += [layer.self_attn._w_qkv, layer.self_attn.o_proj.weight,
                           layer.mlp._w_gu, layer.mlp.down_proj.weight]
                return ws + [model.lm_head.weight]

            def build_graph(KK=1):
                nonlocal cache
                cache = new_cache()
                st = torch.zeros((1, 1), dtype=torch.long, device=dev)
                sp = torch.zeros((1,), dtype=torch.long, device=dev)
                idx = torch.zeros((1,), dtype=torch.long, device=dev)
                buf = torch.zeros((NEW,), dtype=torch.long, device=dev)
                t0 = prefill()
                for i_, layer_ in enumerate(model.model.layers):   # S4: kernel reads the cache's own length counter
                    layer_.self_attn._len = cache.layers[i_].cumulative_length

                def full():
                    lg = model(input_ids=st, past_key_values=cache, cache_position=sp, use_cache=True).logits
                    nxt = lg[:, -1].argmax(-1)
                    buf.index_copy_(0, idx, nxt)
                    st.copy_(nxt.view(1, 1)); sp.add_(1); idx.add_(1)
                    return lg

                def arm():
                    reset_counters(N_PROMPT)
                    st.copy_(t0.view(1, 1)); sp.fill_(N_PROMPT); idx.fill_(1); buf.zero_(); buf[0] = t0[0]

                arm()
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(3):
                        full()
                torch.cuda.current_stream().wait_stream(s)
                sync()
                arm()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    for _k in range(KK):          # S9: KK decode steps per graph replay
                        full()
                keep.append(g)

                def run():
                    arm()
                    sync(); t = time.perf_counter()
                    for _ in range(TOK // KK):
                        g.replay()
                    sync()
                    return (time.perf_counter() - t) * 1000 / TOK, buf.tolist()[:TOK + 1]
                return run

            def warm(name, run):
                run(); run()
                ms_all, toks = [], None
                for _ in range(5):
                    ms, toks = run()
                    ms_all.append(ms)
                ms = statistics.median(ms_all)
                log(f"W {name:<30} warm {ms:6.2f} ms/step ({1000 / ms:6.1f} tok/s) | tokens first diff vs generate: "
                    f"{first_diff(toks, ref[:TOK + 1])}")
                return round(ms, 2), first_diff(toks, ref[:TOK + 1])

            orig_get = mq.ALL_ATTENTION_FUNCTIONS.get_interface
            sdpa_fn = orig_get("sdpa", mq.eager_attention_forward)

            def our_att(module, q, k, v, mask, dropout=0.0, scaling=None, sliding_window=None, **kw):
                if q.shape[2] != 1:
                    return sdpa_fn(module, q, k, v, mask, dropout=dropout, scaling=scaling,
                                   sliding_window=sliding_window, **kw)
                return kda.decode_attn(q, k, v, module._len, scaling), None

            def use_ours(on):
                mq.ALL_ATTENTION_FUNCTIONS.get_interface = (lambda *a, **k: our_att) if on else orig_get

            # B4: best stack = fused norms + merged + fused q/k-norm-RoPE + split-KV attention v1 (S=32) + small-op
            # fusions + add+RMSNorm v2 + 8 tokens per graph. fp16 and int8-g32.
            import fuse_small
            import kernels_v2
            kda.SPLITS = 32
            undo_norm, _ = fuse_rmsnorms(model)
            undo_merge = merge_projections(model)
            undo_att = fuse_attention(model, merged=True)
            use_ours(True)
            uf = fuse_small.install(model)
            kernels_v2.kernels()
            v1n = fuse_small.add_rmsnorm
            fuse_small.add_rmsnorm = kernels_v2.add_rmsnorm
            import kernels_v3                     # B5: attention v3 (S=16) adopted after S11-S13
            kernels_v3.kernels()
            kernels_v3.SPLITS = 16
            v1a = kda.decode_attn
            if os.environ.get("ATTN", "v3") == "v3":
                kda.decode_attn = kernels_v3.decode_attn
            log(f"W attention kernel: {os.environ.get('ATTN', 'v3')}")
            table = {}
            first = build_graph(8)
            for _ in range(5):
                first()
            table["fp16_best"] = warm("fp16 best (B4)", first)
            u = kernel_int8.install(qweights(), group=32)
            table["int8_best"] = warm("int8-g32 best (B4)", build_graph(8))
            u()
            fuse_small.add_rmsnorm, kda.decode_attn = v1n, v1a
            uf(); use_ours(False)
            undo_att(); undo_merge(); undo_norm()
            results["table"] = table
            log(f"RESULT W: fp16 best {table['fp16_best'][0]} ms ({1000 / table['fp16_best'][0]:.1f} tok/s) | int8 best "
                f"{table['int8_best'][0]} ms ({1000 / table['int8_best'][0]:.1f} tok/s) | token first diff: fp16 "
                f"{table['fp16_best'][1]}, int8 {table['int8_best'][1]}")

        elif stage == "X":
            # S4b (pre-registered, docs/PREREG_S4.md): split-KV decode attention kernel inside the real model.
            # Order: fp16 SDPA -> fp16 ours -> int8-g32 SDPA -> int8-g32 ours -> fp16 SDPA (again). Prompts (q_len > 1)
            # still use SDPA; only single-token decode uses the new kernel.
            import kernel_int8
            import kernel_decode_attn as kda
            from transformers.models.qwen3 import modeling_qwen3 as mq
            kda.get_kernels()
            keep = []
            TOK = 56                  # tokens timed per run (divisible by 1, 2, 4, 8)

            def qweights():
                ws = []
                for layer in model.model.layers:
                    ws += [layer.self_attn._w_qkv, layer.self_attn.o_proj.weight,
                           layer.mlp._w_gu, layer.mlp.down_proj.weight]
                return ws + [model.lm_head.weight]

            def build_graph(KK=1):
                nonlocal cache
                cache = new_cache()
                st = torch.zeros((1, 1), dtype=torch.long, device=dev)
                sp = torch.zeros((1,), dtype=torch.long, device=dev)
                idx = torch.zeros((1,), dtype=torch.long, device=dev)
                buf = torch.zeros((NEW,), dtype=torch.long, device=dev)
                t0 = prefill()
                for i_, layer_ in enumerate(model.model.layers):   # S4: kernel reads the cache's own length counter
                    layer_.self_attn._len = cache.layers[i_].cumulative_length

                def full():
                    lg = model(input_ids=st, past_key_values=cache, cache_position=sp, use_cache=True).logits
                    nxt = lg[:, -1].argmax(-1)
                    buf.index_copy_(0, idx, nxt)
                    st.copy_(nxt.view(1, 1)); sp.add_(1); idx.add_(1)
                    return lg

                def arm():
                    reset_counters(N_PROMPT)
                    st.copy_(t0.view(1, 1)); sp.fill_(N_PROMPT); idx.fill_(1); buf.zero_(); buf[0] = t0[0]

                arm()
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(3):
                        full()
                torch.cuda.current_stream().wait_stream(s)
                sync()
                arm()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    for _k in range(KK):          # S9: KK decode steps per graph replay
                        full()
                keep.append(g)

                def run():
                    arm()
                    sync(); t = time.perf_counter()
                    for _ in range(TOK // KK):
                        g.replay()
                    sync()
                    return (time.perf_counter() - t) * 1000 / TOK, buf.tolist()[:TOK + 1]
                return run

            def warm(name, run):
                run(); run()
                ms_all, toks = [], None
                for _ in range(5):
                    ms, toks = run()
                    ms_all.append(ms)
                ms = statistics.median(ms_all)
                log(f"X {name:<30} warm {ms:6.2f} ms/step ({1000 / ms:6.1f} tok/s) | tokens first diff vs generate: "
                    f"{first_diff(toks, ref[:TOK + 1])}")
                return round(ms, 2), first_diff(toks, ref[:TOK + 1])

            orig_get = mq.ALL_ATTENTION_FUNCTIONS.get_interface
            sdpa_fn = orig_get("sdpa", mq.eager_attention_forward)

            def our_att(module, q, k, v, mask, dropout=0.0, scaling=None, sliding_window=None, **kw):
                if q.shape[2] != 1:
                    return sdpa_fn(module, q, k, v, mask, dropout=dropout, scaling=scaling,
                                   sliding_window=sliding_window, **kw)
                return kda.decode_attn(q, k, v, module._len, scaling), None

            def use_ours(on):
                mq.ALL_ATTENTION_FUNCTIONS.get_interface = (lambda *a, **k: our_att) if on else orig_get

            # S11b: attention v3 inside the B4 stack (8 tokens per graph, norm v2). Split count chosen on fp16 only.
            import fuse_small
            import kernels_v2
            import kernels_v3
            kda.SPLITS = 32
            undo_norm, _ = fuse_rmsnorms(model)
            undo_merge = merge_projections(model)
            undo_att = fuse_attention(model, merged=True)
            use_ours(True)
            uf = fuse_small.install(model)
            kernels_v2.kernels(); kernels_v3.kernels()
            v1n, v1a = fuse_small.add_rmsnorm, kda.decode_attn
            fuse_small.add_rmsnorm = kernels_v2.add_rmsnorm
            table = {}
            first = build_graph(8)
            for _ in range(5):
                first()
            table["fp16_v1"] = warm("fp16 B4 (attention v1)", first)
            kda.decode_attn = kernels_v3.decode_attn
            for S_ in (16, 32, 64):
                kernels_v3.SPLITS = S_
                table[f"fp16_v3_S{S_}"] = warm(f"fp16 attention v3 S={S_}", build_graph(8))
            best_S = min((16, 32, 64), key=lambda S_: table[f"fp16_v3_S{S_}"][0])
            kernels_v3.SPLITS = best_S
            results["best_splits_v3"] = best_S
            u = kernel_int8.install(qweights(), group=32)
            kda.decode_attn = v1a
            table["int8_v1"] = warm("int8 B4 (attention v1)", build_graph(8))
            kda.decode_attn = kernels_v3.decode_attn
            table["int8_v3"] = warm(f"int8 attention v3 S={best_S}", build_graph(8))
            u()
            kda.decode_attn = v1a
            table["fp16_v1_again"] = warm("fp16 B4 (attention v1, again)", build_graph(8))
            fuse_small.add_rmsnorm = v1n
            uf(); use_ours(False)
            undo_att(); undo_merge(); undo_norm()
            results["table"] = table
            b = (table["fp16_v1"][0] + table["fp16_v1_again"][0]) / 2
            log(f"RESULT X: best v3 splits {best_S} | fp16 v1 {b:.2f} -> v3 {table[f'fp16_v3_S{best_S}'][0]} ms | int8 v1 "
                f"{table['int8_v1'][0]} -> v3 {table['int8_v3'][0]} ms ({1000 / table['int8_v3'][0]:.1f} tok/s) | token "
                f"first diff: fp16 v3 {table[f'fp16_v3_S{best_S}'][1]}, int8 v3 {table['int8_v3'][1]}")

        elif stage == "Y":
            # S4b (pre-registered, docs/PREREG_S4.md): split-KV decode attention kernel inside the real model.
            # Order: fp16 SDPA -> fp16 ours -> int8-g32 SDPA -> int8-g32 ours -> fp16 SDPA (again). Prompts (q_len > 1)
            # still use SDPA; only single-token decode uses the new kernel.
            import kernel_int8
            import kernel_decode_attn as kda
            from transformers.models.qwen3 import modeling_qwen3 as mq
            kda.get_kernels()
            keep = []
            TOK = 56                  # tokens timed per run (divisible by 1, 2, 4, 8)

            def qweights():
                ws = []
                for layer in model.model.layers:
                    ws += [layer.self_attn._w_qkv, layer.self_attn.o_proj.weight,
                           layer.mlp._w_gu, layer.mlp.down_proj.weight]
                return ws + [model.lm_head.weight]

            def build_graph(KK=1):
                nonlocal cache
                cache = new_cache()
                st = torch.zeros((1, 1), dtype=torch.long, device=dev)
                sp = torch.zeros((1,), dtype=torch.long, device=dev)
                idx = torch.zeros((1,), dtype=torch.long, device=dev)
                buf = torch.zeros((NEW,), dtype=torch.long, device=dev)
                t0 = prefill()
                for i_, layer_ in enumerate(model.model.layers):   # S4: kernel reads the cache's own length counter
                    layer_.self_attn._len = cache.layers[i_].cumulative_length

                def full():
                    lg = model(input_ids=st, past_key_values=cache, cache_position=sp, use_cache=True).logits
                    nxt = lg[:, -1].argmax(-1)
                    buf.index_copy_(0, idx, nxt)
                    st.copy_(nxt.view(1, 1)); sp.add_(1); idx.add_(1)
                    return lg

                def arm():
                    reset_counters(N_PROMPT)
                    st.copy_(t0.view(1, 1)); sp.fill_(N_PROMPT); idx.fill_(1); buf.zero_(); buf[0] = t0[0]

                arm()
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(3):
                        full()
                torch.cuda.current_stream().wait_stream(s)
                sync()
                arm()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    for _k in range(KK):          # S9: KK decode steps per graph replay
                        full()
                keep.append(g)

                def run():
                    arm()
                    sync(); t = time.perf_counter()
                    for _ in range(TOK // KK):
                        g.replay()
                    sync()
                    return (time.perf_counter() - t) * 1000 / TOK, buf.tolist()[:TOK + 1]
                return run

            def warm(name, run):
                run(); run()
                ms_all, toks = [], None
                for _ in range(5):
                    ms, toks = run()
                    ms_all.append(ms)
                ms = statistics.median(ms_all)
                log(f"Y {name:<30} warm {ms:6.2f} ms/step ({1000 / ms:6.1f} tok/s) | tokens first diff vs generate: "
                    f"{first_diff(toks, ref[:TOK + 1])}")
                return round(ms, 2), first_diff(toks, ref[:TOK + 1])

            orig_get = mq.ALL_ATTENTION_FUNCTIONS.get_interface
            sdpa_fn = orig_get("sdpa", mq.eager_attention_forward)

            def our_att(module, q, k, v, mask, dropout=0.0, scaling=None, sliding_window=None, **kw):
                if q.shape[2] != 1:
                    return sdpa_fn(module, q, k, v, mask, dropout=dropout, scaling=scaling,
                                   sliding_window=sliding_window, **kw)
                return kda.decode_attn(q, k, v, module._len, scaling), None

            def use_ours(on):
                mq.ALL_ATTENTION_FUNCTIONS.get_interface = (lambda *a, **k: our_att) if on else orig_get

            # S12b: attention v4 vs v3 (S=16) inside the B4 stack (8 tokens per graph, norm v2). Splits chosen on fp16.
            import fuse_small
            import kernels_v2
            import kernels_v3
            import kernels_v4
            kda.SPLITS = 32
            kernels_v3.SPLITS = 16
            undo_norm, _ = fuse_rmsnorms(model)
            undo_merge = merge_projections(model)
            undo_att = fuse_attention(model, merged=True)
            use_ours(True)
            uf = fuse_small.install(model)
            kernels_v2.kernels(); kernels_v3.kernels(); kernels_v4.kernels()
            v1n, v1a = fuse_small.add_rmsnorm, kda.decode_attn
            fuse_small.add_rmsnorm = kernels_v2.add_rmsnorm
            kda.decode_attn = kernels_v3.decode_attn
            table = {}
            first = build_graph(8)
            for _ in range(5):
                first()
            table["fp16_v3"] = warm("fp16 attention v3 S=16", first)
            kda.decode_attn = kernels_v4.decode_attn
            for S_ in (8, 16, 32):
                kernels_v4.SPLITS = S_
                table[f"fp16_v4_S{S_}"] = warm(f"fp16 attention v4 S={S_}", build_graph(8))
            best_S = min((8, 16, 32), key=lambda S_: table[f"fp16_v4_S{S_}"][0])
            kernels_v4.SPLITS = best_S
            results["best_splits_v4"] = best_S
            u = kernel_int8.install(qweights(), group=32)
            kda.decode_attn = kernels_v3.decode_attn
            table["int8_v3"] = warm("int8 attention v3 S=16", build_graph(8))
            kda.decode_attn = kernels_v4.decode_attn
            table["int8_v4"] = warm(f"int8 attention v4 S={best_S}", build_graph(8))
            u()
            kda.decode_attn = kernels_v3.decode_attn
            table["fp16_v3_again"] = warm("fp16 attention v3 S=16 (again)", build_graph(8))
            fuse_small.add_rmsnorm, kda.decode_attn = v1n, v1a
            uf(); use_ours(False)
            undo_att(); undo_merge(); undo_norm()
            results["table"] = table
            b = (table["fp16_v3"][0] + table["fp16_v3_again"][0]) / 2
            log(f"RESULT Y: best v4 splits {best_S} | fp16 v3 {b:.2f} -> v4 {table[f'fp16_v4_S{best_S}'][0]} ms | int8 v3 "
                f"{table['int8_v3'][0]} -> v4 {table['int8_v4'][0]} ms ({1000 / table['int8_v4'][0]:.1f} tok/s) | token "
                f"first diff: fp16 v4 {table[f'fp16_v4_S{best_S}'][1]}, int8 v4 {table['int8_v4'][1]}")

        elif stage == "Z":
            # S4b (pre-registered, docs/PREREG_S4.md): split-KV decode attention kernel inside the real model.
            # Order: fp16 SDPA -> fp16 ours -> int8-g32 SDPA -> int8-g32 ours -> fp16 SDPA (again). Prompts (q_len > 1)
            # still use SDPA; only single-token decode uses the new kernel.
            import kernel_int8
            import kernel_decode_attn as kda
            from transformers.models.qwen3 import modeling_qwen3 as mq
            kda.get_kernels()
            keep = []
            TOK = 56                  # tokens timed per run (divisible by 1, 2, 4, 8)

            def qweights():
                ws = []
                for layer in model.model.layers:
                    ws += [layer.self_attn._w_qkv, layer.self_attn.o_proj.weight,
                           layer.mlp._w_gu, layer.mlp.down_proj.weight]
                return ws + [model.lm_head.weight]

            def build_graph(KK=1):
                nonlocal cache
                cache = new_cache()
                st = torch.zeros((1, 1), dtype=torch.long, device=dev)
                sp = torch.zeros((1,), dtype=torch.long, device=dev)
                idx = torch.zeros((1,), dtype=torch.long, device=dev)
                buf = torch.zeros((NEW,), dtype=torch.long, device=dev)
                t0 = prefill()
                for i_, layer_ in enumerate(model.model.layers):   # S4: kernel reads the cache's own length counter
                    layer_.self_attn._len = cache.layers[i_].cumulative_length

                def full():
                    lg = model(input_ids=st, past_key_values=cache, cache_position=sp, use_cache=True).logits
                    nxt = lg[:, -1].argmax(-1)
                    buf.index_copy_(0, idx, nxt)
                    st.copy_(nxt.view(1, 1)); sp.add_(1); idx.add_(1)
                    return lg

                def arm():
                    reset_counters(N_PROMPT)
                    st.copy_(t0.view(1, 1)); sp.fill_(N_PROMPT); idx.fill_(1); buf.zero_(); buf[0] = t0[0]

                arm()
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(3):
                        full()
                torch.cuda.current_stream().wait_stream(s)
                sync()
                arm()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    for _k in range(KK):          # S9: KK decode steps per graph replay
                        full()
                keep.append(g)

                def run():
                    arm()
                    sync(); t = time.perf_counter()
                    for _ in range(TOK // KK):
                        g.replay()
                    sync()
                    return (time.perf_counter() - t) * 1000 / TOK, buf.tolist()[:TOK + 1]
                return run

            def warm(name, run):
                run(); run()
                ms_all, toks = [], None
                for _ in range(5):
                    ms, toks = run()
                    ms_all.append(ms)
                ms = statistics.median(ms_all)
                log(f"Z {name:<30} warm {ms:6.2f} ms/step ({1000 / ms:6.1f} tok/s) | tokens first diff vs generate: "
                    f"{first_diff(toks, ref[:TOK + 1])}")
                return round(ms, 2), first_diff(toks, ref[:TOK + 1])

            orig_get = mq.ALL_ATTENTION_FUNCTIONS.get_interface
            sdpa_fn = orig_get("sdpa", mq.eager_attention_forward)

            def our_att(module, q, k, v, mask, dropout=0.0, scaling=None, sliding_window=None, **kw):
                if q.shape[2] != 1:
                    return sdpa_fn(module, q, k, v, mask, dropout=dropout, scaling=scaling,
                                   sliding_window=sliding_window, **kw)
                return kda.decode_attn(q, k, v, module._len, scaling), None

            def use_ours(on):
                mq.ALL_ATTENTION_FUNCTIONS.get_interface = (lambda *a, **k: our_att) if on else orig_get

            # S13b: attention v5 vs v3 (S=16) inside the B4 stack (8 tokens per graph, norm v2). Splits chosen on fp16.
            import fuse_small
            import kernels_v2
            import kernels_v3
            import kernels_v5
            kda.SPLITS = 32
            kernels_v3.SPLITS = 16
            undo_norm, _ = fuse_rmsnorms(model)
            undo_merge = merge_projections(model)
            undo_att = fuse_attention(model, merged=True)
            use_ours(True)
            uf = fuse_small.install(model)
            kernels_v2.kernels(); kernels_v3.kernels(); kernels_v5.kernels()
            v1n, v1a = fuse_small.add_rmsnorm, kda.decode_attn
            fuse_small.add_rmsnorm = kernels_v2.add_rmsnorm
            kda.decode_attn = kernels_v3.decode_attn
            table = {}
            first = build_graph(8)
            for _ in range(5):
                first()
            table["fp16_v3"] = warm("fp16 attention v3 S=16", first)
            kda.decode_attn = kernels_v5.decode_attn
            for S_ in (8, 16, 32):
                kernels_v5.SPLITS = S_
                table[f"fp16_v5_S{S_}"] = warm(f"fp16 attention v5 S={S_}", build_graph(8))
            best_S = min((8, 16, 32), key=lambda S_: table[f"fp16_v5_S{S_}"][0])
            kernels_v5.SPLITS = best_S
            results["best_splits_v5"] = best_S
            u = kernel_int8.install(qweights(), group=32)
            kda.decode_attn = kernels_v3.decode_attn
            table["int8_v3"] = warm("int8 attention v3 S=16", build_graph(8))
            kda.decode_attn = kernels_v5.decode_attn
            table["int8_v5"] = warm(f"int8 attention v5 S={best_S}", build_graph(8))
            u()
            kda.decode_attn = kernels_v3.decode_attn
            table["fp16_v3_again"] = warm("fp16 attention v3 S=16 (again)", build_graph(8))
            fuse_small.add_rmsnorm, kda.decode_attn = v1n, v1a
            uf(); use_ours(False)
            undo_att(); undo_merge(); undo_norm()
            results["table"] = table
            b = (table["fp16_v3"][0] + table["fp16_v3_again"][0]) / 2
            log(f"RESULT Z: best v5 splits {best_S} | fp16 v3 {b:.2f} -> v5 {table[f'fp16_v5_S{best_S}'][0]} ms | int8 v3 "
                f"{table['int8_v3'][0]} -> v5 {table['int8_v5'][0]} ms ({1000 / table['int8_v5'][0]:.1f} tok/s) | token "
                f"first diff: fp16 v5 {table[f'fp16_v5_S{best_S}'][1]}, int8 v5 {table['int8_v5'][1]}")

        elif stage == "Q1":
            # S4b (pre-registered, docs/PREREG_S4.md): split-KV decode attention kernel inside the real model.
            # Order: fp16 SDPA -> fp16 ours -> int8-g32 SDPA -> int8-g32 ours -> fp16 SDPA (again). Prompts (q_len > 1)
            # still use SDPA; only single-token decode uses the new kernel.
            import kernel_int8
            import kernel_decode_attn as kda
            from transformers.models.qwen3 import modeling_qwen3 as mq
            kda.get_kernels()
            keep = []
            TOK = 56                  # tokens timed per run (divisible by 1, 2, 4, 8)

            def qweights():
                ws = []
                for layer in model.model.layers:
                    ws += [layer.self_attn._w_qkv, layer.self_attn.o_proj.weight,
                           layer.mlp._w_gu, layer.mlp.down_proj.weight]
                return ws + [model.lm_head.weight]

            def build_graph(KK=1):
                nonlocal cache
                cache = new_cache()
                st = torch.zeros((1, 1), dtype=torch.long, device=dev)
                sp = torch.zeros((1,), dtype=torch.long, device=dev)
                idx = torch.zeros((1,), dtype=torch.long, device=dev)
                buf = torch.zeros((NEW,), dtype=torch.long, device=dev)
                t0 = prefill()
                for i_, layer_ in enumerate(model.model.layers):   # S4: kernel reads the cache's own length counter
                    layer_.self_attn._len = cache.layers[i_].cumulative_length

                def full():
                    lg = model(input_ids=st, past_key_values=cache, cache_position=sp, use_cache=True).logits
                    nxt = lg[:, -1].argmax(-1)
                    buf.index_copy_(0, idx, nxt)
                    st.copy_(nxt.view(1, 1)); sp.add_(1); idx.add_(1)
                    return lg

                def arm():
                    reset_counters(N_PROMPT)
                    st.copy_(t0.view(1, 1)); sp.fill_(N_PROMPT); idx.fill_(1); buf.zero_(); buf[0] = t0[0]

                arm()
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(3):
                        full()
                torch.cuda.current_stream().wait_stream(s)
                sync()
                arm()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    for _k in range(KK):          # S9: KK decode steps per graph replay
                        full()
                keep.append(g)

                def run():
                    arm()
                    sync(); t = time.perf_counter()
                    for _ in range(TOK // KK):
                        g.replay()
                    sync()
                    return (time.perf_counter() - t) * 1000 / TOK, buf.tolist()[:TOK + 1]
                return run

            def warm(name, run):
                run(); run()
                ms_all, toks = [], None
                for _ in range(5):
                    ms, toks = run()
                    ms_all.append(ms)
                ms = statistics.median(ms_all)
                log(f"Q1 {name:<29} warm {ms:6.2f} ms/step ({1000 / ms:6.1f} tok/s) | tokens first diff vs generate: "
                    f"{first_diff(toks, ref[:TOK + 1])}")
                return round(ms, 2), first_diff(toks, ref[:TOK + 1])

            orig_get = mq.ALL_ATTENTION_FUNCTIONS.get_interface
            sdpa_fn = orig_get("sdpa", mq.eager_attention_forward)

            def our_att(module, q, k, v, mask, dropout=0.0, scaling=None, sliding_window=None, **kw):
                if q.shape[2] != 1:
                    return sdpa_fn(module, q, k, v, mask, dropout=dropout, scaling=scaling,
                                   sliding_window=sliding_window, **kw)
                return kda.decode_attn(q, k, v, module._len, scaling), None

            def use_ours(on):
                mq.ALL_ATTENTION_FUNCTIONS.get_interface = (lambda *a, **k: our_att) if on else orig_get

            # Q1b (pre-registered, docs/PREREG_Q1.md): int4-g32 vs int8-g32 vs fp16. Speed in the best stack (8 tokens per
            # graph, norm v2, attention v3, small-op fusions); quality vs a float32 reference on 10 prompts.
            import fuse_small
            import kernels_v2
            import kernels_v3
            import kernel_int4
            kda.SPLITS = 32
            undo_norm, _ = fuse_rmsnorms(model)
            undo_merge = merge_projections(model)
            undo_att = fuse_attention(model, merged=True)
            use_ours(True)
            uf = fuse_small.install(model)
            kernels_v2.kernels(); kernels_v3.kernels()
            kernels_v3.SPLITS = 16
            v1n, v1a = fuse_small.add_rmsnorm, kda.decode_attn
            fuse_small.add_rmsnorm, kda.decode_attn = kernels_v2.add_rmsnorm, kernels_v3.decode_attn
            table = {}
            first = build_graph(8)
            for _ in range(5):
                first()
            table["fp16"] = warm("fp16 best", first)
            u = kernel_int8.install(qweights(), group=32)
            table["int8"] = warm("int8-g32 best", build_graph(8))
            u()
            u = kernel_int4.install(qweights())
            table["int4"] = warm("int4-g32 best", build_graph(8))
            u()
            table["fp16_again"] = warm("fp16 best (again)", build_graph(8))
            fuse_small.add_rmsnorm, kda.decode_attn = v1n, v1a
            uf(); use_ours(False)
            results["table"] = table
            log(f"Q1 speed: fp16 {table['fp16'][0]}/{table['fp16_again'][0]} | int8 {table['int8'][0]} | int4 "
                f"{table['int4'][0]} ms/token ({1000 / table['int4'][0]:.1f} tok/s)")

            m32 = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, dtype=torch.float32, attn_implementation="sdpa").to(device=dev).eval()
            texts = ["The capital of France is", "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
                     "Three tips for learning a new programming language are",
                     "Question: What is 17 multiplied by 23?\nAnswer:", "Once upon a time, in a small village by the sea,",
                     "Photosynthesis is the process by which plants",
                     "List of the planets in our solar system, in order from the Sun:\n1.",
                     "The main difference between a CPU and a GPU is that",
                     "Translate to French: 'Good morning, how are you?'\nFrench:"]
            items = [(t_, tok(t_, return_tensors="pt").input_ids.to(dev)) for t_ in texts] + \
                    [("filler 1024 tokens", make_prompt_ids(tok, 1024, dev))]
            kl = lambda a, b: float((a.exp() * (a - b)).sum(-1).mean())
            top = lambda a, b: float((a.argmax(-1) == b.argmax(-1)).float().mean())
            qrows = []
            for label, p_ids in items:
                P = p_ids.shape[1]
                genk = dict(attention_mask=torch.ones_like(p_ids), max_new_tokens=48, min_new_tokens=48,
                            do_sample=False, pad_token_id=tok.eos_token_id)
                seq = model.generate(input_ids=p_ids, **genk)
                l32 = torch.log_softmax(m32(input_ids=seq, use_cache=False).logits[0, P - 1:-1].float(), -1)
                l16 = torch.log_softmax(model(input_ids=seq, use_cache=False).logits[0, P - 1:-1].float(), -1)
                row = {"prompt": label[:30], "fp16_KL": kl(l32, l16), "fp16_top1": top(l32, l16)}
                for tag, inst in (("int8", lambda: kernel_int8.install(qweights(), group=32)),
                                  ("int4", lambda: kernel_int4.install(qweights()))):
                    uq = inst()
                    lq = torch.log_softmax(model(input_ids=seq, use_cache=False).logits[0, P - 1:-1].float(), -1)
                    sq = model.generate(input_ids=p_ids, **genk)
                    uq()
                    row[f"{tag}_KL"] = kl(l32, lq)
                    row[f"{tag}_top1"] = top(l32, lq)
                    row[f"{tag}_text_first_diff"] = first_diff(seq[0, P:].tolist(), sq[0, P:].tolist())
                qrows.append(row)
                log(f"Q1 quality {label[:22]!r:<26} | int8 KL {row['int8_KL']:.2e} top-1 {row['int8_top1']:.3f} diff "
                    f"{row['int8_text_first_diff']} | int4 KL {row['int4_KL']:.2e} top-1 {row['int4_top1']:.3f} diff "
                    f"{row['int4_text_first_diff']}")
            undo_att(); undo_merge(); undo_norm()
            del m32
            results["quality"] = qrows
            mean = lambda k: statistics.mean(r[k] for r in qrows)
            same = lambda tag: sum(r[f"{tag}_text_first_diff"] is None for r in qrows)
            log(f"RESULT Q1: speed fp16 {(table['fp16'][0] + table['fp16_again'][0]) / 2:.2f} | int8 {table['int8'][0]} | "
                f"int4 {table['int4'][0]} ms/token || quality vs fp32 over {len(qrows)} prompts: fp16 KL "
                f"{mean('fp16_KL'):.1e} | int8 KL {mean('int8_KL'):.2e}, top-1 {mean('int8_top1'):.3f}, same text "
                f"{same('int8')}/{len(qrows)} | int4 KL {mean('int4_KL'):.2e}, top-1 {mean('int4_top1'):.3f}, same text "
                f"{same('int4')}/{len(qrows)}")

        elif stage == "Q2":
            # Q2 (pre-registered, docs/PREREG_Q2.md): quality of 4-bit quantization variants via FAKE quantization (each
            # weight replaced by its quantize->dequantize value; the model then runs normally). Measures exactly the
            # accuracy a correct kernel of that format would give, without a kernel per variant. Same 10 prompts as E14/Q1.
            import torch.nn as nn

            def chunked(fn, W, rows=4096):
                out = torch.empty_like(W)
                for r in range(0, W.shape[0], rows):
                    out[r:r + rows] = fn(W[r:r + rows])
                return out

            def fq(W, bits=4, G=32, asym=False, search=False):
                def one(Wc):
                    N, K = Wc.shape
                    Wg = Wc.float().view(N, K // G, G)
                    mn, mx = Wg.amin(-1, keepdim=True), Wg.amax(-1, keepdim=True)
                    amax = Wg.abs().amax(-1, keepdim=True)
                    qmax = 2 ** (bits - 1) - 1
                    best, best_err = None, None
                    for f in ([1.0] if not search else [1.0 - 0.02 * i for i in range(16)]):
                        if asym:
                            lo, hi = mn * f, mx * f
                            d = ((hi - lo) / (2 ** bits - 1)).clamp_min(1e-12).half().float()
                            m = lo.half().float()
                            q = torch.round((Wg - m) / d).clamp(0, 2 ** bits - 1)
                            dq = q * d + m
                        else:
                            sc = (amax * f / qmax).clamp_min(1e-12).half().float()
                            q = torch.round(Wg / sc).clamp(-qmax - 1, qmax)
                            dq = q * sc
                        err = (dq - Wg).pow(2).sum(-1, keepdim=True)
                        if best is None:
                            best, best_err = dq, err
                        else:
                            better = err < best_err
                            best = torch.where(better, dq, best)
                            best_err = torch.minimum(err, best_err)
                    return best.view(N, K).to(Wc.dtype)
                return chunked(one, W)

            lin = []
            for layer in model.model.layers:
                a_, m_ = layer.self_attn, layer.mlp
                lin += [a_.q_proj, a_.k_proj, a_.v_proj, a_.o_proj, m_.gate_proj, m_.up_proj, m_.down_proj]
            orig_cpu = [l_.weight.data.cpu() for l_ in lin]
            lm_orig = model.lm_head.weight          # tied to the embedding: replaced (untied) during a variant, restored after
            variants = [
                ("A int4 sym RTN g32 (Q1)", dict(bits=4, G=32), "same", 0.5625),
                ("B int4 sym search g32", dict(bits=4, G=32, search=True), "same", 0.5625),
                ("C int4 asym g32 (Q4_1-like)", dict(bits=4, G=32, asym=True), "same", 0.625),
                ("D int4 asym search g32", dict(bits=4, G=32, asym=True, search=True), "same", 0.625),
                ("E = D + lm_head int8-g32", dict(bits=4, G=32, asym=True, search=True), "int8", 0.625),
                ("F int4 asym search g16", dict(bits=4, G=16, asym=True, search=True), "same", 0.75),
                ("G int8 sym RTN g32 (reference)", dict(bits=8, G=32), "same", 1.0625),
            ]

            m32 = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, dtype=torch.float32, attn_implementation="sdpa").to(device=dev).eval()
            texts = ["The capital of France is", "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
                     "Three tips for learning a new programming language are",
                     "Question: What is 17 multiplied by 23?\nAnswer:", "Once upon a time, in a small village by the sea,",
                     "Photosynthesis is the process by which plants",
                     "List of the planets in our solar system, in order from the Sun:\n1.",
                     "The main difference between a CPU and a GPU is that",
                     "Translate to French: 'Good morning, how are you?'\nFrench:"]
            items = [(t_, tok(t_, return_tensors="pt").input_ids.to(dev)) for t_ in texts] + \
                    [("filler 1024 tokens", make_prompt_ids(tok, 1024, dev))]
            refs = []
            for label, p_ids in items:                       # fp16 reference text + fp32 log-probs, computed once
                P = p_ids.shape[1]
                genk = dict(attention_mask=torch.ones_like(p_ids), max_new_tokens=48, min_new_tokens=48,
                            do_sample=False, pad_token_id=tok.eos_token_id)
                seq = model.generate(input_ids=p_ids, **genk)
                l32 = torch.log_softmax(m32(input_ids=seq, use_cache=False).logits[0, P - 1:-1].float(), -1)
                refs.append((label, p_ids, P, genk, seq, l32))
            del m32
            torch.cuda.empty_cache()
            kl = lambda a, b: float((a.exp() * (a - b)).sum(-1).mean())
            top = lambda a, b: float((a.argmax(-1) == b.argmax(-1)).float().mean())
            summary = []
            for name, kw, lm_mode, bpw in variants:
                t_v = time.perf_counter()
                for l_ in lin:
                    l_.weight.data.copy_(fq(l_.weight.data, **kw))
                lm_kw = kw if lm_mode == "same" else dict(bits=8, G=32)
                model.lm_head.weight = nn.Parameter(fq(lm_orig.data, **lm_kw), requires_grad=False)
                rows = []
                for label, p_ids, P, genk, seq, l32 in refs:
                    lq = torch.log_softmax(model(input_ids=seq, use_cache=False).logits[0, P - 1:-1].float(), -1)
                    sq = model.generate(input_ids=p_ids, **genk)
                    rows.append((kl(l32, lq), top(l32, lq), first_diff(seq[0, P:].tolist(), sq[0, P:].tolist())))
                for l_, o_ in zip(lin, orig_cpu):
                    l_.weight.data.copy_(o_.to(dev))
                model.lm_head.weight = lm_orig
                mk = statistics.mean(r[0] for r in rows)
                mt = statistics.mean(r[1] for r in rows)
                same = sum(r[2] is None for r in rows)
                med_div = statistics.median([48 if r[2] is None else r[2] for r in rows])
                summary.append({"variant": name, "bytes_per_weight": bpw, "mean_KL": mk, "mean_top1": mt,
                                "same_text": same, "median_first_divergence": med_div,
                                "per_prompt": [(lab_[0][:30], r[0], r[1], r[2]) for lab_, r in zip(refs, rows)]})
                log(f"Q2 {name:<32} {bpw:.4f} B/w | mean KL {mk:.2e} | top-1 {mt:.3f} | same text {same}/10 | median "
                    f"first divergence {med_div} tokens | {time.perf_counter() - t_v:.0f} s")
            results["variants"] = summary
            best4 = min((v for v in summary if not v["variant"].startswith("G")), key=lambda v: v["mean_KL"])
            log(f"RESULT Q2: best 4-bit = {best4['variant']} (KL {best4['mean_KL']:.2e}, top-1 {best4['mean_top1']:.3f}, "
                f"{best4['bytes_per_weight']} B/w) | A (Q1) KL {summary[0]['mean_KL']:.2e} | int8 KL {summary[-1]['mean_KL']:.2e}")

        elif stage == "SD1":
            # SD1 (pre-registered, docs/PREREG_SD1.md): greedy speculative decoding, draft Qwen3-0.6B -> target MODEL_ID.
            # Eager PyTorch (no graphs): measures acceptance and output identity exactly; speed in the fast stack is then
            # modeled from measured step times (A). Prior art: Leviathan et al. 2023; Chen et al. 2023.
            from transformers import DynamicCache
            draft = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen3-0.6B", dtype=torch.float16, attn_implementation="sdpa").to(dev).eval()
            NEWT = 64
            texts = ["The capital of France is", "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
                     "Three tips for learning a new programming language are",
                     "Question: What is 17 multiplied by 23?\nAnswer:", "Once upon a time, in a small village by the sea,",
                     "Photosynthesis is the process by which plants",
                     "List of the planets in our solar system, in order from the Sun:\n1.",
                     "The main difference between a CPU and a GPU is that",
                     "Translate to French: 'Good morning, how are you?'\nFrench:"]
            items = [(t_, tok(t_, return_tensors="pt").input_ids.to(dev)) for t_ in texts] + \
                    [("filler 1024 tokens", make_prompt_ids(tok, 1024, dev))]

            def newc(cfg):
                try:
                    return DynamicCache(config=cfg)
                except TypeError:
                    return DynamicCache()

            def fwd(m, cache, toks):
                ids = torch.tensor([toks], device=dev)
                return m(input_ids=ids, past_key_values=cache, use_cache=True).logits[0]

            def greedy(p_ids):
                prompt = p_ids[0].tolist()
                c = newc(model.config)
                lg = fwd(model, c, prompt)
                out = [int(lg[-1].argmax())]
                while len(out) < NEWT:
                    out.append(int(fwd(model, c, [out[-1]])[-1].argmax()))
                return out

            def spec(p_ids, k):
                prompt = p_ids[0].tolist()
                ct, cd = newc(model.config), newc(draft.config)
                fwd(model, ct, prompt[:-1]); fwd(draft, cd, prompt[:-1])
                out, pend_t, pend_d, rounds, accepted = [], [prompt[-1]], [prompt[-1]], 0, 0
                while len(out) < NEWT:
                    props = []
                    nxt = int(fwd(draft, cd, pend_d)[-1].argmax())
                    props.append(nxt)
                    for _ in range(k - 1):
                        nxt = int(fwd(draft, cd, [nxt])[-1].argmax())
                        props.append(nxt)
                    preds = fwd(model, ct, pend_t + props)[len(pend_t) - 1:].argmax(-1).tolist()   # k+1 target picks
                    n = 0
                    while n < k and props[n] == preds[n]:
                        n += 1
                    new = props[:n] + [preds[n]]
                    if k - n:
                        ct.crop(-(k - n))                      # target keeps pend_t + props[:n]
                    rd = (k - 1) - min(n, k - 1)
                    if rd:
                        cd.crop(-rd)                           # draft keeps pend_d + props[:min(n, k-1)]
                    pend_t = [new[-1]]
                    pend_d = (props[k - 1:k] if n == k else []) + [new[-1]]
                    out += new
                    rounds += 1
                    accepted += n
                return out[:NEWT], rounds, accepted

            KS = (2, 3, 4, 6)
            stats = {k: {"rounds": 0, "accepted": 0, "tokens": 0, "identical": 0, "first_diffs": []} for k in KS}
            for label, p_ids in items:
                ref = greedy(p_ids)
                line = []
                for k in KS:
                    o, r, a = spec(p_ids, k)
                    d = first_diff(o, ref)
                    st = stats[k]
                    st["rounds"] += r; st["accepted"] += a; st["tokens"] += len(o)
                    st["identical"] += d is None
                    st["first_diffs"].append(d)
                    line.append(f"k={k}: {len(o) / r:.2f} tok/round, diff {d}")
                log(f"SD1 {label[:22]!r:<26} | " + " | ".join(line))
            T = {"fp16": (6.60, 3.00), "int8": (4.00, 2.10)}     # measured step times (ms): target 1.7B, draft 0.6B (B5)
            summary = {}
            for k in KS:
                st = stats[k]
                tau = st["tokens"] / st["rounds"]
                alpha = st["accepted"] / (st["rounds"] * k)
                sp = {f: tau * tT / (k * tD + tT) for f, (tT, tD) in T.items()}
                summary[k] = {"tokens_per_round": tau, "acceptance_rate": alpha, "identical": st["identical"],
                              "modeled_speedup": sp, "first_diffs": st["first_diffs"]}
                log(f"SD1 k={k}: {tau:.2f} tokens per target pass, draft acceptance {100 * alpha:.1f}%, identical text "
                    f"{st['identical']}/10 | modeled speedup in our stack (A, verify = 1 target step): fp16 "
                    f"{sp['fp16']:.2f}x, int8 {sp['int8']:.2f}x")
            results["sd1"] = summary
            best = max(KS, key=lambda k: summary[k]["modeled_speedup"]["int8"])
            log(f"RESULT SD1: best k={best}: {summary[best]['tokens_per_round']:.2f} tok/pass, identical "
                f"{summary[best]['identical']}/10, modeled int8 {summary[best]['modeled_speedup']['int8']:.2f}x, fp16 "
                f"{summary[best]['modeled_speedup']['fp16']:.2f}x")

        elif stage == "SD2":
            # SD2 (pre-registered, docs/PREREG_SD2.md): (1) near-tie check for SD1's divergences; (2) cheaper drafts for the
            # same lossless verify: Qwen3-0.6B fp16 / int8-g32 / int4-g32 (fake-quantized; a lossy draft only changes
            # acceptance, the target still decides every token) and prompt-lookup drafting (copy what followed the latest
            # matching n-gram in the context; ~free). Eager PyTorch; speeds modeled from measured step times (A).
            from transformers import DynamicCache
            import copy

            def newc(cfg):
                try:
                    return DynamicCache(config=cfg)
                except TypeError:
                    return DynamicCache()

            def fq_(W, bits, G=32, search=False):
                out = torch.empty_like(W)
                qmax = 2 ** (bits - 1) - 1
                for r in range(0, W.shape[0], 4096):
                    Wc = W[r:r + 4096]
                    N_, K_ = Wc.shape
                    Wg = Wc.float().view(N_, K_ // G, G)
                    amax = Wg.abs().amax(-1, keepdim=True)
                    best, berr = None, None
                    for f in ([1.0] if not search else [1.0 - 0.02 * i for i in range(16)]):
                        sc = (amax * f / qmax).clamp_min(1e-12).half().float()
                        dq = torch.round(Wg / sc).clamp(-qmax - 1, qmax) * sc
                        err = (dq - Wg).pow(2).sum(-1, keepdim=True)
                        if best is None:
                            best, berr = dq, err
                        else:
                            m_ = err < berr
                            best, berr = torch.where(m_, dq, best), torch.minimum(err, berr)
                    out[r:r + 4096] = best.view(N_, K_).to(W.dtype)
                return out

            base_draft = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen3-0.6B", dtype=torch.float16, attn_implementation="sdpa").to(dev).eval()

            def quantized_draft(bits, search):
                d = copy.deepcopy(base_draft)
                for layer in d.model.layers:
                    for l_ in (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj,
                               layer.self_attn.o_proj, layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj):
                        l_.weight.data.copy_(fq_(l_.weight.data, bits, search=search))
                d.lm_head.weight = torch.nn.Parameter(fq_(base_draft.lm_head.weight.data, bits, search=search),
                                                      requires_grad=False)
                return d

            NEWT = 64
            texts = ["The capital of France is", "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
                     "Three tips for learning a new programming language are",
                     "Question: What is 17 multiplied by 23?\nAnswer:", "Once upon a time, in a small village by the sea,",
                     "Photosynthesis is the process by which plants",
                     "List of the planets in our solar system, in order from the Sun:\n1.",
                     "The main difference between a CPU and a GPU is that",
                     "Translate to French: 'Good morning, how are you?'\nFrench:"]
            items = [(t_, tok(t_, return_tensors="pt").input_ids.to(dev)) for t_ in texts] + \
                    [("filler 1024 tokens", make_prompt_ids(tok, 1024, dev))]

            def fwd(m, cache, toks):
                ids = torch.tensor([toks], device=dev)
                return m(input_ids=ids, past_key_values=cache, use_cache=True).logits[0]

            def gap_of(lg):
                t2 = lg.float().topk(2).values
                return float(t2[0] - t2[1])

            def greedy(p_ids):
                prompt = p_ids[0].tolist()
                c = newc(model.config)
                lg = fwd(model, c, prompt)[-1]
                out, gaps = [int(lg.argmax())], [gap_of(lg)]
                while len(out) < NEWT:
                    lg = fwd(model, c, [out[-1]])[-1]
                    out.append(int(lg.argmax())); gaps.append(gap_of(lg))
                return out, gaps

            def lookup_props(ctx, k, nmax=3):
                for n in range(nmax, 0, -1):
                    if len(ctx) <= n:
                        continue
                    tail = ctx[-n:]
                    for i in range(len(ctx) - n - 1, -1, -1):          # latest earlier occurrence
                        if ctx[i:i + n] == tail:
                            return ctx[i + n:i + n + k]
                return []

            def spec(p_ids, k, draft):
                prompt = p_ids[0].tolist()
                ct = newc(model.config)
                fwd(model, ct, prompt[:-1])
                cd = None
                if draft is not None:
                    cd = newc(draft.config)
                    fwd(draft, cd, prompt[:-1])
                out, pend_t, pend_d, rounds, accepted, proposed = [], [prompt[-1]], [prompt[-1]], 0, 0, 0
                while len(out) < NEWT:
                    if draft is None:
                        props = lookup_props(prompt + out, k)
                    else:
                        props = []
                        nxt = int(fwd(draft, cd, pend_d)[-1].argmax())
                        props.append(nxt)
                        for _ in range(k - 1):
                            nxt = int(fwd(draft, cd, [nxt])[-1].argmax())
                            props.append(nxt)
                    m = len(props)
                    preds = fwd(model, ct, pend_t + props)[len(pend_t) - 1:].argmax(-1).tolist()
                    n = 0
                    while n < m and props[n] == preds[n]:
                        n += 1
                    new = props[:n] + [preds[n]]
                    if m - n:
                        ct.crop(-(m - n))
                    if draft is not None:
                        rd = (k - 1) - min(n, k - 1)
                        if rd:
                            cd.crop(-rd)
                        pend_d = (props[k - 1:k] if n == k else []) + [new[-1]]
                    pend_t = [new[-1]]
                    out += new
                    rounds += 1
                    accepted += n
                    proposed += m
                return out[:NEWT], rounds, accepted, proposed

            refs = []
            for label, p_ids in items:
                ref, gaps = greedy(p_ids)
                refs.append((label, p_ids, ref, gaps))
            all_gaps = sorted(g for _, _, _, gs in refs for g in gs)
            med_gap = all_gaps[len(all_gaps) // 2]
            p5_gap = all_gaps[len(all_gaps) // 20]
            log(f"SD2 target logit gap (top-1 minus top-2) over all {len(all_gaps)} positions: median {med_gap:.3f}, "
                f"5th percentile {p5_gap:.3f}")
            drafts = [("0.6B fp16", base_draft, 3.00), ("0.6B int8-g32", None, 2.20), ("0.6B int4-g32 search", None, 1.74),
                      ("prompt lookup", "lookup", 0.05)]
            T = {"fp16": 6.60, "int8": 4.00}
            summary = []
            for dname, dmodel, t_d in drafts:
                if dname == "0.6B int8-g32":
                    dmodel = quantized_draft(8, False)
                elif dname == "0.6B int4-g32 search":
                    dmodel = quantized_draft(4, True)
                d_arg = None if dmodel == "lookup" else dmodel
                for k in ((2, 3, 4) if d_arg is not None else (4, 8)):
                    R = A = P = 0
                    ident, diffs = 0, []
                    for label, p_ids, ref, gaps in refs:
                        o, r, a, pr = spec(p_ids, k, d_arg)
                        d = first_diff(o, ref)
                        R += r; A += a; P += pr
                        ident += d is None
                        diffs.append((label[:16], d, None if d is None else round(gaps[d], 3)))
                    tau = NEWT * len(refs) / R
                    kd = (P / R) if d_arg is None else k
                    sp = {f: tau * tT / (kd * t_d + tT) for f, tT in T.items()}
                    tok_s = {f: 1000 / tT * sp[f] for f, tT in T.items()}
                    summary.append({"draft": dname, "k": k, "tokens_per_pass": tau, "acceptance": A / max(P, 1),
                                    "identical": ident, "diffs": diffs, "modeled_speedup": sp, "modeled_tok_s": tok_s})
                    dd = [x for x in diffs if x[1] is not None]
                    log(f"SD2 {dname:<21} k={k}: {tau:.2f} tok/pass, acceptance {100 * A / max(P, 1):.1f}%, identical "
                        f"{ident}/10 | modeled (A): fp16 {sp['fp16']:.2f}x ({tok_s['fp16']:.0f} tok/s), int8 "
                        f"{sp['int8']:.2f}x ({tok_s['int8']:.0f} tok/s) | divergences (prompt, pos, target gap there): {dd}")
                if dname in ("0.6B int8-g32", "0.6B int4-g32 search"):
                    del dmodel
                    torch.cuda.empty_cache()
            results["sd2"] = summary
            results["gap_median"], results["gap_p5"] = med_gap, p5_gap
            b8 = max(summary, key=lambda v: v["modeled_speedup"]["int8"])
            b16 = max(summary, key=lambda v: v["modeled_speedup"]["fp16"])
            log(f"RESULT SD2: best fp16: {b16['draft']} k={b16['k']} {b16['modeled_speedup']['fp16']:.2f}x "
                f"({b16['modeled_tok_s']['fp16']:.0f} tok/s) | best int8: {b8['draft']} k={b8['k']} "
                f"{b8['modeled_speedup']['int8']:.2f}x ({b8['modeled_tok_s']['int8']:.0f} tok/s) | gap median {med_gap:.3f}")

        elif stage in ("H", "I"):
            # stage I = the same check on LONG prompts (1024 and 2048 filler tokens), see E8b
            # E8: is the merged-matrix model as ACCURATE as the original? Use a float32 copy of the
            # same model as the "truth" and measure how far each fp16 version is from it, on varied prompts.
            # If merged-vs-fp32 error ~= original-vs-fp32 error, merging only changes rounding.
            prompts = [
                "The capital of France is",
                "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
                "Three tips for learning a new programming language are",
                "Question: What is 17 multiplied by 23?\nAnswer:",
                "Once upon a time, in a small village by the sea,",
                "Photosynthesis is the process by which plants",
                "List of the planets in our solar system, in order from the Sun:\n1.",
                "The main difference between a CPU and a GPU is that",
                "Translate to French: 'Good morning, how are you?'\nFrench:",
            ]
            GEN = 48
            m32 = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, dtype=torch.float32, attn_implementation="sdpa").to(device=dev).eval()
            log("float32 reference model loaded")

            def lsm(x):
                return torch.log_softmax(x.float(), dim=-1)

            def compare(a, b):
                la, lb = lsm(a), lsm(b)
                kl = (la.exp() * (la - lb)).sum(-1).mean()  # KL(a || b), a = reference
                return float((a.float() - b.float()).abs().max()), float(kl), float((a.argmax(-1) == b.argmax(-1)).float().mean())

            rows = []
            if stage == "I":
                items = [(f"filler {n} tokens", make_prompt_ids(tok, n, dev)) for n in (1024, 2048)]
            else:
                items = [(t, tok(t, return_tensors="pt").input_ids.to(dev)) for t in prompts]
            for k, (ptxt, p_ids) in enumerate(items):
                P = p_ids.shape[1]
                cont_o = model.generate(input_ids=p_ids, attention_mask=torch.ones_like(p_ids), max_new_tokens=GEN,
                                        min_new_tokens=GEN, do_sample=False, pad_token_id=tok.eos_token_id)
                seq = cont_o  # prompt + original continuation; all models scored on the SAME text
                l32 = m32(input_ids=seq, use_cache=False).logits[0, P - 1:-1]
                lo = model(input_ids=seq, use_cache=False).logits[0, P - 1:-1]
                restore = merge_projections(model)
                lm = model(input_ids=seq, use_cache=False).logits[0, P - 1:-1]
                cont_m = model.generate(input_ids=p_ids, attention_mask=torch.ones_like(p_ids), max_new_tokens=GEN,
                                        min_new_tokens=GEN, do_sample=False, pad_token_id=tok.eos_token_id)
                restore()
                d_o, kl_o, top_o = compare(l32, lo)
                d_m, kl_m, top_m = compare(l32, lm)
                _, kl_mo, top_mo = compare(lo, lm)
                fd = first_diff(cont_o[0, P:].tolist(), cont_m[0, P:].tolist())
                row = {"prompt": ptxt[:40], "orig_vs_fp32_maxdiff": round(d_o, 3), "merged_vs_fp32_maxdiff": round(d_m, 3),
                       "orig_vs_fp32_KL": kl_o, "merged_vs_fp32_KL": kl_m, "merged_vs_orig_KL": kl_mo,
                       "orig_top1_agree_fp32": top_o, "merged_top1_agree_fp32": top_m, "merged_top1_agree_orig": top_mo,
                       "greedy_first_diff_merged_vs_orig": fd}
                rows.append(row)
                log(f"H{k} {ptxt[:28]!r:<32} | max diff vs fp32: orig {d_o:5.3f} merged {d_m:5.3f} | "
                    f"KL vs fp32: orig {kl_o:.2e} merged {kl_m:.2e} | top-1 agree w/ fp32: orig {top_o:.3f} "
                    f"merged {top_m:.3f} | greedy text merged vs orig: first diff {fd}")
            mean = lambda key: statistics.mean(r[key] for r in rows)
            same_text = sum(r["greedy_first_diff_merged_vs_orig"] is None for r in rows)
            results.update(rows=rows, prompts=len(rows), same_text_count=same_text,
                           mean_orig_KL=mean("orig_vs_fp32_KL"), mean_merged_KL=mean("merged_vs_fp32_KL"),
                           mean_orig_top1=mean("orig_top1_agree_fp32"), mean_merged_top1=mean("merged_top1_agree_fp32"))
            log(f"RESULT H: {len(rows)} prompts | mean KL vs fp32: orig {mean('orig_vs_fp32_KL'):.2e}, merged "
                f"{mean('merged_vs_fp32_KL'):.2e} | mean top-1 agreement with fp32: orig {mean('orig_top1_agree_fp32'):.3f}, "
                f"merged {mean('merged_top1_agree_fp32'):.3f} | identical {GEN}-token greedy text: {same_text}/{len(rows)}")

        elif stage == "G":
            # E7: one comparable table. Every method timed two ways:
            #   cold = GPU idle for 2 s, then ONE timed decode (like the earlier C/E/E2c runs)
            #   warm = 1 untimed decode, then 5 timed decodes back to back (like E6)
            IDLE_S, WARM_N, COLD_N = 2.0, 5, 3
            keep = []

            def build_graph():
                nonlocal cache
                cache = new_cache()
                st = torch.zeros((1, 1), dtype=torch.long, device=dev)
                sp = torch.zeros((1,), dtype=torch.long, device=dev)
                idx = torch.zeros((1,), dtype=torch.long, device=dev)
                buf = torch.zeros((NEW,), dtype=torch.long, device=dev)
                t0 = prefill()

                def full():
                    lg = model(input_ids=st, past_key_values=cache, cache_position=sp, use_cache=True).logits
                    nxt = lg[:, -1].argmax(-1)
                    buf.index_copy_(0, idx, nxt)
                    st.copy_(nxt.view(1, 1)); sp.add_(1); idx.add_(1)
                    return lg

                def arm():
                    reset_counters(N_PROMPT)
                    st.copy_(t0.view(1, 1)); sp.fill_(N_PROMPT); idx.fill_(1); buf.zero_(); buf[0] = t0[0]

                arm()
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(3):
                        full()
                torch.cuda.current_stream().wait_stream(s)
                sync()
                arm()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    full()
                keep.append(g)

                def run():
                    arm()
                    sync(); t = time.perf_counter()
                    for _ in range(NEW - 1):
                        g.replay()
                    sync()
                    return (time.perf_counter() - t) * 1000 / (NEW - 1), buf.tolist()
                return run

            def gen_decode_ms():
                sync(); t = time.perf_counter(); out = gen(NEW); sync(); full_s = time.perf_counter() - t
                sync(); t = time.perf_counter(); gen(1); sync(); one_s = time.perf_counter() - t
                return (full_s - one_s) * 1000 / (NEW - 1), out[0, N_PROMPT:].tolist()

            def measure(name, fn):
                cold = []
                for _ in range(COLD_N):
                    sync(); time.sleep(IDLE_S)
                    cold.append(fn()[0])
                fn()  # untimed warm-up decode
                warm, toks = [], None
                for _ in range(WARM_N):
                    ms, toks = fn()
                    warm.append(ms)
                c, w = statistics.median(cold), statistics.median(warm)
                fd = first_diff(toks, ref)
                log(f"G {name:<16} cold {c:6.2f} ms/step | warm {w:6.2f} ms/step | tokens first diff vs generate: {fd}")
                return {"cold_ms": round(c, 2), "warm_ms": round(w, 2), "cold_all": [round(x, 2) for x in cold],
                        "warm_all": [round(x, 2) for x in warm], "first_diff": fd}

            table = {}
            table["generate"] = measure("generate", gen_decode_ms)
            table["graph"] = measure("graph", build_graph())
            restore = merge_projections(model)
            table["graph_merged"] = measure("graph_merged", build_graph())
            restore()
            table["graph_again"] = measure("graph (again)", build_graph())
            results["table"] = table
            gw, gc = table["generate"]["warm_ms"], table["generate"]["cold_ms"]
            for k in ("graph", "graph_merged", "graph_again"):
                log(f"   {k:<14} speedup vs generate: cold {gc / table[k]['cold_ms']:.2f}x | warm {gw / table[k]['warm_ms']:.2f}x")
            log("RESULT G: see table above (same procedure for every method)")

        elif stage == "F":
            # E6: graph decode with original vs merged matrices, order O-M-O, outputs checked.
            def full_graph_decode():
                nonlocal cache
                cache = new_cache()
                st = torch.zeros((1, 1), dtype=torch.long, device=dev)
                sp = torch.zeros((1,), dtype=torch.long, device=dev)
                idx = torch.zeros((1,), dtype=torch.long, device=dev)
                buf = torch.zeros((NEW,), dtype=torch.long, device=dev)
                t0 = prefill()

                def full():
                    lg = model(input_ids=st, past_key_values=cache, cache_position=sp, use_cache=True).logits
                    nxt = lg[:, -1].argmax(-1)
                    buf.index_copy_(0, idx, nxt)
                    st.copy_(nxt.view(1, 1)); sp.add_(1); idx.add_(1)
                    return lg

                def arm():
                    st.copy_(t0.view(1, 1)); sp.fill_(N_PROMPT); idx.fill_(1); buf.zero_(); buf[0] = t0[0]

                arm()
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(3):
                        full()
                torch.cuda.current_stream().wait_stream(s)
                sync()
                reset_counters(N_PROMPT)
                arm()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    full()
                times = []
                for _ in range(1 + REPEAT):  # first run = warm-up
                    reset_counters(N_PROMPT)
                    arm()
                    sync(); t = time.perf_counter()
                    for _ in range(NEW - 1):
                        g.replay()
                    sync()
                    times.append((time.perf_counter() - t) * 1000 / (NEW - 1))
                del g
                return buf.tolist(), statistics.median(times[1:])

            pos = torch.arange(N_PROMPT, device=dev)
            logits_orig = model(input_ids=ids, use_cache=False).logits[0, -8:].float()
            o1_toks, o1 = full_graph_decode()
            log(f"F original #1: {o1:6.2f} ms/step | tokens vs generate: first diff {first_diff(o1_toks, ref)}")
            restore = merge_projections(model)
            logits_merged = model(input_ids=ids, use_cache=False).logits[0, -8:].float()
            ldiff = float((logits_merged - logits_orig).abs().max())
            log(f"F merged: max |logit diff| vs original on the prompt = {ldiff:.4f}")
            m_toks, m = full_graph_decode()
            log(f"F merged    : {m:6.2f} ms/step | tokens vs generate: first diff {first_diff(m_toks, ref)}")
            restore()
            o2_toks, o2 = full_graph_decode()
            log(f"F original #2: {o2:6.2f} ms/step | tokens vs generate: first diff {first_diff(o2_toks, ref)}")
            base = (o1 + o2) / 2
            results.update(original_ms=[round(o1, 2), round(o2, 2)], merged_ms=round(m, 2),
                           saved_ms=round(base - m, 2), merged_logit_diff=round(ldiff, 4),
                           merged_first_diff=first_diff(m_toks, ref))
            log(f"RESULT F: original {o1:.2f}/{o2:.2f} -> merged {m:.2f} ms/step "
                f"(saved {base - m:.2f} ms, {base / m:.2f}x) | merged tokens first diff vs generate: "
                f"{first_diff(m_toks, ref)}")

        elif stage == "E":
            # E4: where do the remaining ~9 ms per graph-decoded token go?
            def capture(full_step):
                """Fresh cache -> prefill -> warm-up -> counter reset -> capture.
                full_step=False: graph = model forward only (as in C).
                full_step=True : graph = forward + argmax + feed token back + record token,
                                 so the Python loop is ONLY g.replay()."""
                nonlocal cache
                cache = new_cache()
                st = torch.zeros((1, 1), dtype=torch.long, device=dev)
                sp = torch.zeros((1,), dtype=torch.long, device=dev)
                idx = torch.zeros((1,), dtype=torch.long, device=dev)
                buf = torch.zeros((NEW,), dtype=torch.long, device=dev)
                t0 = prefill()

                def fwd():
                    return model(input_ids=st, past_key_values=cache, cache_position=sp, use_cache=True).logits

                def full():
                    lg = fwd()
                    nxt = lg[:, -1].argmax(-1)
                    buf.index_copy_(0, idx, nxt)
                    st.copy_(nxt.view(1, 1))
                    sp.add_(1)
                    idx.add_(1)
                    return lg

                body = full if full_step else fwd
                st.copy_(t0.view(1, 1)); sp.fill_(N_PROMPT); idx.fill_(1); buf.zero_(); buf[0] = t0[0]
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(3):
                        body()
                torch.cuda.current_stream().wait_stream(s)
                sync()
                reset_counters(N_PROMPT)
                st.copy_(t0.view(1, 1)); sp.fill_(N_PROMPT); idx.fill_(1); buf.zero_(); buf[0] = t0[0]
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    out = body()
                sync()
                return g, out, st, sp, idx, buf, t0

            # 1. forward-only graph, replays back to back (no argmax/copies between): pure GPU step cost
            g, out, st, sp, idx, buf, t0 = capture(False)
            st.copy_(t0.view(1, 1)); sp.fill_(N_PROMPT)
            sync(); t = time.perf_counter()
            for _ in range(NEW - 1):
                g.replay()
            sync()
            replay_only = (time.perf_counter() - t) * 1000 / (NEW - 1)
            log(f"E1 replay only (forward graph, nothing between): {replay_only:6.2f} ms/step")
            del g

            # 2. C-style loop: forward graph + argmax + 2 copies in Python between replays
            g, out, st, sp, idx, buf, t0 = capture(False)
            st.copy_(t0.view(1, 1)); sp.fill_(N_PROMPT)
            sync(); t = time.perf_counter()
            toks = [t0]
            for _ in range(NEW - 1):
                g.replay()
                cur = out[:, -1].argmax(-1)
                toks.append(cur)
                st.copy_(cur.view(1, 1))
                sp.add_(1)
            sync()
            c_style = (time.perf_counter() - t) * 1000 / (NEW - 1)
            c_ok = first_diff(torch.cat(toks).tolist(), ref)
            log(f"E2 C-style loop (as stage C): {c_style:6.2f} ms/step | tokens vs generate: first diff {c_ok}")
            del g

            # 3. whole step inside the graph: loop is only g.replay()
            g, out, st, sp, idx, buf, t0 = capture(True)
            sync(); t = time.perf_counter()
            for _ in range(NEW - 1):
                g.replay()
            sync()
            full_graph = (time.perf_counter() - t) * 1000 / (NEW - 1)
            f_ok = first_diff(buf.tolist(), ref)
            log(f"E3 full step in graph: {full_graph:6.2f} ms/step | tokens vs generate: first diff {f_ok} "
                f"(None = all {NEW} identical)")
            del g

            # 4. realistic memory-bandwidth floor: read ~1 GiB once, measure achieved GB/s
            x = torch.ones(512 * 2**20, dtype=torch.float16, device=dev)  # 1 GiB
            for _ in range(3):
                x.sum()
            sync(); t = time.perf_counter()
            for _ in range(10):
                x.sum()
            sync()
            sec = (time.perf_counter() - t) / 10
            gbs = x.numel() * 2 / sec / 1e9
            weights_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
            floor = weights_bytes / (gbs * 1e9) * 1000
            log(f"E4 achieved read bandwidth: {gbs:6.1f} GB/s (spec 640) -> weights-read floor {floor:5.2f} ms/step")
            del x

            results.update(replay_only_ms=round(replay_only, 2), c_style_ms=round(c_style, 2),
                           c_style_first_diff=c_ok, full_graph_ms=round(full_graph, 2), full_graph_first_diff=f_ok,
                           achieved_read_GBps=round(gbs, 1), weights_read_floor_ms=round(floor, 2))
            log(f"RESULT E: replay-only {replay_only:.2f} | C-style {c_style:.2f} | full-graph {full_graph:.2f} | "
                f"floor {floor:.2f} ms/step")

            # 5. optional: profiler table (may show CPU only on Windows ROCm)
            try:
                g, out, st, sp, idx, buf, t0 = capture(True)
                from torch.profiler import profile, ProfilerActivity
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                    for _ in range(5):
                        g.replay()
                    sync()
                log("PROFILE (top 12 by device time):")
                print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=12), flush=True)
            except Exception as ex:
                log(f"profiler not available: {type(ex).__name__}: {str(ex).splitlines()[0][:200]}")

        elif stage == "D":
            # Correctness check for C: does ONE graph replay give the same logits as ONE eager
            # step, from the same cache state and the same input token (teacher forcing)?
            K = 8
            cache = new_cache()
            t0 = prefill()
            cur, pos, eager_logits, etoks = t0, torch.tensor([N_PROMPT], device=dev), [], [t0]
            for _ in range(K):
                lg = model(input_ids=cur.view(1, 1), past_key_values=cache, cache_position=pos, use_cache=True).logits
                eager_logits.append(lg[0, -1].float().clone())
                cur = lg[:, -1].argmax(-1)
                etoks.append(cur)
                pos = pos + 1
            cache = new_cache()
            prefill()
            static_tok = torch.zeros((1, 1), dtype=torch.long, device=dev)
            static_pos = torch.zeros((1,), dtype=torch.long, device=dev)

            def step():
                return model(input_ids=static_tok, past_key_values=cache, cache_position=static_pos, use_cache=True).logits

            static_tok.copy_(t0.view(1, 1))
            static_pos.fill_(N_PROMPT)
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    step()
            torch.cuda.current_stream().wait_stream(s)
            sync()
            reset_counters(N_PROMPT)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                static_logits = step()
            rows = []
            for k in range(K):
                static_tok.copy_(etoks[k].view(1, 1))
                static_pos.fill_(N_PROMPT + k)
                g.replay()
                gl = static_logits[0, -1].float()
                el = eager_logits[k]
                top2 = torch.topk(el, 2).values
                row = {"step": k, "max_abs_logit_diff": round(float((gl - el).abs().max()), 4),
                       "same_argmax": bool(gl.argmax() == el.argmax()),
                       "eager_top1_minus_top2": round(float(top2[0] - top2[1]), 4)}
                rows.append(row)
                log(f"D step {k}: max |logit diff| = {row['max_abs_logit_diff']:.4f} | same top token: "
                    f"{row['same_argmax']} | eager margin top1-top2 = {row['eager_top1_minus_top2']:.4f}")
            results["steps"] = rows
            log("RESULT D: tiny diffs + same top tokens = rounding only; large diffs = the graph is computing something different")

        else:  # stage C
            # No cache reuse: each repeat = fresh cache -> prefill -> capture -> decode once.
            def graph_run(i):
                nonlocal cache
                cache = new_cache()
                static_tok = torch.zeros((1, 1), dtype=torch.long, device=dev)
                static_pos = torch.zeros((1,), dtype=torch.long, device=dev)

                def step():
                    return model(input_ids=static_tok, past_key_values=cache,
                                 cache_position=static_pos, use_cache=True).logits

                t0 = prefill()
                static_tok.copy_(t0.view(1, 1))
                static_pos.fill_(N_PROMPT)
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(3):  # warm-up writes position N_PROMPT only; real decode overwrites it
                        step()
                torch.cuda.current_stream().wait_stream(s)
                sync()
                reset_counters(N_PROMPT)
                log(f"C run {i}: capturing graph...")
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    static_logits = step()
                sync()
                log(f"C run {i}: captured, decoding...")
                static_tok.copy_(t0.view(1, 1))
                static_pos.fill_(N_PROMPT)
                sync()
                start = time.perf_counter()
                toks = [t0]
                for _ in range(NEW - 1):
                    g.replay()
                    cur = static_logits[:, -1].argmax(-1)
                    toks.append(cur)
                    static_tok.copy_(cur.view(1, 1))
                    static_pos.add_(1)
                sync()
                ms = (time.perf_counter() - start) * 1000 / (NEW - 1)
                del g
                log(f"C run {i}: {ms:.2f} ms/step")
                return torch.cat(toks).tolist(), ms

            try:
                graph_run(0)  # warm-up run (not timed)
                runs = [graph_run(i + 1) for i in range(REPEAT)]
                toks, ms = runs[-1][0], statistics.median(r[1] for r in runs)
                results.update(graph_works=True, ms_per_step=round(ms, 2), first_diff_vs_ref=first_diff(toks, ref),
                               text_graph=tok.decode(toks[:40]), text_ref=tok.decode(ref[:40]))
                log(f"C text (graph)    : {tok.decode(toks[:40])!r}")
                log(f"C text (generate) : {tok.decode(ref[:40])!r}")
                log(f"RESULT C. manual_graph: {ms:7.2f} ms/step ({1000 / ms:6.1f} tok/s) | "
                    f"first token difference vs generate: {first_diff(toks, ref)} (None = all {NEW} identical)")
            except Exception as ex:
                results.update(graph_works=False, graph_error=f"{type(ex).__name__}: {str(ex).splitlines()[0][:300]}")
                log(f"RESULT C. manual_graph: FAILED -> {results['graph_error']}")

    results["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    tag = f"{MODEL_ID.split('/')[-1]}_N{N_PROMPT}"
    out = HERE / "results" / f"E2_stage_{stage}_{tag}.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log(f"saved {out.name}")
    # E2 v3: every stage finished but the process then hung on exit (Python/ROCm teardown on
    # Windows) until the watchdog killed it. Results are saved, so exit immediately.
    watchdog.cancel()
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
