"""SD7: adaptive speculation + prompt-lookup drafting (pre-registered: docs/PREREG_SD7.md). Built on SD3/SD6 (below).

Target: Qwen3-1.7B fp16, best stack (fused norms, merged projections, fused q/k-norm-RoPE, split-KV attention v3 for
single tokens, small-op fusions, add+RMSNorm v2). Draft: Qwen3-0.6B, same stack + REAL int4-g32 kernel (scale search).
Per round: one captured graph runs the draft's k single-token steps (proposals stay on the GPU); one captured graph runs
the target over a fixed k+1 tokens (verify); the CPU reads proposals + target picks (one small sync), accepts the longest
matching prefix plus the target's own next token, and rolls both StaticCaches back by setting their counters.
If all k proposals were accepted, a 1-step draft graph ingests the last proposal before the next round.
Baseline: the target alone in the same stack, 8 decode steps per graph (S9), same run. Output tokens compared with it.
Prior art: speculative decoding (Leviathan et al. 2023; Chen et al. 2023).

Run: .venv-gpu\\Scripts\\python.exe -u sd3.py    (writes results/SD3.json)
"""
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
import graph_decode as gd  # noqa: E402
import sampler  # noqa: E402   SM1: exact sampling (SD_TEMP, SD_SEED); SD_TEMP=0 = greedy as before
from inference_benchmark import make_prompt_ids  # noqa: E402

NEWT = int(os.environ.get("SD_NEWT", "96"))
PROMPTSET = os.environ.get("SD_PROMPTSET", "b6")   # "b6": the 11 shared prompts of the B6 comparison
MAXL = 1024 + NEWT + 24 + (40 if os.environ.get("FW2") == "1" else 0)   # FW2: room for 32-token verifies
KS = tuple(int(x) for x in os.environ.get("SD_KS", "2,3").split(","))     # draft lengths to test
OUT = os.environ.get("SD_OUT", "SD7b")
OVERLAP = os.environ.get("OVERLAP", "0") == "1"   # INV1: time draft + verify graphs run at the same time on two streams
BI = os.environ.get("BI", "0") == "1"      # BI1: batch-invariant target matmuls (kernel_bi.py)
MQ = os.environ.get("MQ", "1") == "1"      # SD6: multi-token verify kernels (kernels_v6.py) instead of PyTorch fallbacks


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


@torch.no_grad()
def main():
    torch.set_grad_enabled(False)
    wd = threading.Timer(int(os.environ.get("STAGE_TIMEOUT_S", "1200")), lambda: (log("RESULT SD7b: TIMEOUT"), os._exit(3)))
    wd.daemon = True
    wd.start()
    from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
    from transformers.models.qwen3 import modeling_qwen3 as mq
    import fuse_small
    import kernels_v2
    import kernels_v3
    import kernel_int4
    import kernel_decode_attn as kda
    dev = torch.device("cuda")
    TGT = os.environ.get("SD_TARGET", "Qwen/Qwen3-1.7B")
    tok = AutoTokenizer.from_pretrained(TGT)
    T = AutoModelForCausalLM.from_pretrained(TGT, dtype=torch.float16, attn_implementation="sdpa").to(dev).eval()
    D = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float16, attn_implementation="sdpa").to(dev).eval()
    for m in (T, D):
        m.requires_grad_(False)                                   # SD3 leak fix: no autograd history in caches
        gd.fuse_rmsnorms(m)
        gd.merge_projections(m)
        gd.fuse_attention(m, merged=True)
    if os.environ.get("FREE_DUP", "0") == "1":          # S4B: free the target's original q/k/v/gate/up weights,
        for layer in T.model.layers:                    # which merge/fuse copied into _w_qkv / _w_gu (saves GBs)
            for lin in (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj,
                        layer.mlp.gate_proj, layer.mlp.up_proj):
                lin.weight.data = torch.empty(0, dtype=lin.weight.dtype, device=dev)
        torch.cuda.empty_cache()
        log(f"FREE_DUP: target duplicates freed, allocated {torch.cuda.memory_allocated() / 2**30:.2f} GiB")
    kda.get_kernels(); kernels_v3.kernels(); kernels_v2.kernels()
    kernels_v3.SPLITS = 16
    kda.decode_attn = kernels_v3.decode_attn
    sdpa_fn = mq.ALL_ATTENTION_FUNCTIONS.get_interface("sdpa", mq.eager_attention_forward)

    import kernels_v6
    if MQ:
        kernels_v6.kernels()

    import kernels_v7                                   # S4B: any query-heads-per-kv-head ratio (Qwen3-4B: 4)

    def our_att(module, q, k, v, mask, dropout=0.0, scaling=None, sliding_window=None, **kw):
        g2 = q.shape[1] == 2 * k.shape[1]               # 1.7B / 0.6B: validated v3/v6 kernels; else generic v7
        if MQ and 1 < q.shape[2] <= 32:
            return (kernels_v6.attn_mq if g2 else kernels_v7.attn_mq)(q, k, v, module._len, scaling), None
        if q.shape[2] != 1:
            return sdpa_fn(module, q, k, v, mask, dropout=dropout, scaling=scaling, sliding_window=sliding_window, **kw)
        return (kda.decode_attn if g2 else kernels_v7.decode_attn)(q, k, v, module._len, scaling), None
    mq.ALL_ATTENTION_FUNCTIONS.get_interface = lambda *a, **k: our_att
    fuse_small.install(T)
    fuse_small.install(D)
    fuse_small.add_rmsnorm = kernels_v2.add_rmsnorm
    if MQ:
        kernels_v6.install_kv_write(max_n=32)
    dws = []
    for layer in D.model.layers:
        dws += [layer.self_attn._w_qkv, layer.self_attn.o_proj.weight, layer.mlp._w_gu, layer.mlp.down_proj.weight]
    kernel_int4.install(dws + [D.lm_head.weight], search=True)
    worst = 0.0                                                  # SD7 self-test of the multi-row int4 kernel
    for w in (dws[0], dws[1], dws[2], dws[3], D.lm_head.weight):
        pk, sc = kernel_int4._registry[id(w)]
        K = pk.shape[1] * 2
        for nrow in (2, 5, 8):
            x = torch.randn(1, nrow, K, device=dev, generator=torch.Generator(device=dev).manual_seed(nrow)).half() * 0.1
            got = kernel_int4.gemv_n(x, pk, sc).float()
            ref = x.float() @ kernel_int4.dequant(pk, sc).float().T
            worst = max(worst, float((got - ref).abs().max() / (ref.abs().max() + 1e-6)))
    log(f"SD7 int4 multi-row kernel self-test: worst relative error {worst:.2e} -> {'CORRECT' if worst < 2e-2 else 'WRONG'}")
    if os.environ.get("FREE_DUP", "0") == "1":          # S4B-diag: the draft's fp16 copies of int4 weights are unused
        for layer in D.model.layers:
            for lin in (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj,
                        layer.mlp.gate_proj, layer.mlp.up_proj):
                lin.weight.data = torch.empty(0, dtype=lin.weight.dtype, device=dev)
        for w in dws:                                    # lm_head is tied to the embeddings in 0.6B: keep it
            w.data = torch.empty(0, dtype=w.dtype, device=dev)
        torch.cuda.empty_cache()
    ar_stats = {}
    if BI:                                              # BI1: every target matmul with 1..32 rows -> one kernel whose
        import kernel_bi                                # per-row summation order does not depend on the row count
        tw = []
        for layer in T.model.layers:
            tw += [layer.self_attn._w_qkv, layer.self_attn.o_proj.weight, layer.mlp._w_gu, layer.mlp.down_proj.weight]
        tw.append(T.lm_head.weight)
        kernel_bi.install(tw)
        _ar = kernels_v2.add_rmsnorm

        def ar_logged(a, b, w, eps):                    # record which add+RMSNorm kernel each row count takes
            n_ = a.shape[-1]
            v2 = (n_ % 256 == 0 and n_ // 256 in (4, 8) and
                  not any(t.contiguous().data_ptr() % 16 for t in (a, b)) and w.data_ptr() % 16 == 0)
            key = (a.numel() // n_, "v2" if v2 else "v1")
            ar_stats[key] = ar_stats.get(key, 0) + 1
            return _ar(a, b, w, eps)
        fuse_small.add_rmsnorm = ar_logged
        log(f"BI1: {len(tw)} target matrices routed to kernel_bi (1..{kernel_bi.NMAX} rows); prefill uses PyTorch")
    fr, tot = torch.cuda.mem_get_info()
    log(f"VRAM after load: PyTorch allocated {torch.cuda.memory_allocated() / 2**30:.2f} GiB | device free {fr / 2**30:.2f} "
        f"of {tot / 2**30:.2f} GiB")
    log(f"models loaded and patched (target fp16 stack, draft int4-g32 search) | multi-token verify kernels (MQ): {MQ}")
    log(f"grad enabled: {torch.is_grad_enabled()} | any param requires grad: "
        f"{any(p.requires_grad for m in (T, D) for p in m.parameters())}")

    cT = StaticCache(config=T.config, max_cache_len=MAXL)
    cD = StaticCache(config=D.config, max_cache_len=MAXL)

    def log_mem(where):
        torch.cuda.synchronize()
        log(f"MEM {where}: allocated {torch.cuda.memory_allocated() / 2**30:.2f} GiB | reserved "
            f"{torch.cuda.memory_reserved() / 2**30:.2f} GiB | peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")

    def setc(cache, n):
        for layer in cache.layers:
            layer.cumulative_length.fill_(n)

    def prefill(m, cache, ids):
        setc(cache, 0) if all(getattr(l_, "is_initialized", False) for l_ in cache.layers) else None
        P = ids.shape[1]
        lg = m(input_ids=ids, past_key_values=cache, cache_position=torch.arange(P, device=dev), use_cache=True,
               logits_to_keep=1).logits
        for i, layer in enumerate(m.model.layers):
            layer.self_attn._len = cache.layers[i].cumulative_length
        first = int(sampler.pick(lg[0, -1:], torch.tensor([P - 1], device=dev))[0])   # SM1: greedy if SD_TEMP=0
        del lg
        return first

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

    if PROMPTSET == "b6":
        import b6_prompts
        prompts = [(lab, torch.tensor([ids], device=dev)) for lab, ids in b6_prompts.prompts(tok)]
    elif PROMPTSET == "bi3":                            # BI3: 50 varied prompts (bi3_prompts.py)
        import bi3_prompts
        prompts = [(lab, torch.tensor([ids], device=dev)) for lab, ids in bi3_prompts.prompts(tok)]
    else:
      prompts = [("filler 1024 tokens", make_prompt_ids(tok, 1024, dev))] + [
        (t, tok(t, return_tensors="pt").input_ids.to(dev)) for t in
        ["The capital of France is", "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
         "List of the planets in our solar system, in order from the Sun:\n1.",
         "Once upon a time, in a small village by the sea,"]]
    prefill(T, cT, prompts[0][1]); prefill(D, cD, prompts[0][1])      # initialise caches (lazy) before capture

    # ---- target-alone baseline: 8 single-token steps per graph
    t_st = torch.zeros((1, 1), dtype=torch.long, device=dev)
    t_sp = torch.zeros((1,), dtype=torch.long, device=dev)
    t_idx = torch.zeros((1,), dtype=torch.long, device=dev)
    t_out = torch.zeros((NEWT + 8,), dtype=torch.long, device=dev)

    def t_step():
        lg = T(input_ids=t_st, past_key_values=cT, cache_position=t_sp, use_cache=True).logits
        nxt = sampler.pick(lg[:, -1], t_sp)
        t_out.index_copy_(0, t_idx, nxt)
        t_st.copy_(nxt.view(1, 1)); t_sp.add_(1); t_idx.add_(1)

    P0 = prompts[0][1].shape[1]
    G8 = capture(lambda: [t_step() for _ in range(8)], lambda: (setc(cT, P0), t_sp.fill_(P0), t_idx.zero_()))
    log("baseline graph captured")

    def replay_ms(g, before, reps=20):
        ts = []
        for _ in range(reps):
            before(); torch.cuda.synchronize()
            t = time.perf_counter(); g.replay(); torch.cuda.synchronize()
            ts.append((time.perf_counter() - t) * 1000)
        return statistics.median(ts)

    g8 = replay_ms(G8, lambda: (setc(cT, P0), t_sp.fill_(P0), t_idx.zero_()))
    try:                                                          # record the GPU state with every run
        import gpu_state
        ck = torch.cuda._compile_kernel(gpu_state.CLK_SRC, "clk")
        cbuf = torch.zeros(4, dtype=torch.int64, device=dev)
        G8.replay(); G8.replay()
        log(f"CLOCK shader clock right after target graphs: {gpu_state.clock_mhz(ck, cbuf):.0f} MHz")
    except Exception as ex:
        log(f"CLOCK probe failed: {str(ex)[:200]}")
    log(f"SANITY target alone: one 8-token graph replay {g8:.2f} ms = {g8 / 8:.2f} ms/token (B5 measured ~6.6 ms/token)")

    def baseline(ids):
        first = prefill(T, cT, ids)
        P = ids.shape[1]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        setc(cT, P); t_st.fill_(first); t_sp.fill_(P); t_idx.zero_()
        for _ in range((NEWT - 1 + 7) // 8):
            G8.replay()
        toks = [first] + t_out[:NEWT - 1].tolist()
        dt = time.perf_counter() - t0
        return toks, dt * 1000 / (NEWT - 1)

    # ======================= SD7: adaptive speculation + prompt lookup =======================
    KSET = (1, 2, 3, 5, 7)
    VSET = (1, 2, 3, 4, 6, 8)                      # verify sizes = k + 1 (and 1 = a plain decode step)
    props = torch.zeros((7,), dtype=torch.long, device=dev)
    d_st = torch.zeros((1, 1), dtype=torch.long, device=dev)
    d_sp = torch.zeros((1,), dtype=torch.long, device=dev)
    d_idx = torch.zeros((1,), dtype=torch.long, device=dev)
    v_ids = torch.zeros((1, 8), dtype=torch.long, device=dev)
    v_sp = torch.zeros((8,), dtype=torch.long, device=dev)
    base = torch.arange(8, device=dev)
    preds = torch.zeros((8,), dtype=torch.long, device=dev)
    di_ids = torch.zeros((1, 8), dtype=torch.long, device=dev)
    di_sp = torch.zeros((8,), dtype=torch.long, device=dev)

    def d_step():
        lg = D(input_ids=d_st, past_key_values=cD, cache_position=d_sp, use_cache=True).logits
        nxt = sampler.pick(lg[:, -1], d_sp)                 # SM1: same noise as the target (shared randomness)
        props.index_copy_(0, d_idx, nxt)
        d_st.copy_(nxt.view(1, 1)); d_sp.add_(1); d_idx.add_(1)

    def v_step(n):
        lg = T(input_ids=v_ids[:, :n], past_key_values=cT, cache_position=v_sp[:n], use_cache=True).logits
        preds[:n].copy_(sampler.pick(lg[0], v_sp[:n]))

    def ingest(m):
        D(input_ids=di_ids[:, :m], past_key_values=cD, cache_position=di_sp[:m], use_cache=True, logits_to_keep=1)

    Gd, Gv, GI = {}, {}, {}
    for k in KSET:
        Gd[k] = capture(lambda k=k: [d_step() for _ in range(k)], lambda: (setc(cD, P0), d_sp.fill_(P0), d_idx.zero_()))
    for n in VSET:
        Gv[n] = capture(lambda n=n: v_step(n), lambda: (setc(cT, P0), torch.add(base, P0, out=v_sp)))
    for m in range(1, 9):
        GI[m] = capture(lambda m=m: ingest(m), lambda: (setc(cD, P0), torch.add(base, P0, out=di_sp)))
    cost = {}
    for k in KSET:
        cost[("d", k)] = replay_ms(Gd[k], lambda: (setc(cD, P0), d_sp.fill_(P0), d_idx.zero_()), 10) + \
            replay_ms(Gv[k + 1], lambda: (setc(cT, P0), torch.add(base, P0, out=v_sp)), 10)
    for n in VSET:
        cost[("p", n - 1) if n > 1 else ("plain", 0)] = replay_ms(Gv[n], lambda: (setc(cT, P0), torch.add(base, P0, out=v_sp)), 10)
    ci = {m: replay_ms(GI[m], lambda: (setc(cD, P0), torch.add(base, P0, out=di_sp)), 10) for m in range(1, 9)}
    log("SD7 graphs captured | start-up costs (ms): " + " ".join(f"{a}{b}={v:.2f}" for (a, b), v in sorted(cost.items()))
        + " | ingest " + " ".join(f"{m}:{v:.2f}" for m, v in ci.items()))
    log_mem("after SD7 capture")
    fr, tot = torch.cuda.mem_get_info()
    log(f"VRAM after capture: device free {fr / 2**30:.2f} of {tot / 2**30:.2f} GiB")
    ctxd, ctxi, ctxv = [], [], []
    for _ in range(15):                                   # draft step / ingest timed right after a verify graph
        setc(cT, P0); torch.add(base, P0, out=v_sp); torch.cuda.synchronize()
        t = time.perf_counter(); Gv[8].replay(); torch.cuda.synchronize(); ctxv.append((time.perf_counter() - t) * 1000)
        setc(cD, P0); d_sp.fill_(P0); d_idx.zero_(); torch.cuda.synchronize()
        t = time.perf_counter(); Gd[1].replay(); torch.cuda.synchronize(); ctxd.append((time.perf_counter() - t) * 1000)
        setc(cD, P0); torch.add(base, P0, out=di_sp); torch.cuda.synchronize()
        t = time.perf_counter(); GI[1].replay(); torch.cuda.synchronize(); ctxi.append((time.perf_counter() - t) * 1000)
    log(f"DIAG in context: verify 8 tokens {statistics.median(ctxv):.2f} ms | draft 1 step {statistics.median(ctxd):.2f} ms | "
        f"draft ingest 1 token {statistics.median(ctxi):.2f} ms")
    if os.environ.get("SD7_DIAG_ONLY") == "1":
        log("RESULT DIAG done")
        wd.cancel(); sys.stdout.flush(); os._exit(0)
    PLD_SIZES = (7, 5, 3, 2, 1)                    # proposal lengths with a captured verify graph (m + 1 in VSET)

    def E(a, k):
        return (1 - a ** (k + 1)) / (1 - a) if a < 0.9999 else k + 1

    def gen(ids, mode):
        """mode: fixed7 | adapt | adapt_pld. Returns tokens, ms/token, rounds, action counts."""
        P = ids.shape[1]
        first = prefill(T, cT, ids)
        prefill(D, cD, ids)
        ctx = ids[0].tolist() + [first]            # everything committed; ctx[-1] is pending (not yet in target cache)
        Pt = Pd = P
        idx = {g: {} for g in (2, 3, 4)}         # n-gram -> end position of its latest occurrence
        prev_idx = {g: {} for g in (2, 3, 4)}    # n-gram -> end position of the occurrence before that
        for i in range(len(ctx)):
            for g in (2, 3, 4):
                if i + 1 >= g:
                    key = tuple(ctx[i + 1 - g:i + 1])
                    if key in idx[g]:
                        prev_idx[g][key] = idx[g][key]
                    idx[g][key] = i
        st = {"d": [1.5, 0.5], "p": [1.5, 0.5]}    # decayed success / failure counts + prior (alpha ~0.75)
        cst = dict(cost)
        acts = {}
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        def add_tokens(new):
            for tkn in new:
                ctx.append(tkn)
                i = len(ctx) - 1
                for g in (2, 3, 4):
                    if i + 1 >= g:
                        key = tuple(ctx[i + 1 - g:i + 1])
                        if key in idx[g]:
                            prev_idx[g][key] = idx[g][key]
                        idx[g][key] = i

        def pld():
            L = len(ctx)
            for g in (4, 3, 2):
                key = tuple(ctx[L - g:])
                j = prev_idx[g].get(key) if idx[g].get(key) == L - 1 else idx[g].get(key)
                if j is not None and j < L - 1:
                    cont = ctx[j + 1:L][:7]
                    if cont:
                        m = next(s for s in PLD_SIZES if s <= len(cont))
                        return cont[:m]
            return None

        def catch_up():
            nonlocal Pd
            while Pd < Pt:
                m = min(8, Pt - Pd)
                di_ids[0, :m].copy_(torch.tensor(ctx[Pd:Pd + m], device=dev))
                setc(cD, Pd); torch.add(base, Pd, out=di_sp)
                GI[m].replay()
                Pd += m

        rounds = 0
        while len(ctx) - P < NEWT:
            pending = ctx[-1]
            lag = Pt - Pd
            alpha_d = st["d"][0] / sum(st["d"])
            alpha_p = st["p"][0] / sum(st["p"])
            cands = []
            if mode == "fixed7":
                cands = [(0, ("d", 7))]
            else:
                cu = sum(ci[min(8, lag - 8 * c)] for c in range((lag + 7) // 8)) if lag else 0.0
                cands.append((1 / cst[("plain", 0)], ("plain", 0)))
                for k in KSET:
                    cands.append((E(alpha_d, k) / (cst[("d", k)] + cu), ("d", k)))
                prop = pld() if mode == "adapt_pld" else None
                if prop:
                    m = len(prop)
                    cands.append((E(alpha_p, m) / cst[("p", m)], ("p", m)))
            _, act = max(cands)
            acts[act] = acts.get(act, 0) + 1
            ts = time.perf_counter()
            if act[0] == "d":
                k = act[1]
                catch_up()
                ts = time.perf_counter()
                setc(cD, Pd); d_st.fill_(pending); d_sp.fill_(Pd); d_idx.zero_()
                Gd[k].replay()
                v_ids[0, 0].fill_(pending); v_ids[0, 1:k + 1].copy_(props[:k])
                torch.add(base, Pt, out=v_sp); setc(cT, Pt)
                Gv[k + 1].replay()
                pr, pd = props[:k].tolist(), preds[:k + 1].tolist()
                n = 0
                while n < k and pr[n] == pd[n]:
                    n += 1
                new = pr[:n] + [pd[n]]
                Pt += 1 + n
                Pd += 1 + min(n, k - 1)
                if n == k:
                    di_ids[0, 0].fill_(pr[k - 1]); setc(cD, Pd); torch.add(base, Pd, out=di_sp)
                    GI[1].replay()
                    Pd += 1
                st["d"] = [0.8 * st["d"][0] + n, 0.8 * st["d"][1] + (1 if n < k else 0)]
                st["p"] = [0.97 * st["p"][0] + 0.03 * 1.5, 0.97 * st["p"][1] + 0.03 * 0.5]
            elif act[0] == "p":
                m = act[1]
                v_ids[0, :m + 1].copy_(torch.tensor([pending] + prop, device=dev))
                torch.add(base, Pt, out=v_sp); setc(cT, Pt)
                Gv[m + 1].replay()
                pd = preds[:m + 1].tolist()
                n = 0
                while n < m and prop[n] == pd[n]:
                    n += 1
                new = prop[:n] + [pd[n]]
                Pt += 1 + n
                st["p"] = [0.8 * st["p"][0] + n, 0.8 * st["p"][1] + (1 if n < m else 0)]
                st["d"] = [0.97 * st["d"][0] + 0.03 * 1.5, 0.97 * st["d"][1] + 0.03 * 0.5]
            else:
                v_ids[0, 0].fill_(pending); torch.add(base, Pt, out=v_sp); setc(cT, Pt)
                Gv[1].replay()
                new = [int(preds[0])]
                Pt += 1
                st["d"] = [0.97 * st["d"][0] + 0.03 * 1.5, 0.97 * st["d"][1] + 0.03 * 0.5]
                st["p"] = [0.97 * st["p"][0] + 0.03 * 1.5, 0.97 * st["p"][1] + 0.03 * 0.5]
            cst[act] = 0.8 * cst[act] + 0.2 * (time.perf_counter() - ts) * 1000      # learn real round costs online
            add_tokens(new)
            rounds += 1
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        out = ctx[P:P + NEWT]
        return out, dt * 1000 / (NEWT - 1), rounds, acts

    # ======================= SD7b: fixed adaptive policy =======================
    G4 = capture(lambda: [t_step() for _ in range(4)], lambda: (setc(cT, P0), t_sp.fill_(P0), t_idx.zero_()))
    c_g4 = replay_ms(G4, lambda: (setc(cT, P0), t_sp.fill_(P0), t_idx.zero_()), 10)
    d_ctx = statistics.median(ctxd)
    log(f"SD7b: plain 4-token graph {c_g4:.2f} ms ({c_g4 / 4:.2f} ms/token) | draft step in context {d_ctx:.2f} ms")

    if os.environ.get("FW2") == "1":
        # ================= FW2: how much would a token tree gain? (measurement + exact counterfactual) =================
        props2 = torch.zeros((7,), dtype=torch.long, device=dev)

        def d_step2():
            lg = D(input_ids=d_st, past_key_values=cD, cache_position=d_sp, use_cache=True).logits
            top2 = torch.topk(lg[:, -1], 2, dim=-1).indices[0]
            props.index_copy_(0, d_idx, top2[:1])
            props2.index_copy_(0, d_idx, top2[1:2])
            d_st.copy_(top2[:1].view(1, 1)); d_sp.add_(1); d_idx.add_(1)
        Gd7b = capture(lambda: [d_step2() for _ in range(7)], lambda: (setc(cD, P0), d_sp.fill_(P0), d_idx.zero_()))
        wv = torch.zeros((1, 32), dtype=torch.long, device=dev)
        wsp = torch.zeros((32,), dtype=torch.long, device=dev)
        wbase = torch.arange(32, device=dev)

        def wv_step(n):
            T(input_ids=wv[:, :n], past_key_values=cT, cache_position=wsp[:n], use_cache=True)
        vcost = {}
        for n in (8, 12, 16, 24, 32):
            gw = capture(lambda n=n: wv_step(n), lambda: (setc(cT, P0), torch.add(wbase, P0, out=wsp)))
            tsamp = []
            for _ in range(15):                                         # right after a draft burst (realistic clock)
                setc(cD, P0); d_sp.fill_(P0); d_idx.zero_(); Gd[7].replay()
                setc(cT, P0); torch.add(wbase, P0, out=wsp); torch.cuda.synchronize()
                t = time.perf_counter(); gw.replay(); torch.cuda.synchronize()
                tsamp.append((time.perf_counter() - t) * 1000)
            vcost[n] = statistics.median(tsamp)
            del gw
        dsamp = []
        for _ in range(15):
            setc(cT, P0); torch.add(base, P0, out=v_sp); Gv[8].replay()
            setc(cD, P0); d_sp.fill_(P0); d_idx.zero_(); torch.cuda.synchronize()
            t = time.perf_counter(); Gd7b.replay(); torch.cuda.synchronize()
            dsamp.append((time.perf_counter() - t) * 1000)
        d7b = statistics.median(dsamp)
        log("FW2 verify cost (ms) by tokens: " + " | ".join(f"{n}: {v:.2f}" for n, v in vcost.items()) +
            f" | draft 7 steps with top-2: {d7b:.2f} ms")

        def pld_prop(ctx):
            L = len(ctx)
            for g in (4, 3, 2):
                key = ctx[L - g:]
                for j in range(L - 2, g - 2, -1):                       # latest earlier occurrence
                    if ctx[j - g + 1:j + 1] == key:
                        return ctx[j + 1:L][:7]
            return None

        fw = {"verify_ms": vcost, "draft7_top2_ms": d7b, "rows": []}
        for label, ids in prompts:
            P = ids.shape[1]
            first = prefill(T, cT, ids)
            prefill(D, cD, ids)
            ctx = ids[0].tolist() + [first]
            Pt = Pd = P
            log_rounds = []
            while len(ctx) - P < NEWT:
                pending = ctx[-1]
                L = len(ctx)
                pp = pld_prop(ctx)
                setc(cD, Pd); d_st.fill_(pending); d_sp.fill_(Pd); d_idx.zero_()
                Gd7b.replay()
                v_ids[0, 0].fill_(pending); v_ids[0, 1:8].copy_(props[:7])
                torch.add(base, Pt, out=v_sp); setc(cT, Pt)
                Gv[8].replay()
                pr, p2, pd = props[:7].tolist(), props2[:7].tolist(), preds[:8].tolist()
                n = 0
                while n < 7 and pr[n] == pd[n]:
                    n += 1
                new = pr[:n] + [pd[n]]
                Pt += 1 + n
                Pd += 1 + min(n, 6)
                if n == 7:
                    di_ids[0, 0].fill_(pr[6]); setc(cD, Pd); torch.add(base, Pd, out=di_sp)
                    GI[1].replay()
                    Pd += 1
                log_rounds.append((L, pr, p2, pp))
                ctx += new
            true = ctx
            chain, sib, pldb, tree = [], [], [], []
            for (L, pr, p2, pp) in log_rounds:
                fut = true[L:L + 16]
                n = 0
                while n < 7 and n < len(fut) and pr[n] == fut[n]:
                    n += 1
                c = n + 1
                s_ = n + 2 if (n < 7 and n < len(fut) and p2[n] == fut[n]) else c
                j = 0
                if pp:
                    while j < len(pp) and j < len(fut) and pp[j] == fut[j]:
                        j += 1
                pb = j + 1 if pp else 1
                chain.append(c); sib.append(max(c, s_)); pldb.append(max(c, pb)); tree.append(max(c, s_, pb))
            m = lambda v: statistics.mean(v)
            row = {"prompt": label, "rounds": len(log_rounds), "chain": m(chain), "chain_sib": m(sib),
                   "chain_pld": m(pldb), "chain_sib_pld": m(tree)}
            fw["rows"].append(row)
            log(f"FW2 {label[:24]:<24} tokens/round: chain {m(chain):.2f} | +siblings {m(sib):.2f} | +PLD branch "
                f"{m(pldb):.2f} | +both {m(tree):.2f}")
        rs = fw["rows"]
        def vc(n):                                                      # linear interpolation of verify cost
            ks = sorted(vcost)
            for a, b in zip(ks, ks[1:]):
                if a <= n <= b:
                    return vcost[a] + (vcost[b] - vcost[a]) * (n - a) / (b - a)
            return vcost[ks[-1]]
        variants = {"chain (fixed7)": ("chain", 8), "chain+siblings": ("chain_sib", 15),
                    "chain+PLD branch": ("chain_pld", 15), "chain+siblings+PLD": ("chain_sib_pld", 22)}
        out_lines = []
        for name, (key, nodes) in variants.items():
            tpr = statistics.mean(r[key] for r in rs)
            ms_round = d7b + vc(nodes) + 1.0                            # + ~1 ms: ingest/sync/host per round
            out_lines.append(f"{name}: {tpr:.2f} tok/round, verify {nodes} tokens {vc(nodes):.2f} ms -> modeled "
                             f"{1000 * tpr / ms_round:.0f} tok/s")
            fw.setdefault("modeled", {})[name] = {"tok_round": tpr, "nodes": nodes, "tok_s": 1000 * tpr / ms_round}
        log("RESULT FW2 (A, from measured parts): " + " || ".join(out_lines))
        fw["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        (HERE / "results" / f"{OUT}.json").write_text(json.dumps(fw, indent=1), encoding="utf-8")
        wd.cancel(); sys.stdout.flush(); os._exit(0)

    def gen2(ids, use_pld):
        """SD7b policy. Per action: expected tokens/round (draft lengths learned counterfactually: a round with k
        guesses also tells what any k' <= k would have produced) divided by learned ms/round. Plain decoding runs
        4 tokens per graph. Prompt lookup only for >= 3-token matches with >= 3 tokens to copy, cautious prior."""
        P = ids.shape[1]
        first = prefill(T, cT, ids)
        prefill(D, cD, ids)
        ctx = ids[0].tolist() + [first]
        Pt = Pd = P
        idx = {g: {} for g in (3, 4)}
        prev_idx = {g: {} for g in (3, 4)}

        def add_tokens(new, start=None):
            for tkn in new:
                ctx.append(tkn)
                i = len(ctx) - 1
                for g in (3, 4):
                    if i + 1 >= g:
                        key = tuple(ctx[i + 1 - g:i + 1])
                        if key in idx[g]:
                            prev_idx[g][key] = idx[g][key]
                        idx[g][key] = i
        for i in range(len(ctx)):
            for g in (3, 4):
                if i + 1 >= g:
                    key = tuple(ctx[i + 1 - g:i + 1])
                    if key in idx[g]:
                        prev_idx[g][key] = idx[g][key]
                    idx[g][key] = i

        def pld():
            L = len(ctx)
            for g in (4, 3):
                key = tuple(ctx[L - g:])
                j = prev_idx[g].get(key) if idx[g].get(key) == L - 1 else idx[g].get(key)
                if j is not None and j < L - 1:
                    cont = ctx[j + 1:L][:7]
                    if len(cont) >= 3:
                        m = next(s_ for s_ in PLD_SIZES if s_ <= len(cont))
                        return cont[:m]
            return None

        def catch_up():
            nonlocal Pd
            while Pd < Pt:
                m = min(8, Pt - Pd)
                di_ids[0, :m].copy_(torch.tensor(ctx[Pd:Pd + m], device=dev))
                setc(cD, Pd); torch.add(base, Pd, out=di_sp)
                GI[m].replay()
                Pd += m

        Tk = {k: E(0.75, k) for k in KSET}
        Tp = {m: E(0.5, m) for m in PLD_SIZES}
        C = {("d", k): k * d_ctx + cost[("p", k)] + 0.3 for k in KSET}
        C.update({("p", m): cost[("p", m)] + 0.3 for m in PLD_SIZES})
        C[("plain", 4)] = c_g4 + 0.3
        C[("plain", 1)] = cost[("plain", 0)] + 0.3
        since_draft, full_streak = 0, 0
        acts = {}
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        rounds = 0
        while len(ctx) - P < NEWT:
            pending = ctx[-1]
            remaining = NEWT - (len(ctx) - P)
            lag = Pt - Pd
            cu = sum(ci[min(8, lag - 8 * c)] for c in range((lag + 7) // 8)) if lag else 0.0
            pa = ("plain", 4) if remaining >= 4 else ("plain", 1)
            cands = [(pa[1] / C[pa], pa)]
            for k in KSET:
                cands.append((Tk[k] / (C[("d", k)] + cu), ("d", k)))
            prop = pld() if use_pld else None
            if prop:
                m = len(prop)
                cands.append((Tp[m] / C[("p", m)], ("p", m)))
            _, act = max(cands)
            if act[0] != "d" and since_draft >= 12:                       # exploration: re-try drafting
                act = ("d", 3)
            elif act[0] == "d" and full_streak >= 2 and act[1] < 7:        # all accepted twice: try longer
                act = ("d", KSET[KSET.index(act[1]) + 1])
            acts[act] = acts.get(act, 0) + 1
            if act[0] == "d":
                catch_up()
            ts = time.perf_counter()
            if act[0] == "d":
                k = act[1]
                setc(cD, Pd); d_st.fill_(pending); d_sp.fill_(Pd); d_idx.zero_()
                Gd[k].replay()
                v_ids[0, 0].fill_(pending); v_ids[0, 1:k + 1].copy_(props[:k])
                torch.add(base, Pt, out=v_sp); setc(cT, Pt)
                Gv[k + 1].replay()
                pr, pd = props[:k].tolist(), preds[:k + 1].tolist()
                n = 0
                while n < k and pr[n] == pd[n]:
                    n += 1
                new = pr[:n] + [pd[n]]
                Pt += 1 + n
                Pd += 1 + min(n, k - 1)
                if n == k:
                    di_ids[0, 0].fill_(pr[k - 1]); setc(cD, Pd); torch.add(base, Pd, out=di_sp)
                    GI[1].replay()
                    Pd += 1
                for k2 in KSET:                                            # counterfactual updates
                    if k2 <= k:
                        Tk[k2] = 0.75 * Tk[k2] + 0.25 * (min(n, k2) + 1)
                    elif n < k:
                        Tk[k2] = 0.75 * Tk[k2] + 0.25 * (n + 1)
                full_streak = full_streak + 1 if n == k else 0
                since_draft = 0
            elif act[0] == "p":
                m = act[1]
                v_ids[0, :m + 1].copy_(torch.tensor([pending] + prop, device=dev))
                torch.add(base, Pt, out=v_sp); setc(cT, Pt)
                Gv[m + 1].replay()
                pd = preds[:m + 1].tolist()
                n = 0
                while n < m and prop[n] == pd[n]:
                    n += 1
                new = prop[:n] + [pd[n]]
                Pt += 1 + n
                for m2 in PLD_SIZES:
                    if m2 <= m:
                        Tp[m2] = 0.75 * Tp[m2] + 0.25 * (min(n, m2) + 1)
                    elif n < m:
                        Tp[m2] = 0.75 * Tp[m2] + 0.25 * (n + 1)
                since_draft += 1
            elif act == ("plain", 4):
                t_st.fill_(pending); t_sp.fill_(Pt); t_idx.zero_(); setc(cT, Pt)
                G4.replay()
                new = t_out[:4].tolist()
                Pt += 4
                since_draft += 1
            else:
                v_ids[0, 0].fill_(pending); torch.add(base, Pt, out=v_sp); setc(cT, Pt)
                Gv[1].replay()
                new = [int(preds[0])]
                Pt += 1
                since_draft += 1
            C[act] = 0.8 * C[act] + 0.2 * (time.perf_counter() - ts) * 1000
            add_tokens(new)
            rounds += 1
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        return ctx[P:P + NEWT], dt * 1000 / (NEWT - 1), rounds, acts

    def gen3(ids, use_pld):
        """SD7c: the simplest rule. Draft k=7 by default. After 3 rounds in a row with <= 1 guess accepted, run 4 blocks
        of 4-token plain graphs (16 tokens), then probe k=7 again. Prompt lookup only for matches with >= 5 tokens to
        copy; after 2 failed copies in a row, no copying for 8 rounds."""
        P = ids.shape[1]
        first = prefill(T, cT, ids)
        prefill(D, cD, ids)
        ctx = ids[0].tolist() + [first]
        Pt = Pd = P
        idx = {g: {} for g in (3, 4)}
        prev_idx = {g: {} for g in (3, 4)}

        def index(i):
            for g in (3, 4):
                if i + 1 >= g:
                    key = tuple(ctx[i + 1 - g:i + 1])
                    if key in idx[g]:
                        prev_idx[g][key] = idx[g][key]
                    idx[g][key] = i
        for i in range(len(ctx)):
            index(i)

        def pld():
            L = len(ctx)
            for g in (4, 3):
                key = tuple(ctx[L - g:])
                j = prev_idx[g].get(key) if idx[g].get(key) == L - 1 else idx[g].get(key)
                if j is not None and j < L - 1:
                    cont = ctx[j + 1:L][:7]
                    if len(cont) >= 5:
                        return cont[:5] if len(cont) < 7 else cont[:7]
            return None

        def catch_up():
            nonlocal Pd
            while Pd < Pt:
                m = min(8, Pt - Pd)
                di_ids[0, :m].copy_(torch.tensor(ctx[Pd:Pd + m], device=dev))
                setc(cD, Pd); torch.add(base, Pd, out=di_sp)
                GI[m].replay()
                Pd += m

        low, plain_left, pfail, pld_off = 0, 0, 0, 0
        acts = {}
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        rounds = 0
        while len(ctx) - P < NEWT:
            pending = ctx[-1]
            remaining = NEWT - (len(ctx) - P)
            prop = pld() if (use_pld and pld_off == 0 and plain_left == 0) else None
            if plain_left > 0:
                act = ("plain", 4) if remaining >= 4 else ("plain", 1)
            elif prop:
                act = ("p", len(prop))
            else:
                act = ("d", 7)
            acts[act] = acts.get(act, 0) + 1
            pld_off = max(0, pld_off - 1)
            if act[0] == "d":
                catch_up()
                k = 7
                setc(cD, Pd); d_st.fill_(pending); d_sp.fill_(Pd); d_idx.zero_()
                Gd[k].replay()
                v_ids[0, 0].fill_(pending); v_ids[0, 1:k + 1].copy_(props[:k])
                torch.add(base, Pt, out=v_sp); setc(cT, Pt)
                Gv[k + 1].replay()
                pr, pd = props[:k].tolist(), preds[:k + 1].tolist()
                n = 0
                while n < k and pr[n] == pd[n]:
                    n += 1
                new = pr[:n] + [pd[n]]
                Pt += 1 + n
                Pd += 1 + min(n, k - 1)
                if n == k:
                    di_ids[0, 0].fill_(pr[k - 1]); setc(cD, Pd); torch.add(base, Pd, out=di_sp)
                    GI[1].replay()
                    Pd += 1
                low = low + 1 if n <= 1 else 0
                if low >= 3:
                    plain_left, low = 4, 0
            elif act[0] == "p":
                m = act[1]
                v_ids[0, :m + 1].copy_(torch.tensor([pending] + prop, device=dev))
                torch.add(base, Pt, out=v_sp); setc(cT, Pt)
                Gv[m + 1].replay()
                pd = preds[:m + 1].tolist()
                n = 0
                while n < m and prop[n] == pd[n]:
                    n += 1
                new = prop[:n] + [pd[n]]
                Pt += 1 + n
                pfail = pfail + 1 if n < 2 else 0
                if pfail >= 2:
                    pld_off, pfail = 8, 0
            elif act == ("plain", 4):
                t_st.fill_(pending); t_sp.fill_(Pt); t_idx.zero_(); setc(cT, Pt)
                G4.replay()
                new = t_out[:4].tolist()
                Pt += 4
                plain_left -= 1
            else:
                v_ids[0, 0].fill_(pending); torch.add(base, Pt, out=v_sp); setc(cT, Pt)
                Gv[1].replay()
                new = [int(preds[0])]
                Pt += 1
                plain_left = max(0, plain_left - 1)
            for tkn in new:
                ctx.append(tkn)
                index(len(ctx) - 1)
            rounds += 1
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        return ctx[P:P + NEWT], dt * 1000 / (NEWT - 1), rounds, acts

    results = {"rows": [], "label": OUT, "newt": NEWT, "start_costs": {f"{a}{b}": v for (a, b), v in cost.items()}}
    MODES = tuple(os.environ.get("SD7_MODES", "fixed7,fallback,fallback_pld").split(","))
    REPS, WARM = int(os.environ.get("SD_REPS", "3")), os.environ.get("SD_WARM", "1") == "1"
    for label, ids in prompts:
        if WARM:
            baseline(ids)
            for md in MODES:
                (gen(ids, "fixed7") if md == "fixed7" else gen3(ids, md == "fallback_pld"))      # warm-up
        bms, sms = [], {md: [] for md in MODES}
        last = {}
        for _ in range(REPS):
            bt, bm = baseline(ids)
            bms.append(bm)
            for md in MODES:
                o, ms, r, a = gen(ids, "fixed7") if md == "fixed7" else gen3(ids, md == "fallback_pld")
                sms[md].append(ms)
                last[md] = (o, r, a)
        bm = statistics.median(bms)
        row = {"prompt": label, "target_ms": bm, "tok_s_target": 1000 / bm}
        parts = [f"target alone {1000 / bm:5.1f} tok/s"]
        for md in MODES:
            m_ = statistics.median(sms[md])
            o, r, a = last[md]
            d = gd.first_diff(o, bt)
            row[md] = {"ms": m_, "tok_s": 1000 / m_, "rounds": r, "first_diff": d,
                       "actions": {f"{x}{y}": c for (x, y), c in a.items()}, "tokens": o}
            parts.append(f"{md} {1000 / m_:5.1f} ({bm / m_:.2f}x, diff {d})")
        row["tokens_target"] = bt
        results["rows"].append(row)
        log(f"SD7c {label[:24]:<24} | " + " | ".join(parts) + " | fallback_pld actions " +
            str(row.get("fallback_pld", {}).get("actions")))
    for md in MODES:
        rs = results["rows"]
        log(f"SD7c MEAN {md}: {statistics.mean(r[md]['tok_s'] for r in rs):.1f} tok/s | target alone "
            f"{statistics.mean(r['tok_s_target'] for r in rs):.1f} | identical {sum(r[md]['first_diff'] is None for r in rs)}/{len(rs)}"
            f" | slower than target alone on {sum(r[md]['tok_s'] < r['tok_s_target'] for r in rs)} prompts")
    rs = results["rows"]
    log(f"RESULT SD7c (BI={int(BI)}): " + " | ".join(f"{md} {statistics.mean(r[md]['tok_s'] for r in rs):.1f}" for md in MODES)
        + f" tok/s mean over {len(rs)} prompts | target alone {statistics.mean(r['tok_s_target'] for r in rs):.1f} | "
        + " | ".join(f"{md} identical {sum(r[md]['first_diff'] is None for r in rs)}/{len(rs)}" for md in MODES))
    if BI:
        log("BI1 kernel calls by (rows, K, variant): " + str(sorted(kernel_bi.stats.items())))
    log("add+RMSNorm kernel by (rows, kernel): " + str(sorted(ar_stats.items())))
    results["bi"] = BI
    results["temp"], results["seed"] = sampler.TEMP, sampler.SEED
    results["add_rmsnorm_paths"] = {f"{a}_{b}": c for (a, b), c in ar_stats.items()}
    results["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    (HERE / "results" / f"{OUT}.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    wd.cancel()
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
