# Pre-registration BI1: bit-identical speculative decoding (idea 5) (2026-11-15 01:00, before running)
**Claim being tested (C):** speculative decoding in our engine can be made EXACTLY bit-identical to plain decoding,
on every prompt, at small cost, by making the target's matrix multiplies batch-invariant.
**Prior art:** batch-invariant kernels (Thinking Machines Lab 2025; vLLM/SGLang deterministic modes; LLM-42). Known
cost elsewhere: about 1.6x slower (26 s -> 42 s in Thinking Machines' vLLM test). Speculative-decoding papers report
"almost" identical output (e.g. >95%). The open part is: exact, at near-zero cost, batch 1, consumer AMD GPU.
**Why it should work:** attention (split-KV, per query), norms, RoPE, KV write are already per-row in our engine. The
only part whose arithmetic depends on the number of rows is the matmul (PyTorch uses a 1-row path for plain decoding
and a multi-row path for the 8-token verify). kernel_bi.py: one kernel for 1..32 rows whose per-row summation order
is fixed (= kernel_f12.gemv_f16_ref order), so the bits of row t can not depend on the row count.
**Method:** run_BI1.bat. (1) kernel_bi.py self-test on the 10 real target shapes (1.7B, 4B). (2) sd7b.py, fixed k=7
only, 11 prompts, 96 tokens, greedy, FREE_DUP; 1.7B normal, 1.7B BI, 4B normal, 4B BI, same session. Identity =
speculative output vs plain decoding (8-step graph) in the same run with the same kernels.
**Predictions:**
- **H130:** n-row output is bitwise equal to 1-row output for all tested n (2..32) on all 10 shapes, and equal to
  gemv_f16_ref where tested; error vs fp32 comparable to PyTorch.
- **H131:** with BI, speculative output is identical to plain decoding on 11/11 prompts, on BOTH models
  (currently 8/11 on 1.7B, 11/11 on 4B).
- **H132:** BI fixed7 speed >= 0.90x normal fixed7 on both models.
- **H133:** BI target-alone speed >= 0.97x normal target-alone (1-row kernel about as fast as PyTorch, L2: 7.04 vs 7.14 ms).
**Diagnostics logged:** which add+RMSNorm kernel (v1/v2) each row count takes; kernel_bi calls by rows/K/variant.
**Limits stated in advance:** the guarantee is "speculative == plain decoding with the same kernels". BI plain output
may differ from PyTorch-default plain output (a different, equally valid fp16 summation order); prefill (> 32 rows)
uses PyTorch in both paths, so it is shared. Greedy decoding only; sampling would need the same random stream too.
If H131 fails, the log tells which component still depends on row count; that is a finding, not a failure to hide.

## Results BI1 (2026-11-15 00:44-00:54, results/BI1_log.txt, BI1_kernel.json, BI1_17_*.json, BI1_4b_*.json)
| | 1.7B normal | 1.7B BI | 4B normal | 4B BI |
|---|---|---|---|---|
| speculative == plain decoding (B) | 7/11 | **11/11** | 11/11 | **11/11** |
| fixed7 tok/s, mean of 11 (B) | 209.9 | 192.4 (-8.3%) | 134.5 | 99.4 (-26.1%) |
| target alone tok/s (B) | 144.6 | 152.1 (+5.2%) | 65.3 | 68.4 (+4.8%) |
| verify 8 tokens in context (B) | 7.71 ms | 8.84 ms | 17.26 ms | 25.80 ms |
- **H130 PASS:** n-row bits == 1-row bits for every n (2..32) on all 10 shapes, both variants; == gemv_f16_ref on the 9
  shapes it supports; error vs fp32 equal to PyTorch's. (The isolated ms numbers in the self-test are not
  trustworthy - isolated launches run at low clocks; in-graph, our 1-row kernel is FASTER than PyTorch, see H133.)
- **H131 PASS:** 11/11 on both models. The 1.7B normal run had 4 prompts diverge (tokens 77, 31, 41, 52); with BI none.
- **H132 PASS on 1.7B (-8.3%), FAIL on 4B (-26.1%).** Cause: the multi-row kernel; verify 8 tokens +1.1 ms (1.7B),
  +8.5 ms (4B). Suspects: x converted half->float once per output row (8x redundant work), 48 KiB shared-memory
  variant (low occupancy) on the 4B lm_head, and the global-memory variant for K >= 4096.
- **H133 PASS:** plain decoding got faster (+5%) with our kernel.
- Diagnostics: add+RMSNorm uses one kernel per model for all row counts (target 1.7B v2, 4B v1; draft v2) - no mixing.
  BI plain output == PyTorch-default plain output on 8/11 (1.7B) and 11/11 (4B); where they differ it is fp16 rounding
  of a different, equally valid summation order (first differences at tokens 77, 39, 55).
**Conclusion (B):** exact equality of speculative and plain decoding is achieved and measured, with plain decoding
faster. The remaining problem is speed of the multi-row kernel, mainly on 4B -> BI2 (kernel tuning, same summation order).

# Pre-registration BI2a: faster bit-identical multi-row kernel (2026-11-15 01:10, before running)
**Method:** bi2_bench.py. Variants T{512,1024,2048}R{1,2}: input rows converted to float once per K-tile in shared
memory, each warp R output rows, 1-token build for 1-row calls; per-row summation order unchanged. Bit-identity checked
against kernel_bi (rows 1/2/5/8/13, every real shape). Timing: all target matmuls of one forward pass in one graph,
replayed back-to-back, 1 row and 8 rows, vs bi1 and PyTorch.
**Predictions:**
- **H140:** every T variant is bit-identical to bi1 (same order by construction).
- **H141:** the best T variant's 8-row time is <= 1.25x PyTorch's 8-row time on both models (bi1 today: about
  1.15x on 1.7B and 1.5x on 4B, judged from the verify times).
- **H142:** the best 1-row time stays <= PyTorch's 1-row time.
If H141 holds, the winner goes into kernel_bi and BI2b repeats the full BI1 run (identity + speed).

## Results BI2a (2026-11-15 01:03, results/BI2a_log.txt, BI2_bench.json) - one forward's matmuls in a graph, ms (B)
| | PyTorch 1 / 8 rows | BI1 1 / 8 | best: T512R2 1 / 8 |
|---|---|---|---|
| 1.7B | 6.15 / 6.12 | 6.10 / 9.24 | 5.93 / **6.79** (1.11x PyTorch) |
| 4B | 13.83 / 13.89 | 13.42 / 29.90 | 13.43 / **15.27** (1.10x PyTorch) |
- **H140 PASS** (all 6 variants bit-identical). **H141 PASS** (1.11x, 1.10x <= 1.25x). **H142 PASS** (1 row faster than PyTorch).
- R=2 (two output rows per warp, each input read used twice) is the decisive change; R=1 tiled was no better than BI1.
- T512R2 now default in kernel_bi.py (BI_KERNEL=bi1 restores BI1; BI_TILE / BI_R env).

# Pre-registration BI2b (2026-11-15 01:10, before running)
run_BI2b.bat: (1) bench again with R=4 and tile 256 variants (if one is clearly better, next round); (2) kernel
self-test (BI2 vs BI1 bits, n-row vs 1-row); (3) the full BI1 protocol again with the T512R2 kernel.
- **H143:** speculative == plain 11/11 on both models (unchanged guarantee).
- **H144:** BI fixed7 >= 0.95x normal fixed7 on both models (BI1: 0.92x / 0.74x).
- **H145:** BI target alone >= normal target alone on both models.

## Results BI2b (2026-11-15 01:19-01:29, results/BI2b_log.txt, BI2_bench.json, BI2_17_*.json, BI2_4b_*.json)
| | 1.7B normal | 1.7B BI2 | 4B normal | 4B BI2 |
|---|---|---|---|---|
| speculative == plain decoding (B) | 7/11 | **11/11** | 11/11 | **11/11** |
| fixed7 tok/s, mean of 11 (B) | 255.3 | 253.4 (**-0.7%**) | 158.7 | 152.1 (**-4.2%**) |
| target alone tok/s (B) | 154.5 | 159.4 (+3.2%) | 65.7 | 69.0 (+5.0%) |
| verify 8 tokens in context (B) | 7.58 ms | 7.85 ms | 17.03 ms | 17.55 ms |
- **H143 PASS** (11/11 both). **H144 PASS** (0.993x, 0.958x >= 0.95x). **H145 PASS** (plain decoding faster).
- Bench: R=4 is not better than R=2; tile 256 is 0.05-0.17 ms faster for 8 rows, 0.3-0.4 ms slower for 1 row.
  A per-row-count choice (T512R2 for 1 row, T256R2 for 2+ rows) gives the same bits and maybe ~1% - not done yet.
- **Caution on absolute numbers:** both "normal" runs were much faster in this session (1.7B 255 vs 210 at 00:46;
  4B 159 vs 135) with identical code. Something in machine state differs between sessions (other apps, clocks,
  memory). Only same-session comparisons are valid; the 255 tok/s is NOT a new record until it repeats.
**Conclusion (B):** speculative decoding in our engine is bit-for-bit identical to plain decoding on every tested
prompt, on both models, at a cost of 0.7-4.2% speculative speed, while plain decoding gets 3-5% faster.
Prior art for contrast (not directly comparable): Thinking Machines' batch-invariant vLLM ran 26 s -> 42 s
(batched serving, different hardware). Not yet independently replicated (D): wait for the user.

# Pre-registration BI3: does the guarantee hold on a bigger, longer, repeated test? (2026-11-15 01:45, before running)
**Method:** run_BI3.bat, run twice in separate sessions (s1 now, s2 later, e.g. after a reboot). 50 varied prompts
(bi3_prompts.py, fixed before running: 1 filler, 19 raw continuations, 30 chat prompts incl. code, math, JSON, other
languages), 512 new tokens each, greedy, fixed k=7, 1 timed repetition per prompt (identity needs no repetitions;
speed is the mean over 50 prompts). 1.7B normal / BI, 4B normal / BI, same session. Kernel T512R2.
**Predictions:**
- **H150:** BI speculative == BI plain decoding on 50/50 prompts, 512 tokens, both models, both sessions.
- **H151:** normal (PyTorch matmuls) speculative differs from plain on >= 5/50 prompts on 1.7B (shows the test
  can detect divergence at this length).
- **H152:** BI fixed7 >= 0.95x normal fixed7, both models, both sessions.
- **H153 (s2 only):** BI plain and BI speculative outputs are identical between session 1 and session 2 on 50/50
  (run-to-run reproducibility).

## Results BI3 session 1 (2026-11-15 01:41-02:12, results/BI3_s1_log.txt, BI3_*_s1.json)
| 50 prompts x 512 tokens | 1.7B normal | 1.7B BI | 4B normal | 4B BI |
|---|---|---|---|---|
| speculative == plain decoding (B) | 35/50 | **50/50** | 50/50 | **50/50** |
| fixed7 tok/s, mean of 50 (B) | 259.3 | 251.3 (-3.1%) | 151.6 | 145.1 (-4.3%) |
| target alone tok/s (B) | 155.9 | 160.8 (+3.1%) | 66.3 | 69.6 (+5.0%) |
- **H150 PASS for session 1** (50/50 both models; 25,600 tokens per model, every one identical).
- **H151 PASS:** normal 1.7B diverged on 15/50 prompts (first differences at tokens 39 ... 462) - the test detects
  divergence, and the longer the output the more prompts diverge (96 tokens: 3-4/11; 512 tokens: 15/50).
- **H152 PASS** (-3.1%, -4.3%).
- **H153:** waits for session 2.
- Observation (B): on 4B, PyTorch's own matmuls happened to be consistent between 1 and 8 rows on all 50 prompts
  (also 11/11 earlier) - by luck of which internal kernels it picks for those shapes, not by guarantee.
  BI plain vs PyTorch plain: 34/50 (1.7B) and 30/50 (4B) identical over 512 tokens - expected, different rounding order.

## Results BI3 session 2 (2026-11-15 15:18-15:49, after restart; results/BI3_s2_log.txt, BI3_*_s2.json)
| 50 prompts x 512 tokens | 1.7B normal | 1.7B BI | 4B normal | 4B BI |
|---|---|---|---|---|
| speculative == plain (B) | 35/50 | **50/50** | 50/50 | **50/50** |
| fixed7 tok/s (B) | 254.0 | 246.9 (-2.8%) | 147.8 | 143.2 (-3.1%) |
| target alone tok/s (B) | 152.9 | 159.1 (+4.1%) | 65.5 | 68.6 (+4.6%) |
| same tokens as session 1, plain / speculative | 50/50, 50/50 | 50/50, 50/50 | 50/50, 50/50 | 50/50, 50/50 |
- **H150 PASS** (50/50, both models, both sessions). **H152 PASS** (-2.8%, -3.1%). **H153 PASS** (50/50 across sessions).
- Finding (B): the normal engine's divergences are deterministic too - same 15 prompts at the same token positions in
  both sessions. The problem was never randomness; it is that 1-row and 8-row matmuls round differently (batch
  variance). BI removes exactly that.
- Speed varies ~2-3% between sessions (s1 vs s2); the BI cost stays 2.8-4.3% in all four comparisons.
**BI3 conclusion (B, 2 sessions, not independently replicated):** 200 runs of 512 tokens (2 models x 50 prompts x
2 sessions) with BI: speculative output == plain output every time; 102,400 tokens checked. Cost 2.8-4.3% speculative
speed; plain decoding 3-5% faster.
