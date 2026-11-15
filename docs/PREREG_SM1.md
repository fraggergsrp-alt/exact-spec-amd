# Pre-registration SM1: exact speculative SAMPLING (temperature > 0) (2026-11-15 16:05, before running)
**Question:** can speculative decoding WITH random sampling give exactly the same tokens as plain sampling with the
same seed - not just the same distribution (what standard speculative sampling guarantees), but the same sample?
**Method (sampler.py + sd7b.py):** token = argmax_v( logit_v / T + Gumbel(seed, position, v) ), noise from a hash of
(seed, position, v), no random state. Exact sample from softmax(logit/T) (Gumbel-max trick). The target picks with
this rule in plain decoding and at every verify position; the draft uses the same noise (shared randomness) and its
proposals are accepted while they equal the target's picks. With bit-identical logits (BI), the token at each
position is a fixed function of that position's logits -> speculative output == plain sampled output.
**Prior art:** Gumbel-max trick; drafter-invariant speculative decoding with common random numbers (Daliri, Musco,
Suresh, ISIT 2025); Gumbel-max list sampling (Rowan, Phan, Khisti, NeurIPS 2025). The algorithm is theirs; new here is
only showing it holds bit-for-bit in practice with batch-invariant fp16 kernels on a consumer GPU. Temperature only
(no top-k / top-p yet).
**Runs (run_SM1.bat):** sampler self-test; then bi3 prompts (50), 256 new tokens, fixed k=7, 1 repetition, seed 42,
T = 0.7 and 1.0, 1.7B and 4B, normal (PyTorch matmuls) and BI. Same session.
**Predictions:**
- **H160:** sampler self-test passes: frequencies match softmax (chi-square < 37.7, 15 dof, both T), deterministic,
  row-count invariant.
- **H161:** BI speculative == BI plain sampled output on 50/50 prompts, both T, both models.
- **H162:** normal (PyTorch) speculative differs from plain on >= 5/50 prompts on 1.7B at T=0.7 (sampling makes near-ties
  matter at more positions than greedy).
- **H163:** speculative sampling is still faster than plain: BI fixed7 >= 1.3x BI plain at T=0.7 on 1.7B (acceptance
  drops with temperature; shared noise should keep it reasonable).
- **H164:** BI cost vs normal fixed7 <= 5% at both temperatures.

## Results SM1 (2026-11-15 15:59-16:38, results/SM1_log.txt, SM1_*.json). First attempt (14:57 UTC) stopped at the
sampler self-test: compile error (-INFINITY undefined in hipRTC) -> replaced by -__builtin_inff(); no model run happened.
| 50 prompts x 256 tokens | 1.7B normal | 1.7B BI | 4B normal | 4B BI |
|---|---|---|---|---|
| T=0.7 speculative == plain (B) | 38/50 | **50/50** | 50/50 | **50/50** |
| T=0.7 fixed7 / plain tok/s (B) | 225.3 / 152.9 | 215.7 / 156.5 (1.38x) | 128.9 / 65.3 | 123.4 / 68.4 (1.80x) |
| T=1.0 speculative == plain (B) | 36/50 | **50/50** | 50/50 | **50/50** |
| T=1.0 fixed7 / plain tok/s (B) | 189.6 / 152.7 | 186.3 / 157.4 (1.18x) | 116.1 / 65.2 | 112.6 / 68.8 (1.64x) |
| tokens per round T=0.7 / T=1.0 | 3.96 / 3.30 | 3.96 / 3.30 | 3.51 / 3.19 | 3.47 / 3.19 |
- **H160 PASS:** chi-square 16.4 and 17.6 (< 37.7), max frequency error 0.0008 / 0.0014; deterministic; row-count invariant.
- **H161 PASS:** 50/50 at both temperatures on both models (200 sampled outputs, 51,200 tokens).
- **H162 PASS:** normal 1.7B diverged on 12/50 (T=0.7) and 14/50 (T=1.0); with sampling, divergence can start at
  token 1 (a single rounding difference flips a sample).
- **H163 PASS:** 1.38x at T=0.7 on 1.7B (>= 1.3x); 4B 1.80x / 1.64x. Shared noise keeps acceptance high
  (3.3-4.0 tokens/round).
- **H164 PASS:** BI cost -4.3%, -1.7% (1.7B), -4.3%, -3.1% (4B).
- Cost of the sampler itself: BI plain 156.5-157.4 tok/s vs 160.8 greedy (BI3) on 1.7B, about 2-3%: one thread block
  scans all 151,936 logits per token. Could be split over more blocks later (same bits if the order is fixed).
**Conclusion (B, one session):** exact speculative sampling works in practice: same seed -> same text with or
without speculation, at 1.2-1.8x speed. Algorithm = prior art (Daliri et al. 2025); the bit-exact practical result
on fp16 consumer hardware depends on our batch-invariant kernels (without them 1.7B diverged on 24-28% of prompts).
