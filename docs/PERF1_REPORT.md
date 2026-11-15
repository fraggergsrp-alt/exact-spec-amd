# PERF1 report: unexplained GPU performance variability (2026-11-15)
Pre-registration, every hypothesis, every failed run: docs/PREREG_PERF1.md. Labels: (B) measured on this PC,
(C) hypothesis / interpretation. Nothing here is claimed as novel; all mechanisms below are known in general.

## 1. Setup
- PC: RX 9070 XT 16 GB, Ryzen 5 3600, Windows 11, ROCm PyTorch 2.14. Engine: exact speculative decoding (Qwen3-1.7B
  target, Qwen3-0.6B int4 draft, batch-invariant kernels), frozen as frozen/ENGINE_v1_2026-11-15 (SHA-256 manifest).
  Every PERF1 bat checked the frozen hashes before and after: unchanged in all runs.
- Research scripts (research/perf1/): perf1_env.py (environment + telemetry capability), perf1_a.py (draft timing),
  perf1_b.py / perf1_b2.py (session variability, frozen engine as subprocess), perf1_c2.py (VRAM guard),
  events.ps1 (read-only Windows event dump). Bats: run_PERF1.bat, run_PERF1A.bat, run_PERF1B2.bat, run_PERF1C.bat,
  run_PERF1_events.bat. Raw data: results/PERF1*_trials.jsonl, PERF1B_runs.jsonl, PERF1B2_runs.jsonl, PERF1*_log.txt,
  PERF1_env_*.json, PERF1_events.txt.
- **Telemetry limits:** no amd-smi, rocm-smi or rocprof in this Windows ROCm install; python amdsmi not installed.
  Measured: GPU-event execution time, CPU enqueue/sync time, shader clock (own kernel: cycle counter / 100 MHz wall
  clock), copy bandwidth, Windows GPU-memory counters. NOT measured (not estimated either): memory clock,
  temperature, power, cache hit rates.

## 2. Experiment A - why is the draft step sometimes 2.1-2.3 ms instead of ~1.6 ms? (B)
30 randomized trials per condition, fixed idle reset, then a pre-workload, then 7 draft steps timed on the GPU.
| idle before | pre-workload | draft ms/step | shader clock at start |
|---|---|---|---|
| 300 ms | none | 2.11 (IQR 1.69-2.46) | 1189 MHz |
| 30 ms | none | 1.58 | 1189 MHz |
| 300 ms | verify / memory streaming / ALU spin | 1.58 / 1.59 / 1.67 | ~2750 MHz |
| 300 ms | verify, then 20 / 100 ms idle | 1.63 / 1.64 | 1188 MHz |
| 1000 ms | real 1024-token prefill | 1.57 | 2733 MHz |
**Finding:** the slow draft happens only after the GPU has been idle for more than ~100-300 ms. It is NOT caused by
the shader clock value at the start (equally low in slow and fast cases), NOT by caches (draft after draft is not
faster), NOT by the CPU (the difference is GPU execution time). Memory traffic wakes the state fully, arithmetic
mostly -> consistent with a deep idle / memory power state (C: cannot read memory clock or power to confirm).
**Engineering consequence:** none needed. Steady decoding has no idle gaps, and a real request starts with a prefill
that wakes the GPU (first draft 1.57 ms/step; prefill itself 137.4 ms with or without warm-up). The earlier
"isolated 2.3 ms draft" was an artefact of measuring after idle.

## 3. Experiment B - why did identical code run ~20% slower in some sessions? (B, partly unexplained)
- **Stability:** within one session the frozen engine varies 0.3% (5 runs) and 1.2% (9 runs, B2).
- **VRAM oversubscription is a strong, sharp effect:** a helper process holding 6 GiB (everything still fits): no
  change. Holding 10 GiB (11.9 GB in use before the engine loads): speculative -54%, plain decoding -78%, output still
  identical. (C: consistent with Windows moving GPU memory to system RAM.)
- **The actual slow sessions are NOT explained by this:** only Explorer, a terminal and Claude were open. All slow
  1.7B runs fall in one period (Nov 14 18:50 - Nov 15 00:54); every run since (30+, including after a reboot) is fast.
  Slow runs mostly coincided with a lower shader clock under load (~2750 vs ~3230 MHz), but one slow run had a high
  clock, so the clock is a correlate, not the cause. B2 could not provoke the slow state (idle, after a 4B run,
  back-to-back: all fast). Windows event logs show nothing between the last slow and first fast run.
  **Status: unexplained, not currently reproducible.** Every run now logs environment and clock automatically.

## 4. Experiment C - can the causes be removed? (B)
- C1 idle wake-up: not needed in real use (see 2). A ~2 ms memory warm-up would save ~1.4 ms on a draft started
  after a long idle without prefill - a case that does not occur in the engine.
- C2 VRAM guard: warns at start-up if free VRAM < measured peak need + 0.5 GiB. Correct decisions in all three tests
  (OK / OK / WARN at 0 / 6 / 10 GiB held elsewhere for the 1.7B setup) - **only with the Windows counter**.
  **Finding (B): on Windows, hipMemGetInfo / torch.cuda.mem_get_info does not see other processes' GPU memory**
  (reported 15.77 GB free while 11.7 GB were in use). Candidate for the next engine version, after re-validation.

## 5. Negative results and process errors (kept on purpose)
- H170 (shader clock causes the slow draft): disproved. H184/H186 (slow state provokable): not reproduced.
  H191, H192: failed (warm-up saves slightly more than predicted; prefill shows no wake-up penalty).
- My script errors: Experiment A run 1 and C run 1 crashed (cache-length pointer missing - the second time because I
  edited an outdated copy); C run 2 hung (cache write counter not reset before re-prefill -> out-of-bounds writes);
  the environment summary wrongly printed "amdsmi: true"; the first VRAM guard used a query that is blind on Windows.
  All were in research copies; the frozen engine was never touched.

## 6. Implications for specialized hardware (C, careful)
These observations are about one consumer GPU under one OS and driver; they do not automatically transfer to custom
silicon, and none of the mechanisms is new.
- **Idle-state wake-up:** interactive one-user inference is bursty (the user reads and types). Power states that take
  milliseconds to exit after ~100-300 ms of idle add latency to the first work after a pause. An accelerator for local,
  interactive use would want fast exit from idle for its memory system, or a predictable "ready" state.
- **Memory capacity cliff:** performance does not degrade gracefully when the working set no longer fits: it falls off
  a cliff (-54% to -78%). Capacity planning (weights + KV cache + other users of the memory) matters as much as
  bandwidth. Plain decoding suffered more than speculative decoding (-78% vs -54%) - consistent with speculation reading
  the big model's weights about once per 4 tokens; worth testing before any claim.
- **Observability:** without power, temperature and memory-clock counters, two of our questions could only be
  narrowed, not answered. Good on-device telemetry is itself a design requirement.

## 7. Open items
- H183: catch a slow session with logging (and ideally AMD Software metrics logging for temperature / power / clocks).
- Test (C): "speculative decoding degrades less under memory oversubscription than plain decoding" with a
  pre-registered sweep of held VRAM.
- Decide whether to adopt the C2 guard into a new, re-validated engine version.
