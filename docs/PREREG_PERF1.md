# Pre-registration PERF1: unexplained GPU performance variability (2026-11-15 18:00, before running)
**Scope:** measurement only. No new architecture, no claim of novelty. Validated engine frozen in
frozen/ENGINE_v1_2026-11-15 (hashes); research scripts live in research/perf1/ and only import root modules
(check_frozen.py runs first and last in run_PERF1.bat).
**Telemetry actually available (checked 2026-11-15):** no amd-smi, rocm-smi or rocprof in this Windows ROCm install
(only hipInfo, clinfo, hipcc). Measured: GPU execution time (GPU events), CPU enqueue/sync time, shader clock (our
kernel: cycle counter / 100 MHz wall clock), copy bandwidth, VRAM free, Windows counters if available (perf1_env.py
logs which). NOT measured, will not be estimated as measured: memory clock, temperature, power, cache hit rates.
Optional: AMD Adrenalin "Performance -> Metrics -> logging" CSV by the user (temperature, power, clocks).

## Experiment A - draft timing (research/perf1/perf1_a.py)
Draft graph D7 (7 steps, 0.6B int4) timed after a fixed reset (GPU idle 300 ms, then a second pass with 30 ms) and a
pre-workload: none / D7 / verify V8 / ALU-only spin (no memory traffic, same duration as V8) / memory streaming (same
duration) / V8 + idle gap 1, 5, 20, 100 ms. 30 trials each, randomized and interleaved (seed 1).
- **H170 (clock state):** shader clock right before D7 is higher after V8 than after idle, and the draft time follows
  the clock. Disproof: equal clock_before in "verify" and "alone" but different draft times; or "alu" raises the clock
  as much as V8 but the draft stays slow.
- **H171 (memory-side state):** "mem" (memory traffic, little ALU) makes the draft about as fast as "verify", "alu" does
  not. Supports a memory-subsystem warm-up (memory clock / power state) that we cannot measure directly -> would be
  reported as narrowed, not proven. Disproof: "alu" alone reproduces the speed-up.
- **H172 (cache):** NOT the cause. The draft's working set per step (~0.3 GB int4 weights + 0.3 GB lm_head) is far
  larger than the GPU's caches, and V8 leaves target data in them. Disproof: "verify" faster than both "alu" and "mem"
  at equal clock_before.
- **H173 (CPU launch/sync):** NOT the cause. CPU enqueue time is equal (within 0.05 ms) across conditions and the effect
  is in GPU execution time. Disproof: GPU times equal, only CPU times differ.
- **H174 (decay):** the advantage after V8 shrinks with idle gap; the gap at which half of it is gone is reported as
  the governor time constant (estimate from data, with IQR).
- **H175:** "alone" reproduces ~2.3 ms/step and "verify" ~1.65 ms/step within +-15% (otherwise the original anomaly
  does not reproduce under controlled conditions - also a result).

## Experiment B - session variability (research/perf1/perf1_b.py)
Frozen engine (sd7b.py, b6 prompts, 96 tokens, fixed7, BI=1) run 9 times per session in randomized A-B-A order:
A = no extra GPU memory use, B = a helper process holding 6 or 10 GiB of VRAM. Conditions logged before each run.
Runs append to results/PERF1B_runs.jsonl; the bat is meant to be run in several sessions (fresh boot / normal apps).
- **H180:** within one session at A, run-to-run spread of fixed7 tok/s is <= 5% (max/min).
- **H181:** helper 6 GiB (engine still fits in VRAM) changes fixed7 by < 5%.
- **H182:** helper 10 GiB (VRAM near/over capacity) lowers fixed7 by >= 15% (WDDM paging), reproducing the size of
  the earlier session differences. Disproof: no drop -> VRAM pressure is not the mechanism.
- **H183 (cross-session, after >= 3 sessions):** session-to-session differences at A correlate with logged conditions
  (free RAM, GPU memory in use by other processes, clock after load); if no logged condition explains them, report
  "unexplained" with the distribution.

## Experiment C - only after A and B: test whether an identified cause can be removed in software (e.g. a warm-up
kernel, a clock-holding schedule, reserving VRAM). Separate pre-registration.
**Hardware implications:** discussed only for causes established in A/B; no claim that GPU behaviour transfers to
custom silicon.

## Results run 1 (2026-11-15 17:56-18:16, results/PERF1_20261115_1756_log.txt, PERF1B_runs.jsonl, PERF1_env_*.json)
**Experiment A: did not run** - script bug (short prompt reached the multi-token attention kernel before the layer's
cache-length pointer existed; the validated engine always prefills the 1024-token prompt first). Fixed in research
copy (prefill the 1024-token prompt first). Engine untouched (frozen check OK before and after). Also fixed: the env
summary printed "python amdsmi: true" although the import failed (ModuleNotFoundError) - amdsmi is NOT available.
**Experiment B (session boot 15:01, 9 runs, randomized A-B-A):**
| helper VRAM held | runs | fixed7 tok/s median (min-max) | target alone tok/s median | identical |
|---|---|---|---|---|
| 0 GiB | 5 | 254.0 (253.5-254.3) | 159.4 | 11/11 every run |
| 6 GiB | 2 | 253.6 (253.5-253.8) | 159.3 | 11/11 |
| 10 GiB | 2 | 116.3 (116.1-116.6) | 35.5 | 11/11 |
- **H180 PASS:** within-session spread at 0 GiB 0.3% (max/min) - the engine itself is very stable.
- **H181 PASS:** 6 GiB held elsewhere (everything still fits in 16 GiB): no change (-0.2%).
- **H182 PASS:** 10 GiB held (GPU memory in use before the engine started: ~11.9 GB of 16): fixed7 -54%, plain -78%.
  Output still identical. Mechanism consistent with Windows moving GPU memory to system RAM when VRAM is
  oversubscribed (not directly observable with our counters - narrowed, not proven).
- Observation (B numbers, C interpretation): plain decoding lost much more (-78%) than speculative decoding (-54%).
  Plausible reason: speculation reads the big model's weights once per ~4 tokens, so paged-out weights cost less per
  token. To be tested, not claimed.
- **H183:** needs more sessions. Current evidence: the earlier ~20% session differences are of the kind produced by
  GPU memory pressure from other programs (only a step change at oversubscription, no gradual effect at 6 GiB).
  Earlier slow sessions were not logged with GPU memory in use, so this is not yet confirmed for them.

## Results Experiment A (2026-11-15 18:19-18:22, results/PERF1A_20261115_1819_log.txt, PERF1A300_*, PERF1A30_*)
Draft D7 = 7 steps; ms/step = GPU-event time / 7; 30 randomized trials per condition; clock = our probe (MHz).
| pre-workload | reset 300 ms idle: ms/step (clock before) | reset 30 ms idle: ms/step (clock before) |
|---|---|---|
| none ("alone") | **2.114** (IQR 1.69-2.46) (1189) | **1.582** (1189) |
| draft (b2b) | 1.613 (2751) | 1.613 (2751) |
| verify | 1.581 (2749) | 1.580 (2749) |
| ALU spin, no memory | 1.670 (2745) | 1.576 (2755) |
| memory streaming | 1.586 (2745) | 1.582 (2748) |
| verify + 1 / 5 ms idle | 1.574 / 1.577 (2755 / 2721) | 1.573 / 1.577 |
| verify + 20 / 100 ms idle | 1.634 / 1.639 (**1188**) | 1.582 / 1.641 (1188) |
Shader clock right AFTER the draft: ~2750 MHz in every condition. Verify alone: 8.40 ms in every condition.
- **H175 PASS (reproduced, but only after a long idle):** 2.11 ms/step after 300 ms idle (earlier "isolated" 2.3),
  1.58 after verify (earlier 1.65). With only 30 ms idle the draft is equally fast WITHOUT any pre-workload (1.582).
- **H170 (shader clock) DISPROVED in its simple form:** the shader clock before the draft is ~1189 MHz in both the slow
  case (300 ms idle) and the fast cases (30 ms idle; verify + 20/100 ms gap). The shader clock drops within 5-20 ms of
  idle but the draft stays fast; it climbs to ~2750 MHz during every draft. What predicts the slow draft is idle time
  > ~100 ms, not the shader clock at the start.
- **H171 (memory-side) WEAKLY SUPPORTED:** after 300 ms idle, a memory-only pre-workload restores full speed (1.586),
  an ALU-only one restores most (1.670). Some state that memory traffic wakes up and arithmetic only partly wakes up
  - consistent with a memory-clock / memory power state, which we cannot read. Narrowed, not proven.
- **H172 (cache) PASS as predicted:** after verify (target data in caches) the draft is not faster than after memory
  streaming or ALU at the same idle; draft-after-draft (same weights) is even 2% slower (1.613). No cache benefit.
- **H173 (CPU) PASS as predicted:** the difference is in GPU execution time; CPU enqueue time varies (2-13 ms) only
  because the call waits while earlier GPU work is still queued.
- **H174 (decay):** the speed advantage survives 100 ms of idle (1.64) and is gone at 300 ms (2.11, large spread):
  the slow state starts between 100 and 300 ms of idle (not resolved further in this run).
**Conclusion A (B, one session):** the "draft anomaly" is a GPU power-management effect after >100-300 ms of idle (a
deeper idle state whose wake-up costs ~0.5 ms per draft step for the first ~15 ms), NOT a property of drafting, of
caching, of the shader clock value, or of the CPU. In steady speculative decoding there is no idle gap, so it does not
affect throughput; it only affects the first round after a pause (one-time ~3-4 ms) and any "isolated" profiling.
Prior art: GPU idle power states and DVFS wake-up latency are well known in general - nothing novel claimed.

## Exploratory re-analysis of earlier logs (2026-11-15 18:35; post-hoc, NOT pre-registered - treat as a lead)
User reports: during the slow and the fast sessions only File Explorer, a terminal and Claude were open. So VRAM
pressure from other programs (H182 mechanism) probably does NOT explain the earlier slow sessions. H183 stays open.
Every engine run logs "CLOCK shader clock right after target graphs". Across all logged runs (1.7B, greedy):
| run (session time) | clock after target graphs | 1.7B fixed7 tok/s | target alone |
|---|---|---|---|
| SD7 (18:50, Nov 14) | 2837 MHz | 215.9 | 143.8 |
| SD7c (22:45) | 2744 | 215.8 | 144.6 |
| BI1 normal (00:46) | 2746 | 209.9 | 144.6 |
| BI1 BI (00:48) | 2801 | 192.4 | 152.1 |
| SD7b (22:36) | 3174 | 197.1 (exception) | 143.3 |
| BI2b normal / BI (01:21-01:23) | 3227 / 3226 | 255.3 / 253.4 | 154.5 / 159.4 |
| BI3 s1 / s2 normal (01:45, 15:22) | 3240 / 3246 | 259.3 / 254.0 (50 prompts x 512) | 155.9 / 152.9 |
Same code, identical tokens and rounds (BI1 vs BI2b: same number of rounds on every prompt; time per round 18.6-23.1
vs 16.2-17.4 ms; plain 4-token graph 7.48 vs 6.57 ms/token; isolated start-up draft cost d7 23.6 vs 17.7 ms).
**Lead (C):** the shader clock the GPU settles at under the 1.7B load is bimodal across runs (~2750 vs ~3230 MHz);
slow 1.7B runs coincide with the low state in 4 of 5 cases (exception: SD7b at 3174 MHz). The 4B load always shows
~3300 MHz, but BI1 4B was also slow (134.5 vs 158.7) at 3308 MHz - so the clock does not explain everything.
Possible mechanism (unverified): the driver's power/performance governor choosing a lower level for the lighter,
bursty 1.7B workload in some runs. Needs telemetry we lack (power, temperature, memory clock) -> Adrenalin logging.

# Pre-registration B2: can the slow 1.7B state be provoked? (2026-11-15 18:40, before running)
research/perf1/perf1_b2.py: frozen engine, 1.7B, 9 runs, randomized: 3x after 120 s GPU idle, 3x right after a 4B
run, 3x back-to-back. Logged per run: engine clock probe, plain ms/token, verify/draft in context, tok/s.
Optional user telemetry: AMD Software metrics logging (temperature, power, clocks) running during the whole bat.
- **H184:** the clock probe is bimodal (some runs < 3000 MHz, some > 3100 MHz) also within one session.
- **H185:** runs in the low state are >= 10% slower (fixed7) than runs in the high state; runs in the same state differ
  by < 3%.
- **H186:** the preceding condition (idle / after 4B / back-to-back) changes the probability of the low state.
Disproof / outcomes: if all 9 runs are in the high state and within 3%, the slow state cannot be provoked by these
conditions -> report "not reproduced" and fall back to logging future sessions.

## Results B2 (2026-11-15 18:30-19:00, results/PERF1B2_20261115_1830_log.txt, PERF1B2_runs.jsonl)
All 9 runs in the high state: clock 3213-3237 MHz, fixed7 251.8-254.7 tok/s (spread 1.2%), plain 6.36-6.38 ms/token,
11/11 identical. idle120 / after4b / b2b medians 254.0 / 252.8 / 254.2 tok/s.
- **H184 FAIL:** no bimodality within this session. **H185:** same-state spread 1.2% (< 3%) - holds; no low-state runs to
  compare. **H186 FAIL / no effect:** preceding condition does not change anything (after4b ~0.5% slower at most).
- **Slow state NOT reproduced.** Updated picture of all logs: every run from 2026-11-15 01:19 on (BI2b, BI3 s1, BI3 s2
  after a reboot, SM1, PERF1B, B2: >30 engine runs) is fast and in the high-clock state. All slow 1.7B runs are from one
  period, 2026-11-14 18:50 to 2026-11-15 00:54. SD7b (22:36) was slow at a high clock (3174 MHz) with slow plain
  decoding (7.14 ms/token), so the clock level is a correlate, not the cause.
- Status: **unexplained, not currently reproducible.** Something about the machine changed between 00:54 and 01:19 on
  Nov 15 and has persisted since (including across a reboot). Next check: read-only Windows event/update history for
  that window (run_PERF1_events.bat). Future sessions are now logged with environment + clock automatically.

## Windows event check (results/PERF1_events.txt, read-only, 2026-11-15 19:05)
Window 2026-11-14 18:00 to 2026-11-15 02:00:
- Windows Update: updates found and downloaded 19:20-19:21 (Nov 14); security update KB5132061 dated Nov 14 (date only).
  The first slow run (SD7, 18:50) is BEFORE these downloads, so they cannot have started the slow period.
- Services installed 19:52 and 19:55 (names not in the summary line); IsolationSession / BITS start-type changes
  (routine update plumbing); time-zone refresh 20:17.
- Kernel-Power 566 "system session transitioned" at 00:02 and 00:18 (Nov 15) - still inside the slow period.
- **Nothing logged between 00:54 (last slow run) and 01:19 (first fast run).** No Defender scan events found;
  Task Scheduler log disabled (cannot check scheduled maintenance).
**Verdict:** no logged system event explains the slow -> fast change. Slow period stays UNEXPLAINED (negative result
preserved). Not reproducible with B/B2 conditions. Automatic env + clock logging is in place for future sessions.

# Pre-registration PERF1-C: can the established causes be removed? (2026-11-15 19:15, before running)
**C1 - idle wake-up (cause from A).** perf1_a.py with 1000 ms idle reset, 20 randomized trials per condition:
alone (D7 after idle) / warm2 (~2 ms memory streaming, then D7) / prefill (1024-token prefill of both models, as at a
real request start, then D7) / warm2_prefill (warm-up, prefill, D7). Prefill GPU time logged.
- **H190:** after a real prefill the first draft is already fast (<= 1.65 ms/step): in actual use the prefill wakes the
  GPU, so no extra warm-up is needed for decoding.
- **H191:** an explicit 2 ms warm-up before D7 makes D7 fast, but warm-up + D7 is NOT cheaper than D7 alone by more
  than 1 ms (the warm-up costs about what it saves) -> no net benefit for decoding.
- **H192:** the wake-up penalty moves to the prefill: prefill after idle is slower than after a warm-up by >= 1 ms;
  report whether warm-up + prefill is faster than prefill alone (if not, nothing to fix).
**C2 - VRAM guard (cause from B).** perf1_c2.py: warn at start-up if free VRAM < measured peak need + 0.5 GiB.
- **H193:** the guard says OK at helper 0 and 6 GiB and WARN at 10 GiB for the 1.7B configuration - matching the
  measured PERF1B slowdowns (none / none / -54%).
The validated engine stays frozen; any adopted fix would be a separate, re-validated version.

## PERF1-C run 1 (2026-11-15 19:09, results/PERF1C_20261115_1909_log.txt) - both parts invalid, fixed, rerun
- C1 crashed with the same '_len' error as Experiment A run 1: my edit for C1 was made on an older copy of
  perf1_a.py and overwrote the earlier fix. Fix re-applied (prefill the 1024-token prompt first). My process error.
- C2 ran but **H193 FAILED as implemented**: hipMemGetInfo (torch.cuda.mem_get_info) reported 15.77 GiB free in every
  case, also while the helper held 10 GiB. **Finding (B): on Windows, the HIP free-memory query does not see other
  processes' GPU memory**, so a guard built on it cannot work. The Windows counter "GPU Process Memory\Dedicated
  Usage" does see it (PERF1B logged 11.9 GB in use with the 10 GiB helper). C2 now uses that counter (logs both).

## PERF1-C run 2 (2026-11-15 19:13) - hung, stopped by the user; cause found
C1 stopped making progress right after calibration (no trial completed in 20 min). Cause (my bug): the research
prefill() did not reset the StaticCache write counter before re-prefilling 1024 tokens inside a trial (sd7b.py does:
setc(cache, 0)). The counter stood at ~1032, so the 1024-token prefill wrote past the end of the 1184-slot cache ->
out-of-bounds GPU writes -> hang. Experiment A never re-prefilled, so it was unaffected. Fixed (reset first).
The frozen engine was not involved (check OK before the run).

## Results PERF1-C run 3 (2026-11-15 19:34, results/PERF1C_20261115_1934_log.txt, PERF1C1_*)
1000 ms idle before every trial, 20 randomized trials per condition.
| before the draft | draft ms/step | clock before | prefill (1024 tok, both models) |
|---|---|---|---|
| nothing | 2.050 | 1190 | - |
| ~2 ms memory warm-up | 1.586 | 2732 | - |
| real prefill | 1.568 | 2733 | 137.44 ms |
| warm-up + prefill | 1.567 | 2732 | 137.45 ms |
- **H190 PASS:** after a real prefill the first draft is already fast (1.568). In real use the prompt prefill wakes the GPU.
- **H191 FAIL (narrowly):** warm-up + draft (~1.8 + 11.10 ms) is ~1.4 ms cheaper than the slow draft alone (14.35 ms),
  slightly more than the 1 ms predicted. Irrelevant in practice because every request starts with a prefill (H190).
- **H192 FAIL:** the wake-up penalty does NOT move into the prefill: 137.44 vs 137.45 ms with/without warm-up. A long
  prefill absorbs the wake-up. **Conclusion C1: nothing to fix for real use; no change to the engine.**
- **H193 PASS (with the Windows counter):** in use 1.43 / 7.66 / 11.67 GB -> 1.7B decisions OK / OK / WARN, matching the
  measured speeds (full / full / -54%). hipMemGetInfo said 15.77 GB free in all three cases (blind to other processes).
  4B (needs 14.14 GB): WARN already at 6 GB held elsewhere. **Conclusion C2: a start-up guard is worthwhile and must use
  the Windows counter.** Candidate for the next engine version (would be re-validated before adoption).
