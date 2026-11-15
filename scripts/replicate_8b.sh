#!/usr/bin/env bash
# Qwen3-8B, lossless B12 weights + exact speculative decoding, 11 prompts x 96 tokens.
cd "$(dirname "$0")/.." && mkdir -p results
SD_PROMPTSET=b6 SD_NEWT=96 SD_REPS=2 SD_WARM=1 SD_OUT=REPL_8b python -u research/cap1/sd8b.py 2>&1 | tee results/REPL_8b_log.txt | grep -E "CAP1c|RESULT|Error|Traceback"
