#!/usr/bin/env bash
# Linux (untested): Qwen3-8B lossless B12 + exact speculative decoding, 11 prompts x 96 tokens.
cd "$(dirname "$0")/.." && mkdir -p results
PY=${PY:-$( [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3 )}
SD_PROMPTSET=b6 SD_NEWT=96 SD_REPS=2 SD_WARM=1 SD_OUT=REPL_8b $PY -u benchmarks/sd8b.py 2>&1 | tee results/REPL_8b_log.txt | grep -E "CAP1c|RESULT|Error|Traceback"
