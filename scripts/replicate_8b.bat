@echo off
REM Qwen3-8B, lossless B12 weights + exact speculative decoding, 11 prompts x 96 tokens. ~10 min (first run downloads ~16 GB).
cd /d "%~dp0\.."
set PY=python
if exist ".venv-gpu\Scripts\python.exe" set PY=".venv-gpu\Scripts\python.exe"
if not exist results mkdir results
set SD_PROMPTSET=b6
set SD_NEWT=96
set SD_REPS=2
set SD_WARM=1
set SD_OUT=REPL_8b
%PY% -u research\cap1\sd8b.py > results\REPL_8b_log.txt 2>&1
findstr /C:"CAP1c" /C:"RESULT" /C:"Error" /C:"Traceback" results\REPL_8b_log.txt
echo Full log: results\REPL_8b_log.txt
pause
