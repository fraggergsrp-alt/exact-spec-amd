@echo off
REM Qwen3-1.7B: speculative vs plain decoding, standard PyTorch matmuls (BI=0) vs batch-invariant kernels (BI=1). ~10 min.
cd /d "%~dp0\.."
if not defined PY if exist ".venv\Scripts\python.exe" set PY=".venv\Scripts\python.exe"
if not defined PY set PY=python
if not exist results mkdir results
set FREE_DUP=1
set SD7_MODES=fixed7
set SD_PROMPTSET=b6
set BI=0
set SD_OUT=REPL_17_standard
%PY% -u benchmarks\sd7b.py > results\REPL_17_log.txt 2>&1
set BI=1
set SD_OUT=REPL_17_exact
%PY% -u benchmarks\sd7b.py >> results\REPL_17_log.txt 2>&1
findstr /C:"RESULT" /C:"Error" /C:"Traceback" results\REPL_17_log.txt
echo Full log: results\REPL_17_log.txt
if not defined NOPAUSE pause
