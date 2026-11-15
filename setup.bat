@echo off
REM One-time setup for Windows: makes a private Python environment in .venv, installs AMD's ROCm build of PyTorch
REM and the other packages into it (nothing system-wide), then checks that PyTorch can see the GPU.
REM Needs: Python 3.12 (python.org), a recent AMD Adrenalin driver, an RX 9000-series GPU with 16 GB. ~5 GB download.
cd /d "%~dp0"
set BASEPY=
py -3.12 -c "1" >nul 2>nul && set BASEPY=py -3.12
if not defined BASEPY python -c "import sys; assert sys.version_info[:2]==(3,12)" >nul 2>nul && set BASEPY=python
if not defined BASEPY (
  echo Python 3.12 was not found. Install it from https://www.python.org/downloads/ and run setup.bat again.
  pause & exit /b 1
)
if not exist .venv\Scripts\python.exe (
  echo [1/4] creating .venv ...
  %BASEPY% -m venv .venv || (echo Could not create .venv & pause & exit /b 1)
)
set PY=.venv\Scripts\python.exe
%PY% -m pip install --upgrade pip >nul
REM RX 9070 / 9070 XT = gfx1201, RX 9060 XT = gfx1200. Override with: set GFX=gfx1200
if not defined GFX (
  powershell -NoProfile -Command "if ((Get-CimInstance Win32_VideoController).Name -match '9060') { exit 1 } else { exit 0 }"
  if errorlevel 1 (set GFX=gfx1200) else (set GFX=gfx1201)
)
echo [2/4] installing ROCm PyTorch for %GFX% (several GB, a few minutes) ...
%PY% -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ "torch[device-%GFX%]==2.14.0+rocm10.1.0" || (echo PyTorch install failed & pause & exit /b 1)
echo [3/4] installing the other packages ...
%PY% -m pip install -r requirements.txt || (echo Package install failed & pause & exit /b 1)
echo [4/4] checking the GPU ...
%PY% -c "import torch; ok = torch.cuda.is_available(); print('torch', torch.__version__, '| GPU visible:', ok, '|', torch.cuda.get_device_name(0) if ok else '-'); raise SystemExit(0 if ok else 1)" || (echo PyTorch cannot see the GPU. Update the AMD driver and try again. & pause & exit /b 1)
echo.
echo Setup done. Try:  .venv\Scripts\python generate.py "Explain what a hash table is."
echo (the first run downloads Qwen3-8B, about 16 GB)
if not defined NOPAUSE pause
