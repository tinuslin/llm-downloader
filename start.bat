@echo off
setlocal
cd /d "%~dp0"

echo ==================================================
echo   LLM Downloader  -  Web UI Launcher
echo ==================================================
echo.

where python >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python not found in PATH.
  echo         Install Python 3.10+ and check "Add python.exe to PATH".
  pause
  exit /b 1
)

set "PORT=%~1"
if "%PORT%"=="" set "PORT=8765"

if not exist ".venv\Scripts\python.exe" (
  echo [1/3] Creating local virtual environment ".venv" ...
  python -m venv .venv
  if errorlevel 1 goto :fail
) else (
  echo [1/3] Using existing .venv
)

set "PY=%~dp0.venv\Scripts\python.exe"

"%PY%" -c "import fastapi, uvicorn" >nul 2>nul
if not errorlevel 1 goto :deps_ok

echo [2/3] Installing Web dependencies ...
echo       Trying TUNA mirror ...
"%PY%" -m pip install --disable-pip-version-check -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
if not errorlevel 1 goto :deps_ok
echo       TUNA failed. Trying Aliyun mirror ...
"%PY%" -m pip install --disable-pip-version-check -i https://mirrors.aliyun.com/pypi/simple -r requirements.txt
if not errorlevel 1 goto :deps_ok
echo       Aliyun failed. Trying official PyPI ...
"%PY%" -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 goto :fail

:deps_ok
echo [2/3] Dependencies OK

set "OPEN_ARG=--open"
if not "%LLD_NO_OPEN%"=="" set "OPEN_ARG="

echo [3/3] Starting Web UI  --  Ctrl+C to stop
echo       Open the URL printed below. Default: http://127.0.0.1:8765
echo       Want another port? Run: start.bat 9000
echo       CLI needs no install: python -m llm_downloader recommend
echo.
"%PY%" -m llm_downloader serve %OPEN_ARG% --port %PORT%
if errorlevel 1 goto :fail
goto :eof

:fail
echo.
echo [ERROR] Launch failed. See the messages above.
pause
exit /b 1
