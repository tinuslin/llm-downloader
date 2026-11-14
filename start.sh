#!/usr/bin/env bash
# LLM Downloader - Web UI launcher (Linux / macOS)
set -euo pipefail
cd "$(dirname "$0")"

PORT="${1:-8765}"
PY=".venv/bin/python"

if [ ! -x "$PY" ]; then
  echo "[1/3] Creating local virtual environment .venv ..."
  python3 -m venv .venv
fi

if ! "$PY" -c "import fastapi, uvicorn" >/dev/null 2>&1; then
  if [ -n "${PIP_INDEX_URL:-}" ]; then
    echo "[2/3] Installing Web dependencies (custom index) ..."
    "$PY" -m pip install --disable-pip-version-check -r requirements.txt
  else
    for M in "https://pypi.tuna.tsinghua.edu.cn/simple" "https://mirrors.aliyun.com/pypi/simple" ""; do
      if [ -n "$M" ]; then
        echo "[2/3] Installing Web dependencies (mirror: $M) ..."
        "$PY" -m pip install --disable-pip-version-check -i "$M" -r requirements.txt && break
      else
        echo "[2/3] Installing Web dependencies (official PyPI) ..."
        "$PY" -m pip install --disable-pip-version-check -r requirements.txt && break
      fi
    done
  fi
  "$PY" -c "import fastapi, uvicorn" >/dev/null 2>&1 || { echo "[ERROR] dependency install failed"; exit 1; }
else
  echo "[2/3] Dependencies OK"
fi

echo "[3/3] Starting at http://127.0.0.1:$PORT  --  Ctrl+C to stop"
exec "$PY" -m llm_downloader serve --open --port "$PORT"
