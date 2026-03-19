#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${WEBARENA_PYTHON:-.venv/bin/python}"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Expected Python interpreter at $PYTHON_BIN"
  exit 1
fi

if ! "$PYTHON_BIN" -c "import browsergym.webarena, webarena, playwright" >/dev/null 2>&1; then
  echo "Installing BrowserGym WebArena runtime..."
  uv pip install --python "$PYTHON_BIN" \
    browsergym-core==0.14.3 \
    browsergym-webarena==0.14.3 \
    libwebarena==0.0.4 \
    playwright
  uv pip install --python "$PYTHON_BIN" --no-deps \
    git+https://github.com/web-arena-x/webarena.git
else
  echo "BrowserGym WebArena runtime already installed."
fi

echo "Ensuring Playwright Chromium is installed..."
"$PYTHON_BIN" -m playwright install chromium

echo "WebArena bootstrap completed."
