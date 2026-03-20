#!/usr/bin/env bash
set -euo pipefail

WEBARENA_REPO_URL="${WEBARENA_REPO_URL:-https://github.com/web-arena-x/webarena.git}"
WEBARENA_REPO_REV="${WEBARENA_REPO_REV:-dce04686a56253aefba7b18a4fa0937cf1dc987b}"
WEBARENA_REPO_DIR="${WEBARENA_REPO_DIR:-.cache/webarena-src}"

if [ -n "${WEBARENA_PYTHON:-}" ]; then
  PYTHON_BIN="$WEBARENA_PYTHON"
elif [ -x ".venv/bin/python" ]; then
  PYTHON_BIN=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
else
  echo "Could not locate a Python interpreter for WebArena bootstrap."
  exit 1
fi

NLTK_DATA_DIR="${NLTK_DATA:-.nltk_data}"
mkdir -p "$NLTK_DATA_DIR"
export NLTK_DATA="$NLTK_DATA_DIR"

mkdir -p "$(dirname "$WEBARENA_REPO_DIR")"
if [ ! -d "$WEBARENA_REPO_DIR/.git" ]; then
  echo "Cloning WebArena source checkout into $WEBARENA_REPO_DIR..."
  git clone "$WEBARENA_REPO_URL" "$WEBARENA_REPO_DIR"
fi
git -C "$WEBARENA_REPO_DIR" fetch --depth 1 origin "$WEBARENA_REPO_REV"
git -C "$WEBARENA_REPO_DIR" checkout --detach "$WEBARENA_REPO_REV"

if ! "$PYTHON_BIN" -c "import browsergym.webarena, webarena, playwright" >/dev/null 2>&1; then
  echo "Installing BrowserGym WebArena runtime..."
  uv pip install --python "$PYTHON_BIN" \
    browsergym-core==0.14.3 \
    browsergym-webarena==0.14.3 \
    libwebarena==0.0.4 \
    playwright
  uv pip install --python "$PYTHON_BIN" --no-deps \
    "git+$WEBARENA_REPO_URL@$WEBARENA_REPO_REV"
else
  echo "BrowserGym WebArena runtime already installed."
fi

echo "Ensuring NLTK punkt_tab is installed..."
"$PYTHON_BIN" - <<'PY'
import os
import nltk

download_dir = os.environ["NLTK_DATA"]
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", download_dir=download_dir, quiet=False, raise_on_error=True)
PY

echo "Ensuring Playwright Chromium is installed..."
"$PYTHON_BIN" -m playwright install chromium

echo "WebArena bootstrap completed."
