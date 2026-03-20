#!/usr/bin/env bash
set -euo pipefail

ANDROIDWORLD_REPO_URL="${ANDROIDWORLD_REPO_URL:-https://github.com/google-research/android_world.git}"
ANDROIDWORLD_REPO_REF="${ANDROIDWORLD_REPO_REF:-d9c569f764b3a5629321858de03ff653d0f24056}"
ANDROIDWORLD_REPO_DIR="${ANDROIDWORLD_REPO_DIR:-.cache/android_world-src}"

if [ -n "${ANDROIDWORLD_PYTHON:-}" ]; then
  PYTHON_BIN="$ANDROIDWORLD_PYTHON"
elif [ -x ".venv/bin/python" ]; then
  PYTHON_BIN=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
else
  echo "Could not locate a Python interpreter for AndroidWorld bootstrap."
  exit 1
fi

mkdir -p "$(dirname "$ANDROIDWORLD_REPO_DIR")"
if [ ! -d "$ANDROIDWORLD_REPO_DIR/.git" ]; then
  echo "Cloning AndroidWorld source into $ANDROIDWORLD_REPO_DIR..."
  git clone "$ANDROIDWORLD_REPO_URL" "$ANDROIDWORLD_REPO_DIR"
fi

git -C "$ANDROIDWORLD_REPO_DIR" fetch --depth 1 origin "$ANDROIDWORLD_REPO_REF"
git -C "$ANDROIDWORLD_REPO_DIR" checkout --detach "$ANDROIDWORLD_REPO_REF"

echo "Installing AndroidWorld requirements..."
uv pip install --python "$PYTHON_BIN" -r "$ANDROIDWORLD_REPO_DIR/requirements.txt"
uv pip install --python "$PYTHON_BIN" setuptools
uv pip install --python "$PYTHON_BIN" --no-build-isolation -e "$ANDROIDWORLD_REPO_DIR"

echo "AndroidWorld bootstrap completed."
