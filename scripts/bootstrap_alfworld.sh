#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${ALFWORLD_DATA:-/datasets/alfworld}"

echo "Using ALFWorld dataset dir: $DATA_DIR"
mkdir -p "$DATA_DIR"

if [ -d "$DATA_DIR/json_2.1.1" ]; then
  echo "ALFWorld dataset already installed."
  exit 0
fi

echo "Installing ALFWorld dataset..."
export ALFWORLD_DATA="$DATA_DIR"

uv run alfworld-download

echo "ALFWorld dataset installed successfully."