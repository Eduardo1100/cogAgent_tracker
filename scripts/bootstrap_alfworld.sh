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
cd "$DATA_DIR"

wget -q https://storage.googleapis.com/alfworld/json_2.1.1_json.zip
unzip -q json_2.1.1_json.zip
rm json_2.1.1_json.zip

wget -q https://storage.googleapis.com/alfworld/json_2.1.1_pddl.zip
unzip -q json_2.1.1_pddl.zip
rm json_2.1.1_pddl.zip

wget -q https://storage.googleapis.com/alfworld/json_2.1.3_tw-pddl.zip
unzip -q json_2.1.3_tw-pddl.zip
rm json_2.1.3_tw-pddl.zip

wget -q https://storage.googleapis.com/alfworld/mrcnn_alfred_objects_sep13_004.pth

echo "ALFWorld dataset installed successfully."