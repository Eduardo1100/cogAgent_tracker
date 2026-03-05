#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="/app/datasets/alfworld"

if [ -d "$DATA_DIR/json_2.1.1" ]; then
  echo "ALFWorld dataset already installed."
  exit 0
fi

echo "Installing ALFWorld dataset..."

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

# JSON tasks
wget -q https://storage.googleapis.com/alfworld/json_2.1.1_json.zip
unzip -q json_2.1.1_json.zip
rm json_2.1.1_json.zip

# PDDL files
wget -q https://storage.googleapis.com/alfworld/json_2.1.1_pddl.zip
unzip -q json_2.1.1_pddl.zip
rm json_2.1.1_pddl.zip

# TextWorld game files
wget -q https://storage.googleapis.com/alfworld/json_2.1.3_tw-pddl.zip
unzip -q json_2.1.3_tw-pddl.zip
rm json_2.1.3_tw-pddl.zip

# Mask-RCNN detector
wget -q https://storage.googleapis.com/alfworld/mrcnn_alfred_objects_sep13_004.pth

echo "ALFWorld dataset installed successfully."