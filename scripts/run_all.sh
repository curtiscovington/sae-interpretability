#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/default.yaml}"

python -m src.collect_acts --config "$CONFIG_PATH"
python -m src.train_sae --config "$CONFIG_PATH"
python -m src.interpret --config "$CONFIG_PATH" --label A
python -m src.interpret --config "$CONFIG_PATH" --label B
python -m src.eval --config "$CONFIG_PATH"
python -m src.viz --config "$CONFIG_PATH"

echo "Done. See outputs/results.json, outputs/tables, outputs/figures, outputs/features"
