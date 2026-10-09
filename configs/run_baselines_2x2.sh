#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
for model in dkt simplekt dkt-sem; do
 bash "$ROOT/configs/train_2x2.sh" "${1:-mathdial}" "$model"
done
