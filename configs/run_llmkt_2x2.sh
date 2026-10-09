#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
bash "$ROOT/configs/train_2x2.sh" "${1:-mathdial}" lmkt
