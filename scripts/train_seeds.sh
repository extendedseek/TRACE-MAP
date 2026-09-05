#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

cd "$PROJECT_ROOT"
for seed in 0 1 2 3 4; do
  python -m trace_map.cli train \
    --config configs/base.yaml \
    --override configs/environment/e1.yaml \
    --override configs/communication/c0.yaml \
    --set "run.seed=$seed" \
    --output "results/generated/train/seed_$seed"
done
