#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

ablations=(
  full
  no_memory
  no_regime_compatibility
  no_counterfactual_credibility
  factuality_only
  no_opponent_belief
  no_deviation_regularizer
  no_trust_gate
  no_language_credit
  single_timescale
)

cd "$PROJECT_ROOT"
echo "This launches paper-scale retraining for every ablation and five seeds."
for ablation in "${ablations[@]}"; do
  for seed in 0 1 2 3 4; do
    python -m trace_map.cli train \
      --config configs/base.yaml \
      --override configs/environment/e1.yaml \
      --override configs/communication/c0.yaml \
      --override "configs/ablation/$ablation.yaml" \
      --set "run.seed=$seed" \
      --output "results/generated/ablations/$ablation/seed_$seed"
  done
done
