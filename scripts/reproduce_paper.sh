#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$PROJECT_ROOT/results/generated/train}"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

environment_overlays=(
  "e1:configs/environment/e1.yaml"
  "e2:configs/environment/e2.yaml"
  "e3:configs/environment/e3.yaml"
)
communication_overlays=(
  "c0:configs/communication/c0.yaml"
  "c1:configs/communication/c1.yaml"
  "c2_010:configs/communication/c2_010.yaml"
  "c2_030:configs/communication/c2_030.yaml"
  "c2_050:configs/communication/c2_050.yaml"
  "c3:configs/communication/c3.yaml"
)
shift_overlays=(
  "e1_to_e2:configs/environment/e1_to_e2.yaml"
  "e1_to_e3:configs/environment/e1_to_e3.yaml"
)

cd "$PROJECT_ROOT"
for seed in 0 1 2 3 4; do
  checkpoint="$CHECKPOINT_ROOT/seed_$seed/checkpoints/step_000300000.pt"
  if [[ ! -f "$checkpoint" ]]; then
    echo "Missing frozen checkpoint: $checkpoint" >&2
    echo "Run scripts/train_seeds.sh or set CHECKPOINT_ROOT." >&2
    exit 1
  fi

  for environment_entry in "${environment_overlays[@]}"; do
    environment_name="${environment_entry%%:*}"
    environment_file="${environment_entry#*:}"
    for communication_entry in "${communication_overlays[@]}"; do
      communication_name="${communication_entry%%:*}"
      communication_file="${communication_entry#*:}"
      python -m trace_map.cli evaluate \
        --config configs/base.yaml \
        --override "$environment_file" \
        --override "$communication_file" \
        --set "run.seed=$seed" \
        --checkpoint "$checkpoint" \
        --output "results/generated/evaluation/${environment_name}_${communication_name}/seed_$seed"
    done
  done

  for shift_entry in "${shift_overlays[@]}"; do
    shift_name="${shift_entry%%:*}"
    shift_file="${shift_entry#*:}"
    python -m trace_map.cli evaluate \
      --config configs/base.yaml \
      --override "$shift_file" \
      --override configs/communication/c0.yaml \
      --set "run.seed=$seed" \
      --checkpoint "$checkpoint" \
      --output "results/generated/evaluation/${shift_name}_c0/seed_$seed"
  done
done

python scripts/summarize_results.py \
  --input results/generated/evaluation \
  --csv results/generated/evaluation_summary.csv \
  --markdown results/generated/evaluation_summary.md
