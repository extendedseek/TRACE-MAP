#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TRACE_TMP_DIR="$(mktemp -d)"
trap 'rm -rf -- "$TRACE_TMP_DIR"' EXIT
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

cd "$PROJECT_ROOT"
python -m compileall -q src tests
python -m unittest discover -s tests -v
python -m trace_map.cli smoke --config configs/smoke.yaml --output "$TRACE_TMP_DIR/smoke"
test -s "$TRACE_TMP_DIR/smoke/metrics.json"
test -s "$TRACE_TMP_DIR/smoke/decision_trace.jsonl"
echo "Repository verification passed."
