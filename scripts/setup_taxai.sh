#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TAXAI_DIR="$PROJECT_ROOT/third_party/TaxAI"
TAXAI_URL="https://github.com/jidiai/TaxAI.git"
TAXAI_COMMIT="04e7cb17071d942366eb0cad4fb4ba57d02bf612"

if [[ -d "$TAXAI_DIR/.git" ]]; then
  actual_commit="$(git -C "$TAXAI_DIR" rev-parse HEAD)"
  if [[ "$actual_commit" != "$TAXAI_COMMIT" ]]; then
    echo "Existing TaxAI checkout is at $actual_commit; expected $TAXAI_COMMIT." >&2
    echo "Move that directory aside and rerun this script; it will not overwrite local work." >&2
    exit 1
  fi
  echo "TaxAI is already pinned at $TAXAI_COMMIT"
  exit 0
fi

if [[ -e "$TAXAI_DIR" ]]; then
  echo "$TAXAI_DIR exists but is not a Git checkout; refusing to overwrite it." >&2
  exit 1
fi

mkdir -p "$PROJECT_ROOT/third_party"
git clone "$TAXAI_URL" "$TAXAI_DIR"
git -C "$TAXAI_DIR" checkout --detach "$TAXAI_COMMIT"
echo "Installed TaxAI at $TAXAI_COMMIT"
