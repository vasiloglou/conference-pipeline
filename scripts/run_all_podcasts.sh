#!/bin/bash
# Compatibility wrapper for the canonical workflow CLI podcast stage.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python3}"
fi

cd "$PROJECT_ROOT"

"$PYTHON_BIN" -m conference_pipeline.cli workflow run --workspace conferences/neurips2025 --stages podcast "$@"
