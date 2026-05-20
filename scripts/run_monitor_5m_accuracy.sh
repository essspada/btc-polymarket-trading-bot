#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ -z "${PYTHON_BIN}" || ! -x "$(command -v "${PYTHON_BIN}")" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON_BIN=".venv/bin/python"
  fi
fi

"${PYTHON_BIN}" -m src.main --config configs/default.yaml --mode monitor-5m "$@"
