#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
if (( $# != 0 )); then
    echo "usage: ./run_zkgpt_formal.sh" >&2
    exit 2
fi

if [[ -n "${PYTHON:-}" ]]; then
    PYTHON_BIN="$PYTHON"
else
    PYTHON_BIN="$(command -v python3 || command -v python || true)"
fi
if [[ -z "$PYTHON_BIN" ]]; then
    echo "error: no Python interpreter found" >&2
    exit 127
fi

exec "$PYTHON_BIN" "$ROOT/scripts/run_formal.py"
