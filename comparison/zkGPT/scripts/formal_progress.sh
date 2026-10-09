#!/usr/bin/env bash
set -euo pipefail
SCRIPT_PATH="$(realpath -e -- "${BASH_SOURCE[0]}")"
ROOT="$(realpath -e -- "$(dirname -- "$SCRIPT_PATH")/..")"
exec python3 "$ROOT/scripts/formal_progress.py" "$@"
