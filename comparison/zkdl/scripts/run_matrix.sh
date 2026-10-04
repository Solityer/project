#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
if [[ "$#" -ne 0 ]]; then
    echo "run-matrix: compatibility alias accepts no arguments; use run_five.sh as the sole formal entry" >&2
    exit 2
fi
exec "$script_dir/run_five.sh"
