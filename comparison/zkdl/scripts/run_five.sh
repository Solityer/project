#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
if [[ "$#" -ne 0 ]]; then
    echo "run-five: usage: $0" >&2
    exit 2
fi
exec python3 "$script_dir/formal_runner.py"
