#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
if (( $# != 0 )); then
    echo "usage: ./run_zkgpt_formal.sh" >&2
    exit 2
fi

exec "$ROOT/run_zkgpt_formal.sh"
