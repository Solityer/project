#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"${ROOT}/scripts/build_cuda_backend.sh"
"${ROOT}/scripts/gat_zkgpt_experiment.py" --datasets all --backends all "$@"

