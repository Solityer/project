#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <cpu|gpu|all> [dataset|all] [extra args...]" >&2
  exit 2
fi

BACKEND="$1"
shift
if [[ $# -gt 0 ]]; then
  DATASETS="$1"
  shift
else
  DATASETS="all"
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"${ROOT}/scripts/gat_zkgpt_experiment.py" --datasets "${DATASETS}" --backends "${BACKEND}" "$@"
