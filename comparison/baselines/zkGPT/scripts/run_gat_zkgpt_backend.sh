#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <cpu|gpu> [dataset-list|all] [extra args...]" >&2
  exit 2
fi

BACKEND="$1"
DATASETS="${2:-all}"
shift || true
shift || true

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-/home/pzh/project}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi

cd "${ROOT_DIR}"
"${PYTHON_BIN}" scripts/gat_zkgpt_experiment.py \
  --project-root "${PROJECT_ROOT}" \
  --results-dir "${ROOT_DIR}/results" \
  --datasets "${DATASETS}" \
  --backends "${BACKEND}" \
  "$@"
