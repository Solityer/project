#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ZKGPT_PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(python3 "${ROOT}/scripts/ensure_environment.py" --print-python)"
fi
case "${PYTHON_BIN}" in
  "${ROOT}/".venv/bin/python) ;;
  *) echo "zkGPT build requires baseline-local interpreter: ${ROOT}/.venv/bin/python" >&2; exit 1 ;;
esac
MODE="${1:---all}"
case "${MODE}" in
  --cpu)
    "${PYTHON_BIN}" "${ROOT}/scripts/formal_preflight.py" --source-only
    "${PYTHON_BIN}" -m py_compile "${ROOT}"/src/zkgpt_gat/*.py "${ROOT}"/scripts/*.py
    ;;
  --gpu)
    "${PYTHON_BIN}" "${ROOT}/scripts/formal_preflight.py" --source-only
    "${ROOT}/scripts/build_cuda_backend.sh"
    ;;
  --all)
    "${ROOT}/scripts/build.sh" --cpu
    "${ROOT}/scripts/build.sh" --gpu
    ;;
  *)
    echo "usage: $0 [--cpu|--gpu|--all]" >&2
    exit 2
    ;;
esac
