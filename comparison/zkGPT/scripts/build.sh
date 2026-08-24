#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:---all}"
case "${MODE}" in
  --cpu)
    python3 "${ROOT}/scripts/formal_preflight.py" --source-only
    python3 -m py_compile "${ROOT}"/src/zkgpt_gat/*.py "${ROOT}"/scripts/*.py
    ;;
  --gpu)
    python3 "${ROOT}/scripts/formal_preflight.py" --source-only
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
