#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${ROOT}/backend/cuda/zkgpt_cuda_digest.cu"
OUT="${ROOT}/build/zkgpt_cuda_digest"

mkdir -p "${ROOT}/build" "${ROOT}/runs_tmp"
if [[ -x "${OUT}" && "${OUT}" -nt "${SRC}" ]]; then
  exit 0
fi

nvcc -std=c++17 -O3 "${SRC}" -o "${OUT}"

