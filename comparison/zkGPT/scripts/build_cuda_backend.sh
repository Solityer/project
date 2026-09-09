#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${ROOT}/backend/cuda/zkgpt_cuda_proof_hot_path.cu"
OUT="${ROOT}/build/zkgpt_cuda_native_prover"
MANIFEST="${ROOT}/build/cuda_build_manifest.json"

PYTHON_BIN="${ZKGPT_PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(python3 "${ROOT}/scripts/ensure_environment.py" --print-python)"
fi
if [[ "${PYTHON_BIN}" != "${ROOT}/.venv/bin/python" ]]; then
  echo "zkGPT CUDA build requires baseline-local interpreter: ${ROOT}/.venv/bin/python" >&2
  exit 1
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "zkGPT baseline-local interpreter is missing: ${PYTHON_BIN}" >&2
  exit 1
fi

NVCC="${ZKGPT_NVCC_BIN:-}"
if [[ -z "${NVCC}" ]]; then
  NVCC="$(command -v nvcc || true)"
fi
if [[ -z "${NVCC}" || ! -x "${NVCC}" ]]; then
  echo "nvcc is required for the native zkGPT GPU backend" >&2
  exit 1
fi
NVCC="$(realpath -e "${NVCC}")"

compute_capability="${ZKGPT_CUDA_COMPUTE_CAPABILITY:-}"
if [[ -z "${compute_capability}" && -x "$(command -v nvidia-smi || true)" ]]; then
  compute_capability="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | awk 'NF {print $1; exit}')"
fi
if [[ -z "${compute_capability}" ]]; then
  echo "cannot determine CUDA compute capability; set ZKGPT_CUDA_COMPUTE_CAPABILITY (for example 12.1)" >&2
  exit 1
fi
if [[ ! "${compute_capability}" =~ ^[0-9]+\.[0-9]+$ ]]; then
  echo "invalid CUDA compute capability: ${compute_capability}" >&2
  exit 1
fi
arch_flag="${compute_capability/./}"
supported_codes="$(${NVCC} --list-gpu-code 2>/dev/null || true)"
if ! printf '%s\n' "${supported_codes}" | grep -Eq "(^|[[:space:]])sm_${arch_flag}($|[[:space:]])"; then
  echo "nvcc does not support the requested native CUDA target sm_${arch_flag}" >&2
  exit 1
fi

for required_dir in "${ROOT}/build" "${ROOT}/runs_tmp"; do
  if [[ -L "${required_dir}" || ( -e "${required_dir}" && ! -d "${required_dir}" ) ]]; then
    echo "CUDA build directory must be a real directory: ${required_dir}" >&2
    exit 1
  fi
  mkdir -p "${required_dir}"
done
if [[ -L "${OUT}" || ( -e "${OUT}" && ! -f "${OUT}" ) ]]; then
  echo "CUDA output must be a regular file, not an alias or special file: ${OUT}" >&2
  exit 1
fi
if [[ -f "${OUT}" && "$(stat -c '%h' "${OUT}")" != "1" ]]; then
  echo "CUDA output hardlink count must be one: ${OUT}" >&2
  exit 1
fi
if [[ -x "${OUT}" && -f "${MANIFEST}" ]]; then
  if PYTHONPATH="${ROOT}/src" "${PYTHON_BIN}" -c '
import pathlib, sys
from zkgpt_gat.build_identity import verify_build_manifest
manifest = verify_build_manifest(pathlib.Path(sys.argv[1]), require_nvcc=True)
if manifest.get("target", {}).get("compute_capability") != sys.argv[2]:
    raise ValueError("CUDA build target differs from the requested device capability")
' "${ROOT}" "${compute_capability}"; then
      exit 0
  fi
fi

compile_command=(
  "${NVCC}"
  -std=c++17
  -O3
  -lineinfo
  "-gencode=arch=compute_${arch_flag},code=sm_${arch_flag}"
  "${SRC}"
  -o
  "${OUT}"
)
"${compile_command[@]}"
"${PYTHON_BIN}" "${ROOT}/scripts/write_build_manifest.py" \
  --root "${ROOT}" \
  --binary "${OUT}" \
  --nvcc "${NVCC}" \
  --target-compute-capability "${compute_capability}" \
  --arch-flag "${arch_flag}" \
  --compile-command "${compile_command[@]}"
