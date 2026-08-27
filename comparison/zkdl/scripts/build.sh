#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
mode="${1:---all}"
case "${mode}" in
  --cpu|--gpu|--all) ;;
  *) echo "usage: $0 [--cpu|--gpu|--all]" >&2; exit 2 ;;
esac
mkdir -p bin
if [[ "${mode}" == "--cpu" || "${mode}" == "--all" ]]; then
  CARGO_TARGET_DIR=target/formal-cpu cargo build --release --locked --bin zkdl-formal
  cp target/formal-cpu/release/zkdl-formal bin/zkdl-formal-cpu
  python3 scripts/record_build.py --device cpu --binary bin/zkdl-formal-cpu
fi
if [[ "${mode}" == "--gpu" || "${mode}" == "--all" ]]; then
  command -v "${NVCC:-nvcc}" >/dev/null || { echo "CUDA nvcc is required for the formal GPU binary" >&2; exit 1; }
  CARGO_TARGET_DIR=target/formal-gpu cargo build --release --locked --bin zkdl-formal --features gpu
  cp target/formal-gpu/release/zkdl-formal bin/zkdl-formal-gpu
  python3 scripts/record_build.py --device gpu --binary bin/zkdl-formal-gpu
fi
