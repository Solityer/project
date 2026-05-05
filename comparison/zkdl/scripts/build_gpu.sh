#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
cmake -S . -B build/gpu -DCMAKE_BUILD_TYPE=Release -DZKDL_ENABLE_CUDA_BACKEND=ON -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build/gpu -j
