#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
cmake -S . -B build_cuda -DCMAKE_BUILD_TYPE=Release -DZKDL_GAT_ENABLE_CUDA=ON
cmake --build build_cuda -j
