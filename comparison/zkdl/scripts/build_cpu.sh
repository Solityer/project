#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
cmake -S . -B build/cpu -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpu -j
