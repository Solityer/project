#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

source /home/pzh/.cargo/env 2>/dev/null || true
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-2}"
EZKL_BUILD_JOBS="${EZKL_BUILD_JOBS:-2}"
EZKL_CARGO_FEATURES="${EZKL_CARGO_FEATURES:-ezkl,no-banner,mimalloc}"
cargo test native_gat --no-default-features --features "${EZKL_CARGO_FEATURES}"
cargo build --release --bin ezkl --no-default-features --features "${EZKL_CARGO_FEATURES}"
cmake -S cpp/gat_ezkl_adapter -B build_gat_adapter -DCMAKE_BUILD_TYPE=Release
cmake --build build_gat_adapter -j"${EZKL_BUILD_JOBS}"
