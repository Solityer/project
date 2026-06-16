#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

source /home/pzh/.cargo/env 2>/dev/null || true
cargo test native_gat --no-default-features --features ezkl,no-banner
cargo build --release --bin ezkl --no-default-features --features ezkl,no-banner
cmake -S cpp/gat_ezkl_adapter -B build_gat_adapter -DCMAKE_BUILD_TYPE=Release
cmake --build build_gat_adapter -j
