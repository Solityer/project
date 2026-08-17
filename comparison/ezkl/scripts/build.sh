#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-2}"
cargo build --release --locked --bin ezkl --no-default-features --features ezkl,no-banner,mimalloc
