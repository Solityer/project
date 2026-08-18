#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
mkdir -p bin
CARGO_TARGET_DIR=target/formal-cpu cargo build --release --locked --bin zkdl-formal
cp target/formal-cpu/release/zkdl-formal bin/zkdl-formal-cpu
CARGO_TARGET_DIR=target/formal-gpu cargo build --release --locked --bin zkdl-formal --features gpu
cp target/formal-gpu/release/zkdl-formal bin/zkdl-formal-gpu
