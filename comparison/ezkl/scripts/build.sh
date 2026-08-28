#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
export PYTHONDONTWRITEBYTECODE=1

# The baseline is self-contained at the path/configuration level.  Toolchain
# discovery is intentionally delegated to PATH; do not source a developer
# machine's shell fragment or inherit a path outside this baseline.
if ! CARGO_BIN="$(command -v cargo)" || [[ -z "${CARGO_BIN}" ]]; then
  echo "missing cargo in PATH" >&2
  exit 2
fi
if ! RUSTC_BIN="$(command -v rustc)" || [[ -z "${RUSTC_BIN}" ]]; then
  echo "missing rustc in PATH" >&2
  exit 2
fi

# rust-toolchain pins the nightly channel used by the vendored Halo2 stack.
# Keep the release line explicit and fail closed when a different compiler is
# selected instead of silently producing an unauthenticated build identity.
REQUIRED_CARGO_VERSION="1.93.0-nightly"
REQUIRED_RUSTC_VERSION="1.93.0-nightly"
cargo_version="$(${CARGO_BIN} --version 2>/dev/null | awk '{print $2}')"
rustc_version="$(${RUSTC_BIN} --version 2>/dev/null | awk '{print $2}')"
if [[ "${cargo_version}" != "${REQUIRED_CARGO_VERSION}" ]]; then
  echo "unsupported cargo version: got ${cargo_version:-<unknown>}, require ${REQUIRED_CARGO_VERSION}" >&2
  exit 2
fi
if [[ "${rustc_version}" != "${REQUIRED_RUSTC_VERSION}" ]]; then
  echo "unsupported rustc version: got ${rustc_version:-<unknown>}, require ${REQUIRED_RUSTC_VERSION}" >&2
  exit 2
fi
export RUSTC="${RUSTC_BIN}"

export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-2}"
EZKL_BUILD_JOBS="${EZKL_BUILD_JOBS:-2}"
EZKL_CARGO_FEATURES="${EZKL_CARGO_FEATURES:-formal}"
if [[ "${EZKL_CARGO_FEATURES}" != "formal" ]]; then
  echo "formal ezkl build accepts only the formal production feature set" >&2
  exit 2
fi
"${CARGO_BIN}" metadata --offline --locked --no-deps --format-version 1 >/dev/null
"${CARGO_BIN}" build --offline --locked --release --bin ezkl_formal --no-default-features --features "${EZKL_CARGO_FEATURES}"
if [[ ! -x "${ROOT}/target/release/ezkl_formal" ]]; then
  echo "formal CPU binary was not produced" >&2
  exit 2
fi
if [[ "${EZKL_BUILD_GPU:-0}" == "1" ]]; then
  CARGO_TARGET_DIR="${ROOT}/target-gpu" \
    "${CARGO_BIN}" build --offline --locked --release --bin ezkl_formal --no-default-features \
      --features "${EZKL_CARGO_FEATURES},gpu-accelerated"
  if [[ ! -x "${ROOT}/target-gpu/release/ezkl_formal" ]]; then
    echo "formal CUDA binary was not produced" >&2
    exit 2
  fi
fi
mkdir -p "${ROOT}/build_identity"
cmake -S cpp/gat_ezkl_adapter -B build_gat_adapter -DCMAKE_BUILD_TYPE=Release
cmake --build build_gat_adapter -j"${EZKL_BUILD_JOBS}"
python3 scripts/write_build_identity.py \
  --backend cpu \
  --binary target/release/ezkl_formal \
  --adapter build_gat_adapter/gat_ezkl_export \
  --features "${EZKL_CARGO_FEATURES}" \
  --output build_identity/cpu.json
if [[ "${EZKL_BUILD_GPU:-0}" == "1" ]]; then
  python3 scripts/write_build_identity.py \
    --backend cuda \
    --binary target-gpu/release/ezkl_formal \
    --adapter build_gat_adapter/gat_ezkl_export \
    --features "${EZKL_CARGO_FEATURES},gpu-accelerated" \
    --output build_identity/cuda.json
fi
