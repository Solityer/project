#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/home/pzh/project/comparison/ezkl
LOG_DIR="$(cd "$(dirname "$0")" && pwd -P)"
cd "$ROOT"

exec >> "$LOG_DIR/runner.log" 2>&1

finish() {
    rc=$?
    if (( rc == 0 )); then
        printf 'PASSED\n' > "$LOG_DIR/status"
    else
        printf 'FAILED\n' > "$LOG_DIR/status"
    fi
    printf '%s\n' "$rc" > "$LOG_DIR/exit_code"
    printf 'FINISHED time=%s exit_code=%s\n' "$(date -Is)" "$rc"
}
trap finish EXIT

printf 'RUNNING\n' > "$LOG_DIR/status"
printf 'START time=%s commit=%s\n' "$(date -Is)" "$(git rev-parse HEAD)"

unset LD_PRELOAD RUSTFLAGS CARGO_ENCODED_RUSTFLAGS CARGO_TARGET_DIR CARGO_BUILD_TARGET
export RUSTUP_TOOLCHAIN=nightly-2025-12-01
export CARGO_HOME="$ROOT/.cargo_local"
export CARGO_BUILD_JOBS=2
export PATH="$HOME/.cargo/bin:$PATH"
export ICICLE_BACKEND_INSTALL_DIR="$ROOT/artifacts/build/icicle-cuda"
export ICICLE_NATIVE_LIB_DIR="$ICICLE_BACKEND_INSTALL_DIR/lib"
export ICICLE_NATIVE_MANIFEST_SHA256="$(sha256sum "$ICICLE_BACKEND_INSTALL_DIR/native_backend_manifest.json" | awk '{print $1}')"

free -h
df -h "$ROOT"
nvidia-smi || true

run_test() {
    name="$1"
    shift
    printf 'START_TEST time=%s name=%s\n' "$(date -Is)" "$name"

    if "$@" > "$LOG_DIR/$name.log" 2>&1; then
        printf 'PASS_TEST time=%s name=%s\n' "$(date -Is)" "$name"
    else
        rc=$?
        printf 'FAIL_TEST time=%s name=%s exit_code=%s\n' "$(date -Is)" "$name" "$rc"
        tail -n 100 "$LOG_DIR/$name.log"
        return "$rc"
    fi
}

run_test 01_region_increment \
    cargo test --offline --locked --release --no-default-features \
    --features formal-gpu --target-dir target/gpu \
    --test gat_adapter_relation \
    region_bulk_increment_matches_elementwise_next_at_column_boundaries \
    -- --exact --nocapture

run_test 02_native_gat \
    cargo test --offline --locked --release --no-default-features \
    --features formal-gpu --target-dir target/gpu \
    --test gat_adapter_relation \
    native_gat_ \
    -- --nocapture --test-threads=1

echo "ALL_TARGETED_TESTS_PASSED"
