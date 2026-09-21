#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$ROOT"

if [[ -n "${PYTHON:-}" ]]; then
    PYTHON_BIN="$PYTHON"
else
    PYTHON_BIN="$(command -v python3 || command -v python || true)"
fi
if [[ -z "${PYTHON_BIN}" ]]; then
    echo "error: no current Python interpreter found (python3 or python)" >&2
    exit 127
fi

mkdir -p "$ROOT/logs"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$ROOT/logs/zkgpt_cpu_formal_${STAMP}.log"

# Keep the complete stage stream visible to the caller while retaining the
# exact same bytes in the per-run log for postmortem/debugging.
exec > >(tee -a "$LOG") 2>&1

echo "MAIN_LOG=$LOG"

echo "============================================================"
echo "zkGPT CPU formal run"
echo "start_time=$(date --iso-8601=seconds)"
echo "root=$ROOT"
echo "pid=$$"
echo "log=$LOG"
echo "============================================================"

run_step() {
    name="$1"
    shift

    echo
    echo "============================================================"
    echo "START: $name"
    echo "time=$(date --iso-8601=seconds)"
    echo "============================================================"

    "$@"
    rc=$?

    if [ "$rc" -ne 0 ]; then
        echo
        echo "============================================================"
        echo "FAILED: $name"
        echo "exit_code=$rc"
        echo "time=$(date --iso-8601=seconds)"
        echo "============================================================"
        return "$rc"
    fi

    echo
    echo "============================================================"
    echo "OK: $name"
    echo "time=$(date --iso-8601=seconds)"
    echo "============================================================"

    return 0
}

echo
echo "===== per-cell stale-output cleanup ====="
echo "cleanup is performed by gat_zkgpt_experiment.py with the run-directory allowlist"

echo
echo "===== formal execution ====="

run_step \
    "prepare_resources" \
    "$PYTHON_BIN" scripts/prepare_resources.py --check \
    || exit $?

run_step \
    "build_cpu" \
    "$ROOT/build.sh" --cpu \
    || exit $?

run_step \
    "formal_preflight_cpu" \
    "$PYTHON_BIN" scripts/formal_preflight.py --backend cpu \
    || exit $?

run_step \
    "five_datasets_cpu" \
    "$PYTHON_BIN" scripts/run_five_datasets.py --backend cpu \
    || exit $?

run_step \
    "collect_results_cpu" \
    "$PYTHON_BIN" scripts/collect_results.py --backends cpu \
    || exit $?

run_step \
    "validate_final_results_cpu" \
    "$PYTHON_BIN" scripts/validate_final_results.py \
        --input results/collected_results.json --backends cpu \
    || exit $?

echo
echo "============================================================"
echo "ALL_FIVE_DATASETS_COMPLETED"
echo "finish_time=$(date --iso-8601=seconds)"
echo "============================================================"

exit 0
