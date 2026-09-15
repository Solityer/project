#!/usr/bin/env bash

set -u
set -o pipefail

ROOT="/home/pzh/project/comparison/zkGPT"
cd "$ROOT" || exit 1

mkdir -p "$ROOT/logs"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$ROOT/logs/zkgpt_cpu_formal_${STAMP}.log"

echo "MAIN_LOG=$LOG"

exec >> "$LOG" 2>&1

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
    python3 scripts/prepare_resources.py --check \
    || exit $?

run_step \
    "build_cpu" \
    ./build.sh --cpu \
    || exit $?

run_step \
    "formal_preflight_cpu" \
    python3 scripts/formal_preflight.py --backend cpu \
    || exit $?

run_step \
    "five_datasets_cpu" \
    python3 scripts/run_five_datasets.py --backend cpu \
    || exit $?

echo
echo "============================================================"
echo "ALL_FIVE_DATASETS_COMPLETED"
echo "finish_time=$(date --iso-8601=seconds)"
echo "============================================================"

exit 0
