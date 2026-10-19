#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(realpath -e -- "${BASH_SOURCE[0]}")"
ROOT="$(realpath -e -- "$(dirname -- "$SCRIPT_PATH")/..")"
cd -- "$ROOT"
command -v nohup >/dev/null
command -v setsid >/dev/null

DATASETS=(cora citeseer pubmed ppi ogbn_arxiv)
SELECTOR_EXPLICIT=0
SELECTED_DATASETS=()
if (($# != 0)); then
    if [[ "$1" != "--datasets" || $# -lt 2 ]]; then
        echo "usage: $SCRIPT_PATH [--datasets cora citeseer pubmed ppi ogbn_arxiv]" >&2
        exit 2
    fi
    SELECTOR_EXPLICIT=1
    shift
    declare -A REQUESTED=()
    for dataset in "$@"; do
        case "$dataset" in
            cora|citeseer|pubmed|ppi|ogbn_arxiv) ;;
            *) echo "unknown zkGPT dataset: $dataset" >&2; exit 2 ;;
        esac
        if [[ "${REQUESTED[$dataset]-}" == 1 ]]; then
            echo "duplicate zkGPT dataset: $dataset" >&2
            exit 2
        fi
        REQUESTED["$dataset"]=1
    done
    for dataset in "${DATASETS[@]}"; do
        if [[ "${REQUESTED[$dataset]-}" == 1 ]]; then
            SELECTED_DATASETS+=("$dataset")
        fi
    done
fi

PYTHON_BIN="$(python3 -c 'import sys; print(sys.executable)')"
LAUNCH_ID="$("$PYTHON_BIN" -c 'import uuid; print(uuid.uuid4().hex)')"
LAUNCH_ROOT="$ROOT/baseline-local/formal-detached"
LAUNCH_DIR="$LAUNCH_ROOT/$LAUNCH_ID"
mkdir -p -- "$LAUNCH_DIR"
chmod 700 -- "$LAUNCH_ROOT" "$LAUNCH_DIR"
CONSOLE_REL="baseline-local/formal-detached/$LAUNCH_ID/console.log"
POINTER_TMP="$LAUNCH_ROOT/.latest_launch.$$.tmp"
printf '{"schema":"zkgpt-gat-detached-launch-v1","launch_id":"%s","console_path":"%s","status_path":"baseline-local/formal-detached/%s/status.json"}\n' \
    "$LAUNCH_ID" "$CONSOLE_REL" "$LAUNCH_ID" > "$POINTER_TMP"
chmod 600 -- "$POINTER_TMP"
mv -f -- "$POINTER_TMP" "$LAUNCH_ROOT/latest_launch.json"

SUPERVISOR_ARGS=(--root "$ROOT" --launch-id "$LAUNCH_ID")
if ((SELECTOR_EXPLICIT)); then
    SUPERVISOR_ARGS+=(--datasets "${SELECTED_DATASETS[@]}")
fi

nohup setsid "$PYTHON_BIN" "$ROOT/scripts/formal_detached_supervisor.py" \
    "${SUPERVISOR_ARGS[@]}" \
    >> "$LAUNCH_DIR/console.log" 2>&1 < /dev/null &

if ((SELECTOR_EXPLICIT)); then
    SCOPE_TEXT="$(IFS=,; printf '%s' "${SELECTED_DATASETS[*]}")"
else
    SCOPE_TEXT="all5"
fi
printf 'ZKGPT_FORMAL_LAUNCHED launch_id=%s datasets=%s console=%s status=%s\n' \
    "$LAUNCH_ID" "$SCOPE_TEXT" "$CONSOLE_REL" \
    "baseline-local/formal-detached/$LAUNCH_ID/status.json"
printf 'Progress: %s/scripts/formal_progress.sh\n' "$ROOT"
