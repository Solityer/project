#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
root="$(cd -- "$script_dir/.." && pwd -P)"
if [[ "$#" -ne 0 ]]; then
    echo "usage: $0" >&2
    exit 2
fi
cd -- "$root"
native_backend=cpu
# The official entry owns this legacy helper gate; users do not select a
# backend or set internal orchestration flags.
export ZKDL_ALLOW_FULL_DATASET_RUN=1
command -v setsid >/dev/null 2>&1 || { echo "run-five: setsid is required for owned-stage cleanup" >&2; exit 2; }
mkdir -p runs results
session="$(mktemp -d "$root/runs/formal_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXXXX")"
mkdir -p "$session/logs"
result_dir="$root/results/$(basename -- "$session")"
mkdir -- "$result_dir"
step="initialization"
active_stage_pid=""
active_tee_pid=""
active_fifo=""
active_step_name=""
active_step_started=""
interrupted_signal=""
cleanup_active_stage() {
    if [[ -n "$active_stage_pid" ]]; then
        kill -TERM -- "-$active_stage_pid" 2>/dev/null || true
        sleep 1
        kill -KILL -- "-$active_stage_pid" 2>/dev/null || true
        wait "$active_stage_pid" 2>/dev/null || true
        active_stage_pid=""
    fi
    if [[ -n "$active_tee_pid" ]]; then
        kill -TERM "$active_tee_pid" 2>/dev/null || true
        wait "$active_tee_pid" 2>/dev/null || true
        active_tee_pid=""
    fi
    if [[ -n "$active_fifo" ]]; then
        rm -f -- "$active_fifo"
        active_fifo=""
    fi
}
handle_signal() {
    local signal="$1"
    local exit_code finished elapsed
    trap - INT TERM HUP
    interrupted_signal="$signal"
    case "$signal" in
        INT) exit_code=130 ;;
        TERM) exit_code=143 ;;
        HUP) exit_code=129 ;;
    esac
    if [[ -n "$active_step_started" ]]; then
        finished="$(date +%s)"
        elapsed="$((finished - active_step_started))"
        printf '%s\t%s\t%s\n' "$active_step_name" "$elapsed" "$exit_code" >> "$session/logs/stages.tsv"
        printf 'formal_step=INTERRUPTED step=%s signal=%s exit=%s log=%s\n' \
            "$active_step_name" "$signal" "$exit_code" "$session/logs/$active_step_name.log" >&2
        active_step_name=""
        active_step_started=""
    fi
    cleanup_active_stage
    exit "$exit_code"
}
trap 'handle_signal INT' INT
trap 'handle_signal TERM' TERM
trap 'handle_signal HUP' HUP
on_exit() {
    local status=$?
    cleanup_active_stage
    if [[ -n "$interrupted_signal" ]]; then
        printf "formal_run=INTERRUPTED signal=%s step=%s exit=%s runs=%s logs=%s\n" \
            "$interrupted_signal" "$step" "$status" "$session" "$session/logs" >&2
    elif [[ "$status" -ne 0 ]]; then
        printf "formal_run=FAILED step=%s exit=%s runs=%s logs=%s\n" "$step" "$status" "$session" "$session/logs" >&2
    fi
}
trap on_exit EXIT
run_step() {
    step="$1"
    shift
    local started finished status tee_status log_path
    started="$(date +%s)"
    active_step_name="$step"
    active_step_started="$started"
    log_path="$session/logs/$step.log"
    active_fifo="$session/logs/.${step}.$$.fifo"
    mkfifo -- "$active_fifo"
    printf 'formal_step=START step=%s log=%s\n' "$step" "$log_path"
    tee "$log_path" <"$active_fifo" &
    active_tee_pid=$!
    setsid "$@" >"$active_fifo" 2>&1 &
    active_stage_pid=$!
    if wait "$active_stage_pid"; then
        status=0
    else
        status=$?
    fi
    active_stage_pid=""
    if wait "$active_tee_pid"; then
        tee_status=0
    else
        tee_status=$?
    fi
    active_tee_pid=""
    rm -f -- "$active_fifo"
    active_fifo=""
    if [[ "$status" -eq 0 && "$tee_status" -ne 0 ]]; then
        status="$tee_status"
    fi
    finished="$(date +%s)"
    printf '%s\t%s\t%s\n' "$step" "$((finished - started))" "$status" >> "$session/logs/stages.tsv"
    active_step_name=""
    active_step_started=""
    return "$status"
}
export ZKDL_PREFLIGHT_REPORT="$session/dataset_prerequisites.json"
run_step dataset_prerequisites python3 "$script_dir/formal_preflight.py"
unset ZKDL_PREFLIGHT_REPORT
run_step snapshots python3 "$script_dir/formal_session.py" --session "$session"
run_step build python3 "$script_dir/formal_provenance.py" --output "$session/build_manifest.json"
export ZKDL_BUILD_MANIFEST="$session/build_manifest.json"
export ZKDL_RUNNER_BINARY="$session/build/gat_runner"
session_id="$(basename -- "$session")"
for config in cora citeseer pubmed ppi ogbn_arxiv; do
    run_step "${config}_setup" "$script_dir/setup_register.sh" "$root/configs/${session_id}_${config}.cfg"
done
for config in cora citeseer pubmed ppi ogbn_arxiv; do
    run_step "${config}_prepare" "$script_dir/prepare_local.sh" "$root/configs/${session_id}_${config}.cfg" "$session/${config}"
done
for config in cora citeseer pubmed ppi ogbn_arxiv; do
    run_step "${config}_build_identity" python3 "$script_dir/formal_provenance.py" --check "$ZKDL_BUILD_MANIFEST"
    run_step "${config}_prove" "$ZKDL_RUNNER_BINARY" prove --config "$root/configs/${session_id}_${config}.cfg" --backend "$native_backend" --run-dir "$session/${config}"
    run_step "${config}_verify" "$script_dir/verify_single.sh" "$root/configs/${session_id}_${config}.cfg" "$session/${config}"
done
run_step collect python3 "$script_dir/collect_results.py" --runs "$session" --output "$result_dir/collected.json"
run_step validate python3 "$script_dir/final_validate.py" --runs "$session" --collection "$result_dir/collected.json"
printf 'formal_run=PASS runs=%s results=%s logs=%s\n' "$session" "$result_dir" "$session/logs"
