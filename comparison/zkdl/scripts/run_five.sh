#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
root="$(cd -- "$script_dir/.." && pwd -P)"
if [[ "${1:-cpu}" != "cpu" || "$#" -gt 1 ]]; then
    echo "usage: $0 cpu" >&2
    exit 2
fi
if [[ "${ZKDL_ALLOW_FULL_DATASET_RUN:-0}" != "1" ]]; then
    echo "run-five: full datasets are user-owned; set ZKDL_ALLOW_FULL_DATASET_RUN=1 for the formal run" >&2
    exit 3
fi
cd -- "$root"
mkdir -p runs results
session="$(mktemp -d "$root/runs/formal_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXXXX")"
mkdir -p "$session/logs"
result_dir="$root/results/$(basename -- "$session")"
mkdir -- "$result_dir"
step="initialization"
trap 'status=$?; if [[ "$status" -ne 0 ]]; then printf "formal_run=FAILED step=%s exit=%s runs=%s logs=%s\n" "$step" "$status" "$session" "$session/logs" >&2; fi' EXIT
run_step() {
    step="$1"
    shift
    local started finished status
    started="$(date +%s)"
    printf 'formal_step=START step=%s log=%s\n' "$step" "$session/logs/$step.log"
    if "$@" 2>&1 | tee "$session/logs/$step.log"; then
        status=0
    else
        status=$?
    fi
    finished="$(date +%s)"
    printf '%s\t%s\t%s\n' "$step" "$((finished - started))" "$status" >> "$session/logs/stages.tsv"
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
    run_step "${config}_prepare" "$script_dir/prepare_local.sh" "$root/configs/${session_id}_${config}.cfg" "$session/${config}_cpu"
done
for config in cora citeseer pubmed ppi ogbn_arxiv; do
    run_step "${config}_build_identity" python3 "$script_dir/formal_provenance.py" --check "$ZKDL_BUILD_MANIFEST"
    run_step "${config}_prove" "$ZKDL_RUNNER_BINARY" prove --config "$root/configs/${session_id}_${config}.cfg" --backend cpu --run-dir "$session/${config}_cpu"
    run_step "${config}_verify" "$script_dir/verify_single.sh" "$root/configs/${session_id}_${config}.cfg" "$session/${config}_cpu"
done
run_step collect python3 "$script_dir/collect_results.py" --runs "$session" --output "$result_dir/collected.json" --require-selected-backend cpu
run_step validate python3 "$script_dir/final_validate.py" --runs "$session" --backend cpu
printf 'formal_run=PASS runs=%s results=%s logs=%s\n' "$session" "$result_dir" "$session/logs"
