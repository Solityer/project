#!/usr/bin/env bash
set -uo pipefail
umask 077
root=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P) || exit 1
cd -- "$root" || exit 1
script_path="$root/scripts/run_ablation_matrix.sh"
helper="$root/scripts/ablation_matrix_support.py"
gpu_root="$root/runs/ablation_sessions/gpu_matrix"
datasets=(cora citeseer pubmed ppi ogbn_arxiv)
all_variants=(wo_shared_neighborhood wo_feature_folding)
usage() {
    cat <<'USAGE'
Usage: bash scripts/run_ablation_matrix.sh [--background] [--backend gpu]
       bash scripts/run_ablation_matrix.sh --protocol-variant wo_shared_neighborhood|wo_feature_folding [--background] [--backend gpu]
       bash scripts/run_ablation_matrix.sh --summary|--failures|--follow|--follow-current
No-argument run order is wo_shared_neighborhood then wo_feature_folding, each over
Cora, Citeseer, Pubmed, PPI, OGBN-Arxiv. GPU only; cells run serially.
USAGE
}
protocol_variant="" backend=gpu background=0 worker_session="" worker_variants="" view_mode=""
while (($#)); do
    case "$1" in
        --protocol-variant) (($# >= 2)) || { echo 'missing --protocol-variant value' >&2; exit 2; }; protocol_variant=$2; shift 2 ;;
        --backend) (($# >= 2)) || { echo 'missing --backend value' >&2; exit 2; }; backend=$2; shift 2 ;;
        --background) background=1; shift ;;
        --summary) view_mode=summary; shift ;;
        --failures) view_mode=failures; shift ;;
        --follow) view_mode=follow; shift ;;
        --follow-current) view_mode=follow-current; shift ;;
        --_worker-session) (($# >= 2)) || exit 2; worker_session=$2; shift 2 ;;
        --_worker-variants) (($# >= 2)) || exit 2; worker_variants=$2; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) printf 'unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done
[[ $backend == gpu ]] || { echo 'error: only --backend gpu is supported' >&2; exit 2; }
if [[ -n $view_mode ]]; then
    [[ -z $protocol_variant && $background == 0 && -z $worker_session ]] || { echo 'view mode cannot be combined with run options' >&2; exit 2; }
    exec python3 "$helper" view --project-root "$root" --mode "$view_mode"
fi
variants=()
if [[ -n $worker_session ]]; then
    [[ $worker_session =~ ^[A-Za-z0-9_.-]+$ && -n $worker_variants && $background == 0 && -z $protocol_variant ]] || { echo 'invalid internal worker arguments' >&2; exit 2; }
    IFS=, read -r -a variants <<< "$worker_variants"
else
    case "$protocol_variant" in
        '') variants=("${all_variants[@]}") ;;
        wo_shared_neighborhood|wo_feature_folding) variants=("$protocol_variant") ;;
        *) echo 'error: --protocol-variant must be wo_shared_neighborhood or wo_feature_folding' >&2; exit 2 ;;
    esac
fi
for variant in "${variants[@]}"; do
    [[ $variant == wo_shared_neighborhood || $variant == wo_feature_folding ]] || { echo "invalid variant: $variant" >&2; exit 2; }
done

if [[ -z $worker_session ]]; then
    mkdir -p -- "$gpu_root" || exit 1
    session_id=""
    for attempt in 1 2 3 4 5 6 7 8; do
        candidate="gpu_$(date -u +%Y%m%dT%H%M%S)_$(date +%N)_${BASHPID}_${attempt}"
        if mkdir -- "$gpu_root/$candidate" 2>/dev/null; then session_id=$candidate; break; fi
    done
    [[ -n $session_id ]] || { echo 'cannot reserve unique GPU session' >&2; exit 1; }
    session_root="$gpu_root/$session_id"
    : > "$session_root/global.log" || exit 1
    variants_csv=$(IFS=,; printf '%s' "${variants[*]}")
    if ((background)); then
        command -v setsid >/dev/null 2>&1 || { echo 'setsid is required for --background' >&2; exit 1; }
        nohup setsid bash "$script_path" --_worker-session "$session_id" --_worker-variants "$variants_csv" </dev/null >/dev/null 2>&1 &
        launcher_pid=$!
        worker_json="$session_root/worker.json"
        ready=0
        for _ in $(seq 1 300); do
            if [[ -s $worker_json ]]; then ready=1; break; fi
            sleep 0.05
        done
        if ((ready == 0)); then echo "worker PID handshake failed; see $session_root/global.log" >&2; exit 1; fi
        worker_pid=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["pid"])' "$worker_json") || exit 1
        [[ $worker_pid =~ ^[1-9][0-9]*$ ]] && kill -0 "$worker_pid" 2>/dev/null || { echo 'handshaken worker is not alive' >&2; exit 1; }
        detached=$(python3 -c 'import json,os,sys; x=json.load(open(sys.argv[1])); p=int(x["pid"]); g=int(x["process_group_id"]); s=int(x["session_id"]); print("yes" if os.getpgid(p)==g==s and os.getsid(p)==s and os.getsid(0)!=s else "no")' "$worker_json")
        [[ $detached == yes ]] || { kill "$worker_pid" 2>/dev/null || true; echo 'worker is not detached in its own session' >&2; exit 1; }
        latest_tmp="$gpu_root/.latest.tmp.${BASHPID}.${RANDOM}"
        printf '%s\n' "$session_id" > "$latest_tmp" && mv -f -- "$latest_tmp" "$gpu_root/latest" || { rm -f "$latest_tmp"; exit 1; }
        : > "$session_root/.launch-release" || { kill "$worker_pid" 2>/dev/null || true; exit 1; }
        printf 'GPU_ABLATION_SESSION_STARTED session_id=%s worker_pid=%s variants=%s backend=gpu\n' "$session_id" "$worker_pid" "$variants_csv"
        printf 'GLOBAL_LOG=%s\nFOLLOW_GLOBAL=bash %s --follow\nFOLLOW_CURRENT=bash %s --follow-current\nSUMMARY=bash %s --summary\nFAILURES=bash %s --failures\n' \
            "$session_root/global.log" "$script_path" "$script_path" "$script_path" "$script_path"
        exit 0
    fi
else
    session_id=$worker_session
    session_root="$gpu_root/$session_id"
fi
session_rel="runs/ablation_sessions/gpu_matrix/$session_id"
global_log="$session_root/global.log"
build_log="$session_root/logs/build_cuda.log"
matrix_initialized=0 worker_finalized=0
current_variant="" current_dataset="" cell_log="" cell_log_rel=""
cell_log_ready=0 cell_logger_rc=0 global_logger_rc=0
child_rc=0 cell_tee_rc=0 global_tee_rc=0 build_rc=0 build_tee_rc=0
capture_setup_rc=0 cell_log_setup_rc=0 cell_log_fallback_rc=0 cell_matrix_tracking=0 cell_matrix_rc=0 cell_matrix_logger_rc=0 cell_matrix_error=""
capture_output_missing=0
session_matrix_rc=0 session_matrix_logger_rc=0 session_matrix_error=""
session_global_logger_rc=0
session_failure_reason="" session_failure_rc=null
session_failure_args=()
last_matrix_helper_rc=0 last_matrix_logger_rc=0 last_matrix_detail="" last_matrix_output=""
cell_matrix_detail="" session_matrix_detail=""

now_utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }
refresh_session_failure() {
    local prior_reason=$session_failure_reason prior_rc=$session_failure_rc
    session_failure_reason="" session_failure_rc=null
    if ((session_matrix_rc != 0 || session_matrix_logger_rc != 0)); then
        session_failure_reason="session matrix update failed helper_rc=$session_matrix_rc logger_rc=$session_matrix_logger_rc"
        if [[ -n $session_matrix_detail ]]; then session_failure_reason="$session_failure_reason; detail=$session_matrix_detail"; fi
        if ((session_matrix_rc != 0)); then session_failure_rc=$session_matrix_rc; else session_failure_rc=$session_matrix_logger_rc; fi
    fi
    if ((session_global_logger_rc != 0)); then
        if [[ -n $session_failure_reason ]]; then session_failure_reason="$session_failure_reason; global logger failed rc=$session_global_logger_rc"
        else session_failure_reason="global logger failed rc=$session_global_logger_rc"; fi
        if [[ $session_failure_rc == null ]]; then session_failure_rc=$session_global_logger_rc; fi
    fi
    if [[ -n $prior_reason ]]; then
        if [[ -n $session_failure_reason && $session_failure_reason != *"$prior_reason"* ]]; then
            session_failure_reason="$prior_reason; $session_failure_reason"
        elif [[ -z $session_failure_reason ]]; then
            session_failure_reason=$prior_reason
        fi
        if [[ $session_failure_rc == null && $prior_rc != null ]]; then session_failure_rc=$prior_rc; fi
    fi
    session_failure_args=()
    if [[ -n $session_failure_reason ]]; then
        session_failure_args=(--session-failure-reason "$session_failure_reason" --session-failure-rc "$session_failure_rc")
    fi
}
remember_error() {
    local name=$1 value=$2 old=${!1}
    if [[ $value =~ ^[0-9]+$ ]] && ((value != 0)) && [[ $old == 0 || $old == null ]]; then printf -v "$name" '%s' "$value"; fi
}
global_line() {
    local -a ps
    printf '%s\n' "$1" | tee -a "$global_log"
    ps=("${PIPESTATUS[@]}")
    remember_error global_logger_rc "${ps[1]:-1}"
    remember_error session_global_logger_rc "${ps[1]:-1}"
    return "${ps[1]:-1}"
}
cell_line() {
    local -a ps
    if ((cell_log_ready)); then
        printf '%s\n' "$1" | tee -a "$cell_log" | tee -a "$global_log"
        ps=("${PIPESTATUS[@]}")
        remember_error cell_logger_rc "${ps[1]:-1}"
        remember_error global_logger_rc "${ps[2]:-1}"
        remember_error session_global_logger_rc "${ps[2]:-1}"
        return "$(( ${ps[1]:-1} != 0 ? ${ps[1]:-1} : ${ps[2]:-1} ))"
    fi
    printf '%s\n' "$1" | tee -a "$global_log"
    ps=("${PIPESTATUS[@]}")
    remember_error global_logger_rc "${ps[1]:-1}"
    remember_error session_global_logger_rc "${ps[1]:-1}"
    return 1
}
cmd_line() {
    local rendered='$' arg dest=$1
    shift
    for arg in "$@"; do printf -v rendered '%s %q' "$rendered" "$arg"; done
    if [[ $dest == cell ]]; then cell_line "$rendered"; else build_line "$rendered"; fi
}
build_line() {
    local -a ps
    printf '%s\n' "$1" | tee -a "$build_log" | tee -a "$global_log"
    ps=("${PIPESTATUS[@]}")
    remember_error build_tee_rc "${ps[1]:-1}"
    remember_error global_logger_rc "${ps[2]:-1}"
    remember_error session_global_logger_rc "${ps[2]:-1}"
    return "$(( ${ps[1]:-1} != 0 ? ${ps[1]:-1} : ${ps[2]:-1} ))"
}
run_cell() {
    local -a ps
    cmd_line cell "$@" || true
    child_rc=0 cell_tee_rc=0 global_tee_rc=0
    if ((cell_log_ready)); then
        "$@" 2>&1 | tee -a "$cell_log" | tee -a "$global_log"
        ps=("${PIPESTATUS[@]}")
        child_rc=${ps[0]:-1} cell_tee_rc=${ps[1]:-1} global_tee_rc=${ps[2]:-1}
        remember_error cell_logger_rc "$cell_tee_rc"
        remember_error global_logger_rc "$global_tee_rc"
        remember_error session_global_logger_rc "$global_tee_rc"
    else
        "$@" 2>&1 | tee -a "$global_log"
        ps=("${PIPESTATUS[@]}")
        child_rc=${ps[0]:-1} cell_tee_rc=0 global_tee_rc=${ps[1]:-1}
        remember_error global_logger_rc "$global_tee_rc"
        remember_error session_global_logger_rc "$global_tee_rc"
    fi
    ((child_rc == 0 && cell_tee_rc == 0 && global_tee_rc == 0 && cell_log_ready))
}
run_build() {
    local -a ps
    cmd_line build "$@" || true
    "$@" 2>&1 | tee -a "$build_log" | tee -a "$global_log"
    ps=("${PIPESTATUS[@]}")
    build_rc=${ps[0]:-1}
    remember_error build_tee_rc "${ps[1]:-1}"
    remember_error global_logger_rc "${ps[2]:-1}"
    remember_error session_global_logger_rc "${ps[2]:-1}"
    ((build_rc == 0 && ${ps[1]:-1} == 0 && ${ps[2]:-1} == 0))
}
run_global() {
    local -a ps
    "$@" 2>&1 | tee -a "$global_log"
    ps=("${PIPESTATUS[@]}")
    global_command_rc=${ps[0]:-1}
    remember_error global_logger_rc "${ps[1]:-1}"
    remember_error session_global_logger_rc "${ps[1]:-1}"
    ((global_command_rc == 0 && ${ps[1]:-1} == 0))
}
run_capture_cell() {
    local stdout_path=$1 stderr_path=$2; shift 2
    local -a ps
    local shell_error_path="$session_root/.capture_shell_${current_variant}_${current_dataset}.stderr"
    local capture_rc=0
    capture_output_missing=0
    cmd_line cell "$@" || true
    : > "$stdout_path" 2>> "$global_log"
    capture_setup_rc=$?
    if ((capture_setup_rc == 0)); then
        : > "$stderr_path" 2>> "$global_log"
        capture_setup_rc=$?
    fi
    if ((capture_setup_rc == 0)); then
        : > "$shell_error_path" 2>> "$global_log"
        capture_setup_rc=$?
    fi
    if ((capture_setup_rc != 0)); then
        child_rc=null
        cell_tee_rc=0 global_tee_rc=0
        cell_line "CAPTURE_SETUP_FAILED variant=$current_variant dataset=$current_dataset rc=$capture_setup_rc" || true
        return 1
    fi
    { "$@" > "$stdout_path" 2> "$stderr_path"; } 2> "$shell_error_path"
    capture_rc=$?
    child_rc=$capture_rc
    if [[ -s $shell_error_path ]]; then
        child_rc=null
        capture_setup_rc=$capture_rc
    fi
    cell_tee_rc=0 global_tee_rc=0
    if [[ -f $stdout_path ]]; then
        cat -- "$stdout_path" | tee -a "$cell_log" | tee -a "$global_log"
        ps=("${PIPESTATUS[@]}")
        cell_tee_rc=${ps[1]:-1} global_tee_rc=${ps[2]:-1}
        remember_error cell_logger_rc "$cell_tee_rc"; remember_error global_logger_rc "$global_tee_rc"
        remember_error session_global_logger_rc "$global_tee_rc"
    else
        capture_output_missing=1
    fi
    if [[ -s $shell_error_path ]]; then
        cell_line '[capture setup stderr]' || true
        cat -- "$shell_error_path" | tee -a "$cell_log" | tee -a "$global_log"
        ps=("${PIPESTATUS[@]}")
        if ((${ps[1]:-1} != 0)); then cell_tee_rc=${ps[1]}; fi
        if ((${ps[2]:-1} != 0)); then global_tee_rc=${ps[2]}; fi
        remember_error cell_logger_rc "$cell_tee_rc"; remember_error global_logger_rc "$global_tee_rc"
        remember_error session_global_logger_rc "$global_tee_rc"
    fi
    if [[ -s $stderr_path ]]; then
        cell_line '[stderr]' || true
        cat -- "$stderr_path" | tee -a "$cell_log" | tee -a "$global_log"
        ps=("${PIPESTATUS[@]}")
        if ((${ps[1]:-1} != 0)); then cell_tee_rc=${ps[1]}; fi
        if ((${ps[2]:-1} != 0)); then global_tee_rc=${ps[2]}; fi
        remember_error cell_logger_rc "$cell_tee_rc"; remember_error global_logger_rc "$global_tee_rc"
        remember_error session_global_logger_rc "$global_tee_rc"
    fi
    rm -f -- "$shell_error_path" 2>> "$global_log" || true
    if ((capture_output_missing)); then cell_line "CAPTURE_STDOUT_MISSING variant=$current_variant dataset=$current_dataset child_rc=$child_rc" || true; fi
    [[ $child_rc == 0 ]] && ((cell_tee_rc == 0 && global_tee_rc == 0 && capture_output_missing == 0))
}
matrix_update() {
    local -a ps
    local has_variant=0 has_dataset=0 arg output_logger_rc=0
    for arg in "$@"; do
        [[ $arg == --variant ]] && has_variant=1
        [[ $arg == --dataset ]] && has_dataset=1
    done
    last_matrix_output=$(python3 "$helper" matrix-update --session-root "$session_root" "$@" 2>&1)
    last_matrix_helper_rc=$?
    last_matrix_detail=$(printf '%s\n' "$last_matrix_output" | awk 'NF { line=$0 } END { print line }' | tr '\t\r\n' '   ' | cut -c1-600)
    last_matrix_logger_rc=0
    if [[ -n $last_matrix_output ]]; then
        if ((cell_matrix_tracking && has_variant && has_dataset && cell_log_ready)); then
            printf '%s\n' "$last_matrix_output" | tee -a "$cell_log" | tee -a "$global_log"
            ps=("${PIPESTATUS[@]}")
            remember_error cell_logger_rc "${ps[1]:-1}"
            remember_error global_logger_rc "${ps[2]:-1}"
            remember_error session_global_logger_rc "${ps[2]:-1}"
            if ((${ps[1]:-1} != 0)); then output_logger_rc=${ps[1]}
            else output_logger_rc=${ps[2]:-1}; fi
        else
            printf '%s\n' "$last_matrix_output" | tee -a "$global_log"
            ps=("${PIPESTATUS[@]}")
            remember_error global_logger_rc "${ps[1]:-1}"
            remember_error session_global_logger_rc "${ps[1]:-1}"
            output_logger_rc=${ps[1]:-1}
        fi
        last_matrix_logger_rc=$output_logger_rc
    fi
    if ((last_matrix_helper_rc != 0 || last_matrix_logger_rc != 0)); then
        if ((cell_matrix_tracking && has_variant && has_dataset)); then
            remember_error cell_matrix_rc "$last_matrix_helper_rc"
            remember_error cell_matrix_logger_rc "$last_matrix_logger_rc"
            if [[ -z $cell_matrix_error ]]; then
                cell_matrix_error="helper_rc=$last_matrix_helper_rc logger_rc=$last_matrix_logger_rc"
                cell_matrix_detail=$last_matrix_detail
            fi
        else
            remember_error session_matrix_rc "$last_matrix_helper_rc"
            remember_error session_matrix_logger_rc "$last_matrix_logger_rc"
            if [[ -z $session_matrix_error ]]; then
                session_matrix_error="helper_rc=$last_matrix_helper_rc logger_rc=$last_matrix_logger_rc"
                session_matrix_detail=$last_matrix_detail
            fi
        fi
    fi
    if ((last_matrix_helper_rc != 0)); then
        if ((cell_matrix_tracking && has_variant && has_dataset && cell_log_ready)); then
            cell_line "MATRIX_UPDATE_ERROR helper_rc=$last_matrix_helper_rc logger_rc=$last_matrix_logger_rc detail=$last_matrix_detail" || true
        else
            global_line "MATRIX_UPDATE_ERROR helper_rc=$last_matrix_helper_rc logger_rc=$last_matrix_logger_rc detail=$last_matrix_detail" || true
        fi
        return "$last_matrix_helper_rc"
    fi
    return "$last_matrix_logger_rc"
}
init_matrix() {
    local variants_csv matrix_worker_pid
    variants_csv=$(IFS=,; printf '%s' "${variants[*]}")
    matrix_worker_pid=${BASHPID:-$$}
    local -a ps
    python3 "$helper" matrix-init --session-root "$session_root" --variants "$variants_csv" --worker-pid "$matrix_worker_pid" \
        2>&1 | tee -a "$global_log"
    ps=("${PIPESTATUS[@]}")
    remember_error global_logger_rc "${ps[1]:-1}"
    if ((${ps[0]:-1} != 0 || ${ps[1]:-1} != 0)); then return 1; fi
    matrix_initialized=1
}
publish_latest() {
    local tmp="$gpu_root/.latest.tmp.${BASHPID}.${RANDOM}"
    printf '%s\n' "$session_id" > "$tmp" && mv -f -- "$tmp" "$gpu_root/latest"
}
begin_cell() {
    current_variant=$1 current_dataset=$2
    cell_log_rel="logs/${current_variant}_${current_dataset}.log" cell_log="$session_root/$cell_log_rel"
    cell_logger_rc=0 global_logger_rc=0 cell_matrix_rc=0 cell_matrix_logger_rc=0 cell_matrix_error=""
    cell_matrix_tracking=1 cell_log_ready=0 cell_log_fallback_rc=0
    local setup_error="$session_root/.cell_log_setup_${current_variant}_${current_dataset}.stderr"
    : > "$setup_error" 2>> "$global_log"
    local scratch_rc=$?
    if ((scratch_rc == 0)); then
        { mkdir -p -- "$session_root/logs" && : >> "$cell_log"; } 2> "$setup_error"
    else
        { mkdir -p -- "$session_root/logs" && : >> "$cell_log"; } 2>> "$global_log"
    fi
    local log_setup_rc=$?
    cell_log_setup_rc=$log_setup_rc
    if ((log_setup_rc == 0)); then
        cell_log_ready=1
    else
        cell_log_rel="${current_variant}_${current_dataset}.failure.log"
        cell_log="$session_root/$cell_log_rel"
        : >> "$cell_log" 2>> "$global_log"
        cell_log_fallback_rc=$?
        if ((cell_log_fallback_rc == 0)); then cell_log_ready=1; fi
        if ((cell_log_ready)); then
            cell_line "CELL_LOG_CREATE_FAILED variant=$current_variant dataset=$current_dataset primary_rc=$log_setup_rc fallback_rc=$cell_log_fallback_rc log=$cell_log" || true
        else
            global_line "CELL_LOG_CREATE_FAILED variant=$current_variant dataset=$current_dataset primary_rc=$log_setup_rc fallback_rc=$cell_log_fallback_rc log=$cell_log" || true
        fi
        if ((scratch_rc == 0)) && [[ -s $setup_error ]]; then
            if ((cell_log_ready)); then cell_line '[primary cell log setup stderr]' || true; fi
            if ((cell_log_ready)); then
                cat -- "$setup_error" | tee -a "$cell_log" | tee -a "$global_log"
                local -a setup_ps=("${PIPESTATUS[@]}")
                remember_error cell_logger_rc "${setup_ps[1]:-1}"
                remember_error global_logger_rc "${setup_ps[2]:-1}"
                remember_error session_global_logger_rc "${setup_ps[2]:-1}"
            else
                cat -- "$setup_error" | tee -a "$global_log"
                local -a setup_ps=("${PIPESTATUS[@]}")
                remember_error global_logger_rc "${setup_ps[1]:-1}"
                remember_error session_global_logger_rc "${setup_ps[1]:-1}"
            fi
        elif ((scratch_rc != 0)); then
            global_line "CELL_LOG_SETUP_DIAGNOSTIC_CAPTURE_FAILED variant=$current_variant dataset=$current_dataset rc=$scratch_rc" || true
        fi
    fi
    rm -f -- "$setup_error" 2>> "$global_log" || true
    matrix_update --variant "$current_variant" --dataset "$current_dataset" --cell-status running --cell-phase startup_inventory \
        --cell-log "$cell_log_rel" --current-active true --phase startup_inventory --log "$session_rel/$cell_log_rel" || true
    cell_line "===== CELL_START variant=$current_variant dataset=$current_dataset phase=startup_inventory started_at=$(now_utc) =====" || true
}
record_stage() {
    local stage=$1 crc=$2 lrc=$3 grc=$4 reason=$5 status=passed forced_status=${6:-}
    if [[ $crc != 0 || $lrc != 0 || $grc != 0 ]]; then status=failed; fi
    [[ -n $forced_status ]] && status=$forced_status
    matrix_update --variant "$current_variant" --dataset "$current_dataset" --cell-phase "$stage" --stage "$stage" --stage-status "$status" \
        --stage-command-rc "$crc" --cell-logger-rc "$lrc" --global-logger-rc "$grc" --stage-reason "$reason" || true
}
failure_detail() {
    local log_path=${1:-$cell_log} detail=""
    if [[ -f $log_path && ! -L $log_path ]]; then
        detail=$(awk '
            index($0, "ablation-entrypoint: ") { exact = substr($0, index($0, "ablation-entrypoint: ") + 21); next }
            tolower($0) ~ /error|fail|exception|reject|mismatch|invalid|missing|cuda|srs|oom|proof/ { candidate = $0 }
            END { if (exact != "") print exact; else if (candidate != "") print candidate }
        ' "$log_path" | tail -n 1 | tr '\t\r\n' '   ' | cut -c1-600)
    fi
    printf '%s' "$detail"
}
announce_phase() {
    local stage=$1
    cell_line "CELL_PHASE variant=$current_variant dataset=$current_dataset phase=$stage started_at=$(now_utc)" || true
    matrix_update --variant "$current_variant" --dataset "$current_dataset" --cell-phase "$stage" --current-active true \
        --phase "$stage" --log "$session_rel/$cell_log_rel" || true
}
fail_cell() {
    local phase=$1 reason=$2 crc=${3:-null} vrc=${4:-null}
    local detail failure_write_reason
    detail=$(failure_detail)
    if [[ -n $detail && $reason != *"$detail"* ]]; then reason="$reason; detail=$detail"; fi
    if ((cell_log_setup_rc != 0)); then reason="$reason; per-cell log setup failed primary_rc=$cell_log_setup_rc fallback_rc=$cell_log_fallback_rc"; fi
    if [[ -n $cell_matrix_error ]]; then
        reason="$reason; per-cell matrix update failed helper_rc=${cell_matrix_rc:-0} logger_rc=${cell_matrix_logger_rc:-0}"
        if [[ -n $cell_matrix_detail ]]; then reason="$reason; detail=$cell_matrix_detail"; fi
    fi
    cell_line "CELL_FAILED variant=$current_variant dataset=$current_dataset phase=$phase finished_at=$(now_utc) command_rc=$crc logger_rc=$cell_logger_rc validation_rc=$vrc reason=$reason" || true
    if ! matrix_update --variant "$current_variant" --dataset "$current_dataset" --cell-status failed --cell-phase "$phase" --reason "$reason" \
        --command-rc "$crc" --logger-rc "$cell_logger_rc" --validation-rc "$vrc" --current-active false --phase "$phase"; then
        failure_write_reason="failure status update failed helper_rc=$last_matrix_helper_rc logger_rc=$last_matrix_logger_rc"
        cell_line "CELL_STATUS_WRITE_FAILED variant=$current_variant dataset=$current_dataset phase=$phase $failure_write_reason" || true
        matrix_update --variant "$current_variant" --dataset "$current_dataset" --cell-status failed --cell-phase "$phase" \
            --reason "$reason; $failure_write_reason" --command-rc "$crc" --logger-rc "$cell_logger_rc" \
            --validation-rc "$vrc" --current-active false --phase "$phase" || true
    fi
}

process_cell() {
    local variant=$1 dataset=$2 base="$root/configs/${2}_secure.cfg"
    local config_rel="configs/$variant/$dataset.cfg" registry_rel="$session_rel/registries/$variant/$dataset.registry"
    local raw_rel="preflight/$variant/$dataset.registry.stdout" err_rel="preflight/$variant/$dataset.registry.stderr" insp_rel="preflight/$variant/$dataset.registry.json"
    local config="$session_root/$config_rel" registry="$session_root/registries/$variant/$dataset.registry" raw="$session_root/$raw_rel" err="$session_root/$err_rel" inspection="$session_root/$insp_rel"
    local export_rel="$session_rel/results/$variant/$dataset" tag="${session_id}_${variant}_${dataset}"
    local run_dir="$session_root/results/$variant/$dataset/${session_id}_${variant}_${dataset}/single"
    local proof="$run_dir/proof.bin" metrics="$session_root/evidence/${variant}_${dataset}.metrics.json" evidence="$session_root/evidence/${variant}_${dataset}.json"
    local rc=0 reason="" phase="" digest="" native_rc=0 validation_rc=0 proof_logger_cell=0 proof_logger_global=0
    begin_cell "$variant" "$dataset"
    if ! run_cell python3 "$helper" inventory-check --input "$session_root/startup_inventory.json" --dataset "$dataset"; then
        if ((cell_log_setup_rc != 0)); then
            reason="per-cell log creation failed (primary_rc=$cell_log_setup_rc fallback_rc=$cell_log_fallback_rc); inventory child rc=$child_rc log=$cell_log"
        else
            reason="startup inventory failed for $dataset (rc=$child_rc)"
        fi
        phase=startup_inventory
        if ((cell_log_setup_rc != 0)); then record_stage "$phase" "$child_rc" "$cell_logger_rc" "$global_tee_rc" "$reason" failed
        else record_stage "$phase" "$child_rc" "$cell_tee_rc" "$global_tee_rc" "$reason"; fi
        fail_cell "$phase" "$reason" null null; return 1
    fi
    if ((cell_log_setup_rc != 0)); then
        reason="per-cell log creation failed (primary_rc=$cell_log_setup_rc fallback_rc=$cell_log_fallback_rc); log=$cell_log"
        record_stage startup_inventory "$child_rc" "$cell_logger_rc" "$global_logger_rc" "$reason" failed
        fail_cell startup_inventory "$reason" null null; return 1
    fi
    record_stage startup_inventory 0 "$cell_tee_rc" "$global_tee_rc" 'cached data, checkpoint and SRS manifest metadata passed'
    if [[ $cuda_status != visible ]]; then
        reason="CUDA probe status is $cuda_status; this GPU cell was not launched"; phase=startup_cuda
        record_stage "$phase" null 0 0 "$reason" failed; fail_cell "$phase" "$reason" null null; return 1
    fi
    phase=write_config
    announce_phase "$phase"
    if ! run_cell python3 "$helper" write-config --base "$base" --output "$config" --variant "$variant" --export-dir "$export_rel" --registry-path "$registry_rel"; then
        reason="config generation failed (rc=$child_rc)"; record_stage "$phase" "$child_rc" "$cell_tee_rc" "$global_tee_rc" "$reason"; fail_cell "$phase" "$reason" null null; return 1
    fi
    record_stage "$phase" 0 "$cell_tee_rc" "$global_tee_rc" 'formal base config preserved except variant/session paths'
    phase=register
    announce_phase "$phase"
    if ! run_cell "$registry_register" --config "$config" --output "$registry_rel"; then
        reason="verifier registry registration failed (rc=$child_rc)"; record_stage "$phase" "$child_rc" "$cell_tee_rc" "$global_tee_rc" "$reason"; fail_cell "$phase" "$reason" null null; return 1
    fi
    digest=$(python3 "$helper" extract-registry-digest --log "$cell_log" 2>&1)
    rc=$?
    if ((rc != 0)) || [[ ! $digest =~ ^[0-9a-f]{64}$ ]]; then
        local digest_detail
        digest_detail=$(printf '%s' "$digest" | tr '\t\r\n' '   ' | cut -c1-600)
        reason="registration log has no canonical registry digest (extract_rc=$rc)"
        if [[ -n $digest_detail ]]; then reason="$reason; detail=$digest_detail"; fi
        cell_line "REGISTRY_DIGEST_EXTRACTION_FAILED rc=$rc" || true
        if [[ -n $digest ]]; then
            printf '%s\n' "$digest" | tee -a "$cell_log" | tee -a "$global_log"
            local -a digest_log_ps=("${PIPESTATUS[@]}")
            remember_error cell_logger_rc "${digest_log_ps[1]:-1}"
            remember_error global_logger_rc "${digest_log_ps[2]:-1}"
            remember_error session_global_logger_rc "${digest_log_ps[2]:-1}"
        fi
        record_stage "$phase" "$rc" "$cell_logger_rc" "$global_logger_rc" "$reason" failed
        fail_cell "$phase" "$reason" null null; return 1
    fi
    matrix_update --variant "$variant" --dataset "$dataset" --registry-digest "$digest" || true
    record_stage "$phase" 0 "$cell_tee_rc" "$global_tee_rc" "registered candidate digest=$digest"
    phase=pin_config
    announce_phase "$phase"
    if ! run_cell python3 "$helper" write-config --base "$base" --output "$config" --variant "$variant" --export-dir "$export_rel" --registry-path "$registry_rel" --registry-digest "$digest"; then
        reason="config digest pinning failed (rc=$child_rc)"; record_stage "$phase" "$child_rc" "$cell_tee_rc" "$global_tee_rc" "$reason"; fail_cell "$phase" "$reason" null null; return 1
    fi
    record_stage "$phase" 0 "$cell_tee_rc" "$global_tee_rc" 'pinned generated config to its independent registry'
    phase=inspect
    announce_phase "$phase"
    if ! run_cell mkdir -p -- "$(dirname -- "$inspection")"; then
        reason="inspection output directory creation failed (rc=$child_rc)"
        record_stage "$phase" "$child_rc" "$cell_tee_rc" "$global_tee_rc" "$reason"
        fail_cell "$phase" "$reason" null null; return 1
    fi
    if ! run_capture_cell "$raw" "$err" "$registry_inspect" --registry "$registry_rel" --trusted-digest "$digest"; then
        if [[ $child_rc == null ]]; then reason="independent registry inspector could not start (capture_setup_rc=$capture_setup_rc)"
        elif ((capture_output_missing)); then reason="independent registry inspector exited rc=$child_rc without a captured stdout file"
        else reason="independent registry inspector failed (rc=$child_rc)"; fi
        if ((capture_output_missing)); then record_stage "$phase" "$child_rc" "$cell_tee_rc" "$global_tee_rc" "$reason" failed
        else record_stage "$phase" "$child_rc" "$cell_tee_rc" "$global_tee_rc" "$reason"; fi
        fail_cell "$phase" "$reason" null null; return 1
    fi
    if ! run_cell python3 "$helper" extract-inspection --input "$raw" --output "$inspection"; then
        reason="independent inspection output failed schema parsing (rc=$child_rc)"; record_stage "$phase" "$child_rc" "$cell_tee_rc" "$global_tee_rc" "$reason"; fail_cell "$phase" "$reason" null null; return 1
    fi
    record_stage "$phase" 0 "$cell_tee_rc" "$global_tee_rc" 'independent registry inspection completed'
    phase=registry_binding
    announce_phase "$phase"
    if ! run_cell python3 "$helper" check-registry --project-root "$root" --config "$config" --variant "$variant" --registry "$registry" --registry-digest "$digest" --inspection "$inspection"; then
        reason="variant, schema, SRS, or registry metadata binding failed (rc=$child_rc)"; record_stage "$phase" "$child_rc" "$cell_tee_rc" "$global_tee_rc" "$reason"; fail_cell "$phase" "$reason" null null; return 1
    fi
    record_stage "$phase" 0 "$cell_tee_rc" "$global_tee_rc" 'variant identity and trusted resource binding verified'
    phase=proof
    announce_phase "$phase"
    if ((cell_matrix_rc != 0 || cell_matrix_logger_rc != 0)); then
        reason="per-cell matrix update failed helper_rc=$cell_matrix_rc logger_rc=$cell_matrix_logger_rc; proof was not started"
        if [[ -n $cell_matrix_detail ]]; then reason="$reason; detail=$cell_matrix_detail"; fi
        fail_cell matrix_update "$reason" null null; return 1
    fi
    run_cell "$run_binary" --config "$config" --protocol-variant "$variant" --compute-backend cuda_hotspots --benchmark-mode single --export-tag "$tag"
    native_rc=$child_rc proof_logger_cell=$cell_tee_rc proof_logger_global=$global_tee_rc
    reason=''; ((native_rc == 0)) || reason="formal prover exited with code $native_rc"
    record_stage "$phase" "$native_rc" "$proof_logger_cell" "$proof_logger_global" "$reason"
    phase=independent_validation
    announce_phase "$phase"
    run_cell python3 "$helper" validate-cell --project-root "$root" --run-dir "$run_dir" --config "$config" --variant "$variant" --dataset "$dataset" --backend gpu --registry-digest "$digest" --registry "$registry" --inspection "$inspection" --binary "$run_binary" --result-log "$cell_log" --receipt-path "$evidence" --metrics-output "$metrics" || true
    validation_rc=$child_rc
    if ((validation_rc != 0)); then
        reason="proof serialization, independent verification, or four-metric validation failed (rc=$validation_rc)"
    elif ((cell_tee_rc != 0 || global_tee_rc != 0)); then
        reason="independent validation exited 0 but its output logger failed (cell_logger_rc=$cell_tee_rc global_logger_rc=$global_tee_rc)"
    fi
    local validation_cell_tee=$cell_tee_rc validation_global_tee=$global_tee_rc
    record_stage "$phase" "$validation_rc" "$validation_cell_tee" "$validation_global_tee" "$reason"
    if ((native_rc != 0 || validation_rc != 0 || proof_logger_cell != 0 || proof_logger_global != 0 || validation_cell_tee != 0 || validation_global_tee != 0 || cell_logger_rc != 0 || global_logger_rc != 0 || cell_matrix_rc != 0 || cell_matrix_logger_rc != 0)); then
        [[ -n $reason ]] || reason="formal result has a failed log pipeline (cell logger=$cell_logger_rc global logger=$global_logger_rc)"
        if ((cell_matrix_rc != 0 || cell_matrix_logger_rc != 0)); then reason="$reason; per-cell matrix update failed helper_rc=$cell_matrix_rc logger_rc=$cell_matrix_logger_rc"; fi
        fail_cell proof "$reason" "$native_rc" "$validation_rc"; return 1
    fi
    cell_line "CELL_SUCCESS variant=$variant dataset=$dataset phase=complete finished_at=$(now_utc) VERIFY_OK" || true
    if ((cell_logger_rc != 0 || global_logger_rc != 0)); then
        reason="final success log write failed (cell_logger_rc=$cell_logger_rc global_logger_rc=$global_logger_rc)"
        fail_cell proof "$reason" "$native_rc" "$validation_rc"; return 1
    fi
    if ! matrix_update --variant "$variant" --dataset "$dataset" --cell-status success --cell-phase complete --reason 'serialized proof independently verified; all four formal metrics validated' --command-rc 0 --logger-rc 0 --validation-rc 0 --registry-digest "$digest" --metrics-json "$metrics" --evidence-path "$evidence" --proof-path "$proof" --current-active false --phase complete; then
        reason="matrix success transition failed helper_rc=$last_matrix_helper_rc logger_rc=$last_matrix_logger_rc; success was not retained"
        fail_cell matrix_update "$reason" "$native_rc" "$validation_rc"; return 1
    fi
    return 0
}

on_exit() {
    local rc=$?
    trap - EXIT
    if ((matrix_initialized && !worker_finalized)); then
        global_line "WORKER_ABORTED exit_code=$rc at=$(now_utc); unresolved cells are marked failed" || true
        local variant dataset state
        for variant in "${variants[@]}"; do for dataset in "${datasets[@]}"; do
            state=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["cells"].get(sys.argv[2],{}).get("status","pending"))' "$session_root/matrix_summary.json" "$variant/$dataset" 2>/dev/null)
            if [[ $state == pending || $state == running ]]; then
                begin_cell "$variant" "$dataset"
                fail_cell worker_abort "worker exited unexpectedly with code $rc" null null
            fi
        done; done
        matrix_update --worker-status finished_with_failures --phase matrix_complete --current-active false \
            --session-failure-reason "worker aborted unexpectedly with exit_code=$rc" --session-failure-rc "$rc" || true
    fi
    exit "$rc"
}

if [[ -z $worker_session ]]; then
    if ! init_matrix; then echo 'matrix initialization failed' >&2; exit 1; fi
    publish_latest || { echo 'cannot publish latest GPU session' >&2; exit 1; }
else
    [[ -d $session_root && -f $global_log ]] || { echo 'internal worker reservation missing' >&2; exit 1; }
    init_matrix || { echo 'matrix initialization failed' >&2; exit 1; }
    released=0
    for _ in $(seq 1 300); do [[ -f $session_root/.launch-release ]] && { released=1; break; }; sleep 0.05; done
    if ((released == 0)); then global_line "detached launch handshake timeout at=$(now_utc)" || true; matrix_update --worker-status finished_with_failures --phase launch_handshake_failed || true; exit 1; fi
fi
trap on_exit EXIT
matrix_update --worker-status running --phase startup_inventory || true
global_line "MATRIX_START session_id=$session_id worker_pid=${BASHPID:-$$} variants=$(IFS=,; printf '%s' "${variants[*]}") datasets=cora,citeseer,pubmed,ppi,ogbn_arxiv backend=gpu started_at=$(now_utc)" || true
inventory_file="$session_root/startup_inventory.json"
global_line "GLOBAL_PHASE phase=startup_inventory started_at=$(now_utc) checking_all_dataset_metadata_and_cuda" || true
inventory_rc=0
run_global python3 "$helper" inventory --project-root "$root" --output "$inventory_file" || inventory_rc=$global_command_rc
cuda_status=unknown
if [[ -f $inventory_file ]]; then cuda_status=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("cuda",{}).get("status","unknown"))' "$inventory_file" 2>/dev/null) || cuda_status=unknown; fi
if [[ ! -f $inventory_file ]]; then inventory_rc=${inventory_rc:-1}; global_line "INVENTORY_MISSING rc=$inventory_rc" || true; fi
matrix_update --phase startup_inventory --inventory-path "$inventory_file" || true
global_line "GLOBAL_PHASE phase=startup_inventory finished_at=$(now_utc) inventory_rc=$inventory_rc cuda_status=$cuda_status" || true

mkdir -p -- "$session_root/logs" || true
: > "$build_log" || build_tee_rc=1
matrix_update --phase build --build-status running --log logs/build_cuda.log || true
global_line "GLOBAL_PHASE phase=cuda_release_build started_at=$(now_utc) backend=gpu" || true
build_ok=1 build_failure_reason=''
build_jobs=$(python3 "$helper" build-jobs 2>&1); jobs_rc=$?
if ((jobs_rc != 0)) || [[ ! $build_jobs =~ ^[1-9][0-9]*$ ]]; then
    build_ok=0 build_rc=$jobs_rc
    build_failure_reason="invalid effective-memory CUDA build job count: $build_jobs"
    build_line "$build_failure_reason" || true
else
    build_dir="$root/build_cuda"
    configure=(cmake -S "$root" -B "$build_dir" -DCMAKE_BUILD_TYPE=Release -DGATZK_ENABLE_CUDA_BACKEND=ON -DGATZK_BUILD_CORE_EXECUTION_REGRESSION=OFF -DGATZK_BUILD_PROTOCOL_VARIANT_REGRESSION=OFF)
    if [[ ! -f $build_dir/CMakeCache.txt ]] || ! grep -q '^CMAKE_CUDA_ARCHITECTURES:' "$build_dir/CMakeCache.txt"; then configure+=( -DCMAKE_CUDA_ARCHITECTURES=native ); fi
    if ! run_build "${configure[@]}"; then build_ok=0; build_failure_reason="CUDA CMake configure failed (rc=$build_rc logger_rc=$build_tee_rc)"
    elif ! run_build cmake --build "$build_dir" --parallel "$build_jobs" --target gatzk_run gatzk_registry_register gatzk_registry_inspect; then build_ok=0; build_failure_reason="CUDA production target build failed (rc=$build_rc logger_rc=$build_tee_rc)"
    elif ! run_build python3 "$helper" check-build --project-root "$root" --build-dir "$build_dir" --cuda; then build_ok=0; build_failure_reason="CUDA source/build provenance check failed (rc=$build_rc logger_rc=$build_tee_rc)"
    fi
fi
failures=0
if ((build_ok)); then
    matrix_update --build-status passed --command-rc 0 --logger-rc "$build_tee_rc" --reason 'CUDA Release production targets and build provenance verified' --phase cells || true
    global_line "GLOBAL_PHASE phase=cuda_release_build finished_at=$(now_utc) status=passed log=$build_log" || true
    registry_register="$root/build_cuda/gatzk_registry_register" registry_inspect="$root/build_cuda/gatzk_registry_inspect" run_binary="$root/build_cuda/gatzk_run"
    for variant in "${variants[@]}"; do for dataset in "${datasets[@]}"; do
        if process_cell "$variant" "$dataset"; then global_line "CELL_COMPLETE variant=$variant dataset=$dataset status=success finished_at=$(now_utc)" || true
        else failures=$((failures + 1)); global_line "CELL_COMPLETE variant=$variant dataset=$dataset status=failed finished_at=$(now_utc)" || true
        fi
    done; done
else
    matrix_update --build-status failed --command-rc "$build_rc" --logger-rc "$build_tee_rc" --reason "$build_failure_reason" --phase build || true
    global_line "GLOBAL_PHASE phase=cuda_release_build finished_at=$(now_utc) status=failed command_rc=$build_rc logger_rc=$build_tee_rc reason=$build_failure_reason log=$build_log" || true
    for variant in "${variants[@]}"; do for dataset in "${datasets[@]}"; do
        begin_cell "$variant" "$dataset"
        cell_line "CELL_PRECHECK_FAILED variant=$variant dataset=$dataset phase=build at=$(now_utc) build_rc=$build_rc reason=$build_failure_reason" || true
        record_stage build "$build_rc" "$build_tee_rc" "$global_logger_rc" "$build_failure_reason"
        fail_cell build "$build_failure_reason" null null
        failures=$((failures + 1))
    done; done
fi
refresh_session_failure
terminal=complete
if ((failures != 0)) || [[ -n $session_failure_reason ]]; then terminal=finished_with_failures; fi
worker_finalized=0
if matrix_update --worker-status "$terminal" --phase matrix_complete --current-active false "${session_failure_args[@]}"; then
    worker_finalized=1
else
    session_failure_reason="final worker terminal-state update failed helper_rc=$last_matrix_helper_rc logger_rc=$last_matrix_logger_rc"
    if [[ -n $session_matrix_error ]]; then session_failure_reason="$session_failure_reason; $session_matrix_error"; fi
    if ((last_matrix_helper_rc != 0)); then session_failure_rc=$last_matrix_helper_rc
    elif ((last_matrix_logger_rc != 0)); then session_failure_rc=$last_matrix_logger_rc
    else session_failure_rc=null; fi
    session_failure_args=(--session-failure-reason "$session_failure_reason" --session-failure-rc "$session_failure_rc")
    terminal=finished_with_failures
    if matrix_update --worker-status "$terminal" --phase matrix_complete --current-active false "${session_failure_args[@]}"; then
        worker_finalized=1
    else
        global_line "WORKER_TERMINAL_UPDATE_RETRY_FAILED helper_rc=$last_matrix_helper_rc logger_rc=$last_matrix_logger_rc reason=$session_failure_reason" || true
        worker_finalized=1
    fi
fi
global_line "MATRIX_FINISHED status=$terminal finished_at=$(now_utc) session_id=$session_id" || true
summary_view_failed=0
run_global python3 "$helper" view --project-root "$root" --mode summary || summary_view_failed=1
if ((summary_view_failed)); then
    if [[ -z $session_failure_reason ]]; then
        session_failure_reason="final summary view failed command_rc=$global_command_rc global_logger_rc=$session_global_logger_rc"
        if ((global_command_rc != 0)); then session_failure_rc=$global_command_rc
        elif ((session_global_logger_rc != 0)); then session_failure_rc=$session_global_logger_rc
        else session_failure_rc=1; fi
        session_failure_args=(--session-failure-reason "$session_failure_reason" --session-failure-rc "$session_failure_rc")
    fi
    terminal=finished_with_failures
    if ! matrix_update --worker-status "$terminal" --phase matrix_complete --current-active false "${session_failure_args[@]}"; then
        local_retry_reason="summary-view failure status update failed helper_rc=$last_matrix_helper_rc logger_rc=$last_matrix_logger_rc"
        session_failure_reason="$session_failure_reason; $local_retry_reason"
        if ((last_matrix_helper_rc != 0)); then session_failure_rc=$last_matrix_helper_rc
        elif ((last_matrix_logger_rc != 0)); then session_failure_rc=$last_matrix_logger_rc; fi
        session_failure_args=(--session-failure-reason "$session_failure_reason" --session-failure-rc "$session_failure_rc")
        if ! matrix_update --worker-status "$terminal" --phase matrix_complete --current-active false "${session_failure_args[@]}"; then
            global_line "WORKER_SUMMARY_VIEW_FINALIZE_FAILED helper_rc=$last_matrix_helper_rc logger_rc=$last_matrix_logger_rc reason=$session_failure_reason" || true
        fi
    fi
fi
refresh_session_failure
if [[ $terminal == complete && -n $session_failure_reason ]]; then
    terminal=finished_with_failures
    if ! matrix_update --worker-status "$terminal" --phase matrix_complete --current-active false "${session_failure_args[@]}"; then
        local_retry_reason="post-terminal logger/view failure status update failed helper_rc=$last_matrix_helper_rc logger_rc=$last_matrix_logger_rc"
        session_failure_reason="$session_failure_reason; $local_retry_reason"
        if ((last_matrix_helper_rc != 0)); then session_failure_rc=$last_matrix_helper_rc
        elif ((last_matrix_logger_rc != 0)); then session_failure_rc=$last_matrix_logger_rc; fi
        session_failure_args=(--session-failure-reason "$session_failure_reason" --session-failure-rc "$session_failure_rc")
        if ! matrix_update --worker-status "$terminal" --phase matrix_complete --current-active false "${session_failure_args[@]}"; then
            global_line "POST_TERMINAL_FAILURE_UPDATE_FAILED helper_rc=$last_matrix_helper_rc logger_rc=$last_matrix_logger_rc reason=$session_failure_reason" || true
        fi
    fi
fi
if ((failures == 0)) && [[ -z $session_failure_reason ]]; then exit 0; else exit 1; fi
