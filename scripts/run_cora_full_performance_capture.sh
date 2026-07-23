#!/usr/bin/env bash
set -Eeuo pipefail

# One-row, full-Cora performance capture.  This script is deliberately kept
# separate from the matrix runner: it never selects a subgraph or a fixture.

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
CONFIG_REL="configs/cora_secure.cfg"
CONFIG="$ROOT/$CONFIG_REL"
CAPTURE_ROOT="$ROOT/results/performance_engineering/cora_capture"

BACKEND=cpu
TIMEOUT_SECONDS=21600
SAMPLE_INTERVAL=5
SKIP_BUILD=0
DRY_RUN=0
JOBS=${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '1')}
THREADS=0

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN_DIR="$CAPTURE_ROOT/$STAMP"
if [[ -e "$RUN_DIR" ]]; then
    RUN_DIR="${RUN_DIR}_$$"
fi
TRACE="$RUN_DIR/CORA_FULL_PERFORMANCE_TRACE.md"
RESOURCE_CSV="$RUN_DIR/resource_samples.csv"
LOG_DIR="$RUN_DIR/logs"
SRS_DIR="$RUN_DIR/srs_preflight"

ACTIVE_PID=""
ACTIVE_PHASE=""
RESOURCE_PID=""
STDOUT_MONITOR_PID=""
STDERR_MONITOR_PID=""
INTERRUPTED=0
FINALIZED=0
FINAL_STATUS="not_started"
FINAL_EXIT_CODE="not_available"
FAILURE_STAGE="not_available"
FAILURE_REASON="not_available"
MANIFEST_PATH=""
FORMAL_EXPORT_ROOT="$ROOT/runs/cora_secure/cora_capture/$STAMP"

mkdir -p "$LOG_DIR" "$SRS_DIR"

config_value() {
    local key=$1
    awk -F= -v wanted="$key" '
        $1 ~ /^[[:space:]]*#/ { next }
        {
            left=$1
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", left)
            if (left == wanted) {
                value=$2
                sub(/^[[:space:]]+/, "", value)
                sub(/[[:space:]]+$/, "", value)
                print value
                exit
            }
        }
    ' "$CONFIG"
}

append_trace() {
    printf '%s\n' "$1" >> "$TRACE"
}

command_text() {
    local rendered=
    printf -v rendered '%q ' "$@"
    printf '%s' "${rendered% }"
}

append_command() {
    append_trace "- command: \`$(command_text "$@")\`"
}

append_log_excerpt() {
    local title=$1
    local path=$2
    append_trace "### $title"
    append_trace "- full_log: \`$path\`"
    {
        printf '%s\n' '```text'
        if [[ -s "$path" ]]; then
            tail -n 80 "$path"
        else
            printf '%s\n' 'not_available'
        fi
        printf '%s\n' '```'
    } >> "$TRACE"
}

sample_resources_once() {
    local phase=$1
    local pid=$2
    local stamp_now rss cpu threads mem gpu
    stamp_now=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    rss=$(awk '/^VmRSS:/ {print $2; exit}' "/proc/$pid/status" 2>/dev/null || true)
    [[ -n "$rss" ]] || rss=not_available
    cpu=$(ps -o %cpu= -p "$pid" 2>/dev/null | awk 'NF {print $1; exit}' || true)
    [[ -n "$cpu" ]] || cpu=not_available
    threads=$(find "/proc/$pid/task" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l || true)
    [[ "$threads" =~ ^[0-9]+$ ]] || threads=not_available
    mem=$(free -m 2>/dev/null | awk '/^Mem:/ {print $2 "/" $3 "/" $7; exit}' || true)
    [[ -n "$mem" ]] || mem=not_available
    gpu=not_available
    if command -v nvidia-smi >/dev/null 2>&1; then
        gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total \
            --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr ',' ';' || true)
        [[ -n "$gpu" ]] || gpu=not_available
    fi
    printf '%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$stamp_now" "$phase" "$pid" "$rss" "$cpu" "$threads" "$mem" "$gpu" \
        >> "$RESOURCE_CSV"
    printf '| %s | %s | %s | %s | %s | %s | %s | %s |\n' \
        "$stamp_now" "$phase" "$pid" "$rss" "$cpu" "$threads" "$mem" "$gpu" \
        >> "$TRACE"
}

resource_sampler() {
    local phase=$1
    local pid=$2
    while kill -0 "$pid" 2>/dev/null; do
        sample_resources_once "$phase" "$pid"
        sleep "$SAMPLE_INTERVAL"
    done
    sample_resources_once "$phase" "$pid" || true
}

follow_log() {
    local stream=$1
    local path=$2
    while [[ ! -f "$path" ]]; do
        sleep 0.1
    done
    tail -n 0 -F "$path" 2>/dev/null | while IFS= read -r line; do
        [[ -n "$line" ]] || continue
        if [[ "$stream" == stderr && "$line" == \[progress\]* ]]; then
            append_trace "- live_${stream}: \`$line\`"
        elif [[ "$stream" == stdout ]]; then
            append_trace "- live_${stream}: \`$line\`"
        fi
    done
}

stop_monitor() {
    local pid=$1
    if [[ -n "$pid" ]]; then
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    fi
}

terminate_active_process() {
    if [[ -n "$ACTIVE_PID" ]] && kill -0 "$ACTIVE_PID" 2>/dev/null; then
        kill -TERM -- "-$ACTIVE_PID" 2>/dev/null || kill -TERM "$ACTIVE_PID" 2>/dev/null || true
    fi
}

handle_signal() {
    INTERRUPTED=1
    FAILURE_STAGE=${ACTIVE_PHASE:-not_available}
    FAILURE_REASON="received signal"
    append_trace "- signal: received while phase=\`${ACTIVE_PHASE:-not_available}\`; termination requested"
    terminate_active_process
}

run_phase() {
    local phase=$1
    local stdout_log=$2
    local stderr_log=$3
    shift 3
    local -a command=("$@")
    local start_seconds elapsed timed_out code

    append_trace "### phase: $phase"
    append_command "${command[@]}"
    : > "$stdout_log"
    : > "$stderr_log"
    {
        printf '$'
        printf ' %q' "${command[@]}"
        printf '\n'
    } >> "$stdout_log"
    append_trace "- stdout_log: \`$stdout_log\`"
    append_trace "- stderr_log: \`$stderr_log\`"

    follow_log stdout "$stdout_log" &
    STDOUT_MONITOR_PID=$!
    follow_log stderr "$stderr_log" &
    STDERR_MONITOR_PID=$!

    ACTIVE_PHASE=$phase
    setsid env GATZK_RUN_PROGRESS=1 "${command[@]}" \
        >> "$stdout_log" 2>> "$stderr_log" &
    ACTIVE_PID=$!
    resource_sampler "$phase" "$ACTIVE_PID" &
    RESOURCE_PID=$!
    start_seconds=$SECONDS
    timed_out=0

    while kill -0 "$ACTIVE_PID" 2>/dev/null; do
        elapsed=$((SECONDS - start_seconds))
        if (( elapsed >= TIMEOUT_SECONDS )); then
            timed_out=1
            FAILURE_STAGE=$phase
            FAILURE_REASON="timeout after ${TIMEOUT_SECONDS}s"
            append_trace "- timeout: phase=$phase after ${TIMEOUT_SECONDS}s"
            terminate_active_process
            for _ in $(seq 1 30); do
                kill -0 "$ACTIVE_PID" 2>/dev/null || break
                sleep 1
            done
            if kill -0 "$ACTIVE_PID" 2>/dev/null; then
                kill -KILL -- "-$ACTIVE_PID" 2>/dev/null || kill -KILL "$ACTIVE_PID" 2>/dev/null || true
            fi
            break
        fi
        sleep 1
    done

    set +e
    wait "$ACTIVE_PID"
    code=$?
    set -e
    [[ "$timed_out" -eq 1 ]] && code=124
    [[ "$INTERRUPTED" -eq 1 ]] && code=130
    FINAL_EXIT_CODE=$code

    stop_monitor "$STDOUT_MONITOR_PID"
    stop_monitor "$STDERR_MONITOR_PID"
    wait "$RESOURCE_PID" 2>/dev/null || true
    STDOUT_MONITOR_PID=""
    STDERR_MONITOR_PID=""
    RESOURCE_PID=""
    ACTIVE_PID=""
    ACTIVE_PHASE=""
    append_trace "- exit_code: $code"
    if [[ "$code" -ne 0 ]]; then
        FAILURE_STAGE=$phase
        if [[ "$timed_out" -eq 0 && "$INTERRUPTED" -eq 0 ]]; then
            FAILURE_REASON="phase exited with code $code"
        fi
        return "$code"
    fi
    return 0
}

validate_srs() {
    local srs_path=$1
    local expected_artifact=$2
    local manifest="$srs_path/srs_manifest.json"
    python3 - "$srs_path" "$expected_artifact" "$SRS_ID" "$SRS_TYPE" "$SRS_DEGREE_BOUND" <<'PY'
import hashlib
import json
import pathlib
import sys

srs = pathlib.Path(sys.argv[1])
expected_artifact = pathlib.Path(sys.argv[2])
expected_id = sys.argv[3]
expected_type = sys.argv[4]
expected_degree = int(sys.argv[5])
domains = {
    "C": 8,
    "FH": 4194304,
    "N": 4096,
    "cat": 128,
    "d": 16,
    "edge": 32768,
    "in": 2048,
}

def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

manifest_path = srs / "srs_manifest.json"
if not manifest_path.is_file():
    raise SystemExit("missing srs_manifest.json")
data = json.loads(manifest_path.read_text(encoding="utf-8"))
checks = {
    "manifest_format": "zkgat_srs_manifest_v7",
    "srs_id": expected_id,
    "srs_type": expected_type,
    "degree_bound": expected_degree,
    "domain_sizes": domains,
}
for key, value in checks.items():
    if data.get(key) != value:
        raise SystemExit(f"{key} mismatch: {data.get(key)!r} != {value!r}")
if not expected_artifact.is_file():
    raise SystemExit("missing current compiler artifact")
artifact = srs / "protocol_compiler_artifact.txt"
expected_artifact_hash = sha256(expected_artifact)
if sha256(artifact) != expected_artifact_hash:
    raise SystemExit("protocol compiler artifact hash mismatch")
if data.get("protocol_compiler_artifact_hash") != expected_artifact_hash:
    raise SystemExit("manifest compiler artifact hash mismatch")
for key in ("pk_coef_path", "vk_path"):
    path = srs / data[key]
    if sha256(path) != data[key[:-5] + "_hash"]:
        raise SystemExit(f"{key} hash mismatch")
for domain in domains:
    rel = data.get("pk_lagrange_paths", {}).get(domain)
    expected_hash = data.get("pk_lagrange_hashes", {}).get(domain)
    if not rel or not expected_hash:
        raise SystemExit(f"missing Lagrange metadata for {domain}")
    path = (srs / rel).resolve()
    if path.parent != srs.resolve():
        raise SystemExit(f"Lagrange path escapes SRS directory: {domain}")
    if sha256(path) != expected_hash:
        raise SystemExit(f"Lagrange hash mismatch: {domain}")
text = manifest_path.read_text(encoding="utf-8")
marker = '  "manifest_hash": '
position = text.find(marker)
if position < 0:
    raise SystemExit("manifest_hash field missing")
if data.get("manifest_hash") != hashlib.sha256(text[:position].encode("utf-8")).hexdigest():
    raise SystemExit("manifest hash mismatch")
print("SRS_VALID")
PY
}

append_manifest_summary() {
    local manifest=$1
    if [[ ! -f "$manifest" ]]; then
        append_trace "## Final metrics"
        append_trace "not_available (run_manifest.json was not produced)"
        return
    fi
    python3 - "$manifest" "$TRACE" <<'PY'
import json
import pathlib
import sys

manifest = pathlib.Path(sys.argv[1])
trace = pathlib.Path(sys.argv[2])
row = json.loads(manifest.read_text(encoding="utf-8"))

def value(key):
    item = row.get(key)
    return "not_available" if item in (None, "") else str(item)

def number(key):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return None

formal_sum = number("formal_time_sum_ms")
with trace.open("a", encoding="utf-8") as out:
    out.write("## Final metrics from current program artifact\n\n")
    out.write("| metric | value |\n| --- | ---: |\n")
    for key in (
        "commitment_time_ms", "prove_time_ms", "verify_time_ms",
        "formal_time_sum_ms", "total_wall_clock_ms", "non_formal_overhead_ms",
        "proof_size_bytes", "private_plain_value_count", "verifier_reads_witness",
    ):
        out.write(f"| `{key}` | {value(key)} |\n")
    out.write("\n## Captured stage metrics\n\n")
    out.write("| stage | milliseconds | share of formal sum |\n| --- | ---: | ---: |\n")
    stages = [
        ("context", "context_build_ms"),
        ("forward", "forward_ms"),
        ("trace", "trace_generation_ms"),
        ("dynamic commitment MSM", "dynamic_commit_msm_ms"),
        ("derived commitment", "derived_commitment_point_ms"),
        ("quotient commitment", "quotient_commitment_ms"),
        ("quotient materialization", "quotient_build_ms"),
        ("quotient t_FH", "quotient_t_fh_ms"),
        ("quotient t_edge", "quotient_t_edge_ms"),
        ("domain opening", "domain_opening_ms"),
        ("external opening", "external_opening_ms"),
        ("serialization", "proof_serialization_ms"),
        ("strict parse", "proof_parse_ms"),
        ("verify transcript", "verify_transcript_ms"),
        ("verify domain opening", "verify_domain_opening_ms"),
        ("verify quotient", "verify_quotient_ms"),
        ("verify external opening", "verify_external_opening_ms"),
    ]
    for label, key in stages:
        measured = number(key)
        share = "not_available"
        if measured is not None and formal_sum not in (None, 0):
            share = f"{100.0 * measured / formal_sum:.3f}%"
        out.write(f"| {label} | {value(key)} | {share} |\n")
    out.write("\n## Main hotspot ranking from captured values\n\n")
    out.write("| hotspot | milliseconds | share of formal sum |\n| --- | ---: | ---: |\n")
    hotspots = [
        ("dynamic commitments/MSM", "dynamic_commit_msm_ms"),
        ("quotient non-commitment", "prove_quotient_noncommitment_ms"),
        ("t_edge", "quotient_t_edge_ms"),
        ("domain + external opening", None),
        ("derived commitments", "derived_commitment_point_ms"),
    ]
    for label, key in hotspots:
        if key is None:
            a = number("domain_opening_ms")
            b = number("external_opening_ms")
            measured = None if a is None or b is None else a + b
            shown = "not_available" if measured is None else f"{measured:.3f}"
        else:
            measured = number(key)
            shown = value(key)
        share = "not_available"
        if measured is not None and formal_sum not in (None, 0):
            share = f"{100.0 * measured / formal_sum:.3f}%"
        out.write(f"| {label} | {shown} | {share} |\n")
PY
}

append_progress_availability() {
    local stderr_log=$1
    append_trace "## Program stage progress availability"
    append_trace "| stage family | status |"
    append_trace "| --- | --- |"
    local label pattern status
    while IFS='|' read -r label pattern; do
        if rg -qi "$pattern" "$stderr_log" 2>/dev/null; then
            status=present
        else
            status=not_available
        fi
        append_trace "| $label | $status |"
    done <<'EOF'
context|context
trace|trace
commitment|commit|dynamic
quotient|quotient
t_edge|t_edge
NTT/IFFT|ntt|ifft
opening|opening
serialization|serializ
parse|parse
verify|verify
EOF
}

finalize_capture() {
    local previous latest_target
    if [[ "$FINALIZED" -eq 1 ]]; then
        return
    fi
    FINALIZED=1
    if [[ -z "$TRACE" || ! -f "$TRACE" ]]; then
        return
    fi
    append_trace "## Final status"
    append_trace "- status: $FINAL_STATUS"
    append_trace "- exit_code: $FINAL_EXIT_CODE"
    append_trace "- failure_stage: $FAILURE_STAGE"
    append_trace "- failure_or_exception: $FAILURE_REASON"
    if [[ -n "$MANIFEST_PATH" ]]; then
        append_manifest_summary "$MANIFEST_PATH"
    else
        append_manifest_summary ""
    fi
    local stderr_log="$LOG_DIR/main.stderr.log"
    local stdout_log="$LOG_DIR/main.stdout.log"
    if [[ -f "$stderr_log" ]]; then
        append_progress_availability "$stderr_log"
        append_log_excerpt "stdout/stderr final excerpts" "$stdout_log"
        append_log_excerpt "stderr final excerpt" "$stderr_log"
        local last_progress failure_lines
        last_progress=$(rg '\[progress\]' "$stderr_log" | tail -n 1 || true)
        failure_lines=$(rg -i 'fatal:|error:|timeout|exception' "$stderr_log" | tail -n 20 || true)
        append_trace "- last_progress: \`${last_progress:-not_available}\`"
        append_trace "- exception_lines: \`${failure_lines:-not_available}\`"
    fi
    append_trace "- raw resource CSV: \`$RESOURCE_CSV\`"
    latest_target="$CAPTURE_ROOT/latest"
    if [[ -e "$latest_target" || -L "$latest_target" ]]; then
        previous="$CAPTURE_ROOT/latest.previous_$STAMP"
        mv "$latest_target" "$previous" 2>/dev/null || true
        append_trace "- previous_latest_preserved: \`$previous\`"
    fi
    ln -s "$(basename "$RUN_DIR")" "$latest_target" 2>/dev/null || true
    append_trace "- latest: \`$latest_target\`"
}

trap 'handle_signal' INT TERM HUP
trap 'finalize_capture' EXIT

usage() {
    cat <<'EOF'
usage: scripts/run_cora_full_performance_capture.sh [options]

Options:
  --backend cpu|gpu              Formal compute backend (default: cpu)
  --timeout-seconds N            Per setup/main phase timeout (default: 21600)
  --sample-interval SECONDS     CPU/GPU/RSS sampling interval (default: 5)
  --threads N                   Bind each phase to CPUs 0..N-1
  --jobs N                      CMake build parallelism (default: host CPUs)
  --skip-build                   Reuse the selected existing build
  --dry-run                      Record commands and exit without setup or Cora
  --help                         Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)
            BACKEND=${2:?missing value for --backend}
            shift 2
            ;;
        --timeout-seconds)
            TIMEOUT_SECONDS=${2:?missing value for --timeout-seconds}
            shift 2
            ;;
        --sample-interval)
            SAMPLE_INTERVAL=${2:?missing value for --sample-interval}
            shift 2
            ;;
        --threads)
            THREADS=${2:?missing value for --threads}
            shift 2
            ;;
        --jobs)
            JOBS=${2:?missing value for --jobs}
            shift 2
            ;;
        --skip-build)
            SKIP_BUILD=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            printf 'unknown option: %s\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

case "$BACKEND" in
    cpu) COMPUTE_BACKEND=cpu; BUILD_DIR="$ROOT/build" ;;
    gpu) COMPUTE_BACKEND=cuda_hotspots; BUILD_DIR="$ROOT/build_cuda" ;;
    *) printf '%s\n' '--backend must be cpu or gpu' >&2; exit 2 ;;
esac
[[ "$TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] || { printf '%s\n' '--timeout-seconds must be a positive integer' >&2; exit 2; }
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || { printf '%s\n' '--jobs must be a positive integer' >&2; exit 2; }
[[ "$THREADS" =~ ^[0-9]+$ ]] || { printf '%s\n' '--threads must be a non-negative integer' >&2; exit 2; }
[[ "$SAMPLE_INTERVAL" =~ ^[0-9]+([.][0-9]+)?$ ]] || { printf '%s\n' '--sample-interval must be numeric' >&2; exit 2; }
awk -v interval="$SAMPLE_INTERVAL" 'BEGIN { exit !(interval > 0) }'
command -v python3 >/dev/null 2>&1 || { printf '%s\n' 'python3 is required' >&2; exit 2; }
command -v cmake >/dev/null 2>&1 || { printf '%s\n' 'cmake is required' >&2; exit 2; }
command -v setsid >/dev/null 2>&1 || { printf '%s\n' 'setsid is required' >&2; exit 2; }

DATASET=$(config_value dataset)
SRS_TYPE=$(config_value srs_type)
SRS_ID=$(config_value srs_id)
SRS_REL=$(config_value srs_path)
[[ "$DATASET" == cora ]] || { printf '%s\n' 'configs/cora_secure.cfg is not a Cora config' >&2; exit 2; }
[[ -n "$SRS_TYPE" && -n "$SRS_ID" && -n "$SRS_REL" ]] || { printf '%s\n' 'Cora config lacks SRS fields' >&2; exit 2; }
SRS_PATH="$ROOT/$SRS_REL"

mkdir -p "$RUN_DIR"
: > "$RESOURCE_CSV"
printf '%s\n' 'timestamp_utc,phase,pid,rss_kib,cpu_percent,thread_count,mem_total_used_available_mib,gpu_util_percent_used_total_mib' >> "$RESOURCE_CSV"
cat > "$TRACE" <<EOF
# Cora full performance capture

- capture_dir: \`$RUN_DIR\`
- trace_file: \`$TRACE\`
- start_utc: \`$(date -u +%Y-%m-%dT%H:%M:%SZ)\`
- config: \`$CONFIG_REL\`
- backend: \`$BACKEND\` (formal compute backend: \`$COMPUTE_BACKEND\`)
- timeout_seconds: $TIMEOUT_SECONDS
- sample_interval_seconds: $SAMPLE_INTERVAL
- requested_threads: $THREADS
- build_jobs: $JOBS
- skip_build: $SKIP_BUILD
- dry_run: $DRY_RUN

## Execution contract

The formal entry is the current \`gatzk_run\` warm path. It must complete the
full Cora graph, context, trace, commitments, quotient, openings, canonical
serialization, strict parse, verifier, and artifact export. No fixture,
subgraph, local benchmark, or old SRS fallback is permitted.

## Environment

EOF
{
    printf '%s\n' '```text'
    uname -a 2>&1 || true
    printf 'hostname=%s\n' "$(hostname 2>/dev/null || printf 'not_available')"
    printf 'online_cpus=%s\n' "$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf 'not_available')"
    printf 'affinity=%s\n' "$(taskset -pc $$ 2>/dev/null || printf 'not_available')"
    printf 'memory=%s\n' "$(free -h 2>/dev/null | awk '/^Mem:/ {print $0; exit}' || printf 'not_available')"
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>&1 || true
    else
        printf '%s\n' 'nvidia-smi=not_available'
    fi
    printf '%s\n' '```'
} >> "$TRACE"
append_trace "## Resource samples"
append_trace '| timestamp_utc | phase | pid | rss_kib | cpu_percent | threads | mem_total/used/available_mib | gpu_util/used/total_mib |'
append_trace '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |'

append_trace "## SRS preflight"
append_trace "- srs_type: \`$SRS_TYPE\`"
append_trace "- srs_id: \`$SRS_ID\`"
append_trace "- srs_path: \`$SRS_PATH\`"
append_trace "- compiler_artifact: \`$SRS_DIR/current_protocol_compiler_artifact.txt\`"

if [[ "$THREADS" -gt 0 ]]; then
    command -v taskset >/dev/null 2>&1 || { append_trace '- error: taskset is required for --threads'; exit 2; }
    HOST_CPUS=$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '0')
    (( THREADS <= HOST_CPUS )) || { append_trace "- error: --threads=$THREADS exceeds online CPUs=$HOST_CPUS"; exit 2; }
    RUN_PREFIX=(taskset -c "0-$((THREADS - 1))")
else
    RUN_PREFIX=()
fi

BUILD_BINARY="$BUILD_DIR/gatzk_run"
SETUP_BINARY="$BUILD_DIR/gatzk_srs_setup"

if [[ "$SKIP_BUILD" -eq 0 ]]; then
    CMAKE_CUDA=OFF
    [[ "$BACKEND" == gpu ]] && CMAKE_CUDA=ON
    if ! run_phase build "$LOG_DIR/build.stdout.log" "$LOG_DIR/build.stderr.log" \
        cmake -S . -B "${BUILD_DIR#$ROOT/}" \
        "-DGATZK_ENABLE_CUDA_BACKEND=$CMAKE_CUDA" \
        -DGATZK_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release; then
        FINAL_STATUS=failed
        FAILURE_REASON="build configure failed"
        exit "$FINAL_EXIT_CODE"
    fi
    if ! run_phase build_compile "$LOG_DIR/build_compile.stdout.log" "$LOG_DIR/build_compile.stderr.log" \
        cmake --build "${BUILD_DIR#$ROOT/}" --target gatzk_run gatzk_srs_setup -j "$JOBS"; then
        FINAL_STATUS=failed
        FAILURE_REASON="build compile failed"
        exit "$FINAL_EXIT_CODE"
    fi
fi

if [[ ! -x "$SETUP_BINARY" || ! -x "$BUILD_BINARY" ]]; then
    FINAL_STATUS=failed
    FAILURE_STAGE=build
    FAILURE_REASON="selected formal/setup binaries are missing; remove --skip-build or build them first"
    exit 1
fi

EXPECTED_ARTIFACT="$SRS_DIR/current_protocol_compiler_artifact.txt"
SRS_DEGREE_BOUND=16777215
SRS_DOMAINS=(C=8 FH=4194304 N=4096 cat=128 d=16 edge=32768 in=2048)

if [[ "$DRY_RUN" -eq 1 ]]; then
    FINAL_STATUS=dry_run
    FINAL_EXIT_CODE=0
    append_trace '- dry_run: setup-only and Cora execution intentionally skipped'
    append_command "$SETUP_BINARY" compile "$EXPECTED_ARTIFACT" "$SRS_DEGREE_BOUND" \
        "$ROOT/configs/srs_cora_max_envelope_v1.cfg" "${SRS_DOMAINS[@]}"
    append_command "$SETUP_BINARY" ceremony "$SRS_DIR/cora_formal_srs_v2.staging" \
        "$EXPECTED_ARTIFACT" "$SRS_ID"
    append_command "${RUN_PREFIX[@]}" "$BUILD_BINARY" --config "$CONFIG_REL" --benchmark-mode warm \
        --compute-backend "$COMPUTE_BACKEND" --fast-msm on --parallel-fft on \
        --fft-backend-upgrade on --fft-kernel-upgrade on --trace-layout-upgrade on \
        --fast-verify-pairing on --export-tag "cora_capture/$STAMP"
    exit 0
fi

if ! run_phase srs_compile "$LOG_DIR/srs_compile.stdout.log" "$LOG_DIR/srs_compile.stderr.log" \
    "$SETUP_BINARY" compile "$EXPECTED_ARTIFACT" "$SRS_DEGREE_BOUND" \
    "$ROOT/configs/srs_cora_max_envelope_v1.cfg" "${SRS_DOMAINS[@]}"; then
    FINAL_STATUS=failed
    FAILURE_REASON="current protocol compiler artifact generation failed"
    exit "$FINAL_EXIT_CODE"
fi

VALIDATION_LOG="$LOG_DIR/srs_validate.log"
if validate_srs "$SRS_PATH" "$EXPECTED_ARTIFACT" > "$VALIDATION_LOG" 2>&1; then
    append_trace '- result: existing Cora SRS matches current config, current artifact, domain catalogue, manifest and all material hashes; reused'
    append_log_excerpt 'SRS validation output' "$VALIDATION_LOG"
else
    append_trace '- result: existing Cora SRS is missing or mismatched; no old SRS fallback will be used'
    append_log_excerpt 'SRS validation output' "$VALIDATION_LOG"
    SRS_STAGE="$SRS_DIR/cora_formal_srs_v2.staging"
    if [[ -e "$SRS_STAGE" ]]; then
        FINAL_STATUS=failed
        FAILURE_REASON="unexpected non-empty SRS staging path"
        exit 1
    fi
    if ! run_phase srs_ceremony "$LOG_DIR/srs_ceremony.stdout.log" "$LOG_DIR/srs_ceremony.stderr.log" \
        "$SETUP_BINARY" ceremony "$SRS_STAGE" "$EXPECTED_ARTIFACT" "$SRS_ID"; then
        FINAL_STATUS=failed
        FAILURE_REASON="setup-only Cora SRS ceremony failed"
        exit "$FINAL_EXIT_CODE"
    fi
    if ! validate_srs "$SRS_STAGE" "$EXPECTED_ARTIFACT" > "$VALIDATION_LOG" 2>&1; then
        FINAL_STATUS=failed
        FAILURE_REASON="generated Cora SRS failed post-generation validation"
        append_log_excerpt 'generated SRS validation output' "$VALIDATION_LOG"
        exit 1
    fi
    append_trace '- generated SRS post-validation: SRS_VALID'
    if [[ -e "$SRS_PATH" || -L "$SRS_PATH" ]]; then
        BACKUP_SRS="${SRS_PATH}.mismatch_${STAMP}"
        mv "$SRS_PATH" "$BACKUP_SRS"
        append_trace "- preserved mismatched SRS at: \`$BACKUP_SRS\`"
    fi
    mv "$SRS_STAGE" "$SRS_PATH"
    append_trace "- installed newly generated validated SRS at: \`$SRS_PATH\`"
fi

MAIN_STDOUT="$LOG_DIR/main.stdout.log"
MAIN_STDERR="$LOG_DIR/main.stderr.log"
EXPORT_TAG="cora_capture/$STAMP"
MAIN_COMMAND=(
    "${RUN_PREFIX[@]}" "$BUILD_BINARY" --config "$CONFIG_REL"
    --benchmark-mode warm --compute-backend "$COMPUTE_BACKEND"
    --fast-msm on --parallel-fft on --fft-backend-upgrade on
    --fft-kernel-upgrade on --trace-layout-upgrade on --fast-verify-pairing on
    --export-tag "$EXPORT_TAG"
)
append_trace '## Formal Cora execution'
append_command "${MAIN_COMMAND[@]}"
if run_phase main "$MAIN_STDOUT" "$MAIN_STDERR" "${MAIN_COMMAND[@]}"; then
    FINAL_STATUS=completed
else
    FINAL_STATUS=failed
    [[ "$INTERRUPTED" -eq 1 ]] && FINAL_STATUS=interrupted
    [[ "$FINAL_EXIT_CODE" -eq 124 ]] && FINAL_STATUS=timeout
    FAILURE_STAGE=main
    FAILURE_REASON=${FAILURE_REASON:-formal Cora entry failed}
fi
MANIFEST_PATH=$(find "$FORMAL_EXPORT_ROOT" -type f -name run_manifest.json -print -quit 2>/dev/null || true)
[[ -n "$MANIFEST_PATH" ]] || MANIFEST_PATH=""
exit "$FINAL_EXIT_CODE"
