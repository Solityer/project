#!/usr/bin/env bash

set -Eeuo pipefail

root=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P) || exit 1
cd "$root" || exit 1
umask 077

usage() {
    cat <<'USAGE'
Usage:
  bash scripts/run_ablation_matrix.sh --protocol-variant wo_shared_neighborhood [--backend both|cpu|gpu]
  bash scripts/run_ablation_matrix.sh --protocol-variant wo_feature_folding [--backend both|cpu|gpu]

The default backend is both: GPU hotspots followed by CPU, over all five official datasets.
This command builds the selected formal binaries, creates and independently inspects
variant-specific verifier registrations for every dataset, then runs each formal cell
in a new isolated session under runs/ablation_sessions/.
USAGE
}

protocol_variant=""
backend_selection="both"
while (($#)); do
    case "$1" in
    --protocol-variant)
        (($# >= 2)) || { printf 'missing value for --protocol-variant\n' >&2; exit 2; }
        protocol_variant=$2
        shift 2
        ;;
    --backend)
        (($# >= 2)) || { printf 'missing value for --backend\n' >&2; exit 2; }
        backend_selection=$2
        shift 2
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

case "$protocol_variant" in
    wo_shared_neighborhood|wo_feature_folding) ;;
    *)
        printf 'error: --protocol-variant must be wo_shared_neighborhood or wo_feature_folding\n' >&2
        usage >&2
        exit 2
        ;;
esac
case "$backend_selection" in
    both) backends=(gpu cpu) ;;
    cpu|gpu) backends=("$backend_selection") ;;
    *)
        printf 'error: --backend must be both, cpu, or gpu\n' >&2
        exit 2
        ;;
esac

helper="$root/scripts/ablation_matrix_support.py"
datasets=(cora citeseer pubmed ppi ogbn_arxiv)
session_id="${protocol_variant}_$(date -u +%Y%m%dT%H%M%SZ)_$$_$(date +%N)"
session_rel="runs/ablation_sessions/${protocol_variant}/${session_id}"
session_root="$root/$session_rel"
mkdir -p -- "$root/runs/ablation_sessions/$protocol_variant"
mkdir -- "$session_root" || { printf 'session path already exists: %s\n' "$session_root" >&2; exit 1; }
mkdir -p -- "$session_root/logs" "$session_root/configs" "$session_root/registries" "$session_root/preflight" \
    "$session_root/results" "$session_root/evidence"
summary="$session_root/status.tsv"
printf 'variant\tbackend\tdataset\tnative_status\tlog_status\tfinal_status\tregistry_digest\trun_dir\tlog\n' > "$summary"
session_created=1

write_index() {
    python3 "$helper" write-index --project-root "$root" --session-root "$session_root" --variant "$protocol_variant" \
        --backends "$(IFS=,; printf '%s' "${backends[*]}")"
}
on_exit() {
    local exit_status=$?
    trap - EXIT
    if ((exit_status != 0)) && [[ ${session_created:-0} -eq 1 ]]; then
        write_index || true
        printf 'ABLATION_SESSION_FAILED status=%s variant=%s session=%s index=%s\n' \
            "$exit_status" "$protocol_variant" "$session_root" "$session_root/ablation_session_index.json" >&2
    fi
    exit "$exit_status"
}
trap on_exit EXIT

run_logged() {
    local log=$1
    shift
    printf '$' >> "$log"
    printf ' %q' "$@" >> "$log"
    printf '\n' >> "$log"
    local -a pipe_status
    set +e
    "$@" 2>&1 | tee -a "$log"
    pipe_status=("${PIPESTATUS[@]}")
    set -e
    if ((pipe_status[0] != 0)); then
        return "${pipe_status[0]}"
    fi
    if ((pipe_status[1] != 0)); then
        return "${pipe_status[1]}"
    fi
}

build_jobs=$(python3 "$helper" build-jobs)
[[ $build_jobs =~ ^[1-9][0-9]*$ ]] || { printf 'invalid effective-memory build job count: %s\n' "$build_jobs" >&2; exit 1; }
printf 'variant=%s session=%s backend=%s effective-memory-build-jobs=%s\n' \
    "$protocol_variant" "$session_root" "$backend_selection" "$build_jobs"

configure_build() {
    local build_name=$1 cuda=$2
    local build_dir="$root/$build_name" log="$session_root/logs/build_${build_name}.log"
    local -a configure_args
    configure_args=(cmake -S "$root" -B "$build_dir" -DCMAKE_BUILD_TYPE=Release "-DGATZK_ENABLE_CUDA_BACKEND=$cuda")
    if [[ $cuda == ON ]] && ! grep -q '^CMAKE_CUDA_ARCHITECTURES:' "$build_dir/CMakeCache.txt" 2>/dev/null; then
        configure_args+=( -DCMAKE_CUDA_ARCHITECTURES=native )
    fi
    run_logged "$log" "${configure_args[@]}"
    run_logged "$log" cmake --build "$build_dir" --parallel "$build_jobs" --target \
        gatzk_run gatzk_registry_register gatzk_registry_inspect
    if [[ $cuda == ON ]]; then
        python3 "$helper" check-build --project-root "$root" --build-dir "$build_dir" --cuda
    else
        python3 "$helper" check-build --project-root "$root" --build-dir "$build_dir"
    fi
}

# CPU registration tools generate the five pinned candidate registries.
configure_build build OFF
if [[ $backend_selection == both || $backend_selection == gpu ]]; then
    configure_build build_cuda ON
fi

registry_register="$root/build/gatzk_registry_register"
registry_inspect="$root/build/gatzk_registry_inspect"
for dataset in "${datasets[@]}"; do
    base_config="configs/${dataset}_secure.cfg"
    config_rel="$session_rel/configs/${dataset}.cfg"
    registry_rel="$session_rel/registries/${dataset}.registry"
    export_rel="$session_rel/results/${dataset}"
    config="$root/$config_rel"
    registry="$root/$registry_rel"
    [[ -f $root/$base_config ]] || { printf 'missing canonical base config: %s\n' "$root/$base_config" >&2; exit 1; }

    python3 "$helper" write-config --base "$root/$base_config" --output "$config" --variant "$protocol_variant" \
        --export-dir "$export_rel" --registry-path "$registry_rel"
    register_log="$session_root/logs/register_${dataset}.log"
    register_resource="$session_root/logs/register_${dataset}.resource.txt"
    printf '===== PREFLIGHT REGISTER %s variant=%s =====\n' "$dataset" "$protocol_variant"
    if [[ -x /usr/bin/time ]]; then
        run_logged "$register_log" /usr/bin/time -v -o "$register_resource" -- \
            "$registry_register" --config "$config" --output "$registry_rel"
    else
        run_logged "$register_log" "$registry_register" --config "$config" --output "$registry_rel"
    fi
    digest=$(sed -n 's/.*candidate_registry_digest=\([0-9a-f]\{64\}\).*/\1/p' "$register_log" | tail -n 1)
    [[ $digest =~ ^[0-9a-f]{64}$ ]] || { printf 'registration did not emit a canonical candidate digest for %s\n' "$dataset" >&2; exit 1; }

    python3 "$helper" write-config --base "$root/$base_config" --output "$config" --variant "$protocol_variant" \
        --export-dir "$export_rel" --registry-path "$registry_rel" --registry-digest "$digest"
    inspect_json="$session_root/preflight/${dataset}.registry.json"
    inspect_log="$session_root/logs/inspect_${dataset}.log"
    inspect_resource="$session_root/logs/inspect_${dataset}.resource.txt"
    printf '===== PREFLIGHT INDEPENDENT INSPECT %s variant=%s digest=%s =====\n' "$dataset" "$protocol_variant" "$digest"
    if [[ -x /usr/bin/time ]]; then
        /usr/bin/time -v -o "$inspect_resource" -- "$registry_inspect" --registry "$registry_rel" --trusted-digest "$digest" \
            > "$inspect_json" 2> "$inspect_log" || { cat "$inspect_log" >&2; exit 1; }
    else
        "$registry_inspect" --registry "$registry_rel" --trusted-digest "$digest" > "$inspect_json" 2> "$inspect_log" || {
            cat "$inspect_log" >&2
            exit 1
        }
    fi
    cat "$inspect_json"
    python3 "$helper" check-registry --project-root "$root" --config "$config" --variant "$protocol_variant" --registry "$registry" \
        --registry-digest "$digest" --inspection "$inspect_json"
    write_index
done

python3 "$helper" write-index --project-root "$root" --session-root "$session_root" --variant "$protocol_variant" \
    --backends "$(IFS=,; printf '%s' "${backends[*]}")"
index_status=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["status"])' \
    "$session_root/ablation_session_index.json")
[[ $index_status == formal_runs_in_progress ]] || { printf 'all dataset registry preflights did not complete\n' >&2; exit 1; }

attempted=0
passed=0
for backend in "${backends[@]}"; do
    for dataset in "${datasets[@]}"; do
        attempted=$((attempted + 1))
        config="$session_root/configs/${dataset}.cfg"
        registry_digest=$(sed -n 's/^verifier_registry_digest[[:space:]]*=[[:space:]]*\([0-9a-f]\{64\}\)[[:space:]]*$/\1/p' "$config")
        if [[ $backend == gpu ]]; then
            binary="$root/build_cuda/gatzk_run"
            compute_backend=cuda_hotspots
        else
            binary="$root/build/gatzk_run"
            compute_backend=cpu
        fi
        tag="${session_id}_${backend}_${dataset}"
        run_dir="$root/$session_rel/results/${dataset}/${tag}/single"
        cell_log="$session_root/logs/${backend}_${dataset}.log"
        resource_log="$session_root/logs/${backend}_${dataset}.resource.txt"
        {
            printf 'start_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
            printf 'variant=%s\nbackend=%s\ndataset=%s\nconfig=%s\nregistry_digest=%s\nexport_tag=%s\nbinary=%s\n' \
                "$protocol_variant" "$backend" "$dataset" "$config" "$registry_digest" "$tag" "$binary"
            printf 'git_head=%s\n' "$(git rev-parse HEAD)"
            printf 'binary_sha256=%s\n' "$(sha256sum "$binary" | cut -d ' ' -f1)"
        } > "$cell_log"
        command=("$binary" --config "$config" --protocol-variant "$protocol_variant" --compute-backend "$compute_backend" \
            --benchmark-mode single --export-tag "$tag")
        if [[ -x /usr/bin/time ]]; then
            command=(/usr/bin/time -v -o "$resource_log" -- "${command[@]}")
        else
            printf 'Resource receipt unavailable: /usr/bin/time is missing.\n' >> "$cell_log"
        fi
        printf '===== START variant=%s backend=%s dataset=%s =====\n' "$protocol_variant" "$backend" "$dataset"
        native_status=0
        log_status=0
        if run_logged "$cell_log" "${command[@]}"; then
            native_status=0
        else
            native_status=$?
        fi
        if ! grep -Eq "^dataset=${dataset} .*backend=${backend} protocol_variant=${protocol_variant} status=VERIFY_OK " "$cell_log"; then
            log_status=1
        fi
        final_status=$native_status
        if ((final_status == 0 && log_status != 0)); then
            final_status=1
            printf 'Missing matching native VERIFY_OK result.\n' >> "$cell_log"
        fi
        if ((final_status == 0)); then
            if ! python3 "$helper" validate-cell --project-root "$root" --run-dir "$run_dir" --config "$config" \
                --variant "$protocol_variant" --dataset "$dataset" --backend "$backend" --registry-digest "$registry_digest" \
                --binary "$binary" --receipt-path "$session_root/evidence/${backend}_${dataset}.json" >> "$cell_log" 2>&1; then
                final_status=1
            fi
        fi
        printf 'end_utc=%s\nnative_status=%s\nlog_status=%s\nfinal_status=%s\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$native_status" "$log_status" "$final_status" >> "$cell_log"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$protocol_variant" "$backend" "$dataset" \
            "$native_status" "$log_status" "$final_status" "$registry_digest" "$run_dir" "$cell_log" >> "$summary"
        write_index
        printf '===== END variant=%s backend=%s dataset=%s status=%s =====\n' "$protocol_variant" "$backend" "$dataset" "$final_status"
        if ((final_status != 0)); then
            printf 'formal cell failed; stopping at first failure. Session evidence: %s\n' "$session_root" >&2
            exit "$final_status"
        fi
        passed=$((passed + 1))
    done
done

printf 'ATTEMPTED=%s PASSED=%s FAILED=0 VARIANT=%s STATUS_FILE=%s SESSION_INDEX=%s RESULTS=%s\n' \
    "$attempted" "$passed" "$protocol_variant" "$summary" "$session_root/ablation_session_index.json" "$session_root/results"
((attempted == passed && attempted == ${#backends[@]} * ${#datasets[@]}))
