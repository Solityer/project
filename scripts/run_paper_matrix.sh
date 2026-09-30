#!/usr/bin/env bash

root=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P) || exit 1
cd "$root" || exit 1
umask 077
suite_tag="paper_$(date -u +%Y%m%dT%H%M%S)_$$_$(date +%N)"
log_dir="$root/runs/full_dataset_logs/$suite_tag"
mkdir -p -- "$log_dir" || exit 1
summary="$log_dir/status.tsv"
printf 'backend\tdataset\tnative_status\tlog_status\tfinal_status\tlog\n' > "$summary" || exit 1

run_cell() {
    local backend=$1 dataset=$2 binary compute_backend config tag cell_log
    local native_status log_status final_status
    local -a pipeline_status command
    config="configs/${dataset}_secure.cfg"
    if [[ $backend == gpu ]]; then
        binary="$root/build_cuda/gatzk_run"
        compute_backend=cuda_hotspots
    else
        binary="$root/build/gatzk_run"
        compute_backend=cpu
    fi
    tag="${suite_tag}_${backend}_${dataset}"
    cell_log="$log_dir/${backend}_${dataset}.log"
    if [[ ! -x $binary || ! -f $config ]]; then
        printf 'Missing executable or config: %s %s\n' "$binary" "$config" > "$cell_log"
        printf '%s\t%s\t127\t0\t127\t%s\n' "$backend" "$dataset" "$cell_log" >> "$summary"
        return 127
    fi
    {
        printf 'start_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        printf 'git_head=%s\n' "$(git rev-parse HEAD)"
        printf 'binary_sha256=%s\n' "$(sha256sum "$binary" | cut -d ' ' -f1)"
        printf 'config=%s\nexport_tag=%s\n' "$config" "$tag"
    } > "$cell_log" || return 1
    command=("$binary" --config "$config" --compute-backend "$compute_backend"
             --benchmark-mode single --export-tag "$tag")
    if [[ -x /usr/bin/time ]]; then
        command=(/usr/bin/time -v -o "$log_dir/${backend}_${dataset}.resource.txt" -- "${command[@]}")
    else
        printf 'Resource receipt unavailable: /usr/bin/time is missing.\n' >> "$cell_log"
    fi
    printf '===== START %s %s =====\n' "$backend" "$dataset"
    "${command[@]}" 2>&1 | tee -a "$cell_log"
    pipeline_status=("${PIPESTATUS[@]}")
    native_status=${pipeline_status[0]}
    log_status=${pipeline_status[1]}
    final_status=$native_status
    if ((final_status == 0 && log_status != 0)); then
        final_status=$log_status
    fi
    if ((final_status == 0)) && ! grep -Eq "^dataset=${dataset} .*backend=${backend} status=VERIFY_OK " "$cell_log"; then
        final_status=1
        printf 'Missing matching native VERIFY_OK result.\n' >> "$cell_log"
    fi
    printf 'end_utc=%s\nnative_status=%s\nlog_status=%s\nfinal_status=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$native_status" "$log_status" "$final_status" >> "$cell_log" || final_status=1
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$backend" "$dataset" "$native_status" \
        "$log_status" "$final_status" "$cell_log" >> "$summary" || final_status=1
    printf '===== END %s %s status=%s =====\n' "$backend" "$dataset" "$final_status"
    return "$final_status"
}

attempted=0
passed=0
failed=0
for backend in gpu cpu; do
    for dataset in cora citeseer pubmed ppi ogbn_arxiv; do
        attempted=$((attempted + 1))
        if run_cell "$backend" "$dataset"; then
            passed=$((passed + 1))
        else
            failed=$((failed + 1))
        fi
    done
done
printf 'ATTEMPTED=%s PASSED=%s FAILED=%s STATUS_FILE=%s\n' "$attempted" "$passed" "$failed" "$summary"
((attempted == 10 && failed == 0))
