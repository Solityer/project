#!/usr/bin/env bash

root=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P) || exit 1
cd "$root" || exit 1
umask 077

# Run all five CUDA datasets, or one named dataset.
if (($# == 0)); then
    datasets=(cora citeseer pubmed ppi ogbn_arxiv)
elif (($# == 1)); then
    case "$1" in
        cora|citeseer|pubmed|ppi|ogbn_arxiv) datasets=("$1") ;;
        *) printf 'Unknown dataset: %s\nUsage: bash scripts/run_paper_matrix.sh [cora|citeseer|pubmed|ppi|ogbn_arxiv]\n' "$1" >&2; exit 2 ;;
    esac
else
    printf 'Usage: bash scripts/run_paper_matrix.sh [cora|citeseer|pubmed|ppi|ogbn_arxiv]\n' >&2
    exit 2
fi

suite_tag="gpu_paper_$(date -u +%Y%m%dT%H%M%S)_$$_$(date +%N)"
log_dir="$root/runs/full_dataset_logs/$suite_tag"
mkdir -p -- "$log_dir" || exit 1
printf '%s\n' "$log_dir" > "$root/runs/full_dataset_logs/LATEST_GPU" || exit 1
summary="$log_dir/status.tsv"
printf 'backend\tdataset\tnative_status\tlog_status\tfinal_status\tlog\n' > "$summary" || exit 1

run_cell() {
    local dataset=$1
    local backend=gpu
    local binary="$root/build_cuda/gatzk_run"
    local config="configs/${dataset}_secure.cfg"
    local tag="${suite_tag}_gpu_${dataset}"
    local cell_log="$log_dir/gpu_${dataset}.log"
    local native_status log_status final_status
    local -a pipeline_status command

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

    command=("$binary" --config "$config" --compute-backend cuda_hotspots
             --benchmark-mode single --export-tag "$tag")
    if [[ -x /usr/bin/time ]]; then
        command=(/usr/bin/time -v -o "$log_dir/gpu_${dataset}.resource.txt" -- "${command[@]}")
    else
        printf 'Resource receipt unavailable: /usr/bin/time is missing.\n' >> "$cell_log"
    fi

    printf '===== START gpu %s =====\n' "$dataset"
    "${command[@]}" 2>&1 | tee -a "$cell_log"
    pipeline_status=("${PIPESTATUS[@]}")
    native_status=${pipeline_status[0]}
    log_status=${pipeline_status[1]}
    final_status=$native_status
    if ((final_status == 0 && log_status != 0)); then
        final_status=$log_status
    fi
    if ((final_status == 0)) && ! grep -Eq "^dataset=${dataset} .*backend=gpu status=VERIFY_OK " "$cell_log"; then
        final_status=1
        printf 'Missing matching CUDA VERIFY_OK result.\n' >> "$cell_log"
    fi

    printf 'end_utc=%s\nnative_status=%s\nlog_status=%s\nfinal_status=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$native_status" "$log_status" "$final_status" >> "$cell_log" || final_status=1
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$backend" "$dataset" "$native_status" \
        "$log_status" "$final_status" "$cell_log" >> "$summary" || final_status=1
    printf '===== END gpu %s status=%s =====\n' "$dataset" "$final_status"
    return "$final_status"
}

attempted=0
passed=0
failed=0
for dataset in "${datasets[@]}"; do
    attempted=$((attempted + 1))
    if run_cell "$dataset"; then
        passed=$((passed + 1))
    else
        failed=$((failed + 1))
    fi
done

printf 'ATTEMPTED=%s PASSED=%s FAILED=%s STATUS_FILE=%s\n' "$attempted" "$passed" "$failed" "$summary"
((failed == 0 && attempted == ${#datasets[@]}))
