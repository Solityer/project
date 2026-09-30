#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
root="$(cd -- "$script_dir/.." && pwd -P)"
backend="${1:-cpu}"
if [[ "$backend" != "cpu" ]]; then
    echo "usage: $0 cpu" >&2
    exit 2
fi
if [[ "${ZKDL_ALLOW_FULL_DATASET_RUN:-0}" != "1" ]]; then
    echo "run-five: refusing full-dataset proving; set ZKDL_ALLOW_FULL_DATASET_RUN=1 after review" >&2
    exit 3
fi
for config in cora citeseer pubmed ppi ogbn_arxiv; do
    run_dir="runs/${config}_${backend}"
    "$script_dir/setup_register.sh" "$root/configs/${config}_secure.cfg"
    "$script_dir/run_single.sh" "$root/configs/${config}_secure.cfg" "$backend" "$run_dir"
    "$script_dir/verify_single.sh" "$root/configs/${config}_secure.cfg" "$run_dir"
done

python3 "$script_dir/collect_results.py" --runs runs --output results/collected.json \
    --require-selected-backend cpu
python3 "$script_dir/final_validate.py" --runs runs --backend cpu
