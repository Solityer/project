#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
"$(dirname "$0")/build_cpu.sh"

datasets=(
  cora_full
  citeseer_full
  pubmed_full
  ppi_full_formal
  ogbn_arxiv_full
)

for dataset in "${datasets[@]}"; do
  ./build/cpu/zkdl_gat_full_forward \
    --route cpu \
    --config "configs/datasets/${dataset}.cfg" \
    --experiment-root . \
    --output-root runs
done

python3 scripts/collect_results.py --route cpu --root .
python3 scripts/validate_results.py --route cpu --root .
