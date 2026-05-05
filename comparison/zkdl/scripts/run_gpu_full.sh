#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
"$(dirname "$0")/build_gpu.sh"

mkdir -p runs
./build/gpu/zkdl_gat_cuda_probe --output runs/gpu_probe.json

datasets=(
  cora_full
  citeseer_full
  pubmed_full
  ppi_full_formal
  ogbn_arxiv_full
)

for dataset in "${datasets[@]}"; do
  ./build/gpu/zkdl_gat_full_forward \
    --route gpu \
    --config "configs/datasets/${dataset}.cfg" \
    --experiment-root . \
    --gpu-probe runs/gpu_probe.json \
    --output-root runs
done

python3 scripts/collect_results.py --route gpu --root .
python3 scripts/validate_results.py --route gpu --root .
