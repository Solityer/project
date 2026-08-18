#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
: "${EZKL_SRS_PATH:?set EZKL_SRS_PATH to the trusted ceremony SRS matching formal logrows}"
: "${EZKL_CEREMONY_DESCRIPTOR:?set EZKL_CEREMONY_DESCRIPTOR to its exact trusted descriptor}"

# Formal matrix entry.  CUDA is included by default and fails closed if the
# distinct gpu-accelerated binary or runtime device is unavailable.
read -r -a devices <<< "${EZKL_DEVICES:-cpu cuda}"
datasets=(cora_full citeseer_full pubmed_full ppi_full_formal ogbn_arxiv_full)
for dataset in "${datasets[@]}"; do
  for device in "${devices[@]}"; do
    python3 scripts/run_gat_ezkl_baseline.py \
      --dataset "${dataset}" --device "${device}" --srs-path "${EZKL_SRS_PATH}" \
      --ceremony-descriptor "${EZKL_CEREMONY_DESCRIPTOR}" --clean --run
  done
done

python3 scripts/validate_final_results.py
python3 scripts/collect_results.py
