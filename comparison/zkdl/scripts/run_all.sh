#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

# This is the sole matrix entry point.  Build both native backends and
# authenticate all baseline-local resources before the first cell is allowed
# to create a statement, register a model, or run a proof.  The commands are
# intentionally fail-closed: a stale/missing binary or resource manifest
# cannot be hidden by a later matrix result.
bash scripts/build.sh --all
python3 scripts/prepare_resources.py --check
python3 scripts/setup_resources.py --binary bin/zkdl-formal-cpu

status=0
for dataset in cora citeseer pubmed ppi ogbn-arxiv; do
  for device in cpu gpu; do
    if ! python3 scripts/run_single.py --dataset "${dataset}" --device "${device}"; then
      status=1
    fi
  done
done
if ! python3 scripts/collect_results.py; then
  status=1
fi
if ! python3 scripts/validate_final_results.py \
  --runs "${ROOT}/runs" \
  --output "${ROOT}/results/final_validation.json"; then
  status=1
fi
exit "${status}"
