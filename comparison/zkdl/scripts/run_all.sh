#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
status=0
for dataset in cora citeseer pubmed ppi ogbn-arxiv; do
  for device in cpu gpu; do
    if ! python3 scripts/run_single.py --dataset "${dataset}" --device "${device}" "$@"; then
      status=1
    fi
  done
done
if ! python3 scripts/collect_results.py; then
  status=1
fi
exit "${status}"
