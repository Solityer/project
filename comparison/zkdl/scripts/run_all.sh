#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
for dataset in cora citeseer pubmed ppi ogbn-arxiv; do
  for device in cpu gpu; do
    python3 scripts/run_single.py --dataset "${dataset}" --device "${device}" "$@"
  done
done
