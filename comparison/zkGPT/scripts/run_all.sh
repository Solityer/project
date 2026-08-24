#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
if [[ $# -eq 0 ]]; then
  echo "usage: $0 --public-y-root <independent-public-y-directory> [extra args...]" >&2
  exit 2
fi
python3 scripts/gat_zkgpt_experiment.py --datasets all --backends all --force "$@"
