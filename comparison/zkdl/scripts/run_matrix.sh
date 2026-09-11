#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

if [[ "${ZKDL_ALLOW_FULL_DATASET_RUN:-0}" != "1" ]]; then
  echo "run-matrix: refusing full-dataset proving; set ZKDL_ALLOW_FULL_DATASET_RUN=1 only for the user-authorized formal matrix" >&2
  exit 3
fi

"$script_dir/run_five.sh" cpu
"$script_dir/run_five.sh" gpu
