#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
scripts/run_cpu_full.sh
scripts/run_gpu_full.sh
python3 scripts/validate_results.py --route all --root .
