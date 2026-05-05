#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 scripts/run_full_baseline.py --route cpu --cpu-build-dir build
