#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
: "${EZKL_SRS_PATH:?set EZKL_SRS_PATH to a trusted ceremony SRS}"
: "${EZKL_CEREMONY_DESCRIPTOR:?set EZKL_CEREMONY_DESCRIPTOR to its exact trusted descriptor}"
python3 scripts/run_until_complete.py --srs-path "${EZKL_SRS_PATH}" \
  --ceremony-descriptor "${EZKL_CEREMONY_DESCRIPTOR}"
