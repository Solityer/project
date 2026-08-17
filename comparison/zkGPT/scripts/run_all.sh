#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
python3 scripts/gat_zkgpt_experiment.py --datasets all --backends all --force
python3 scripts/rebuild_summary_from_artifacts.py
python3 scripts/run_negative_tests.py
python3 scripts/audit_formal_proof_semantics.py
