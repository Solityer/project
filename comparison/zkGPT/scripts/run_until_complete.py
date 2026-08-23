#!/usr/bin/env python3
"""Fail-closed formal driver; no retained reference graft is executed."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    preflight = subprocess.run([sys.executable, ROOT / "scripts" / "formal_preflight.py"], cwd=ROOT, check=False)
    subprocess.run([sys.executable, ROOT / "scripts" / "collect_results.py"], cwd=ROOT, check=True)
    validation = subprocess.run([sys.executable, ROOT / "scripts" / "validate_final_results.py"], cwd=ROOT, check=False)
    state = {
        "schema": "zkGPT_gat_matrix_state_v1",
        "status": "privacy_blocked",
        "formal_experiment_eligible": False,
        "preflight_return_code": preflight.returncode,
        "validator_return_code": validation.returncode,
        "completed_rows": [],
        "pending_rows": [
            {"dataset": dataset, "backend": backend}
            for backend in ("cpu", "gpu")
            for dataset in ("cora", "citeseer", "pubmed", "ppi", "ogbn-arxiv")
        ],
    }
    output = ROOT / "results" / "run_until_complete_state.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 2 if preflight.returncode == 2 and validation.returncode == 2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
