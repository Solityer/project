#!/usr/bin/env python3
"""Execute the complete formal matrix and publish an honest matrix state."""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all zkGPT datasets/backends")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--public-y-root", required=True, help="independent public-Y file or per-dataset directory")
    args = parser.parse_args()
    command = [sys.executable, str(ROOT / "scripts" / "gat_zkgpt_experiment.py"), "--datasets", "all", "--backends", "all", "--public-y-root", args.public_y_root]
    if args.force:
        command.append("--force")
    run = subprocess.run(command, cwd=ROOT, check=False)
    collect = subprocess.run([sys.executable, str(ROOT / "scripts" / "collect_results.py")], cwd=ROOT, check=False)
    validate = subprocess.run([sys.executable, str(ROOT / "scripts" / "validate_final_results.py"), "--public-y-root", args.public_y_root], cwd=ROOT, check=False)
    state = {
        "schema": "zkGPT_gat_matrix_state_v2",
        "status": "complete" if run.returncode == 0 and collect.returncode == 0 and validate.returncode == 0 else "incomplete",
        "formal_experiment_eligible": run.returncode == 0 and validate.returncode == 0,
        "runner_return_code": run.returncode,
        "collector_return_code": collect.returncode,
        "validator_return_code": validate.returncode,
    }
    output = ROOT / "results" / "run_until_complete_state.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if state["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
