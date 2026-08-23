#!/usr/bin/env python3
"""Serialized formal matrix driver with durable progress and fail-closed exits."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["cora", "citeseer", "pubmed", "ppi", "ogbn-arxiv"]
DEVICES = ["cpu", "gpu"]


def write_state(path: Path, **values: object) -> None:
    payload = {
        "schema": "zkDL_gat_matrix_state_v1",
        "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        **values,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=ROOT.parents[1])
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()
    state = ROOT / "results" / "run_until_complete_state.json"
    if not args.skip_build:
        completed = subprocess.run([ROOT / "scripts" / "build.sh", "--all"], cwd=ROOT, check=False)
        if completed.returncode:
            write_state(state, status="failed", phase="build", return_code=completed.returncode)
            return completed.returncode
    completed_rows: list[dict[str, str]] = []
    for dataset in DATASETS:
        for device in DEVICES:
            command = [
                sys.executable,
                ROOT / "scripts" / "run_single.py",
                "--dataset",
                dataset,
                "--device",
                device,
                "--project-root",
                args.project_root,
            ]
            write_state(
                state,
                status="running",
                phase="formal_row",
                dataset=dataset,
                device=device,
                completed_rows=completed_rows,
                command=[str(item) for item in command],
            )
            completed = subprocess.run(command, cwd=ROOT, check=False)
            if completed.returncode:
                write_state(
                    state,
                    status="failed",
                    phase="formal_row",
                    dataset=dataset,
                    device=device,
                    completed_rows=completed_rows,
                    return_code=completed.returncode,
                )
                return completed.returncode
            completed_rows.append({"dataset": dataset, "device": device})
    subprocess.run([sys.executable, ROOT / "scripts" / "collect_results.py"], cwd=ROOT, check=True)
    validation = subprocess.run(
        [sys.executable, ROOT / "scripts" / "validate_final_results.py"], cwd=ROOT, check=False
    )
    write_state(
        state,
        status="privacy_blocked" if validation.returncode == 2 else "complete" if validation.returncode == 0 else "failed",
        phase="validation",
        completed_rows=completed_rows,
        validator_return_code=validation.returncode,
        privacy_profile_status="unsupported_privacy_profile",
        formal_experiment_eligible=False,
    )
    return validation.returncode


if __name__ == "__main__":
    raise SystemExit(main())
