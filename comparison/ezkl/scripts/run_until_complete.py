#!/usr/bin/env python3
"""Resume missing formal matrix cells without changing proof semantics."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verified(dataset: str, device: str) -> bool:
    path = ROOT / "results" / "runs" / device / dataset / "run_manifest.json"
    if not path.is_file():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("status") == "verified" and payload.get("verify_passed") is True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", nargs="+", choices=("cpu", "cuda"), default=["cpu", "cuda"])
    parser.add_argument("--datasets", nargs="+",
                        choices=("cora_full", "citeseer_full", "pubmed_full",
                                 "ppi_full_formal", "ogbn_arxiv_full"),
                        default=None,
                        help="datasets to resume; omitted means all five datasets")
    parser.add_argument(
        "--finalize",
        action="store_true",
        help="after the selected resume work, require and strictly validate the complete 5x2 matrix",
    )
    args = parser.parse_args()
    datasets = json.loads((ROOT / "configs" / "full_datasets.json").read_text(encoding="utf-8"))
    selected_datasets = tuple(args.datasets or [entry["run_id"] for entry in datasets["datasets"]])
    selected_devices = tuple(args.devices)
    selected_cells = {
        (dataset, device)
        for dataset in selected_datasets
        for device in selected_devices
    }
    complete_cells = {
        (entry["run_id"], device)
        for entry in datasets["datasets"]
        for device in ("cpu", "cuda")
    }
    selected_full_matrix = selected_cells == complete_cells
    for entry in datasets["datasets"]:
        if entry["run_id"] not in selected_datasets:
            continue
        for device in args.devices:
            if verified(entry["run_id"], device):
                continue
            command = [
                sys.executable,
                str(ROOT / "scripts" / "run_gat_ezkl_baseline.py"),
                "--dataset", entry["run_id"], "--device", device,
                "--run",
                "--clean",
            ]
            subprocess.run(command, cwd=ROOT, check=True)

    collect_command = [
        sys.executable,
        str(ROOT / "scripts" / "collect_results.py"),
    ]
    if args.finalize:
        # Finalization is an explicit matrix-level action.  Do not let a
        # partial selector weaken the collector's complete-matrix requirement.
        collect_command.extend(["--datasets", *[entry["run_id"] for entry in datasets["datasets"]]])
        collect_command.extend(["--devices", "cpu", "cuda"])
    else:
        collect_command.extend(["--datasets", *selected_datasets])
        collect_command.extend(["--devices", *selected_devices])
        if not selected_full_matrix:
            collect_command.append("--allow-incomplete")
    subprocess.run(collect_command, cwd=ROOT, check=True)

    if not selected_full_matrix and not args.finalize:
        print(json.dumps({
            "status": "partial_matrix_collected",
            "requested_datasets": list(selected_datasets),
            "requested_devices": list(selected_devices),
            "final_validation": "deferred_until_complete_5x2_or_explicit_finalize",
        }, sort_keys=True))
        return 0

    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "validate_final_results.py")],
        cwd=ROOT,
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
