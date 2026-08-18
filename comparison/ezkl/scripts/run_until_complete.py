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
    path = ROOT / "runs" / dataset / device / "run_manifest.json"
    if not path.is_file():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("status") == "verified" and payload.get("verify_passed") is True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", nargs="+", choices=("cpu", "cuda"), default=["cpu", "cuda"])
    parser.add_argument("--srs-path", type=Path, required=True)
    parser.add_argument("--ceremony-descriptor", type=Path, required=True)
    args = parser.parse_args()
    datasets = json.loads((ROOT / "configs" / "full_datasets.json").read_text(encoding="utf-8"))
    for entry in datasets["datasets"]:
        for device in args.devices:
            if verified(entry["run_id"], device):
                continue
            command = [
                sys.executable,
                str(ROOT / "scripts" / "run_gat_ezkl_baseline.py"),
                "--dataset", entry["run_id"], "--device", device,
                "--srs-path", str(args.srs_path.resolve()), "--run",
                "--ceremony-descriptor", str(args.ceremony_descriptor.resolve()),
            ]
            subprocess.run(command, cwd=ROOT, check=True)
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "validate_final_results.py")], cwd=ROOT, check=True
    )
    subprocess.run([sys.executable, str(ROOT / "scripts" / "collect_results.py")], cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
