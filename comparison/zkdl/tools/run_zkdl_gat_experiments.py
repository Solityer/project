#!/usr/bin/env python3
"""Compatibility CLI for the current formal zkDL runner."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["cora", "citeseer", "pubmed", "ppi", "ogbn-arxiv"]
DEVICES = ["cpu", "gpu"]


def selection(value: str, allowed: list[str]) -> list[str]:
    result = allowed if value == "all" else [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(result) - set(allowed))
    if unknown:
        raise SystemExit(f"unknown selection: {','.join(unknown)}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--device", default="all")
    parser.add_argument("--project-root", type=Path, default=ROOT.parents[1])
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs")
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()
    datasets = selection(args.dataset, DATASETS)
    devices = selection(args.device, DEVICES)
    if not args.skip_build:
        mode = "--all" if len(devices) == 2 else f"--{devices[0]}"
        subprocess.run([ROOT / "scripts" / "build.sh", mode], cwd=ROOT, check=True)
    for dataset in datasets:
        for device in devices:
            completed = subprocess.run(
                [
                    sys.executable,
                    ROOT / "scripts" / "run_single.py",
                    "--dataset",
                    dataset,
                    "--device",
                    device,
                    "--project-root",
                    args.project_root,
                    "--output-root",
                    args.output_root,
                ],
                cwd=ROOT,
                check=False,
            )
            if completed.returncode:
                return completed.returncode
    return subprocess.run(
        [sys.executable, ROOT / "scripts" / "collect_results.py", "--runs", args.output_root],
        cwd=ROOT,
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
