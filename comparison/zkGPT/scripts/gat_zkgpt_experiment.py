#!/usr/bin/env python3
"""Formal single-dataset/backend and matrix runner for the zkGPT baseline."""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from zkgpt_gat.runner import run_many, selected_backends, selected_datasets  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the native GAT-ZKML-zkGPT proof path")
    parser.add_argument("--project-root", default=str(ROOT.parents[1]))
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--backends", default="all")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--public-y-root", default=None, help="independent public-Y file or per-dataset directory; never read from proof artifacts")
    parser.add_argument("--setup-only", action="store_true", help="register canonical model commitments without proving")
    args = parser.parse_args()
    datasets = selected_datasets(args.datasets)
    if args.setup_only:
        for dataset in datasets:
            command = [sys.executable, str(ROOT / "scripts" / "register_model.py"), "--dataset", dataset, "--project-root", args.project_root]
            completed = subprocess.run(command, cwd=ROOT, check=False)
            if completed.returncode != 0:
                return completed.returncode
        return 0
    if not args.public_y_root:
        parser.error("--public-y-root is required for formal proving and independent verification")
    backends = selected_backends(args.backends)
    run_many(
        datasets,
        backends,
        pathlib.Path(args.project_root).resolve(),
        ROOT,
        pathlib.Path(args.results_dir).resolve(),
        args.force,
        pathlib.Path(args.public_y_root).resolve() if args.public_y_root else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
