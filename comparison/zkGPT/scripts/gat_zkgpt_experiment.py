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
from zkgpt_gat.result_validation import ResultValidationError, resolve_results_dir  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the native GAT-ZKML-zkGPT proof path")
    parser.add_argument("--project-root", default=str(ROOT), help="baseline root; external project roots are rejected")
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--backends", default="all")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--setup-only", action="store_true", help="register canonical model commitments without proving")
    args = parser.parse_args()
    if pathlib.Path(args.project_root).expanduser().resolve() != ROOT.resolve():
        parser.error("--project-root must equal comparison/zkGPT")
    try:
        results_dir = resolve_results_dir(ROOT, args.results_dir)
    except ResultValidationError as exc:
        parser.error(str(exc))
    datasets = selected_datasets(args.datasets)
    if args.setup_only:
        for dataset in datasets:
            command = [sys.executable, str(ROOT / "scripts" / "register_model.py"), "--dataset", dataset, "--project-root", str(ROOT), "--zk-root", str(ROOT)]
            completed = subprocess.run(command, cwd=ROOT, check=False)
            if completed.returncode != 0:
                return completed.returncode
        return 0
    backends = selected_backends(args.backends)
    run_many(
        datasets,
        backends,
        pathlib.Path(args.project_root).resolve(),
        ROOT,
        results_dir,
        args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
