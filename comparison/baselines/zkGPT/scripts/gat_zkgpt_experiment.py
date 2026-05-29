#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import pathlib
import sys


def _ensure_numpy_python() -> None:
    try:
        import numpy  # noqa: F401
        return
    except ModuleNotFoundError:
        venv_python = pathlib.Path("/home/pzh/project/.venv/bin/python")
        if venv_python.exists() and pathlib.Path(sys.executable) != venv_python:
            os.execv(str(venv_python), [str(venv_python), *sys.argv])
        raise


def main() -> None:
    _ensure_numpy_python()
    zk_root = pathlib.Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(zk_root / "src"))
    from zkgpt_gat.runner import run_many, selected_backends, selected_datasets

    parser = argparse.ArgumentParser(description="Run independent GAT-ZKML-zkGPT baseline experiments")
    parser.add_argument("--project-root", default="/home/pzh/project")
    parser.add_argument("--results-dir", default=str(zk_root / "results"))
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--backends", default="all")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--allow-invalid-digest-replay",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    if not args.allow_invalid_digest_replay:
        raise SystemExit(
            "fatal: the legacy Python digest/replay adapter is disabled for official zkGPT runs. "
            "It used digest commitments, residual checks, and local replay verification, so it "
            "cannot produce formal VERIFY_OK results under comparison/GAT-ZKML-zkGPT.md."
        )
    run_many(
        datasets=selected_datasets(args.datasets),
        backends=selected_backends(args.backends),
        project_root=pathlib.Path(args.project_root).resolve(),
        zk_root=zk_root,
        results_dir=pathlib.Path(args.results_dir).resolve(),
        force=args.force,
    )


if __name__ == "__main__":
    main()
