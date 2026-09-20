#!/usr/bin/env python3
"""Fail-closed validator for the exact zkDL formal result matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

from result_artifacts import ArtifactError, audit_run, require_exact_matrix, require_selected_backend


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument(
        "--backend", choices=("cpu", "gpu"), default="cpu",
        help="selected backend to validate when --require-matrix is absent (default: cpu)",
    )
    parser.add_argument(
        "--require-matrix", action="store_true",
        help="explicitly require the complete five-dataset CPU/GPU matrix",
    )
    parser.add_argument("--require-pairs", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    runs = (root / args.runs).resolve()
    try:
        runs.relative_to(root.resolve())
    except ValueError:
        raise SystemExit("final-validate: runs directory must be baseline-local")
    if not runs.is_dir():
        raise SystemExit(f"final-validate: runs directory is missing: {runs}")
    records = sorted(runs.rglob("run.record"))
    if not records:
        raise SystemExit("final-validate: no production run records")
    rows: list[dict[str, str]] = []
    failures: list[str] = []
    for record_path in records:
        try:
            rows.append(audit_run(root, record_path.parent))
        except ArtifactError as exc:
            failures.append(str(exc))
    if failures:
        raise SystemExit("final-validate: " + "; ".join(failures))
    if args.require_pairs:
        args.require_matrix = True
    try:
        if args.require_matrix:
            require_exact_matrix(rows)
        else:
            require_selected_backend(rows, args.backend)
    except ArtifactError as exc:
        raise SystemExit(f"final-validate: {exc}") from exc
    if args.require_matrix:
        print("final_validation=PASS records=10 exact_matrix=5x2 independent_verify=10")
    else:
        print(f"final_validation=PASS records=5 selected_backend={args.backend} independent_verify=5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
