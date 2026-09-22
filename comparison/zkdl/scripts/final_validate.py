#!/usr/bin/env python3
"""Fail-closed validator for the exact zkDL formal result matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

from result_artifacts import ArtifactError, audit_run, require_exact_matrix, require_selected_backend


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=None,
                        help="baseline-local run directory (legacy spelling)")
    parser.add_argument("--runs-dir", type=Path, default=None,
                        help="run directory relative to --root (fixture mode supports a temp root)")
    parser.add_argument("--root", type=Path, default=None,
                        help="explicit artifact root; requires --fixture-mode")
    parser.add_argument("--fixture-mode", action="store_true",
                        help="audit explicitly-marked tiny CPU fixture artifacts, never formal results")
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
    if args.runs is not None and args.runs_dir is not None:
        raise SystemExit("final-validate: --runs and --runs-dir are mutually exclusive")
    formal_root = Path(__file__).resolve().parent.parent
    if args.root is not None and not args.fixture_mode:
        raise SystemExit("final-validate: --root requires explicit --fixture-mode")
    root = (args.root if args.root is not None else formal_root).resolve()
    runs_arg = args.runs_dir if args.runs_dir is not None else args.runs
    runs = (root / (runs_arg if runs_arg is not None else Path("runs"))).resolve()
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
            rows.append(audit_run(root, record_path.parent, fixture_mode=args.fixture_mode))
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
        suffix = " fixture_mode=1" if args.fixture_mode else ""
        print(f"final_validation=PASS records=5 selected_backend={args.backend} independent_verify=5{suffix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
