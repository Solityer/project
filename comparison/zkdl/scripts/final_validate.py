#!/usr/bin/env python3
"""Fail-closed validator for one verified zkDL result per canonical dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from result_artifacts import ArtifactError, audit_run, require_formal_five


def validate_collection(root: Path, runs: Path, collection_path: Path,
                        audited_rows: list[dict[str, str]], *, fixture_mode: bool) -> None:
    path = collection_path if collection_path.is_absolute() else root / collection_path
    if path.is_symlink() or not path.is_file():
        raise ArtifactError(f"collection summary is missing, non-regular or symlinked: {path}")
    path = path.resolve()
    try:
        collection_relative = path.relative_to(root.resolve()).as_posix()
        runs_relative = runs.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ArtifactError("collection summary or runs path escapes baseline root") from exc
    expected_path = f"results/{runs.name}/collected.json"
    expected_mode = "fixture" if fixture_mode else "formal_single_path"
    if not fixture_mode and collection_relative != expected_path:
        raise ArtifactError(f"formal collection summary must be session-bound at {expected_path}")
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"collection summary is invalid: {exc}") from exc
    if not isinstance(summary, dict) or summary.get("schema") != "zkDL_gat_result_collection_v3":
        raise ArtifactError("collection summary schema is not the single-path result contract")
    if (summary.get("mode") != expected_mode or summary.get("session_id") != runs.name or
            summary.get("runs_path") != runs_relative or summary.get("collection_path") != collection_relative):
        raise ArtifactError("collection summary is not bound to the current formal session and output path")
    summary_rows = summary.get("rows")
    if not isinstance(summary_rows, list) or any(not isinstance(row, dict) for row in summary_rows):
        raise ArtifactError("collection summary rows are malformed")
    normalized_rows = [{str(key): str(value) for key, value in row.items()} for row in summary_rows]
    try:
        require_formal_five(normalized_rows)
    except ArtifactError as exc:
        raise ArtifactError(f"collection summary: {exc}") from exc
    audited = sorted(audited_rows, key=lambda row: row["dataset"])
    collected = sorted(normalized_rows, key=lambda row: row["dataset"])
    if collected != audited:
        raise ArtifactError("collection summary does not exactly match this session's audited proof records and metrics")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=None,
                        help="baseline-local run directory (legacy spelling)")
    parser.add_argument("--runs-dir", type=Path, default=None,
                        help="run directory relative to --root (fixture mode supports a temp root)")
    parser.add_argument("--root", type=Path, default=None,
                        help="explicit artifact root; requires --fixture-mode")
    parser.add_argument("--collection", type=Path, required=True,
                        help="collector summary bound to this exact run session")
    parser.add_argument("--fixture-mode", action="store_true",
                        help="audit explicitly-marked tiny CPU fixture artifacts, never formal results")
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
    if not args.fixture_mode and (runs.parent != root / "runs" or not runs.name.startswith("formal_")):
        raise SystemExit("final-validate: formal validation requires one fresh runs/formal_<session> directory")
    records = sorted(runs.rglob("run.record"))
    if not records:
        raise SystemExit("final-validate: no production run records")
    rows: list[dict[str, str]] = []
    failures: list[str] = []
    for record_path in records:
        try:
            rows.append(audit_run(root, record_path.parent,
                                  expected_gpu_offload_stage=None if args.fixture_mode else "hyrax_ipa_fold",
                                  fixture_mode=args.fixture_mode))
        except ArtifactError as exc:
            failures.append(str(exc))
    if failures:
        raise SystemExit("final-validate: " + "; ".join(failures))
    try:
        require_formal_five(rows)
        validate_collection(root, runs, args.collection, rows, fixture_mode=args.fixture_mode)
    except ArtifactError as exc:
        raise SystemExit(f"final-validate: {exc}") from exc
    suffix = " fixture_mode=1" if args.fixture_mode else ""
    print(f"final_validation=PASS records=5 datasets=5 independent_verify=5{suffix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
