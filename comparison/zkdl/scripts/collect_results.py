#!/usr/bin/env python3
"""Collect only fully verified native zkDL result artifacts.

Collection is deliberately fail-closed: a timing record, a proof envelope,
or a hand-written VERIFIED marker is not enough. Each row must pass the
baseline-local proof/resource/independent-verifier audit in
``result_artifacts.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from result_artifacts import ArtifactError, audit_run, require_formal_five


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=None,
                        help="baseline-local run directory (legacy spelling)")
    parser.add_argument("--runs-dir", type=Path, default=None,
                        help="run directory relative to --root (fixture mode supports a temp root)")
    parser.add_argument("--root", type=Path, default=None,
                        help="explicit artifact root; requires --fixture-mode")
    parser.add_argument("--output", type=Path, default=Path("results/collected.json"))
    parser.add_argument("--fixture-mode", action="store_true",
                        help="audit explicitly-marked tiny CPU fixture artifacts, never formal results")
    args = parser.parse_args()
    if args.runs is not None and args.runs_dir is not None:
        raise SystemExit("collect-results: --runs and --runs-dir are mutually exclusive")
    formal_root = Path(__file__).resolve().parent.parent
    if args.root is not None and not args.fixture_mode:
        raise SystemExit("collect-results: --root requires explicit --fixture-mode")
    root = (args.root if args.root is not None else formal_root).resolve()
    runs_arg = args.runs_dir if args.runs_dir is not None else args.runs
    runs = (root / (runs_arg if runs_arg is not None else Path("runs"))).resolve()
    try:
        runs.relative_to(root.resolve())
    except ValueError:
        raise SystemExit("collect-results: runs directory must be baseline-local")
    if not runs.is_dir():
        raise SystemExit(f"collect-results: runs directory is missing: {runs}")
    if not args.fixture_mode and (runs.parent != root / "runs" or not runs.name.startswith("formal_")):
        raise SystemExit("collect-results: formal collection requires one fresh runs/formal_<session> directory")
    records = sorted(runs.rglob("run.record"))
    if not records:
        raise SystemExit("collect-results: no production run records")
    rows: list[dict[str, str]] = []
    failures: list[str] = []
    for record_path in records:
        try:
            rows.append(audit_run(root, record_path.parent, fixture_mode=args.fixture_mode))
        except ArtifactError as exc:
            failures.append(str(exc))
    if failures:
        raise SystemExit("collect-results: " + "; ".join(failures))
    try:
        require_formal_five(rows)
    except ArtifactError as exc:
        raise SystemExit(f"collect-results: {exc}") from exc
    order = {"cora": 0, "citeseer": 1, "pubmed": 2, "ppi": 3, "ogbn_arxiv": 4}
    rows.sort(key=lambda row: order[row["dataset"]])
    output = (root / args.output).resolve()
    try:
        output.relative_to(root.resolve())
    except ValueError:
        raise SystemExit("collect-results: output escapes baseline root")
    runs_relative = runs.relative_to(root.resolve()).as_posix()
    output_relative = output.relative_to(root.resolve()).as_posix()
    if not args.fixture_mode:
        expected_output = f"results/{runs.name}/collected.json"
        if output_relative != expected_output:
            raise SystemExit(f"collect-results: formal summary must be written to {expected_output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({
            "schema": "zkDL_gat_result_collection_v3",
            "mode": "fixture" if args.fixture_mode else "formal_single_path",
            "session_id": runs.name,
            "runs_path": runs_relative,
            "collection_path": output_relative,
            **({"fixture_mode": True} if args.fixture_mode else {}),
            "rows": rows,
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        fields = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"collected_rows={len(rows)} output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
