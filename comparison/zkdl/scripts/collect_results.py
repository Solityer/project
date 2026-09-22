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

from result_artifacts import ArtifactError, audit_run, require_exact_matrix, require_selected_backend


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
    parser.add_argument(
        "--require-matrix",
        action="store_true",
        help="require exactly the five canonical datasets on both CPU and GPU",
    )
    parser.add_argument(
        "--require-selected-backend",
        choices=("cpu", "gpu"),
        help="require exactly one fully audited cell for each canonical dataset on this backend",
    )
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
    if args.require_matrix and args.require_selected_backend:
        raise SystemExit("collect-results: --require-matrix and --require-selected-backend are mutually exclusive")
    if args.require_matrix:
        try:
            require_exact_matrix(rows)
        except ArtifactError as exc:
            raise SystemExit(f"collect-results: {exc}") from exc
    elif args.require_selected_backend:
        try:
            require_selected_backend(rows, args.require_selected_backend)
        except ArtifactError as exc:
            raise SystemExit(f"collect-results: {exc}") from exc
    order = {"cora": 0, "citeseer": 1, "pubmed": 2, "ppi": 3, "ogbn_arxiv": 4}
    backend_order = {"cpu": 0, "gpu": 1}
    rows.sort(key=lambda row: (order[row["dataset"]], backend_order[row["backend"]]))
    output = (root / args.output).resolve()
    try:
        output.relative_to(root.resolve())
    except ValueError:
        raise SystemExit("collect-results: output escapes baseline root")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({
            "schema": "zkDL_gat_result_collection_v2",
            "mode": "matrix" if args.require_matrix else "selected_backend" if args.require_selected_backend else "unconstrained",
            "selected_backend": args.require_selected_backend,
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
