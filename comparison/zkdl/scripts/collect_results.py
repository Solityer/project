#!/usr/bin/env python3
"""Collect only production run records; never infer verification from timing."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REQUIRED = {
    "schema", "dataset", "backend", "statement_id", "proof_id", "verification_status",
    "proof_scope", "proof_size_bytes", "commitment_size_bytes", "verifier_reads_witness",
    "gpu_kernel_executed", "silent_fallback", "output_bound",
}


def read_record(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            raise ValueError(f"malformed line in {path}")
        key, value = line.split("=", 1)
        if key in values:
            raise ValueError(f"duplicate key {key} in {path}")
        values[key] = value
    missing = REQUIRED - values.keys()
    if missing:
        raise ValueError(f"missing fields in {path}: {sorted(missing)}")
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--output", type=Path, default=Path("results/collected.json"))
    parser.add_argument("--allow-blocked", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    runs = (root / args.runs).resolve()
    if not runs.is_relative_to(root) or not runs.is_dir():
        raise SystemExit("collect-results: runs directory must be baseline-local")
    rows: list[dict[str, str]] = []
    failures: list[str] = []
    for record_path in sorted(runs.rglob("run.record")):
        try:
            row = read_record(record_path)
        except ValueError as exc:
            failures.append(str(exc))
            continue
        row["run_dir"] = record_path.parent.relative_to(root).as_posix()
        if not args.allow_blocked and row["verification_status"] != "VERIFIED":
            failures.append(f"{record_path}: status={row['verification_status']}")
        rows.append(row)
    if failures:
        raise SystemExit("collect-results: " + "; ".join(failures))
    output = (root / args.output).resolve()
    if not output.is_relative_to(root):
        raise SystemExit("collect-results: output escapes baseline root")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"schema": "zkDL_gat_result_collection_v1", "rows": rows}, indent=2) + "\n", encoding="utf-8")
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
