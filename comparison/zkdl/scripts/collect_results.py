#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "zkdl_gat_outputs" / "results"
FINAL_JSON = RESULTS / "final_zkdl_results.json"
FINAL_CSV = RESULTS / "final_zkdl_results.csv"
FINAL_REPORT = RESULTS / "final_zkdl_report.md"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    completed = subprocess.run([sys.executable, "tools/update_zkdl_gat_results.py"], cwd=ROOT, check=False)
    latest = read_json(RESULTS / "latest.json")
    rows = latest.get("rows", [])
    write_json(
        FINAL_JSON,
        {
            "baseline": "zkdl",
            "generated_at": latest.get("generated_at"),
            "rows": rows,
            "source": "zkdl_gat_outputs/results/latest.json",
        },
    )
    fields = ["device", "dataset", "status", "benchmark_mode", "commitment_time_ms", "prove_time_ms", "verify_time_ms", "proof_size_bytes", "blocker"]
    with FINAL_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    lines = [
        "# zkDL Final Local Results",
        "",
        "| device | dataset | status | commitment_time_ms | prove_time_ms | verify_time_ms | proof_size_bytes |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('device')} | {row.get('dataset')} | {row.get('status')} | "
            f"{row.get('commitment_time_ms')} | {row.get('prove_time_ms')} | "
            f"{row.get('verify_time_ms')} | {row.get('proof_size_bytes')} |"
        )
    FINAL_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
