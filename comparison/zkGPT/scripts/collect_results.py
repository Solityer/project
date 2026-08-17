#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    rc = subprocess.run([sys.executable, "scripts/rebuild_summary_from_artifacts.py"], cwd=ROOT, check=False).returncode
    summary = read_json(RESULTS / "summary.json")
    rows = summary.get("rows", [])
    write_json(
        RESULTS / "final_zkgpt_results.json",
        {
            "baseline": "zkGPT",
            "generated_at": summary.get("generated_at"),
            "rows": rows,
            "source": "results/summary.json",
        },
    )
    fields = ["dataset", "backend", "status", "N", "E", "G_batch", "commitment_time_ms", "prove_time_ms", "verify_time_ms", "proof_size_bytes", "gpu_hot_path_status", "cpu_fallback"]
    with (RESULTS / "final_zkgpt_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    lines = [
        "# zkGPT Final Local Results",
        "",
        "| backend | dataset | status | prove_time_ms | verify_time_ms | proof_size_bytes | gpu_hot_path_status |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('backend')} | {row.get('dataset')} | {row.get('status')} | "
            f"{row.get('prove_time_ms')} | {row.get('verify_time_ms')} | "
            f"{row.get('proof_size_bytes')} | {row.get('gpu_hot_path_status', '')} |"
        )
    (RESULTS / "final_zkgpt_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
