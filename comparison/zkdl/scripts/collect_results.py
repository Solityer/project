#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["cora", "citeseer", "pubmed", "ppi", "ogbn-arxiv"]
DEVICES = ["cpu", "gpu"]


def read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=ROOT / "runs")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    runs = args.runs.resolve()
    output = args.output.resolve()
    base = ROOT.resolve()
    for path, label in ((runs, "runs"), (output, "output")):
        if path != base and base not in path.parents:
            raise SystemExit(f"{label} must stay inside comparison/zkdl")
    rows = []
    for dataset in DATASETS:
        for device in DEVICES:
            path = runs / dataset / device / "warm" / "result.json"
            row = read_json(path) or {
                "dataset": dataset,
                "device": device,
                "status": "missing",
                "proof_generated": False,
                "verified": False,
                "formal_experiment_eligible": False,
                "blocker": f"missing result: {path.relative_to(base)}",
            }
            if row.get("dataset") != dataset or row.get("device") != device:
                raise SystemExit(f"result key mismatch: {path}")
            rows.append(row)
    matrix_eligible = all(
        row.get("status") == "ok"
        and row.get("proof_generated") is True
        and row.get("verified") is True
        and row.get("formal_experiment_eligible") is True
        and row.get("privacy_profile_status") == "supported_multi_session_registered_model"
        for row in rows
    )
    payload = {
        "schema": "zkDL_gat_result_matrix_v1",
        "baseline": "zkdl",
        "privacy_profile_status": "supported_multi_session_registered_model",
        "formal_experiment_eligible": matrix_eligible,
        "rows": rows,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "final_zkdl_results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    fields = [
        "device",
        "dataset",
        "status",
        "proof_generated",
        "verified",
        "adapter_time_ms",
        "compile_time_ms",
        "prove_time_ms",
        "verify_time_ms",
        "proof_size_bytes",
        "privacy_profile_status",
        "formal_experiment_eligible",
        "proof_format",
        "model_session_commitments",
        "model_rerandomization_proofs",
        "model_session_rerandomization",
        "blocker",
    ]
    with (output / "final_zkdl_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# zkDL formal result matrix",
        "",
        "Strict reusable-model privacy status: `supported_multi_session_registered_model`.",
        "",
        "| dataset | device | status | verified | prove ms | verify ms | proof bytes |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('dataset')} | {row.get('device')} | {row.get('status')} | "
            f"{row.get('verified')} | {row.get('prove_time_ms', '')} | "
            f"{row.get('verify_time_ms', '')} | {row.get('proof_size_bytes', '')} |"
        )
    (output / "final_zkdl_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
