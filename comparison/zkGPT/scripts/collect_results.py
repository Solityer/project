#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["cora", "citeseer", "pubmed", "ppi", "ogbn-arxiv"]
BACKENDS = ["cpu", "gpu"]


def main() -> int:
    capabilities = json.loads((ROOT / "configs" / "formal_capabilities.json").read_text(encoding="utf-8"))
    blocker_ids = [item["id"] for item in capabilities["blockers"]]
    rows = [
        {
            "dataset": dataset,
            "backend": backend,
            "status": "unsupported_privacy_profile",
            "proof_generated": False,
            "verified": False,
            "formal_experiment_eligible": False,
            "blockers": blocker_ids,
        }
        for backend in BACKENDS
        for dataset in DATASETS
    ]
    payload = {
        "schema": "zkGPT_gat_formal_result_matrix_v1",
        "baseline": "zkGPT",
        "status": "unsupported_privacy_profile",
        "formal_experiment_eligible": False,
        "rows": rows,
    }
    results = ROOT / "results"
    results.mkdir(parents=True, exist_ok=True)
    (results / "final_zkgpt_results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    fields = ["dataset", "backend", "status", "proof_generated", "verified", "formal_experiment_eligible", "blockers"]
    with (results / "final_zkgpt_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# zkGPT formal result matrix",
        "",
        "No formal result is accepted. The strict profile is `unsupported_privacy_profile`.",
        "",
        "| backend | dataset | status | proof generated | verified |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines.extend(
        f"| {row['backend']} | {row['dataset']} | {row['status']} | false | false |" for row in rows
    )
    (results / "final_zkgpt_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
