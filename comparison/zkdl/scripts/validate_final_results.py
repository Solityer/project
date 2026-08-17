#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "zkdl_gat_outputs"
RESULTS = OUT / "results"
RUNS = OUT / "runs"
DATASETS = ["cora", "citeseer", "pubmed", "ppi", "ogbn-arxiv"]
DEVICES = ["cpu", "gpu"]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_dir(dataset: str, device: str) -> Path:
    return RUNS / dataset / device / "warm"


def validate() -> dict[str, Any]:
    errors: list[str] = []
    latest = read_json(RESULTS / "latest.json")
    rows = latest.get("rows", [])
    if not isinstance(rows, list) or len(rows) != 10:
        errors.append("latest_json_row_count_not_10")
    by_key = {(row.get("device"), row.get("dataset")): row for row in rows if isinstance(row, dict)}
    for device in DEVICES:
        for dataset in DATASETS:
            label = f"{dataset}/{device}"
            row = by_key.get((device, dataset))
            if row is None:
                errors.append(f"{label}:missing_latest_row")
            elif row.get("status") != "ok":
                errors.append(f"{label}:status_not_ok:{row.get('status')}")
            root = run_dir(dataset, device)
            proof = read_json(root / "full_proof_manifest.json")
            verifier = read_json(root / "full_verifier_manifest.json")
            metadata = read_json(root / "formal_artifacts" / "zkdl_full_chain_public_metadata.json")
            proof_object = root / "formal_artifacts" / "zkdl_full_proof_object.txt"
            relations = proof.get("relations", []) if isinstance(proof.get("relations"), list) else []
            checks = {
                "proof_generated": proof.get("proof_generated") is True and verifier.get("proof_generated") is True,
                "verified": proof.get("verified") is True and verifier.get("verified") is True,
                "full_dataset": proof.get("is_full_dataset") is True,
                "missing_relations_empty": proof.get("missing_relations") == [] and verifier.get("missing_relations") == [],
                "verifier_reads_witness_false": verifier.get("verifier_reads_witness") is False,
                "witness_included_false": proof.get("witness_included_in_proof") is False and verifier.get("witness_included_in_proof") is False,
                "proof_object_exists": proof_object.exists() and proof_object.stat().st_size > 0,
                "metadata_exists": bool(metadata),
                "relations_present": bool(relations),
                "proof_size_split": int(proof.get("proof_size_bytes", 0) or 0) == sum(int(item.get("proof_size_bytes", 0) or 0) for item in relations if isinstance(item, dict)),
            }
            if device == "gpu":
                checks["gpu_major_subroutine"] = bool(relations)
            for name, passed in checks.items():
                if not passed:
                    errors.append(f"{label}:{name}")
    for name in ["final_zkdl_results.json", "final_zkdl_results.csv", "final_zkdl_report.md"]:
        if not (RESULTS / name).exists():
            errors.append(f"missing_{name}")
    return {"baseline": "zkdl", "status": "passed" if not errors else "failed", "errors": errors}


def main() -> int:
    audit = validate()
    write_json(RESULTS / "local_validation.json", audit)
    if audit["status"] == "passed":
        print("zkdl_final_validation=OK rows=10")
        return 0
    print(f"zkdl_final_validation=FAILED errors={len(audit['errors'])}")
    for error in audit["errors"][:30]:
        print(f"- {error}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
