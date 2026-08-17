#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATASETS = ["cora", "citeseer", "pubmed", "ppi", "ogbn-arxiv"]
BACKENDS = ["cpu", "gpu"]


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


def validate() -> dict[str, Any]:
    errors: list[str] = []
    summary = read_json(RESULTS / "summary.json")
    rows = summary.get("rows", [])
    if not isinstance(rows, list) or len(rows) != 10:
        errors.append("summary_json_row_count_not_10")
        rows = []
    by_key = {(row.get("backend"), row.get("dataset")): row for row in rows if isinstance(row, dict)}
    for backend in BACKENDS:
        for dataset in DATASETS:
            label = f"{dataset}/{backend}"
            row = by_key.get((backend, dataset))
            if row is None:
                errors.append(f"{label}:missing_summary_row")
                continue
            if row.get("status") != "VERIFY_OK":
                errors.append(f"{label}:status_not_verify_ok")
            artifact_dir = RESULTS / backend / dataset
            manifest = read_json(artifact_dir / "run_manifest.json")
            verification = read_json(artifact_dir / "verification.json")
            benchmark = read_json(artifact_dir / "benchmark_summary.json")
            proof = artifact_dir / "proof_payload.json"
            if manifest.get("status") != "VERIFY_OK" or verification.get("status") != "VERIFY_OK":
                errors.append(f"{label}:manifest_or_verification_not_ok")
            if manifest.get("full_dataset") is not True:
                errors.append(f"{label}:not_full_dataset")
            if manifest.get("verifier_reads_witness") is not False:
                errors.append(f"{label}:verifier_reads_witness")
            if manifest.get("witness_included_in_proof") is not False:
                errors.append(f"{label}:witness_included_in_proof")
            if not proof.exists() or proof.stat().st_size <= 0:
                errors.append(f"{label}:missing_proof_payload")
            if int(benchmark.get("proof_size_bytes", 0) or 0) <= 0:
                errors.append(f"{label}:missing_proof_size")
            if backend == "gpu":
                gpu = read_json(artifact_dir / "gpu_proof_hot_path.json")
                if not (gpu.get("proof_hot_path") is True and gpu.get("cuda_kernel_executed") is True and gpu.get("cpu_fallback") is False):
                    errors.append(f"{label}:invalid_gpu_hot_path")
                if float(gpu.get("total_time_ms", 0.0) or 0.0) <= 0.0:
                    errors.append(f"{label}:missing_gpu_timing")
    negative = read_json(RESULTS / "negative_tests" / "formal_negative_tests.json")
    if negative.get("status") != "PASS":
        errors.append("negative_tests_not_passed")
    formal = read_json(RESULTS / "formal_proof_semantics_audit.json")
    if formal.get("status") != "PASS":
        errors.append("formal_semantics_audit_not_passed")
    for name in ["final_zkgpt_results.json", "final_zkgpt_results.csv", "final_zkgpt_report.md"]:
        if not (RESULTS / name).exists():
            errors.append(f"missing_{name}")
    return {"baseline": "zkGPT", "status": "passed" if not errors else "failed", "errors": errors}


def main() -> int:
    audit = validate()
    write_json(RESULTS / "local_validation.json", audit)
    if audit["status"] == "passed":
        print("zkgpt_final_validation=OK rows=10")
        return 0
    print(f"zkgpt_final_validation=FAILED errors={len(audit['errors'])}")
    for error in audit["errors"][:30]:
        print(f"- {error}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
