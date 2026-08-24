#!/usr/bin/env python3
"""Collect only artifacts produced by the native zkGPT formal runner."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from zkgpt_gat.gpu_execution import validate_gpu_execution_evidence
DATASETS = ["cora", "citeseer", "pubmed", "ppi", "ogbn-arxiv"]
BACKENDS = ["cpu", "gpu"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_rows() -> list[dict]:
    rows: list[dict] = []
    for backend in BACKENDS:
        for dataset in DATASETS:
            manifest = ROOT / "results" / "runs" / backend / dataset / "run_manifest.json"
            if not manifest.exists():
                rows.append({"dataset": dataset, "backend": backend, "status": "not_run", "proof_generated": False, "verified": False, "formal_experiment_eligible": False, "artifact_path": str(manifest.parent)})
                continue
            value = json.loads(manifest.read_text(encoding="utf-8"))
            status = value.get("status", "invalid")
            artifact_dir = manifest.parent
            required = [
                "metadata.json",
                "commitments.json",
                "transcript.json",
                "proof_payload.json",
                "lookup_proof.json",
                "matrix_sumcheck_proof.json",
                "gkr_proof.json",
                "combine_proof.json",
                "hyrax_mle_openings.json",
                "verification.json",
                "independent_verification.json",
            ]
            proof_generated = all((artifact_dir / name).is_file() for name in required)
            verification = json.loads((artifact_dir / "verification.json").read_text(encoding="utf-8")) if (artifact_dir / "verification.json").is_file() else {}
            independent = json.loads((artifact_dir / "independent_verification.json").read_text(encoding="utf-8")) if (artifact_dir / "independent_verification.json").is_file() else {}
            independent_path = artifact_dir / "independent_verification.json"
            independent_digest_ok = (
                independent_path.is_file()
                and value.get("independent_verification_sha256") == _sha256(independent_path)
            )
            verified = (
                status == "VERIFY_OK"
                and proof_generated
                and value.get("final_logits_bound") is True
                and value.get("full_gat_forward_proved") is True
                and value.get("scope_mismatch") is False
                and verification.get("status") == "VERIFY_OK"
                and independent.get("status") == "VERIFY_OK"
                and value.get("independent_verification_status") == "VERIFY_OK"
                and independent_digest_ok
            )
            if backend == "gpu":
                gpu = value.get("gpu_backend", {})
                gpu_contract_valid, gpu_contract_errors = validate_gpu_execution_evidence(gpu)
                verified = (
                    verified
                    and (artifact_dir / "gpu_proof_hot_path.json").is_file()
                    and gpu_contract_valid
                )
            else:
                gpu_contract_errors = []
            rows.append({
                "dataset": dataset,
                "backend": backend,
                "status": status,
                "proof_generated": proof_generated,
                "verified": verified,
                "formal_experiment_eligible": verified,
                "artifact_path": str(manifest.parent),
                "run_digest": value.get("run_digest", ""),
                "gpu_contract_errors": ";".join(gpu_contract_errors),
            })
    return rows


def main() -> int:
    rows = _load_rows()
    eligible = len(rows) == len(DATASETS) * len(BACKENDS) and all(row["formal_experiment_eligible"] for row in rows)
    payload = {"schema": "zkGPT_gat_formal_result_matrix_v3", "baseline": "zkGPT", "status": "complete" if eligible else "not_complete", "formal_experiment_eligible": eligible, "rows": rows}
    results = ROOT / "results"
    results.mkdir(parents=True, exist_ok=True)
    (results / "final_zkgpt_results.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fields = ["dataset", "backend", "status", "proof_generated", "verified", "formal_experiment_eligible", "artifact_path", "run_digest", "gpu_contract_errors"]
    with (results / "final_zkgpt_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# zkGPT formal result matrix", "", f"formal_experiment_eligible = {str(eligible).lower()}", "", "| backend | dataset | status | proof generated | verified |", "| --- | --- | --- | --- | --- |"]
    lines.extend(f"| {row['backend']} | {row['dataset']} | {row['status']} | {str(row['proof_generated']).lower()} | {str(row['verified']).lower()} |" for row in rows)
    (results / "final_zkgpt_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0 if eligible else 1


if __name__ == "__main__":
    raise SystemExit(main())
