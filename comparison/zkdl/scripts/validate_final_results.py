#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["cora", "citeseer", "pubmed", "ppi", "ogbn-arxiv"]
DEVICES = ["cpu", "gpu"]


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=ROOT / "runs")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "local_validation.json")
    args = parser.parse_args()
    runs = args.runs.resolve()
    output = args.output.resolve()
    base = ROOT.resolve()
    for path in (runs, output):
        if path != base and base not in path.parents:
            raise SystemExit("validation paths must stay inside comparison/zkdl")
    errors: list[str] = []
    verified_rows = 0
    for dataset in DATASETS:
        for device in DEVICES:
            label = f"{dataset}/{device}"
            run = runs / dataset / device / "warm"
            result = read_json(run / "result.json")
            proof = read_json(run / "proof_manifest.json")
            verifier = read_json(run / "verifier_manifest.json")
            preflight = read_json(run / "formal_artifacts" / "preflight.json")
            proof_path = run / "formal_artifacts" / "proof.bin"
            checks = {
                "result_key": result.get("dataset") == dataset and result.get("device") == device,
                "accepted_preflight": preflight.get("status") == "accepted",
                "proof_generated": result.get("proof_generated") is True and proof.get("proof_generated") is True,
                "verified": result.get("verified") is True and verifier.get("verified") is True,
                "independent_verifier": verifier.get("independent_verifier") is True,
                "verifier_reads_witness_false": verifier.get("verifier_reads_witness") is False,
                "context_match": all(
                    proof.get(key) == verifier.get(key) == result.get(key)
                    for key in ("statement_id", "model_registry_id", "device", "proof_sha256", "proof_size_bytes")
                ),
                "proof_file": proof_path.is_file()
                and proof_path.stat().st_size == proof.get("proof_size_bytes")
                and sha256(proof_path) == proof.get("proof_sha256"),
                "privacy_status_explicit": result.get("privacy_profile_status") == "supported_multi_session_registered_model"
                and result.get("formal_experiment_eligible") is True,
                "proof_format": proof.get("proof_format") == "zkDL-GAT-proof-envelope-v3"
                and verifier.get("proof_format") == "zkDL-GAT-proof-envelope-v3",
                "model_session_rerandomization": result.get("model_session_rerandomization") == "pedersen_blinder_schnorr_v1"
                and proof.get("model_session_rerandomization") == "pedersen_blinder_schnorr_v1"
                and verifier.get("model_session_rerandomization") == "pedersen_blinder_schnorr_v1"
                and verifier.get("model_rerandomization_verified") is True
                and int(proof.get("model_session_commitments", 0)) == int(proof.get("model_rerandomization_proofs", 0))
                and int(proof.get("model_session_commitments", 0)) > 0,
            }
            if device == "gpu":
                telemetry = proof.get("gpu_telemetry") or {}
                checks["gpu_provider"] = telemetry.get("provider_id") == "zkDL-native-CUDA-BLS12-381-v1"
                checks["gpu_kat"] = telemetry.get("cross_backend_kat_passed") is True
                checks["gpu_kernel_coverage"] = all(
                    int(telemetry.get(key, 0)) > 0
                    for key in (
                        "msm_kernel_invocations",
                        "field_inner_product_invocations",
                        "scalar_fold_invocations",
                        "point_fold_invocations",
                        "sumcheck_kernel_invocations",
                    )
                ) and verifier.get("gpu_telemetry_validated") is True
            failed = [name for name, passed in checks.items() if not passed]
            if failed:
                errors.extend(f"{label}:{name}" for name in failed)
            else:
                verified_rows += 1
    audit = {
        "schema": "zkDL_gat_local_validation_v1",
        "baseline": "zkdl",
        "verified_rows": verified_rows,
        "expected_rows": 10,
        "computational_validation": "passed" if not errors else "failed",
        "privacy_profile_status": "supported_multi_session_registered_model",
        "formal_experiment_eligible": not errors,
        "status": "complete" if not errors else "failed",
        "errors": errors,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if errors:
        print(f"zkdl_final_validation=FAILED errors={len(errors)}")
        return 1
    print("zkdl_computational_validation=OK rows=10 privacy_profile=SUPPORTED_MULTI_SESSION")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
