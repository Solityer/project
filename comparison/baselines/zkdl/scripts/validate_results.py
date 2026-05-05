#!/usr/bin/env python3
import json
from pathlib import Path


DATASETS = ["cora_full", "citeseer_full", "pubmed_full", "ppi_full_formal", "ogbn_arxiv_full"]
REQUIRED = [
    "dataset",
    "route_id",
    "backend",
    "status",
    "strict_public_input_equivalent",
    "full_gat_forward_in_zkdl_proof",
    "final_logits_derivation_complete",
    "no_precomputed_logits_bypass",
    "commitment_time_ms",
    "prove_time_ms",
    "verify_time_ms",
    "proof_size_bytes",
    "block_reason",
    "failure_step",
]
GPU_REQUIRED = ["gpu_backend_active", "gpu_device_name", "gpu_device_count", "gpu_backend_name", "cpu_fallback_used"]


def load(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def main():
    root = Path(__file__).resolve().parents[1]
    errors = []
    for dataset in DATASETS:
        for route in ("cpu", "gpu"):
            path = root / "runs" / dataset / route / "run_manifest.json"
            if not path.exists():
                errors.append(f"missing manifest: {path}")
                continue
            manifest = load(path)
            for key in REQUIRED:
                if key not in manifest:
                    errors.append(f"{path}: missing {key}")
            if route == "gpu":
                for key in GPU_REQUIRED:
                    if key not in manifest:
                        errors.append(f"{path}: missing {key}")
                if manifest.get("cpu_fallback_used") and manifest.get("status") == "verified":
                    errors.append(f"{path}: GPU fallback cannot be verified")
            if manifest.get("status") != "verified":
                if not manifest.get("failure_step") or not manifest.get("block_reason"):
                    errors.append(f"{path}: blocked row must carry failure_step and block_reason")
                if "capability_check" in str(manifest.get("failure_step")):
                    errors.append(f"{path}: failure_step must name the reached proof stage, not a generic capability check")
            if manifest.get("status") not in {"blocked", "verified"}:
                errors.append(f"{path}: unsupported status {manifest.get('status')}")
            if manifest.get("status") == "verified":
                gates = [
                    manifest.get("strict_public_input_equivalent"),
                    manifest.get("full_gat_forward_in_zkdl_proof"),
                    manifest.get("final_logits_derivation_complete"),
                    manifest.get("no_precomputed_logits_bypass"),
                ]
                if not all(gates):
                    errors.append(f"{path}: verified status without all proof gates")
                if manifest.get("commitment_time_ms") is None or manifest.get("prove_time_ms") is None:
                    errors.append(f"{path}: verified status without commitment/prove metrics")
                if manifest.get("verify_time_ms") is None or manifest.get("proof_size_bytes") in (None, 0):
                    errors.append(f"{path}: verified status without verify metrics and proof bytes")
            for key in ["attempted_stages", "unsupported_protocol_requirements"]:
                if key not in manifest or not isinstance(manifest[key], list):
                    errors.append(f"{path}: missing {key} array")
    for result in [
        "zkdl_cpu_full_results.json",
        "zkdl_cpu_full_results.csv",
        "zkdl_gpu_full_results.json",
        "zkdl_gpu_full_results.csv",
        "main_vs_zkdl_cpu_full_comparison.json",
        "main_vs_zkdl_cpu_full_comparison.csv",
        "main_vs_zkdl_gpu_full_comparison.json",
        "main_vs_zkdl_gpu_full_comparison.csv",
        "summary.md",
        "control_variable_audit.md",
        "code_cleanliness_audit.md",
        "cleanup_report.md",
    ]:
        if not (root / "results" / result).exists():
            errors.append(f"missing result file: results/{result}")

    scanned = []
    for folder in ["include", "src", "scripts", "configs", "manifests"]:
        scanned.extend((root / folder).rglob("*"))
    forbidden = ["/home/pzh/project/" + item for item in ("src", "include", "scripts")]
    for path in scanned:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for needle in forbidden:
            if needle in text:
                errors.append(f"{path}: forbidden dependency reference {needle}")

    if errors:
        for error in errors:
            print(error)
        raise SystemExit(1)
    print("validation=ok")


if __name__ == "__main__":
    main()
