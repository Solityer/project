#!/usr/bin/env python3
import json
from pathlib import Path


DATASETS = ["cora_full", "citeseer_full", "pubmed_full", "ppi_full_formal", "ogbn_arxiv_full"]
REQUIRED = [
    "dataset",
    "route_id",
    "backend",
    "status",
    "proof_backend",
    "verifier_backend",
    "proof_artifact_type",
    "precheck_artifact_path",
    "full_dataset",
    "full_dataset_gate_passed",
    "strict_public_input_equivalent",
    "semantic_precheck_passed",
    "zkdl_proof_generated",
    "zkdl_proof_verified",
    "zkdl_verifier_called",
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
FULL_DATASET_GATE_FIELDS = [
    "dataset_id",
    "config_path",
    "full_dataset",
    "node_count",
    "edge_count",
    "feature_dim",
    "class_count",
    "layer_count",
    "hidden_heads_per_layer",
    "output_heads",
    "hidden_dim",
    "output_dim",
    "graph_batch_count",
    "node_ptr",
    "edge_ptr",
    "self_loop_policy",
    "edge_sort_policy",
    "feature_tensor_hash",
    "edge_tensor_hash",
    "checkpoint_tensor_count",
    "checkpoint_tensor_hashes",
    "public_input_hash",
    "witness_coverage_hash",
    "final_logits_shape",
    "final_logits_hash",
    "main_manifest_match",
]


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
                for metric_key in ["commitment_time_ms", "prove_time_ms", "verify_time_ms", "proof_size_bytes"]:
                    if manifest.get(metric_key) is not None:
                        errors.append(f"{path}: blocked row must keep formal metric {metric_key}=null")
                if manifest.get("zkdl_proof_generated"):
                    stats = manifest.get("dataset_stats", {})
                    for debug_key in [
                        "debug_commitment_attempt_time_ms",
                        "debug_prove_attempt_time_ms",
                        "debug_verify_attempt_time_ms",
                        "verifier_failed_after_ms",
                    ]:
                        if not stats.get(debug_key):
                            errors.append(f"{path}: generated blocked attempt missing dataset_stats.{debug_key}")
            if manifest.get("status") not in {"blocked", "verified"}:
                errors.append(f"{path}: unsupported status {manifest.get('status')}")

            stats = manifest.get("dataset_stats")
            if not isinstance(stats, dict):
                errors.append(f"{path}: missing dataset_stats object")
                stats = {}
            for key in FULL_DATASET_GATE_FIELDS:
                if key not in stats:
                    errors.append(f"{path}: missing full dataset gate field dataset_stats.{key}")
            if manifest.get("full_dataset_gate_passed"):
                for key in FULL_DATASET_GATE_FIELDS:
                    if stats.get(key) in ("", None):
                        errors.append(f"{path}: empty full dataset gate field dataset_stats.{key}")
                if stats.get("full_dataset") != "true":
                    errors.append(f"{path}: gate passed but dataset_stats.full_dataset is not true")

            if manifest.get("proof_artifact_type") != "zkdl_proof":
                if manifest.get("proof_artifact_path"):
                    errors.append(f"{path}: non-zkDL proof artifact must not be exposed as proof_artifact_path")
                if manifest.get("proof_artifact_generated"):
                    errors.append(f"{path}: non-zkDL artifact cannot set proof_artifact_generated")
            if manifest.get("zkdl_proof_verified") and manifest.get("status") != "verified":
                errors.append(f"{path}: zkdl_proof_verified true but status is not verified")
            if manifest.get("verifier_called") and not manifest.get("zkdl_verifier_called"):
                errors.append(f"{path}: verifier_called must mean zkDL verifier, not semantic checker")
            if manifest.get("status") == "verified":
                gates = [
                    manifest.get("full_dataset"),
                    manifest.get("full_dataset_gate_passed"),
                    manifest.get("strict_public_input_equivalent"),
                    manifest.get("zkdl_proof_generated"),
                    manifest.get("zkdl_proof_verified"),
                    manifest.get("zkdl_verifier_called"),
                    manifest.get("full_gat_forward_in_zkdl_proof"),
                    manifest.get("final_logits_derivation_complete"),
                    manifest.get("no_precomputed_logits_bypass"),
                ]
                if not all(gates):
                    errors.append(f"{path}: verified status without all proof gates")
                if manifest.get("proof_backend") != "zkDL":
                    errors.append(f"{path}: verified status without proof_backend=zkDL")
                if manifest.get("verifier_backend") != "zkDL":
                    errors.append(f"{path}: verified status without verifier_backend=zkDL")
                if manifest.get("proof_artifact_type") != "zkdl_proof":
                    errors.append(f"{path}: verified status without a zkDL proof artifact type")
                if manifest.get("commitment_time_ms") is None or manifest.get("prove_time_ms") is None:
                    errors.append(f"{path}: verified status without commitment/prove metrics")
                if manifest.get("verify_time_ms") is None or manifest.get("proof_size_bytes") in (None, 0):
                    errors.append(f"{path}: verified status without verify metrics and proof bytes")
            else:
                if manifest.get("proof_backend") == "zkDL" and not manifest.get("zkdl_proof_generated"):
                    errors.append(f"{path}: blocked row cannot claim zkDL proof backend without a generated proof")
                if manifest.get("verifier_backend") == "zkDL" and not manifest.get("zkdl_verifier_called"):
                    errors.append(f"{path}: blocked row cannot claim zkDL verifier backend without a verifier call")
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

    for legacy in (root / "runs").glob("*/**/zkdl_gat_composed_proof.json"):
        errors.append(f"legacy misleading proof artifact still present: {legacy}")

    if errors:
        for error in errors:
            print(error)
        raise SystemExit(1)
    print("validation=ok")


if __name__ == "__main__":
    main()
