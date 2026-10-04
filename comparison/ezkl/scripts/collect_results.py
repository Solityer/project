#!/usr/bin/env python3
"""Collect baseline-local ezkl result cells fail-closed.

Formal collection requires the five canonical hybrid CPU+GPU results.
Missing artifacts, unverified native instances, or stale identities fail closed.
Explicit native CPU/GPU selections remain available for local component checks.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from formal_common import (
    BACKENDS,
    CANONICAL_OUTPUT_MODE,
    CANONICAL_PRIVACY,
    CANONICAL_PROOF_SCOPE,
    CANONICAL_RELATION,
    CANONICAL_STATEMENT_MODE,
    DATASETS,
    FORMAL_BACKEND,
    FORMAL_GPU_FEATURES,
    FORMAL_PHASE_BACKENDS,
    FORMAL_SESSION_ID,
    NATIVE_FORMAL_BACKEND,
    ROOT,
    ContractError,
    abort,
    canonical_json_digest,
    load_json,
    local_path,
    print_error,
    reject_forbidden_path,
    require_circuit_ref,
    require_digest,
    require_exact_keys,
    require_file,
    require_positive_timings,
    public_output_values,
    resource_manifest_hash,
    sha256,
    source_tree_hash,
    validate_artifact_identity,
    validate_commitment_metrics,
    validate_gpu_evidence,
    validate_gpu_evidence_document,
    validate_gpu_environment,
    validate_public_output,
    validate_public_statement,
)

REQUIRED_RESULT_KEYS = {
    "schema_version",
    "status",
    "dataset",
    "backend",
    "proof_scope",
    "privacy_policy",
    "public_statement_mode",
    "output_mode",
    "relation_id",
    "quant_cfg_id",
    "static_tables_id",
    "public_statement_sha256",
    "public_output_sha256",
    "srs_sha256",
    "srs_provenance",
    "settings_sha256",
    "compiled_circuit_sha256",
    "vk_sha256",
    "pk_sha256",
    "proof_sha256",
    "proof_size_bytes",
    "binary_sha256",
    "source_tree_sha256",
    "resource_manifest_sha256",
    "verifier_reads_witness",
    "proof_contains_private_witness",
    "native_verify_result",
    "native_public_instance_binding",
    "phase_timings_ms",
    "commitment_time_ms",
    "commitment_metric",
    "gpu_evidence",
    "circuit_ref",
    "proof_instances",
    "public_statement_path",
    "public_output_path",
    "srs_path",
    "settings_path",
    "compiled_circuit_path",
    "vk_path",
    "pk_path",
    "binary_path",
    "artifact_manifest_path",
    "artifact_manifest_sha256",
    "proof_path",
    "core_run_manifest_sha256",
}
PATH_DIGESTS = {
    "public_statement_path": "public_statement_sha256",
    "public_output_path": "public_output_sha256",
    "srs_path": "srs_sha256",
    "settings_path": "settings_sha256",
    "compiled_circuit_path": "compiled_circuit_sha256",
    "vk_path": "vk_sha256",
    "pk_path": "pk_sha256",
    "binary_path": "binary_sha256",
    "proof_path": "proof_sha256",
    "artifact_manifest_path": "artifact_manifest_sha256",
}
PAIR_FIELDS = (
    "relation_id",
    "proof_scope",
    "privacy_policy",
    "public_statement_mode",
    "output_mode",
    "quant_cfg_id",
    "static_tables_id",
    "public_statement_sha256",
    "public_output_sha256",
    "srs_sha256",
    "srs_provenance",
    "settings_sha256",
    "compiled_circuit_sha256",
    "vk_sha256",
    "source_tree_sha256",
    "resource_manifest_sha256",
    "circuit_ref",
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--runs-root", default="runs/formal")
    result.add_argument("--output", default="results/formal_matrix.json")
    result.add_argument("--session-id", help="active formal session; required for hybrid result collection")
    result.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASETS,
        default=list(DATASETS),
        help="dataset axis to collect; default requires the strict five-dataset matrix",
    )
    result.add_argument(
        "--backends",
        nargs="+",
        choices=(*BACKENDS, FORMAL_BACKEND),
        default=[FORMAL_BACKEND],
        help="formal default: five hybrid results; explicit cpu/gpu selections are local checks",
    )
    return result


def path_from_result(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        abort(f"result field {label} is not a path")
    path = local_path(value, label)
    reject_forbidden_path(path, label)
    return require_file(path, label)


def validate_proof_instances(result: dict[str, Any], proof: Path) -> dict[str, Any]:
    document = load_json(proof, "verifier-facing proof")
    instances = document.get("instances")
    if not isinstance(instances, list) or not instances or any(not isinstance(item, list) for item in instances):
        abort(f"{proof} does not expose a native instances array")
    count = sum(len(item) for item in instances)
    if count <= 0:
        abort(f"{proof} has no native public instances")
    expected = result["proof_instances"]
    if not isinstance(expected, dict):
        abort("result proof_instances must be an object")
    required = {
        "proof_field",
        "encoding",
        "sha256",
        "count",
        "public_statement_sha256",
        "public_output_sha256",
        "native_binding",
    }
    if not required.issubset(expected):
        abort(f"result proof_instances is missing fields: {sorted(required - set(expected))}")
    digest = canonical_json_digest(instances)
    if expected["proof_field"] != "instances" or expected["encoding"] != "canonical-json":
        abort("result proof_instances encoding is not canonical native instances")
    if expected["sha256"] != digest or expected["count"] != count:
        abort("proof instance digest/count mismatch")
    if expected["public_statement_sha256"] != result["public_statement_sha256"]:
        abort("proof instances are not bound to the external public statement")
    if expected["public_output_sha256"] != result["public_output_sha256"]:
        abort("proof instances are not bound to external public output Y")
    if expected["native_binding"] is not True:
        abort("proof instances lack native public-instance binding")
    return expected


def validate_cell(path: Path, *, expected_session_id: str | None = None) -> dict[str, Any]:
    result = load_json(path, f"result cell {path}")
    formal_gpu_identity_fields = {
        "selected_cuda_device", "build_receipt_sha256", "build_environment_sha256",
        "gpu_prover_record_sha256", "independent_verifier_log_sha256",
    }
    unknown = set(result) - REQUIRED_RESULT_KEYS - {"execution_mode", "phase_backends", "native_backend", "formal_session_id"} - formal_gpu_identity_fields
    if unknown:
        abort(f"{path} has unknown result fields: {sorted(unknown)}")
    missing = REQUIRED_RESULT_KEYS - set(result)
    if missing:
        abort(f"{path} is missing result fields: {sorted(missing)}")
    if result["schema_version"] != 1 or result["status"] != "VERIFY_OK":
        abort(f"{path} is not a verified native result")
    dataset = result["dataset"]
    backend = result["backend"]
    if dataset not in DATASETS or backend not in (*BACKENDS, FORMAL_BACKEND):
        abort(f"{path} has an unknown dataset/backend")
    native_backend = NATIVE_FORMAL_BACKEND if backend == FORMAL_BACKEND else backend
    if backend == FORMAL_BACKEND:
        session_id = result.get("formal_session_id")
        if not isinstance(session_id, str) or not FORMAL_SESSION_ID.fullmatch(session_id):
            abort(f"{path} has no valid formal session identity")
        if expected_session_id is not None and session_id != expected_session_id:
            abort(f"{path} belongs to a different formal session")
        if result.get("native_backend") != native_backend or result.get("execution_mode") != FORMAL_BACKEND:
            abort(f"{path} does not identify the formal hybrid execution/native engine")
        if canonical_json_digest(result.get("phase_backends")) != canonical_json_digest(FORMAL_PHASE_BACKENDS):
            abort(f"{path} has an invalid hybrid phase device schedule")
        if path.parent.name != FORMAL_BACKEND or path.parent.parent.name != dataset:
            abort(f"{path} is outside its canonical dataset/hybrid result directory")
        for field in formal_gpu_identity_fields:
            if field not in result:
                abort(f"{path} is missing current-session GPU identity field {field}")
        selected_device = result["selected_cuda_device"]
        device_keys = {"index", "uuid", "name", "driver_version", "compute_capability", "logical_device_id"}
        if not isinstance(selected_device, dict) or set(selected_device) != device_keys:
            abort(f"{path} selected CUDA device identity has missing or unknown fields")
        if (
            isinstance(selected_device["index"], bool) or not isinstance(selected_device["index"], int) or selected_device["index"] < 0
            or isinstance(selected_device["logical_device_id"], bool) or selected_device["logical_device_id"] != 0
            or any(not isinstance(selected_device[key], str) or not selected_device[key] for key in ("uuid", "name", "driver_version", "compute_capability"))
        ):
            abort(f"{path} selected CUDA device identity contains invalid values")
        current_gpu = validate_gpu_environment()
        if current_gpu["selected_cuda_device"] != selected_device:
            abort(f"{path} selected CUDA device does not match the live formal execution device")
        for field in ("build_receipt_sha256", "build_environment_sha256", "gpu_prover_record_sha256", "independent_verifier_log_sha256"):
            require_digest(result[field], f"{path} {field}")
        receipt_path = ROOT / "artifacts/build/gpu/build_receipt.json"
        require_file(receipt_path, "current GPU build receipt")
        if sha256(receipt_path) != result["build_receipt_sha256"]:
            abort(f"{path} build receipt does not belong to the current formal session")
        receipt = load_json(receipt_path, "current GPU build receipt")
        environment = receipt.get("build_environment")
        if (
            receipt.get("selected_cuda_device") != selected_device
            or receipt.get("features") != list(FORMAL_GPU_FEATURES)
            or receipt.get("build_environment_sha256") != result["build_environment_sha256"]
            or not isinstance(environment, dict)
            or environment.get("selected_cuda_device") != selected_device
            or environment.get("features") != list(FORMAL_GPU_FEATURES)
            or environment.get("build_environment_sha256") != result["build_environment_sha256"]
        ):
            abort(f"{path} GPU result/build identity does not match")
        environment_payload = dict(environment)
        recorded_environment_digest = environment_payload.pop("build_environment_sha256", None)
        if canonical_json_digest(environment_payload) != recorded_environment_digest:
            abort(f"{path} GPU build environment digest is invalid")
        verifier_log = require_file(path.parent / "logs/verify.log", "independent verifier log")
        if sha256(verifier_log) != result["independent_verifier_log_sha256"]:
            abort(f"{path} independent verifier log identity is stale")
    elif "execution_mode" in result or "phase_backends" in result:
        expected_mode = "mixed" if backend == "gpu" else "cpu"
        legacy_phases = {"prepare_compile": "cpu", "setup": expected_mode, "public_statement": "cpu", "witness_generation": "cpu", "model_commitment": "cpu", "prove": expected_mode, "verify": "cpu"}
        cpu_phases = {phase: {"cpu": True, "gpu": False} for phase in FORMAL_PHASE_BACKENDS}
        allowed_phases = (legacy_phases, cpu_phases) if backend == "cpu" else (legacy_phases,)
        if result.get("execution_mode") != expected_mode or result.get("phase_backends") not in allowed_phases:
            abort(f"{path} has an invalid native phase device schedule")
    if "native_backend" in result and result["native_backend"] != native_backend:
        abort(f"{path} has inconsistent native engine identity")
    if result["proof_scope"] != CANONICAL_PROOF_SCOPE:
        abort(f"{path} is not full_gat")
    if result["privacy_policy"] != CANONICAL_PRIVACY:
        abort(f"{path} has the wrong privacy policy")
    if result["public_statement_mode"] != CANONICAL_STATEMENT_MODE or result["output_mode"] != CANONICAL_OUTPUT_MODE:
        abort(f"{path} has the wrong public statement/output mode")
    if result["relation_id"] != CANONICAL_RELATION:
        abort(f"{path} has the wrong relation id")
    if not isinstance(result["quant_cfg_id"], str) or not result["quant_cfg_id"]:
        abort(f"{path} has no quantization identity")
    if not isinstance(result["static_tables_id"], str) or not result["static_tables_id"]:
        abort(f"{path} has no static-table identity")
    if result["verifier_reads_witness"] is not False or result["proof_contains_private_witness"] is not False:
        abort(f"{path} violates the verifier/witness boundary")
    if result["native_verify_result"] is not True or result["native_public_instance_binding"] is not True:
        abort(f"{path} has no native verify/public-instance attestation")
    circuit_ref = require_circuit_ref(result["circuit_ref"])
    for key in (
        "public_statement_sha256", "public_output_sha256", "srs_sha256", "settings_sha256", "compiled_circuit_sha256", "vk_sha256", "pk_sha256", "proof_sha256",
        "binary_sha256", "source_tree_sha256", "resource_manifest_sha256", "artifact_manifest_sha256", "core_run_manifest_sha256"
    ):
        require_digest(result[key], f"{path}:{key}")
    if isinstance(result["proof_size_bytes"], bool) or not isinstance(result["proof_size_bytes"], int) or result["proof_size_bytes"] <= 0:
        abort(f"{path} has no positive proof size")
    require_positive_timings(result["phase_timings_ms"], f"{path}:phase_timings_ms")
    for field, digest_field in PATH_DIGESTS.items():
        artifact = path_from_result(result[field], field)
        if sha256(artifact) != result[digest_field]:
            abort(f"{path} digest mismatch for {field}")
    proof = path_from_result(result["proof_path"], "proof")
    if proof.stat().st_size != result["proof_size_bytes"]:
        abort(f"{path} proof_size_bytes does not match proof file")
    statement = path_from_result(result["public_statement_path"], "public statement")
    public_output = path_from_result(result["public_output_path"], "public output")
    statement_document = validate_public_statement(
        statement,
        dataset=dataset,
        circuit_ref=circuit_ref,
        public_output_sha256=result["public_output_sha256"],
        quant_cfg_id=result["quant_cfg_id"],
        static_tables_id=result["static_tables_id"],
    )
    if public_output_values(public_output, dataset=dataset) != statement_document["public_ezkl"]["y"]:
        abort(f"{path} public output Y disagrees element-by-element with statement")
    metric = require_exact_keys(result["commitment_metric"], {"operation", "included_in", "path", "sha256"}, "result commitment metric")
    commitment_path = path_from_result(metric["path"], "native commitment metrics")
    if commitment_path != path.parent / "commitment_metrics.json":
        abort(f"{path} commitment metrics do not belong to this formal cell")
    require_digest(metric["sha256"], "native commitment metrics digest")
    if sha256(commitment_path) != metric["sha256"]:
        abort(f"{path} native commitment metrics digest mismatch")
    commitment = validate_commitment_metrics(commitment_path, dataset=dataset, model_hash=statement_document["public_ezkl"]["model_hash"])
    if metric["operation"] != commitment["operation"] or metric["included_in"] != commitment["included_in"]:
        abort(f"{path} native commitment measurement scope mismatch")
    duration = result["commitment_time_ms"]
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0 or duration != commitment["duration_ms"]:
        abort(f"{path} commitment_time_ms does not match the native measured duration")
    validate_proof_instances(result, proof)
    artifact_manifest = path_from_result(result["artifact_manifest_path"], "artifact identity manifest")
    artifact = validate_artifact_identity(
        artifact_manifest,
        dataset=dataset,
        backend=native_backend,
        binary=path_from_result(result["binary_path"], "binary"),
        srs=path_from_result(result["srs_path"], "SRS"),
        settings=path_from_result(result["settings_path"], "settings"),
        compiled=path_from_result(result["compiled_circuit_path"], "compiled circuit"),
        vk=path_from_result(result["vk_path"], "verification key"),
        pk=path_from_result(result["pk_path"], "proving key"),
    )
    if artifact["statement_sha256"] != result["public_statement_sha256"]:
        abort(f"{path} artifact statement identity is not externally bound")
    artifact_records = {record["path"]: record for record in artifact["files"]}
    for name, digest in (
        ("statement.json", result["public_statement_sha256"]),
        ("public_output.json", result["public_output_sha256"]),
    ):
        if name not in artifact_records or artifact_records[name]["sha256"] != digest:
            abort(f"{path} artifact file {name} is not bound to the public boundary")
    core_path = path.parent / "core_run_manifest.json"
    core = load_json(core_path, "core run manifest")
    require_exact_keys(
        core,
        {
            "schema_version",
            "relation_id",
            "dataset",
            "backend",
            "proof_scope",
            "privacy_policy",
            "public_statement_mode",
            "output_mode",
            "native_verify_result",
            "native_public_instance_binding",
            "verifier_reads_witness",
            "proof_contains_private_witness",
            "public_statement_sha256",
            "public_output_sha256",
            "proof_instances_sha256",
            "proof_instance_count",
            "circuit_ref",
        },
        "core run manifest",
    )
    if (
        core["schema_version"] != 1 or core["relation_id"] != CANONICAL_RELATION or core["dataset"] != dataset or core["backend"] != native_backend or
        core["proof_scope"] != CANONICAL_PROOF_SCOPE or core["privacy_policy"] != CANONICAL_PRIVACY or
        core["public_statement_mode"] != CANONICAL_STATEMENT_MODE or core["output_mode"] != CANONICAL_OUTPUT_MODE or core["native_verify_result"] is not True or
        core["native_public_instance_binding"] is not True or core["verifier_reads_witness"] is not False or
        core["proof_contains_private_witness"] is not False or core["public_statement_sha256"] != result["public_statement_sha256"] or
        core["public_output_sha256"] != result["public_output_sha256"] or core["circuit_ref"] != artifact["circuit_ref"]
    ):
        abort(f"{path} core run manifest is not a successful native external binding")
    require_digest(core["public_statement_sha256"], "core public statement digest")
    require_digest(core["public_output_sha256"], "core public output digest")
    require_digest(core["proof_instances_sha256"], "core proof instance digest")
    if not isinstance(core["proof_instance_count"], int) or core["proof_instance_count"] <= 0:
        abort(f"{path} core proof instance count is invalid")
    if (core["proof_instances_sha256"] != result["proof_instances"]["sha256"] or core["proof_instance_count"] != result["proof_instances"]["count"]):
        abort(f"{path} core proof-instance attestation does not match the native proof")
    if sha256(core_path) != result["core_run_manifest_sha256"]:
        abort(f"{path} core run manifest digest mismatch")
    if require_circuit_ref(artifact["circuit_ref"], "artifact circuit_ref") != circuit_ref:
        abort(f"{path} circuit_ref does not match artifact identity")
    if result["srs_provenance"] != artifact["srs_provenance"]:
        abort(f"{path} SRS provenance does not match artifact identity")
    validate_gpu_evidence_from_result(result)
    if native_backend == "gpu":
        record = load_json(proof.parent / "gpu_prover_record.json", "native GPU prover record")
        require_exact_keys(record, {"proof_sha256", "msm_calls", "ntt_calls", "synchronized_calls", "device_id", "evidence"}, "native GPU prover record")
        counts = [record[key] for key in ("msm_calls", "ntt_calls", "synchronized_calls", "device_id")]
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts):
            abort("native GPU prover record has invalid device/call counters")
        calls = record["msm_calls"] + record["ntt_calls"]
        if record["proof_sha256"] != result["proof_sha256"] or calls <= 0 or record["synchronized_calls"] != calls or record["evidence"] != result["gpu_evidence"] or result["gpu_evidence"]["gpu_kernel_invocation_count"] != calls:
            abort("native GPU prover record is not synchronized and bound to the verified proof")
        if backend == FORMAL_BACKEND:
            if record["device_id"] != result["selected_cuda_device"]["logical_device_id"]:
                abort("native proof logical device ordinal does not match the selected physical CUDA UUID")
            if sha256(proof.parent / "gpu_prover_record.json") != result["gpu_prover_record_sha256"]:
                abort("native GPU prover record identity is stale")

    if result["source_tree_sha256"] != source_tree_hash() or result["resource_manifest_sha256"] != resource_manifest_hash():
        abort(f"{path} source/resource identity is stale")
    result["_result_path"] = path.relative_to(ROOT).as_posix()
    return result


def validate_gpu_evidence_from_result(result: dict[str, Any]) -> None:
    validate_gpu_evidence_document(result["gpu_evidence"], backend=NATIVE_FORMAL_BACKEND if result["backend"] == FORMAL_BACKEND else result["backend"])


def check_pair_consistency(cells: list[dict[str, Any]]) -> None:
    if {cell["backend"] for cell in cells} != set(BACKENDS):
        return
    by_dataset = {(cell["dataset"], cell["backend"]): cell for cell in cells}
    for dataset in DATASETS:
        cpu = by_dataset[(dataset, "cpu")]
        gpu = by_dataset[(dataset, "gpu")]
        for field in PAIR_FIELDS:
            if cpu[field] != gpu[field]:
                abort(f"CPU/GPU relation mismatch for {dataset}: field={field}")


def main(args: argparse.Namespace) -> int:
    selected_datasets = tuple(args.datasets)
    if len(set(selected_datasets)) != len(selected_datasets):
        abort("--datasets must not contain duplicates")
    selected_backends = tuple(args.backends)
    if len(set(selected_backends)) != len(selected_backends):
        abort("--backends must not contain duplicates")
    if FORMAL_BACKEND in selected_backends and (selected_backends != (FORMAL_BACKEND,) or selected_datasets != DATASETS):
        abort("formal collection requires exactly the five canonical hybrid results")
    if FORMAL_BACKEND in selected_backends:
        if not isinstance(args.session_id, str) or not FORMAL_SESSION_ID.fullmatch(args.session_id):
            abort("formal collection requires the active --session-id")
    elif args.session_id is not None:
        abort("--session-id is only valid for formal hybrid collection")
    runs_root = local_path(args.runs_root, "runs-root")
    output = local_path(args.output, "collector output")
    reject_forbidden_path(output, "collector output")
    if output.is_symlink():
        abort(f"collector output must not be a symlink: {output}")
    cells: list[dict[str, Any]] = []
    for dataset in selected_datasets:
        for backend in selected_backends:
            result_path = runs_root / dataset / backend / "result.json"
            reject_forbidden_path(result_path, f"{dataset}/{backend} result")
            cell = validate_cell(
                result_path,
                expected_session_id=args.session_id if FORMAL_BACKEND in selected_backends else None,
            )
            if (cell["dataset"], cell["backend"]) != (dataset, backend):
                abort(f"result identity does not match its collection path: {result_path}")
            cells.append(cell)
    if len(cells) != len(selected_datasets) * len(selected_backends):
        abort("collector did not obtain the requested dataset/backend cells")
    check_pair_consistency(cells)
    output.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "schema_version": 1,
        "baseline": "ezkl",
        "collection_status": "COLLECTED_PENDING_FINAL_VALIDATION",
        "relation_id": CANONICAL_RELATION,
        "datasets": list(selected_datasets),
        "backends": list(selected_backends),
        "cell_count": len(cells),
        "source_tree_sha256": source_tree_hash(),
        "resource_manifest_sha256": resource_manifest_hash(),
        "pair_consistency_checked": set(selected_backends) == set(BACKENDS),
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "cells": cells,
    }
    if FORMAL_BACKEND in selected_backends:
        document["formal_session_id"] = args.session_id
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(
        f"collected {len(cells)} verified result cells ({', '.join(selected_backends)}) into "
        f"{output.relative_to(ROOT)}; final validation remains required"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(parser().parse_args()))
    except (ContractError, OSError, json.JSONDecodeError) as exc:
        raise SystemExit(print_error(exc))
