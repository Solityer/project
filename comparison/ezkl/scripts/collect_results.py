#!/usr/bin/env python3
"""Collect the five formal EZKL/Halo2 results or explicitly local-only checks.

Formal collection has one result per canonical dataset and no backend axis.
Explicit local-backend collection exists only for the tiny harness.
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
    FORMAL_GPU_FEATURES,
    FORMAL_GPU_EXECUTION_POLICY,
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
    validate_canonical_public_inputs,
    validate_config,
    validate_native_proof_metrics,
    validate_native_pk_read_timing,
    validate_native_verify_metrics,
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
    "prove_time_ms",
    "verify_time_ms",
    "commitment_metric",
    "verify_metric",
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
    result.add_argument("--session-id", help="active formal session; required for five-dataset formal collection")
    result.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASETS,
        default=list(DATASETS),
        help="dataset axis to collect; default requires the strict five-dataset matrix",
    )
    result.add_argument("--local-backends", nargs="+", choices=BACKENDS, help="tiny/local-only results; cannot produce a formal result matrix")
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
    formal_pk_timing_fields = {
        "native_pk_read_timing_sha256",
        "prove_command_wall_ms",
        "prove_wall_scope",
        "prove_command_residual_excluding_pk_artifact_read_and_sigma_prepare_ms",
    }
    unknown = set(result) - REQUIRED_RESULT_KEYS - {"backend", "execution_mode", "phase_backends", "native_backend", "formal_session_id"} - formal_gpu_identity_fields - formal_pk_timing_fields
    if unknown:
        abort(f"{path} has unknown result fields: {sorted(unknown)}")
    missing = REQUIRED_RESULT_KEYS - set(result)
    if missing:
        abort(f"{path} is missing result fields: {sorted(missing)}")
    if result["schema_version"] != 1 or result["status"] != "VERIFY_OK":
        abort(f"{path} is not a verified native result")
    dataset = result["dataset"]
    formal = "formal_session_id" in result
    if formal and not formal_pk_timing_fields.issubset(result):
        abort(f"{path} formal result omits its authenticated PK materialization timing boundary")
    secure_config = None
    backend = result.get("backend")
    if dataset not in DATASETS:
        abort(f"{path} has an unknown dataset")
    if formal:
        if not formal_pk_timing_fields.issubset(result):
            abort(f"{path} formal result omits its authenticated compact PK restore timing")
        if backend is not None or "execution_mode" in result or "phase_backends" in result:
            abort(f"{path} exposes a backend execution axis in the formal result")
        native_backend = result.get("native_backend")
        if native_backend != NATIVE_FORMAL_BACKEND:
            abort(f"{path} does not identify the authenticated native formal engine")
        session_id = result.get("formal_session_id")
        if not isinstance(session_id, str) or not FORMAL_SESSION_ID.fullmatch(session_id):
            abort(f"{path} has no valid formal session identity")
        if expected_session_id is None:
            abort(f"{path} is a formal result but no active session was supplied")
        if expected_session_id is not None and session_id != expected_session_id:
            abort(f"{path} belongs to a different formal session")
        if path.parent.name != dataset:
            abort(f"{path} is outside its canonical dataset result directory")
        secure_config = validate_config(ROOT / "resources/canonical/configs" / f"{dataset}_secure.cfg", dataset)
        if result["quant_cfg_id"] != secure_config["quant_cfg_id"] or result["static_tables_id"] != secure_config["static_table_id"]:
            abort(f"{path} quantization/static-table identities differ from the canonical config")
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
    else:
        if formal_pk_timing_fields.intersection(result):
            abort(f"{path} nonformal result contains formal compact PK restore timing fields")
        if expected_session_id is not None:
            abort(f"{path} is not bound to the active formal session")
        if backend not in BACKENDS:
            abort(f"{path} is not an explicitly local native result")
        native_backend = backend
    if not formal and ("execution_mode" in result or "phase_backends" in result):
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
    if formal:
        production_timing_fields = ("public_statement_production", "final_statement_binding")
        phase_timings = result["phase_timings_ms"]
        if not isinstance(phase_timings, dict) or any(
            field not in phase_timings
            or isinstance(phase_timings[field], bool)
            or not isinstance(phase_timings[field], (int, float))
            or not math.isfinite(phase_timings[field])
            or phase_timings[field] <= 0
            for field in production_timing_fields
        ):
            abort(f"{path} formal result omits its authenticated producer/final stage wall timings")
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
        quant_cfg_id=secure_config["quant_cfg_id"] if secure_config is not None else result["quant_cfg_id"],
        static_tables_id=secure_config["static_table_id"] if secure_config is not None else result["static_tables_id"],
        degree_bound_id=secure_config["degree_bound_id"] if secure_config is not None else None,
        degree_bound_digest=secure_config["degree_bound_digest"] if secure_config is not None else None,
    )
    if public_output_values(public_output, dataset=dataset) != statement_document["public_ezkl"]["y"]:
        abort(f"{path} public output Y disagrees element-by-element with statement")
    metric = require_exact_keys(result["commitment_metric"], {"operation", "included_in", "path", "sha256"}, "result commitment metric")
    commitment_path = path_from_result(metric["path"], "native proof metrics")
    if commitment_path != path.parent / "native_proof_metrics.json":
        abort(f"{path} native proof metrics do not belong to this formal cell")
    require_digest(metric["sha256"], "native proof metrics digest")
    if sha256(commitment_path) != metric["sha256"]:
        abort(f"{path} native proof metrics digest mismatch")
    proof_metrics = validate_native_proof_metrics(commitment_path, proof_sha256=result["proof_sha256"])
    if metric["operation"] != proof_metrics["operation"] or metric["included_in"] != proof_metrics["included_in"]:
        abort(f"{path} native commitment measurement scope mismatch")
    duration = result["commitment_time_ms"]
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0 or duration != proof_metrics["commitment_time_ms"]:
        abort(f"{path} commitment_time_ms does not match the native measured duration")
    if result["prove_time_ms"] != proof_metrics["prove_time_ms"]:
        abort(f"{path} prove_time_ms does not match native proof timing")
    if not math.isclose(
        float(result["phase_timings_ms"].get("prove", float("nan"))),
        float(proof_metrics["prove_time_ms"]),
        rel_tol=0.0,
        abs_tol=0.000001,
    ):
        abort(f"{path} phase prove timing does not match the native composed proof metric")
    verify_metric = require_exact_keys(result["verify_metric"], {"operation", "included_in", "path", "sha256"}, "result native verification metric")
    verify_metrics_path = path_from_result(verify_metric["path"], "native verification metrics")
    if verify_metrics_path != path.parent / "native_verify_metrics.json":
        abort(f"{path} native verification metrics do not belong to this formal cell")
    require_digest(verify_metric["sha256"], "native verification metrics digest")
    if sha256(verify_metrics_path) != verify_metric["sha256"]:
        abort(f"{path} native verification metrics digest mismatch")
    native_verify_metrics = validate_native_verify_metrics(verify_metrics_path, proof_sha256=result["proof_sha256"])
    if verify_metric["operation"] != native_verify_metrics["operation"] or verify_metric["included_in"] != native_verify_metrics["included_in"]:
        abort(f"{path} native verification measurement scope mismatch")
    if result["verify_time_ms"] != native_verify_metrics["verify_time_ms"]:
        abort(f"{path} verify_time_ms does not match native verification timing")
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
    if secure_config is not None:
        validate_canonical_public_inputs(
            statement_document,
            dataset=dataset,
            secure_config=secure_config,
            setup_context_path=artifact_manifest.parent / "producer_context.json",
            expected_resource_manifest_sha256=result["resource_manifest_sha256"],
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
    if formal:
        native_receipt_names = ("setup_receipt.json", "native_keygen_budget.json", "native_keygen_receipt.json", "native_prove_budget.json")
        if any(name not in artifact_records for name in native_receipt_names):
            abort(f"{path} artifact manifest omits native setup/prove resource receipts")
        setup_receipt = load_json(artifact_manifest.parent / native_receipt_names[0], "formal setup resource receipt")
        keygen_budget = load_json(artifact_manifest.parent / native_receipt_names[1], "native keygen budget")
        keygen_receipt = load_json(artifact_manifest.parent / native_receipt_names[2], "native keygen census receipt")
        prove_budget = load_json(artifact_manifest.parent / native_receipt_names[3], "native proving-key read budget")
        if any(not isinstance(value, dict) for value in (setup_receipt, keygen_budget, keygen_receipt, prove_budget)):
            abort(f"{path} native resource receipts must be JSON objects")
        receipt_digests = {name: artifact_records[name]["sha256"] for name in native_receipt_names}
        pk_timing_path = artifact_manifest.parent / "native_pk_read_timing.json"
        timing_file = artifact_records.get("native_pk_read_timing.json")
        if timing_file is None or timing_file.get("sha256") != result["native_pk_read_timing_sha256"] or sha256(pk_timing_path) != result["native_pk_read_timing_sha256"]:
            abort(f"{path} artifact manifest does not bind the native PK restore timing")
        pk_timing = validate_native_pk_read_timing(
            pk_timing_path,
            session_id=session_id,
            dataset=dataset,
            native_prove_budget_sha256=receipt_digests["native_prove_budget.json"],
            pk_sha256=result["pk_sha256"],
        )
        if any(
            pk_timing.get(name) != value
            for name, value in (
                ("admission_sha256", prove_budget.get("admission_sha256")),
                ("resource_manifest_sha256", result["resource_manifest_sha256"]),
                ("setup_receipt_sha256", receipt_digests["setup_receipt.json"]),
                ("native_keygen_budget_sha256", receipt_digests["native_keygen_budget.json"]),
                ("native_keygen_receipt_sha256", receipt_digests["native_keygen_receipt.json"]),
                ("build_receipt_sha256", result["build_receipt_sha256"]),
                ("build_environment_sha256", result["build_environment_sha256"]),
                ("binary_sha256", result["binary_sha256"]),
                ("settings_sha256", result["settings_sha256"]),
                ("compiled_circuit_sha256", result["compiled_circuit_sha256"]),
                ("vk_sha256", result["vk_sha256"]),
            )
        ):
            abort(f"{path} native PK restore timing disagrees with authenticated setup/build identities")
        phase_timings = result["phase_timings_ms"]
        if (
            not math.isclose(phase_timings.get("pk_artifact_read_validate", float("nan")), pk_timing["artifact_read_validate_ms"], rel_tol=0.0, abs_tol=0.000001)
            or not math.isclose(phase_timings.get("pk_materialize", float("nan")), pk_timing["pk_materialize_ms"], rel_tol=0.0, abs_tol=0.000001)
            or not math.isclose(phase_timings.get("setup_pk_sigma_deferred_source_prepare", float("nan")), pk_timing["sigma_deferred_source_prepare_wall_ms"], rel_tol=0.0, abs_tol=0.000001)
            or not math.isclose(proof_metrics["pk_materialize_time_ms"], pk_timing["pk_materialize_ms"], rel_tol=0.0, abs_tol=0.000001)
        ):
            abort(f"{path} phase timing does not preserve the measured PK restore subphases")
        command_wall = result["prove_command_wall_ms"]
        if isinstance(command_wall, bool) or not isinstance(command_wall, (int, float)) or not math.isfinite(command_wall) or command_wall <= 0:
            abort(f"{path} native prove command wall duration is invalid")
        residual = command_wall - pk_timing["artifact_read_validate_ms"] - pk_timing["sigma_deferred_source_prepare_wall_ms"]
        recorded_residual = result["prove_command_residual_excluding_pk_artifact_read_and_sigma_prepare_ms"]
        if (
            isinstance(recorded_residual, bool)
            or not isinstance(recorded_residual, (int, float))
            or not math.isfinite(recorded_residual)
            or not math.isclose(recorded_residual, residual, rel_tol=0.0, abs_tol=0.001)
        ):
            abort(f"{path} outer prove residual does not preserve its mutually exclusive restore phases")
        if residual > 0.0:
            if result["prove_wall_scope"] != "outer_command_residual_excludes_artifact_read_and_sigma_prepare_includes_pk_materialize":
                abort(f"{path} outer prove wall residual does not preserve PK materialization")
        elif residual >= -0.01:
            if result["prove_wall_scope"] != "outer_command_subphase_overlap_within_0_01ms_includes_pk_materialize":
                abort(f"{path} sub-resolution PK restore overlap is not transparently recorded")
        else:
            abort(f"{path} PK restore stages exceed its prove command wall time")
        expected_setup = {
            "schema": "ezkl.setup-reuse-receipt.v1",
            "session_id": session_id,
            "dataset": dataset,
            "admission_sha256": prove_budget.get("admission_sha256"),
            "native_keygen_receipt_sha256": receipt_digests["native_keygen_receipt.json"],
        }
        if any(setup_receipt.get(name) != value for name, value in expected_setup.items()):
            abort(f"{path} setup receipt identity does not match its formal cell")
        setup_files = setup_receipt.get("setup_files")
        if not isinstance(setup_files, dict) or any(
            setup_files.get(name) != digest
            for name, digest in (
                ("settings.json", result["settings_sha256"]),
                ("compiled.circuit", result["compiled_circuit_sha256"]),
                ("vk.key", result["vk_sha256"]),
                ("pk.key", result["pk_sha256"]),
                ("native_keygen_budget.json", receipt_digests["native_keygen_budget.json"]),
                ("native_keygen_receipt.json", receipt_digests["native_keygen_receipt.json"]),
            )
        ):
            abort(f"{path} setup receipt does not bind its exact key/circuit artifacts")
        expected_keygen = {
            "schema": "ezkl.native-keygen-receipt.v2",
            "session_id": session_id,
            "dataset": dataset,
            "budget_receipt_sha256": receipt_digests["native_keygen_budget.json"],
            "settings_sha256": result["settings_sha256"],
            "compiled_circuit_sha256": result["compiled_circuit_sha256"],
            "vk_sha256": result["vk_sha256"],
            "pk_sha256": result["pk_sha256"],
        }
        if any(keygen_receipt.get(name) != value for name, value in expected_keygen.items()):
            abort(f"{path} native keygen census receipt does not match its formal cell")
        relation_identity = keygen_receipt.get("relation_identity")
        if not isinstance(relation_identity, dict) or relation_identity.get("dataset") != dataset or relation_identity.get("resource_manifest_sha256") != result["resource_manifest_sha256"]:
            abort(f"{path} native keygen relation identity does not match the canonical result")
        expected_native_budget = {
            "schema": "ezkl.native-keygen-budget.v1",
            "session_id": session_id,
            "dataset": dataset,
            "admission_sha256": setup_receipt.get("admission_sha256"),
            "settings_sha256": result["settings_sha256"],
            "binary_sha256": result["binary_sha256"],
            "resource_manifest_sha256": result["resource_manifest_sha256"],
            "build_receipt_sha256": result["build_receipt_sha256"],
            "build_environment_sha256": result["build_environment_sha256"],
        }
        if any(keygen_budget.get(name) != value for name, value in expected_native_budget.items()):
            abort(f"{path} native keygen budget does not match its formal build/resource identity")
        native_census = keygen_receipt.get("native_census")
        if not isinstance(native_census, dict) or not isinstance(native_census.get("preflight"), dict):
            abort(f"{path} native keygen receipt has no exact synthesis census")
        preflight = native_census["preflight"]
        stage_caps = keygen_receipt.get("stage_budget_caps")
        if not isinstance(stage_caps, list) or len(stage_caps) != 2:
            abort(f"{path} native keygen receipt has no VK/PK stage caps")
        caps: dict[str, dict[str, int]] = {}
        for index, stage in enumerate(("vk", "pk")):
            value = stage_caps[index]
            if not isinstance(value, dict) or value.get("stage") != stage:
                abort(f"{path} native keygen stage caps are not ordered VK then PK")
            selector_cap = value.get("max_selector_activation_bytes")
            permutation_cap = value.get("max_permutation_backing_bytes")
            if any(isinstance(cap, bool) or not isinstance(cap, int) or cap < 0 or cap >= 1 << 64 for cap in (selector_cap, permutation_cap)):
                abort(f"{path} native {stage.upper()} allocation cap is invalid")
            host_snapshot = keygen_budget.get("host_available_bytes")
            disk_snapshot = keygen_budget.get("native_permutation_backing_budget_bytes")
            selector_initial = preflight.get("selector_activation_initial_backing_bytes")
            source_components = (
                preflight.get("permutation_data_virtual_bytes"),
                preflight.get("permutation_directory_virtual_bytes"),
                keygen_budget.get("permutation_successor_backing_file_upper_bytes"),
            )
            if (
                any(isinstance(bound, bool) or not isinstance(bound, int) or bound < 0 or bound > sys.maxsize for bound in (host_snapshot, disk_snapshot, selector_initial, *source_components))
            ):
                abort(f"{path} native {stage.upper()} resource ceiling is malformed")
            source_backing_upper = sum(source_components)
            if (
                source_backing_upper > sys.maxsize
                or selector_cap < selector_initial
                or selector_cap > host_snapshot
                or permutation_cap > min(disk_snapshot, source_backing_upper)
            ):
                abort(f"{path} native {stage.upper()} allocation cap exceeds its authenticated budget envelope")
            caps[stage] = {"selector": selector_cap, "permutation": permutation_cap}
        selector_peak = native_census.get("selector_activation_storage_peak_bytes")
        permutation_values = [native_census.get(name) for name in ("permutation_data_allocated_bytes", "permutation_directory_allocated_bytes", "permutation_successor_edges")]
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > sys.maxsize for value in permutation_values):
            abort(f"{path} native permutation census byte counts are invalid")
        successor_logical_bytes = 16 * permutation_values[2]
        successor_reserved_bytes = ((successor_logical_bytes + 4095) // 4096) * 4096
        target_bitmap_virtual_bytes = (permutation_values[2] + 7) // 8
        target_bitmap_reserved_bytes = ((target_bitmap_virtual_bytes + 4095) // 4096) * 4096
        exact_pk_read_backing_bytes = successor_reserved_bytes + target_bitmap_reserved_bytes
        permutation_backing = permutation_values[0] + permutation_values[1] + successor_reserved_bytes
        if (
            successor_reserved_bytes > sys.maxsize
            or target_bitmap_reserved_bytes > sys.maxsize
            or exact_pk_read_backing_bytes > sys.maxsize
            or isinstance(selector_peak, bool)
            or not isinstance(selector_peak, int)
            or selector_peak < 0
            or selector_peak > sys.maxsize
            or selector_peak > min(caps["vk"]["selector"], caps["pk"]["selector"])
            or permutation_backing < 0
            or permutation_backing > sys.maxsize
            or permutation_backing > min(caps["vk"]["permutation"], caps["pk"]["permutation"])
        ):
            abort(f"{path} native keygen census exceeds its VK/PK resource caps")
        expected_prove = {
            "schema": "ezkl.native-prove-budget.v1",
            "session_id": session_id,
            "dataset": dataset,
            "admission_sha256": setup_receipt.get("admission_sha256"),
            "setup_receipt_sha256": receipt_digests["setup_receipt.json"],
            "native_keygen_budget_sha256": receipt_digests["native_keygen_budget.json"],
            "native_keygen_receipt_sha256": receipt_digests["native_keygen_receipt.json"],
            "binary_sha256": result["binary_sha256"],
            "settings_sha256": result["settings_sha256"],
            "compiled_circuit_sha256": result["compiled_circuit_sha256"],
            "vk_sha256": result["vk_sha256"],
            "pk_sha256": result["pk_sha256"],
            "resource_manifest_sha256": result["resource_manifest_sha256"],
            "build_receipt_sha256": result["build_receipt_sha256"],
            "build_environment_sha256": result["build_environment_sha256"],
        }
        if any(prove_budget.get(name) != value for name, value in expected_prove.items()):
            abort(f"{path} native prove budget does not bind its setup, keygen, and binary identities")
        for name in (
            "disk_available_bytes",
            "disk_required_known_bytes",
            "permutation_grand_product_set_count",
            "permutation_grand_product_bank_mapped_bytes",
            "permutation_grand_product_window_mapped_bytes",
            "permutation_grand_product_commit_workspace_bytes",
            "proof_permutation_grand_product_commit_host_peak_bytes",
            "native_pk_read_backing_required_bytes",
            "max_total_backing_bytes",
            "host_available_bytes",
            "host_required_known_bytes",
            "native_pk_read_host_scratch_upper_bytes",
            "native_pk_read_successor_cache_upper_bytes",
            "native_pk_read_target_bitmap_fixed_cache_upper_bytes",
            "native_pk_read_target_bitmap_allocation_bitmap_upper_bytes",
            "native_pk_read_validation_buffers_upper_bytes",
            "native_pk_read_stream_buffer_upper_bytes",
        ):
            value = prove_budget.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > sys.maxsize:
                abort(f"{path} native prove budget has invalid {name}")
        target_bitmap_page_count = (target_bitmap_reserved_bytes // 4096)
        target_bitmap_allocation_upper = (target_bitmap_page_count + 7) // 8
        expected_read_host_scratch = (
            prove_budget["native_pk_read_successor_cache_upper_bytes"]
            + prove_budget["native_pk_read_target_bitmap_fixed_cache_upper_bytes"]
            + target_bitmap_allocation_upper
            + prove_budget["native_pk_read_validation_buffers_upper_bytes"]
            + prove_budget["native_pk_read_stream_buffer_upper_bytes"]
        )
        if (
            prove_budget["disk_required_known_bytes"] > prove_budget["disk_available_bytes"]
            or prove_budget["permutation_grand_product_bank_mapped_bytes"]
            != prove_budget["permutation_grand_product_set_count"]
            * prove_budget["permutation_grand_product_window_mapped_bytes"]
            or prove_budget["permutation_grand_product_commit_workspace_bytes"]
            != 4 * prove_budget["permutation_grand_product_window_mapped_bytes"]
            or prove_budget["permutation_grand_product_bank_mapped_bytes"]
            > prove_budget["disk_required_known_bytes"]
            or prove_budget["proof_permutation_grand_product_commit_host_peak_bytes"]
            > prove_budget["host_required_known_bytes"]
            or prove_budget["native_pk_read_backing_required_bytes"] != exact_pk_read_backing_bytes
            or prove_budget["max_total_backing_bytes"] != exact_pk_read_backing_bytes
            or prove_budget["native_pk_read_target_bitmap_allocation_bitmap_upper_bytes"] != target_bitmap_allocation_upper
            or prove_budget["native_pk_read_validation_buffers_upper_bytes"] != 4 * 4096 * 16
            or prove_budget["native_pk_read_successor_cache_upper_bytes"] != keygen_budget.get("native_successor_read_cache_bytes")
            or prove_budget["native_pk_read_target_bitmap_fixed_cache_upper_bytes"] != keygen_budget.get("native_target_bitmap_fixed_cache_bytes")
            or prove_budget["native_pk_read_stream_buffer_upper_bytes"] != keygen_budget.get("native_pk_read_buffer_capacity_bytes")
            or prove_budget["native_pk_read_host_scratch_upper_bytes"] != expected_read_host_scratch
            or prove_budget["native_pk_read_host_scratch_upper_bytes"] > prove_budget["host_required_known_bytes"]
            or prove_budget["host_required_known_bytes"] > prove_budget["host_available_bytes"]
        ):
            abort(f"{path} native PK reader resources do not fit the authenticated host/disk envelope")
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
        require_exact_keys(record, {"proof_sha256", "native_execution_policy", "msm_operations", "gpu_msm_chunk_calls", "ntt_calls", "cpu_fft_calls", "synchronized_calls", "device_id", "evidence"}, "native GPU prover record")
        counts = [record[key] for key in ("msm_operations", "gpu_msm_chunk_calls", "ntt_calls", "cpu_fft_calls", "synchronized_calls", "device_id")]
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts):
            abort("native GPU prover record has invalid device/call counters")
        calls = record["gpu_msm_chunk_calls"] + record["ntt_calls"]
        if record["proof_sha256"] != result["proof_sha256"] or record["native_execution_policy"] != FORMAL_GPU_EXECUTION_POLICY or record["msm_operations"] <= 0 or record["gpu_msm_chunk_calls"] < record["msm_operations"] or record["ntt_calls"] != 0 or record["cpu_fft_calls"] <= 0 or calls <= 0 or record["synchronized_calls"] != calls or record["evidence"] != result["gpu_evidence"] or result["gpu_evidence"]["gpu_kernel_invocation_count"] != calls:
            abort("native GPU prover record is not synchronized and bound to the verified proof")
        if formal:
            if record["device_id"] != result["selected_cuda_device"]["logical_device_id"]:
                abort("native proof logical device ordinal does not match the selected physical CUDA UUID")
            if sha256(proof.parent / "gpu_prover_record.json") != result["gpu_prover_record_sha256"]:
                abort("native GPU prover record identity is stale")

    if result["source_tree_sha256"] != source_tree_hash() or result["resource_manifest_sha256"] != resource_manifest_hash():
        abort(f"{path} source/resource identity is stale")
    result["_result_path"] = path.relative_to(ROOT).as_posix()
    return result


def validate_gpu_evidence_from_result(result: dict[str, Any]) -> None:
    validate_gpu_evidence_document(result["gpu_evidence"], backend=result.get("native_backend", result.get("backend")))


def check_pair_consistency(cells: list[dict[str, Any]]) -> None:
    if not cells or any("backend" not in cell for cell in cells) or {cell["backend"] for cell in cells} != set(BACKENDS):
        return
    by_dataset = {(cell["dataset"], cell["backend"]): cell for cell in cells}
    for dataset in DATASETS:
        cpu = by_dataset[(dataset, "cpu")]
        gpu = by_dataset[(dataset, "gpu")]
        for field in PAIR_FIELDS:
            if cpu[field] != gpu[field]:
                abort(f"CPU/GPU relation mismatch for {dataset}: field={field}")


def build_matrix_document(cells: list[dict[str, Any]], *, datasets: tuple[str, ...], source_sha256: str, resource_sha256: str, collected_at_utc: str, formal_session_id: str | None, backends: tuple[str, ...] | None) -> dict[str, Any]:
    document = {"schema_version": 1, "baseline": "ezkl", "collection_status": "COLLECTED_PENDING_FINAL_VALIDATION", "relation_id": CANONICAL_RELATION, "datasets": list(datasets), "cell_count": len(cells), "source_tree_sha256": source_sha256, "resource_manifest_sha256": resource_sha256, "collected_at_utc": collected_at_utc, "cells": cells}
    if formal_session_id is not None:
        document["formal_session_id"] = formal_session_id
    else:
        if backends is None:
            abort("local matrix construction requires explicit backends")
        document["backends"] = list(backends)
        document["pair_consistency_checked"] = set(backends) == set(BACKENDS)
    return document


def main(args: argparse.Namespace) -> int:
    selected_datasets = tuple(args.datasets)
    if len(set(selected_datasets)) != len(selected_datasets):
        abort("--datasets must not contain duplicates")
    selected_backends = tuple(args.local_backends or ())
    formal = not selected_backends
    if formal and selected_datasets != DATASETS:
        abort("formal collection requires exactly the five canonical datasets")
    if formal:
        if not isinstance(args.session_id, str) or not FORMAL_SESSION_ID.fullmatch(args.session_id):
            abort("formal collection requires the active --session-id")
    elif args.session_id is not None:
        abort("--session-id cannot be combined with local-only collection")
    elif len(selected_datasets) != 1 or len(selected_backends) != 1:
        abort("local-only collection is restricted to one tiny dataset and one backend")
    runs_root = local_path(args.runs_root, "runs-root")
    output = local_path(args.output, "collector output")
    reject_forbidden_path(output, "collector output")
    if not formal and output == ROOT / "results/formal_matrix.json":
        abort("local-only collection cannot write the formal result matrix path")
    if output.is_symlink():
        abort(f"collector output must not be a symlink: {output}")
    cells: list[dict[str, Any]] = []
    for dataset in selected_datasets:
        for backend in selected_backends if not formal else (None,):
            result_path = runs_root / dataset / "result.json" if formal else runs_root / dataset / str(backend) / "result.json"
            reject_forbidden_path(result_path, f"{dataset}/{backend} result")
            cell = validate_cell(
                result_path,
                expected_session_id=args.session_id if formal else None,
            )
            if cell["dataset"] != dataset or (not formal and cell["backend"] != backend):
                abort(f"result identity does not match its collection path: {result_path}")
            cells.append(cell)
    expected_count = len(selected_datasets) if formal else len(selected_datasets) * len(selected_backends)
    if len(cells) != expected_count:
        abort("collector did not obtain the requested result cells")
    check_pair_consistency(cells)
    output.parent.mkdir(parents=True, exist_ok=True)
    document = build_matrix_document(cells, datasets=selected_datasets, source_sha256=source_tree_hash(), resource_sha256=resource_manifest_hash(), collected_at_utc=datetime.now(timezone.utc).isoformat(), formal_session_id=args.session_id if formal else None, backends=None if formal else selected_backends)
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    mode = "formal dataset results" if formal else "local-only results"
    print(f"collected {len(cells)} verified {mode} into {output.relative_to(ROOT)}; final validation remains required")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(parser().parse_args()))
    except (ContractError, OSError, json.JSONDecodeError) as exc:
        raise SystemExit(print_error(exc))
