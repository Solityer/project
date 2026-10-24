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
    FORMAL_GPU_H_KERNEL_LOCAL_MEMORY_UPPER_BYTES,
    FORMAL_GPU_H_PROGRAM_METADATA_UPPER_BYTES,
    FORMAL_PHASE_BACKENDS,
    FORMAL_SESSION_ID,
    NATIVE_FORMAL_BACKEND,
    ROOT,
    ContractError,
    abort,
    canonical_json_digest,
    load_json,
    load_json_with_sha256,
    local_path,
    print_error,
    reject_forbidden_path,
    require_circuit_ref,
    require_digest,
    require_exact_keys,
    require_file,
    require_positive_timings,
    public_output_values,
    validate_compact_native_proof_v2,
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
    validate_public_output,
    validate_public_statement,
    require_authenticated_formal_context,
    native_pk_v4_read_backing_geometry,
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


def validate_proof_instances(
    result: dict[str, Any],
    proof: Path,
    *,
    statement: Path,
    public_output: Path,
    artifact_manifest: Path,
    artifact: dict[str, Any],
) -> dict[str, Any]:
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
    reference = validate_compact_native_proof_v2(
        proof,
        statement_path=statement,
        public_output_path=public_output,
        artifact_manifest_path=artifact_manifest,
        artifact=artifact,
    )
    if reference is not None:
        if expected["proof_field"] != "instances_ref" or expected["encoding"] != "canonical-json":
            abort("result proof_instances does not identify the compact native reference encoding")
        if (
            "column_counts" not in expected
            or expected["sha256"] != reference["sha256"]
            or expected["count"] != reference["count"]
            or expected["column_counts"] != reference["column_counts"]
        ):
            abort("compact proof instance digest/count/column shape mismatch")
    else:
        document = load_json(proof, "verifier-facing proof")
        instances = document.get("instances")
        if not isinstance(instances, list) or not instances or any(not isinstance(item, list) for item in instances):
            abort(f"{proof} does not expose a native instances array")
        count = sum(len(item) for item in instances)
        if count <= 0:
            abort(f"{proof} has no native public instances")
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


def _validate_native_ntt_profile(
    path: Path,
    admission_path: Path,
    admission: dict[str, Any],
    prove_budget: dict[str, Any],
) -> dict[str, int]:
    memory = admission.get("native_memory_model")
    if not isinstance(memory, dict) or memory.get("schema") != "ezkl-halo2-static-envelope-v20":
        abort(f"{path} admission has an incompatible native NTT memory model")
    proof_domain = memory.get("proof_domain_size")
    scalar_bytes = memory.get("size_of_fr_bytes")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in (proof_domain, scalar_bytes)):
        abort(f"{path} admission has malformed native NTT geometry")
    if scalar_bytes != 32 or proof_domain & (proof_domain - 1):
        abort(f"{path} admission native NTT geometry is not BN254 power-of-two Fr")
    base_domain = admission.get("domain_size")
    if isinstance(base_domain, bool) or not isinstance(base_domain, int) or base_domain <= 0:
        abort(f"{path} admission has malformed base-domain geometry")
    capacity = min(max(base_domain, proof_domain), 1 << 25)
    plan = {
        "icicle_ntt_min_log_size": 16,
        "icicle_ntt_max_log_size": 25,
        "icicle_ntt_capacity_elements": capacity,
        "icicle_ntt_device_workspace_upper_bytes": 64 * capacity,
        "icicle_ntt_domain_cache_upper_bytes": 32 * (capacity + 1 + 64 + 6),
        "icicle_ntt_init_scratch_upper_bytes": 4 * 64 * 32,
        "icicle_ntt_device_peak_upper_bytes": 96 * capacity + 10464,
        "icicle_ntt_host_staging_upper_bytes": 32 * capacity,
        "icicle_msm_max_concurrency": 1,
        "icicle_msm_precompute_factor": 4,
        "icicle_msm_precomputed_basis_count": 1,
        "icicle_msm_precomputed_basis_device_upper_bytes": base_domain * 64 * 4,
        "icicle_msm_point_staging_host_upper_bytes": min(base_domain, 1 << 20) * 64,
    }
    for document_name, document in (("Admission", memory), ("native prove budget", prove_budget)):
        for key, expected in plan.items():
            observed = document.get(key)
            if isinstance(observed, bool) or not isinstance(observed, int) or observed != expected:
                abort(f"{path} {document_name} {key} does not match its authenticated source formula")
    if (
        prove_budget.get("schema") != "ezkl.native-prove-budget.v6"
        or prove_budget.get("native_execution_policy") != FORMAL_GPU_EXECUTION_POLICY
        or prove_budget.get("admission_sha256") != sha256(admission_path)
        or prove_budget.get("h_workspace_borrows_ntt") is not True
        or prove_budget.get("h_kernel_local_memory_upper_bytes") != FORMAL_GPU_H_KERNEL_LOCAL_MEMORY_UPPER_BYTES
        or prove_budget.get("h_program_metadata_upper_bytes") != FORMAL_GPU_H_PROGRAM_METADATA_UPPER_BYTES
    ):
        abort(f"{path} native prove budget does not bind the current GPU NTT Admission profile")
    if prove_budget.get("icicle_msm_workspace_query_schema") != "ezkl.icicle.cub_msm_workspace_query.v3":
        abort(f"{path} native prove budget does not bind the two-path CUB workspace schema")
    ordinary_peak = prove_budget.get("icicle_msm_ordinary_factor1_device_workspace_peak_bytes")
    prepared_peak = prove_budget.get("icicle_msm_prepared_factor4_device_workspace_peak_bytes")
    query_peak = prove_budget.get("device_workspace_peak_bytes")
    if (
        any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in (ordinary_peak, prepared_peak, query_peak))
        or query_peak != max(ordinary_peak, prepared_peak)
    ):
        abort(f"{path} native prove budget has inconsistent CUB workspace path peaks")
    expected_cuda_required = (
        query_peak
        + plan["icicle_ntt_device_peak_upper_bytes"]
        + plan["icicle_msm_precomputed_basis_device_upper_bytes"]
        + FORMAL_GPU_H_KERNEL_LOCAL_MEMORY_UPPER_BYTES
    )
    host_required = prove_budget.get("host_required_known_bytes")
    host_available = prove_budget.get("host_available_bytes")
    integrated_device = prove_budget.get("integrated_device")
    minimum_host_required = FORMAL_GPU_H_PROGRAM_METADATA_UPPER_BYTES + (
        expected_cuda_required if integrated_device is True else 0
    )
    if (
        prove_budget.get("cuda_required_workspace_bytes") != expected_cuda_required
        or isinstance(host_required, bool)
        or not isinstance(host_required, int)
        or isinstance(host_available, bool)
        or not isinstance(host_available, int)
        or not isinstance(integrated_device, bool)
        or host_required < minimum_host_required
        or host_required > host_available
    ):
        abort(f"{path} native prove budget omits the fused H CUDA or host metadata reserve")
    plan["h_kernel_local_memory_upper_bytes"] = FORMAL_GPU_H_KERNEL_LOCAL_MEMORY_UPPER_BYTES
    plan["h_program_metadata_upper_bytes"] = FORMAL_GPU_H_PROGRAM_METADATA_UPPER_BYTES
    plan["cuda_required_workspace_bytes"] = expected_cuda_required
    return plan


def _validate_shplonk_fixed_cache_budget(
    path: Path,
    admission: dict[str, Any],
    prove_budget: dict[str, Any],
) -> dict[str, int]:
    memory = admission.get("native_memory_model")
    if not isinstance(memory, dict) or memory.get("schema") != "ezkl-halo2-static-envelope-v20":
        abort(f"{path} admission has an incompatible SHPLONK cache memory model")
    cache_domain = admission.get("domain_size")
    cache_extended_domain = memory.get("proof_domain_size")
    cache_scalar = memory.get("size_of_fr_bytes")
    cache_group_layout = memory.get("size_of_shplonk_fixed_aggregate_cache_group_bytes")
    cache_usize = memory.get("size_of_usize_bytes")
    cache_vec_header = memory.get("size_of_vec_header_bytes")
    cache_sets = memory.get("shplonk_rotation_set_count")
    cache_fixed = memory.get("shplonk_max_commitments_per_rotation_set")
    cache_points = memory.get("shplonk_max_rotation_count")
    cache_source_values = (
        cache_domain,
        cache_extended_domain,
        cache_scalar,
        cache_group_layout,
        cache_usize,
        cache_vec_header,
        cache_sets,
        cache_fixed,
        cache_points,
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
        or value > sys.maxsize
        for value in cache_source_values
    ):
        abort(
            f"{path} admission has malformed fixed-cache source bounds: "
            f"base_domain={cache_domain!r}, extended_proof_domain={cache_extended_domain!r}, "
            f"Fr_bytes={cache_scalar!r}, aggregate_group_bytes={cache_group_layout!r}"
        )

    cache_group_host = (
        cache_group_layout
        + cache_domain * cache_scalar
        + cache_fixed * cache_usize
        + cache_points * cache_scalar
        + 3 * 4096
    )
    cache_group_payload = cache_domain * cache_scalar
    cache_group_disk = ((cache_group_payload + 4095) // 4096) * 4096 + 4096
    cache_bank_header = cache_vec_header + 4096
    cache_static = {
        "shplonk_fixed_cache_group_host_bound_bytes": cache_group_host,
        "shplonk_fixed_cache_group_disk_bound_bytes": cache_group_disk,
        "shplonk_fixed_cache_bank_header_bytes": cache_bank_header,
        "shplonk_fixed_cache_domain_size": cache_domain,
        "shplonk_fixed_cache_scalar_bytes": cache_scalar,
        "shplonk_fixed_cache_max_fixed_sources_per_group": cache_fixed,
        "shplonk_fixed_cache_max_points_per_group": cache_points,
        "shplonk_fixed_cache_rotation_set_count": cache_sets,
    }
    cache_static_mismatches = {
        name: {"actual": prove_budget.get(name), "expected": value}
        for name, value in cache_static.items()
        if prove_budget.get(name) != value
    }
    if cache_static_mismatches:
        abort(
            f"{path} native prove budget SHPLONK cache shape does not match its admitted source geometry: "
            f"base_domain_expected={cache_domain}, extended_proof_domain={cache_extended_domain}, "
            f"Fr_bytes={cache_scalar}, aggregate_group_bytes={cache_group_layout}, "
            f"group_host_bytes_budget={prove_budget.get('shplonk_fixed_cache_group_host_bound_bytes')!r} "
            f"expected={cache_group_host}, "
            f"group_disk_bytes_budget={prove_budget.get('shplonk_fixed_cache_group_disk_bound_bytes')!r} "
            f"expected={cache_group_disk}, mismatches={cache_static_mismatches!r}"
        )

    for name in (
        "disk_available_bytes",
        "disk_required_known_bytes",
        "host_available_bytes",
        "host_required_known_bytes",
        "shplonk_fixed_cache_max_groups",
        "shplonk_fixed_cache_host_limit_bytes",
        "shplonk_fixed_cache_bank_header_limit_bytes",
        "shplonk_fixed_cache_disk_limit_bytes",
    ):
        value = prove_budget.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > sys.maxsize:
            abort(f"{path} native prove budget has invalid {name}")

    expected_cache_groups = min(
        2,
        cache_sets,
        max(
            0,
            prove_budget["host_available_bytes"]
            - prove_budget["host_required_known_bytes"]
            - cache_bank_header,
        )
        // cache_group_host,
        max(
            0,
            prove_budget["disk_available_bytes"]
            - prove_budget["disk_required_known_bytes"],
        )
        // cache_group_disk,
    )
    expected_cache_host = cache_bank_header + expected_cache_groups * cache_group_host if expected_cache_groups else 0
    expected_cache_bank_header = cache_bank_header if expected_cache_groups else 0
    expected_cache_disk = expected_cache_groups * cache_group_disk
    cache_limit_expected = {
        "shplonk_fixed_cache_max_groups": expected_cache_groups,
        "shplonk_fixed_cache_host_limit_bytes": expected_cache_host,
        "shplonk_fixed_cache_bank_header_limit_bytes": expected_cache_bank_header,
        "shplonk_fixed_cache_disk_limit_bytes": expected_cache_disk,
    }
    cache_limit_mismatches = {
        name: {"actual": prove_budget.get(name), "expected": value}
        for name, value in cache_limit_expected.items()
        if prove_budget.get(name) != value
    }
    if cache_limit_mismatches:
        abort(
            f"{path} native prove budget SHPLONK cache limits do not match the admitted resource envelope: "
            f"base_domain={cache_domain}, extended_proof_domain={cache_extended_domain}, "
            f"Fr_bytes={cache_scalar}, aggregate_group_bytes={cache_group_layout}, "
            f"group_host_bytes={cache_group_host}, group_disk_bytes={cache_group_disk}, "
            f"host_available={prove_budget['host_available_bytes']}, "
            f"host_required={prove_budget['host_required_known_bytes']}, "
            f"disk_available={prove_budget['disk_available_bytes']}, "
            f"disk_required={prove_budget['disk_required_known_bytes']}, "
            f"mismatches={cache_limit_mismatches!r}"
        )
    return {
        "rotation_set_count": cache_sets,
        "group_host_bytes": cache_group_host,
        "group_disk_bytes": cache_group_disk,
        "bank_header_bytes": cache_bank_header,
    }


def require_positive_finite_duration(result: dict[str, Any], field: str, label: str) -> int | float:
    if field not in result:
        abort(f"{label} is missing {field}")
    value = result[field]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        abort(f"{label} {field} must be a positive finite duration")
    return value


def validate_cell(
    path: Path,
    *,
    expected_session_id: str | None = None,
    expected_source_tree_sha256: str | None = None,
    expected_resource_manifest_sha256: str | None = None,
    expected_config_values: dict[str, str] | None = None,
    expected_result_sha256: str | None = None,
) -> dict[str, Any]:
    if expected_result_sha256 is None:
        result = load_json(path, f"result cell {path}")
    else:
        require_digest(expected_result_sha256, f"{path} expected checkpoint digest")
        result, actual_result_sha256 = load_json_with_sha256(path, f"result cell {path}")
        if actual_result_sha256 != expected_result_sha256:
            abort(f"{path} result bytes differ from the active session checkpoint")
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
    native_ntt_plan: dict[str, int] | None = None
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
        secure_config = (
            validate_config(ROOT / "resources/canonical/configs" / f"{dataset}_secure.cfg", dataset)
            if expected_config_values is None
            else dict(expected_config_values)
        )
        if secure_config.get("dataset") != dataset:
            abort(f"{path} authenticated config values belong to another dataset")
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
    duration = require_positive_finite_duration(result, "commitment_time_ms", str(path))
    if duration != proof_metrics["commitment_time_ms"]:
        abort(f"{path} commitment_time_ms does not match the native measured duration")
    prove_time = require_positive_finite_duration(result, "prove_time_ms", str(path))
    if prove_time != proof_metrics["prove_time_ms"]:
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
    verify_time = require_positive_finite_duration(result, "verify_time_ms", str(path))
    if verify_time != native_verify_metrics["verify_time_ms"]:
        abort(f"{path} verify_time_ms does not match native verification timing")
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
    validate_proof_instances(
        result,
        proof,
        statement=statement,
        public_output=public_output,
        artifact_manifest=artifact_manifest,
        artifact=artifact,
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
        native_receipt_names = (
            "setup_receipt.json",
            "native_keygen_budget.json",
            "native_keygen_receipt.json",
            "native_prove_budget.json",
            "admission.json",
        )
        if any(name not in artifact_records for name in native_receipt_names):
            abort(f"{path} artifact manifest omits native setup/prove resource receipts")
        setup_receipt = load_json(artifact_manifest.parent / native_receipt_names[0], "formal setup resource receipt")
        keygen_budget = load_json(artifact_manifest.parent / native_receipt_names[1], "native keygen budget")
        keygen_receipt = load_json(artifact_manifest.parent / native_receipt_names[2], "native keygen census receipt")
        prove_budget = load_json(artifact_manifest.parent / native_receipt_names[3], "native proving-key read budget")
        admission_path = artifact_manifest.parent / "admission.json"
        admission = load_json(admission_path, "formal native admission")
        current_admission_sha256 = sha256(admission_path)
        if any(not isinstance(value, dict) for value in (setup_receipt, keygen_budget, keygen_receipt, prove_budget)):
            abort(f"{path} native resource receipts must be JSON objects")
        native_ntt_plan = _validate_native_ntt_profile(path, admission_path, admission, prove_budget)
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
                ("admission.json", current_admission_sha256),
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
            "schema": "ezkl.native-keygen-receipt.v4",
            "session_id": session_id,
            "dataset": dataset,
            "budget_receipt_sha256": receipt_digests["native_keygen_budget.json"],
            "settings_sha256": result["settings_sha256"],
            "compiled_circuit_sha256": result["compiled_circuit_sha256"],
            "vk_sha256": result["vk_sha256"],
            "pk_sha256": result["pk_sha256"],
            "evidence_kind": "mandatory_vk_pk_native_keygen_synthesis_census",
            "coverage": "actual_vk_and_pk_synthesis_census; exact_selector_advice_fixed_permutation_and_lookup_bounds",
        }
        if any(keygen_receipt.get(name) != value for name, value in expected_keygen.items()):
            abort(f"{path} native keygen census receipt does not match its formal cell")
        relation_identity = keygen_receipt.get("relation_identity")
        if not isinstance(relation_identity, dict) or relation_identity.get("dataset") != dataset or relation_identity.get("resource_manifest_sha256") != result["resource_manifest_sha256"]:
            abort(f"{path} native keygen relation identity does not match the canonical result")
        expected_native_budget = {
            "schema": "ezkl.native-keygen-budget.v3",
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
        setup_disk_required = keygen_budget.get("setup_disk_required_known_bytes")
        proof_spill_reserve = keygen_budget.get("proof_spill_reserve_bytes")
        disk_required = keygen_budget.get("disk_required_known_bytes")
        disk_available = keygen_budget.get("disk_available_bytes")
        if (
            any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                or value > sys.maxsize
                for value in (setup_disk_required, proof_spill_reserve, disk_required, disk_available)
            )
            or proof_spill_reserve != 0
            or keygen_budget.get("disk_budget_scope")
            != "setup_keygen_peak_and_source_bounded_permutation_backing; proof_disk_checked_after_actual_setup_census"
            or setup_disk_required != disk_required
            or disk_required > disk_available
        ):
            abort(f"{path} native keygen budget confuses Setup storage with the post-census proof disk gate")
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
        constraint_system = admission.get("constraint_system")
        permutation_domain = admission.get("domain_size")
        permutation_columns = constraint_system.get("num_permutation_columns") if isinstance(constraint_system, dict) else None
        if (
            isinstance(permutation_domain, bool) or not isinstance(permutation_domain, int) or permutation_domain <= 0
            or isinstance(permutation_columns, bool) or not isinstance(permutation_columns, int) or permutation_columns <= 0
        ):
            abort(f"{path} native admission has invalid permutation bitmap geometry")
        permutation_cells = permutation_domain * permutation_columns
        if permutation_cells > sys.maxsize or permutation_values[2] > permutation_cells:
            abort(f"{path} native permutation census exceeds its admitted bitmap domain")
        successor_logical_bytes = 16 * permutation_values[2]
        successor_reserved_bytes = ((successor_logical_bytes + 4095) // 4096) * 4096
        target_bitmap_geometry = _validate_native_pk_v4_read_backing_budget(
            admission, permutation_values[2], prove_budget
        )
        target_bitmap_virtual_bytes = target_bitmap_geometry["target_bitmap_virtual_bytes"]
        target_bitmap_page_count = target_bitmap_geometry["target_bitmap_page_count"]
        target_bitmap_reserved_bytes = target_bitmap_geometry["target_bitmap_backing_bytes"]
        exact_pk_read_backing_bytes = target_bitmap_reserved_bytes
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
            "schema": "ezkl.native-prove-budget.v6",
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
            "native_execution_policy": FORMAL_GPU_EXECUTION_POLICY,
            "h_workspace_borrows_ntt": True,
            **native_ntt_plan,
        }
        prove_budget_exact_keys = set(expected_prove) | {
            "icicle_msm_workspace_query_schema",
            "icicle_msm_ordinary_factor1_device_workspace_peak_bytes",
            "icicle_msm_prepared_factor4_device_workspace_peak_bytes",
            "native_pk_format_version",
            "native_pk_sigma_wire_encoding",
            "native_pk_sigma_index_width_bits",
            "native_pk_sigma_packed_payload_bytes",
            "disk_available_bytes",
            "disk_required_known_bytes",
            "permutation_grand_product_set_count",
            "permutation_grand_product_bank_mapped_bytes",
            "permutation_grand_product_window_mapped_bytes",
            "permutation_grand_product_commit_workspace_bytes",
            "proof_permutation_grand_product_commit_host_peak_bytes",
            "shplonk_fixed_cache_group_host_bound_bytes",
            "shplonk_fixed_cache_group_disk_bound_bytes",
            "shplonk_fixed_cache_bank_header_bytes",
            "shplonk_fixed_cache_domain_size",
            "shplonk_fixed_cache_scalar_bytes",
            "shplonk_fixed_cache_max_fixed_sources_per_group",
            "shplonk_fixed_cache_max_points_per_group",
            "shplonk_fixed_cache_rotation_set_count",
            "shplonk_fixed_cache_max_groups",
            "shplonk_fixed_cache_host_limit_bytes",
            "shplonk_fixed_cache_bank_header_limit_bytes",
            "shplonk_fixed_cache_disk_limit_bytes",
            "native_pk_read_backing_required_bytes",
            "native_pk_read_mapped_peak_upper_bytes",
            "native_pk_read_mapped_extra_upper_bytes",
            "max_total_backing_bytes",
            "native_pk_read_host_scratch_upper_bytes",
            "native_pk_materialize_host_scratch_upper_bytes",
            "native_pk_read_successor_cache_upper_bytes",
            "native_pk_read_target_bitmap_fixed_cache_upper_bytes",
            "native_pk_read_target_bitmap_allocation_bitmap_upper_bytes",
            "native_pk_read_validation_buffers_upper_bytes",
            "native_pk_read_stream_buffer_upper_bytes",
            "host_available_bytes",
            "host_required_known_bytes",
            "h_kernel_local_memory_upper_bytes",
            "h_program_metadata_upper_bytes",
            "cuda_required_workspace_bytes",
            "integrated_device",
            "device_workspace_peak_bytes",
            "backing_scope",
            "host_scope",
        }
        require_exact_keys(prove_budget, prove_budget_exact_keys, "native prove budget")
        if any(prove_budget.get(name) != value for name, value in expected_prove.items()):
            abort(f"{path} native prove budget does not bind its setup, keygen, and binary identities")
        if (
            not isinstance(admission, dict)
            or receipt_digests["admission.json"] != current_admission_sha256
            or current_admission_sha256 != prove_budget.get("admission_sha256")
            or current_admission_sha256 != keygen_budget.get("admission_sha256")
            or current_admission_sha256 != setup_receipt.get("admission_sha256")
        ):
            abort(f"{path} formal setup/prove budgets do not bind the current Admission bytes")
        _validate_shplonk_fixed_cache_budget(path, admission, prove_budget)
        for name in (
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
            "icicle_msm_max_concurrency",
            "icicle_msm_precompute_factor",
            "icicle_msm_precomputed_basis_count",
            "icicle_msm_precomputed_basis_device_upper_bytes",
            "icicle_msm_point_staging_host_upper_bytes",
            "icicle_ntt_min_log_size",
            "icicle_ntt_max_log_size",
            "icicle_ntt_capacity_elements",
            "icicle_ntt_device_workspace_upper_bytes",
            "icicle_ntt_domain_cache_upper_bytes",
            "icicle_ntt_init_scratch_upper_bytes",
            "icicle_ntt_device_peak_upper_bytes",
            "icicle_ntt_host_staging_upper_bytes",
        ):
            value = prove_budget.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > sys.maxsize:
                abort(f"{path} native prove budget has invalid {name}")
        target_bitmap_allocation_upper = target_bitmap_geometry["target_bitmap_allocation_bitmap_bytes"]
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
        require_exact_keys(record, {
            "schema_version", "proof_sha256", "admission_sha256", "native_execution_policy", "icicle_msm_max_concurrency",
            "icicle_msm_precompute_factor", "gpu_msm_precompute_calls", "gpu_msm_precomputed_chunk_calls",
            "gpu_msm_precomputed_cache_hits", "icicle_msm_precomputed_basis_device_upper_bytes",
            "icicle_msm_precomputed_basis_device_bytes", "icicle_msm_precomputed_basis_device_peak_bytes",
            "msm_operations", "gpu_msm_chunk_calls", "ntt_calls", "cpu_fft_calls", "synchronized_calls", "device_id",
            "gpu_ntt_elements", "ntt_max_size", "ntt_workspace_capacity_bytes", "ntt_domain_capacity_bytes",
            "ntt_host_staging_capacity_bytes", "ntt_device_peak_upper_bound_bytes",
            "gpu_h_eval_groups", "gpu_h_eval_calls", "gpu_h_eval_rows", "gpu_h_eval_fallbacks", "gpu_h_eval_chunks",
            "gpu_h_eval_syncs", "gpu_h_workspace_high_water_bytes", "gpu_h_local_memory_peak_upper_bytes", "evidence",
        }, "native GPU prover record")
        counts = [record[key] for key in (
            "schema_version", "icicle_msm_max_concurrency", "icicle_msm_precompute_factor", "gpu_msm_precompute_calls",
            "gpu_msm_precomputed_chunk_calls", "gpu_msm_precomputed_cache_hits",
            "icicle_msm_precomputed_basis_device_upper_bytes", "icicle_msm_precomputed_basis_device_bytes",
            "icicle_msm_precomputed_basis_device_peak_bytes", "msm_operations", "gpu_msm_chunk_calls", "ntt_calls",
            "cpu_fft_calls", "synchronized_calls", "device_id", "gpu_ntt_elements", "ntt_max_size",
            "ntt_workspace_capacity_bytes", "ntt_domain_capacity_bytes", "ntt_host_staging_capacity_bytes",
            "ntt_device_peak_upper_bound_bytes", "gpu_h_eval_groups", "gpu_h_eval_calls", "gpu_h_eval_rows", "gpu_h_eval_fallbacks",
            "gpu_h_eval_chunks", "gpu_h_eval_syncs", "gpu_h_workspace_high_water_bytes",
            "gpu_h_local_memory_peak_upper_bytes",
        )]
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts):
            abort("native GPU prover record has invalid device/call counters")
        kernel_calls = record["gpu_msm_chunk_calls"] + record["gpu_msm_precompute_calls"] + record["ntt_calls"] + record["gpu_h_eval_chunks"]
        sync_calls = record["gpu_msm_chunk_calls"] + record["gpu_msm_precompute_calls"] + record["ntt_calls"] + record["gpu_h_eval_syncs"]
        max_size = record["ntt_max_size"]
        plan = native_ntt_plan
        if plan is None:
            abort("formal GPU prover record has no authenticated NTT Admission plan")
        expected_workspace = max_size * 64
        expected_domain = 0 if max_size == 0 else 32 * (max_size + 1 + 64 + 6)
        expected_host = max_size * 32
        expected_device_peak = 0 if max_size == 0 else 96 * max_size + 10464
        planned_ntt_below_minimum = plan["icicle_ntt_capacity_elements"] < (1 << 16)
        minimum_ntt_elements = (1 << 16) * record["ntt_calls"]
        maximum_ntt_elements = max_size * record["ntt_calls"]
        if (
            record["schema_version"] != 4
            or record["proof_sha256"] != result["proof_sha256"]
            or record["admission_sha256"] != sha256(proof.parent / "admission.json")
            or record["native_execution_policy"] != FORMAL_GPU_EXECUTION_POLICY
            or record["icicle_msm_max_concurrency"] != plan["icicle_msm_max_concurrency"]
            or record["icicle_msm_precompute_factor"] != plan["icicle_msm_precompute_factor"]
            or record["icicle_msm_precomputed_basis_device_upper_bytes"] != plan["icicle_msm_precomputed_basis_device_upper_bytes"]
            or record["icicle_msm_precomputed_basis_device_bytes"] > record["icicle_msm_precomputed_basis_device_upper_bytes"]
            or record["icicle_msm_precomputed_basis_device_peak_bytes"] < record["icicle_msm_precomputed_basis_device_bytes"]
            or record["icicle_msm_precomputed_basis_device_peak_bytes"] > record["icicle_msm_precomputed_basis_device_upper_bytes"]
            or record["msm_operations"] <= 0
            or record["gpu_msm_chunk_calls"] < record["msm_operations"]
            or record["gpu_msm_precomputed_chunk_calls"] > record["gpu_msm_chunk_calls"]
            or record["gpu_msm_precompute_calls"] > record["gpu_msm_precomputed_chunk_calls"]
            or record["gpu_msm_precomputed_cache_hits"] > record["gpu_msm_precomputed_chunk_calls"]
            or record["gpu_msm_precomputed_chunk_calls"] != record["gpu_msm_precompute_calls"] + record["gpu_msm_precomputed_cache_hits"]
            or (planned_ntt_below_minimum and record["ntt_calls"] != 0)
            or (record["ntt_calls"] == 0 and record["gpu_ntt_elements"] != 0)
            or (record["ntt_calls"] > 0 and max_size < (1 << 16))
            or (record["ntt_calls"] > 0 and record["gpu_ntt_elements"] < minimum_ntt_elements)
            or (record["ntt_calls"] > 0 and record["gpu_ntt_elements"] > maximum_ntt_elements)
            or record["gpu_h_eval_calls"] <= 0
            or record["gpu_h_eval_groups"] != record["gpu_h_eval_calls"] + record["gpu_h_eval_fallbacks"]
            or record["gpu_h_eval_rows"] <= 0
            or record["gpu_h_eval_chunks"] <= 0
            or record["gpu_h_eval_syncs"] != record["gpu_h_eval_chunks"]
            or record["gpu_h_workspace_high_water_bytes"] > plan["icicle_ntt_device_workspace_upper_bytes"]
            or record["gpu_h_local_memory_peak_upper_bytes"] > plan["h_kernel_local_memory_upper_bytes"]
            or (max_size != 0 and max_size & (max_size - 1) != 0)
            or max_size > plan["icicle_ntt_capacity_elements"]
            or record["ntt_workspace_capacity_bytes"] != expected_workspace
            or record["ntt_domain_capacity_bytes"] != expected_domain
            or record["ntt_host_staging_capacity_bytes"] != expected_host
            or record["ntt_device_peak_upper_bound_bytes"] != expected_device_peak
            or expected_workspace > plan["icicle_ntt_device_workspace_upper_bytes"]
            or expected_domain > plan["icicle_ntt_domain_cache_upper_bytes"]
            or expected_device_peak > plan["icicle_ntt_device_peak_upper_bytes"]
            or expected_host > plan["icicle_ntt_host_staging_upper_bytes"]
            or kernel_calls <= 0
            or record["synchronized_calls"] != sync_calls
            or record["evidence"] != result["gpu_evidence"]
            or result["gpu_evidence"]["gpu_kernel_invocation_count"] != kernel_calls
        ):
            abort("native GPU prover record is not synchronized and bound to the verified proof")
        if formal:
            if record["device_id"] != result["selected_cuda_device"]["logical_device_id"]:
                abort("native proof logical device ordinal does not match the selected physical CUDA UUID")
            if sha256(proof.parent / "gpu_prover_record.json") != result["gpu_prover_record_sha256"]:
                abort("native GPU prover record identity is stale")

    source_identity = source_tree_hash() if expected_source_tree_sha256 is None else expected_source_tree_sha256
    resource_identity = resource_manifest_hash() if expected_resource_manifest_sha256 is None else expected_resource_manifest_sha256
    if result["source_tree_sha256"] != source_identity or result["resource_manifest_sha256"] != resource_identity:
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


def _validate_native_pk_v4_read_backing_budget(
    admission: dict[str, Any], successor_edges: int, prove_budget: dict[str, Any]
) -> dict[str, int]:
    constraint_system = admission.get("constraint_system")
    permutation_domain = admission.get("domain_size")
    permutation_columns = constraint_system.get("num_permutation_columns") if isinstance(constraint_system, dict) else None
    if (
        isinstance(permutation_domain, bool)
        or not isinstance(permutation_domain, int)
        or permutation_domain <= 0
        or isinstance(permutation_columns, bool)
        or not isinstance(permutation_columns, int)
        or permutation_columns <= 0
    ):
        abort("formal native admission has invalid permutation bitmap geometry")
    permutation_cells = permutation_domain * permutation_columns
    geometry = native_pk_v4_read_backing_geometry(permutation_cells, successor_edges)
    expected = {
        "native_pk_read_backing_required_bytes": geometry["target_bitmap_backing_bytes"],
        "max_total_backing_bytes": geometry["target_bitmap_backing_bytes"],
        "native_pk_read_target_bitmap_allocation_bitmap_upper_bytes": geometry[
            "target_bitmap_allocation_bitmap_bytes"
        ],
    }
    if any(
        isinstance(prove_budget.get(name), bool)
        or not isinstance(prove_budget.get(name), int)
        or prove_budget.get(name) != value
        for name, value in expected.items()
    ):
        abort("native prove budget does not match the v4 PK reader's target-bitmap backing")
    return geometry


def _pending_formal_cell(
    path: Path,
    *,
    dataset: str,
    session_id: str,
    source_sha256: str,
    resource_sha256: str,
) -> dict[str, Any]:
    """Load only enough identity to construct the pending matrix; final validation owns acceptance."""
    result = load_json(path, f"pending formal result cell {path}")
    if (
        result.get("dataset") != dataset
        or result.get("status") != "VERIFY_OK"
        or result.get("formal_session_id") != session_id
        or result.get("native_verify_result") is not True
        or result.get("source_tree_sha256") != source_sha256
        or result.get("resource_manifest_sha256") != resource_sha256
        or path.parent.name != dataset
    ):
        abort(f"{path} is not a current-session independently verified result for the pending matrix")
    missing = REQUIRED_RESULT_KEYS - set(result)
    if missing:
        abort(f"{path} is missing pending formal result fields: {sorted(missing)}")
    result["_result_path"] = path.relative_to(ROOT).as_posix()
    return result


def main(
    args: argparse.Namespace,
    *,
    authenticated_context: Any | None = None,
    formal_session_id: str | None = None,
) -> int:
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
        if authenticated_context is not None:
            if formal_session_id != args.session_id:
                abort("sealed result collection session does not match its command-line session")
            require_authenticated_formal_context(
                authenticated_context,
                session_id=args.session_id,
                dataset=DATASETS[0],
                config=ROOT / "resources/canonical/configs" / f"{DATASETS[0]}_secure.cfg",
            )
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
    if formal and authenticated_context is not None:
        source_sha256 = authenticated_context.source_tree_sha256
        resource_sha256 = authenticated_context.resource_manifest_sha256
    else:
        source_sha256 = source_tree_hash()
        resource_sha256 = resource_manifest_hash()
    cells: list[dict[str, Any]] = []
    for dataset in selected_datasets:
        for backend in selected_backends if not formal else (None,):
            result_path = runs_root / dataset / "result.json" if formal else runs_root / dataset / str(backend) / "result.json"
            reject_forbidden_path(result_path, f"{dataset}/{backend} result")
            if formal:
                cell = _pending_formal_cell(
                    result_path,
                    dataset=dataset,
                    session_id=args.session_id,
                    source_sha256=source_sha256,
                    resource_sha256=resource_sha256,
                )
            else:
                cell = validate_cell(result_path, expected_session_id=None)
            if cell["dataset"] != dataset or (not formal and cell["backend"] != backend):
                abort(f"result identity does not match its collection path: {result_path}")
            cells.append(cell)
    expected_count = len(selected_datasets) if formal else len(selected_datasets) * len(selected_backends)
    if len(cells) != expected_count:
        abort("collector did not obtain the requested result cells")
    check_pair_consistency(cells)
    output.parent.mkdir(parents=True, exist_ok=True)
    document = build_matrix_document(cells, datasets=selected_datasets, source_sha256=source_sha256, resource_sha256=resource_sha256, collected_at_utc=datetime.now(timezone.utc).isoformat(), formal_session_id=args.session_id if formal else None, backends=None if formal else selected_backends)
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    mode = "formal dataset results" if formal else "local-only results"
    print(f"collected {len(cells)} verified {mode} into {output.relative_to(ROOT)}; final validation remains required")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(parser().parse_args()))
    except (ContractError, OSError, json.JSONDecodeError) as exc:
        raise SystemExit(print_error(exc))
