#!/usr/bin/env python3
"""Collect the exact ten zkDL result cells without manufacturing success."""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

from runner_contract import (
    BASELINE_ROOT,
    CELL_FILES,
    CONFIGS,
    DATASETS,
    DEVICES,
    PP_FIELDS,
    atomic_write_bytes,
    artifact_inventory,
    config_digest,
    config_path,
    ensure_contained,
    ensure_directory,
    ensure_regular_file,
    build_manifest_path,
    load_build_manifest,
    load_resource_manifest,
    pp_binding_from_manifest,
    pp_path_from_manifest,
    protocol_digest,
    private_publication_fields,
    relative_manifest_path,
    read_json,
    resource_inventory_digest,
    sha256_file,
    validate_pp_file,
)
from validate_final_results import _public_instance, current_source_identities


IDENTITY_FIELDS = (
    "config_sha256",
    "protocol_sha256",
    "resource_manifest_id",
    "resource_inventory_digest",
    "integer_bound_ledger_id",
    "input_preprocess_id",
    "quant_cfg_id",
    "data_identity",
    "source_input_id",
    "graph_features_digest",
    "topology_digest",
    "public_instance",
    "checkpoint_manifest_id",
    "model_shape_id",
    "static_tables_id",
    "static_table_rows_id",
    "model_arch_id",
    "model_param_id",
    "L",
    "hidden_profile",
    "d_in_profile",
    "K_out",
    "checkpoint_binding",
    "provenance",
    "commitment_time_ms",
    "prove_time_ms",
)

def _current_execution_identity(
    dataset: str,
    device: str,
    resource_manifest: dict,
    *,
    graph_count: int,
    feature_dim: int,
) -> dict:
    """Recompute every current binding used by a result cell.

    The collector is deliberately independent of result provenance: it reads
    current files and current release-build manifests, then passes those
    identities into inspect_cell for exact cross-checking.
    """
    identity = dict(
        current_source_identities(
            dataset,
            resource_manifest,
            graph_count=graph_count,
            feature_dim=feature_dim,
            device=device,
        )
    )
    pp = pp_path_from_manifest(resource_manifest)
    validate_pp_file(pp, resource_manifest)
    identity.update(pp_binding_from_manifest(resource_manifest))
    cpu_binary = ensure_regular_file(BASELINE_ROOT / "bin" / "zkdl-formal-cpu", label="CPU formal binary")
    prover_binary = ensure_regular_file(
        BASELINE_ROOT / "bin" / f"zkdl-formal-{device}", label=f"{device} formal binary"
    )
    cpu_build = load_build_manifest("cpu", cpu_binary)
    prover_build = load_build_manifest(device, prover_binary)
    identity.update(
        {
            "cpu_binary_sha256": cpu_build["binary_sha256"],
            "prover_binary_sha256": prover_build["binary_sha256"],
            "cpu_binary_size_bytes": cpu_build["binary_size_bytes"],
            "prover_binary_size_bytes": prover_build["binary_size_bytes"],
            "cpu_build_identity": cpu_build["build_identity"],
            "prover_build_identity": prover_build["build_identity"],
            "build_identity": {"cpu": cpu_build["build_identity"], "prover": prover_build["build_identity"]},
            "cpu_build_manifest_sha256": sha256_file(build_manifest_path("cpu")),
            "prover_build_manifest_sha256": sha256_file(build_manifest_path(device)),
        }
    )
    return identity


def _telemetry_errors(
    proof: dict, verifier: dict, result: dict, success: dict, device: str
) -> list[str]:
    errors: list[str] = []
    telemetry = proof.get("gpu_telemetry")
    if device == "cpu":
        for label, value in (
            ("proof", proof.get("gpu_telemetry")),
            ("verifier", verifier.get("gpu_telemetry")),
            ("result", result.get("gpu_telemetry")),
            ("success", success.get("gpu_telemetry")),
        ):
            if value not in (None, {}):
                errors.append(f"cpu_{label}_gpu_telemetry_present")
        if verifier.get("gpu_telemetry_validated") is not True:
            errors.append("cpu_telemetry_boundary")
        return errors
    if not isinstance(telemetry, dict):
        return ["gpu_telemetry_missing"]
    for label, value in (
        ("verifier", verifier.get("gpu_telemetry")),
        ("result", result.get("gpu_telemetry")),
        ("success", success.get("gpu_telemetry")),
    ):
        if value != telemetry:
            errors.append(f"gpu_{label}_telemetry_mismatch")
    if telemetry.get("provider_id") != "zkDL-native-CUDA-BLS12-381-v1":
        errors.append("gpu_provider")
    if telemetry.get("cross_backend_kat_passed") is not True:
        errors.append("gpu_kat")
    for key in (
        "msm_kernel_invocations",
        "field_inner_product_invocations",
        "scalar_fold_invocations",
        "point_fold_invocations",
        "sumcheck_kernel_invocations",
    ):
        try:
            if int(telemetry.get(key, 0)) <= 0:
                errors.append(f"gpu_{key}")
        except (TypeError, ValueError):
            errors.append(f"gpu_{key}")
    if verifier.get("gpu_telemetry_validated") is not True:
        errors.append("gpu_telemetry_not_validated")
    return errors


def _require_equal(errors: list[str], label: str, left: object, right: object) -> None:
    if left != right:
        errors.append(label)


def inspect_cell(
    run: Path,
    dataset: str,
    device: str,
    *,
    resource_manifest: dict | None = None,
    resource_manifest_id: str | None = None,
    current_identity: dict | None = None,
) -> tuple[dict, list[str]]:
    result_path = run / "result.json"
    if not result_path.is_file():
        return (
            {
                "dataset": dataset,
                "device": device,
                "status": "missing",
                "proof_generated": False,
                "verified": False,
                "formal_experiment_eligible": False,
                "artifact_inventory_valid": False,
                "blocker": f"missing result: {run}",
            },
            ["missing_result"],
        )
    result = read_json(result_path, label="cell result")
    errors: list[str] = []
    for manifest_name, manifest in (("result", result),):
        for field in private_publication_fields(manifest):
            errors.append(f"{manifest_name}_private_field:{field}")
    if resource_manifest is None or resource_manifest_id is None:
        errors.append("collector_resource_manifest_not_supplied")
    if current_identity is None:
        errors.append("collector_current_identity_not_supplied")
    if result.get("dataset") != dataset:
        errors.append("result_dataset_mismatch")
    if result.get("device") != device:
        errors.append("result_device_mismatch")
    if result.get("status") != "ok":
        errors.append("cell_not_success")
    try:
        inventory = artifact_inventory(run, require_complete=True)
        preflight = read_json(run / "formal_artifacts" / "preflight.json", label="cell preflight")
        relation = read_json(run / "formal_artifacts" / "canonical_relation.json", label="cell canonical relation")
        proof = read_json(run / "proof_manifest.json", label="cell prover manifest")
        verifier = read_json(run / "verifier_manifest.json", label="cell verifier manifest")
        success = read_json(run / "success_manifest.json", label="cell success manifest")
        for manifest_name, manifest in (
            ("proof", proof),
            ("verifier", verifier),
            ("success", success),
        ):
            for field in private_publication_fields(manifest):
                errors.append(f"{manifest_name}_private_field:{field}")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        errors.append(f"artifact_inventory:{error}")
        row = dict(result)
        row["artifact_inventory_valid"] = False
        row["collector_errors"] = errors
        row["formal_experiment_eligible"] = False
        return row, errors

    # The native prover and native independent verifier are the authority for
    # success.  Result flags are only accepted after these records and the
    # bytes they describe have been cross-checked.
    if proof.get("proof_generated") is not True:
        errors.append("prover_did_not_generate_proof")
    if verifier.get("verified") is not True:
        errors.append("independent_verifier_not_verified")
    if verifier.get("independent_verifier") is not True:
        errors.append("independent_verifier_flag")
    if verifier.get("verifier_reads_witness") is not False:
        errors.append("verifier_witness_boundary")
    if success.get("schema") != "zkDL_gat_cell_success_manifest_v1":
        errors.append("success_manifest_schema")
    if success.get("status") != "verified":
        errors.append("success_manifest_status")
    if success.get("dataset") != dataset or success.get("device") != device:
        errors.append("success_manifest_cell")
    if success.get("files") != {key: inventory[key] for key in inventory if key != "success_manifest.json"}:
        errors.append("success_manifest_inventory")
    for field in IDENTITY_FIELDS:
        if field not in result:
            errors.append(f"result_missing_{field}")
        if field not in success:
            errors.append(f"success_missing_{field}")
    if result.get("proof_format") != proof.get("proof_format"):
        errors.append("result_proof_format")
    for field in (
        "protocol_id", "statement_id", "model_registry_id", "device",
        "proof_sha256", "proof_size_bytes", "proof_format",
        "model_session_commitments", "model_rerandomization_proofs",
        "commitment_time_ms", "prove_time_ms",
        "dataset", "input_preprocess_id", "quant_cfg_id", "data_identity",
        "source_input_id", "checkpoint_manifest_id", "model_shape_id",
        "model_arch_id", "model_param_id", "graph_features_digest", "topology_digest",
    ):
        _require_equal(errors, f"proof_verifier_{field}", proof.get(field), verifier.get(field))
        _require_equal(errors, f"proof_result_{field}", proof.get(field), result.get(field))
    proof_file = run / "formal_artifacts" / "proof.bin"
    try:
        ensure_regular_file(proof_file, label="cell proof")
        actual_proof_sha = sha256_file(proof_file)
        actual_proof_size = proof_file.stat().st_size
        _require_equal(errors, "proof_actual_digest", proof.get("proof_sha256"), actual_proof_sha)
        _require_equal(errors, "proof_actual_size", proof.get("proof_size_bytes"), actual_proof_size)
    except (OSError, ValueError, TypeError):
        errors.append("proof_file_invalid")
    for manifest in (proof, verifier):
        try:
            relative_manifest_path(manifest.get("proof_path"), run=run, label="proof path")
        except (OSError, ValueError):
            errors.append("proof_path_invalid")
    try:
        public = _public_instance(relation)
        if relation.get("public_logits") is None:
            errors.append("public_statement_logits_missing")
        for key, value in public.items():
            _require_equal(errors, f"public_{key}_preflight", preflight.get(key), value)
        _require_equal(errors, "public_instance_result", result.get("public_instance"), public)
        _require_equal(errors, "public_instance_success", success.get("public_instance"), public)
        _require_equal(errors, "preflight_status", preflight.get("status"), "accepted")
    except (KeyError, TypeError, ValueError, OverflowError):
        errors.append("public_statement_invalid")

    if resource_manifest is not None:
        try:
            expected_pp = pp_binding_from_manifest(resource_manifest)
            pp = pp_path_from_manifest(resource_manifest)
            actual_pp = ensure_regular_file(pp, label="canonical PP")
            validate_pp_file(pp, resource_manifest)
            _require_equal(errors, "pp_actual_digest", expected_pp["pp_sha256"], f"sha256:{sha256_file(actual_pp)}")
            _require_equal(errors, "pp_actual_size", expected_pp["pp_size_bytes"], actual_pp.stat().st_size)
            if resource_manifest_id is not None:
                _require_equal(errors, "result_resource_manifest_id", result.get("resource_manifest_id"), resource_manifest_id)
            for manifest_name, manifest in (("proof", proof), ("verifier", verifier), ("result", result), ("success", success)):
                for key, value in expected_pp.items():
                    _require_equal(errors, f"{manifest_name}_{key}", manifest.get(key), value)
        except (OSError, ValueError, KeyError, TypeError):
            errors.append("pp_binding_invalid")

    if current_identity is not None:
        for key in (
            "config_sha256", "protocol_sha256", "resource_manifest_id",
            "resource_inventory_digest", "checkpoint_manifest_id", "model_shape_id",
            "model_arch_id", "model_param_id", "source_identity",
            "L", "hidden_profile", "d_in_profile", "K_out",
            "input_preprocess_id", "quant_cfg_id",
        ):
            for manifest_name, manifest in (("result", result), ("success", success)):
                _require_equal(errors, f"{manifest_name}_current_{key}", manifest.get(key), current_identity.get(key))
        provenance = result.get("provenance")
        success_provenance = success.get("provenance")
        if not isinstance(provenance, dict) or not isinstance(success_provenance, dict):
            errors.append("provenance_missing")
        else:
            _require_equal(errors, "provenance_match", provenance, success_provenance)
            for key in (
                "config_sha256", "protocol_sha256", "resource_manifest_id",
                "resource_inventory_digest", "source_identity", "registry_sha256",
                "registry_size_bytes", "cpu_binary_sha256", "prover_binary_sha256",
                "cpu_binary_size_bytes", "prover_binary_size_bytes",
                "cpu_build_identity", "prover_build_identity",
                *PP_FIELDS,
                "input_preprocess_id", "quant_cfg_id",
            ):
                _require_equal(errors, f"provenance_current_{key}", provenance.get(key), current_identity.get(key))
        for key in PP_FIELDS:
            _require_equal(errors, f"result_current_{key}", result.get(key), current_identity.get(key))
            _require_equal(errors, f"success_current_{key}", success.get(key), current_identity.get(key))
        for key in ("registry_sha256", "registry_size_bytes"):
            _require_equal(errors, f"result_current_{key}", result.get(key), current_identity.get(key))
            _require_equal(errors, f"success_current_{key}", success.get(key), current_identity.get(key))
        _require_equal(errors, "result_build_identity", result.get("build_identity"), current_identity.get("build_identity"))
        _require_equal(errors, "success_build_identity", success.get("build_identity"), current_identity.get("build_identity"))
        try:
            registry = ensure_regular_file(
                BASELINE_ROOT / "artifacts" / "model_registry" / dataset / "model_registry.bin",
                label="model registry",
            )
            _require_equal(errors, "registry_actual_digest", current_identity.get("registry_sha256"), sha256_file(registry))
            _require_equal(errors, "registry_actual_size", current_identity.get("registry_size_bytes"), registry.stat().st_size)
        except (OSError, ValueError):
            errors.append("registry_invalid")
    errors.extend(_telemetry_errors(proof, verifier, result, success, device))
    row = dict(result)
    row["artifact_inventory_valid"] = not errors
    row["collector_errors"] = errors
    if errors:
        row["formal_experiment_eligible"] = False
    return row, errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=BASELINE_ROOT / "runs")
    parser.add_argument("--output", type=Path, default=BASELINE_ROOT / "results")
    args = parser.parse_args()
    runs = ensure_contained(args.runs, label="runs root")
    output = ensure_contained(args.output, label="result output")
    output.mkdir(parents=True, exist_ok=True)
    resource_manifest, resource_manifest_id = load_resource_manifest(require_staged=True)
    rows: list[dict] = []
    errors: list[str] = []
    for dataset in DATASETS:
        for device in DEVICES:
            run = runs / dataset / device / "warm"
            identity = None
            try:
                preflight = read_json(
                    run / "formal_artifacts" / "preflight.json",
                    label=f"{dataset}/{device} preflight",
                )
                identity = _current_execution_identity(
                    dataset,
                    device,
                    resource_manifest,
                    graph_count=int(preflight["graph_count"]),
                    feature_dim=int(preflight["feature_dim"]),
                )
            except (OSError, KeyError, TypeError, ValueError) as error:
                errors.append(f"{dataset}/{device}:current_identity:{error}")
            row, row_errors = inspect_cell(
                run,
                dataset,
                device,
                resource_manifest=resource_manifest,
                resource_manifest_id=resource_manifest_id,
                current_identity=identity,
            )
            rows.append(row)
            errors.extend(f"{dataset}/{device}:{error}" for error in row_errors)
    matrix_eligible = len(errors) == 0 and all(
        row.get("status") == "ok"
        and row.get("proof_generated") is True
        and row.get("verified") is True
        and row.get("formal_experiment_eligible") is True
        and row.get("privacy_profile_status") == "supported_multi_session_registered_model"
        for row in rows
    )
    payload = {
        "schema": "zkDL_gat_result_matrix_v2",
        "baseline": "zkdl",
        "dataset_order": list(DATASETS),
        "device_order": list(DEVICES),
        "row_count": len(rows),
        "privacy_profile_status": "supported_multi_session_registered_model",
        "formal_experiment_eligible": matrix_eligible,
        "collector_errors": errors,
        "rows": rows,
    }
    atomic_write_bytes(
        output / "final_zkdl_results.json",
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    fields = [
        "device",
        "dataset",
        "status",
        "proof_generated",
        "verified",
        "artifact_inventory_valid",
        "adapter_time_ms",
        "compile_time_ms",
        "commitment_time_ms",
        "prove_time_ms",
        "verify_time_ms",
        "proof_size_bytes",
        "privacy_profile_status",
        "formal_experiment_eligible",
        "proof_format",
        "model_session_commitments",
        "model_rerandomization_proofs",
        "model_session_rerandomization",
        "config_sha256",
        "protocol_sha256",
        "resource_manifest_id",
        "resource_inventory_digest",
        "integer_bound_ledger_id",
        "input_preprocess_id",
        "quant_cfg_id",
        "data_identity",
        "source_input_id",
        "graph_features_digest",
        "topology_digest",
        "checkpoint_manifest_id",
        "model_shape_id",
        "static_tables_id",
        "static_table_rows_id",
        "model_arch_id",
        "model_param_id",
        "public_instance",
        "gpu_telemetry",
        "provenance",
        "blocker",
    ]
    handle = io.StringIO()
    writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_bytes(output / "final_zkdl_results.csv", handle.getvalue().encode("utf-8"))
    lines = [
        "# zkDL formal result matrix",
        "",
        "Strict reusable-model privacy status: `supported_multi_session_registered_model`.",
        "",
        f"Matrix eligibility: `{matrix_eligible}`; collector errors: `{len(errors)}`.",
        "",
        "| dataset | device | status | verified | artifacts | prove ms | verify ms | proof bytes |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('dataset')} | {row.get('device')} | {row.get('status')} | "
            f"{row.get('verified')} | {row.get('artifact_inventory_valid')} | "
            f"{row.get('prove_time_ms', '')} | {row.get('verify_time_ms', '')} | "
            f"{row.get('proof_size_bytes', '')} |"
        )
    atomic_write_bytes(output / "final_zkdl_report.md", ("\n".join(lines) + "\n").encode("utf-8"))
    print(f"ZKDL_COLLECT rows={len(rows)} eligible={matrix_eligible} errors={len(errors)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
