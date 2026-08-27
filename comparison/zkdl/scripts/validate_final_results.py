#!/usr/bin/env python3
"""Independent matrix-level validator for the ten formal zkDL cells.

The validator authenticates each cell against the current baseline-local
configuration, protocol document, staged-resource inventory, and checkpoint
metadata.  It never trusts a result's self-reported provenance as the source
of those identities.  The relation/preflight checks are structural: they
hash and cross-check formal artifacts without executing a dataset or a
prover/verifier.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Any

from runner_contract import (
    BASELINE_ROOT,
    DATASETS,
    DEVICES,
    PP_FIELDS,
    atomic_write_json,
    artifact_inventory,
    build_manifest_path,
    config_digest,
    config_path,
    ensure_contained,
    ensure_directory,
    ensure_regular_file,
    load_build_manifest,
    load_resource_manifest,
    parse_config,
    pp_binding_from_manifest,
    pp_path_from_manifest,
    private_publication_fields,
    protocol_digest,
    read_json,
    resource_inventory_digest,
    sha256_file,
    source_identity,
    validate_pp_file,
    validate_config_resources,
)


FR = 0x73EDA753299D7D483339D80809A1D80553BDA402FFFE5BFEFFFFFFFF00000001
CURRENT_BINDINGS = (
    "config_sha256",
    "protocol_sha256",
    "resource_manifest_id",
    "resource_inventory_digest",
)


def _json_id(value: Any, *, domain: bytes = b"") -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(domain + payload).hexdigest()}"


def _int_list(value: Any, label: str) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an integer list")
    result: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(f"{label} contains a non-integer")
        result.append(int(item))
    return result


def _public_instance(relation: dict[str, Any]) -> dict[str, Any]:
    node_ptr = _int_list(relation.get("node_ptr"), "node_ptr")
    edge_ptr = _int_list(relation.get("edge_ptr"), "edge_ptr")
    src = _int_list(relation.get("src"), "src")
    dst = _int_list(relation.get("dst"), "dst")
    n = int(relation["node_count"])
    e = int(relation["edge_count"])
    g = int(relation["graph_count"])
    d = int(relation["feature_dim"])
    c = int(relation["class_count"])
    if n < 1 or e < 1 or g < 1 or d < 1 or c < 1:
        raise ValueError("relation dimensions must be positive")
    if len(src) != e or len(dst) != e:
        raise ValueError("relation edge_count does not match src/dst")
    if len(node_ptr) != g + 1 or len(edge_ptr) != g + 1:
        raise ValueError("relation pointer lengths do not match graph_count")
    if node_ptr[0] != 0 or node_ptr[-1] != n or any(a >= b for a, b in zip(node_ptr, node_ptr[1:])):
        raise ValueError("relation node_ptr is not canonical")
    if edge_ptr[0] != 0 or edge_ptr[-1] != e or any(a > b for a, b in zip(edge_ptr, edge_ptr[1:])):
        raise ValueError("relation edge_ptr is not canonical")
    if any(value < 0 or value >= n for value in src + dst):
        raise ValueError("relation endpoint escapes node range")
    return {
        "node_count": n,
        "edge_count": e,
        "graph_count": g,
        "feature_dim": d,
        "class_count": c,
        "node_ptr": node_ptr,
        "edge_ptr": edge_ptr,
    }


def _topology_digest(relation: dict[str, Any], public: dict[str, Any]) -> str:
    src = _int_list(relation["src"], "src")
    dst = _int_list(relation["dst"], "dst")
    digest = hashlib.sha256()
    digest.update(b"zkDL-GAT-public-topology-v1\0")
    digest.update(struct.pack(">QQQ", public["node_count"], public["edge_count"], public["graph_count"]))
    for values in (public["node_ptr"], public["edge_ptr"], src, dst):
        digest.update(struct.pack(">Q", len(values)))
        for value in values:
            digest.update(struct.pack(">Q", value))
    return digest.hexdigest()


def _feature_digest(relation: dict[str, Any], public: dict[str, Any]) -> str:
    features = _int_list(relation.get("public_features"), "public_features")
    expected = public["node_count"] * public["feature_dim"]
    if len(features) != expected:
        raise ValueError("relation public_features length mismatch")
    digest = hashlib.sha256()
    digest.update(b"zkDL-GAT-public-features-v1\0")
    digest.update(struct.pack(">QQ", public["node_count"], public["feature_dim"]))
    for value in features:
        digest.update(struct.pack(">q", value))
    return digest.hexdigest()


def _table_rows_digest(operation_families: Any) -> str:
    if not isinstance(operation_families, list):
        raise ValueError("operation_families must be a list")
    entries: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()

    def visit(expr: Any) -> None:
        if not isinstance(expr, dict):
            return
        table = expr.get("PublicTableLookup")
        if isinstance(table, dict):
            values = tuple(int(value) % FR for value in table.get("values", []))
            bits = tuple(int(value) for value in table.get("bit_columns", []))
            if not values or not bits:
                raise ValueError("malformed public table lookup")
            entries.add((values, bits))
            return
        for key in ("Add", "Sub", "Mul"):
            if key in expr:
                children = expr[key]
                if not isinstance(children, list) or len(children) != 2:
                    raise ValueError("malformed arithmetic expression")
                visit(children[0])
                visit(children[1])
                return

    for operation in operation_families:
        if not isinstance(operation, dict):
            raise ValueError("malformed operation family")
        for expression in operation.get("residuals", []):
            visit(expression)
    ordered = sorted(entries, key=lambda item: (len(item[0]), item[0], len(item[1]), item[1]))
    digest = hashlib.sha256()
    digest.update(b"zkDL-GAT-registry-static-table-rows-v1\0")
    for values, bits in ordered:
        digest.update(struct.pack(">Q", len(values)))
        for value in values:
            digest.update(int(value).to_bytes(32, "big"))
        digest.update(struct.pack(">Q", len(bits)))
        for bit in bits:
            digest.update(struct.pack(">I", bit))
    return f"sha256:{digest.hexdigest()}"


def _checkpoint_binding(config: dict[str, str], manifest: dict[str, Any], graph_count: int, feature_dim: int) -> dict[str, Any]:
    dataset = config["dataset"]
    if manifest.get("dataset", dataset) != dataset:
        raise ValueError("checkpoint manifest dataset mismatch")
    for key in ("L", "K_out", "C"):
        if int(manifest.get(key, -1)) != int(config[key]):
            raise ValueError(f"checkpoint/config {key} mismatch")
    profile: list[tuple[int, int]] = []
    for token in str(config["hidden_profile"]).split(","):
        parts = token.strip().split("x")
        if len(parts) != 2 or any(not part.isdigit() for part in parts):
            raise ValueError("malformed hidden_profile config")
        profile.append((int(parts[0]), int(parts[1])))
    actual_profile = [
        (int(item.get("head_count", -1)), int(item.get("head_dim", -1)))
        for item in manifest.get("hidden_profile", [])
    ]
    if actual_profile != profile:
        raise ValueError("checkpoint/config hidden_profile mismatch")
    if [int(value) for value in manifest.get("d_in_profile", [])] != [int(config["d_in_profile"])]:
        raise ValueError("checkpoint/config d_in_profile mismatch")
    layers = manifest.get("hidden_layers", [])
    if len(layers) != len(profile) or not layers:
        raise ValueError("checkpoint hidden layer/profile count mismatch")
    if int(layers[0].get("input_dim", -1)) != feature_dim:
        raise ValueError("checkpoint input_dim mismatch")
    for layer, expected in zip(layers, profile):
        if (int(layer.get("head_count", -1)), int(layer.get("head_dim", -1))) != expected:
            raise ValueError("checkpoint hidden layer dimensions mismatch")
    if len(manifest.get("hidden_head_specs", [])) != sum(heads for heads, _ in profile):
        raise ValueError("checkpoint hidden head count mismatch")
    if len(manifest.get("output_head_specs", [])) != int(config["K_out"]):
        raise ValueError("checkpoint output head count mismatch")
    default_rule = "whole_graph_single" if graph_count == 1 else "multi_graph_batch"
    actual_rule = manifest.get("batching_rule", default_rule)
    actual_graphs = int(manifest.get("batch_graphs", 1 if actual_rule == "whole_graph_single" else graph_count))
    if actual_rule != config["batching_rule"] or actual_graphs != int(config["batch_graphs"]):
        raise ValueError("checkpoint/config batching mismatch")
    return {
        "dataset": dataset,
        "L": int(manifest["L"]),
        "hidden_profile": actual_profile,
        "d_in_profile": [int(value) for value in manifest["d_in_profile"]],
        "K_out": int(manifest["K_out"]),
        "batching_rule": actual_rule,
        "batch_graphs": actual_graphs,
    }


def current_source_identities(
    dataset: str,
    resource_manifest: dict[str, Any],
    *,
    graph_count: int,
    feature_dim: int,
    device: str | None = None,
) -> dict[str, Any]:
    """Read/hash current source bindings without executing data computation."""
    config_file = config_path(dataset)
    config = parse_config(config_file)
    validate_config_resources(dataset, config, resource_manifest)
    entry = resource_manifest["datasets"][dataset]
    checkpoint = ensure_directory(
        BASELINE_ROOT / entry["checkpoint_bundle"], label=f"{dataset} checkpoint"
    )
    manifest_file = ensure_regular_file(checkpoint / "manifest.json", label=f"{dataset} model manifest")
    tensors_file = ensure_regular_file(checkpoint / "tensors.txt", label=f"{dataset} model tensors")
    checkpoint_manifest = read_json(manifest_file, label=f"{dataset} model manifest")
    binding = _checkpoint_binding(config, checkpoint_manifest, graph_count, feature_dim)
    shape_payload = {
        "hidden_layers": checkpoint_manifest["hidden_layers"],
        "hidden_head_specs": checkpoint_manifest["hidden_head_specs"],
        "output_head_specs": checkpoint_manifest["output_head_specs"],
        "class_count": int(checkpoint_manifest["C"]),
        "feature_dim": feature_dim,
    }
    pp = pp_path_from_manifest(resource_manifest)
    validate_pp_file(pp, resource_manifest)
    registry = ensure_regular_file(
        BASELINE_ROOT / "artifacts" / "model_registry" / dataset / "model_registry.bin",
        label=f"{dataset} model registry",
    )
    result = {
        "config_sha256": config_digest(config_file),
        "protocol_sha256": protocol_digest(),
        "resource_manifest_id": str(resource_manifest["manifest_id"]),
        "resource_inventory_digest": resource_inventory_digest(dataset, resource_manifest),
        "checkpoint_manifest_id": f"sha256:{sha256_file(manifest_file)}",
        "checkpoint_manifest_sha256": sha256_file(manifest_file),
        "checkpoint_tensors_sha256": sha256_file(tensors_file),
        "model_shape_id": _json_id(shape_payload),
        "model_arch_id": checkpoint_manifest["model_arch_id"],
        "model_param_id": checkpoint_manifest["model_param_id"],
        "source_static_table_id": checkpoint_manifest["static_table_id"],
        "checkpoint_binding": binding,
        "source_identity": source_identity(),
        "registry_sha256": f"sha256:{sha256_file(registry)}",
        "registry_size_bytes": registry.stat().st_size,
        **pp_binding_from_manifest(resource_manifest),
    }
    if device is not None:
        if device not in DEVICES:
            raise ValueError(f"invalid build device: {device}")
        cpu_binary = ensure_regular_file(BASELINE_ROOT / "bin" / "zkdl-formal-cpu", label="CPU formal binary")
        prover_binary = ensure_regular_file(
            BASELINE_ROOT / "bin" / f"zkdl-formal-{device}", label=f"{device} formal binary"
        )
        cpu_build = load_build_manifest("cpu", cpu_binary)
        prover_build = load_build_manifest(device, prover_binary)
        result.update(
            {
                "cpu_binary_sha256": cpu_build["binary_sha256"],
                "prover_binary_sha256": prover_build["binary_sha256"],
                "cpu_binary_size_bytes": cpu_build["binary_size_bytes"],
                "prover_binary_size_bytes": prover_build["binary_size_bytes"],
                "cpu_build_identity": cpu_build["build_identity"],
                "prover_build_identity": prover_build["build_identity"],
                "build_identity": {
                    "cpu": cpu_build["build_identity"],
                    "prover": prover_build["build_identity"],
                },
                "cpu_build_manifest_sha256": sha256_file(build_manifest_path("cpu")),
                "prover_build_manifest_sha256": sha256_file(build_manifest_path(device)),
            }
        )
    return result


def _check_current_bindings(
    label: str,
    errors: list[str],
    obj: dict[str, Any],
    expected: dict[str, Any],
    prefix: str,
    *,
    required: bool = True,
) -> None:
    for key in CURRENT_BINDINGS:
        if (required or key in obj) and obj.get(key) != expected.get(key):
            errors.append(f"{label}:{prefix}_{key}")


def _telemetry_errors(label: str, device: str, proof: dict[str, Any], verifier: dict[str, Any], result: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    telemetry = proof.get("gpu_telemetry")
    if device == "cpu":
        if telemetry not in (None, {}):
            errors.append(f"{label}:cpu_gpu_telemetry_present")
        if verifier.get("gpu_telemetry") not in (None, {}):
            errors.append(f"{label}:cpu_verifier_gpu_telemetry_present")
        if result.get("gpu_telemetry") not in (None, {}):
            errors.append(f"{label}:cpu_result_gpu_telemetry_present")
        if verifier.get("gpu_telemetry_validated") is not True:
            errors.append(f"{label}:cpu_telemetry_boundary")
        return errors
    if not isinstance(telemetry, dict):
        return [f"{label}:gpu_telemetry_missing"]
    if verifier.get("gpu_telemetry") != telemetry or result.get("gpu_telemetry") != telemetry:
        errors.append(f"{label}:gpu_telemetry_provenance")
    if telemetry.get("provider_id") != "zkDL-native-CUDA-BLS12-381-v1":
        errors.append(f"{label}:gpu_provider")
    if telemetry.get("cross_backend_kat_passed") is not True:
        errors.append(f"{label}:gpu_kat")
    for key in (
        "msm_kernel_invocations",
        "field_inner_product_invocations",
        "scalar_fold_invocations",
        "point_fold_invocations",
        "sumcheck_kernel_invocations",
    ):
        try:
            valid = int(telemetry.get(key, 0)) > 0
        except (TypeError, ValueError):
            valid = False
        if not valid:
            errors.append(f"{label}:gpu_{key}")
    if verifier.get("gpu_telemetry_validated") is not True:
        errors.append(f"{label}:gpu_telemetry_not_validated")
    return errors


def check_cell(
    run: Path,
    dataset: str,
    device: str,
    resource_manifest: dict[str, Any],
    resource_manifest_id: str,
    *,
    current_identity: dict[str, Any] | None = None,
) -> list[str]:
    label = f"{dataset}/{device}"
    errors: list[str] = []
    try:
        result = read_json(run / "result.json", label=f"{label} result")
        preflight = read_json(run / "formal_artifacts" / "preflight.json", label=f"{label} preflight")
        relation = read_json(run / "formal_artifacts" / "canonical_relation.json", label=f"{label} canonical relation")
        proof = read_json(run / "proof_manifest.json", label=f"{label} prover manifest")
        verifier = read_json(run / "verifier_manifest.json", label=f"{label} verifier manifest")
        success = read_json(run / "success_manifest.json", label=f"{label} success manifest")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return [f"{label}:missing_or_malformed:{error}"]
    try:
        inventory = artifact_inventory(run, require_complete=True)
    except (OSError, ValueError) as error:
        return [f"{label}:artifact_inventory:{error}"]

    def require(condition: bool, name: str) -> None:
        if not condition:
            errors.append(f"{label}:{name}")

    require(result.get("dataset") == dataset and result.get("device") == device, "result_key")
    for manifest_name, manifest in (("result", result), ("proof", proof), ("verifier", verifier), ("success", success)):
        for field in private_publication_fields(manifest):
            errors.append(f"{label}:{manifest_name}_private_field:{field}")
    require(result.get("status") == "ok", "result_status")
    require(result.get("proof_generated") is True and proof.get("proof_generated") is True, "proof_generated")
    require(result.get("verified") is True and verifier.get("verified") is True, "verified")
    require(result.get("formal_experiment_eligible") is True, "formal_eligibility")
    require(preflight.get("status") == "accepted", "accepted_preflight")
    require(success.get("schema") == "zkDL_gat_cell_success_manifest_v1", "success_schema")
    require(success.get("status") == "verified", "success_status")
    require(success.get("dataset") == dataset and success.get("device") == device, "success_key")
    require(success.get("files") == {key: inventory[key] for key in inventory if key != "success_manifest.json"}, "success_inventory")
    require(verifier.get("independent_verifier") is True, "independent_verifier")
    require(verifier.get("verifier_reads_witness") is False, "verifier_reads_witness")
    require(proof.get("proof_format") == "zkDL-GAT-proof-envelope-v3", "proof_format")
    require(verifier.get("proof_format") == proof.get("proof_format"), "proof_format_match")
    for key in ("protocol_id", "statement_id", "model_registry_id", "device", "proof_sha256", "proof_size_bytes", "model_session_commitments", "model_rerandomization_proofs"):
        require(proof.get(key) == verifier.get(key), f"proof_verifier_{key}")
    for key in ("statement_id", "model_registry_id", "device", "proof_sha256", "proof_size_bytes", "proof_format"):
        require(proof.get(key) == result.get(key), f"result_{key}")
    proof_file = run / "formal_artifacts" / "proof.bin"
    require(proof_file.is_file(), "proof_file_present")
    if proof_file.is_file():
        try:
            declared_size = int(proof.get("proof_size_bytes", -1))
        except (TypeError, ValueError):
            declared_size = -1
        require(proof_file.stat().st_size == declared_size, "proof_size")
        require(sha256_file(proof_file) == proof.get("proof_sha256"), "proof_digest")
    errors.extend(_telemetry_errors(label, device, proof, verifier, result))
    if device == "cpu":
        require(success.get("gpu_telemetry") in (None, {}), "cpu_success_gpu_telemetry_present")
    else:
        require(success.get("gpu_telemetry") == proof.get("gpu_telemetry"), "gpu_success_telemetry_provenance")

    ledger: dict[str, Any] | None = None
    profile: dict[str, Any] | None = None
    try:
        public = _public_instance(relation)
        expected_topology = _topology_digest(relation, public)
        expected_features = _feature_digest(relation, public)
        ledger_value = preflight["ledger"]
        profile_value = preflight["static_tables"]
        if not isinstance(ledger_value, dict) or not isinstance(profile_value, dict):
            raise ValueError("preflight identity object is malformed")
        ledger = ledger_value
        profile = profile_value
        static_profile_material = dict(profile)
        static_profile_material.pop("static_tables_id", None)
        static_profile_id = _json_id(static_profile_material, domain=b"zkDL-GAT-static-table-profile-v1\0")
        relation_rows_id = _table_rows_digest(relation.get("operation_families"))
        ledger_material = dict(ledger)
        ledger_material.pop("integer_bound_ledger_id", None)
        expected_ledger_id = _json_id(ledger_material)
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        errors.append(f"{label}:static_identity_reconstruction:{error}")
    else:
        require(relation.get("schema") == "zkDL_gat_canonical_adapter_v2", "relation_schema")
        require(relation.get("dataset") == dataset, "relation_dataset")
        require(relation.get("integer_bound_ledger") == ledger, "relation_ledger_copy")
        require(relation.get("static_table_profile") == profile, "relation_static_profile")
        for key, value in public.items():
            require(relation.get(key) == value, f"relation_{key}")
            require(preflight.get(key) == value, f"preflight_{key}")
        require(preflight.get("public_instance") == public, "preflight_public_instance")
        require(result.get("public_instance") == public, "result_public_instance")
        require(success.get("public_instance") == public, "success_public_instance")
        require(relation.get("topology_digest") == f"sha256:{expected_topology}", "relation_topology_digest")
        require(preflight.get("topology_digest") == f"sha256:{expected_topology}", "preflight_topology_digest")
        require(relation.get("graph_features_digest") == f"sha256:{expected_features}", "relation_features_digest")
        require(preflight.get("graph_features_digest") == f"sha256:{expected_features}", "preflight_features_digest")
        require(relation.get("static_tables_id") == static_profile_id, "relation_static_tables_id")
        require(preflight.get("static_tables_id") == static_profile_id, "preflight_static_tables_id")
        require(relation.get("static_table_rows_id") == relation_rows_id, "relation_static_table_rows_id")
        require(preflight.get("static_table_rows_id") == relation_rows_id, "preflight_static_table_rows_id")
        require(ledger.get("integer_bound_ledger_id") == expected_ledger_id, "ledger_digest")
        require(relation.get("integer_bound_ledger_id") == expected_ledger_id, "relation_ledger_id")
        table_domains = ledger.get("table_domains", {})
        for key in ("lrelu", "elu", "exp"):
            table = profile.get(key, {})
            require(table_domains.get(key) == [table.get("min"), table.get("max")], f"ledger_{key}_domain")

    try:
        if current_identity is None:
            current_identity = current_source_identities(
                dataset,
                resource_manifest,
                graph_count=int(preflight["graph_count"]),
                feature_dim=int(preflight["feature_dim"]),
                device=device,
            )
    except (OSError, KeyError, TypeError, ValueError) as error:
        errors.append(f"{label}:current_source_identity:{error}")
        current_identity = None
    if current_identity is not None:
        require(current_identity.get("resource_manifest_id") == resource_manifest_id, "current_resource_manifest_id")
        _check_current_bindings(label, errors, preflight, current_identity, "preflight")
        _check_current_bindings(label, errors, relation, current_identity, "relation", required=False)
        _check_current_bindings(label, errors, result, current_identity, "result")
        _check_current_bindings(label, errors, success, current_identity, "success")
        for manifest_name, manifest in (
            ("proof", proof),
            ("verifier", verifier),
            ("result", result),
            ("success", success),
        ):
            for key in PP_FIELDS:
                require(manifest.get(key) == current_identity.get(key), f"{manifest_name}_{key}")
        require(result.get("registry_sha256") == current_identity.get("registry_sha256"), "result_registry_sha256")
        require(result.get("registry_size_bytes") == current_identity.get("registry_size_bytes"), "result_registry_size_bytes")
        require(success.get("registry_sha256") == current_identity.get("registry_sha256"), "success_registry_sha256")
        require(success.get("registry_size_bytes") == current_identity.get("registry_size_bytes"), "success_registry_size_bytes")
        require(result.get("source_identity") == current_identity.get("source_identity"), "result_source_identity")
        require(success.get("source_identity") == current_identity.get("source_identity"), "success_source_identity")
        require(result.get("build_identity") == current_identity.get("build_identity"), "result_build_identity")
        require(success.get("build_identity") == current_identity.get("build_identity"), "success_build_identity")
        provenance = result.get("provenance")
        success_provenance = success.get("provenance")
        require(isinstance(provenance, dict), "provenance_object")
        require(isinstance(success_provenance, dict), "success_provenance_object")
        if isinstance(provenance, dict):
            _check_current_bindings(label, errors, provenance, current_identity, "provenance")
        if isinstance(success_provenance, dict):
            require(success_provenance == provenance, "success_provenance_match")
        for key in ("checkpoint_manifest_id", "model_shape_id", "model_arch_id", "model_param_id"):
            require(preflight.get(key) == current_identity.get(key), f"preflight_{key}")
            require(relation.get(key) == current_identity.get(key), f"relation_{key}")
            require(ledger is not None and ledger.get(key) == current_identity.get(key), f"ledger_{key}")
            require(result.get(key) == current_identity.get(key), f"result_{key}")
            require(success.get(key) == current_identity.get(key), f"success_{key}")
        if ledger is not None:
            for key in ("config_sha256", "resource_manifest_id", "checkpoint_manifest_id"):
                require(ledger.get(key) == current_identity.get(key), f"ledger_{key}")
        require(preflight.get("checkpoint_manifest_sha256") == current_identity.get("checkpoint_manifest_sha256"), "checkpoint_manifest_sha256")
        require(preflight.get("checkpoint_tensors_sha256") == current_identity.get("checkpoint_tensors_sha256"), "checkpoint_tensors_sha256")
        require(preflight.get("checkpoint_binding") == current_identity.get("checkpoint_binding"), "checkpoint_binding")
        if profile is not None:
            require(profile.get("source_static_table_id") == current_identity.get("source_static_table_id"), "source_static_table_id")
        for key in ("graph_count", "feature_dim", "class_count", "static_tables_id", "static_table_rows_id"):
            require(relation.get(key) == preflight.get(key), f"relation_preflight_{key}")

    provenance = result.get("provenance")
    if isinstance(provenance, dict):
        for key, path in (
            ("preflight_sha256", run / "formal_artifacts" / "preflight.json"),
            ("canonical_relation_sha256", run / "formal_artifacts" / "canonical_relation.json"),
            ("statement_sha256", run / "formal_artifacts" / "statement.bin"),
            ("proof_manifest_sha256", run / "proof_manifest.json"),
            ("verifier_manifest_sha256", run / "verifier_manifest.json"),
        ):
            require(provenance.get(key) == sha256_file(path), f"provenance_{key}")
    try:
        commitment_count = int(proof.get("model_session_commitments", 0))
        rerandomization_count = int(proof.get("model_rerandomization_proofs", 0))
    except (TypeError, ValueError):
        commitment_count = rerandomization_count = 0
    require(
        result.get("privacy_profile_status") == "supported_multi_session_registered_model"
        and result.get("model_session_rerandomization") == "pedersen_blinder_schnorr_v1"
        and verifier.get("model_session_rerandomization") == "pedersen_blinder_schnorr_v1"
        and verifier.get("model_rerandomization_verified") is True
        and commitment_count == rerandomization_count > 0,
        "privacy_profile",
    )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=BASELINE_ROOT / "runs")
    parser.add_argument("--output", type=Path, default=BASELINE_ROOT / "results" / "local_validation.json")
    args = parser.parse_args()
    runs = ensure_contained(args.runs, label="runs root")
    output = ensure_contained(args.output, label="validation output")
    resource_manifest, resource_manifest_id = load_resource_manifest(require_staged=True)
    errors: list[str] = []
    verified_rows = 0
    for dataset in DATASETS:
        for device in DEVICES:
            run = runs / dataset / device / "warm"
            try:
                preflight = read_json(run / "formal_artifacts" / "preflight.json", label=f"{dataset}/{device} preflight")
                identity = current_source_identities(
                    dataset,
                    resource_manifest,
                    graph_count=int(preflight["graph_count"]),
                    feature_dim=int(preflight["feature_dim"]),
                    device=device,
                )
            except (OSError, KeyError, TypeError, ValueError) as error:
                identity = None
                errors.append(f"{dataset}/{device}:current_source_identity:{error}")
            cell_errors = check_cell(
                run,
                dataset,
                device,
                resource_manifest,
                resource_manifest_id,
                current_identity=identity,
            )
            errors.extend(cell_errors)
            if not cell_errors:
                verified_rows += 1
    audit = {
        "schema": "zkDL_gat_local_validation_v3",
        "baseline": "zkdl",
        "verified_rows": verified_rows,
        "expected_rows": 10,
        "computational_validation": "passed" if not errors else "failed",
        "privacy_profile_status": "supported_multi_session_registered_model",
        "formal_experiment_eligible": not errors and verified_rows == 10,
        "status": "complete" if not errors and verified_rows == 10 else "failed",
        "errors": errors,
    }
    atomic_write_json(output, audit)
    if errors:
        print(f"zkdl_final_validation=FAILED errors={len(errors)}")
        return 1
    print("zkdl_computational_validation=OK rows=10 privacy_profile=SUPPORTED_MULTI_SESSION")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
