#!/usr/bin/env python3
"""Print source-derived five-dataset zkDL capacity bounds without preparation.

This tool invokes only ``gat_runner resource-plan`` and feeds its typed
ANALYSIS_ONLY records through the same Python phase-bound equations used by the
formal runner.  It never creates SRS material, prepares a dataset, admits a
formal run, or proves/verifies a statement.
"""
from __future__ import annotations

import json
import hashlib
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import formal_phase
import formal_runner
import formal_provenance
from prepare_form_resources import audit_formal_prerequisites

ROOT = Path(__file__).resolve().parent.parent
DATASETS = formal_phase.DATASETS
MIB = 1 << 20


class ResourceReportError(RuntimeError):
    pass


PLANNED_SHAPE_SCHEMA = "zkdl_analysis_only_planned_shape_v1"
PLANNED_SHAPE_FIELDS = {"schema", "status", "dataset", "config_sha256", "scale_chain_sha256", "input_identity", "graph", "checkpoint", "table_plan", "sorted_lookup_plan", "family_domains", "family_domain_count", "family_domain_sha256", "srs_plan", "ledger_plan", "analysis_source_identity", "prepared_resources_claimed", "authenticated_resource_digests_present"}
INPUT_IDENTITY_FIELDS = {"metadata_sha256", "edge_file_sha256", "node_ptr_file_sha256", "checkpoint_manifest_sha256", "checkpoint_tensor_export_sha256"}


def _config_values(config: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in config.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def _planned_input_paths(config: Path, dataset: str) -> dict[str, Path | None]:
    values = _config_values(config)
    cache_root = Path(values.get("cache_root", "data/cache"))
    checkpoint_bundle = Path(values.get("checkpoint_bundle", ""))
    if (cache_root.is_absolute() or ".." in cache_root.parts or checkpoint_bundle.is_absolute() or
            not checkpoint_bundle.parts or ".." in checkpoint_bundle.parts):
        raise ResourceReportError(f"{dataset}: planned input paths must remain baseline-local")
    graph_root = ROOT / cache_root / dataset
    checkpoint_root = ROOT / checkpoint_bundle
    return {"metadata_sha256": graph_root / "meta.cfg", "edge_file_sha256": graph_root / "edges.txt", "node_ptr_file_sha256": graph_root / "node_ptr.txt" if dataset == "ppi" else None, "checkpoint_manifest_sha256": checkpoint_root / "manifest.json", "checkpoint_tensor_export_sha256": checkpoint_root / "tensors.txt", "scale_chain_sha256": ROOT / "configs" / "quantization_scale_chain.cfg"}


def _hash_planned_inputs(paths: dict[str, Path | None]) -> dict[str, str | None]:
    for key, path in paths.items():
        if path is not None and (path.is_symlink() or not path.is_file()):
            raise ResourceReportError(f"planned input {key} is missing or symlinked: {path}")
    return {key: formal_phase.sha256_file(path) if path is not None else None for key, path in paths.items()}


def _require_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ResourceReportError(f"planned metadata has unknown or missing {label} fields")
    return value


def _uint128_decimal(value: Any, label: str) -> int:
    if not isinstance(value, str) or not re.fullmatch(r"(?:0|[1-9][0-9]*)", value):
        raise ResourceReportError(f"{label} must be a canonical unsigned decimal string")
    parsed = int(value)
    if parsed > (1 << 128) - 1:
        raise ResourceReportError(f"{label} exceeds uint128")
    return parsed


def _build_planned_shape(planned: dict[str, Any], dataset: str) -> dict[str, Any]:
    checkpoint = planned.get("checkpoint")
    graph = planned.get("graph")
    if not isinstance(checkpoint, dict) or not isinstance(graph, dict):
        raise ResourceReportError(f"{dataset}: canonical preflight omitted graph/checkpoint shape metadata")
    planned_shape = {}
    planned_shape["schema"] = PLANNED_SHAPE_SCHEMA
    planned_shape["status"] = "ANALYSIS_ONLY_PLANNED"
    planned_shape["dataset"] = dataset
    planned_shape["config_sha256"] = planned["config_sha256"]
    planned_shape["scale_chain_sha256"] = planned["quantization_scale_chain_sha256"]
    planned_shape["input_identity"] = {"metadata_sha256": graph["meta_sha256"], "edge_file_sha256": graph["edge_file_sha256"], "node_ptr_file_sha256": graph["node_ptr_file_sha256"], "checkpoint_manifest_sha256": checkpoint["manifest_sha256"], "checkpoint_tensor_export_sha256": checkpoint["tensor_export_sha256"]}
    planned_shape["graph"] = {"nodes": graph["nodes"], "normalized_edges": graph["normalized_edges"], "graphs": graph["graphs"], "raw_edge_rows": graph["raw_edge_rows"], "csr_max_in_degree": graph["csr_max_in_degree"], "features": planned["input_shape"]["features"], "classes": planned["input_shape"]["classes"]}
    planned_shape["checkpoint"] = {"tensor_count": checkpoint["tensor_count"], "max_quantized_model_abs": checkpoint["max_quantized_model_abs"], "model_shapes": checkpoint["model_shapes"], "model_padded_domains": checkpoint["model_padded_domains"]}
    planned_shape["table_plan"] = planned["table_plan"]
    planned_shape["sorted_lookup_plan"] = planned["sorted_lookup_plan"]
    planned_shape["family_domains"] = planned["family_domains"]
    planned_shape["family_domain_count"] = planned["family_domain_count"]
    planned_shape["family_domain_sha256"] = planned["family_domain_sha256"]
    planned_shape["srs_plan"] = planned["srs_plan"]
    planned_shape["ledger_plan"] = planned["ledger_plan"]
    planned_shape["analysis_source_identity"] = {"planner_sha256": formal_phase.sha256_file(Path(__file__).resolve().parent / "prepare_form_resources.py"), "resource_report_sha256": formal_phase.sha256_file(Path(__file__).resolve()), "formal_phase_sha256": formal_phase.sha256_file(Path(formal_phase.__file__).resolve()), "formal_runner_sha256": formal_phase.sha256_file(Path(formal_runner.__file__).resolve())}
    planned_shape["prepared_resources_claimed"] = False
    planned_shape["authenticated_resource_digests_present"] = False
    _validate_planned_shape(planned_shape, dataset)
    return planned_shape


def _validate_planned_shape(value: Any, dataset: str) -> None:
    shape = _require_keys(value, PLANNED_SHAPE_FIELDS, "top-level")
    if (shape.get("schema") != PLANNED_SHAPE_SCHEMA or shape.get("status") != "ANALYSIS_ONLY_PLANNED" or
            shape.get("dataset") != dataset or shape.get("prepared_resources_claimed") is not False or
            shape.get("authenticated_resource_digests_present") is not False):
        raise ResourceReportError(f"{dataset}: planned metadata schema/status/side-effect flags are invalid")
    for field in ("config_sha256", "scale_chain_sha256", "family_domain_sha256"):
        if not isinstance(shape.get(field), str) or not re.fullmatch(r"[0-9a-f]{64}", shape[field]):
            raise ResourceReportError(f"{dataset}: planned metadata has an invalid {field}")
    identity = _require_keys(shape.get("input_identity"), INPUT_IDENTITY_FIELDS, "input identity")
    for field, digest in identity.items():
        if field == "node_ptr_file_sha256" and dataset != "ppi":
            if digest is not None:
                raise ResourceReportError(f"{dataset}: non-PPI planned metadata unexpectedly binds node_ptr")
        elif not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ResourceReportError(f"{dataset}: planned metadata has an invalid input identity {field}")
    graph = _require_keys(shape.get("graph"), {"nodes", "normalized_edges", "graphs", "raw_edge_rows", "csr_max_in_degree", "features", "classes"}, "graph")
    if any(type(graph[field]) is not int or graph[field] < 0 for field in graph):
        raise ResourceReportError(f"{dataset}: planned graph fields must be nonnegative integers")
    if not graph["nodes"] or not graph["normalized_edges"] or not graph["graphs"] or not graph["features"] or not graph["classes"]:
        raise ResourceReportError(f"{dataset}: planned graph dimensions must be positive")
    checkpoint = _require_keys(shape.get("checkpoint"), {"tensor_count", "max_quantized_model_abs", "model_shapes", "model_padded_domains"}, "checkpoint")
    if (type(checkpoint["tensor_count"]) is not int or checkpoint["tensor_count"] <= 0 or
            type(checkpoint["max_quantized_model_abs"]) is not int or checkpoint["max_quantized_model_abs"] < 0 or
            not isinstance(checkpoint["model_shapes"], dict) or not isinstance(checkpoint["model_padded_domains"], list) or
            any(type(domain) is not int or domain <= 0 for domain in checkpoint["model_padded_domains"])):
        raise ResourceReportError(f"{dataset}: planned checkpoint fields have invalid types or values")
    for name, axes in checkpoint["model_shapes"].items():
        if (not isinstance(name, str) or not name or not isinstance(axes, list) or not axes or
                any(type(axis) is not int or axis <= 0 for axis in axes)):
            raise ResourceReportError(f"{dataset}: planned model tensor shape is malformed")
    table = _require_keys(shape.get("table_plan"), {"schema", "quant_cfg_id", "generator_id", "version", "scale_id", "domain_id", "row_order", "padding_rule", "static_table_id", "lrelu_padded_domain", "elu_padded_domain", "exp_padded_domain", "lrelu_min", "lrelu_max", "elu_min", "elu_max", "exp_max", "spec_hash", "lrelu_rows", "elu_rows", "exp_rows", "lrelu_value_min", "lrelu_value_max", "elu_value_min", "elu_value_max", "exp_value_min", "exp_value_max"}, "table plan")
    if table.get("schema") != "zkDL_gat_static_table_descriptor_v1":
        raise ResourceReportError(f"{dataset}: planned static-table schema is invalid")
    for field in ("quant_cfg_id", "generator_id", "version", "scale_id", "domain_id", "row_order", "padding_rule", "static_table_id"):
        if not isinstance(table.get(field), str) or not table[field]:
            raise ResourceReportError(f"{dataset}: planned static-table {field} is invalid")
    for field in ("spec_hash",):
        if not isinstance(table.get(field), str) or not re.fullmatch(r"[0-9a-f]{64}", table[field]):
            raise ResourceReportError(f"{dataset}: planned static-table {field} is invalid")
    for field in ("lrelu_padded_domain", "elu_padded_domain", "exp_padded_domain", "lrelu_min", "lrelu_max", "elu_min", "elu_max", "exp_max", "lrelu_rows", "elu_rows", "exp_rows", "lrelu_value_min", "lrelu_value_max", "elu_value_min", "elu_value_max", "exp_value_min", "exp_value_max"):
        if type(table.get(field)) is not int:
            raise ResourceReportError(f"{dataset}: planned static-table {field} is not an integer")
    if any(table[field] <= 0 for field in ("lrelu_rows", "elu_rows", "exp_rows", "lrelu_padded_domain", "elu_padded_domain", "exp_padded_domain")):
        raise ResourceReportError(f"{dataset}: planned static-table row/domain shapes must be positive")
    domains = shape.get("family_domains")
    if (not isinstance(domains, dict) or not domains or any(not isinstance(name, str) or not name or
            type(domain) is not int or domain <= 0 for name, domain in domains.items()) or
            type(shape.get("family_domain_count")) is not int or shape["family_domain_count"] != len(domains)):
        raise ResourceReportError(f"{dataset}: planned canonical family domains are malformed")
    domain_digest = hashlib.sha256(json.dumps(domains, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
    if shape["family_domain_sha256"] != domain_digest:
        raise ResourceReportError(f"{dataset}: planned canonical family-domain digest is inconsistent")
    srs = _require_keys(shape.get("srs_plan"), {"basis_domain", "frame_bytes", "families", "max_segment_count", "estimated_materialized_bytes", "available_bytes"}, "SRS plan")
    if any(type(srs[field]) is not int or srs[field] < 0 for field in ("basis_domain", "frame_bytes", "max_segment_count", "estimated_materialized_bytes", "available_bytes")):
        raise ResourceReportError(f"{dataset}: analytic SRS plan has invalid scalar fields")
    if not isinstance(srs, dict) or not isinstance(srs.get("families"), dict) or set(srs["families"]) != set(domains):
        raise ResourceReportError(f"{dataset}: analytic SRS shapes do not cover the planned families")
    for family, domain in domains.items():
        plan = srs["families"].get(family)
        if (not isinstance(plan, dict) or set(plan) != {"logical_domain", "segment_size", "segment_count"} or
                plan.get("logical_domain") != domain or
                type(plan.get("segment_size")) is not int or plan["segment_size"] <= 0 or
                type(plan.get("segment_count")) is not int or
                plan["segment_count"] != (domain + plan["segment_size"] - 1) // plan["segment_size"]):
            raise ResourceReportError(f"{dataset}: malformed analytic SRS family shape for {family}")
    ledger = _require_keys(shape.get("ledger_plan"), {"schema", "ledger_id", "degree_bound_id", "quant_cfg_id", "static_table_id", "field_modulus", "scale_chain_sha256", "maxima", "max_family_aux_abs", "max_field_product_abs", "families"}, "ledger plan")
    if ledger.get("schema") != "zkDL_gat_integer_bound_ledger_v1":
        raise ResourceReportError(f"{dataset}: planned integer-bound ledger schema is invalid")
    for field in ("ledger_id", "degree_bound_id", "quant_cfg_id", "static_table_id", "field_modulus"):
        if not isinstance(ledger.get(field), str) or not ledger[field]:
            raise ResourceReportError(f"{dataset}: planned integer-bound ledger {field} is invalid")
    if (not isinstance(ledger.get("scale_chain_sha256"), str) or
            ledger["scale_chain_sha256"] != shape["scale_chain_sha256"]):
        raise ResourceReportError(f"{dataset}: planned integer-bound ledger scale identity is inconsistent")
    if not isinstance(ledger, dict) or not isinstance(ledger.get("families"), list) or not ledger["families"]:
        raise ResourceReportError(f"{dataset}: planned integer-bound ledger families are missing")
    family_ids = [item.get("family_id") for item in ledger["families"] if isinstance(item, dict)]
    if len(family_ids) != len(ledger["families"]) or any(not isinstance(item, str) or not item for item in family_ids) or len(set(family_ids)) != len(family_ids):
        raise ResourceReportError(f"{dataset}: planned integer-bound ledger family ids are malformed")
    ledger_fields = {"family_id", "layer_index", "head_index", "output", "max_input_abs", "max_weight_abs", "max_attention_abs", "max_bias_abs", "max_score_abs", "max_table_output_abs", "max_delta", "max_exp", "max_sum", "max_aggregate_num", "max_dividend_abs", "max_aux_abs", "max_divisor", "max_field_product_abs"}
    for family in ledger["families"]:
        _require_keys(family, ledger_fields, "ledger family")
        if any(type(family[field]) is not int or family[field] < 0 for field in ledger_fields - {"family_id"}):
            raise ResourceReportError(f"{dataset}: planned integer-bound family fields are invalid")
    if not isinstance(ledger.get("maxima"), dict) or not ledger["maxima"]:
        raise ResourceReportError(f"{dataset}: planned integer-bound ledger maxima are missing")
    if any(type(value) is not int or value < 0 for value in ledger["maxima"].values()):
        raise ResourceReportError(f"{dataset}: planned integer-bound ledger maxima are invalid")
    for field in ("max_family_aux_abs", "max_field_product_abs"):
        if type(ledger.get(field)) is not int or ledger[field] < 0:
            raise ResourceReportError(f"{dataset}: planned integer-bound ledger {field} is invalid")
    lookup = _require_keys(shape.get("sorted_lookup_plan"), {"segment_size", "table_families", "query_families", "query_count", "query_rows_total", "table_domain_total", "query_domain_total", "merge_domain_total", "scratch_record_payload_bytes_total", "scratch_bytes_total", "scratch_bytes_max_single_file"}, "sorted-lookup plan")
    if (not isinstance(lookup, dict) or not isinstance(lookup.get("table_families"), list) or
            not isinstance(lookup.get("query_families"), list) or
            lookup.get("query_count") != len(lookup["query_families"])):
        raise ResourceReportError(f"{dataset}: analytic sorted-lookup query plan is malformed")
    if len(lookup["table_families"]) != 3 or not lookup["query_families"]:
        raise ResourceReportError(f"{dataset}: analytic sorted-lookup table/query families are incomplete")
    table_family_fields = {"table_ref", "operation", "logical_rows", "padded_domain", "sorted_unique", "input_min", "input_max", "output_min", "output_max", "x_bits", "y_bits", "key_encoding", "key_max_u128", "padding_rule"}
    table_by_operation: dict[str, dict[str, Any]] = {}
    for family in lookup["table_families"]:
        _require_keys(family, table_family_fields, "sorted-lookup table family")
        if (not isinstance(family["operation"], str) or family["operation"] in table_by_operation or
                type(family["logical_rows"]) is not int or family["logical_rows"] <= 0 or
                family["padded_domain"] != 1 << (family["logical_rows"] - 1).bit_length()):
            raise ResourceReportError(f"{dataset}: sorted-lookup table family shape is invalid")
        _uint128_decimal(family["key_max_u128"], f"{dataset}: sorted-lookup table key maximum")
        table_by_operation[family["operation"]] = family
    query_fields = {"operation_id", "operation", "table_ref", "table_rows", "query_rows", "query_shape", "table_domain", "query_domain", "merge_rows", "merge_domain", "segment_size", "table_segment_count", "query_segment_count", "merge_segment_count", "x_bits", "y_bits", "key_max_u128", "scratch_header_bytes", "scratch_record_payload_bytes", "scratch_bytes_upper_bound"}
    for query in lookup["query_families"]:
        _require_keys(query, query_fields, "sorted-lookup query family")
        if (not isinstance(query.get("operation_id"), str) or not query["operation_id"] or
                not isinstance(query.get("operation"), str) or query["operation"] not in table_by_operation or
                type(query.get("query_rows")) is not int or query["query_rows"] <= 0):
            raise ResourceReportError(f"{dataset}: analytic sorted-lookup query rows are invalid")
        query_shape = query.get("query_shape")
        if (not isinstance(query_shape, list) or not query_shape or
                any(type(axis) is not int or axis <= 0 for axis in query_shape) or
                math.prod(query_shape) != query["query_rows"]):
            raise ResourceReportError(f"{dataset}: analytic sorted-lookup query shape is invalid")
        for field in ("table_domain", "query_domain", "merge_domain", "x_bits", "y_bits"):
            if type(query.get(field)) is not int or query[field] <= 0:
                raise ResourceReportError(f"{dataset}: analytic sorted-lookup {field} is invalid")
        table_family = table_by_operation[query["operation"]]
        if (query["table_rows"] != table_family["logical_rows"] or
                query["table_domain"] != table_family["padded_domain"] or
                query["query_domain"] != 1 << (query["query_rows"] - 1).bit_length() or
                query["merge_rows"] != query["table_rows"] + query["query_rows"] or
                query["merge_domain"] != 1 << (query["merge_rows"] - 1).bit_length() or
                query["x_bits"] != table_family["x_bits"] or query["y_bits"] != table_family["y_bits"] or
                _uint128_decimal(query["key_max_u128"], f"{dataset}: sorted-lookup query key maximum") !=
                _uint128_decimal(table_family["key_max_u128"], f"{dataset}: sorted-lookup table key maximum")):
            raise ResourceReportError(f"{dataset}: sorted-lookup query/table shape binding is inconsistent")
    source_identity = _require_keys(shape.get("analysis_source_identity"), {"planner_sha256", "resource_report_sha256", "formal_phase_sha256", "formal_runner_sha256"}, "analysis source identity")
    for field, digest in source_identity.items():
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ResourceReportError(f"{dataset}: planned metadata has an invalid analysis source {field}")


def _ppi_node_ptr_source(config: Path) -> tuple[Path, str] | None:
    if config.stem != "ppi_secure":
        return None
    values: dict[str, str] = {}
    for line in config.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip()
    raw_cache = values.get("cache_root")
    if not raw_cache:
        raise ResourceReportError("PPI config lacks cache_root for node_ptr provenance")
    cache_root = Path(raw_cache)
    if cache_root.is_absolute() or ".." in cache_root.parts:
        raise ResourceReportError("PPI cache_root is not baseline-local")
    path = ROOT / cache_root / "ppi" / "node_ptr.txt"
    if path.is_symlink() or not path.is_file():
        raise ResourceReportError(f"PPI graph ownership metadata is missing or symlinked: {path}")
    return path, formal_phase.sha256_file(path)


def _resource_plan(runner: Path, dataset: str) -> dict[str, Any]:
    config = ROOT / "configs" / f"{dataset}_secure.cfg"
    ppi_node_ptr = _ppi_node_ptr_source(config)
    official_config_bytes = config.read_bytes()
    input_paths = _planned_input_paths(config, dataset)
    input_hashes_before = _hash_planned_inputs(input_paths)
    planned = audit_formal_prerequisites(config)
    if planned.get("dataset") != dataset or planned.get("config_sha256") != hashlib.sha256(official_config_bytes).hexdigest():
        raise ResourceReportError(f"{dataset}: canonical preflight identity does not bind the official config")
    checkpoint = planned.get("checkpoint")
    graph = planned.get("graph")
    if not isinstance(checkpoint, dict) or not isinstance(graph, dict):
        raise ResourceReportError(f"{dataset}: canonical preflight omitted graph/checkpoint shape metadata")
    planned_shape = _build_planned_shape(planned, dataset)
    expected_input_hashes = dict(planned_shape["input_identity"])
    expected_input_hashes["scale_chain_sha256"] = planned_shape["scale_chain_sha256"]
    if input_hashes_before != expected_input_hashes:
        raise ResourceReportError(f"{dataset}: source inputs changed while analytic metadata was built")
    # Keep the explicit top-level identity fields first. The native resource-plan
    # reader scopes nested fragments but uses the shared string reader for the
    # top-level schema; sorting would put ledger_plan.schema ahead of it.
    planned_shape_bytes = json.dumps(planned_shape, sort_keys=False, separators=(",", ":")).encode("utf-8") + b"\n"
    planned_shape = json.loads(planned_shape_bytes)
    _validate_planned_shape(planned_shape, dataset)
    planned_shape_bytes = json.dumps(planned_shape, sort_keys=False, separators=(",", ":")).encode("utf-8") + b"\n"
    sidecar_path: Path | None = None
    try:
        with tempfile.TemporaryDirectory(prefix=f"zkdl-analysis-only-planned-{dataset}-") as temporary:
            sidecar_path = Path(temporary) / "planned-shape.json"
            sidecar_path.write_bytes(planned_shape_bytes)
            os.chmod(sidecar_path, 0o600)
            completed = subprocess.run(
                [str(runner), "resource-plan", "--config", str(config), "--planned-shape", str(sidecar_path)],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
                timeout=180,
            )
            if completed.returncode != 0:
                detail = completed.stderr.strip()[-4000:]
                raise ResourceReportError(f"{dataset}: planned-shape resource-plan exited {completed.returncode}: {detail}")
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ResourceReportError(f"{dataset}: resource-plan could not complete: {exc}") from exc
    if config.read_bytes() != official_config_bytes:
        raise ResourceReportError(f"{dataset}: official config changed during analysis-only resource-plan")
    if _hash_planned_inputs(input_paths) != input_hashes_before:
        raise ResourceReportError(f"{dataset}: graph/model/quantization input changed during analysis-only resource-plan")
    records = []
    for line in completed.stdout.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("schema") == "zkdl_analysis_only_resource_plan_v1":
            records.append(value)
    if len(records) != 1:
        raise ResourceReportError(f"{dataset}: expected exactly one ANALYSIS_ONLY resource-plan record")
    record = records[0]
    if (record.get("status") != "ANALYSIS_ONLY_PLANNED" or record.get("dataset") != dataset or
            record.get("dataset_prepared") is not False or
            record.get("formal_admission_run") is not False or
            record.get("proof_artifact_created") is not False or
            record.get("independent_verification_run") is not False or
            record.get("prepared_resources_claimed") is not False or
            record.get("authenticated_resource_digests_present") is not False):
        raise ResourceReportError(f"{dataset}: resource-plan status/side-effect boundary is invalid")
    if any(key != "authenticated_resource_digests_present" and
           (("auth" in key.lower() and "digest" in key.lower()) or key in
           {"resource_manifest_digest", "srs_manifest_digest", "build_manifest_digest", "proof_digest", "result_digest"}
           ) for key in record):
        raise ResourceReportError(f"{dataset}: ANALYSIS_ONLY record contains an authentication/proof digest field")
    config_digest = hashlib.sha256(official_config_bytes).hexdigest()
    planned_digest = hashlib.sha256(planned_shape_bytes).hexdigest()
    if record.get("config_sha256") != config_digest or record.get("planned_shape_sha256") != planned_digest:
        raise ResourceReportError(f"{dataset}: resource-plan did not bind official config and planned metadata")
    record["official_config_sha256"] = config_digest
    record["planned_shape_sha256"] = planned_digest
    record["planned_metadata_status"] = "ANALYSIS_ONLY_PLANNED"
    source_paths = {
        "gat_runner_source_sha256": ROOT / "gat_adapter" / "gat_runner.cu",
        "analysis_resource_report_sha256": Path(__file__).resolve(),
        "formal_phase_source_sha256": Path(formal_phase.__file__).resolve(),
        "formal_runner_source_sha256": Path(formal_runner.__file__).resolve(),
        "native_bridge_header_sha256": ROOT / "gat_adapter" / "gat_native_bridge.hpp",
        "native_core_source_sha256": ROOT / "gat_adapter" / "gat_native_core.cu",
        "native_core_header_sha256": ROOT / "gat_adapter" / "gat_native_core.cuh",
        "relation_proof_header_sha256": ROOT / "zkrelation-proof.cuh",
    }
    for field, path in source_paths.items():
        if record.get(field) != formal_phase.sha256_file(path):
            raise ResourceReportError(f"{dataset}: ANALYSIS_ONLY binary/source hash mismatch for {field}")
    if ppi_node_ptr is not None:
        node_ptr_path, original_digest = ppi_node_ptr
        final_digest = formal_phase.sha256_file(node_ptr_path)
        if final_digest != original_digest:
            raise ResourceReportError("PPI node_ptr.txt changed while resource-plan was running")
        record["node_ptr_file_sha256"] = final_digest
    else:
        record["node_ptr_file_sha256"] = None
    if record.get("metadata_sha256") != graph["meta_sha256"] or record.get("raw_edge_file_sha256") != graph["edge_file_sha256"]:
        raise ResourceReportError(f"{dataset}: native graph source identities differ from the planned metadata")
    if record.get("checkpoint_manifest_sha256") != checkpoint["manifest_sha256"]:
        raise ResourceReportError(f"{dataset}: native checkpoint manifest identity differs from the planned metadata")
    return record


def _capacity_inputs(records: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    preflight: dict[str, Any] = {"datasets": {}}
    native: dict[str, dict[str, Any]] = {}
    for dataset in DATASETS:
        item = records[dataset]
        domains = item.get("srs_family_domains")
        if not isinstance(domains, list) or not domains:
            raise ResourceReportError(f"{dataset}: typed SRS family domains are missing")
        family_domains: dict[str, int] = {}
        families: dict[str, dict[str, int]] = {}
        basis_domain = item.get("srs_basis_domain")
        if type(basis_domain) is not int or basis_domain != (1 << 18):
            raise ResourceReportError(f"{dataset}: SRS basis domain is not the fixed native domain")
        for domain in domains:
            if not isinstance(domain, dict):
                raise ResourceReportError(f"{dataset}: malformed SRS family domain")
            family_id = domain.get("id")
            logical_domain = domain.get("logical_domain")
            segment_count = domain.get("segment_count")
            if (not isinstance(family_id, str) or not family_id or
                    type(logical_domain) is not int or logical_domain <= 0 or
                    type(segment_count) is not int or segment_count <= 0 or
                    family_id in family_domains):
                raise ResourceReportError(f"{dataset}: invalid or duplicate SRS family descriptor")
            family_domains[family_id] = logical_domain
            families[family_id] = {
                "logical_domain": logical_domain,
                "segment_size": basis_domain,
                "segment_count": segment_count,
            }
        if len(family_domains) != item.get("srs_family_count"):
            raise ResourceReportError(f"{dataset}: SRS family count differs from typed family list")
        if item.get("checkpoint_tensor_count") != item.get("model_tensor_count") or item.get("model_tensor_count", 0) <= 0:
            raise ResourceReportError(f"{dataset}: architecture model tensor count differs from checkpoint manifest")
        cache_count = item.get("cache_file_count")
        checkpoint_count = item.get("checkpoint_file_count")
        if (type(cache_count) is not int or cache_count < 1 or
                type(checkpoint_count) is not int or checkpoint_count < 1):
            raise ResourceReportError(f"{dataset}: exact cache/checkpoint inventory counts are missing")
        audit = {
            "input_shape": {
                "features": item.get("features"),
                "classes": item.get("classes"),
            },
            "family_domains": family_domains,
            "family_domain_count": len(family_domains),
            "srs_plan": {
                "estimated_materialized_bytes": item.get("srs_materialized_bytes_upper_bound"),
                "basis_domain": basis_domain,
                "frame_bytes": item.get("srs_frame_bytes"),
                "families": families,
            },
        }
        preflight["datasets"][dataset] = {
            "resource_prerequisites": audit,
            "input_identity": {
                # _admission_profile uses these only for exact file counts in
                # the retained manifest bound; resource-plan already streamed
                # and counted the current regular files.
                "cache_files": [None] * cache_count,
                "checkpoint_files": [None] * checkpoint_count,
            },
        }
        native_item = dict(item)
        native_item["edges"] = item.get("normalized_edge_count")
        native[dataset] = native_item
    return preflight, native


def build_report(runner: Path) -> dict[str, Any]:
    if runner.is_symlink() or not runner.is_file() or not os.access(runner, os.X_OK):
        raise ResourceReportError(f"native gat_runner is missing, symlinked, or not executable: {runner}")
    source_entries_before = formal_provenance.source_entries(ROOT)
    source_digest_before = formal_provenance.identity(source_entries_before)
    binary_digest_before = formal_phase.sha256_file(runner)
    records = {dataset: _resource_plan(runner, dataset) for dataset in DATASETS}
    preflight, native = _capacity_inputs(records)
    # _admission_profile scans this directory only to attach a filesystem
    # identity and existing-byte count.  Keep the temporary outside formal
    # runs so this analysis cannot create session-like state there.
    with tempfile.TemporaryDirectory(prefix="zkdl-analysis-only-capacity-") as temporary:
        profile, diagnostics = formal_phase._admission_profile(
            Path(temporary), "analysis-only", preflight, native)
    requirements = formal_runner._remaining_capacity_requirements(profile, ())
    available_memory, memory_method = formal_runner._effective_host_memory_available()
    disk = os.statvfs(ROOT)
    available_disk = disk.f_bavail * disk.f_frsize
    source_hashes = {
        "formal_phase.py": formal_phase.sha256_file(Path(formal_phase.__file__).resolve()),
        "formal_runner.py": formal_phase.sha256_file(Path(formal_runner.__file__).resolve()),
        "analysis_only_resource_report.py": formal_phase.sha256_file(Path(__file__).resolve()),
    }
    source_entries_after = formal_provenance.source_entries(ROOT)
    source_digest_after = formal_provenance.identity(source_entries_after)
    binary_digest_after = formal_phase.sha256_file(runner)
    if source_digest_after != source_digest_before or binary_digest_after != binary_digest_before:
        raise ResourceReportError("baseline source tree or native binary changed during the capacity analysis")
    dataset_rows: dict[str, Any] = {}
    for dataset in DATASETS:
        capacity = profile["datasets"][dataset]
        phase_peaks = diagnostics[dataset]["phase_peak_bytes"]
        dataset_rows[dataset] = {
            "nodes": records[dataset]["nodes"],
            "normalized_edges": records[dataset]["normalized_edge_count"],
            "features": records[dataset]["features"],
            "classes": records[dataset]["classes"],
            "padded_edge_domain": records[dataset]["padded_edge_domain"],
            "proof_wire_upper_bound_bytes": records[dataset]["native_proof_wire_bytes_upper_bound"],
            "descriptor_resident_upper_bound_bytes": records[dataset]["native_operation_descriptor_resident_bytes_upper_bound"],
            "private_family_disk_upper_bound_bytes": records[dataset]["private_family_storage_upper_bound_bytes"],
            "sorted_scratch_disk_upper_bound_bytes": records[dataset]["sorted_lookup_scratch_upper_bound_bytes"],
            "sumcheck_spill_call_count": records[dataset]["native_sumcheck_spill_call_count"],
            "sumcheck_spill_disk_peak_bytes_upper_bound": records[dataset]["native_sumcheck_max_spill_disk_peak_bytes_upper_bound"],
            "sumcheck_spill_ram_peak_bytes_upper_bound": records[dataset]["native_sumcheck_max_spill_ram_peak_bytes_upper_bound"],
            "sumcheck_spill_provider_scan_bytes_upper_bound": records[dataset]["native_sumcheck_spill_provider_scan_bytes_upper_bound"],
            "sumcheck_spill_provider_source_bytes_upper_bound": records[dataset]["native_sumcheck_spill_provider_source_bytes_upper_bound"],
            "sumcheck_spill_storage_io_bytes_upper_bound": records[dataset]["native_sumcheck_spill_storage_io_bytes_upper_bound"],
            "srs_bytes_upper_bound": records[dataset]["srs_materialized_bytes_upper_bound"],
            "cell_retained_disk_upper_bound_bytes": capacity["cell_retained_artifact_bytes_upper_bound"],
            "cell_additional_disk_peak_bytes": capacity["additional_disk_peak_bytes"],
            "host_peak_upper_bound_bytes": capacity["host_peak_bytes"],
            "phase_peak_upper_bound_bytes": phase_peaks,
            "resource_plan_source_sha256": {
                "official_config": records[dataset]["official_config_sha256"],
                "analysis_only_planned_shape": records[dataset]["planned_shape_sha256"],
                "metadata": records[dataset]["metadata_sha256"],
                "edges": records[dataset]["raw_edge_file_sha256"],
                "node_ptr": records[dataset]["node_ptr_file_sha256"],
                "checkpoint_manifest": records[dataset]["checkpoint_manifest_sha256"],
                "gat_runner": records[dataset]["gat_runner_source_sha256"],
                "native_bridge_header": records[dataset]["native_bridge_header_sha256"],
                "native_core": records[dataset]["native_core_source_sha256"],
                "native_core_header": records[dataset]["native_core_header_sha256"],
                "proof_header": records[dataset]["relation_proof_header_sha256"],
            },
        }
    memory_required = requirements["host_bytes"]
    disk_required = requirements["disk_bytes"]
    return {
        "schema": "zkdl_analysis_only_five_dataset_capacity_v1",
        "status": "ANALYSIS_ONLY",
        "dataset_order": list(DATASETS),
        "formal_admission_run": False,
        "dataset_prepared": False,
        "proof_artifact_created": False,
        "independent_verification_run": False,
        "result_collection_run": False,
        "memory": {
            "available_bytes": available_memory,
            "available_gib": available_memory / (1 << 30),
            "availability_method": memory_method,
            "required_peak_bytes": memory_required,
            "required_peak_gib": memory_required / (1 << 30),
            "fits": memory_required <= available_memory,
        },
        "disk": {
            "available_bytes": available_disk,
            "available_tib": available_disk / (1 << 40),
            "required_sequential_five_cell_peak_bytes": disk_required,
            "required_sequential_five_cell_peak_tib": disk_required / (1 << 40),
            "finalization_host_peak_bytes": requirements["finalization_host_bytes"],
            "fits": disk_required <= available_disk,
            "lifecycle": "retained-prefix-plus-one-current-cell-scratch; cell scratch is pruned before independent verification",
        },
        "datasets": dataset_rows,
        "capacity_equations": "formal_phase._admission_profile + formal_runner._remaining_capacity_requirements",
        "analysis_tool_source_sha256": source_hashes,
        "baseline_source_tree_digest": source_digest_before,
        "gat_runner_binary_sha256": binary_digest_before,
        "build_manifest_authenticated": False,
        "analysis_only_resource_plan_records": records,
    }


def main(argv: list[str]) -> int:
    if len(argv) > 1:
        raise ResourceReportError("usage: analysis_only_resource_report.py [GAT_RUNNER]")
    runner = Path(argv[0]) if argv else ROOT / "gat_runner"
    report = build_report(runner.absolute())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except (formal_phase.PhaseError, formal_runner.RunnerError, ResourceReportError) as exc:
        print(f"analysis-only-resource-report: {exc}", file=sys.stderr)
        raise SystemExit(2)
