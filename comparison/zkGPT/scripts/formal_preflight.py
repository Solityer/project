#!/usr/bin/env python3
"""Authenticate formal execution preconditions without running development tests.

Only actual dataset proofs and separate native verification produce formal
results. Development proof closure is checked independently under tests/.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from common import (
    FORMAL_BACKENDS,
    DATASETS,
    CAPACITY_PROOF_PATH,
    PREFLIGHT_SCHEMA,
    ChainError,
    atomic_write_json,
    baseline_root,
    binary_fingerprint,
    build_binary,
    config_with_auth,
    file_fingerprint,
    formal_backend_capability,
    formal_child_environment,
    ensure_directory,
    load_json,
    invalidate_results,
    relative_path,
    require_bool,
    require_regular,
    require_manifest,
    sha256_file,
    verifier_binary,
    verify_build_provenance,
    verify_local_resources,
    formal_preflight_path,
    require_formal_session,
    runtime_source_sha256,
    atomic_write_text,
)

CAPACITY_SCHEMA = "zkgpt-gat-capacity-v8"
CAPACITY_U64_MAX = (1 << 64) - 1
CAPACITY_SHAPE_FIELDS = ("num_nodes", "num_edges", "num_features", "num_classes", "graph_count", "model_layers", "model_output_heads", "model_classes", "model_input_dim", "model_hidden_dim", "n_raw", "n_in")
CAPACITY_INTEGER_FIELDS = ("explicit_gate_count", "max_materialized_gate_bytes", "max_materialized_vector_bytes", "context_recipe_bytes_bound", "context_payload_bytes_bound", "lasso_logical_domain_entries", "lasso_private_selected_row_workspace_peak_bytes", "lasso_heap_workspace_peak_bytes", "hyrax_active_domain_workspace_entries", "hyrax_active_domain_workspace_bytes", "hyrax_l_qry_tensor_columns", "hyrax_l_qry_tensor_workspace_bytes", "hyrax_aux_tensor_workspace_bytes", "hyrax_folded_columns_workspace_bytes", "hyrax_tensor_retained_workspace_bytes", "hyrax_tensor_opening_transient_peak_workspace_bytes", "hyrax_tensor_combined_peak_workspace_bytes", "hyrax_w_mle_workspace_bytes", "borrowed_trace_bytes_upper_bound", "lasso_table_and_index_bytes_upper_bound", "gkr_layer_count_upper_bound", "gkr_fconn_layer_count_upper_bound", "gkr_input_source_count_upper_bound", "gkr_message_count_upper_bound", "gkr_wire_bytes_upper_bound", "gkr_prefix_workspace_bytes_upper_bound", "l_query_zero_interval_count_upper_bound", "estimated_native_scratch_bytes", "estimated_peak_host_memory_bytes", "estimated_native_heap_bytes_upper_bound", "estimated_native_mapped_bytes_upper_bound", "max_materialized_fr_bytes", "max_materialized_g1_bytes", "flattened_input_domain_entries", "flattened_input_domain_limit_u64", "disk_requirement_bytes_upper_bound", "proof_message_count_upper_bound", "maximum_transcript_field_entries_upper_bound", "total_transcript_field_entries_upper_bound", "maximum_message_string_bytes_upper_bound", "proof_artifact_wire_bytes_upper_bound", "artifact_disk_bytes_upper_bound", "host_memory_limit_bytes", "native_scratch_available_bytes", "coordinate_domain_limit_u64", "proof_message_count_limit", "maximum_message_string_bytes_limit", "proof_artifact_wire_limit_bytes", "artifact_disk_available_bytes", "shared_scratch_artifact_available_bytes", "retained_relation_witness_entries", "retained_relation_witness_bytes_upper_bound", "relation_public_output_entries", "relation_public_output_bytes_upper_bound", "relation_residual_entries", "relation_residual_bytes_upper_bound", "retained_layer_witness_entries_upper_bound", "retained_layer_witness_bytes_upper_bound", "evaluated_layer_witness_entries_upper_bound", "evaluated_layer_witness_bytes_upper_bound", "native_gkr_witness_heap_bytes_upper_bound", "native_gkr_witness_mapped_bytes_upper_bound", "relation_gate_unary_entries", "relation_gate_binary_entries", "relation_gate_output_entries", "max_single_gate_array_bytes_upper_bound", "max_explicit_unary_gate_buffer_bytes_upper_bound", "max_single_relation_builder_array_bytes_upper_bound", "gkr_subset_source_map_bytes_upper_bound", "gkr_subset_sort_scratch_bytes_upper_bound", "gkr_subset_bitmap_scratch_bytes_upper_bound", "gkr_subset_rank_scratch_bytes_upper_bound", "gkr_subset_source_support_entries_upper_bound", "native_transcript_heap_bytes_upper_bound", "transcript_content_bytes_upper_bound", "transcript_wire_bytes_upper_bound", "transcript_prefix_bytes_upper_bound", "maximum_challenge_suffix_bytes_upper_bound", "transcript_metadata_bytes_upper_bound", "zero_tail_certificate_count_upper_bound", "l_query_zero_interval_message_count_upper_bound")
CAPACITY_INTEGER_FIELDS += ("gkr_point_shape_bitmap_scratch_bytes", "gkr_point_shape_touched_index_scratch_bytes_upper_bound", "gkr_point_shape_layer_count", "gkr_point_shape_fconn_count")
CAPACITY_INTEGER_FIELDS += ("forward_admission_peak_host_memory_bytes_upper_bound",)
CAPACITY_INTEGER_FIELDS += ("explicit_unary_gate_buffer_entry_bytes",)
CAPACITY_INTEGER_FIELDS += ("gkr_prefix_filesystem_allocation_unit_bytes", "gkr_input_fold_cache_late_rounds", "gkr_input_fold_cache_scalar_bytes", "gkr_input_fold_cache_raw_bytes_upper_bound", "gkr_input_fold_cache_mapped_bytes_upper_bound", "gkr_input_fold_cache_heap_bytes_upper_bound", "gkr_input_fold_cache_page_size_bytes")
LASSO_PROFILE_U64_FIELDS = ("padded_queries", "table_domain", "full_domain", "rows", "columns", "private_witness_entries", "private_witness_element_bytes", "private_witness_bytes", "row_blinding_bytes", "row_commitment_bytes", "column_generator_bytes", "row_fold_weights_bytes", "row_fold_values_bytes", "onehot_query_bytes", "t2_column_one_bytes", "onehot_heap_peak_bytes", "masked_opening_generator_bytes", "masked_opening_coefficient_bytes", "masked_opening_response_bytes", "row_ledger_retained_bytes", "commit_peak_bytes", "booleanity_peak_bytes", "row_fold_peak_bytes", "opening_peak_bytes", "family_peak_bytes", "row_commitment_chunk_count", "booleanity_rounds", "onehot_rounds", "lookup_rounds", "hyrax_matrix_masked_openings", "hyrax_l_qry_tensor_openings")
LASSO_FAMILY_IDS = ("lrelu", "exp", "elu", "range")
GKR_LAYER_U64_FIELDS = ("size", "size_u0", "size_u1", "size_v0", "size_v1", "unary_gates", "binary_gates", "native_rows", "native_inner", "native_cols", "phase1_rounds", "phase1_first_round_pair_visits", "phase1_followup_pair_visits", "phase1_total_pair_visits", "phase1_extent_visits", "phase1_followup_extent_visits", "phase1_first_round_field_adds_upper_bound", "phase1_first_round_field_subs_upper_bound", "phase1_first_round_field_muls_upper_bound", "phase1_followup_field_adds_upper_bound", "phase1_followup_field_subs_upper_bound", "phase1_followup_field_muls_upper_bound", "phase1_init_active_scalar_assignments", "phase2_rounds", "phase2_first_round_pair_visits", "phase2_followup_pair_visits", "phase2_total_pair_visits", "phase2_extent_visits", "phase2_followup_extent_visits", "phase2_first_round_field_adds_upper_bound", "phase2_first_round_field_subs_upper_bound", "phase2_first_round_field_muls_upper_bound", "phase2_followup_field_adds_upper_bound", "phase2_followup_field_subs_upper_bound", "phase2_followup_field_muls_upper_bound", "phase2_init_active_scalar_assignments")
GKR_LAYER_PROFILE_FIELDS = frozenset(("layer_type", "bit_length", "max_bl_u", "max_bl_v", "need_phase2", "source_domains_exact", *GKR_LAYER_U64_FIELDS))


def capacity_u64(value: Any, key: str, dataset: str, *, positive: bool = False) -> int:
    if type(value) is not int or value < (1 if positive else 0) or value > CAPACITY_U64_MAX:
        raise ChainError(f"capacity has invalid uint64 {key} for {dataset}")
    return value


def capacity_add(values: Iterable[int], key: str, dataset: str) -> int:
    total = 0
    for value in values:
        if value > CAPACITY_U64_MAX - total:
            raise ChainError(f"capacity uint64 sum overflows {key} for {dataset}")
        total += value
    return total


def capacity_mul(left: int, right: int, key: str, dataset: str) -> int:
    if left and right > CAPACITY_U64_MAX // left:
        raise ChainError(f"capacity uint64 product overflows {key} for {dataset}")
    return left * right


def _validate_shape_admission_capacity(payload: Mapping[str, Any], dataset: str) -> None:
    source_support_entries = payload["gkr_subset_source_support_entries_upper_bound"]
    bitmap_bytes = payload["gkr_point_shape_bitmap_scratch_bytes"]
    touched_bytes = payload["gkr_point_shape_touched_index_scratch_bytes_upper_bound"]
    source_words = source_support_entries // 64 + (1 if source_support_entries % 64 else 0)
    expected_bitmap_bytes = capacity_mul(2, capacity_mul(source_words, 8, "GKR point-shape bitmap", dataset), "GKR point-shape bitmap", dataset)
    expected_touched_bytes = capacity_mul(capacity_mul(2, source_words, "GKR point-shape touched words", dataset), ctypes.sizeof(ctypes.c_size_t), "GKR point-shape touched indices", dataset)
    if bitmap_bytes != expected_bitmap_bytes:
        raise ChainError(f"capacity GKR point-shape bitmap does not match exact logical source support for {dataset}")
    if touched_bytes != expected_touched_bytes:
        raise ChainError(f"capacity GKR point-shape touched-index bound does not match exact logical source support for {dataset}")
    shape_host_bytes = capacity_add((bitmap_bytes, touched_bytes), "GKR point-shape host allocation", dataset)
    if payload["estimated_peak_host_memory_bytes"] < shape_host_bytes:
        raise ChainError(f"capacity host-memory bound omits native GKR point-shape admission for {dataset}")
    breakdown = payload["native_mapped_phase_breakdown"]
    native_heap_limit = 8 << 20
    expected_bitmap_mapped = bitmap_bytes if bitmap_bytes > native_heap_limit else 0
    expected_touched_mapped = touched_bytes if touched_bytes > native_heap_limit else 0
    expected_shape_mapped = capacity_add((expected_bitmap_mapped, expected_touched_mapped), "GKR point-shape mapped allocations", dataset)
    if breakdown["shape_admission"] != expected_shape_mapped or breakdown["component:shape_admission_bitmap"] != expected_bitmap_mapped or breakdown["component:shape_admission_touched_indices"] != expected_touched_mapped:
        raise ChainError(f"capacity shape-admission allocation classes are inconsistent for {dataset}")


def reject_nonfinite_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def reject_floating_json_number(value: str) -> None:
    raise ValueError(f"non-integer capacity JSON number: {value}")


def parse_capacity_integer(value: str) -> int:
    return int(value)


def validate_gkr_layer_work_profile(payload: Mapping[str, Any], dataset: str) -> None:
    profiles = payload.get("gkr_layer_work_profile")
    layer_count = payload.get("gkr_layer_count_upper_bound")
    if not isinstance(profiles, list) or type(layer_count) is not int or len(profiles) != layer_count:
        raise ChainError(f"capacity has invalid GKR layer work profile for {dataset}")
    for index, profile in enumerate(profiles):
        if not isinstance(profile, dict) or set(profile) != GKR_LAYER_PROFILE_FIELDS:
            raise ChainError(f"capacity has invalid GKR layer work profile fields at layer {index} for {dataset}")
        for key in GKR_LAYER_U64_FIELDS:
            capacity_u64(profile.get(key), f"gkr_layer_work_profile[{index}].{key}", dataset)
        layer_type = profile.get("layer_type")
        if type(layer_type) is not int or layer_type < 0 or layer_type > (1 << 32) - 1:
            raise ChainError(f"capacity has invalid GKR layer type at layer {index} for {dataset}")
        for key in ("bit_length", "max_bl_u", "max_bl_v"):
            value = profile.get(key)
            if type(value) is not int or value < -1 or value > (1 << 31) - 1:
                raise ChainError(f"capacity has invalid signed GKR {key} at layer {index} for {dataset}")
        for key in ("need_phase2", "source_domains_exact"):
            if type(profile.get(key)) is not bool:
                raise ChainError(f"capacity has invalid GKR {key} flag at layer {index} for {dataset}")


def validate_canonical_graph_identity(payload: Mapping[str, Any], dataset: str,
                                      config: Mapping[str, str]) -> None:
    identity = payload.get("canonical_graph_identity")
    fields = {"digest", "nodes", "edges", "graphs", "node_ptr", "edge_ptr", "input_preprocess_id", "batching_rule"}
    if not isinstance(identity, dict) or set(identity) != fields:
        raise ChainError(f"capacity has no strict native canonical graph identity for {dataset}")
    digest = identity.get("digest")
    if not isinstance(digest, str) or len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ChainError(f"capacity has an invalid canonical graph digest for {dataset}")
    for identity_key, capacity_key in (("nodes", "num_nodes"), ("edges", "num_edges"), ("graphs", "graph_count")):
        value = capacity_u64(identity.get(identity_key), f"canonical_graph_identity.{identity_key}", dataset, positive=True)
        if value != payload.get(capacity_key):
            raise ChainError(f"native graph identity disagrees with capacity {capacity_key} for {dataset}")
    for key, terminal in (("node_ptr", identity["nodes"]), ("edge_ptr", identity["edges"])):
        values = identity.get(key)
        if not isinstance(values, list) or len(values) != identity["graphs"] + 1:
            raise ChainError(f"native graph identity has invalid {key} boundaries for {dataset}")
        for index, value in enumerate(values):
            capacity_u64(value, f"canonical_graph_identity.{key}[{index}]", dataset)
        if values[0] != 0 or values[-1] != terminal or any(left >= right for left, right in zip(values, values[1:])):
            raise ChainError(f"native graph identity {key} boundaries are not strictly increasing for {dataset}")
    if (identity.get("input_preprocess_id") != config.get("input_preprocess_id") or
            identity.get("batching_rule") != config.get("batching_rule")):
        raise ChainError(f"native graph identity preprocessing/batching differs from config for {dataset}")


def validate_capacity_numbers(payload: Mapping[str, Any], dataset: str, *, require_fits: bool) -> None:
    if not require_fits:
        reason = payload.get("reason")
        if not isinstance(reason, str) or not reason:
            raise ChainError(f"blocked capacity has no native failure reason for {dataset}")
        return
    stale_capacity_keys = ("masked_lasso_matrix_entries", "masked_response_entries_lower_bound", "masked_proof_bytes_lower_bound", "maximum_masked_response_entries", "native_response_vector_limit", "native_message_string_limit_bytes", "native_artifact_wire_limit_bytes", "lasso_family_domains", "max_materialized_gate", "max_materialized_vector", "estimated_peak_memory_bytes")
    if any(key in payload for key in stale_capacity_keys):
        raise ChainError(f"capacity v2 contains legacy dense-path fields for {dataset}")
    for key in CAPACITY_SHAPE_FIELDS:
        capacity_u64(payload.get(key), key, dataset, positive=True)
    if payload["num_classes"] != payload["model_classes"]:
        raise ChainError(f"capacity graph/model class counts disagree for {dataset}")
    for key in CAPACITY_INTEGER_FIELDS:
        capacity_u64(payload.get(key), key, dataset)
    validate_gkr_layer_work_profile(payload, dataset)
    if payload.get("gkr_point_shape_ready") is not True:
        raise ChainError(f"capacity has no successful native GKR point-domain shape admission for {dataset}")
    if payload.get("forward_value_admission_ready") is not True:
        raise ChainError(f"capacity has no successful canonical all-value forward admission for {dataset}")
    point_shape_digest = payload.get("gkr_point_shape_digest")
    if not isinstance(point_shape_digest, str) or len(point_shape_digest) != 64 or any(ch not in "0123456789abcdef" for ch in point_shape_digest):
        raise ChainError(f"capacity has an invalid native GKR point-domain shape digest for {dataset}")
    if (payload["gkr_point_shape_layer_count"] != payload["gkr_layer_count_upper_bound"] or
            payload["gkr_point_shape_fconn_count"] != payload["gkr_fconn_layer_count_upper_bound"]):
        raise ChainError(f"capacity native GKR point-domain shape differs from the circuit profile for {dataset}")
    source_support_entries = payload["gkr_subset_source_support_entries_upper_bound"]
    if source_support_entries > payload["flattened_input_domain_entries"]:
        raise ChainError(f"capacity logical source support exceeds the full source domain for {dataset}")
    source_words = source_support_entries // 64 + (1 if source_support_entries % 64 else 0)
    expected_bitmap_bytes = capacity_mul(source_words, 8, "GKR source-map bitmap", dataset)
    expected_rank_bytes = capacity_mul(source_words, 8, "GKR source-map rank prefix", dataset)
    if (payload["gkr_subset_bitmap_scratch_bytes_upper_bound"] != expected_bitmap_bytes or
            payload["gkr_subset_rank_scratch_bytes_upper_bound"] != expected_rank_bytes or
            payload["gkr_subset_sort_scratch_bytes_upper_bound"] != 0):
        raise ChainError(f"capacity source-map bitmap/rank/sort scratch bound is inconsistent for {dataset}")
    cache_scalar_bytes = capacity_u64(
        payload["gkr_input_fold_cache_scalar_bytes"],
        "gkr_input_fold_cache_scalar_bytes", dataset, positive=True)
    cache_unit_bytes = capacity_u64(
        payload["gkr_prefix_filesystem_allocation_unit_bytes"],
        "gkr_prefix_filesystem_allocation_unit_bytes", dataset, positive=True)
    cache_page_size_bytes = capacity_u64(
        payload["gkr_input_fold_cache_page_size_bytes"],
        "gkr_input_fold_cache_page_size_bytes", dataset, positive=True)
    cache_rounding_unit_bytes = max(cache_unit_bytes, cache_page_size_bytes)
    cache_domain = capacity_u64(payload["n_in"], "n_in", dataset, positive=True)
    cache_segments = payload.get("input_segments")
    if (cache_domain & (cache_domain - 1) or not isinstance(cache_segments, list) or
            len(cache_segments) != 4):
        raise ChainError(f"capacity input-fold cache domain/segments are invalid for {dataset}")
    cache_lengths = [capacity_u64(
        segment.get("logical_length") if isinstance(segment, dict) else None,
        f"input_segments[{index}].logical_length", dataset)
        for index, segment in enumerate(cache_segments)]
    if any(length > cache_domain for length in cache_lengths):
        raise ChainError(f"capacity input-fold cache segment exceeds its domain for {dataset}")
    cache_cap = 128 * 1024 * 1024 * 1024
    cache_heap_limit = 8 << 20
    expected_cache_plan = (0, 0, 0, 0) if cache_domain == 1 else None
    if cache_domain > 1:
        for late_round in range(cache_domain.bit_length() - 1):
            divisor = 1 << late_round
            raw_bytes = 0
            mapped_bytes = 0
            heap_bytes = 0
            for length in cache_lengths:
                entries = length // divisor + (1 if length % divisor else 0)
                array_bytes = capacity_mul(
                    entries, cache_scalar_bytes, "input-fold cache array", dataset)
                raw_bytes = capacity_add(
                    (raw_bytes, 2 * array_bytes), "input-fold cache raw bytes", dataset)
                heap_bytes = capacity_add(
                    (heap_bytes, 2 * min(array_bytes, cache_heap_limit)),
                    "input-fold cache heap bytes", dataset)
                if array_bytes > cache_heap_limit:
                    rounded = ((array_bytes + cache_rounding_unit_bytes - 1) //
                               cache_rounding_unit_bytes) * cache_rounding_unit_bytes
                    mapped_bytes = capacity_add(
                        (mapped_bytes, 2 * rounded),
                        "input-fold cache mapped bytes", dataset)
            if raw_bytes <= cache_cap and mapped_bytes <= cache_cap:
                expected_cache_plan = (late_round, raw_bytes, mapped_bytes, heap_bytes)
                break
    if expected_cache_plan is None:
        raise ChainError(f"capacity input-fold cache has no bounded late-fold round for {dataset}")
    reported_cache_plan = (
        payload["gkr_input_fold_cache_late_rounds"],
        payload["gkr_input_fold_cache_raw_bytes_upper_bound"],
        payload["gkr_input_fold_cache_mapped_bytes_upper_bound"],
        payload["gkr_input_fold_cache_heap_bytes_upper_bound"],
    )
    if reported_cache_plan != expected_cache_plan:
        raise ChainError(f"capacity input-fold cache plan/bounds are inconsistent for {dataset}")
    forward_admission_peak = capacity_u64(payload["forward_admission_peak_host_memory_bytes_upper_bound"],
                                          "forward_admission_peak_host_memory_bytes_upper_bound", dataset, positive=True)
    if payload["estimated_peak_host_memory_bytes"] < forward_admission_peak:
        raise ChainError(f"capacity host-memory bound omits canonical forward admission for {dataset}")
    if forward_admission_peak > payload["host_memory_limit_bytes"]:
        raise ChainError(f"canonical forward-admission host bound exceeds available memory for {dataset}")
    if payload["gkr_subset_source_map_bytes_upper_bound"] > capacity_mul(source_support_entries, 8, "unique source-map support", dataset):
        raise ChainError(f"capacity source-map bound exceeds logical source support for {dataset}")
    breakdown = payload.get("native_mapped_phase_breakdown")
    phase_keys = ("gate_plan", "plan_restore", "gkr_validation", "gkr_proving", "lasso_tensor", "input_opening", "shape_admission")
    component_keys = ("component:proof_persistent", "component:gate_arrays", "component:source_maps", "component:residual_targets", "component:layer_values", "component:input_segments", "component:input_oracle_fold", "component:input_fold_cache", "component:fconn_fold", "component:fconn_source_weights", "component:gkr_prefix", "component:lasso_persistent", "component:tensor_retained", "component:lasso_transient", "component:tensor_opening", "component:w_mle", "component:source_bitmap", "component:source_rank", "component:explicit_unary_gate_buffer", "component:shape_admission_bitmap", "component:shape_admission_touched_indices")
    if not isinstance(breakdown, dict) or set(breakdown) != set(phase_keys + component_keys):
        raise ChainError(f"capacity native mapped phase breakdown is incomplete for {dataset}")
    phase_bytes = {key: capacity_u64(breakdown.get(key), f"native_mapped_phase_breakdown.{key}", dataset) for key in phase_keys}
    for key in component_keys:
        capacity_u64(breakdown.get(key), f"native_mapped_phase_breakdown.{key}", dataset)
    expected_subset_bitmap_mapped = expected_bitmap_bytes if expected_bitmap_bytes > (8 << 20) else 0
    expected_subset_rank_mapped = expected_rank_bytes if expected_rank_bytes > (8 << 20) else 0
    if (breakdown["component:source_bitmap"] != expected_subset_bitmap_mapped or
            breakdown["component:source_rank"] != expected_subset_rank_mapped):
        raise ChainError(f"capacity source-map bitmap/rank allocator classes are inconsistent for {dataset}")
    if (breakdown["component:input_fold_cache"] != expected_cache_plan[2] or
            phase_bytes["gkr_proving"] < expected_cache_plan[2] or
            payload["estimated_native_heap_bytes_upper_bound"] < expected_cache_plan[3]):
        raise ChainError(f"capacity input-fold cache is absent from native resource bounds for {dataset}")
    explicit_unary_bytes = payload["max_explicit_unary_gate_buffer_bytes_upper_bound"]
    explicit_unary_entry_bytes = payload["explicit_unary_gate_buffer_entry_bytes"]
    if explicit_unary_entry_bytes == 0 or explicit_unary_bytes % explicit_unary_entry_bytes != 0 or explicit_unary_bytes > capacity_mul(
            payload["explicit_gate_count"], explicit_unary_entry_bytes,
            "explicit unary gate temporary bound", dataset):
        raise ChainError(f"capacity explicit unary temporary is inconsistent with its reported native entry size for {dataset}")
    expected_explicit_unary_mapped = explicit_unary_bytes if explicit_unary_bytes > (8 << 20) else 0
    if breakdown["component:explicit_unary_gate_buffer"] != expected_explicit_unary_mapped:
        raise ChainError(f"capacity explicit unary mapped temporary omits the allocator threshold for {dataset}")
    _validate_shape_admission_capacity(payload, dataset)
    if max(phase_bytes.values()) != payload["estimated_native_mapped_bytes_upper_bound"] or payload["estimated_native_scratch_bytes"] != max(phase_bytes.values()):
        raise ChainError(f"capacity native scratch does not match the live phase maximum for {dataset}")
    profiles = payload.get("lasso_workspace_profile")
    if not isinstance(profiles, list) or len(profiles) != len(LASSO_FAMILY_IDS):
        raise ChainError(f"capacity has invalid Lasso workspace profile for {dataset}")
    if type(payload.get("scratch_artifact_same_filesystem")) is not bool or type(payload.get("ppi_node_ptr_boundary")) is not bool or type(payload.get("ppi_edge_ptr_boundary")) is not bool:
        raise ChainError(f"capacity has invalid filesystem or graph-boundary flags for {dataset}")
    logical_entries = 0
    lasso_retained_bytes = 0
    largest_lasso_family_transient_bytes = 0
    for index, profile in enumerate(profiles):
        if not isinstance(profile, dict) or profile.get("family_id") != LASSO_FAMILY_IDS[index] or not isinstance(profile.get("workspace"), dict):
            raise ChainError(f"capacity has invalid Lasso profile {index} for {dataset}")
        workspace = profile["workspace"]
        if type(workspace.get("t2_booleanity")) is not bool:
            raise ChainError(f"capacity has invalid Lasso booleanity profile {index} for {dataset}")
        values = {key: capacity_u64(workspace.get(key), f"lasso_workspace_profile[{index}].workspace.{key}", dataset) for key in LASSO_PROFILE_U64_FIELDS}
        wrapper_fields = ("logical_domain_entries", "row_commitment_chunk_count", "row_commitment_chunk_entries", "booleanity_rounds", "a_query_challenges", "onehot_rounds", "lookup_rounds", "hyrax_matrix_masked_openings", "hyrax_l_qry_tensor_openings", "row_commitment_wire_bytes_upper_bound", "family_message_count_upper_bound", "family_wire_bytes_upper_bound")
        wrapper = {key: capacity_u64(profile.get(key), f"lasso_workspace_profile[{index}].{key}", dataset) for key in wrapper_fields}
        if any(values[key] == 0 for key in ("padded_queries", "table_domain", "rows", "columns", "private_witness_element_bytes", "private_witness_entries")):
            raise ChainError(f"capacity has empty Lasso domain {index} for {dataset}")
        expected_domain = capacity_mul(values["padded_queries"], values["table_domain"], "Lasso logical domain", dataset)
        if values["full_domain"] != expected_domain:
            raise ChainError(f"capacity Lasso domain product mismatch for {dataset}")
        if any(value & (value - 1) for value in (values["padded_queries"], values["table_domain"], values["rows"], values["columns"])):
            raise ChainError(f"capacity Lasso MLE dimensions are not powers of two for {dataset}")
        if workspace["t2_booleanity"] != (values["table_domain"] == 2):
            raise ChainError(f"capacity Lasso T2 profile mismatch for {dataset}")
        duplicate_fields = ("row_commitment_chunk_count", "booleanity_rounds", "onehot_rounds", "lookup_rounds", "hyrax_matrix_masked_openings", "hyrax_l_qry_tensor_openings")
        if any(wrapper[key] != values[key] for key in duplicate_fields):
            raise ChainError(f"capacity Lasso duplicated workspace counters disagree for {dataset}")
        if wrapper["logical_domain_entries"] != values["full_domain"] or wrapper["row_commitment_chunk_count"] != (values["rows"] - 1) // 4096 + 1 or wrapper["row_commitment_chunk_entries"] != 4096 or wrapper["booleanity_rounds"] != values["full_domain"].bit_length() - 1 or wrapper["a_query_challenges"] != values["padded_queries"].bit_length() - 1 or wrapper["onehot_rounds"] != values["table_domain"].bit_length() - 1 or wrapper["lookup_rounds"] != values["table_domain"].bit_length() - 1 or wrapper["hyrax_matrix_masked_openings"] != 4 or wrapper["hyrax_l_qry_tensor_openings"] != 1:
            raise ChainError(f"capacity Lasso transcript profile mismatch for {dataset}")
        if values["private_witness_entries"] != values["padded_queries"]:
            raise ChainError(f"capacity Lasso selected-row count mismatch for {dataset}")
        if values["private_witness_bytes"] != capacity_mul(values["private_witness_entries"], values["private_witness_element_bytes"], "Lasso private witness bytes", dataset):
            raise ChainError(f"capacity Lasso private workspace size mismatch for {dataset}")
        expected_ledger = capacity_add((values["row_blinding_bytes"], values["column_generator_bytes"]), "Lasso retained row ledger", dataset)
        expected_commit_peak = capacity_add((expected_ledger, values["row_commitment_bytes"]), "Lasso commit peak", dataset)
        expected_booleanity_peak = capacity_add((expected_ledger, max(values["private_witness_bytes"], values["t2_column_one_bytes"])), "Lasso booleanity peak", dataset)
        expected_row_fold_peak = capacity_add((expected_ledger, values["row_fold_weights_bytes"], values["row_fold_values_bytes"], values["onehot_query_bytes"]), "Lasso row-fold peak", dataset)
        expected_opening_peak = max(expected_row_fold_peak, capacity_add((expected_ledger, values["row_fold_values_bytes"], values["onehot_query_bytes"], values["masked_opening_generator_bytes"], values["masked_opening_coefficient_bytes"], values["masked_opening_response_bytes"]), "Lasso opening peak", dataset))
        expected_ordinary_onehot_peak = capacity_add((expected_ledger, values["onehot_heap_peak_bytes"]), "Lasso ordinary onehot peak", dataset)
        expected_family_peak = max(expected_commit_peak, expected_booleanity_peak, expected_opening_peak, expected_ordinary_onehot_peak)
        if values["row_ledger_retained_bytes"] != expected_ledger or values["commit_peak_bytes"] != expected_commit_peak or values["booleanity_peak_bytes"] != expected_booleanity_peak or values["row_fold_peak_bytes"] != expected_row_fold_peak or values["opening_peak_bytes"] != expected_opening_peak or values["family_peak_bytes"] != expected_family_peak:
            raise ChainError(f"capacity Lasso live-set profile is inconsistent for {dataset}")
        layout_domain = capacity_mul(values["rows"], values["columns"], "Lasso matrix layout", dataset)
        if layout_domain != values["full_domain"]:
            raise ChainError(f"capacity Lasso matrix layout/domain mismatch for {dataset}")
        row_bits = values["rows"].bit_length() - 1
        column_bits = values["columns"].bit_length() - 1
        domain_bits = capacity_add((values["padded_queries"].bit_length() - 1, values["table_domain"].bit_length() - 1), "Lasso domain bits", dataset)
        if row_bits != domain_bits // 2 or column_bits != domain_bits - row_bits:
            raise ChainError(f"capacity Lasso matrix layout disagrees with the native balanced layout for {dataset}")
        if values["family_peak_bytes"] < values["private_witness_bytes"]:
            raise ChainError(f"capacity Lasso peak omits its private witness for {dataset}")
        row_payload_minimum = capacity_add((capacity_mul(wrapper["row_commitment_chunk_count"], 8, "Lasso row chunk count prefix", dataset), capacity_mul(values["rows"], 8 + 64, "Lasso row commitment payload", dataset)), "Lasso row commitment wire", dataset)
        minimum_family_messages = capacity_add((1, wrapper["row_commitment_chunk_count"], wrapper["booleanity_rounds"], wrapper["a_query_challenges"], wrapper["onehot_rounds"], wrapper["lookup_rounds"], wrapper["hyrax_matrix_masked_openings"], wrapper["hyrax_l_qry_tensor_openings"]), "Lasso family message count", dataset)
        if wrapper["row_commitment_wire_bytes_upper_bound"] < row_payload_minimum or wrapper["family_wire_bytes_upper_bound"] < wrapper["row_commitment_wire_bytes_upper_bound"] or wrapper["family_message_count_upper_bound"] < minimum_family_messages:
            raise ChainError(f"capacity Lasso wire bound is understated for {dataset}")
        logical_entries = capacity_add((logical_entries, wrapper["logical_domain_entries"]), "Lasso logical-domain total", dataset)
        lasso_retained_bytes = capacity_add((lasso_retained_bytes, values["row_ledger_retained_bytes"]), "Lasso retained family workspaces", dataset)
        largest_lasso_family_transient_bytes = max(largest_lasso_family_transient_bytes, values["family_peak_bytes"] - values["row_ledger_retained_bytes"])
    if payload["lasso_logical_domain_entries"] != logical_entries:
        raise ChainError(f"capacity Lasso logical-domain total mismatch for {dataset}")
    expected_lasso_workspace_peak = capacity_add((lasso_retained_bytes, largest_lasso_family_transient_bytes), "Lasso private workspace peak", dataset)
    if payload["lasso_private_selected_row_workspace_peak_bytes"] != expected_lasso_workspace_peak:
        raise ChainError(f"capacity Lasso private workspace peak mismatch for {dataset}")
    lasso_heap_persistent = capacity_add(
        (
            profile["workspace"]["row_ledger_retained_bytes"]
            for profile in profiles
        ),
        "Lasso standard-heap retained ledgers",
        dataset,
    )
    lasso_heap_generator_copies = capacity_mul(
        2,
        capacity_add(
            (profile["workspace"]["column_generator_bytes"] for profile in profiles),
            "Lasso duplicated standard-heap generator vectors",
            dataset,
        ),
        "Lasso duplicated standard-heap generator vectors",
        dataset,
    )
    lasso_row_commitment_transient = capacity_add(
        (profile["workspace"]["row_commitment_bytes"] for profile in profiles),
        "Lasso simultaneously retained row commitments",
        dataset,
    )
    lasso_heap_transient = max(
        capacity_add(
            (
                capacity_add(
                    (
                        profile["workspace"]["row_fold_weights_bytes"],
                        profile["workspace"]["row_fold_values_bytes"],
                        profile["workspace"]["onehot_query_bytes"],
                    ),
                    "Lasso standard-heap row-fold transient",
                    dataset,
                ),
                profile["workspace"]["onehot_heap_peak_bytes"],
            ),
            "Lasso onehot/fold transient peak",
            dataset,
        )
        for profile in profiles
    )
    expected_lasso_heap_workspace = capacity_add(
        (
            lasso_heap_persistent,
            lasso_heap_generator_copies,
            max(
                lasso_row_commitment_transient,
                lasso_heap_transient,
            ),
        ),
        "Lasso standard-heap workspace peak",
        dataset,
    )
    if payload["lasso_heap_workspace_peak_bytes"] < expected_lasso_heap_workspace:
        raise ChainError(f"capacity Lasso standard-heap workspace is understated for {dataset}")
    query_logical_length = 0
    zero_interval_count = 0
    for profile in profiles:
        query_block = profile["workspace"]["padded_queries"]
        remainder = query_logical_length % query_block
        if remainder:
            zero_interval_count += 1
            query_logical_length = capacity_add((query_logical_length, query_block - remainder), "L_qry aligned query gap", dataset)
        query_logical_length = capacity_add((query_logical_length, query_block), "L_qry logical length", dataset)
    if query_logical_length > payload["n_in"]:
        raise ChainError(f"capacity L_qry logical length exceeds its padded input domain for {dataset}")
    l_query_active_bits = (query_logical_length - 1).bit_length()
    l_query_active = 1 << l_query_active_bits
    expected_l_query_columns = 1 << (l_query_active_bits // 2)
    l_query_column_rounds = expected_l_query_columns.bit_length() - 1
    hyrax_active_entries = max(profile["workspace"]["columns"] for profile in profiles)
    hyrax_active_bytes = max(capacity_add((profile["workspace"]["masked_opening_generator_bytes"], profile["workspace"]["masked_opening_coefficient_bytes"], profile["workspace"]["masked_opening_response_bytes"]), "Hyrax masked-opening workspace", dataset) for profile in profiles)
    hyrax_folded_columns_bytes = max(capacity_add((profile["workspace"]["row_fold_values_bytes"], profile["workspace"]["onehot_query_bytes"]), "Hyrax folded-column workspace", dataset) for profile in profiles)
    fr_element_bytes = profiles[0]["workspace"]["row_fold_values_bytes"] // profiles[0]["workspace"]["columns"]
    g1_element_bytes = profiles[0]["workspace"]["column_generator_bytes"] // profiles[0]["workspace"]["columns"]
    if fr_element_bytes == 0 or g1_element_bytes == 0 or any(profile["workspace"]["row_fold_values_bytes"] % profile["workspace"]["columns"] or profile["workspace"]["column_generator_bytes"] % profile["workspace"]["columns"] or profile["workspace"]["row_fold_values_bytes"] // profile["workspace"]["columns"] != fr_element_bytes or profile["workspace"]["column_generator_bytes"] // profile["workspace"]["columns"] != g1_element_bytes for profile in profiles):
        raise ChainError(f"capacity Lasso element-size profiles disagree for {dataset}")
    for profile in profiles:
        workspace = profile["workspace"]
        expected_profile_sizes = {
            "row_blinding_bytes": capacity_mul(workspace["rows"], fr_element_bytes, "Lasso row blinding bytes", dataset),
            "row_commitment_bytes": capacity_mul(workspace["rows"], g1_element_bytes, "Lasso row commitment bytes", dataset),
            "column_generator_bytes": capacity_mul(workspace["columns"], g1_element_bytes, "Lasso column generator bytes", dataset),
            "row_fold_weights_bytes": capacity_mul(workspace["rows"], fr_element_bytes, "Lasso row fold weights", dataset),
            "row_fold_values_bytes": capacity_mul(workspace["columns"], fr_element_bytes, "Lasso row fold values", dataset),
            "onehot_query_bytes": capacity_mul(workspace["table_domain"], fr_element_bytes, "Lasso selected-table onehot query", dataset),
            "t2_column_one_bytes": capacity_mul(workspace["padded_queries"], fr_element_bytes, "Lasso T2 NativeVector column", dataset) if workspace["t2_booleanity"] else 0,
            "onehot_heap_peak_bytes": capacity_mul(
                capacity_add(
                    (
                        capacity_mul(2, workspace["table_domain"], "Lasso onehot left/current extent", dataset),
                        workspace["table_domain"] // 2 if workspace["table_domain"] > 1 else 0,
                    ),
                    "Lasso onehot left/current/next extent",
                    dataset,
                ),
                fr_element_bytes,
                "Lasso onehot standard-heap peak",
                dataset,
            ),
            "masked_opening_generator_bytes": capacity_mul(workspace["columns"], g1_element_bytes, "Lasso masked opening generators", dataset),
            "masked_opening_coefficient_bytes": capacity_mul(workspace["columns"], fr_element_bytes, "Lasso masked opening coefficients", dataset),
            "masked_opening_response_bytes": capacity_mul(workspace["columns"], fr_element_bytes, "Lasso masked opening response", dataset),
        }
        if any(workspace[key] != value for key, value in expected_profile_sizes.items()):
            raise ChainError(f"capacity Lasso vector extent profile is inconsistent for {dataset}")
        if workspace["t2_booleanity"] and workspace["t2_column_one_bytes"] != workspace["private_witness_bytes"]:
            raise ChainError(f"capacity T2 NativeVector alias size is inconsistent for {dataset}")
    tensor_rows = l_query_active // expected_l_query_columns
    tensor_row_bytes = capacity_add((g1_element_bytes, capacity_mul(2, fr_element_bytes, "L_qry tensor row size", dataset)), "L_qry tensor row size", dataset)
    tensor_column_bytes = capacity_add((capacity_mul(2, g1_element_bytes, "L_qry tensor column generators", dataset), capacity_mul(3, fr_element_bytes, "L_qry tensor column fields", dataset)), "L_qry tensor column size", dataset)
    expected_l_query_workspace = capacity_add((capacity_mul(tensor_rows, tensor_row_bytes, "L_qry tensor row workspace", dataset), capacity_mul(expected_l_query_columns, tensor_column_bytes, "L_qry tensor column workspace", dataset)), "L_qry tensor workspace", dataset)
    if payload["hyrax_active_domain_workspace_entries"] != hyrax_active_entries or payload["hyrax_active_domain_workspace_bytes"] != hyrax_active_bytes or payload["hyrax_folded_columns_workspace_bytes"] != hyrax_folded_columns_bytes or payload["hyrax_l_qry_tensor_columns"] != expected_l_query_columns or payload["hyrax_l_qry_tensor_workspace_bytes"] != expected_l_query_workspace:
        raise ChainError(f"capacity Hyrax workspace aggregates disagree with Lasso profiles for {dataset}")
    aux_tensor_shape = None
    for active_bits in range(0, 61):
        active = 1 << active_bits
        columns = 1 << (active_bits // 2)
        rows = active // columns
        row_bytes = capacity_add((g1_element_bytes, capacity_mul(2, fr_element_bytes, "Aux tensor row size", dataset)), "Aux tensor row size", dataset)
        column_bytes = capacity_add((capacity_mul(2, g1_element_bytes, "Aux tensor column generators", dataset), capacity_mul(3, fr_element_bytes, "Aux tensor column fields", dataset)), "Aux tensor column size", dataset)
        workspace_bytes = capacity_add((capacity_mul(rows, row_bytes, "Aux tensor row workspace", dataset), capacity_mul(columns, column_bytes, "Aux tensor column workspace", dataset)), "Aux tensor workspace", dataset)
        if payload["hyrax_aux_tensor_workspace_bytes"] == workspace_bytes:
            aux_tensor_shape = (rows, columns)
            break
    if aux_tensor_shape is None:
        raise ChainError(f"capacity Aux tensor workspace does not match the native balanced layout for {dataset}")
    for index, profile in enumerate(profiles):
        workspace = profile["workspace"]
        rounds = workspace["booleanity_rounds"]
        onehot_rounds = workspace["onehot_rounds"]
        lookup_rounds = workspace["lookup_rounds"]
        query_challenges = workspace["padded_queries"].bit_length() - 1
        matrix_rounds = workspace["columns"].bit_length() - 1
        expected_family_messages = capacity_add(
            (
                1,
                workspace["row_commitment_chunk_count"],
                rounds,
                query_challenges,
                4,
                capacity_add((114, capacity_mul(139, rounds, "Lasso Booleanity message rounds", dataset)), "Lasso Booleanity messages", dataset),
                capacity_add((72, capacity_mul(103, onehot_rounds, "Lasso onehot message rounds", dataset)), "Lasso onehot messages", dataset),
                capacity_add((86, capacity_mul(121, lookup_rounds, "Lasso lookup message rounds", dataset)), "Lasso lookup messages", dataset),
                capacity_mul(4, capacity_add((9, capacity_mul(2, matrix_rounds, "Lasso matrix opening rounds", dataset)), "Lasso matrix opening messages", dataset), "Lasso matrix opening messages", dataset),
                capacity_add((10, capacity_mul(2, l_query_column_rounds, "Lasso L_qry tensor rounds", dataset)), "Lasso L_qry tensor messages", dataset),
            ),
            f"lasso_workspace_profile[{index}].family_message_count_upper_bound",
            dataset,
        )
        if profile["family_message_count_upper_bound"] != expected_family_messages:
            raise ChainError(f"capacity Lasso family message count disagrees with the native call path for {dataset}")
        row_payload_bound = capacity_add(
            (
                capacity_mul(workspace["row_commitment_chunk_count"], 8, "Lasso row chunk count prefix", dataset),
                capacity_mul(workspace["rows"], 8 + 64, "Lasso row commitment payload", dataset),
            ),
            "Lasso row commitment wire",
            dataset,
        )
        maximum_family_vector_entries = max(
            9,
            rounds,
            onehot_rounds,
            lookup_rounds,
            query_challenges,
            matrix_rounds,
            l_query_column_rounds,
        )
        family_field_vector_bound = capacity_add(
            (8, capacity_mul(maximum_family_vector_entries, 8 + 64, "Lasso field-vector payload", dataset)),
            "Lasso field-vector payload",
            dataset,
        )
        non_chunk_messages = profile["family_message_count_upper_bound"] - workspace["row_commitment_chunk_count"]
        expected_family_wire = capacity_add(
            (
                row_payload_bound,
                capacity_mul(non_chunk_messages, 1024, "Lasso non-chunk message wire", dataset),
                capacity_mul(4, family_field_vector_bound, "Lasso field-vector wire", dataset),
            ),
            "Lasso family wire",
            dataset,
        )
        if profile["row_commitment_wire_bytes_upper_bound"] < row_payload_bound or profile["family_wire_bytes_upper_bound"] < expected_family_wire:
            raise ChainError(f"capacity Lasso family wire bound omits active protocol payloads for {dataset}")
    l_query_retained = capacity_mul(tensor_rows, capacity_add((g1_element_bytes, fr_element_bytes), "L_qry tensor retained row size", dataset), "L_qry tensor retained workspace", dataset)
    aux_rows, aux_columns = aux_tensor_shape
    aux_retained = capacity_mul(aux_rows, capacity_add((g1_element_bytes, fr_element_bytes), "Aux tensor retained row size", dataset), "Aux tensor retained workspace", dataset)
    aux_transient = capacity_add((capacity_mul(aux_rows, fr_element_bytes, "Aux tensor row weights", dataset), capacity_mul(aux_columns, tensor_column_bytes, "Aux tensor column transient", dataset)), "Aux tensor opening transient", dataset)
    l_query_transient = capacity_add((capacity_mul(tensor_rows, fr_element_bytes, "L_qry tensor row weights", dataset), capacity_mul(expected_l_query_columns, tensor_column_bytes, "L_qry tensor column transient", dataset)), "L_qry tensor opening transient", dataset)
    expected_tensor_retained = capacity_add((l_query_retained, aux_retained), "simultaneous Aux/L tensor retained state", dataset)
    expected_tensor_transient = max(aux_transient, l_query_transient)
    tensor_combined_expected = capacity_add((expected_tensor_retained, expected_tensor_transient), "combined Aux/L tensor workspace", dataset)
    if payload["hyrax_tensor_retained_workspace_bytes"] != expected_tensor_retained or payload["hyrax_tensor_opening_transient_peak_workspace_bytes"] != expected_tensor_transient or payload["hyrax_tensor_combined_peak_workspace_bytes"] != tensor_combined_expected or payload["hyrax_l_qry_tensor_workspace_bytes"] != capacity_add((l_query_retained, l_query_transient), "L_qry tensor opening workspace", dataset) or payload["hyrax_aux_tensor_workspace_bytes"] != capacity_add((aux_retained, aux_transient), "Aux tensor opening workspace", dataset) or payload["l_query_zero_interval_count_upper_bound"] < zero_interval_count:
        raise ChainError(f"capacity simultaneous Aux/L tensor workspace is inconsistent for {dataset}")
    family_messages = capacity_add((profile["family_message_count_upper_bound"] for profile in profiles), "Lasso family messages", dataset)
    family_wire = capacity_add((profile["family_wire_bytes_upper_bound"] for profile in profiles), "Lasso family wire", dataset)
    protocol_messages = capacity_add((family_messages, payload["gkr_message_count_upper_bound"]), "Lasso/GKR global messages", dataset)
    protocol_wire = capacity_add((family_wire, payload["gkr_wire_bytes_upper_bound"]), "Lasso/GKR global wire", dataset)
    if payload["proof_message_count_upper_bound"] < capacity_add((protocol_messages, 256), "minimum fixed protocol messages", dataset):
        raise ChainError(f"capacity global message bound omits Lasso/GKR protocol messages for {dataset}")
    remaining_message_count = payload["proof_message_count_upper_bound"] - protocol_messages
    small_message_wire = capacity_mul(remaining_message_count, 2048, "remaining transcript message wire", dataset)
    one_copy_transcript_wire = capacity_add((protocol_wire, small_message_wire), "one-copy transcript wire", dataset)
    duplicate_transcript_wire = capacity_mul(2, one_copy_transcript_wire, "duplicated proof-section/native transcript wire", dataset)
    structured_tensor_rows = capacity_add((aux_tensor_shape[0], tensor_rows), "structured Aux/L_qry commitment rows", dataset)
    structured_tensor_wire = capacity_mul(structured_tensor_rows, 8 + 64, "structured Aux/L_qry commitment wire", dataset)
    minimum_artifact_wire = capacity_add((duplicate_transcript_wire, structured_tensor_wire, 1 << 20), "serialized artifact wire minimum", dataset)
    if payload["proof_artifact_wire_bytes_upper_bound"] < minimum_artifact_wire:
        raise ChainError(f"capacity global transcript bounds omit Lasso/GKR protocol messages for {dataset}")
    if payload["n_in"] < payload["n_raw"] or payload["n_in"] & (payload["n_in"] - 1):
        raise ChainError(f"capacity input padding length is invalid for {dataset}")
    flattened_input = capacity_mul(4, payload["n_in"], "flattened input domain", dataset)
    if payload["n_in"] > payload["coordinate_domain_limit_u64"] or flattened_input != payload["flattened_input_domain_entries"] or flattened_input > payload["flattened_input_domain_limit_u64"]:
        raise ChainError(f"capacity input coordinates exceed the active domain limit for {dataset}")
    disk_requirement = capacity_add((payload["estimated_native_scratch_bytes"], payload["artifact_disk_bytes_upper_bound"]), "combined scratch/artifact requirement", dataset)
    if payload["disk_requirement_bytes_upper_bound"] != disk_requirement:
        raise ChainError(f"capacity combined disk requirement is inconsistent for {dataset}")
    host_native_and_lasso = capacity_add(
        (payload["estimated_native_heap_bytes_upper_bound"], payload["lasso_heap_workspace_peak_bytes"]),
        "native and Lasso host heap requirements",
        dataset,
    )
    if payload["estimated_peak_host_memory_bytes"] < host_native_and_lasso:
        raise ChainError(f"capacity host peak omits native or Lasso standard-heap workspace for {dataset}")
    if payload["estimated_native_scratch_bytes"] < payload["estimated_native_mapped_bytes_upper_bound"]:
        raise ChainError(f"capacity scratch requirement omits the native mapped-byte upper bound for {dataset}")
    if require_fits and payload["estimated_peak_host_memory_bytes"] > payload["host_memory_limit_bytes"]:
        raise ChainError(f"capacity host-memory estimate exceeds available headroom for {dataset}")
    if require_fits and payload["proof_message_count_upper_bound"] > payload["proof_message_count_limit"]:
        raise ChainError(f"capacity proof message count exceeds its native limit for {dataset}")
    if require_fits and payload["maximum_message_string_bytes_upper_bound"] > payload["maximum_message_string_bytes_limit"]:
        raise ChainError(f"capacity message string exceeds its native limit for {dataset}")
    if require_fits and (payload["proof_artifact_wire_limit_bytes"] == 0 or payload["proof_artifact_wire_bytes_upper_bound"] > payload["proof_artifact_wire_limit_bytes"]):
        raise ChainError(f"capacity proof artifact exceeds its trusted wire limit for {dataset}")
    proof_context_bytes = capacity_add((payload["proof_artifact_wire_bytes_upper_bound"], payload["context_recipe_bytes_bound"], payload["context_payload_bytes_bound"]), "proof/context artifact disk bound", dataset)
    if payload["artifact_disk_bytes_upper_bound"] < proof_context_bytes:
        raise ChainError(f"capacity disk bound omits proof/context outputs for {dataset}")
    expected_artifact_limit = capacity_add(
        (
            1 << 30,
            capacity_mul(2, capacity_add((capacity_mul(profile["workspace"]["rows"], 284, "trusted row artifact overhead", dataset) for profile in profiles), "trusted row artifact overhead", dataset), "trusted row artifact overhead", dataset),
            capacity_mul(2, capacity_add((capacity_mul(profile["row_commitment_chunk_count"], 2048, "trusted row chunk overhead", dataset) for profile in profiles), "trusted row chunk overhead", dataset), "trusted row chunk overhead", dataset),
            capacity_mul(len(profiles), 64 << 10, "trusted family headers", dataset),
        ),
        "trusted artifact limit",
        dataset,
    )
    if expected_artifact_limit > (4 << 30):
        expected_artifact_limit = 0
    if payload["proof_artifact_wire_limit_bytes"] != expected_artifact_limit:
        raise ChainError(f"capacity proof artifact limit disagrees with its trusted shape bound for {dataset}")
    if payload["proof_message_count_limit"] != (1 << 20) or payload["maximum_message_string_bytes_limit"] != (1 << 26) or payload["coordinate_domain_limit_u64"] != (1 << 60) or payload["flattened_input_domain_limit_u64"] != (1 << 62):
        raise ChainError(f"capacity report changed an active native/domain limit for {dataset}")
    input_bits = flattened_input.bit_length() - 1
    gkr_max_fields = max(9, capacity_add((payload["gkr_input_source_count_upper_bound"], 1), "GKR source field width", dataset), input_bits)
    family_max_fields = max(
        9,
        max(
            max(
                profile["workspace"]["booleanity_rounds"],
                profile["workspace"]["onehot_rounds"],
                profile["workspace"]["lookup_rounds"],
                profile["a_query_challenges"],
                profile["workspace"]["columns"].bit_length() - 1,
                l_query_column_rounds,
            )
            for profile in profiles
        ),
    )
    expected_maximum_transcript_fields = max(gkr_max_fields, family_max_fields)
    row_message_maximum = max(
        (
            capacity_add((8, capacity_mul(min(profile["workspace"]["rows"], 4096), 8 + 64, "maximum Lasso row chunk payload", dataset)), "maximum Lasso row chunk payload", dataset)
            for profile in profiles
        ),
        default=0,
    )
    gkr_message_maximum = max(2048, capacity_add((8, capacity_mul(gkr_max_fields, 8 + 64, "maximum GKR message payload", dataset)), "maximum GKR message payload", dataset))
    family_message_maximum = capacity_add((8, capacity_mul(family_max_fields, 8 + 64, "maximum Lasso field vector payload", dataset)), "maximum Lasso field vector payload", dataset)
    # The construction material is committed to a fixed SHA-256 hex digest
    # before transcript absorption; only this 64-byte payload is serialized.
    source_ledger_maximum = 64
    expected_message_string_maximum = max(row_message_maximum, gkr_message_maximum, family_message_maximum, source_ledger_maximum)
    if payload["maximum_transcript_field_entries_upper_bound"] != expected_maximum_transcript_fields or payload["maximum_message_string_bytes_upper_bound"] < expected_message_string_maximum:
        raise ChainError(f"capacity transcript field/string maxima omit an active protocol payload for {dataset}")
    if payload["scratch_artifact_same_filesystem"] and (payload["shared_scratch_artifact_available_bytes"] > payload["native_scratch_available_bytes"] or payload["shared_scratch_artifact_available_bytes"] > payload["artifact_disk_available_bytes"]):
        raise ChainError(f"capacity shared filesystem headroom exceeds a per-path observation for {dataset}")
    if not payload["scratch_artifact_same_filesystem"] and payload["shared_scratch_artifact_available_bytes"] != 0:
        raise ChainError(f"capacity reports shared filesystem headroom for distinct filesystems for {dataset}")
    if require_fits and payload["scratch_artifact_same_filesystem"]:
        if disk_requirement > payload["shared_scratch_artifact_available_bytes"]:
            raise ChainError(f"capacity combined scratch/artifact requirement exceeds shared filesystem headroom for {dataset}")
    elif require_fits and (payload["estimated_native_scratch_bytes"] > payload["native_scratch_available_bytes"] or payload["artifact_disk_bytes_upper_bound"] > payload["artifact_disk_available_bytes"]):
        raise ChainError(f"capacity scratch/artifact requirement exceeds filesystem headroom for {dataset}")
    expected_total_transcript_fields = capacity_mul(payload["proof_message_count_upper_bound"], payload["maximum_transcript_field_entries_upper_bound"], "total transcript field entries", dataset)
    if payload["total_transcript_field_entries_upper_bound"] != expected_total_transcript_fields:
        raise ChainError(f"capacity transcript field-count bounds are inconsistent for {dataset}")
    if require_fits and any(payload[key] == 0 for key in ("estimated_peak_host_memory_bytes", "host_memory_limit_bytes", "native_scratch_available_bytes", "artifact_disk_available_bytes", "max_materialized_fr_bytes", "max_materialized_g1_bytes", "max_materialized_gate_bytes", "max_materialized_vector_bytes", "hyrax_active_domain_workspace_entries", "hyrax_active_domain_workspace_bytes", "hyrax_l_qry_tensor_columns", "hyrax_l_qry_tensor_workspace_bytes", "hyrax_aux_tensor_workspace_bytes", "hyrax_folded_columns_workspace_bytes", "hyrax_tensor_retained_workspace_bytes", "hyrax_tensor_opening_transient_peak_workspace_bytes", "hyrax_tensor_combined_peak_workspace_bytes", "hyrax_w_mle_workspace_bytes", "proof_message_count_upper_bound", "maximum_transcript_field_entries_upper_bound", "total_transcript_field_entries_upper_bound", "maximum_message_string_bytes_upper_bound", "proof_artifact_wire_bytes_upper_bound", "artifact_disk_bytes_upper_bound")):
        raise ChainError(f"capacity reports zero for a required active resource on {dataset}")
    if require_fits and (payload["ppi_node_ptr_boundary"] is not True or payload["ppi_edge_ptr_boundary"] is not True):
        raise ChainError(f"capacity has invalid graph-batch boundaries for {dataset}")
    native_backing_upper = capacity_add(
        (payload["estimated_native_heap_bytes_upper_bound"], payload["estimated_native_mapped_bytes_upper_bound"]),
        "native heap and mapped workspace upper bound",
        dataset,
    )
    mixed_native_workspace_upper = capacity_add(
        (native_backing_upper, payload["lasso_heap_workspace_peak_bytes"]),
        "native and Lasso mixed workspace upper bound",
        dataset,
    )
    for key in ("hyrax_active_domain_workspace_bytes", "hyrax_folded_columns_workspace_bytes", "gkr_prefix_workspace_bytes_upper_bound", "lasso_private_selected_row_workspace_peak_bytes", "max_materialized_fr_bytes", "max_materialized_g1_bytes", "max_materialized_gate_bytes", "max_materialized_vector_bytes"):
        if payload[key] > mixed_native_workspace_upper:
            raise ChainError(f"capacity {key} exceeds checked native heap/mapped backing for {dataset}")
    for key in ("hyrax_l_qry_tensor_workspace_bytes", "hyrax_aux_tensor_workspace_bytes", "hyrax_tensor_combined_peak_workspace_bytes", "hyrax_w_mle_workspace_bytes", "borrowed_trace_bytes_upper_bound", "lasso_table_and_index_bytes_upper_bound"):
        if payload[key] > payload["estimated_peak_host_memory_bytes"]:
            raise ChainError(f"capacity {key} exceeds estimated host peak for {dataset}")


def check_source(root: Path) -> List[str]:
    required = (
        root / "gat_adapter" / "gat_cli.cpp",
        root / "gat_adapter" / "gat_verifier_cli.cpp",
        root / "gat_adapter" / "gat_proof.cpp",
        root / "gat_adapter" / "CMakeLists.txt",
        root / "build.sh",
    )
    errors: List[str] = []
    for path in required:
        try:
            require_regular(path, "production source")
        except ChainError as exc:
            errors.append(str(exc))
    return errors


def check_capabilities(root: Path, backend: str, executable: Path) -> Dict[str, Any]:
    completed = subprocess.run([str(executable), "--capabilities"], cwd=str(root),
                               text=True, capture_output=True, env=formal_child_environment(), check=False)
    if completed.returncode != 0:
        raise ChainError(f"{backend} verifier capabilities failed with {completed.returncode}")
    try:
        payload = json.loads(completed.stdout)
    except ValueError as exc:
        raise ChainError(f"{backend} verifier capabilities are not JSON") from exc
    if not isinstance(payload, dict) or payload.get("schema") != "zkgpt-gat-verifier-capabilities-v2":
        raise ChainError(f"{backend} verifier capabilities schema is unsupported")
    for key in ("independent", "reads_witness", "reads_checkpoint"):
        require_bool(payload, key, key == "independent", f"{backend} capabilities")
    if payload.get("masked_full") is not True:
        raise ChainError(f"{backend} verifier declares masked_full unsupported")
    return payload


def run_capacity(root: Path, dataset: str, backend: str, log_dir: Path) -> Dict[str, Any]:
    config_path, config = config_with_auth(root, dataset)
    prover = build_binary(root, backend)
    ensure_directory(log_dir, "capacity diagnostic log directory", root)
    stdout_log = log_dir / f"{dataset}-{backend}.stdout.log"
    stderr_log = log_dir / f"{dataset}-{backend}.stderr.log"
    print(f"CAPACITY_ADMISSION_BEGIN dataset={dataset} backend={backend}", file=sys.stderr, flush=True)
    completed = subprocess.run(
        [str(prover), "--capacity-only", "--dataset", dataset, "--backend", backend,
         "--baseline-root", str(root), "--config", relative_path(root, config_path, "dataset config")],
        cwd=str(root), text=True, encoding="utf-8", errors="replace", stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, env=formal_child_environment(), check=False,
    )
    atomic_write_text(stdout_log, completed.stdout, root)
    atomic_write_text(stderr_log, completed.stderr, root)
    print(f"CAPACITY_NATIVE_LOGS dataset={dataset} stdout={relative_path(root, stdout_log, 'capacity stdout log')} stderr={relative_path(root, stderr_log, 'capacity stderr log')}", file=sys.stderr, flush=True)
    if completed.stderr:
        sys.stderr.write(completed.stderr)
        if not completed.stderr.endswith("\n"):
            sys.stderr.write("\n")
        sys.stderr.flush()
    try:
        payload = json.loads(completed.stdout, parse_constant=reject_nonfinite_json_constant, parse_float=reject_floating_json_number, parse_int=parse_capacity_integer)
    except ValueError as exc:
        raise ChainError(
            f"capacity output is not strict integer JSON for {dataset}/{backend} "
            f"(exit={completed.returncode}, stdout_chars={len(completed.stdout)}, "
            f"stdout_prefix={completed.stdout[:128]!r}, parse_error={exc}); "
            f"native logs: stdout={relative_path(root, stdout_log, 'capacity stdout log')}, "
            f"stderr={relative_path(root, stderr_log, 'capacity stderr log')}"
        ) from exc
    if not isinstance(payload, dict) or payload.get("schema") != CAPACITY_SCHEMA:
        raise ChainError(f"unsupported capacity schema for {dataset}/{backend}")
    if payload.get("profile_kind") != "formal_admission":
        raise ChainError(f"analysis-only capacity profile cannot satisfy formal admission for {dataset}/{backend}")
    if payload.get("dataset") != dataset or payload.get("backend") != backend or payload.get("status") not in ("READY", "BLOCKED"):
        raise ChainError(f"capacity has an invalid status for {dataset}/{backend}")
    if payload.get("memory_accounting_kind") != "host-only":
        raise ChainError(f"capacity has an unsupported memory accounting mode for {dataset}/{backend}")
    if capacity_u64(payload.get("device_peak_bytes"), "device_peak_bytes", dataset) != 0:
        raise ChainError(f"CPU formal capacity unexpectedly accounts for device allocation on {dataset}/{backend}")
    if type(payload.get("ready")) is not bool or payload["ready"] != (payload["status"] == "READY"):
        raise ChainError(f"capacity status/ready mismatch for {dataset}/{backend}")
    if type(payload.get("full_relation_bound")) is not bool or not isinstance(payload.get("proof_path"), str) or not isinstance(payload.get("reason"), str):
        raise ChainError(f"capacity has invalid path or relation metadata for {dataset}/{backend}")
    validate_capacity_numbers(payload, dataset, require_fits=payload["ready"])
    if payload.get("status") == "READY" and (payload.get("proof_path") != CAPACITY_PROOF_PATH or payload.get("full_relation_bound") is not True):
        raise ChainError(f"capacity selected an unsupported or incomplete full-GAT proof path for {dataset}/{backend}")
    if payload.get("status") == "READY" and payload.get("reason", "") != "":
        raise ChainError(f"capacity READY report carries a blocker reason for {dataset}/{backend}")
    if payload.get("status") == "BLOCKED" and (not isinstance(payload.get("reason"), str) or not payload["reason"]):
        raise ChainError(f"capacity blocker has no reason for {dataset}/{backend}")
    try:
        validate_canonical_graph_identity(payload, dataset, config)
    except ChainError as exc:
        raise ChainError(f"{exc}; native logs: stdout={relative_path(root, stdout_log, 'capacity stdout log')}, stderr={relative_path(root, stderr_log, 'capacity stderr log')}") from exc

    print(
        f"CAPACITY_ADMISSION_END dataset={dataset} status={payload.get('status')} "
        f"point_shape={payload.get('gkr_point_shape_ready')} "
        f"all_value={payload.get('forward_value_admission_ready')} "
        f"host_peak_bytes={payload.get('estimated_peak_host_memory_bytes')} "
        f"disk_peak_bytes={payload.get('disk_requirement_bytes_upper_bound')} "
        f"artifact_bytes={payload.get('artifact_disk_bytes_upper_bound')}",
        file=sys.stderr,
        flush=True,
    )
    if completed.returncode != 0 and payload.get("status") != "BLOCKED":
        raise ChainError(f"capacity preflight failed for {dataset}/{backend} with exit={completed.returncode}")
    return payload


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(list(argv))
    root = baseline_root()
    try:
        session_id, session, session_manifest_sha256 = require_formal_session(root)
        selected = FORMAL_BACKENDS
        preflight_path = formal_preflight_path(root, session_id)
        if preflight_path.exists():
            preflight_path.unlink()
    except (ChainError, OSError, ValueError) as exc:
        print(f"FORMAL_PREFLIGHT_BLOCKED: {exc}", file=sys.stderr)
        return 2
    blockers: List[str] = []
    source_errors = check_source(root)
    blockers.extend(source_errors)
    resource_errors = verify_local_resources(root)
    blockers.extend(resource_errors)
    backend_report: Dict[str, Any] = {}
    capacity: Dict[str, Any] = {}
    resource_manifest_digest: Optional[str] = None
    resource_manifest_fp: Optional[Dict[str, Any]] = None
    try:
        resource_manifest = require_manifest(root, "resource_manifest.json", "zkgpt-gat-resource-manifest-v1")
        del resource_manifest
        resource_manifest_path = root / "results" / "resource_manifest.json"
        resource_manifest_digest = sha256_file(resource_manifest_path)
        resource_manifest_fp = file_fingerprint(resource_manifest_path, "resource manifest")
    except ChainError as exc:
        blockers.append(str(exc))
    for backend in selected:
        entry: Dict[str, Any] = {"prover": None, "verifier": None, "capabilities": None, "ready": False}
        capability = formal_backend_capability(backend)
        entry["capability_status"] = capability["status"]
        entry["formal_backend_supported"] = capability["supported"]
        if not capability["supported"]:
            entry["blocker"] = capability["reason"]
            blockers.append(capability["reason"])
            backend_report[backend] = entry
            continue
        try:
            prover = build_binary(root, backend)
            verifier = verifier_binary(root, backend)
            entry["source_tree_sha256"] = verify_build_provenance(root, backend)
            entry["prover"] = relative_path(root, prover, "prover executable")
            entry["verifier"] = relative_path(root, verifier, "verifier executable")
            entry["prover_fingerprint"] = binary_fingerprint(prover, "prover executable")
            entry["verifier_fingerprint"] = binary_fingerprint(verifier, "verifier executable")
            entry["capabilities"] = check_capabilities(root, backend, verifier)
            entry["ready"] = True
        except (ChainError, OSError, ValueError, subprocess.SubprocessError) as exc:
            entry["blocker"] = str(exc)
            blockers.append(str(exc))
        backend_report[backend] = entry
    capacity_log_dir = preflight_path.parent / "capacity_logs"
    if not blockers and "cpu" in selected:
        for dataset in DATASETS:
            try:
                capacity[dataset] = run_capacity(root, dataset, "cpu", capacity_log_dir)
            except (ChainError, OSError, ValueError, subprocess.SubprocessError) as exc:
                stdout_log = capacity_log_dir / f"{dataset}-cpu.stdout.log"
                stderr_log = capacity_log_dir / f"{dataset}-cpu.stderr.log"
                log_note = ""
                if stdout_log.is_file() and stderr_log.is_file():
                    log_note = f"; native logs: stdout={relative_path(root, stdout_log, 'capacity stdout log')}, stderr={relative_path(root, stderr_log, 'capacity stderr log')}"
                capacity[dataset] = {"schema": CAPACITY_SCHEMA, "status": "BLOCKED", "dataset": dataset, "backend": "cpu", "reason": f"{exc}{log_note}"}
    all_capacity_ready = set(capacity) == set(DATASETS) and all(
        isinstance(capacity.get(dataset), dict) and capacity[dataset].get("status") == "READY"
        for dataset in DATASETS
    )
    capacity_matrix_valid = set(capacity) == set(DATASETS) and all(
        isinstance(capacity.get(dataset), dict) and capacity[dataset].get("status") in ("READY", "BLOCKED")
        for dataset in DATASETS
    )
    preflight_status = "BLOCKED" if blockers or not capacity_matrix_valid else ("READY" if all_capacity_ready else "PARTIAL")
    payload = {
        "schema": PREFLIGHT_SCHEMA,
        "session_id": session_id,
        "session_manifest_sha256": session_manifest_sha256,
        "runtime_source_sha256": runtime_source_sha256(root),
        "native_source_sha256": session.get("native_source_sha256"),
        "status": preflight_status,
        "formal_experiment_eligible": preflight_status == "READY",
        "shared_preconditions_ready": not blockers,
        "source_ready": not source_errors,
        "resources_ready": not resource_errors,
        "backends": backend_report,
        "capacity": capacity,
        "readiness_scope": "formal_execution_preconditions",
        "static_checks_passed": preflight_status == "READY",
        "development_e2e_checked": False,
        "development_e2e_check": "tests/check_formal_closure.py (separate development evidence)",
        "formal_results_complete": False,
        "resource_manifest_sha256": resource_manifest_digest,
        "resource_manifest_fingerprint": resource_manifest_fp,
        "blockers": blockers,
        "formal_datasets_started": False,
        "no_fallback": True,
        "verifier_reads_checkpoint": False,
        "verifier_reads_witness": False,
    }
    try:
        atomic_write_json(preflight_path, payload)
    except ChainError as exc:
        print(f"PREFLIGHT_OUTPUT_FAILED: {exc}", file=sys.stderr)
        return 2
    if preflight_status == "BLOCKED":
        print("FORMAL_PREFLIGHT_BLOCKED", file=sys.stderr)
        for blocker in blockers:
            print(f"- {blocker}", file=sys.stderr)
        return 2
    if preflight_status == "PARTIAL":
        print("FORMAL_PREFLIGHT_PARTIAL")
        for dataset in DATASETS:
            item = capacity[dataset]
            if item.get("status") == "BLOCKED":
                print(f"CAPACITY_BLOCKED dataset={dataset}: {item.get('reason')}")
        return 0
    print("FORMAL_PREFLIGHT_READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
