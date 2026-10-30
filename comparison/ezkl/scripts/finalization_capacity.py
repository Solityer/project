#!/usr/bin/env python3
"""Source-derived, fail-closed capacity gate for EZKL final collection."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from collect_results import build_matrix_document
from formal_common import (
    DATASETS,
    ROOT,
    has_compact_native_proof_v2_prefix,
    normalize_selected_datasets,
    selected_datasets_for_session,
    sha256,
)

MAX_RESULT_JSON_BYTES = 4 * 1024 * 1024
MAX_SESSION_JSON_BYTES = 1024 * 1024
MAX_SMALL_JSON_BYTES = 1024 * 1024
MAX_MATRIX_JSON_BYTES = 32 * 1024 * 1024
MAX_META_CFG_BYTES = 64 * 1024
MAX_RESOURCE_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_EDGE_SOURCE_BYTES = 64 * 1024 * 1024
SESSION_TEMP_ALLOWANCE_BYTES = 64 * 1024
INTERPRETER_FIXED_FLOOR_BYTES = 64 * 1024 * 1024
PROOF_DOM_WIRE_FACTOR_MIN = 112.5
PROOF_TEXT_WIRE_FACTOR = 4.0
PROOF_CANONICAL_WIRE_FACTOR = 6.0
PROOF_RESCALED_FLOAT_TEXT_MAX_BYTES = 328
PROOF_SCHEMA_FIXED_DOM_ALLOWANCE_BYTES = 1024 * 1024
CPYTHON_ALLOCATOR_SLACK_BYTES_PER_OBJECT = 16
STATEMENT_METADATA_WIRE_ALLOWANCE_BYTES = 64 * 1024
STATEMENT_METADATA_DOM_ALLOWANCE_BYTES = 1024 * 1024
ARRAY_ELEMENT_PUNCTUATION_BYTES = 12


class FinalizationCapacityError(RuntimeError):
    pass


def _fail(message: str) -> None:
    raise FinalizationCapacityError(message)


def _rooted_path(value: Path, label: str) -> Path:
    path = value if value.is_absolute() else ROOT / value
    if ".." in path.parts:
        _fail(f"{label} contains a parent-directory component")
    try:
        relative = path.relative_to(ROOT)
    except ValueError:
        _fail(f"{label} escapes the EZKL baseline")
    cursor = ROOT
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            _fail(f"{label} must not contain a symlink: {cursor}")
    if path.resolve(strict=False).parent != path.parent.resolve(strict=False) and path.exists():
        _fail(f"{label} resolves through an unexpected path: {path}")
    return path


def _regular_file(path: Path, label: str) -> os.stat_result:
    try:
        info = path.stat(follow_symlinks=False)
    except OSError as exc:
        _fail(f"cannot stat {label}: {exc}")
    if not stat.S_ISREG(info.st_mode):
        _fail(f"{label} is not a regular file: {path}")
    return info


def _stat_record(info: os.stat_result) -> dict[str, int]:
    return {"device_id": int(info.st_dev), "inode": int(info.st_ino), "size_bytes": int(info.st_size), "mtime_ns": int(info.st_mtime_ns)}


def _read_json(path: Path, label: str, limit: int) -> Any:
    info = _regular_file(path, label)
    if info.st_size > limit:
        _fail(f"{label} exceeds the bounded metadata JSON read limit ({info.st_size} > {limit})")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        _fail(f"cannot read bounded {label}: {exc}")


def _current_host_available() -> tuple[int, dict[str, Any]]:
    available: int | None = None
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            key, separator, value = line.partition(":")
            if separator and key == "MemAvailable":
                parts = value.split()
                if parts and parts[0].isdigit():
                    available = int(parts[0]) * 1024
                    break
    except OSError as exc:
        _fail(f"cannot sample host MemAvailable: {exc}")
    if available is None:
        _fail("host MemAvailable is absent or malformed")
    effective = available
    limits: list[dict[str, Any]] = []
    try:
        cgroups = Path("/proc/self/cgroup").read_text(encoding="ascii").splitlines()
    except OSError as exc:
        _fail(f"cannot sample cgroup memory limits: {exc}")
    candidates: list[Path] = []
    for line in cgroups:
        hierarchy, _controllers, relative = line.split(":", 2)
        if hierarchy == "0":
            base = Path("/sys/fs/cgroup") / relative.lstrip("/")
            candidates.extend((base, *base.parents))
            break
    candidates.append(Path("/sys/fs/cgroup"))
    for base in dict.fromkeys(candidates):
        try:
            maximum = (base / "memory.max").read_text(encoding="ascii").strip()
            current = int((base / "memory.current").read_text(encoding="ascii").strip())
        except (OSError, ValueError):
            continue
        if maximum == "max":
            continue
        if not maximum.isdigit():
            _fail(f"malformed cgroup memory.max at {base}")
        cgroup_available = max(0, int(maximum) - current)
        effective = min(effective, cgroup_available)
        limits.append({"path": str(base), "limit_bytes": int(maximum), "current_bytes": current, "available_bytes": cgroup_available})
    return effective, {"mem_available_bytes": available, "cgroup_limits": limits, "effective_available_bytes": effective}


def _filesystem(path: Path) -> tuple[int, int]:
    cursor = path
    while not cursor.exists():
        if cursor == cursor.parent:
            _fail(f"no existing parent filesystem for {path}")
        cursor = cursor.parent
    if cursor.is_symlink() or not cursor.is_dir():
        _fail(f"filesystem probe path is not a real directory: {cursor}")
    try:
        info = os.statvfs(cursor)
        device_id = int(os.stat(cursor, follow_symlinks=False).st_dev)
    except OSError as exc:
        _fail(f"cannot sample filesystem capacity for {path}: {exc}")
    return int(info.f_bavail) * int(info.f_frsize), device_id


def _runtime_dom_wire_factor() -> float:
    pointer = struct.calcsize("P")
    sizes = (sys.getsizeof({}), sys.getsizeof([]), sys.getsizeof(""), sys.getsizeof(0), sys.getsizeof(0.0))
    # Seven container/scalar header sets per sixteen wire bytes deliberately
    # overcounts CPython string headers, float/int objects, dict/list headers,
    # and pointer slots for the pinned Snark JSON shape.
    measured = math.ceil((7 * sum(sizes) + 32 * pointer) / 16)
    return max(PROOF_DOM_WIRE_FACTOR_MIN, float(measured))


def typed_statement_requirements(*, num_nodes: int, num_features: int, num_classes: int, raw_edge_lines: int, graph_count: int) -> dict[str, int]:
    """Bound the canonical producer's typed JSON lists without parsing the artifact."""
    values = (num_nodes, num_features, num_classes, raw_edge_lines, graph_count)
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values):
        _fail("canonical EZKL statement shape contains an invalid dimension")
    if graph_count > num_nodes:
        _fail("canonical EZKL graph count exceeds its node count")
    feature_count = num_nodes * num_features
    y_count = num_nodes * num_classes
    # The producer can materialize both directions for every source edge and
    # then add at most one self-loop per node.
    edge_count_upper = 2 * raw_edge_lines + num_nodes
    int64_digits = len(str(1 << 63)) + 1
    endpoint_digits = len(str(max(num_nodes, edge_count_upper)))
    list_item_wire = lambda digits: digits + ARRAY_ELEMENT_PUNCTUATION_BYTES
    statement_wire = (
        STATEMENT_METADATA_WIRE_ALLOWANCE_BYTES
        + feature_count * list_item_wire(int64_digits)
        + num_nodes * list_item_wire(len(str(max(0, num_nodes - 1))))
        + 2 * edge_count_upper * list_item_wire(len(str(max(0, num_nodes - 1))))
        + (graph_count + 1) * list_item_wire(len(str(num_nodes)) + 1)
        + (graph_count + 1) * list_item_wire(endpoint_digits)
        + y_count * list_item_wire(int64_digits)
    )
    public_output_wire = 4096 + y_count * list_item_wire(int64_digits)
    pointer_bytes = struct.calcsize("P")
    # All large statement/output values are Python ints loaded from the producer's
    # int64 arrays. getsizeof(2**63) covers the widest signed-int64 value.
    int_item_bytes = sys.getsizeof(1 << 63) + CPYTHON_ALLOCATOR_SLACK_BYTES_PER_OBJECT + 3 * pointer_bytes
    list_headers = 12 * sys.getsizeof([]) + 8 * sys.getsizeof({})
    metadata_dom = STATEMENT_METADATA_DOM_ALLOWANCE_BYTES
    statement_dom = (
        metadata_dom
        + list_headers
        + (feature_count + num_nodes + 2 * edge_count_upper + 2 * (graph_count + 1) + y_count) * int_item_bytes
    )
    output_dom = metadata_dom // 4 + 2 * sys.getsizeof([]) + sys.getsizeof({}) + y_count * int_item_bytes
    # validate_public_statement constructs set(node_ids), then copies one graph's
    # src/dst slices, and builds bytearray(16*Y) before bytes(output_bytes).
    set_slots_upper = 6 * pointer_bytes * num_nodes
    graph_slice_upper = 2 * pointer_bytes * edge_count_upper + 2 * sys.getsizeof([])
    y_digest_buffers_upper = 34 * y_count + sys.getsizeof(bytearray()) + len(b"ezkl-gat-public-y-v1")
    statement_validation_work = set_slots_upper + graph_slice_upper + y_digest_buffers_upper
    statement_text_upper = 4 * statement_wire
    output_text_upper = 4 * public_output_wire
    return {
        "num_nodes": num_nodes,
        "num_features": num_features,
        "num_classes": num_classes,
        "raw_edge_lines": raw_edge_lines,
        "graph_count": graph_count,
        "feature_values": feature_count,
        "edge_count_upper_after_undirected_expansion_and_self_loops": edge_count_upper,
        "public_y_values": y_count,
        "statement_wire_bytes_upper": statement_wire,
        "public_output_wire_bytes_upper": public_output_wire,
        "statement_text_bytes_upper_unicode_safe": statement_text_upper,
        "public_output_text_bytes_upper_unicode_safe": output_text_upper,
        "statement_dom_bytes_upper_typed_int64": statement_dom,
        "public_output_dom_bytes_upper_typed_int64": output_dom,
        "node_id_set_bytes_upper": set_slots_upper,
        "graph_src_dst_slice_bytes_upper": graph_slice_upper,
        "public_y_digest_bytearray_and_bytes_upper": y_digest_buffers_upper,
        "statement_validation_work_bytes_upper": statement_validation_work,
    }


def typed_proof_requirements(
    *,
    proof_size_bytes: int,
    statement_shape: Mapping[str, int],
    compact_v2: bool = False,
) -> dict[str, int | float | bool]:
    """Bound legacy Snark and compact-v2 JSON by their source-defined schemas.

    Legacy proofs retain the original Fr/proof/PrettyElements lower bounds.
    Compact v2 uses a conservative small-document DOM and read-text envelope,
    since its instances are reconstructed from authenticated statement files.
    """
    if isinstance(proof_size_bytes, bool) or not isinstance(proof_size_bytes, int) or proof_size_bytes <= 0:
        _fail("native proof JSON stat size is invalid")
    required = ("num_nodes", "num_features", "num_classes", "graph_count", "feature_values", "edge_count_upper_after_undirected_expansion_and_self_loops", "public_y_values")
    if any(isinstance(statement_shape.get(key), bool) or not isinstance(statement_shape.get(key), int) or statement_shape[key] <= 0 for key in required):
        _fail("native proof capacity shape is missing a positive producer-derived dimension")

    if compact_v2:
        # Compact v2 contains only a native transcript hex string and fixed-size
        # authenticated references. It has no embedded Fr or PrettyElements
        # arrays, so the legacy lower bound is inapplicable. Retain a deliberately
        # conservative recursive-DOM multiplier and transient-text allowance.
        proof_text_bytes_upper = math.ceil(PROOF_TEXT_WIRE_FACTOR * proof_size_bytes)
        proof_document_dom = (
            math.ceil(PROOF_DOM_WIRE_FACTOR_MIN * proof_size_bytes)
            + PROOF_SCHEMA_FIXED_DOM_ALLOWANCE_BYTES
        )
        proof_json_load_peak = proof_text_bytes_upper + proof_document_dom
        return {
            "proof_file_stat_size_bytes": proof_size_bytes,
            "compact_proof_v2": True,
            "proof_document_dom_bytes_upper_compact_v2": proof_document_dom,
            "proof_text_bytes_upper_ascii_read_and_decode": proof_text_bytes_upper,
            "proof_json_load_peak_bytes_upper": proof_json_load_peak,
            "proof_canonical_digest_peak_bytes_upper": proof_document_dom,
            "proof_peak_bytes_source_derived_typed_snark": proof_json_load_peak,
        }

    nodes = statement_shape["num_nodes"]
    feature_values = statement_shape["feature_values"]
    graph_count = statement_shape["graph_count"]
    edge_upper = statement_shape["edge_count_upper_after_undirected_expansion_and_self_loops"]
    y_values = statement_shape["public_y_values"]
    pointer_bytes = struct.calcsize("P")
    try:
        u8_cache_probe = json.loads("[0,0,255,255]")
    except json.JSONDecodeError as exc:
        _fail(f"cannot confirm current CPython JSON u8 scalar behavior: {exc}")
    json_u8_values_are_cached = u8_cache_probe[0] is u8_cache_probe[1] and u8_cache_probe[2] is u8_cache_probe[3]
    canonical_probe = list(
        json.JSONEncoder(ensure_ascii=False, sort_keys=True, separators=(",", ":")).iterencode(
            [["0" * 64 for _ in range(8)]]
        )
    )
    if len(canonical_probe) > 2 * 8 + 3 or any(len(chunk) > 69 for chunk in canonical_probe):
        _fail("current JSON encoder exceeds the bounded canonical-instance fragment model")

    # execute.rs builds six public graph tensors, then public Y and one hashed
    # model instance. The edge source's deduplication can only reduce E, so the
    # upper bound covers native undirected expansion and self-loops. It is used
    # for allocations while wire subtraction below uses a separately derived
    # lower count that does not pretend E_upper is exact.
    graph_pointer_values = 2 * (graph_count + 1)
    public_values_upper = feature_values + nodes + 2 * edge_upper + graph_pointer_values + y_values
    public_values_lower = feature_values + nodes + graph_pointer_values + y_values
    instances_upper = public_values_upper + 1
    instances_lower = public_values_lower + 1

    instance_wire_upper = 67 * instances_upper + 3
    instance_wire_lower = 67 * instances_lower + 3
    pretty_debug_wire_lower = 68 * public_values_lower
    pretty_rescaled_wire_lower = 3 * public_values_lower
    processed_model_debug_wire_lower = 68
    known_native_wire_lower = (
        instance_wire_lower
        + pretty_debug_wire_lower
        + pretty_rescaled_wire_lower
        + processed_model_debug_wire_lower
    )
    if proof_size_bytes < known_native_wire_lower:
        _fail("native proof JSON is smaller than its source-required public-instance/PrettyElements payload")

    # Every u8 contributes at least one decimal byte plus an array delimiter;
    # the matching 0x string contributes two hex bytes per u8. Thus the two
    # native proof encodings consume at least 4P bytes together. Remaining
    # unclassified bytes are conservatively assigned to P without charging
    # public tensor payloads a generic 112.5x recursive-DOM multiplier.
    proof_vector_bytes_upper = max(0, (proof_size_bytes - known_native_wire_lower) // 4)

    instance_string_object = sys.getsizeof("0" * 64) + CPYTHON_ALLOCATOR_SLACK_BYTES_PER_OBJECT + 3 * pointer_bytes
    pretty_debug_string_object = sys.getsizeof("0x" + "0" * 64) + CPYTHON_ALLOCATOR_SLACK_BYTES_PER_OBJECT + 3 * pointer_bytes
    pretty_float_string_object = sys.getsizeof("0" * PROOF_RESCALED_FLOAT_TEXT_MAX_BYTES) + CPYTHON_ALLOCATOR_SLACK_BYTES_PER_OBJECT + 3 * pointer_bytes
    instance_dom = instances_upper * instance_string_object + 2 * sys.getsizeof([]) + 3 * pointer_bytes

    # PrettyElements stores one raw/debug and one rescaled copy of each public
    # input/output; processed_params adds one model-hash debug string. The
    # separate Fr `instances` copy is accounted for above.
    pretty_dom = (
        public_values_upper * pretty_debug_string_object
        + public_values_upper * pretty_float_string_object
        + pretty_debug_string_object
        + 22 * sys.getsizeof([])
        + 15 * 3 * pointer_bytes
        + sys.getsizeof({})
    )
    proof_vector_per_item = 3 * pointer_bytes
    if not json_u8_values_are_cached:
        proof_vector_per_item += sys.getsizeof(255) + CPYTHON_ALLOCATOR_SLACK_BYTES_PER_OBJECT
    proof_vector_dom = proof_vector_bytes_upper * proof_vector_per_item + sys.getsizeof([])
    if json_u8_values_are_cached:
        proof_vector_dom += 256 * sys.getsizeof(0)
    hex_proof_dom = sys.getsizeof("") + CPYTHON_ALLOCATOR_SLACK_BYTES_PER_OBJECT + 2 * proof_vector_bytes_upper + 2
    proof_document_dom = (
        instance_dom
        + pretty_dom
        + proof_vector_dom
        + hex_proof_dom
        + PROOF_SCHEMA_FIXED_DOM_ALLOWANCE_BYTES
    )

    canonical_instance_wire = instance_wire_upper
    canonical_chunk_object = sys.getsizeof("0" * 69) + CPYTHON_ALLOCATOR_SLACK_BYTES_PER_OBJECT
    canonical_chunk_count_upper = 2 * instances_upper + 3
    canonical_fragment_peak = (
        canonical_instance_wire
        + canonical_chunk_count_upper * canonical_chunk_object
        + 2 * sys.getsizeof([])
        + 3 * canonical_chunk_count_upper * pointer_bytes
    )
    canonical_encode_peak = 2 * canonical_instance_wire  # temporary ASCII str + encoded bytes
    canonical_instances_buffers = max(canonical_fragment_peak, canonical_encode_peak)

    # The Rust serde fields in this formal path are ASCII-only (hex field
    # values, decimal f64 strings, fixed keys/version, and numeric metadata).
    # Two wire bytes per source byte cover the decoded text and a transient
    # read/decode copy. Protocol and split are None in this producer call.
    proof_text_bytes_upper = 2 * proof_size_bytes
    proof_json_load_peak = proof_text_bytes_upper + proof_document_dom
    proof_canonical_digest_peak = proof_document_dom + canonical_instances_buffers
    proof_peak = max(proof_json_load_peak, proof_canonical_digest_peak)
    return {
        "proof_file_stat_size_bytes": proof_size_bytes,
        "public_instance_values_upper": instances_upper,
        "public_instance_values_lower_for_wire_subtraction": instances_lower,
        "public_instance_wire_bytes_upper": instance_wire_upper,
        "known_native_wire_bytes_lower_for_proof_vector_bound": known_native_wire_lower,
        "proof_vector_bytes_upper_from_remaining_wire_div_4": proof_vector_bytes_upper,
        "proof_vector_and_hex_wire_minimum_divisor": 4,
        "public_instance_dom_bytes_upper_typed_fr64": instance_dom,
        "pretty_public_inputs_values_upper": public_values_upper,
        "pretty_debug_string_chars_exact": 66,
        "pretty_rescaled_float_string_chars_upper": PROOF_RESCALED_FLOAT_TEXT_MAX_BYTES,
        "pretty_public_inputs_dom_bytes_upper_typed": pretty_dom,
        "proof_vector_dom_bytes_upper_typed_u8": proof_vector_dom,
        "json_u8_scalar_cache_confirmed": json_u8_values_are_cached,
        "proof_vector_list_pointer_bytes_per_item_upper": proof_vector_per_item,
        "hex_proof_dom_bytes_upper_typed_hex": hex_proof_dom,
        "proof_document_dom_bytes_upper_typed_snark": proof_document_dom,
        "proof_text_bytes_upper_ascii_read_and_decode": proof_text_bytes_upper,
        "canonical_instance_text_and_bytes_source_derived": 2 * canonical_instance_wire,
        "canonical_instance_fragment_peak_bytes_upper": canonical_fragment_peak,
        "canonical_encoder_chunk_count_upper": canonical_chunk_count_upper,
        "canonical_encoder_runtime_probe_confirmed": True,
        "canonical_instance_buffers_bytes_upper": canonical_instances_buffers,
        "proof_json_load_peak_bytes_upper": proof_json_load_peak,
        "proof_canonical_digest_peak_bytes_upper": proof_canonical_digest_peak,
        "proof_peak_bytes_source_derived_typed_snark": proof_peak,
        "protocol_and_split_source_defined_null": True,
    }


def _host_requirements(
    proof_sizes: Mapping[str, int],
    result_sizes: Mapping[str, int],
    matrix_bytes: int,
    *,
    statement_requirements: Mapping[str, Mapping[str, int]] | None = None,
    proof_requirements: Mapping[str, Mapping[str, Any]] | None = None,
    small_json_sizes: Mapping[str, int] | None = None,
    phase: str = "collector",
    datasets: tuple[str, ...] = DATASETS,
) -> dict[str, int | float | str]:
    selected_datasets = normalize_selected_datasets(datasets)
    if set(proof_sizes) != set(selected_datasets) or set(result_sizes) != set(selected_datasets):
        _fail("finalization capacity input does not match the immutable selected EZKL datasets")
    if any(isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in (*proof_sizes.values(), *result_sizes.values())):
        _fail("finalization capacity input has an invalid artifact size")
    if phase not in {"collector", "validator"}:
        _fail("finalization capacity phase must be collector or validator")
    if statement_requirements is not None and set(statement_requirements) != set(selected_datasets):
        _fail("finalization capacity statement bounds do not match the selected EZKL datasets")
    if proof_requirements is not None and set(proof_requirements) != set(selected_datasets):
        _fail("finalization capacity proof bounds do not match the selected EZKL datasets")
    small_json_sizes = small_json_sizes or {}
    if any(isinstance(size, bool) or not isinstance(size, int) or size <= 0 or size > MAX_SMALL_JSON_BYTES for size in small_json_sizes.values()):
        _fail("finalization capacity input has an invalid or oversized small JSON artifact")
    dom_factor = _runtime_dom_wire_factor()
    proof_factor = dom_factor + PROOF_TEXT_WIRE_FACTOR + PROOF_CANONICAL_WIRE_FACTOR
    if proof_requirements is None:
        proof_peaks = {dataset: math.ceil(proof_sizes[dataset] * proof_factor) for dataset in selected_datasets}
        proof_document_doms = proof_peaks
    else:
        if any(proof_requirements[dataset].get("proof_file_stat_size_bytes") != proof_sizes[dataset] for dataset in selected_datasets):
            _fail("typed native proof capacity bounds are not tied to current proof stat sizes")
        proof_peaks = {dataset: int(proof_requirements[dataset]["proof_peak_bytes_source_derived_typed_snark"]) for dataset in selected_datasets}
        proof_document_doms = {dataset: int(proof_requirements[dataset]["proof_document_dom_bytes_upper_typed_snark"]) for dataset in selected_datasets}
    proof_peak = max(proof_peaks.values())
    result_dom = math.ceil(sum(result_sizes.values()) * dom_factor)
    result_text = math.ceil(sum(result_sizes.values()) * PROOF_TEXT_WIRE_FACTOR)
    metadata_dom = math.ceil(sum(small_json_sizes.values()) * dom_factor)
    metadata_text = math.ceil(sum(small_json_sizes.values()) * PROOF_TEXT_WIRE_FACTOR)
    matrix_dom = math.ceil(matrix_bytes * dom_factor)
    matrix_text = math.ceil(matrix_bytes * PROOF_TEXT_WIRE_FACTOR)
    matrix_write_buffers = math.ceil(matrix_bytes * 5) if phase == "collector" else 0
    fixed_floor = INTERPRETER_FIXED_FLOOR_BYTES + sum(sys.getsizeof(value) for value in (sys.modules, sys.path, sys.argv, {}, [], "", 0, 0.0))
    cell_peaks: dict[str, int] = {}
    statement_dom_max = statement_text_max = output_dom_max = output_text_max = validation_work_max = 0
    if statement_requirements is not None:
        for dataset in selected_datasets:
            shape = statement_requirements[dataset]
            statement_wire = shape["statement_wire_bytes_upper"]
            output_wire = shape["public_output_wire_bytes_upper"]
            statement_dom = shape["statement_dom_bytes_upper_typed_int64"]
            output_dom = shape["public_output_dom_bytes_upper_typed_int64"]
            statement_text = shape["statement_text_bytes_upper_unicode_safe"]
            output_text = shape["public_output_text_bytes_upper_unicode_safe"]
            validation_work = shape["statement_validation_work_bytes_upper"]
            statement_dom_max = max(statement_dom_max, statement_dom)
            statement_text_max = max(statement_text_max, statement_text)
            output_dom_max = max(output_dom_max, output_dom)
            output_text_max = max(output_text_max, output_text)
            validation_work_max = max(validation_work_max, validation_work)
            statement_parse_peak = statement_dom + statement_text
            statement_validate_peak = statement_dom + validation_work
            output_parse_peak = statement_dom + output_dom + output_text
            proof_with_statement_peak = proof_peaks[dataset] + statement_dom
            output_with_proof_peak = proof_document_doms[dataset] + statement_dom + output_dom + output_text
            cell_peaks[dataset] = max(statement_parse_peak, statement_validate_peak, output_parse_peak, proof_with_statement_peak, output_with_proof_peak)
    else:
        cell_peaks = {dataset: proof_peaks[dataset] for dataset in selected_datasets}
    current_cell_peak = max(cell_peaks.values())
    collector_peak = fixed_floor + result_dom + result_text + metadata_dom + metadata_text + current_cell_peak + matrix_write_buffers
    validator_peak = fixed_floor + matrix_dom + matrix_text + result_dom + result_text + metadata_dom + metadata_text + current_cell_peak
    accounting: dict[str, int | float | str] = {
        "selected_dataset_count": len(selected_datasets),
        "proof_wire_bytes_max": max(proof_sizes.values()),
        "bounded_metadata_dom_bytes_per_wire_byte": dom_factor,
        "proof_peak_bytes_source_derived_single_current_proof": proof_peak,
        "proof_peak_model": "typed-native-Snark-schema" if proof_requirements is not None else "generic-CPython-wire-upper-bound",
        "proof_document_dom_bytes_max_typed_snark": max(proof_document_doms.values()),
        "result_cells_dom_bytes_source_derived": result_dom,
        "result_json_text_bytes_upper": result_text,
        "bounded_small_json_dom_bytes_source_derived": metadata_dom,
        "bounded_small_json_text_bytes_upper": metadata_text,
        "statement_dom_bytes_max_typed_int64": statement_dom_max,
        "statement_text_bytes_max_unicode_safe": statement_text_max,
        "public_output_dom_bytes_max_typed_int64": output_dom_max,
        "public_output_text_bytes_max_unicode_safe": output_text_max,
        "statement_validation_work_bytes_max": validation_work_max,
        "current_cell_peak_bytes_source_derived": current_cell_peak,
        "matrix_dom_bytes_source_derived": matrix_dom,
        "matrix_text_bytes_upper": matrix_text,
        "matrix_serialization_write_buffers_bytes_upper": matrix_write_buffers,
        "interpreter_fixed_floor_bytes": fixed_floor,
        "collector_peak_bytes_source_derived": collector_peak,
        "validator_peak_bytes_source_derived": validator_peak,
        "host_required_bytes": max(collector_peak, validator_peak),
    }
    if proof_requirements is None:
        accounting["proof_dom_bytes_per_wire_byte"] = dom_factor
        accounting["proof_text_bytes_per_wire_byte"] = PROOF_TEXT_WIRE_FACTOR
        accounting["canonical_instances_text_and_bytes_per_wire_byte"] = PROOF_CANONICAL_WIRE_FACTOR
    else:
        accounting["proof_text_bytes_per_wire_byte"] = 2.0
        accounting["canonical_instances_text_and_bytes_per_wire_byte"] = "exact upper from typed Fr instance count"
    return accounting


def estimate_finalization_capacity(
    *,
    proof_sizes: Mapping[str, int],
    result_sizes: Mapping[str, int],
    matrix_bytes: int,
    session_temp_bytes: int,
    host_available_bytes: int,
    filesystem_available_bytes: Mapping[int, int],
    matrix_device_id: int,
    session_device_id: int,
    statement_requirements: Mapping[str, Mapping[str, int]] | None = None,
    proof_requirements: Mapping[str, Mapping[str, Any]] | None = None,
    small_json_sizes: Mapping[str, int] | None = None,
    phase: str = "collector",
    datasets: tuple[str, ...] = DATASETS,
) -> dict[str, Any]:
    if isinstance(matrix_bytes, bool) or not isinstance(matrix_bytes, int) or matrix_bytes <= 0 or isinstance(session_temp_bytes, bool) or not isinstance(session_temp_bytes, int) or session_temp_bytes <= 0:
        _fail("finalization matrix/session temporary size is invalid")
    if isinstance(host_available_bytes, bool) or not isinstance(host_available_bytes, int) or host_available_bytes < 0:
        _fail("host available capacity is invalid")
    if phase not in {"collector", "validator"}:
        _fail("finalization capacity phase must be collector or validator")
    selected_datasets = normalize_selected_datasets(datasets)
    host = _host_requirements(
        proof_sizes,
        result_sizes,
        matrix_bytes,
        statement_requirements=statement_requirements,
        proof_requirements=proof_requirements,
        small_json_sizes=small_json_sizes,
        phase=phase,
        datasets=selected_datasets,
    )
    disk_required: dict[int, int] = {}
    disk_required[matrix_device_id] = disk_required.get(matrix_device_id, 0) + (matrix_bytes if phase == "collector" else 0)
    disk_required[session_device_id] = disk_required.get(session_device_id, 0) + session_temp_bytes
    if int(host["host_required_bytes"]) > host_available_bytes:
        _fail(f"EZKL final {phase} host gate needs {host['host_required_bytes']} bytes; only {host_available_bytes} are currently available")
    for device_id, required in disk_required.items():
        available = filesystem_available_bytes.get(device_id)
        if isinstance(available, bool) or not isinstance(available, int) or available < required:
            _fail(f"EZKL finalization filesystem device {device_id} needs {required} new bytes; only {available} are available")
    return {
        "phase": phase,
        "selected_datasets": list(selected_datasets),
        "write_matrix": phase == "collector",
        "host": {**host, "available_bytes": host_available_bytes},
        "filesystems": {str(device): {"available_bytes": filesystem_available_bytes[device], "required_new_bytes": required} for device, required in sorted(disk_required.items())},
        "matrix_output_bytes": matrix_bytes if phase == "collector" else 0,
        "matrix_existing_bytes": matrix_bytes if phase == "validator" else 0,
        "session_atomic_temp_bytes": session_temp_bytes,
        "existing_proof_and_result_bytes_already_accounted_by_statvfs": True,
        "existing_matrix_bytes_already_accounted_by_statvfs": phase == "validator",
        "coverage": "source-derived conservative host bound for one active typed statement/output/proof cell plus bounded finalization metadata; partial, not a complete native peak estimate",
    }


def _atomic_json_bytes(document: Mapping[str, Any]) -> int:
    return len((json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def _require_matrix_size_within_upper(actual_bytes: int, predicted_upper_bytes: int) -> None:
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in (actual_bytes, predicted_upper_bytes)):
        _fail("formal matrix size bounds are invalid")
    if actual_bytes > predicted_upper_bytes:
        _fail(f"existing formal matrix exceeds the shared-schema serialization upper bound ({actual_bytes} > {predicted_upper_bytes})")


def _session_temp_size(
    session_document: dict[str, Any],
    receipt: dict[str, Any],
    *,
    receipt_field: str,
    terminal: bool = False,
    terminal_status: str = "FINAL_VALIDATION_OK",
) -> int:
    candidate = SESSION_TEMP_ALLOWANCE_BYTES
    for _ in range(12):
        receipt["session_atomic_temp_bytes"] = candidate
        candidate_document = dict(session_document)
        candidate_document[receipt_field] = receipt
        receipt_size = _atomic_json_bytes(candidate_document)
        if terminal:
            terminal_document = dict(candidate_document)
            terminal_document["status"] = terminal_status
            terminal_document["stage"] = "final_validation_complete"
            terminal_document["updated_at_utc"] = "2099-12-31T23:59:59.999999+00:00"
            terminal_document["completed_at_utc"] = "2099-12-31T23:59:59.999999+00:00"
            actual = max(receipt_size, _atomic_json_bytes(terminal_document))
        else:
            actual = receipt_size
        if actual == candidate:
            return actual
        candidate = actual
    _fail("cannot reach a stable session receipt/terminal temporary-size estimate")


def _result_proof_relative(result: dict[str, Any], dataset: str, proof_path: Path) -> None:
    value = result.get("proof_path")
    if not isinstance(value, str) or not value or Path(value).is_absolute() or ".." in Path(value).parts:
        _fail(f"{dataset} result has an unsafe verifier-facing proof path")
    referenced = _rooted_path(Path(value), f"{dataset} result proof path")
    if referenced != proof_path:
        _fail(f"{dataset} result proof path does not match its formal output directory")


def _manifest_source_entry(
    entry: Any,
    *,
    dataset: str,
    role: str,
    relative_path: str,
    actual_size: int,
) -> str:
    if (
        not isinstance(entry, dict)
        or entry.get("path") != relative_path
        or entry.get("dataset") != dataset
        or entry.get("role") != role
        or isinstance(entry.get("size"), bool)
        or entry.get("size") != actual_size
        or not isinstance(entry.get("sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])
    ):
        _fail(f"{dataset} {role} source does not match its authenticated manifest path/size/hash")
    return entry["sha256"]


def _read_hashed_bounded_bytes(path: Path, label: str, limit: int, expected_sha256: str) -> tuple[bytes, dict[str, Any]]:
    before = _regular_file(path, label)
    if before.st_size <= 0 or before.st_size > limit:
        _fail(f"{label} exceeds its bounded source size")
    try:
        contents = path.read_bytes()
    except OSError as exc:
        _fail(f"cannot read {label}: {exc}")
    digest = hashlib.sha256(contents).hexdigest()
    after = _regular_file(path, label)
    if _stat_record(after) != _stat_record(before) or len(contents) != before.st_size:
        _fail(f"{label} changed while its authenticated bytes were read")
    if digest != expected_sha256:
        _fail(f"{label} content digest does not match the resource manifest")
    return contents, {"stat": _stat_record(before), "sha256": digest}


def _read_meta_cfg(path: Path, dataset: str, expected_sha256: str) -> tuple[dict[str, int], dict[str, Any]]:
    raw, identity = _read_hashed_bounded_bytes(path, f"{dataset} canonical metadata", MAX_META_CFG_BYTES, expected_sha256)
    values: dict[str, str] = {}
    try:
        for raw_line in raw.decode("utf-8").splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            key, separator, value = line.partition("=")
            if not separator or not key.strip() or key.strip() in values:
                _fail(f"{dataset} canonical metadata has a malformed key/value line")
            values[key.strip()] = value.strip()
        if values.get("name") != dataset:
            _fail(f"{dataset} canonical metadata names a different dataset")
        shape = {key: int(values[key]) for key in ("num_nodes", "num_features", "num_classes")}
        shape["graph_count"] = int(values.get("graph_count", "1"))
    except (OSError, UnicodeError, ValueError, KeyError) as exc:
        _fail(f"cannot read {dataset} canonical metadata: {exc}")
    return shape, identity


def _scan_edge_source(path: Path, dataset: str, expected_sha256: str) -> tuple[int, dict[str, Any]]:
    before = _regular_file(path, f"{dataset} authenticated edge source")
    if before.st_size <= 0 or before.st_size > MAX_EDGE_SOURCE_BYTES:
        _fail(f"{dataset} edge source exceeds the bounded canonical source size")
    digest = hashlib.sha256()
    newline_count = 0
    last_byte = b""
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                newline_count += chunk.count(b"\n")
                last_byte = chunk[-1:]
    except OSError as exc:
        _fail(f"cannot stream {dataset} authenticated edge source: {exc}")
    after = _regular_file(path, f"{dataset} authenticated edge source")
    digest_hex = digest.hexdigest()
    if _stat_record(after) != _stat_record(before):
        _fail(f"{dataset} edge source changed while its rows and digest were collected")
    if digest_hex != expected_sha256:
        _fail(f"{dataset} edge source digest does not match the resource manifest")
    raw_lines = newline_count + (1 if last_byte and last_byte != b"\n" else 0)
    return raw_lines, {"stat": _stat_record(before), "sha256": digest_hex}


def _check_shape_source_identities(source_stats: Mapping[str, Any]) -> None:
    manifest_path = ROOT / "resources/resource_manifest.json"
    manifest_identity = source_stats["resource_manifest"]
    manifest_stat = _regular_file(manifest_path, "canonical resource manifest")
    if _stat_record(manifest_stat) != manifest_identity["stat"] or sha256(manifest_path) != manifest_identity["sha256"]:
        _fail("canonical resource manifest changed during finalization capacity admission")
    for dataset, names in source_stats["datasets"].items():
        for field, filename in (("meta_cfg", "meta.cfg"), ("edge_source", "edges.txt")):
            path = ROOT / "resources/canonical/cache" / dataset / filename
            if _stat_record(_regular_file(path, f"{dataset} {filename}")) != names[field]["stat"]:
                _fail(f"{dataset} {filename} identity changed during finalization capacity admission")


def _dataset_shape_bounds(
    expected_resource_manifest_sha256: str,
    datasets: tuple[str, ...] = DATASETS,
) -> tuple[dict[str, dict[str, int]], dict[str, Any]]:
    selected_datasets = normalize_selected_datasets(datasets)
    manifest_path = _rooted_path(ROOT / "resources/resource_manifest.json", "canonical resource manifest")
    manifest_stat_before = _regular_file(manifest_path, "canonical resource manifest")
    manifest = _read_json(manifest_path, "canonical resource manifest", MAX_RESOURCE_MANIFEST_BYTES)
    if not isinstance(manifest, dict) or manifest.get("baseline") != "ezkl" or not isinstance(manifest.get("files"), list):
        _fail("canonical resource manifest has an invalid top-level schema")
    manifest_digest = sha256(manifest_path)
    if _stat_record(_regular_file(manifest_path, "canonical resource manifest")) != _stat_record(manifest_stat_before) or manifest_digest != expected_resource_manifest_sha256:
        _fail("canonical resource manifest changed or no longer matches the active session identity")
    manifest_entries: dict[str, dict[str, Any]] = {}
    for entry in manifest["files"]:
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("path"), str)
            or not entry["path"]
            or Path(entry["path"]).is_absolute()
            or ".." in Path(entry["path"]).parts
            or "\\" in entry["path"]
            or Path(entry["path"]).parts[0] != "resources"
            or isinstance(entry.get("size"), bool)
            or not isinstance(entry.get("size"), int)
            or entry["size"] <= 0
            or not isinstance(entry.get("sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])
            or not isinstance(entry.get("role"), str)
            or not entry["role"]
            or ("dataset" in entry and (not isinstance(entry["dataset"], str) or entry["dataset"] not in {*DATASETS, "all"}))
            or entry["path"] in manifest_entries
        ):
            _fail("canonical resource manifest contains a malformed file record")
        manifest_entries[entry["path"]] = entry
    shapes: dict[str, dict[str, int]] = {}
    source_stats: dict[str, Any] = {"resource_manifest": {"stat": _stat_record(manifest_stat_before), "sha256": manifest_digest}, "datasets": {}}
    for dataset in selected_datasets:
        data_root = _rooted_path(ROOT / "resources/canonical/cache" / dataset, f"{dataset} resource cache")
        meta_path = _rooted_path(data_root / "meta.cfg", f"{dataset} canonical metadata")
        meta_stat = _regular_file(meta_path, f"{dataset} canonical metadata")
        meta_relative = meta_path.relative_to(ROOT).as_posix()
        meta_digest = _manifest_source_entry(
            manifest_entries.get(meta_relative),
            dataset=dataset,
            role="canonical_dataset_cache",
            relative_path=meta_relative,
            actual_size=meta_stat.st_size,
        )
        shape, meta_identity = _read_meta_cfg(meta_path, dataset, meta_digest)
        if shape["num_nodes"] <= 0 or shape["num_features"] <= 0 or shape["num_classes"] <= 0 or shape["graph_count"] <= 0:
            _fail(f"{dataset} canonical metadata contains a nonpositive graph dimension")
        edge_path = _rooted_path(data_root / "edges.txt", f"{dataset} authenticated edge source")
        edge_stat = _regular_file(edge_path, f"{dataset} authenticated edge source")
        edge_relative = edge_path.relative_to(ROOT).as_posix()
        edge_digest = _manifest_source_entry(
            manifest_entries.get(edge_relative),
            dataset=dataset,
            role="canonical_dataset_cache",
            relative_path=edge_relative,
            actual_size=edge_stat.st_size,
        )
        raw_edge_lines, edge_identity = _scan_edge_source(edge_path, dataset, edge_digest)
        shape["raw_edge_lines"] = raw_edge_lines
        shapes[dataset] = typed_statement_requirements(**shape)
        source_stats["datasets"][dataset] = {"meta_cfg": meta_identity, "edge_source": edge_identity}
    _check_shape_source_identities(source_stats)
    return shapes, source_stats


def _small_metadata_paths(output_root: Path, datasets: tuple[str, ...] = DATASETS) -> dict[str, Path]:
    selected_datasets = normalize_selected_datasets(datasets)
    paths: dict[str, Path] = {
        "gpu_build_receipt": ROOT / "artifacts/build/gpu/build_receipt.json",
        "native_backend_manifest": ROOT / "artifacts/build/icicle-cuda/native_backend_manifest.json",
    }
    for dataset in selected_datasets:
        directory = output_root / dataset
        for name in (
            "artifact_manifest.json",
            "core_run_manifest.json",
            "native_proof_metrics.json",
            "native_pk_read_timing.json",
            "native_verify_metrics.json",
            "gpu_prover_record.json",
        ):
            paths[f"{dataset}/{name}"] = directory / name
    return paths


def _result_path_matches(result: dict[str, Any], field: str, expected: Path, dataset: str) -> None:
    value = result.get(field)
    if not isinstance(value, str) or not value:
        _fail(f"{dataset} result has no bounded canonical {field}")
    value_path = Path(value)
    if not value_path.is_absolute() and ".." in value_path.parts:
        _fail(f"{dataset} result contains an unsafe {field}")
    if _rooted_path(value_path, f"{dataset} result {field}") != expected:
        _fail(f"{dataset} result {field} does not match the statted finalization artifact")


def check_finalization_capacity(
    *,
    output_root: Path,
    matrix_path: Path,
    session_file: Path,
    session_id: str,
    source_tree_sha256: str,
    resource_manifest_sha256: str,
    phase: str = "collector",
) -> dict[str, Any]:
    if phase not in {"collector", "validator"}:
        _fail("finalization capacity phase must be collector or validator")
    if not re.fullmatch(r"[0-9a-f]{32}", session_id):
        _fail("finalization capacity requires the active EZKL session id")
    if not re.fullmatch(r"[0-9a-f]{64}", source_tree_sha256) or not re.fullmatch(r"[0-9a-f]{64}", resource_manifest_sha256):
        _fail("finalization capacity source/resource identity is malformed")
    output_root = _rooted_path(Path(output_root), "formal output root")
    matrix_path = _rooted_path(Path(matrix_path), "formal matrix")
    session_file = _rooted_path(Path(session_file), "formal session record")
    expected_session_file = output_root / ".sessions" / f"{session_id}.json"
    if session_file != expected_session_file:
        _fail("finalization capacity session file is outside the active output-root session directory")
    matrix_exists = matrix_path.exists()
    if phase == "collector" and matrix_exists:
        _fail("formal matrix already exists before the collector capacity gate")
    if phase == "validator" and not matrix_exists:
        _fail("formal matrix is missing before the final validator capacity gate")
    session_stat = _regular_file(session_file, "active formal session record")
    if session_stat.st_size > MAX_SESSION_JSON_BYTES:
        _fail("active formal session record exceeds the bounded JSON read limit")
    session_document = _read_json(session_file, "active formal session record", MAX_SESSION_JSON_BYTES)
    if not isinstance(session_document, dict):
        _fail("active formal session record is not an object")
    selected_datasets = selected_datasets_for_session(session_document)
    expected_stage = "collect_selected_results" if phase == "collector" else "final_matrix_validation"
    if not isinstance(session_document, dict) or session_document.get("baseline") != "ezkl" or session_document.get("session_id") != session_id or session_document.get("status") != "RUNNING" or session_document.get("stage") != expected_stage or session_document.get("source_tree_sha256") != source_tree_sha256 or session_document.get("resource_manifest_sha256") != resource_manifest_sha256:
        _fail(f"active session record is not bound to the current EZKL {phase} source/resource stage")
    artifact_paths: dict[str, dict[str, Path]] = {}
    artifact_stats: dict[str, dict[str, dict[str, int]]] = {}
    proof_sizes: dict[str, int] = {}
    result_sizes: dict[str, int] = {}
    statement_sizes: dict[str, int] = {}
    output_sizes: dict[str, int] = {}
    small_paths = _small_metadata_paths(output_root, selected_datasets)
    small_stats: dict[str, dict[str, int]] = {}
    small_sizes: dict[str, int] = {}
    for name, path in small_paths.items():
        rooted = _rooted_path(path, f"small JSON artifact {name}")
        info = _regular_file(rooted, f"small JSON artifact {name}")
        if info.st_size <= 0 or info.st_size > MAX_SMALL_JSON_BYTES:
            _fail(f"small JSON artifact {name} exceeds the bounded 1 MiB read limit")
        small_paths[name] = rooted
        small_stats[name] = _stat_record(info)
        small_sizes[name] = int(info.st_size)
    small_sizes["active_session_record"] = int(session_stat.st_size)
    for dataset in selected_datasets:
        directory = _rooted_path(output_root / dataset, f"{dataset} formal output directory")
        paths = {name: _rooted_path(directory / name, f"{dataset} {name}") for name in ("result.json", "proof.json", "statement.json", "public_output.json")}
        stats = {name.removesuffix(".json"): _regular_file(path, f"{dataset} {name}") for name, path in paths.items()}
        if any(info.st_size <= 0 for info in stats.values()):
            _fail(f"{dataset} formal result/proof/statement/output artifact is empty")
        if stats["result"].st_size > MAX_RESULT_JSON_BYTES:
            _fail(f"{dataset} result JSON exceeds the bounded metadata read limit")
        artifact_paths[dataset] = paths
        artifact_stats[dataset] = {name: _stat_record(info) for name, info in stats.items()}
        result_sizes[dataset] = int(stats["result"].st_size)
        proof_sizes[dataset] = int(stats["proof"].st_size)
        statement_sizes[dataset] = int(stats["statement"].st_size)
        output_sizes[dataset] = int(stats["public_output"].st_size)
    shapes, shape_source_stats = _dataset_shape_bounds(resource_manifest_sha256, selected_datasets)
    proof_bounds = {
        dataset: typed_proof_requirements(
            proof_size_bytes=proof_sizes[dataset],
            statement_shape=shapes[dataset],
            compact_v2=has_compact_native_proof_v2_prefix(artifact_paths[dataset]["proof.json"]),
        )
        for dataset in selected_datasets
    }
    for dataset in selected_datasets:
        if statement_sizes[dataset] > shapes[dataset]["statement_wire_bytes_upper"]:
            _fail(f"{dataset} statement JSON exceeds the producer-derived typed-list upper bound before parsing")
        if output_sizes[dataset] > shapes[dataset]["public_output_wire_bytes_upper"]:
            _fail(f"{dataset} public output JSON exceeds the producer-derived int64-Y upper bound before parsing")
        shapes[dataset]["statement_file_stat_size_bytes"] = statement_sizes[dataset]
        shapes[dataset]["public_output_file_stat_size_bytes"] = output_sizes[dataset]
    matrix_path = _rooted_path(matrix_path, "formal matrix")
    matrix_directory = matrix_path.parent
    while not matrix_directory.exists():
        if matrix_directory == matrix_directory.parent:
            _fail(f"no existing matrix output parent for {matrix_path}")
        matrix_directory = matrix_directory.parent
    early_matrix_bytes = max(1, 8 * sum(result_sizes.values()) + 1024 * 1024)
    matrix_info = _regular_file(matrix_path, "existing formal matrix") if phase == "validator" else None
    matrix_initial_stat = _stat_record(matrix_info) if matrix_info is not None else None
    if matrix_info is not None and (matrix_info.st_size <= 0 or matrix_info.st_size > MAX_MATRIX_JSON_BYTES):
        _fail("existing formal matrix exceeds the bounded 32 MiB JSON read limit")
    host_available, early_host_accounting = _current_host_available()
    session_available, session_device = _filesystem(session_file.parent)
    matrix_available, matrix_device = _filesystem(matrix_directory)
    if matrix_info is not None and int(matrix_info.st_dev) != matrix_device:
        _fail("existing formal matrix filesystem identity changed")
    early_capacity = estimate_finalization_capacity(
        proof_sizes=proof_sizes,
        result_sizes=result_sizes,
        matrix_bytes=matrix_info.st_size if matrix_info is not None else early_matrix_bytes,
        session_temp_bytes=session_stat.st_size + SESSION_TEMP_ALLOWANCE_BYTES,
        host_available_bytes=host_available,
        filesystem_available_bytes={session_device: session_available, matrix_device: matrix_available},
        matrix_device_id=matrix_device,
        session_device_id=session_device,
        statement_requirements=shapes,
        proof_requirements=proof_bounds,
        small_json_sizes=small_sizes,
        phase=phase,
        datasets=selected_datasets,
    )
    cells: list[dict[str, Any]] = []
    for dataset in selected_datasets:
        result_path = artifact_paths[dataset]["result.json"]
        proof_path = artifact_paths[dataset]["proof.json"]
        cell = _read_json(result_path, f"{dataset} result", MAX_RESULT_JSON_BYTES)
        if not isinstance(cell, dict) or cell.get("dataset") != dataset or cell.get("formal_session_id") != session_id or cell.get("status") != "VERIFY_OK" or cell.get("source_tree_sha256") != source_tree_sha256 or cell.get("resource_manifest_sha256") != resource_manifest_sha256:
            _fail(f"{dataset} result metadata does not belong to the active verified formal session")
        if cell.get("proof_size_bytes") != proof_sizes[dataset]:
            _fail(f"{dataset} result proof size does not match the statted proof artifact")
        _result_proof_relative(cell, dataset, proof_path)
        for field, name in (("public_statement_path", "statement.json"), ("public_output_path", "public_output.json"), ("artifact_manifest_path", "artifact_manifest.json")):
            _result_path_matches(cell, field, artifact_paths[dataset][name] if name in artifact_paths[dataset] else small_paths[f"{dataset}/{name}"], dataset)
        commitment_metric = cell.get("commitment_metric")
        verify_metric = cell.get("verify_metric")
        if not isinstance(commitment_metric, dict) or not isinstance(verify_metric, dict):
            _fail(f"{dataset} result does not identify bounded proof/verify metrics")
        _result_path_matches({"path": commitment_metric.get("path")}, "path", small_paths[f"{dataset}/native_proof_metrics.json"], dataset)
        _result_path_matches({"path": verify_metric.get("path")}, "path", small_paths[f"{dataset}/native_verify_metrics.json"], dataset)
        cell["_result_path"] = result_path.relative_to(ROOT).as_posix()
        cells.append(cell)
    matrix_document = build_matrix_document(cells, datasets=selected_datasets, source_sha256=source_tree_sha256, resource_sha256=resource_manifest_sha256, collected_at_utc="2099-12-31T23:59:59.999999+00:00", formal_session_id=session_id, backends=None)
    predicted_matrix_bytes = _atomic_json_bytes(matrix_document)
    if predicted_matrix_bytes > MAX_MATRIX_JSON_BYTES:
        _fail("predicted final matrix exceeds the bounded 32 MiB final JSON limit")
    if matrix_info is not None:
        _require_matrix_size_within_upper(matrix_info.st_size, predicted_matrix_bytes)
    # The timestamp maximum makes predicted_matrix_bytes an upper bound. The
    # validator's existing file is admitted and charged by its actual stat size.
    matrix_bytes = predicted_matrix_bytes if phase == "collector" else int(matrix_info.st_size)
    if matrix_info is not None:
        # The matrix is bounded by stat and the exact shared-builder size before
        # this parse; the independent final validator still performs all checks.
        parsed_matrix = _read_json(matrix_path, "existing formal matrix", MAX_MATRIX_JSON_BYTES)
        if not isinstance(parsed_matrix, dict) or parsed_matrix.get("baseline") != "ezkl" or parsed_matrix.get("formal_session_id") != session_id or parsed_matrix.get("source_tree_sha256") != source_tree_sha256 or parsed_matrix.get("resource_manifest_sha256") != resource_manifest_sha256 or parsed_matrix.get("datasets") != list(selected_datasets) or parsed_matrix.get("cell_count") != len(selected_datasets):
            _fail("existing formal matrix is not bound to the current final validation session/source/resources")
    host_available, host_accounting = _current_host_available()
    session_available, session_device = _filesystem(session_file.parent)
    matrix_available, matrix_device = _filesystem(matrix_directory)
    if matrix_info is not None and int(matrix_info.st_dev) != matrix_device:
        _fail("existing formal matrix filesystem identity changed during validation capacity check")
    final_small_stats = {name: _stat_record(_regular_file(path, f"small JSON artifact {name}")) for name, path in small_paths.items()}
    if final_small_stats != small_stats:
        _fail("a small final-validation JSON artifact changed during capacity admission")
    small_stats = final_small_stats
    artifact_stats_final: dict[str, dict[str, dict[str, int]]] = {}
    for dataset, paths in artifact_paths.items():
        artifact_stats_final[dataset] = {name.removesuffix(".json"): _stat_record(_regular_file(path, f"{dataset} {name}")) for name, path in paths.items()}
        if artifact_stats_final[dataset] != artifact_stats[dataset]:
            _fail(f"{dataset} result/proof/statement/output artifacts changed during capacity admission")
    final_matrix_info = _regular_file(matrix_path, "existing formal matrix") if phase == "validator" else None
    matrix_stat_record = _stat_record(final_matrix_info) if final_matrix_info is not None else None
    if phase == "validator":
        _require_matrix_size_within_upper(final_matrix_info.st_size, predicted_matrix_bytes)
        if matrix_stat_record != matrix_initial_stat or final_matrix_info.st_dev != matrix_device:
            _fail("existing formal matrix changed during final validation capacity admission")
    if _stat_record(_regular_file(session_file, "active formal session record")) != _stat_record(session_stat):
        _fail("active formal session record changed during finalization capacity admission")
    _check_shape_source_identities(shape_source_stats)
    receipt_field = "finalization_capacity" if phase == "collector" else "final_validation_capacity"
    terminal_session = phase == "validator"
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "baseline": "ezkl",
        "phase": phase,
        "write_matrix": phase == "collector",
        "session_id": session_id,
        "selected_datasets": list(selected_datasets),
        "source_tree_sha256": source_tree_sha256,
        "resource_manifest_sha256": resource_manifest_sha256,
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_root": output_root.relative_to(ROOT).as_posix(),
        "matrix_path": matrix_path.relative_to(ROOT).as_posix(),
        "session_file": session_file.relative_to(ROOT).as_posix(),
        "session_file_stat": _stat_record(session_stat),
        "file_system_identity": {"session_device_id": session_device, "matrix_device_id": matrix_device},
        "artifacts": artifact_stats_final,
        "small_json_artifacts": small_stats,
        "dataset_shapes_source_derived": shapes,
        "proof_shapes_source_derived": proof_bounds,
        "shape_source_file_stats": shape_source_stats,
        "matrix_output_bytes": predicted_matrix_bytes if phase == "collector" else 0,
        "matrix_output_bytes_upper": predicted_matrix_bytes,
        "matrix_existing_bytes": int(final_matrix_info.st_size) if final_matrix_info is not None else 0,
        "matrix_file_stat": matrix_stat_record,
        "host_accounting": host_accounting,
        "early_capacity_accounting": early_capacity,
        "coverage": "SOURCE_DERIVED conservative CPython host/output bound from one active selected statement/output/proof cell plus selected result/matrix metadata; partial and not a complete native peak estimate",
    }
    receipt["host"] = _host_requirements(
        proof_sizes,
        result_sizes,
        matrix_bytes,
        statement_requirements=shapes,
        proof_requirements=proof_bounds,
        small_json_sizes=small_sizes,
        phase=phase,
        datasets=selected_datasets,
    )
    receipt["host"]["available_bytes"] = host_available
    available_by_device = {session_device: session_available, matrix_device: min(session_available if session_device == matrix_device else matrix_available, matrix_available)}
    temp_size = SESSION_TEMP_ALLOWANCE_BYTES
    filesystem_plan: dict[str, Any] = {}
    for _ in range(12):
        receipt["session_atomic_temp_bytes"] = temp_size
        required_by_device: dict[int, int] = {matrix_device: matrix_bytes if phase == "collector" else 0}
        required_by_device[session_device] = required_by_device.get(session_device, 0) + temp_size
        filesystem_plan = {str(device): {"available_bytes": available_by_device[device], "required_new_bytes": required} for device, required in sorted(required_by_device.items())}
        receipt["filesystems"] = filesystem_plan
        actual_size = _session_temp_size(
            dict(session_document),
            receipt,
            receipt_field=receipt_field,
            terminal=terminal_session,
            terminal_status="FINAL_VALIDATION_OK" if selected_datasets == DATASETS else "SUBSET_COMPLETE",
        )
        if actual_size == temp_size:
            break
        temp_size = actual_size
    else:
        _fail("cannot stabilize EZKL session receipt/terminal atomic-write capacity estimate")
    receipt["session_atomic_temp_bytes"] = temp_size
    receipt["filesystems"] = filesystem_plan
    estimate_finalization_capacity(
        proof_sizes=proof_sizes,
        result_sizes=result_sizes,
        matrix_bytes=matrix_bytes,
        session_temp_bytes=temp_size,
        host_available_bytes=host_available,
        filesystem_available_bytes=available_by_device,
        matrix_device_id=matrix_device,
        session_device_id=session_device,
        statement_requirements=shapes,
        proof_requirements=proof_bounds,
        small_json_sizes=small_sizes,
        phase=phase,
        datasets=selected_datasets,
    )
    return receipt
