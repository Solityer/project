"""Fresh-process verifier for the test-only synthetic zkGPT bundle.

This module is deliberately kept under ``tests/``.  It is an adapter around
the formal verifier, not a production runner and not an alternate proof
implementation.  The adapter has a small, explicit trust root consisting of
the serialized proof messages and the public synthetic graph/config/model
descriptors.  In particular it never loads a checkpoint bundle, model tensor,
raw witness, or forward trace.

The production verifier context currently also supports the formal dataset
runner.  That context intentionally loads a checkpoint manifest and tensor
file, so it is not suitable for a test bundle whose public model descriptor
contains architecture only.  This adapter reconstructs the same layout and
native family descriptors from the public synthetic files and then calls the
formal verifier directly with that external statement.
"""
from __future__ import annotations

import argparse
import copy
import json
import pathlib
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zkgpt_gat.constants import (  # noqa: E402
    CIRCUIT_LAYOUT_ID,
    COMMITMENT_BACKEND_ID,
    ENCODING_ID,
    LASSO_CFG_ID,
    MATRIX_PROOF_CFG_ID,
    PADDING_RULE_ID,
    PROOF_BACKEND_ID,
    PROTOCOL_ID,
)
from zkgpt_gat.data import LocalGraph  # noqa: E402
from zkgpt_gat.field import next_power_of_two  # noqa: E402
from zkgpt_gat.model import QUANT_SCALE_PROFILES  # noqa: E402
from zkgpt_gat.privacy import PROFILE_ID, PROFILE_STATUS, transcript_domain  # noqa: E402
from zkgpt_gat.registry import canonical_registry_public_path  # noqa: E402
from zkgpt_gat.rounding import validate_rounding_mode  # noqa: E402
from zkgpt_gat.util import digest_json  # noqa: E402
from zkgpt_gat.verifier_context import (  # noqa: E402
    _aux_arrays,
    _dimensions,
    _families,
    _ledger,
    _matrix_families,
    _model_arrays,
)


PROOF_FILES = (
    "metadata.json",
    "commitments.json",
    "transcript.json",
    "lookup_proof.json",
    "fixed_lookup_tables.json",
    "matrix_sumcheck_proof.json",
    "gkr_proof.json",
    "combine_proof.json",
    "hyrax_mle_openings.json",
    "privacy_profile.json",
    "proof_payload.json",
)
PUBLIC_ROOT_FILES = {
    "public_config.json",
    "public_graph.json",
    "public_model.json",
    "public_y.json",
}
PUBLIC_STATEMENT_SCHEMA = "zkGPT_external_public_statement_v1"

# These are exact raw-material names.  Native zkGPT proof messages may contain
# names such as ``committed_witness_opening_digest`` as part of a cryptographic
# binding; those are not witness values and remain allowed.  The adapter must
# nevertheless reject any field that would carry raw witness/trace material.
FORBIDDEN_RAW_FIELDS = {
    "checkpoint",
    "checkpoint_bundle",
    "checkpoint_tensors",
    "tensor",
    "tensors",
    "tensor_values",
    "raw_tensor",
    "raw_tensors",
    "witness",
    "raw_witness",
    "private_witness",
    "witness_values",
    "witness_arrays",
    "trace",
    "raw_trace",
    "private_trace",
    "trace_arrays",
}
FORBIDDEN_PATH_TOKENS = {
    "checkpoint",
    "checkpoints",
    "checkpoint_bundle",
    "checkpoint_tensors",
    "tensor",
    "tensors",
    "witness",
    "private",
    "trace",
}


class SyntheticVerificationError(ValueError):
    """Fail-closed error for malformed or out-of-scope synthetic input."""


def _normal_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def _contains_forbidden_field(value: Any, path: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = _normal_key(key)
            child_path = f"{path}.{key}"
            if normalized in FORBIDDEN_RAW_FIELDS:
                found.append(child_path)
            found.extend(_contains_forbidden_field(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_contains_forbidden_field(child, f"{path}[{index}]"))
    return found


@dataclass
class PublicInputAccessAudit:
    """Machine-readable audit of the adapter's file and field boundary."""

    root: pathlib.Path
    files_read: list[str] = field(default_factory=list)
    forbidden_paths_read: list[str] = field(default_factory=list)
    forbidden_fields_seen: list[str] = field(default_factory=list)

    def _relative(self, path: pathlib.Path) -> str:
        try:
            return str(path.resolve().relative_to(self.root.resolve()))
        except ValueError:
            return str(path.resolve())

    def record_path(self, path: pathlib.Path, *, allowed: bool) -> None:
        relative = self._relative(path)
        tokens = {
            _normal_key(part)
            for part in pathlib.PurePath(relative).parts
        }
        if not allowed or tokens & FORBIDDEN_PATH_TOKENS:
            self.forbidden_paths_read.append(relative)
            raise SyntheticVerificationError(
                f"forbidden private/checkpoint path requested: {relative}"
            )
        if relative not in self.files_read:
            self.files_read.append(relative)

    def scan_public_fields(self, payload: Any, label: str) -> None:
        found = _contains_forbidden_field(payload, label)
        if found:
            self.forbidden_fields_seen.extend(found)
            raise SyntheticVerificationError(
                f"public input contains forbidden private/checkpoint fields: {found}"
            )

    def finish(self, *, status: str, errors: Iterable[str] = ()) -> Dict[str, Any]:
        return {
            "schema": "zkGPT_synthetic_public_input_access_audit_v1",
            "status": status,
            "root": str(self.root),
            "allowed_root_files": sorted(PUBLIC_ROOT_FILES | {"static_descriptor.json"}),
            "allowed_proof_files": list(PROOF_FILES),
            "files_read": sorted(self.files_read),
            "forbidden_paths_read": sorted(set(self.forbidden_paths_read)),
            "forbidden_fields_seen": sorted(set(self.forbidden_fields_seen)),
            "verifier_reads_witness": False,
            "witness_paths_read": [],
            "witness_fields_read": [],
            "errors": list(errors),
        }


def _read_json(
    audit: PublicInputAccessAudit,
    path: pathlib.Path,
    *,
    label: str,
    allowed_relative: set[str],
    scan_fields: bool,
) -> Dict[str, Any]:
    path = path.expanduser().resolve()
    relative = audit._relative(path)
    audit.record_path(path, allowed=relative in allowed_relative)
    if not path.is_file():
        raise SyntheticVerificationError(f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SyntheticVerificationError(f"cannot read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SyntheticVerificationError(f"{label} must be a JSON object: {path}")
    if scan_fields:
        audit.scan_public_fields(value, label)
    return value


def _read_proof_json(
    audit: PublicInputAccessAudit,
    artifact: pathlib.Path,
    name: str,
) -> Dict[str, Any]:
    return _read_json(
        audit,
        artifact / name,
        label=f"serialized proof {name}",
        allowed_relative={f"proof/{item}" for item in PROOF_FILES},
        # Raw witness names are rejected, while native cryptographic
        # committed-witness-opening descriptors are intentionally allowed.
        scan_fields=True,
    )


def _int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise SyntheticVerificationError(f"{label} must be an integer")
    try:
        converted = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SyntheticVerificationError(f"{label} must be an integer") from exc
    if isinstance(value, float) and value != converted:
        raise SyntheticVerificationError(f"{label} must be integral")
    return converted


def _int_list(value: Any, label: str, *, nonempty: bool = False) -> list[int]:
    if not isinstance(value, list) or (nonempty and not value):
        raise SyntheticVerificationError(f"{label} must be a {'non-empty ' if nonempty else ''}integer list")
    return [_int(item, f"{label}[{index}]") for index, item in enumerate(value)]


def _array_digest(label: str, value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    import hashlib

    digest = hashlib.sha256()
    digest.update(label.encode("utf-8"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(str(array.dtype).encode("ascii"))
    if array.dtype == object:
        for item in array.ravel().tolist():
            digest.update(str(item).encode("ascii"))
            digest.update(b",")
    else:
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _graph_digest(graph: LocalGraph) -> str:
    return digest_json(
        {
            "absolute_ids": _array_digest("absolute_ids", graph.absolute_ids),
            "src": _array_digest("src", graph.src),
            "dst": _array_digest("dst", graph.dst),
            "node_ptr": graph.node_ptr.tolist(),
            "edge_ptr": graph.edge_ptr.tolist(),
        }
    )


def _public_x_digest(graph: LocalGraph) -> str:
    features = np.asarray(graph.features, dtype=np.int64)
    return digest_json(
        {
            "features": _array_digest("X_pub.H_input", features),
            "topology": _graph_digest(graph),
            "selectors": {
                "segment_order": ["X_pub", "W", "Aux", "L_qry"],
                "graph_count": graph.graph_count,
            },
        }
    )


def _local_graph(payload: Mapping[str, Any]) -> LocalGraph:
    expected = {
        "schema", "name", "features", "labels", "absolute_ids", "src", "dst",
        "edge_graph", "node_ptr", "edge_ptr", "graph_count", "n_total",
        "num_nodes", "num_features", "num_classes", "task_type", "report_unit",
        "batching_rule", "self_loop_rule", "edge_sort_rule", "chunking_rule",
    }
    if set(payload) != expected or payload.get("schema") != "zkGPT_test_public_graph_v1":
        raise SyntheticVerificationError("synthetic public graph schema mismatch")
    features_data = payload.get("features", {})
    labels_data = payload.get("labels", {})
    if not isinstance(features_data, Mapping) or not isinstance(labels_data, Mapping):
        raise SyntheticVerificationError("synthetic graph features/labels must be objects")
    feature_shape = tuple(_int_list(features_data.get("shape"), "features.shape"))
    label_shape = tuple(_int_list(labels_data.get("shape"), "labels.shape"))
    feature_values = _int_list(features_data.get("values"), "features.values")
    label_values = _int_list(labels_data.get("values"), "labels.values")
    if len(feature_shape) != 2 or len(label_shape) != 1:
        raise SyntheticVerificationError("synthetic graph feature/label rank mismatch")
    if int(np.prod(feature_shape, dtype=np.int64)) != len(feature_values):
        raise SyntheticVerificationError("synthetic graph feature shape mismatch")
    n = _int(payload.get("num_nodes"), "num_nodes")
    e = len(payload.get("src", []))
    g = _int(payload.get("graph_count"), "graph_count")
    if feature_shape != (n, _int(payload.get("num_features"), "num_features")):
        raise SyntheticVerificationError("synthetic graph feature dimensions mismatch")
    if label_shape != (n,) or len(label_values) != n:
        raise SyntheticVerificationError("synthetic graph label dimensions mismatch")
    src = np.asarray(_int_list(payload.get("src"), "src"), dtype=np.int64)
    dst = np.asarray(_int_list(payload.get("dst"), "dst"), dtype=np.int64)
    edge_graph = np.asarray(_int_list(payload.get("edge_graph"), "edge_graph"), dtype=np.int64)
    absolute_ids = np.asarray(_int_list(payload.get("absolute_ids"), "absolute_ids"), dtype=np.int64)
    node_ptr = np.asarray(_int_list(payload.get("node_ptr"), "node_ptr"), dtype=np.int64)
    edge_ptr = np.asarray(_int_list(payload.get("edge_ptr"), "edge_ptr"), dtype=np.int64)
    if e <= 0 or len(dst) != e or len(edge_graph) != e or len(absolute_ids) != n:
        raise SyntheticVerificationError("synthetic graph edge/node cardinality mismatch")
    if len(node_ptr) != g + 1 or len(edge_ptr) != g + 1:
        raise SyntheticVerificationError("synthetic graph pointer cardinality mismatch")
    if list(node_ptr) != sorted(set(int(v) for v in node_ptr)) or int(node_ptr[0]) != 0 or int(node_ptr[-1]) != n:
        raise SyntheticVerificationError("synthetic node_ptr is not strictly increasing")
    if list(edge_ptr) != sorted(set(int(v) for v in edge_ptr)) or int(edge_ptr[0]) != 0 or int(edge_ptr[-1]) != e:
        raise SyntheticVerificationError("synthetic edge_ptr is not strictly increasing")
    if len(set(int(v) for v in absolute_ids)) != n or any(v < 0 or v >= _int(payload.get("n_total"), "n_total") for v in absolute_ids):
        raise SyntheticVerificationError("synthetic absolute node ids are invalid")
    if any(v < 0 or v >= n for v in src) or any(v < 0 or v >= n for v in dst):
        raise SyntheticVerificationError("synthetic edge endpoint leaves node domain")
    if any(v < 0 or v >= g for v in edge_graph):
        raise SyntheticVerificationError("synthetic edge graph id leaves batch domain")
    for graph_id in range(g):
        n_begin, n_end = int(node_ptr[graph_id]), int(node_ptr[graph_id + 1])
        e_begin, e_end = int(edge_ptr[graph_id]), int(edge_ptr[graph_id + 1])
        if e_begin >= e_end or n_begin >= n_end:
            raise SyntheticVerificationError("synthetic graph batch contains an empty graph")
        if any(int(edge_graph[index]) != graph_id for index in range(e_begin, e_end)):
            raise SyntheticVerificationError("synthetic edge_ptr/edge_graph mismatch")
        if any(int(src[index]) < n_begin or int(src[index]) >= n_end for index in range(e_begin, e_end)):
            raise SyntheticVerificationError("synthetic source crosses graph boundary")
        if any(int(dst[index]) < n_begin or int(dst[index]) >= n_end for index in range(e_begin, e_end)):
            raise SyntheticVerificationError("synthetic destination crosses graph boundary")
    if any(not any(int(dst[index]) == node for index in range(e)) for node in range(n)):
        raise SyntheticVerificationError("synthetic graph misses a destination group/self-loop")
    return LocalGraph(
        name=str(payload["name"]),
        num_nodes=n,
        num_features=_int(payload["num_features"], "num_features"),
        num_classes=_int(payload["num_classes"], "num_classes"),
        features=np.asarray(feature_values, dtype=np.int64).reshape(feature_shape),
        labels=np.asarray(label_values, dtype=np.int64).reshape(label_shape),
        absolute_ids=absolute_ids,
        src=src,
        dst=dst,
        edge_graph=edge_graph,
        node_ptr=node_ptr,
        edge_ptr=edge_ptr,
        graph_count=g,
        n_total=_int(payload["n_total"], "n_total"),
        task_type=str(payload["task_type"]),
        report_unit=str(payload["report_unit"]),
        batching_rule=str(payload["batching_rule"]),
        self_loop_rule=str(payload["self_loop_rule"]),
        edge_sort_rule=str(payload["edge_sort_rule"]),
        chunking_rule=str(payload["chunking_rule"]),
    )


def _manifest_and_model(payload: Mapping[str, Any]) -> Dict[str, Any]:
    expected = {
        "manifest", "hidden_layers", "output_heads", "quant_scales",
        "model_arch_id", "model_param_id", "quant_cfg_id", "static_table_id",
        "class_count", "k_out",
    }
    if set(payload) != expected:
        raise SyntheticVerificationError("synthetic public model schema mismatch")
    manifest = payload.get("manifest")
    if not isinstance(manifest, Mapping):
        raise SyntheticVerificationError("synthetic public model manifest is missing")
    if _contains_forbidden_field(manifest, "public_model.manifest"):
        raise SyntheticVerificationError("synthetic public model contains private tensor/checkpoint fields")
    try:
        hidden, classes, k_out, layers = _dimensions(manifest)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise SyntheticVerificationError(f"synthetic public model architecture is invalid: {exc}") from exc
    if classes != _int(payload["class_count"], "class_count") or k_out != _int(payload["k_out"], "k_out"):
        raise SyntheticVerificationError("synthetic public model cardinality mismatch")
    for key in ("model_arch_id", "model_param_id", "quant_cfg_id", "static_table_id"):
        if str(payload[key]) != str(manifest.get(key, payload[key])) and key == "model_arch_id":
            raise SyntheticVerificationError(f"synthetic public model {key} mismatch")
    quant_id = str(payload["quant_cfg_id"])
    if quant_id not in QUANT_SCALE_PROFILES or payload["quant_scales"] != QUANT_SCALE_PROFILES[quant_id]:
        raise SyntheticVerificationError("synthetic public model quantization profile mismatch")
    if [
        {
            "layer_index": _int(layer["layer_index"], "layer_index"),
            "input_dim": _int(layer["input_dim"], "input_dim"),
            "head_count": _int(layer["head_count"], "head_count"),
            "head_dim": _int(layer["head_dim"], "head_dim"),
        }
        for layer in payload["hidden_layers"]
    ] != hidden:
        raise SyntheticVerificationError("synthetic public model hidden architecture mismatch")
    if len(payload["output_heads"]) != k_out:
        raise SyntheticVerificationError("synthetic public model output head count mismatch")
    return dict(manifest)


def _validate_static_descriptor(descriptor: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    required = {"schema", "static_tables_id", "quant_cfg_id", "families", "content", "content_digest", "descriptor_digest"}
    if set(descriptor) != required or descriptor.get("schema") != "zkGPT_static_tables_descriptor_v1":
        raise SyntheticVerificationError("synthetic static descriptor schema mismatch")
    if descriptor.get("static_tables_id") != config.get("static_table_id") or descriptor.get("quant_cfg_id") != config.get("quant_cfg_id"):
        raise SyntheticVerificationError("synthetic static descriptor identity mismatch")
    if descriptor.get("families") != ["T_LReLU", "T_ELU", "T_exp", "T_range"]:
        raise SyntheticVerificationError("synthetic static descriptor family order mismatch")
    content = descriptor.get("content")
    if not isinstance(content, Mapping) or content.get("range_table", {}).get("range_bits") != _int(config.get("range_bits"), "range_bits"):
        raise SyntheticVerificationError("synthetic static descriptor range policy mismatch")
    if descriptor.get("content_digest") != digest_json(content):
        raise SyntheticVerificationError("synthetic static descriptor content digest mismatch")
    without_digest = dict(descriptor)
    without_digest.pop("descriptor_digest", None)
    if descriptor.get("descriptor_digest") != digest_json(without_digest):
        raise SyntheticVerificationError("synthetic static descriptor digest mismatch")


def _logical_segment(name: str, arrays: Sequence[tuple[str, Sequence[int]]]) -> Dict[str, Any]:
    offset = 0
    descriptors: list[Dict[str, Any]] = []
    for array_name, shape in arrays:
        shape = [int(item) for item in shape]
        length = int(np.prod(shape, dtype=np.int64))
        descriptors.append({
            "name": array_name,
            "offset": offset,
            "length": length,
            "shape": shape,
            "encoding": "canonical_bn254_fr_integer",
        })
        offset += length
    return {"name": name, "logical_length": offset, "arrays": descriptors}


def _expected_layout(
    graph: LocalGraph,
    manifest: Mapping[str, Any],
    range_bits: int,
    ledger: Mapping[str, Any],
    fixed_tables: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[Dict[str, Any], list[Dict[str, Any]], list[Dict[str, Any]], list[Dict[str, Any]]]:
    """Rebuild layout plus public native family descriptors without witness data."""
    lasso = _families(graph, manifest, range_bits, ledger)
    if fixed_tables is not None:
        # Function-family padding is authenticated by the fixed table's
        # canonical domain/evaluator.  Rebuild those two descriptors from the
        # public table registry so the layout digest is independent of any
        # private trace values.
        import zkgpt_gat.proof as proof_module

        for family in lasso:
            if family.get("kind") != "function":
                continue
            table_id = str(family.get("table_id", ""))
            table = fixed_tables.get(table_id)
            if not isinstance(table, Mapping):
                raise SyntheticVerificationError(
                    f"missing fixed table for Lasso family: {table_id}"
                )
            certified_domain = table.get("certified_domain")
            tuple_encoding = table.get("tuple_encoding")
            if (
                not isinstance(certified_domain, list)
                or len(certified_domain) != 2
                or not isinstance(tuple_encoding, Mapping)
            ):
                raise SyntheticVerificationError(
                    f"fixed table descriptor is incomplete: {table_id}"
                )
            domain_min = _int(certified_domain[0], f"{table_id}.certified_domain[0]")
            family["a_pad"] = str(
                proof_module._pair_key(
                    dict(table),
                    domain_min,
                    proof_module._fixed_table_eval_q(table_id, domain_min),
                )
            )
            family["tuple_encoding"] = copy.deepcopy(dict(tuple_encoding))
    model_arrays = _model_arrays(manifest)
    aux_arrays = _aux_arrays(graph, manifest, range_bits, ledger)
    segments = [
        _logical_segment("X_pub", [
            ("X_pub.H_input", [graph.num_nodes, graph.num_features]),
            ("X_pub.absolute_ids", [graph.num_nodes]),
            ("X_pub.dimension_scalars", [7]),
        ]),
        _logical_segment("W", [(str(item["name"]), item["shape"]) for item in model_arrays]),
        _logical_segment("Aux", aux_arrays),
        _logical_segment("L_qry", [(str(item["family"]), item["shape"]) for item in lasso]),
    ]
    n_raw = max(int(item["logical_length"]) for item in segments)
    n_in = next_power_of_two(max(1, n_raw))
    selector_encoding = {"00": "X_pub", "01": "W", "10": "Aux", "11": "L_qry"}
    selectors = {segment: selector for selector, segment in selector_encoding.items()}
    for segment in segments:
        segment["padded_length"] = n_in
        segment["padding_begin"] = int(segment["logical_length"])
        segment["padding_value"] = "0"
        segment["segment_selector"] = selectors[segment["name"]]
        if segment["name"] == "L_qry":
            segment["lasso_families"] = [copy.deepcopy(item) for item in lasso]
    layout: Dict[str, Any] = {
        "schema": "zkGPT_gat_four_segment_layout_v1",
        "segment_order": ["X_pub", "W", "Aux", "L_qry"],
        "selector_encoding": selector_encoding,
        "N_raw": n_raw,
        "N_in": n_in,
        "input_layer_length": 4 * n_in,
        "padding_rule": "logical prefix followed by canonical zero padding; family-internal Lasso padding is part of logical L_qry",
        "segments": segments,
    }
    # ``shape`` is an external layout-array property.  Native Lasso family
    # registry entries intentionally carry only protocol family fields; keep
    # that exact wire shape in the authenticated L_qry descriptor.
    layout["segments"][-1]["lasso_families"] = [
        {
            key: value
            for key, value in item.items()
            if key != "shape"
        }
        for item in lasso
    ]
    layout["layout_digest"] = digest_json(layout)
    return layout, lasso, _matrix_families(graph, manifest), model_arrays


def _public_statement(graph: LocalGraph, public_y: np.ndarray, metadata: Mapping[str, Any]) -> Dict[str, Any]:
    statement: Dict[str, Any] = {
        "schema": PUBLIC_STATEMENT_SCHEMA,
        "protocol_id": metadata["protocol_id"],
        "rounding_mode": validate_rounding_mode(metadata.get("rounding_mode")),
        "integer_bound_ledger_digest": metadata["integer_bound_ledger_digest"],
        "features": {
            "shape": list(graph.features.shape),
            "values": [int(value) for value in graph.features.ravel()],
        },
        "graph": {
            "absolute_ids": [int(value) for value in graph.absolute_ids],
            "src": [int(value) for value in graph.src],
            "dst": [int(value) for value in graph.dst],
            "edge_graph": [int(value) for value in graph.edge_graph],
            "node_ptr": [int(value) for value in graph.node_ptr],
            "edge_ptr": [int(value) for value in graph.edge_ptr],
            "self_loop_rule": graph.self_loop_rule,
            "edge_sort_rule": graph.edge_sort_rule,
            "batching_rule": graph.batching_rule,
        },
        "public_output": {
            "shape": list(public_y.shape),
            "values": [str(int(value)) for value in public_y.ravel()],
        },
    }
    # The digest is over the canonical statement object without the digest
    # member itself.  This mirrors ``verifier_context.validate_external_context``
    # and keeps the synthetic adapter on the same binding rule as production.
    statement["statement_digest"] = digest_json(statement)
    return statement


def _validate_public_statement(statement: Mapping[str, Any]) -> Dict[str, Any]:
    """Require the external statement schema and its canonical digest."""
    if not isinstance(statement, Mapping):
        raise SyntheticVerificationError("synthetic public statement must be an object")
    if statement.get("schema") != PUBLIC_STATEMENT_SCHEMA:
        raise SyntheticVerificationError("synthetic public statement schema mismatch")
    payload = dict(statement)
    statement_digest = payload.pop("statement_digest", None)
    if not isinstance(statement_digest, str) or statement_digest != digest_json(payload):
        raise SyntheticVerificationError("synthetic public statement digest mismatch")
    return payload


def _build_external_context(
    graph: LocalGraph,
    public_y: np.ndarray,
    metadata: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> Dict[str, Any]:
    statement = _public_statement(graph, public_y, metadata)
    static_vk: Dict[str, Any] = {
        "schema": "zkGPT-static-table-vk-v1",
        "rounding_mode": validate_rounding_mode(metadata.get("rounding_mode")),
        "static_table_id": metadata["static_table_id"],
        "quant_cfg_id": metadata["quant_cfg_id"],
        "static_table_descriptor_schema": metadata["static_table_descriptor_schema"],
        "static_table_content_digest": metadata["static_table_content_digest"],
        "static_table_descriptor_digest": metadata["static_table_descriptor_digest"],
        "integer_bound_ledger_digest": metadata["integer_bound_ledger_digest"],
        "table_registry_digest": metadata["static_table_registry_digest"],
        "tables": copy.deepcopy(metadata["static_table_registry"]),
    }
    static_vk["vk_digest"] = digest_json(static_vk)
    setup: Dict[str, Any] = {
        "schema": "zkGPT-gat-verifier-setup-v1",
        "protocol_id": metadata["protocol_id"],
        "proof_backend_id": metadata["proof_backend_id"],
        "commitment_backend_id": metadata["commitment_backend_id"],
        "circuit_layout_id": metadata["circuit_layout_id"],
        "model_arch_id": metadata["model_arch_id"],
        "model_param_id": metadata["model_param_id"],
        "quant_cfg_id": metadata["quant_cfg_id"],
        "rounding_mode": validate_rounding_mode(metadata.get("rounding_mode")),
        "static_table_id": metadata["static_table_id"],
        "static_table_descriptor_schema": metadata["static_table_descriptor_schema"],
        "static_table_content_digest": metadata["static_table_content_digest"],
        "static_table_descriptor_digest": metadata["static_table_descriptor_digest"],
        "registry_digest": registry["registry_digest"],
        "Layout_in": metadata["Layout_in"],
        "Layout_in_digest": metadata["Layout_in_digest"],
        "domain_cfg": metadata["domain_cfg"],
        "dim_cfg": metadata["dim_cfg"],
        "integer_bound_ledger": metadata["integer_bound_ledger"],
        "integer_bound_ledger_digest": metadata["integer_bound_ledger_digest"],
    }
    setup["setup_digest"] = digest_json(setup)
    context = {
        "schema": "zkGPT_external_verifier_context_v1",
        "public_statement": statement,
        "model_registry": copy.deepcopy(registry),
        "static_table_vk": static_vk,
        "setup_context": setup,
    }
    context["context_digest"] = digest_json(context)
    return context


def _registry_metadata_seed(
    config_payload: Mapping[str, Any],
    metadata: Mapping[str, Any],
    descriptor: Mapping[str, Any],
    ledger: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build the test adapter's authenticated static-table seed.

    ``rounding_mode`` is part of the table identity.  Keep its source
    explicit here: both public config and serialized proof metadata must
    resolve to a supported mode and must agree before the seed is handed to
    the native table-registry constructor.
    """
    try:
        config_rounding_mode = validate_rounding_mode(config_payload.get("rounding_mode"))
        metadata_rounding_mode = validate_rounding_mode(metadata.get("rounding_mode"))
    except (TypeError, ValueError) as exc:
        raise SyntheticVerificationError(str(exc)) from exc
    if metadata_rounding_mode != config_rounding_mode:
        raise SyntheticVerificationError(
            "synthetic config/metadata rounding_mode mismatch"
        )
    return {
        "quant_cfg_id": config_payload["quant_cfg_id"],
        "static_table_id": config_payload["static_table_id"],
        "rounding_mode": metadata_rounding_mode,
        "static_table_descriptor_schema": descriptor["schema"],
        "static_table_content_digest": descriptor["content_digest"],
        "static_table_descriptor_digest": descriptor["descriptor_digest"],
        "encoding_id": ENCODING_ID,
        "integer_bound_ledger": ledger,
        "integer_bound_ledger_digest": digest_json(ledger),
    }


def _invoke_formal_verifier(
    metadata: Dict[str, Any],
    commitments: Dict[str, Any],
    transcript: Dict[str, Any],
    lasso: Dict[str, Any],
    matrix: Dict[str, Any],
    gkr: Dict[str, Any],
    combine: Dict[str, Any],
    openings: Dict[str, Any],
    privacy_profile: Dict[str, Any],
    backend: str,
    external_context: Dict[str, Any],
) -> Dict[str, Any]:
    # Keep the adapter compatible with the named public API when it is added,
    # while accepting the current source's private spelling during transition.
    import zkgpt_gat.proof as proof_module

    verifier = getattr(proof_module, "verify_formal_proof", None)
    if verifier is None:
        verifier = proof_module._verify_formal
    return verifier(
        metadata=metadata,
        commitments=commitments,
        transcript=transcript,
        lasso=lasso,
        matrix=matrix,
        gkr=gkr,
        combine=combine,
        openings=openings,
        privacy_profile=privacy_profile,
        gpu_claim=None,
        backend=backend,
        external_context=external_context,
    )


def verify_serialized_bundle(bundle_root: pathlib.Path) -> Dict[str, Any]:
    """Verify a synthetic bundle using only its public/test trust boundary."""
    root = pathlib.Path(bundle_root).expanduser().resolve()
    audit = PublicInputAccessAudit(root)
    errors: list[str] = []
    try:
        if not root.is_dir():
            raise SyntheticVerificationError(f"synthetic bundle root is not a directory: {root}")
        artifact = (root / "proof").resolve()
        if artifact.parent != root or not artifact.is_dir():
            raise SyntheticVerificationError("synthetic proof must be exactly bundle_root/proof")
        config_payload = _read_json(audit, root / "public_config.json", label="public synthetic config", allowed_relative=PUBLIC_ROOT_FILES, scan_fields=True)
        graph_payload = _read_json(audit, root / "public_graph.json", label="public synthetic graph", allowed_relative=PUBLIC_ROOT_FILES, scan_fields=True)
        model_payload = _read_json(audit, root / "public_model.json", label="public synthetic model", allowed_relative=PUBLIC_ROOT_FILES, scan_fields=True)
        y_payload = _read_json(audit, root / "public_y.json", label="public synthetic Y", allowed_relative=PUBLIC_ROOT_FILES, scan_fields=True)
        descriptor_name = str(config_payload.get("static_descriptor_path", ""))
        descriptor_path = (root / descriptor_name).resolve()
        if descriptor_path.parent != root or descriptor_path.name != descriptor_name:
            raise SyntheticVerificationError("synthetic static descriptor must be a direct bundle-root file")
        descriptor = _read_json(audit, descriptor_path, label="public synthetic static descriptor", allowed_relative=PUBLIC_ROOT_FILES | {descriptor_name}, scan_fields=True)
        _validate_static_descriptor(descriptor, config_payload)
        graph = _local_graph(graph_payload)
        manifest = _manifest_and_model(model_payload)
        if config_payload.get("model") != model_payload:
            raise SyntheticVerificationError("synthetic config/model public descriptors differ")
        if config_payload.get("dataset") != "synthetic-e2e":
            raise SyntheticVerificationError("synthetic adapter accepts only the dedicated synthetic dataset")
        if config_payload.get("task_type") != graph.task_type or config_payload.get("report_unit") != graph.report_unit:
            raise SyntheticVerificationError("synthetic config/graph task identity mismatch")
        config_rounding_mode = validate_rounding_mode(config_payload.get("rounding_mode"))
        range_bits = _int(config_payload.get("range_bits"), "range_bits")
        if range_bits <= 0 or range_bits > 16:
            raise SyntheticVerificationError("synthetic range_bits is outside the bounded adapter domain")
        y_shape = tuple(_int_list(y_payload.get("shape"), "public_y.shape"))
        y_values = _int_list(y_payload.get("values"), "public_y.values", nonempty=True)
        if y_shape != (graph.num_nodes, graph.num_classes) or int(np.prod(y_shape, dtype=np.int64)) != len(y_values):
            raise SyntheticVerificationError("synthetic public Y shape mismatch")
        public_y = np.asarray(y_values, dtype=object).reshape(y_shape)

        model_param_id = str(config_payload.get("model_param_id", model_payload.get("model_param_id", "")))
        registry_relative = str(config_payload.get("registry_relative_path", ""))
        registry_path = (root / registry_relative).resolve()
        canonical_registry = canonical_registry_public_path(root, model_param_id).resolve()
        if registry_path != canonical_registry or registry_path.parent == artifact:
            raise SyntheticVerificationError("synthetic registry path is not the canonical public com_W registry")
        registry = _read_json(audit, registry_path, label="trusted public com_W registry", allowed_relative={registry_relative}, scan_fields=True)
        if registry.get("protocol") != "zkGPT-native-model-registration-v1":
            raise SyntheticVerificationError("synthetic registry protocol mismatch")
        registry_without_digest = dict(registry)
        registry_digest = registry_without_digest.pop("registry_digest", None)
        if registry_digest != digest_json(registry_without_digest):
            raise SyntheticVerificationError("synthetic registry digest mismatch")
        binding = registry.get("registration_binding")
        if not isinstance(binding, Mapping) or registry.get("registration_binding_digest") != digest_json(binding):
            raise SyntheticVerificationError("synthetic registry binding digest mismatch")
        for key in ("model_arch_id", "model_param_id", "quant_cfg_id", "static_table_id"):
            if str(binding.get(key, "")) != str(config_payload.get(key, model_payload.get(key, ""))):
                raise SyntheticVerificationError(f"synthetic registry identity mismatch: {key}")
        registry_rounding_mode = validate_rounding_mode(binding.get("rounding_mode"))
        if registry_rounding_mode != config_rounding_mode:
            raise SyntheticVerificationError("synthetic config/registry rounding_mode mismatch")

        metadata = _read_proof_json(audit, artifact, "metadata.json")
        commitments = _read_proof_json(audit, artifact, "commitments.json")
        transcript = _read_proof_json(audit, artifact, "transcript.json")
        lasso = _read_proof_json(audit, artifact, "lookup_proof.json")
        fixed_tables = _read_proof_json(audit, artifact, "fixed_lookup_tables.json")
        matrix = _read_proof_json(audit, artifact, "matrix_sumcheck_proof.json")
        gkr = _read_proof_json(audit, artifact, "gkr_proof.json")
        combine = _read_proof_json(audit, artifact, "combine_proof.json")
        openings = _read_proof_json(audit, artifact, "hyrax_mle_openings.json")
        privacy_profile = _read_proof_json(audit, artifact, "privacy_profile.json")
        proof_payload = _read_proof_json(audit, artifact, "proof_payload.json")
        metadata_rounding_mode = validate_rounding_mode(metadata.get("rounding_mode"))
        if metadata_rounding_mode != config_rounding_mode:
            raise SyntheticVerificationError("synthetic config/metadata rounding_mode mismatch")
        backend = str(metadata.get("backend", ""))
        if backend != "cpu":
            raise SyntheticVerificationError("synthetic adapter only supports the CPU fixture")
        expected_payload = {
            "Pi_Lasso": lasso,
            "FixedLookupTables": fixed_tables,
            "Pi_MatMul": matrix,
            "Pi_GKR": gkr,
            "Pi_combine": combine,
            "Open_in": openings,
            "PrivacyProfile": privacy_profile,
        }
        for key, expected in expected_payload.items():
            if proof_payload.get(key) != expected:
                raise SyntheticVerificationError(f"serialized proof payload/{key} mismatch")
        if proof_payload.get("GPU_proof_contribution") is not None:
            raise SyntheticVerificationError("synthetic CPU proof unexpectedly contains GPU contribution")

        # Rebuild every public cardinality and descriptor used by the formal
        # verifier from graph/architecture/registry files, never from a trace.
        ledger = _ledger(
            graph,
            manifest,
            binding.get("registered_model_bounds", {}),
            str(config_payload["quant_cfg_id"]),
            config_rounding_mode,
            range_bits=range_bits,
        )
        # Construct the expected fixed-table registry using only the trusted
        # descriptor identity and the public bound ledger.  The same public
        # table descriptors also determine function-family L_qry padding.
        import zkgpt_gat.proof as proof_module

        registry_metadata_seed = _registry_metadata_seed(
            config_payload,
            metadata,
            descriptor,
            ledger,
        )
        expected_tables = proof_module._static_lookup_registry(registry_metadata_seed, range_bits)
        layout, lasso_families, matrix_families, model_arrays = _expected_layout(
            graph,
            manifest,
            range_bits,
            ledger,
            fixed_tables=expected_tables,
        )
        n_in = _int(binding.get("N_in"), "registry N_in")
        if n_in != layout["N_in"] or binding.get("Layout_in_digest") != layout["layout_digest"]:
            raise SyntheticVerificationError("registry N_in/Layout_in does not match independently rebuilt layout")
        if fixed_tables != metadata.get("static_table_registry"):
            raise SyntheticVerificationError("serialized fixed tables differ from metadata")
        if fixed_tables != expected_tables:
            raise SyntheticVerificationError("fixed-table registry is not derived from the public descriptor/ledger")

        hidden, classes, k_out, layers = _dimensions(manifest)
        dcat = [int(item["head_count"]) * int(item["head_dim"]) for item in hidden]
        expected_dims = {
            "G_batch": graph.graph_count,
            "N": graph.num_nodes,
            "E": int(graph.src.size),
            "N_total": graph.n_total,
            "node_ptr": graph.node_ptr.tolist(),
            "edge_ptr": graph.edge_ptr.tolist(),
            "L": layers,
            "d_in": [int(manifest["d_in_profile"][0])] + dcat,
            "d_h": [int(item["head_dim"]) for item in hidden],
            "d_cat": dcat,
            "C": classes,
            "B": range_bits,
            "K_out": k_out,
            "hidden_profile": [{"head_count": int(item["head_count"]), "head_dim": int(item["head_dim"])} for item in hidden],
        }
        expected_domain = {
            "n_FH": next_power_of_two(graph.num_nodes * graph.num_features),
            "n_edge": next_power_of_two(max(int(graph.src.size), graph.num_nodes, graph.num_nodes * max([int(item["head_dim"]) for item in hidden] + [classes]))),
            "n_in": n_in,
            "n_d_h": next_power_of_two(max(int(item["head_dim"]) for item in hidden)),
            "n_cat": next_power_of_two(max(dcat)),
            "n_C": next_power_of_two(classes),
            "n_N": next_power_of_two(graph.num_nodes),
        }
        topology = {
            "absolute_ids": [int(value) for value in graph.absolute_ids],
            "src": [int(value) for value in graph.src],
            "dst": [int(value) for value in graph.dst],
            "node_ptr": graph.node_ptr.tolist(),
            "edge_ptr": graph.edge_ptr.tolist(),
        }
        expected_x_digest = _public_x_digest(graph)
        expected_y_digest = _array_digest("Y", public_y)
        expected_registry_metadata = {
            "protocol": registry["protocol"],
            "registry_digest": registry["registry_digest"],
            "registration_binding_digest": registry["registration_binding_digest"],
            "public_path": registry_relative,
            "commitment_backend": registry["commitment_backend"],
            "public_descriptor": registry,
        }
        expected_metadata: Dict[str, Any] = {
            "protocol_id": PROTOCOL_ID,
            "proof_backend_id": PROOF_BACKEND_ID,
            "commitment_backend_id": COMMITMENT_BACKEND_ID,
            "model_arch_id": model_payload["model_arch_id"],
            "model_param_id": model_payload["model_param_id"],
            "static_table_id": config_payload["static_table_id"],
            "quant_cfg_id": config_payload["quant_cfg_id"],
            "static_table_descriptor_schema": descriptor["schema"],
            "static_table_content_digest": descriptor["content_digest"],
            "static_table_descriptor_digest": descriptor["descriptor_digest"],
            "static_table_descriptor_path": descriptor_name,
            "domain_cfg": expected_domain,
            "dim_cfg": expected_dims,
            "encoding_id": ENCODING_ID,
            "padding_rule_id": PADDING_RULE_ID,
            "circuit_layout_id": CIRCUIT_LAYOUT_ID,
            "lasso_cfg_id": LASSO_CFG_ID,
            "matrix_proof_cfg_id": MATRIX_PROOF_CFG_ID,
            "dataset": config_payload["dataset"],
            "dataset_display_name": config_payload.get("dataset_display_name", graph.name),
            "task_type": graph.task_type,
            "report_unit": graph.report_unit,
            "batching_rule": graph.batching_rule,
            "self_loop_rule": graph.self_loop_rule,
            "edge_sort_rule": graph.edge_sort_rule,
            "chunking_rule": graph.chunking_rule,
            "full_dataset": True,
            "X_pub_digest": expected_x_digest,
            "public_input_layout": "X_pub=features+topology+selectors; X_pub is verifier-reconstructed and uncommitted",
            "public_output_policy": "Y is a public statement output claim and is not part of X_pub or any private commitment",
            "private_input_layout": "X_pub|W|Aux|L_qry",
            "segment_order": ["X_pub", "W", "Aux", "L_qry"],
            "privacy_extension_profile_id": PROFILE_ID,
            "privacy_profile_status": PROFILE_STATUS,
            "registered_model_commitment_scope": "canonical_integer_GAT_model_weights_with_layout_and_bound_ledger",
            "Layout_in": layout,
            "Layout_in_digest": layout["layout_digest"],
            "integer_bound_ledger": ledger,
            "integer_bound_ledger_digest": digest_json(ledger),
            "static_table_registry": expected_tables,
            "static_table_registry_digest": digest_json(expected_tables),
            "public_output_digest": expected_y_digest,
            "model_registry": expected_registry_metadata,
            "model_registry_digest": digest_json(expected_registry_metadata),
        }
        for key, expected in expected_metadata.items():
            if metadata.get(key) != expected:
                raise SyntheticVerificationError(f"metadata field is not source-matched: {key}")
        session_nonce = str(metadata.get("session_nonce", ""))
        if not session_nonce or metadata.get("privacy_transcript_domain") != transcript_domain(session_nonce):
            raise SyntheticVerificationError("metadata session nonce/transcript domain is invalid")
        if metadata.get("metadata_order") is None:
            raise SyntheticVerificationError("metadata ordering descriptor is missing")

        # Check the native family descriptors before entering the formal
        # verifier.  These checks make the adapter independent of any proof
        # selected family labels or private cardinality.
        actual_matrix_specs = [item.get("spec") for item in matrix.get("claims", []) if isinstance(item, Mapping)]
        expected_matrix_specs = [item["spec"] for item in matrix_families]
        if actual_matrix_specs != expected_matrix_specs:
            raise SyntheticVerificationError("native MatMul descriptors differ from public architecture/graph")
        actual_lasso_families = lasso.get("families", [])
        if [item.get("family") for item in actual_lasso_families] != [item["family"] for item in lasso_families]:
            raise SyntheticVerificationError("native Lasso family order differs from public layout")
        if any(int(item.get("query_count", -1)) != int(expected["padded_count"]) for item, expected in zip(actual_lasso_families, lasso_families)):
            raise SyntheticVerificationError("native Lasso query cardinality differs from public layout")

        import zkgpt_gat.proof as proof_module

        graph_statement = _public_statement(graph, public_y, metadata)["graph"]
        native_specs = proof_module._native_gkr_specs(metadata, graph_statement, [int(value) for value in public_y.ravel()])
        actual_gkr_specs = [
            (item.get("kind"), item.get("proof", {}).get("spec"))
            for item in gkr.get("claims", [])
            if isinstance(item, Mapping) and isinstance(item.get("proof"), Mapping)
        ]
        expected_gkr_specs = [(kind, spec.descriptor()) for kind, spec in native_specs]
        if actual_gkr_specs != expected_gkr_specs:
            raise SyntheticVerificationError("native GKR descriptors differ from public architecture/graph")

        external_context = _build_external_context(graph, public_y, metadata, registry)
        statement = external_context.get("public_statement")
        _validate_public_statement(statement)
        formal = _invoke_formal_verifier(
            metadata, commitments, transcript, lasso, matrix, gkr, combine,
            openings, privacy_profile, backend, external_context,
        )
        result = dict(formal)
        result["adapter"] = "zkGPT.tests.verify_synthetic_proof"
        result["statement_schema"] = statement["schema"]
        result["statement_digest"] = statement["statement_digest"]
        result["expected_layout_digest"] = layout["layout_digest"]
        result["expected_native_descriptor_counts"] = {
            "lasso": len(lasso_families),
            "matmul": len(matrix_families),
            "gkr": len(native_specs),
            "model_arrays": len(model_arrays),
        }
        result["public_input_access_audit"] = audit.finish(status=str(result.get("status", "VERIFY_FAILED")))
        return result
    except (SyntheticVerificationError, KeyError, TypeError, ValueError, OverflowError) as exc:
        errors.append(str(exc))
        result = {
            "status": "VERIFY_FAILED",
            "reason": str(exc),
            "adapter": "zkGPT.tests.verify_synthetic_proof",
            "public_input_access_audit": audit.finish(status="VERIFY_FAILED", errors=errors),
        }
        return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify a test-only serialized zkGPT synthetic proof")
    parser.add_argument("--bundle-root", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, default=None, help="test-only JSON report path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = verify_serialized_bundle(args.bundle_root)
    output = args.output.expanduser().resolve() if args.output is not None else args.bundle_root.expanduser().resolve() / "public_input_access_audit.json"
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except OSError as exc:
        result = {
            **result,
            "status": "VERIFY_FAILED",
            "reason": f"cannot write test-only adapter report: {exc}",
        }
        print(json.dumps(result, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("status") == "VERIFY_OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FORBIDDEN_RAW_FIELDS",
    "PublicInputAccessAudit",
    "SyntheticVerificationError",
    "main",
    "verify_serialized_bundle",
]
