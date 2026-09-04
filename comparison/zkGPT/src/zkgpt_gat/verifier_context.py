"""External source-matched verifier context for serialized zkGPT proofs.

The proof directory is untrusted.  This module reconstructs the public
statement from an explicit config, canonical graph cache, model manifest,
trusted model-registry public JSON, and an independently supplied public Y.
Artifact metadata is compared with this context before the payload verifier
is called; it is never used to choose the model, layout, or table identities.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import pathlib
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from .config import AppConfig, load_app_config
from .constants import (
    CIRCUIT_LAYOUT_ID,
    COMMITMENT_BACKEND_ID,
    ENCODING_ID,
    EXTERNAL_VERIFIER_CONTEXT_FIELDS,
    EXTERNAL_PUBLIC_STATEMENT_SCHEMA,
    EXTERNAL_VERIFIER_CONTEXT_SCHEMA,
    FUNCTION_LOOKUP_DOMAIN_BITS,
    LASSO_CFG_ID,
    LOOKUP_TUPLE_ENCODING_ID,
    MATRIX_PROOF_CFG_ID,
    PADDING_RULE_ID,
    PROOF_BACKEND_ID,
    PROTOCOL_ID,
    PUBLIC_STATEMENT_FIELDS,
    STATIC_TABLE_VK_SCHEMA,
    STATIC_TABLE_VK_FIELDS,
    STATIC_TABLE_FAMILY_IDS,
    VERIFIER_SETUP_FIELDS,
    VERIFIER_SETUP_SCHEMA,
)
from .data import GraphDataset, LocalGraph, load_dataset, normalize_graph
from .field import BN254_FR_MODULUS, next_power_of_two, to_field_int
from .forward import _exp_map_fixed, _elu_table_value
from .native_gkr import LinearGateSpec, LinearTerm, OracleRef, ProductGateSpec, identity_oracle
from .model import QUANT_SCALE_PROFILES, _validate_canonical_manifest
from .quantization import CANONICAL_QUANT_CFG_ID, canonical_input_preprocess_id, quantize_ties_away
from .reference_forward import REFERENCE_FORWARD_ENGINE, reference_forward_source_sha256
from .registry import canonical_registry_public_path, require_registry_transaction_id
from .resources import STATIC_TABLE_DESCRIPTOR_RELATIVE_PATH, canonical_dataset_name, required_resource_paths, validate_static_table_descriptor
from .rounding import (
    validate_rounding_mode,
)
from .bound_ledger import build_integer_bound_ledger
from .util import digest_json

_FIELD_HALF = (BN254_FR_MODULUS - 1) // 2
_HYRAX_BACKEND = COMMITMENT_BACKEND_ID
PUBLIC_Y_SCHEMA = "zkGPT_public_y_v1"

class ContextError(ValueError):
    """The external statement or a trust-boundary path is invalid."""


def _inside(path: pathlib.Path, root: pathlib.Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _file(path: pathlib.Path, label: str) -> pathlib.Path:
    path = pathlib.Path(path).expanduser().resolve()
    if not path.is_file():
        raise ContextError(f"{label} does not exist or is not a file: {path}")
    return path


def _owned_file(path: pathlib.Path, label: str) -> pathlib.Path:
    """Load a verifier input only from an owned one-link regular file."""
    raw = pathlib.Path(path).expanduser()
    try:
        info = raw.lstat()
    except OSError as exc:
        raise ContextError(f"cannot inspect {label}: {raw}") from exc
    import stat

    if stat.S_ISLNK(info.st_mode):
        raise ContextError(f"{label} must not be a symlink: {raw}")
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ContextError(f"{label} must be an owned regular file: {raw}")
    return _file(raw, label)


def _directory(path: pathlib.Path, label: str) -> pathlib.Path:
    path = pathlib.Path(path).expanduser().resolve()
    if not path.is_dir():
        raise ContextError(f"{label} does not exist or is not a directory: {path}")
    return path


def _json(path: pathlib.Path, label: str) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContextError(f"cannot read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContextError(f"{label} must be a JSON object: {path}")
    return value


def _array_digest(label: str, value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    h = hashlib.sha256()
    h.update(label.encode("utf-8")); h.update(str(array.shape).encode("ascii")); h.update(str(array.dtype).encode("ascii"))
    if array.dtype == object:
        for item in array.ravel().tolist(): h.update(str(item).encode("ascii")); h.update(b",")
    else:
        h.update(memoryview(array).cast("B"))
    return h.hexdigest()


def _structured_function_vk(table_id: str, function: str, rule: str) -> Dict[str, Any]:
    """Rebuild the native fixed-function verification-key descriptor.

    This is intentionally kept in the external-context module instead of
    accepting a producer-supplied table digest.  The fresh verifier therefore
    has one source of truth for the tuple encoding, evaluator identity, and
    complete function-table domain.
    """
    input_scale = {"T_LReLU": 256, "T_ELU": 16, "T_exp": 256}.get(table_id)
    if input_scale is None:
        raise ContextError(f"unknown structured table {table_id}")
    vk = {
        "protocol": "structured-fixed-function-verification-key-v2",
        "table_id": table_id,
        "function": function,
        "evaluation_rule": rule,
        "domain_bits": FUNCTION_LOOKUP_DOMAIN_BITS,
        "input_domain": {
            "signed": True,
            "min_inclusive": -(1 << (FUNCTION_LOOKUP_DOMAIN_BITS - 1)),
            "max_exclusive": 1 << (FUNCTION_LOOKUP_DOMAIN_BITS - 1),
            "fixed_scale": input_scale,
        },
        "field_modulus": str(BN254_FR_MODULUS),
        "evaluator_id": {
            "T_LReLU": "signed_fixed_point_leaky_relu_alpha_0p2_nearest_q",
            "T_ELU": "signed_fixed_point_elu_expm1_nearest_q",
            "T_exp": "signed_fixed_point_exp_neg_nearest_q",
        }[table_id],
        "selected_opening_rule": "fixed verifier-key evaluator bound to the complete native Lasso query/table commitment argument",
    }
    vk["vk_digest"] = digest_json({key: value for key, value in vk.items() if key != "vk_digest"})
    return vk


def _fixed_table_eval_q(table_id: str, x_value: int) -> int:
    value = int(x_value)
    if value > _FIELD_HALF:
        value -= BN254_FR_MODULUS
    if table_id == "T_LReLU":
        return _fixed_lrelu(value)
    if table_id == "T_ELU":
        return int(_elu_table_value(value))
    if table_id == "T_exp":
        return int(_exp_map_fixed(value))
    raise ContextError(f"unsupported fixed function table {table_id}")


def _canonical_function_rows_digest(
    table_id: str,
    domain_min: int,
    domain_max: int,
    tuple_encoding: Mapping[str, Any],
) -> str:
    h = hashlib.sha256()
    base = int(tuple_encoding["base"])
    x_bound = int(tuple_encoding["x_bound"])
    y_bound = int(tuple_encoding["y_bound"])
    for x_value in range(int(domain_min), int(domain_max) + 1):
        y_value = _fixed_table_eval_q(table_id, x_value)
        packed = (x_value + x_bound) * base + (y_value + y_bound)
        h.update(f"{x_value},{y_value},{packed}\n".encode("ascii"))
    return h.hexdigest()


def _field_chunk_digests(values: Sequence[int], chunk_size: int = 8192) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    for chunk_index, begin in enumerate(range(0, len(values), chunk_size)):
        h = hashlib.sha256()
        chunk_values = values[begin: begin + chunk_size]
        for value in chunk_values:
            h.update(str(to_field_int(value)).encode("ascii"))
            h.update(b",")
        chunks.append({"chunk_index": chunk_index, "count": len(chunk_values), "sha256": h.hexdigest()})
    return chunks


def _canonical_static_table_registry(metadata: Mapping[str, Any]) -> Dict[str, Any]:
    """Reconstruct the exact fixed-table registry from authenticated inputs."""
    required = {
        "quant_cfg_id", "rounding_mode", "static_table_id", "static_table_descriptor_schema",
        "static_table_content_digest", "static_table_descriptor_digest", "encoding_id",
        "integer_bound_ledger", "integer_bound_ledger_digest", "dim_cfg",
    }
    if not required <= set(metadata):
        raise ContextError("metadata is incomplete for canonical static-table registry")
    ledger = metadata.get("integer_bound_ledger")
    if not isinstance(ledger, Mapping):
        raise ContextError("integer-bound ledger is missing for canonical static-table registry")
    if metadata.get("integer_bound_ledger_digest") != digest_json(ledger):
        raise ContextError("integer-bound ledger digest is self-inconsistent")
    rounding_mode = validate_rounding_mode(metadata.get("rounding_mode"))
    range_bits = int(metadata.get("dim_cfg", {}).get("B", 0))
    if range_bits <= 0:
        raise ContextError("static-table range_bits is invalid")
    common = {
        "quant_cfg_id": metadata["quant_cfg_id"],
        "rounding_mode": rounding_mode,
        "quant_scale_chain": dict(ledger.get("scales", {})),
        "static_table_id": metadata["static_table_id"],
        "static_table_descriptor_schema": metadata["static_table_descriptor_schema"],
        "static_table_content_digest": metadata["static_table_content_digest"],
        "static_table_descriptor_digest": metadata["static_table_descriptor_digest"],
        "encoding_id": metadata["encoding_id"],
        "range_bits": range_bits,
        "field_encoding_scale": 1,
        "integer_bound_ledger_digest": metadata["integer_bound_ledger_digest"],
        "certified_table_domains": ledger.get("table_domains", {}),
    }
    if common["encoding_id"] != ENCODING_ID:
        raise ContextError("static-table tuple encoding identity mismatch")

    def function_table(table_id: str, function: str, rule: str) -> Dict[str, Any]:
        vk = _structured_function_vk(table_id, function, rule)
        scales = {"T_LReLU": (256, 256), "T_ELU": (16, 16), "T_exp": (256, 16)}
        domain_key = {"T_LReLU": "lrelu", "T_ELU": "elu", "T_exp": "exp"}[table_id]
        domain = common["certified_table_domains"].get(domain_key)
        if not isinstance(domain, (list, tuple)) or len(domain) != 2:
            raise ContextError(f"missing certified domain for {table_id}")
        certified_domain = [int(domain[0]), int(domain[1])]
        domain_min, domain_max = certified_domain
        if domain_max < domain_min:
            raise ContextError(f"invalid certified domain for {table_id}")
        endpoint_outputs = [_fixed_table_eval_q(table_id, value) for value in {domain_min, domain_max, 0} if domain_min <= value <= domain_max]
        x_bound = max(abs(domain_min), abs(domain_max))
        y_bound = max((abs(int(value)) for value in endpoint_outputs), default=0)
        pack_base = 2 * y_bound + 1
        max_packed = 2 * x_bound * pack_base + 2 * y_bound
        if max_packed >= _FIELD_HALF:
            raise ContextError(f"{table_id} tuple packing exceeds BN254 signed field range")
        input_scale, output_scale = scales[table_id]
        descriptor: Dict[str, Any] = {
            "table_id": table_id,
            "table_kind": "structured_fixed_function_table",
            "function": function,
            "evaluation_rule": rule,
            "field_modulus": str(BN254_FR_MODULUS),
            "input_scale": input_scale,
            "output_scale": output_scale,
            "domain_policy": f"signed fixed-point representatives with abs(x) < 2^{FUNCTION_LOOKUP_DOMAIN_BITS - 1}",
            "input_domain": {
                "signed": True,
                "min_inclusive": -(1 << (FUNCTION_LOOKUP_DOMAIN_BITS - 1)),
                "max_exclusive": 1 << (FUNCTION_LOOKUP_DOMAIN_BITS - 1),
                "fixed_scale": input_scale,
            },
            "certified_domain": certified_domain,
            "tuple_encoding": {
                "encoding_id": LOOKUP_TUPLE_ENCODING_ID,
                "x_bound": x_bound,
                "y_bound": y_bound,
                "base": pack_base,
                "max_packed": max_packed,
            },
            "canonical_row_order": "ascending signed x; row=(x,y,Pack_f(x,y))",
            "row_count": domain_max - domain_min + 1,
            "integer_bound_ledger_digest": common["integer_bound_ledger_digest"],
            "table_materialization": "preprocessed structured evaluator with fixed verification key; selected openings are verified by the fixed VK and are not accepted as query-generated table side",
            "structured_verification_key": vk,
            "chunks": [{"chunk_index": 0, "count": 1, "sha256": digest_json({"structured_table_vk": vk})}],
            **common,
        }
        descriptor["rows_digest"] = _canonical_function_rows_digest(table_id, domain_min, domain_max, descriptor["tuple_encoding"])
        descriptor["table_digest"] = digest_json(descriptor)
        descriptor["table_commitment"] = digest_json({
            "commitment_backend": "structured-fixed-function-vk-commitment-v2",
            "table_id": table_id,
            "structured_verification_key_digest": digest_json(vk),
            "chunk_roots": descriptor["chunks"],
        })
        return descriptor

    range_values = [to_field_int(value) for value in range(1 << range_bits)]
    range_descriptor: Dict[str, Any] = {
        "table_id": "T_range",
        "table_kind": "materialized_fixed_range_table",
        "function": "Range",
        "domain_policy": f"all limbs in [0,{(1 << range_bits) - 1}]",
        "table_materialization": "preprocessed fixed limb table",
        "entry_count": len(range_values),
        "chunks": _field_chunk_digests(range_values),
        **common,
    }
    range_descriptor["table_digest"] = digest_json({key: value for key, value in range_descriptor.items() if key not in {"table_digest", "table_commitment"}})
    range_descriptor["table_commitment"] = digest_json({
        "commitment_backend": "materialized-fixed-range-table-commitment-v2",
        "table_id": "T_range",
        "range_bits": range_bits,
        "entry_count": len(range_values),
        "chunk_roots": range_descriptor["chunks"],
    })
    return {
        "T_LReLU": function_table("T_LReLU", "LeakyReLU", "y=x if x>=0 else round(0.2*x)"),
        "T_ELU": function_table("T_ELU", "ELU", "y=x if x>=0 else round(exp(x)-1)"),
        "T_exp": function_table("T_exp", "ExpMap", "y=round(exp(-x)) for nonnegative x"),
        "T_range": range_descriptor,
    }


def _authenticated_static_table_registry(metadata: Mapping[str, Any]) -> Tuple[Dict[str, Any], str]:
    registry = metadata.get("static_table_registry")
    if not isinstance(registry, Mapping) or set(registry) != set(STATIC_TABLE_FAMILY_IDS):
        raise ContextError("static-table registry family set is invalid")
    digest = digest_json(dict(registry))
    if metadata.get("static_table_registry_digest") != digest:
        raise ContextError("static-table registry digest is self-inconsistent")
    canonical = _canonical_static_table_registry(metadata)
    if dict(registry) != canonical:
        raise ContextError("static-table registry does not match authenticated canonical descriptor")
    return copy.deepcopy(dict(registry)), digest


def _graph_digest(graph: LocalGraph) -> str:
    return digest_json({"absolute_ids": _array_digest("absolute_ids", graph.absolute_ids), "src": _array_digest("src", graph.src), "dst": _array_digest("dst", graph.dst), "node_ptr": graph.node_ptr.tolist(), "edge_ptr": graph.edge_ptr.tolist()})


def _round_feature(value: float, scale: int = 1) -> int:
    scaled = float(value) * int(scale)
    return int(np.floor(scaled + 0.5) if scaled >= 0 else np.ceil(scaled - 0.5))


def _x_digest(graph: LocalGraph, layer_count: int, class_count: int, k_out: int, input_scale: int = 1) -> str:
    features = np.asarray([_round_feature(v, input_scale) for v in np.ravel(graph.features)], dtype=np.int64).reshape(graph.features.shape)
    return digest_json({"features": _array_digest("X_pub.H_input", features), "topology": _graph_digest(graph), "selectors": {"segment_order": ["X_pub", "W", "Aux", "L_qry"], "graph_count": graph.graph_count}})


def _integer_array(value: Any, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.size == 0 or not (np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.floating) or array.dtype == object):
        raise ContextError(f"{label} must be a non-empty integer array")
    if np.issubdtype(array.dtype, np.floating) and (not np.all(np.isfinite(array)) or not np.all(array == np.floor(array))):
        raise ContextError(f"{label} contains non-integral values")
    output = np.empty(array.shape, dtype=object)
    for index in np.ndindex(array.shape):
        try: item = int(array[index])
        except (TypeError, ValueError, OverflowError) as exc: raise ContextError(f"{label} contains a non-integer value") from exc
        if float(array[index]) != item: raise ContextError(f"{label} contains a non-integral value")
        output[index] = item
    return output


def load_public_y(path: pathlib.Path) -> np.ndarray:
    path = _file(path, "public Y")
    if path.suffix.lower() == ".npy":
        try: return _integer_array(np.load(path, allow_pickle=False), "public Y")
        except (OSError, ValueError) as exc: raise ContextError(f"cannot load public Y: {path}: {exc}") from exc
    if path.suffix.lower() == ".json":
        try: raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc: raise ContextError(f"cannot read public Y: {path}: {exc}") from exc
        shape = None
        if isinstance(raw, dict):
            keys = [key for key in ("Y", "public_y", "values", "logits") if key in raw]
            if len(keys) != 1: raise ContextError("public Y JSON must contain exactly one of Y/public_y/values/logits")
            value = raw[keys[0]]; shape = raw.get("shape")
        elif isinstance(raw, list): value = raw
        else: raise ContextError("public Y JSON must contain an integer array")
        array = _integer_array(value, "public Y")
        if shape is not None:
            target = tuple(int(item) for item in shape)
            if any(item < 0 for item in target) or int(np.prod(target, dtype=np.int64)) != array.size: raise ContextError("public Y shape does not match values")
            array = array.reshape(target)
        return array
    try: values = [int(item) for item in path.read_text(encoding="utf-8").split()]
    except (OSError, ValueError) as exc: raise ContextError(f"public Y text is not an integer vector: {path}") from exc
    return _integer_array(values, "public Y")


def _file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_tree_sha256(root: pathlib.Path) -> str:
    if root.is_symlink() or not root.is_dir():
        raise ContextError(f"public-Y cache root is not a real directory: {root}")
    files: list[pathlib.Path] = []
    for current, directories, names in os.walk(root, followlinks=False):
        current_path = pathlib.Path(current)
        if any((current_path / name).is_symlink() for name in directories):
            raise ContextError("public-Y cache contains a symlink directory")
        files.extend(current_path / name for name in names)
    files.sort(key=lambda path: path.relative_to(root).as_posix())
    digest = hashlib.sha256()
    for path in files:
        _owned_file(path, "public-Y cache file")
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(path.stat().st_size).encode("ascii"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\n")
    return digest.hexdigest()


def _public_y_provenance(
    root: pathlib.Path,
    config: AppConfig,
    graph: LocalGraph,
    manifest: Mapping[str, Any],
    public_y_path: pathlib.Path,
    output: np.ndarray,
    y_digest: str,
) -> Dict[str, Any]:
    """Authenticate the descriptor/provenance of the external public Y.

    The verifier never loads private tensor values or replays the producer
    forward.  It does authenticate the independently generated descriptor,
    its exact Y digest, source implementation digest, checkpoint/cache
    identities, and all numerical/model fields that bind the statement.
    """
    try:
        payload = _json(_owned_file(public_y_path, "public Y descriptor"), "public Y descriptor")
    except ContextError:
        raise
    if payload.get("schema") != PUBLIC_Y_SCHEMA:
        raise ContextError("public Y descriptor schema is not canonical")
    descriptor = dict(payload)
    descriptor_digest = descriptor.pop("descriptor_digest", None)
    if not isinstance(descriptor_digest, str) or descriptor_digest != digest_json(descriptor):
        raise ContextError("public Y descriptor digest is invalid")
    canonical = canonical_dataset_name(config.dataset)
    paths = required_resource_paths(root, canonical)
    checkpoint_manifest = paths["checkpoint_bundle"] / "manifest.json"
    checkpoint_npz = paths["checkpoint_bundle"] / "tensors.npz"
    checkpoint_tensors = paths["checkpoint_bundle"] / "tensors.txt"
    expected_scale_chain = {key: int(value) for key, value in QUANT_SCALE_PROFILES[CANONICAL_QUANT_CFG_ID].items()}
    expected = {
        "dataset": canonical,
        "config_dataset": config.dataset,
        "model_arch_id": config.entries.get("model_arch_id"),
        "model_param_id": config.entries.get("model_param_id"),
        "quant_cfg_id": CANONICAL_QUANT_CFG_ID,
        "source_quant_cfg_id": manifest.get("quant_cfg_id"),
        "rounding_mode": config.rounding_mode,
        "L": int(manifest.get("L", -1)),
        "hidden_profile": manifest.get("hidden_profile"),
        "d_in_profile": manifest.get("d_in_profile"),
        "K_out": int(manifest.get("K_out", -1)),
        "C": int(manifest.get("C", -1)),
        "shape": [int(graph.num_nodes), int(manifest.get("C", -1))],
        "quant_scale_chain": expected_scale_chain,
        "input_preprocess_id": config.input_preprocess_id,
        "canonical_scale_chain": manifest.get("quant_scale_chain"),
        "output_average_rule": manifest.get("output_average_rule"),
        "cache_dir": str(paths["cache_dir"].relative_to(root)),
        "checkpoint_bundle": str(paths["checkpoint_bundle"].relative_to(root)),
        "graph_count": int(graph.graph_count),
        "edge_count": int(graph.src.size),
        "input_feature_rounding": "nearest_ties_away",
        "forward_engine": "zkgpt_gat.forward.forward_public_y",
        "reference_forward_engine": REFERENCE_FORWARD_ENGINE,
        "reference_forward_source_sha256": reference_forward_source_sha256(),
        "checkpoint_manifest_sha256": _file_sha256(checkpoint_manifest),
        "checkpoint_npz_sha256": _file_sha256(checkpoint_npz),
        "checkpoint_tensors_sha256": _file_sha256(checkpoint_tensors),
        "cache_tree_sha256": _cache_tree_sha256(paths["cache_dir"]),
    }
    checks: Dict[str, bool] = {}
    for key, value in expected.items():
        checks[key] = payload.get(key) == value
    checks["y_digest"] = payload.get("y_digest") == y_digest
    checks["reference_y_digest"] = payload.get("reference_y_digest") == y_digest
    checks["independent_reference_forward"] = payload.get("independent_reference_forward") is True
    checks["descriptor_digest"] = True
    if payload.get("values") is None or not np.array_equal(load_public_y(public_y_path), output):
        checks["values"] = False
    else:
        checks["values"] = True
    return {
        "status": "VERIFY_OK" if all(checks.values()) else "VERIFY_FAILED",
        "checks": checks,
        "descriptor_digest": descriptor_digest,
        "reference_forward_source_sha256": expected["reference_forward_source_sha256"],
        "public_y_path": str(public_y_path.relative_to(root)),
    }


def _manifest(bundle: pathlib.Path) -> Dict[str, Any]:
    return _json(_file(bundle / "manifest.json", "model manifest"), "model manifest")


def _dimensions(manifest: Mapping[str, Any]) -> Tuple[List[Dict[str, int]], int, int, int]:
    hidden = [{"layer_index": int(item["layer_index"]), "input_dim": int(item["input_dim"]), "head_count": int(item["head_count"]), "head_dim": int(item["head_dim"])} for item in manifest.get("hidden_layers", [])]
    hidden.sort(key=lambda item: item["layer_index"])
    outputs = sorted((int(item["head_index"]) for item in manifest.get("output_head_specs", [])))
    classes, k_out, layers = int(manifest.get("C", 0)), int(manifest.get("K_out", 0)), int(manifest.get("L", len(hidden) + 1))
    d_in_profile = manifest.get("d_in_profile")
    if not isinstance(d_in_profile, (list, tuple)) or len(d_in_profile) != len(hidden):
        raise ContextError("model manifest d_in_profile cardinality is inconsistent")
    try:
        canonical_d_in = [int(value) for value in d_in_profile]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ContextError("model manifest d_in_profile is not integral") from exc
    if (
        not hidden
        or [item["layer_index"] for item in hidden] != list(range(len(hidden)))
        or outputs != list(range(len(outputs)))
        or classes <= 0
        or k_out <= 0
        or k_out != len(outputs)
        or layers != len(hidden) + 1
        or canonical_d_in != [item["input_dim"] for item in hidden]
        or any(value <= 0 for value in canonical_d_in)
        or any(canonical_d_in[index] != hidden[index - 1]["head_count"] * hidden[index - 1]["head_dim"] for index in range(1, len(hidden)))
    ):
        raise ContextError("model manifest architecture/cardinality is inconsistent")
    return hidden, classes, k_out, layers


def _canonical_model_architecture(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract only the public architecture portion of a checkpoint manifest.

    A checkpoint manifest is a public input to the independent verifier, but
    it is still an untrusted JSON document.  Native checkpoint manifests may
    carry loader hints, tensor names, digests, or other producer-only fields;
    none of those fields are part of the verifier's model binding.  Keep only
    the dimensions and identifiers needed to rebuild the GAT relation.  In
    particular, never copy a raw manifest (or a tensor/value field from it)
    into the public statement or setup context.
    """
    hidden, classes, k_out, layers = _dimensions(manifest)
    model_arch_id = str(manifest.get("model_arch_id", ""))
    if not model_arch_id:
        raise ContextError("model manifest has no model_arch_id")
    d_in_profile = manifest.get("d_in_profile")
    if not isinstance(d_in_profile, (list, tuple)) or not d_in_profile:
        raise ContextError("model manifest has no d_in_profile")
    try:
        canonical_d_in = [int(value) for value in d_in_profile]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ContextError("model manifest d_in_profile is not integral") from exc
    if any(value <= 0 for value in canonical_d_in):
        raise ContextError("model manifest d_in_profile contains a non-positive dimension")
    if canonical_d_in != [int(item["input_dim"]) for item in hidden]:
        raise ContextError("model manifest d_in_profile does not match hidden input dimensions")
    output_average_rule = str(manifest.get("output_average_rule", ""))
    if not output_average_rule:
        raise ContextError("model manifest has no output_average_rule")
    if output_average_rule != "per_head_bias_then_arithmetic_mean":
        raise ContextError("model manifest output_average_rule is not the canonical GAT rule")
    return {
        "dataset": str(manifest.get("dataset", "")),
        "input_preprocess_id": str(manifest.get("input_preprocess_id", "")),
        "quant_cfg_id": str(manifest.get("quant_cfg_id", "")),
        "quant_scale_chain": copy.deepcopy(manifest.get("quant_scale_chain", {})),
        "model_arch_id": model_arch_id,
        "hidden_layers": [
            {
                "layer_index": int(item["layer_index"]),
                "input_dim": int(item["input_dim"]),
                "head_count": int(item["head_count"]),
                "head_dim": int(item["head_dim"]),
            }
            for item in hidden
        ],
        # Output-head specs are reduced to their public cardinality.  Raw
        # tensor-name fields that may accompany these entries are deliberately
        # not retained.
        "output_head_specs": [{"head_index": index} for index in range(k_out)],
        "C": classes,
        "K_out": k_out,
        "L": layers,
        "d_in_profile": canonical_d_in,
        "hidden_profile": [
            {"head_count": int(item["head_count"]), "head_dim": int(item["head_dim"])}
            for item in hidden
        ],
        "output_average_rule": output_average_rule,
    }


def _model_arrays(manifest: Mapping[str, Any]) -> List[Dict[str, Any]]:
    hidden, classes, _, _ = _dimensions(manifest); previous = int(manifest.get("d_in_profile", [0])[0])
    if previous <= 0: raise ContextError("model manifest has no positive input dimension")
    arrays: List[Dict[str, Any]] = []
    for layer in hidden:
        if previous != layer["input_dim"]: raise ContextError("model manifest input chain mismatch")
        for index in range(layer["head_count"]):
            prefix = f"hidden.l{layer['layer_index'] + 1}.h{index}"
            arrays.extend([{"name": f"{prefix}.W", "shape": [layer["input_dim"], layer["head_dim"]]}, {"name": f"{prefix}.a_src", "shape": [layer["head_dim"]]}, {"name": f"{prefix}.a_dst", "shape": [layer["head_dim"]]}])
        previous = layer["head_count"] * layer["head_dim"]
    for index in range(int(manifest["K_out"])):
        prefix = f"output.h{index}"
        arrays.extend([{"name": f"{prefix}.W", "shape": [previous, classes]}, {"name": f"{prefix}.a_src", "shape": [classes]}, {"name": f"{prefix}.a_dst", "shape": [classes]}, {"name": f"{prefix}.b_out", "shape": [classes]}])
    for item in arrays:
        length = 1
        for dim in item["shape"]: length *= int(dim)
        item["length"] = length
    return arrays


def _validate_registry(path: pathlib.Path, artifact: pathlib.Path, root: pathlib.Path, manifest: Mapping[str, Any], identities: Mapping[str, str]) -> Dict[str, Any]:
    path = _owned_file(path, "trusted model registry public file")
    if _inside(path, artifact): raise ContextError("trusted registry public file must not be inside proof artifact")
    if not _inside(path, root): raise ContextError("trusted registry public file must be inside project-root")
    expected_path = canonical_registry_public_path(root, identities["model_param_id"]).resolve()
    if path.resolve() != expected_path:
        raise ContextError(f"trusted model registry path is not canonical for config model identity: {expected_path}")
    registry = _json(path, "trusted model registry")
    if registry.get("protocol") != "zkGPT-native-model-registration-v1": raise ContextError("trusted registry protocol mismatch")
    if set(registry) != {
        "protocol", "model_param_id", "model_arch_id", "registration_binding",
        "registration_binding_digest", "registration_transaction_id", "commitment_backend",
        "entries", "registry_digest",
    }:
        raise ContextError("trusted registry public schema contains private or unsupported fields")
    try:
        require_registry_transaction_id(registry.get("registration_transaction_id"))
    except RuntimeError as exc:
        raise ContextError(str(exc)) from exc
    payload = dict(registry); registry_digest = payload.pop("registry_digest", None)
    if registry_digest != digest_json(payload): raise ContextError("trusted registry digest is invalid")
    binding = registry.get("registration_binding", {})
    if not isinstance(binding, dict) or registry.get("registration_binding_digest") != digest_json(binding): raise ContextError("trusted registry binding digest is invalid")
    if set(binding) != {
        "protocol", "model_arch_id", "model_param_id", "quant_cfg_id", "input_preprocess_id", "quant_scale_chain_digest", "rounding_mode",
        "static_table_id", "layout", "segment_order", "padding_rule_id",
        "integer_bound_ledger_digest", "Layout_in_digest", "N_in",
        "registered_model_bounds", "arrays",
    }:
        raise ContextError("trusted registry binding schema contains private or unsupported fields")
    for key in ("model_arch_id", "model_param_id", "quant_cfg_id", "static_table_id"):
        top_level = registry.get(key)
        if top_level is not None and str(top_level) != str(identities[key]): raise ContextError(f"trusted registry {key} mismatch")
        if str(binding.get(key, "")) != str(identities[key]): raise ContextError(f"trusted registry binding {key} mismatch")
    if str(binding.get("input_preprocess_id", "")) != canonical_input_preprocess_id(str(manifest.get("dataset", ""))):
        raise ContextError("trusted registry input_preprocess_id mismatch")
    if binding.get("quant_scale_chain_digest") != digest_json(manifest.get("quant_scale_chain", {})):
        raise ContextError("trusted registry quant_scale_chain identity mismatch")
    if registry.get("commitment_backend") != COMMITMENT_BACKEND_ID or binding.get("layout") != "X_pub|W|Aux|L_qry" or binding.get("segment_order") != ["X_pub", "W", "Aux", "L_qry"]: raise ContextError("trusted registry commitment/layout binding mismatch")
    arrays = _model_arrays(manifest); names = ["com_W"]; model_names = [item["name"] for item in arrays]
    binding_arrays = binding.get("arrays", [])
    if [item.get("name") for item in binding_arrays] != names or len(binding_arrays) != 1 or int(binding_arrays[0].get("length", -1)) != int(binding.get("N_in", -2)): raise ContextError("trusted registry model commitment labels or lengths mismatch")
    entries = registry.get("entries")
    if not isinstance(entries, dict) or set(entries) != set(names): raise ContextError("trusted registry has missing, extra, or duplicate model arrays")
    bounds = binding.get("registered_model_bounds")
    if not isinstance(bounds, dict) or set(bounds) != {name.rsplit(".", 1)[0] for name in model_names}: raise ContextError("trusted registry model-bound labels mismatch")
    for family, bound in bounds.items():
        if not isinstance(bound, dict): raise ContextError(f"trusted registry bound is not an object: {family}")
        prefix = str(family)
        shape_map = {item["name"]: item["shape"] for item in arrays if item["name"].rsplit(".", 1)[0] == prefix}
        if shape_map.get(f"{prefix}.W") != bound.get("W_shape") or shape_map.get(f"{prefix}.a_src") != bound.get("a_src_shape") or shape_map.get(f"{prefix}.a_dst") != bound.get("a_dst_shape"): raise ContextError(f"trusted registry bound shape mismatch: {family}")
        if f"{prefix}.b_out" in shape_map and shape_map[f"{prefix}.b_out"] != bound.get("b_shape"): raise ContextError(f"trusted registry output bias shape mismatch: {family}")
    entry = entries["com_W"]
    if set(entry) != {"backend", "parameter_id", "label", "vector_length", "commitment", "padded_length"}:
        raise ContextError("trusted registry com_W entry contains private or unsupported fields")
    if not isinstance(entry, dict) or str(entry.get("label", "com_W")) != "com_W" or int(entry.get("vector_length", -1)) != int(binding.get("N_in", -2)) or int(entry.get("padded_length", -1)) != int(entry.get("vector_length", -2)): raise ContextError("trusted registry com_W entry mismatch")
    if entry.get("backend") not in {None, _HYRAX_BACKEND}: raise ContextError("trusted registry com_W backend mismatch")
    return registry


def _fixed_lrelu(value: int) -> int:
    return int(value) if int(value) >= 0 else -((-int(value) + 2) // 5)


def _ledger(
    graph: LocalGraph,
    manifest: Mapping[str, Any],
    bounds: Mapping[str, Any],
    quant_cfg_id: str,
    rounding_mode: str,
    range_bits: int = 12,
) -> Dict[str, Any]:
    """Rebuild the exact producer ledger from external public inputs."""
    mode = validate_rounding_mode(rounding_mode)
    hidden, classes, k_out, _ = _dimensions(manifest)
    hidden_profile = [
        {"layer_index": int(item["layer_index"]), "head_count": int(item["head_count"]), "head_dim": int(item["head_dim"])}
        for item in hidden
    ]
    return build_integer_bound_ledger(
        mode=mode,
        quant_cfg_id=quant_cfg_id,
        authenticated_quant_scales=dict(QUANT_SCALE_PROFILES[quant_cfg_id]),
        registered_model_bounds=bounds,
        hidden_profile=hidden_profile,
        d_in=[int(manifest["d_in_profile"][0])] + [int(item["head_count"]) * int(item["head_dim"]) for item in hidden],
        classes=int(classes),
        k_out=int(k_out),
        max_degree=max((int(v) for v in np.bincount(graph.dst, minlength=graph.num_nodes)), default=1),
        feature_bound=max((abs(_round_feature(float(v), QUANT_SCALE_PROFILES[quant_cfg_id]["input"])) for v in np.ravel(graph.features)), default=0),
        node_count=int(graph.num_nodes),
        range_bits=int(range_bits),
    )


def _shape(name: str, shape: Sequence[int]) -> Tuple[str, Sequence[int]]: return name, [int(v) for v in shape]


def _division_names(prefix: str, shape: Sequence[int]) -> List[Tuple[str, Sequence[int]]]:
    return [_shape(f"{prefix}.{name}", shape) for name in ("x", "d", "y", "sigma", "z", "a", "a_inv", "d_inv", "q", "r", "delta")]


def _division_gkr_names(prefix: str, shape: Sequence[int]) -> List[Tuple[str, Sequence[int]]]:
    return [_shape(f"{prefix}.gkr.{name}", shape) for name in (
        "sigma_a", "sigma_q", "a_a_inv", "d_d_inv", "d_q", "z_a", "z_sigma", "z_a_inv",
    )]


def _head_aux_names(tag: str, n: int, e: int, width: int, *, output: bool) -> List[Tuple[str, Sequence[int]]]:
    out = [
        _shape(f"{tag}.H_prime", [n, width]),
        # E_src/E_dst are node-wise attention projections.  Edge routing
        # selects their entries through public src/dst maps; they are not
        # edge-length witness arrays even when E happens to equal N.
        _shape(f"{tag}.E_src", [n]),
        _shape(f"{tag}.E_dst", [n]),
        _shape(f"{tag}.S", [e]),
        _shape(f"{tag}.Z", [e]),
        _shape(f"{tag}.M", [n]),
        _shape(f"{tag}.Delta", [e]),
        _shape(f"{tag}.U", [e]),
        _shape(f"{tag}.Sum", [n]),
        _shape(f"{tag}.Inv", [n]),
        _shape(f"{tag}.Alpha", [e]),
        _shape(f"{tag}.H_agg_pre", [n, width]),
        _shape(f"{tag}.H_agg", [n, width]),
        _shape(f"{tag}.H_or_Y_agg_num", [n, width]),
        _shape(f"{tag}.SignedRoundDiv.output", [n, width]),
        _shape(f"{tag}.field_aggregate", [n, width]),
        _shape(f"{tag}.SignedRoundDiv.denominator", [n]),
        _shape(f"{tag}.Q_grp", [e]),
        _shape(f"{tag}.group_count", [n]),
        _shape(f"{tag}.s_max", [e]),
        _shape(f"{tag}.R_U", [e]),
        _shape(f"{tag}.R_num", [e, width]),
        _shape(f"{tag}.R_field", [e, width]),
    ]
    out.extend(_division_names(f"{tag}.SignedRoundDiv.aggregate", [n, width]))
    if output:
        out.extend(_division_names(f"{tag}.SignedRoundDiv.scale_alignment", [n, width]))
        out.append(_shape(f"{tag}.Y_pre_aligned", [n, width]))
    out.extend(_shape(f"{tag}.gkr.{name}", shape) for name, shape in (
        ("sum_inv", [n]), ("s_max_delta", [e]), ("double_aggregate", [n, width]),
    ))
    out.extend(_division_gkr_names(f"{tag}.SignedRoundDiv.aggregate", [n, width]))
    if output:
        out.extend(_division_gkr_names(f"{tag}.SignedRoundDiv.scale_alignment", [n, width]))
    out.extend(_shape(f"{tag}.gkr.{name}", shape) for name, shape in (
        ("one_minus_Q", [e]),
        ("R_U_prev", [e]),
        ("R_U_carry", [e]),
        ("R_smax", [e]),
        ("R_smax_prev", [e]),
        ("R_smax_carry", [e]),
        ("R_num_prev", [e, width]),
        ("R_num_carry", [e, width]),
        ("U_H", [e, width]),
        ("R_field_prev", [e, width]),
        ("R_field_carry", [e, width]),
        ("Alpha_H", [e, width]),
    ))
    return out


def _aux_arrays(
    graph: LocalGraph,
    manifest: Mapping[str, Any],
    range_bits: int | None = None,
    ledger: Mapping[str, Any] | None = None,
) -> List[Tuple[str, Sequence[int]]]:
    """Rebuild the exact current ``com_Aux`` array order and shapes.

    This is deliberately derived from the public architecture and graph only.
    The order mirrors ``proof._segment_arrays``: trace wires, native-GKR
    product advice, sparse running-state advice, matrix residuals, and the
    final-average wires.  In particular, ``R_smax`` is a committed wire and
    must not be omitted from the external Layout_in reconstruction.
    """
    hidden, classes, k_out, _ = _dimensions(manifest)
    n, e = int(graph.num_nodes), int(graph.src.size)
    out: List[Tuple[str, Sequence[int]]] = []
    for layer in hidden:
        layer_index = int(layer["layer_index"]) + 1
        width = int(layer["head_dim"])
        out.append(_shape(f"hidden.l{layer_index}.concat", [n, int(layer["head_count"]) * width]))
        for index in range(int(layer["head_count"])):
            out.extend(_head_aux_names(f"hidden.l{layer_index}.h{index}", n, e, width, output=False))
    for index in range(int(k_out)):
        out.extend(_head_aux_names(f"output.h{index}", n, e, int(classes), output=True))

    # The matrix residuals and their range-advice arrays have public cardinality
    # determined by the graph/architecture.  The native proof's residual wire
    # is always present even when the witness values happen to be zero.
    previous = int(hidden[-1]["head_count"]) * int(hidden[-1]["head_dim"])
    matrix_items: List[Tuple[str, Sequence[int]]] = []
    for layer in hidden:
        width = int(layer["head_dim"])
        for index in range(int(layer["head_count"])):
            tag = f"hidden.l{int(layer['layer_index']) + 1}.h{index}"
            matrix_items.extend((
                _shape(f"{tag}.projection.rescale_residual", [n, width]),
                _shape(f"{tag}.attention_src.rescale_residual", [n]),
                _shape(f"{tag}.attention_dst.rescale_residual", [n]),
            ))
    for index in range(int(k_out)):
        tag = f"output.h{index}"
        matrix_items.extend((
            _shape(f"{tag}.projection.rescale_residual", [n, int(classes)]),
            _shape(f"{tag}.attention_src.rescale_residual", [n]),
            _shape(f"{tag}.attention_dst.rescale_residual", [n]),
        ))
    out.extend(matrix_items)
    residual_count = sum(int(np.prod(shape, dtype=np.int64)) for _, shape in matrix_items)
    out.extend(_shape(f"rescale_rounding_range.gkr.{name}", [residual_count]) for name in ("value", "sign", "abs", "sign_abs"))

    final_shape = [n, int(classes)]
    out.extend(_division_names("output.final_average", final_shape))
    out.extend(_division_gkr_names("output.final_average", final_shape))
    out.extend([_shape("output.Y_sum", final_shape), _shape("output.Y", final_shape)])
    if range_bits is not None and ledger is not None and "bit_width" in ledger:
        lasso_families = _families(graph, manifest, int(range_bits), ledger)
        for family in lasso_families:
            if family["kind"] == "range":
                out.append(_shape(
                    f"lasso.{family['family']}.range_bits",
                    [int(family["padded_count"]), int(family["range_bits"])],
                ))
    return out


def _logical(name: str, arrays: Sequence[Tuple[str, Sequence[int]]]) -> Dict[str, Any]:
    offset, descriptors = 0, []
    for array_name, shape in arrays:
        shape = [int(v) for v in shape]; length = int(np.prod(shape, dtype=np.int64)); descriptors.append({"name": array_name, "offset": offset, "length": length, "shape": shape, "encoding": "canonical_bn254_fr_integer"}); offset += length
    return {"name": name, "logical_length": offset, "arrays": descriptors}


def _families(graph: LocalGraph, manifest: Mapping[str, Any], bits: int, ledger: Mapping[str, Any]) -> List[Dict[str, Any]]:
    hidden, classes, k_out, _ = _dimensions(manifest); n, e = int(graph.num_nodes), int(graph.src.size); families: List[Dict[str, Any]] = []; offset = 0
    policy_map = ledger.get("range_family_policy", {}).get("families", {})
    def add(label: str, kind: str, count: int, table: str, signed: bool = False) -> None:
        nonlocal offset
        policy = policy_map.get(label) if kind == "range" else None
        if kind == "range":
            if not isinstance(policy, Mapping):
                raise ContextError(f"missing canonical range-family policy: {label}")
            if int(policy.get("range_bits", -1)) != int(bits):
                raise ContextError(f"range-family bits differ from authenticated ledger: {label}")
            limbs = int(policy.get("limb_count", 0))
            bit_width = int(policy.get("bit_width", 0))
            bound = int(policy.get("bound", -1))
            if limbs <= 0 or bit_width <= 0 or bound < 0:
                raise ContextError(f"invalid canonical range-family policy: {label}")
        else:
            limbs = 1; bit_width = 0; bound = 0
        real = int(count) if kind == "function" else int(count) * limbs
        # Keep the external descriptor in lockstep with the native prover:
        # committed Sumcheck requires at least one round even for a one-query
        # synthetic family.
        padded = next_power_of_two(max(2, real))
        item = {"family": label, "kind": kind, "table_id": table, "offset": offset, "real_count": real, "padded_count": padded, "shape": [padded]}
        if kind == "range": item.update({"range_bits": int(bits), "bit_width": bit_width, "limbs_per_value": limbs, "bound": bound, "signed_magnitude": bool(signed), "a_pad": "0", "Q_pad": {"zero_prefix": real, "one_suffix": padded - real}})
        else: item.update({"a_pad": "0", "Q_pad": {"zero_prefix": real, "one_suffix": padded - real}})
        families.append(item); offset += padded
    for layer in hidden:
        for index in range(layer["head_count"]):
            tag, d = f"hidden.l{layer['layer_index'] + 1}.h{index}", layer["head_dim"]; add(f"{tag}.range_delta", "range", e, "T_range"); add(f"{tag}.LeakyReLU", "function", e, "T_LReLU"); add(f"{tag}.ExpMap", "function", e, "T_exp"); add(f"{tag}.ELU", "function", n * d, "T_ELU"); add(f"{tag}.SignedRoundDiv.aggregate.range", "range", 5 * n * d, "T_range")
    for index in range(k_out):
        tag = f"output.h{index}"; add(f"{tag}.range_delta", "range", e, "T_range"); add(f"{tag}.LeakyReLU", "function", e, "T_LReLU"); add(f"{tag}.ExpMap", "function", e, "T_exp"); add(f"{tag}.SignedRoundDiv.aggregate.range", "range", 5 * n * classes, "T_range"); add(f"{tag}.SignedRoundDiv.scale_alignment.range", "range", 5 * n * classes, "T_range")
    add("output.final_average.SignedRoundDiv.range", "range", 5 * n * classes, "T_range")
    residual_values = sum(layer["head_count"] * (n * layer["head_dim"] + 2 * n) for layer in hidden) + k_out * (n * classes + 2 * n)
    add("rescale_rounding_range", "range", residual_values, "T_range", signed=True)
    return families


def _matrix_families(graph: LocalGraph, manifest: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Rebuild the exact ``MatMulSpec.to_dict`` wire descriptors.

    Native MatMul does not serialize a separate top-level ``kind`` field.  In
    particular, matrix-vector claims are ordinary three-dimensional specs with
    ``m == 1`` (``[n, k, 1]``).  Keeping these descriptors in the external
    context lets the CLI reject a claim before the native verifier is called,
    while still comparing exactly the same object as ``verify_matmul``.
    """
    hidden, classes, k_out, _ = _dimensions(manifest); n = int(graph.num_nodes); out: List[Dict[str, Any]] = []

    def spec(label: str, shape: Sequence[int], left: Tuple[str, str], right: Tuple[str, str], output: Tuple[str, str], residual: Tuple[str, str]) -> Dict[str, Any]:
        return {
            "label": label,
            "shape": [int(value) for value in shape],
            "left": {"segment": left[0], "array_name": left[1]},
            "right": {"segment": right[0], "array_name": right[1]},
            "output": {"segment": output[0], "array_name": output[1]},
            "residual": {"segment": residual[0], "array_name": residual[1]},
            "output_scale": 1,
            "residual_semantics": "exact_zero",
        }

    for layer in hidden:
        for index in range(layer["head_count"]):
            tag = f"hidden.l{layer['layer_index'] + 1}.h{index}"
            layer_index = int(layer["layer_index"]) + 1
            left = ("X_pub", "X_pub.H_input") if layer["layer_index"] == 0 else ("Aux", f"hidden.l{layer_index - 1}.concat")
            projection = f"{tag}.projection"
            out.append({"label": projection, "spec": spec(projection, [n, layer["input_dim"], layer["head_dim"]], left, ("W", f"{tag}.W"), ("Aux", f"{tag}.H_prime"), ("Aux", f"{projection}.rescale_residual"))})
            for direction in ("src", "dst"):
                label = f"{tag}.attention_{direction}"
                out.append({"label": label, "spec": spec(label, [n, layer["head_dim"], 1], ("Aux", f"{tag}.H_prime"), ("W", f"{tag}.a_{direction}"), ("Aux", f"{tag}.E_{direction}"), ("Aux", f"{label}.rescale_residual"))})
    previous = hidden[-1]["head_count"] * hidden[-1]["head_dim"]
    for index in range(k_out):
        tag = f"output.h{index}"; projection = f"{tag}.projection"
        out.append({"label": projection, "spec": spec(projection, [n, previous, classes], ("Aux", f"hidden.l{len(hidden)}.concat"), ("W", f"{tag}.W"), ("Aux", f"{tag}.H_prime"), ("Aux", f"{projection}.rescale_residual"))})
        for direction in ("src", "dst"):
            label = f"{tag}.attention_{direction}"
            out.append({"label": label, "spec": spec(label, [n, classes, 1], ("Aux", f"{tag}.H_prime"), ("W", f"{tag}.a_{direction}"), ("Aux", f"{tag}.E_{direction}"), ("Aux", f"{label}.rescale_residual"))})
    return out


def _gkr_record(kind: str, spec: LinearGateSpec | ProductGateSpec) -> Dict[str, Any]:
    """Serialize the same nested native-gkr descriptor used by the wire."""
    return {"kind": kind, "spec": spec.descriptor()}


def _native_division_records(prefix: str, count: int, rounding_mode: str) -> List[Dict[str, Any]]:
    rounding_mode = validate_rounding_mode(rounding_mode)
    aux = lambda name: identity_oracle("Aux", f"{prefix}.{name}", count)
    product = lambda name: identity_oracle("Aux", f"{prefix}.gkr.{name}", count)
    zero = (0,) * count
    one = (1,) * count
    inv_two = pow(2, BN254_FR_MODULUS - 2, BN254_FR_MODULUS)
    if rounding_mode == "nearest_ties_away":
        equation = ("nearest_equation", aux("a"), (LinearTerm(1, product("d_q")), LinearTerm(inv_two, aux("r")), LinearTerm(-inv_two, aux("d"))), (0,) * count)
        slack_terms = ((LinearTerm(2, aux("d")), LinearTerm(-1, aux("r"))), (-1,) * count)
    else:
        equation = ("toward_zero_equation", aux("a"), (LinearTerm(1, product("d_q")), LinearTerm(1, aux("r"))), (0,) * count)
        slack_terms = ((LinearTerm(1, aux("d")), LinearTerm(-1, aux("r"))), (-1,) * count)
    specs: List[Tuple[str, LinearGateSpec | ProductGateSpec]] = [
        ("product", ProductGateSpec(f"{prefix}.sigma_boolean", aux("sigma"), aux("sigma"), aux("sigma"))),
        ("product", ProductGateSpec(f"{prefix}.z_boolean", aux("z"), aux("z"), aux("z"))),
        ("product", ProductGateSpec(f"{prefix}.sigma_a", product("sigma_a"), aux("sigma"), aux("a"))),
        ("product", ProductGateSpec(f"{prefix}.sigma_q", product("sigma_q"), aux("sigma"), aux("q"))),
        ("product", ProductGateSpec(f"{prefix}.a_a_inv", product("a_a_inv"), aux("a"), aux("a_inv"))),
        ("product", ProductGateSpec(f"{prefix}.d_d_inv", product("d_d_inv"), aux("d"), aux("d_inv"))),
        ("product", ProductGateSpec(f"{prefix}.d_q", product("d_q"), aux("d"), aux("q"))),
        ("product", ProductGateSpec(f"{prefix}.z_a", product("z_a"), aux("z"), aux("a"))),
        ("product", ProductGateSpec(f"{prefix}.z_sigma", product("z_sigma"), aux("z"), aux("sigma"))),
        ("product", ProductGateSpec(f"{prefix}.z_a_inv", product("z_a_inv"), aux("z"), aux("a_inv"))),
        ("linear", LinearGateSpec(f"{prefix}.signed_input", aux("x"), (LinearTerm(1, aux("a")), LinearTerm(-2, product("sigma_a"))), zero)),
        ("linear", LinearGateSpec(f"{prefix}.signed_output", aux("y"), (LinearTerm(1, aux("q")), LinearTerm(-2, product("sigma_q"))), zero)),
        ("linear", LinearGateSpec(f"{prefix}.a_inverse", product("a_a_inv"), (LinearTerm(-1, aux("z")),), one)),
        ("linear", LinearGateSpec(f"{prefix}.d_inverse", product("d_d_inv"), (), one)),
        ("linear", LinearGateSpec(f"{prefix}.z_a_zero", product("z_a"), (), zero)),
        ("linear", LinearGateSpec(f"{prefix}.z_sigma_zero", product("z_sigma"), (), zero)),
        ("linear", LinearGateSpec(f"{prefix}.z_a_inv_zero", product("z_a_inv"), (), zero)),
        ("linear", LinearGateSpec(f"{prefix}.{equation[0]}", equation[1], equation[2], equation[3])),
        ("linear", LinearGateSpec(f"{prefix}.rounding_slack", aux("delta"), slack_terms[0], slack_terms[1])),
    ]
    return [_gkr_record(kind, spec) for kind, spec in specs]


def _function_tuple_encoding(table_id: str, ledger: Mapping[str, Any]) -> Tuple[int, int, int, int]:
    domain_name = {"T_LReLU": "lrelu", "T_ELU": "elu", "T_exp": "exp"}[table_id]
    domain = ledger.get("table_domains", {}).get(domain_name)
    if not isinstance(domain, (list, tuple)) or len(domain) != 2:
        raise ContextError(f"missing certified domain for {table_id}")
    domain_min, domain_max = int(domain[0]), int(domain[1])
    def evaluate(value: int) -> int:
        if table_id == "T_LReLU":
            return _fixed_lrelu(value)
        if table_id == "T_ELU":
            return int(_elu_table_value(value))
        return int(_exp_map_fixed(value))
    endpoint_values = [evaluate(value) for value in (domain_min, domain_max, 0) if domain_min <= value <= domain_max]
    x_bound = max(abs(domain_min), abs(domain_max))
    y_bound = max((abs(value) for value in endpoint_values), default=0)
    base = 2 * y_bound + 1
    pad = (domain_min + x_bound) * base + (evaluate(domain_min) + y_bound)
    return base, x_bound, y_bound, pad


def _native_lasso_consistency_records(
    graph: LocalGraph,
    manifest: Mapping[str, Any],
    families: Sequence[Mapping[str, Any]],
    ledger: Mapping[str, Any],
    range_bits: int,
) -> List[Dict[str, Any]]:
    aux_arrays = dict((name, int(np.prod(shape, dtype=np.int64))) for name, shape in _aux_arrays(graph, manifest, range_bits, ledger))
    records: List[Dict[str, Any]] = []
    for family in families:
        label = str(family["family"])
        real_count, padded_count = int(family["real_count"]), int(family["padded_count"])
        lasso_real = OracleRef("L_qry", label, tuple(range(real_count)))
        if family["kind"] == "function":
            table_id = str(family["table_id"])
            base, x_bound, y_bound, pad = _function_tuple_encoding(table_id, ledger)
            if label.endswith(".LeakyReLU"):
                prefix = label.removesuffix(".LeakyReLU")
                x_name, y_name = f"{prefix}.S", f"{prefix}.Z"
            elif label.endswith(".ExpMap"):
                prefix = label.removesuffix(".ExpMap")
                x_name, y_name = f"{prefix}.Delta", f"{prefix}.U"
            elif label.endswith(".ELU"):
                prefix = label.removesuffix(".ELU")
                x_name, y_name = f"{prefix}.SignedRoundDiv.output", f"{prefix}.H_agg"
            else:
                raise ContextError(f"unknown formal function family: {label}")
            if aux_arrays.get(x_name) != real_count or aux_arrays.get(y_name) != real_count:
                raise ContextError(f"function lookup Aux/L cardinality mismatch: {label}")
            records.append(_gkr_record("linear", LinearGateSpec(
                f"lasso/{label}/packed_query", lasso_real,
                (LinearTerm(base, identity_oracle("Aux", x_name, real_count)), LinearTerm(1, identity_oracle("Aux", y_name, real_count))),
                (x_bound * base + y_bound,) * real_count,
            )))
            if padded_count > real_count:
                records.append(_gkr_record("linear", LinearGateSpec(
                    f"lasso/{label}/padding", OracleRef("L_qry", label, tuple(range(real_count, padded_count))), (), (pad,) * (padded_count - real_count),
                )))
            continue

        policy = ledger.get("range_family_policy", {}).get("families", {}).get(label)
        if not isinstance(policy, Mapping):
            raise ContextError(f"range family is absent from authenticated ledger: {label}")
        if any(int(family.get(key, -1)) != int(policy.get(source, -2)) for key, source in (("range_bits", "range_bits"), ("bit_width", "bit_width"), ("limbs_per_value", "limb_count"), ("bound", "bound"))):
            raise ContextError(f"range family geometry differs from authenticated ledger: {label}")
        limb_count = int(family["limbs_per_value"])
        base = 1 << int(family["range_bits"])
        bit_name = f"lasso.{label}.range_bits"
        bit_count = padded_count * int(family["range_bits"])
        records.append(_gkr_record("product", ProductGateSpec(
            f"lasso/{label}/range-bit-boolean", identity_oracle("Aux", bit_name, bit_count), identity_oracle("Aux", bit_name, bit_count), identity_oracle("Aux", bit_name, bit_count),
        )))
        records.append(_gkr_record("linear", LinearGateSpec(
            f"lasso/{label}/range-limb-reconstruction", OracleRef("L_qry", label, tuple(range(padded_count))),
            tuple(LinearTerm(1 << bit, OracleRef("Aux", bit_name, tuple(index * int(family["range_bits"]) + bit for index in range(padded_count)))) for bit in range(int(family["range_bits"]))),
            (0,) * padded_count,
        )))
        if label == "rescale_rounding_range":
            sources = [("rescale_rounding_range.gkr.abs", aux_arrays["rescale_rounding_range.gkr.abs"])]
        elif label.endswith(".range_delta"):
            prefix = label.removesuffix(".range_delta")
            sources = [(f"{prefix}.Delta", aux_arrays[f"{prefix}.Delta"])]
        elif label.endswith(".range"):
            prefix = label.removesuffix(".range")
            if prefix == "output.final_average.SignedRoundDiv":
                prefix = "output.final_average"
            sources = [(f"{prefix}.{name}", aux_arrays[f"{prefix}.{name}"]) for name in ("a", "q", "r", "delta", "d")]
        else:
            raise ContextError(f"unknown formal range family: {label}")
        cursor = 0
        for source_name, count in sources:
            terms = tuple(LinearTerm(
                pow(base, limb, BN254_FR_MODULUS),
                OracleRef("L_qry", label, tuple(cursor + index * limb_count + limb for index in range(count))),
            ) for limb in range(limb_count))
            records.append(_gkr_record("linear", LinearGateSpec(
                f"lasso/{label}/reconstruct/{source_name}", identity_oracle("Aux", source_name, count), terms, (0,) * count,
            )))
            cursor += count * limb_count
        if cursor != real_count:
            raise ContextError(f"range lookup limb layout differs from certified family: {label}")
        if padded_count > real_count:
            records.append(_gkr_record("linear", LinearGateSpec(
                f"lasso/{label}/padding", OracleRef("L_qry", label, tuple(range(real_count, padded_count))), (), (0,) * (padded_count - real_count),
            )))
    return records


def _gkr_families(
    graph: LocalGraph,
    manifest: Mapping[str, Any],
    public_y: Sequence[int] | None = None,
    lasso_families: Sequence[Mapping[str, Any]] | None = None,
    ledger: Mapping[str, Any] | None = None,
    range_bits: int = 12,
) -> List[Dict[str, Any]]:
    """Rebuild the exact ordered native GKR gate/spec descriptors.

    The old context exposed label/shape summaries, which allowed a proof to
    select a different oracle wiring underneath an otherwise matching label.
    The current native wire carries a nested ``proof.spec`` descriptor; this
    function reconstructs that descriptor from public graph/model data and
    compares it byte-for-byte before the payload verifier runs.
    """
    hidden, classes, k_out, _ = _dimensions(manifest)
    rounding_mode = validate_rounding_mode((ledger or {}).get("rounding_mode"))
    n, e = int(graph.num_nodes), int(graph.src.size)
    src = tuple(int(value) for value in graph.src)
    dst = tuple(int(value) for value in graph.dst)
    groups: List[List[int]] = [[] for _ in range(n)]
    for index, target in enumerate(dst):
        if target < 0 or target >= n:
            raise ContextError("public destination leaves the canonical node domain")
        groups[target].append(index)
    if any(not group for group in groups):
        raise ContextError("formal GAT graph must contain a canonical self-loop group for every node")
    group_starts = {group[0] for group in groups}
    expected_q = tuple(1 if index in group_starts else 0 for index in range(e))
    expected_count = tuple(len(group) for group in groups)
    group_ends = tuple(group[-1] for group in groups)
    records: List[Dict[str, Any]] = []

    def add_head(prefix: str, width: int, *, output: bool, h_prime_scale: int) -> None:
        aux = lambda name, mapping: OracleRef("Aux", f"{prefix}.{name}", tuple(int(value) for value in mapping))
        identity_edges, identity_nodes = tuple(range(e)), tuple(range(n))
        records.extend([
            _gkr_record("linear", LinearGateSpec(f"{prefix}.Q_grp_public_boundary", aux("Q_grp", identity_edges), (), expected_q)),
            _gkr_record("linear", LinearGateSpec(f"{prefix}.group_count_public", aux("group_count", identity_nodes), (), expected_count)),
            _gkr_record("linear", LinearGateSpec(f"{prefix}.score_routing", aux("S", identity_edges), (LinearTerm(1, aux("E_src", src)), LinearTerm(1, aux("E_dst", dst))), (0,) * e)),
            _gkr_record("linear", LinearGateSpec(f"{prefix}.max_shift", aux("Delta", identity_edges), (LinearTerm(1, aux("M", dst)), LinearTerm(-1, aux("Z", identity_edges))), (0,) * e)),
            _gkr_record("product", ProductGateSpec(f"{prefix}.alpha_product", aux("Alpha", identity_edges), aux("U", identity_edges), aux("Inv", dst))),
            _gkr_record("product", ProductGateSpec(f"{prefix}.sum_inverse_product", aux("gkr.sum_inv", identity_nodes), aux("Sum", identity_nodes), aux("Inv", identity_nodes))),
            _gkr_record("linear", LinearGateSpec(f"{prefix}.sum_inverse_one", aux("gkr.sum_inv", identity_nodes), (), (1,) * n)),
            _gkr_record("product", ProductGateSpec(f"{prefix}.Q_grp_boolean", aux("Q_grp", identity_edges), aux("Q_grp", identity_edges), aux("Q_grp", identity_edges))),
            _gkr_record("product", ProductGateSpec(f"{prefix}.s_max_boolean", aux("s_max", identity_edges), aux("s_max", identity_edges), aux("s_max", identity_edges))),
            _gkr_record("product", ProductGateSpec(f"{prefix}.s_max_delta_product", aux("gkr.s_max_delta", identity_edges), aux("s_max", identity_edges), aux("Delta", identity_edges))),
            _gkr_record("linear", LinearGateSpec(f"{prefix}.s_max_selected", aux("gkr.s_max_delta", identity_edges), (), (0,) * e)),
        ])
        repeated_edges = tuple(edge for edge in range(e) for _ in range(width))
        h_edge_map = tuple(src[edge] * width + column for edge in range(e) for column in range(width))
        flat_count = e * width
        previous_edge_map = tuple(-1 if edge in group_starts else edge - 1 for edge in range(e))
        previous_flat_map = tuple(-1 if edge in group_starts else (edge - 1) * width + column for edge in range(e) for column in range(width))
        records.extend([
            _gkr_record("linear", LinearGateSpec(f"{prefix}.one_minus_Q", aux("gkr.one_minus_Q", identity_edges), (LinearTerm(-1, aux("Q_grp", identity_edges)),), (1,) * e)),
            _gkr_record("linear", LinearGateSpec(f"{prefix}.R_U_prev_binding", aux("gkr.R_U_prev", identity_edges), (LinearTerm(1, aux("R_U", previous_edge_map)),), (0,) * e)),
            _gkr_record("product", ProductGateSpec(f"{prefix}.R_U_carry_product", aux("gkr.R_U_carry", identity_edges), aux("gkr.one_minus_Q", identity_edges), aux("gkr.R_U_prev", identity_edges))),
            _gkr_record("linear", LinearGateSpec(f"{prefix}.R_U_recurrence", aux("R_U", identity_edges), (LinearTerm(1, aux("gkr.R_U_carry", identity_edges)), LinearTerm(1, aux("U", identity_edges))), (0,) * e)),
            _gkr_record("linear", LinearGateSpec(f"{prefix}.Sum_terminal", aux("Sum", identity_nodes), (LinearTerm(1, aux("R_U", group_ends)),), (0,) * n)),
            _gkr_record("linear", LinearGateSpec(f"{prefix}.R_smax_prev_binding", aux("gkr.R_smax_prev", identity_edges), (LinearTerm(1, aux("gkr.R_smax", previous_edge_map)),), (0,) * e)),
            _gkr_record("product", ProductGateSpec(f"{prefix}.R_smax_carry_product", aux("gkr.R_smax_carry", identity_edges), aux("gkr.one_minus_Q", identity_edges), aux("gkr.R_smax_prev", identity_edges))),
            _gkr_record("linear", LinearGateSpec(f"{prefix}.R_smax_recurrence", aux("gkr.R_smax", identity_edges), (LinearTerm(1, aux("gkr.R_smax_carry", identity_edges)), LinearTerm(1, aux("s_max", identity_edges))), (0,) * e)),
            _gkr_record("linear", LinearGateSpec(f"{prefix}.s_max_one_per_group", aux("gkr.sum_inv", identity_nodes), (LinearTerm(1, aux("gkr.R_smax", group_ends)),), (0,) * n)),
            _gkr_record("linear", LinearGateSpec(f"{prefix}.R_num_prev_binding", aux("gkr.R_num_prev", tuple(range(flat_count))), (LinearTerm(1, aux("R_num", previous_flat_map)),), (0,) * flat_count)),
            _gkr_record("product", ProductGateSpec(f"{prefix}.R_num_carry_product", aux("gkr.R_num_carry", tuple(range(flat_count))), aux("gkr.one_minus_Q", repeated_edges), aux("gkr.R_num_prev", tuple(range(flat_count))))),
            _gkr_record("product", ProductGateSpec(f"{prefix}.U_H_product", aux("gkr.U_H", tuple(range(flat_count))), aux("U", repeated_edges), aux("H_prime", h_edge_map))),
            _gkr_record("linear", LinearGateSpec(f"{prefix}.R_num_recurrence", aux("R_num", tuple(range(flat_count))), (LinearTerm(1, aux("gkr.R_num_carry", tuple(range(flat_count)))), LinearTerm(1, aux("gkr.U_H", tuple(range(flat_count))))), (0,) * flat_count)),
            _gkr_record("linear", LinearGateSpec(f"{prefix}.R_field_prev_binding", aux("gkr.R_field_prev", tuple(range(flat_count))), (LinearTerm(1, aux("R_field", previous_flat_map)),), (0,) * flat_count)),
            _gkr_record("product", ProductGateSpec(f"{prefix}.R_field_carry_product", aux("gkr.R_field_carry", tuple(range(flat_count))), aux("gkr.one_minus_Q", repeated_edges), aux("gkr.R_field_prev", tuple(range(flat_count))))),
            _gkr_record("product", ProductGateSpec(f"{prefix}.Alpha_H_product", aux("gkr.Alpha_H", tuple(range(flat_count))), aux("Alpha", repeated_edges), aux("H_prime", h_edge_map))),
            _gkr_record("linear", LinearGateSpec(f"{prefix}.R_field_recurrence", aux("R_field", tuple(range(flat_count))), (LinearTerm(1, aux("gkr.R_field_carry", tuple(range(flat_count)))), LinearTerm(1, aux("gkr.Alpha_H", tuple(range(flat_count))))), (0,) * flat_count)),
        ])
        node_flat_count = n * width
        end_flat_maps = tuple(group_ends[node] * width + column for node in range(n) for column in range(width))
        records.extend([
            _gkr_record("linear", LinearGateSpec(f"{prefix}.agg_num_terminal", identity_oracle("Aux", f"{prefix}.H_or_Y_agg_num", node_flat_count), (LinearTerm(1, aux("R_num", end_flat_maps)),), (0,) * node_flat_count)),
            _gkr_record("linear", LinearGateSpec(f"{prefix}.field_aggregate_terminal", identity_oracle("Aux", f"{prefix}.field_aggregate", node_flat_count), (LinearTerm(1, aux("R_field", end_flat_maps)),), (0,) * node_flat_count)),
            _gkr_record("linear", LinearGateSpec(f"{prefix}.field_aggregate_output", identity_oracle("Aux", f"{prefix}.H_agg_pre", node_flat_count), (LinearTerm(1, identity_oracle("Aux", f"{prefix}.field_aggregate", node_flat_count)),), (0,) * node_flat_count)),
            _gkr_record("product", ProductGateSpec(
                f"{prefix}.double_aggregate_product",
                identity_oracle("Aux", f"{prefix}.gkr.double_aggregate", node_flat_count),
                aux("Sum", tuple(row for row in range(n) for _ in range(width))),
                identity_oracle("Aux", f"{prefix}.H_agg_pre", node_flat_count),
            )),
            _gkr_record("linear", LinearGateSpec(
                f"{prefix}.double_aggregate_equality",
                identity_oracle("Aux", f"{prefix}.H_or_Y_agg_num", node_flat_count),
                (LinearTerm(1, identity_oracle("Aux", f"{prefix}.gkr.double_aggregate", node_flat_count)),),
                (0,) * node_flat_count,
            )),
        ])
        aggregate_prefix = f"{prefix}.SignedRoundDiv.aggregate"
        aggregate_count = n * width
        repeated_nodes = tuple(row for row in range(n) for _ in range(width))
        records.extend([
            _gkr_record("linear", LinearGateSpec(f"{aggregate_prefix}.numerator_wiring", identity_oracle("Aux", f"{aggregate_prefix}.x", aggregate_count), (LinearTerm(16, identity_oracle("Aux", f"{prefix}.H_or_Y_agg_num", aggregate_count)),), (0,) * aggregate_count)),
            _gkr_record("linear", LinearGateSpec(f"{aggregate_prefix}.denominator_wiring", identity_oracle("Aux", f"{aggregate_prefix}.d", aggregate_count), (LinearTerm(h_prime_scale, aux("Sum", repeated_nodes)),), (0,) * aggregate_count)),
            _gkr_record("linear", LinearGateSpec(f"{aggregate_prefix}.output_wiring", identity_oracle("Aux", f"{prefix}.SignedRoundDiv.output", aggregate_count), (LinearTerm(1, identity_oracle("Aux", f"{aggregate_prefix}.y", aggregate_count)),), (0,) * aggregate_count)),
        ])
        records.extend(_native_division_records(aggregate_prefix, aggregate_count, rounding_mode))
        if output:
            alignment_prefix = f"{prefix}.SignedRoundDiv.scale_alignment"
            alignment_count = n * width
            records.extend(_native_division_records(alignment_prefix, alignment_count, rounding_mode))
            records.extend([
                _gkr_record("linear", LinearGateSpec(f"{alignment_prefix}.numerator_wiring", identity_oracle("Aux", f"{alignment_prefix}.x", alignment_count), (LinearTerm(16, identity_oracle("Aux", f"{prefix}.SignedRoundDiv.output", alignment_count)),), (0,) * alignment_count)),
                _gkr_record("linear", LinearGateSpec(f"{alignment_prefix}.denominator_wiring", identity_oracle("Aux", f"{alignment_prefix}.d", alignment_count), (), (16,) * alignment_count)),
                _gkr_record("linear", LinearGateSpec(f"{alignment_prefix}.output_wiring", identity_oracle("Aux", f"{prefix}.Y_pre_aligned", alignment_count), (LinearTerm(1, identity_oracle("Aux", f"{alignment_prefix}.y", alignment_count)),), (0,) * alignment_count)),
            ])
            value_count = n * width
            bias_map = tuple(index % width for index in range(value_count))
            records.append(_gkr_record("linear", LinearGateSpec(f"{prefix}.bias_add", identity_oracle("Aux", f"{prefix}.H_agg", value_count), (LinearTerm(1, identity_oracle("Aux", f"{prefix}.Y_pre_aligned", value_count)), LinearTerm(1, OracleRef("W", f"{prefix}.b_out", bias_map))), (0,) * value_count)))

    for layer in hidden:
        position, width = int(layer["layer_index"]) + 1, int(layer["head_dim"])
        for index in range(int(layer["head_count"])):
            add_head(f"hidden.l{position}.h{index}", width, output=False, h_prime_scale=16 if position == 1 else 256)
        concat_width = int(layer["head_count"]) * width
        for head_index in range(int(layer["head_count"])):
            target_map = tuple(row * concat_width + head_index * width + column for row in range(n) for column in range(width))
            records.append(_gkr_record("linear", LinearGateSpec(f"hidden.l{position}.concat.h{head_index}", OracleRef("Aux", f"hidden.l{position}.concat", target_map), (LinearTerm(1, identity_oracle("Aux", f"hidden.l{position}.h{head_index}.H_agg", n * width)),), (0,) * (n * width))))
    for index in range(int(k_out)):
        add_head(f"output.h{index}", int(classes), output=True, h_prime_scale=256)

    final_prefix, final_count = "output.final_average", n * int(classes)
    values = tuple(to_field_int(value) for value in (public_y if public_y is not None else [0] * final_count))
    if len(values) != final_count:
        raise ContextError("external public Y shape differs from final GKR layer")
    records.extend([
        _gkr_record("linear", LinearGateSpec("output.head_sum_post_bias", identity_oracle("Aux", "output.Y_sum", final_count), tuple(LinearTerm(1, identity_oracle("Aux", f"output.h{head_index}.H_agg", final_count)) for head_index in range(int(k_out))), (0,) * final_count)),
        _gkr_record("linear", LinearGateSpec("output.final_average.numerator_wiring", identity_oracle("Aux", f"{final_prefix}.x", final_count), (LinearTerm(1, identity_oracle("Aux", "output.Y_sum", final_count)),), (0,) * final_count)),
        _gkr_record("linear", LinearGateSpec("output.final_average.denominator_wiring", identity_oracle("Aux", f"{final_prefix}.d", final_count), (), (int(k_out),) * final_count)),
    ])
    records.extend(_native_division_records(final_prefix, final_count, rounding_mode))
    records.extend([
        _gkr_record("linear", LinearGateSpec("output.final_average.public_tensor_binding", identity_oracle("Aux", "output.Y", final_count), (LinearTerm(1, identity_oracle("Aux", f"{final_prefix}.y", final_count)),), (0,) * final_count)),
        _gkr_record("linear", LinearGateSpec("output.external_public_Y", identity_oracle("Aux", "output.Y", final_count), (), values)),
    ])
    residual_items = _aux_arrays(
        graph, manifest, range_bits,
        ledger if ledger and "bit_width" in ledger else None,
    )
    residual_count = next((int(np.prod(shape, dtype=np.int64)) for name, shape in residual_items if name == "rescale_rounding_range.gkr.value"), 0)
    if residual_count <= 0:
        raise ContextError("external residual cardinality is empty")
    residual = lambda name: identity_oracle("Aux", f"rescale_rounding_range.gkr.{name}", residual_count)
    records.extend([
        _gkr_record("product", ProductGateSpec("rescale_rounding_range.sign_boolean", residual("sign"), residual("sign"), residual("sign"))),
        _gkr_record("product", ProductGateSpec("rescale_rounding_range.sign_abs_product", residual("sign_abs"), residual("sign"), residual("abs"))),
        _gkr_record("linear", LinearGateSpec("rescale_rounding_range.signed_magnitude", residual("value"), (LinearTerm(1, residual("abs")), LinearTerm(-2, residual("sign_abs"))), (0,) * residual_count)),
    ])
    for item in _matrix_families(graph, manifest):
        matrix_spec = item["spec"]
        shape = [int(value) for value in matrix_spec["shape"]]
        residual_size = int(shape[0]) * int(shape[2])
        residual_source = matrix_spec["residual"]
        residual_oracle = identity_oracle(
            str(residual_source["segment"]), str(residual_source["array_name"]), residual_size,
        )
        records.append(_gkr_record("linear", LinearGateSpec(
            f"{matrix_spec['label']}.residual_zero",
            residual_oracle,
            (),
            (0,) * residual_size,
        )))
    if lasso_families is not None and ledger is not None:
        records.extend(_native_lasso_consistency_records(graph, manifest, lasso_families, ledger, int(range_bits)))
    return records


def _dims(config: AppConfig, dataset: GraphDataset, graph: LocalGraph, manifest: Mapping[str, Any], n_in: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    hidden, classes, k_out, layers = _dimensions(manifest)
    manifest_d_in = [int(value) for value in manifest["d_in_profile"]]
    if config.d_in_profile != manifest_d_in:
        raise ContextError("formal d_in_profile does not match canonical model dimensions")
    dcat = [item["head_count"] * item["head_dim"] for item in hidden]
    dims = {
        "G_batch": graph.graph_count,
        "N": graph.num_nodes,
        "E": int(graph.src.size),
        "N_total": graph.n_total,
        "node_ptr": graph.node_ptr.tolist(),
        "edge_ptr": graph.edge_ptr.tolist(),
        "L": layers,
        # ``d_in`` is the internal circuit chain: public input, each hidden
        # concat, and the output-layer input.  The config/manifest
        # ``d_in_profile`` remains the one-entry-per-hidden-layer wire.
        "d_in": manifest_d_in + [dcat[-1]],
        "d_in_profile": manifest_d_in,
        "d_h": [item["head_dim"] for item in hidden],
        "d_cat": dcat,
        "C": classes,
        "B": int(config.range_bits),
        "K_out": k_out,
        "hidden_profile": [
            {"head_count": item["head_count"], "head_dim": item["head_dim"]}
            for item in hidden
        ],
    }
    domain = {
        "n_FH": next_power_of_two(max(dataset.num_nodes * graph.num_features, graph.num_nodes * graph.num_features)),
        "n_edge": next_power_of_two(max(int(graph.src.size), graph.num_nodes, graph.num_nodes * max([item["head_dim"] for item in hidden] + [classes]))),
        "n_in": int(n_in),
        "n_d_h": next_power_of_two(max(item["head_dim"] for item in hidden)),
        "n_cat": next_power_of_two(max(dcat)),
        "n_C": next_power_of_two(classes),
        "n_N": next_power_of_two(graph.num_nodes),
    }
    return dims, domain


@dataclass(frozen=True)
class VerifierContext:
    project_root: pathlib.Path; config_path: pathlib.Path; artifact_dir: pathlib.Path; registry_public_path: pathlib.Path; public_y_path: pathlib.Path; config: AppConfig; dataset: GraphDataset; graph: LocalGraph; model_manifest: Dict[str, Any]; registry_public: Dict[str, Any]; public_y_shape: Tuple[int, ...]; public_y_digest: str; expected_metadata: Dict[str, Any]; expected_model_arrays: Tuple[Dict[str, Any], ...]; expected_lasso_families: Tuple[Dict[str, Any], ...]; expected_matrix_families: Tuple[Dict[str, Any], ...]; expected_gkr_families: Tuple[Dict[str, Any], ...]; public_statement: Dict[str, Any]; context_digest: str; public_y_provenance: Dict[str, Any]

    def proof_context(self) -> Dict[str, Any]:
        """Return only externally reconstructed public verifier material.

        The producer-side compatibility context contains private trace-derived
        values.  The accepted replay path receives this object from the CLI,
        where every field below was rebuilt from files outside the artifact.
        """
        static_vk = static_table_vk(self.expected_metadata)
        setup = setup_context(self.expected_metadata, self.registry_public, static_vk)
        context = {
            "schema": EXTERNAL_VERIFIER_CONTEXT_SCHEMA,
            "public_statement": copy.deepcopy(self.public_statement),
            "model_registry": copy.deepcopy(self.registry_public),
            "static_table_vk": static_vk,
            "setup_context": setup,
        }
        # Recompute the envelope digest from exactly the bytes supplied to
        # the fresh verifier.  The stored dataclass value is source-derived,
        # but the returned object must remain self-authenticating if callers
        # serialize or copy it.
        context["context_digest"] = digest_json(context)
        return context

    def validate_metadata(self, metadata: Mapping[str, Any]) -> Dict[str, Any]:
        errors = [f"metadata.{key} does not match external context" for key, expected in self.expected_metadata.items() if key not in {"identities", "Layout_in"} and metadata.get(key) != expected]
        layout, binding = metadata.get("Layout_in"), self.registry_public.get("registration_binding", {})
        if not isinstance(layout, dict): errors.append("metadata.Layout_in is missing")
        else:
            if layout.get("layout_digest") != binding.get("Layout_in_digest") or metadata.get("Layout_in_digest") != binding.get("Layout_in_digest") or int(layout.get("N_in", -1)) != int(binding.get("N_in", -2)): errors.append("metadata Layout_in digest/domain mismatch")
            copy_layout = dict(layout); digest = copy_layout.pop("layout_digest", None)
            if digest != digest_json(copy_layout): errors.append("metadata Layout_in self-digest is invalid")
            segments = layout.get("segments", [])
            if [item.get("name") for item in segments] != ["X_pub", "W", "Aux", "L_qry"]: errors.append("metadata Layout_in segment order mismatch")
            else:
                public_arrays = [_shape("X_pub.H_input", [self.graph.num_nodes, self.graph.num_features]), _shape("X_pub.absolute_ids", [self.graph.num_nodes]), _shape("X_pub.dimension_scalars", [7])]
                for descriptor, arrays in zip(segments[:3], (public_arrays, [_shape(item["name"], item["shape"]) for item in self.expected_model_arrays], _aux_arrays(self.graph, self.model_manifest, self.config.range_bits, self.expected_metadata["integer_bound_ledger"]))):
                    expected = _logical(str(descriptor["name"]), arrays)
                    if descriptor.get("arrays") != expected["arrays"] or descriptor.get("logical_length") != expected["logical_length"] or int(descriptor.get("padded_length", -1)) != int(binding.get("N_in", -2)): errors.append(f"metadata Layout_in {descriptor.get('name')} array cardinality mismatch")
                actual = segments[3].get("lasso_families", []); labels = [item.get("family") for item in actual]; expected_labels = [item["family"] for item in self.expected_lasso_families]
                if labels != expected_labels or len(labels) != len(set(labels)): errors.append("metadata Lasso family labels/order/cardinality mismatch")
                else:
                    for item, expected in zip(actual, self.expected_lasso_families):
                        for field in ("family", "kind", "table_id", "offset", "real_count", "padded_count", "range_bits", "bit_width", "limbs_per_value", "bound", "signed_magnitude"):
                            if field not in item and field not in expected:
                                continue
                            if item.get(field) != expected.get(field): errors.append(f"metadata Lasso family {expected['family']} {field} mismatch")
        static = metadata.get("static_table_registry", {})
        if set(static) != {"T_LReLU", "T_ELU", "T_exp", "T_range"} or metadata.get("static_table_registry_digest") != digest_json(static): errors.append("static table registry is missing, duplicated, or self-inconsistent")
        for key, table in static.items():
            expected_domains = self.expected_metadata["integer_bound_ledger"].get("table_domains", {})
            if not isinstance(table, dict) or table.get("table_id") != key or table.get("quant_cfg_id") != self.expected_metadata["quant_cfg_id"] or table.get("rounding_mode") != self.expected_metadata["rounding_mode"] or table.get("static_table_id") != self.expected_metadata["static_table_id"] or table.get("static_table_descriptor_schema") != self.expected_metadata["static_table_descriptor_schema"] or table.get("static_table_content_digest") != self.expected_metadata["static_table_content_digest"] or table.get("static_table_descriptor_digest") != self.expected_metadata["static_table_descriptor_digest"] or table.get("encoding_id") != ENCODING_ID or table.get("range_bits") != int(self.expected_metadata["dim_cfg"]["B"]) or table.get("field_encoding_scale") != 1 or table.get("certified_table_domains") != expected_domains or table.get("integer_bound_ledger_digest") != self.expected_metadata["integer_bound_ledger_digest"]: errors.append(f"static table {key} descriptor mismatch")
        registry_metadata = metadata.get("model_registry")
        if registry_metadata != self.expected_metadata.get("model_registry") or metadata.get("model_registry_digest") != digest_json(registry_metadata): errors.append("metadata model registry is not the external trusted descriptor")
        return {"status": "VERIFY_OK" if not errors else "VERIFY_FAILED", "errors": errors, "context_digest": self.context_digest}

    def validate_proof_payload(self, metadata: Mapping[str, Any], proof_payload: Mapping[str, Any]) -> Dict[str, Any]:
        errors = []
        if metadata.get("public_output_digest") != self.public_y_digest or proof_payload.get("Y_digest") != self.public_y_digest: errors.append("public Y digest does not match independent Y")
        if proof_payload.get("M_pub_digest") != digest_json(dict(metadata)): errors.append("proof M_pub digest does not match metadata")
        if self.public_y_provenance.get("status") != "VERIFY_OK": errors.append("public Y independent-reference provenance is invalid")
        return {"status": "VERIFY_OK" if not errors else "VERIFY_FAILED", "errors": errors, "context_digest": self.context_digest}

    def validate_subproof_families(self, lasso: Mapping[str, Any] | None, matrix: Mapping[str, Any] | None, gkr: Mapping[str, Any] | None) -> Dict[str, Any]:
        errors = []
        if lasso is not None:
            actual = lasso.get("families", []); labels = [item.get("family") for item in actual]; expected = list(self.expected_lasso_families)
            if labels != [item["family"] for item in expected] or len(labels) != len(set(labels)): errors.append("Lasso proof family labels/order mismatch")
            for item, exp in zip(actual, expected):
                if item.get("table") not in {None, exp.get("table_id")} and item.get("table_id") != exp.get("table_id"): errors.append(f"Lasso proof table mismatch: {exp['family']}")
                # Both function rows and range limbs are bound to the
                # canonical L_qry layout.  A proof-selected query count (for
                # example one obtained from a private maximum limb) is not an
                # accepted formal descriptor.
                if int(item.get("query_count", -1)) != int(exp.get("padded_count", -2)): errors.append(f"Lasso proof query count mismatch: {exp['family']}")
        if matrix is not None:
            actual = matrix.get("claims", []); expected = list(self.expected_matrix_families)
            actual_labels = [item.get("spec", {}).get("label") if isinstance(item.get("spec"), Mapping) else None for item in actual]
            if actual_labels != [item["label"] for item in expected] or len(actual_labels) != len(set(actual_labels)): errors.append("MatMul family labels/order mismatch")
            for item, exp in zip(actual, expected):
                # Native verifier compares ``proof["spec"]`` to
                # MatMulSpec.to_dict() byte-for-byte at the object level.
                # Perform the same exact comparison here; do not infer a
                # private shape or accept a proof-selected descriptor.
                if not isinstance(item.get("spec"), Mapping) or dict(item["spec"]) != exp["spec"]: errors.append(f"MatMul family descriptor mismatch: {exp['label']}")
        if gkr is not None:
            actual = gkr.get("claims", []); expected = list(self.expected_gkr_families)
            actual_specs = [item.get("proof", {}).get("spec") if isinstance(item.get("proof"), Mapping) else None for item in actual]
            expected_specs = [item["spec"] for item in expected]
            actual_labels = [item.get("label") if isinstance(item, Mapping) else None for item in actual_specs if isinstance(item, Mapping)]
            if actual_specs != expected_specs or len(actual_specs) != len(set(json.dumps(item, sort_keys=True) for item in actual_specs)): errors.append("GKR family nested spec/order/uniqueness mismatch")
            for item, exp in zip(actual, expected):
                actual_spec = item.get("proof", {}).get("spec") if isinstance(item.get("proof"), Mapping) else None
                if item.get("kind") != exp.get("kind") or actual_spec != exp.get("spec"): errors.append(f"GKR family nested descriptor mismatch: {exp['spec'].get('label', '<unknown>')}")
        return {"status": "VERIFY_OK" if not errors else "VERIFY_FAILED", "errors": errors, "context_digest": self.context_digest}


def build_verifier_context(project_root: pathlib.Path, config_path: pathlib.Path, registry_public_path: pathlib.Path, artifact_dir: pathlib.Path, public_y_path: pathlib.Path) -> VerifierContext:
    root, artifact, config_path = _directory(project_root, "project-root"), _directory(artifact_dir, "artifact directory"), _file(config_path, "config")
    if _inside(config_path, artifact): raise ContextError("config must not be loaded from proof artifact")
    if not _inside(config_path, root): raise ContextError("config must be inside project-root")
    config = load_app_config(config_path, root)
    static_descriptor = validate_static_table_descriptor(root)
    if config.entries.get("static_table_descriptor_digest") != static_descriptor["descriptor_digest"]:
        raise ContextError("formal config is not bound to the canonical static-table descriptor")
    raw_manifest = _manifest(config.checkpoint_bundle)
    try:
        _validate_canonical_manifest(raw_manifest)
    except (TypeError, ValueError) as exc:
        raise ContextError(f"checkpoint manifest is not canonical: {exc}") from exc
    manifest = _canonical_model_architecture(raw_manifest)
    identities = {}
    configured_arch = str(config.entries.get("model_arch_id", ""))
    raw_arch = str(manifest.get("model_arch_id", ""))
    if not configured_arch or not raw_arch or configured_arch != raw_arch:
        raise ContextError("formal model architecture does not match the raw checkpoint manifest")
    # The public manifest supplies only the canonical architecture.  Model
    # parameter/value binding is taken from the trusted public registry below;
    # no raw tensor identity is used as a verifier input.
    for key in ("model_arch_id", "model_param_id", "quant_cfg_id", "static_table_id"):
        value = str(config.entries.get(key, ""))
        if not value:
            raise ContextError(f"formal config is missing {key}")
        identities[key] = value
    if identities["quant_cfg_id"] != "gat_l3_scale_chain":
        raise ContextError("formal verifier requires the registered GAT scale chain")
    if config.input_preprocess_id != canonical_input_preprocess_id(str(raw_manifest.get("dataset", ""))):
        raise ContextError("formal config/checkpoint input_preprocess_id mismatch")
    hidden_profile = config.hidden_profile
    manifest_hidden = _dimensions(manifest)[0]
    if [(int(item.head_count), int(item.head_dim)) for item in hidden_profile] != [(int(item["head_count"]), int(item["head_dim"])) for item in manifest_hidden]:
        raise ContextError("formal hidden_profile does not match raw checkpoint architecture")
    if config.d_in_profile and [int(value) for value in config.d_in_profile] != [int(value) for value in manifest.get("d_in_profile", [])]:
        raise ContextError("formal d_in_profile does not match raw checkpoint architecture")
    if int(config.k_out) != int(manifest.get("K_out", -1)) or int(config.num_classes) != int(manifest.get("C", -1)):
        raise ContextError("formal output cardinality does not match raw checkpoint architecture")
    dataset, graph = load_dataset(config), None
    graph = normalize_graph(dataset, config); registry = _validate_registry(registry_public_path, artifact, root, manifest, identities)
    if registry.get("registration_binding", {}).get("rounding_mode") != config.rounding_mode:
        raise ContextError("trusted registry SignedRoundDiv rounding_mode mismatch")
    y_path = _file(public_y_path, "public Y")
    if _inside(y_path, artifact): raise ContextError("public Y must be outside proof artifact")
    if not _inside(y_path, root):
        raise ContextError("public Y must be inside baseline project-root")
    expected_y_path = required_resource_paths(root, canonical_dataset_name(config.dataset))["public_y_path"].resolve()
    if y_path != expected_y_path:
        raise ContextError("public Y must use the canonical baseline-local resource path")
    output = load_public_y(y_path); hidden, classes, k_out, layers = _dimensions(manifest); expected_shape = (graph.num_nodes, classes)
    if tuple(output.shape) != expected_shape: raise ContextError(f"public Y shape {tuple(output.shape)} != expected {expected_shape}")
    y_digest = _array_digest("Y", output)
    public_y_provenance = _public_y_provenance(root, config, graph, raw_manifest, y_path, output, y_digest)
    if public_y_provenance.get("status") != "VERIFY_OK":
        failed = sorted(key for key, value in public_y_provenance.get("checks", {}).items() if value is not True)
        raise ContextError("public Y independent-reference provenance is invalid: " + ",".join(failed))
    binding = registry["registration_binding"]; bounds = binding.get("registered_model_bounds", {}); rounding_mode = config.rounding_mode; ledger = _ledger(graph, manifest, bounds, identities["quant_cfg_id"], rounding_mode, config.range_bits)
    if binding.get("integer_bound_ledger_digest") != digest_json(ledger): raise ContextError("trusted registry bound ledger does not match external graph/model context")
    n_in, layout_digest = int(binding.get("N_in", 0)), str(binding.get("Layout_in_digest", ""))
    if n_in <= 0 or n_in & (n_in - 1) or len(layout_digest) != 64: raise ContextError("trusted registry has invalid Layout_in identity")
    dims, domain = _dims(config, dataset, graph, manifest, n_in); model_arrays = _model_arrays(manifest); lasso = _families(graph, manifest, config.range_bits, ledger); matrix = _matrix_families(graph, manifest); gkr = _gkr_families(graph, manifest, np.ravel(output).tolist(), lasso, ledger, config.range_bits)
    # ``project_root`` is the baseline root for the comparison CLI, not the
    # repository root.  Registry/public-path identities therefore use this
    # same canonical root and can never silently resolve through
    # ``root/comparison/zkGPT``.
    zk_root = root.resolve(); registry_path = pathlib.Path(registry_public_path).resolve()
    if not _inside(registry_path, zk_root):
        raise ContextError("trusted registry public file must be inside baseline project-root")
    public_path = str(registry_path.relative_to(zk_root))
    registry_metadata = {"protocol": registry["protocol"], "registry_digest": registry["registry_digest"], "registration_binding_digest": registry["registration_binding_digest"], "public_path": public_path, "commitment_backend": registry["commitment_backend"], "public_descriptor": copy.deepcopy(registry)}
    expected = {"protocol_id": PROTOCOL_ID, "proof_backend_id": PROOF_BACKEND_ID, "commitment_backend_id": COMMITMENT_BACKEND_ID, **identities, "input_preprocess_id": config.input_preprocess_id, "quant_scale_chain": raw_manifest["quant_scale_chain"], "static_table_descriptor_schema": static_descriptor["schema"], "static_table_content_digest": static_descriptor["content_digest"], "static_table_descriptor_digest": static_descriptor["descriptor_digest"], "static_table_descriptor_path": STATIC_TABLE_DESCRIPTOR_RELATIVE_PATH, "domain_cfg": domain, "dim_cfg": dims, "encoding_id": ENCODING_ID, "padding_rule_id": PADDING_RULE_ID, "circuit_layout_id": CIRCUIT_LAYOUT_ID, "lasso_cfg_id": LASSO_CFG_ID, "matrix_proof_cfg_id": MATRIX_PROOF_CFG_ID, "dataset": config.entries.get("dataset"), "dataset_display_name": dataset.name, "task_type": graph.task_type, "report_unit": graph.report_unit, "batching_rule": graph.batching_rule, "self_loop_rule": graph.self_loop_rule, "edge_sort_rule": graph.edge_sort_rule, "chunking_rule": graph.chunking_rule, "full_dataset": graph.num_nodes == dataset.num_nodes, "X_pub_digest": _x_digest(graph, layers, classes, k_out, QUANT_SCALE_PROFILES[identities["quant_cfg_id"]]["input"]), "public_input_layout": "X_pub=features+topology+selectors; X_pub is verifier-reconstructed and uncommitted", "public_output_policy": "Y is a public statement output claim and is not part of X_pub or any private commitment", "private_input_layout": "X_pub|W|Aux|L_qry", "segment_order": ["X_pub", "W", "Aux", "L_qry"], "privacy_extension_profile_id": "masked_full_multi_session_v1", "privacy_profile_status": "supported_multi_session_masked_full", "registered_model_commitment_scope": "canonical_integer_GAT_model_weights_with_layout_and_bound_ledger", "Layout_in_digest": layout_digest, "integer_bound_ledger": ledger, "integer_bound_ledger_digest": digest_json(ledger), "public_output_digest": y_digest, "model_registry": registry_metadata, "model_registry_digest": digest_json(registry_metadata), "identities": identities}
    expected["rounding_mode"] = rounding_mode
    # Static tables are verifier setup reconstructed from authenticated
    # baseline resources, never a proof-selected artifact sidecar.
    canonical_tables = _canonical_static_table_registry(expected)
    expected["static_table_registry"] = copy.deepcopy(canonical_tables)
    expected["static_table_registry_digest"] = digest_json(canonical_tables)
    features = np.asarray([_round_feature(value) for value in np.ravel(graph.features)], dtype=np.int64).reshape(graph.features.shape)
    # Use the same canonical statement constructor as the producer path.  The
    # richer context material remains authenticated in model_registry,
    # static_table_vk, and setup_context below; it is not duplicated in the
    # statement wire object.
    statement = _canonical_public_statement(
        graph,
        features,
        output,
        {
            "protocol_id": PROTOCOL_ID,
            "dataset": config.entries.get("dataset", dataset.name),
            "dataset_display_name": dataset.name,
            "input_preprocess_id": config.input_preprocess_id,
            "rounding_mode": rounding_mode,
            "integer_bound_ledger_digest": digest_json(ledger),
        },
    )
    static_vk = static_table_vk(expected)
    setup = setup_context(expected, registry, static_vk)
    context_payload = {
        "schema": EXTERNAL_VERIFIER_CONTEXT_SCHEMA,
        "public_statement": statement,
        "model_registry": copy.deepcopy(registry),
        "static_table_vk": static_vk,
        "setup_context": setup,
    }
    context_digest = digest_json(context_payload)
    return VerifierContext(root, config_path, artifact, registry_path, y_path, config, dataset, graph, dict(manifest), registry, expected_shape, y_digest, expected, tuple(model_arrays), tuple(lasso), tuple(matrix), tuple(gkr), statement, context_digest, public_y_provenance)


# Compatibility helpers retained for code that imported the earlier context
# module.  They are deliberately not used by the external CLI path.
def _canonical_public_statement(
    graph: LocalGraph,
    features: Any,
    output: Any,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    feature_array = np.asarray(features, dtype=np.int64)
    output_array = np.asarray(output, dtype=object)
    statement: dict[str, Any] = {
        "schema": EXTERNAL_PUBLIC_STATEMENT_SCHEMA,
        "protocol_id": metadata.get("protocol_id", PROTOCOL_ID),
        "dataset": metadata["dataset"],
        "dataset_display_name": metadata["dataset_display_name"],
        "task_type": graph.task_type,
        "features": {
            "shape": list(feature_array.shape),
            "values": [int(value) for value in feature_array.ravel()],
        },
        "graph": {
            "absolute_ids": [int(value) for value in graph.absolute_ids],
            "src": [int(value) for value in graph.src],
            "dst": [int(value) for value in graph.dst],
            "node_ptr": graph.node_ptr.tolist(),
            "edge_ptr": graph.edge_ptr.tolist(),
            "edge_graph": [int(value) for value in graph.edge_graph],
            "self_loop_rule": graph.self_loop_rule,
            "edge_sort_rule": graph.edge_sort_rule,
            "batching_rule": graph.batching_rule,
        },
        "public_output": {
            "shape": list(output_array.shape),
            "values": [str(int(value)) for value in output_array.ravel()],
        },
        "input_preprocess_id": str(metadata["input_preprocess_id"]),
        "rounding_mode": validate_rounding_mode(metadata.get("rounding_mode")),
        "integer_bound_ledger_digest": metadata["integer_bound_ledger_digest"],
    }
    if set(statement) != PUBLIC_STATEMENT_FIELDS:
        raise ContextError("canonical public statement field set is invalid")
    statement["statement_digest"] = digest_json(statement)
    return statement


def public_statement(graph: LocalGraph, trace: Any, metadata: Mapping[str, Any]) -> dict[str, Any]:
    return _canonical_public_statement(
        graph,
        np.asarray(trace.input_features, dtype=np.int64),
        np.asarray(trace.y, dtype=object),
        metadata,
    )


def static_table_vk(metadata: Mapping[str, Any]) -> dict[str, Any]:
    _, table_registry_digest = _authenticated_static_table_registry(metadata)
    vk = {
        "schema": STATIC_TABLE_VK_SCHEMA,
        "rounding_mode": validate_rounding_mode(metadata.get("rounding_mode")),
        "static_table_id": metadata["static_table_id"],
        "quant_cfg_id": metadata["quant_cfg_id"],
        "input_preprocess_id": metadata["input_preprocess_id"],
        "static_table_descriptor_schema": metadata["static_table_descriptor_schema"],
        "static_table_content_digest": metadata["static_table_content_digest"],
        "static_table_descriptor_digest": metadata["static_table_descriptor_digest"],
        "integer_bound_ledger_digest": metadata["integer_bound_ledger_digest"],
        "table_registry_digest": table_registry_digest,
    }
    if set(vk) | {"vk_digest"} != set(STATIC_TABLE_VK_FIELDS):
        raise ContextError("static-table VK schema contains unsupported fields")
    vk["vk_digest"] = digest_json(vk)
    return vk


def setup_context(metadata: Mapping[str, Any], registry: Mapping[str, Any], table_vk: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(registry, Mapping) or registry.get("registry_digest") != digest_json({key: value for key, value in registry.items() if key != "registry_digest"}):
        raise ContextError("model registry digest is invalid in setup context")
    expected_vk = static_table_vk(metadata)
    if dict(table_vk) != expected_vk:
        raise ContextError("setup context static-table VK is not canonical")
    setup = {
        "schema": VERIFIER_SETUP_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "proof_backend_id": PROOF_BACKEND_ID,
        "commitment_backend_id": COMMITMENT_BACKEND_ID,
        "circuit_layout_id": CIRCUIT_LAYOUT_ID,
        "model_arch_id": metadata["model_arch_id"],
        "model_param_id": metadata["model_param_id"],
        "quant_cfg_id": metadata["quant_cfg_id"],
        "input_preprocess_id": metadata["input_preprocess_id"],
        "rounding_mode": validate_rounding_mode(metadata.get("rounding_mode")),
        "static_table_id": metadata["static_table_id"],
        "registry_digest": registry["registry_digest"],
        "Layout_in_digest": metadata["Layout_in_digest"],
        "domain_cfg": metadata["domain_cfg"],
        "dim_cfg": metadata["dim_cfg"],
        "integer_bound_ledger_digest": metadata["integer_bound_ledger_digest"],
    }
    if set(setup) | {"setup_digest"} != set(VERIFIER_SETUP_FIELDS):
        raise ContextError("verifier setup schema contains unsupported fields")
    setup["setup_digest"] = digest_json(setup)
    return setup


def make_external_context(graph: LocalGraph, trace: Any, metadata: Mapping[str, Any], registry: Mapping[str, Any]) -> dict[str, Any]:
    statement = public_statement(graph, trace, metadata)
    tables = static_table_vk(metadata)
    setup = setup_context(metadata, registry, tables)
    context = {
        "schema": EXTERNAL_VERIFIER_CONTEXT_SCHEMA,
        "public_statement": statement,
        "model_registry": dict(registry),
        "static_table_vk": tables,
        "setup_context": setup,
    }
    if set(context) | {"context_digest"} != set(EXTERNAL_VERIFIER_CONTEXT_FIELDS):
        raise ContextError("external verifier context schema contains unsupported fields")
    context["context_digest"] = digest_json(context)
    return context


def validate_external_context(metadata: Mapping[str, Any], context: Mapping[str, Any]) -> dict[str, bool]:
    """Validate the one canonical producer/fresh verifier context envelope.

    Every digest is checked before its self-authenticating field is removed.
    The public statement is then compared as an exact canonical payload, so a
    missing or extra field cannot be smuggled through a ``>=`` check.
    """
    result = {
        "external_context_envelope": False,
        "external_public_statement": False,
        "external_model_registry": False,
        "external_static_table_vk": False,
        "external_setup_context": False,
    }
    try:
        if not isinstance(context, Mapping) or set(context) != {
            "schema", "public_statement", "model_registry", "static_table_vk",
            "setup_context", "context_digest",
        }:
            return result
        if context.get("schema") != EXTERNAL_VERIFIER_CONTEXT_SCHEMA:
            return result
        envelope = dict(context)
        context_digest = envelope.get("context_digest")
        envelope.pop("context_digest", None)
        if not isinstance(context_digest, str) or context_digest != digest_json(envelope):
            return result
        result["external_context_envelope"] = True

        raw_statement = context.get("public_statement")
        if not isinstance(raw_statement, Mapping):
            return result
        statement = dict(raw_statement)
        # Validate the outer schema and digest while the digest member is
        # still present.  Only then remove it for exact payload validation.
        if statement.get("schema") != EXTERNAL_PUBLIC_STATEMENT_SCHEMA:
            return result
        statement_digest = statement.get("statement_digest")
        if not isinstance(statement_digest, str) or statement_digest != digest_json({
            key: value for key, value in statement.items() if key != "statement_digest"
        }):
            return result
        statement.pop("statement_digest", None)
        if set(statement) != PUBLIC_STATEMENT_FIELDS:
            return result
        features = statement["features"]
        if not isinstance(features, Mapping) or set(features) != {"shape", "values"}:
            return result
        feature_shape = tuple(int(value) for value in features["shape"])
        feature_values = np.asarray(features["values"], dtype=object).reshape(feature_shape)
        if not all(int(value) == value for value in feature_values.ravel()):
            return result
        graph_data = statement["graph"]
        graph_fields = {
            "absolute_ids", "src", "dst", "node_ptr", "edge_ptr", "edge_graph",
            "self_loop_rule", "edge_sort_rule", "batching_rule",
        }
        if not isinstance(graph_data, Mapping) or set(graph_data) != graph_fields:
            return result
        topology = {
            "absolute_ids": _array_digest("absolute_ids", np.asarray(graph_data["absolute_ids"], dtype=np.int64)),
            "src": _array_digest("src", np.asarray(graph_data["src"], dtype=np.int64)),
            "dst": _array_digest("dst", np.asarray(graph_data["dst"], dtype=np.int64)),
            "node_ptr": [int(value) for value in graph_data["node_ptr"]],
            "edge_ptr": [int(value) for value in graph_data["edge_ptr"]],
        }
        graph_count = len(topology["node_ptr"]) - 1
        x_digest = digest_json({
            "features": _array_digest("X_pub.H_input", feature_values.astype(np.int64)),
            "topology": digest_json(topology),
            "selectors": {
                "segment_order": ["X_pub", "W", "Aux", "L_qry"],
                "graph_count": graph_count,
            },
        })
        public_output = statement["public_output"]
        if not isinstance(public_output, Mapping) or set(public_output) != {"shape", "values"}:
            return result
        output_shape = tuple(int(value) for value in public_output["shape"])
        output_values = np.asarray(
            [int(value) for value in public_output["values"]], dtype=object
        ).reshape(output_shape)
        expected_shape = (
            int(metadata.get("dim_cfg", {}).get("N", -1)),
            int(metadata.get("dim_cfg", {}).get("C", -1)),
        )
        result["external_public_statement"] = bool(
            statement.get("protocol_id") == metadata.get("protocol_id", PROTOCOL_ID)
            and statement.get("dataset") == metadata.get("dataset")
            and statement.get("dataset_display_name") == metadata.get("dataset_display_name")
            and statement.get("task_type") == metadata.get("task_type")
            and statement.get("input_preprocess_id") == metadata.get("input_preprocess_id")
            and statement.get("rounding_mode") == metadata.get("rounding_mode")
            and statement.get("integer_bound_ledger_digest") == metadata.get("integer_bound_ledger_digest")
            and metadata.get("X_pub_digest") == x_digest
            and metadata.get("public_output_digest") == _array_digest("Y", output_values)
            and output_shape == expected_shape
        )

        registry_value = context.get("model_registry")
        if not isinstance(registry_value, Mapping):
            return result
        registry = dict(registry_value)
        registry_digest = registry.get("registry_digest")
        binding = registry.get("registration_binding", {})
        result["external_model_registry"] = bool(
            isinstance(registry_digest, str)
            and registry_digest == digest_json({key: value for key, value in registry.items() if key != "registry_digest"})
            and isinstance(binding, Mapping)
            and binding.get("rounding_mode") == metadata.get("rounding_mode")
            and registry.get("registration_binding_digest") == digest_json(binding)
            and registry_value == metadata.get("model_registry", {}).get("public_descriptor")
        )

        tables = context.get("static_table_vk")
        if not isinstance(tables, Mapping):
            return result
        table_payload = dict(tables)
        vk_digest = table_payload.get("vk_digest")
        table_payload.pop("vk_digest", None)
        expected_table_fields = {
            "schema", "rounding_mode", "static_table_id", "quant_cfg_id", "input_preprocess_id",
            "static_table_descriptor_schema", "static_table_content_digest",
            "static_table_descriptor_digest", "integer_bound_ledger_digest",
            "table_registry_digest",
        }
        result["external_static_table_vk"] = bool(
            set(table_payload) == expected_table_fields
            and tables.get("schema") == STATIC_TABLE_VK_SCHEMA
            and isinstance(vk_digest, str)
            and vk_digest == digest_json(table_payload)
            and table_payload.get("rounding_mode") == metadata.get("rounding_mode")
            and table_payload.get("static_table_id") == metadata.get("static_table_id")
            and table_payload.get("quant_cfg_id") == metadata.get("quant_cfg_id")
            and table_payload.get("input_preprocess_id") == metadata.get("input_preprocess_id")
            and table_payload.get("static_table_descriptor_schema") == metadata.get("static_table_descriptor_schema")
            and table_payload.get("static_table_content_digest") == metadata.get("static_table_content_digest")
            and table_payload.get("static_table_descriptor_digest") == metadata.get("static_table_descriptor_digest")
            and table_payload.get("integer_bound_ledger_digest") == metadata.get("integer_bound_ledger_digest")
            and table_payload.get("table_registry_digest") == metadata.get("static_table_registry_digest")
        )

        setup = context.get("setup_context")
        if not isinstance(setup, Mapping):
            return result
        setup_payload = dict(setup)
        setup_digest = setup_payload.get("setup_digest")
        setup_payload.pop("setup_digest", None)
        expected_setup_fields = {
            "schema", "protocol_id", "proof_backend_id", "commitment_backend_id",
            "circuit_layout_id", "model_arch_id", "model_param_id", "quant_cfg_id", "input_preprocess_id",
            "rounding_mode", "static_table_id", "registry_digest", "Layout_in_digest",
            "domain_cfg", "dim_cfg", "integer_bound_ledger_digest",
        }
        result["external_setup_context"] = bool(
            set(setup_payload) == expected_setup_fields
            and setup_payload.get("schema") == VERIFIER_SETUP_SCHEMA
            and isinstance(setup_digest, str)
            and setup_digest == digest_json(setup_payload)
            and setup_payload.get("protocol_id") == metadata.get("protocol_id")
            and setup_payload.get("proof_backend_id") == metadata.get("proof_backend_id")
            and setup_payload.get("commitment_backend_id") == metadata.get("commitment_backend_id")
            and setup_payload.get("circuit_layout_id") == metadata.get("circuit_layout_id")
            and setup_payload.get("model_arch_id") == metadata.get("model_arch_id")
            and setup_payload.get("model_param_id") == metadata.get("model_param_id")
            and setup_payload.get("quant_cfg_id") == metadata.get("quant_cfg_id")
            and setup_payload.get("input_preprocess_id") == metadata.get("input_preprocess_id")
            and setup_payload.get("rounding_mode") == metadata.get("rounding_mode")
            and setup_payload.get("static_table_id") == metadata.get("static_table_id")
            and setup_payload.get("registry_digest") == metadata.get("model_registry", {}).get("registry_digest")
            and setup_payload.get("Layout_in_digest") == metadata.get("Layout_in_digest")
            and setup_payload.get("domain_cfg") == metadata.get("domain_cfg")
            and setup_payload.get("dim_cfg") == metadata.get("dim_cfg")
            and setup_payload.get("integer_bound_ledger_digest") == metadata.get("integer_bound_ledger_digest")
        )
        return result
    except (KeyError, TypeError, ValueError, OverflowError, IndexError):
        return result


__all__ = ["ContextError", "EXTERNAL_PUBLIC_STATEMENT_SCHEMA", "VerifierContext", "build_verifier_context", "load_public_y", "public_statement", "static_table_vk", "setup_context", "make_external_context", "validate_external_context"]
