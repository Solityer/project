#!/usr/bin/env python3
"""Shared fail-closed contracts for the formal zkDL execution chain.

This module is production runner infrastructure, not a test adapter.  It keeps
all paths, resources, manifests, and result artifacts inside this baseline and
uses a single exact cell layout for both the runner and the independent
collector/validator.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any, Iterable


BASELINE_ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("cora", "citeseer", "pubmed", "ppi", "ogbn-arxiv")
DEVICES = ("cpu", "gpu")
CONFIGS = {
    "cora": "cora_full.cfg",
    "citeseer": "citeseer_full.cfg",
    "pubmed": "pubmed_full.cfg",
    "ppi": "ppi_full.cfg",
    "ogbn-arxiv": "ogbn_arxiv_full.cfg",
}
PROTOCOL_PATH = BASELINE_ROOT / "GAT-ZKML-zkDL.md"
RESOURCE_MANIFEST_PATH = BASELINE_ROOT / "resources" / "resource_manifest.json"
STAGED_RESOURCE_MANIFEST_PATH = BASELINE_ROOT / "resources" / "staged_resource_manifest.json"
RESOURCE_POLICY_SCHEMA = "zkDL_gat_resource_policy_v1"
STAGED_RESOURCE_MANIFEST_SCHEMA = "zkDL_gat_staged_resource_manifest_v1"
# Kept as a public alias for callers that only need to identify the source
# policy.  Production code must call load_resource_manifest(require_staged=True)
# and therefore never consumes this file as an authenticated resource set.
RESOURCE_MANIFEST_SCHEMA = RESOURCE_POLICY_SCHEMA
PP_INFO_SCHEMA = "zkDL_gat_pp_info_v1"
PP_CANONICAL_RELATIVE = "artifacts/setup/pp.bin"
PP_DEFAULT_MAX_VECTOR_LEN = 1 << 28
# Must stay in lockstep with ``expected_table`` in src/formal/protocol.rs:
# native relation validation materializes a full public vector and accepts at
# most 2^27 raw rows.  Python callers fail closed before attempting the same
# materialization, rather than changing the relation to a compact encoding.
STATIC_TABLE_ROW_CAP = 1 << 27
PP_SETUP_ID = "zkDL-GAT-transparent-PP-v1"
PP_SECURITY_ID = "zkdl_gat_bls12_381_multi_session_v2"
PP_PROTOCOL = "zkDL-hiding-Hyrax-IPA-BLS12-381-v1"
PP_SECURITY_BITS = 128
PP_FIELDS = (
    "pp_path",
    "pp_sha256",
    "pp_size_bytes",
    "pp_schema",
    "pp_id",
    "pp_max_vector_len",
    "pp_setup_id",
    "pp_security_id",
    "pp_protocol",
    "pp_security_bits",
)
DATASET_CACHE_SCHEMA = "zkDL_gat_dataset_cache_v1"
CHECKPOINT_BUNDLE_SCHEMA = "zkDL_gat_checkpoint_bundle_v1"
BUILD_MANIFEST_SCHEMA = "zkDL_gat_build_manifest_v1"
# The comparison baseline is intentionally pinned to the same current formal
# GAT shape as the main experiment.  These values are consumed by every
# production entry point (preflight, adapter, collector, and validator); a
# stale L2/single-head config must fail closed instead of silently describing a
# different relation.
CANONICAL_MODEL_L = 3
CANONICAL_HIDDEN_PROFILE = ((8, 8), (8, 8))
CANONICAL_MODEL_K_OUT = 2
CANONICAL_QUANT_CFG_ID = "gat_l3_scale_chain"
CANONICAL_INPUT_PREPROCESS_IDS = {
    "cora": "row_normalize",
    "citeseer": "row_normalize",
    "pubmed": "row_normalize",
    "ppi": "row_normalize",
    "ogbn-arxiv": "ogbn_arxiv_raw_float32",
}


def canonical_input_preprocess_id(dataset: str) -> str:
    """Return the one preprocessing identity authorized for *dataset*."""
    try:
        return CANONICAL_INPUT_PREPROCESS_IDS[dataset]
    except KeyError as error:
        raise ValueError(f"unknown dataset: {dataset}") from error
CANONICAL_SCALE_VALUES = {
    # ``feature_scale`` is the native integer input boundary and is therefore
    # the canonical scale-chain S_input, not a cache encoding hint.
    "feature_scale": 16,
    "hidden_weight_scale": 16,
    "hidden_attention_scale": 16,
    "hidden_projection_scale": 16,
    "delta_scale": 256,
    "exp_scale": 16,
    "hidden_recovery_num_scale": 1,
    "hidden_recovery_den_scale": 16,
    "elu_input_scale": 16,
    "elu_output_scale": 16,
    "agg_scale": 16,
    "output_weight_scale": 16,
    "output_attention_scale": 1,
    "output_projection_scale": 256,
    "output_pre_scale": 16,
    "output_head_scale": 16,
    "output_bias_scale": 16,
    "output_scale": 16,
}
CELL_SCHEMA = "zkDL_gat_cell_success_manifest_v1"
CELL_FILES = frozenset(
    {
        "formal_artifacts/preflight.json",
        "formal_artifacts/canonical_relation.json",
        "formal_artifacts/statement.bin",
        "formal_artifacts/proof.bin",
        "proof_manifest.json",
        "verifier_manifest.json",
        "result.json",
        "success_manifest.json",
    }
)
CELL_DIRS = frozenset({"formal_artifacts"})

# Private witness/model-opening bytes are process-local inputs.  These names
# are deliberately not part of the public cell schema; callers can use this
# small guard when checking JSON manifests for accidental publication.
PRIVATE_PUBLICATION_KEYS = frozenset(
    {
        "witness",
        "witness_path",
        "witness_sha256",
        "model_secret",
        "model_secret_path",
        "model_secret_sha256",
        "secret",
        "secret_path",
        "secret_sha256",
    }
)


def private_publication_fields(value: Any, *, prefix: str = "") -> list[str]:
    """Return public-manifest fields that would expose private inputs.

    The verifier-boundary boolean is intentionally allowed: it attests that
    the independent verifier did *not* read a witness and carries no secret
    bytes or path.  Raw witness/model-secret paths and digests are rejected.
    """
    findings: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            location = f"{prefix}.{key_text}" if prefix else key_text
            if key_text in PRIVATE_PUBLICATION_KEYS:
                findings.append(location)
            if key_text != "verifier_reads_witness":
                lowered = key_text.lower()
                if "witness.private.bin" in lowered or "model_opening.private.bin" in lowered:
                    findings.append(location)
            findings.extend(private_publication_fields(item, prefix=location))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            findings.extend(private_publication_fields(item, prefix=f"{prefix}[{index}]"))
    elif isinstance(value, str):
        lowered = value.lower()
        if "witness.private.bin" in lowered or "model_opening.private.bin" in lowered:
            findings.append(prefix or "value")
    return findings


def _relative(path: Path, root: Path = BASELINE_ROOT) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def ensure_contained(path: Path, *, root: Path = BASELINE_ROOT, label: str = "path") -> Path:
    """Resolve *path* and reject absolute escapes and symlink components."""
    root_resolved = root.resolve()
    raw_candidate = path if path.is_absolute() else root_resolved / path
    # Check the lexical path before resolving it.  A symlink whose target is
    # still inside the baseline is forbidden as well: artifacts must have one
    # unambiguous owner and may not alias another cell/resource.
    raw_candidate = Path(os.path.abspath(str(raw_candidate)))
    try:
        raw_parts = raw_candidate.relative_to(root_resolved).parts
    except ValueError as error:
        raise ValueError(f"{label} escapes baseline root: {path}") from error
    current = root_resolved
    for part in raw_parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{label} contains a symlink component: {current}")
    candidate = raw_candidate.resolve(strict=False)
    if candidate != root_resolved and root_resolved not in candidate.parents:
        raise ValueError(f"{label} must stay inside {root_resolved}: {path}")
    return candidate


def ensure_regular_file(path: Path, *, label: str, allow_missing: bool = False) -> Path:
    path = ensure_contained(path, label=label)
    if not path.exists():
        if allow_missing:
            return path
        raise FileNotFoundError(f"missing {label}: {path}")
    st = path.lstat()
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
        raise ValueError(f"{label} must be a regular non-symlink file: {path}")
    if st.st_nlink != 1:
        raise ValueError(f"{label} must not be hard-linked: {path}")
    return path


def ensure_directory(path: Path, *, label: str, allow_missing: bool = False) -> Path:
    path = ensure_contained(path, label=label)
    if not path.exists():
        if allow_missing:
            return path
        raise FileNotFoundError(f"missing {label}: {path}")
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{label} must be a real directory: {path}")
    return path


def sha256_file(path: Path) -> str:
    path = ensure_regular_file(path, label="hashed file")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def canonical_data_identity(
    *,
    dataset: str,
    input_preprocess_id: str,
    node_count: int,
    edge_count: int,
    graph_count: int,
    feature_dim: int,
    graph_features_digest: str,
    topology_digest: str,
) -> str:
    """Return the explicit public data identity bound into every artifact."""
    if input_preprocess_id != canonical_input_preprocess_id(dataset):
        raise ValueError("data identity input preprocessing is not canonical")
    material = {
        "dataset": dataset,
        "input_preprocess_id": input_preprocess_id,
        "node_count": int(node_count),
        "edge_count": int(edge_count),
        "graph_count": int(graph_count),
        "feature_dim": int(feature_dim),
        "graph_features_digest": graph_features_digest,
        "topology_digest": topology_digest,
    }
    return f"sha256:{sha256_bytes(canonical_json(material))}"


def atomic_write_bytes(path: Path, payload: bytes, *, mode: int = 0o644) -> None:
    path = ensure_contained(path, label="output")
    path.parent.mkdir(parents=True, exist_ok=True)
    ensure_directory(path.parent, label="output parent")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def atomic_write_json(path: Path, value: Any, *, mode: int = 0o644) -> None:
    atomic_write_bytes(path, canonical_json(value), mode=mode)


def read_json(path: Path, *, label: str = "JSON") -> dict[str, Any]:
    ensure_regular_file(path, label=label)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def config_path(dataset: str) -> Path:
    if dataset not in CONFIGS:
        raise ValueError(f"unknown dataset: {dataset}")
    return ensure_regular_file(
        BASELINE_ROOT / "configs" / CONFIGS[dataset], label=f"{dataset} config"
    )


def parse_config(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def parse_int_profile(value: Any, *, label: str) -> list[int]:
    """Parse a comma-separated integer profile from config or metadata.

    Both ``d_in_profile = 1433,64`` and JSON-style ``[1433, 64]`` are
    accepted at the boundary, while all downstream code receives a canonical
    integer list.  A scalar is deliberately rejected: it was the source of
    the old L2/single-input assumption.
    """
    if isinstance(value, list):
        raw_items = value
    elif isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1].strip()
        if not text:
            raise ValueError(f"{label} must not be empty")
        raw_items = [item.strip() for item in text.split(",")]
    else:
        raise ValueError(f"{label} must be a comma-separated integer profile")
    result: list[int] = []
    for item in raw_items:
        if isinstance(item, bool):
            raise ValueError(f"{label} contains a boolean")
        try:
            number = int(item)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{label} contains a non-integer") from error
        if number <= 0:
            raise ValueError(f"{label} contains a non-positive dimension")
        result.append(number)
    if not result:
        raise ValueError(f"{label} must not be empty")
    return result


def parse_hidden_profile(value: Any, *, label: str = "hidden_profile") -> list[tuple[int, int]]:
    """Parse the canonical ``head_count x head_dim`` hidden profile."""
    if isinstance(value, list):
        tokens = value
    elif isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1].strip()
        tokens = [item.strip() for item in text.split(",")] if text else []
    else:
        raise ValueError(f"{label} must be a comma-separated head profile")
    result: list[tuple[int, int]] = []
    for token in tokens:
        if isinstance(token, dict):
            try:
                heads = int(token["head_count"])
                width = int(token["head_dim"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{label} contains a malformed object") from error
        else:
            parts = str(token).strip().lower().split("x")
            if len(parts) != 2:
                raise ValueError(f"{label} contains a malformed entry")
            try:
                heads, width = int(parts[0]), int(parts[1])
            except ValueError as error:
                raise ValueError(f"{label} contains a malformed entry") from error
        if heads <= 0 or width <= 0:
            raise ValueError(f"{label} contains a non-positive dimension")
        result.append((heads, width))
    if not result:
        raise ValueError(f"{label} must not be empty")
    return result


def validate_quantization_config(config: dict[str, str]) -> dict[str, int]:
    """Authenticate the native zkDL integer scale constants.

    The 5-way LeakyReLU divisor and table input/output scales are protocol
    constants.  Configurable model-dependent scales are checked here too, so
    adapter code cannot accidentally reinterpret a layer with a stale
    single-layer scale.
    """
    values: dict[str, int] = {}
    for key, expected in CANONICAL_SCALE_VALUES.items():
        try:
            actual = int(config[key])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"missing or malformed zkDL scale: {key}") from error
        if actual != expected:
            raise ValueError(f"zkDL scale {key}={actual} is not canonical ({expected})")
        values[key] = actual
    if config.get("quant_cfg_id") != CANONICAL_QUANT_CFG_ID:
        raise ValueError(
            "zkDL config quant_cfg_id must be the unversioned canonical scale chain"
        )
    dataset = config.get("dataset")
    expected_preprocess = CANONICAL_INPUT_PREPROCESS_IDS.get(dataset)
    if expected_preprocess is None:
        raise ValueError(f"unsupported formal dataset for input preprocessing: {dataset!r}")
    if config.get("input_preprocess_id") != expected_preprocess:
        raise ValueError(
            f"zkDL config input_preprocess_id must be {expected_preprocess} for {dataset}"
        )
    try:
        configured_input_scale = int(config["s_input"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("zkDL config s_input is missing or malformed") from error
    if configured_input_scale != values["feature_scale"]:
        raise ValueError("zkDL config s_input and feature_scale disagree")
    return values


def validate_model_structure_config(
    config: dict[str, str], *, feature_dim: int | None = None
) -> dict[str, Any]:
    """Validate the current L3, two-hidden-layer, two-output-head shape."""
    try:
        layer_count = int(config["L"])
        output_head_count = int(config["K_out"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("config L/K_out are malformed") from error
    profile = parse_hidden_profile(config.get("hidden_profile"), label="config hidden_profile")
    d_in_profile = parse_int_profile(config.get("d_in_profile"), label="config d_in_profile")
    if layer_count != CANONICAL_MODEL_L or output_head_count != CANONICAL_MODEL_K_OUT:
        raise ValueError("zkDL config must use the current L=3/K_out=2 model")
    if tuple(profile) != CANONICAL_HIDDEN_PROFILE:
        raise ValueError("zkDL config must use hidden_profile=8x8,8x8")
    if layer_count != len(profile) + 1 or len(d_in_profile) != len(profile):
        raise ValueError("L, hidden_profile, and d_in_profile lengths are inconsistent")
    if feature_dim is not None and d_in_profile[0] != int(feature_dim):
        raise ValueError("config d_in_profile[0] does not match public feature dimension")
    expected_d_in = [d_in_profile[0]] + [heads * width for heads, width in profile[:-1]]
    if d_in_profile != expected_d_in:
        raise ValueError(
            "config d_in_profile must be [dataset_feature_dim, previous_layer_concat_width]"
        )
    if not config.get("model_arch_id") or not config.get("model_param_id"):
        raise ValueError("config model_arch_id/model_param_id are required")
    return {
        "L": layer_count,
        "hidden_profile": profile,
        "d_in_profile": d_in_profile,
        "K_out": output_head_count,
    }


def validate_checkpoint_structure(
    config: dict[str, str],
    manifest: dict[str, Any],
    *,
    graph_count: int,
    feature_dim: int,
) -> dict[str, Any]:
    """Bind checkpoint metadata to every layer/input/head dimension."""
    structure = validate_model_structure_config(config, feature_dim=feature_dim)
    dataset = config["dataset"]
    manifest_dataset = manifest.get("dataset")
    if manifest_dataset not in (None, dataset):
        raise ValueError("checkpoint manifest dataset mismatch")
    for key in ("L", "K_out", "C"):
        if int(manifest.get(key, -1)) != int(config[key]):
            raise ValueError(f"checkpoint/config {key} mismatch")
    if manifest.get("model_arch_id") != config["model_arch_id"]:
        raise ValueError("checkpoint/config model_arch_id mismatch")
    if manifest.get("model_param_id") != config["model_param_id"]:
        raise ValueError("checkpoint/config model_param_id mismatch")
    if manifest.get("quant_cfg_id") != CANONICAL_QUANT_CFG_ID:
        raise ValueError("checkpoint manifest quant_cfg_id is not canonical")
    expected_preprocess = CANONICAL_INPUT_PREPROCESS_IDS.get(dataset)
    if manifest.get("input_preprocess_id") != expected_preprocess:
        raise ValueError("checkpoint manifest input_preprocess_id is not canonical")
    actual_profile = parse_hidden_profile(
        manifest.get("hidden_profile"), label="checkpoint hidden_profile"
    )
    if actual_profile != structure["hidden_profile"]:
        raise ValueError("checkpoint/config hidden_profile mismatch")
    actual_d_in = parse_int_profile(
        manifest.get("d_in_profile"), label="checkpoint d_in_profile"
    )
    if actual_d_in != structure["d_in_profile"]:
        raise ValueError("checkpoint/config d_in_profile mismatch")
    layers = manifest.get("hidden_layers")
    if not isinstance(layers, list) or len(layers) != len(actual_profile):
        raise ValueError("checkpoint hidden layer/profile count mismatch")
    for index, (layer, (head_count, head_dim), input_dim) in enumerate(
        zip(layers, actual_profile, actual_d_in)
    ):
        if not isinstance(layer, dict):
            raise ValueError("checkpoint hidden layer metadata is malformed")
        if (
            int(layer.get("layer_index", -1)) != index
            or int(layer.get("input_dim", -1)) != input_dim
            or int(layer.get("head_count", -1)) != head_count
            or int(layer.get("head_dim", -1)) != head_dim
        ):
            raise ValueError("checkpoint hidden layer/profile/input mapping mismatch")
    hidden_specs = manifest.get("hidden_head_specs")
    expected_hidden_count = sum(heads for heads, _ in actual_profile)
    if not isinstance(hidden_specs, list) or len(hidden_specs) != expected_hidden_count:
        raise ValueError("checkpoint hidden head count mismatch")
    cursor = 0
    global_indices: list[int] = []
    for layer_index, (head_count, _) in enumerate(actual_profile):
        for local_head_index in range(head_count):
            spec = hidden_specs[cursor]
            if not isinstance(spec, dict) or (
                int(spec.get("layer_index", -1)) != layer_index
                or int(spec.get("local_head_index", -1)) != local_head_index
            ):
                raise ValueError("checkpoint hidden head layer/local/global mapping mismatch")
            global_index = spec.get("global_head_index")
            if global_index is not None:
                try:
                    global_index = int(global_index)
                except (TypeError, ValueError) as error:
                    raise ValueError("checkpoint global_head_index is malformed") from error
                if global_index < 0:
                    raise ValueError("checkpoint global_head_index is negative")
                global_indices.append(global_index)
            cursor += 1
    if len(global_indices) != len(set(global_indices)):
        raise ValueError("checkpoint global_head_index values collide")
    output_specs = manifest.get("output_head_specs")
    if not isinstance(output_specs, list) or len(output_specs) != structure["K_out"]:
        raise ValueError("checkpoint output head count mismatch")
    for index, spec in enumerate(output_specs):
        if not isinstance(spec, dict) or int(spec.get("head_index", -1)) != index:
            raise ValueError("checkpoint output head index mapping mismatch")
    configured_rule = config["batching_rule"]
    default_rule = "whole_graph_single" if graph_count == 1 else "multi_graph_batch"
    actual_rule = manifest.get("batching_rule", default_rule)
    if actual_rule != configured_rule:
        raise ValueError("checkpoint/config batching_rule mismatch")
    configured_graphs = int(config["batch_graphs"])
    actual_graphs = int(
        manifest.get("batch_graphs", 1 if actual_rule == "whole_graph_single" else graph_count)
    )
    if actual_graphs != configured_graphs:
        raise ValueError("checkpoint/config batch_graphs mismatch")
    return {
        "dataset": dataset,
        "L": int(manifest["L"]),
        "hidden_profile": [list(item) for item in actual_profile],
        "d_in_profile": actual_d_in,
        "K_out": int(manifest["K_out"]),
        "model_arch_id": manifest["model_arch_id"],
        "model_param_id": manifest["model_param_id"],
        "batching_rule": actual_rule,
        "batch_graphs": actual_graphs,
    }


def validate_checkpoint_tensor_shapes(
    manifest: dict[str, Any], tensor_shapes: dict[str, Any]
) -> None:
    """Authenticate every checkpoint tensor name and shape used by the adapter.

    The canonical bundles use a leading singleton axis for matrices and
    attention vectors: ``(1, d_in, d_out)``, ``(1, d_out, 1)``, and ``(1,)``.
    This inventory check is deliberately independent of tensor values and is
    shared by preflight and the adapter so a stale/partial bundle cannot reach
    registration.
    """
    layers = manifest.get("hidden_layers")
    hidden_specs = manifest.get("hidden_head_specs")
    output_specs = manifest.get("output_head_specs")
    if not isinstance(layers, list) or not isinstance(hidden_specs, list) or not isinstance(output_specs, list):
        raise ValueError("checkpoint tensor inventory metadata is malformed")
    expected: dict[str, tuple[int, ...]] = {}

    def add(name: Any, shape: tuple[int, ...]) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("checkpoint tensor name is malformed")
        if name in expected:
            raise ValueError(f"checkpoint tensor name collides: {name}")
        expected[name] = shape

    for spec in hidden_specs:
        layer_index = int(spec["layer_index"])
        layer = layers[layer_index]
        input_dim = int(layer["input_dim"])
        head_dim = int(layer["head_dim"])
        add(spec["seq_kernel"], (1, input_dim, head_dim))
        add(spec["attn_src_kernel"], (1, head_dim, 1))
        add(spec["attn_dst_kernel"], (1, head_dim, 1))
        add(spec["attn_src_bias"], (1,))
        add(spec["attn_dst_bias"], (1,))
        add(spec["output_bias"], (head_dim,))
    if not layers:
        raise ValueError("checkpoint has no hidden layers")
    output_input_dim = int(layers[-1]["head_count"]) * int(layers[-1]["head_dim"])
    class_count = int(manifest["C"])
    for spec in output_specs:
        add(spec["seq_kernel"], (1, output_input_dim, class_count))
        add(spec["attn_src_kernel"], (1, class_count, 1))
        add(spec["attn_dst_kernel"], (1, class_count, 1))
        add(spec["attn_src_bias"], (1,))
        add(spec["attn_dst_bias"], (1,))
        add(spec["output_bias"], (class_count,))

    actual: dict[str, tuple[int, ...]] = {}
    for name, shape in tensor_shapes.items():
        try:
            normalized = tuple(int(value) for value in shape)
        except (TypeError, ValueError) as error:
            raise ValueError(f"checkpoint tensor shape is malformed: {name}") from error
        if name in actual:
            raise ValueError(f"checkpoint tensor name is duplicated: {name}")
        actual[name] = normalized
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise ValueError(f"checkpoint tensor inventory mismatch missing={missing} extra={extra}")
    mismatches = [
        f"{name}: expected {expected[name]} got {actual[name]}"
        for name in sorted(expected)
        if actual[name] != expected[name]
    ]
    if mismatches:
        raise ValueError("checkpoint tensor shape mismatch: " + "; ".join(mismatches))


def static_table_row_cap_blocker(tables: Any) -> dict[str, Any] | None:
    """Return a structured blocker before native full-table materialization.

    Rust ``expected_table`` checks the raw (unpadded) domain length against
    ``1 << 27``.  Keep that exact policy here; do not truncate or replace the
    authenticated full-vector table.  ``None`` means every profile entry is
    safe to hand to the native relation builder.
    """
    entries: list[dict[str, int | str]] = []
    malformed: list[str] = []
    if not isinstance(tables, dict):
        return {
            "code": "static_table_profile_malformed",
            "message": "static table profile must be an object",
            "native_row_cap": STATIC_TABLE_ROW_CAP,
            "native_row_cap_expression": "1 << 27",
        }
    for kind in ("lrelu", "elu", "exp"):
        entry = tables.get(kind)
        if not isinstance(entry, dict):
            malformed.append(kind)
            continue
        try:
            minimum = int(entry["min"])
            maximum = int(entry["max"])
            declared_rows = int(entry["rows"])
        except (KeyError, TypeError, ValueError):
            malformed.append(kind)
            continue
        raw_rows = maximum - minimum + 1
        entries.append(
            {
                "kind": kind,
                "min": minimum,
                "max": maximum,
                "rows": raw_rows,
                "declared_rows": declared_rows,
            }
        )
        if raw_rows <= 0 or declared_rows != raw_rows:
            malformed.append(kind)
    if malformed:
        return {
            "code": "static_table_profile_malformed",
            "message": "static table domain bounds and row counts are inconsistent",
            "native_row_cap": STATIC_TABLE_ROW_CAP,
            "native_row_cap_expression": "1 << 27",
            "tables": entries,
            "malformed_tables": sorted(set(malformed)),
        }
    offenders = [entry for entry in entries if int(entry["rows"]) > STATIC_TABLE_ROW_CAP]
    if not offenders:
        return None
    return {
        "code": "static_table_domain_exceeds_native_cap",
        "message": (
            "native expected_table materializes a full public vector and "
            "accepts raw rows <= 2^27; relation construction is blocked"
        ),
        "native_row_cap": STATIC_TABLE_ROW_CAP,
        "native_row_cap_expression": "1 << 27",
        "tables": entries,
        "offending_tables": offenders,
    }


def read_checkpoint_tensor_shapes(path: Path) -> dict[str, tuple[int, ...]]:
    """Read only tensor names/shapes from a canonical ``tensors.txt`` bundle."""
    tokens = path.read_text(encoding="utf-8").split()
    index = 0
    shapes: dict[str, tuple[int, ...]] = {}
    while index < len(tokens):
        if tokens[index] != "TENSOR" or index + 2 >= len(tokens):
            raise ValueError("checkpoint tensor marker is malformed")
        name = tokens[index + 1]
        rank = int(tokens[index + 2])
        index += 3
        if name in shapes or rank < 0 or index + rank >= len(tokens):
            raise ValueError(f"checkpoint tensor header is malformed: {name}")
        shape = tuple(int(value) for value in tokens[index : index + rank])
        index += rank
        if index >= len(tokens):
            raise ValueError(f"checkpoint tensor count is missing: {name}")
        count = int(tokens[index])
        index += 1
        if count < 0 or index + count > len(tokens):
            raise ValueError(f"checkpoint tensor payload is truncated: {name}")
        index += count
        shapes[name] = shape
    return shapes


def model_shape_payload(manifest: dict[str, Any], *, feature_dim: int) -> dict[str, Any]:
    """Return the shape-only identity payload shared by all production stages."""
    profile = parse_hidden_profile(manifest.get("hidden_profile"), label="checkpoint hidden_profile")
    d_in_profile = parse_int_profile(manifest.get("d_in_profile"), label="checkpoint d_in_profile")
    return {
        "L": int(manifest["L"]),
        "hidden_profile": [list(item) for item in profile],
        "d_in_profile": d_in_profile,
        "K_out": int(manifest["K_out"]),
        "hidden_layers": manifest["hidden_layers"],
        "hidden_head_specs": manifest["hidden_head_specs"],
        "output_head_specs": manifest["output_head_specs"],
        "class_count": int(manifest["C"]),
        "feature_dim": int(feature_dim),
    }


def validate_preflight_metadata(report: dict[str, Any], expected: dict[str, Any]) -> None:
    """Compare every adapter/preflight binding field fail-closed."""
    for key, value in expected.items():
        if key not in report:
            raise ValueError("preflight metadata missing: " + key)
        if report[key] != value:
            raise ValueError("canonical/preflight metadata mismatch: " + key)


def _manifest_digest(manifest: dict[str, Any]) -> str:
    material = dict(manifest)
    material.pop("manifest_id", None)
    return f"sha256:{sha256_bytes(canonical_json(material))}"


def _sha256_id(value: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("sha256 value must be a 64-character hexadecimal digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError("sha256 value is not hexadecimal") from error
    return f"sha256:{value}"


def _require_sha256_id(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise ValueError(f"{label} must be a tagged sha256 digest")
    _sha256_id(value[7:])
    return value


def _validate_pp_policy(pp: Any) -> dict[str, Any]:
    """Validate the immutable PP policy without accepting an identity claim."""
    if not isinstance(pp, dict):
        raise ValueError("resource policy PP entry must be an object")
    expected = {
        "path": PP_CANONICAL_RELATIVE,
        "schema": PP_INFO_SCHEMA,
        "max_vector_len": PP_DEFAULT_MAX_VECTOR_LEN,
        "setup_id": PP_SETUP_ID,
        "security_id": PP_SECURITY_ID,
        "protocol": PP_PROTOCOL,
        "security_bits": PP_SECURITY_BITS,
    }
    if set(pp) - set(expected):
        raise ValueError("resource policy PP contains an identity or unknown field")
    for key, value in expected.items():
        if pp.get(key) != value:
            raise ValueError(f"resource policy PP {key} mismatch")
    maximum = pp["max_vector_len"]
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum <= 0 or maximum & (maximum - 1):
        raise ValueError("resource policy PP max_vector_len must be a power of two")
    return pp


def _validate_pp_spec(pp: Any) -> dict[str, Any]:
    """Validate an actual PP identity from the staged manifest."""
    if not isinstance(pp, dict):
        raise ValueError("staged resource manifest PP entry must be an object")
    policy = _validate_pp_policy({key: pp.get(key) for key in (
        "path", "schema", "max_vector_len", "setup_id", "security_id",
        "protocol", "security_bits",
    )})
    if not isinstance(pp.get("pp_id"), str) or len(pp["pp_id"]) != 64:
        raise ValueError("staged PP pp_id is malformed")
    try:
        int(pp["pp_id"], 16)
    except ValueError as error:
        raise ValueError("staged PP pp_id is not hexadecimal") from error
    if set(pp["pp_id"]) == {"0"}:
        raise ValueError("staged PP pp_id may not be a placeholder")
    _require_sha256_id(pp.get("sha256"), label="staged PP sha256")
    size = pp.get("size_bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError("staged PP size_bytes must be positive")
    return {**policy, "pp_id": pp["pp_id"], "sha256": pp["sha256"], "size_bytes": size}


def pp_path_from_manifest(manifest: dict[str, Any]) -> Path:
    pp = _validate_pp_spec(manifest.get("pp"))
    path = ensure_contained(BASELINE_ROOT / pp["path"], label="canonical PP path")
    if path.relative_to(BASELINE_ROOT).as_posix() != PP_CANONICAL_RELATIVE:
        raise ValueError("resource manifest PP path is not canonical")
    return path


def pp_binding_from_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    pp = _validate_pp_spec(manifest.get("pp"))
    return {
        "pp_path": pp["path"],
        "pp_sha256": pp["sha256"],
        "pp_size_bytes": pp["size_bytes"],
        "pp_schema": pp["schema"],
        "pp_id": pp["pp_id"],
        "pp_max_vector_len": pp["max_vector_len"],
        "pp_setup_id": pp["setup_id"],
        "pp_security_id": pp["security_id"],
        "pp_protocol": pp["protocol"],
        "pp_security_bits": pp["security_bits"],
    }


def validate_pp_file(path: Path, manifest: dict[str, Any], pp_info: dict[str, Any] | None = None) -> dict[str, Any]:
    """Authenticate the one baseline-local PP and its native pp-info record."""
    expected = pp_binding_from_manifest(manifest)
    canonical = pp_path_from_manifest(manifest)
    actual_path = ensure_contained(path, label="PP path")
    if actual_path != canonical:
        raise ValueError(f"PP path is not the manifest canonical path: {actual_path}")
    actual_path = ensure_regular_file(actual_path, label="PP")
    actual_sha = _sha256_id(sha256_file(actual_path))
    actual_size = actual_path.stat().st_size
    if actual_sha != expected["pp_sha256"]:
        raise ValueError("PP sha256 does not match the authenticated resource manifest")
    if actual_size != expected["pp_size_bytes"]:
        raise ValueError("PP size does not match the authenticated resource manifest")
    if pp_info is not None:
        # The native pp-info JSON has a canonical `pp_*` projection plus
        # compatibility aliases used by older setup tooling.  Authenticate
        # every field the binary emits, while keeping the parser forward
        # compatible with a binary that omits an optional alias.
        required_native = {
            "schema": "pp_schema",
            "pp_id": "pp_id",
            "max_vector_len": "pp_max_vector_len",
        }
        optional_native = {
            "pp_path": "pp_path",
            "pp_sha256": "pp_sha256",
            "pp_size_bytes": "pp_size_bytes",
            "pp_schema": "pp_schema",
            "pp_max_vector_len": "pp_max_vector_len",
            "pp_setup_id": "pp_setup_id",
            "pp_security_id": "pp_security_id",
            "pp_protocol": "pp_protocol",
            "pp_security_bits": "pp_security_bits",
            "setup_id": "pp_setup_id",
            "security_id": "pp_security_id",
            "protocol": "pp_protocol",
            "security_bits": "pp_security_bits",
            "sha256": "pp_sha256",
            "size_bytes": "pp_size_bytes",
            "digest": "pp_digest",
            "pp_digest": "pp_digest",
        }
        expected_aliases = {
            **expected,
            "pp_sha256": expected["pp_sha256"],
            "pp_size_bytes": expected["pp_size_bytes"],
            "pp_digest": f"sha256:{expected['pp_id']}",
        }
        for native_key, binding_key in required_native.items():
            if pp_info.get(native_key) != expected_aliases[binding_key]:
                raise ValueError(f"native PP info {native_key} mismatch")
        for native_key, binding_key in optional_native.items():
            if native_key in pp_info and pp_info.get(native_key) != expected_aliases[binding_key]:
                raise ValueError(f"native PP info {native_key} mismatch")
    return expected


def _production_source_files() -> list[Path]:
    """List source inputs that determine a formal binary/execution identity."""
    roots = [BASELINE_ROOT / "src", BASELINE_ROOT / "scripts", BASELINE_ROOT / "configs"]
    explicit = [BASELINE_ROOT / "Cargo.toml", BASELINE_ROOT / "Cargo.lock", BASELINE_ROOT / "build.rs"]
    paths = [path for path in explicit if path.is_file()]
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            if path.suffix not in {".rs", ".cu", ".cuh", ".py", ".sh", ".cfg", ".json", ".toml"}:
                continue
            if path.parts and "tests" in path.parts:
                continue
            paths.append(path)
    return sorted(set(paths))


def source_identity() -> str:
    records: list[dict[str, Any]] = []
    for path in _production_source_files():
        ensure_regular_file(path, label="production source")
        records.append({"path": _relative(path), "size": path.stat().st_size, "sha256": sha256_file(path)})
    return f"sha256:{sha256_bytes(canonical_json(records))}"


def build_manifest_path(device: str) -> Path:
    if device not in DEVICES:
        raise ValueError(f"invalid build device: {device}")
    return ensure_contained(BASELINE_ROOT / "bin" / f"zkdl-formal-{device}.build.json", label="build manifest")


def write_build_manifest(device: str, binary: Path) -> dict[str, Any]:
    binary = ensure_regular_file(binary, label=f"{device} formal binary")
    if binary != ensure_contained(BASELINE_ROOT / "bin" / f"zkdl-formal-{device}", label="formal binary"):
        raise ValueError("build binary path is not canonical")
    payload: dict[str, Any] = {
        "schema": BUILD_MANIFEST_SCHEMA,
        "baseline": "zkdl",
        "device": device,
        "binary_path": binary.relative_to(BASELINE_ROOT).as_posix(),
        "binary_sha256": _sha256_id(sha256_file(binary)),
        "binary_size_bytes": binary.stat().st_size,
        "source_identity": source_identity(),
        "cargo_lock_sha256": _sha256_id(sha256_file(BASELINE_ROOT / "Cargo.lock")),
        "profile": "release",
        "feature": "gpu" if device == "gpu" else "cpu",
    }
    material = canonical_json(payload)
    payload["build_identity"] = f"sha256:{sha256_bytes(material)}"
    write_path = build_manifest_path(device)
    atomic_write_json(write_path, payload)
    return payload


def load_build_manifest(device: str, binary: Path) -> dict[str, Any]:
    binary = ensure_regular_file(binary, label=f"{device} formal binary")
    manifest = read_json(build_manifest_path(device), label=f"{device} build manifest")
    if manifest.get("schema") != BUILD_MANIFEST_SCHEMA or manifest.get("baseline") != "zkdl":
        raise ValueError(f"{device} build manifest schema mismatch")
    if manifest.get("device") != device or manifest.get("profile") != "release":
        raise ValueError(f"{device} build manifest identity mismatch")
    expected_binary = ensure_contained(BASELINE_ROOT / str(manifest.get("binary_path", "")), label="build binary path")
    if expected_binary != binary:
        raise ValueError(f"{device} build manifest binary path mismatch")
    if manifest.get("binary_sha256") != _sha256_id(sha256_file(binary)):
        raise ValueError(f"{device} binary digest differs from build manifest")
    if manifest.get("binary_size_bytes") != binary.stat().st_size:
        raise ValueError(f"{device} binary size differs from build manifest")
    if manifest.get("source_identity") != source_identity():
        raise ValueError(f"{device} build is stale for current source")
    if manifest.get("cargo_lock_sha256") != _sha256_id(sha256_file(BASELINE_ROOT / "Cargo.lock")):
        raise ValueError(f"{device} build lockfile identity mismatch")
    material = dict(manifest)
    build_id = material.pop("build_identity", None)
    if build_id != f"sha256:{sha256_bytes(canonical_json(material))}":
        raise ValueError(f"{device} build manifest digest mismatch")
    return manifest


def _validate_dataset_policy_entry(dataset: str, entry: Any) -> dict[str, str]:
    if not isinstance(entry, dict):
        raise ValueError(f"resource policy entry is not an object: {dataset}")
    expected = {
        "cache": f"resources/datasets/{'ogbn_arxiv' if dataset == 'ogbn-arxiv' else dataset}",
        "checkpoint_bundle": f"resources/models/{dataset.replace('-', '_')}_gat",
    }
    for key, value in expected.items():
        raw = entry.get(key)
        if raw != value or Path(str(raw)).is_absolute():
            raise ValueError(f"resource policy {dataset} {key} path is not canonical")
        ensure_contained(BASELINE_ROOT / value, label=f"{dataset} {key}")
    return expected


def load_resource_policy() -> tuple[dict[str, Any], str]:
    """Load only the immutable source policy; it carries no PP/resource identity."""
    policy = read_json(RESOURCE_MANIFEST_PATH, label="resource policy")
    if policy.get("schema") != RESOURCE_POLICY_SCHEMA or policy.get("baseline") != "zkdl":
        raise ValueError("unsupported zkDL resource policy schema")
    allowed = {"schema", "baseline", "datasets", "pp", "static_tables"}
    if set(policy) != allowed:
        raise ValueError("source resource policy contains staged identity or unknown fields")
    _validate_pp_policy(policy.get("pp"))
    datasets = policy.get("datasets")
    if not isinstance(datasets, dict) or set(datasets) != set(DATASETS):
        raise ValueError("resource policy must enumerate exactly five datasets")
    for dataset in DATASETS:
        _validate_dataset_policy_entry(dataset, datasets[dataset])
    static_tables = policy.get("static_tables")
    if static_tables != {"scope": "baseline-local formal relation"}:
        raise ValueError("resource policy static_tables scope mismatch")
    policy_id = _manifest_digest(policy)
    return policy, policy_id


def _resource_bundle_records(dataset: str, root: Path) -> list[dict[str, Any]]:
    root = ensure_directory(root, label=f"{dataset} resource bundle")
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            if path.is_symlink():
                raise ValueError(f"resource directory is a symlink: {path}")
            continue
        ensure_regular_file(path, label=f"{dataset} resource")
        records.append({
            "path": _relative(path),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_id(sha256_file(path)),
        })
    if not records:
        raise ValueError(f"{dataset} resource bundle is empty")
    return records


def _bundle_identity(dataset: str, root: Path, schema: str) -> dict[str, Any]:
    records = _resource_bundle_records(dataset, root)
    return {
        "schema": schema,
        "size_bytes": sum(int(record["size_bytes"]) for record in records),
        "sha256": f"sha256:{sha256_bytes(canonical_json(records))}",
        "files": records,
    }


def build_staged_resource_manifest(
    *, require_all: bool = True, pp_info: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Authenticate staged bundles and the native PP into one atomic manifest."""
    policy, policy_id = load_resource_policy()
    staged: dict[str, Any] = {
        "schema": STAGED_RESOURCE_MANIFEST_SCHEMA,
        "baseline": "zkdl",
        "policy_id": policy_id,
        "datasets": {},
        "pp": {},
    }
    pp_policy = policy["pp"]
    pp_path = ensure_regular_file(
        BASELINE_ROOT / pp_policy["path"], label="canonical PP", allow_missing=not require_all
    )
    if pp_path.exists():
        pp_sha = _sha256_id(sha256_file(pp_path))
        pp_size = pp_path.stat().st_size
        if pp_size <= 0:
            raise ValueError("canonical PP must not be empty")
        pp_id = str((pp_info or {}).get("pp_id", ""))
    else:
        pp_sha = ""
        pp_size = 0
        pp_id = ""
    staged["pp"] = {
        **pp_policy,
        "pp_id": pp_id,
        "sha256": pp_sha,
        "size_bytes": pp_size,
    }
    for dataset in DATASETS:
        entry_policy = _validate_dataset_policy_entry(dataset, policy["datasets"][dataset])
        cache = BASELINE_ROOT / entry_policy["cache"]
        checkpoint = BASELINE_ROOT / entry_policy["checkpoint_bundle"]
        cache_exists = cache.exists()
        checkpoint_exists = checkpoint.exists()
        if require_all and (not cache_exists or not checkpoint_exists):
            missing = entry_policy["cache"] if not cache_exists else entry_policy["checkpoint_bundle"]
            raise FileNotFoundError(f"missing staged {dataset} resource bundle: {missing}")
        if not cache_exists or not checkpoint_exists:
            continue
        validate_staged_dataset_resources(dataset, {"datasets": {dataset: entry_policy}})
        cache_id = _bundle_identity(dataset, cache, DATASET_CACHE_SCHEMA)
        checkpoint_id = _bundle_identity(dataset, checkpoint, CHECKPOINT_BUNDLE_SCHEMA)
        staged["datasets"][dataset] = {
            **entry_policy,
            "cache_schema": cache_id["schema"],
            "cache_sha256": cache_id["sha256"],
            "cache_size_bytes": cache_id["size_bytes"],
            "cache_files": cache_id["files"],
            "checkpoint_schema": checkpoint_id["schema"],
            "checkpoint_sha256": checkpoint_id["sha256"],
            "checkpoint_size_bytes": checkpoint_id["size_bytes"],
            "checkpoint_files": checkpoint_id["files"],
        }
    if require_all and set(staged["datasets"]) != set(DATASETS):
        raise FileNotFoundError("all five dataset resource bundles must be staged before setup")
    if not staged["pp"]["pp_id"]:
        raise ValueError("native PP identity is required; invoke setup_resources with formal binary")
    if pp_info is not None:
        if pp_info.get("schema") != PP_INFO_SCHEMA:
            raise ValueError("native PP info schema mismatch")
        if pp_info.get("pp_id") != staged["pp"]["pp_id"]:
            raise ValueError("native PP info id mismatch")
        if int(pp_info.get("max_vector_len", -1)) != int(staged["pp"]["max_vector_len"]):
            raise ValueError("native PP info max_vector_len mismatch")
    staged["manifest_id"] = _manifest_digest(staged)
    return staged


def load_resource_manifest(*, require_staged: bool = True) -> tuple[dict[str, Any], str]:
    """Load the authenticated staged manifest; production defaults fail closed."""
    if not require_staged:
        return load_resource_policy()
    policy, policy_id = load_resource_policy()
    manifest = read_json(STAGED_RESOURCE_MANIFEST_PATH, label="staged resource manifest")
    if manifest.get("schema") != STAGED_RESOURCE_MANIFEST_SCHEMA or manifest.get("baseline") != "zkdl":
        raise ValueError("unsupported zkDL staged resource manifest schema")
    if manifest.get("policy_id") != policy_id:
        raise ValueError("staged resource manifest is for a different source policy")
    if manifest.get("manifest_id") != _manifest_digest(manifest):
        raise ValueError("staged resource manifest digest mismatch")
    pp = manifest.get("pp")
    _validate_pp_spec(pp)
    pp_path = pp_path_from_manifest(manifest)
    validate_pp_file(pp_path, manifest)
    datasets = manifest.get("datasets")
    if not isinstance(datasets, dict) or set(datasets) != set(DATASETS):
        raise ValueError("staged resource manifest must enumerate exactly five datasets")
    for dataset in DATASETS:
        entry = datasets[dataset]
        expected_paths = _validate_dataset_policy_entry(dataset, entry)
        for key, value in expected_paths.items():
            if entry.get(key) != value:
                raise ValueError(f"staged {dataset} {key} does not match source policy")
        validate_staged_dataset_resources(dataset, manifest)
        _validate_bundle_identity(dataset, entry, "cache", DATASET_CACHE_SCHEMA)
        _validate_bundle_identity(dataset, entry, "checkpoint", CHECKPOINT_BUNDLE_SCHEMA)
    return manifest, str(manifest["manifest_id"])


def validate_config_resources(dataset: str, config: dict[str, str], manifest: dict[str, Any]) -> None:
    entry = manifest["datasets"][dataset]
    if config.get("dataset") != dataset:
        raise ValueError(f"config dataset mismatch: {dataset}")
    for key in ("cache", "checkpoint_bundle"):
        if config.get(key) != entry[key]:
            raise ValueError(f"config/resource manifest mismatch for {dataset}: {key}")
        ensure_contained(BASELINE_ROOT / config[key], label=f"{dataset} {key}")


def validate_staged_dataset_resources(dataset: str, manifest: dict[str, Any]) -> None:
    """Require only the selected dataset's local resource bundle."""
    entry = manifest["datasets"][dataset]
    cache = ensure_directory(BASELINE_ROOT / entry["cache"], label=f"{dataset} cache")
    checkpoint = ensure_directory(
        BASELINE_ROOT / entry["checkpoint_bundle"], label=f"{dataset} checkpoint"
    )
    ensure_regular_file(cache / "meta.cfg", label=f"{dataset} cache metadata")
    ensure_regular_file(cache / "edges.txt", label=f"{dataset} edges")
    metadata: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        (cache / "meta.cfg").read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if "=" not in line:
            raise ValueError(f"{dataset} cache metadata line {line_number} is malformed")
        key, value = (part.strip() for part in line.split("=", 1))
        if not key or key in metadata:
            raise ValueError(f"{dataset} cache metadata has duplicate/empty key")
        metadata[key] = value
    if metadata.get("name") != dataset:
        raise ValueError(f"{dataset} cache metadata name is not canonical")
    expected_preprocess = CANONICAL_INPUT_PREPROCESS_IDS.get(dataset)
    if metadata.get("input_preprocess_id") != expected_preprocess:
        raise ValueError(f"{dataset} cache input_preprocess_id is not canonical")
    if dataset == "ogbn-arxiv":
        if not (cache / "features.npy").is_file() or (cache / "features.txt").exists():
            raise ValueError("ogbn-arxiv raw_float32 cache must use only features.npy")
    feature_files = [cache / "features.txt", cache / "features.npy"]
    for feature_file in feature_files:
        if feature_file.exists() or feature_file.is_symlink():
            ensure_regular_file(feature_file, label=f"{dataset} features")
    if not any(feature_file.is_file() for feature_file in feature_files):
        raise FileNotFoundError(f"{dataset} cache has no feature resource")
    ensure_regular_file(checkpoint / "manifest.json", label=f"{dataset} model manifest")
    ensure_regular_file(checkpoint / "tensors.txt", label=f"{dataset} model tensors")


def _validate_bundle_identity(
    dataset: str, entry: dict[str, Any], prefix: str, schema: str
) -> None:
    root_key = "cache" if prefix == "cache" else "checkpoint_bundle"
    records_key = f"{prefix}_files"
    root = ensure_directory(BASELINE_ROOT / entry[root_key], label=f"{dataset} {prefix}")
    actual = _resource_bundle_records(dataset, root)
    if entry.get(f"{prefix}_schema") != schema:
        raise ValueError(f"{dataset} {prefix} schema mismatch")
    if entry.get(records_key) != actual:
        raise ValueError(f"{dataset} {prefix} file inventory mismatch")
    if entry.get(f"{prefix}_size_bytes") != sum(int(item["size_bytes"]) for item in actual):
        raise ValueError(f"{dataset} {prefix} size mismatch")
    if entry.get(f"{prefix}_sha256") != f"sha256:{sha256_bytes(canonical_json(actual))}":
        raise ValueError(f"{dataset} {prefix} digest mismatch")


def resource_inventory_digest(dataset: str, manifest: dict[str, Any]) -> str:
    """Digest staged resource identities without exposing private model data."""
    entry = manifest["datasets"][dataset]
    _validate_bundle_identity(dataset, entry, "cache", DATASET_CACHE_SCHEMA)
    _validate_bundle_identity(dataset, entry, "checkpoint", CHECKPOINT_BUNDLE_SCHEMA)
    records = list(entry["cache_files"]) + list(entry["checkpoint_files"])
    return f"sha256:{sha256_bytes(canonical_json(records))}"


def cell_path(output_root: Path, dataset: str, device: str) -> Path:
    if dataset not in DATASETS or device not in DEVICES:
        raise ValueError(f"invalid cell: {dataset}/{device}")
    output_root = ensure_contained(output_root, label="runs root")
    if output_root == BASELINE_ROOT:
        raise ValueError("runs root must be a baseline-local child directory")
    if output_root.exists() and output_root.is_symlink():
        raise ValueError("runs root may not be a symlink")
    return ensure_contained(output_root / dataset / device / "warm", label="run cell")


def prepare_cell(run: Path, *, replace: bool = False) -> None:
    run = ensure_contained(run, label="run cell")
    if run.exists():
        if run.is_symlink() or not run.is_dir():
            raise ValueError(f"run cell is not a real directory: {run}")
        if any(run.iterdir()):
            if not replace:
                raise FileExistsError(
                    f"run cell already contains artifacts; pass --replace explicitly: {run}"
                )
            # The target is a single validated dataset/device cell, never a
            # workspace or recursive baseline root.
            for child in run.iterdir():
                if child.is_symlink() or child.is_dir():
                    if child.is_symlink():
                        child.unlink()
                    else:
                        import shutil

                        shutil.rmtree(child)
                else:
                    child.unlink()
    else:
        run.mkdir(parents=True, exist_ok=False)
    ensure_directory(run, label="run cell")


def artifact_inventory(run: Path, *, require_complete: bool = False) -> dict[str, dict[str, int | str]]:
    """Return a strict cell inventory and reject symlink/hardlink/special files."""
    run = ensure_directory(run, label="run cell")
    files: dict[str, dict[str, int | str]] = {}
    directories: set[str] = set()
    for path in sorted(run.rglob("*")):
        rel = path.relative_to(run).as_posix()
        if path.is_symlink():
            raise ValueError(f"run artifact symlink is forbidden: {rel}")
        if path.is_dir():
            directories.add(rel)
            continue
        ensure_regular_file(path, label=f"run artifact {rel}")
        if rel in files:
            raise ValueError(f"duplicate run artifact: {rel}")
        files[rel] = {"size": path.stat().st_size, "sha256": sha256_file(path)}
    if not directories.issubset(CELL_DIRS):
        extra = sorted(directories - CELL_DIRS)
        raise ValueError(f"unexpected run artifact directories: {extra}")
    if require_complete:
        missing = sorted(CELL_FILES - files.keys())
        extra = sorted(files.keys() - CELL_FILES)
        if missing or extra:
            raise ValueError(f"cell artifact inventory mismatch missing={missing} extra={extra}")
    return files


def relative_manifest_path(raw: Any, *, run: Path, label: str) -> Path:
    if not isinstance(raw, str) or not raw or Path(raw).is_absolute():
        raise ValueError(f"{label} must be a relative path")
    path = ensure_contained(BASELINE_ROOT / raw, label=label)
    try:
        path.relative_to(run)
    except ValueError as error:
        raise ValueError(f"{label} escapes run cell: {raw}") from error
    return path


def protocol_digest() -> str:
    return f"sha256:{sha256_file(PROTOCOL_PATH)}"


def config_digest(path: Path) -> str:
    return f"sha256:{sha256_file(path)}"
