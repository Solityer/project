#!/usr/bin/env python3
"""Generate baseline-local integer public-Y resources without proving.

This is deliberately a resource-only entry point.  It loads one canonical
zkGPT config/cache/checkpoint, executes the native integer public-output
relation, and atomically publishes only the corresponding public output.  It
never imports the proof builder, model registry, setup code, or runner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import stat
import sys
import time
from typing import Any, Iterable

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from zkgpt_gat.config import generated_config, load_app_config  # noqa: E402
from zkgpt_gat.constants import (  # noqa: E402
    DATASET_ORDER,
    FORMAL_GAT_CLASS_COUNTS,
    FORMAL_GAT_D_IN_PROFILE,
    FORMAL_GAT_HIDDEN_PROFILE,
    FORMAL_GAT_K_OUT,
    FORMAL_GAT_LAYER_COUNT,
    FORMAL_GAT_MODEL_ARCH_IDS,
    PUBLIC_Y_SCHEMA,
    REFERENCE_FORWARD_PROFILE_ID,
)
from zkgpt_gat.data import load_dataset, normalize_graph  # noqa: E402
from zkgpt_gat.forward import forward_public_y  # noqa: E402
from zkgpt_gat.reference_forward import REFERENCE_FORWARD_ENGINE, reference_forward, reference_forward_source_sha256  # noqa: E402
from zkgpt_gat.model import load_model  # noqa: E402
from zkgpt_gat.resources import (  # noqa: E402
    canonical_dataset_name,
    check_resource_layout,
    required_resource_paths,
)
from zkgpt_gat.util import digest_file, digest_json, write_json  # noqa: E402
from zkgpt_gat.verifier_context import _x_digest, canonical_graph_identity, load_public_y  # noqa: E402


def _owned_regular(path: pathlib.Path, label: str) -> pathlib.Path:
    try:
        info = path.lstat()
    except OSError as exc:
        raise RuntimeError(f"cannot inspect {label}: {path}") from exc
    if stat.S_ISLNK(info.st_mode):
        raise RuntimeError(f"{label} must not be a symlink: {path}")
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise RuntimeError(f"{label} must be an owned regular file: {path}")
    return path


def _tree_digest(root: pathlib.Path, label: str) -> str:
    """Hash every owned cache file in deterministic relative-path order."""
    root = root.resolve()
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"{label} must be a real directory: {root}")
    records: list[pathlib.Path] = []
    for current, directories, files in os.walk(root, followlinks=False):
        current_path = pathlib.Path(current)
        for name in directories:
            directory = current_path / name
            if directory.is_symlink():
                raise RuntimeError(f"{label} contains a symlink directory: {directory}")
        for name in files:
            records.append(current_path / name)
    records.sort(key=lambda item: item.relative_to(root).as_posix())
    digest = hashlib.sha256()
    for path in records:
        _owned_regular(path, f"{label} file")
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(str(path.stat().st_size).encode("ascii"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\n")
    return digest.hexdigest()


def _array_digest(values: np.ndarray) -> str:
    """Match the independent verifier's object-array Y digest exactly."""
    array = np.ascontiguousarray(np.asarray(values, dtype=object))
    digest = hashlib.sha256()
    digest.update(b"Y")
    digest.update(str(array.shape).encode("ascii"))
    digest.update(str(array.dtype).encode("ascii"))
    for value in array.ravel().tolist():
        digest.update(str(int(value)).encode("ascii"))
        digest.update(b",")
    return digest.hexdigest()


def _checkpoint_manifest(bundle: pathlib.Path) -> dict[str, Any]:
    path = _owned_regular(bundle / "manifest.json", "checkpoint manifest")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot parse checkpoint manifest: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"checkpoint manifest must be an object: {path}")
    return payload


def _expected_contract(config, dataset_name: str, graph, model, raw_manifest: dict[str, Any]) -> dict[str, Any]:
    canonical = canonical_dataset_name(dataset_name)
    expected_d_in = list(FORMAL_GAT_D_IN_PROFILE[canonical])
    expected_hidden = [
        {"head_count": int(item.head_count), "head_dim": int(item.head_dim)}
        for item in model.hidden_layers
    ]
    expected = {
        "dataset": canonical,
        "config_dataset": config.dataset,
        "model_arch_id": FORMAL_GAT_MODEL_ARCH_IDS[canonical],
        "model_param_id": config.entries["model_param_id"],
        "quant_cfg_id": config.entries["quant_cfg_id"],
        "source_quant_cfg_id": raw_manifest.get("quant_cfg_id"),
        "rounding_mode": config.rounding_mode,
        "L": FORMAL_GAT_LAYER_COUNT,
        "hidden_profile": expected_hidden,
        "d_in_profile": expected_d_in,
        "K_out": FORMAL_GAT_K_OUT,
        "C": FORMAL_GAT_CLASS_COUNTS[canonical],
        "shape": [int(graph.num_nodes), int(model.class_count)],
        "quant_scale_chain": {key: int(value) for key, value in model.quant_scales.items()},
        "input_preprocess_id": config.input_preprocess_id,
        "canonical_scale_chain": raw_manifest.get("quant_scale_chain"),
        "output_average_rule": model.output_average_rule,
    }
    if expected["model_arch_id"] != model.model_arch_id:
        raise RuntimeError("loaded model architecture is not the canonical L3 identity")
    if expected["model_param_id"] != model.model_param_id:
        raise RuntimeError("loaded model parameter identity differs from formal config")
    if expected["quant_cfg_id"] != model.quant_cfg_id:
        raise RuntimeError("loaded model quantization identity differs from formal config")
    if model.rounding_mode != config.rounding_mode:
        raise RuntimeError("loaded model rounding mode differs from formal config")
    if model.manifest.get("output_average_rule") != "per_head_bias_then_arithmetic_mean":
        raise RuntimeError("checkpoint output-average rule is not canonical")
    return expected


def _validate_public_y(path: pathlib.Path, contract: dict[str, Any], values: np.ndarray) -> dict[str, Any]:
    """Reload through the independent public-Y loader and verify the contract."""
    _owned_regular(path, "public Y")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot reload generated public Y: {path}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != PUBLIC_Y_SCHEMA:
        raise RuntimeError("generated public Y schema is invalid")
    loaded = load_public_y(path)
    expected_shape = tuple(int(item) for item in contract["shape"])
    if tuple(loaded.shape) != expected_shape or tuple(values.shape) != expected_shape:
        raise RuntimeError("generated public Y shape is not bound to the model contract")
    if not np.array_equal(np.asarray(loaded, dtype=object), np.asarray(values, dtype=object)):
        raise RuntimeError("generated public Y reload differs from native integer forward")
    expected_digest = _array_digest(values)
    if payload.get("y_digest") != expected_digest:
        raise RuntimeError("generated public Y digest does not match its integer values")
    if payload.get("reference_y_digest") != expected_digest or payload.get("independent_reference_forward") is not True:
        raise RuntimeError("generated public Y lacks independent reference-forward evidence")
    if payload.get("public_y_schema") != PUBLIC_Y_SCHEMA or payload.get("public_y_shape") != [int(item) for item in values.shape]:
        raise RuntimeError("generated public Y canonical schema/shape binding is invalid")
    if payload.get("public_y_path") != str(path.relative_to(ROOT)):
        raise RuntimeError("generated public Y path binding is not canonical")
    if payload.get("reference_forward_profile_id") != REFERENCE_FORWARD_PROFILE_ID:
        raise RuntimeError("generated public Y reference-forward profile is not canonical")
    descriptor = dict(payload)
    descriptor_digest = descriptor.pop("descriptor_digest", None)
    if descriptor_digest != digest_json(descriptor):
        raise RuntimeError("generated public Y descriptor digest is invalid")
    for key, expected in contract.items():
        if payload.get(key) != expected:
            raise RuntimeError(f"generated public Y contract mismatch: {key}")
    return payload


def generate_one(dataset_name: str) -> dict[str, Any]:
    canonical = canonical_dataset_name(dataset_name)
    check_resource_layout(ROOT, [canonical], require_staged=True, require_public_y=False)
    config_path = generated_config(canonical, "cpu", ROOT, ROOT)
    config = load_app_config(config_path, ROOT)
    paths = required_resource_paths(ROOT, canonical)
    cache_digest = _tree_digest(paths["cache_dir"], f"{canonical} cache")
    manifest_path = _owned_regular(paths["checkpoint_bundle"] / "manifest.json", "checkpoint manifest")
    npz_path = _owned_regular(paths["checkpoint_bundle"] / "tensors.npz", "checkpoint tensors.npz")
    tensors_path = _owned_regular(paths["checkpoint_bundle"] / "tensors.txt", "checkpoint tensors")
    raw_manifest = _checkpoint_manifest(paths["checkpoint_bundle"])
    checkpoint_manifest_digest = digest_file(manifest_path)
    checkpoint_npz_digest = digest_file(npz_path)
    checkpoint_tensors_digest = digest_file(tensors_path)

    dataset = load_dataset(config)
    graph = normalize_graph(dataset, config)
    if graph.num_nodes != dataset.num_nodes:
        raise RuntimeError(f"{canonical} resource-only forward is not full-dataset")
    model = load_model(
        config.checkpoint_bundle,
        model_arch_id=config.entries.get("model_arch_id"),
        model_param_id=config.entries.get("model_param_id"),
        quant_cfg_id=config.entries.get("quant_cfg_id"),
        static_table_id=config.entries.get("static_table_id"),
        rounding_mode=config.rounding_mode,
    )
    contract = _expected_contract(config, canonical, graph, model, raw_manifest)
    started = time.perf_counter()
    # This is the native zkGPT integer relation with proof-only advice
    # discarded as it is produced.  It is mathematically identical to
    # ``forward_full`` (validated against the complete Cora trace) but keeps
    # PPI/OGBN resources bounded instead of retaining every edge/head trace.
    values = np.asarray(forward_public_y(graph, model, config.rounding_mode), dtype=object)
    reference_values = np.asarray(reference_forward(graph, model, config.rounding_mode), dtype=object)
    if not np.array_equal(values, reference_values):
        raise RuntimeError(f"{canonical} production/reference integer forward mismatch")
    if values.ndim != 2 or tuple(values.shape) != tuple(contract["shape"]):
        raise RuntimeError(f"{canonical} native integer forward returned an unexpected shape: {values.shape}")
    if any(not isinstance(value, (int, np.integer)) for value in values.ravel().tolist()):
        raise RuntimeError(f"{canonical} native integer forward returned a non-integer value")

    payload: dict[str, Any] = {
        "schema": PUBLIC_Y_SCHEMA,
        **contract,
        "cache_dir": str(paths["cache_dir"].relative_to(ROOT)),
        "cache_tree_sha256": cache_digest,
        "checkpoint_bundle": str(paths["checkpoint_bundle"].relative_to(ROOT)),
        "checkpoint_manifest_sha256": checkpoint_manifest_digest,
        "checkpoint_npz_sha256": checkpoint_npz_digest,
        "checkpoint_tensors_sha256": checkpoint_tensors_digest,
        "graph_count": int(graph.graph_count),
        "edge_count": int(graph.src.size),
        "input_feature_rounding": "nearest_ties_away",
        "forward_engine": "zkgpt_gat.forward.forward_public_y",
        "reference_forward_engine": REFERENCE_FORWARD_ENGINE,
        "reference_forward_source_sha256": reference_forward_source_sha256(),
        "reference_forward_profile_id": REFERENCE_FORWARD_PROFILE_ID,
        "public_y_schema": PUBLIC_Y_SCHEMA,
        "public_y_path": str(paths["public_y_path"].relative_to(ROOT)),
        "public_y_shape": [int(value) for value in values.shape],
        "X_pub_digest": _x_digest(
            graph,
            int(model.layer_count if hasattr(model, "layer_count") else len(model.hidden_layers) + 1),
            int(model.class_count),
            int(model.k_out),
            int(model.quant_scales["input"]),
        ),
        "full_dataset": bool(graph.num_nodes == dataset.num_nodes),
        "graph_identity": canonical_graph_identity(
            graph,
            canonical,
            full_dataset=graph.num_nodes == dataset.num_nodes,
        ),
        "reference_y_digest": _array_digest(reference_values),
        "independent_reference_forward": bool(np.array_equal(values, reference_values)),
        "y_digest": _array_digest(values),
        "values": [int(value) for value in values.ravel().tolist()],
    }
    payload["descriptor_digest"] = digest_json(payload)
    output_path = paths["public_y_path"]
    # Public Y is a trusted setup input.  A second invocation must never
    # replace the registered statement (even with an atomically written file);
    # the operator may remove an explicitly invalid staging result before a
    # fresh setup, but this script itself is non-overwriting.
    if os.path.lexists(output_path):
        _owned_regular(output_path, f"existing {canonical} public Y")
        raise RuntimeError(f"canonical public Y already exists; refusing overwrite: {output_path}")
    write_json(output_path, payload, root=ROOT, label=f"{canonical} public Y")
    _validate_public_y(output_path, payload | {"descriptor_digest": payload["descriptor_digest"]}, values)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "dataset": canonical,
        "path": str(output_path.relative_to(ROOT)),
        "shape": list(values.shape),
        "y_digest": payload["y_digest"],
        "descriptor_digest": payload["descriptor_digest"],
        "cache_tree_sha256": cache_digest,
        "checkpoint_manifest_sha256": checkpoint_manifest_digest,
        "checkpoint_npz_sha256": checkpoint_npz_digest,
        "checkpoint_tensors_sha256": checkpoint_tensors_digest,
        "elapsed_forward_ms": elapsed_ms,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate zkGPT baseline-local integer public Y without proving")
    parser.add_argument("--datasets", default="all", help="comma-separated canonical datasets or all")
    args = parser.parse_args()
    if args.datasets == "all":
        selected = DATASET_ORDER[:]
    else:
        selected = [item.strip() for item in args.datasets.split(",") if item.strip()]
    unknown = sorted(set(selected) - set(DATASET_ORDER))
    if unknown:
        parser.error(f"unsupported dataset(s): {unknown}")
    for dataset in selected:
        result = generate_one(dataset)
        print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
