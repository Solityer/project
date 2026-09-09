#!/usr/bin/env python3
"""One-time model preprocessing for the zkGPT GAT baseline."""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import stat
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from zkgpt_gat.config import generated_config, load_app_config  # noqa: E402
from zkgpt_gat.data import load_dataset, normalize_graph  # noqa: E402
from zkgpt_gat.forward import forward_public_y  # noqa: E402
from zkgpt_gat.formal_gate import require_formal_experiment  # noqa: E402
from zkgpt_gat.model import load_model  # noqa: E402
from zkgpt_gat.proof import _model_arrays, _padded_segment_chunks, make_metadata  # noqa: E402
from zkgpt_gat.reference_forward import (  # noqa: E402
    REFERENCE_FORWARD_ENGINE,
    reference_forward,
    reference_forward_source_sha256,
)
from zkgpt_gat.registry_layout import build_model_registry_layout  # noqa: E402
from zkgpt_gat.registry import (  # noqa: E402
    canonical_registry_public_path,
    canonical_registry_secret_path,
    ensure_model_registry,
)
from zkgpt_gat.resources import (  # noqa: E402
    DATASET_NAMES,
    check_resource_layout,
    canonical_dataset_name,
    read_resource_manifest,
    resolve_local,
    validate_static_table_descriptor,
)
from zkgpt_gat.util import digest_file, digest_json, require_dir  # noqa: E402
from zkgpt_gat.verifier_context import (  # noqa: E402
    _array_digest,
    _x_digest,
    canonical_graph_identity,
    load_public_y,
)
from zkgpt_gat.constants import PUBLIC_Y_SCHEMA, REFERENCE_FORWARD_PROFILE_ID  # noqa: E402


def _public_architecture(manifest: dict) -> dict:
    """Project the checkpoint descriptor to public shape/identity fields."""
    return {
        "model_arch_id": manifest.get("model_arch_id"),
        "model_param_id": manifest.get("model_param_id"),
        "quant_cfg_id": manifest.get("quant_cfg_id"),
        "input_preprocess_id": manifest.get("input_preprocess_id"),
        "rounding_mode": manifest.get("rounding_mode"),
        "static_table_id": manifest.get("static_table_id"),
        "hidden_layers": [
            {
                "layer_index": int(item["layer_index"]),
                "input_dim": int(item["input_dim"]),
                "head_count": int(item["head_count"]),
                "head_dim": int(item["head_dim"]),
            }
            for item in manifest.get("hidden_layers", [])
        ],
        "output_head_specs": [
            {"head_index": int(item["head_index"])}
            for item in manifest.get("output_head_specs", [])
        ],
        "C": int(manifest.get("C", 0)),
        "K_out": int(manifest.get("K_out", 0)),
        "L": int(manifest.get("L", 0)),
        "d_in_profile": [int(value) for value in manifest.get("d_in_profile", [])],
        "output_average_rule": manifest.get("output_average_rule"),
        "dataset": manifest.get("dataset"),
        "quant_scale_chain": manifest.get("quant_scale_chain"),
    }


def _checkpoint_provenance(config, root: pathlib.Path) -> dict:
    bundle = config.checkpoint_bundle
    cache_root = config.cache_dir
    tree = hashlib.sha256()
    if cache_root.is_dir():
        for path in sorted(item for item in cache_root.rglob("*") if item.is_file()):
            relative = path.relative_to(cache_root).as_posix()
            tree.update(relative.encode("utf-8"))
            tree.update(b"\0")
            tree.update(str(path.stat().st_size).encode("ascii"))
            tree.update(b"\0")
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    tree.update(chunk)
            tree.update(b"\n")
    return {
        "manifest_sha256": digest_file(bundle / "manifest.json"),
        "npz_sha256": digest_file(bundle / "tensors.npz"),
        "tensors_sha256": digest_file(bundle / "tensors.txt"),
        "cache_tree_sha256": tree.hexdigest(),
        "resource_identity": str(bundle.relative_to(root)),
    }


def _read_checkpoint_manifest(config) -> dict:
    """Read only the authenticated checkpoint descriptor before tensor I/O."""
    path = resolve_local(
        ROOT,
        config.checkpoint_bundle / "manifest.json",
        "checkpoint manifest",
    )
    if not path.is_file():
        raise FileNotFoundError(f"baseline-local checkpoint manifest is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot parse baseline-local checkpoint manifest: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("checkpoint manifest must be a JSON object")
    return payload


def _validate_registration_identity(config, checkpoint_manifest: dict, static_descriptor: dict) -> None:
    """Fail closed on mode, static-table, architecture and model identities."""
    expected = {
        "model_arch_id": config.entries.get("model_arch_id"),
        "model_param_id": config.entries.get("model_param_id"),
        "static_table_id": config.entries.get("static_table_id"),
        "rounding_mode": config.rounding_mode,
    }
    if config.entries.get("static_tables_id") != static_descriptor.get("static_tables_id"):
        raise ValueError("formal config static_tables_id does not match canonical descriptor")
    if config.entries.get("static_table_id") != static_descriptor.get("static_tables_id"):
        raise ValueError("formal config static_table_id does not match canonical descriptor")
    if config.entries.get("quant_cfg_id") != static_descriptor.get("quant_cfg_id"):
        raise ValueError("formal config quant_cfg_id does not match canonical descriptor")
    if expected["rounding_mode"] != "nearest_ties_away":
        raise ValueError("formal registration requires the authenticated nearest_ties_away mode")
    for key, wanted in expected.items():
        actual = checkpoint_manifest.get(key)
        if actual is None or str(actual) != str(wanted):
            raise ValueError(f"checkpoint manifest {key} does not match formal config")
    d_in_profile = checkpoint_manifest.get("d_in_profile")
    if [int(value) for value in d_in_profile or []] != config.d_in_profile:
        raise ValueError("checkpoint manifest d_in_profile does not match formal config")
    if int(checkpoint_manifest.get("L", -1)) != config.layer_count:
        raise ValueError("checkpoint manifest L does not match formal config")
    hidden = checkpoint_manifest.get("hidden_layers")
    if not isinstance(hidden, list) or len(hidden) != len(config.hidden_profile):
        raise ValueError("checkpoint architecture does not match formal hidden profile")
    actual_hidden = [
        (int(item.get("head_count", -1)), int(item.get("head_dim", -1)))
        for item in sorted(hidden, key=lambda item: int(item.get("layer_index", -1)))
        if isinstance(item, dict)
    ]
    expected_hidden = [(item.head_count, item.head_dim) for item in config.hidden_profile]
    if actual_hidden != expected_hidden:
        raise ValueError("checkpoint hidden head profile does not match formal config")
    if int(checkpoint_manifest.get("K_out", -1)) != config.k_out or int(checkpoint_manifest.get("C", -1)) != config.num_classes:
        raise ValueError("checkpoint output cardinality does not match formal config")


def _validate_registration_paths(config_path: pathlib.Path, config) -> None:
    """Ensure every setup input/output path remains baseline-local."""
    root = require_dir(ROOT, label="registration baseline root")
    paths = {
        "config": config_path,
        "resource_manifest": resolve_local(root, config.entries.get("resource_manifest", ""), "resource_manifest"),
        "data_root": config.data_root,
        "cache_root": config.cache_root,
        "cache_dir": config.cache_dir,
        "checkpoint_bundle": config.checkpoint_bundle,
        "static_table": config.static_table_path,
        "public_y": config.public_y_path,
        "results": resolve_local(root, "results", "results root"),
        "private_setup": resolve_local(root, "artifacts/private_setup", "private setup root"),
    }
    for label, path in paths.items():
        candidate = pathlib.Path(path).expanduser().resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"{label} escapes baseline root: {candidate}") from exc
    model_param_id = str(config.entries.get("model_param_id", ""))
    # Explicitly authenticate both canonical registry trees before any model
    # tensor loading or registration work.  The registry helpers inspect the
    # lexical components first, so an in-baseline alias cannot be hidden by
    # path resolution.
    canonical_registry_public_path(root, model_param_id)
    canonical_registry_secret_path(root, model_param_id)


def _read_canonical_public_y(config, dataset: str) -> tuple[pathlib.Path, dict]:
    """Load only the canonical Y descriptor before setup/model work.

    Registration is not allowed to create or replace public output.  The
    descriptor must already be present at the one generated resource path and
    must be self-consistent before private checkpoint tensors are loaded.
    """

    expected = required_resource_paths(ROOT, dataset)["public_y_path"]
    configured = resolve_local(ROOT, config.entries.get("public_y_path", ""), "public_y_path")
    if configured != expected or config.entries.get("public_y_path", "") != str(expected.relative_to(ROOT)):
        raise ValueError(f"config public_y_path must equal canonical path: {expected.relative_to(ROOT)}")
    try:
        info = expected.lstat()
    except OSError as exc:
        raise ValueError(f"cannot inspect canonical public Y: {expected}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError("canonical public Y must be an owned regular file")
    try:
        payload = json.loads(expected.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read canonical public Y descriptor: {expected}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != PUBLIC_Y_SCHEMA:
        raise ValueError("canonical public Y descriptor schema is unsupported")
    descriptor = dict(payload)
    descriptor_digest = descriptor.pop("descriptor_digest", None)
    if not isinstance(descriptor_digest, str) or descriptor_digest != digest_json(descriptor):
        raise ValueError("canonical public Y descriptor digest is invalid")
    return expected, payload


def registration_preflight(dataset: str):
    """Gate and authenticate all setup inputs before checkpoint tensor I/O."""
    # Keep this exact capability gate as the first executable operation in
    # registration.  A blocked capability manifest must not read a checkpoint
    # or create either side of the model registry.
    require_formal_experiment()
    dataset = canonical_dataset_name(dataset)
    if dataset not in DATASET_NAMES:
        raise ValueError(f"unsupported formal dataset: {dataset}")
    check_resource_layout(ROOT, [dataset], require_staged=True, require_public_y=True)
    manifest = read_resource_manifest(ROOT)
    config_path = generated_config(dataset, "cpu", ROOT, ROOT)
    config = load_app_config(config_path, ROOT)
    if canonical_dataset_name(config.dataset) != dataset:
        raise ValueError("generated formal config dataset does not match registration request")
    _validate_registration_paths(config_path, config)
    _read_canonical_public_y(config, dataset)
    descriptor = validate_static_table_descriptor(ROOT)
    checkpoint_manifest = _read_checkpoint_manifest(config)
    _validate_registration_identity(config, checkpoint_manifest, descriptor)
    return config, checkpoint_manifest, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Register canonical zkGPT GAT model weights")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--project-root", type=pathlib.Path, default=ROOT)
    parser.add_argument("--zk-root", type=pathlib.Path, default=ROOT)
    args = parser.parse_args()
    project_root = require_dir(args.project_root, label="registration project root")
    zk_root = require_dir(args.zk_root, label="registration zkGPT root")
    expected_root = require_dir(ROOT, label="registration baseline root")
    if project_root != expected_root or zk_root != expected_root:
        raise SystemExit("model registration is baseline-local; project/zk roots must equal comparison/zkGPT")
    config, _checkpoint_manifest, _resource_manifest = registration_preflight(args.dataset)
    dataset = load_dataset(config)
    graph = normalize_graph(dataset, config)
    model = load_model(
        config.checkpoint_bundle,
        model_arch_id=config.entries.get("model_arch_id"),
        model_param_id=config.entries.get("model_param_id"),
        quant_cfg_id=config.entries.get("quant_cfg_id"),
        static_table_id=config.entries.get("static_table_id"),
        rounding_mode=config.rounding_mode,
    )
    public_y_path, public_y_descriptor = _read_canonical_public_y(
        config, canonical_dataset_name(config.dataset)
    )
    canonical_y = np.asarray(load_public_y(public_y_path), dtype=object)
    expected_shape = (int(graph.num_nodes), int(model.class_count))
    if tuple(canonical_y.shape) != expected_shape:
        raise ValueError(f"canonical public Y shape {tuple(canonical_y.shape)} != {expected_shape}")
    native_y = np.asarray(forward_public_y(graph, model, config.rounding_mode), dtype=object)
    reference_y = np.asarray(reference_forward(graph, model, config.rounding_mode), dtype=object)
    if not np.array_equal(native_y, reference_y):
        raise ValueError("native and independent reference forward outputs differ")
    if not np.array_equal(canonical_y, native_y):
        raise ValueError("canonical public Y differs from native/reference forward output")
    y_digest = _array_digest("Y", canonical_y)
    x_digest = _x_digest(
        graph,
        int(len(model.hidden_layers) + 1),
        int(model.class_count),
        int(model.k_out),
        int(model.quant_scales["input"]),
    )
    graph_identity = canonical_graph_identity(
        graph,
        canonical_dataset_name(config.dataset),
        full_dataset=graph.num_nodes == dataset.num_nodes,
    )
    descriptor_digest = str(public_y_descriptor.get("descriptor_digest", ""))
    expected_descriptor_fields = {
        "schema": PUBLIC_Y_SCHEMA,
        "public_y_schema": PUBLIC_Y_SCHEMA,
        "public_y_path": str(public_y_path.relative_to(ROOT)),
        "public_y_shape": [int(value) for value in expected_shape],
        "y_digest": y_digest,
        "reference_y_digest": y_digest,
        "independent_reference_forward": True,
        "reference_forward_profile_id": REFERENCE_FORWARD_PROFILE_ID,
        "reference_forward_engine": REFERENCE_FORWARD_ENGINE,
        "reference_forward_source_sha256": reference_forward_source_sha256(),
        "X_pub_digest": x_digest,
        "full_dataset": True,
        "graph_identity": graph_identity,
    }
    for key, expected in expected_descriptor_fields.items():
        if public_y_descriptor.get(key) != expected:
            raise ValueError(f"canonical public Y descriptor mismatch: {key}")
    if not descriptor_digest:
        raise ValueError("canonical public Y descriptor digest is missing")
    # Registration commits only the canonical model weights.  Derive the
    # proof-equivalent layout and integer-bound ledger from graph/model shapes
    # instead of executing the complete dataset forward trace during setup.
    input_layout, _ledger = build_model_registry_layout(config, dataset, graph, model)
    metadata = make_metadata(config, dataset, graph, model, "cpu", {"backend": "cpu"}, input_layout)
    metadata["model_architecture"] = _public_architecture(_checkpoint_manifest)
    metadata["checkpoint_provenance"] = _checkpoint_provenance(config, ROOT)
    metadata.update(
        {
            "public_y_schema": PUBLIC_Y_SCHEMA,
            "public_y_path": str(public_y_path.relative_to(ROOT)),
            "public_y_digest": y_digest,
            "public_output_digest": y_digest,
            "public_y_descriptor_digest": descriptor_digest,
            "public_y_shape": [int(value) for value in expected_shape],
            "reference_y_digest": y_digest,
            "independent_reference_forward": True,
            "reference_forward_profile_id": REFERENCE_FORWARD_PROFILE_ID,
            "reference_forward_engine": REFERENCE_FORWARD_ENGINE,
            "reference_forward_source_sha256": reference_forward_source_sha256(),
            "X_pub_digest": x_digest,
            "full_dataset": True,
            "graph_identity": graph_identity,
        }
    )
    w_values = [
        int(value)
        for _, array in _model_arrays(model)
        for value in array.ravel().tolist()
    ]
    n_in = int(input_layout["N_in"])
    if len(w_values) > n_in:
        raise ValueError("canonical W vector exceeds shape-only input layout")
    w_values.extend([0] * (n_in - len(w_values)))
    registry = ensure_model_registry(ROOT, metadata, list(_padded_segment_chunks("com_W", w_values)))
    print(registry["public_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
