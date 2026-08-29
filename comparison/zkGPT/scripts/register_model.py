#!/usr/bin/env python3
"""One-time model preprocessing for the zkGPT GAT baseline."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from zkgpt_gat.config import generated_config, load_app_config  # noqa: E402
from zkgpt_gat.data import load_dataset, normalize_graph  # noqa: E402
from zkgpt_gat.formal_gate import require_formal_experiment  # noqa: E402
from zkgpt_gat.forward import forward_full  # noqa: E402
from zkgpt_gat.model import load_model  # noqa: E402
from zkgpt_gat.proof import _padded_segment_chunks, make_metadata, prepare_input_segments_and_layout  # noqa: E402
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
from zkgpt_gat.util import require_dir  # noqa: E402


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
        # The shared checkpoint exporter does not serialize the formal
        # SignedRoundDiv mode; the zkGPT config is the authenticated owner of
        # that choice.  All formal configs use the descriptor's explicit
        # nearest-ties-away mode, so absence is accepted only for this source
        # manifest field.  The source quantization identity is likewise
        # mapped to the baseline's exact-integer scale profile below.
        if key == "rounding_mode" and actual is None and str(wanted) == "nearest_ties_away":
            continue
        if key == "quant_cfg_id" and str(actual) == "fp32_torch_checkpoint" and str(wanted) == "exact_integer_gat_v1":
            continue
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


def registration_preflight(dataset: str):
    """Gate and authenticate all setup inputs before checkpoint tensor I/O."""
    # Keep this exact capability gate as the first executable operation in
    # registration.  A blocked capability manifest must not read a checkpoint
    # or create either side of the model registry.
    require_formal_experiment()
    dataset = canonical_dataset_name(dataset)
    if dataset not in DATASET_NAMES:
        raise ValueError(f"unsupported formal dataset: {dataset}")
    check_resource_layout(ROOT, [dataset], require_staged=True, require_public_y=False)
    manifest = read_resource_manifest(ROOT)
    config_path = generated_config(dataset, "cpu", ROOT, ROOT)
    config = load_app_config(config_path, ROOT)
    if canonical_dataset_name(config.dataset) != dataset:
        raise ValueError("generated formal config dataset does not match registration request")
    _validate_registration_paths(config_path, config)
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
    trace = forward_full(graph, model, config.rounding_mode)
    _, input_layout, padded_segments = prepare_input_segments_and_layout(config, dataset, graph, model, trace)
    metadata = make_metadata(config, dataset, graph, model, "cpu", {"backend": "cpu"}, input_layout)
    registry = ensure_model_registry(ROOT, metadata, list(_padded_segment_chunks("com_W", padded_segments["W"])))
    print(registry["public_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
