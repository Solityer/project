#!/usr/bin/env python3
"""Produce the baseline-local canonical no-bias zkGPT checkpoint bundle.

The historical ``*_gat`` bundles are intentionally not accepted by the
formal loader: they contain hidden/output attention-bias fields and an older
scale identity.  This preparation stage reads those baseline-local source
bundles, keeps only the authenticated kernels and output bias, and writes a
new content-described bundle under ``*_gat_canonical``.  It never invents
weights, copies a public output, or reads an external project path.

The generated bundle is a cache.  Its manifest records the source bundle and
both source digests, the exact input-cache digest, the complete canonical
scale chain, and every emitted tensor digest.  Any existing bundle with an
invalid or mismatched identity fails closed instead of being silently reused.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import pathlib
import stat
import sys
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from zkgpt_gat.config import generated_config, load_app_config  # noqa: E402
from zkgpt_gat.constants import DATASET_ORDER, FORMAL_GAT_CLASS_COUNTS, FORMAL_GAT_MODEL_ARCH_IDS  # noqa: E402
from zkgpt_gat.model import CANONICAL_CHECKPOINT_SCHEMA, _load_npz_tensors, _load_text_tensors, _tensor_digest, _validate_canonical_manifest, _validate_tensor_index  # noqa: E402
from zkgpt_gat.quantization import (  # noqa: E402
    CANONICAL_QUANT_CFG_ID,
    canonical_input_preprocess_id,
    canonical_scale_chain_descriptor,
)
from zkgpt_gat.resources import canonical_dataset_name, required_resource_paths  # noqa: E402
from zkgpt_gat.rounding import validate_rounding_mode  # noqa: E402
from zkgpt_gat.util import digest_file, digest_json, ensure_dir, write_bytes_atomic, write_json, write_text_atomic  # noqa: E402


def _owned_regular(path: pathlib.Path, label: str) -> pathlib.Path:
    try:
        info = path.lstat()
    except OSError as exc:
        raise RuntimeError(f"cannot inspect {label}: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise RuntimeError(f"{label} must be an owned regular file: {path}")
    return path


def _tree_digest(root: pathlib.Path, label: str) -> str:
    root = root.resolve()
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"{label} must be a real directory: {root}")
    files: list[pathlib.Path] = []
    for current, directories, names in os.walk(root, followlinks=False):
        current_path = pathlib.Path(current)
        for name in directories:
            if (current_path / name).is_symlink():
                raise RuntimeError(f"{label} contains a symlink directory: {current_path / name}")
        files.extend(current_path / name for name in names)
    files.sort(key=lambda item: item.relative_to(root).as_posix())
    digest = hashlib.sha256()
    for path in files:
        _owned_regular(path, f"{label} file")
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\n")
    return digest.hexdigest()


def _source_manifest(path: pathlib.Path) -> dict[str, Any]:
    _owned_regular(path, "source checkpoint manifest")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot parse source checkpoint manifest: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError("source checkpoint manifest must be an object")
    return value


def _tensor_text(selected: list[tuple[str, np.ndarray]]) -> str:
    lines: list[str] = []
    for name, array in selected:
        values = np.asarray(array, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise RuntimeError(f"source tensor is non-finite: {name}")
        shape = [int(value) for value in values.shape]
        lines.append("TENSOR " + name + " " + str(len(shape)) + " " + " ".join(map(str, shape)) + " " + str(int(values.size)))
        lines.append(" ".join(format(float(value), ".17g") for value in values.ravel(order="C")))
    return "\n".join(lines) + "\n"


def _canonical_tensor_names(source: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    hidden_specs = source.get("hidden_head_specs")
    output_specs = source.get("output_head_specs")
    if not isinstance(hidden_specs, list) or not isinstance(output_specs, list):
        raise RuntimeError("source checkpoint head specs are missing")
    hidden_specs = sorted(hidden_specs, key=lambda item: (int(item["layer_index"]), int(item["local_head_index"])))
    output_specs = sorted(output_specs, key=lambda item: int(item["head_index"]))
    if len(hidden_specs) != 16 or len(output_specs) != 2:
        raise RuntimeError("source checkpoint is not the canonical [8x8,8x8], K_out=2 shape")
    selected_hidden: list[dict[str, Any]] = []
    names: list[str] = []
    for spec in hidden_specs:
        required = {"layer_index", "local_head_index", "seq_kernel", "attn_src_kernel", "attn_dst_kernel"}
        if not required <= set(spec):
            raise RuntimeError("source hidden head is missing a kernel")
        selected_hidden.append({key: spec[key] for key in sorted(required)})
        names.extend([spec["seq_kernel"], spec["attn_src_kernel"], spec["attn_dst_kernel"]])
    selected_output: list[dict[str, Any]] = []
    for spec in output_specs:
        required = {"head_index", "seq_kernel", "attn_src_kernel", "attn_dst_kernel", "output_bias"}
        if not required <= set(spec):
            raise RuntimeError("source output head is missing a kernel/output bias")
        selected_output.append({key: spec[key] for key in sorted(required)})
        names.extend([spec["seq_kernel"], spec["attn_src_kernel"], spec["attn_dst_kernel"], spec["output_bias"]])
    return selected_hidden, selected_output, names


def _safe_existing_bundle(bundle: pathlib.Path) -> None:
    if not bundle.exists() and not os.path.lexists(bundle):
        return
    if bundle.is_symlink() or not bundle.is_dir():
        raise RuntimeError(f"canonical checkpoint path must be a real directory: {bundle}")
    for child in bundle.iterdir():
        info = child.lstat()
        if child.is_symlink() or stat.S_ISDIR(info.st_mode) or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise RuntimeError(f"canonical checkpoint contains an unsafe entry: {child}")
        if child.name not in {"manifest.json", "tensors.npz", "tensors.txt"}:
            raise RuntimeError(f"canonical checkpoint contains an unsupported entry: {child}")


def _tensor_npz(tensors: list[tuple[str, np.ndarray]]) -> bytes:
    buffer = io.BytesIO()
    arrays = {name.replace("/", "__").replace(":", "__"): np.ascontiguousarray(array, dtype=np.float32) for name, array in tensors}
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def generate_one(dataset_name: str) -> dict[str, Any]:
    dataset = canonical_dataset_name(dataset_name)
    config_path = generated_config(dataset, "cpu", ROOT, ROOT)
    config = load_app_config(config_path, ROOT)
    paths = required_resource_paths(ROOT, dataset)
    source_bundle = ROOT / "artifacts" / "checkpoints" / f"{paths['checkpoint_bundle'].name.removesuffix('_canonical')}"
    source_manifest_path = _owned_regular(source_bundle / "manifest.json", "source checkpoint manifest")
    source_tensors_path = _owned_regular(source_bundle / "tensors.txt", "source checkpoint tensors")
    source = _source_manifest(source_manifest_path)
    source_hidden, source_output, names = _canonical_tensor_names(source)
    source_tensors = _load_text_tensors(source_tensors_path)
    missing = sorted(set(names) - set(source_tensors))
    if missing:
        raise RuntimeError(f"source tensor inventory is missing canonical kernels: {missing[:3]}")
    # The source bundle is intentionally bias-enabled; its attention and hidden
    # output-bias entries are not copied.  Only the output-layer bias is legal
    # in the canonical relation.
    selected: list[tuple[str, np.ndarray]] = []
    for name in names:
        selected.append((name, np.asarray(source_tensors[name], dtype=np.float32)))
    index: dict[str, Any] = {}
    for name, array in selected:
        index[name] = {
            "archive_key": name.replace("/", "__").replace(":", "__"),
            "dtype": "float32",
            "shape": [int(value) for value in array.shape],
            "count": int(array.size),
            "sha256": _tensor_digest(array),
        }
    manifest = {
        "schema": CANONICAL_CHECKPOINT_SCHEMA,
        "dataset": str(config.dataset),
        "model_arch_id": str(config.entries["model_arch_id"]),
        "model_param_id": str(config.entries["model_param_id"]),
        "quant_cfg_id": CANONICAL_QUANT_CFG_ID,
        "input_preprocess_id": config.input_preprocess_id,
        "rounding_mode": validate_rounding_mode(config.rounding_mode),
        "static_table_id": str(config.entries["static_table_id"]),
        "quant_scale_chain": canonical_scale_chain_descriptor(hidden_profile=[{"head_count": 8, "head_dim": 8}, {"head_count": 8, "head_dim": 8}], k_out=2),
        "L": 3,
        "hidden_layers": [
            {"layer_index": int(item["layer_index"]), "input_dim": int(layer["input_dim"]), "head_count": int(layer["head_count"]), "head_dim": int(layer["head_dim"])}
            for item, layer in zip(source_hidden[::8], sorted(source.get("hidden_layers", []), key=lambda value: int(value["layer_index"])))
        ],
        "hidden_profile": [{"head_count": 8, "head_dim": 8}, {"head_count": 8, "head_dim": 8}],
        "d_in_profile": [int(value) for value in source.get("d_in_profile", [])],
        "C": int(config.num_classes),
        "K_out": 2,
        "hidden_head_specs": source_hidden,
        "output_head_specs": source_output,
        "output_average_rule": "per_head_bias_then_arithmetic_mean",
        "tensor_count": len(index),
        "tensor_index": index,
        "tensor_index_digest": digest_json(index),
        "training_procedure_id": "baseline_local_source_checkpoint_no_bias_projection_v1",
        "source_data_digest": _tree_digest(paths["cache_dir"], f"{dataset} cache"),
        "source_bundle_path": str(source_bundle.relative_to(ROOT)),
        "source_manifest_sha256": digest_file(source_manifest_path),
        "source_tensors_sha256": digest_file(source_tensors_path),
        "conversion_policy": "drop_hidden_and_attention_bias_terms; retain_output_bias_only",
        "family_schema_version": "multi_layer_multi_head_v2_no_bias",
        "task_type": source.get("task_type", config.entries.get("task_type", "transductive_node_classification")),
        "report_unit": source.get("report_unit", config.entries.get("report_unit", "node")),
        "batching_rule": source.get("batching_rule", config.entries.get("batching_rule", "whole_graph_single")),
        "self_loop_rule": source.get("self_loop_rule", config.entries.get("self_loop_rule", "per_node")),
        "edge_sort_rule": source.get("edge_sort_rule", config.entries.get("edge_sort_rule", "edge_gid_then_dst_stable")),
        "chunking_rule": config.entries.get("chunking_rule", "none"),
        "label_format": source.get("label_format", "single_label"),
        "split_format": source.get("split_format", "index_file"),
        "degree_bound_id": source.get("degree_bound_id", "auto"),
    }
    if manifest["d_in_profile"] != [int(value) for value in config.d_in_profile]:
        raise RuntimeError("source checkpoint d_in_profile does not match formal config")
    _validate_canonical_manifest(manifest)
    text = _tensor_text(selected)
    output_bundle = paths["checkpoint_bundle"]
    _safe_existing_bundle(output_bundle)
    ensure_dir(output_bundle, root=ROOT, label=f"{dataset} canonical checkpoint")
    existing_manifest = output_bundle / "manifest.json"
    existing_npz = output_bundle / "tensors.npz"
    existing_tensors = output_bundle / "tensors.txt"
    npz_payload = _tensor_npz(selected)
    if existing_manifest.exists() or existing_npz.exists() or existing_tensors.exists():
        if not existing_manifest.is_file() or not existing_npz.is_file() or not existing_tensors.is_file():
            raise RuntimeError(f"canonical checkpoint cache is partial: {output_bundle}")
        try:
            old = json.loads(existing_manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"canonical checkpoint cache manifest is invalid: {output_bundle}") from exc
        if old != manifest or existing_tensors.read_text(encoding="utf-8") != text:
            raise RuntimeError(f"canonical checkpoint cache identity differs from current source: {output_bundle}")
        try:
            existing_text_tensors = _load_text_tensors(existing_tensors)
            _validate_tensor_index(old, existing_text_tensors)
            npz_tensors = _load_npz_tensors(existing_npz, old)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"canonical checkpoint cache tensors are invalid: {output_bundle}") from exc
        for name, array in existing_text_tensors.items():
            if npz_tensors[name].tobytes(order="C") != np.asarray(array, dtype=np.float32).tobytes(order="C"):
                raise RuntimeError(f"canonical checkpoint cache NPZ/TXT mismatch: {name}")
    else:
        write_text_atomic(existing_tensors, text, root=ROOT, label=f"{dataset} canonical checkpoint tensors")
        write_bytes_atomic(existing_npz, npz_payload, root=ROOT, label=f"{dataset} canonical checkpoint tensors.npz")
        write_json(existing_manifest, manifest, root=ROOT, label=f"{dataset} canonical checkpoint manifest")
    return {"dataset": dataset, "bundle": str(output_bundle.relative_to(ROOT)), "tensor_count": len(index), "manifest_digest": digest_file(existing_manifest), "tensor_digest": digest_file(existing_tensors), "npz_digest": digest_file(existing_npz)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare baseline-local canonical no-bias zkGPT checkpoint bundles")
    parser.add_argument("--datasets", default="all")
    args = parser.parse_args()
    selected = DATASET_ORDER[:] if args.datasets == "all" else [canonical_dataset_name(item.strip()) for item in args.datasets.split(",") if item.strip()]
    for dataset in selected:
        print(json.dumps(generate_one(dataset), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
