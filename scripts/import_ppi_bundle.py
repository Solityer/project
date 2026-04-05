#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import tempfile
from typing import Dict, Iterable

import numpy as np

from checkpoint_reader_compat import read_checkpoint_tensors
from export_gat_checkpoint import archive_key, build_manifest, write_text_tensor_dump
from export_torch_gat_bundle import build_tensors_and_manifest_spec


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_manifest(bundle_dir: pathlib.Path) -> Dict[str, object]:
    manifest_path = bundle_dir / "manifest.json"
    require(manifest_path.exists(), f"缺少 manifest.json: {bundle_dir}")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(isinstance(data, dict), "manifest.json 必须是 JSON 对象")
    return data


def require_keys(manifest: Dict[str, object], keys: Iterable[str]) -> None:
    for key in keys:
        require(key in manifest, f"manifest 缺少必填字段: {key}")


def parse_hidden_profile(manifest: Dict[str, object]) -> str:
    raw = manifest.get("hidden_profile")
    require(isinstance(raw, list) and raw, "hidden_profile 必须是非空数组")
    parts = []
    for entry in raw:
        require(isinstance(entry, dict), "hidden_profile 每项都必须是对象")
        parts.append(f"{int(entry['head_count'])}x{int(entry['head_dim'])}")
    return ",".join(parts)


def validate_ppi_bundle(manifest: Dict[str, object], bundle_dir: pathlib.Path) -> None:
    require_keys(
        manifest,
        (
            "family_schema_version",
            "L",
            "hidden_profile",
            "hidden_layers",
            "hidden_head_specs",
            "d_in_profile",
            "K_out",
            "output_head_specs",
            "C",
            "output_average_rule",
            "model_arch_id",
            "model_param_id",
            "quant_cfg_id",
            "static_table_id",
            "degree_bound_id",
            "task_type",
            "report_unit",
            "batching_rule",
            "subgraph_rule",
            "self_loop_rule",
            "edge_sort_rule",
        ),
    )
    require((bundle_dir / "tensors.txt").exists(), f"缺少 tensors.txt: {bundle_dir}")
    require(str(manifest["family_schema_version"]) == "multi_layer_multi_head_v2", "family_schema_version 不受支持")
    require(int(manifest["L"]) == 2, "PPI 正式配置要求 L=2")
    require(parse_hidden_profile(manifest) == "1x8", "PPI 正式配置要求 hidden_profile=[1x8]")
    d_in_profile = manifest["d_in_profile"]
    require(isinstance(d_in_profile, list) and [int(value) for value in d_in_profile] == [50], "PPI 正式配置要求 d_in_profile=[50]")
    require(int(manifest["K_out"]) == 1, "PPI 正式配置要求 K_out=1")
    require(int(manifest["C"]) == 121, "PPI 正式配置要求 C=121")
    require(str(manifest["task_type"]) == "inductive_multi_graph_node_classification", "PPI bundle 的 task_type 不匹配")
    require(str(manifest["report_unit"]) == "graph", "PPI bundle 的 report_unit 不匹配")
    require(str(manifest["batching_rule"]) == "multi_graph_batch", "PPI bundle 的 batching_rule 不匹配")
    require(str(manifest["subgraph_rule"]) == "whole_graph", "PPI bundle 的 subgraph_rule 不匹配")
    require(str(manifest["self_loop_rule"]) == "per_node", "PPI bundle 的 self_loop_rule 不匹配")
    require(str(manifest["edge_sort_rule"]) == "edge_gid_then_dst_stable", "PPI bundle 的 edge_sort_rule 不匹配")
    require(str(manifest["output_average_rule"]) == "per_head_bias_then_arithmetic_mean", "PPI bundle 的 output_average_rule 不匹配")


def install_bundle(source_dir: pathlib.Path, output_dir: pathlib.Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.copytree(source_dir, output_dir)


def write_bundle_from_torch_checkpoint(checkpoint_path: pathlib.Path, bundle_dir: pathlib.Path) -> None:
    (
        tensors,
        d_in_profile,
        hidden_layers,
        hidden_head_specs,
        output_head_specs,
        num_classes,
        bundle,
    ) = build_tensors_and_manifest_spec(checkpoint_path)
    manifest = {
        "checkpoint_prefix": str(checkpoint_path),
        "family_schema_version": "multi_layer_multi_head_v2",
        "output_average_rule": str(bundle.get("output_average_rule", "per_head_bias_then_arithmetic_mean")),
        "L": len(hidden_layers) + 1,
        "d_in_profile": [int(value) for value in d_in_profile],
        "hidden_profile": [
            {"head_count": int(layer["head_count"]), "head_dim": int(layer["head_dim"])}
            for layer in hidden_layers
        ],
        "hidden_layers": hidden_layers,
        "hidden_head_specs": hidden_head_specs,
        "K_out": len(output_head_specs),
        "output_head_specs": output_head_specs,
        "C": int(num_classes),
        "task_type": str(bundle.get("task_type", "inductive_multi_graph_node_classification")),
        "report_unit": str(bundle.get("report_unit", "graph")),
        "batching_rule": str(bundle.get("batching_rule", "multi_graph_batch")),
        "subgraph_rule": str(bundle.get("subgraph_rule", "whole_graph")),
        "self_loop_rule": str(bundle.get("self_loop_rule", "per_node")),
        "edge_sort_rule": str(bundle.get("edge_sort_rule", "edge_gid_then_dst_stable")),
        "tensor_count": len(tensors),
        "model_arch_id": str(bundle.get("model_arch_id", checkpoint_path.stem)),
        "model_param_id": str(bundle.get("model_param_id", checkpoint_path.stem)),
        "quant_cfg_id": str(bundle.get("quant_cfg_id", "fp32_torch_checkpoint")),
        "static_table_id": str(bundle.get("static_table_id", "tables:lrelu+elu+exp+range")),
        "degree_bound_id": str(bundle.get("degree_bound_id", "auto")),
        "tensor_index": {},
    }
    for name, tensor in sorted(tensors.items()):
        manifest["tensor_index"][name] = {
            "archive_key": archive_key(name),
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
        }

    bundle_dir.mkdir(parents=True, exist_ok=True)
    arrays = {archive_key(name): tensor for name, tensor in tensors.items()}
    np.savez_compressed(bundle_dir / "tensors.npz", **arrays)
    write_text_tensor_dump(bundle_dir / "tensors.txt", tensors)
    (bundle_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def write_bundle_from_tf_checkpoint(checkpoint_prefix: pathlib.Path, bundle_dir: pathlib.Path) -> None:
    tensors = read_checkpoint_tensors(str(checkpoint_prefix))
    manifest = build_manifest(tensors, str(checkpoint_prefix))
    manifest["task_type"] = "inductive_multi_graph_node_classification"
    manifest["report_unit"] = "graph"
    manifest["batching_rule"] = "multi_graph_batch"
    manifest["subgraph_rule"] = "whole_graph"
    manifest["self_loop_rule"] = "per_node"
    manifest["edge_sort_rule"] = "edge_gid_then_dst_stable"
    manifest["model_param_id"] = pathlib.Path(str(checkpoint_prefix)).name
    manifest["quant_cfg_id"] = "fp32_bundle_export"

    bundle_dir.mkdir(parents=True, exist_ok=True)
    arrays = {archive_key(name): tensor for name, tensor in tensors.items()}
    np.savez_compressed(bundle_dir / "tensors.npz", **arrays)
    write_text_tensor_dump(bundle_dir / "tensors.txt", tensors)
    (bundle_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--bundle-dir")
    source.add_argument("--torch-checkpoint")
    source.add_argument("--tf-checkpoint-prefix")
    parser.add_argument("--output-dir", default="artifacts/checkpoints/ppi_gat")
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir).resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    if args.bundle_dir:
        source_dir = pathlib.Path(args.bundle_dir).resolve()
        require(source_dir.exists(), f"bundle 目录不存在: {source_dir}")
        require(source_dir.is_dir(), f"bundle 路径不是目录: {source_dir}")
        manifest = load_manifest(source_dir)
        validate_ppi_bundle(manifest, source_dir)
        install_bundle(source_dir, output_dir)
        print(str(output_dir))
        return

    if args.torch_checkpoint:
        checkpoint_path = pathlib.Path(args.torch_checkpoint).resolve()
        require(checkpoint_path.exists(), f"torch checkpoint 不存在: {checkpoint_path}")
        require(checkpoint_path.is_file(), f"torch checkpoint 路径不是文件: {checkpoint_path}")
        with tempfile.TemporaryDirectory(prefix="ppi_bundle_") as tmp_dir:
            temp_bundle = pathlib.Path(tmp_dir) / "bundle"
            write_bundle_from_torch_checkpoint(checkpoint_path, temp_bundle)
            manifest = load_manifest(temp_bundle)
            validate_ppi_bundle(manifest, temp_bundle)
            install_bundle(temp_bundle, output_dir)
        print(str(output_dir))
        return

    checkpoint_prefix = pathlib.Path(args.tf_checkpoint_prefix).resolve()
    require(checkpoint_prefix.exists() or checkpoint_prefix.with_suffix(".index").exists(), f"tensorflow checkpoint 前缀不存在: {checkpoint_prefix}")
    with tempfile.TemporaryDirectory(prefix="ppi_bundle_") as tmp_dir:
        temp_bundle = pathlib.Path(tmp_dir) / "bundle"
        write_bundle_from_tf_checkpoint(checkpoint_prefix, temp_bundle)
        manifest = load_manifest(temp_bundle)
        validate_ppi_bundle(manifest, temp_bundle)
        install_bundle(temp_bundle, output_dir)
    print(str(output_dir))


if __name__ == "__main__":
    main()
