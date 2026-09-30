#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import tempfile
from typing import Dict, Iterable

import numpy as np

from checkpoint_bundle import (
    archive_key,
    build_family_manifest,
    load_checkpoint_manifest,
    validate_checkpoint_manifest,
    validate_tensor_encodings,
    write_text_tensor_dump,
)
from export_torch_gat_bundle import build_tensors_and_manifest_spec
from quantization_scale_chain import canonical_input_preprocess_id, canonical_quantization_scale_chain


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_manifest(bundle_dir: pathlib.Path) -> Dict[str, object]:
    manifest_path = bundle_dir / "manifest.json"
    require(manifest_path.exists(), f"缺少 manifest.json: {bundle_dir}")
    return load_checkpoint_manifest(manifest_path)


def require_keys(manifest: Dict[str, object], keys: Iterable[str]) -> None:
    for key in keys:
        require(key in manifest, f"manifest 缺少必填字段: {key}")


def parse_hidden_profile(manifest: Dict[str, object]) -> str:
    raw = manifest["hidden_profile"]
    require(isinstance(raw, list) and raw, "hidden_profile 必须是非空数组")
    parts = []
    for entry in raw:
        require(isinstance(entry, dict), "hidden_profile 每项都必须是对象")
        parts.append(f"{entry['head_count']}x{entry['head_dim']}")
    return ",".join(parts)


def validate_ppi_bundle(manifest: Dict[str, object], bundle_dir: pathlib.Path) -> None:
    validate_checkpoint_manifest(manifest)
    require_keys(
        manifest,
        (
            "family_schema",
            "dataset",
            "input_preprocess_id",
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
            "quantization_scale_chain",
            "static_table_id",
            "degree_bound_id",
            "task_type",
            "report_unit",
            "batching_rule",
            "subgraph_rule",
            "self_loop_rule",
            "edge_sort_rule",
            "label_format",
            "loss",
            "selection_metric",
            "split_format",
        ),
    )
    require((bundle_dir / "tensors.txt").exists(), f"缺少 tensors.txt: {bundle_dir}")
    require((bundle_dir / "tensors.npz").exists(), f"缺少 tensors.npz: {bundle_dir}")
    require(manifest["family_schema"] == "multi_layer_multi_head", "family_schema 不受支持")
    require(manifest["dataset"] == "ppi", "PPI bundle dataset identity 不匹配")
    require(manifest["input_preprocess_id"] == canonical_input_preprocess_id("ppi"), "PPI bundle input_preprocess_id 不匹配")
    require(manifest["L"] == 3, "PPI 正式配置要求 L=3")
    require(parse_hidden_profile(manifest) == "8x8,8x8", "PPI 正式配置要求 hidden_profile=[8x8,8x8]")
    d_in_profile = manifest["d_in_profile"]
    require(isinstance(d_in_profile, list) and d_in_profile == [50, 64], "PPI 正式配置要求 d_in_profile=[50,64]")
    require(manifest["K_out"] == 2, "PPI 正式配置要求 K_out=2")
    require(manifest["C"] == 121, "PPI 正式配置要求 C=121")
    require(manifest["model_arch_id"] == "gat_family_L3_hid8x8-8x8_kout2_c121", "PPI bundle 的 model_arch_id 不匹配")
    require(manifest["model_param_id"] == "ppi_gat_seed11", "PPI bundle 的 model_param_id 不匹配")
    require(manifest["quant_cfg_id"] == "gat_l3_scale_chain", "PPI bundle 必须声明 canonical scale chain")
    expected_chain = canonical_quantization_scale_chain(manifest["quant_cfg_id"], manifest["hidden_profile"], manifest["K_out"])
    require(manifest["quantization_scale_chain"] == expected_chain.canonical_serialization(), "PPI bundle quantization_scale_chain metadata is malformed")
    require(manifest["task_type"] == "inductive_multi_graph_node_classification", "PPI bundle 的 task_type 不匹配")
    require(manifest["report_unit"] == "graph", "PPI bundle 的 report_unit 不匹配")
    require(manifest["batching_rule"] == "multi_graph_batch", "PPI bundle 的 batching_rule 不匹配")
    require(manifest["subgraph_rule"] == "whole_graph", "PPI bundle 的 subgraph_rule 不匹配")
    require(manifest["self_loop_rule"] == "per_node", "PPI bundle 的 self_loop_rule 不匹配")
    require(manifest["edge_sort_rule"] == "edge_gid_then_dst_stable", "PPI bundle 的 edge_sort_rule 不匹配")
    require(manifest["output_average_rule"] == "per_head_bias_then_arithmetic_mean", "PPI bundle 的 output_average_rule 不匹配")
    require(manifest["label_format"] == "multi_hot", "PPI bundle 必须保留 multi_hot 标签语义")
    require(manifest["loss"] == "BCEWithLogitsLoss", "PPI bundle 必须声明 BCEWithLogitsLoss")
    require(manifest["selection_metric"] == "validation_loss", "PPI bundle 必须只按 validation_loss 选择")
    require(manifest["split_format"] == "node_masks", "PPI bundle 必须声明 node_masks split")

    hidden_layers = manifest["hidden_layers"]
    require(isinstance(hidden_layers, list) and len(hidden_layers) == 2, "PPI bundle 必须包含两个 hidden layer")
    require(all(isinstance(layer, dict) for layer in hidden_layers), "PPI hidden_layers 每项必须是对象")
    require(
        [(layer["input_dim"], layer["head_count"], layer["head_dim"]) for layer in hidden_layers] == [(50, 8, 8), (64, 8, 8)], "PPI hidden_layers shape 不匹配"
    )
    hidden_specs = manifest["hidden_head_specs"]
    output_specs = manifest["output_head_specs"]
    require(isinstance(hidden_specs, list) and len(hidden_specs) == 16, "PPI bundle 必须包含 16 个 hidden head")
    require(isinstance(output_specs, list) and len(output_specs) == 2, "PPI bundle 必须包含 2 个 output head")
    require(all(isinstance(spec, dict) for spec in hidden_specs), "PPI hidden_head_specs 每项必须是对象")
    require(all(isinstance(spec, dict) for spec in output_specs), "PPI output_head_specs 每项必须是对象")
    # Formal identity is the explicit layer/local pair; a global-only manifest
    # is rejected fail-closed.
    expected_pairs = [(layer_index, head_index) for layer_index in range(2) for head_index in range(8)]
    actual_pairs = []
    for spec in hidden_specs:
        require("layer_index" in spec and "local_head_index" in spec, "PPI hidden head spec 必须包含 layer_index/local_head_index")
        actual_pairs.append((spec["layer_index"], spec["local_head_index"]))
    require(actual_pairs == expected_pairs, "PPI hidden head specs 必须按 layer/local pair canonical 顺序排列")

    tensor_index = manifest["tensor_index"]
    require(isinstance(tensor_index, dict), "PPI bundle tensor_index 必须是对象")

    def tensor_shape(name: object) -> tuple[int, ...]:
        require(isinstance(name, str) and name in tensor_index, f"PPI bundle 缺少 tensor: {name}")
        entry = tensor_index[name]
        require(isinstance(entry, dict) and isinstance(entry["shape"], list), f"PPI tensor shape 缺失: {name}")
        return tuple(entry["shape"])

    def validate_head(spec: Dict[str, object], input_dim: int, output_dim: int, *, require_output_bias: bool) -> None:
        require("attn_dst_bias" not in spec and "attn_src_bias" not in spec, "PPI head must reject attention scalar bias fields")
        keys = ("seq_kernel", "attn_dst_kernel", "attn_src_kernel")
        if require_output_bias:
            keys += ("output_bias",)
        else:
            require("output_bias" not in spec, "PPI hidden head must not expose output_bias")
        for key in keys:
            require(key in spec, f"PPI head spec 缺少 {key}")
        require(tensor_shape(spec["seq_kernel"]) == (1, input_dim, output_dim), "PPI seq kernel shape 不匹配")
        require(tensor_shape(spec["attn_dst_kernel"]) == (1, output_dim, 1), "PPI destination attention shape 不匹配")
        require(tensor_shape(spec["attn_src_kernel"]) == (1, output_dim, 1), "PPI source attention shape 不匹配")
        if require_output_bias:
            require(tensor_shape(spec["output_bias"]) == (output_dim,), "PPI output bias shape 不匹配")

    for layer_index, input_dim in enumerate((50, 64)):
        layer_specs = [spec for spec in hidden_specs if spec["layer_index"] == layer_index]
        require(len(layer_specs) == 8, f"PPI hidden layer {layer_index} head count 不匹配")
        for spec in layer_specs:
            validate_head(spec, input_dim, 8, require_output_bias=False)
    for spec in output_specs:
        validate_head(spec, 64, 121, require_output_bias=True)
    validate_tensor_encodings(manifest, bundle_dir / "tensors.npz", bundle_dir / "tensors.txt")


def install_bundle(source_dir: pathlib.Path, output_dir: pathlib.Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    # A canonical checkpoint directory also contains best_model.pt and its
    # training summary. Replace only bundle-owned files so importing the
    # freshly trained checkpoint cannot erase its source artifact.
    bundle_files = ("manifest.json", "tensors.npz", "tensors.txt")
    for name in bundle_files:
        destination = output_dir / name
        if destination.exists():
            destination.unlink()
        source = source_dir / name
        require(source.exists(), f"临时 bundle 缺少 {name}")
        shutil.copy2(source, destination)


def write_bundle_from_torch_checkpoint(checkpoint_path: pathlib.Path, bundle_dir: pathlib.Path) -> None:
    tensors, d_in_profile, hidden_layers, hidden_head_specs, output_head_specs, num_classes, bundle = build_tensors_and_manifest_spec(checkpoint_path)
    manifest = build_family_manifest(
        tensors,
        str(checkpoint_path),
        d_in_profile=d_in_profile,
        hidden_layers=hidden_layers,
        hidden_head_specs=hidden_head_specs,
        output_head_specs=output_head_specs,
        num_classes=num_classes,
        task_type=bundle["task_type"],
        report_unit=bundle["report_unit"],
        quant_cfg_id=bundle["quant_cfg_id"],
        quantization_scale_chain=bundle["quantization_scale_chain"],
        static_table_id=bundle["static_table_id"],
        degree_bound_id=bundle["degree_bound_id"],
        model_arch_id=bundle["model_arch_id"],
        model_param_id=bundle["model_param_id"],
        dataset="ppi",
        input_preprocess_id=bundle["input_preprocess_id"],
    )

    bundle_dir.mkdir(parents=True, exist_ok=True)
    arrays = {archive_key(name): tensor for name, tensor in tensors.items()}
    np.savez_compressed(bundle_dir / "tensors.npz", **arrays)
    write_text_tensor_dump(bundle_dir / "tensors.txt", tensors)
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    validate_tensor_encodings(manifest, bundle_dir / "tensors.npz", bundle_dir / "tensors.txt")


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--bundle-dir")
    source.add_argument("--torch-checkpoint")
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

    raise RuntimeError("an explicit --bundle-dir or --torch-checkpoint source is required")


if __name__ == "__main__":
    main()
