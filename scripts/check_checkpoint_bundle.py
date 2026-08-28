#!/usr/bin/env python3
"""Strictly check one exported checkpoint bundle against a formal dataset shape."""
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, Iterable, Tuple

import numpy as np
import torch


EXPECTED: Dict[str, Tuple[Tuple[int, ...], int, str, str, str, str]] = {
    "cora": ((1433, 64), 7, "gat_family_L3_hid8x8-8x8_kout2_c7", "cora_planetoid_seed11", "transductive_node_classification", "node"),
    "citeseer": ((3703, 64), 6, "gat_family_L3_hid8x8-8x8_kout2_c6", "citeseer_planetoid_seed11", "transductive_node_classification", "node"),
    "pubmed": ((500, 64), 3, "gat_family_L3_hid8x8-8x8_kout2_c3", "pubmed_planetoid_seed11", "transductive_node_classification", "node"),
    "ppi": ((50, 64), 121, "gat_family_L3_hid8x8-8x8_kout2_c121", "ppi_gat_seed11", "inductive_multi_graph_node_classification", "graph"),
    "ogbn-arxiv": ((128, 64), 40, "gat_family_L3_hid8x8-8x8_kout2_c40", "ogbn_arxiv_seed11", "transductive_node_classification", "node"),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def tensor_shape(tensor_index: object, name: object) -> tuple[int, ...]:
    require(isinstance(tensor_index, dict), "manifest tensor_index must be an object")
    require(isinstance(name, str) and name in tensor_index, f"manifest missing tensor {name}")
    entry = tensor_index[name]
    require(isinstance(entry, dict) and isinstance(entry.get("shape"), list), f"manifest missing shape for {name}")
    return tuple(int(value) for value in entry["shape"])


def head_names(spec: object) -> Iterable[str]:
    require(isinstance(spec, dict), "head spec must be an object")
    names = ("seq_kernel", "attn_dst_kernel", "attn_dst_bias", "attn_src_kernel", "attn_src_bias", "output_bias")
    for name in names:
        require(name in spec, f"head spec missing {name}")
        yield str(spec[name])


def check_head(spec: object, tensor_index: object, input_dim: int, output_dim: int) -> None:
    names = list(head_names(spec))
    require(tensor_shape(tensor_index, names[0]) == (1, input_dim, output_dim), "seq kernel shape mismatch")
    require(tensor_shape(tensor_index, names[1]) == (1, output_dim, 1), "destination attention shape mismatch")
    require(tensor_shape(tensor_index, names[2]) == (1,), "destination attention bias shape mismatch")
    require(tensor_shape(tensor_index, names[3]) == (1, output_dim, 1), "source attention shape mismatch")
    require(tensor_shape(tensor_index, names[4]) == (1,), "source attention bias shape mismatch")
    require(tensor_shape(tensor_index, names[5]) == (output_dim,), "output bias shape mismatch")


def check_source_checkpoint(
    checkpoint: pathlib.Path,
    manifest: Dict[str, object],
    dataset: str,
    d_in_profile: Tuple[int, ...],
    classes: int,
    arch_id: str,
    model_param_id: str,
) -> None:
    source = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    require(isinstance(source, dict), "source checkpoint must be an object")
    bundle = source.get("bundle")
    state = source.get("state_dict")
    require(isinstance(bundle, dict), "source checkpoint is missing bundle metadata")
    require(isinstance(state, dict), "source checkpoint is missing state_dict")
    require(str(bundle.get("dataset")) == dataset, "source checkpoint dataset mismatch")
    require(int(bundle.get("L", 0)) == 3, "source checkpoint L mismatch")
    require(bundle.get("d_in_profile") == list(d_in_profile), "source checkpoint d_in_profile mismatch")
    require(bundle.get("hidden_profile") == [{"head_count": 8, "head_dim": 8}, {"head_count": 8, "head_dim": 8}], "source checkpoint hidden_profile mismatch")
    require(int(bundle.get("K_out", 0)) == 2, "source checkpoint K_out mismatch")
    require(int(bundle.get("C", 0)) == classes, "source checkpoint class count mismatch")
    require(bundle.get("model_arch_id") == arch_id, "source checkpoint model_arch_id mismatch")
    require(bundle.get("quant_cfg_id") == "fp32_torch_checkpoint", "source checkpoint quantization metadata mismatch")
    require(bundle.get("model_param_id") == model_param_id, "source checkpoint model parameter identity mismatch")
    if dataset == "ppi":
        require(bundle.get("label_format") == "multi_hot", "source checkpoint PPI label_format mismatch")
        require(bundle.get("loss") == "BCEWithLogitsLoss", "source checkpoint PPI loss metadata mismatch")
        require(bundle.get("selection_metric") == "validation_loss", "source checkpoint PPI selection metric mismatch")
        require(bundle.get("split_format") == "node_masks_v1", "source checkpoint PPI split metadata mismatch")
    require(
        str(manifest.get("checkpoint_prefix", ""))
        in {str(checkpoint.resolve()), str(checkpoint)},
        "source checkpoint provenance does not match the bundle check path",
    )

    expected: Dict[str, Tuple[int, ...]] = {}
    for layer_index, input_dim in enumerate(d_in_profile):
        for head_index in range(8):
            prefix = f"hidden_layers.{layer_index}.heads.{head_index}"
            expected.update(
                {
                    f"{prefix}.seq.weight": (8, input_dim),
                    f"{prefix}.attn_dst.weight": (1, 8),
                    f"{prefix}.attn_dst.bias": (1,),
                    f"{prefix}.attn_src.weight": (1, 8),
                    f"{prefix}.attn_src.bias": (1,),
                }
            )
    for head_index in range(2):
        prefix = f"output_heads.{head_index}"
        expected.update(
            {
                f"{prefix}.seq.weight": (classes, 64),
                f"{prefix}.attn_dst.weight": (1, classes),
                f"{prefix}.attn_dst.bias": (1,),
                f"{prefix}.attn_src.weight": (1, classes),
                f"{prefix}.attn_src.bias": (1,),
                f"{prefix}.output_bias": (classes,),
            }
        )
    require(set(state) == set(expected), "source checkpoint state_dict keys do not match canonical GAT")
    for name, shape in expected.items():
        value = state[name]
        require(isinstance(value, torch.Tensor), f"source checkpoint tensor is invalid: {name}")
        require(tuple(value.shape) == shape, f"source checkpoint tensor shape mismatch: {name}")


def check_bundle(bundle_dir: pathlib.Path, dataset: str, checkpoint: pathlib.Path | None) -> None:
    d_in_profile, classes, arch_id, model_param_id, task_type, report_unit = EXPECTED[dataset]
    manifest_path = bundle_dir / "manifest.json"
    require(manifest_path.exists(), f"missing {manifest_path}")
    require((bundle_dir / "tensors.txt").exists(), f"missing {bundle_dir / 'tensors.txt'}")
    require((bundle_dir / "tensors.npz").exists(), f"missing {bundle_dir / 'tensors.npz'}")
    if checkpoint is not None:
        require(checkpoint.exists() and checkpoint.is_file(), f"missing source checkpoint {checkpoint}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(isinstance(manifest, dict), "manifest must be an object")
    require(manifest.get("family_schema_version") == "multi_layer_multi_head_v2", "unsupported family schema")
    require(int(manifest.get("L", 0)) == 3, "formal bundle requires L=3")
    require(int(manifest.get("K_out", 0)) == 2, "formal bundle requires K_out=2")
    require(int(manifest.get("C", 0)) == classes, "bundle class count mismatch")
    require(manifest.get("d_in_profile") == list(d_in_profile), "bundle d_in_profile mismatch")
    require(manifest.get("hidden_profile") == [{"head_count": 8, "head_dim": 8}, {"head_count": 8, "head_dim": 8}], "bundle hidden_profile mismatch")
    require(manifest.get("model_arch_id") == arch_id, "bundle model_arch_id mismatch")
    require(manifest.get("model_param_id") == model_param_id, "bundle model_param_id mismatch")
    require(manifest.get("task_type") == task_type, "bundle task_type mismatch")
    require(manifest.get("report_unit") == report_unit, "bundle report_unit mismatch")
    require(manifest.get("output_average_rule") == "per_head_bias_then_arithmetic_mean", "bundle output average rule mismatch")
    if dataset == "ppi":
        require(manifest.get("label_format") == "multi_hot", "PPI bundle label_format mismatch")
        require(manifest.get("loss") == "BCEWithLogitsLoss", "PPI bundle loss metadata mismatch")
        require(manifest.get("selection_metric") == "validation_loss", "PPI bundle selection metric mismatch")
        require(manifest.get("split_format") == "node_masks_v1", "PPI bundle split metadata mismatch")
    if checkpoint is not None:
        check_source_checkpoint(
            checkpoint,
            manifest,
            dataset,
            d_in_profile,
            classes,
            arch_id,
            model_param_id,
        )
    tensor_index = manifest.get("tensor_index")
    hidden_layers = manifest.get("hidden_layers")
    hidden_specs = manifest.get("hidden_head_specs")
    output_specs = manifest.get("output_head_specs")
    require(isinstance(hidden_layers, list) and len(hidden_layers) == 2, "bundle must contain two hidden layers")
    require(isinstance(hidden_specs, list) and len(hidden_specs) == 16, "bundle must contain sixteen hidden heads")
    require(isinstance(output_specs, list) and len(output_specs) == 2, "bundle must contain two output heads")
    require(all(isinstance(layer, dict) for layer in hidden_layers), "hidden layer entries must be objects")
    require(all(isinstance(spec, dict) for spec in hidden_specs), "hidden head entries must be objects")
    require(all(isinstance(spec, dict) for spec in output_specs), "output head entries must be objects")
    require(
        [
            (int(layer["input_dim"]), int(layer["head_count"]), int(layer["head_dim"]))
            for layer in hidden_layers
        ]
        == [(d_in_profile[0], 8, 8), (d_in_profile[1], 8, 8)],
        "bundle hidden layer shape mismatch",
    )
    for layer_index, input_dim in enumerate(d_in_profile):
        layer_heads = [spec for spec in hidden_specs if int(spec.get("layer_index", -1)) == layer_index]
        require(len(layer_heads) == 8, f"bundle hidden layer {layer_index} head count mismatch")
        for spec in layer_heads:
            check_head(spec, tensor_index, input_dim, 8)
    for spec in output_specs:
        check_head(spec, tensor_index, 64, classes)

    with np.load(bundle_dir / "tensors.npz") as archive:
        for logical_name, entry in tensor_index.items():
            require(isinstance(entry, dict), f"tensor index entry must be an object: {logical_name}")
            archive_key = entry.get("archive_key")
            require(isinstance(archive_key, str) and archive_key in archive, f"archive missing {logical_name}")
            require(tuple(archive[archive_key].shape) == tuple(int(value) for value in entry["shape"]), f"archive shape mismatch: {logical_name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--dataset", required=True, choices=sorted(EXPECTED))
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    check_bundle(
        pathlib.Path(args.bundle_dir).resolve(),
        args.dataset,
        pathlib.Path(args.checkpoint).resolve() if args.checkpoint else None,
    )
    print(f"checkpoint bundle shape check passed: {args.dataset}")


if __name__ == "__main__":
    main()
