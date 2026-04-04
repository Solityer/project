#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import re
from typing import Dict, List, Sequence

import numpy as np

from checkpoint_reader_compat import read_checkpoint_tensors


def archive_key(name: str) -> str:
    return name.replace("/", "__").replace(":", "__")


def write_text_tensor_dump(path: pathlib.Path, tensors: Dict[str, np.ndarray]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for name, tensor in sorted(tensors.items()):
            flat = tensor.astype(np.float64, copy=False).reshape(-1)
            shape = " ".join(str(dim) for dim in tensor.shape)
            handle.write(f"TENSOR {name} {tensor.ndim} {shape} {flat.size}\n")
            handle.write(" ".join(format(float(value), ".17g") for value in flat))
            handle.write("\n")


def hidden_head_manifest(head_index: int) -> Dict[str, str]:
    if head_index == 0:
        seq_kernel = "conv1d/kernel"
        dst_kernel = "conv1d_1/kernel"
        dst_bias = "conv1d_1/bias"
        src_kernel = "conv1d_2/kernel"
        src_bias = "conv1d_2/bias"
        output_bias = "BiasAdd/biases"
    else:
        base = head_index * 3
        seq_kernel = f"conv1d_{base}/kernel"
        dst_kernel = f"conv1d_{base + 1}/kernel"
        dst_bias = f"conv1d_{base + 1}/bias"
        src_kernel = f"conv1d_{base + 2}/kernel"
        src_bias = f"conv1d_{base + 2}/bias"
        output_bias = f"BiasAdd_{head_index}/biases"
    return {
        "seq_kernel": seq_kernel,
        "attn_dst_kernel": dst_kernel,
        "attn_dst_bias": dst_bias,
        "attn_src_kernel": src_kernel,
        "attn_src_bias": src_bias,
        "output_bias": output_bias,
    }


def infer_hidden_head_count(tensors: Dict[str, np.ndarray]) -> int:
    pattern = re.compile(r"^BiasAdd(?:_(\d+))?/biases$")
    indices: List[int] = []
    for name in tensors:
        match = pattern.match(name)
        if not match:
            continue
        indices.append(0 if match.group(1) is None else int(match.group(1)))
    if not indices:
        raise ValueError("failed to infer hidden head count from checkpoint tensors")
    hidden_head_count = max(indices)
    if hidden_head_count <= 0:
        raise ValueError("checkpoint tensors do not expose a valid output attention head bias")
    return hidden_head_count


def normalize_hidden_layers(
    d_in_profile: Sequence[int],
    hidden_profile: Sequence[Dict[str, int]],
) -> List[Dict[str, int]]:
    if len(d_in_profile) != len(hidden_profile):
        raise ValueError("d_in_profile must align with hidden_profile")
    hidden_layers: List[Dict[str, int]] = []
    for layer_index, (input_dim, shape) in enumerate(zip(d_in_profile, hidden_profile)):
        hidden_layers.append(
            {
                "layer_index": layer_index,
                "input_dim": int(input_dim),
                "head_count": int(shape["head_count"]),
                "head_dim": int(shape["head_dim"]),
            }
        )
    return hidden_layers


def build_family_manifest(
    tensors: Dict[str, np.ndarray],
    checkpoint_prefix: str,
    *,
    d_in_profile: Sequence[int],
    hidden_layers: Sequence[Dict[str, int]],
    hidden_head_specs: Sequence[Dict[str, object]],
    output_head_specs: Sequence[Dict[str, object]],
    num_classes: int,
    task_type: str = "transductive_node_classification",
    report_unit: str = "node",
    quant_cfg_id: str = "fp32_bundle_export",
    static_table_id: str = "tables:lrelu+elu+exp+range",
    degree_bound_id: str = "auto",
    model_arch_id: str | None = None,
    model_param_id: str | None = None,
) -> Dict[str, object]:
    hidden_profile = [
        {
            "head_count": int(layer["head_count"]),
            "head_dim": int(layer["head_dim"]),
        }
        for layer in hidden_layers
    ]
    k_out = len(output_head_specs)
    if k_out <= 0:
        raise ValueError("output_head_specs must be non-empty")
    if len(hidden_layers) + 1 < 2:
        raise ValueError("family manifest must expose at least one hidden layer and one output layer")
    if model_arch_id is None:
        hidden_label = "-".join(f"{layer['head_count']}x{layer['head_dim']}" for layer in hidden_layers)
        model_arch_id = f"gat_family_L{len(hidden_layers) + 1}_hid{hidden_label}_kout{k_out}_c{num_classes}"
    if model_param_id is None:
        model_param_id = pathlib.Path(checkpoint_prefix).name or "checkpoint_bundle"

    manifest: Dict[str, object] = {
        "checkpoint_prefix": checkpoint_prefix,
        "family_schema_version": "multi_layer_multi_head_v2",
        "output_average_rule": "per_head_bias_then_arithmetic_mean",
        "L": len(hidden_layers) + 1,
        "d_in_profile": [int(value) for value in d_in_profile],
        "hidden_profile": hidden_profile,
        "hidden_layers": list(hidden_layers),
        "hidden_head_specs": list(hidden_head_specs),
        "K_out": k_out,
        "output_head_specs": list(output_head_specs),
        "C": int(num_classes),
        "task_type": task_type,
        "report_unit": report_unit,
        "tensor_count": len(tensors),
        "model_arch_id": model_arch_id,
        "model_param_id": model_param_id,
        "quant_cfg_id": quant_cfg_id,
        "static_table_id": static_table_id,
        "degree_bound_id": degree_bound_id,
        "tensor_index": {},
    }
    tensor_index = manifest["tensor_index"]
    assert isinstance(tensor_index, dict)
    for name, tensor in sorted(tensors.items()):
        tensor_index[name] = {
            "archive_key": archive_key(name),
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
        }
    return manifest


def build_manifest(tensors: Dict[str, np.ndarray], checkpoint_prefix: str) -> Dict[str, object]:
    hidden_head_count = infer_hidden_head_count(tensors)
    hidden_head_dim = int(tensors["conv1d/kernel"].shape[2])
    d_in = int(tensors["conv1d/kernel"].shape[1])
    output_bias_name = f"BiasAdd_{hidden_head_count}/biases"
    if output_bias_name not in tensors:
        raise ValueError(f"missing output attention bias tensor: {output_bias_name}")
    c_out = int(tensors[output_bias_name].shape[0])
    hidden_layers = normalize_hidden_layers(
        [d_in],
        [{"head_count": hidden_head_count, "head_dim": hidden_head_dim}],
    )
    hidden_head_specs = [
        {
            "layer_index": 0,
            "local_head_index": head_index,
            "global_head_index": head_index,
            **hidden_head_manifest(head_index),
        }
        for head_index in range(hidden_head_count)
    ]
    output_head_specs = [
        {
            "head_index": 0,
            "seq_kernel": f"conv1d_{hidden_head_count * 3}/kernel",
            "attn_dst_kernel": f"conv1d_{hidden_head_count * 3 + 1}/kernel",
            "attn_dst_bias": f"conv1d_{hidden_head_count * 3 + 1}/bias",
            "attn_src_kernel": f"conv1d_{hidden_head_count * 3 + 2}/kernel",
            "attn_src_bias": f"conv1d_{hidden_head_count * 3 + 2}/bias",
            "output_bias": output_bias_name,
        }
    ]
    return build_family_manifest(
        tensors,
        checkpoint_prefix,
        d_in_profile=[d_in],
        hidden_layers=hidden_layers,
        hidden_head_specs=hidden_head_specs,
        output_head_specs=output_head_specs,
        num_classes=c_out,
        model_param_id=pathlib.Path(checkpoint_prefix).name or "checkpoint_bundle",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-prefix", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tensors = read_checkpoint_tensors(args.checkpoint_prefix)
    manifest = build_manifest(tensors, args.checkpoint_prefix)
    arrays = {archive_key(name): tensor for name, tensor in tensors.items()}
    np.savez_compressed(output_dir / "tensors.npz", **arrays)
    write_text_tensor_dump(output_dir / "tensors.txt", tensors)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
