#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from export_gat_checkpoint import archive_key, build_family_manifest, write_text_tensor_dump
from train_planetoid_gat import load_checkpoint


def export_linear_kernel(weight: torch.Tensor) -> np.ndarray:
    return weight.detach().cpu().numpy().T[np.newaxis, :, :].astype(np.float32, copy=False)


def export_attention_kernel(weight: torch.Tensor) -> np.ndarray:
    return weight.detach().cpu().numpy().reshape(1, -1, 1).astype(np.float32, copy=False)


def export_scalar_bias(bias: torch.Tensor) -> np.ndarray:
    return bias.detach().cpu().numpy().reshape(1).astype(np.float32, copy=False)


def export_vector_bias(bias: torch.Tensor) -> np.ndarray:
    return bias.detach().cpu().numpy().reshape(-1).astype(np.float32, copy=False)


def parse_hidden_profile(bundle: Dict[str, object]) -> List[Dict[str, int]]:
    raw = bundle.get("hidden_profile")
    if isinstance(raw, list) and raw:
        normalized: List[Dict[str, int]] = []
        for entry in raw:
            if not isinstance(entry, dict):
                raise ValueError("bundle hidden_profile entries must be objects")
            normalized.append(
                {
                    "head_count": int(entry["head_count"]),
                    "head_dim": int(entry["head_dim"]),
                }
            )
        return normalized
    hidden_heads = bundle.get("hidden_heads")
    hidden_dim = bundle.get("hidden_dim")
    if hidden_heads is None or hidden_dim is None:
        raise ValueError("bundle metadata must expose hidden_profile or hidden_heads/hidden_dim")
    return [{"head_count": int(hidden_heads), "head_dim": int(hidden_dim)}]


def parse_d_in_profile(bundle: Dict[str, object], input_dim: int) -> List[int]:
    raw = bundle.get("d_in_profile")
    if isinstance(raw, list) and raw:
        return [int(value) for value in raw]
    hidden_profile = parse_hidden_profile(bundle)
    profile = [int(input_dim)]
    for layer in hidden_profile[:-1]:
        profile.append(int(layer["head_count"]) * int(layer["head_dim"]))
    return profile


def parse_k_out(bundle: Dict[str, object]) -> int:
    return int(bundle.get("K_out", 1))


def structured_hidden_prefix(layer_index: int, head_index: int) -> str:
    return f"hidden/layer{layer_index}/head{head_index}"


def structured_output_prefix(head_index: int) -> str:
    return f"output/head{head_index}"


def export_head_tensors(
    tensors: Dict[str, np.ndarray],
    state: Dict[str, torch.Tensor],
    state_prefix: str,
    tensor_prefix: str,
) -> Dict[str, str]:
    seq_name = f"{tensor_prefix}/seq/kernel"
    dst_kernel_name = f"{tensor_prefix}/attn_dst/kernel"
    dst_bias_name = f"{tensor_prefix}/attn_dst/bias"
    src_kernel_name = f"{tensor_prefix}/attn_src/kernel"
    src_bias_name = f"{tensor_prefix}/attn_src/bias"
    output_bias_name = f"{tensor_prefix}/output_bias"
    tensors[seq_name] = export_linear_kernel(state[f"{state_prefix}.seq.weight"])
    tensors[dst_kernel_name] = export_attention_kernel(state[f"{state_prefix}.attn_dst.weight"])
    tensors[dst_bias_name] = export_scalar_bias(state[f"{state_prefix}.attn_dst.bias"])
    tensors[src_kernel_name] = export_attention_kernel(state[f"{state_prefix}.attn_src.weight"])
    tensors[src_bias_name] = export_scalar_bias(state[f"{state_prefix}.attn_src.bias"])
    if f"{state_prefix}.output_bias" in state:
        tensors[output_bias_name] = export_vector_bias(state[f"{state_prefix}.output_bias"])
    else:
        out_dim = int(state[f"{state_prefix}.seq.weight"].shape[0])
        tensors[output_bias_name] = np.zeros((out_dim,), dtype=np.float32)
    return {
        "seq_kernel": seq_name,
        "attn_dst_kernel": dst_kernel_name,
        "attn_dst_bias": dst_bias_name,
        "attn_src_kernel": src_kernel_name,
        "attn_src_bias": src_bias_name,
        "output_bias": output_bias_name,
    }


def build_tensors_and_manifest_spec(
    checkpoint_path: pathlib.Path,
) -> Tuple[Dict[str, np.ndarray], List[int], List[Dict[str, int]], List[Dict[str, object]], List[Dict[str, object]], int, Dict[str, object]]:
    checkpoint = load_checkpoint(checkpoint_path, torch.device("cpu"))
    bundle = checkpoint["bundle"]
    state = checkpoint["state_dict"]
    hidden_profile = parse_hidden_profile(bundle)
    num_classes = int(bundle.get("C", bundle.get("num_classes")))
    k_out = parse_k_out(bundle)
    d_in_profile = parse_d_in_profile(bundle, int(bundle.get("num_features")))
    expected_layers = int(bundle.get("L", len(hidden_profile) + 1))
    if expected_layers != len(hidden_profile) + 1:
        raise ValueError("bundle L conflicts with hidden_profile")

    tensors: Dict[str, np.ndarray] = {}
    hidden_layers: List[Dict[str, int]] = []
    hidden_head_specs: List[Dict[str, object]] = []
    output_head_specs: List[Dict[str, object]] = []

    if any(key.startswith("hidden_layers.") for key in state):
        global_head_index = 0
        for layer_index, layer in enumerate(hidden_profile):
            hidden_layers.append(
                {
                    "layer_index": layer_index,
                    "input_dim": int(d_in_profile[layer_index]),
                    "head_count": int(layer["head_count"]),
                    "head_dim": int(layer["head_dim"]),
                }
            )
            for local_head_index in range(int(layer["head_count"])):
                state_prefix = f"hidden_layers.{layer_index}.heads.{local_head_index}"
                tensor_prefix = structured_hidden_prefix(layer_index, local_head_index)
                hidden_head_specs.append(
                    {
                        "layer_index": layer_index,
                        "local_head_index": local_head_index,
                        "global_head_index": global_head_index,
                        **export_head_tensors(tensors, state, state_prefix, tensor_prefix),
                    }
                )
                global_head_index += 1
        for head_index in range(k_out):
            state_prefix = f"output_heads.{head_index}"
            tensor_prefix = structured_output_prefix(head_index)
            output_head_specs.append(
                {
                    "head_index": head_index,
                    **export_head_tensors(tensors, state, state_prefix, tensor_prefix),
                }
            )
    elif any(key.startswith("hidden_heads.") for key in state):
        if len(hidden_profile) != 1:
            raise ValueError("legacy torch checkpoint layout only supports one hidden layer")
        if k_out != 1:
            raise ValueError("legacy torch checkpoint layout only supports K_out=1")
        hidden_layers.append(
            {
                "layer_index": 0,
                "input_dim": int(d_in_profile[0]),
                "head_count": int(hidden_profile[0]["head_count"]),
                "head_dim": int(hidden_profile[0]["head_dim"]),
            }
        )
        for head_index in range(int(hidden_profile[0]["head_count"])):
            state_prefix = f"hidden_heads.{head_index}"
            tensor_prefix = structured_hidden_prefix(0, head_index)
            hidden_head_specs.append(
                {
                    "layer_index": 0,
                    "local_head_index": head_index,
                    "global_head_index": head_index,
                    **export_head_tensors(tensors, state, state_prefix, tensor_prefix),
                }
            )
        output_head_specs.append(
            {
                "head_index": 0,
                **export_head_tensors(tensors, state, "output_head", structured_output_prefix(0)),
            }
        )
    else:
        raise ValueError("checkpoint state_dict does not expose a supported GAT family layout")

    return tensors, d_in_profile, hidden_layers, hidden_head_specs, output_head_specs, num_classes, bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    checkpoint_path = pathlib.Path(args.checkpoint).resolve()
    output_dir = pathlib.Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    (
        tensors,
        d_in_profile,
        hidden_layers,
        hidden_head_specs,
        output_head_specs,
        num_classes,
        bundle,
    ) = build_tensors_and_manifest_spec(checkpoint_path)
    manifest = build_family_manifest(
        tensors,
        str(checkpoint_path),
        d_in_profile=d_in_profile,
        hidden_layers=hidden_layers,
        hidden_head_specs=hidden_head_specs,
        output_head_specs=output_head_specs,
        num_classes=num_classes,
        task_type=str(bundle.get("task_type", "transductive_node_classification")),
        report_unit=str(bundle.get("report_unit", "node")),
        quant_cfg_id=str(bundle.get("quant_cfg_id", "fp32_bundle_export")),
        static_table_id=str(bundle.get("static_table_id", "tables:lrelu+elu+exp+range")),
        degree_bound_id=str(bundle.get("degree_bound_id", "auto")),
        model_arch_id=str(bundle.get("model_arch_id", "")) or None,
        model_param_id=str(bundle.get("model_param_id", "")) or checkpoint_path.name,
    )
    arrays = {archive_key(name): tensor for name, tensor in tensors.items()}
    np.savez_compressed(output_dir / "tensors.npz", **arrays)
    write_text_tensor_dump(output_dir / "tensors.txt", tensors)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
