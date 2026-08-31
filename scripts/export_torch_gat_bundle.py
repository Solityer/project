#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from export_gat_checkpoint import archive_key, build_family_manifest, write_text_tensor_dump
from train_planetoid_gat import CANONICAL_MODEL_SPECS, load_checkpoint
from quantization_scale_chain import canonical_input_preprocess_id


def export_linear_kernel(weight: torch.Tensor) -> np.ndarray:
    return weight.detach().cpu().numpy().T[np.newaxis, :, :].astype(np.float32, copy=False)


def export_attention_kernel(weight: torch.Tensor) -> np.ndarray:
    return weight.detach().cpu().numpy().reshape(1, -1, 1).astype(np.float32, copy=False)


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
    raise ValueError(
        "canonical PyTorch checkpoint metadata must expose hidden_profile"
    )


def parse_d_in_profile(bundle: Dict[str, object], input_dim: int) -> List[int]:
    raw = bundle.get("d_in_profile")
    if isinstance(raw, list) and raw:
        return [int(value) for value in raw]
    raise ValueError("canonical PyTorch checkpoint metadata must expose d_in_profile")


def parse_k_out(bundle: Dict[str, object]) -> int:
    if "K_out" not in bundle:
        raise ValueError("canonical PyTorch checkpoint metadata must expose K_out")
    return int(bundle["K_out"])


def structured_hidden_prefix(layer_index: int, head_index: int) -> str:
    return f"hidden/layer{layer_index}/head{head_index}"


def structured_output_prefix(head_index: int) -> str:
    return f"output/head{head_index}"


def export_head_tensors(
    tensors: Dict[str, np.ndarray],
    state: Dict[str, torch.Tensor],
    state_prefix: str,
    tensor_prefix: str,
    *,
    require_output_bias: bool,
) -> Dict[str, str]:
    for legacy_key in (
        f"{state_prefix}.attn_dst.bias",
        f"{state_prefix}.attn_src.bias",
        f"{state_prefix}.attn_dst_bias",
        f"{state_prefix}.attn_src_bias",
    ):
        if legacy_key in state:
            raise ValueError(f"attention scalar bias is forbidden in formal checkpoint: {legacy_key}")
    seq_name = f"{tensor_prefix}/seq/kernel"
    dst_kernel_name = f"{tensor_prefix}/attn_dst/kernel"
    src_kernel_name = f"{tensor_prefix}/attn_src/kernel"
    output_bias_name = f"{tensor_prefix}/output_bias"
    tensors[seq_name] = export_linear_kernel(state[f"{state_prefix}.seq.weight"])
    tensors[dst_kernel_name] = export_attention_kernel(state[f"{state_prefix}.attn_dst.weight"])
    tensors[src_kernel_name] = export_attention_kernel(state[f"{state_prefix}.attn_src.weight"])
    if f"{state_prefix}.output_bias" in state:
        if not require_output_bias:
            raise ValueError(f"hidden head must not expose output_bias: {state_prefix}")
        tensors[output_bias_name] = export_vector_bias(state[f"{state_prefix}.output_bias"])
    elif require_output_bias:
        raise ValueError(f"output head {state_prefix} must expose output_bias")
    else:
        # Hidden heads intentionally have no post-aggregation bias tensor.
        pass
    result = {
        "seq_kernel": seq_name,
        "attn_dst_kernel": dst_kernel_name,
        "attn_src_kernel": src_kernel_name,
    }
    if require_output_bias:
        result["output_bias"] = output_bias_name
    return result


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
    dataset = str(bundle.get("dataset", ""))
    dataset_key = "ogbn_arxiv" if dataset == "ogbn-arxiv" else dataset
    if dataset_key not in CANONICAL_MODEL_SPECS:
        raise ValueError(f"unsupported canonical checkpoint dataset: {dataset}")
    expected_d_in, expected_classes, expected_arch, _ = CANONICAL_MODEL_SPECS[dataset_key]
    if (
        int(bundle.get("L", 0)) != 3
        or hidden_profile != [
            {"head_count": 8, "head_dim": 8},
            {"head_count": 8, "head_dim": 8},
        ]
        or d_in_profile != expected_d_in
        or k_out != 2
        or num_classes != expected_classes
        or bundle.get("model_arch_id") != expected_arch
        or bundle.get("quant_cfg_id") != "gat_l3_scale_chain_v1"
    ):
        raise ValueError("PyTorch checkpoint does not match the canonical L=3 multi-head formal shape")
    expected_preprocess_id = canonical_input_preprocess_id(dataset)
    if bundle.get("input_preprocess_id") != expected_preprocess_id:
        raise ValueError(
            "PyTorch checkpoint input_preprocess_id does not match its dataset"
        )
    if dataset_key == "ppi":
        ppi_semantics = {
            "label_format": "multi_hot",
            "loss": "BCEWithLogitsLoss",
            "selection_metric": "validation_loss",
            "split_format": "node_masks_v1",
        }
        if any(bundle.get(key) != value for key, value in ppi_semantics.items()):
            raise ValueError(
                "PPI checkpoint must declare multi_hot/BCEWithLogitsLoss/"
                "validation_loss/node_masks_v1 metadata"
            )
    expected_layers = int(bundle.get("L", len(hidden_profile) + 1))
    if expected_layers != len(hidden_profile) + 1:
        raise ValueError("bundle L conflicts with hidden_profile")

    tensors: Dict[str, np.ndarray] = {}
    hidden_layers: List[Dict[str, int]] = []
    hidden_head_specs: List[Dict[str, object]] = []
    output_head_specs: List[Dict[str, object]] = []

    if any(key.startswith("hidden_layers.") for key in state):
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
                        **export_head_tensors(
                            tensors,
                            state,
                            state_prefix,
                            tensor_prefix,
                            require_output_bias=False,
                        ),
                    }
                )
        for head_index in range(k_out):
            state_prefix = f"output_heads.{head_index}"
            tensor_prefix = structured_output_prefix(head_index)
            output_head_specs.append(
                {
                    "head_index": head_index,
                    **export_head_tensors(
                        tensors,
                        state,
                        state_prefix,
                        tensor_prefix,
                        require_output_bias=True,
                    ),
                }
            )
    else:
        raise ValueError(
            "canonical checkpoint state_dict must expose structured multi-layer keys"
        )

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
        quant_cfg_id=str(bundle.get("quant_cfg_id", "gat_l3_scale_chain_v1")),
        quantization_scale_chain=str(bundle["quantization_scale_chain"]),
        static_table_id=str(bundle.get("static_table_id", "tables:lrelu+elu+exp+range")),
        degree_bound_id=str(bundle.get("degree_bound_id", "auto")),
        model_arch_id=str(bundle.get("model_arch_id", "")) or None,
        model_param_id=str(bundle.get("model_param_id", "")) or checkpoint_path.name,
        dataset=str(bundle["dataset"]),
        input_preprocess_id=str(bundle["input_preprocess_id"]),
    )
    for metadata_key in ("label_format", "loss", "selection_metric", "split_format"):
        if metadata_key in bundle:
            manifest[metadata_key] = bundle[metadata_key]
    arrays = {archive_key(name): tensor for name, tensor in tensors.items()}
    np.savez_compressed(output_dir / "tensors.npz", **arrays)
    write_text_tensor_dump(output_dir / "tensors.txt", tensors)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
