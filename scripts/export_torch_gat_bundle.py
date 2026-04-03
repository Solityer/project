#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict

import numpy as np
import torch

from export_gat_checkpoint import archive_key, build_manifest, write_text_tensor_dump
from train_planetoid_gat import load_checkpoint


def hidden_seq_name(head_index: int) -> str:
    if head_index == 0:
        return "conv1d/kernel"
    return f"conv1d_{head_index * 3}/kernel"


def hidden_dst_name(head_index: int) -> str:
    index = 1 if head_index == 0 else head_index * 3 + 1
    return f"conv1d_{index}"


def hidden_src_name(head_index: int) -> str:
    index = 2 if head_index == 0 else head_index * 3 + 2
    return f"conv1d_{index}"


def output_bias_name(head_index: int) -> str:
    if head_index == 0:
        return "BiasAdd/biases"
    return f"BiasAdd_{head_index}/biases"


def export_linear_kernel(weight: torch.Tensor) -> np.ndarray:
    return weight.detach().cpu().numpy().T[np.newaxis, :, :].astype(np.float32, copy=False)


def export_attention_kernel(weight: torch.Tensor) -> np.ndarray:
    return weight.detach().cpu().numpy().reshape(1, -1, 1).astype(np.float32, copy=False)


def export_scalar_bias(bias: torch.Tensor) -> np.ndarray:
    return bias.detach().cpu().numpy().reshape(1).astype(np.float32, copy=False)


def build_tensors(checkpoint_path: pathlib.Path) -> Dict[str, np.ndarray]:
    checkpoint = load_checkpoint(checkpoint_path, torch.device("cpu"))
    bundle = checkpoint["bundle"]
    state = checkpoint["state_dict"]
    hidden_dim = int(bundle["hidden_dim"])
    num_classes = int(bundle["num_classes"])
    hidden_heads = int(bundle["hidden_heads"])

    tensors: Dict[str, np.ndarray] = {}
    for head_index in range(hidden_heads):
        prefix = f"hidden_heads.{head_index}"
        tensors[hidden_seq_name(head_index)] = export_linear_kernel(state[f"{prefix}.seq.weight"])
        dst_name = hidden_dst_name(head_index)
        src_name = hidden_src_name(head_index)
        tensors[f"{dst_name}/kernel"] = export_attention_kernel(state[f"{prefix}.attn_dst.weight"])
        tensors[f"{dst_name}/bias"] = export_scalar_bias(state[f"{prefix}.attn_dst.bias"])
        tensors[f"{src_name}/kernel"] = export_attention_kernel(state[f"{prefix}.attn_src.weight"])
        tensors[f"{src_name}/bias"] = export_scalar_bias(state[f"{prefix}.attn_src.bias"])
        tensors[output_bias_name(head_index)] = np.zeros((hidden_dim,), dtype=np.float32)

    tensors["conv1d_24/kernel"] = export_linear_kernel(state["output_head.seq.weight"])
    tensors["conv1d_25/kernel"] = export_attention_kernel(state["output_head.attn_dst.weight"])
    tensors["conv1d_25/bias"] = export_scalar_bias(state["output_head.attn_dst.bias"])
    tensors["conv1d_26/kernel"] = export_attention_kernel(state["output_head.attn_src.weight"])
    tensors["conv1d_26/bias"] = export_scalar_bias(state["output_head.attn_src.bias"])
    tensors["BiasAdd_8/biases"] = np.zeros((num_classes,), dtype=np.float32)
    return tensors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    checkpoint_path = pathlib.Path(args.checkpoint).resolve()
    output_dir = pathlib.Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tensors = build_tensors(checkpoint_path)
    manifest = build_manifest(tensors, str(checkpoint_path))
    arrays = {archive_key(name): tensor for name, tensor in tensors.items()}
    np.savez_compressed(output_dir / "tensors.npz", **arrays)
    write_text_tensor_dump(output_dir / "tensors.txt", tensors)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
