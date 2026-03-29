#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict

import numpy as np

from checkpoint_reader_compat import read_checkpoint_tensors


def archive_key(name: str) -> str:
    return name.replace("/", "__").replace(":", "__")


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


def build_manifest(tensors: Dict[str, np.ndarray], checkpoint_prefix: str) -> Dict[str, object]:
    manifest: Dict[str, object] = {
        "checkpoint_prefix": checkpoint_prefix,
        "tensor_count": len(tensors),
        "hidden_head_count": 8,
        "hidden_heads": [],
        "output_head": {
            "seq_kernel": "conv1d_24/kernel",
            "attn_dst_kernel": "conv1d_25/kernel",
            "attn_dst_bias": "conv1d_25/bias",
            "attn_src_kernel": "conv1d_26/kernel",
            "attn_src_bias": "conv1d_26/bias",
            "output_bias": "BiasAdd_8/biases",
        },
        "tensor_index": {},
    }
    hidden_heads = manifest["hidden_heads"]
    assert isinstance(hidden_heads, list)
    for head_index in range(8):
        hidden_heads.append({"head_index": head_index, **hidden_head_manifest(head_index)})
    tensor_index = manifest["tensor_index"]
    assert isinstance(tensor_index, dict)
    for name, tensor in sorted(tensors.items()):
        tensor_index[name] = {
            "archive_key": archive_key(name),
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
        }
    return manifest


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
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
