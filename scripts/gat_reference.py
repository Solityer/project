#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, Tuple

import numpy as np

from planetoid_utils import adj_to_bias, labels_to_int, load_raw_planetoid_dataset, row_normalize_features


def load_exported_tensors(export_dir: pathlib.Path) -> Tuple[Dict[str, object], Dict[str, np.ndarray]]:
    manifest = json.loads((export_dir / "manifest.json").read_text(encoding="utf-8"))
    archive = np.load(export_dir / "tensors.npz")
    tensors = {
        name: archive[meta["archive_key"]]
        for name, meta in manifest["tensor_index"].items()
    }
    return manifest, tensors


def leaky_relu(values: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    return np.where(values >= 0.0, values, alpha * values)


def elu(values: np.ndarray) -> np.ndarray:
    return np.where(values >= 0.0, values, np.expm1(values))


def save_array(path: pathlib.Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array.astype(np.float32, copy=False))


def head_forward(
    features: np.ndarray,
    bias: np.ndarray,
    seq_kernel: np.ndarray,
    dst_kernel: np.ndarray,
    dst_bias: np.ndarray,
    src_kernel: np.ndarray,
    src_bias: np.ndarray,
    output_bias: np.ndarray,
) -> Dict[str, np.ndarray]:
    h_prime = features @ seq_kernel.reshape(seq_kernel.shape[1], seq_kernel.shape[2])
    e_dst = h_prime @ dst_kernel.reshape(dst_kernel.shape[1]) + dst_bias.reshape(())
    e_src = h_prime @ src_kernel.reshape(src_kernel.shape[1]) + src_bias.reshape(())
    s = e_dst[:, None] + e_src[None, :]
    z = leaky_relu(s)
    masked_logits = z + bias
    m = np.max(masked_logits, axis=1)
    delta = m[:, None] - masked_logits
    u = np.exp(masked_logits - m[:, None])
    sum_u = np.sum(u, axis=1)
    inv = 1.0 / sum_u
    alpha = u * inv[:, None]
    h_agg_pre_bias = alpha @ h_prime
    h_agg = elu(h_agg_pre_bias + output_bias.reshape(1, -1))
    return {
        "H_prime": h_prime,
        "E_src": e_src,
        "E_dst": e_dst,
        "S": s,
        "Z": z,
        "M": m,
        "Delta": delta,
        "U": u,
        "Sum": sum_u,
        "inv": inv,
        "alpha": alpha,
        "H_agg_pre_bias": h_agg_pre_bias,
        "H_agg": h_agg,
    }


def export_group(output_dir: pathlib.Path, group_name: str, tensors: Dict[str, np.ndarray]) -> None:
    for name, value in tensors.items():
        save_array(output_dir / group_name / f"{name}.npy", value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--dataset", required=True, choices=["cora"])
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    checkpoint_dir = pathlib.Path(args.checkpoint_dir)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest, tensors = load_exported_tensors(checkpoint_dir)
    sparse_features, labels, adjacency, _ = load_raw_planetoid_dataset(pathlib.Path(args.data_root), args.dataset)
    features = row_normalize_features(sparse_features)
    bias = adj_to_bias(adjacency, features.shape[0])

    save_array(output_dir / "inputs" / "H.npy", features)
    save_array(output_dir / "inputs" / "bias.npy", bias)
    save_array(output_dir / "inputs" / "labels.npy", labels_to_int(labels))

    hidden_outputs = []
    head_summaries = []
    for head in manifest["hidden_heads"]:
        head_tensors = head_forward(
            features,
            bias,
            tensors[head["seq_kernel"]],
            tensors[head["attn_dst_kernel"]],
            tensors[head["attn_dst_bias"]],
            tensors[head["attn_src_kernel"]],
            tensors[head["attn_src_bias"]],
            tensors[head["output_bias"]],
        )
        hidden_outputs.append(head_tensors["H_agg"])
        export_group(output_dir, f"hidden_head_{head['head_index']}", head_tensors)
        head_summaries.append(
            {
                "head_index": head["head_index"],
                "node_count": int(head_tensors["H_prime"].shape[0]),
                "width": int(head_tensors["H_prime"].shape[1]),
            }
        )

    hidden_concat = np.concatenate(hidden_outputs, axis=1)
    save_array(output_dir / "hidden_concat.npy", hidden_concat)

    output_head = manifest["output_head"]
    output_tensors = head_forward(
        hidden_concat,
        bias,
        tensors[output_head["seq_kernel"]],
        tensors[output_head["attn_dst_kernel"]],
        tensors[output_head["attn_dst_bias"]],
        tensors[output_head["attn_src_kernel"]],
        tensors[output_head["attn_src_bias"]],
        tensors[output_head["output_bias"]],
    )
    save_array(output_dir / "output" / "Y_lin.npy", output_tensors["H_prime"])
    save_array(output_dir / "output" / "Y.npy", output_tensors["H_agg"])
    export_group(output_dir, "output", output_tensors)

    summary = {
        "dataset": args.dataset,
        "node_count": int(features.shape[0]),
        "feature_count": int(features.shape[1]),
        "class_count": int(output_tensors["H_agg"].shape[1]),
        "hidden_head_count": len(manifest["hidden_heads"]),
        "head_summaries": head_summaries,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
