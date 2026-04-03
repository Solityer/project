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
    *,
    apply_output_bias: bool,
    apply_activation: bool,
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
    if apply_output_bias:
        h_agg_input = h_agg_pre_bias + output_bias.reshape(1, -1)
    else:
        h_agg_input = h_agg_pre_bias
    h_agg = elu(h_agg_input) if apply_activation else h_agg_input
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


def run_bundle_forward(
    checkpoint_dir: pathlib.Path,
    data_root: pathlib.Path,
    dataset: str,
    semantics: str,
) -> Dict[str, object]:
    manifest, tensors = load_exported_tensors(checkpoint_dir)
    sparse_features, labels, adjacency, _ = load_raw_planetoid_dataset(data_root, dataset)
    features = row_normalize_features(sparse_features)
    bias = adj_to_bias(adjacency, features.shape[0])

    hidden_apply_output_bias = semantics == "reference_style"
    hidden_apply_activation = True
    output_apply_output_bias = semantics == "reference_style"
    output_apply_activation = semantics == "reference_style"

    hidden_outputs = []
    hidden_head_traces = []
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
            apply_output_bias=hidden_apply_output_bias,
            apply_activation=hidden_apply_activation,
        )
        hidden_outputs.append(head_tensors["H_agg"])
        hidden_head_traces.append((head["head_index"], head_tensors))
        head_summaries.append(
            {
                "head_index": head["head_index"],
                "node_count": int(head_tensors["H_prime"].shape[0]),
                "width": int(head_tensors["H_prime"].shape[1]),
            }
        )

    hidden_concat = np.concatenate(hidden_outputs, axis=1)
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
        apply_output_bias=output_apply_output_bias,
        apply_activation=output_apply_activation,
    )

    summary = {
        "dataset": dataset,
        "semantics": semantics,
        "node_count": int(features.shape[0]),
        "feature_count": int(features.shape[1]),
        "class_count": int(output_tensors["H_agg"].shape[1]),
        "hidden_head_count": len(manifest["hidden_heads"]),
        "head_summaries": head_summaries,
    }
    return {
        "features": features,
        "bias": bias,
        "labels": labels_to_int(labels),
        "hidden_head_traces": hidden_head_traces,
        "hidden_concat": hidden_concat,
        "output_tensors": output_tensors,
        "summary": summary,
    }


def export_group(output_dir: pathlib.Path, group_name: str, tensors: Dict[str, np.ndarray]) -> None:
    for name, value in tensors.items():
        save_array(output_dir / group_name / f"{name}.npy", value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--dataset", required=True, choices=["cora", "citeseer", "pubmed"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--semantics", choices=["reference_style", "formal_note"], default="reference_style")
    args = parser.parse_args()

    checkpoint_dir = pathlib.Path(args.checkpoint_dir)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = run_bundle_forward(checkpoint_dir, pathlib.Path(args.data_root), args.dataset, args.semantics)
    save_array(output_dir / "inputs" / "H.npy", results["features"])
    save_array(output_dir / "inputs" / "bias.npy", results["bias"])
    save_array(output_dir / "inputs" / "labels.npy", results["labels"])

    for head_index, head_tensors in results["hidden_head_traces"]:
        export_group(output_dir, f"hidden_head_{head_index}", head_tensors)

    save_array(output_dir / "hidden_concat.npy", results["hidden_concat"])
    save_array(output_dir / "output" / "Y_lin.npy", results["output_tensors"]["H_prime"])
    save_array(output_dir / "output" / "Y.npy", results["output_tensors"]["H_agg"])
    export_group(output_dir, "output", results["output_tensors"])

    (output_dir / "summary.json").write_text(
        json.dumps(results["summary"], indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
