#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict

import numpy as np
import torch

from train_planetoid_gat import run_trace_from_bundle, run_trace_from_checkpoint


def compare_arrays(name: str, expected: np.ndarray, actual: np.ndarray) -> Dict[str, object]:
    diff = np.abs(expected.astype(np.float64) - actual.astype(np.float64))
    return {
        "name": name,
        "shape": list(expected.shape),
        "max_abs_error": float(diff.max(initial=0.0)),
        "mean_abs_error": float(diff.mean() if diff.size else 0.0),
        "passed": bool(np.allclose(expected, actual, atol=1e-5, rtol=1e-5)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--dataset", required=True, choices=["cora", "citeseer", "pubmed"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    checkpoint_path = pathlib.Path(args.checkpoint).resolve()
    bundle_dir = pathlib.Path(args.bundle_dir).resolve()
    data_root = pathlib.Path(args.data_root).resolve()
    output_path = pathlib.Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    training_trace = run_trace_from_checkpoint(checkpoint_path, data_root, args.dataset, torch.device("cpu"))
    bundle_trace = run_trace_from_bundle(bundle_dir, data_root, args.dataset, torch.device("cpu"))

    comparisons = []
    comparisons.append(compare_arrays("hidden_concat", training_trace["hidden_concat"], bundle_trace["hidden_concat"]))
    comparisons.append(compare_arrays("output/Y_lin", training_trace["output"]["H_prime"], bundle_trace["output"]["H_prime"]))
    comparisons.append(compare_arrays("output/Y", training_trace["logits"], bundle_trace["logits"]))

    hidden_pairs = min(len(training_trace["hidden_heads"]), len(bundle_trace["hidden_heads"]))
    for head_index in range(hidden_pairs):
        training_head = training_trace["hidden_heads"][head_index]
        bundle_head = bundle_trace["hidden_heads"][head_index]
        for key in ["H_prime", "E_src", "E_dst", "S", "Z", "M", "Delta", "U", "Sum", "inv", "alpha", "H_agg_pre_bias", "H_agg"]:
            comparisons.append(compare_arrays(f"hidden_head_{head_index}/{key}", training_head[key], bundle_head[key]))

    summary = {
        "dataset": args.dataset,
        "comparison_count": len(comparisons),
        "passed": all(item["passed"] for item in comparisons),
        "comparisons": comparisons,
    }
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
