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
    if expected.shape != actual.shape:
        return {
            "name": name,
            "shape": list(expected.shape),
            "actual_shape": list(actual.shape),
            "max_abs_error": None,
            "mean_abs_error": None,
            "passed": False,
        }
    diff = np.abs(expected.astype(np.float64) - actual.astype(np.float64))
    return {
        "name": name,
        "shape": list(expected.shape),
        "actual_shape": list(actual.shape),
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

    comparisons = [
        compare_arrays(
            "output/Y", training_trace["logits"], bundle_trace["logits"]
        )
    ]
    training_layers = training_trace["hidden_layers"]
    bundle_layers = bundle_trace["hidden_layers"]
    if len(training_layers) != len(bundle_layers):
        raise ValueError("checkpoint and bundle hidden-layer counts differ")
    trace_keys = [
        "H_prime", "E_src", "E_dst", "S", "Z", "M", "Delta",
        "U", "Sum", "inv", "alpha", "H_agg_pre_bias", "H_agg",
    ]
    for layer_index, (training_layer, bundle_layer) in enumerate(
        zip(training_layers, bundle_layers)
    ):
        comparisons.append(
            compare_arrays(
                f"hidden_layer_{layer_index}/concat",
                training_layer["concat"],
                bundle_layer["concat"],
            )
        )
        if len(training_layer["heads"]) != len(bundle_layer["heads"]):
            raise ValueError(
                f"checkpoint and bundle head counts differ at layer {layer_index}"
            )
        for head_index, (training_head, bundle_head) in enumerate(
            zip(training_layer["heads"], bundle_layer["heads"])
        ):
            for key in trace_keys:
                comparisons.append(
                    compare_arrays(
                        f"hidden_layer_{layer_index}/head_{head_index}/{key}",
                        training_head[key],
                        bundle_head[key],
                    )
                )

    training_outputs = training_trace["output_heads"]
    bundle_outputs = bundle_trace["output_heads"]
    if len(training_outputs) != len(bundle_outputs):
        raise ValueError("checkpoint and bundle K_out differ")
    for head_index, (training_head, bundle_head) in enumerate(
        zip(training_outputs, bundle_outputs)
    ):
        for key in trace_keys:
            comparisons.append(
                compare_arrays(
                    f"output/head_{head_index}/{key}",
                    training_head[key],
                    bundle_head[key],
                )
            )
        comparisons.append(
            compare_arrays(
                f"output/head_{head_index}/biased_value",
                training_head["H_agg"],
                bundle_head["H_agg"],
            )
        )

    summary = {
        "dataset": args.dataset,
        "comparison_count": len(comparisons),
        "passed": all(item["passed"] for item in comparisons),
        "comparisons": comparisons,
    }
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if not summary["passed"]:
        raise RuntimeError(
            f"checkpoint/bundle parity failed; see {output_path}"
        )


if __name__ == "__main__":
    main()
