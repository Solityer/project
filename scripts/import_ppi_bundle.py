#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
from typing import Dict, Iterable


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_manifest(bundle_dir: pathlib.Path) -> Dict[str, object]:
    manifest_path = bundle_dir / "manifest.json"
    require(manifest_path.exists(), f"missing manifest.json in {bundle_dir}")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(isinstance(data, dict), "manifest.json must contain a JSON object")
    return data


def require_keys(manifest: Dict[str, object], keys: Iterable[str]) -> None:
    for key in keys:
        require(key in manifest, f"missing required manifest field: {key}")


def parse_hidden_profile(manifest: Dict[str, object]) -> str:
    raw = manifest.get("hidden_profile")
    require(isinstance(raw, list) and raw, "hidden_profile must be a non-empty array")
    parts = []
    for entry in raw:
        require(isinstance(entry, dict), "hidden_profile entries must be objects")
        parts.append(f"{int(entry['head_count'])}x{int(entry['head_dim'])}")
    return ",".join(parts)


def validate_ppi_bundle(manifest: Dict[str, object], bundle_dir: pathlib.Path) -> None:
    require_keys(
        manifest,
        (
            "family_schema_version",
            "L",
            "hidden_profile",
            "hidden_layers",
            "hidden_head_specs",
            "d_in_profile",
            "K_out",
            "output_head_specs",
            "C",
            "output_average_rule",
            "model_arch_id",
            "model_param_id",
            "quant_cfg_id",
            "static_table_id",
            "degree_bound_id",
            "task_type",
            "report_unit",
            "batching_rule",
            "subgraph_rule",
            "self_loop_rule",
            "edge_sort_rule",
        ),
    )
    require((bundle_dir / "tensors.txt").exists(), f"missing tensors.txt in {bundle_dir}")
    require(str(manifest["family_schema_version"]) == "multi_layer_multi_head_v2", "unsupported family_schema_version")
    require(int(manifest["L"]) == 2, "PPI formal config requires L=2")
    require(parse_hidden_profile(manifest) == "1x8", "PPI formal config requires hidden_profile=[1x8]")
    d_in_profile = manifest["d_in_profile"]
    require(isinstance(d_in_profile, list) and [int(value) for value in d_in_profile] == [50], "PPI formal config requires d_in_profile=[50]")
    require(int(manifest["K_out"]) == 1, "PPI formal config requires K_out=1")
    require(int(manifest["C"]) == 121, "PPI formal config requires C=121")
    require(str(manifest["task_type"]) == "inductive_multi_graph_node_classification", "PPI bundle task_type mismatch")
    require(str(manifest["report_unit"]) == "graph", "PPI bundle report_unit mismatch")
    require(str(manifest["batching_rule"]) == "multi_graph_batch", "PPI bundle batching_rule mismatch")
    require(str(manifest["subgraph_rule"]) == "whole_graph", "PPI bundle subgraph_rule mismatch")
    require(str(manifest["self_loop_rule"]) == "per_node", "PPI bundle self_loop_rule mismatch")
    require(str(manifest["edge_sort_rule"]) == "edge_gid_then_dst_stable", "PPI bundle edge_sort_rule mismatch")
    require(str(manifest["output_average_rule"]) == "per_head_bias_then_arithmetic_mean", "PPI bundle output_average_rule mismatch")


def install_bundle(source_dir: pathlib.Path, output_dir: pathlib.Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.copytree(source_dir, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--output-dir", default="artifacts/checkpoints/ppi_gat")
    args = parser.parse_args()

    source_dir = pathlib.Path(args.bundle_dir).resolve()
    output_dir = pathlib.Path(args.output_dir).resolve()
    require(source_dir.exists(), f"bundle dir does not exist: {source_dir}")
    require(source_dir.is_dir(), f"bundle dir is not a directory: {source_dir}")

    manifest = load_manifest(source_dir)
    validate_ppi_bundle(manifest, source_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    install_bundle(source_dir, output_dir)
    print(str(output_dir))


if __name__ == "__main__":
    main()
