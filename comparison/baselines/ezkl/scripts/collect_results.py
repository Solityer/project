#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
DATASETS_PATH = BASE_DIR / "configs" / "full_datasets.json"
RESULTS_DIR = BASE_DIR / "results"
DATASET_ORDER = ["cora_full", "citeseer_full", "pubmed_full", "ppi_full_formal", "ogbn_arxiv_full"]


COMPARISON_COLUMNS = [
    "dataset",
    "full_dataset",
    "node_count",
    "edge_count",
    "num_classes",
    "model_arch_id",
    "model_param_id",
    "quant_cfg_id",
    "preprocess_id",
    "batching_rule",
    "subgraph_rule",
    "self_loop_rule",
    "edge_sort_rule",
    "chunking_rule",
    "main_commitment_time_ms",
    "main_prove_time_ms",
    "main_verify_time_ms",
    "main_proof_size_bytes",
    "ezkl_status",
    "ezkl_route_id",
    "ezkl_input_visibility",
    "ezkl_output_visibility",
    "ezkl_param_visibility",
    "ezkl_k",
    "ezkl_num_inner_cols",
    "ezkl_commitment_time_ms",
    "ezkl_commitment_time_semantics",
    "ezkl_prove_time_ms",
    "ezkl_verify_time_ms",
    "ezkl_proof_size_bytes",
    "ezkl_setup_ms",
    "ezkl_gen_witness_ms",
    "ezkl_compile_ms",
    "ezkl_gen_settings_ms",
    "ezkl_verify_passed",
    "ezkl_block_reason",
]


EZKL_COLUMNS = [
    "dataset",
    "full_dataset",
    "route_id",
    "status",
    "node_count",
    "edge_count",
    "num_classes",
    "model_arch_id",
    "model_param_id",
    "quant_cfg_id",
    "preprocess_id",
    "batching_rule",
    "subgraph_rule",
    "self_loop_rule",
    "edge_sort_rule",
    "chunking_rule",
    "input_visibility",
    "output_visibility",
    "param_visibility",
    "k",
    "num_inner_cols",
    "commitment_time_ms",
    "commitment_time_semantics",
    "prove_time_ms",
    "verify_time_ms",
    "proof_size_bytes",
    "setup_ms",
    "gen_witness_ms",
    "compile_ms",
    "gen_settings_ms",
    "verify_passed",
    "block_reason",
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def parse_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (int, float, bool)):
        return value
    text = str(value)
    if text.lower() == "true":
        return True
    if text.lower() == "false":
        return False
    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        return text


def dataset_specs() -> dict[str, Any]:
    return json.loads(DATASETS_PATH.read_text())["datasets"]


def pending_manifest(dataset_id: str, spec: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": dataset_id,
        "full_dataset": True,
        "route_id": "ezkl_timing_private_input_public_output_full",
        "status": "pending_full_resource",
        "node_count": parse_value(reference.get("node_count")),
        "edge_count": parse_value(reference.get("edge_count")),
        "num_classes": parse_value(reference.get("C")),
        "model_arch_id": reference.get("model_arch_id"),
        "model_param_id": reference.get("model_param_id"),
        "quant_cfg_id": reference.get("quant_cfg_id"),
        "preprocess_id": spec.get("preprocess_id"),
        "batching_rule": spec.get("batching_rule"),
        "subgraph_rule": spec.get("subgraph_rule"),
        "self_loop_rule": spec.get("self_loop_rule"),
        "edge_sort_rule": spec.get("edge_sort_rule"),
        "chunking_rule": spec.get("chunking_rule"),
        "input_visibility": "private",
        "output_visibility": "public",
        "param_visibility": "fixed",
        "k": None,
        "num_inner_cols": None,
        "commitment_time_ms": None,
        "commitment_time_semantics": "included_in_ezkl_prove_ms_unless_instrumented",
        "prove_time_ms": None,
        "verify_time_ms": None,
        "proof_size_bytes": None,
        "setup_ms": None,
        "gen_witness_ms": None,
        "compile_ms": None,
        "gen_settings_ms": None,
        "verify_passed": False,
        "block_reason": "not_run",
    }


def normalize_manifest(manifest: dict[str, Any], dataset_id: str) -> dict[str, Any]:
    return {
        key: manifest.get(key)
        for key in EZKL_COLUMNS
    } | {"dataset": dataset_id}


def comparison_row(dataset_id: str, spec: dict[str, Any], reference: dict[str, Any], ezkl: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": dataset_id,
        "full_dataset": True,
        "node_count": ezkl.get("node_count") or parse_value(reference.get("node_count")),
        "edge_count": ezkl.get("edge_count") or parse_value(reference.get("edge_count")),
        "num_classes": ezkl.get("num_classes") or parse_value(reference.get("C")),
        "model_arch_id": ezkl.get("model_arch_id") or reference.get("model_arch_id"),
        "model_param_id": ezkl.get("model_param_id") or reference.get("model_param_id"),
        "quant_cfg_id": ezkl.get("quant_cfg_id") or reference.get("quant_cfg_id"),
        "preprocess_id": ezkl.get("preprocess_id") or spec.get("preprocess_id"),
        "batching_rule": ezkl.get("batching_rule") or spec.get("batching_rule"),
        "subgraph_rule": ezkl.get("subgraph_rule") or spec.get("subgraph_rule"),
        "self_loop_rule": ezkl.get("self_loop_rule") or spec.get("self_loop_rule"),
        "edge_sort_rule": ezkl.get("edge_sort_rule") or spec.get("edge_sort_rule"),
        "chunking_rule": ezkl.get("chunking_rule") or spec.get("chunking_rule"),
        "main_commitment_time_ms": parse_value(reference.get("commitment_time_ms")),
        "main_prove_time_ms": parse_value(reference.get("prove_time_ms")),
        "main_verify_time_ms": parse_value(reference.get("verify_time_ms")),
        "main_proof_size_bytes": parse_value(reference.get("proof_size_bytes")),
        "ezkl_status": ezkl.get("status"),
        "ezkl_route_id": ezkl.get("route_id"),
        "ezkl_input_visibility": ezkl.get("input_visibility"),
        "ezkl_output_visibility": ezkl.get("output_visibility"),
        "ezkl_param_visibility": ezkl.get("param_visibility"),
        "ezkl_k": ezkl.get("k"),
        "ezkl_num_inner_cols": ezkl.get("num_inner_cols"),
        "ezkl_commitment_time_ms": ezkl.get("commitment_time_ms"),
        "ezkl_commitment_time_semantics": ezkl.get("commitment_time_semantics"),
        "ezkl_prove_time_ms": ezkl.get("prove_time_ms"),
        "ezkl_verify_time_ms": ezkl.get("verify_time_ms"),
        "ezkl_proof_size_bytes": ezkl.get("proof_size_bytes"),
        "ezkl_setup_ms": ezkl.get("setup_ms"),
        "ezkl_gen_witness_ms": ezkl.get("gen_witness_ms"),
        "ezkl_compile_ms": ezkl.get("compile_ms"),
        "ezkl_gen_settings_ms": ezkl.get("gen_settings_ms"),
        "ezkl_verify_passed": ezkl.get("verify_passed"),
        "ezkl_block_reason": ezkl.get("block_reason"),
    }


def write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    specs = dataset_specs()
    ezkl_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []

    for dataset_id in DATASET_ORDER:
        spec = specs[dataset_id]
        reference = load_json(BASE_DIR / spec["reference_manifest"])
        manifest_path = BASE_DIR / spec["run_dir"] / "run_manifest.json"
        manifest = load_json(manifest_path)
        ezkl = normalize_manifest(manifest, dataset_id) if manifest else pending_manifest(dataset_id, spec, reference)
        ezkl_rows.append(ezkl)
        comparison_rows.append(comparison_row(dataset_id, spec, reference, ezkl))

    (RESULTS_DIR / "ezkl_full_results.json").write_text(json.dumps(ezkl_rows, indent=2))
    (RESULTS_DIR / "main_vs_ezkl_full_comparison.json").write_text(json.dumps(comparison_rows, indent=2))
    write_csv(RESULTS_DIR / "ezkl_full_results.csv", EZKL_COLUMNS, ezkl_rows)
    write_csv(RESULTS_DIR / "main_vs_ezkl_full_comparison.csv", COMPARISON_COLUMNS, comparison_rows)
    print(json.dumps({"rows": len(comparison_rows), "results_dir": RESULTS_DIR.as_posix()}, indent=2))


if __name__ == "__main__":
    main()
