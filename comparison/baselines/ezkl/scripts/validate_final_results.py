#!/usr/bin/env python3
"""Validate the final strict ezkl native GAT comparison artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path


EZKL_ROOT = Path(__file__).resolve().parents[1]
RESULTS = EZKL_ROOT / "results"
RUNS = EZKL_ROOT / "runs"
ROUTE_ID = "ezkl_gat_native_compressed_csr_attention_full"
DATASETS = [
    "cora_full",
    "citeseer_full",
    "pubmed_full",
    "ppi_full_formal",
    "ogbn_arxiv_full",
]
DISPLAY_BY_DATASET = {
    "cora_full": "Cora",
    "citeseer_full": "Citeseer",
    "pubmed_full": "Pubmed",
    "ppi_full_formal": "PPI",
    "ogbn_arxiv_full": "ogbn-arxiv",
}
REQUIRED_RUN_ARTIFACTS = [
    "run_manifest.json",
    "model_info.json",
    "graph_attention_spec.json",
    "final_logits_binding.json",
    "formal_tables.json",
    "formal_table_hashes.json",
    "native_full_witness_preflight.json",
    "keygen_memory_profile.json",
    "prove_memory_profile.json",
]
METRICS = [
    "commitment_time_ms",
    "prove_time_ms",
    "verify_time_ms",
    "proof_size_bytes",
]


def read_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def fail(message: str) -> None:
    raise SystemExit(f"final_result_validation=FAILED: {message}")


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def rows_by_dataset(rows: list[dict]) -> dict[str, dict]:
    return {str(row.get("dataset")): row for row in rows}


def validate_rows(rows: list[dict]) -> dict[str, dict]:
    require(len(rows) == 5, f"expected 5 result rows, got {len(rows)}")
    by_dataset = rows_by_dataset(rows)
    require(set(by_dataset) == set(DATASETS), "result datasets do not match final full-dataset set")
    for dataset in DATASETS:
        row = by_dataset[dataset]
        require(row.get("status") == "verified", f"{dataset} is not verified")
        require(row.get("route_id") == ROUTE_ID, f"{dataset} route_id mismatch")
        require(row.get("strict_public_input_equivalent") is True, f"{dataset} strict flag false")
        require(row.get("full_gat_forward_in_ezkl_proof") is True, f"{dataset} full GAT flag false")
        require(row.get("final_logits_derivation_complete") is True, f"{dataset} final logits flag false")
        require(row.get("no_precomputed_logits_bypass") is True, f"{dataset} precomputed logits bypass flag false")
        for metric in METRICS:
            require(row.get(metric) is not None, f"{dataset} missing {metric}")
    return by_dataset


def validate_result_csv(path: Path, expected: dict[str, dict]) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 5, f"{path.name} row count mismatch")
    for row in rows:
        dataset = row.get("dataset")
        require(dataset in expected, f"{path.name} unexpected dataset {dataset}")
        require(row.get("route_id", ROUTE_ID) == ROUTE_ID, f"{dataset} CSV route mismatch")
        require(row.get("status", "verified") == "verified", f"{dataset} CSV status mismatch")
        for metric in METRICS:
            if metric in row:
                require(row[metric] != "", f"{dataset} CSV missing {metric}")


def validate_comparison_csv(path: Path) -> None:
    expected_displays = set(DISPLAY_BY_DATASET.values())
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 5, f"{path.name} row count mismatch")
    require({row.get("dataset") for row in rows} == expected_displays, "comparison CSV dataset mismatch")
    for row in rows:
        dataset = row.get("dataset")
        require(row.get("ezkl_status") == "verified", f"{dataset} comparison status mismatch")
        require(row.get("ezkl_route_id") == ROUTE_ID, f"{dataset} comparison route mismatch")
        for metric in METRICS:
            require(row.get(f"ezkl_{metric}") not in {"", None}, f"{dataset} comparison missing {metric}")


def validate_manifests(expected: dict[str, dict]) -> None:
    for dataset in DATASETS:
        run_dir = RUNS / dataset
        for name in REQUIRED_RUN_ARTIFACTS:
            require((run_dir / name).exists(), f"{dataset} missing {name}")
        manifest = read_json(run_dir / "run_manifest.json")
        require(isinstance(manifest, dict), f"{dataset} manifest is not an object")
        row = expected[dataset]
        require(manifest.get("status") == row.get("status"), f"{dataset} manifest status mismatch")
        require(manifest.get("route_id") == ROUTE_ID, f"{dataset} manifest route mismatch")
        for metric in METRICS:
            require(manifest.get(metric) == row.get(metric), f"{dataset} manifest {metric} mismatch")
        binding = read_json(run_dir / "final_logits_binding.json")
        require(isinstance(binding, dict), f"{dataset} binding is not an object")
        require(binding.get("no_precomputed_logits_bypass") is True, f"{dataset} binding bypass flag false")
        preflight = read_json(run_dir / "native_full_witness_preflight.json")
        require(isinstance(preflight, dict), f"{dataset} preflight is not an object")
        require(preflight.get("witness_specific_table") is False, f"{dataset} witness-specific table")


def validate_reports(expected: dict[str, dict]) -> None:
    summary = (RESULTS / "summary.md").read_text(encoding="utf-8")
    audit = (RESULTS / "control_variable_audit.md").read_text(encoding="utf-8")
    comparison = read_json(RESULTS / "main_vs_ezkl_full_comparison.json")
    require(isinstance(comparison, dict), "comparison JSON is not an object")
    comparison_rows = comparison.get("rows")
    require(isinstance(comparison_rows, list) and len(comparison_rows) == 5, "comparison row count mismatch")
    require("Verified ezkl full-dataset rows: 5/5" in summary, "summary verified count mismatch")
    require(ROUTE_ID in summary, "summary route missing")
    require("ogbn_arxiv_full" in audit and "ppi_full_formal" in audit, "control audit dataset coverage missing")
    for dataset, row in expected.items():
        require(str(row["proof_size_bytes"]) in summary, f"{dataset} proof size missing from summary")


def main() -> int:
    results = read_json(RESULTS / "gat_ezkl_full_results.json")
    require(isinstance(results, dict), "result JSON is not an object")
    rows = results.get("rows")
    require(isinstance(rows, list), "result JSON rows missing")
    expected = validate_rows(rows)
    validate_result_csv(RESULTS / "gat_ezkl_full_results.csv", expected)
    validate_comparison_csv(RESULTS / "main_vs_ezkl_full_comparison.csv")
    validate_manifests(expected)
    validate_reports(expected)
    print("final_result_validation=OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
