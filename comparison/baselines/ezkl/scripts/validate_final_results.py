#!/usr/bin/env python3
"""Validate the final strict ezkl native GAT comparison artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path


EZKL_ROOT = Path(__file__).resolve().parents[1]
RESULTS = EZKL_ROOT / "results"
RUNS = EZKL_ROOT / "runs"
CPU_ROUTE_ID = "ezkl_gat_native_compressed_csr_attention_full"
GPU_ROUTE_ID = "ezkl_gat_native_compressed_csr_attention_full_gpu"
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
FORBIDDEN_ROUTE_TOKENS = [
    "output-binding",
    "output_binding",
    "digest",
    "folding",
    "identity",
    "subset",
    "shard",
    "sample",
    "precomputed_logits",
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


def validate_rows(rows: list[dict], route_id: str, require_verified: bool) -> dict[str, dict]:
    require(len(rows) == 5, f"expected 5 result rows, got {len(rows)}")
    by_dataset = rows_by_dataset(rows)
    require(set(by_dataset) == set(DATASETS), "result datasets do not match final full-dataset set")
    for dataset in DATASETS:
        row = by_dataset[dataset]
        require(row.get("route_id") == route_id, f"{dataset} route_id mismatch")
        if not require_verified:
            if row.get("status") == "verified":
                require(row.get("strict_public_input_equivalent") is True, f"{dataset} strict flag false")
                require(row.get("full_gat_forward_in_ezkl_proof") is True, f"{dataset} full GAT flag false")
                require(row.get("final_logits_derivation_complete") is True, f"{dataset} final logits flag false")
                require(row.get("no_precomputed_logits_bypass") is True, f"{dataset} precomputed logits bypass flag false")
                for metric in METRICS:
                    require(row.get(metric) is not None, f"{dataset} missing {metric}")
            else:
                require(row.get("strict_public_input_equivalent") is not True, f"{dataset} unverified strict flag true")
            continue
        require(row.get("status") == "verified", f"{dataset} is not verified")
        require(row.get("strict_public_input_equivalent") is True, f"{dataset} strict flag false")
        require(row.get("full_gat_forward_in_ezkl_proof") is True, f"{dataset} full GAT flag false")
        require(row.get("final_logits_derivation_complete") is True, f"{dataset} final logits flag false")
        require(row.get("no_precomputed_logits_bypass") is True, f"{dataset} precomputed logits bypass flag false")
        for metric in METRICS:
            require(row.get(metric) is not None, f"{dataset} missing {metric}")
    return by_dataset


def validate_result_csv(path: Path, expected: dict[str, dict], route_id: str, require_verified: bool) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 5, f"{path.name} row count mismatch")
    for row in rows:
        dataset = row.get("dataset")
        require(dataset in expected, f"{path.name} unexpected dataset {dataset}")
        require(row.get("route_id", route_id) == route_id, f"{dataset} CSV route mismatch")
        if require_verified:
            require(row.get("status", "verified") == "verified", f"{dataset} CSV status mismatch")
        for metric in METRICS:
            if metric in row:
                if require_verified:
                    require(row[metric] != "", f"{dataset} CSV missing {metric}")


def validate_comparison_csv(path: Path, route_id: str, require_verified: bool) -> None:
    expected_displays = set(DISPLAY_BY_DATASET.values())
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 5, f"{path.name} row count mismatch")
    require({row.get("dataset") for row in rows} == expected_displays, "comparison CSV dataset mismatch")
    for row in rows:
        dataset = row.get("dataset")
        if require_verified:
            require(row.get("ezkl_status") == "verified", f"{dataset} comparison status mismatch")
        require(row.get("ezkl_route_id") == route_id, f"{dataset} comparison route mismatch")
        for metric in METRICS:
            if require_verified:
                require(row.get(f"ezkl_{metric}") not in {"", None}, f"{dataset} comparison missing {metric}")


def validate_manifests(expected: dict[str, dict], route_id: str, run_suffix: str = "") -> None:
    for dataset in DATASETS:
        run_dir = RUNS / dataset / run_suffix if run_suffix else RUNS / dataset
        for name in REQUIRED_RUN_ARTIFACTS:
            require((run_dir / name).exists(), f"{dataset} missing {name}")
        manifest = read_json(run_dir / "run_manifest.json")
        require(isinstance(manifest, dict), f"{dataset} manifest is not an object")
        row = expected[dataset]
        require(manifest.get("status") == row.get("status"), f"{dataset} manifest status mismatch")
        require(manifest.get("route_id") == route_id, f"{dataset} manifest route mismatch")
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
    require("已验证全量数据集行数：5/5" in summary, "summary verified count mismatch")
    require(CPU_ROUTE_ID in summary, "summary route missing")
    require("ogbn_arxiv_full" in audit and "ppi_full_formal" in audit, "control audit dataset coverage missing")
    for dataset, row in expected.items():
        require(str(row["proof_size_bytes"]) in summary, f"{dataset} proof size missing from summary")


def validate_gpu_verified_rows(rows: list[dict]) -> None:
    for row in rows:
        dataset = str(row.get("dataset"))
        require(row.get("route_id") == GPU_ROUTE_ID, f"{dataset} GPU route mismatch")
        require(row.get("backend") == "gpu", f"{dataset} GPU backend mismatch")
        require(row.get("gpu_backend_required") is True, f"{dataset} GPU required flag false")
        require(row.get("gpu_backend_active") is True, f"{dataset} GPU active flag false")
        require(row.get("cpu_fallback_used") is False, f"{dataset} CPU fallback flag true")
        require(row.get("full_dataset") is True, f"{dataset} full dataset flag false")
        require(row.get("verify_passed") is True, f"{dataset} verify flag false")
        require(row.get("gpu_backend_library_arch") not in {"", None}, f"{dataset} GPU library arch missing")
        require(row.get("gpu_device_name") not in {"", None}, f"{dataset} GPU device name missing")
        require(row.get("gpu_device_count") not in {"", None, 0}, f"{dataset} GPU device count missing")
        text = " ".join(
            str(row.get(key, ""))
            for key in ["route_id", "command", "block_reason", "source_run_dir"]
        ).lower()
        for token in FORBIDDEN_ROUTE_TOKENS:
            require(token not in text, f"{dataset} forbidden GPU route token present: {token}")


def main() -> int:
    results = read_json(RESULTS / "gat_ezkl_full_results.json")
    require(isinstance(results, dict), "result JSON is not an object")
    rows = results.get("rows")
    require(isinstance(rows, list), "result JSON rows missing")
    expected = validate_rows(rows, CPU_ROUTE_ID, True)
    validate_result_csv(RESULTS / "gat_ezkl_full_results.csv", expected, CPU_ROUTE_ID, True)
    validate_comparison_csv(RESULTS / "main_vs_ezkl_full_comparison.csv", CPU_ROUTE_ID, True)
    validate_manifests(expected, CPU_ROUTE_ID)
    validate_reports(expected)

    gpu_status = "missing"
    gpu_json = RESULTS / "gat_ezkl_gpu_full_results.json"
    if gpu_json.exists():
        gpu_results = read_json(gpu_json)
        require(isinstance(gpu_results, dict), "GPU result JSON is not an object")
        gpu_rows = gpu_results.get("rows")
        require(isinstance(gpu_rows, list), "GPU result JSON rows missing")
        gpu_expected = validate_rows(gpu_rows, GPU_ROUTE_ID, False)
        validate_result_csv(RESULTS / "gat_ezkl_gpu_full_results.csv", gpu_expected, GPU_ROUTE_ID, False)
        validate_comparison_csv(RESULTS / "main_vs_ezkl_gpu_full_comparison.csv", GPU_ROUTE_ID, False)
        gpu_verified = [row for row in gpu_rows if row.get("status") == "verified"]
        if len(gpu_verified) == 5:
            validate_gpu_verified_rows(gpu_rows)
            validate_manifests(gpu_expected, GPU_ROUTE_ID, "gpu")
            gpu_status = "verified_5/5"
        else:
            status_md = RESULTS / "gpu_route_status.md"
            require(status_md.exists(), "GPU route is not fully verified and gpu_route_status.md is missing")
            status_text = status_md.read_text(encoding="utf-8")
            require("未标记为 verified" in status_text, "GPU status report must state that GPU is not verified")
            gpu_status = f"not_verified_{len(gpu_verified)}/5"

    print(f"final_result_validation=OK cpu_verified=5/5 gpu_status={gpu_status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
