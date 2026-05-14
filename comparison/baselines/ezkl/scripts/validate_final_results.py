#!/usr/bin/env python3
"""Validate final CPU-only strict ezkl native GAT comparison artifacts."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from typing import Any


EZKL_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EZKL_ROOT.parents[2]
RESULTS = EZKL_ROOT / "results"
RUNS = EZKL_ROOT / "runs"
CPU_ROUTE_ID = "ezkl_gat_native_compressed_csr_attention_full"
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
    "proof.json",
    "verify.json",
    "model_info.json",
    "graph_attention_spec.json",
    "final_logits_binding.json",
    "formal_tables.json",
    "formal_table_hashes.json",
    "native_full_witness_preflight.json",
    "keygen_memory_profile.json",
    "prove_memory_profile.json",
]
REQUIRED_RESULTS = [
    "gat_ezkl_full_results.json",
    "gat_ezkl_full_results.csv",
    "main_vs_ezkl_full_comparison.json",
    "main_vs_ezkl_full_comparison.csv",
    "summary.md",
    "strict_audit_report.md",
    "control_variable_audit.md",
    "native_full_experiment_report.md",
    "ppi_ogbn_arxiv_audit.md",
    "cleanup_report.md",
]
METRICS = [
    "commitment_time_ms",
    "prove_time_ms",
    "verify_time_ms",
    "proof_size_bytes",
]
STRICT_TIMINGS = [
    "gen_settings_ms",
    "compile_ms",
    "setup_ms",
    "gen_witness_ms",
    "prove_time_ms",
    "verify_time_ms",
]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def fail(message: str) -> None:
    raise SystemExit(f"final_result_validation=FAILED: {message}")


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def rows_by_dataset(rows: list[dict]) -> dict[str, dict]:
    return {str(row.get("dataset")): row for row in rows}


def empty_field(value: object) -> bool:
    return value is None or value == ""


def metric_cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def contains_manual_continuation(value: object) -> bool:
    return isinstance(value, str) and "manual continuation" in value.lower()


def validate_rows(rows: list[dict], audits: dict) -> dict[str, dict]:
    require(len(rows) == 5, f"expected 5 result rows, got {len(rows)}")
    by_dataset = rows_by_dataset(rows)
    require(set(by_dataset) == set(DATASETS), "result datasets do not match final full-dataset set")
    require(set(audits) == set(DATASETS), "result audits do not cover final full-dataset set")
    for dataset in DATASETS:
        row = by_dataset[dataset]
        status = row.get("status")
        require(row.get("backend") == "cpu", f"{dataset} backend mismatch")
        require(row.get("route_id") == CPU_ROUTE_ID, f"{dataset} route_id mismatch")
        require(row.get("statement_route_id") == CPU_ROUTE_ID, f"{dataset} statement route mismatch")
        require(row.get("final_logits_derivation_complete") is True, f"{dataset} final logits flag false")
        require(row.get("no_precomputed_logits_bypass") is True, f"{dataset} precomputed logits bypass flag false")
        require(row.get("audit_error_count") == 0, f"{dataset} audit errors present")
        for key in ["run_id", "run_started_at", "run_completed_at"]:
            require(bool(row.get(key)), f"{dataset} missing {key}")
        audit = audits[dataset]
        require(audit.get("errors") == [], f"{dataset} audit errors non-empty")

        if status == "verified":
            require(row.get("verify_passed") is True, f"{dataset} verify_passed false")
            require(row.get("proof_generated") is True, f"{dataset} proof_generated false")
            require(row.get("strict_public_input_equivalent") is True, f"{dataset} strict flag false")
            require(row.get("full_gat_forward_in_ezkl_proof") is True, f"{dataset} full GAT flag false")
            require(row.get("manifest_strict_check") == "passed", f"{dataset} manifest strict check failed")
            require(row.get("proof_size_matches_manifest") is True, f"{dataset} proof size audit mismatch")
            require(row.get("verify_json_status") == "verified", f"{dataset} verify JSON status mismatch")
            require(row.get("verify_json_verify_passed") is True, f"{dataset} verify JSON flag false")
            require(row.get("final_benchmark_included") is True, f"{dataset} verified row not included")
            require(empty_field(row.get("block_reason")), f"{dataset} block_reason is not empty")
            require(empty_field(row.get("failure_step")), f"{dataset} failure_step is not empty")
            for metric in METRICS:
                require(row.get(metric) is not None, f"{dataset} missing {metric}")
            require(audit.get("status") == "passed", f"{dataset} audit status failed")
        elif status in {"blocked", "failed"}:
            require(row.get("verify_passed") is False, f"{dataset} blocked verify_passed not false")
            require(row.get("proof_generated") is False, f"{dataset} blocked proof_generated not false")
            require(row.get("manifest_strict_check") == status, f"{dataset} blocked audit status mismatch")
            require(row.get("final_benchmark_included") is False, f"{dataset} blocked row included in benchmark")
            require(not empty_field(row.get("block_reason")), f"{dataset} blocked missing block_reason")
            require(not empty_field(row.get("failure_step")), f"{dataset} blocked missing failure_step")
            require(row.get("verify_json_status") in {"blocked", "failed"}, f"{dataset} blocked verify JSON status mismatch")
            require(empty_field(row.get("verify_json_verify_passed")), f"{dataset} blocked verify JSON flag should be absent")
            require(audit.get("status") == status, f"{dataset} blocked audit status failed")
        else:
            fail(f"{dataset} has unexpected status {status}")
    return by_dataset


def validate_result_csv(path: Path, expected: dict[str, dict]) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 5, f"{path.name} row count mismatch")
    for row in rows:
        dataset = row.get("dataset")
        require(dataset in expected, f"{path.name} unexpected dataset {dataset}")
        require(row.get("backend") == "cpu", f"{dataset} CSV backend mismatch")
        require(row.get("route_id") == CPU_ROUTE_ID, f"{dataset} CSV route mismatch")
        expected_status = expected[dataset].get("status")
        require(row.get("status") == expected_status, f"{dataset} CSV status mismatch")
        if expected_status == "verified":
            require(row.get("manifest_strict_check") == "passed", f"{dataset} CSV strict audit mismatch")
            require(row.get("final_benchmark_included") in {"True", "true", True}, f"{dataset} CSV inclusion mismatch")
            for metric in METRICS:
                require(row.get(metric) not in {"", None}, f"{dataset} CSV missing {metric}")
        else:
            require(row.get("manifest_strict_check") == expected_status, f"{dataset} CSV blocked audit mismatch")
            require(row.get("final_benchmark_included") in {"False", "false", False}, f"{dataset} CSV blocked inclusion mismatch")
            require(row.get("block_reason") not in {"", None}, f"{dataset} CSV blocked reason missing")


def validate_comparison_csv(path: Path, expected: dict[str, dict]) -> None:
    display_to_dataset = {display: dataset for dataset, display in DISPLAY_BY_DATASET.items()}
    expected_displays = set(DISPLAY_BY_DATASET.values())
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 5, f"{path.name} row count mismatch")
    require({row.get("dataset") for row in rows} == expected_displays, "comparison CSV dataset mismatch")
    for row in rows:
        dataset = row.get("dataset")
        source_dataset = display_to_dataset[str(dataset)]
        expected_status = expected[source_dataset].get("status")
        require(row.get("backend") == "cpu", f"{dataset} comparison backend mismatch")
        require(row.get("ezkl_status") == expected_status, f"{dataset} comparison status mismatch")
        require(row.get("ezkl_route_id") == CPU_ROUTE_ID, f"{dataset} comparison route mismatch")
        if expected_status == "verified":
            require(row.get("manifest_strict_check") == "passed", f"{dataset} comparison audit mismatch")
            for metric in METRICS:
                require(row.get(f"ezkl_{metric}") not in {"", None}, f"{dataset} comparison missing {metric}")
        else:
            require(row.get("manifest_strict_check") == expected_status, f"{dataset} comparison blocked audit mismatch")
            require(row.get("ezkl_block_reason") not in {"", None}, f"{dataset} comparison block reason missing")
            for metric in METRICS:
                require(row.get(f"ezkl_{metric}") in {"", None}, f"{dataset} blocked comparison metric should be blank")


def validate_manifest_text_tails(dataset: str, manifest: dict) -> None:
    require(not contains_manual_continuation(manifest.get("stdout_tail")), f"{dataset} stdout manual continuation")
    require(not contains_manual_continuation(manifest.get("stderr_tail")), f"{dataset} stderr manual continuation")
    stage_records = manifest.get("stage_records")
    status = manifest.get("status")
    require(isinstance(stage_records, list) and len(stage_records) >= 1, f"{dataset} missing stage records")
    if status == "verified":
        require(len(stage_records) == 6, f"{dataset} verified stage record count mismatch")
    for index, record in enumerate(stage_records):
        require(isinstance(record, dict), f"{dataset} stage record is not object")
        if status == "verified" or index < len(stage_records) - 1:
            require(record.get("returncode") == 0, f"{dataset} stage {record.get('phase')} returncode nonzero")
        elif status in {"blocked", "failed"}:
            require(record.get("returncode") not in {0, None}, f"{dataset} blocked final stage returncode missing")
        require(record.get("manual_continuation_detected") is False, f"{dataset} stage manual flag")
        require(not contains_manual_continuation(record.get("stdout_tail")), f"{dataset} stage stdout manual")
        require(not contains_manual_continuation(record.get("stderr_tail")), f"{dataset} stage stderr manual")


def validate_manifests(expected: dict[str, dict]) -> None:
    for dataset in DATASETS:
        run_dir = RUNS / dataset
        for name in REQUIRED_RUN_ARTIFACTS:
            require((run_dir / name).exists(), f"{dataset} missing {name}")
        manifest = read_json(run_dir / "run_manifest.json")
        verify = read_json(run_dir / "verify.json")
        binding = read_json(run_dir / "final_logits_binding.json")
        preflight = read_json(run_dir / "native_full_witness_preflight.json")
        prove_profile = read_json(run_dir / "prove_memory_profile.json")
        row = expected[dataset]

        require(isinstance(manifest, dict), f"{dataset} manifest is not an object")
        require(manifest.get("dataset") == dataset, f"{dataset} manifest dataset mismatch")
        status = manifest.get("status")
        require(status in {"verified", "blocked", "failed"}, f"{dataset} manifest status unexpected")
        require(manifest.get("backend") == "cpu", f"{dataset} manifest backend mismatch")
        require(manifest.get("route_id") == CPU_ROUTE_ID, f"{dataset} manifest route mismatch")
        require(manifest.get("statement_route_id") == CPU_ROUTE_ID, f"{dataset} manifest statement route mismatch")
        require(manifest.get("final_logits_derivation_complete") is True, f"{dataset} final logits false")
        require(manifest.get("forbidden_substitute_used") is False, f"{dataset} forbidden fallback")
        require(manifest.get("precomputed_logits_binding_used") is False, f"{dataset} precomputed logits binding")
        require(manifest.get("clean_run") is True, f"{dataset} was not a clean run")
        require(bool(manifest.get("run_id")), f"{dataset} missing run_id")
        validate_manifest_text_tails(dataset, manifest)
        require(manifest.get("commitment_time_ms") == manifest.get("setup_ms"), f"{dataset} commitment/setup mismatch")

        require(isinstance(binding, dict), f"{dataset} binding is not an object")
        require(binding.get("no_precomputed_logits_bypass") is True, f"{dataset} binding bypass flag false")
        require(binding.get("final_logits_derivation_complete") is True, f"{dataset} binding derivation false")
        require(isinstance(preflight, dict), f"{dataset} preflight is not an object")
        require(preflight.get("status") == "ready", f"{dataset} preflight not ready")
        require(preflight.get("witness_specific_table") is False, f"{dataset} witness-specific table")
        require(isinstance(prove_profile, dict), f"{dataset} prove profile is not object")

        if status == "verified":
            require(manifest.get("check_mode") == "safe", f"{dataset} check_mode not safe")
            require(manifest.get("verify_passed") is True, f"{dataset} manifest verify false")
            require(manifest.get("proof_generated") is True, f"{dataset} manifest proof false")
            require(manifest.get("strict_public_input_equivalent") is True, f"{dataset} manifest strict false")
            require(manifest.get("full_gat_forward_in_ezkl_proof") is True, f"{dataset} manifest full GAT false")
            require(manifest.get("no_precomputed_logits_bypass") is True, f"{dataset} bypass flag false")
            require(empty_field(manifest.get("block_reason")), f"{dataset} block_reason residue")
            require(empty_field(manifest.get("failure_step")), f"{dataset} failure_step residue")
            require(empty_field(manifest.get("failed_command")), f"{dataset} failed_command residue")
            for key in STRICT_TIMINGS:
                require(isinstance(manifest.get(key), (int, float)), f"{dataset} null/non-numeric {key}")
            proof_size = (run_dir / "proof.json").stat().st_size
            require(manifest.get("proof_size_bytes") == proof_size, f"{dataset} proof size mismatch")
            require(row.get("proof_size_bytes") == proof_size, f"{dataset} result proof size mismatch")
            for metric in METRICS:
                require(manifest.get(metric) == row.get(metric), f"{dataset} manifest {metric} mismatch")
            require(isinstance(verify, dict), f"{dataset} verify JSON is not object")
            require(verify.get("status") == "verified", f"{dataset} verify JSON status mismatch")
            require(verify.get("verify_passed") is True, f"{dataset} verify JSON false")
            require(verify.get("strict_public_input_equivalent") is True, f"{dataset} verify strict false")
            require(verify.get("full_gat_forward_in_ezkl_proof") is True, f"{dataset} verify full GAT false")
            require(verify.get("route_id") == CPU_ROUTE_ID, f"{dataset} verify route mismatch")
            require(prove_profile.get("proof_generated") is True, f"{dataset} prove profile proof false")
        else:
            require(manifest.get("check_mode") == "not_started", f"{dataset} blocked check_mode mismatch")
            require(manifest.get("verify_passed") is False, f"{dataset} blocked manifest verify not false")
            require(manifest.get("proof_generated") is False, f"{dataset} blocked manifest proof not false")
            require(not empty_field(manifest.get("block_reason")), f"{dataset} blocked block_reason missing")
            require(not empty_field(manifest.get("failure_step")), f"{dataset} blocked failure_step missing")
            require(not empty_field(manifest.get("failed_command")), f"{dataset} blocked failed_command missing")
            for key in ["gen_settings_ms", "compile_ms", "setup_ms", "gen_witness_ms"]:
                require(isinstance(manifest.get(key), (int, float)), f"{dataset} blocked null/non-numeric {key}")
            require(isinstance(manifest.get(manifest.get("failure_step")), (int, float)), f"{dataset} blocked failed stage untimed")
            require(manifest.get("verify_time_ms") is None, f"{dataset} blocked verify_time_ms should be null")
            require(manifest.get("proof_size_bytes") is None, f"{dataset} blocked proof_size_bytes should be null")
            require(row.get("final_benchmark_included") is False, f"{dataset} blocked row included")
            require(isinstance(verify, dict), f"{dataset} blocked verify JSON is not object")
            require(verify.get("status") == status, f"{dataset} blocked verify JSON status mismatch")
            require(verify.get("verified") is False, f"{dataset} blocked verify JSON verified not false")
            require(verify.get("verify_invoked") is False, f"{dataset} blocked verify was invoked")
            require(prove_profile.get("proof_generated") is False, f"{dataset} blocked prove profile proof true")
            require(prove_profile.get("returncode") not in {0, None}, f"{dataset} blocked prove returncode missing")


def validate_reports(expected: dict[str, dict]) -> None:
    for name in REQUIRED_RESULTS:
        require((RESULTS / name).exists(), f"missing results/{name}")
    summary = (RESULTS / "summary.md").read_text(encoding="utf-8")
    readme = (EZKL_ROOT / "README.md").read_text(encoding="utf-8")
    strict_audit = (RESULTS / "strict_audit_report.md").read_text(encoding="utf-8")
    ppi_audit = (RESULTS / "ppi_ogbn_arxiv_audit.md").read_text(encoding="utf-8")
    verified_count = sum(1 for row in expected.values() if row.get("status") == "verified")
    require(f"已验证全量数据集行数：{verified_count}/5" in summary, "summary verified count mismatch")
    require("CPU-only strict baseline" in readme, "README CPU-only language missing")
    require("不存在 GPU baseline" in readme, "README GPU non-claim missing")
    require(CPU_ROUTE_ID in summary and CPU_ROUTE_ID in readme, "route missing from summary/README")
    require("manual continuation" in readme, "README audit rule missing manual continuation")
    require("ppi_full_formal" in ppi_audit and "ogbn_arxiv_full" in ppi_audit, "PPI/ogbn audit coverage missing")
    require("returns a non-zero BLOCKED status unless every dataset is a verified benchmark row" in strict_audit, "strict audit enforcement missing")
    for dataset, row in expected.items():
        display = DISPLAY_BY_DATASET[dataset]
        require(display in summary and display in readme, f"{dataset} missing from summary/README")
        require(str(row.get("status")) in summary and str(row.get("status")) in readme, f"{dataset} status missing")
        if row.get("status") == "verified":
            for metric in METRICS:
                rendered = metric_cell(row[metric])
                require(rendered in summary, f"{dataset} {metric} missing from summary")
                require(rendered in readme, f"{dataset} {metric} missing from README")
        else:
            require(str(row.get("block_reason")) in strict_audit or str(row.get("block_reason")) in readme, f"{dataset} block reason missing from reports")


def validate_git_boundary() -> None:
    completed = subprocess.run(
        ["git", "status", "--short"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    for raw_line in completed.stdout.splitlines():
        if not raw_line.strip():
            continue
        path_part = raw_line[3:] if len(raw_line) > 3 else ""
        paths = [part.strip() for part in path_part.split(" -> ")]
        for path in paths:
            require(path.startswith("comparison/baselines/ezkl/"), f"external modified path in git status: {raw_line}")


def validate_no_junk_files() -> None:
    patterns = ["*.bak", "*.tmp", "*scratch*", "*manual*", "*old*", "*.orig"]
    ignored_dirs = {".cargo_local", ".rustup", "target", "build_gat_adapter", "__pycache__"}
    matches: list[Path] = []
    for pattern in patterns:
        for path in EZKL_ROOT.rglob(pattern):
            if any(part in ignored_dirs for part in path.relative_to(EZKL_ROOT).parts):
                continue
            matches.append(path)
    require(not matches, "junk files present: " + ", ".join(str(path.relative_to(EZKL_ROOT)) for path in matches))


def main() -> int:
    results = read_json(RESULTS / "gat_ezkl_full_results.json")
    require(isinstance(results, dict), "result JSON is not an object")
    rows = results.get("rows")
    audits = results.get("audits")
    require(isinstance(rows, list), "result JSON rows missing")
    require(isinstance(audits, dict), "result JSON audits missing")
    expected = validate_rows(rows, audits)
    validate_result_csv(RESULTS / "gat_ezkl_full_results.csv", expected)
    validate_comparison_csv(RESULTS / "main_vs_ezkl_full_comparison.csv", expected)
    validate_manifests(expected)
    validate_reports(expected)
    validate_git_boundary()
    validate_no_junk_files()
    verified = [dataset for dataset, row in expected.items() if row.get("status") == "verified"]
    blocked = [dataset for dataset, row in expected.items() if row.get("status") in {"blocked", "failed"}]
    if len(verified) == len(DATASETS):
        print("final_result_validation=OK cpu_verified=5/5")
        return 0
    print(
        f"final_result_validation=BLOCKED cpu_verified={len(verified)}/{len(DATASETS)} "
        f"blocked={','.join(blocked) or 'none'}"
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
