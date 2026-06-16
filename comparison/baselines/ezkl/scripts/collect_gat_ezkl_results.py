#!/usr/bin/env python3
"""Collect only strict, clean, CPU verified GAT-ZKML ezkl baseline results."""

from __future__ import annotations

import csv
import datetime as dt
import json
from pathlib import Path
from typing import Any


EZKL_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path("/home/pzh/project")
RESULTS = EZKL_ROOT / "results"
RUNS = EZKL_ROOT / "runs"
CPU_ROUTE_ID = "ezkl_gat_native_compressed_csr_attention_full"
DATASETS = [
    ("cora_full", "cora", "Cora", "runs/cora_full"),
    ("citeseer_full", "citeseer", "Citeseer", "runs/citeseer_full"),
    ("pubmed_full", "pubmed", "Pubmed", "runs/pubmed_full"),
    ("ppi_full_formal", "ppi", "PPI", "runs/ppi_full_formal"),
    ("ogbn_arxiv_full", "ogbn-arxiv", "ogbn-arxiv", "runs/ogbn_arxiv_full"),
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


def now_utc_iso8601() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def metric_cell(row: dict, key: str) -> str:
    value = row.get(key)
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def empty_field(value: object) -> bool:
    return value is None or value == ""


def contains_manual_continuation(value: object) -> bool:
    return isinstance(value, str) and "manual continuation" in value.lower()


def main_rows() -> dict[str, dict]:
    # Comparison baselines must not depend on main-experiment result tables.
    # Keep the legacy output shape, but leave main columns empty.
    return {}


def load_optional_json(path: Path) -> dict:
    if not path.exists():
        return {}
    data = read_json(path)
    return data if isinstance(data, dict) else {}


def strict_manifest_audit(dataset: str, run_dir: Path) -> dict:
    manifest_path = run_dir / "run_manifest.json"
    errors: list[str] = []
    if not manifest_path.exists():
        return {"dataset": dataset, "status": "failed", "errors": ["missing_run_manifest_json"]}

    manifest = load_optional_json(manifest_path)
    verify = load_optional_json(run_dir / "verify.json")
    binding = load_optional_json(run_dir / "final_logits_binding.json")
    preflight = load_optional_json(run_dir / "native_full_witness_preflight.json")
    prove_profile = load_optional_json(run_dir / "prove_memory_profile.json")

    for name in REQUIRED_RUN_ARTIFACTS:
        if not (run_dir / name).exists():
            errors.append(f"missing_{name}")

    proof_path = run_dir / "proof.json"
    proof_file_size = proof_path.stat().st_size if proof_path.exists() else None
    status = manifest.get("status")
    is_verified = status == "verified"
    is_blocked_or_failed = status in {"blocked", "failed"}

    common_checks = {
        "dataset_matches": manifest.get("dataset") == dataset,
        "route_is_cpu_strict": manifest.get("route_id") == CPU_ROUTE_ID,
        "backend_is_cpu": manifest.get("backend") == "cpu",
        "forbidden_substitute_false": manifest.get("forbidden_substitute_used") is False,
        "precomputed_logits_binding_false": manifest.get("precomputed_logits_binding_used") is False,
        "preflight_ready": preflight.get("status") == "ready",
        "preflight_not_witness_specific": preflight.get("witness_specific_table") is False,
        "run_manifest_from_clean_runner": bool(manifest.get("run_id")) and manifest.get("clean_run") is True,
    }

    manual_sources = [
        manifest.get("stdout_tail"),
        manifest.get("stderr_tail"),
    ]
    for record in manifest.get("stage_records", []) if isinstance(manifest.get("stage_records"), list) else []:
        if isinstance(record, dict):
            manual_sources.extend([record.get("stdout_tail"), record.get("stderr_tail")])
            if record.get("manual_continuation_detected") is True:
                errors.append("stage_record_manual_continuation_detected")
    common_checks["no_manual_continuation"] = not any(contains_manual_continuation(item) for item in manual_sources)

    if is_verified:
        checks = common_checks | {
            "status_verified": True,
            "verify_passed_true": manifest.get("verify_passed") is True
            and verify.get("verify_passed") is True,
            "proof_generated_true": manifest.get("proof_generated") is True
            and prove_profile.get("proof_generated") is True,
            "strict_public_input_equivalent_true": manifest.get("strict_public_input_equivalent") is True
            and verify.get("strict_public_input_equivalent") is True,
            "full_gat_forward_true": manifest.get("full_gat_forward_in_ezkl_proof") is True
            and verify.get("full_gat_forward_in_ezkl_proof") is True,
            "final_logits_complete_true": manifest.get("final_logits_derivation_complete") is True
            and binding.get("final_logits_derivation_complete") is True,
            "no_precomputed_logits_bypass_true": manifest.get("no_precomputed_logits_bypass") is True
            and binding.get("no_precomputed_logits_bypass") is True,
            "block_reason_empty": empty_field(manifest.get("block_reason")),
            "failure_step_empty": empty_field(manifest.get("failure_step")),
            "failed_command_empty": empty_field(manifest.get("failed_command")),
            "check_mode_safe": manifest.get("check_mode") == "safe",
            "proof_size_matches_file": proof_file_size is not None
            and manifest.get("proof_size_bytes") == proof_file_size,
            "stage_records_cover_six_phases": len(manifest.get("stage_records", [])) == 6,
        }
        for key in STRICT_TIMINGS:
            checks[f"{key}_present"] = isinstance(manifest.get(key), (int, float))
        checks["commitment_equals_setup"] = manifest.get("commitment_time_ms") == manifest.get("setup_ms")
        audit_status = "passed" if not errors else "failed"
    elif is_blocked_or_failed:
        checks = common_checks | {
            "status_blocked_or_failed": True,
            "verify_passed_false": manifest.get("verify_passed") is False,
            "proof_generated_false": manifest.get("proof_generated") is False
            and prove_profile.get("proof_generated") is not True,
            "block_reason_present": not empty_field(manifest.get("block_reason")),
            "failure_step_present": not empty_field(manifest.get("failure_step")),
            "failed_command_present": not empty_field(manifest.get("failed_command")),
            "verify_not_invoked": verify.get("verify_invoked") is False or verify.get("verified") is False,
            "failed_stage_timed": isinstance(manifest.get(str(manifest.get("failure_step"))), (int, float)),
            "stage_records_cover_attempted_phases": len(manifest.get("stage_records", [])) >= 1,
        }
        for key in ["gen_settings_ms", "compile_ms", "setup_ms", "gen_witness_ms"]:
            checks[f"{key}_present_before_block"] = isinstance(manifest.get(key), (int, float))
        checks["commitment_equals_setup"] = manifest.get("commitment_time_ms") == manifest.get("setup_ms")
        audit_status = "blocked" if status == "blocked" and not errors else "failed"
    else:
        checks = common_checks | {"status_recognized": False}
        audit_status = "failed"

    for name, passed in checks.items():
        if not passed:
            errors.append(name)

    return {
        "dataset": dataset,
        "status": audit_status if not errors else "failed",
        "manifest_status": status,
        "errors": errors,
        "checks": checks,
        "proof_file_size_bytes": proof_file_size,
        "block_reason": manifest.get("block_reason", ""),
        "failure_step": manifest.get("failure_step", ""),
        "manifest_path": str(manifest_path.relative_to(EZKL_ROOT)),
    }


def strict_audits() -> dict[str, dict]:
    audits = {}
    for run_id, _main_key, _display, run_dir in DATASETS:
        audits[run_id] = strict_manifest_audit(run_id, EZKL_ROOT / run_dir)
    failures = {dataset: audit["errors"] for dataset, audit in audits.items() if audit["status"] == "failed"}
    if failures:
        raise SystemExit(
            "strict_result_collection=FAILED: "
            + "; ".join(f"{dataset}: {', '.join(errors)}" for dataset, errors in failures.items())
        )
    return audits


def collect_cpu(audits: dict[str, dict]) -> tuple[list[dict], list[dict], dict[str, dict]]:
    main_by_dataset = main_rows()
    ezkl_rows: list[dict] = []
    comparison_rows: list[dict] = []
    manifests: dict[str, dict] = {}
    for run_id, main_key, display, run_dir_raw in DATASETS:
        run_dir = EZKL_ROOT / run_dir_raw
        manifest = load_optional_json(run_dir / "run_manifest.json")
        binding = load_optional_json(run_dir / "final_logits_binding.json")
        preflight = load_optional_json(run_dir / "native_full_witness_preflight.json")
        verify = load_optional_json(run_dir / "verify.json")
        manifests[run_id] = manifest

        ezkl_row = {
            "dataset": run_id,
            "display_dataset": display,
            "backend": manifest.get("backend"),
            "statement_route_id": manifest.get("statement_route_id", CPU_ROUTE_ID),
            "source_run_dir": run_dir_raw,
            "route_id": manifest.get("route_id", ""),
            "status": manifest.get("status"),
            "block_reason": manifest.get("block_reason", ""),
            "strict_public_input_equivalent": manifest.get("strict_public_input_equivalent"),
            "full_gat_forward_in_ezkl_proof": manifest.get("full_gat_forward_in_ezkl_proof"),
            "final_logits_derivation_complete": manifest.get("final_logits_derivation_complete"),
            "output_derivation_required": manifest.get(
                "output_derivation_required",
                binding.get("output_derivation_required", False),
            ),
            "output_derivation_complete_candidate": manifest.get(
                "output_derivation_complete_candidate",
                binding.get("output_derivation_complete_candidate", False),
            ),
            "failure_step": manifest.get("failure_step", ""),
            "failure_gate": manifest.get("failure_gate", ""),
            "failure_region": manifest.get("failure_region", ""),
            "failure_row": manifest.get("failure_row", ""),
            "failure_relation": manifest.get("failure_relation", ""),
            "failure_values": manifest.get("failure_values", []),
            "preflight_node_index": preflight.get("node_index"),
            "preflight_degree": preflight.get("degree"),
            "preflight_sum_fp32_reference": preflight.get("sum_fp32_reference"),
            "preflight_inv_fp32_reference": preflight.get("inv_fp32_reference"),
            "preflight_quantized_sum_scale16": preflight.get("quantized_sum_scale16"),
            "preflight_quantized_inv_scale16": preflight.get("quantized_inv_scale16"),
            "preflight_quantized_product": preflight.get("quantized_product"),
            "preflight_inverse_policy": preflight.get("inverse_policy"),
            "preflight_expected_inv_q": preflight.get("expected_inv_q"),
            "preflight_inverse_scale_product": preflight.get("inverse_scale_product"),
            "preflight_inverse_band": preflight.get("inverse_band"),
            "preflight_alpha_rescale_policy": preflight.get("alpha_rescale_policy"),
            "preflight_alpha_rescale_divisor": preflight.get("alpha_rescale_divisor"),
            "preflight_alpha_rounding_rule": preflight.get("alpha_rounding_rule"),
            "preflight_inverse_table_hash": preflight.get("inverse_table_hash"),
            "preflight_inverse_accepted": preflight.get("inverse_all_nodes_accepted"),
            "preflight_alpha_accepted": preflight.get("alpha_all_edges_accepted"),
            "final_logits_binding_exists": bool(binding),
            "final_logits_binding_status": binding.get("binding_status", ""),
            "final_logits_shape": binding.get("final_logits_shape", manifest.get("final_logits_shape")),
            "public_instance_length": binding.get("public_instance_length"),
            "public_instance_order": binding.get("public_instance_order"),
            "public_instance_layout": binding.get(
                "public_instance_layout", manifest.get("public_instance_layout", "")
            ),
            "public_instance_chunk_size": binding.get(
                "public_instance_chunk_size", manifest.get("public_instance_chunk_size")
            ),
            "public_instance_column_count": binding.get(
                "public_instance_column_count", manifest.get("public_instance_column_count")
            ),
            "public_instance_total_length": binding.get(
                "public_instance_total_length", manifest.get("public_instance_total_length")
            ),
            "final_logits_hash": binding.get("final_logits_hash"),
            "public_instance_hash": binding.get("public_instance_hash"),
            "no_precomputed_logits_bypass": binding.get("no_precomputed_logits_bypass"),
            "full_dataset": manifest.get("full_dataset"),
            "command": manifest.get("command", ""),
            "verify_passed": manifest.get("verify_passed"),
            "verify_json_status": verify.get("status"),
            "verify_json_verify_passed": verify.get("verify_passed"),
            "proof_generated": manifest.get("proof_generated"),
            "proof_file_size_bytes": audits[run_id]["proof_file_size_bytes"],
            "proof_size_matches_manifest": audits[run_id]["checks"].get("proof_size_matches_file", False),
            "manifest_strict_check": audits[run_id]["status"],
            "audit_error_count": len(audits[run_id]["errors"]),
            "final_benchmark_included": manifest.get("status") == "verified"
            and audits[run_id]["status"] == "passed",
            "run_id": manifest.get("run_id", ""),
            "run_started_at": manifest.get("run_started_at", ""),
            "run_completed_at": manifest.get("run_completed_at", ""),
        }
        for metric in METRICS:
            ezkl_row[metric] = manifest.get(metric)
        ezkl_rows.append(ezkl_row)

        main_row = main_by_dataset.get(main_key, {})
        comparison = {
            "dataset": display,
            "backend": "cpu",
            "ezkl_status": ezkl_row["status"],
            "ezkl_block_reason": ezkl_row["block_reason"],
            "ezkl_route_id": ezkl_row["route_id"],
            "manifest_strict_check": ezkl_row["manifest_strict_check"],
        }
        for metric in METRICS:
            comparison[f"main_cpu_{metric}"] = main_row.get(metric)
            comparison[f"ezkl_{metric}"] = manifest.get(metric) if ezkl_row["final_benchmark_included"] else None
        comparison_rows.append(comparison)
    return ezkl_rows, comparison_rows, manifests


def write_summary(ezkl_rows: list[dict], comparison_rows: list[dict]) -> None:
    lines: list[str] = []
    lines.append("# GAT-ZKML 与 ezkl CPU 严格全量数据集结果")
    lines.append("")
    verified_count = sum(1 for row in ezkl_rows if row.get("status") == "verified")
    lines.append(f"- 已验证全量数据集行数：{verified_count}/{len(ezkl_rows)}。")
    lines.append(f"- 路线：`{CPU_ROUTE_ID}`。")
    lines.append("- 结果只来自 clean run directory 中完整自动执行链生成的 manifest。")
    lines.append("- 未使用 GPU、output-binding workaround、precomputed logits、identity、digest、folding、subset、shard 或 sample route。")
    lines.append("")
    lines.append("## 数据集状态")
    lines.append("")
    lines.append("| 数据集 | status | route_id | strict | full GAT forward in proof | final logits derivation | manifest strict check |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in ezkl_rows:
        lines.append(
            f"| {row['display_dataset']} | {row['status']} | {row.get('route_id', '')} | "
            f"{str(row.get('strict_public_input_equivalent', False)).lower()} | "
            f"{str(row.get('full_gat_forward_in_ezkl_proof', False)).lower()} | "
            f"{str(row.get('final_logits_derivation_complete', False)).lower()} | "
            f"{row.get('manifest_strict_check', '')} |"
        )
    lines.append("")
    lines.append("## CPU 四项指标")
    lines.append("")
    lines.append("| 数据集 | status | final benchmark | commitment_time_ms | prove_time_ms | verify_time_ms | proof_size_bytes |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: |")
    for row in ezkl_rows:
        lines.append(
            f"| {row['display_dataset']} | {row['status']} | "
            f"{str(row.get('final_benchmark_included', False)).lower()} | "
            f"{metric_cell(row, 'commitment_time_ms')} | "
            f"{metric_cell(row, 'prove_time_ms')} | {metric_cell(row, 'verify_time_ms')} | "
            f"{metric_cell(row, 'proof_size_bytes')} |"
        )
    lines.append("")
    lines.append("## 独立结果占位表")
    lines.append("")
    lines.append(
        "| 数据集 | main commitment_time_ms | ezkl commitment_time_ms | main prove_time_ms | ezkl prove_time_ms | main verify_time_ms | ezkl verify_time_ms | main proof_size_bytes | ezkl proof_size_bytes |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in comparison_rows:
        lines.append(
            "| "
            + row["dataset"]
            + " | "
            + metric_cell(row, "main_cpu_commitment_time_ms")
            + " | "
            + metric_cell(row, "ezkl_commitment_time_ms")
            + " | "
            + metric_cell(row, "main_cpu_prove_time_ms")
            + " | "
            + metric_cell(row, "ezkl_prove_time_ms")
            + " | "
            + metric_cell(row, "main_cpu_verify_time_ms")
            + " | "
            + metric_cell(row, "ezkl_verify_time_ms")
            + " | "
            + metric_cell(row, "main_cpu_proof_size_bytes")
            + " | "
            + metric_cell(row, "ezkl_proof_size_bytes")
            + " |"
        )
    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_native_full_experiment_report(ezkl_rows: list[dict]) -> None:
    lines: list[str] = []
    lines.append("# Native Full Experiment Report")
    lines.append("")
    lines.append("Main-experiment result columns are intentionally blank; this baseline exports only ezkl-local rerun artifacts.")
    lines.append("")
    lines.append("| Dataset | status | strict_public_input_equivalent | full_gat_forward_in_ezkl_proof | final_logits_derivation_complete | manifest_strict_check |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for row in ezkl_rows:
        lines.append(
            f"| {row['display_dataset']} | {row.get('status', '')} | "
            f"{str(row.get('strict_public_input_equivalent', False)).lower()} | "
            f"{str(row.get('full_gat_forward_in_ezkl_proof', False)).lower()} | "
            f"{str(row.get('final_logits_derivation_complete', False)).lower()} | "
            f"{row.get('manifest_strict_check', '')} |"
        )
    lines.append("")
    lines.append("## Final Logits Binding")
    lines.append("")
    lines.append("| Dataset | binding_exists | binding_status | shape | public_instance_length | layout | columns | chunk_size | order | no_precomputed_logits_bypass |")
    lines.append("| --- | --- | --- | --- | ---: | --- | ---: | ---: | --- | --- |")
    for row in ezkl_rows:
        lines.append(
            f"| {row['display_dataset']} | {str(row.get('final_logits_binding_exists', False)).lower()} | "
            f"{row.get('final_logits_binding_status', '')} | {row.get('final_logits_shape', '')} | "
            f"{row.get('public_instance_length', '')} | {row.get('public_instance_layout', '')} | "
            f"{row.get('public_instance_column_count', '')} | {row.get('public_instance_chunk_size', '')} | "
            f"{row.get('public_instance_order', '')} | "
            f"{str(row.get('no_precomputed_logits_bypass', '')).lower()} |"
        )
    lines.append("")
    lines.append("No GPU baseline, output-binding workaround, precomputed logits, subset, shard, or sample route was used.")
    (RESULTS / "native_full_experiment_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_control_variable_audit(ezkl_rows: list[dict], manifests: dict[str, dict]) -> None:
    lines = [
        "# Control Variable Audit",
        "",
        f"- Route: `{CPU_ROUTE_ID}`",
        "- Backend: `cpu` for every final row.",
        "- Verified benchmark rows require strict public input equivalence, complete full GAT forward proof, and no precomputed logits bypass.",
        "- The collector records real blocked resource runs as audit rows, but excludes them from benchmark metrics and returns a non-zero BLOCKED status.",
        "",
        "| Dataset | status | backend | route_id | clean_run | check_mode | proof_generated | verify_passed |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in ezkl_rows:
        manifest = manifests[row["dataset"]]
        lines.append(
            f"| {row['dataset']} | {row['status']} | {row['backend']} | {row['route_id']} | "
            f"{str(manifest.get('clean_run')).lower()} | {manifest.get('check_mode')} | "
            f"{str(row.get('proof_generated')).lower()} | {str(row.get('verify_passed')).lower()} |"
        )
    lines.append("")
    lines.append("Only `manifest_strict_check=passed` rows are included as final benchmark rows; blocked rows are retained only as audit evidence.")
    (RESULTS / "control_variable_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def size_by_name(manifest: dict, name: str) -> object:
    profile = manifest.get("prove_memory_profile")
    if not isinstance(profile, dict):
        profile = {}
    for payload in profile.get("largest_payloads_loaded", []):
        if isinstance(payload, dict) and payload.get("name") == name:
            return payload.get("size_bytes")
    return ""


def ppi_ogbn_row(dataset: str, manifest: dict) -> dict:
    model_info_path = RUNS / dataset / "model_info.json"
    model_info = load_optional_json(model_info_path)
    profile = manifest.get("prove_memory_profile") if isinstance(manifest.get("prove_memory_profile"), dict) else {}
    estimate = manifest.get("compressed_row_estimate") if isinstance(manifest.get("compressed_row_estimate"), dict) else {}
    final_shape = manifest.get("final_logits_shape") or []
    return {
        "dataset": dataset,
        "status": manifest.get("status"),
        "N": manifest.get("N"),
        "E": manifest.get("E"),
        "G_batch": manifest.get("G_batch"),
        "num_classes": model_info.get("num_classes") or (final_shape[1] if len(final_shape) == 2 else ""),
        "final_logits_shape": final_shape,
        "public_instance_total_length": manifest.get("public_instance_total_length"),
        "proof_size_bytes": manifest.get("proof_size_bytes"),
        "commitment_time_ms": manifest.get("commitment_time_ms"),
        "prove_time_ms": manifest.get("prove_time_ms"),
        "verify_time_ms": manifest.get("verify_time_ms"),
        "setup_ms": manifest.get("setup_ms"),
        "gen_witness_ms": manifest.get("gen_witness_ms"),
        "compressed_row_estimate.total_rows_estimated": estimate.get("total_rows_estimated"),
        "assigned_row_upper_bound": profile.get("assigned_row_upper_bound"),
        "logrows": profile.get("logrows"),
        "circuit_rows": profile.get("circuit_rows"),
        "pk.key size": profile.get("pk_size") or size_by_name(manifest, "pk.key"),
        "formal_tables.bin size": size_by_name(manifest, "formal_tables.bin"),
        "node_features.f64.bin size": size_by_name(manifest, "node_features.f64.bin"),
        "csr_segment_layout.bin size": size_by_name(manifest, "csr_segment_layout.bin"),
        "peak_rss_bytes": profile.get("peak_rss_bytes"),
        "final_judgment": "verified benchmark row" if manifest.get("status") == "verified" else "resource-blocked audit row",
    }


def write_ppi_ogbn_audit(manifests: dict[str, dict]) -> None:
    rows = [
        ppi_ogbn_row("ppi_full_formal", manifests["ppi_full_formal"]),
        ppi_ogbn_row("ogbn_arxiv_full", manifests["ogbn_arxiv_full"]),
    ]
    headers = list(rows[0])
    lines = [
        "# PPI vs ogbn-arxiv Strict Audit",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    ppi = rows[0]
    ogbn = rows[1]
    ppi_status = manifests["ppi_full_formal"].get("status")
    ogbn_status = manifests["ogbn_arxiv_full"].get("status")
    ppi_verified = ppi_status == "verified"
    ogbn_verified = ogbn_status == "verified"
    prove_delta = ""
    if isinstance(ppi.get("prove_time_ms"), (int, float)) and isinstance(ogbn.get("prove_time_ms"), (int, float)):
        prove_delta = f"{abs(float(ppi['prove_time_ms']) - float(ogbn['prove_time_ms'])):.3f} ms"
    ppi_outcome = (
        "- PPI is a verified benchmark row: `proof.json`, `verify.json`, final logits binding, formal tables, run manifest, and memory profiles all pass strict audit."
        if ppi_verified
        else "- PPI remains a resource-blocked audit row and is excluded from verified benchmark metrics."
    )
    ogbn_outcome = (
        "- ogbn-arxiv is a verified benchmark row after this clean rerun and remains the large-graph regression/control case."
        if ogbn_verified
        else "- ogbn-arxiv is not verified in this run; final collection must remain non-OK."
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- PPI status is `{ppi_status}` and ogbn-arxiv status is `{ogbn_status}` in this clean run.",
            f"- PPI/ogbn-arxiv prove-time delta: {prove_delta or 'not available because one side lacks a numeric prove time'}.",
            ppi_outcome,
            ogbn_outcome,
            "- The two large attempts use the same CPU strict route, same Halo2/KZG backend family, and large chunked public-instance columns.",
            "- `public_instance_total_length`, proof size, PK/SRS scale, fixed logrows/circuit row class, and class-count differences are the primary controls to inspect before treating close proving times as suspicious.",
            "- `compressed_row_estimate.total_rows_estimated` is a static planner estimate. `logrows` and `circuit_rows` come from the instantiated Halo2 circuit profile. They need not be equal because the compressed native backend uses a separate diagnostic/full-model layout bound by the selected logrows.",
            "- No manual continuation, forbidden fallback, or precomputed-logits bypass is present. Verified rows must have empty failed-command/block fields and proof sizes matching the real proof files.",
        ]
    )
    (RESULTS / "ppi_ogbn_arxiv_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_strict_audit_report(audits: dict[str, dict]) -> None:
    lines = [
        "# Strict Reproducibility Audit",
        "",
        f"- Generated at: `{now_utc_iso8601()}`",
        f"- Required route: `{CPU_ROUTE_ID}`",
        "- The collector writes an audit report for blocked resource runs, but returns a non-zero BLOCKED status unless every dataset is a verified benchmark row.",
        "",
        "| Dataset | status | proof_file_size_bytes | block_reason | failure_step | failed_checks |",
        "| --- | --- | ---: | --- | --- | --- |",
    ]
    for dataset, audit in audits.items():
        lines.append(
            f"| {dataset} | {audit['status']} | {audit.get('proof_file_size_bytes', '')} | "
            f"{audit.get('block_reason', '')} | {audit.get('failure_step', '')} | "
            f"{', '.join(audit.get('errors', []))} |"
        )
    lines.extend(
        [
            "",
            "## Enforced Rules",
            "",
            "- Verified benchmark rows require `status=verified`, `verify_passed=true`, and `proof_generated=true`.",
            "- Verified benchmark rows require `proof_size_bytes` to equal the real `proof.json` file size.",
            "- Verified benchmark rows require `gen_settings_ms`, `compile_ms`, `setup_ms`, `gen_witness_ms`, `prove_time_ms`, and `verify_time_ms` to be numeric.",
            "- `failed_command`, `failure_step`, and `block_reason` must be empty for verified rows.",
            "- Blocked rows must keep `verify_passed=false`, `proof_generated=false`, non-empty `block_reason`, non-empty `failure_step`, and the real failed command.",
            "- `stdout_tail`, `stderr_tail`, and stage tails must not contain `manual continuation`.",
            "- `proof.json`, `verify.json`, `run_manifest.json`, model metadata, formal table metadata, preflight, and memory profiles must exist.",
        ]
    )
    (RESULTS / "strict_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_cleanup_report() -> None:
    lines = [
        "# Cleanup Report",
        "",
        "- Final run directories are regenerated with `--clean --run` for every dataset.",
        "- The collector accepts only final strict CPU run directories under `comparison/baselines/ezkl/runs/<dataset>`.",
        "- Debug/scratch/manual/old/bak/tmp/orig artifacts are not part of the final result set.",
        "- Large binary/proof/key payloads remain reproducible run artifacts and are ignored by `.gitignore`; JSON manifests and audit files record their measured sizes and hashes where available.",
    ]
    (RESULTS / "cleanup_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readme(ezkl_rows: list[dict]) -> None:
    lines: list[str] = []
    lines.append("# GAT-ZKML 的 ezkl CPU 严格全量数据集对比实验")
    lines.append("")
    lines.append("本目录是主仓库 GAT-ZKML 的 ezkl CPU-only strict baseline。主仓库提供正式语义真源；本目录只改变证明系统承载方式，以及 ezkl/Halo2 对稀疏图、公开实例列和查找表的必要适配。")
    lines.append("")
    lines.append("## 正式路线")
    lines.append("")
    lines.append(f"唯一最终路线：`{CPU_ROUTE_ID}`。该路线证明 full GAT forward。最终 logits 在电路内完整推导并公开绑定，`strict_public_input_equivalent=true` 只在真实 prove 与 verify 完成后成立。")
    lines.append("")
    lines.append("不存在 GPU baseline；本目录最终结果全部为 `backend=cpu`。")
    lines.append("")
    lines.append("## 全量数据集状态")
    lines.append("")
    lines.append("| 数据集 | status | final benchmark | strict_public_input_equivalent | full_gat_forward_in_ezkl_proof | final_logits_derivation_complete | no_precomputed_logits_bypass | manifest_strict_check |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in ezkl_rows:
        lines.append(
            f"| {row['display_dataset']} | {row['status']} | "
            f"{str(row.get('final_benchmark_included', False)).lower()} | "
            f"{str(row['strict_public_input_equivalent']).lower()} | "
            f"{str(row['full_gat_forward_in_ezkl_proof']).lower()} | "
            f"{str(row['final_logits_derivation_complete']).lower()} | "
            f"{str(row['no_precomputed_logits_bypass']).lower()} | "
            f"{row['manifest_strict_check']} |"
        )
    lines.append("")
    lines.append("## CPU 四项指标")
    lines.append("")
    lines.append("| 数据集 | status | final benchmark | commitment_time_ms | prove_time_ms | verify_time_ms | proof_size_bytes |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: |")
    for row in ezkl_rows:
        lines.append(
            f"| {row['display_dataset']} | {row['status']} | "
            f"{str(row.get('final_benchmark_included', False)).lower()} | "
            f"{metric_cell(row, 'commitment_time_ms')} | "
            f"{metric_cell(row, 'prove_time_ms')} | {metric_cell(row, 'verify_time_ms')} | "
            f"{metric_cell(row, 'proof_size_bytes')} |"
        )
    lines.append("")
    lines.append("## 结果可信性检查")
    lines.append("")
    lines.append("最终表格由 `scripts/collect_gat_ezkl_results.py` 从 run manifest 自动生成。collector 和 validator 会拒绝以下情况：")
    lines.append("")
    lines.append("- `status=verified` 但 `verify_passed` 或 `proof_generated` 不为 `true`。")
    lines.append("- `status=verified` 但 `failed_command`、`failure_step` 或 `block_reason` 非空。")
    lines.append("- `stdout_tail`、`stderr_tail` 或阶段日志包含 `manual continuation`。")
    lines.append("- `gen_settings_ms`、`compile_ms`、`setup_ms`、`gen_witness_ms`、`prove_time_ms` 或 `verify_time_ms` 缺失。")
    lines.append("- `proof_size_bytes` 与真实 `proof.json` 文件大小不一致。")
    lines.append("- 缺少 `proof.json`、`verify.json`、`run_manifest.json`、模型/图/绑定/形式化表/预检查/内存 profile 文件。")
    lines.append("- route 不是 CPU strict route，或出现 forbidden fallback / precomputed logits bypass。")
    lines.append("")
    lines.append("## 结果文件")
    lines.append("")
    for name in [
        "results/gat_ezkl_full_results.json",
        "results/gat_ezkl_full_results.csv",
        "results/main_vs_ezkl_full_comparison.json",
        "results/main_vs_ezkl_full_comparison.csv",
        "results/summary.md",
        "results/strict_audit_report.md",
        "results/control_variable_audit.md",
        "results/native_full_experiment_report.md",
        "results/ppi_ogbn_arxiv_audit.md",
        "results/cleanup_report.md",
    ]:
        lines.append(f"- `{name}`")
    lines.append("")
    lines.append("## 构建、运行与验证")
    lines.append("")
    lines.append("```bash")
    lines.append("cargo test native_gat --no-default-features --features ezkl,no-banner")
    lines.append("cargo build --release --bin ezkl --no-default-features --features ezkl,no-banner")
    lines.append("cmake -S cpp/gat_ezkl_adapter -B build_gat_adapter -DCMAKE_BUILD_TYPE=Release")
    lines.append("cmake --build build_gat_adapter -j")
    lines.append("python3 -m py_compile scripts/run_gat_ezkl_baseline.py scripts/collect_gat_ezkl_results.py scripts/validate_final_results.py")
    lines.append("python3 scripts/collect_gat_ezkl_results.py")
    lines.append("python3 scripts/validate_final_results.py")
    lines.append("```")
    lines.append("")
    lines.append("五个数据集的 clean rerun 命令见 `scripts/run_gat_ezkl_baseline.py --help` 和本目录正式审计报告。")
    (EZKL_ROOT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    audits = strict_audits()
    cpu_rows, cpu_comparison_rows, manifests = collect_cpu(audits)

    result_fields = [
        "dataset",
        "display_dataset",
        "backend",
        "statement_route_id",
        "source_run_dir",
        "route_id",
        "status",
        "block_reason",
        "strict_public_input_equivalent",
        "full_gat_forward_in_ezkl_proof",
        "final_logits_derivation_complete",
        "output_derivation_required",
        "output_derivation_complete_candidate",
        "failure_step",
        "failure_gate",
        "failure_region",
        "failure_row",
        "failure_relation",
        "failure_values",
        "preflight_node_index",
        "preflight_degree",
        "preflight_sum_fp32_reference",
        "preflight_inv_fp32_reference",
        "preflight_quantized_sum_scale16",
        "preflight_quantized_inv_scale16",
        "preflight_quantized_product",
        "preflight_inverse_policy",
        "preflight_expected_inv_q",
        "preflight_inverse_scale_product",
        "preflight_inverse_band",
        "preflight_alpha_rescale_policy",
        "preflight_alpha_rescale_divisor",
        "preflight_alpha_rounding_rule",
        "preflight_inverse_table_hash",
        "preflight_inverse_accepted",
        "preflight_alpha_accepted",
        "final_logits_binding_exists",
        "final_logits_binding_status",
        "final_logits_shape",
        "public_instance_length",
        "public_instance_order",
        "public_instance_layout",
        "public_instance_chunk_size",
        "public_instance_column_count",
        "public_instance_total_length",
        "final_logits_hash",
        "public_instance_hash",
        "no_precomputed_logits_bypass",
        "full_dataset",
        "command",
        "verify_passed",
        "verify_json_status",
        "verify_json_verify_passed",
        "proof_generated",
        "proof_file_size_bytes",
        "proof_size_matches_manifest",
        "manifest_strict_check",
        "audit_error_count",
        "final_benchmark_included",
        "run_id",
        "run_started_at",
        "run_completed_at",
    ] + METRICS
    comparison_fields = [
        "dataset",
        "backend",
        "ezkl_status",
        "ezkl_block_reason",
        "ezkl_route_id",
        "manifest_strict_check",
    ]
    for metric in METRICS:
        comparison_fields.extend([f"main_cpu_{metric}", f"ezkl_{metric}"])

    verified_rows = [row for row in cpu_rows if row.get("final_benchmark_included")]
    blocked_rows = [row for row in cpu_rows if row.get("status") in {"blocked", "failed"}]
    write_json(
        RESULTS / "gat_ezkl_full_results.json",
        {
            "rows": cpu_rows,
            "verified_benchmark_rows": verified_rows,
            "blocked_or_failed_rows": blocked_rows,
            "audits": audits,
        },
    )
    write_json(RESULTS / "main_vs_ezkl_full_comparison.json", {"rows": cpu_comparison_rows})
    write_csv(RESULTS / "gat_ezkl_full_results.csv", cpu_rows, result_fields)
    write_csv(RESULTS / "main_vs_ezkl_full_comparison.csv", cpu_comparison_rows, comparison_fields)
    write_summary(cpu_rows, cpu_comparison_rows)
    write_strict_audit_report(audits)
    write_control_variable_audit(cpu_rows, manifests)
    write_native_full_experiment_report(cpu_rows)
    write_ppi_ogbn_audit(manifests)
    write_cleanup_report()
    write_readme(cpu_rows)
    verified_count = len(verified_rows)
    if verified_count == len(DATASETS):
        print(f"strict_result_collection=OK cpu_verified={verified_count}/{len(DATASETS)} results_dir={RESULTS}")
        return 0
    blocked = ",".join(row["dataset"] for row in blocked_rows) or "none"
    print(
        f"strict_result_collection=BLOCKED cpu_verified={verified_count}/{len(DATASETS)} "
        f"blocked={blocked} results_dir={RESULTS}"
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
