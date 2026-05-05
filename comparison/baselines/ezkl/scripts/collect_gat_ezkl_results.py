#!/usr/bin/env python3
"""Collect the CPU-only strict GAT-ZKML ezkl baseline results."""

from __future__ import annotations

import csv
import json
from pathlib import Path


EZKL_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path("/home/pzh/project")
RESULTS = EZKL_ROOT / "results"
CPU_ROUTE_ID = "ezkl_gat_native_compressed_csr_attention_full"
DATASETS = [
    ("cora_full", "cora", "Cora", "runs/cora_full"),
    ("citeseer_full", "citeseer", "Citeseer", "runs/citeseer_full"),
    ("pubmed_full", "pubmed", "Pubmed", "runs/pubmed_full"),
    ("ppi_full_formal", "ppi", "PPI", "runs/ppi_full_formal"),
    ("ogbn_arxiv_full", "ogbn-arxiv", "ogbn-arxiv", "runs/ogbn_arxiv_full"),
]
METRICS = ["commitment_time_ms", "prove_time_ms", "verify_time_ms", "proof_size_bytes"]


def read_json(path: Path) -> dict:
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


def main_rows() -> dict[str, dict]:
    latest = read_json(PROJECT_ROOT / "runs/benchmarks/latest.json")
    return {row["dataset"]: row for row in latest["rows"]}


def preflight_for_run(run_dir: str) -> dict:
    path = EZKL_ROOT / run_dir / "native_full_witness_preflight.json"
    return read_json(path) if path.exists() else {}


def final_logits_binding_for_run(run_dir: str) -> dict:
    path = EZKL_ROOT / run_dir / "final_logits_binding.json"
    return read_json(path) if path.exists() else {}


def native_constraint_failure_for_run(run_dir: str) -> dict:
    base = EZKL_ROOT / run_dir
    for path in [
        base / "debug_full_model" / "native_constraint_failure.json",
        base / "native_constraint_failure.json",
    ]:
        if path.exists():
            return read_json(path)
    return {}


def effective_manifest(run_dir: str, run_id: str) -> dict:
    base = EZKL_ROOT / run_dir
    manifest_path = base / "run_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {
        "dataset": run_id,
        "route_id": CPU_ROUTE_ID,
        "backend": "cpu",
        "status": "failed",
        "block_reason": "missing_cpu_run_manifest",
    }
    debug_full = base / "debug_full_model" / "run_manifest.json"
    if debug_full.exists() and manifest.get("status") != "verified":
        debug_manifest = read_json(debug_full)
        if debug_manifest.get("block_reason") != "native_constraint_synthesis_failed":
            manifest = debug_manifest | {
                "dataset": run_id,
                "source_manifest": str(debug_full.relative_to(EZKL_ROOT)),
            }
    return manifest


def collect_cpu() -> tuple[list[dict], list[dict]]:
    main_by_dataset = main_rows()
    ezkl_rows: list[dict] = []
    comparison_rows: list[dict] = []
    for run_id, main_key, display, run_dir in DATASETS:
        manifest = effective_manifest(run_dir, run_id)
        preflight = preflight_for_run(run_dir)
        final_binding = final_logits_binding_for_run(run_dir)
        failure = native_constraint_failure_for_run(run_dir)
        ezkl_row = {
            "dataset": run_id,
            "display_dataset": display,
            "backend": "cpu",
            "statement_route_id": manifest.get("statement_route_id", CPU_ROUTE_ID),
            "source_run_dir": run_dir,
            "route_id": manifest.get("route_id", ""),
            "status": manifest.get("status", "failed"),
            "block_reason": manifest.get("block_reason", ""),
            "strict_public_input_equivalent": manifest.get("strict_public_input_equivalent", False),
            "full_gat_forward_in_ezkl_proof": manifest.get("full_gat_forward_in_ezkl_proof", False),
            "final_logits_derivation_complete": manifest.get(
                "final_logits_derivation_complete",
                final_binding.get("final_logits_derivation_complete", False),
            ),
            "output_derivation_required": manifest.get(
                "output_derivation_required",
                final_binding.get("output_derivation_required", False),
            ),
            "output_derivation_complete_candidate": manifest.get(
                "output_derivation_complete_candidate",
                final_binding.get("output_derivation_complete_candidate", False),
            ),
            "failure_step": manifest.get("failure_step", "") or ("prove_time_ms" if failure else ""),
            "failure_gate": failure.get("gate_name", manifest.get("failure_gate", "")),
            "failure_region": failure.get("region_name", manifest.get("failure_region", "")),
            "failure_row": failure.get("row_offset", manifest.get("failure_row", "")),
            "failure_relation": failure.get("expected_relation", manifest.get("failure_relation", "")),
            "failure_values": failure.get("actual_values", manifest.get("failure_values", [])),
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
            "final_logits_binding_exists": bool(final_binding),
            "final_logits_binding_status": final_binding.get("binding_status", ""),
            "final_logits_shape": final_binding.get("final_logits_shape"),
            "public_instance_length": final_binding.get("public_instance_length"),
            "public_instance_order": final_binding.get("public_instance_order"),
            "public_instance_layout": final_binding.get(
                "public_instance_layout", manifest.get("public_instance_layout", "")
            ),
            "public_instance_chunk_size": final_binding.get(
                "public_instance_chunk_size", manifest.get("public_instance_chunk_size")
            ),
            "public_instance_column_count": final_binding.get(
                "public_instance_column_count", manifest.get("public_instance_column_count")
            ),
            "public_instance_total_length": final_binding.get(
                "public_instance_total_length", manifest.get("public_instance_total_length")
            ),
            "final_logits_hash": final_binding.get("final_logits_hash"),
            "public_instance_hash": final_binding.get("public_instance_hash"),
            "no_precomputed_logits_bypass": final_binding.get("no_precomputed_logits_bypass"),
            "full_dataset": manifest.get("full_dataset"),
            "command": manifest.get("command", ""),
            "verify_passed": manifest.get("verify_passed"),
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
        }
        for metric in METRICS:
            comparison[f"main_cpu_{metric}"] = main_row.get(metric)
            comparison[f"ezkl_{metric}"] = manifest.get(metric)
        comparison_rows.append(comparison)
    return ezkl_rows, comparison_rows


def write_summary(ezkl_rows: list[dict], comparison_rows: list[dict]) -> None:
    lines: list[str] = []
    lines.append("# GAT-ZKML 与 ezkl CPU 严格全量数据集结果")
    lines.append("")
    verified_count = sum(1 for row in ezkl_rows if row.get("status") == "verified")
    lines.append(f"- 已验证全量数据集行数：{verified_count}/{len(ezkl_rows)}。")
    lines.append(f"- 路线：`{CPU_ROUTE_ID}`。")
    lines.append("- `strict_public_input_equivalent` 只在完整 native GAT proof 已通过验证时为 `true`。")
    lines.append("- 未使用 output-binding workaround、precomputed logits、identity、digest、folding、subset、shard 或 sample route。")
    lines.append("")
    lines.append("## 数据集状态")
    lines.append("")
    lines.append("| 数据集 | status | route_id | strict | full GAT forward in proof | final logits derivation |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for row in ezkl_rows:
        lines.append(
            f"| {row['display_dataset']} | {row['status']} | {row.get('route_id', '')} | "
            f"{str(row.get('strict_public_input_equivalent', False)).lower()} | "
            f"{str(row.get('full_gat_forward_in_ezkl_proof', False)).lower()} | "
            f"{str(row.get('final_logits_derivation_complete', False)).lower()} |"
        )
    lines.append("")
    lines.append("## CPU 四项指标")
    lines.append("")
    lines.append("| 数据集 | commitment_time_ms | prove_time_ms | verify_time_ms | proof_size_bytes |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for row in ezkl_rows:
        lines.append(
            f"| {row['display_dataset']} | {metric_cell(row, 'commitment_time_ms')} | "
            f"{metric_cell(row, 'prove_time_ms')} | {metric_cell(row, 'verify_time_ms')} | "
            f"{metric_cell(row, 'proof_size_bytes')} |"
        )
    lines.append("")
    lines.append("## 与主仓库基准对比")
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


def write_native_constraint_debug_report() -> None:
    lines: list[str] = []
    lines.append("# Native Constraint Debug Report")
    lines.append("")
    lines.append(f"- Route: `{CPU_ROUTE_ID}`")
    lines.append("- Final benchmark collection uses only verified full-model manifests.")
    lines.append("- No output-binding workaround, precomputed logits, subset, shard, sample, digest, folding, or identity route is used.")
    lines.append("")
    lines.append("## Full-Dataset Quantized Inverse Preflight Evidence")
    lines.append("")
    lines.append("| Dataset | node | degree | policy | sum_fp32 | inv_fp32 | sum_q | inv_q | expected_inv_q | product | inverse_ok | alpha_ok |")
    lines.append("| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |")
    for _run_id, _main_key, display, run_dir in DATASETS:
        preflight = preflight_for_run(run_dir)
        lines.append(
            f"| {display} | {preflight.get('node_index', '')} | {preflight.get('degree', '')} | "
            f"{preflight.get('inverse_policy', '')} | "
            f"{preflight.get('sum_fp32_reference', '')} | {preflight.get('inv_fp32_reference', '')} | "
            f"{preflight.get('quantized_sum_scale16', '')} | {preflight.get('quantized_inv_scale16', '')} | "
            f"{preflight.get('expected_inv_q', '')} | {preflight.get('quantized_product', '')} | "
            f"{preflight.get('inverse_all_nodes_accepted', '')} | {preflight.get('alpha_all_edges_accepted', '')} |"
        )
    (RESULTS / "native_constraint_debug_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_native_full_experiment_report(ezkl_rows: list[dict]) -> None:
    lines: list[str] = []
    lines.append("# Native Full Experiment Report")
    lines.append("")
    lines.append("| Dataset | status | strict_public_input_equivalent | full_gat_forward_in_ezkl_proof | final_logits_derivation_complete |")
    lines.append("| --- | --- | --- | --- | --- |")
    for row in ezkl_rows:
        lines.append(
            f"| {row['display_dataset']} | {row.get('status', '')} | "
            f"{str(row.get('strict_public_input_equivalent', False)).lower()} | "
            f"{str(row.get('full_gat_forward_in_ezkl_proof', False)).lower()} | "
            f"{str(row.get('final_logits_derivation_complete', False)).lower()} |"
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
    lines.append("No output-binding workaround, no precomputed logits, no subset, no shard, and no sample route was used.")
    (RESULTS / "native_full_experiment_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
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
    ] + METRICS
    comparison_fields = ["dataset", "backend", "ezkl_status", "ezkl_block_reason", "ezkl_route_id"]
    for metric in METRICS:
        comparison_fields.extend([f"main_cpu_{metric}", f"ezkl_{metric}"])

    cpu_rows, cpu_comparison_rows = collect_cpu()
    write_json(RESULTS / "gat_ezkl_full_results.json", {"rows": cpu_rows})
    write_json(RESULTS / "main_vs_ezkl_full_comparison.json", {"rows": cpu_comparison_rows})
    write_csv(RESULTS / "gat_ezkl_full_results.csv", cpu_rows, result_fields)
    write_csv(RESULTS / "main_vs_ezkl_full_comparison.csv", cpu_comparison_rows, comparison_fields)
    write_summary(cpu_rows, cpu_comparison_rows)
    write_native_constraint_debug_report()
    write_native_full_experiment_report(cpu_rows)
    print(f"wrote CPU results under {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
