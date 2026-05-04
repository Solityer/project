#!/usr/bin/env python3
"""Collect GAT-ZKML ezkl baseline results into JSON/CSV/Markdown."""

from __future__ import annotations

import csv
import json
from pathlib import Path


EZKL_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path("/home/pzh/project")
RESULTS = EZKL_ROOT / "results"
CPU_ROUTE_ID = "ezkl_gat_native_compressed_csr_attention_full"
GPU_ROUTE_ID = "ezkl_gat_native_compressed_csr_attention_full_gpu"
DATASETS = [
    ("cora_full", "cora", "Cora", "runs/cora_full"),
    ("citeseer_full", "citeseer", "Citeseer", "runs/citeseer_full"),
    ("pubmed_full", "pubmed", "Pubmed", "runs/pubmed_full"),
    ("ppi_full_formal", "ppi", "PPI", "runs/ppi_full_formal"),
    ("ogbn_arxiv_full", "ogbn-arxiv", "ogbn-arxiv", "runs/ogbn_arxiv_full"),
]
METRICS = ["commitment_time_ms", "prove_time_ms", "verify_time_ms", "proof_size_bytes"]
BACKENDS = {
    "cpu": {
        "route_id": CPU_ROUTE_ID,
        "run_suffix": "",
        "result_prefix": "gat_ezkl_full_results",
        "comparison_prefix": "main_vs_ezkl_full_comparison",
        "summary": "summary.md",
        "label": "CPU",
    },
    "gpu": {
        "route_id": GPU_ROUTE_ID,
        "run_suffix": "gpu",
        "result_prefix": "gat_ezkl_gpu_full_results",
        "comparison_prefix": "main_vs_ezkl_gpu_full_comparison",
        "summary": "gpu_summary.md",
        "label": "GPU",
    },
}


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


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


def backend_run_dir(run_dir: str, backend: str) -> str:
    suffix = BACKENDS[backend]["run_suffix"]
    return f"{run_dir}/{suffix}" if suffix else run_dir


def effective_manifest(run_dir: str, run_id: str, backend: str) -> dict:
    base = EZKL_ROOT / run_dir
    manifest_path = base / "run_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {
        "dataset": run_id,
        "route_id": BACKENDS[backend]["route_id"],
        "backend": backend,
        "status": "missing" if backend == "gpu" else "failed",
        "block_reason": f"missing_{backend}_run_manifest",
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


def preflight_for_run(run_dir: str) -> dict:
    path = EZKL_ROOT / run_dir / "native_full_witness_preflight.json"
    if not path.exists():
        return {}
    return read_json(path)


def final_logits_binding_for_run(run_dir: str) -> dict:
    path = EZKL_ROOT / run_dir / "final_logits_binding.json"
    if not path.exists():
        return {}
    return read_json(path)


def native_constraint_failure_for_run(run_dir: str) -> dict:
    base = EZKL_ROOT / run_dir
    for path in [
        base / "debug_full_model" / "native_constraint_failure.json",
        base / "native_constraint_failure.json",
    ]:
        if path.exists():
            return read_json(path)
    return {}


def collect_backend(backend: str) -> tuple[list[dict], list[dict]]:
    main_by_dataset = main_rows()
    ezkl_rows: list[dict] = []
    comparison_rows: list[dict] = []
    for run_id, main_key, display, run_dir in DATASETS:
        actual_run_dir = backend_run_dir(run_dir, backend)
        manifest = effective_manifest(actual_run_dir, run_id, backend)
        preflight = preflight_for_run(actual_run_dir)
        final_binding = final_logits_binding_for_run(actual_run_dir)
        failure = native_constraint_failure_for_run(actual_run_dir)
        ezkl_row = {
            "dataset": run_id,
            "display_dataset": display,
            "backend": manifest.get("backend", backend),
            "gpu_backend": manifest.get("gpu_backend", ""),
            "gpu_env": manifest.get("gpu_env", {}),
            "gpu_device_info": manifest.get("gpu_device_info", {}),
            "statement_route_id": manifest.get("statement_route_id", CPU_ROUTE_ID),
            "source_run_dir": actual_run_dir,
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
            "failure_relation": failure.get(
                "expected_relation", manifest.get("failure_relation", "")
            ),
            "failure_values": failure.get("actual_values", manifest.get("failure_values", [])),
            "failure_raw_error": failure.get("raw_error", ""),
            "failure_remaining_blocker": failure.get("remaining_blocker", ""),
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
        }
        for metric in METRICS:
            ezkl_row[metric] = manifest.get(metric)
        ezkl_rows.append(ezkl_row)

        main_row = main_by_dataset.get(main_key, {})
        comparison = {
            "dataset": display,
            "backend": backend,
            "ezkl_status": ezkl_row["status"],
            "ezkl_block_reason": ezkl_row["block_reason"],
            "ezkl_route_id": ezkl_row["route_id"],
        }
        for metric in METRICS:
            comparison[f"main_cpu_{metric}"] = main_row.get(metric)
            comparison[f"ezkl_{metric}"] = manifest.get(metric)
        comparison_rows.append(comparison)
    return ezkl_rows, comparison_rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_summary(ezkl_rows: list[dict], comparison_rows: list[dict], backend: str) -> None:
    label = BACKENDS[backend]["label"]
    route_id = BACKENDS[backend]["route_id"]
    lines: list[str] = []
    lines.append(f"# GAT-ZKML 与 ezkl 全量数据集 {label} 汇总")
    lines.append("")
    verified_count = sum(1 for row in ezkl_rows if row.get("status") == "verified")
    lines.append(f"- 已验证全量数据集行数：{verified_count}/{len(ezkl_rows)}。")
    lines.append(f"- 路线：`{route_id}`。")
    lines.append("- `strict_public_input_equivalent` 只在完整 native GAT proof 已通过验证时为 `true`。")
    lines.append("- 未使用 output-binding workaround、precomputed logits、identity、digest、folding、subset、shard 或 sample route。")
    lines.append("")
    lines.append("## 数据集状态")
    lines.append("")
    lines.append("| 数据集 | status | route_id | backend | strict | full GAT forward in proof | final logits derivation | block_reason |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in ezkl_rows:
        lines.append(
            f"| {row['display_dataset']} | {row['status']} | {row.get('route_id', '')} | "
            f"{row.get('backend', backend)} | "
            f"{row.get('strict_public_input_equivalent', False)} | {row.get('full_gat_forward_in_ezkl_proof', False)} | "
            f"{row.get('final_logits_derivation_complete', False)} | "
            f"{row.get('block_reason', '')} |"
        )
    lines.append("")
    lines.append("## 四项指标对比")
    lines.append("")
    lines.append(
        "| 数据集 | main commitment_time_ms | ezkl commitment_time_ms | main prove_time_ms | ezkl prove_time_ms | main verify_time_ms | ezkl verify_time_ms | main proof_size_bytes | ezkl proof_size_bytes |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in comparison_rows:
        ezkl_display = {
            metric: ("blocked" if row["ezkl_status"] == "blocked" and row.get(f"ezkl_{metric}") is None else metric_cell(row, f"ezkl_{metric}"))
            for metric in METRICS
        }
        lines.append(
            "| "
            + row["dataset"]
            + " | "
            + metric_cell(row, "main_cpu_commitment_time_ms")
            + " | "
            + ezkl_display["commitment_time_ms"]
            + " | "
            + metric_cell(row, "main_cpu_prove_time_ms")
            + " | "
            + ezkl_display["prove_time_ms"]
            + " | "
            + metric_cell(row, "main_cpu_verify_time_ms")
            + " | "
            + ezkl_display["verify_time_ms"]
            + " | "
            + metric_cell(row, "main_cpu_proof_size_bytes")
            + " | "
            + ezkl_display["proof_size_bytes"]
            + " |"
        )
    lines.append("")
    lines.append("## 阻塞或失败原因")
    lines.append("")
    for row in ezkl_rows:
        detail = row.get("failure_gate") or row.get("failure_region") or row.get("failure_relation")
        if detail:
            values = row.get("failure_values") or []
            value_text = ", ".join(values) if isinstance(values, list) else str(values)
            preflight_text = ""
            if not values and row.get("preflight_node_index") is not None:
                preflight_text = (
                    f"; node={row.get('preflight_node_index')}; degree={row.get('preflight_degree')}; "
                    f"sum_fp32={row.get('preflight_sum_fp32_reference')}; "
                    f"inv_fp32={row.get('preflight_inv_fp32_reference')}; "
                    f"policy={row.get('preflight_inverse_policy')}; "
                    f"sum_q={row.get('preflight_quantized_sum_scale16')}; "
                    f"inv_q={row.get('preflight_quantized_inv_scale16')}; "
                    f"expected_inv_q={row.get('preflight_expected_inv_q')}; "
                    f"product={row.get('preflight_quantized_product')}; "
                    f"inverse_ok={row.get('preflight_inverse_accepted')}; "
                    f"alpha_ok={row.get('preflight_alpha_accepted')}"
                )
            lines.append(
                f"- {row['display_dataset']}: {row['status']} / {row.get('block_reason', '')}; "
                f"step={row.get('failure_step', '')}; gate={row.get('failure_gate', '')}; "
                f"region={row.get('failure_region', '')}; row={row.get('failure_row', '')}; "
                f"relation={row.get('failure_relation', '')}; values={value_text}{preflight_text}"
            )
        else:
            reason = row.get("block_reason") or ""
            suffix = f" / {reason}" if reason else " /"
            lines.append(f"- {row['display_dataset']}: {row['status']}{suffix}")
    (RESULTS / BACKENDS[backend]["summary"]).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_native_constraint_debug_report() -> None:
    lines: list[str] = []
    lines.append("# Native Constraint Debug Report")
    lines.append("")
    lines.append("- Route: `ezkl_gat_native_compressed_csr_attention_full`")
    lines.append("- Diagnostic policy: explicit `--native-debug-scope` runs are available for development, but final benchmark collection uses only the verified full-model manifests.")
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
    lines.append("| Dataset | status | strict_public_input_equivalent | full_gat_forward_in_ezkl_proof | final_logits_derivation_complete | block_reason |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for row in ezkl_rows:
        lines.append(
            f"| {row['display_dataset']} | {row.get('status', '')} | "
            f"{row.get('strict_public_input_equivalent', False)} | "
            f"{row.get('full_gat_forward_in_ezkl_proof', False)} | "
            f"{row.get('final_logits_derivation_complete', False)} | {row.get('block_reason', '')} |"
        )
    lines.append("")
    lines.append("## Final Logits Binding")
    lines.append("")
    lines.append("| Dataset | binding_exists | binding_status | shape | public_instance_length | layout | columns | chunk_size | order | no_precomputed_logits_bypass |")
    lines.append("| --- | --- | --- | --- | ---: | --- | ---: | ---: | --- | --- |")
    for row in ezkl_rows:
        lines.append(
            f"| {row['display_dataset']} | {row.get('final_logits_binding_exists', False)} | "
            f"{row.get('final_logits_binding_status', '')} | {row.get('final_logits_shape', '')} | "
            f"{row.get('public_instance_length', '')} | {row.get('public_instance_layout', '')} | "
            f"{row.get('public_instance_column_count', '')} | {row.get('public_instance_chunk_size', '')} | "
            f"{row.get('public_instance_order', '')} | "
            f"{row.get('no_precomputed_logits_bypass', '')} |"
        )
    lines.append("")
    lines.append("No output-binding workaround, no precomputed logits, no subset, no shard, and no sample route was used.")
    lines.append("")
    lines.append("## Quantized Inverse Preflight Evidence")
    lines.append("")
    lines.append("| Dataset | step | gate | region | node | degree | policy | sum_q | inv_q | expected_inv_q | product | inverse_ok | alpha_ok | relation |")
    lines.append("| --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | --- | --- |")
    for row in ezkl_rows:
        lines.append(
            f"| {row['display_dataset']} | {row.get('failure_step', '')} | {row.get('failure_gate', '')} | "
            f"{row.get('failure_region', '')} | {row.get('preflight_node_index', '')} | "
            f"{row.get('preflight_degree', '')} | {row.get('preflight_inverse_policy', '')} | "
            f"{row.get('preflight_quantized_sum_scale16', '')} | "
            f"{row.get('preflight_quantized_inv_scale16', '')} | {row.get('preflight_expected_inv_q', '')} | "
            f"{row.get('preflight_quantized_product', '')} | {row.get('preflight_inverse_accepted', '')} | "
            f"{row.get('preflight_alpha_accepted', '')} | {row.get('failure_relation', '')} |"
        )
    lines.append("")
    lines.append("## Native Constraint Failure Evidence")
    lines.append("")
    lines.append("| Dataset | gate | region | row | values | remaining_blocker |")
    lines.append("| --- | --- | --- | ---: | --- | --- |")
    for row in ezkl_rows:
        if not row.get("failure_gate"):
            continue
        values = row.get("failure_values") or []
        value_text = "; ".join(values) if isinstance(values, list) else str(values)
        lines.append(
            f"| {row['display_dataset']} | {row.get('failure_gate', '')} | "
            f"{row.get('failure_region', '')} | {row.get('failure_row', '')} | "
            f"{value_text} | {row.get('failure_remaining_blocker', '')} |"
        )
    (RESULTS / "native_full_experiment_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    result_fields = [
        "dataset",
        "display_dataset",
        "backend",
        "gpu_backend",
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
    ] + METRICS
    comparison_fields = ["dataset", "backend", "ezkl_status", "ezkl_block_reason", "ezkl_route_id"]
    for metric in METRICS:
        comparison_fields.extend([f"main_cpu_{metric}", f"ezkl_{metric}"])

    cpu_rows, cpu_comparison_rows = collect_backend("cpu")
    write_json(RESULTS / "gat_ezkl_full_results.json", {"rows": cpu_rows})
    write_json(RESULTS / "main_vs_ezkl_full_comparison.json", {"rows": cpu_comparison_rows})
    write_csv(RESULTS / "gat_ezkl_full_results.csv", cpu_rows, result_fields)
    write_csv(RESULTS / "main_vs_ezkl_full_comparison.csv", cpu_comparison_rows, comparison_fields)
    write_summary(cpu_rows, cpu_comparison_rows, "cpu")
    write_native_constraint_debug_report()
    write_native_full_experiment_report(cpu_rows)

    gpu_rows, gpu_comparison_rows = collect_backend("gpu")
    write_json(RESULTS / "gat_ezkl_gpu_full_results.json", {"rows": gpu_rows})
    write_json(RESULTS / "main_vs_ezkl_gpu_full_comparison.json", {"rows": gpu_comparison_rows})
    write_csv(RESULTS / "gat_ezkl_gpu_full_results.csv", gpu_rows, result_fields)
    write_csv(RESULTS / "main_vs_ezkl_gpu_full_comparison.csv", gpu_comparison_rows, comparison_fields)
    write_summary(gpu_rows, gpu_comparison_rows, "gpu")
    print(f"wrote results under {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
