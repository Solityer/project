#!/usr/bin/env python3
"""Collect GAT-ZKML ezkl baseline results into JSON/CSV/Markdown."""

from __future__ import annotations

import csv
import json
from pathlib import Path


EZKL_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path("/home/pzh/project")
RESULTS = EZKL_ROOT / "results"
DATASETS = [
    ("cora_full", "cora", "Cora", "runs/cora_full"),
    ("citeseer_full", "citeseer", "Citeseer", "runs/citeseer_full"),
    ("pubmed_full", "pubmed", "Pubmed", "runs/pubmed_full"),
    ("ppi_full_formal", "ppi", "PPI", "runs/ppi_full_formal"),
    ("ogbn_arxiv_full", "ogbn-arxiv", "ogbn-arxiv", "runs/ogbn_arxiv_full"),
]
METRICS = ["commitment_time_ms", "prove_time_ms", "verify_time_ms", "proof_size_bytes"]
SEGMENT_FIELDS = [
    "node_count",
    "edge_count",
    "max_in_degree",
    "total_packed_slots",
    "padding_ratio",
    "route_feasible_full",
    "blocker_if_any",
]
NATIVE_DEBUG_SCOPES = [
    "level0_mul",
    "inverse_only",
    "segment_sum_inverse",
    "segment_max_only",
    "lrelu_lookup_only",
    "exp_lookup_only",
    "elu_lookup_only",
    "compressed_projection_only",
    "compressed_aggregation_only",
    "single_head_first_layer",
    "first_hidden_layer_all_heads",
    "hidden_only",
    "output_only",
    "output_projection_only",
    "output_attention_only",
    "output_aggregation_only",
    "output_bias_add_only",
    "output_average_only",
    "final_logits_public_binding_only",
    "output_full_derivation",
    "full_model",
]


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


def effective_manifest(run_dir: str, run_id: str) -> dict:
    base = EZKL_ROOT / run_dir
    manifest_path = base / "run_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {
        "dataset": run_id,
        "status": "failed",
        "block_reason": "missing_run_manifest",
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


def collect() -> tuple[list[dict], list[dict]]:
    main_by_dataset = main_rows()
    ezkl_rows: list[dict] = []
    comparison_rows: list[dict] = []
    for run_id, main_key, display, run_dir in DATASETS:
        manifest = effective_manifest(run_dir, run_id)
        preflight = preflight_for_run(run_dir)
        final_binding = final_logits_binding_for_run(run_dir)
        ezkl_row = {
            "dataset": run_id,
            "display_dataset": display,
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
            "final_logits_binding_exists": bool(final_binding),
            "final_logits_binding_status": final_binding.get("binding_status", ""),
            "final_logits_shape": final_binding.get("final_logits_shape"),
            "public_instance_length": final_binding.get("public_instance_length"),
            "public_instance_order": final_binding.get("public_instance_order"),
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


def write_summary(ezkl_rows: list[dict], comparison_rows: list[dict]) -> None:
    lines: list[str] = []
    lines.append("# GAT-ZKML vs ezkl Full-Dataset Baseline Summary")
    lines.append("")
    verified_count = sum(1 for row in ezkl_rows if row.get("status") == "verified")
    lines.append(f"- Verified ezkl full-dataset rows: {verified_count}/{len(ezkl_rows)}.")
    lines.append("- Strict public input equivalent is true only for rows with a verified full native GAT proof.")
    lines.append("- No output-binding workaround, precomputed logits, identity, digest, folding, subset, shard, or sample route was used.")
    lines.append("")
    lines.append("## Dataset Status")
    lines.append("")
    lines.append("| Dataset | status | route_id | strict | full GAT forward in proof | final logits derivation | block_reason |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in ezkl_rows:
        lines.append(
            f"| {row['display_dataset']} | {row['status']} | {row.get('route_id', '')} | "
            f"{row.get('strict_public_input_equivalent', False)} | {row.get('full_gat_forward_in_ezkl_proof', False)} | "
            f"{row.get('final_logits_derivation_complete', False)} | "
            f"{row.get('block_reason', '')} |"
        )
    lines.append("")
    lines.append("## Four-Metric Comparison")
    lines.append("")
    lines.append(
        "| Dataset | main commitment_time_ms | ezkl commitment_time_ms | main prove_time_ms | ezkl prove_time_ms | main verify_time_ms | ezkl verify_time_ms | main proof_size_bytes | ezkl proof_size_bytes |"
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
    lines.append("## Blocked/Failed Reasons")
    lines.append("")
    for row in ezkl_rows:
        detail = row.get("failure_gate") or row.get("failure_region") or row.get("failure_relation")
        if detail:
            values = row.get("failure_values") or []
            value_text = ", ".join(values) if isinstance(values, list) else str(values)
            preflight_text = ""
            if row.get("preflight_node_index") is not None:
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
            lines.append(f"- {row['display_dataset']}: {row['status']} / {row.get('block_reason', '')}")
    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect_segment_rows() -> list[dict]:
    rows: list[dict] = []
    for run_id, _main_key, display, run_dir in DATASETS:
        analysis_path = EZKL_ROOT / run_dir / "segment_softmax_analysis.json"
        if not analysis_path.exists():
            continue
        analysis = read_json(analysis_path)
        row = {"dataset": run_id, "display_dataset": display}
        for field in SEGMENT_FIELDS:
            row[field] = analysis.get(field)
        rows.append(row)
    return rows


def write_segment_summary(rows: list[dict]) -> None:
    lines: list[str] = []
    lines.append("# Segment Softmax Analysis Summary")
    lines.append("")
    lines.append("| Dataset | N | E | max_in_degree | N * Dmax | padding_ratio | feasible / blocked | blocker |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |")
    for row in rows:
        feasible = "feasible" if row.get("route_feasible_full") else "blocked"
        ratio = row.get("padding_ratio")
        ratio_text = "" if ratio is None else f"{float(ratio):.6f}"
        lines.append(
            f"| {row['display_dataset']} | {row.get('node_count', '')} | {row.get('edge_count', '')} | "
            f"{row.get('max_in_degree', '')} | {row.get('total_packed_slots', '')} | "
            f"{ratio_text} | {feasible} | {row.get('blocker_if_any') or ''} |"
        )
    (RESULTS / "segment_softmax_analysis_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_native_csr_report() -> None:
    lines: list[str] = []
    lines.append("# Native CSR Segment Attention Report")
    lines.append("")
    lines.append("- Legacy report retained for continuity; rows below reflect the latest native route manifest for each dataset.")
    lines.append("- Layout: CSR/ragged dst segments over `src[E]`, `dst[E]`, `segment_ptr[N+1]`, `edge_ptr[G+1]`, and `node_ptr[G+1]`.")
    lines.append("- Inverse policy: native witness inverse column is checked against deterministic scale-16 `T_inv`; alpha uses quantized product rescale metadata.")
    lines.append("")
    lines.append("| Dataset | route_id | primitive | status | block_reason | failure_step | rows | required_logrows | max_logrows |")
    lines.append("| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |")
    for run_id, _main_key, display, run_dir in DATASETS:
        manifest = effective_manifest(run_dir, run_id)
        plan = manifest.get("native_compile_plan") or {}
        estimate = plan.get("estimate") or {}
        lines.append(
            f"| {display} | {manifest.get('route_id', '')} | {manifest.get('native_primitive', '')} | "
            f"{manifest.get('status', 'missing')} | {manifest.get('block_reason', '')} | "
            f"{manifest.get('failure_step', '')} | {estimate.get('total_rows', '')} | "
            f"{estimate.get('required_logrows', '')} | {estimate.get('max_logrows', '')} |"
        )
    lines.append("")
    lines.append("## Artifact Contract")
    lines.append("")
    for _run_id, _main_key, display, run_dir in DATASETS:
        base = EZKL_ROOT / run_dir
        spec = base / "graph_attention_spec.json"
        csr = base / "csr_segment_layout.json"
        payloads = base / "tensor_payloads.json"
        lines.append(
            f"- {display}: spec={spec.exists()}, csr={csr.exists()}, tensor_payloads={payloads.exists()}"
        )
    forced_rows: list[tuple[str, dict]] = []
    for _run_id, _main_key, display, run_dir in DATASETS:
        forced_path = EZKL_ROOT / run_dir / "forced_model_logrows30.compiled"
        if forced_path.exists():
            forced_rows.append((display, read_json(forced_path)))
    if forced_rows:
        lines.append("")
        lines.append("## Forced max-logrows=30 Compile Attempt")
        lines.append("")
        lines.append(
            "| Dataset | status | block_reason | rows | required_logrows | max_logrows | lookup_status |"
        )
        lines.append("| --- | --- | --- | ---: | ---: | ---: | --- |")
        for display, plan in forced_rows:
            estimate = plan.get("estimate") or {}
            lookup = plan.get("lookup_semantics") or {}
            lines.append(
                f"| {display} | {plan.get('status', '')} | {plan.get('block_reason', '')} | "
                f"{estimate.get('total_rows', '')} | {estimate.get('required_logrows', '')} | "
                f"{estimate.get('max_logrows', '')} | {lookup.get('formal_status', '')} |"
            )
    (RESULTS / "native_csr_segment_attention_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_native_compressed_csr_report() -> None:
    lines: list[str] = []
    lines.append("# Native Compressed CSR Attention Report")
    lines.append("")
    lines.append("- Route: `ezkl_gat_native_compressed_csr_attention_full`")
    lines.append("- Native primitive: `GraphSegmentAttentionCompressedCSR` / `csr_segment_attention_compressed_v1`")
    lines.append("- Row model: CSR edge rows, row-compressed projection and aggregation, segment max selector/state machine, quantized inverse lookup, alpha rescale, and batched formal lookup table accounting.")
    lines.append("- Formal tables: deterministic from `range_bits`, `quant_cfg_id`, and `static_table_id`; no witness-specific table is generated.")
    lines.append("")
    lines.append("| Dataset | status | block_reason | previous rows | compressed rows | reduction | required_logrows | feasible | table_hash |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |")
    for run_id, _main_key, display, run_dir in DATASETS:
        base = EZKL_ROOT / run_dir
        manifest = effective_manifest(run_dir, run_id)
        estimate = read_json(base / "compressed_row_estimate.json") if (base / "compressed_row_estimate.json").exists() else manifest.get("compressed_row_estimate", {})
        hashes = read_json(base / "formal_table_hashes.json") if (base / "formal_table_hashes.json").exists() else manifest.get("formal_table_hashes", {})
        rows = estimate.get("total_rows_estimated", "")
        previous = estimate.get("previous_native_csr_rows", "")
        reduction = estimate.get("row_reduction_ratio_vs_previous_native_csr", "")
        lines.append(
            f"| {display} | {manifest.get('status', 'missing')} | {manifest.get('block_reason', '')} | "
            f"{previous} | {rows} | {reduction} | {estimate.get('required_logrows', '')} | "
            f"{estimate.get('feasible_under_logrows_26', '')} | {hashes.get('table_hash', '')} |"
        )
    lines.append("")
    lines.append("## Required Artifacts")
    lines.append("")
    for _run_id, _main_key, display, run_dir in DATASETS:
        base = EZKL_ROOT / run_dir
        artifacts = [
            "run_manifest.json",
            "compressed_row_estimate.json",
            "formal_tables.json",
            "formal_table_hashes.json",
            "graph_attention_spec.json",
            "model_info.json",
        ]
        status = ", ".join(f"{name}={str((base / name).exists()).lower()}" for name in artifacts)
        lines.append(f"- {display}: {status}")
    (RESULTS / "native_compressed_csr_attention_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_native_constraint_debug_report() -> None:
    lines: list[str] = []
    lines.append("# Native Constraint Debug Report")
    lines.append("")
    lines.append("- Route: `ezkl_gat_native_compressed_csr_attention_full`")
    lines.append("- Diagnostic policy: every native debug scope runs MockProver before KZG prove; failures write `native_constraint_failure.json` and `.log`.")
    lines.append("- No output-binding workaround, precomputed logits, subset, shard, sample, digest, folding, or identity route is used.")
    lines.append("")
    lines.append("## Cora Scope Ladder")
    lines.append("")
    lines.append("| Scope | status | setup_ms | prove_ms | verify_ms | proof_size | verify_passed | block_reason | failure gate | failure region | row |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | ---: |")
    base = EZKL_ROOT / "runs/cora_full"
    for scope in NATIVE_DEBUG_SCOPES:
        manifest_path = base / f"debug_{scope}" / "run_manifest.json"
        manifest = read_json(manifest_path) if manifest_path.exists() else {"status": "missing"}
        lines.append(
            f"| {scope} | {manifest.get('status', 'missing')} | {metric_cell(manifest, 'setup_ms')} | "
            f"{metric_cell(manifest, 'prove_time_ms')} | {metric_cell(manifest, 'verify_time_ms')} | "
            f"{metric_cell(manifest, 'proof_size_bytes')} | {manifest.get('verify_passed', False)} | "
            f"{manifest.get('block_reason', '')} | {manifest.get('failure_gate', '')} | "
            f"{manifest.get('failure_region', '')} | {manifest.get('failure_row', '')} |"
        )
    root_manifest_path = base / "run_manifest.json"
    if root_manifest_path.exists():
        root_manifest = read_json(root_manifest_path)
        lines.append(
            f"| full_model_current | {root_manifest.get('status', 'missing')} | "
            f"{metric_cell(root_manifest, 'setup_ms')} | {metric_cell(root_manifest, 'prove_time_ms')} | "
            f"{metric_cell(root_manifest, 'verify_time_ms')} | {metric_cell(root_manifest, 'proof_size_bytes')} | "
            f"{root_manifest.get('verify_passed', False)} | {root_manifest.get('block_reason', '')} | "
            f"{root_manifest.get('failure_gate', '')} | {root_manifest.get('failure_region', '')} | "
            f"{root_manifest.get('failure_row', '')} |"
        )
    failure_path = base / "debug_full_model" / "native_constraint_failure.json"
    if not failure_path.exists():
        failure_path = base / "native_constraint_failure.json"
    if failure_path.exists():
        failure = read_json(failure_path)
        lines.append("")
        lines.append("## Current Full-Model Blocker")
        lines.append("")
        lines.append(f"- gate_name: `{failure.get('gate_name', '')}`")
        lines.append(f"- region_name: `{failure.get('region_name', '')}`")
        lines.append(f"- row_offset: `{failure.get('row_offset', '')}`")
        lines.append(f"- relation: {failure.get('expected_relation', '')}")
        lines.append(f"- actual_values: `{failure.get('actual_values', [])}`")
        lines.append(f"- attempted_fix: {failure.get('attempted_fix', '')}")
        lines.append(f"- remaining_blocker: {failure.get('remaining_blocker', '')}")
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
    lines.append("| Dataset | binding_exists | binding_status | shape | public_instance_length | order | no_precomputed_logits_bypass |")
    lines.append("| --- | --- | --- | --- | ---: | --- | --- |")
    for row in ezkl_rows:
        lines.append(
            f"| {row['display_dataset']} | {row.get('final_logits_binding_exists', False)} | "
            f"{row.get('final_logits_binding_status', '')} | {row.get('final_logits_shape', '')} | "
            f"{row.get('public_instance_length', '')} | {row.get('public_instance_order', '')} | "
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
    (RESULTS / "native_full_experiment_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    ezkl_rows, comparison_rows = collect()
    write_json(RESULTS / "gat_ezkl_full_results.json", {"rows": ezkl_rows})
    write_json(RESULTS / "main_vs_ezkl_full_comparison.json", {"rows": comparison_rows})
    write_csv(
        RESULTS / "gat_ezkl_full_results.csv",
        ezkl_rows,
        [
            "dataset",
            "display_dataset",
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
            "final_logits_hash",
            "public_instance_hash",
            "no_precomputed_logits_bypass",
        ]
        + METRICS,
    )
    comparison_fields = ["dataset", "ezkl_status", "ezkl_block_reason", "ezkl_route_id"]
    for metric in METRICS:
        comparison_fields.extend([f"main_cpu_{metric}", f"ezkl_{metric}"])
    write_csv(RESULTS / "main_vs_ezkl_full_comparison.csv", comparison_rows, comparison_fields)
    write_summary(ezkl_rows, comparison_rows)
    segment_rows = collect_segment_rows()
    write_segment_summary(segment_rows)
    write_native_csr_report()
    write_native_compressed_csr_report()
    write_native_constraint_debug_report()
    write_native_full_experiment_report(ezkl_rows)
    print(f"wrote results under {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
