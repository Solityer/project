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


def collect() -> tuple[list[dict], list[dict]]:
    main_by_dataset = main_rows()
    ezkl_rows: list[dict] = []
    comparison_rows: list[dict] = []
    for run_id, main_key, display, run_dir in DATASETS:
        manifest_path = EZKL_ROOT / run_dir / "run_manifest.json"
        if manifest_path.exists():
            manifest = read_json(manifest_path)
        else:
            manifest = {
                "dataset": run_id,
                "status": "failed",
                "block_reason": "missing_run_manifest",
            }
        ezkl_row = {
            "dataset": run_id,
            "display_dataset": display,
            "route_id": manifest.get("route_id", ""),
            "status": manifest.get("status", "failed"),
            "block_reason": manifest.get("block_reason", ""),
            "strict_public_input_equivalent": manifest.get("strict_public_input_equivalent", False),
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
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_summary(ezkl_rows: list[dict], comparison_rows: list[dict]) -> None:
    lines: list[str] = []
    lines.append("# GAT-ZKML vs ezkl Full-Dataset Baseline Summary")
    lines.append("")
    lines.append("- Strict public input equivalent: false for all completed rows because no full GAT ezkl proof verified.")
    lines.append("- Full GAT forward entered ezkl proof: false for all rows.")
    lines.append("- No precomputed logits, identity, reshape, digest, folding, subset, shard, or scaled route was used.")
    lines.append("")
    lines.append("## Dataset Status")
    lines.append("")
    lines.append("| Dataset | status | route_id | block_reason |")
    lines.append("| --- | --- | --- | --- |")
    for row in ezkl_rows:
        lines.append(
            f"| {row['display_dataset']} | {row['status']} | {row.get('route_id', '')} | {row.get('block_reason', '')} |"
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
    lines.append("- Route: `ezkl_gat_native_csr_segment_attention_full`")
    lines.append("- Native primitive: `GraphSegmentAttention` / `csr_segment_attention_v1`")
    lines.append("- Layout: CSR/ragged dst segments over `src[E]`, `dst[E]`, `segment_ptr[N+1]`, `edge_ptr[G+1]`, and `node_ptr[G+1]`.")
    lines.append("- Exact inverse attempt: native witness inverse column with `Sum_i * Inv_i = 1`; no epsilon reciprocal is used.")
    lines.append("")
    lines.append("| Dataset | status | block_reason | failure_step | rows | required_logrows | max_logrows |")
    lines.append("| --- | --- | --- | --- | ---: | ---: | ---: |")
    for run_id, _main_key, display, run_dir in DATASETS:
        manifest_path = EZKL_ROOT / run_dir / "run_manifest.json"
        manifest = read_json(manifest_path) if manifest_path.exists() else {}
        plan = manifest.get("native_compile_plan") or {}
        estimate = plan.get("estimate") or {}
        lines.append(
            f"| {display} | {manifest.get('status', 'missing')} | {manifest.get('block_reason', '')} | "
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


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    ezkl_rows, comparison_rows = collect()
    write_json(RESULTS / "gat_ezkl_full_results.json", {"rows": ezkl_rows})
    write_json(RESULTS / "main_vs_ezkl_full_comparison.json", {"rows": comparison_rows})
    write_csv(
        RESULTS / "gat_ezkl_full_results.csv",
        ezkl_rows,
        ["dataset", "display_dataset", "route_id", "status", "block_reason", "strict_public_input_equivalent"] + METRICS,
    )
    comparison_fields = ["dataset", "ezkl_status", "ezkl_block_reason", "ezkl_route_id"]
    for metric in METRICS:
        comparison_fields.extend([f"main_cpu_{metric}", f"ezkl_{metric}"])
    write_csv(RESULTS / "main_vs_ezkl_full_comparison.csv", comparison_rows, comparison_fields)
    write_summary(ezkl_rows, comparison_rows)
    segment_rows = collect_segment_rows()
    write_segment_summary(segment_rows)
    write_native_csr_report()
    print(f"wrote results under {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
