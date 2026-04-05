#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import pathlib
from datetime import datetime, timezone
from typing import Dict, List, Tuple


def parse_assignment(value: str, flag: str) -> Tuple[str, str]:
    if "=" not in value:
        raise ValueError(f"{flag} expects dataset=path_or_reason")
    dataset, payload = value.split("=", 1)
    dataset = dataset.strip()
    payload = payload.strip()
    if not dataset or not payload:
        raise ValueError(f"{flag} expects dataset=path_or_reason")
    return dataset, payload


def maybe_number(value: str):
    try:
        if any(ch in value for ch in ".eE"):
            return float(value)
        return int(value)
    except ValueError:
        return value


def parse_bool(value: str) -> bool | None:
    lowered = value.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    return None


def normalize_scalar(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    if value is None:
        return None
    if not isinstance(value, str):
        return str(value)
    parsed_bool = parse_bool(value)
    if parsed_bool is not None:
        return parsed_bool
    return maybe_number(value)


def load_manifest(path: pathlib.Path) -> Dict[str, object]:
    manifest_path = path / "run_manifest.json" if path.is_dir() else path
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"run manifest must be a JSON object: {manifest_path}")
    return {key: normalize_scalar(value) for key, value in data.items()}


def infer_benchmark_mode(manifest: Dict[str, object]) -> str:
    explicit = manifest.get("benchmark_mode")
    if isinstance(explicit, str) and explicit:
        return explicit
    notes = str(manifest.get("notes", ""))
    marker = "benchmark_mode="
    start = notes.find(marker)
    if start == -1:
        raise ValueError("run manifest missing benchmark_mode")
    start += len(marker)
    end = notes.find(";", start)
    if end == -1:
        end = len(notes)
    value = notes[start:end].strip()
    if not value:
        raise ValueError("run manifest benchmark_mode note is empty")
    return value


def hottest_stage(manifest: Dict[str, object]) -> str:
    candidates = [
        ("context_build_ms", float(manifest.get("context_build_ms", 0.0))),
        ("trace_generation_ms", float(manifest.get("trace_generation_ms", 0.0))),
        ("commit_dynamic_ms", float(manifest.get("commit_dynamic_ms", 0.0))),
        ("quotient_build_ms", float(manifest.get("quotient_build_ms", 0.0))),
        ("domain_opening_ms", float(manifest.get("domain_opening_ms", 0.0))),
        ("verify_quotient_ms", float(manifest.get("verify_quotient_ms", 0.0))),
    ]
    name, value = max(candidates, key=lambda item: item[1])
    return f"{name}:{value:.3f}"


def infer_route2_label(manifest: Dict[str, object]) -> str:
    explicit = str(manifest.get("route2_label", "")).strip()
    if explicit:
        return explicit
    labels: List[str] = []
    if manifest.get("enabled_fast_msm") is True:
        labels.append("msm")
    if manifest.get("enabled_parallel_fft") is True:
        labels.append("fft")
    if manifest.get("enabled_fft_backend_upgrade") is True:
        labels.append("packed")
    if manifest.get("enabled_fft_backend_upgrade") is True and manifest.get("enabled_fft_kernel_upgrade") is True:
        labels.append("kernel")
    if manifest.get("enabled_trace_layout_upgrade") is True:
        labels.append("layout")
    if manifest.get("enabled_fast_verify_pairing") is True:
        labels.append("pairing")
    return "_".join(labels) if labels else "legacy"


def make_success_row(dataset: str, manifest: Dict[str, object]) -> Dict[str, object]:
    return {
        "dataset": dataset,
        "status": "ok",
        "blocker": "",
        "benchmark_mode": infer_benchmark_mode(manifest),
        "prove_time_ms": float(manifest["prove_time_ms"]),
        "verify_time_ms": float(manifest["verify_time_ms"]),
        "proof_size_bytes": int(manifest["proof_size_bytes"]),
        "node_count": int(manifest["node_count"]),
        "edge_count": int(manifest["edge_count"]),
        "is_full_dataset": bool(manifest["is_full_dataset"]),
        "backend_name": str(manifest.get("backend_name", manifest.get("backend", ""))),
        "route2_label": infer_route2_label(manifest),
        "fft_backend_route": str(manifest.get("fft_backend_route", "")),
        "trace_generation_ms": float(manifest.get("trace_generation_ms", 0.0)),
        "commit_dynamic_ms": float(manifest.get("commit_dynamic_ms", 0.0)),
        "quotient_build_ms": float(manifest.get("quotient_build_ms", 0.0)),
        "domain_opening_ms": float(manifest.get("domain_opening_ms", 0.0)),
        "external_opening_ms": float(manifest.get("external_opening_ms", 0.0)),
        "verify_metadata_ms": float(manifest.get("verify_metadata_ms", 0.0)),
        "verify_transcript_ms": float(manifest.get("verify_transcript_ms", 0.0)),
        "verify_domain_opening_ms": float(manifest.get("verify_domain_opening_ms", 0.0)),
        "verify_quotient_ms": float(manifest.get("verify_quotient_ms", 0.0)),
        "verify_external_fold_ms": float(manifest.get("verify_external_fold_ms", 0.0)),
        "verify_misc_ms": float(manifest.get("verify_misc_ms", 0.0)),
        "verified": bool(manifest.get("verified", False)),
        "model_arch_id": str(manifest.get("model_arch_id", "")),
        "model_param_id": str(manifest.get("model_param_id", "")),
        "quant_cfg_id": str(manifest.get("quant_cfg_id", "")),
        "hotspot": hottest_stage(manifest),
    }


def make_blocked_row(dataset: str, benchmark_mode: str, reason: str) -> Dict[str, object]:
    return {
        "dataset": dataset,
        "status": "blocked",
        "blocker": reason,
        "benchmark_mode": benchmark_mode,
        "prove_time_ms": None,
        "verify_time_ms": None,
        "proof_size_bytes": None,
        "node_count": None,
        "edge_count": None,
        "is_full_dataset": None,
        "backend_name": "",
        "route2_label": "",
        "fft_backend_route": "",
        "trace_generation_ms": None,
        "commit_dynamic_ms": None,
        "quotient_build_ms": None,
        "domain_opening_ms": None,
        "external_opening_ms": None,
        "verify_metadata_ms": None,
        "verify_transcript_ms": None,
        "verify_domain_opening_ms": None,
        "verify_quotient_ms": None,
        "verify_external_fold_ms": None,
        "verify_misc_ms": None,
        "verified": False,
        "model_arch_id": "",
        "model_param_id": "",
        "quant_cfg_id": "",
        "hotspot": "",
    }


def ensure_consistent_benchmark_mode(rows: List[Dict[str, object]]) -> str:
    modes = {str(row["benchmark_mode"]) for row in rows if row["status"] == "ok"}
    if not modes:
        blocked_modes = {str(row["benchmark_mode"]) for row in rows}
        return next(iter(blocked_modes)) if blocked_modes else "single"
    if len(modes) != 1:
        raise ValueError(f"inconsistent benchmark_mode across successful rows: {sorted(modes)}")
    return next(iter(modes))


def markdown_cell(value) -> str:
    if value is None or value == "":
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def status_label(status: object) -> str:
    text = str(status)
    if text == "ok":
        return "已完成"
    if text == "blocked":
        return "阻塞"
    return text


def write_outputs(output_dir: pathlib.Path, mode: str, rows: List[Dict[str, object]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_mode": mode,
        "rows": rows,
    }
    (output_dir / "latest.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    csv_columns = [
        "dataset",
        "status",
        "blocker",
        "benchmark_mode",
        "prove_time_ms",
        "verify_time_ms",
        "proof_size_bytes",
        "node_count",
        "edge_count",
        "is_full_dataset",
        "backend_name",
        "route2_label",
        "fft_backend_route",
        "trace_generation_ms",
        "commit_dynamic_ms",
        "quotient_build_ms",
        "domain_opening_ms",
        "verify_quotient_ms",
        "hotspot",
    ]
    with (output_dir / "latest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in csv_columns})

    lines = [
        "# 最新基准结果",
        "",
        f"- 主表口径：`{mode}`",
        "",
        "| 数据集 | 状态 | prove_time_ms | verify_time_ms | proof_size_bytes | node_count | edge_count | route2 | 主要热点 | 阻塞说明 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    markdown_cell(row["dataset"]),
                    markdown_cell(status_label(row["status"])),
                    markdown_cell(row["prove_time_ms"]),
                    markdown_cell(row["verify_time_ms"]),
                    markdown_cell(row["proof_size_bytes"]),
                    markdown_cell(row["node_count"]),
                    markdown_cell(row["edge_count"]),
                    markdown_cell(row["route2_label"]),
                    markdown_cell(row["hotspot"]),
                    markdown_cell(row["blocker"]),
                ]
            )
            + " |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", default=[], help="dataset=run_manifest.json_or_dir")
    parser.add_argument("--blocked", action="append", default=[], help="dataset=reason")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-order", default="cora,citeseer,pubmed,ppi,ogbn-arxiv")
    parser.add_argument("--blocked-benchmark-mode", default="")
    args = parser.parse_args()

    rows_by_dataset: Dict[str, Dict[str, object]] = {}
    for item in args.run:
        dataset, path_value = parse_assignment(item, "--run")
        rows_by_dataset[dataset] = make_success_row(dataset, load_manifest(pathlib.Path(path_value)))

    mode = ensure_consistent_benchmark_mode(list(rows_by_dataset.values()))
    blocked_mode = args.blocked_benchmark_mode or mode
    for item in args.blocked:
        dataset, reason = parse_assignment(item, "--blocked")
        rows_by_dataset[dataset] = make_blocked_row(dataset, blocked_mode, reason)

    ordered_rows: List[Dict[str, object]] = []
    ordered_datasets = [entry.strip() for entry in args.dataset_order.split(",") if entry.strip()]
    seen = set()
    for dataset in ordered_datasets:
        row = rows_by_dataset.get(dataset)
        if row is not None:
            ordered_rows.append(row)
            seen.add(dataset)
    for dataset, row in rows_by_dataset.items():
        if dataset not in seen:
            ordered_rows.append(row)

    if not ordered_rows:
        raise ValueError("no benchmark rows supplied")
    write_outputs(pathlib.Path(args.output_dir), mode, ordered_rows)


if __name__ == "__main__":
    main()
