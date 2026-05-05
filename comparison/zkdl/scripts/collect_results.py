#!/usr/bin/env python3
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


METRICS = [
    "commitment_time_ms",
    "prove_time_ms",
    "verify_time_ms",
    "proof_size_bytes",
]


def read_dataset_order(root: Path) -> list[str]:
    return [
        line.strip()
        for line in (root / "configs" / "dataset_order.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_manifest(root: Path, dataset: str, route: str) -> dict:
    path = root / "runs" / dataset / route / "run_manifest.json"
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def collect_route(root: Path, route: str) -> None:
    datasets = read_dataset_order(root)
    rows = [read_manifest(root, dataset, route) for dataset in datasets]
    route_id = f"zkdl_gat_full_forward_{route}"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "route_id": route_id,
        "backend": route,
        "rows": rows,
    }

    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / f"zkdl_{route}_full_results.json"
    csv_path = results_dir / f"zkdl_{route}_full_results.csv"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    fields = [
        "dataset",
        "route_id",
        "backend",
        "status",
        *METRICS,
        "strict_public_input_equivalent",
        "full_gat_forward_in_zkdl_proof",
        "final_logits_derivation_complete",
        "no_precomputed_logits_bypass",
        "failure_step",
        "block_reason",
    ]
    if route == "gpu":
        fields.extend([
            "gpu_backend_active",
            "gpu_device_name",
            "gpu_device_count",
            "gpu_backend_name",
            "cpu_fallback_used",
        ])
    write_csv(csv_path, rows, fields)

    reference = json.loads((root / "configs" / "main_reference_results.json").read_text(encoding="utf-8"))[route]
    comparison_rows = []
    for row in rows:
        dataset = row["dataset"]
        main = reference[dataset]
        comparison = {
            "dataset": dataset,
            "main_route": route,
            "zkdl_route_id": row["route_id"],
            "zkdl_status": row["status"],
            "comparison_status": "verified" if row["status"] == "verified" else "blocked",
            "failure_step": row.get("failure_step", ""),
            "block_reason": row.get("block_reason", ""),
        }
        for metric in METRICS:
            comparison[f"main_{metric}"] = main[metric]
            comparison[f"zkdl_{metric}"] = row.get(metric)
        if route == "gpu":
            comparison["gpu_backend_active"] = row.get("gpu_backend_active")
            comparison["cpu_fallback_used"] = row.get("cpu_fallback_used")
            comparison["gpu_device_name"] = row.get("gpu_device_name")
        comparison_rows.append(comparison)

    comparison_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "route": route,
        "rows": comparison_rows,
    }
    comparison_json = results_dir / f"main_vs_zkdl_{route}_full_comparison.json"
    comparison_csv = results_dir / f"main_vs_zkdl_{route}_full_comparison.csv"
    comparison_json.write_text(json.dumps(comparison_payload, indent=2, sort_keys=True), encoding="utf-8")
    comparison_fields = [
        "dataset",
        "main_route",
        "zkdl_route_id",
        "zkdl_status",
        "comparison_status",
        *[f"main_{metric}" for metric in METRICS],
        *[f"zkdl_{metric}" for metric in METRICS],
        "failure_step",
        "block_reason",
    ]
    if route == "gpu":
        comparison_fields.extend(["gpu_backend_active", "cpu_fallback_used", "gpu_device_name"])
    write_csv(comparison_csv, comparison_rows, comparison_fields)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route", choices=["cpu", "gpu"], required=True)
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    collect_route(Path(args.root).resolve(), args.route)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
