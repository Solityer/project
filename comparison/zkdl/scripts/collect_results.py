#!/usr/bin/env python3
"""Collect the exact ten zkDL result cells without manufacturing success."""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

from runner_contract import (
    BASELINE_ROOT,
    CELL_FILES,
    DATASETS,
    DEVICES,
    atomic_write_bytes,
    artifact_inventory,
    ensure_contained,
    read_json,
)


IDENTITY_FIELDS = (
    "config_sha256",
    "protocol_sha256",
    "resource_manifest_id",
    "resource_inventory_digest",
    "integer_bound_ledger_id",
    "graph_features_digest",
    "topology_digest",
    "public_instance",
    "checkpoint_manifest_id",
    "model_shape_id",
    "static_tables_id",
    "static_table_rows_id",
    "model_arch_id",
    "model_param_id",
    "provenance",
)


def _telemetry_errors(result: dict, success: dict, device: str) -> list[str]:
    errors: list[str] = []
    telemetry = result.get("gpu_telemetry")
    if device == "cpu":
        if telemetry not in (None, {}):
            errors.append("cpu_gpu_telemetry_present")
        if success.get("gpu_telemetry") not in (None, {}):
            errors.append("cpu_success_gpu_telemetry_present")
        return errors
    if not isinstance(telemetry, dict):
        return ["gpu_telemetry_missing"]
    if success.get("gpu_telemetry") != telemetry:
        errors.append("gpu_success_telemetry_mismatch")
    if telemetry.get("provider_id") != "zkDL-native-CUDA-BLS12-381-v1":
        errors.append("gpu_provider")
    if telemetry.get("cross_backend_kat_passed") is not True:
        errors.append("gpu_kat")
    for key in (
        "msm_kernel_invocations",
        "field_inner_product_invocations",
        "scalar_fold_invocations",
        "point_fold_invocations",
        "sumcheck_kernel_invocations",
    ):
        try:
            if int(telemetry.get(key, 0)) <= 0:
                errors.append(f"gpu_{key}")
        except (TypeError, ValueError):
            errors.append(f"gpu_{key}")
    return errors


def inspect_cell(run: Path, dataset: str, device: str) -> tuple[dict, list[str]]:
    result_path = run / "result.json"
    if not result_path.is_file():
        return (
            {
                "dataset": dataset,
                "device": device,
                "status": "missing",
                "proof_generated": False,
                "verified": False,
                "formal_experiment_eligible": False,
                "artifact_inventory_valid": False,
                "blocker": f"missing result: {run}",
            },
            ["missing_result"],
        )
    result = read_json(result_path, label="cell result")
    errors: list[str] = []
    if result.get("dataset") != dataset:
        errors.append("result_dataset_mismatch")
    if result.get("device") != device:
        errors.append("result_device_mismatch")
    if result.get("status") == "ok":
        try:
            inventory = artifact_inventory(run, require_complete=True)
            success = read_json(run / "success_manifest.json", label="cell success manifest")
            if success.get("schema") != "zkDL_gat_cell_success_manifest_v1":
                errors.append("success_manifest_schema")
            if success.get("status") != "verified":
                errors.append("success_manifest_status")
            if success.get("dataset") != dataset or success.get("device") != device:
                errors.append("success_manifest_cell")
            if success.get("files") != {key: inventory[key] for key in inventory if key != "success_manifest.json"}:
                errors.append("success_manifest_inventory")
            for field in IDENTITY_FIELDS:
                if field not in result:
                    errors.append(f"result_missing_{field}")
                if field not in success:
                    errors.append(f"success_missing_{field}")
            if isinstance(result.get("provenance"), dict) and isinstance(success.get("provenance"), dict):
                if result["provenance"] != success["provenance"]:
                    errors.append("provenance_mismatch")
            errors.extend(_telemetry_errors(result, success, device))
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append(f"artifact_inventory:{error}")
    else:
        errors.append("cell_not_success")
    row = dict(result)
    row["artifact_inventory_valid"] = not errors
    row["collector_errors"] = errors
    if errors:
        row["formal_experiment_eligible"] = False
    return row, errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=BASELINE_ROOT / "runs")
    parser.add_argument("--output", type=Path, default=BASELINE_ROOT / "results")
    args = parser.parse_args()
    runs = ensure_contained(args.runs, label="runs root")
    output = ensure_contained(args.output, label="result output")
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    errors: list[str] = []
    for dataset in DATASETS:
        for device in DEVICES:
            run = runs / dataset / device / "warm"
            row, row_errors = inspect_cell(run, dataset, device)
            rows.append(row)
            errors.extend(f"{dataset}/{device}:{error}" for error in row_errors)
    matrix_eligible = len(errors) == 0 and all(
        row.get("status") == "ok"
        and row.get("proof_generated") is True
        and row.get("verified") is True
        and row.get("formal_experiment_eligible") is True
        and row.get("privacy_profile_status") == "supported_multi_session_registered_model"
        for row in rows
    )
    payload = {
        "schema": "zkDL_gat_result_matrix_v2",
        "baseline": "zkdl",
        "dataset_order": list(DATASETS),
        "device_order": list(DEVICES),
        "row_count": len(rows),
        "privacy_profile_status": "supported_multi_session_registered_model",
        "formal_experiment_eligible": matrix_eligible,
        "collector_errors": errors,
        "rows": rows,
    }
    atomic_write_bytes(
        output / "final_zkdl_results.json",
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    fields = [
        "device",
        "dataset",
        "status",
        "proof_generated",
        "verified",
        "artifact_inventory_valid",
        "adapter_time_ms",
        "compile_time_ms",
        "prove_time_ms",
        "verify_time_ms",
        "proof_size_bytes",
        "privacy_profile_status",
        "formal_experiment_eligible",
        "proof_format",
        "model_session_commitments",
        "model_rerandomization_proofs",
        "model_session_rerandomization",
        "config_sha256",
        "protocol_sha256",
        "resource_manifest_id",
        "resource_inventory_digest",
        "integer_bound_ledger_id",
        "graph_features_digest",
        "topology_digest",
        "checkpoint_manifest_id",
        "model_shape_id",
        "static_tables_id",
        "static_table_rows_id",
        "model_arch_id",
        "model_param_id",
        "public_instance",
        "gpu_telemetry",
        "provenance",
        "blocker",
    ]
    handle = io.StringIO()
    writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_bytes(output / "final_zkdl_results.csv", handle.getvalue().encode("utf-8"))
    lines = [
        "# zkDL formal result matrix",
        "",
        "Strict reusable-model privacy status: `supported_multi_session_registered_model`.",
        "",
        f"Matrix eligibility: `{matrix_eligible}`; collector errors: `{len(errors)}`.",
        "",
        "| dataset | device | status | verified | artifacts | prove ms | verify ms | proof bytes |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('dataset')} | {row.get('device')} | {row.get('status')} | "
            f"{row.get('verified')} | {row.get('artifact_inventory_valid')} | "
            f"{row.get('prove_time_ms', '')} | {row.get('verify_time_ms', '')} | "
            f"{row.get('proof_size_bytes', '')} |"
        )
    atomic_write_bytes(output / "final_zkdl_report.md", ("\n".join(lines) + "\n").encode("utf-8"))
    print(f"ZKDL_COLLECT rows={len(rows)} eligible={matrix_eligible} errors={len(errors)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
