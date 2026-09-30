#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import pathlib
from datetime import datetime, timezone
from typing import Dict, List, Tuple

CANONICAL_TIMING_SCHEMA = "online_trace_commit_prove_protocol_verify_v2"

_AGGREGATE_PATH = pathlib.Path(__file__).resolve().with_name("aggregate_secure_results.py")
_AGGREGATE_SPEC = importlib.util.spec_from_file_location("gatzk_aggregate_secure_results_for_export", _AGGREGATE_PATH)
if _AGGREGATE_SPEC is None or _AGGREGATE_SPEC.loader is None:
    raise RuntimeError(f"cannot load acceptance contract: {_AGGREGATE_PATH}")
_AGGREGATE = importlib.util.module_from_spec(_AGGREGATE_SPEC)
_AGGREGATE_SPEC.loader.exec_module(_AGGREGATE)
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]


class LoadedManifest(dict):
    """Manifest carrying the inputs needed for repeatable attestation re-auth."""

    def __init__(self, values: Dict[str, object], manifest_path: pathlib.Path, attestation_path: str, attestation_digest: str):
        super().__init__(values)
        if not isinstance(manifest_path, pathlib.Path):
            raise ValueError("loaded manifest requires its regular manifest path")
        if not isinstance(attestation_path, str) or not attestation_path.strip():
            raise ValueError("loaded manifest requires an external attestation path")
        if not isinstance(attestation_digest, str) or not attestation_digest.strip():
            raise ValueError("loaded manifest requires an external attestation digest")
        self._manifest_path = manifest_path
        self._attestation_path = attestation_path
        self._attestation_digest = attestation_digest


def parse_assignment(value: str, flag: str) -> Tuple[str, str]:
    if "=" not in value:
        raise ValueError(f"{flag} expects dataset=path_or_reason")
    dataset, payload = value.split("=", 1)
    dataset = dataset.strip()
    payload = payload.strip()
    if dataset not in _AGGREGATE.CANONICAL_DATASETS or not payload:
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


def _validate_verifier_report_binding(manifest: Dict[str, object], manifest_path: pathlib.Path, proof_hash: str) -> None:
    """Bind the public acceptance state to the binary-produced verifier report.

    The full aggregate command additionally checks the external registry,
    source inventory, build receipt, and all proof contents.  The per-backend
    exporter has no trusted-registry CLI input, but it must still reject a
    manifest whose local verifier report or proof bytes disagree with the
    claimed acceptance state.
    """
    context = f"benchmark export/{manifest_path.parent}"
    report_name = _AGGREGATE.nonempty_string(manifest.get("verifier_report_path"), "verifier_report_path", context)
    report_relative = pathlib.Path(report_name)
    if report_relative.is_absolute() or report_relative.parent != pathlib.Path("."):
        raise ValueError(f"{context}: verifier_report_path must be a run-local basename")
    report_path = manifest_path.parent / report_relative
    if not report_path.is_file() or report_path.is_symlink():
        raise ValueError(f"{context}: missing verifier report: {report_path}")
    report = _AGGREGATE.load_json_object(report_path, context)
    if set(report) != _AGGREGATE.VERIFIER_REPORT_KEYS:
        raise ValueError(f"{context}: verifier report has unexpected or missing fields")
    if report.get("format") != "zkgat_verifier_report":
        raise ValueError(f"{context}: verifier report format mismatch")

    manifest_verifier = _AGGREGATE.as_bool(manifest.get("verifier_accepted"), "verifier_accepted", context)
    report_verifier = _AGGREGATE.as_bool(report.get("verify_result"), "verify_result", context)
    if report_verifier is not manifest_verifier or not report_verifier:
        raise ValueError(f"{context}: verifier report acceptance is not manifest-bound")
    for field in ("reference_consistency_passed", "security_budget_passed"):
        if not _AGGREGATE.as_bool(report.get(field), field, context):
            raise ValueError(f"{context}: verifier report {field} is not true")
        if _AGGREGATE.as_bool(manifest.get(field), field, context) is not True:
            raise ValueError(f"{context}: manifest {field} is not true")

    report_tamper_passed = _AGGREGATE.as_bool(report.get("tamper_tests_passed"), "tamper_tests_passed", context)
    report_tamper_ms = _AGGREGATE.as_float(report.get("tamper_tests_ms"), "tamper_tests_ms", context)
    if report_tamper_passed:
        report_tamper_map = _AGGREGATE.canonical_tamper_case_string(report.get("tamper_test_case_map"), "tamper_test_case_map", context)
        if report_tamper_ms <= 0.0:
            raise ValueError(f"{context}: verifier report tamper_tests_ms must be positive")
    else:
        if report.get("tamper_test_case_map") != "" or report_tamper_ms != 0.0:
            raise ValueError(f"{context}: verifier report pending tamper state is malformed")
        report_tamper_map = ""

    manifest_tamper_keys = {"tamper_tests_passed", "tamper_test_case_map", "tamper_tests_ms"}
    present_manifest_tamper_keys = manifest_tamper_keys & set(manifest)
    if present_manifest_tamper_keys:
        if present_manifest_tamper_keys != manifest_tamper_keys:
            raise ValueError(f"{context}: run manifest tamper state is partial")
        manifest_tamper_passed = _AGGREGATE.as_bool(manifest.get("tamper_tests_passed"), "tamper_tests_passed", context)
        manifest_tamper_ms = _AGGREGATE.as_float(manifest.get("tamper_tests_ms"), "tamper_tests_ms", context)
        if manifest_tamper_passed:
            manifest_tamper_map = _AGGREGATE.canonical_tamper_case_string(manifest.get("tamper_test_case_map"), "tamper_test_case_map", context)
            if manifest_tamper_map != report_tamper_map or manifest_tamper_ms != report_tamper_ms:
                raise ValueError(f"{context}: verifier report tamper state is not manifest-bound")
        elif manifest.get("tamper_test_case_map") != "" or manifest_tamper_ms != 0.0:
            raise ValueError(f"{context}: run manifest pending tamper state is malformed")
        if manifest_tamper_passed is not report_tamper_passed:
            raise ValueError(f"{context}: verifier report tamper state is not manifest-bound")
    for field in (
        "security_budget_ledger_digest",
        "security_budget_exact",
        "public_input_digest",
        "topology_digest",
        "source_tree_hash",
        "binary_hash",
        "build_identity",
    ):
        if report.get(field) != manifest.get(field):
            raise ValueError(f"{context}: verifier report {field} is not manifest-bound")
    for field in ("trusted_registry_digest", "authenticated_registry_digest"):
        _AGGREGATE.sha256_string(report.get(field), field, context)
    if report.get("trusted_registry_digest") != report.get("authenticated_registry_digest"):
        raise ValueError(f"{context}: verifier report registry trust binding mismatch")
    if report.get("trusted_registry_digest") != manifest.get("trusted_registry_digest"):
        raise ValueError(f"{context}: verifier report registry digest is not manifest-bound")
    if report.get("trusted_registry_digest") != manifest.get("verifier_registry_digest"):
        raise ValueError(f"{context}: verifier report verifier registry digest is not manifest-bound")
    if report.get("proof_bin_sha256") != proof_hash:
        raise ValueError(f"{context}: verifier report proof hash is not proof-bound")
    parsed_digest = _AGGREGATE.sha256_string(report.get("parsed_output_digest"), "parsed_output_digest", context)
    report_output_digest = _AGGREGATE.sha256_string(report.get("proof_public_output_hash"), "proof_public_output_hash", context)
    if parsed_digest != report_output_digest or report_output_digest != manifest.get("proof_public_output_hash"):
        raise ValueError(f"{context}: verifier report output digest is not manifest-bound")
    if _AGGREGATE.as_int(report.get("parsed_output_count"), "parsed_output_count", context) <= 0:
        raise ValueError(f"{context}: verifier report parsed output is empty")
    report_hash = _AGGREGATE.sha256_file(report_path)
    if report_hash != manifest.get("verifier_report_hash"):
        raise ValueError(f"{context}: verifier_report_hash does not match verifier_report.json")


def load_manifest(path: pathlib.Path, attestation_path: str | None = None, attestation_digest: str | None = None) -> Dict[str, object]:
    manifest_path = path / "run_manifest.json" if path.is_dir() else path
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError(f"run manifest must be a regular file: {manifest_path}")
    data = _AGGREGATE.load_json_object(manifest_path, f"benchmark export/{manifest_path.parent}")
    manifest = {key: normalize_scalar(value) for key, value in data.items()}
    _AGGREGATE.nonempty_string(manifest.get("input_preprocess_id"), "input_preprocess_id", f"benchmark export/{manifest_path.parent}")
    proof_path = manifest_path.parent / "proof.bin"
    if not proof_path.is_file() or proof_path.is_symlink():
        raise ValueError(f"official benchmark export requires a regular proof.bin: {proof_path}")
    actual_proof_hash = _AGGREGATE.sha256_file(proof_path)
    if manifest.get("proof_hash") != actual_proof_hash:
        raise ValueError(f"official benchmark export proof_hash does not match proof.bin: {manifest_path.parent}")
    _validate_verifier_report_binding(manifest, manifest_path, actual_proof_hash)
    attestation = _AGGREGATE.check_tamper_attestation_manifest_binding(
        PROJECT_ROOT, manifest_path.parent, proof_path, manifest, attestation_path, attestation_digest, f"benchmark export/{manifest_path.parent}"
    )
    return LoadedManifest(manifest, manifest_path, attestation_path, attestation_digest)


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


def make_success_row(dataset: str, manifest: Dict[str, object]) -> Dict[str, object]:
    context = f"benchmark export/{dataset}"
    if not isinstance(manifest, LoadedManifest):
        raise ValueError(f"{context}: benchmark export requires a manifest loaded with external attestation inputs")
    manifest_path = manifest._manifest_path
    proof_path = manifest_path.parent / "proof.bin"
    attestation = _AGGREGATE.check_tamper_attestation_manifest_binding(
        PROJECT_ROOT, manifest_path.parent, proof_path, manifest, manifest._attestation_path, manifest._attestation_digest, context
    )
    _AGGREGATE.check_manifest_tamper_state(manifest, attestation, context)
    _, _, input_preprocess_id = _AGGREGATE.check_manifest_config_binding(PROJECT_ROOT, dataset, manifest, context)
    benchmark_mode = infer_benchmark_mode(manifest)
    if benchmark_mode != "single":
        raise ValueError(f"official benchmark export only supports single manifests, got {benchmark_mode!r} for {dataset}")
    if _AGGREGATE.as_bool(manifest.get("is_full_dataset"), "is_full_dataset", f"benchmark export/{dataset}") is not True:
        raise ValueError(f"official benchmark export only supports full-dataset manifests: {dataset}")
    if _AGGREGATE.as_bool(manifest.get("full_dataset"), "full_dataset", f"benchmark export/{dataset}") is not True:
        raise ValueError(f"official benchmark export only supports full-dataset manifests: {dataset}")
    if _AGGREGATE.as_bool(manifest.get("stale_result"), "stale_result", f"benchmark export/{dataset}") is not False:
        raise ValueError(f"official benchmark export refuses stale manifests: {dataset}")
    _AGGREGATE.check_formal_acceptance_state(manifest, f"benchmark export/{dataset}")
    timing_schema = manifest.get("timing_schema")
    if timing_schema != CANONICAL_TIMING_SCHEMA:
        raise ValueError(f"official benchmark export requires timing_schema={CANONICAL_TIMING_SCHEMA!r} for {dataset}, got {timing_schema!r}")
    _AGGREGATE.validate_online_timing(manifest, context)
    timing_values = {}
    for field in ("commitment_time_ms", "prove_time_ms", "verify_time_ms"):
        value = _AGGREGATE.as_float(manifest.get(field), field, context)
        if value < 0.0:
            raise ValueError(f"{context}: {field} must not be negative")
        timing_values[field] = value
    proof_size = _AGGREGATE.as_int(manifest.get("proof_size_bytes"), "proof_size_bytes", context)
    if proof_size <= 0:
        raise ValueError(f"{context}: proof_size_bytes must be positive")
    return {
        "dataset": dataset,
        "status": "ok",
        "blocker": "",
        "benchmark_mode": benchmark_mode,
        "timing_schema": timing_schema,
        "input_preprocess_id": input_preprocess_id,
        "commitment_time_ms": timing_values["commitment_time_ms"],
        "prove_time_ms": timing_values["prove_time_ms"],
        "verify_time_ms": timing_values["verify_time_ms"],
        "proof_size_bytes": proof_size,
        "tamper_tests_passed": True,
        "tamper_test_case_map": attestation["tamper_test_case_map"],
        "tamper_tests_ms": attestation["tamper_tests_ms"],
        "tamper_attestation_path": attestation["tamper_attestation_path"],
        "tamper_attestation_sha256": attestation["tamper_attestation_sha256"],
        "tamper_attestation_format": attestation["format"],
        "tamper_attestation_suite_id": attestation["suite_id"],
        "is_full_dataset": bool(manifest["is_full_dataset"]),
        "backend_name": str(manifest.get("backend_name", manifest.get("backend", ""))),
    }


def make_blocked_row(dataset: str, benchmark_mode: str, reason: str) -> Dict[str, object]:
    return {
        "dataset": dataset,
        "status": "blocked",
        "blocker": reason,
        "benchmark_mode": benchmark_mode,
        "commitment_time_ms": None,
        "prove_time_ms": None,
        "verify_time_ms": None,
        "proof_size_bytes": None,
        "is_full_dataset": None,
        "backend_name": "",
    }


def ensure_consistent_benchmark_mode(rows: List[Dict[str, object]]) -> str:
    modes = {str(row["benchmark_mode"]) for row in rows if row["status"] == "ok"}
    if not modes:
        blocked_modes = {str(row["benchmark_mode"]) for row in rows}
        mode = next(iter(blocked_modes)) if blocked_modes else "single"
        if mode != "single":
            raise ValueError(f"official benchmark export only supports single mode, got {mode!r}")
        return mode
    if len(modes) != 1:
        raise ValueError(f"inconsistent benchmark_mode across successful rows: {sorted(modes)}")
    mode = next(iter(modes))
    if mode != "single":
        raise ValueError(f"official benchmark export only supports single mode, got {mode!r}")
    return mode


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

    payload = {"generated_at": datetime.now(timezone.utc).isoformat(), "benchmark_mode": mode, "rows": rows}
    (output_dir / "latest.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    csv_columns = ["dataset", "status", "blocker", "benchmark_mode", "commitment_time_ms", "prove_time_ms", "verify_time_ms", "proof_size_bytes"]
    with (output_dir / "latest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in csv_columns})

    lines = ["# 最新正式结果", ""]
    lines.extend(
        [
            "| 数据集 | 状态 | commitment_time_ms | prove_time_ms | verify_time_ms | proof_size_bytes | 阻塞说明 |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in rows:
        lines.append(
            "| " + " | ".join(
                [
                    markdown_cell(row["dataset"]),
                    markdown_cell(status_label(row["status"])),
                    markdown_cell(row["commitment_time_ms"]),
                    markdown_cell(row["prove_time_ms"]),
                    markdown_cell(row["verify_time_ms"]),
                    markdown_cell(row["proof_size_bytes"]),
                    markdown_cell(row["blocker"]),
                ]
            ) + " |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", default=[], help="dataset=run_manifest.json_or_dir")
    parser.add_argument("--attestation", action="append", default=[], help="dataset=external tamper attestation path (one per --run)")
    parser.add_argument("--attestation-digest", action="append", default=[], help="dataset=external SHA-256 tamper-attestation digest (one per --run)")
    parser.add_argument("--blocked", action="append", default=[], help="dataset=reason")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-order", default="cora,citeseer,pubmed,ppi,ogbn_arxiv")
    parser.add_argument("--blocked-benchmark-mode", default="")
    args = parser.parse_args()

    rows_by_dataset: Dict[str, Dict[str, object]] = {}
    attestation_paths: Dict[str, str] = {}
    attestation_digests: Dict[str, str] = {}
    for item in args.attestation:
        dataset, value = parse_assignment(item, "--attestation")
        if dataset in attestation_paths:
            raise ValueError(f"duplicate attestation assignment: {dataset}")
        attestation_paths[dataset] = value
    for item in args.attestation_digest:
        dataset, value = parse_assignment(item, "--attestation-digest")
        if dataset in attestation_digests:
            raise ValueError(f"duplicate attestation digest assignment: {dataset}")
        attestation_digests[dataset] = value
    for item in args.run:
        dataset, path_value = parse_assignment(item, "--run")
        if dataset in rows_by_dataset:
            raise ValueError(f"duplicate run assignment: {dataset}")
        if dataset not in attestation_paths or dataset not in attestation_digests:
            raise ValueError(f"missing external attestation path or digest for {dataset}")
        rows_by_dataset[dataset] = make_success_row(dataset, load_manifest(pathlib.Path(path_value), attestation_paths[dataset], attestation_digests[dataset]))

    mode = ensure_consistent_benchmark_mode(list(rows_by_dataset.values()))
    blocked_mode = args.blocked_benchmark_mode or mode
    for item in args.blocked:
        dataset, reason = parse_assignment(item, "--blocked")
        if dataset in rows_by_dataset:
            raise ValueError(f"duplicate benchmark assignment: {dataset}")
        rows_by_dataset[dataset] = make_blocked_row(dataset, blocked_mode, reason)

    successful_datasets = {dataset for dataset, row in rows_by_dataset.items() if row["status"] == "ok"}
    if set(attestation_paths) != successful_datasets:
        raise ValueError("external attestation paths must cover exactly the successful --run datasets")
    if set(attestation_digests) != successful_datasets:
        raise ValueError("external attestation digests must cover exactly the successful --run datasets")

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
