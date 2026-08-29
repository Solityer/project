#!/usr/bin/env python3
"""Aggregate the ten verified formal runs into one source-locked result set.

This command is intentionally fail-closed.  It does not create any result
file until exactly one complete warm CPU and CUDA run has passed all manifest,
proof, provenance, and security checks for each canonical dataset.  Controlled
smoke/stage runs therefore cannot be promoted into the paper result directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pathlib
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Tuple


DATASET_ALIASES = {
    "cora": "cora",
    "citeseer": "citeseer",
    "pubmed": "pubmed",
    "ppi": "ppi",
    "ogbn-arxiv": "ogbn_arxiv",
}
DATASET_ORDER = ("cora", "citeseer", "pubmed", "ppi", "ogbn-arxiv")
EXPECTED_MODEL_ARCH = {
    "cora": "gat_family_L3_hid8x8-8x8_kout2_c7",
    "citeseer": "gat_family_L3_hid8x8-8x8_kout2_c6",
    "pubmed": "gat_family_L3_hid8x8-8x8_kout2_c3",
    "ppi": "gat_family_L3_hid8x8-8x8_kout2_c121",
    "ogbn-arxiv": "gat_family_L3_hid8x8-8x8_kout2_c40",
}
EXPECTED_MODEL_PARAM = {
    "cora": "cora_planetoid_seed11",
    "citeseer": "citeseer_planetoid_seed11",
    "pubmed": "pubmed_planetoid_seed11",
    "ppi": "ppi_gat_seed11",
    "ogbn-arxiv": "ogbn_arxiv_seed11",
}
EXPECTED_SRS_ID = {
    "cora": "cora_formal_srs",
    "citeseer": "citeseer_formal_srs",
    "pubmed": "pubmed_formal_srs",
    "ppi": "ppi_formal_srs",
    "ogbn-arxiv": "ogbn_arxiv_formal_srs",
}
EXPECTED_D_IN = {
    "cora": [1433, 64],
    "citeseer": [3703, 64],
    "pubmed": [500, 64],
    "ppi": [50, 64],
    "ogbn-arxiv": [128, 64],
}
EXPECTED_CLASSES = {
    "cora": 7,
    "citeseer": 6,
    "pubmed": 3,
    "ppi": 121,
    "ogbn-arxiv": 40,
}
EXPECTED_SRS_DEGREE = {
    "cora": 16777215,
    "citeseer": 67108863,
    "pubmed": 67108863,
    "ppi": 33554431,
    "ogbn-arxiv": 134217727,
}
EXPECTED_SRS_DOMAINS = {
    "cora": {"C": 16, "FH": 4194304, "N": 4096, "cat": 128, "d": 16, "edge": 32768, "in": 2048},
    "citeseer": {"C": 8, "FH": 16777216, "N": 4096, "cat": 128, "d": 16, "edge": 32768, "in": 4096},
    "pubmed": {"C": 8, "FH": 16777216, "N": 32768, "cat": 128, "d": 16, "edge": 262144, "in": 512},
    "ppi": {"C": 128, "FH": 4194304, "N": 65536, "cat": 64, "d": 16, "edge": 8388608, "in": 64},
    "ogbn-arxiv": {"C": 64, "FH": 33554432, "N": 262144, "cat": 64, "d": 16, "edge": 8388608, "in": 256},
}
REQUIRED_HASHES = (
    "proof_hash",
    "checkpoint_model_hash",
    "quant_config_hash",
    "srs_manifest_hash",
    "public_metadata_hash",
    "static_table_hash",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def parse_assignment(value: str, flag: str) -> Tuple[str, str]:
    if "=" not in value:
        raise ValueError(f"{flag} expects dataset=run_manifest.json_or_dir")
    dataset, payload = value.split("=", 1)
    dataset = dataset.strip()
    payload = payload.strip()
    require(dataset in DATASET_ALIASES, f"{flag} has unsupported dataset {dataset!r}")
    require(payload != "", f"{flag} has an empty path for {dataset}")
    return dataset, payload


def as_bool(value: Any, field: str, context: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    raise ValueError(f"{context}: {field} must be a boolean or true/false string")


def as_int(value: Any, field: str, context: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{context}: {field} must be an integer")
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: {field} must be an integer") from exc
    return parsed


def as_float(value: Any, field: str, context: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{context}: {field} must be numeric")
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: {field} must be numeric") from exc
    require(math.isfinite(parsed), f"{context}: {field} must be finite")
    return parsed


def nonempty_string(value: Any, field: str, context: str) -> str:
    require(isinstance(value, str) and value.strip() != "", f"{context}: {field} is missing")
    return value.strip()


def resolve_input_path(project_root: pathlib.Path, value: str) -> pathlib.Path:
    candidate = pathlib.Path(value)
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return candidate.resolve()


def load_manifest(project_root: pathlib.Path, value: str) -> Tuple[pathlib.Path, pathlib.Path, Dict[str, Any]]:
    input_path = resolve_input_path(project_root, value)
    manifest_path = input_path / "run_manifest.json" if input_path.is_dir() else input_path
    require(manifest_path.is_file(), f"run manifest does not exist: {manifest_path}")
    run_dir = manifest_path.parent
    proof_path = run_dir / "proof.bin"
    require(proof_path.is_file(), f"run manifest has no proof.bin: {proof_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read run manifest {manifest_path}: {exc}") from exc
    require(isinstance(manifest, dict), f"run manifest must be a JSON object: {manifest_path}")
    return run_dir, proof_path, manifest


def load_json_object(path: pathlib.Path, context: str) -> Dict[str, Any]:
    require(path.is_file(), f"{context}: missing JSON artifact {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{context}: cannot read JSON artifact {path}: {exc}") from exc
    require(isinstance(value, dict), f"{context}: JSON artifact must be an object: {path}")
    return value


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_config(project_root: pathlib.Path, manifest: Dict[str, Any], context: str) -> pathlib.Path:
    config_value = nonempty_string(manifest.get("config"), "config", context)
    config_path = resolve_input_path(project_root, config_value)
    require(config_path.is_file(), f"{context}: config does not exist: {config_path}")
    try:
        config_path.relative_to(project_root)
    except ValueError as exc:
        raise ValueError(f"{context}: config must be inside the project root") from exc
    return config_path


def load_config_entries(path: pathlib.Path, context: str) -> Dict[str, str]:
    entries: Dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"{context}: cannot read config {path}: {exc}") from exc
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"{context}: malformed config line {line_number}")
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        require(key and value, f"{context}: malformed config line {line_number}")
        entries[key] = value
    return entries


def check_canonical_config(path: pathlib.Path, dataset: str, context: str) -> None:
    entries = load_config_entries(path, context)
    expected_dataset = DATASET_ALIASES[dataset]
    expected_values = {
        "dataset": expected_dataset,
        "L": "3",
        "hidden_profile": "[8x8,8x8]",
        "d_in_profile": "[" + ",".join(str(value) for value in EXPECTED_D_IN[dataset]) + "]",
        "K_out": "2",
        "num_classes": str(EXPECTED_CLASSES[dataset]),
        "range_bits": str({"cora": 12, "citeseer": 12, "pubmed": 12, "ppi": 14, "ogbn-arxiv": 21}[dataset]),
        "model_arch_id": EXPECTED_MODEL_ARCH[dataset],
        "model_param_id": EXPECTED_MODEL_PARAM[dataset],
        "quant_cfg_id": "gat_l3_scale_chain_v1",
        "static_table_id": "tables:lrelu+elu+exp+range",
        "degree_bound_id": "auto",
        "checkpoint_bundle": f"artifacts/checkpoints/{dataset.replace('-', '_')}_gat",
        "srs_id": EXPECTED_SRS_ID[dataset],
        "srs_path": f"artifacts/srs/{EXPECTED_SRS_ID[dataset]}",
    }
    for key, expected in expected_values.items():
        actual = entries.get(key, "")
        if key in {"hidden_profile", "d_in_profile"}:
            actual = actual.replace(" ", "")
            expected = expected.replace(" ", "")
        require(actual == expected, f"{context}: canonical config {key} mismatch")


def check_checkpoint_artifacts(
    project_root: pathlib.Path,
    dataset: str,
    run_manifest: Dict[str, Any],
    context: str,
) -> Dict[str, Any]:
    bundle_root = (
        project_root / "artifacts" / "checkpoints" / f"{dataset.replace('-', '_')}_gat"
    ).resolve()
    require(bundle_root.is_dir(), f"{context}: canonical checkpoint bundle is missing: {bundle_root}")
    checkpoint_manifest_path = bundle_root / "manifest.json"
    bundle_manifest = load_json_object(checkpoint_manifest_path, context)
    required_files = {
        "checkpoint_bundle_manifest_hash": checkpoint_manifest_path,
        "checkpoint_bundle_tensors_hash": bundle_root / "tensors.txt",
        "checkpoint_bundle_archive_hash": bundle_root / "tensors.npz",
        "checkpoint_source_hash": bundle_root / "best_model.pt",
    }
    for field, path in required_files.items():
        require(path.is_file(), f"{context}: canonical checkpoint file is missing: {path}")
        recorded = nonempty_string(run_manifest.get(field), field, context)
        actual = sha256_file(path)
        require(recorded == actual, f"{context}: {field} does not match {path}")

    expected_hidden_profile = [
        {"head_count": 8, "head_dim": 8},
        {"head_count": 8, "head_dim": 8},
    ]
    require(bundle_manifest.get("family_schema_version") == "multi_layer_multi_head_v2",
            f"{context}: checkpoint bundle family schema is not current")
    require(as_int(bundle_manifest.get("L"), "checkpoint L", context) == 3,
            f"{context}: checkpoint bundle L is not 3")
    require(bundle_manifest.get("d_in_profile") == EXPECTED_D_IN[dataset],
            f"{context}: checkpoint d_in_profile is not canonical")
    require(bundle_manifest.get("hidden_profile") == expected_hidden_profile,
            f"{context}: checkpoint hidden_profile is not canonical")
    require(as_int(bundle_manifest.get("K_out"), "checkpoint K_out", context) == 2,
            f"{context}: checkpoint K_out is not 2")
    require(as_int(bundle_manifest.get("C"), "checkpoint C", context) == EXPECTED_CLASSES[dataset],
            f"{context}: checkpoint class count is not canonical")
    require(bundle_manifest.get("model_arch_id") == EXPECTED_MODEL_ARCH[dataset],
            f"{context}: checkpoint model_arch_id is not canonical")
    require(bundle_manifest.get("model_param_id") == EXPECTED_MODEL_PARAM[dataset],
            f"{context}: checkpoint model_param_id is not canonical")
    require(bundle_manifest.get("quant_cfg_id") == "gat_l3_scale_chain_v1",
            f"{context}: checkpoint quant_cfg_id is not canonical")
    require(bundle_manifest.get("static_table_id") == "tables:lrelu+elu+exp+range",
            f"{context}: checkpoint static_table_id is not canonical")
    require(bundle_manifest.get("degree_bound_id") == "auto",
            f"{context}: checkpoint degree_bound_id is not canonical")
    source_value = nonempty_string(bundle_manifest.get("checkpoint_prefix"), "checkpoint_prefix", context)
    source_path = resolve_input_path(project_root, source_value)
    require(source_path == bundle_root / "best_model.pt",
            f"{context}: checkpoint_prefix does not point to canonical best_model.pt")
    return {
        "checkpoint_bundle_manifest_hash": run_manifest["checkpoint_bundle_manifest_hash"],
        "checkpoint_bundle_tensors_hash": run_manifest["checkpoint_bundle_tensors_hash"],
        "checkpoint_bundle_archive_hash": run_manifest["checkpoint_bundle_archive_hash"],
        "checkpoint_source_hash": run_manifest["checkpoint_source_hash"],
    }


def check_srs_artifacts(
    project_root: pathlib.Path,
    dataset: str,
    run_manifest: Dict[str, Any],
    context: str,
) -> Dict[str, Any]:
    srs_id = EXPECTED_SRS_ID[dataset]
    srs_root = (project_root / "artifacts" / "srs" / srs_id).resolve()
    srs_manifest_path = srs_root / "srs_manifest.json"
    srs_manifest = load_json_object(srs_manifest_path, context)
    require(srs_manifest.get("srs_id") == srs_id, f"{context}: SRS id is not canonical")
    recorded_srs_id = nonempty_string(run_manifest.get("srs_id"), "srs_id", context)
    require(recorded_srs_id == srs_id, f"{context}: run manifest srs_id is not canonical")
    require(srs_manifest.get("srs_type") == "trusted_srs", f"{context}: SRS type is not trusted_srs")
    require(as_int(srs_manifest.get("degree_bound"), "SRS degree_bound", context) == EXPECTED_SRS_DEGREE[dataset],
            f"{context}: SRS degree_bound is not canonical")
    raw_domains = srs_manifest.get("domain_sizes")
    require(isinstance(raw_domains, dict), f"{context}: SRS domain_sizes is missing")
    try:
        domain_sizes = {str(key): int(value) for key, value in raw_domains.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: SRS domain_sizes are malformed") from exc
    require(domain_sizes == EXPECTED_SRS_DOMAINS[dataset],
            f"{context}: SRS domain catalogue is not canonical")

    raw_manifest = srs_manifest_path.read_bytes()
    actual_hash = hashlib.sha256(raw_manifest).hexdigest()
    recorded_hash = nonempty_string(run_manifest.get("srs_manifest_hash"), "srs_manifest_hash", context)
    require(recorded_hash == actual_hash,
            f"{context}: srs_manifest_hash does not match {srs_manifest_path}")
    declared_hash = nonempty_string(srs_manifest.get("manifest_hash"), "manifest_hash", context)
    marker = b'  "manifest_hash"'
    marker_position = raw_manifest.rfind(marker)
    require(marker_position >= 0 and hashlib.sha256(raw_manifest[:marker_position]).hexdigest() == declared_hash,
            f"{context}: SRS manifest canonical hash field is invalid")
    return {"srs_manifest_hash": recorded_hash, "srs_id": srs_id}


def check_security(manifest: Dict[str, Any], proof_path: pathlib.Path, context: str) -> Dict[str, Any]:
    require(as_bool(manifest.get("verifier_reads_witness"), "verifier_reads_witness", context) is False,
            f"{context}: verifier_reads_witness must be false")
    require(as_int(manifest.get("private_plain_value_count"), "private_plain_value_count", context) == 0,
            f"{context}: private_plain_value_count must be zero")
    require(nonempty_string(manifest.get("proof_payload_audit_status"), "proof_payload_audit_status", context) == "passed",
            f"{context}: proof payload audit did not pass")
    for field in ("reference_consistency_passed", "cpu_gpu_reference_consistency_passed", "tamper_tests_passed"):
        require(as_bool(manifest.get(field), field, context), f"{context}: {field} must be true")
    tamper_ms = as_float(manifest.get("tamper_tests_ms"), "tamper_tests_ms", context)
    require(tamper_ms > 0.0, f"{context}: tamper_tests_ms must be positive")

    for field in REQUIRED_HASHES:
        nonempty_string(manifest.get(field), field, context)
    actual_proof_hash = sha256_file(proof_path)
    require(manifest["proof_hash"] == actual_proof_hash,
            f"{context}: proof_hash does not match proof.bin ({actual_proof_hash})")
    return {
        "verifier_reads_witness": False,
        "private_plain_value_count": 0,
        "proof_payload_audit_status": "passed",
        "reference_consistency_passed": True,
        "cpu_gpu_reference_consistency_passed": True,
        "tamper_tests_passed": True,
        "tamper_tests_ms": tamper_ms,
        "proof_hash_audit_passed": True,
    }


def check_manifest(
    project_root: pathlib.Path,
    dataset: str,
    backend: str,
    value: str,
) -> Dict[str, Any]:
    run_dir, proof_path, manifest = load_manifest(project_root, value)
    context = f"{backend}/{dataset} ({run_dir})"
    expected_manifest_dataset = DATASET_ALIASES[dataset]
    require(manifest.get("dataset") == expected_manifest_dataset,
            f"{context}: manifest dataset is not {expected_manifest_dataset}")
    require(manifest.get("status") == "VERIFY_OK", f"{context}: status is not VERIFY_OK")
    require(as_bool(manifest.get("verified"), "verified", context), f"{context}: verified is not true")
    require(as_bool(manifest.get("full_dataset"), "full_dataset", context), f"{context}: full_dataset is not true")
    require(as_bool(manifest.get("is_full_dataset"), "is_full_dataset", context), f"{context}: is_full_dataset is not true")
    require(as_bool(manifest.get("stale_result"), "stale_result", context) is False,
            f"{context}: stale_result must be false")
    require(manifest.get("benchmark_mode") == "warm", f"{context}: benchmark_mode must be warm")

    expected_compute_backend = "cpu" if backend == "cpu" else "cuda_hotspots"
    expected_backend = "cpu" if backend == "cpu" else "gpu"
    require(manifest.get("backend") == expected_backend,
            f"{context}: backend must be {expected_backend}")
    require(manifest.get("compute_backend_name") == expected_compute_backend,
            f"{context}: compute_backend_name must be {expected_compute_backend}")
    requested_gpu = as_bool(manifest.get("gpu_backend_requested"), "gpu_backend_requested", context)
    require(requested_gpu is (backend == "cuda"), f"{context}: gpu_backend_requested is inconsistent")
    invocation_count = as_int(manifest.get("gpu_kernel_invocation_count"), "gpu_kernel_invocation_count", context)
    if backend == "cpu":
        require(invocation_count == 0, f"{context}: CPU run reports CUDA kernel invocations")
    else:
        runtime_present = as_bool(manifest.get("gpu_runtime_present"), "gpu_runtime_present", context)
        runtime_loaded = as_bool(manifest.get("gpu_runtime_loaded"), "gpu_runtime_loaded", context)
        require(runtime_present or runtime_loaded, f"{context}: CUDA runtime evidence is missing")
        require(invocation_count > 0, f"{context}: CUDA run reports no kernel invocations")
        nonempty_string(manifest.get("gpu_device"), "gpu_device", context)
        nonempty_string(manifest.get("gpu_acceleration_status"), "gpu_acceleration_status", context)

    config_path = resolve_config(project_root, manifest, context)
    expected_config = (project_root / "configs" / f"{dataset.replace('-', '_')}_secure.cfg").resolve()
    require(config_path == expected_config,
            f"{context}: manifest config must be canonical {expected_config.relative_to(project_root)}")
    check_canonical_config(config_path, dataset, context)
    config_hash = sha256_file(config_path)
    recorded_config_hash = nonempty_string(manifest.get("config_sha256"), "config_sha256", context)
    require(recorded_config_hash == config_hash,
            f"{context}: config_sha256 does not match {config_path}")
    checkpoint_hashes = check_checkpoint_artifacts(project_root, dataset, manifest, context)
    srs_info = check_srs_artifacts(project_root, dataset, manifest, context)
    security = check_security(manifest, proof_path, context)
    timings = {
        "commitment_time_ms": as_float(manifest.get("commitment_time_ms"), "commitment_time_ms", context),
        "prove_time_ms": as_float(manifest.get("prove_time_ms"), "prove_time_ms", context),
        "verify_time_ms": as_float(manifest.get("verify_time_ms"), "verify_time_ms", context),
        "total_wall_clock_ms": as_float(manifest.get("total_wall_clock_ms"), "total_wall_clock_ms", context),
    }
    for field, value_number in timings.items():
        require(value_number >= 0.0, f"{context}: {field} must not be negative")
    proof_size = as_int(manifest.get("proof_size_bytes"), "proof_size_bytes", context)
    require(proof_size > 0 and proof_size == proof_path.stat().st_size,
            f"{context}: proof_size_bytes does not match proof.bin")

    row: Dict[str, Any] = {
        "dataset": dataset,
        "manifest_dataset": expected_manifest_dataset,
        "backend": backend,
        "status": "VERIFY_OK",
        "benchmark_mode": "warm",
        "run_dir": str(run_dir.relative_to(project_root)),
        "config": str(config_path.relative_to(project_root)),
        "config_sha256": recorded_config_hash,
        "proof_size_bytes": proof_size,
        "proof_hash": manifest["proof_hash"],
        "checkpoint_model_hash": manifest["checkpoint_model_hash"],
        "quant_config_hash": manifest["quant_config_hash"],
        "srs_manifest_hash": manifest["srs_manifest_hash"],
        "public_metadata_hash": manifest["public_metadata_hash"],
        "static_table_hash": manifest["static_table_hash"],
        "srs_id": srs_info["srs_id"],
        "model_arch": nonempty_string(manifest.get("model_arch"), "model_arch", context),
        "model_params": nonempty_string(manifest.get("model_params"), "model_params", context),
        "quant_config": nonempty_string(manifest.get("quant_config"), "quant_config", context),
        "static_tables": nonempty_string(manifest.get("static_tables"), "static_tables", context),
        "srs_degree_bound": nonempty_string(manifest.get("srs_degree_bound"), "srs_degree_bound", context),
        "gpu_runtime_present": as_bool(manifest.get("gpu_runtime_present"), "gpu_runtime_present", context),
        "gpu_runtime_loaded": as_bool(manifest.get("gpu_runtime_loaded"), "gpu_runtime_loaded", context),
        "gpu_kernel_invocation_count": invocation_count,
        "gpu_device": str(manifest.get("gpu_device", "")),
        "gpu_acceleration_status": str(manifest.get("gpu_acceleration_status", "")),
        "tamper_tests_ms": security["tamper_tests_ms"],
        **timings,
        **checkpoint_hashes,
        **security,
    }
    require(row["srs_id"] == EXPECTED_SRS_ID[dataset],
            f"{context}: srs_id is not the canonical {EXPECTED_SRS_ID[dataset]}")
    require(row["model_arch"] == EXPECTED_MODEL_ARCH[dataset],
            f"{context}: model_arch is not the canonical architecture")
    require(row["model_params"].startswith("registry_sha256="),
            f"{context}: model_params is not a bound model registry")
    require(row["quant_config"].startswith("gat_l3_scale_chain_v1|"),
            f"{context}: quant_config is not the canonical scale-chain binding")
    require(row["static_tables"].startswith("registry_sha256="),
            f"{context}: static_tables is not a bound static-table registry")
    return row


def ensure_pair_consistency(rows: Iterable[Dict[str, Any]]) -> None:
    grouped: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["dataset"], {})[row["backend"]] = row
    for dataset in DATASET_ORDER:
        pair = grouped[dataset]
        require(set(pair) == {"cpu", "cuda"}, f"{dataset}: CPU/CUDA pair is incomplete")
        cpu = pair["cpu"]
        cuda = pair["cuda"]
        for field in (
            "config",
            "config_sha256",
            "checkpoint_bundle_manifest_hash",
            "checkpoint_bundle_tensors_hash",
            "checkpoint_bundle_archive_hash",
            "checkpoint_source_hash",
            "checkpoint_model_hash",
            "quant_config_hash",
            "srs_manifest_hash",
            "static_table_hash",
            "srs_id",
            "model_arch",
            "model_params",
            "quant_config",
            "static_tables",
            "srs_degree_bound",
        ):
            require(cpu[field] == cuda[field], f"{dataset}: CPU/CUDA {field} mismatch")


def write_outputs(output_dir: pathlib.Path, rows: List[Dict[str, Any]]) -> None:
    # All validation, including pair consistency, has completed before this function is called.
    output_dir.mkdir(parents=True, exist_ok=True)
    security_rows = [
        {
            "dataset": row["dataset"],
            "backend": row["backend"],
            "run_dir": row["run_dir"],
            "proof_hash": row["proof_hash"],
            "proof_hash_audit_passed": row["proof_hash_audit_passed"],
            "verifier_reads_witness": row["verifier_reads_witness"],
            "private_plain_value_count": row["private_plain_value_count"],
            "proof_payload_audit_status": row["proof_payload_audit_status"],
            "reference_consistency_passed": row["reference_consistency_passed"],
            "cpu_gpu_reference_consistency_passed": row["cpu_gpu_reference_consistency_passed"],
            "tamper_tests_passed": row["tamper_tests_passed"],
            "tamper_tests_ms": row["tamper_tests_ms"],
        }
        for row in rows
    ]
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_mode": "warm",
        "run_count": len(rows),
        "rows": rows,
        "security_tests": {
            "all_passed": all(
                item["proof_hash_audit_passed"]
                and item["verifier_reads_witness"] is False
                and item["private_plain_value_count"] == 0
                and item["proof_payload_audit_status"] == "passed"
                and item["reference_consistency_passed"]
                and item["cpu_gpu_reference_consistency_passed"]
                and item["tamper_tests_passed"]
                for item in security_rows
            ),
            "run_count": len(security_rows),
            "checks": security_rows,
        },
    }
    (output_dir / "latest.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    csv_columns = [
        "dataset",
        "backend",
        "status",
        "benchmark_mode",
        "commitment_time_ms",
        "prove_time_ms",
        "verify_time_ms",
        "total_wall_clock_ms",
        "proof_size_bytes",
        "proof_hash",
        "config",
        "config_sha256",
        "checkpoint_bundle_manifest_hash",
        "checkpoint_bundle_tensors_hash",
        "checkpoint_bundle_archive_hash",
        "checkpoint_source_hash",
        "checkpoint_model_hash",
        "quant_config_hash",
        "srs_manifest_hash",
        "static_table_hash",
        "srs_id",
        "gpu_runtime_present",
        "gpu_kernel_invocation_count",
        "gpu_device",
    ]
    with (output_dir / "latest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in csv_columns})

    lines = [
        "# 最新正式结果",
        "",
        "本表只由十个 `VERIFY_OK`、full-dataset、warm manifest 聚合生成；CPU/CUDA 每个数据集的配置、模型、量化、静态表和 SRS 标识已逐项配对校验。",
        "",
        "| 数据集 | 后端 | 状态 | commitment (ms) | prove (ms) | verify (ms) | proof bytes | GPU kernels |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['backend']} | {row['status']} | "
            f"{row['commitment_time_ms']:.3f} | {row['prove_time_ms']:.3f} | "
            f"{row['verify_time_ms']:.3f} | {row['proof_size_bytes']} | "
            f"{row['gpu_kernel_invocation_count']} |"
        )
    lines.extend(
        [
            "",
            "安全字段详见 `security_tests_report.md`；当前仓库只有在十组正式运行完成后才应生成本目录。",
            "",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    security_lines = [
        "# 正式运行安全核验",
        "",
        "以下字段直接来自十个正式 run manifest，并额外核验了 `proof.bin` 的 SHA-256。所有行都必须通过，聚合器才会写出本报告。",
        "",
        "| 数据集 | 后端 | proof hash | proof 审计 | witness 读取 | private plain values | payload audit | reference | CPU/GPU reference | tamper | tamper ms |",
        "| --- | --- | --- | --- | --- | ---: | --- | --- | --- | --- | ---: |",
    ]
    for item in security_rows:
        security_lines.append(
            f"| {item['dataset']} | {item['backend']} | `{item['proof_hash']}` | "
            f"{str(item['proof_hash_audit_passed']).lower()} | "
            f"{str(item['verifier_reads_witness']).lower()} | {item['private_plain_value_count']} | "
            f"{item['proof_payload_audit_status']} | "
            f"{str(item['reference_consistency_passed']).lower()} | "
            f"{str(item['cpu_gpu_reference_consistency_passed']).lower()} | "
            f"{str(item['tamper_tests_passed']).lower()} | {item['tamper_tests_ms']:.3f} |"
        )
    security_lines.extend(
        [
            "",
            "结论：`all_passed=true`（由 `latest.json` 的 `security_tests` 字段记录）。这不是未执行 run 的占位结论。",
            "",
        ]
    )
    (output_dir / "security_tests_report.md").write_text("\n".join(security_lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail-closed aggregation of the five CPU and five CUDA formal runs")
    parser.add_argument("--output-dir", required=True, help="root results/paper_secure directory")
    parser.add_argument("--cpu-run", action="append", default=[], help="dataset=run_manifest.json_or_dir (exactly five)")
    parser.add_argument("--cuda-run", action="append", default=[], help="dataset=run_manifest.json_or_dir (exactly five)")
    args = parser.parse_args()

    project_root = pathlib.Path(__file__).resolve().parents[1]
    assignments = {"cpu": args.cpu_run, "cuda": args.cuda_run}
    rows: List[Dict[str, Any]] = []
    for backend, values in assignments.items():
        require(len(values) == len(DATASET_ORDER), f"exactly five --{backend}-run assignments are required")
        seen = set()
        for value in values:
            dataset, path_value = parse_assignment(value, f"--{backend}-run")
            require(dataset not in seen, f"duplicate {backend} dataset assignment: {dataset}")
            seen.add(dataset)
            rows.append(check_manifest(project_root, dataset, backend, path_value))
        require(seen == set(DATASET_ORDER), f"--{backend}-run must cover exactly {', '.join(DATASET_ORDER)}")

    require(len(rows) == 10, "exactly ten formal runs are required")
    ensure_pair_consistency(rows)
    ordered_rows = sorted(rows, key=lambda row: (DATASET_ORDER.index(row["dataset"]), row["backend"]))
    write_outputs(resolve_input_path(project_root, args.output_dir), ordered_rows)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(f"aggregate_secure_results: {exc}") from exc
