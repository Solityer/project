#!/usr/bin/env python3
"""Run strict native zkGPT GAT formal cells.

The native prover and the separate native verifier are the only components
allowed to produce/accept a proof.  This launcher adds freshness, resource,
identity and path checks around that chain; it never turns a partial artifact
or a metadata-only result into success.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from common import (
    DATASETS,
    CAPACITY_PROOF_PATH,
    EXECUTION_SCHEMA,
    PREFLIGHT_SCHEMA,
    PAPER_TIMING_SEMANTICS,
    REPORT_SCHEMA,
    ChainError,
    artifact_fields,
    assert_fresh_files,
    atomic_write_json,
    atomic_write_text,
    binary_fingerprint,
    build_binary,
    cleanup_run_outputs,
    config_with_auth,
    cross_check_identity,
    file_fingerprint,
    formal_backend_capability,
    FORMAL_BACKENDS,
    load_json,
    paper_metrics,
    run_logged,
    run_captured,
    local_path,
    derive_run_nonce,
    require_formal_session,
    runtime_source_sha256,
    relative_path,
    require_bool,
    require_circuit_id,
    require_fixed_digest,
    require_manifest,
    require_prover_secret,
    require_result_shape,
    require_string,
    require_timing,
    resource_identity,
    run_directory,
    sha256_file,
    validate_registration_artifact,
    verifier_binary,
    verify_build_provenance,
)


def require_run_file(root: Path, run_path: Path, relative: Any, description: str) -> Tuple[str, Path]:
    if not isinstance(relative, str) or not relative:
        raise ChainError(f"result has no {description} path")
    path = local_path(root, relative, description)
    if path.parent != run_path:
        raise ChainError(f"{description} is outside its formal run directory: {path}")
    if not path.is_file():
        raise ChainError(f"missing {description}: {path}")
    return relative, path


def _result_identity(result: Mapping[str, Any], path: Path) -> Dict[str, Any]:
    identity: Dict[str, Any] = {}
    for key in ("statement_digest", "plan_digest", "proof_digest",
                "registration_digest", "registration_binding_digest",
                "registration_layout_digest", "com_W_point"):
        identity[key] = require_fixed_digest(result.get(key), key, path)
    identity["circuit_id"] = require_circuit_id(result.get("circuit_id"), path)
    for key in ("full_dataset_identity", "checkpoint_identity", "cache_identity"):
        identity[key] = require_string(result, key, path)
    return identity


def _resource_result_identity(result: Mapping[str, Any], resource: Mapping[str, Any], path: Path) -> None:
    expected = {
        "full_dataset_identity": resource["full_dataset_identity"],
        "checkpoint_identity": resource["checkpoint_identity"],
        "cache_identity": resource["cache_identity"],
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise ChainError(f"{path} resource identity mismatch for {key}")
    for key in ("config_sha256", "quantization_sha256", "cache_manifest_sha256",
                "checkpoint_manifest_sha256", "checkpoint_manifest_entry_sha256"):
        if key in result:
            require_fixed_digest(result[key], key, path)
            expected_key = key
            if result[key] != resource[expected_key]:
                raise ChainError(f"{path} resource identity mismatch for {key}")


def _validate_report(
    report: Mapping[str, Any],
    result: Mapping[str, Any],
    dataset: str,
    backend: str,
    resource: Mapping[str, Any],
    paths: Mapping[str, str],
    artifact_digest: str,
    artifact_size: int,
    report_path: Path,
) -> None:
    if report.get("schema") != REPORT_SCHEMA:
        raise ChainError(f"unsupported independent verifier report schema: {report_path}")
    if report.get("status") != "VERIFY_OK" or report.get("independent") is not True or report.get("masked_full") is not True:
        raise ChainError(f"independent verifier did not return VERIFY_OK: {report_path}")
    if report.get("dataset") != dataset or report.get("backend") != backend:
        raise ChainError(f"independent verifier dataset/backend mismatch: {report_path}")
    require_bool(report, "verifier_reads_witness", False, report_path)
    require_bool(report, "verifier_reads_checkpoint", False, report_path)
    for key in ("statement_digest", "plan_digest", "proof_digest",
                "registration_digest", "registration_binding_digest",
                "registration_layout_digest", "com_W_point"):
        require_fixed_digest(report.get(key), key, report_path)
        if report[key] != result[key]:
            raise ChainError(f"independent verifier {key} mismatch: {report_path}")
    require_circuit_id(report.get("circuit_id"), report_path)
    if report["circuit_id"] != result["circuit_id"]:
        raise ChainError(f"independent verifier circuit_id mismatch: {report_path}")
    for key in ("full_dataset_identity", "checkpoint_identity", "cache_identity"):
        require_string(report, key, report_path)
        if report[key] != resource[key] or report[key] != result[key]:
            raise ChainError(f"independent verifier {key} mismatch: {report_path}")
    if report.get("artifact_path") != paths["artifact"] or report.get("verifier_context_path") != paths["context"] or report.get("public_logits_path") != paths["logits"]:
        raise ChainError(f"independent verifier input path mismatch: {report_path}")
    if report.get("artifact_sha256") != artifact_digest or report.get("artifact_size_bytes") != artifact_size:
        raise ChainError(f"independent verifier artifact identity mismatch: {report_path}")
    if report.get("proof_digest") != result["proof_digest"]:
        raise ChainError(f"independent verifier proof digest mismatch: {report_path}")


def ensure_registration(
    root: Path,
    dataset: str,
    backend: str,
    run_relative: str,
    run_path: Path,
    config: Mapping[str, str],
    session_id: str,
    session_manifest_sha256: str,
) -> Tuple[str, str, Dict[str, Any]]:
    """Reuse or create one identity-checked long-lived model registration."""

    registration_path = run_path / "model_registration.json"
    secret_path = run_path / "com_W.prover.secret"
    registration_exists = registration_path.exists()
    secret_exists = secret_path.exists()
    if registration_exists != secret_exists:
        raise ChainError(f"incomplete model registration pair for {dataset}/{backend}")
    if not registration_exists or load_json(registration_path).get("schema") != "zkgpt-gat-model-registration-v2":
        command = [sys.executable, str(root / "scripts" / "register_model.py"),
                   "--dataset", dataset]
        registered = run_captured(command, root)
        atomic_write_text(run_path / "runner-registration.stdout.log", registered.stdout, root)
        atomic_write_text(run_path / "runner-registration.stderr.log", registered.stderr, root)
        if registered.returncode != 0:
            raise ChainError(f"automatic model registration failed ({registered.returncode}); see {run_relative}/runner-registration.stderr.log")
    if not registration_path.is_file() or not secret_path.is_file():
        raise ChainError(f"automatic registration did not produce a complete pair: {run_path}")
    registration = validate_registration_artifact(
        root, dataset, backend, config, registration_path, secret_path
    )
    return (
        relative_path(root, registration_path, "model registration"),
        relative_path(root, secret_path, "prover secret"),
        registration,
    )


def _enrich_result(
    root: Path,
    result: Dict[str, Any],
    result_path: Path,
    resource: Mapping[str, Any],
    registration: Mapping[str, Any],
    artifact: Path,
    run_nonce: str,
    run_started_ns: int,
    resource_manifest_digest: str,
    session_id: str,
    session_manifest_sha256: str,
) -> Dict[str, Any]:
    artifact_fp = file_fingerprint(artifact, "proof artifact")
    result["artifact_sha256"] = artifact_fp["sha256"]
    result["artifact_size_bytes"] = artifact_fp["size_bytes"]
    result["artifact_mtime_ns"] = artifact_fp["mtime_ns"]
    if (result.get("session_id") != session_id or result.get("run_nonce") != run_nonce or
            result.get("session_manifest_sha256") != session_manifest_sha256):
        raise ChainError("native result did not preserve the current formal session binding")
    if result.get("paper_timing_semantics") != PAPER_TIMING_SEMANTICS:
        raise ChainError("native result uses stale paper timing semantics")
    result["run_started_unix_ns"] = run_started_ns
    result["resource_manifest_sha256"] = resource_manifest_digest
    for key in ("config_sha256", "quantization_sha256", "cache_manifest_sha256",
                "checkpoint_manifest_sha256", "checkpoint_manifest_entry_sha256"):
        result[key] = resource[key]
    result["registration_backend"] = result.get("registration_backend", "path-bound:" + result["backend"])
    result["registration_schema"] = registration["schema"]
    result["native_result_fingerprint"] = file_fingerprint(result_path, "native result")
    result["run_dir"] = str(result_path.parent.relative_to(root))
    result["registration_path"] = result.get("registration_path", str((result_path.parent / "model_registration.json").relative_to(root)))
    result["prover_secret_path"] = result.get("prover_secret_path", str((result_path.parent / "com_W.prover.secret").relative_to(root)))
    return result


def run_cell(root: Path, dataset: str, backend: str, preflight: Mapping[str, Any], resource_manifest_digest: str,
             session_id: str, session: Mapping[str, Any], session_manifest_sha256: str,
             preflight_path: Path) -> None:
    capability = formal_backend_capability(backend)
    if not capability["supported"] or backend not in FORMAL_BACKENDS:
        raise ChainError(capability["reason"])
    config_path, config = config_with_auth(root, dataset)
    if config.get("privacy_policy") != "model_and_trace_hidden" or config.get("privacy_profile", "masked_full") not in ("masked_full", ""):
        raise ChainError(f"{dataset} config does not declare the strict privacy boundary")
    resource = resource_identity(root, dataset)
    run_relative, run_path = run_directory(root, dataset, session_id, config)
    cleanup_run_outputs(root, run_path)
    run_started_ns = time.time_ns()
    run_nonce = derive_run_nonce(session_id, dataset)
    prover = build_binary(root, backend)
    verifier = verifier_binary(root, backend)
    prover_fp = binary_fingerprint(prover, "prover executable")
    verifier_fp = binary_fingerprint(verifier, "verifier executable")
    config_relative = relative_path(root, config_path, "dataset config")
    registration_relative, secret_relative, registration = ensure_registration(
        root, dataset, backend, run_relative, run_path, config, session_id,
        session_manifest_sha256
    )
    process_telemetry: Dict[str, Any] = {"prover": {}, "verifier": {}}
    prover_start = time.perf_counter()
    completed_returncode = run_logged([
        str(prover), "--dataset", dataset, "--backend", backend,
        "--baseline-root", str(root), "--config", config_relative,
        "--registration", registration_relative, "--com-w-secret", secret_relative,
        "--formal-session-id", session_id, "--run-nonce", run_nonce,
        "--session-manifest-sha256", session_manifest_sha256,
        "--formal-run-directory", run_relative,
    ], root, run_path / "prover.stdout.log", run_path / "prover.stderr.log",
       f"{dataset}/prover", process_telemetry["prover"])
    prover_wall_ms = (time.perf_counter() - prover_start) * 1000.0
    result_path = run_path / "result.json"
    if completed_returncode != 0:
        raise ChainError(f"native prover failed with {completed_returncode}; see {run_relative}/prover.stderr.log")
    result = require_result_shape(load_json(result_path, "native result"), result_path)
    if result.get("dataset") != dataset or result.get("backend") != backend or result.get("status") != "VERIFY_OK" or result.get("verification_status") != "VERIFY_OK":
        raise ChainError(f"native result is not a matching VERIFY_OK cell: {result_path}")
    if (result.get("session_id") != session_id or result.get("run_nonce") != run_nonce or
            result.get("session_manifest_sha256") != session_manifest_sha256):
        raise ChainError(f"native result does not belong to the current session: {result_path}")
    require_bool(result, "proof_generated", True, result_path)
    require_bool(result, "verifier_called", True, result_path)
    require_bool(result, "verifier_ok", True, result_path)
    require_bool(result, "verifier_reads_witness", False, result_path)
    require_bool(result, "cpu_fallback", False, result_path)
    identity = _result_identity(result, result_path)
    _resource_result_identity(result, resource, result_path)
    timing = require_timing(result, result_path)
    artifact, declared_artifact_digest = artifact_fields(root, result, run_path)
    artifact_fp = file_fingerprint(artifact, "proof artifact")
    if declared_artifact_digest and declared_artifact_digest != artifact_fp["sha256"]:
        raise ChainError(f"native artifact digest mismatch: {artifact}")
    context_rel, context_path = require_run_file(root, run_path, result.get("verifier_context_path"), "verifier context")
    logits_rel, logits_path = require_run_file(root, run_path, result.get("public_logits_path"), "public logits")
    statement_rel, statement_path = require_run_file(root, run_path, result.get("statement_path"), "public statement")
    assert_fresh_files((result_path, context_path, logits_path, statement_path, artifact), run_started_ns)
    report_rel = f"{run_relative}/verifier_report.json"
    verifier_start = time.perf_counter()
    verified_returncode = run_logged([str(verifier), "--baseline-root", str(root), "--verifier-context", context_rel, "--public-logits", logits_rel, "--artifact", relative_path(root, artifact, "proof artifact"), "--report", report_rel], root, run_path / "verifier.stdout.log", run_path / "verifier.stderr.log", f"{dataset}/{backend}/verifier", process_telemetry["verifier"])
    verifier_wall_ms = (time.perf_counter() - verifier_start) * 1000.0
    report_path = run_path / "verifier_report.json"
    if verified_returncode != 0:
        raise ChainError(f"independent verifier rejected the cell ({verified_returncode}); see {report_rel}")
    report = load_json(report_path, "independent verifier report")
    if not isinstance(report, dict):
        raise ChainError(f"independent verifier report is not an object: {report_path}")
    paths = {"artifact": relative_path(root, artifact, "proof artifact"),
             "context": context_rel, "logits": logits_rel, "statement": statement_rel}
    _validate_report(report, result, dataset, backend, resource, paths,
                     artifact_fp["sha256"], artifact_fp["size_bytes"], report_path)
    assert_fresh_files((report_path, run_path / "prover.stdout.log",
                        run_path / "prover.stderr.log", run_path / "verifier.stdout.log",
                        run_path / "verifier.stderr.log"), run_started_ns)
    result = _enrich_result(root, result, result_path, resource, registration,
                            artifact, run_nonce, run_started_ns, resource_manifest_digest,
                            session_id, session_manifest_sha256)
    atomic_write_json(result_path, result, root)
    execution = {
        "schema": EXECUTION_SCHEMA, "dataset": dataset, "backend": backend,
        "session_id": session_id, "session_manifest_sha256": session_manifest_sha256,
        "paper_timing_semantics": PAPER_TIMING_SEMANTICS,
        "preflight_sha256": sha256_file(preflight_path),
        "paper_metrics": paper_metrics(result, report, report_path), "process_telemetry": process_telemetry,
        "status": "VERIFY_OK", "no_fallback": True, "native_backend": result["native_backend"],
        "run_nonce": run_nonce, "run_started_unix_ns": run_started_ns,
        "result_path": relative_path(root, result_path, "native result"),
        "verifier_report_path": report_rel, "statement_path": statement_rel,
        "verifier_context_path": context_rel, "public_logits_path": logits_rel,
        "registration_path": registration_relative,
        "prover_secret_path": secret_relative,
        "artifact_path": paths["artifact"], "artifact_sha256": artifact_fp["sha256"],
        "artifact_size_bytes": artifact_fp["size_bytes"],
        "artifact_mtime_ns": artifact_fp["mtime_ns"],
        "statement_digest": identity["statement_digest"], "plan_digest": identity["plan_digest"],
        "circuit_id": identity["circuit_id"], "proof_digest": identity["proof_digest"],
        "registration_digest": identity["registration_digest"],
        "registration_binding_digest": identity["registration_binding_digest"],
        "registration_layout_digest": identity["registration_layout_digest"],
        "com_W_point": identity["com_W_point"],
        "full_dataset_identity": resource["full_dataset_identity"],
        "checkpoint_identity": resource["checkpoint_identity"], "cache_identity": resource["cache_identity"],
        "config_sha256": resource["config_sha256"], "quantization_sha256": resource["quantization_sha256"],
        "cache_manifest_sha256": resource["cache_manifest_sha256"],
        "checkpoint_manifest_sha256": resource["checkpoint_manifest_sha256"],
        "checkpoint_manifest_entry_sha256": resource["checkpoint_manifest_entry_sha256"],
        "resource_manifest_sha256": resource_manifest_digest,
        "prover_binary": {"path": relative_path(root, prover, "prover executable"), **prover_fp},
        "verifier_binary": {"path": relative_path(root, verifier, "verifier executable"), **verifier_fp},
        "timing_ms": timing,
        "prover_wall_ms": prover_wall_ms, "verifier_wall_ms": verifier_wall_ms,
        "verifier_reads_witness": False, "verifier_reads_checkpoint": False,
    }
    execution["file_fingerprints"] = {
        "result": file_fingerprint(result_path, "native result"),
        "statement": file_fingerprint(statement_path, "public statement"),
        "verifier_context": file_fingerprint(context_path, "verifier context"),
        "public_logits": file_fingerprint(logits_path, "public logits"),
        "proof_artifact": artifact_fp,
        "verifier_report": file_fingerprint(report_path, "verifier report"),
    }
    atomic_write_json(run_path / "execution.json", execution, root)
    print(f"VERIFY_OK dataset={dataset} backend={backend} result={run_relative}/result.json")


def validate_preflight_binding(root: Path, preflight: Mapping[str, Any], resource_manifest_digest: str,
                               backend: str, session_id: str, session: Mapping[str, Any],
                               session_manifest_sha256: str) -> None:
    if preflight.get("schema") != PREFLIGHT_SCHEMA or preflight.get("status") != "READY" or preflight.get("formal_experiment_eligible") is not True:
        raise ChainError("formal preflight is not eligible; no proving was started")
    if preflight.get("resource_manifest_sha256") != resource_manifest_digest:
        raise ChainError("formal preflight resource manifest is stale")
    if (preflight.get("session_id") != session_id or
            preflight.get("session_manifest_sha256") != session_manifest_sha256 or
            preflight.get("runtime_source_sha256") != session.get("runtime_source_sha256") or
            preflight.get("runtime_source_sha256") != runtime_source_sha256(root) or
            preflight.get("native_source_sha256") != session.get("native_source_sha256")):
        raise ChainError("formal preflight is not bound to the current session/runtime/native source")
    entry = preflight.get("backends", {}).get(backend)
    if not isinstance(entry, dict) or entry.get("ready") is not True:
        raise ChainError(f"formal preflight has no ready {backend} backend")
    if entry.get("source_tree_sha256") != verify_build_provenance(root, backend):
        raise ChainError("formal preflight source identity is stale")
    prover = build_binary(root, backend)
    verifier = verifier_binary(root, backend)
    if entry.get("prover_fingerprint", {}).get("sha256") != sha256_file(prover) or entry.get("verifier_fingerprint", {}).get("sha256") != sha256_file(verifier):
        raise ChainError("formal preflight binary fingerprint is stale")
    capacities = preflight.get("capacity", {})
    if set(capacities) != set(DATASETS) or any(not isinstance(capacities[d], dict) or (capacities[d].get("status") != "READY" or capacities[d].get("proof_path") != CAPACITY_PROOF_PATH or capacities[d].get("full_relation_bound") is not True) for d in DATASETS):
        raise ChainError("formal preflight capacity matrix is incomplete")


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(list(argv))
    print("Direct zkGPT cell execution is private to scripts/run_formal.py; "
          "use that local formal entrypoint.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
