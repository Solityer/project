#!/usr/bin/env python3
"""Own the lock, session, and bounded stage process groups for formal zkGPT."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from common import (
    ChainError,
    DATASETS,
    RESOURCE_MANIFEST_SCHEMA,
    atomic_write_json,
    atomic_write_text,
    baseline_root,
    binary_fingerprint,
    create_formal_session,
    ensure_directory,
    formal_final_path,
    formal_preflight_path,
    formal_results_directory,
    formal_run_lock,
    formal_session_path,
    formal_session_status_path,
    load_json,
    install_process_signal_cleanup,
    invalidate_results,
    local_path,
    paper_metrics,
    relative_path,
    run_logged,
    run_directory,
    runtime_source_sha256,
    sha256_file,
    source_tree_sha256,
    validate_formal_cell,
    verify_build_provenance,
    build_binary,
    verifier_binary,
    require_formal_session,
)

from collect_results import _validate_preflight
from gat_zkgpt_experiment import run_cell


ROOT_CAPACITY_PROFILE_SCHEMA = "comparison-baseline-capacity-profile-v2"
CELL_INTERNAL_LOG_RESERVE_BYTES = 16 * 1024 * 1024
FINALIZATION_INTERNAL_DISK_RESERVE_BYTES = 64 * 1024 * 1024
HOST_ACCOUNTING_OVERHEAD_BYTES = 64 * 1024 * 1024
SESSION_ID_RE = re.compile(r"^[0-9a-f]{32}$")


def _write_status(root: Path, session_id: str, status: str, active_stage: str,
                  completed: Sequence[str], stage_status: Dict[str, str],
                  started_unix_ns: int, failure: str = "") -> None:
    atomic_write_json(formal_session_status_path(root, session_id), {
        "schema": "zkgpt-gat-formal-session-v1",
        "session_id": session_id,
        "status": status,
        "active_stage": active_stage,
        "completed_stages": list(completed),
        "stage_status": dict(stage_status),
        "started_unix_ns": started_unix_ns,
        "updated_unix_ns": time.time_ns(),
        "failure": failure,
    }, root)


def _session_dir(root: Path, value: str, *, allow_empty: bool) -> tuple[str, Path]:
    if not value:
        raise ChainError("internal formal phase requires --session-dir")
    supplied = Path(value)
    if not supplied.is_absolute():
        raise ChainError("internal formal --session-dir must be absolute")
    session_id = supplied.name
    if not SESSION_ID_RE.fullmatch(session_id):
        raise ChainError("internal formal session directory must end in a 32-hex root session id")
    expected = root / "results" / "formal" / session_id
    if str(supplied) != str(expected):
        raise ChainError(f"internal formal session path is not the canonical baseline-local path: {supplied}")
    if supplied.is_symlink():
        raise ChainError("internal formal session directory must not be a symlink")
    if supplied.exists() and not supplied.is_dir():
        raise ChainError("internal formal session path is not a directory")
    if not allow_empty and not supplied.is_dir():
        raise ChainError("internal formal session has not been prepared")
    return session_id, supplied


def _set_status(root: Path, session_id: str, status: str, active_stage: str,
                failure: str = "") -> None:
    path = formal_session_status_path(root, session_id)
    current = load_json(path, "formal session status")
    if not isinstance(current, dict) or current.get("session_id") != session_id:
        raise ChainError("formal session status identity is invalid")
    current.update({
        "status": status,
        "active_stage": active_stage,
        "updated_unix_ns": time.time_ns(),
        "failure": failure,
    })
    atomic_write_json(path, current, root)


def _scratch_directory(root: Path, session_dir: Path) -> Path:
    scratch = (root / "artifacts" / "private" / "native_scratch").resolve()
    try:
        scratch.relative_to(root.resolve())
    except ValueError as exc:
        raise ChainError("native scratch path escapes the zkGPT baseline") from exc
    scratch.mkdir(parents=True, exist_ok=True)
    if scratch.is_symlink() or not scratch.is_dir():
        raise ChainError("native scratch path is aliased or not a directory")
    comparison_root = root.parent.resolve()
    expected_device = os.stat(comparison_root, follow_symlinks=False).st_dev
    if os.stat(scratch, follow_symlinks=False).st_dev != expected_device:
        raise ChainError("native scratch directory is not on the verified comparison results filesystem")
    if os.stat(session_dir.parent.parent, follow_symlinks=False).st_dev != expected_device:
        raise ChainError("zkGPT formal session is not on the verified comparison results filesystem")
    os.environ["ZKGPT_NATIVE_SCRATCH_DIR"] = str(scratch)
    return scratch


def _run_logged_stage(root: Path, command: Sequence[str], logs: Path,
                      name: str) -> tuple[Path, Path]:
    stdout_path = logs / f"{name}.stdout.log"
    stderr_path = logs / f"{name}.stderr.log"
    returncode = run_logged(command, root, stdout_path, stderr_path, name)
    if returncode != 0:
        raise ChainError(f"{name} exited with code {returncode}; see {stderr_path}")
    return stdout_path, stderr_path


def _tree_regular_bytes(path: Path) -> int:
    total = 0
    for current, directories, filenames in os.walk(path, followlinks=False):
        parent = Path(current)
        for name in directories:
            entry = parent / name
            if entry.is_symlink() or not entry.is_dir():
                raise ChainError(f"formal session contains an aliased/non-directory path: {entry}")
        for name in filenames:
            entry = parent / name
            info = entry.lstat()
            if entry.is_symlink() or not entry.is_file() or info.st_nlink != 1:
                raise ChainError(f"formal session contains an aliased/non-file path: {entry}")
            total += info.st_size
    return total


def _capacity_profile(session_id: str, session_dir: Path,
                      preflight: Mapping[str, Any]) -> Dict[str, Any]:
    capacities = preflight.get("capacity")
    if not isinstance(capacities, dict) or set(capacities) != set(DATASETS):
        raise ChainError("zkGPT capacity matrix does not contain the five canonical datasets")
    datasets: Dict[str, Any] = {}
    for dataset in DATASETS:
        item = capacities[dataset]
        if not isinstance(item, dict) or item.get("status") != "READY" or item.get("proof_path") != "formal_native_masked_lasso_hyrax+native_masked_layered_gkr" or item.get("full_relation_bound") is not True:
            raise ChainError(f"zkGPT full relation/capacity admission is not READY for {dataset}")
        scratch = item.get("estimated_native_scratch_bytes")
        disk = item.get("disk_requirement_bytes_upper_bound")
        retained = item.get("artifact_disk_bytes_upper_bound")
        host = item.get("estimated_peak_host_memory_bytes")
        wire = item.get("proof_artifact_wire_bytes_upper_bound")
        context = item.get("context_payload_bytes_bound")
        values = (scratch, disk, retained, host, wire, context)
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values):
            raise ChainError(f"zkGPT source-derived capacity fields are incomplete for {dataset}")
        if disk < scratch + retained:
            raise ChainError(f"zkGPT disk bound does not include scratch plus retained artifacts for {dataset}")
        # The capacity report covers the source-derived native live phases and
        # disk artifacts. The extra wire/context terms cover the exact-sized
        # in-memory canonical proof encoding and the bound verifier context.
        additional_disk = disk + CELL_INTERNAL_LOG_RESERVE_BYTES
        retained_upper = retained + CELL_INTERNAL_LOG_RESERVE_BYTES
        host_upper = host + wire + context + HOST_ACCOUNTING_OVERHEAD_BYTES
        datasets[dataset] = {
            "additional_disk_peak_bytes": additional_disk,
            "cell_retained_artifact_bytes_upper_bound": retained_upper,
            "host_peak_bytes": host_upper,
            "device_peak_bytes": 0,
            "phase_coverage": ["setup", "witness", "commitment", "prove", "verify"],
        }
    fs_device = os.stat(session_dir.parent.parent, follow_symlinks=False).st_dev
    return {
        "schema": ROOT_CAPACITY_PROFILE_SCHEMA,
        "baseline": "zkgpt",
        "session_id": session_id,
        "fs_device_id": str(fs_device),
        "preexisting_baseline_artifact_bytes": _tree_regular_bytes(session_dir),
        "finalization_disk_bytes_upper_bound": FINALIZATION_INTERNAL_DISK_RESERVE_BYTES,
        "memory_accounting_kind": "host-only",
        "device": None,
        "bound_methods": {
            "additional_disk_peak": "source_derived_upper_bound",
            "host_peak": "source_derived_upper_bound",
        },
        "datasets": datasets,
    }


def _load_prepared(root: Path, session_id: str,
                   session_dir: Path) -> tuple[Dict[str, Any], Dict[str, Any], str,
                                              Dict[str, Any], Path, str]:
    os.environ["ZKGPT_FORMAL_SESSION_ID"] = session_id
    session_id, session, session_manifest_sha256 = require_formal_session(root)
    context_path = session_dir / "coordinator_context.json"
    context = load_json(context_path, "coordinator session context")
    if not isinstance(context, dict) or context.get("schema") != "zkgpt-gat-coordinator-context-v1" or context.get("root_session_id") != session_id:
        raise ChainError("zkGPT coordinator context is missing or bound to another root session")
    if (context.get("session_manifest_sha256") != session_manifest_sha256 or
            context.get("runtime_source_sha256") != runtime_source_sha256(root) or
            context.get("native_source_sha256") != source_tree_sha256(root)):
        raise ChainError("zkGPT prepared source/session context is stale")
    preflight_path = formal_preflight_path(root, session_id)
    preflight = load_json(preflight_path, "formal preflight")
    if not isinstance(preflight, dict):
        raise ChainError("zkGPT formal preflight is malformed")
    resource_manifest = load_json(root / "results" / "resource_manifest.json", "resource manifest")
    if not isinstance(resource_manifest, dict) or resource_manifest.get("schema") != RESOURCE_MANIFEST_SCHEMA or resource_manifest.get("status") not in ("AUTHENTICATED", "READY"):
        raise ChainError("zkGPT resource manifest is not authenticated")
    resource_digest = sha256_file(root / "results" / "resource_manifest.json")
    if context.get("resource_manifest_sha256") != resource_digest or context.get("preflight_sha256") != sha256_file(preflight_path):
        raise ChainError("zkGPT prepared resource/preflight digest is stale")
    _validate_preflight(root, "cpu", preflight, resource_digest,
                        session_id, session_manifest_sha256,
                        session["native_source_sha256"])
    source_digest = verify_build_provenance(root, "cpu")
    if context.get("source_tree_sha256") != source_digest:
        raise ChainError("zkGPT build provenance no longer matches its prepare receipt")
    prover = build_binary(root, "cpu")
    verifier = verifier_binary(root, "cpu")
    if (context.get("prover_fingerprint") != binary_fingerprint(prover, "zkGPT prover") or
            context.get("verifier_fingerprint") != binary_fingerprint(verifier, "zkGPT verifier")):
        raise ChainError("zkGPT source-matched binary changed after prepare")
    build_log_value = context.get("build_log_path")
    if not isinstance(build_log_value, str):
        raise ChainError("zkGPT prepare context has no build log path")
    build_log = local_path(root, build_log_value, "formal build log")
    if context.get("build_log_sha256") != sha256_file(build_log):
        raise ChainError("zkGPT formal build log changed after prepare")
    _scratch_directory(root, session_dir)
    return session, preflight, resource_digest, context, preflight_path, session_manifest_sha256


def _prepare_phase(root: Path, session_id: str, session_dir: Path) -> Dict[str, Any]:
    if session_dir.exists() and any(session_dir.iterdir()):
        raise ChainError("zkGPT coordinator session directory must be fresh and empty")
    sid, session, session_manifest_sha256 = create_formal_session(root, session_id)
    os.environ["ZKGPT_FORMAL_SESSION_ID"] = sid
    _scratch_directory(root, session_dir)
    logs = session_dir / "logs"
    ensure_directory(logs, "formal session logs", root)
    _set_status(root, sid, "RUNNING", "resource-authentication")
    _run_logged_stage(root, [sys.executable, str(root / "scripts" / "prepare_resources.py"), "--check"], logs, "resource-authentication")

    _set_status(root, sid, "RUNNING", "native-build")
    _run_logged_stage(root, [str(root / "build.sh"), "--cpu"], logs, "native-build")
    build_stdout = logs / "native-build.stdout.log"
    build_stderr = logs / "native-build.stderr.log"
    build_log = logs / "build.log"
    atomic_write_text(build_log,
                      "=== build stdout ===\n" + build_stdout.read_text(encoding="utf-8", errors="replace") +
                      "\n=== build stderr ===\n" + build_stderr.read_text(encoding="utf-8", errors="replace"), root)
    build_source_digest = verify_build_provenance(root, "cpu")
    prover = build_binary(root, "cpu")
    verifier = verifier_binary(root, "cpu")
    build_fingerprints = {
        "prover": binary_fingerprint(prover, "zkGPT prover"),
        "verifier": binary_fingerprint(verifier, "zkGPT verifier"),
    }

    _set_status(root, sid, "RUNNING", "five-dataset-preflight")
    _run_logged_stage(root, [sys.executable, str(root / "scripts" / "formal_preflight.py")], logs, "five-dataset-preflight")
    preflight_path = formal_preflight_path(root, sid)
    preflight = load_json(preflight_path, "formal preflight")
    if not isinstance(preflight, dict) or preflight.get("status") != "READY" or preflight.get("formal_experiment_eligible") is not True:
        raise ChainError("zkGPT all-five native relation/shape/capacity admission did not pass")
    resource_manifest = load_json(root / "results" / "resource_manifest.json", "resource manifest")
    if not isinstance(resource_manifest, dict) or resource_manifest.get("schema") != RESOURCE_MANIFEST_SCHEMA or resource_manifest.get("status") not in ("AUTHENTICATED", "READY"):
        raise ChainError("zkGPT resource manifest is not authenticated")
    resource_digest = sha256_file(root / "results" / "resource_manifest.json")
    _validate_preflight(root, "cpu", preflight, resource_digest,
                        sid, session_manifest_sha256,
                        session["native_source_sha256"])
    context = {
        "schema": "zkgpt-gat-coordinator-context-v1",
        "status": "PREPARED",
        "root_session_id": sid,
        "session_manifest_sha256": session_manifest_sha256,
        "runtime_source_sha256": session["runtime_source_sha256"],
        "native_source_sha256": session["native_source_sha256"],
        "source_tree_sha256": build_source_digest,
        "resource_manifest_sha256": resource_digest,
        "preflight_sha256": sha256_file(preflight_path),
        "prover_fingerprint": build_fingerprints["prover"],
        "verifier_fingerprint": build_fingerprints["verifier"],
        "build_log_path": relative_path(root, build_log, "build log"),
        "build_log_sha256": sha256_file(build_log),
        "datasets": list(DATASETS),
        "created_unix_ns": time.time_ns(),
    }
    atomic_write_json(session_dir / "coordinator_context.json", context, root)
    profile = _capacity_profile(sid, session_dir, preflight)
    graph_datasets: Dict[str, Any] = {}
    capacity_matrix = preflight.get("capacity")
    if not isinstance(capacity_matrix, dict) or set(capacity_matrix) != set(DATASETS):
        raise ChainError("zkGPT graph identity source does not cover exactly five native capacity reports")
    for dataset in DATASETS:
        report = capacity_matrix[dataset]
        identity = report.get("canonical_graph_identity") if isinstance(report, dict) else None
        if not isinstance(identity, dict):
            raise ChainError(f"zkGPT capacity report has no actual graph identity for {dataset}")
        graph_datasets[dataset] = identity
    shared_graph_identity = {
        "schema": "comparison-canonical-graph-set-v1",
        "datasets": graph_datasets,
    }
    _set_status(root, sid, "RUNNING", "prepared")
    return {
        "phase": "prepare", "status": "PASS", "session_id": sid,
        "session_dir": str(session_dir),
        "build_log_path": context["build_log_path"],
        "build_log_sha256": context["build_log_sha256"],
        "shared_capacity_profile": profile,
        "shared_graph_identity": shared_graph_identity,
    }


def _cell_phase(root: Path, session_id: str, session_dir: Path,
                dataset: str) -> Dict[str, Any]:
    session, preflight, resource_digest, context, preflight_path, session_manifest_sha256 = _load_prepared(root, session_id, session_dir)
    receipts = session_dir / "receipts"
    ensure_directory(receipts, "formal cell receipts", root)
    receipt_path = receipts / f"{dataset}.json"
    if receipt_path.exists():
        raise ChainError(f"zkGPT cell receipt already exists; refusing to rerun {dataset}")
    run_relative, run_path = run_directory(root, dataset, session_id)
    if any(run_path.iterdir()):
        raise ChainError(f"zkGPT formal cell has prior partial or stale output; refusing to overwrite {dataset}")
    _set_status(root, session_id, "RUNNING", f"cell:{dataset}")
    run_cell(root, dataset, "cpu", preflight, resource_digest, session_id,
             session, session_manifest_sha256, preflight_path)
    audited = validate_formal_cell(root, dataset, "cpu", preflight, resource_digest,
                                   session_id, session_manifest_sha256,
                                   session["native_source_sha256"], preflight_path)
    result_path = root / audited["result_path"]
    execution_path = result_path.parent / "execution.json"
    result = load_json(result_path, "native result")
    execution = load_json(execution_path, "execution record")
    proof_path = root / audited["artifact_path"]
    verify_report_path = root / audited["verifier_report_path"]
    receipt = {
        "schema": "zkgpt-gat-cell-receipt-v1",
        "status": "VERIFY_OK",
        "baseline": "zkgpt",
        "dataset": dataset,
        "session_id": session_id,
        "session_manifest_sha256": session_manifest_sha256,
        "runtime_source_sha256": context["runtime_source_sha256"],
        "native_source_sha256": context["native_source_sha256"],
        "source_tree_sha256": context["source_tree_sha256"],
        "resource_manifest_sha256": resource_digest,
        "preflight_sha256": sha256_file(preflight_path),
        "result_path": audited["result_path"],
        "result_sha256": sha256_file(result_path),
        "execution_path": str(execution_path.relative_to(root)),
        "execution_sha256": sha256_file(execution_path),
        "proof_path": audited["artifact_path"],
        "proof_sha256": audited["artifact_sha256"],
        "proof_size_bytes": audited["proof_size_bytes"],
        "verification_report_path": audited["verifier_report_path"],
        "verification_report_sha256": sha256_file(verify_report_path),
        "independent_verification": True,
        "paper_metrics": audited["paper_metrics"],
        "timing_ms": audited["timing_ms"],
        "native_backend": result["native_backend"],
        "prover_binary": context["prover_fingerprint"],
        "verifier_binary": context["verifier_fingerprint"],
        "created_unix_ns": time.time_ns(),
    }
    atomic_write_json(receipt_path, receipt, root)
    _set_status(root, session_id, "RUNNING", f"cell-complete:{dataset}")
    report = load_json(verify_report_path, "independent verifier report")
    metrics = audited["paper_metrics"]
    return {
        "phase": "cell", "status": "PASS", "dataset": dataset,
        "session_id": session_id,
        "verification_status": "VERIFY_OK",
        "commitment_ms": metrics["commitment_time_ms"],
        "prove_ms": metrics["prove_time_ms"],
        "verify_ms": metrics["verify_time_ms"],
        "proof_size_bytes": audited["proof_size_bytes"],
        "receipt_path": str(receipt_path.relative_to(root)),
        "receipt_sha256": sha256_file(receipt_path),
        "proof_path": str(proof_path.relative_to(root)),
        "proof_sha256": sha256_file(proof_path),
        "independent_verification": report.get("independent") is True,
        "verification_report_path": str(verify_report_path.relative_to(root)),
        "verification_report_sha256": sha256_file(verify_report_path),
    }


def _finalize_phase(root: Path, session_id: str,
                    session_dir: Path) -> Dict[str, Any]:
    session, preflight, resource_digest, context, preflight_path, session_manifest_sha256 = _load_prepared(root, session_id, session_dir)
    receipts = session_dir / "receipts"
    for dataset in DATASETS:
        receipt_path = receipts / f"{dataset}.json"
        receipt = load_json(receipt_path, f"zkGPT {dataset} cell receipt")
        if not isinstance(receipt, dict) or receipt.get("status") != "VERIFY_OK" or receipt.get("session_id") != session_id or receipt.get("dataset") != dataset:
            raise ChainError(f"zkGPT verified cell receipt is missing or stale: {dataset}")
        if receipt.get("proof_sha256") != sha256_file(root / receipt["proof_path"]):
            raise ChainError(f"zkGPT proof artifact changed after cell receipt: {dataset}")
        if receipt.get("verification_report_sha256") != sha256_file(root / receipt["verification_report_path"]):
            raise ChainError(f"zkGPT timed independent-verifier report changed after receipt: {dataset}")
    timed_reports = {
        dataset: sha256_file(root / load_json(receipts / f"{dataset}.json", "cell receipt")["verification_report_path"])
        for dataset in DATASETS
    }
    _set_status(root, session_id, "RUNNING", "collection")
    logs = session_dir / "logs"
    _run_logged_stage(root, [sys.executable, str(root / "scripts" / "collect_results.py")], logs, "collection")
    _set_status(root, session_id, "RUNNING", "final-validation")
    _run_logged_stage(root, [sys.executable, str(root / "scripts" / "validate_final_results.py")], logs, "final-validation")
    for dataset, digest in timed_reports.items():
        receipt = load_json(receipts / f"{dataset}.json", "cell receipt")
        report_path = root / receipt["verification_report_path"]
        if sha256_file(report_path) != digest:
            raise ChainError(f"final independent verifier overwrote the timed cell report: {dataset}")
        final_report_path = report_path.with_name("final_verifier_report.json")
        if not final_report_path.is_file():
            raise ChainError(f"final independent verifier report is missing: {dataset}")
    final_path = formal_final_path(root, session_id)
    final = load_json(final_path, "final validation result")
    if not isinstance(final, dict) or final.get("status") != "VALIDATED" or final.get("cells_validated") != len(DATASETS) or final.get("session_id") != session_id:
        raise ChainError("zkGPT final native validator did not validate exactly five current-session cells")
    _set_status(root, session_id, "COMPLETE", "")
    return {
        "phase": "finalize", "status": "PASS", "session_id": session_id,
        "final_validation_path": str(final_path.relative_to(root)),
        "final_validation_sha256": sha256_file(final_path),
        "validated_cells": list(DATASETS),
        "independent_verification": True,
        "paper_timing_semantics": final.get("paper_timing_semantics"),
    }


def _phase_marker(payload: Mapping[str, Any]) -> None:
    print("ZKGPT_COORDINATOR_STATE=" + json.dumps(payload, sort_keys=True, separators=(",", ":")), flush=True)


def _run_internal_phase(root: Path, phase: str, session_dir_arg: str,
                        dataset: str | None) -> int:
    session_id, session_dir = _session_dir(root, session_dir_arg,
                                            allow_empty=(phase == "prepare"))
    if phase == "cell" and dataset not in DATASETS:
        raise ChainError("cell phase requires one canonical --dataset")
    if phase != "cell" and dataset is not None:
        raise ChainError("--dataset is only valid for a cell phase")
    with formal_run_lock(root):
        install_process_signal_cleanup()
        if phase == "prepare":
            payload = _prepare_phase(root, session_id, session_dir)
        elif phase == "cell":
            payload = _cell_phase(root, session_id, session_dir, dataset or "")
        elif phase == "finalize":
            payload = _finalize_phase(root, session_id, session_dir)
        else:
            raise ChainError(f"unsupported internal phase: {phase}")
    _phase_marker(payload)
    return 0


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--internal-phase", choices=("prepare", "cell", "finalize"),
                        help=argparse.SUPPRESS)
    parser.add_argument("--session-dir", help=argparse.SUPPRESS)
    parser.add_argument("--dataset", choices=DATASETS, help=argparse.SUPPRESS)
    args = parser.parse_args(list(argv))
    root = baseline_root()
    if args.internal_phase is None:
        print("This baseline is coordinated only by /home/pzh/project/comparison/run_formal.sh", file=sys.stderr)
        return 2
    try:
        if not args.session_dir:
            raise ChainError("internal formal phase requires --session-dir")
        return _run_internal_phase(root, args.internal_phase, args.session_dir, args.dataset)
    except KeyboardInterrupt:
        print("ZKGPT_INTERNAL_PHASE_INTERRUPTED: owned stage group was stopped", file=sys.stderr, flush=True)
        return 130
    except Exception as exc:
        try:
            if args.session_dir:
                failed_id = Path(args.session_dir).name
                if SESSION_ID_RE.fullmatch(failed_id) and formal_session_status_path(root, failed_id).is_file():
                    _set_status(root, failed_id, "FAILED", args.internal_phase or "", str(exc))
        except Exception as status_exc:
            print(f"ZKGPT_SESSION_STATUS_WRITE_FAILED: {status_exc}", file=sys.stderr, flush=True)
        print(f"ZKGPT_INTERNAL_PHASE_FAILED phase={args.internal_phase}: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
