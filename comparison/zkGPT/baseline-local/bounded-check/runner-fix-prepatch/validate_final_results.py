#!/usr/bin/env python3
"""Re-audit exactly five current-session results and independently verify each proof."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from common import (
    DATASETS,
    FORMAL_BACKENDS,
    FINAL_SCHEMA,
    COLLECTED_SCHEMA,
    PREFLIGHT_SCHEMA,
    RESOURCE_MANIFEST_SCHEMA,
    PAPER_TIMING_SEMANTICS,
    ChainError,
    assert_fresh_files,
    atomic_write_json,
    baseline_root,
    formal_collection_path,
    formal_final_path,
    formal_preflight_path,
    formal_session_status_path,
    invalidate_results,
    load_json,
    local_path,
    relative_path,
    require_manifest,
    require_regular,
    require_formal_session,
    run_logged,
    sha256_file,
    validate_formal_cell,
    verifier_binary,
    verify_build_provenance,
    build_binary,
    finite_nonnegative,
    verifier_timing,
    CAPACITY_PROOF_PATH,
)


def _validate_preflight(root: Path, preflight: Mapping[str, Any], resource_digest: str,
                        session_id: str, session_manifest_sha256: str,
                        runtime_source_sha256: str, native_source_sha256: str) -> None:
    if (preflight.get("schema") != PREFLIGHT_SCHEMA or preflight.get("status") != "READY" or
            preflight.get("formal_experiment_eligible") is not True):
        raise ChainError("formal preflight is not current schema-v3 READY")
    if (preflight.get("session_id") != session_id or
            preflight.get("session_manifest_sha256") != session_manifest_sha256 or
            preflight.get("runtime_source_sha256") != runtime_source_sha256 or
            preflight.get("native_source_sha256") != native_source_sha256 or
            preflight.get("resource_manifest_sha256") != resource_digest):
        raise ChainError("formal preflight session/runtime/native/resource binding is stale")
    capacities = preflight.get("capacity")
    if not isinstance(capacities, dict) or set(capacities) != set(DATASETS):
        raise ChainError("formal preflight capacity set is incomplete")
    for dataset in DATASETS:
        item = capacities[dataset]
        if not isinstance(item, dict) or item.get("status") != "READY" or item.get("proof_path") != CAPACITY_PROOF_PATH or item.get("full_relation_bound") is not True:
            raise ChainError(f"formal preflight capacity is not READY for {dataset}")
    for backend in FORMAL_BACKENDS:
        entry = preflight.get("backends", {}).get(backend)
        if not isinstance(entry, dict) or entry.get("ready") is not True:
            raise ChainError(f"formal preflight backend is not READY: {backend}")
        if entry.get("source_tree_sha256") != verify_build_provenance(root, backend):
            raise ChainError("formal preflight native source identity is stale")
        prover, verifier = build_binary(root, backend), verifier_binary(root, backend)
        for key, executable in (("prover_fingerprint", prover), ("verifier_fingerprint", verifier)):
            expected = {"sha256": sha256_file(executable), "size_bytes": executable.stat().st_size,
                        "mtime_ns": executable.stat().st_mtime_ns}
            if entry.get(key) != expected:
                raise ChainError(f"formal preflight {key} is stale")


def independently_verify_final_cell(root: Path, cell: Mapping[str, Any]) -> Dict[str, Any]:
    """Run the independent native verifier again; metadata cannot grant acceptance."""
    dataset, backend = cell["dataset"], cell["backend"]
    result_path = local_path(root, cell["result_path"], "final native result")
    run_path = result_path.parent
    report_path = run_path / "final_verifier_report.json"
    if report_path.exists():
        require_regular(report_path, "stale final verifier report").unlink()
    started_ns = time.time_ns()
    telemetry: Dict[str, Any] = {}
    command = [
        str(verifier_binary(root, backend)), "--baseline-root", str(root),
        "--verifier-context", cell["verifier_context_path"],
        "--public-logits", cell["public_logits_path"],
        "--artifact", cell["artifact_path"],
        "--report", relative_path(root, report_path, "final verifier report"),
    ]
    returncode = run_logged(command, root, run_path / "final-verifier.stdout.log",
                            run_path / "final-verifier.stderr.log",
                            f"final/{dataset}", telemetry)
    if returncode != 0:
        raise ChainError(f"final independent verifier rejected {dataset} (exit={returncode})")
    assert_fresh_files((report_path,), started_ns, "final verifier report")
    report = load_json(report_path, "final verifier report")
    original = load_json(local_path(root, cell["verifier_report_path"], "original verifier report"),
                         "original verifier report")
    if not isinstance(report, dict) or not isinstance(original, dict):
        raise ChainError("final independent verifier report is malformed")
    if (report.get("status") != "VERIFY_OK" or report.get("independent") is not True or
            report.get("masked_full") is not True or
            report.get("verifier_reads_witness") is not False or
            report.get("verifier_reads_checkpoint") is not False):
        raise ChainError("final independent verifier did not cryptographically accept")
    timing_fields = {"decode_crypto_ms", "verify_crypto_ms", "crypto_verify_ms"}
    for key, value in original.items():
        if key not in timing_fields and report.get(key) != value:
            raise ChainError(f"final verifier identity mismatch for {key}: {dataset}")
    verifier_timing(report, report_path)
    return {
        "dataset": dataset, "backend": backend, "status": "VERIFY_OK",
        "report_path": relative_path(root, report_path, "final verifier report"),
        "report_sha256": sha256_file(report_path), "process_telemetry": telemetry,
    }


def require_formal_cells(cells: Any) -> List[Mapping[str, Any]]:
    """Require one canonical, ordered cell for each formal dataset."""
    if not isinstance(cells, list) or len(cells) != len(DATASETS):
        raise ChainError("formal result must contain exactly five cells")
    for expected_dataset, cell in zip(DATASETS, cells):
        if not isinstance(cell, dict) or cell.get("dataset") != expected_dataset or cell.get("backend") != "cpu":
            raise ChainError(f"formal result cell order/identity mismatch for {expected_dataset}")
    return cells


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(list(argv))
    root = baseline_root()
    session_id = ""
    final_path: Path | None = None
    try:
        invalidate_results(root, ("final_validation.json",))
        session_id, session, session_manifest_sha256 = require_formal_session(root)
        final_path = formal_final_path(root, session_id)
        if final_path.exists():
            raise ChainError("current formal session already has a final result; refusing stale-result reuse")
        status_path = formal_session_status_path(root, session_id)
        session_status = load_json(status_path, "formal session status")
        if not isinstance(session_status, dict) or session_status.get("session_id") != session_id or session_status.get("status") != "RUNNING" or session_status.get("active_stage") != "final-validation":
            raise ChainError("final validation is not running as the current official session stage")
        require_manifest(root, "resource_manifest.json", RESOURCE_MANIFEST_SCHEMA)
        resource_digest = sha256_file(root / "results" / "resource_manifest.json")
        preflight_path = formal_preflight_path(root, session_id)
        preflight = load_json(preflight_path, "formal preflight")
        if not isinstance(preflight, dict):
            raise ChainError("formal preflight is not an object")
        _validate_preflight(root, preflight, resource_digest, session_id,
                            session_manifest_sha256, session["runtime_source_sha256"],
                            session["native_source_sha256"])

        collection_path = formal_collection_path(root, session_id)
        assert_fresh_files((collection_path,), int(session["started_unix_ns"]), "current formal collection")
        collection = load_json(collection_path, "current formal collection")
        if not isinstance(collection, dict) or collection.get("schema") != COLLECTED_SCHEMA or collection.get("status") != "COLLECTED":
            raise ChainError("current-session collection schema/status is invalid")
        if (collection.get("session_id") != session_id or
                collection.get("session_manifest_sha256") != session_manifest_sha256 or
                collection.get("paper_timing_semantics") != PAPER_TIMING_SEMANTICS or
                collection.get("runtime_source_sha256") != session["runtime_source_sha256"] or
                collection.get("resource_manifest_sha256") != resource_digest or
                collection.get("preflight") != str(preflight_path.relative_to(root)) or
                collection.get("preflight_sha256") != sha256_file(preflight_path)):
            raise ChainError("collection does not belong to the current session/preflight/runtime")
        matrix = collection.get("matrix")
        if (not isinstance(matrix, dict) or set(matrix) != {"datasets", "cell_count"} or
                matrix.get("datasets") != list(DATASETS) or matrix.get("cell_count") != len(DATASETS)):
            raise ChainError("collection must declare exactly five canonical datasets without a backend axis")
        cells = require_formal_cells(collection.get("cells"))
        audited_cells: List[Dict[str, Any]] = []
        final_verification: List[Dict[str, Any]] = []
        seen_paths: set = set()
        seen_digests: set = set()
        for expected_dataset, cell in zip(DATASETS, cells):
            audited = validate_formal_cell(
                root, expected_dataset, "cpu", preflight, resource_digest,
                session_id, session_manifest_sha256,
                session["native_source_sha256"], preflight_path,
                seen_paths, seen_digests,
            )
            if cell != audited:
                raise ChainError(f"collected row differs from current independent audit: {expected_dataset}")
            audited_cells.append(audited)
            final_verification.append(independently_verify_final_cell(root, audited))

        result = {
            "schema": FINAL_SCHEMA,
            "status": "VALIDATED",
            "baseline": "zkGPT",
            "session_id": session_id,
            "session_manifest_sha256": session_manifest_sha256,
            "paper_timing_semantics": PAPER_TIMING_SEMANTICS,
            "matrix": {"datasets": list(DATASETS), "cell_count": len(DATASETS)},
            "cells_validated": len(audited_cells),
            "cells": audited_cells,
            "resource_scope": "baseline-local",
            "proof_scope": "full_gat",
            "privacy_profile": "masked_full",
            "independent_verifier": True,
            "no_fallback": True,
            "witness_isolated": True,
            "collection_path": str(collection_path.relative_to(root)),
            "collection_sha256": sha256_file(collection_path),
            "resource_manifest_sha256": resource_digest,
            "preflight_path": str(preflight_path.relative_to(root)),
            "preflight_sha256": sha256_file(preflight_path),
            "runtime_source_sha256": session["runtime_source_sha256"],
            "final_native_verification": final_verification,
        }
        atomic_write_json(final_path, result, root)
    except (ChainError, OSError, ValueError) as exc:
        invalidate_results(root)
        if final_path is not None and final_path.exists():
            require_regular(final_path, "rejected final validation output").unlink()
        print(f"FINAL_VALIDATION_FAILED: {exc}", file=sys.stderr)
        return 2
    print(f"FINAL_VALIDATION_OK cells={len(audited_cells)} session={session_id} path=results/formal/{session_id}/final_validation.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
