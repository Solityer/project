#!/usr/bin/env python3
"""Collect only fresh, independently verified native zkGPT cells."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

from common import (
    DATASETS,
    canonical_dataset_scope,
    CAPACITY_PROOF_PATH,
    PREFLIGHT_SCHEMA,
    RESOURCE_MANIFEST_SCHEMA,
    ChainError,
    atomic_write_json,
    baseline_root,
    build_binary,
    load_json,
    invalidate_results,
    require_manifest,
    sha256_file,
    validate_formal_cell,
    verifier_binary,
    verify_build_provenance,
    formal_preflight_path,
    formal_collection_path,
    require_formal_session,
    runtime_source_sha256,
    PAPER_TIMING_SEMANTICS,
    COLLECTED_SCHEMA,
)


def _validate_preflight(root: Path, backend: str, preflight: Dict[str, Any], resource_digest: str,
                        session_id: str, session_manifest_sha256: str,
                        native_source_sha256: str, *, allow_partial: bool = False) -> None:
    status = preflight.get("status")
    if preflight.get("schema") != PREFLIGHT_SCHEMA or status not in (("READY", "PARTIAL") if allow_partial else ("READY",)):
        raise ChainError("formal preflight has an unsupported schema/status")
    if (preflight.get("shared_preconditions_ready") is not True or
            preflight.get("source_ready") is not True or preflight.get("resources_ready") is not True or
            preflight.get("blockers") != [] or
            preflight.get("formal_experiment_eligible") is not (status == "READY")):
        raise ChainError("formal preflight shared source/resource/backend/session preconditions are not ready")
    if preflight.get("static_checks_passed") is not (status == "READY"):
        raise ChainError("formal preflight static-check status is inconsistent")
    if preflight.get("resource_manifest_sha256") != resource_digest:
        raise ChainError("formal preflight resource manifest is stale")
    if (preflight.get("session_id") != session_id or
            preflight.get("session_manifest_sha256") != session_manifest_sha256 or
            preflight.get("runtime_source_sha256") != runtime_source_sha256(root) or
            preflight.get("native_source_sha256") != native_source_sha256):
        raise ChainError("formal preflight session/runtime/native source binding is stale")
    try:
        dataset_scope = canonical_dataset_scope(preflight.get("dataset_scope"))
        _scope_session_id, session, _scope_manifest_sha256 = require_formal_session(root)
    except ChainError as exc:
        raise ChainError("formal preflight dataset scope is invalid") from exc
    if canonical_dataset_scope(session.get("dataset_scope")) != dataset_scope:
        raise ChainError("formal preflight dataset scope differs from its immutable session")
    capacities = preflight.get("capacity")
    if not isinstance(capacities, dict) or set(capacities) != set(DATASETS):
        raise ChainError("formal preflight capacity matrix is incomplete")
    ready_count = 0
    for dataset in DATASETS:
        item = capacities[dataset]
        if not isinstance(item, dict) or item.get("dataset") != dataset or item.get("backend") != "cpu":
            raise ChainError(f"formal preflight capacity identity is invalid for {dataset}")
        if dataset not in dataset_scope:
            if (item.get("status") != "BLOCKED" or item.get("stage") != "scope" or
                    item.get("reason") != "OUT_OF_SCOPE:dataset_not_requested"):
                raise ChainError(f"formal preflight out-of-scope capacity marker is invalid for {dataset}")
            continue
        if item.get("status") == "READY":
            if item.get("proof_path") != CAPACITY_PROOF_PATH or item.get("full_relation_bound") is not True or item.get("reason") != "":
                raise ChainError(f"formal preflight capacity is not a complete full-relation admission for {dataset}")
            ready_count += 1
        elif item.get("status") == "BLOCKED":
            if not allow_partial or not isinstance(item.get("reason"), str) or not item["reason"]:
                raise ChainError(f"formal preflight capacity is blocked for {dataset}")
        else:
            raise ChainError(f"formal preflight capacity status is invalid for {dataset}")
    all_ready = ready_count == len(DATASETS)
    if (status == "READY") != all_ready or (status == "PARTIAL") == all_ready:
        raise ChainError("formal preflight status does not match its per-dataset capacity matrix")
    entry = preflight.get("backends", {}).get(backend)
    if not isinstance(entry, dict) or entry.get("ready") is not True:
        raise ChainError(f"formal preflight backend is not READY: {backend}")
    if entry.get("source_tree_sha256") != verify_build_provenance(root, backend):
        raise ChainError("formal preflight source identity is stale")
    prover = build_binary(root, backend)
    verifier = verifier_binary(root, backend)
    if entry.get("prover_fingerprint") != {"sha256": sha256_file(prover), "size_bytes": prover.stat().st_size, "mtime_ns": prover.stat().st_mtime_ns}:
        raise ChainError("formal preflight prover fingerprint is stale")
    if entry.get("verifier_fingerprint") != {"sha256": sha256_file(verifier), "size_bytes": verifier.stat().st_size, "mtime_ns": verifier.stat().st_mtime_ns}:
        raise ChainError("formal preflight verifier fingerprint is stale")


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(list(argv))
    root = baseline_root()
    try:
        invalidate_results(root)
        session_id, session, session_manifest_sha256 = require_formal_session(root)
        backend = "cpu"
        resource_manifest = require_manifest(root, "resource_manifest.json", RESOURCE_MANIFEST_SCHEMA)
        del resource_manifest
        resource_manifest_path = root / "results" / "resource_manifest.json"
        resource_digest = sha256_file(resource_manifest_path)
        preflight_path = formal_preflight_path(root, session_id)
        preflight = load_json(preflight_path, "formal preflight")
        if not isinstance(preflight, dict):
            raise ChainError("formal preflight is not an object")
        _validate_preflight(root, backend, preflight, resource_digest,
                            session_id, session_manifest_sha256,
                            session["native_source_sha256"])
        if session.get("runtime_source_sha256") != runtime_source_sha256(root):
            raise ChainError("formal session runtime source fingerprint is stale")
    except (ChainError, OSError, ValueError) as exc:
        print(f"COLLECT_FAILED: {exc}", file=sys.stderr)
        return 2
    cells: List[Dict[str, Any]] = []
    seen_paths: set = set()
    seen_digests: set = set()
    try:
        for dataset in DATASETS:
            cells.append(validate_formal_cell(
                root, dataset, backend, preflight, resource_digest,
                session_id, session_manifest_sha256,
                session["native_source_sha256"], preflight_path,
                seen_paths, seen_digests,
            ))
    except (ChainError, OSError, ValueError) as exc:
        print(f"COLLECT_FAILED: {exc}", file=sys.stderr)
        return 2
    payload = {
        "schema": COLLECTED_SCHEMA, "status": "COLLECTED", "baseline": "zkGPT",
        "session_id": session_id, "session_manifest_sha256": session_manifest_sha256,
        "paper_timing_semantics": PAPER_TIMING_SEMANTICS,
        "matrix": {"datasets": list(DATASETS), "cell_count": len(cells)},
        "cells": cells, "resource_manifest": "results/resource_manifest.json",
        "preflight": str(preflight_path.relative_to(root)), "resource_manifest_sha256": resource_digest,
        "preflight_sha256": sha256_file(preflight_path),
        "runtime_source_sha256": session["runtime_source_sha256"],
    }
    try:
        collection_path = formal_collection_path(root, session_id)
        if collection_path.exists():
            raise ChainError("current formal session already has a collection; refusing to reuse or overwrite it")
        atomic_write_json(collection_path, payload, root)
    except ChainError as exc:
        print(f"COLLECT_OUTPUT_FAILED: {exc}", file=sys.stderr)
        return 2
    print(f"COLLECTED {len(cells)} strict cells session={session_id} path=results/formal/{session_id}/collected.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
