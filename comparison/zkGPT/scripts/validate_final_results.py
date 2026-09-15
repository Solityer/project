#!/usr/bin/env python3
"""Fail-closed validation of the selected zkGPT result matrix."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Set, Tuple

from common import (
    DATASETS,
    FORMAL_BACKENDS,
    PREFLIGHT_SCHEMA,
    RESOURCE_MANIFEST_SCHEMA,
    ChainError,
    atomic_write_json,
    baseline_root,
    build_binary,
    formal_backend_capability,
    load_json,
    require_manifest,
    sha256_file,
    validate_formal_cell,
    verifier_binary,
)


def selection(raw: str, allowed: Sequence[str], label: str) -> Tuple[str, ...]:
    if raw == "all":
        return tuple(allowed)
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not values or len(set(values)) != len(values) or any(item not in allowed for item in values):
        raise ChainError(f"{label} must be all or a comma-separated list of supported values")
    return values


def validate_cell(
    root: Path,
    cell: Mapping[str, Any],
    expected_keys: Optional[Set[Tuple[str, str]]] = None,
    preflight: Optional[Mapping[str, Any]] = None,
    resource_digest: Optional[str] = None,
    seen_artifact_paths: Optional[set] = None,
    seen_artifact_digests: Optional[set] = None,
) -> Dict[str, Any]:
    dataset = cell.get("dataset")
    backend = cell.get("backend")
    if not isinstance(dataset, str) or not isinstance(backend, str):
        raise ChainError("collected cell is missing dataset/backend")
    allowed = expected_keys if expected_keys is not None else {(d, b) for d in DATASETS for b in FORMAL_BACKENDS}
    if (dataset, backend) not in allowed:
        raise ChainError(f"unsupported collected cell identity: {dataset}/{backend}")
    if preflight is None:
        preflight_value = load_json(root / "results" / "formal_preflight.json", "formal preflight")
        if not isinstance(preflight_value, dict):
            raise ChainError("formal preflight is not an object")
        preflight = preflight_value
    if resource_digest is None:
        resource_digest = sha256_file(root / "results" / "resource_manifest.json")
    audited = validate_formal_cell(root, dataset, backend, preflight, resource_digest,
                                   seen_artifact_paths, seen_artifact_digests)
    # The collector record is itself an authenticated view.  Compare every
    # identity/path/stat/timing field that the audit returns; a caller cannot
    # edit the matrix JSON to point at a different live cell.
    for key, expected in audited.items():
        if cell.get(key) != expected:
            raise ChainError(f"collected cell field mismatch for {key}: {dataset}/{backend}")
    return audited


def _validate_preflight_binding(root: Path, preflight: Mapping[str, Any], resource_digest: str, backends: Sequence[str]) -> None:
    if preflight.get("schema") != PREFLIGHT_SCHEMA or preflight.get("status") != "READY" or preflight.get("formal_experiment_eligible") is not True:
        raise ChainError("formal preflight is not schema-v2 READY")
    if preflight.get("resource_manifest_sha256") != resource_digest:
        raise ChainError("formal preflight resource manifest is stale")
    capacities = preflight.get("capacity")
    if not isinstance(capacities, dict) or set(capacities) != set(DATASETS):
        raise ChainError("formal preflight capacity matrix is incomplete")
    if any(not isinstance(capacities[d], dict) or capacities[d].get("status") != "READY" for d in DATASETS):
        raise ChainError("formal preflight capacity is not READY for every dataset")
    for backend in backends:
        entry = preflight.get("backends", {}).get(backend)
        if not isinstance(entry, dict) or entry.get("ready") is not True:
            raise ChainError(f"formal preflight backend is not READY: {backend}")
        prover = build_binary(root, backend)
        verifier = verifier_binary(root, backend)
        current_prover = {"sha256": sha256_file(prover), "size_bytes": prover.stat().st_size, "mtime_ns": prover.stat().st_mtime_ns}
        current_verifier = {"sha256": sha256_file(verifier), "size_bytes": verifier.stat().st_size, "mtime_ns": verifier.stat().st_mtime_ns}
        if entry.get("prover_fingerprint") != current_prover or entry.get("verifier_fingerprint") != current_verifier:
            raise ChainError(f"formal preflight binary fingerprint is stale: {backend}")


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/collected_results.json")
    parser.add_argument("--backends", default="cpu")
    args = parser.parse_args(list(argv))
    root = baseline_root()
    try:
        backends = selection(args.backends, FORMAL_BACKENDS, "backends")
        require_manifest(root, "resource_manifest.json", RESOURCE_MANIFEST_SCHEMA)
        resource_digest = sha256_file(root / "results" / "resource_manifest.json")
        preflight = load_json(root / "results" / "formal_preflight.json", "formal preflight")
        if not isinstance(preflight, dict):
            raise ChainError("formal preflight is not an object")
        _validate_preflight_binding(root, preflight, resource_digest, backends)
        input_path = root / args.input
        # Keep input path baseline-local and reject aliases before opening it.
        from common import local_path
        input_path = local_path(root, args.input, "collected results")
        payload = load_json(input_path, "collected results")
        if not isinstance(payload, dict) or payload.get("schema") != "zkgpt-gat-collected-results-v2" or payload.get("status") != "COLLECTED":
            raise ChainError(f"unsupported collected-results schema: {input_path}")
        if payload.get("resource_manifest_sha256") != resource_digest or payload.get("preflight_sha256") != sha256_file(root / "results" / "formal_preflight.json"):
            raise ChainError("collected result provenance is stale")
        matrix = payload.get("matrix")
        expected_keys = {(dataset, backend) for dataset in DATASETS for backend in backends}
        if not isinstance(matrix, dict) or matrix.get("datasets") != list(DATASETS) or matrix.get("backends") != list(backends) or matrix.get("cell_count") != len(expected_keys):
            raise ChainError("collected matrix declaration is incomplete")
        cells = payload.get("cells")
        if not isinstance(cells, list) or len(cells) != len(expected_keys):
            raise ChainError("collected result does not contain the requested formal cells")
        seen_keys: Set[Tuple[str, str]] = set()
        seen_paths: set = set()
        seen_digests: set = set()
        for cell in cells:
            if not isinstance(cell, dict):
                raise ChainError("collected cell is not an object")
            key = (cell.get("dataset"), cell.get("backend"))
            if key in seen_keys:
                raise ChainError(f"duplicate collected cell: {key}")
            seen_keys.add(key)
            validate_cell(root, cell, expected_keys, preflight, resource_digest, seen_paths, seen_digests)
        if seen_keys != expected_keys:
            raise ChainError("collected matrix is missing a canonical dataset/backend cell")
        result = {
            "schema": "zkgpt-gat-final-validation-v2", "status": "VALIDATED", "baseline": "zkGPT",
            "matrix": matrix, "cells_validated": len(cells), "resource_scope": "baseline-local",
            "proof_scope": "full_gat", "privacy_profile": "masked_full", "independent_verifier": True,
            "no_fallback": True, "witness_isolated": True, "input": args.input,
            "resource_manifest_sha256": resource_digest,
            "preflight_sha256": sha256_file(root / "results" / "formal_preflight.json"),
        }
        atomic_write_json(root / "results" / "final_validation.json", result)
    except (ChainError, OSError, ValueError) as exc:
        print(f"FINAL_VALIDATION_FAILED: {exc}", file=sys.stderr)
        return 2
    print(f"FINAL_VALIDATION_OK cells={len(cells)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

