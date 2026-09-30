#!/usr/bin/env python3
"""Collect only fresh, independently verified native zkGPT cells."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from common import (
    DATASETS,
    FORMAL_BACKENDS,
    PREFLIGHT_SCHEMA,
    RESOURCE_MANIFEST_SCHEMA,
    ChainError,
    atomic_write_json,
    baseline_root,
    binary_fingerprint,
    build_binary,
    formal_backend_capability,
    load_json,
    require_fixed_digest,
    require_manifest,
    resource_identity,
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


def _validate_preflight(root: Path, backend: str, preflight: Dict[str, Any], resource_digest: str) -> None:
    if preflight.get("schema") != PREFLIGHT_SCHEMA or preflight.get("status") != "READY" or preflight.get("formal_experiment_eligible") is not True:
        raise ChainError("formal preflight is not an eligible schema-v2 READY record")
    if preflight.get("resource_manifest_sha256") != resource_digest:
        raise ChainError("formal preflight resource manifest is stale")
    capacities = preflight.get("capacity")
    if not isinstance(capacities, dict) or set(capacities) != set(DATASETS):
        raise ChainError("formal preflight capacity matrix is incomplete")
    for dataset in DATASETS:
        if not isinstance(capacities[dataset], dict) or capacities[dataset].get("status") != "READY":
            raise ChainError(f"formal preflight capacity is not READY for {dataset}")
    entry = preflight.get("backends", {}).get(backend)
    if not isinstance(entry, dict) or entry.get("ready") is not True:
        raise ChainError(f"formal preflight backend is not READY: {backend}")
    prover = build_binary(root, backend)
    verifier = verifier_binary(root, backend)
    if entry.get("prover_fingerprint") != {"sha256": sha256_file(prover), "size_bytes": prover.stat().st_size, "mtime_ns": prover.stat().st_mtime_ns}:
        raise ChainError("formal preflight prover fingerprint is stale")
    if entry.get("verifier_fingerprint") != {"sha256": sha256_file(verifier), "size_bytes": verifier.stat().st_size, "mtime_ns": verifier.stat().st_mtime_ns}:
        raise ChainError("formal preflight verifier fingerprint is stale")


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", default="cpu")
    args = parser.parse_args(list(argv))
    root = baseline_root()
    try:
        backends = selection(args.backends, FORMAL_BACKENDS, "backends")
        resource_manifest = require_manifest(root, "resource_manifest.json", RESOURCE_MANIFEST_SCHEMA)
        del resource_manifest
        resource_manifest_path = root / "results" / "resource_manifest.json"
        resource_digest = sha256_file(resource_manifest_path)
        preflight_path = root / "results" / "formal_preflight.json"
        preflight = load_json(preflight_path, "formal preflight")
        if not isinstance(preflight, dict):
            raise ChainError("formal preflight is not an object")
        for backend in backends:
            _validate_preflight(root, backend, preflight, resource_digest)
    except (ChainError, OSError, ValueError) as exc:
        print(f"COLLECT_FAILED: {exc}", file=sys.stderr)
        return 2
    cells: List[Dict[str, Any]] = []
    seen_paths: set = set()
    seen_digests: set = set()
    try:
        for dataset in DATASETS:
            for backend in backends:
                cells.append(validate_formal_cell(root, dataset, backend, preflight,
                                                  resource_digest, seen_paths, seen_digests))
    except (ChainError, OSError, ValueError) as exc:
        print(f"COLLECT_FAILED: {exc}", file=sys.stderr)
        return 2
    payload = {
        "schema": "zkgpt-gat-collected-results-v2", "status": "COLLECTED", "baseline": "zkGPT",
        "matrix": {"datasets": list(DATASETS), "backends": list(backends), "cell_count": len(cells)},
        "cells": cells, "resource_manifest": "results/resource_manifest.json",
        "preflight": "results/formal_preflight.json", "resource_manifest_sha256": resource_digest,
        "preflight_sha256": sha256_file(root / "results" / "formal_preflight.json"),
    }
    try:
        atomic_write_json(root / "results" / "collected_results.json", payload)
    except ChainError as exc:
        print(f"COLLECT_OUTPUT_FAILED: {exc}", file=sys.stderr)
        return 2
    print(f"COLLECTED {len(cells)} strict cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
