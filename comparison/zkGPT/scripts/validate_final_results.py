#!/usr/bin/env python3
"""Fail-closed final validator for the selected zkGPT GAT result matrix."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from common import (
    BACKENDS,
    DATASETS,
    ChainError,
    atomic_write_json,
    baseline_root,
    formal_backend_capability,
    load_json,
    local_path,
    require_result_shape,
    sha256_file,
    verify_local_resources,
)


EXPECTED_KEYS = {(dataset, backend) for dataset in DATASETS for backend in BACKENDS}


def selection(raw: str, allowed: Sequence[str], label: str) -> Tuple[str, ...]:
    if raw == "all":
        return tuple(allowed)
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not values or len(set(values)) != len(values) or any(item not in allowed for item in values):
        raise ChainError(f"{label} must be all or a comma-separated list of supported values")
    return values


def require_text(value: Mapping[str, Any], key: str, where: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise ChainError(f"{where} is missing non-empty {key}")
    return item


def require_false(value: Mapping[str, Any], key: str, where: str) -> None:
    if value.get(key) is not False:
        raise ChainError(f"{where} requires {key}=false")


def require_true(value: Mapping[str, Any], key: str, where: str) -> None:
    if value.get(key) is not True:
        raise ChainError(f"{where} requires {key}=true")


def validate_cell(root: Path, cell: Mapping[str, Any], expected_keys: Optional[set] = None) -> None:
    dataset = require_text(cell, "dataset", "collected cell")
    backend = require_text(cell, "backend", "collected cell")
    allowed_keys = EXPECTED_KEYS if expected_keys is None else expected_keys
    if (dataset, backend) not in allowed_keys:
        raise ChainError(f"unsupported collected cell identity: {dataset}/{backend}")
    artifact_rel = require_text(cell, "artifact_path", f"{dataset}/{backend}")
    artifact = local_path(root, artifact_rel, "collected artifact")
    if not artifact.is_file():
        raise ChainError(f"missing collected artifact: {artifact}")
    digest = require_text(cell, "artifact_sha256", f"{dataset}/{backend}").lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ChainError(f"malformed artifact digest: {artifact}")
    actual = sha256_file(artifact)
    if actual != digest:
        raise ChainError(f"collected artifact digest mismatch: {artifact}")
    proof_size = cell.get("proof_size_bytes")
    if isinstance(proof_size, bool) or not isinstance(proof_size, int) or proof_size <= 0 or proof_size != artifact.stat().st_size:
        raise ChainError(f"collected artifact size mismatch: {artifact}")
    for key in ("statement_digest", "proof_digest", "circuit_id", "plan_digest", "full_dataset_identity", "checkpoint_identity", "cache_identity", "com_W_point"):
        require_text(cell, key, f"{dataset}/{backend}")
    require_true(cell, "no_fallback", f"{dataset}/{backend}")
    if cell.get("witness_isolated") is not True:
        raise ChainError(f"{dataset}/{backend} does not declare witness isolation")
    timing = cell.get("timing_ms")
    if not isinstance(timing, dict):
        raise ChainError(f"{dataset}/{backend} is missing timing_ms")
    for key in ("load", "registration", "witness", "prove", "verify", "serialize"):
        item = timing.get(key)
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)) or float(item) < 0:
            raise ChainError(f"{dataset}/{backend} has invalid timing_ms.{key}")
    wall = cell.get("wall_ms")
    if not isinstance(wall, dict):
        raise ChainError(f"{dataset}/{backend} is missing wall_ms")
    for key in ("prover_wall_ms", "verifier_wall_ms"):
        item = wall.get(key)
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)) or float(item) < 0:
            raise ChainError(f"{dataset}/{backend} has invalid wall_ms.{key}")

    result_rel = require_text(cell, "result_path", f"{dataset}/{backend}")
    result_path = local_path(root, result_rel, "collected result")
    result = require_result_shape(load_json(result_path, "native result"), result_path)
    if result.get("dataset") != dataset or result.get("backend") != backend or result.get("status") != "VERIFY_OK" or result.get("verification_status") != "VERIFY_OK":
        raise ChainError(f"native result is not a matching VERIFY_OK cell: {result_path}")
    if result.get("proof_size_bytes") != proof_size or result.get("proof_digest") != cell.get("proof_digest"):
        raise ChainError(f"native result proof identity mismatch: {result_path}")
    if result.get("artifact_sha256") and result.get("artifact_sha256") != digest:
        raise ChainError(f"native result artifact digest mismatch: {result_path}")
    require_false(result, "cpu_fallback", str(result_path))
    require_false(result, "verifier_reads_witness", str(result_path))
    if result.get("proof_scope") != "full_gat" or result.get("privacy_profile") != "masked_full":
        raise ChainError(f"native result scope mismatch: {result_path}")

    report_rel = require_text(cell, "verifier_report_path", f"{dataset}/{backend}")
    report_path = local_path(root, report_rel, "verifier report")
    report = load_json(report_path, "independent verifier report")
    if not isinstance(report, dict) or report.get("schema") != "zkgpt-gat-independent-verifier-v2" or report.get("status") != "VERIFY_OK" or report.get("independent") is not True or report.get("masked_full") is not True:
        raise ChainError(f"independent verifier is not a matching VERIFY_OK report: {report_path}")
    if report.get("dataset") != dataset or report.get("backend") != backend or report.get("statement_digest") != cell.get("statement_digest"):
        raise ChainError(f"independent verifier identity mismatch: {report_path}")
    require_false(report, "verifier_reads_witness", str(report_path))
    require_false(report, "verifier_reads_checkpoint", str(report_path))

    execution_rel = f"{Path(result_rel).parent}/execution.json"
    execution_path = local_path(root, execution_rel, "execution record")
    execution = load_json(execution_path, "execution timing")
    if not isinstance(execution, dict) or execution.get("schema") != "zkgpt-gat-execution-v1" or execution.get("status") != "VERIFY_OK":
        raise ChainError(f"execution record is not VERIFY_OK: {execution_path}")
    require_true(execution, "no_fallback", str(execution_path))


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/collected_results.json", help="baseline-relative collected result JSON")
    parser.add_argument(
        "--backends", default="cpu",
        help="formal backends represented in the collected matrix (default: cpu; gpu is unsupported)",
    )
    args = parser.parse_args(list(argv))
    root = baseline_root()
    try:
        selected_backends = selection(args.backends, BACKENDS, "backends")
        expected_keys = {(dataset, backend) for dataset in DATASETS for backend in selected_backends}
        for backend in selected_backends:
            capability = formal_backend_capability(backend)
            if not capability["supported"]:
                raise ChainError(capability["reason"])
        resource_errors = verify_local_resources(root)
        if resource_errors:
            raise ChainError("resource authentication failed: " + " | ".join(resource_errors))
        provenance = root / "UPSTREAM_PROVENANCE.md"
        if not provenance.is_file():
            raise ChainError(f"missing provenance document: {provenance}")
        input_path = local_path(root, args.input, "collected results")
        payload = load_json(input_path, "collected results")
        if not isinstance(payload, dict) or payload.get("schema") != "zkgpt-gat-collected-results-v1" or payload.get("status") != "COLLECTED":
            raise ChainError(f"unsupported collected-results schema: {input_path}")
        matrix = payload.get("matrix")
        if not isinstance(matrix, dict) or matrix.get("datasets") != list(DATASETS) or matrix.get("backends") != list(selected_backends) or matrix.get("cell_count") != len(expected_keys):
            raise ChainError("collected matrix declaration is incomplete")
        cells = payload.get("cells")
        if not isinstance(cells, list) or len(cells) != len(expected_keys):
            raise ChainError("collected result does not contain the requested formal cells")
        seen = set()
        for cell in cells:
            if not isinstance(cell, dict):
                raise ChainError("collected cell is not an object")
            key = (cell.get("dataset"), cell.get("backend"))
            if key in seen:
                raise ChainError(f"duplicate collected cell: {key}")
            seen.add(key)
            validate_cell(root, cell, expected_keys)
        if seen != expected_keys:
            raise ChainError("collected matrix is missing a canonical dataset/backend cell")
        result = {
            "schema": "zkgpt-gat-final-validation-v1",
            "status": "VALIDATED",
            "baseline": "zkGPT",
            "matrix": matrix,
            "cells_validated": len(cells),
            "resource_scope": "baseline-local",
            "proof_scope": "full_gat",
            "privacy_profile": "masked_full",
            "independent_verifier": True,
            "no_fallback": True,
            "witness_isolated": True,
            "input": args.input,
        }
        atomic_write_json(root / "results" / "final_validation.json", result)
    except (ChainError, OSError, ValueError) as exc:
        print(f"FINAL_VALIDATION_FAILED: {exc}", file=sys.stderr)
        return 2
    print(f"FINAL_VALIDATION_OK cells={len(cells)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
