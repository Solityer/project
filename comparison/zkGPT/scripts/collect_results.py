#!/usr/bin/env python3
"""Collect the complete ten-cell zkGPT GAT matrix after independent checks."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from common import (
    BACKENDS,
    DATASETS,
    ChainError,
    artifact_fields,
    atomic_write_json,
    baseline_root,
    config_with_auth,
    load_json,
    local_path,
    require_directory,
    require_result_shape,
    sha256_file,
    verify_cache,
    verify_checkpoint,
)


def require_bool(value: Mapping[str, Any], key: str, expected: bool, path: Path) -> None:
    if not isinstance(value.get(key), bool) or value[key] is not expected:
        raise ChainError(f"{path} requires {key}={str(expected).lower()}")


def require_string(value: Mapping[str, Any], key: str, path: Path) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise ChainError(f"{path} requires non-empty {key}")
    return item


def require_timing(result: Mapping[str, Any], path: Path) -> Dict[str, float]:
    timing = result.get("timing_ms")
    if not isinstance(timing, dict):
        raise ChainError(f"{path} is missing timing_ms")
    required = ("load", "registration", "witness", "prove", "verify", "serialize")
    normalized: Dict[str, float] = {}
    for key in required:
        item = timing.get(key)
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)) or float(item) < 0:
            raise ChainError(f"{path} has invalid timing_ms.{key}")
        normalized[key] = float(item)
    return normalized


def collect_cell(root: Path, dataset: str, backend: str) -> Dict[str, Any]:
    config_path, config = config_with_auth(root, dataset)
    verify_cache(root, dataset)
    verify_checkpoint(root, dataset)
    export = config.get("export_dir") or f"baseline-local/runs/{dataset}_secure"
    run_relative = f"{export}/{backend}"
    run_path = local_path(root, run_relative, "formal run directory")
    require_directory(run_path, "formal run directory")
    result_path = run_path / "result.json"
    result = require_result_shape(load_json(result_path, "native result"), result_path)
    if result.get("dataset") != dataset or result.get("backend") != backend:
        raise ChainError(f"result identity mismatch for {dataset}/{backend}")
    if result.get("status") != "VERIFY_OK" or result.get("verification_status") != "VERIFY_OK":
        raise ChainError(f"result is not VERIFY_OK for {dataset}/{backend}")
    for key in ("proof_generated", "verifier_called", "verifier_ok"):
        require_bool(result, key, True, result_path)
    require_bool(result, "verifier_reads_witness", False, result_path)
    require_bool(result, "cpu_fallback", False, result_path)
    if backend == "gpu":
        for key in ("gpu_compiled", "gpu_device_available", "gpu_kernel_executed"):
            require_bool(result, key, True, result_path)
        invocations = result.get("gpu_kernel_invocations")
        if isinstance(invocations, bool) or not isinstance(invocations, int) or invocations <= 0:
            raise ChainError(f"{result_path} requires positive gpu_kernel_invocations")
    else:
        require_bool(result, "gpu_kernel_executed", False, result_path)
        if result.get("gpu_kernel_invocations") != 0:
            raise ChainError(f"CPU result reports GPU kernel invocations: {result_path}")
    timing = require_timing(result, result_path)
    identity_keys = ("statement_digest", "proof_digest", "circuit_id", "plan_digest",
                     "full_dataset_identity", "checkpoint_identity", "cache_identity", "com_W_point")
    identity = {key: require_string(result, key, result_path) for key in identity_keys}
    artifact, declared_digest = artifact_fields(root, result, run_path)
    actual_digest = sha256_file(artifact)
    if declared_digest and declared_digest != actual_digest:
        raise ChainError(f"artifact_sha256 mismatch: {artifact}")
    proof_size = result.get("proof_size_bytes")
    if isinstance(proof_size, bool) or not isinstance(proof_size, int) or proof_size <= 0:
        raise ChainError(f"result has invalid proof_size_bytes: {result_path}")
    if proof_size != artifact.stat().st_size:
        raise ChainError(f"proof_size_bytes does not match artifact: {artifact}")

    report_path = run_path / "verifier_report.json"
    report = load_json(report_path, "independent verifier report")
    if not isinstance(report, dict) or report.get("schema") != "zkgpt-gat-independent-verifier-v1":
        raise ChainError(f"unsupported independent verifier report schema: {report_path}")
    if report.get("status") != "VERIFY_OK" or report.get("independent") is not True:
        raise ChainError(f"independent verifier is not VERIFY_OK: {report_path}")
    if report.get("dataset") != dataset or report.get("backend") != backend:
        raise ChainError(f"independent verifier identity mismatch: {report_path}")
    require_bool(report, "verifier_reads_witness", False, report_path)
    require_bool(report, "verifier_reads_checkpoint", False, report_path)
    if report.get("statement_digest") != result.get("statement_digest"):
        raise ChainError(f"independent verifier statement mismatch: {report_path}")

    execution_path = run_path / "execution.json"
    execution = load_json(execution_path, "execution timing")
    if not isinstance(execution, dict) or execution.get("schema") != "zkgpt-gat-execution-v1" or execution.get("status") != "VERIFY_OK":
        raise ChainError(f"unsupported execution timing record: {execution_path}")
    require_bool(execution, "no_fallback", True, execution_path)
    wall = {}
    for key in ("prover_wall_ms", "verifier_wall_ms"):
        item = execution.get(key)
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)) or float(item) < 0:
            raise ChainError(f"{execution_path} has invalid {key}")
        wall[key] = float(item)
    return {
        "dataset": dataset,
        "backend": backend,
        "result_path": str(result_path.relative_to(root)),
        "verifier_report_path": str(report_path.relative_to(root)),
        "artifact_path": str(artifact.relative_to(root)),
        "artifact_sha256": actual_digest,
        "proof_size_bytes": proof_size,
        "proof_digest": identity["proof_digest"],
        "statement_digest": identity["statement_digest"],
        "circuit_id": identity["circuit_id"],
        "plan_digest": identity["plan_digest"],
        "full_dataset_identity": identity["full_dataset_identity"],
        "checkpoint_identity": identity["checkpoint_identity"],
        "cache_identity": identity["cache_identity"],
        "com_W_point": identity["com_W_point"],
        "timing_ms": timing,
        "wall_ms": wall,
        "native_backend": require_string(result, "native_backend", result_path),
        "no_fallback": True,
        "witness_isolated": True,
    }


def main(argv: Sequence[str]) -> int:
    del argv
    root = baseline_root()
    cells: List[Dict[str, Any]] = []
    try:
        for dataset in DATASETS:
            for backend in BACKENDS:
                cells.append(collect_cell(root, dataset, backend))
    except (ChainError, OSError, ValueError) as exc:
        print(f"COLLECT_FAILED: {exc}", file=sys.stderr)
        return 2
    payload = {
        "schema": "zkgpt-gat-collected-results-v1",
        "status": "COLLECTED",
        "baseline": "zkGPT",
        "matrix": {"datasets": list(DATASETS), "backends": list(BACKENDS), "cell_count": len(cells)},
        "cells": cells,
        "resource_manifest": "results/resource_manifest.json",
        "preflight": "results/formal_preflight.json",
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
