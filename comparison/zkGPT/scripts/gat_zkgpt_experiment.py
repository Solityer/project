#!/usr/bin/env python3
"""Execute one or more strict zkGPT GAT formal cells.

The script is orchestration only: the C++ adapter produces the proof and the
separate C++ verifier consumes the serialized public context, public logits and
artifact.  Any missing capability, failed child, fallback, or malformed
artifact aborts a cell and never becomes a successful result.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from common import (
    BACKENDS,
    DATASETS,
    ChainError,
    atomic_write_json,
    atomic_write_text,
    baseline_root,
    build_binary,
    config_with_auth,
    artifact_fields,
    formal_backend_capability,
    launch_native,
    load_json,
    local_path,
    require_prover_secret,
    require_result_shape,
    run_directory,
    verifier_binary,
    verify_cache,
    verify_checkpoint,
)


def selection(raw: str, allowed: Sequence[str], label: str) -> Tuple[str, ...]:
    if raw == "all":
        return tuple(allowed)
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not values or len(set(values)) != len(values) or any(item not in allowed for item in values):
        raise ChainError(f"{label} must be all or a comma-separated list of supported values")
    return values


def relative_path(root: Path, path: Path, description: str) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError as exc:
        raise ChainError(f"{description} is outside baseline root: {path}") from exc


def require_run_file(root: Path, run_path: Path, relative: Any, description: str) -> Tuple[str, Path]:
    if not isinstance(relative, str) or not relative:
        raise ChainError(f"result has no {description} path")
    path = local_path(root, relative, description)
    if path.parent != run_path:
        raise ChainError(f"{description} is outside its formal run directory: {path}")
    if not path.is_file():
        raise ChainError(f"missing {description}: {path}")
    return relative, path


def require_manifest(root: Path, name: str, expected_schema: str) -> Mapping[str, Any]:
    path = root / "results" / name
    payload = load_json(path, name)
    if not isinstance(payload, dict) or payload.get("schema") != expected_schema:
        raise ChainError(f"unsupported {name} schema: {path}")
    if payload.get("status") not in ("AUTHENTICATED", "READY"):
        raise ChainError(f"{name} is not successful: {path}")
    return payload


def ensure_registration(root: Path, dataset: str, backend: str,
                        run_relative: str,
                        run_path: Path) -> Tuple[str, str]:
    """Create one registration/secret pair automatically, never copy secrets."""

    registration_relative = f"{run_relative}/model_registration.json"
    secret_relative = f"{run_relative}/com_W.prover.secret"
    registration_path = run_path / "model_registration.json"
    secret_path = run_path / "com_W.prover.secret"
    registration_exists = registration_path.exists()
    secret_exists = secret_path.exists()
    if registration_exists != secret_exists:
        raise ChainError(
            f"incomplete model registration pair for {dataset}/{backend}; "
            "both registration and prover secret are required"
        )
    if not registration_exists:
        command = [
            sys.executable,
            str(root / "scripts" / "register_model.py"),
            "--dataset", dataset,
            "--backend", backend,
        ]
        registered = subprocess.run(
            command, cwd=str(root), text=True, capture_output=True, check=False,
        )
        atomic_write_text(run_path / "runner-registration.stdout.log", registered.stdout)
        atomic_write_text(run_path / "runner-registration.stderr.log", registered.stderr)
        if registered.returncode != 0:
            raise ChainError(
                f"automatic model registration failed ({registered.returncode}); "
                f"see {run_relative}/runner-registration.stderr.log"
            )
    if not registration_path.is_file():
        raise ChainError(f"automatic registration did not produce {registration_path}")
    require_prover_secret(secret_path)
    return registration_relative, secret_relative


def run_cell(root: Path, dataset: str, backend: str) -> None:
    capability = formal_backend_capability(backend)
    config_path, config = config_with_auth(root, dataset)
    if config.get("privacy_policy") != "model_and_trace_hidden" or config.get("privacy_profile", "masked_full") not in ("masked_full", ""):
        raise ChainError(f"{dataset} config does not declare the strict privacy boundary")
    if not capability["supported"]:
        # Keep an explicit machine-readable gate record for an intentionally
        # unavailable formal backend.  No registration, CUDA evidence, or
        # native prover process is started on this path.
        run_relative, run_path = run_directory(root, dataset, backend, config)
        atomic_write_json(run_path / "execution.json", {
            "schema": "zkgpt-gat-execution-v1",
            "dataset": dataset,
            "backend": backend,
            "status": "UNSUPPORTED",
            "no_fallback": True,
            "formal_backend_supported": False,
            "capability_status": capability["status"],
            "native_invoked": False,
            "reason": capability["reason"],
        })
        raise ChainError(capability["reason"])
    verify_cache(root, dataset)
    verify_checkpoint(root, dataset)
    build_binary(root, backend)
    verifier_binary(root, backend)
    run_relative, run_path = run_directory(root, dataset, backend, config)
    config_relative = relative_path(root, config_path, "dataset config")
    registration_relative, secret_relative = ensure_registration(
        root, dataset, backend, run_relative, run_path)

    prover_start = time.perf_counter()
    completed = launch_native(
        root, dataset, backend, config_relative, operation="prove",
        registration_path_value=registration_relative,
        com_w_secret_path_value=secret_relative,
    )
    prover_wall_ms = (time.perf_counter() - prover_start) * 1000.0
    atomic_write_text(run_path / "prover.stdout.log", completed.stdout)
    atomic_write_text(run_path / "prover.stderr.log", completed.stderr)
    if completed.returncode != 0:
        raise ChainError(f"native prover failed with {completed.returncode}; see {run_relative}/prover.stderr.log")

    result_path = run_path / "result.json"
    result = require_result_shape(load_json(result_path, "native result"), result_path)
    if result.get("dataset") != dataset or result.get("backend") != backend:
        raise ChainError(f"native result identity mismatch for {dataset}/{backend}")
    if result.get("capability_status") != capability["status"] or result.get("formal_backend_supported") is not True:
        raise ChainError(f"native result violates the formal backend capability gate for {dataset}/{backend}")
    if result.get("status") != "VERIFY_OK" or result.get("verification_status") != "VERIFY_OK":
        raise ChainError(f"native result is not VERIFY_OK for {dataset}/{backend}")
    if result.get("cpu_fallback") is not False or result.get("verifier_reads_witness") is not False:
        raise ChainError(f"native result violates no-fallback/witness isolation for {dataset}/{backend}")
    artifact_relative, _ = artifact_fields(root, result, run_path)
    context_relative, _ = require_run_file(root, run_path, result.get("verifier_context_path"), "verifier context")
    logits_relative, _ = require_run_file(root, run_path, result.get("public_logits_path"), "public logits")
    report_relative = f"{run_relative}/verifier_report.json"
    verifier = verifier_binary(root, backend)
    verifier_start = time.perf_counter()
    verifier_args = [
        str(verifier),
        "--baseline-root", str(root),
        "--verifier-context", context_relative,
        "--public-logits", logits_relative,
        "--artifact", artifact_relative,
        "--report", report_relative,
    ]
    verified = subprocess.run(
        verifier_args, cwd=str(root), text=True, capture_output=True, check=False
    )
    verifier_wall_ms = (time.perf_counter() - verifier_start) * 1000.0
    atomic_write_text(run_path / "verifier.stdout.log", verified.stdout)
    atomic_write_text(run_path / "verifier.stderr.log", verified.stderr)
    if verified.returncode != 0:
        raise ChainError(f"independent verifier rejected the cell ({verified.returncode}); see {run_relative}/verifier_report.json")
    report_path = run_path / "verifier_report.json"
    report = load_json(report_path, "independent verifier report")
    if not isinstance(report, dict) or report.get("schema") != "zkgpt-gat-independent-verifier-v2":
        raise ChainError(f"unsupported independent verifier report schema: {report_path}")
    if report.get("status") != "VERIFY_OK" or report.get("independent") is not True:
        raise ChainError(f"independent verifier did not return VERIFY_OK for {dataset}/{backend}")
    if report.get("verifier_reads_witness") is not False or report.get("verifier_reads_checkpoint") is not False:
        raise ChainError(f"independent verifier violates public-only boundary for {dataset}/{backend}")
    atomic_write_json(run_path / "execution.json", {
        "schema": "zkgpt-gat-execution-v1",
        "dataset": dataset,
        "backend": backend,
        "status": "VERIFY_OK",
        "no_fallback": True,
        "prover_wall_ms": prover_wall_ms,
        "verifier_wall_ms": verifier_wall_ms,
        "prover_returncode": completed.returncode,
        "verifier_returncode": verified.returncode,
        "result_path": relative_path(root, result_path, "native result"),
        "verifier_report_path": report_relative,
    })
    print(f"VERIFY_OK dataset={dataset} backend={backend} result={run_relative}/result.json")


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="all", help="all or comma-separated canonical datasets")
    parser.add_argument("--backends", default="all", help="all or comma-separated cpu/gpu backends")
    parser.add_argument("--continue-on-failure", action="store_true", help="record all failures before returning nonzero")
    args = parser.parse_args(list(argv))
    root = baseline_root()
    try:
        datasets = selection(args.datasets, DATASETS, "datasets")
        backends = selection(args.backends, BACKENDS, "backends")
        require_manifest(root, "resource_manifest.json", "zkgpt-gat-resource-manifest-v1")
        preflight = require_manifest(root, "formal_preflight.json", "zkgpt-gat-formal-preflight-v1")
        if preflight.get("formal_experiment_eligible") is not True:
            raise ChainError("formal preflight is not eligible; no proving was started")
    except ChainError as exc:
        print(f"EXPERIMENT_BLOCKED: {exc}", file=sys.stderr)
        return 2

    failures: List[str] = []
    for dataset in datasets:
        for backend in backends:
            try:
                run_cell(root, dataset, backend)
            except (ChainError, OSError, ValueError) as exc:
                message = f"{dataset}/{backend}: {exc}"
                failures.append(message)
                print(f"CELL_FAILED: {message}", file=sys.stderr)
                if not args.continue_on_failure:
                    return 2
    if failures:
        print("EXPERIMENT_FAILED", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
