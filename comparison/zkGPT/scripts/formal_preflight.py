#!/usr/bin/env python3
"""Authenticate formal execution preconditions without running development tests.

Only actual dataset proofs and separate native verification produce formal
results. Development proof closure is checked independently under tests/.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from common import (
    BACKENDS,
    FORMAL_BACKENDS,
    DATASETS,
    PREFLIGHT_SCHEMA,
    ChainError,
    atomic_write_json,
    baseline_root,
    binary_fingerprint,
    build_binary,
    config_with_auth,
    file_fingerprint,
    formal_backend_capability,
    load_json,
    invalidate_results,
    relative_path,
    require_bool,
    require_regular,
    require_manifest,
    sha256_file,
    verifier_binary,
    verify_build_provenance,
    verify_local_resources,
)


def check_source(root: Path) -> List[str]:
    required = (
        root / "gat_adapter" / "gat_cli.cpp",
        root / "gat_adapter" / "gat_verifier_cli.cpp",
        root / "gat_adapter" / "gat_proof.cpp",
        root / "gat_adapter" / "CMakeLists.txt",
        root / "build.sh",
    )
    errors: List[str] = []
    for path in required:
        try:
            require_regular(path, "production source")
        except ChainError as exc:
            errors.append(str(exc))
    return errors


def check_capabilities(root: Path, backend: str, executable: Path) -> Dict[str, Any]:
    completed = subprocess.run([str(executable), "--capabilities"], cwd=str(root),
                               text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise ChainError(f"{backend} verifier capabilities failed with {completed.returncode}")
    try:
        import json
        payload = json.loads(completed.stdout)
    except (ImportError, ValueError) as exc:
        raise ChainError(f"{backend} verifier capabilities are not JSON") from exc
    if not isinstance(payload, dict) or payload.get("schema") != "zkgpt-gat-verifier-capabilities-v2":
        raise ChainError(f"{backend} verifier capabilities schema is unsupported")
    for key in ("independent", "reads_witness", "reads_checkpoint"):
        require_bool(payload, key, key == "independent", f"{backend} capabilities")
    if payload.get("masked_full") is not True:
        raise ChainError(f"{backend} verifier declares masked_full unsupported")
    return payload


def run_capacity(root: Path, dataset: str, backend: str) -> Dict[str, Any]:
    config_path, config = config_with_auth(root, dataset)
    prover = build_binary(root, backend)
    completed = subprocess.run(
        [str(prover), "--capacity-only", "--dataset", dataset, "--backend", backend,
         "--baseline-root", str(root), "--config", relative_path(root, config_path, "dataset config")],
        cwd=str(root), text=True, capture_output=True, check=False, timeout=300,
    )
    try:
        import json
        payload = json.loads(completed.stdout)
    except (ImportError, ValueError) as exc:
        raise ChainError(f"capacity output is not JSON for {dataset}/{backend}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != "zkgpt-gat-capacity-v1":
        raise ChainError(f"unsupported capacity schema for {dataset}/{backend}")
    if payload.get("dataset") != dataset or payload.get("backend") != backend or payload.get("status") not in ("READY", "BLOCKED"):
        raise ChainError(f"capacity has an invalid status for {dataset}/{backend}")
    if payload.get("status") == "READY" and (payload.get("proof_path") != "native_layered_gkr" or payload.get("full_relation_bound") is not True):
        raise ChainError(f"capacity selected an unsupported or incomplete full-GAT proof path for {dataset}/{backend}")

    if completed.returncode != 0 and payload.get("status") != "BLOCKED":
        raise ChainError(f"capacity preflight failed for {dataset}/{backend}: {completed.stderr[-1000:]}")

    for key in ("layer_count", "max_layer_size", "max_materialized_vector_elements", "estimated_peak_memory_bytes"):
        if key in payload:
            value = payload[key]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ChainError(f"capacity has invalid {key}: {dataset}")
    return payload


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("all",) + BACKENDS, default="cpu")
    args = parser.parse_args(list(argv))
    root = baseline_root()
    invalidate_results(root, ("formal_preflight.json", "collected_results.json", "final_validation.json"))
    selected = FORMAL_BACKENDS if args.backend == "all" else (args.backend,)
    blockers: List[str] = []
    source_errors = check_source(root)
    blockers.extend(source_errors)
    resource_errors = verify_local_resources(root)
    blockers.extend(resource_errors)
    backend_report: Dict[str, Any] = {}
    capacity: Dict[str, Any] = {}
    resource_manifest_digest: Optional[str] = None
    resource_manifest_fp: Optional[Dict[str, Any]] = None
    try:
        resource_manifest = require_manifest(root, "resource_manifest.json", "zkgpt-gat-resource-manifest-v1")
        del resource_manifest
        resource_manifest_path = root / "results" / "resource_manifest.json"
        resource_manifest_digest = sha256_file(resource_manifest_path)
        resource_manifest_fp = file_fingerprint(resource_manifest_path, "resource manifest")
    except ChainError as exc:
        blockers.append(str(exc))
    for backend in selected:
        entry: Dict[str, Any] = {"prover": None, "verifier": None, "capabilities": None, "ready": False}
        capability = formal_backend_capability(backend)
        entry["capability_status"] = capability["status"]
        entry["formal_backend_supported"] = capability["supported"]
        if not capability["supported"]:
            entry["blocker"] = capability["reason"]
            blockers.append(capability["reason"])
            backend_report[backend] = entry
            continue
        try:
            prover = build_binary(root, backend)
            verifier = verifier_binary(root, backend)
            entry["source_tree_sha256"] = verify_build_provenance(root, backend)
            entry["prover"] = relative_path(root, prover, "prover executable")
            entry["verifier"] = relative_path(root, verifier, "verifier executable")
            entry["prover_fingerprint"] = binary_fingerprint(prover, "prover executable")
            entry["verifier_fingerprint"] = binary_fingerprint(verifier, "verifier executable")
            entry["capabilities"] = check_capabilities(root, backend, verifier)
            entry["ready"] = True
        except (ChainError, OSError, ValueError, subprocess.SubprocessError) as exc:
            entry["blocker"] = str(exc)
            blockers.append(str(exc))
        backend_report[backend] = entry
    if not blockers and "cpu" in selected:
        for dataset in DATASETS:
            try:
                capacity[dataset] = run_capacity(root, dataset, "cpu")
                if capacity[dataset].get("status") == "BLOCKED":
                    blockers.append(
                        f"capacity blocked for {dataset}/cpu: "
                        f"{capacity[dataset].get('reason', 'unspecified native capacity blocker')}"
                    )
            except (ChainError, OSError, ValueError, subprocess.SubprocessError) as exc:
                blockers.append(str(exc))
                capacity[dataset] = {"schema": "zkgpt-gat-capacity-v1", "status": "BLOCKED", "dataset": dataset, "backend": "cpu", "reason": str(exc)}
    payload = {
        "schema": PREFLIGHT_SCHEMA,
        "status": "READY" if not blockers else "BLOCKED",
        "formal_experiment_eligible": not blockers,
        "source_ready": not source_errors,
        "resources_ready": not resource_errors,
        "backends": backend_report,
        "capacity": capacity,
        "readiness_scope": "formal_execution_preconditions",
        "static_checks_passed": not blockers,
        "development_e2e_checked": False,
        "development_e2e_check": "tests/check_formal_closure.py (separate development evidence)",
        "formal_results_complete": False,
        "resource_manifest_sha256": resource_manifest_digest,
        "resource_manifest_fingerprint": resource_manifest_fp,
        "blockers": blockers,
        "formal_datasets_started": False,
        "no_fallback": True,
        "verifier_reads_checkpoint": False,
        "verifier_reads_witness": False,
    }
    try:
        atomic_write_json(root / "results" / "formal_preflight.json", payload)
    except ChainError as exc:
        print(f"PREFLIGHT_OUTPUT_FAILED: {exc}", file=sys.stderr)
        return 2
    if blockers:
        print("FORMAL_PREFLIGHT_BLOCKED", file=sys.stderr)
        for blocker in blockers:
            print(f"- {blocker}", file=sys.stderr)
        return 2
    print("FORMAL_PREFLIGHT_READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
