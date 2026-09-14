#!/usr/bin/env python3
"""Run bounded, non-formal readiness checks for the zkGPT GAT matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

from common import (
    BACKENDS,
    ChainError,
    atomic_write_json,
    baseline_root,
    build_binary,
    load_json,
    require_regular,
    verifier_binary,
    verify_local_resources,
)


def check_source(root: Path) -> List[str]:
    required = (
        root / "gat_adapter" / "gat_cli.cpp",
        root / "gat_adapter" / "gat_verifier_cli.cpp",
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
    completed = subprocess.run(
        [str(executable), "--capabilities"],
        cwd=str(root),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ChainError(f"{backend} verifier capabilities failed with {completed.returncode}")
    try:
        payload = json.loads(completed.stdout)
    except ValueError as exc:
        raise ChainError(f"{backend} verifier capabilities are not JSON") from exc
    if not isinstance(payload, dict) or payload.get("schema") != "zkgpt-gat-verifier-capabilities-v2":
        raise ChainError(f"{backend} verifier capabilities schema is unsupported")
    if payload.get("independent") is not True or payload.get("reads_witness") is not False or payload.get("reads_checkpoint") is not False:
        raise ChainError(f"{backend} verifier does not declare the isolated public-only boundary")
    return payload


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("all",) + BACKENDS, default="all")
    args = parser.parse_args(list(argv))
    root = baseline_root()
    selected = BACKENDS if args.backend == "all" else (args.backend,)
    blockers: List[str] = []
    source_errors = check_source(root)
    blockers.extend(source_errors)
    resource_errors = verify_local_resources(root)
    blockers.extend(resource_errors)
    backend_report: Dict[str, Any] = {}
    for backend in selected:
        entry: Dict[str, Any] = {"prover": None, "verifier": None, "capabilities": None, "ready": False}
        try:
            prover = build_binary(root, backend)
            entry["prover"] = str(prover.relative_to(root))
        except ChainError as exc:
            entry["blocker"] = str(exc)
            blockers.append(str(exc))
            backend_report[backend] = entry
            continue
        try:
            verifier = verifier_binary(root, backend)
            entry["verifier"] = str(verifier.relative_to(root))
            capabilities = check_capabilities(root, backend, verifier)
            entry["capabilities"] = capabilities
            if capabilities.get("masked_full") is not True:
                raise ChainError(f"{backend} verifier declares masked_full unsupported")
            entry["ready"] = True
        except ChainError as exc:
            entry["blocker"] = str(exc)
            blockers.append(str(exc))
        backend_report[backend] = entry
    payload = {
        "schema": "zkgpt-gat-formal-preflight-v1",
        "status": "READY" if not blockers else "BLOCKED",
        "formal_experiment_eligible": not blockers,
        "source_ready": not source_errors,
        "resources_ready": not resource_errors,
        "backends": backend_report,
        "blockers": blockers,
        "formal_datasets_started": False,
        "no_fallback": True,
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
