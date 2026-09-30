#!/usr/bin/env python3
"""Run the bounded CPU proof/serialization/verifier readiness gate.

This is deliberately a real tiny native proof closure, not a capability or
metadata check.  The fixture is created in a fresh temporary directory and
is never reused from a previous run.  Full datasets are not proved here;
their native capacity-only path is checked for every canonical dataset.
"""

from __future__ import annotations

import argparse
import os
import signal
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from common import (
    BACKENDS,
    DATASETS,
    PREFLIGHT_SCHEMA,
    REPORT_SCHEMA,
    ChainError,
    assert_fresh_files,
    atomic_write_json,
    baseline_root,
    binary_fingerprint,
    build_binary,
    config_with_auth,
    file_fingerprint,
    formal_backend_capability,
    load_json,
    local_path,
    relative_path,
    require_bool,
    require_circuit_id,
    require_fixed_digest,
    require_regular,
    require_string,
    resource_identity,
    require_manifest,
    sha256_file,
    smoke_binary,
    verifier_binary,
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


def _fixture_file(root: Path, directory: Path, name: str, fixture: Mapping[str, Any], field: str, started_ns: int) -> Path:
    value = fixture.get(field)
    if value in (None, ""):
        path = directory / name
    elif not isinstance(value, str):
        raise ChainError(f"formal fixture has malformed {field}")
    else:

        if value.startswith("/"):
            raise ChainError(f"formal fixture {field} must be baseline-relative")
        candidate = local_path(root, value, f"formal fixture {field}")
        path = candidate
        if path.parent != directory:
            raise ChainError(f"formal fixture {field} escapes fresh directory")
    require_regular(path, f"formal fixture {field}")
    if path.stat().st_mtime_ns < started_ns:
        raise ChainError(f"formal fixture {field} is stale: {path}")
    return path


def _fixture_identity(fixture: Mapping[str, Any], where: Path) -> Dict[str, Any]:
    identity: Dict[str, Any] = {}
    for key in ("statement_digest", "plan_digest", "proof_digest",
                "registration_digest", "registration_binding_digest",
                "registration_layout_digest", "com_W_point"):
        identity[key] = require_fixed_digest(fixture.get(key), key, where)
    identity["circuit_id"] = require_circuit_id(fixture.get("circuit_id"), where)
    for key in ("full_dataset_identity", "checkpoint_identity", "cache_identity"):
        identity[key] = require_string(fixture, key, where)
    return identity


def _run_streaming(command: Sequence[str], cwd: Path, label: str,
                   diagnostic_dir: Path, timeout: float) -> subprocess.CompletedProcess:
    """Run one native stage while teeing both streams and retaining them.

    ``subprocess.run(capture_output=True)`` hid the useful native progress when
    the old tiny proof timed out.  This bounded runner forwards each line to
    the caller's terminal and writes an independent copy into the fresh
    preflight directory, including output produced before a timeout.
    """
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = diagnostic_dir / f"{label}.stdout.log"
    stderr_path = diagnostic_dir / f"{label}.stderr.log"
    process = subprocess.Popen(
        list(command), cwd=str(cwd), text=True, bufsize=1,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        start_new_session=True,
    )
    chunks = {"stdout": [], "stderr": []}

    def pump(stream_name: str, stream: Any, path: Path) -> None:
        target = sys.stdout if stream_name == "stdout" else sys.stderr
        with path.open("w", encoding="utf-8") as output:
            for line in iter(stream.readline, ""):
                chunks[stream_name].append(line)
                output.write(line)
                output.flush()
                print(f"[{label}/{stream_name}] {line}", end="", file=target, flush=True)
        stream.close()

    assert process.stdout is not None and process.stderr is not None
    threads = [
        threading.Thread(target=pump, args=("stdout", process.stdout, stdout_path), daemon=True),
        threading.Thread(target=pump, args=("stderr", process.stderr, stderr_path), daemon=True),
    ]
    for thread in threads:
        thread.start()
    timed_out = False
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
    for thread in threads:
        thread.join(timeout=5)
    result = subprocess.CompletedProcess(
        list(command), process.returncode,
        "".join(chunks["stdout"]), "".join(chunks["stderr"]),
    )
    result.timed_out = timed_out  # type: ignore[attr-defined]
    result.diagnostic_dir = diagnostic_dir  # type: ignore[attr-defined]
    return result


def run_tiny_formal(root: Path, backend: str) -> Dict[str, Any]:
    if backend != "cpu":
        raise ChainError("tiny formal preflight is CPU-only")
    smoke = smoke_binary(root, backend)
    verifier = verifier_binary(root, backend)
    smoke_fp = binary_fingerprint(smoke, "CPU formal smoke executable")
    verifier_fp = binary_fingerprint(verifier, "CPU verifier executable")
    results_dir = root / "results"
    results_dir.mkdir(exist_ok=True)
    try:
        timeout = float(os.environ.get("ZKGPT_PREFLIGHT_TINY_TIMEOUT_SECONDS", "300"))
    except ValueError as exc:
        raise ChainError("ZKGPT_PREFLIGHT_TINY_TIMEOUT_SECONDS must be numeric") from exc
    if timeout <= 0:
        raise ChainError("ZKGPT_PREFLIGHT_TINY_TIMEOUT_SECONDS must be positive")
    fresh = Path(tempfile.mkdtemp(prefix=".formal-preflight-", dir=str(results_dir)))
    started_ns = time.time_ns()
    keep_fresh = False
    stage = "smoke"
    try:
        completed = _run_streaming(
            [str(smoke), "--formal", "--standalone-dir", str(fresh)],
            root, "smoke", fresh, timeout,
        )
        if getattr(completed, "timed_out", False):
            keep_fresh = True
            raise ChainError(
                f"formal adapter smoke timed out after {timeout:g}s; partial diagnostics: {fresh}"
            )
        if completed.returncode != 0:
            keep_fresh = True
            raise ChainError(
                f"formal adapter smoke failed with {completed.returncode}; partial diagnostics: {fresh}"
            )
        fixture_path = fresh / "fixture.json"
        fixture = load_json(fixture_path, "formal fixture")
        if not isinstance(fixture, dict) or fixture.get("schema") not in ("zkgpt-gat-formal-fixture-v1", "zkgpt-gat-formal-fixture-v2"):
            raise ChainError(f"unsupported formal fixture schema: {fixture_path}")
        if fixture.get("backend") != backend:
            raise ChainError("formal fixture backend mismatch")
        identity = _fixture_identity(fixture, fixture_path)
        context = _fixture_file(root, fresh, "verifier_context.bin", fixture, "verifier_context_path", started_ns)
        logits = _fixture_file(root, fresh, "public_logits.txt", fixture, "public_logits_path", started_ns)
        artifact = _fixture_file(root, fresh, "proof.artifact", fixture, "artifact_path", started_ns)
        report = fresh / "verifier_report.json"
        context_rel = relative_path(root, context, "formal fixture context")
        logits_rel = relative_path(root, logits, "formal fixture logits")
        artifact_rel = relative_path(root, artifact, "formal fixture artifact")
        report_rel = relative_path(root, report, "formal fixture report")
        stage = "verifier"
        verified = _run_streaming(
            [str(verifier), "--baseline-root", str(root), "--verifier-context", context_rel,
             "--public-logits", logits_rel, "--artifact", artifact_rel, "--report", report_rel],
            root, "verifier", fresh, timeout,
        )
        if getattr(verified, "timed_out", False):
            keep_fresh = True
            raise ChainError(
                f"standalone independent verifier timed out after {timeout:g}s; partial diagnostics: {fresh}"
            )
        if verified.returncode != 0:
            keep_fresh = True
            raise ChainError(
                f"standalone independent verifier rejected tiny fixture ({verified.returncode}); partial diagnostics: {fresh}"
            )
        report_payload = load_json(report, "tiny independent verifier report")
        if not isinstance(report_payload, dict) or report_payload.get("schema") != REPORT_SCHEMA:
            raise ChainError(f"tiny verifier report is not schema v3: {report}")
        if report_payload.get("status") != "VERIFY_OK" or report_payload.get("independent") is not True or report_payload.get("masked_full") is not True:
            raise ChainError("tiny independent verifier did not return VERIFY_OK")
        require_bool(report_payload, "verifier_reads_checkpoint", False, report)
        require_bool(report_payload, "verifier_reads_witness", False, report)
        for key in ("statement_digest", "plan_digest", "proof_digest",
                    "registration_digest", "registration_binding_digest",
                    "registration_layout_digest", "com_W_point"):
            require_fixed_digest(report_payload.get(key), key, report)
            if report_payload[key] != identity[key]:
                raise ChainError(f"tiny verifier identity mismatch for {key}")
        if report_payload.get("circuit_id") != identity["circuit_id"]:
            raise ChainError("tiny verifier circuit identity mismatch")
        for key in ("full_dataset_identity", "checkpoint_identity", "cache_identity"):
            if report_payload.get(key) != identity[key]:
                raise ChainError(f"tiny verifier resource identity mismatch for {key}")
        artifact_fp = file_fingerprint(artifact, "tiny proof artifact")
        if report_payload.get("artifact_sha256") != artifact_fp["sha256"] or report_payload.get("artifact_size_bytes") != artifact_fp["size_bytes"]:
            raise ChainError("tiny verifier artifact identity mismatch")
        files = {
            "verifier_context": file_fingerprint(context, "tiny verifier context"),
            "public_logits": file_fingerprint(logits, "tiny public logits"),
            "proof_artifact": artifact_fp,
            "fixture": file_fingerprint(fixture_path, "tiny fixture"),
            "verifier_report": file_fingerprint(report, "tiny verifier report"),
        }
        assert_fresh_files((fixture_path, context, logits, artifact, report), started_ns, "tiny formal output")
        return {
            "schema": "zkgpt-gat-tiny-formal-v1", "status": "READY", "backend": backend,
            "independent_verifier": True, "verifier_reads_checkpoint": False,
            "verifier_reads_witness": False, "fixture_schema": fixture["schema"],
            "identity": identity, "files": files,
            "smoke_binary": {"path": relative_path(root, smoke, "smoke executable"), **smoke_fp},
            "verifier_binary": {"path": relative_path(root, verifier, "verifier executable"), **verifier_fp},
        }
    except Exception:
        keep_fresh = True
        raise
    finally:
        if keep_fresh:
            print(f"PREFLIGHT_PARTIAL_DIAGNOSTICS stage={stage} directory={fresh}", file=sys.stderr)
        else:

            shutil.rmtree(fresh, ignore_errors=False)


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
    selected = BACKENDS if args.backend == "all" else (args.backend,)
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
    tiny: Optional[Dict[str, Any]] = None
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
            entry["prover"] = relative_path(root, prover, "prover executable")
            entry["verifier"] = relative_path(root, verifier, "verifier executable")
            entry["prover_fingerprint"] = binary_fingerprint(prover, "prover executable")
            entry["verifier_fingerprint"] = binary_fingerprint(verifier, "verifier executable")
            entry["capabilities"] = check_capabilities(root, backend, verifier)
            if backend == "cpu":
                tiny = run_tiny_formal(root, backend)
                entry["tiny_formal"] = tiny
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
        "tiny_formal": tiny,
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
