#!/usr/bin/env python3
"""Run the baseline-local ezkl formal chain for all five datasets.

The command authenticates the prepared five-dataset resources and official
native ``get-srs`` cache, reuses a source-matched GPU-enabled build or builds it once,
and then runs setup, independent production,
final binding, witness, prove, independent verify, collection and validation.
No SRS path, digest, or other ceremony input is accepted from the operator.
"""

from __future__ import annotations

import argparse
import fcntl
import importlib
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from formal_common import (
    CONTRACT,
    DATASETS,
    FORMAL_GPU_FEATURES,
    FORMAL_SESSION_ID,
    NATIVE_FORMAL_BACKEND,
    ROOT,
    ContractError,
    abort,
    local_path,
    native_backend_identity,
    formal_build_environment_identity,
    load_cli_contract,
    load_json,
    parse_cfg,
    print_error,
    require_file,
    resource_manifest_hash,
    sha256,
    source_tree_hash,
    create_authenticated_formal_context,
    validate_config,
    validate_native_srs_manifest,
    validate_gpu_environment,
    validate_resources,
    validate_formal_python_environment,
    select_formal_cuda_device,
    resource_file_identities,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--binary")
    result.add_argument("--output-root", default="runs/formal")
    result.add_argument("--matrix", default="results/formal_matrix.json")
    result.add_argument("--preflight-only", action="store_true", help="authenticate all prerequisites without setup/build/prove")
    return result


def run(command: list[str], label: str, *, gpu: bool = False) -> float:
    print(f"[ezkl-formal] {label}: {' '.join(command)}", flush=True)
    env = os.environ.copy()
    for key in (
        "ENABLE_ICICLE_GPU",
        "EZKL_GAT_BACKEND",
        "ICICLE_SMALL_K",
        "ICICLE_SMALL_K_FFT",
        "ICICLE_BACKEND_INSTALL_DIR",
        "ICICLE_NATIVE_LIB_DIR",
        "ICICLE_NATIVE_MANIFEST_SHA256",
        "CUDA_ARCH",
    ):
        env.pop(key, None)
    if gpu:
        identity = native_backend_identity()
        install_dir = ROOT / "artifacts" / "build" / "icicle-cuda"
        env["ENABLE_ICICLE_GPU"] = "1"
        env["EZKL_GAT_BACKEND"] = "gpu"
        env["ICICLE_SMALL_K"] = "2"
        env["ICICLE_SMALL_K_FFT"] = "2"
        env["ICICLE_BACKEND_INSTALL_DIR"] = str(install_dir)
        env["ICICLE_NATIVE_LIB_DIR"] = str(install_dir / "lib")
        env["ICICLE_NATIVE_MANIFEST_SHA256"] = identity["native_backend_manifest_sha256"]
        env["CUDA_ARCH"] = str(identity["cuda_architecture"]).removeprefix("sm_")
    started = time.monotonic()
    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=(os.name == "posix"))
    if process.stdout is None:
        abort(f"{label} subprocess has no output stream")
    output_queue: queue.Queue[str | None] = queue.Queue()

    def read_output() -> None:
        try:
            for line in process.stdout:
                output_queue.put(line)
        finally:
            output_queue.put(None)

    reader = threading.Thread(target=read_output, name="ezkl-formal-command-output", daemon=True)
    reader.start()
    started = time.monotonic()
    try:
        while True:
            try:
                line = output_queue.get(timeout=20)
            except queue.Empty:
                elapsed_seconds = int(time.monotonic() - started)
                print(f"[ezkl-formal] {label} still running ({elapsed_seconds}s)", flush=True)
                continue
            if line is None:
                break
            sys.stdout.write(line)
            sys.stdout.flush()
        reader.join(timeout=2)
        if reader.is_alive():
            abort(f"{label} output stream did not close after its leader exited")
        returncode = process.wait()
    except BaseException:
        print(f"[ezkl-formal] stopping {label} process group after interruption or runner failure", file=sys.stderr, flush=True)
        _terminate_process_group(process)
        reader.join(timeout=5)
        raise
    if returncode != 0:
        abort(f"{label} failed with exit code {returncode}")
    return (time.monotonic() - started) * 1000.0


def run_prerequisite(command: list[str], label: str, *, timeout_seconds: int) -> None:
    print(f"[ezkl-formal] prerequisite {label} (timeout={timeout_seconds}s): {' '.join(command)}", flush=True)
    process = subprocess.Popen(command, cwd=ROOT, start_new_session=(os.name == "posix"))
    try:
        returncode = process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        _terminate_process_group(process)
        abort(f"prerequisite {label} timed out after {timeout_seconds}s; process group was terminated")
    except KeyboardInterrupt:
        print(f"[ezkl-formal] interrupted during prerequisite {label}; terminating process group", file=sys.stderr, flush=True)
        _terminate_process_group(process)
        raise
    except BaseException:
        print(f"[ezkl-formal] prerequisite {label} failed; terminating process group", file=sys.stderr, flush=True)
        _terminate_process_group(process)
        raise
    if returncode != 0:
        abort(f"prerequisite {label} failed with exit code {returncode}")


def run_resource_prepare_with_auth_handoff(
    command: list[str],
    *,
    session_id: str,
    timeout_seconds: int,
) -> dict[str, object]:
    """Stream the official auto-preparer and capture its one complete-auth result."""
    marker_prefix = "EZKL_FORMAL_RESOURCE_AUTH_V1="
    print(
        f"[ezkl-formal] prerequisite official resource preparation/authentication "
        f"(timeout={timeout_seconds}s): {' '.join(command)}",
        flush=True,
    )
    env = os.environ.copy()
    env.pop("EZKL_FORMAL_RESOURCE_AUTH_SESSION_ID", None)
    env["EZKL_FORMAL_RESOURCE_AUTH_SESSION_ID"] = session_id
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=(os.name == "posix"),
    )
    if process.stdout is None:
        abort("formal resource preparation has no output stream")
    output_queue: queue.Queue[str | None] = queue.Queue()

    def read_output() -> None:
        try:
            for line in process.stdout:
                output_queue.put(line)
        finally:
            output_queue.put(None)

    reader = threading.Thread(target=read_output, name="ezkl-resource-auth-output", daemon=True)
    reader.start()
    started = time.monotonic()
    deadline = started + timeout_seconds
    marker: dict[str, object] | None = None
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _terminate_process_group(process)
                abort(f"official resource preparation/authentication timed out after {timeout_seconds}s")
            try:
                line = output_queue.get(timeout=min(20, remaining))
            except queue.Empty:
                elapsed_seconds = int(time.monotonic() - started)
                print(f"[ezkl-formal] official resource preparation/authentication still running ({elapsed_seconds}s)", flush=True)
                continue
            if line is None:
                break
            sys.stdout.write(line)
            sys.stdout.flush()
            if line.startswith(marker_prefix):
                if marker is not None:
                    abort("official resource preparation emitted multiple authentication handoffs")
                try:
                    parsed = json.loads(line[len(marker_prefix):])
                except json.JSONDecodeError as exc:
                    abort(f"official resource preparation emitted an invalid authentication handoff: {exc}")
                if not isinstance(parsed, dict):
                    abort("official resource preparation emitted a malformed authentication handoff")
                marker = parsed
        reader.join(timeout=2)
        returncode = process.wait(timeout=max(0.1, deadline - time.monotonic()))
    except BaseException:
        print("[ezkl-formal] stopping official resource preparation process group after interruption or runner failure", file=sys.stderr, flush=True)
        _terminate_process_group(process)
        reader.join(timeout=5)
        raise
    if returncode != 0:
        abort(f"official resource preparation/authentication failed with exit code {returncode}")
    if marker is None:
        abort("official resource preparation exited successfully without a complete-authentication handoff")
    return marker


def validate_resource_auth_handoff(
    marker: dict[str, object],
    *,
    session_id: str,
    source_sha256: str,
    config_sha256s: tuple[tuple[str, str], ...],
    resource_manifest_sha256: str,
    resource_identities: tuple[tuple[str, tuple[int, int, int, int, int]], ...],
) -> None:
    expected_keys = {
        "schema_version",
        "authenticated",
        "session_id",
        "source_tree_sha256",
        "config_sha256s",
        "resource_manifest_sha256",
        "resource_file_identities_before",
        "resource_file_identities_after",
    }
    if set(marker) != expected_keys or marker.get("schema_version") != 1 or marker.get("authenticated") != "complete_resource_manifest":
        abort("official resource preparation authentication handoff schema is invalid")
    if marker.get("session_id") != session_id:
        abort("official resource preparation authentication handoff belongs to another session")
    if marker.get("source_tree_sha256") != source_sha256:
        abort("source tree changed during official resource preparation/authentication")
    expected_configs = [[dataset, digest] for dataset, digest in config_sha256s]
    if marker.get("config_sha256s") != expected_configs:
        abort("secure config identity changed during official resource preparation/authentication")
    expected_files = [[path, list(identity)] for path, identity in resource_identities]
    for field in ("resource_file_identities_before", "resource_file_identities_after"):
        rows = marker.get(field)
        if not isinstance(rows, list) or any(
            not isinstance(row, list)
            or len(row) != 2
            or not isinstance(row[0], str)
            or not isinstance(row[1], list)
            or len(row[1]) != 5
            or any(isinstance(value, bool) or not isinstance(value, int) for value in row[1])
            for row in rows
        ):
            abort(f"official resource preparation authentication handoff has invalid {field}")
        if rows != expected_files:
            abort(f"manifest-listed resources changed during or after official complete authentication ({field})")
    if marker.get("resource_manifest_sha256") != resource_manifest_sha256:
        abort("resource manifest changed after official complete authentication")


def _prepare_and_authenticate_resources(
    python: str,
    *,
    session_id: str,
    preflight_config_sha256s: tuple[tuple[str, str], ...],
) -> tuple[tuple[tuple[str, tuple[int, int, int, int, int]], ...], tuple[tuple[str, str], ...]]:
    """Run official auto-preparation and accept only its current-session auth handoff."""
    prepare_script = ROOT / "scripts" / "prepare_gat_resources.py"
    require_file(prepare_script, "canonical resource preparation entrypoint")
    source_before = source_tree_hash()
    config_before = tuple(preflight_config_sha256s)
    if any(sha256(ROOT / "resources/canonical/configs" / f"{dataset}_secure.cfg") != digest for dataset, digest in config_before):
        abort("secure config changed before official resource preparation")
    marker = run_resource_prepare_with_auth_handoff(
        [python, "-u", str(prepare_script), "--formal-auto"],
        session_id=session_id,
        timeout_seconds=21600,
    )
    _preflight_dataset_configs()
    config_after = tuple(
        (dataset, sha256(ROOT / "resources/canonical/configs" / f"{dataset}_secure.cfg"))
        for dataset in DATASETS
    )
    identities_after = resource_file_identities()
    source_after = source_tree_hash()
    if source_after != source_before:
        abort("source tree changed during official resource preparation/authentication")
    validate_resource_auth_handoff(
        marker,
        session_id=session_id,
        source_sha256=source_before,
        config_sha256s=config_after,
        resource_manifest_sha256=resource_manifest_hash(),
        resource_identities=identities_after,
    )
    print(
        f"[ezkl-formal] resource-authentication=PASS files={len(identities_after)} "
        "handoff=official-formal-auto current-session",
        flush=True,
    )
    return identities_after, config_after


def _terminate_process_group(process: subprocess.Popen[object]) -> None:
    if os.name == "posix":
        try:
            os.killpg(process.pid, 15)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        deadline = time.monotonic() + 5
        while True:
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                break
            if time.monotonic() >= deadline:
                try:
                    os.killpg(process.pid, 9)
                except ProcessLookupError:
                    pass
                break
            time.sleep(0.1)
    elif process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
    if process.poll() is None:
        process.wait()


def local_cli(path: Path, label: str) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        abort(f"{label} is outside the ezkl baseline: {path}")


def reusable_build(
    binary: Path,
    backend: str,
    runtime_identity: dict[str, object],
    build_environment: dict[str, object] | None = None,
) -> bool:
    receipt_path = ROOT / "artifacts" / "build" / backend / "build_receipt.json"
    if not binary.is_file() or binary.is_symlink() or not os.access(binary, os.X_OK):
        return False
    if not receipt_path.is_file() or receipt_path.is_symlink() or not CONTRACT.is_file():
        return False
    try:
        receipt = load_json(receipt_path, "native build receipt")
    except ContractError:
        return False
    contract = load_cli_contract()
    expected = {
        "backend": backend, "binary_path": binary.relative_to(ROOT).as_posix(), "binary_sha256": sha256(binary),
        "source_tree_sha256": source_tree_hash(), "resource_manifest_sha256": resource_manifest_hash(),
        "adapter_contract_id": contract["contract_id"], "adapter_contract_sha256": sha256(CONTRACT),
        "rustc": runtime_identity["rustc"], "cargo": runtime_identity["cargo"],
    }
    if backend == "gpu":
        if build_environment is None:
            abort("GPU build receipt check has no resolved formal build environment")
        expected.update(native_backend_identity())
        expected["build_environment"] = build_environment
        expected["build_environment_sha256"] = build_environment["build_environment_sha256"]
    return (
        receipt.get("schema_version") == (2 if backend == "gpu" else 1)
        and receipt.get("baseline") == "ezkl"
        and receipt.get("relation_id") == load_cli_contract()["relation_id"]
        and all(receipt.get(key) == value for key, value in expected.items())
        and receipt.get("features") == (list(FORMAL_GPU_FEATURES) if backend == "gpu" else ["ezkl"])
    )


def check_output_paths(output_root: Path, matrix: Path) -> None:
    for path, label in ((output_root, "output-root"), (matrix.parent, "matrix parent")):
        for parent in (path, *path.parents):
            if parent == ROOT.parent:
                break
            if parent.exists() and not parent.is_dir():
                abort(f"{label} has a non-directory path: {parent}")
    if matrix.exists() and not matrix.is_file():
        abort(f"matrix is not a regular file: {matrix}")
    for dataset in DATASETS:
        for path in (output_root / dataset, output_root / ".incomplete", output_root / ".incomplete" / dataset):
            if path.is_symlink() or (path.exists() and not path.is_dir()):
                abort(f"formal output path is not a regular directory: {path}")


def _write_session_record(path: Path, document: dict[str, object]) -> None:
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    os.replace(temp, path)


def begin_formal_session(output_root: Path, matrix: Path) -> tuple[str, Path, int, list[str]]:
    lock_dir = ROOT / "runs"
    if lock_dir.is_symlink() or (lock_dir.exists() and not lock_dir.is_dir()):
        abort(f"formal lock directory is not a regular directory: {lock_dir}")
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / ".ezkl-formal.lock"
    if lock_path.is_symlink():
        abort(f"formal session lock must not be a symlink: {lock_path}")
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    lock_fd = os.open(lock_path, flags, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(lock_fd)
        abort("another formal ezkl session currently holds the baseline lock")

    try:
        session_id = uuid.uuid4().hex
        archive_root = output_root / ".incomplete" / session_id / "previous"
        archived: list[str] = []
        archive_root.mkdir(parents=True, exist_ok=False)
        for dataset in DATASETS:
            old_output = output_root / dataset
            if old_output.exists() or old_output.is_symlink():
                if old_output.is_symlink() or not old_output.is_dir():
                    abort(f"prior formal output is not a regular directory: {old_output}")
                destination = archive_root / "datasets" / dataset
                destination.parent.mkdir(parents=True, exist_ok=True)
                old_output.rename(destination)
                archived.append(str(destination.relative_to(ROOT)))
                print(f"[ezkl-formal] archived prior result directory: {old_output} -> {destination}", flush=True)
        if matrix.exists() or matrix.is_symlink():
            if matrix.is_symlink() or not matrix.is_file():
                abort(f"prior formal matrix is not a regular file: {matrix}")
            matrix_archive = archive_root / "matrix" / matrix.name
            matrix_archive.parent.mkdir(parents=True, exist_ok=True)
            matrix.rename(matrix_archive)
            archived.append(str(matrix_archive.relative_to(ROOT)))
            print(f"[ezkl-formal] archived prior active matrix: {matrix} -> {matrix_archive}", flush=True)
        session_dir = output_root / ".sessions"
        if session_dir.is_symlink() or (session_dir.exists() and not session_dir.is_dir()):
            abort(f"formal session history path must be a regular directory: {session_dir}")
        session_dir.mkdir(parents=True, exist_ok=True)
        session_file = session_dir / f"{session_id}.json"
        _write_session_record(
            session_file,
            {
                "schema_version": 1,
                "baseline": "ezkl",
                "session_id": session_id,
                "status": "PREFLIGHT_STARTED",
                "stage": "session_created",
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "archived_previous_outputs": archived,
            },
        )
        lock_payload = f"pid={os.getpid()} session_id={session_id}\n".encode()
        os.ftruncate(lock_fd, 0)
        os.write(lock_fd, lock_payload)
        os.fsync(lock_fd)
        print(f"[ezkl-formal] started fresh formal session {session_id}; prior active outputs are archived", flush=True)
        return session_id, session_file, lock_fd, archived
    except BaseException:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
        raise


def update_session_record(
    path: Path,
    *,
    status: str,
    stage: str,
    error: str | None = None,
) -> None:
    document = json.loads(path.read_text())
    document["status"] = status
    document["stage"] = stage
    document["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    if error is not None:
        document["error"] = error[:2000]
    else:
        document.pop("error", None)
    _write_session_record(path, document)


def invoke_python_entrypoint(module_name: str, argv: list[str], label: str, *, context: object, session_id: str) -> None:
    module = importlib.import_module(module_name)
    parsed = module.parser().parse_args(argv)
    print(f"[ezkl-formal] {label}: in-process authenticated session call", flush=True)
    result = module.main(parsed, authenticated_context=context, formal_session_id=session_id)
    if result != 0:
        abort(f"{label} failed with return code {result}")


def prepare_output_dir(output_root: Path, dataset: str) -> Path:
    output = output_root / dataset
    if output.exists() or output.is_symlink():
        abort(f"fresh formal session output path already exists: {output}")
    output.mkdir(parents=True)
    return output


def _set_stage(session_file: Path, stage: str) -> None:
    update_session_record(session_file, status="RUNNING", stage=stage)


def _preflight_dataset_configs() -> dict[str, dict[str, str]]:
    require_file(ROOT / "resources/canonical/configs/quantization_scale_chain.cfg", "canonical scale chain")
    checked: dict[str, dict[str, str]] = {}
    for dataset in DATASETS:
        config = ROOT / "resources/canonical/configs" / f"{dataset}_secure.cfg"
        values = validate_config(config, dataset)
        data_dir = ROOT / values["cache_root"] / dataset
        meta = parse_cfg(require_file(data_dir / "meta.cfg", f"{dataset} canonical graph metadata"))
        require_file(data_dir / "edges.txt", f"{dataset} canonical edges")
        features = data_dir / "features.npy"
        require_file(features if features.exists() else data_dir / "features.txt", f"{dataset} canonical features")
        try:
            graph_count = int(meta.get("graph_count", "1"))
        except ValueError:
            abort(f"{dataset} graph_count is not an integer")
        if graph_count <= 0:
            abort(f"{dataset} graph_count must be positive")
        if graph_count > 1:
            require_file(data_dir / "node_ptr.txt", f"{dataset} canonical graph boundaries")
        checked[dataset] = values
        print(
            f"[ezkl-formal] dataset preflight PASS: {dataset} config, dataset-scoped metadata, "
            "checkpoint manifest/tensors, graph files, and schema",
            flush=True,
        )
    return checked


def _authenticate_or_prepare_resources(
    python: str,
    *,
    allow_prepare: bool,
    session_id: str,
    preflight_config_sha256s: tuple[tuple[str, str], ...],
) -> tuple[tuple[tuple[str, tuple[int, int, int, int, int]], ...], tuple[tuple[str, str], ...]]:
    try:
        identities_before = resource_file_identities()
        validate_resources()
        identities_after = resource_file_identities()
        if identities_before != identities_after:
            abort("manifest-listed resource files changed during authentication; current session is invalid")
        return identities_after, tuple(preflight_config_sha256s)
    except ContractError as exc:
        if "timed out after" in str(exc) or "changed during authentication" in str(exc):
            raise
        print(f"[ezkl-formal] complete resource validation needs repair: {exc}", flush=True)
        if not allow_prepare:
            raise
    return _prepare_and_authenticate_resources(
        python,
        session_id=session_id,
        preflight_config_sha256s=tuple(preflight_config_sha256s),
    )


def _authenticate_or_provision_srs(python: str, *, allow_provision: bool) -> dict[str, object]:
    try:
        return validate_native_srs_manifest()
    except ContractError as exc:
        print(f"[ezkl-formal] official SRS preflight needs repair: {exc}", flush=True)
        if not allow_provision:
            raise
    provisioner = ROOT / "scripts" / "provision_srs.py"
    require_file(provisioner, "official native SRS provisioner")
    run_prerequisite(
        [python, "-u", str(provisioner)],
        "provision and authenticate official native get-srs material",
        timeout_seconds=21600,
    )
    return validate_native_srs_manifest()


def _run_session(args: argparse.Namespace, *, session_id: str, session_file: Path) -> int:
    binary = local_path(args.binary or f"target/{NATIVE_FORMAL_BACKEND}/release/ezkl", "binary")
    output_root = local_path(args.output_root, "output-root")
    matrix = local_path(args.matrix, "matrix")
    if binary.exists() and binary.is_symlink():
        abort(f"binary must not be a symlink: {binary}")
    if output_root.exists() and output_root.is_symlink():
        abort(f"output-root must not be a symlink: {output_root}")
    if matrix.exists() and matrix.is_symlink():
        abort(f"matrix must not be a symlink: {matrix}")

    python = sys.executable
    resource_validator = ROOT / "scripts" / "validate_resources.py"
    build = ROOT / "scripts" / "build_ezkl.sh"
    producer = ROOT / "scripts" / "produce_public_statement.py"
    single = ROOT / "scripts" / "run_single_dataset.py"
    collect = ROOT / "scripts" / "collect_results.py"
    validate = ROOT / "scripts" / "validate_final.py"
    for path in (resource_validator, build, producer, single, collect, validate):
        if not path.is_file():
            abort(f"missing formal runner component: {path}")

    check_output_paths(output_root, matrix)
    _set_stage(session_file, "runtime_identity")
    runtime_identity = validate_formal_python_environment()
    selected_device = select_formal_cuda_device(runtime_identity["nvidia_smi_path"])
    build_environment = formal_build_environment_identity(runtime_identity, selected_device)
    session_document = json.loads(session_file.read_text())
    session_document["python_environment"] = runtime_identity
    session_document["selected_cuda_device"] = selected_device
    session_document["build_environment"] = build_environment
    _write_session_record(session_file, session_document)
    # Preserve the validated host environment while clearing only the variables
    # that can redirect backend selection; native commands receive explicit
    # source-matched values below.
    for name in (
        "ENABLE_ICICLE_GPU",
        "EZKL_GAT_BACKEND",
        "ICICLE_SMALL_K",
        "ICICLE_SMALL_K_FFT",
        "ICICLE_BACKEND_INSTALL_DIR",
        "ICICLE_NATIVE_LIB_DIR",
        "ICICLE_NATIVE_MANIFEST_SHA256",
        "CUDA_ARCH",
        "EZKL_FORMAL_RESOURCE_AUTH_SESSION_ID",
    ):
        os.environ.pop(name, None)

    _set_stage(session_file, "five_dataset_config_preflight")
    config_values = _preflight_dataset_configs()
    preflight_config_sha256s = tuple(
        (
            dataset,
            sha256(ROOT / "resources/canonical/configs" / f"{dataset}_secure.cfg"),
        )
        for dataset in DATASETS
    )
    _set_stage(session_file, "canonical_resource_authentication")
    authenticated_resource_file_identities, preflight_config_sha256s = _authenticate_or_prepare_resources(
        python,
        allow_prepare=not args.preflight_only,
        session_id=session_id,
        preflight_config_sha256s=preflight_config_sha256s,
    )
    print("[ezkl-formal] authenticated complete five-dataset canonical resources", flush=True)

    _set_stage(session_file, "official_srs_authentication")
    srs_manifest = _authenticate_or_provision_srs(python, allow_provision=not args.preflight_only)
    entries = srs_manifest["entries"]
    shared_paths = {entry["path"] for entry in entries.values()}
    if len(shared_paths) != 1:
        abort("native SRS manifest does not expose one shared official cache")
    srs = ROOT / next(iter(shared_paths))
    if not srs.is_file() or srs.is_symlink():
        abort(f"shared official SRS is missing or symlinked: {srs}")
    print(f"[ezkl-formal] reuse authenticated official SRS: {local_cli(srs, 'SRS')}", flush=True)

    _set_stage(session_file, "gpu_runtime_identity")
    gpu_identity: dict[str, object] | None = None
    if NATIVE_FORMAL_BACKEND == "gpu":
        try:
            gpu_identity = validate_gpu_environment()
            print("[ezkl-formal] native ICICLE/GPU runtime identity PASS", flush=True)
        except ContractError as exc:
            if args.preflight_only:
                print(f"[ezkl-formal] GPU_SETUP_REQUIRED: {exc}", flush=True)
            else:
                run_prerequisite(
                    [
                        "bash", "setup-gpu.sh", "--yes", "--python", python,
                        "--nvcc", runtime_identity["nvcc_path"],
                        "--cmake", runtime_identity["cmake_path"],
                        "--make", runtime_identity["make_path"],
                        "--cc", runtime_identity["cc_path"],
                        "--cxx", runtime_identity["cxx_path"],
                        "--nvidia-smi", runtime_identity["nvidia_smi_path"],
                        "--selected-device-json", json.dumps(selected_device, sort_keys=True, separators=(",", ":")),
                    ],
                    "prepare source-matched native ICICLE CUDA libraries once",
                    timeout_seconds=21600,
                )
                gpu_identity = validate_gpu_environment()
                print("[ezkl-formal] native ICICLE/GPU runtime identity PASS after setup", flush=True)

    _set_stage(session_file, "native_build_identity")
    build_reusable = gpu_identity is not None and reusable_build(binary, NATIVE_FORMAL_BACKEND, runtime_identity, build_environment)
    if not build_reusable:
        print("[ezkl-formal] GPU_BUILD_REQUIRED: no source/resource/runtime-matched formal binary receipt", flush=True)
    else:
        print("[ezkl-formal] source/resource/runtime-matched native build identity PASS", flush=True)

    if args.preflight_only:
        if not build_reusable:
            print("[ezkl-formal] preflight reached GPU build gate; no setup/prove was run", flush=True)
        else:
            print("[ezkl-formal] PREFLIGHT_ONLY_READY: resources/SRS/GPU/build identities passed; no setup/prove was run", flush=True)
        return 0

    if not build_reusable:
        if binary != ROOT / "target" / NATIVE_FORMAL_BACKEND / "release/ezkl":
            abort(f"stale custom --binary cannot be rebuilt; use target/{NATIVE_FORMAL_BACKEND}/release/ezkl")
        build_args = [
            "bash", local_cli(build, "build script"), f"--{NATIVE_FORMAL_BACKEND}", "--python", python,
            "--cargo", str(runtime_identity["cargo_path"]),
            "--rustc", str(runtime_identity["rustc_path"]),
            "--cmake", str(runtime_identity["cmake_path"]),
            "--make", str(runtime_identity["make_path"]),
            "--nvcc", str(runtime_identity["nvcc_path"]),
            "--cc", str(runtime_identity["cc_path"]),
            "--cxx", str(runtime_identity["cxx_path"]),
            "--nvidia-smi", str(runtime_identity["nvidia_smi_path"]),
            "--selected-device-json", json.dumps(selected_device, sort_keys=True, separators=(",", ":")),
            "--build-environment-json", json.dumps(build_environment, sort_keys=True, separators=(",", ":")),
        ]
        run(build_args, "build source-matched official ezkl native binary")
        if not reusable_build(binary, NATIVE_FORMAL_BACKEND, runtime_identity, build_environment):
            abort("native build did not produce a matching binary/resource/adapter receipt")
    else:
        print("[ezkl-formal] reuse source-matched official ezkl native build", flush=True)

    if gpu_identity is None:
        abort("GPU formal execution has no authenticated native ICICLE runtime identity")
    # Parent authentication is process-local and tied to the exact current
    # resource/SRS manifests, configuration digests, and source identity.
    build_receipt = ROOT / "artifacts" / "build" / NATIVE_FORMAL_BACKEND / "build_receipt.json"
    session_document = json.loads(session_file.read_text())
    session_document["build_receipt_sha256"] = sha256(build_receipt)
    session_document["build_environment_sha256"] = build_environment["build_environment_sha256"]
    _write_session_record(session_file, session_document)
    os.environ["EZKL_FORMAL_BUILD_RECEIPT_SHA256"] = session_document["build_receipt_sha256"]
    os.environ["EZKL_FORMAL_BUILD_ENVIRONMENT_SHA256"] = str(session_document["build_environment_sha256"])
    build_receipt_document = load_json(build_receipt, "current formal GPU build receipt")
    if build_receipt_document.get("source_tree_sha256") != source_tree_hash():
        abort("current source tree changed after the formal GPU build; session build identity is stale")
    if build_receipt_document.get("resource_manifest_sha256") != resource_manifest_hash():
        abort("resource manifest changed after the formal GPU build; session build identity is stale")
    context = create_authenticated_formal_context(
        session_id,
        srs_manifest,
        authenticated_resource_file_identities=authenticated_resource_file_identities,
        preflight_config_sha256s=preflight_config_sha256s,
    )
    authenticated_environment = {
        "ICICLE_BACKEND_INSTALL_DIR": str(ROOT / "artifacts" / "build" / "icicle-cuda"),
        "ICICLE_NATIVE_LIB_DIR": str(ROOT / "artifacts" / "build" / "icicle-cuda" / "lib"),
        "ICICLE_NATIVE_MANIFEST_SHA256": gpu_identity["native_backend_manifest_sha256"],
        "CUDA_ARCH": str(gpu_identity["cuda_architecture"]).removeprefix("sm_"),
    }
    os.environ.update({key: str(value) for key, value in authenticated_environment.items()})

    for dataset in DATASETS:
        _set_stage(session_file, f"{dataset}_setup_prove_verify")
        output = prepare_output_dir(output_root, dataset)
        setup = [
            local_cli(binary, "binary"),
            "gat-prepare",
            "--dataset",
            dataset,
            "--phase",
            "setup",
            "--resource-manifest",
            "resources/resource_manifest.json",
            "--output-dir",
            local_cli(output, "output-dir"),
            "--srs-path",
            local_cli(srs, "SRS"),
        ]
        if NATIVE_FORMAL_BACKEND == "gpu":
            setup.append("--gpu")
        setup_ms = run(setup, f"{dataset} native setup", gpu=NATIVE_FORMAL_BACKEND == "gpu")
        (output / "setup_timing.json").write_text(json.dumps({"duration_ms": setup_ms}) + "\n")
        invoke_python_entrypoint(
            "produce_public_statement",
            [
                "--dataset", dataset,
                "--resource-manifest", "resources/resource_manifest.json",
                "--setup-context", local_cli(output / "producer_context.json", "producer setup context"),
                "--output-dir", local_cli(output, "output-dir"),
                "--config", local_cli(ROOT / "resources/canonical/configs" / f"{dataset}_secure.cfg", "config"),
                "--statement", local_cli(output / "statement.json", "public statement"),
                "--public-output", local_cli(output / "public_output.json", "public output"),
                "--producer-manifest", local_cli(output / "producer_manifest.json", "producer manifest"),
            ],
            f"{dataset} independent public statement/Y producer",
            context=context,
            session_id=session_id,
        )
        invoke_python_entrypoint(
            "run_single_dataset",
            [
                "--dataset", dataset,
                "--backend", NATIVE_FORMAL_BACKEND,
                "--execute-formal",
                "--reuse-setup",
                "--config", local_cli(ROOT / "resources/canonical/configs" / f"{dataset}_secure.cfg", "config"),
                "--public-statement", local_cli(output / "statement.json", "public statement"),
                "--public-output", local_cli(output / "public_output.json", "public output"),
                "--producer-manifest", local_cli(output / "producer_manifest.json", "producer manifest"),
                "--binary", local_cli(binary, "binary"),
                "--output-dir", local_cli(output, "output-dir"),
            ],
            f"{dataset} final/witness/prove/independent verify",
            context=context,
            session_id=session_id,
        )

    _set_stage(session_file, "collect_five_results")
    run(
        [
            python,
            local_cli(collect, "result collector"),
            "--runs-root",
            local_cli(output_root, "output-root"),
            "--output",
            local_cli(matrix, "matrix"),
            "--session-id",
            session_id,
        ],
        "collect five-dataset result matrix",
    )
    _set_stage(session_file, "final_matrix_validation")
    run(
        [
            python,
            local_cli(validate, "result validator"),
            "--matrix",
            local_cli(matrix, "matrix"),
            "--session-id",
            session_id,
        ],
        "validate five-dataset result matrix",
    )
    print(f"validated five formal results: {matrix.relative_to(ROOT)}")
    print(f"view: {python} -m json.tool {matrix.relative_to(ROOT)}")
    return 0


def _raise_on_sigterm(signum: int, frame: object) -> None:
    raise KeyboardInterrupt(f"received signal {signum}")


def main(args: argparse.Namespace) -> int:
    previous_sigterm = signal.signal(signal.SIGTERM, _raise_on_sigterm)
    try:
        return _main(args)
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)


def _main(args: argparse.Namespace) -> int:
    binary = local_path(args.binary or f"target/{NATIVE_FORMAL_BACKEND}/release/ezkl", "binary")
    output_root = local_path(args.output_root, "output-root")
    matrix = local_path(args.matrix, "matrix")
    if binary.exists() and binary.is_symlink():
        abort(f"binary must not be a symlink: {binary}")
    if output_root.exists() and output_root.is_symlink():
        abort(f"output-root must not be a symlink: {output_root}")
    if matrix.exists() and matrix.is_symlink():
        abort(f"matrix must not be a symlink: {matrix}")
    check_output_paths(output_root, matrix)

    session_id, session_file, lock_fd, _archived = begin_formal_session(output_root, matrix)
    try:
        result = _run_session(args, session_id=session_id, session_file=session_file)
        update_session_record(
            session_file,
            status="FINAL_VALIDATION_OK" if not args.preflight_only else "PREFLIGHT_ONLY_COMPLETE",
            stage="final_matrix_validation" if not args.preflight_only else "preflight_complete",
        )
        return result
    except KeyboardInterrupt as exc:
        active_stage = json.loads(session_file.read_text()).get("stage", "unknown")
        update_session_record(session_file, status="INTERRUPTED", stage=active_stage, error="KeyboardInterrupt")
        raise
    except BaseException as exc:
        active_stage = json.loads(session_file.read_text()).get("stage", "unknown")
        update_session_record(session_file, status="FAILED", stage=active_stage, error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


if __name__ == "__main__":
    try:
        raise SystemExit(main(parser().parse_args()))
    except (ContractError, OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        raise SystemExit(print_error(exc))
