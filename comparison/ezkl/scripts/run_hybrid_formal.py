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
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


def restart_without_inherited_preload() -> None:
    if not os.environ.get("LD_PRELOAD"):
        return
    clean_environment = dict(os.environ)
    clean_environment.pop("LD_PRELOAD", None)
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--_ld-preload-cleared"]
    os.execve(sys.executable, command, clean_environment)


restart_without_inherited_preload()

from formal_common import (
    CONTRACT,
    DATASETS,
    FORMAL_GPU_FEATURES,
    FORMAL_SESSION_ID,
    NATIVE_FORMAL_BACKEND,
    ROOT,
    SRS_MANIFEST,
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
    require_formal_srs,
    resource_manifest_hash,
    sha256,
    source_tree_hash,
    create_authenticated_formal_context,
    validate_config,
    validate_native_srs_manifest,
    validate_root_owned_phase,
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
    result.add_argument("--internal-phase", choices=("prepare", "cell", "finalize"), help=argparse.SUPPRESS)
    result.add_argument("--session-dir", help=argparse.SUPPRESS)
    result.add_argument("--dataset", help=argparse.SUPPRESS)
    result.add_argument("--_ld-preload-cleared", action="store_true", help=argparse.SUPPRESS)
    return result


def _cgroup_memory_paths() -> tuple[Path, ...]:
    candidates: list[Path] = []
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            hierarchy, _controllers, relative = line.split(":", 2)
            if hierarchy == "0":
                base = Path("/sys/fs/cgroup") / relative.lstrip("/")
                candidates.extend((base, *base.parents))
                break
    except (OSError, ValueError):
        pass
    root = Path("/sys/fs/cgroup")
    candidates.append(root)
    return tuple(dict.fromkeys(candidates))


def _sample_process_resources(pid: int, profile: dict[str, object]) -> None:
    status_path = Path(f"/proc/{pid}/status")
    try:
        status = status_path.read_text()
        values: dict[str, int] = {}
        for line in status.splitlines():
            key, separator, value = line.partition(":")
            if separator and key in {"VmRSS", "VmHWM", "VmSize"}:
                pieces = value.split()
                if pieces and pieces[0].isdigit():
                    values[key] = int(pieces[0]) * 1024
        for key, source in (("process_peak_rss_bytes", "VmHWM"), ("process_peak_vms_bytes", "VmSize")):
            if source in values:
                profile[key] = max(int(profile.get(key, 0)), values[source])
    except OSError:
        pass
    try:
        meminfo = Path("/proc/meminfo").read_text().splitlines()
        host_values = {
            line.split(":", 1)[0]: int(line.split()[1]) * 1024
            for line in meminfo
            if ":" in line and len(line.split()) >= 2 and line.split()[1].isdigit()
        }
        for key, source in (("host_memory_total_bytes", "MemTotal"), ("host_mem_available_bytes_min", "MemAvailable")):
            if source in host_values:
                current = host_values[source]
                if key.endswith("_min"):
                    old = profile.get(key)
                    profile[key] = current if not isinstance(old, int) else min(old, current)
                else:
                    profile[key] = current
    except OSError:
        pass
    cgroup_limits: list[tuple[int, str]] = []
    cgroup_samples: list[dict[str, object]] = []
    for base in _cgroup_memory_paths():
        try:
            current_text = (base / "memory.current").read_text().strip()
            current = int(current_text)
            profile["cgroup_memory_current_peak_bytes"] = max(
                int(profile.get("cgroup_memory_current_peak_bytes", 0)), current
            )
            sample: dict[str, object] = {"path": str(base), "current_bytes": current}
            for name in ("memory.peak", "memory.low", "memory.high", "memory.min", "memory.max"):
                try:
                    value = (base / name).read_text().strip()
                except OSError:
                    continue
                sample[name.replace(".", "_")] = value if value == "max" else int(value)
                if name == "memory.max" and value != "max":
                    cgroup_limits.append((int(value), str(base)))
            try:
                memory_stat = {}
                for line in (base / "memory.stat").read_text().splitlines():
                    key, value = line.split()
                    if key in {"anon", "file", "inactive_file", "active_file", "slab", "slab_reclaimable", "slab_unreclaimable"}:
                        memory_stat[key] = int(value)
                sample["memory_stat"] = memory_stat
            except (OSError, ValueError):
                pass
            try:
                events = {}
                for line in (base / "memory.events").read_text().splitlines():
                    key, value = line.split()
                    events[key] = int(value)
                sample["memory_events"] = events
            except (OSError, ValueError):
                pass
            previous_ancestors = profile.get("cgroup_memory_ancestors")
            if isinstance(previous_ancestors, list):
                for previous in previous_ancestors:
                    if not isinstance(previous, dict) or previous.get("path") != str(base):
                        continue
                    old_current_peak = previous.get("observed_current_peak_bytes", previous.get("current_bytes", 0))
                    if isinstance(old_current_peak, int):
                        sample["observed_current_peak_bytes"] = max(current, old_current_peak)
                    old_stat = previous.get("memory_stat")
                    new_stat = sample.get("memory_stat")
                    if isinstance(old_stat, dict) and isinstance(new_stat, dict):
                        sample["memory_stat_peak"] = {
                            key: max(value, old_stat.get(key, 0))
                            for key, value in new_stat.items()
                            if isinstance(value, int) and isinstance(old_stat.get(key, 0), int)
                        }
                    break
            cgroup_samples.append(sample)
        except (OSError, ValueError):
            continue
    profile["cgroup_memory_ancestors"] = cgroup_samples
    if cgroup_limits:
        limit, source = min(cgroup_limits)
        profile["effective_memory_limit_bytes"] = limit
        profile["effective_memory_limit_source"] = source


def run(
    command: list[str],
    label: str,
    *,
    gpu: bool = False,
    log_path: Path | None = None,
    memory_profile: dict[str, object] | None = None,
) -> float:
    print(f"[ezkl-formal] {label}: {' '.join(command)}", flush=True)
    log_stream = None
    if log_path is not None:
        if log_path.is_symlink() or (log_path.exists() and not log_path.is_file()):
            abort(f"formal command log path is not a regular file: {log_path}")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_stream = log_path.open("a", encoding="utf-8")
        print(f"[ezkl-formal] independent build log: {log_path}", flush=True)
    env = os.environ.copy()
    for key in (
        "ENABLE_ICICLE_GPU",
        "EZKL_GAT_BACKEND",
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
        env["ICICLE_BACKEND_INSTALL_DIR"] = str(install_dir)
        env["ICICLE_NATIVE_LIB_DIR"] = str(install_dir / "lib")
        env["ICICLE_NATIVE_MANIFEST_SHA256"] = identity["native_backend_manifest_sha256"]
        env["CUDA_ARCH"] = str(identity["cuda_architecture"]).removeprefix("sm_")
    started = time.monotonic()
    try:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=(os.name == "posix"))
    except BaseException:
        if log_stream is not None:
            log_stream.close()
        raise
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
    monitor_stop = threading.Event()
    monitor = None
    if memory_profile is not None:
        memory_profile.update({
            "schema_version": 1,
            "process_pid": process.pid,
            "process_peak_rss_bytes": 0,
            "process_peak_vms_bytes": 0,
            "host_memory_total_bytes": 0,
            "host_mem_available_bytes_min": None,
            "effective_memory_limit_bytes": None,
            "effective_memory_limit_source": "host",
            "cgroup_memory_current_peak_bytes": 0,
        })
        _sample_process_resources(process.pid, memory_profile)

        def sample_until_stop() -> None:
            while not monitor_stop.wait(0.5):
                _sample_process_resources(process.pid, memory_profile)

        monitor = threading.Thread(target=sample_until_stop, name="ezkl-formal-memory-profile", daemon=True)
        monitor.start()
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
            if log_stream is not None:
                log_stream.write(line)
                log_stream.flush()
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
        monitor_stop.set()
        if monitor is not None:
            monitor.join(timeout=2)
        if log_stream is not None:
            log_stream.close()
        raise
    monitor_stop.set()
    if monitor is not None:
        monitor.join(timeout=2)
        _sample_process_resources(process.pid, memory_profile)
    if log_stream is not None:
        log_stream.close()
    if returncode != 0:
        abort(f"{label} failed with exit code {returncode}")
    return (time.monotonic() - started) * 1000.0


def run_prerequisite(command: list[str], label: str, *, timeout_seconds: int, log_path: Path | None = None) -> None:
    print(f"[ezkl-formal] prerequisite {label} (timeout={timeout_seconds}s): {' '.join(command)}", flush=True)
    log_stream = None
    if log_path is not None:
        if log_path.is_symlink() or (log_path.exists() and not log_path.is_file()):
            abort(f"formal prerequisite log path is not a regular file: {log_path}")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_stream = log_path.open("a", encoding="utf-8")
        print(f"[ezkl-formal] independent build log: {log_path}", flush=True)
    try:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=log_stream if log_stream is not None else None,
            stderr=subprocess.STDOUT if log_stream is not None else None,
            start_new_session=(os.name == "posix"),
        )
    except BaseException:
        if log_stream is not None:
            log_stream.close()
        raise
    try:
        deadline = time.monotonic() + timeout_seconds
        while True:
            try:
                returncode = process.wait(timeout=min(20, max(0.1, deadline - time.monotonic())))
                break
            except subprocess.TimeoutExpired:
                if time.monotonic() >= deadline:
                    raise
                print(f"[ezkl-formal] prerequisite {label} still running", flush=True)
    except subprocess.TimeoutExpired:
        _terminate_process_group(process)
        if log_stream is not None:
            log_stream.close()
        abort(f"prerequisite {label} timed out after {timeout_seconds}s; process group was terminated")
    except KeyboardInterrupt:
        print(f"[ezkl-formal] interrupted during prerequisite {label}; terminating process group", file=sys.stderr, flush=True)
        _terminate_process_group(process)
        if log_stream is not None:
            log_stream.close()
        raise
    except BaseException:
        print(f"[ezkl-formal] prerequisite {label} failed; terminating process group", file=sys.stderr, flush=True)
        _terminate_process_group(process)
        if log_stream is not None:
            log_stream.close()
        raise
    if log_stream is not None:
        log_stream.close()
    if returncode != 0:
        # The leader may have exited while helpers in its private process
        # group remain alive. Reap the whole group before reporting its real
        # exit status so no failed prerequisite can leak background work.
        _terminate_process_group(process)
        abort(f"prerequisite {label} failed with exit code {returncode}")


def _build_log_path(session_id: str) -> Path:
    return ROOT / "logs" / "formal_builds" / f"ezkl_gpu_release_{session_id}.log"


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
        "rustc_path": runtime_identity["rustc_path"], "cargo_path": runtime_identity["cargo_path"],
        "rustdoc": runtime_identity["rustdoc"], "rustdoc_path": runtime_identity["rustdoc_path"],
    }
    if backend == "gpu":
        if build_environment is None:
            abort("GPU build receipt check has no resolved formal build environment")
        expected.update(native_backend_identity())
        expected["build_environment"] = build_environment
        expected["build_environment_sha256"] = build_environment["build_environment_sha256"]
    return (
        receipt.get("schema_version") == 2
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


def begin_formal_session(output_root: Path, matrix: Path) -> tuple[str, Path, int]:
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
                "archived_previous_outputs": [],
            },
        )
        lock_payload = f"pid={os.getpid()} session_id={session_id}\n".encode()
        os.ftruncate(lock_fd, 0)
        os.write(lock_fd, lock_payload)
        os.fsync(lock_fd)
        print(f"[ezkl-formal] started formal session {session_id}; prior outputs remain intact until admission passes", flush=True)
        return session_id, session_file, lock_fd
    except BaseException:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
        raise


def archive_prior_outputs(output_root: Path, matrix: Path, session_id: str, session_file: Path) -> None:
    """Move old active outputs only after all prerequisite admissions pass."""
    for dataset in DATASETS:
        old_output = output_root / dataset
        if (old_output.exists() or old_output.is_symlink()) and (old_output.is_symlink() or not old_output.is_dir()):
            abort(f"prior formal output is not a regular directory: {old_output}")
    if (matrix.exists() or matrix.is_symlink()) and (matrix.is_symlink() or not matrix.is_file()):
        abort(f"prior formal matrix is not a regular file: {matrix}")
    archive_root = output_root / ".incomplete" / session_id / "previous"
    archive_root.mkdir(parents=True, exist_ok=False)
    archived: list[str] = []
    for dataset in DATASETS:
        old_output = output_root / dataset
        if old_output.exists():
            destination = archive_root / "datasets" / dataset
            destination.parent.mkdir(parents=True, exist_ok=True)
            old_output.rename(destination)
            archived.append(str(destination.relative_to(ROOT)))
            print(f"[ezkl-formal] archived prior result directory: {old_output} -> {destination}", flush=True)
    if matrix.exists():
        matrix_archive = archive_root / "matrix" / matrix.name
        matrix_archive.parent.mkdir(parents=True, exist_ok=True)
        matrix.rename(matrix_archive)
        archived.append(str(matrix_archive.relative_to(ROOT)))
        print(f"[ezkl-formal] archived prior active matrix: {matrix} -> {matrix_archive}", flush=True)
    document = json.loads(session_file.read_text())
    document["archived_previous_outputs"] = archived
    document["archived_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_session_record(session_file, document)


def native_admit_all_datasets(
    binary: Path,
    srs: Path,
    output_root: Path,
    session_id: str,
) -> dict[str, dict[str, object]]:
    admission_root = output_root / ".sessions" / session_id / "admission"
    admission_root.mkdir(parents=True, exist_ok=False)
    admissions: dict[str, dict[str, object]] = {}
    expected_keys = {
        "schema_version", "relation_id", "dataset", "model_arch_id", "quant_cfg_id", "logrows",
        "domain_size", "max_public_srs_logrows", "num_rows", "total_assignments", "total_const_size",
        "total_dynamic_col_size", "max_dynamic_input_len", "num_dynamic_lookups", "num_shuffles",
        "module_sizes", "constraint_system", "scalar_domain_vector_bytes", "resource_manifest_sha256",
        "relation_identity", "settings_sha256", "compiled_circuit_sha256", "native_memory_model",
    }
    cs_keys = {
        "degree", "num_fixed_columns", "num_advice_columns", "num_instance_columns", "num_selectors",
        "num_lookup_arguments", "num_shuffle_arguments", "num_permutation_columns", "num_blinding_factors",
    }
    for dataset in DATASETS:
        admission_dir = admission_root / dataset
        admission_dir.mkdir()
        command = [
            local_cli(binary, "binary"), "gat-prepare", "--dataset", dataset, "--phase", "admission",
            "--resource-manifest", "resources/resource_manifest.json", "--output-dir",
            local_cli(admission_dir, "admission output"), "--srs-path", local_cli(srs, "SRS"), "--gpu",
        ]
        run(command, f"{dataset} native full-relation/capacity admission", gpu=True)
        admission_path = admission_dir / "admission.json"
        admission = load_json(admission_path, f"{dataset} native admission")
        if not isinstance(admission, dict) or set(admission) != expected_keys:
            abort(f"{dataset} native admission JSON has an incompatible schema")
        if (
            admission["schema_version"] != 2
            or admission["relation_id"] != "R_GAT-ezkl"
            or admission["dataset"] != dataset
            or isinstance(admission["logrows"], bool)
            or not isinstance(admission["logrows"], int)
            or admission["logrows"] <= 0
            or admission["logrows"] > admission["max_public_srs_logrows"]
            or admission["domain_size"] != 1 << admission["logrows"]
        ):
            abort(f"{dataset} native admission failed relation/logrows/SRS checks")
        relation_identity = admission["relation_identity"]
        if (
            not isinstance(relation_identity, dict)
            or relation_identity.get("dataset") != dataset
            or relation_identity.get("resource_manifest_sha256") != resource_manifest_hash()
            or admission["resource_manifest_sha256"] != resource_manifest_hash()
            or not isinstance(admission["settings_sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", admission["settings_sha256"]) is None
            or not isinstance(admission["compiled_circuit_sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", admission["compiled_circuit_sha256"]) is None
        ):
            abort(f"{dataset} native admission does not bind its canonical resources and handoff files")
        for name, expected_digest in (
            ("settings.json", admission["settings_sha256"]),
            ("compiled.circuit", admission["compiled_circuit_sha256"]),
        ):
            handoff_file = admission_dir / name
            require_file(handoff_file, f"{dataset} admitted {name}")
            if sha256(handoff_file) != expected_digest:
                abort(f"{dataset} native admission {name} digest mismatch")
        cs = admission["constraint_system"]
        if not isinstance(cs, dict) or set(cs) != cs_keys or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in cs.values()
        ):
            abort(f"{dataset} native admission has malformed exact Halo2 constraint-shape counts")
        memory = admission["native_memory_model"]
        memory_keys = {
            "schema", "proof_logrows", "proof_domain_size", "advice_column_phase_ids",
            "size_of_fr_bytes", "size_of_assigned_fr_bytes", "size_of_option_fr_bytes",
            "size_of_usize_bytes", "size_of_usize_pair_bytes", "size_of_g1_affine_bytes",
            "cuda_uuid", "cuda_free_bytes_at_admission", "cuda_total_memory_bytes",
        }
        if not isinstance(memory, dict) or set(memory) != memory_keys or memory.get("schema") != "ezkl-halo2-memory-model-v1":
            abort(f"{dataset} native admission has an incompatible source memory model")
        positive_memory_keys = {
            "proof_logrows", "proof_domain_size", "size_of_fr_bytes",
            "size_of_assigned_fr_bytes", "size_of_option_fr_bytes", "size_of_usize_bytes",
            "size_of_usize_pair_bytes", "size_of_g1_affine_bytes",
            "cuda_total_memory_bytes",
        }
        if any(
            isinstance(memory.get(key), bool) or not isinstance(memory.get(key), int) or memory[key] <= 0
            for key in positive_memory_keys
        ) or (
            isinstance(memory.get("cuda_free_bytes_at_admission"), bool)
            or not isinstance(memory.get("cuda_free_bytes_at_admission"), int)
            or memory["cuda_free_bytes_at_admission"] < 0
        ):
            abort(f"{dataset} native admission has non-positive or malformed source memory values")
        if (
            memory["proof_domain_size"] != 1 << memory["proof_logrows"]
            or memory["proof_logrows"] < admission["logrows"]
            or memory["proof_domain_size"] < admission["domain_size"]
            or memory["size_of_usize_pair_bytes"] != 2 * memory["size_of_usize_bytes"]
            or memory["size_of_option_fr_bytes"] < memory["size_of_fr_bytes"]
            or memory["cuda_free_bytes_at_admission"] > memory["cuda_total_memory_bytes"]
            or not isinstance(memory["cuda_uuid"], str)
            or re.fullmatch(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", memory["cuda_uuid"]) is None
        ):
            abort(f"{dataset} native admission memory model is internally inconsistent")
        phase_ids = memory["advice_column_phase_ids"]
        if (
            not isinstance(phase_ids, list)
            or len(phase_ids) != cs["num_advice_columns"]
            or any(isinstance(phase, bool) or not isinstance(phase, int) or phase < 0 for phase in phase_ids)
        ):
            abort(f"{dataset} native admission advice phase layout is malformed")
        admissions[dataset] = {
            "path": str(admission_path.relative_to(ROOT)),
            "sha256": sha256(admission_path),
            "prepared_dir": str(admission_dir.relative_to(ROOT)),
            "settings_sha256": admission["settings_sha256"],
            "compiled_circuit_sha256": admission["compiled_circuit_sha256"],
            "logrows": admission["logrows"],
            "domain_size": admission["domain_size"],
            "constraint_system": cs,
            "native_memory_model": memory,
            "relation_identity": relation_identity,
        }
        print(
            f"[ezkl-formal] native relation admission PASS: {dataset} k={admission['logrows']} "
            f"N={admission['domain_size']} advice={cs['num_advice_columns']} "
            f"fixed={cs['num_fixed_columns']} instance={cs['num_instance_columns']} "
            f"lookups={cs['num_lookup_arguments']} permutation_columns={cs['num_permutation_columns']}",
            flush=True,
        )
    return admissions


def _shared_graph_identity_from_admissions(admissions: dict[str, dict[str, object]]) -> dict[str, object]:
    datasets: dict[str, object] = {}
    for dataset in DATASETS:
        record = admissions.get(dataset)
        relation = record.get("relation_identity") if isinstance(record, dict) else None
        graph = relation.get("graph") if isinstance(relation, dict) else None
        if not isinstance(relation, dict) or not isinstance(graph, dict):
            abort(f"{dataset} native admission lacks consumed graph identity")
        digest = graph.get("canonical_graph_digest")
        nodes, edges, graphs = graph.get("n_total"), graph.get("e"), graph.get("g_batch")
        node_ptr, edge_ptr = graph.get("node_ptr"), graph.get("edge_ptr")
        preprocess = relation.get("input_preprocess_id")
        batching = relation.get("batching_rule")
        if (
            not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in (nodes, edges, graphs))
            or not isinstance(node_ptr, list) or not isinstance(edge_ptr, list)
            or len(node_ptr) != graphs + 1 or len(edge_ptr) != graphs + 1
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in node_ptr + edge_ptr)
            or node_ptr[0] != 0 or node_ptr[-1] != nodes
            or edge_ptr[0] != 0 or edge_ptr[-1] != edges
            or any(left >= right for left, right in zip(node_ptr, node_ptr[1:]))
            or any(left >= right for left, right in zip(edge_ptr, edge_ptr[1:]))
            or not isinstance(preprocess, str) or not preprocess
            or batching != ("multi_graph_batch" if dataset == "ppi" else "whole_graph_single")
        ):
            abort(f"{dataset} native admission graph identity is malformed")
        datasets[dataset] = {
            "digest": digest,
            "nodes": nodes,
            "edges": edges,
            "graphs": graphs,
            "node_ptr": node_ptr,
            "edge_ptr": edge_ptr,
            "input_preprocess_id": preprocess,
            "batching_rule": batching,
        }
    return {"schema": "comparison-canonical-graph-set-v1", "datasets": datasets}


_INTERNAL_SESSION_FIELDS = {
    "schema_version", "baseline", "root_session_id", "session_id", "status", "stage",
    "started_at_utc", "updated_at_utc", "output_root", "matrix", "python_environment",
    "selected_cuda_device", "build_environment", "gpu_identity", "build_receipt_path",
    "build_receipt_sha256", "binary_path", "binary_sha256", "source_tree_sha256",
    "build_log_path", "build_log_sha256",
    "resource_manifest_sha256", "resource_file_identities", "config_sha256s",
    "srs_manifest_sha256", "srs_path", "srs_sha256", "srs_file_identity",
    "native_admissions", "archived_previous_outputs", "cell_receipts", "active_dataset",
    "final_matrix_path", "final_matrix_sha256", "shared_capacity_profile", "shared_graph_identity", "error",
}


def _internal_session_location(value: str) -> tuple[Path, str]:
    path = Path(value)
    if not path.is_absolute():
        abort("internal --session-dir must be an absolute path")
    if path.name != "ezkl" or re.fullmatch(r"[0-9a-f]{32}", path.parent.name) is None:
        abort("internal EZKL session directory must be <comparison-session>/<32-hex-id>/ezkl")
    comparison_root = ROOT.parent.resolve()
    resolved = path.resolve(strict=False)
    try:
        relative = resolved.relative_to(comparison_root)
    except ValueError:
        abort("internal EZKL session directory must be inside the comparison tree")
    if tuple(relative.parts[-4:]) != ("runs", "formal", path.parent.name, "ezkl"):
        abort("internal EZKL session directory must be comparison/runs/formal/<32-hex-id>/ezkl")
    cursor = comparison_root
    for component in relative.parts:
        cursor = cursor / component
        if cursor.is_symlink():
            abort(f"internal session path contains a symlink: {cursor}")
    if resolved != path:
        abort("internal EZKL session directory is not canonical")
    return path, path.parent.name


def _new_internal_session_document(
    session_id: str,
    *,
    output_root: Path,
    matrix: Path,
) -> dict[str, object]:
    timestamp = datetime.now(timezone.utc).isoformat()
    document: dict[str, object] = {
        "schema_version": 1,
        "baseline": "ezkl",
        "root_session_id": session_id,
        "session_id": session_id,
        "status": "PREPARING",
        "stage": "session_created",
        "started_at_utc": timestamp,
        "updated_at_utc": timestamp,
        "output_root": output_root.relative_to(ROOT).as_posix(),
        "matrix": matrix.relative_to(ROOT).as_posix(),
        "python_environment": None,
        "selected_cuda_device": None,
        "build_environment": None,
        "gpu_identity": None,
        "build_receipt_path": None,
        "build_receipt_sha256": None,
        "build_log_path": None,
        "build_log_sha256": None,
        "binary_path": None,
        "binary_sha256": None,
        "source_tree_sha256": None,
        "resource_manifest_sha256": None,
        "resource_file_identities": None,
        "config_sha256s": None,
        "srs_manifest_sha256": None,
        "srs_path": None,
        "srs_sha256": None,
        "srs_file_identity": None,
        "native_admissions": None,
        "archived_previous_outputs": [],
        "cell_receipts": {},
        "active_dataset": None,
        "final_matrix_path": None,
        "final_matrix_sha256": None,
        "shared_capacity_profile": None,
        "shared_graph_identity": None,
        "error": None,
    }
    return document


def _load_internal_session(path: Path, session_id: str) -> dict[str, object]:
    document = load_json(path, "EZKL internal formal session")
    if set(document) != _INTERNAL_SESSION_FIELDS:
        abort("EZKL internal session schema mismatch")
    if (
        document.get("schema_version") != 1
        or document.get("baseline") != "ezkl"
        or document.get("root_session_id") != session_id
        or document.get("session_id") != session_id
    ):
        abort("EZKL internal session identity/schema mismatch")
    return document


def _save_internal_session(path: Path, document: dict[str, object], *, status: str, stage: str) -> None:
    document["status"] = status
    document["stage"] = stage
    document["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_session_record(path, document)


def _formal_gpu_environment(session_id: str) -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
    runtime_identity = validate_formal_python_environment()
    selected_device = select_formal_cuda_device(str(runtime_identity["nvidia_smi_path"]))
    os.environ["EZKL_FORMAL_SELECTED_CUDA_DEVICE"] = json.dumps(
        selected_device, sort_keys=True, separators=(",", ":")
    )
    build_environment = formal_build_environment_identity(runtime_identity, selected_device)
    for name in (
        "ENABLE_ICICLE_GPU", "EZKL_GAT_BACKEND", "ICICLE_SMALL_K", "ICICLE_SMALL_K_FFT",
        "ICICLE_BACKEND_INSTALL_DIR", "ICICLE_NATIVE_LIB_DIR", "ICICLE_NATIVE_MANIFEST_SHA256",
        "CUDA_ARCH", "EZKL_FORMAL_RESOURCE_AUTH_SESSION_ID",
    ):
        os.environ.pop(name, None)
    try:
        gpu_identity = validate_gpu_environment()
    except ContractError:
        gpu_identity = {}
    return runtime_identity, selected_device, build_environment, gpu_identity


def _serialized_resource_identities(
    identities: tuple[tuple[str, tuple[int, int, int, int, int]], ...],
) -> list[list[object]]:
    return [[name, list(identity)] for name, identity in identities]


def _parsed_resource_identities(value: object) -> tuple[tuple[str, tuple[int, int, int, int, int]], ...]:
    if not isinstance(value, list):
        abort("EZKL session resource identities are malformed")
    parsed: list[tuple[str, tuple[int, int, int, int, int]]] = []
    for record in value:
        if (
            not isinstance(record, list) or len(record) != 2 or not isinstance(record[0], str)
            or not isinstance(record[1], list) or len(record[1]) != 5
            or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in record[1])
        ):
            abort("EZKL session resource identity entry is malformed")
        parsed.append((record[0], tuple(record[1])))
    if tuple(name for name, _ in parsed) != tuple(sorted(name for name, _ in parsed)) or len({name for name, _ in parsed}) != len(parsed):
        abort("EZKL session resource identities are not strictly sorted and unique")
    return tuple(parsed)


def _internal_prepare(args: argparse.Namespace, session_dir: Path, session_id: str, session_file: Path) -> None:
    binary = local_path(args.binary or f"target/{NATIVE_FORMAL_BACKEND}/release/ezkl", "binary")
    output_root = local_path(args.output_root, "output-root")
    matrix = local_path(args.matrix, "matrix")
    if binary.exists() and binary.is_symlink():
        abort(f"binary must not be a symlink: {binary}")
    if output_root.exists() and output_root.is_symlink():
        abort(f"output-root must not be a symlink: {output_root}")
    if matrix.exists() and matrix.is_symlink():
        abort(f"matrix must not be a symlink: {matrix}")
    if binary != ROOT / "target/gpu/release/ezkl":
        abort("internal formal prepare requires the fixed target/gpu/release/ezkl binary")
    check_output_paths(output_root, matrix)
    if session_file.exists() or session_file.is_symlink():
        abort(f"EZKL internal session already exists and cannot be reused: {session_file}")
    if session_dir.exists():
        if session_dir.is_symlink() or not session_dir.is_dir() or any(session_dir.iterdir()):
            abort(f"EZKL internal session directory must be empty before prepare: {session_dir}")
    else:
        session_dir.mkdir(parents=True, exist_ok=False)
    document = _new_internal_session_document(session_id, output_root=output_root, matrix=matrix)
    _write_session_record(session_file, document)

    _set_stage(session_file, "runtime_identity")
    runtime_identity, selected_device, build_environment, gpu_identity = _formal_gpu_environment(session_id)
    document["python_environment"] = runtime_identity
    document["selected_cuda_device"] = selected_device
    document["build_environment"] = build_environment
    document["gpu_identity"] = gpu_identity
    _write_session_record(session_file, document)

    _set_stage(session_file, "five_dataset_config_preflight")
    _preflight_dataset_configs()
    preflight_config_sha256s = tuple(
        (dataset, sha256(ROOT / "resources/canonical/configs" / f"{dataset}_secure.cfg"))
        for dataset in DATASETS
    )
    _set_stage(session_file, "canonical_resource_authentication")
    authenticated_resource_file_identities, preflight_config_sha256s = _authenticate_or_prepare_resources(
        sys.executable,
        allow_prepare=True,
        session_id=session_id,
        preflight_config_sha256s=preflight_config_sha256s,
    )
    authenticated_resource_manifest_sha256 = resource_manifest_hash()
    print("[ezkl-formal] authenticated complete five-dataset canonical resources", flush=True)

    _set_stage(session_file, "official_srs_authentication")
    srs_manifest = _authenticate_or_provision_srs(sys.executable, allow_provision=True)
    entries = srs_manifest["entries"]
    shared_paths = {entry["path"] for entry in entries.values()}
    if len(shared_paths) != 1:
        abort("native SRS manifest does not expose one shared official cache")
    srs = ROOT / next(iter(shared_paths))
    if not srs.is_file() or srs.is_symlink():
        abort(f"shared official SRS is missing or symlinked: {srs}")

    _set_stage(session_file, "gpu_runtime_identity")
    try:
        gpu_identity = validate_gpu_environment()
    except ContractError as exc:
        run_prerequisite(
            [
                "bash", "setup-gpu.sh", "--yes", "--python", sys.executable,
                "--nvcc", runtime_identity["nvcc_path"], "--cmake", runtime_identity["cmake_path"],
                "--make", runtime_identity["make_path"], "--cc", runtime_identity["cc_path"],
                "--cxx", runtime_identity["cxx_path"], "--nvidia-smi", runtime_identity["nvidia_smi_path"],
                "--selected-device-json", json.dumps(selected_device, sort_keys=True, separators=(",", ":")),
            ],
            "prepare source-matched native ICICLE CUDA libraries once",
            timeout_seconds=21600,
            log_path=_build_log_path(session_id),
        )
        gpu_identity = validate_gpu_environment()
        print(f"[ezkl-formal] native ICICLE/GPU runtime identity PASS after setup ({exc})", flush=True)

    _set_stage(session_file, "native_build_identity")
    os.environ["CARGO_BUILD_JOBS"] = "2"
    build_args = [
        "bash", local_cli(ROOT / "scripts/build_ezkl.sh", "build script"), f"--{NATIVE_FORMAL_BACKEND}",
        "--python", sys.executable, "--cargo", str(runtime_identity["cargo_path"]),
        "--rustc", str(runtime_identity["rustc_path"]), "--rustdoc", str(runtime_identity["rustdoc_path"]),
        "--cmake", str(runtime_identity["cmake_path"]),
        "--make", str(runtime_identity["make_path"]), "--nvcc", str(runtime_identity["nvcc_path"]),
        "--cc", str(runtime_identity["cc_path"]), "--cxx", str(runtime_identity["cxx_path"]),
        "--nvidia-smi", str(runtime_identity["nvidia_smi_path"]),
        "--selected-device-json", json.dumps(selected_device, sort_keys=True, separators=(",", ":")),
        "--build-environment-json", json.dumps(build_environment, sort_keys=True, separators=(",", ":")),
    ]
    build_log = _build_log_path(session_id)
    run(build_args, "build source-matched official ezkl native binary", log_path=build_log)
    if not reusable_build(binary, NATIVE_FORMAL_BACKEND, runtime_identity, build_environment):
        abort("native build did not produce a matching binary/resource/adapter receipt")
    if not build_log.is_file() or build_log.is_symlink() or build_log.stat().st_size == 0:
        abort("formal native build did not produce a nonempty session-specific build log")

    build_receipt = ROOT / "artifacts/build/gpu/build_receipt.json"
    receipt = load_json(build_receipt, "current formal GPU build receipt")
    current_source_sha256 = source_tree_hash()
    current_binary_sha256 = sha256(binary)
    current_build_receipt_sha256 = sha256(build_receipt)
    current_build_log_sha256 = sha256(build_log)
    if receipt.get("source_tree_sha256") != current_source_sha256:
        abort("current source tree changed after the formal GPU build; session build identity is stale")
    if receipt.get("resource_manifest_sha256") != resource_manifest_hash():
        abort("resource manifest changed after the formal GPU build; session resource identity is stale")
    if receipt.get("binary_sha256") != current_binary_sha256:
        abort("formal GPU build receipt does not match the binary used for native admission")
    srs_stat = srs.stat()
    current_srs_file_identity = (
        srs_stat.st_dev, srs_stat.st_ino, srs_stat.st_size, srs_stat.st_mtime_ns, srs_stat.st_ctime_ns,
    )

    context = create_authenticated_formal_context(
        session_id,
        srs_manifest,
        authenticated_resource_file_identities=authenticated_resource_file_identities,
        preflight_config_sha256s=preflight_config_sha256s,
    )
    os.environ.update({
        "ICICLE_BACKEND_INSTALL_DIR": str(ROOT / "artifacts/build/icicle-cuda"),
        "ICICLE_NATIVE_LIB_DIR": str(ROOT / "artifacts/build/icicle-cuda/lib"),
        "ICICLE_NATIVE_MANIFEST_SHA256": str(gpu_identity["native_backend_manifest_sha256"]),
        "CUDA_ARCH": str(gpu_identity["cuda_architecture"]).removeprefix("sm_"),
        "EZKL_FORMAL_BUILD_RECEIPT_SHA256": sha256(build_receipt),
        "EZKL_FORMAL_BUILD_ENVIRONMENT_SHA256": str(build_environment["build_environment_sha256"]),
    })

    document = _load_internal_session(session_file, session_id)
    document.update({
        "python_environment": runtime_identity,
        "selected_cuda_device": selected_device,
        "build_environment": build_environment,
        "gpu_identity": gpu_identity,
        "build_receipt_path": build_receipt.relative_to(ROOT).as_posix(),
        "build_receipt_sha256": current_build_receipt_sha256,
        "build_log_path": build_log.relative_to(ROOT).as_posix(),
        "build_log_sha256": current_build_log_sha256,
        "binary_path": binary.relative_to(ROOT).as_posix(),
        "binary_sha256": current_binary_sha256,
        "source_tree_sha256": current_source_sha256,
        "resource_manifest_sha256": authenticated_resource_manifest_sha256,
        "resource_file_identities": _serialized_resource_identities(authenticated_resource_file_identities),
        "config_sha256s": [[dataset, digest] for dataset, digest in preflight_config_sha256s],
        "srs_manifest_sha256": sha256(SRS_MANIFEST),
        "srs_path": srs.relative_to(ROOT).as_posix(),
        "srs_sha256": entries[DATASETS[0]]["sha256"],
        "srs_file_identity": list(current_srs_file_identity),
    })
    _write_session_record(session_file, document)

    _set_stage(session_file, "five_dataset_native_relation_capacity_admission")
    admissions = native_admit_all_datasets(binary, srs, output_root, session_id)
    shared_graph_identity = _shared_graph_identity_from_admissions(admissions)
    if (
        source_tree_hash() != current_source_sha256
        or sha256(binary) != current_binary_sha256
        or sha256(build_receipt) != current_build_receipt_sha256
        or sha256(build_log) != current_build_log_sha256
        or resource_manifest_hash() != authenticated_resource_manifest_sha256
        or resource_file_identities() != authenticated_resource_file_identities
        or tuple(
            (dataset, sha256(ROOT / "resources/canonical/configs" / f"{dataset}_secure.cfg"))
            for dataset in DATASETS
        ) != tuple(preflight_config_sha256s)
    ):
        abort("source, build, resource, or config identity changed during five-dataset native admission")
    srs_stat_after = srs.stat()
    if (
        srs_stat_after.st_dev,
        srs_stat_after.st_ino,
        srs_stat_after.st_size,
        srs_stat_after.st_mtime_ns,
        srs_stat_after.st_ctime_ns,
    ) != current_srs_file_identity:
        abort("official SRS identity changed during five-dataset native admission")
    document = _load_internal_session(session_file, session_id)
    document.update({
        "python_environment": runtime_identity,
        "selected_cuda_device": selected_device,
        "build_environment": build_environment,
        "gpu_identity": gpu_identity,
        "build_receipt_path": build_receipt.relative_to(ROOT).as_posix(),
        "build_receipt_sha256": current_build_receipt_sha256,
        "build_log_path": build_log.relative_to(ROOT).as_posix(),
        "build_log_sha256": current_build_log_sha256,
        "binary_path": binary.relative_to(ROOT).as_posix(),
        "binary_sha256": current_binary_sha256,
        "source_tree_sha256": current_source_sha256,
        "resource_manifest_sha256": resource_manifest_hash(),
        "resource_file_identities": _serialized_resource_identities(authenticated_resource_file_identities),
        "config_sha256s": [[dataset, digest] for dataset, digest in preflight_config_sha256s],
        "srs_manifest_sha256": sha256(SRS_MANIFEST),
        "srs_path": srs.relative_to(ROOT).as_posix(),
        "srs_sha256": entries[DATASETS[0]]["sha256"],
        "srs_file_identity": list(current_srs_file_identity),
        "native_admissions": admissions,
        "shared_graph_identity": shared_graph_identity,
    })
    _write_session_record(session_file, document)
    if document.get("shared_capacity_profile") is None:
        abort(
            "full setup/witness/commitment/prove/verify capacity bounds are unavailable; "
            "refusing to emit a successful EZKL prepare marker"
        )
    archive_prior_outputs(output_root, matrix, session_id, session_file)
    document = _load_internal_session(session_file, session_id)
    _save_internal_session(session_file, document, status="PREPARED", stage="prepare_complete")
    print(
        f"EZKL_INTERNAL_PHASE_OK phase=prepare session_id={session_id} "
        + json.dumps({
            "status": "PREPARED",
            "session_path": session_file.relative_to(ROOT.parent).as_posix(),
            "build_log_path": build_log.relative_to(ROOT).as_posix(),
            "build_log_sha256": sha256(build_log),
            "shared_capacity_profile": document["shared_capacity_profile"],
            "shared_graph_identity": document["shared_graph_identity"],
        }, separators=(",", ":")),
        flush=True,
    )


def _restore_internal_context(
    document: dict[str, object], session_id: str,
) -> tuple[object, Path, Path, dict[str, object]]:
    if document.get("status") not in {"PREPARED", "CELL_COMPLETE", "PROOFS_COMPLETE"}:
        abort(f"EZKL internal session is not ready for cell/finalize: {document.get('status')}")
    if source_tree_hash() != document.get("source_tree_sha256"):
        abort("EZKL source changed since authenticated prepare")
    if resource_manifest_hash() != document.get("resource_manifest_sha256"):
        abort("EZKL resource manifest changed since authenticated prepare")
    identities = resource_file_identities()
    if identities != _parsed_resource_identities(document.get("resource_file_identities")):
        abort("manifest-listed data/checkpoint files changed since authenticated prepare")
    config_sha256s_raw = document.get("config_sha256s")
    if not isinstance(config_sha256s_raw, list) or len(config_sha256s_raw) != len(DATASETS):
        abort("EZKL internal session has incomplete secure config identities")
    config_sha256s: list[tuple[str, str]] = []
    for record in config_sha256s_raw:
        if not isinstance(record, list) or len(record) != 2 or not isinstance(record[0], str) or not isinstance(record[1], str):
            abort("EZKL internal secure config identity is malformed")
        config_sha256s.append((record[0], record[1]))
    if tuple(name for name, _ in config_sha256s) != tuple(DATASETS):
        abort("EZKL internal secure config identities do not match canonical dataset order")
    for dataset, digest in config_sha256s:
        if sha256(ROOT / "resources/canonical/configs" / f"{dataset}_secure.cfg") != digest:
            abort(f"{dataset} secure config changed since authenticated prepare")

    srs_manifest_path = SRS_MANIFEST
    if sha256(srs_manifest_path) != document.get("srs_manifest_sha256"):
        abort("official native SRS manifest changed since authenticated prepare")
    srs_manifest = load_json(srs_manifest_path, "official native SRS manifest")
    srs_path = local_path(document.get("srs_path"), "authenticated official SRS")
    srs = require_formal_srs(srs_path)
    srs_info = srs.stat()
    current_srs_identity = [srs_info.st_dev, srs_info.st_ino, srs_info.st_size, srs_info.st_mtime_ns, srs_info.st_ctime_ns]
    if current_srs_identity != document.get("srs_file_identity"):
        abort("official SRS file changed since authenticated prepare")
    entries = srs_manifest.get("entries")
    if not isinstance(entries, dict) or set(entries) != set(DATASETS):
        abort("official native SRS manifest dataset set changed")
    if any(entry.get("path") != document.get("srs_path") or entry.get("sha256") != document.get("srs_sha256") for entry in entries.values()):
        abort("official native SRS dataset bindings changed since authenticated prepare")
    if document.get("srs_sha256") != entries[DATASETS[0]].get("sha256"):
        abort("official SRS digest changed since authenticated prepare")

    runtime_identity, selected_device, build_environment, gpu_identity = _formal_gpu_environment(session_id)
    if selected_device != document.get("selected_cuda_device"):
        abort("current physical CUDA device differs from the authenticated prepare device")
    if build_environment != document.get("build_environment"):
        abort("current formal toolchain/device environment differs from authenticated prepare")
    if gpu_identity != document.get("gpu_identity"):
        abort("current native ICICLE backend identity differs from authenticated prepare")
    binary = local_path(document.get("binary_path"), "authenticated formal binary")
    build_receipt = local_path(document.get("build_receipt_path"), "authenticated GPU build receipt")
    build_log = local_path(document.get("build_log_path"), "authenticated formal build log")
    if (
        sha256(binary) != document.get("binary_sha256")
        or sha256(build_receipt) != document.get("build_receipt_sha256")
        or sha256(build_log) != document.get("build_log_sha256")
    ):
        abort("formal binary/build receipt changed since authenticated prepare")
    if not reusable_build(binary, NATIVE_FORMAL_BACKEND, runtime_identity, build_environment):
        abort("formal binary no longer matches source/resource/toolchain/device identity")
    os.environ.update({
        "ICICLE_BACKEND_INSTALL_DIR": str(ROOT / "artifacts/build/icicle-cuda"),
        "ICICLE_NATIVE_LIB_DIR": str(ROOT / "artifacts/build/icicle-cuda/lib"),
        "ICICLE_NATIVE_MANIFEST_SHA256": str(gpu_identity["native_backend_manifest_sha256"]),
        "CUDA_ARCH": str(gpu_identity["cuda_architecture"]).removeprefix("sm_"),
        "EZKL_FORMAL_BUILD_RECEIPT_SHA256": str(document["build_receipt_sha256"]),
        "EZKL_FORMAL_BUILD_ENVIRONMENT_SHA256": str(build_environment["build_environment_sha256"]),
    })
    resource_ids = _parsed_resource_identities(document.get("resource_file_identities"))
    context = create_authenticated_formal_context(
        session_id,
        srs_manifest,
        authenticated_resource_file_identities=resource_ids,
        preflight_config_sha256s=tuple(config_sha256s),
    )
    admissions = document.get("native_admissions")
    if not isinstance(admissions, dict) or set(admissions) != set(DATASETS):
        abort("EZKL internal session does not contain all five native admissions")
    for dataset in DATASETS:
        record = admissions[dataset]
        if not isinstance(record, dict):
            abort(f"{dataset} native admission receipt is malformed")
        path = local_path(record.get("path"), f"{dataset} native admission")
        if sha256(path) != record.get("sha256"):
            abort(f"{dataset} native admission changed since authenticated prepare")
        admission = load_json(path, f"{dataset} native admission")
        prepared_dir = local_path(record.get("prepared_dir"), f"{dataset} admitted circuit directory")
        settings_path = prepared_dir / "settings.json"
        compiled_path = prepared_dir / "compiled.circuit"
        if (
            admission.get("schema_version") != 2
            or admission.get("dataset") != dataset
            or admission.get("relation_id") != "R_GAT-ezkl"
            or admission.get("resource_manifest_sha256") != document.get("resource_manifest_sha256")
            or admission.get("relation_identity") != record.get("relation_identity")
            or admission.get("settings_sha256") != record.get("settings_sha256")
            or admission.get("compiled_circuit_sha256") != record.get("compiled_circuit_sha256")
            or sha256(settings_path) != record.get("settings_sha256")
            or sha256(compiled_path) != record.get("compiled_circuit_sha256")
            or Path(record.get("path", "")).parent != Path(record.get("prepared_dir", ""))
        ):
            abort(f"{dataset} native admission relation identity mismatch")
    if _shared_graph_identity_from_admissions(admissions) != document.get("shared_graph_identity"):
        abort("EZKL shared graph identity changed since authenticated prepare")
    return context, binary, srs, gpu_identity


def _run_internal_cell(
    args: argparse.Namespace,
    session_dir: Path,
    session_id: str,
    session_file: Path,
    dataset: str,
) -> None:
    if dataset not in DATASETS:
        abort(f"unknown canonical EZKL dataset: {dataset}")
    document = _load_internal_session(session_file, session_id)
    if document.get("status") not in {"PREPARED", "CELL_COMPLETE"}:
        abort("EZKL cell requires a completed prepare phase and no failed/in-progress cell")
    receipts = document.get("cell_receipts")
    if not isinstance(receipts, dict) or dataset in receipts:
        abort(f"EZKL cell has already been run or receipt state is malformed: {dataset}")
    context, binary, srs, _gpu = _restore_internal_context(document, session_id)
    output_root = local_path(args.output_root, "output-root")
    matrix = local_path(args.matrix, "matrix")
    if output_root.relative_to(ROOT).as_posix() != document.get("output_root") or matrix.relative_to(ROOT).as_posix() != document.get("matrix"):
        abort("EZKL cell output/matrix paths differ from authenticated prepare")
    _save_internal_session(session_file, document, status="CELL_RUNNING", stage=f"{dataset}_setup_prove_verify")
    document = _load_internal_session(session_file, session_id)
    document["active_dataset"] = dataset
    _write_session_record(session_file, document)

    output = prepare_output_dir(output_root, dataset)
    admissions = document.get("native_admissions")
    admission_record = admissions.get(dataset) if isinstance(admissions, dict) else None
    if not isinstance(admission_record, dict):
        abort(f"{dataset} has no authenticated native admission handoff")
    admission_dir = local_path(admission_record.get("prepared_dir"), f"{dataset} admitted circuit directory")
    setup = [
        local_cli(binary, "binary"), "gat-prepare", "--dataset", dataset, "--phase", "setup",
        "--resource-manifest", "resources/resource_manifest.json", "--output-dir",
        local_cli(output, "output-dir"), "--admission-dir", local_cli(admission_dir, "admitted circuit"),
        "--srs-path", local_cli(srs, "SRS"), "--gpu",
    ]
    setup_ms = run(setup, f"{dataset} native setup", gpu=True)
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
            "--dataset", dataset, "--backend", NATIVE_FORMAL_BACKEND, "--execute-formal", "--reuse-setup",
            "--config", local_cli(ROOT / "resources/canonical/configs" / f"{dataset}_secure.cfg", "config"),
            "--public-statement", local_cli(output / "statement.json", "public statement"),
            "--public-output", local_cli(output / "public_output.json", "public output"),
            "--producer-manifest", local_cli(output / "producer_manifest.json", "producer manifest"),
            "--binary", local_cli(binary, "binary"), "--output-dir", local_cli(output, "output-dir"),
        ],
        f"{dataset} final/witness/prove/independent verify",
        context=context,
        session_id=session_id,
    )
    result_path = output / "result.json"
    result = load_json(result_path, f"{dataset} formal result")
    if (
        result.get("status") != "VERIFY_OK" or result.get("dataset") != dataset
        or result.get("formal_session_id") != session_id or result.get("native_verify_result") is not True
        or result.get("proof_contains_private_witness") is not False
    ):
        abort(f"{dataset} formal result did not bind successful independent verification/session")
    proof_path = local_path(result.get("proof_path"), f"{dataset} proof")
    artifact_path = local_path(result.get("artifact_manifest_path"), f"{dataset} artifact manifest")
    verify_log = output / "logs/verify.log"
    for path, label in ((proof_path, "proof"), (artifact_path, "artifact manifest"), (verify_log, "independent verifier log")):
        require_file(path, f"{dataset} {label}")
    if sha256(proof_path) != result.get("proof_sha256") or sha256(artifact_path) != result.get("artifact_manifest_sha256"):
        abort(f"{dataset} proof/artifact digest does not match result.json")
    if sha256(verify_log) != result.get("independent_verifier_log_sha256"):
        abort(f"{dataset} independent verifier log digest does not match result.json")
    metrics = {
        "commitment_time_ms": result.get("commitment_time_ms"),
        "prove_time_ms": result.get("prove_time_ms"),
        "verify_time_ms": result.get("verify_time_ms"),
        "proof_size_bytes": result.get("proof_size_bytes"),
    }
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0 for value in metrics.values()):
        abort(f"{dataset} result lacks valid four-metric formal measurements")
    receipt = {
        "dataset": dataset,
        "status": "VERIFY_OK",
        "result_path": result_path.relative_to(ROOT).as_posix(),
        "result_sha256": sha256(result_path),
        "artifact_path": artifact_path.relative_to(ROOT).as_posix(),
        "artifact_sha256": sha256(artifact_path),
        "proof_path": proof_path.relative_to(ROOT).as_posix(),
        "proof_sha256": sha256(proof_path),
        "verify_log_path": verify_log.relative_to(ROOT).as_posix(),
        "verify_log_sha256": sha256(verify_log),
        "native_verify_result": True,
        "independent_verification": True,
        **metrics,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    document = _load_internal_session(session_file, session_id)
    cell_receipts = document.get("cell_receipts")
    if not isinstance(cell_receipts, dict) or dataset in cell_receipts:
        abort("EZKL session cell receipt state changed during execution")
    cell_receipts[dataset] = receipt
    document["active_dataset"] = None
    _save_internal_session(session_file, document, status="CELL_COMPLETE", stage=f"{dataset}_complete")
    print(
        f"EZKL_INTERNAL_PHASE_OK phase=cell session_id={session_id} "
        + json.dumps(receipt, separators=(",", ":"), sort_keys=True),
        flush=True,
    )


def _finalize_internal(
    args: argparse.Namespace,
    session_dir: Path,
    session_id: str,
    session_file: Path,
) -> None:
    document = _load_internal_session(session_file, session_id)
    if document.get("status") not in {"CELL_COMPLETE", "PROOFS_COMPLETE"}:
        abort("EZKL finalize requires all cell executions to have completed")
    receipts = document.get("cell_receipts")
    if not isinstance(receipts, dict) or set(receipts) != set(DATASETS):
        abort("EZKL finalize requires exactly one successful result for each canonical dataset")
    for dataset in DATASETS:
        receipt = receipts[dataset]
        if not isinstance(receipt, dict) or receipt.get("status") != "VERIFY_OK" or receipt.get("native_verify_result") is not True:
            abort(f"EZKL finalize found an unsuccessful cell receipt: {dataset}")
        for path_key, digest_key in (("result_path", "result_sha256"), ("artifact_path", "artifact_sha256"), ("proof_path", "proof_sha256"), ("verify_log_path", "verify_log_sha256")):
            path = local_path(receipt.get(path_key), f"{dataset} {path_key}")
            if sha256(path) != receipt.get(digest_key):
                abort(f"EZKL finalize detected changed {dataset} {path_key}")
    _restore_internal_context(document, session_id)
    output_root = local_path(args.output_root, "output-root")
    matrix = local_path(args.matrix, "matrix")
    if output_root.relative_to(ROOT).as_posix() != document.get("output_root") or matrix.relative_to(ROOT).as_posix() != document.get("matrix"):
        abort("EZKL finalize output/matrix paths differ from authenticated prepare")
    document = _load_internal_session(session_file, session_id)
    _save_internal_session(session_file, document, status="PROOFS_COMPLETE", stage="collect_five_results")
    collect = ROOT / "scripts/collect_results.py"
    validate = ROOT / "scripts/validate_final.py"
    run(
        [sys.executable, local_cli(collect, "result collector"), "--runs-root", local_cli(output_root, "output-root"),
         "--output", local_cli(matrix, "matrix"), "--session-id", session_id],
        "collect five-dataset result matrix",
    )
    _save_internal_session(session_file, _load_internal_session(session_file, session_id), status="PROOFS_COMPLETE", stage="final_matrix_validation")
    run(
        [sys.executable, local_cli(validate, "result validator"), "--matrix", local_cli(matrix, "matrix"), "--session-id", session_id],
        "validate five-dataset result matrix",
    )
    matrix_document = load_json(matrix, "validated EZKL formal matrix")
    if matrix_document.get("session_id") != session_id:
        abort("final EZKL matrix does not bind the current root session")
    document = _load_internal_session(session_file, session_id)
    document["final_matrix_path"] = matrix.relative_to(ROOT).as_posix()
    document["final_matrix_sha256"] = sha256(matrix)
    _save_internal_session(session_file, document, status="FINAL_VALIDATION_OK", stage="final_matrix_validation")
    final_receipt_path = session_dir / "final_validation.json"
    final_receipt = {
        "schema_version": 1,
        "baseline": "ezkl",
        "session_id": session_id,
        "status": "FINAL_VALIDATION_OK",
        "matrix_path": matrix.relative_to(ROOT).as_posix(),
        "matrix_sha256": sha256(matrix),
        "datasets": list(DATASETS),
        "cell_result_sha256s": {dataset: receipts[dataset]["result_sha256"] for dataset in DATASETS},
    }
    _write_session_record(final_receipt_path, final_receipt)
    print(
        f"EZKL_INTERNAL_PHASE_OK phase=finalize session_id={session_id} "
        + json.dumps({
            "status": "PASS", "validation_status": "FINAL_VALIDATION_OK",
            "matrix_path": matrix.relative_to(ROOT).as_posix(), "matrix_sha256": sha256(matrix),
            "receipt_path": final_receipt_path.relative_to(ROOT.parent).as_posix(),
            "receipt_sha256": sha256(final_receipt_path),
        }, separators=(",", ":"), sort_keys=True),
        flush=True,
    )


def _run_internal_phase(args: argparse.Namespace) -> int:
    if args.internal_phase is None or args.session_dir is None:
        abort("internal phase invocation requires --internal-phase and --session-dir")
    if args.preflight_only:
        abort("--preflight-only cannot be combined with the internal phase API")
    if args.internal_phase == "cell" and args.dataset not in DATASETS:
        abort("internal cell phase requires one canonical --dataset")
    if args.internal_phase != "cell" and args.dataset is not None:
        abort("--dataset is accepted only for internal cell phase")
    session_dir, session_id = _internal_session_location(args.session_dir)
    validate_root_owned_phase(
        session_dir, session_id, args.internal_phase, args.dataset, consume_token=True,
    )
    session_file = session_dir / "session.json"
    if session_file.is_symlink():
        abort(f"EZKL internal session file must not be a symlink: {session_file}")
    output_root = local_path(args.output_root, "output-root")
    matrix = local_path(args.matrix, "matrix")
    lock_dir = ROOT / "runs"
    if lock_dir.is_symlink() or (lock_dir.exists() and not lock_dir.is_dir()):
        abort(f"formal lock directory is not a regular directory: {lock_dir}")
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / ".ezkl-formal.lock"
    if lock_path.is_symlink():
        abort(f"formal session lock must not be a symlink: {lock_path}")
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(lock_fd)
        abort("another formal EZKL session currently holds the baseline lock")
    try:
        if args.internal_phase == "prepare":
            _internal_prepare(args, session_dir, session_id, session_file)
        elif args.internal_phase == "cell":
            _run_internal_cell(args, session_dir, session_id, session_file, args.dataset)
        else:
            _finalize_internal(args, session_dir, session_id, session_file)
        return 0
    except BaseException as exc:
        if session_file.exists() and not session_file.is_symlink():
            try:
                document = _load_internal_session(session_file, session_id)
                document["error"] = f"{type(exc).__name__}: {exc}"[:2000]
                _save_internal_session(session_file, document, status="FAILED", stage=str(document.get("stage", "failed")))
            except BaseException:
                pass
        raise
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


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
    os.environ["EZKL_FORMAL_SELECTED_CUDA_DEVICE"] = json.dumps(
        selected_device, sort_keys=True, separators=(",", ":")
    )
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
                    log_path=_build_log_path(session_id),
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
            "--rustdoc", str(runtime_identity["rustdoc_path"]),
            "--cmake", str(runtime_identity["cmake_path"]),
            "--make", str(runtime_identity["make_path"]),
            "--nvcc", str(runtime_identity["nvcc_path"]),
            "--cc", str(runtime_identity["cc_path"]),
            "--cxx", str(runtime_identity["cxx_path"]),
            "--nvidia-smi", str(runtime_identity["nvidia_smi_path"]),
            "--selected-device-json", json.dumps(selected_device, sort_keys=True, separators=(",", ":")),
            "--build-environment-json", json.dumps(build_environment, sort_keys=True, separators=(",", ":")),
        ]
        run(
            build_args,
            "build source-matched official ezkl native binary",
            log_path=_build_log_path(session_id),
        )
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

    _set_stage(session_file, "five_dataset_native_relation_capacity_admission")
    admissions = native_admit_all_datasets(binary, srs, output_root, session_id)
    session_document = json.loads(session_file.read_text())
    session_document["native_admissions"] = admissions
    session_document["resource_manifest_sha256"] = resource_manifest_hash()
    session_document["srs_manifest_sha256"] = sha256(SRS_MANIFEST)
    session_document["source_tree_sha256"] = source_tree_hash()
    session_document["binary_sha256"] = sha256(binary)
    _write_session_record(session_file, session_document)
    archive_prior_outputs(output_root, matrix, session_id, session_file)

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
            "--admission-dir",
            local_cli(local_path(admissions[dataset]["prepared_dir"], f"{dataset} admitted circuit directory"), "admitted circuit"),
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
    if args.internal_phase is not None or args.session_dir is not None or args.dataset is not None:
        return _run_internal_phase(args)
    abort("direct EZKL execution is disabled; start the comparison with ./comparison/run_formal.sh")


if __name__ == "__main__":
    try:
        raise SystemExit(main(parser().parse_args()))
    except (ContractError, OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        raise SystemExit(print_error(exc))
