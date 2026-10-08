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
    validate_gpu_environment,
    validate_resources,
    validate_formal_python_environment,
    select_formal_cuda_device,
    resource_file_identities,
)
from process_guardian import (
    ProcessGuardian,
    check_guardian_health,
    finish_guarded_process,
    run_guarded_capture,
    spawn_guarded_process,
    terminate_guarded_process_locally,
)
from finalization_capacity import FinalizationCapacityError, check_finalization_capacity


_PROCESS_GUARDIAN: ProcessGuardian | None = None


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--binary")
    result.add_argument("--output-root", default="runs/formal")
    result.add_argument("--matrix", default="results/formal_matrix.json")
    result.add_argument("--preflight-only", action="store_true", help="authenticate all prerequisites without setup/build/prove")
    result.add_argument("--_from-ezkl-entry", action="store_true", help=argparse.SUPPRESS)
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
        process = spawn_guarded_process(
            command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
    except BaseException:
        if log_stream is not None:
            log_stream.close()
        raise
    if process.stdout is None:
        _terminate_process_group(process)
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
                line = output_queue.get(timeout=5)
            except queue.Empty:
                check_guardian_health()
                elapsed_seconds = int(time.monotonic() - started)
                print(f"[ezkl-formal] {label} still running ({elapsed_seconds}s)", flush=True)
                continue
            if time.monotonic() - started >= 5:
                check_guardian_health()
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
        terminated_leftovers = finish_guarded_process(process)
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
        if terminated_leftovers:
            abort(f"{label} failed with exit code {returncode}; leftover process-group descendants were cleaned")
        abort(f"{label} failed with exit code {returncode}")
    if terminated_leftovers:
        abort(f"{label} left descendants after its leader exited; the group was cleaned")
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
        process = spawn_guarded_process(
            command,
            cwd=ROOT,
            stdout=log_stream if log_stream is not None else None,
            stderr=subprocess.STDOUT if log_stream is not None else None,
        )
    except BaseException:
        if log_stream is not None:
            log_stream.close()
        raise
    try:
        deadline = time.monotonic() + timeout_seconds
        while True:
            try:
                returncode = process.wait(timeout=min(5, max(0.1, deadline - time.monotonic())))
                break
            except subprocess.TimeoutExpired:
                if time.monotonic() >= deadline:
                    raise
                check_guardian_health()
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
    terminated_leftovers = finish_guarded_process(process) if returncode == 0 else False
    if log_stream is not None:
        log_stream.close()
    if returncode != 0:
        # The leader may have exited while helpers in its private process
        # group remain alive. Reap the whole group before reporting its real
        # exit status so no failed prerequisite can leak background work.
        _terminate_process_group(process)
        abort(f"prerequisite {label} failed with exit code {returncode}")
    if terminated_leftovers:
        abort(f"prerequisite {label} left descendants after its leader exited; the group was cleaned")


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
    process = spawn_guarded_process(
        command,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    if process.stdout is None:
        _terminate_process_group(process)
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
                line = output_queue.get(timeout=min(5, remaining))
            except queue.Empty:
                check_guardian_health()
                elapsed_seconds = int(time.monotonic() - started)
                print(f"[ezkl-formal] official resource preparation/authentication still running ({elapsed_seconds}s)", flush=True)
                continue
            if time.monotonic() - started >= 5:
                check_guardian_health()
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
        terminated_leftovers = finish_guarded_process(process) if process.returncode == 0 else False
    except BaseException:
        print("[ezkl-formal] stopping official resource preparation process group after interruption or runner failure", file=sys.stderr, flush=True)
        _terminate_process_group(process)
        reader.join(timeout=5)
        raise
    if returncode != 0:
        _terminate_process_group(process)
        abort(f"official resource preparation/authentication failed with exit code {returncode}")
    if terminated_leftovers:
        abort("official resource preparation left descendants after its leader exited; the group was cleaned")
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
    if getattr(process, "_ezkl_process_group", None) is not None:
        try:
            finish_guarded_process(process, terminate=True)
        except BaseException as exc:
            print(f"[ezkl-formal] guardian cleanup failed; terminating the owned process group locally: {exc}", file=sys.stderr, flush=True)
            terminate_guarded_process_locally(process)
        if process.poll() is None:
            process.wait()
        return
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


def formal_local_path(value: str, label: str) -> Path:
    """Resolve a formal CLI path only after rejecting symlink components."""
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        return local_path(value, label)
    cursor = ROOT
    for component in candidate.parts:
        if component in {"", "."}:
            continue
        cursor = cursor / component
        if cursor.is_symlink():
            abort(f"{label} path must not contain a symlink: {cursor}")
    return local_path(value, label)


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


def _mark_released_session_interrupted(lock_fd: int, output_root: Path) -> None:
    os.lseek(lock_fd, 0, os.SEEK_SET)
    raw = os.read(lock_fd, 8192).decode("utf-8", errors="replace").strip()
    if not raw:
        return
    previous_id: str | None = None
    previous_path: str | None = None
    try:
        owner = json.loads(raw)
        if isinstance(owner, dict) and owner.get("baseline") == "ezkl":
            previous_id = owner.get("session_id") if isinstance(owner.get("session_id"), str) else None
            previous_path = owner.get("session_file") if isinstance(owner.get("session_file"), str) else None
    except json.JSONDecodeError:
        match = re.search(r"(?:^|\s)session_id=([0-9a-f]{32})(?:\s|$)", raw)
        previous_id = match.group(1) if match else None
    if not isinstance(previous_id, str) or re.fullmatch(r"[0-9a-f]{32}", previous_id) is None:
        return
    if previous_path is None:
        previous_path = (output_root / ".sessions" / f"{previous_id}.json").relative_to(ROOT).as_posix()
    path = formal_local_path(previous_path, "previous EZKL session record")
    if not path.is_file() or path.is_symlink():
        return
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if (
        not isinstance(document, dict)
        or document.get("baseline") != "ezkl"
        or document.get("session_id") != previous_id
        or document.get("status") not in {"PREFLIGHT_STARTED", "RUNNING"}
    ):
        return
    update_session_record(
        path,
        status="INTERRUPTED",
        stage=str(document.get("stage", "interrupted")),
        error="previous EZKL owner released the baseline lock before recording a terminal state",
    )
    print(f"[ezkl-formal] marked released prior session {previous_id} as INTERRUPTED", flush=True)


def begin_formal_session(output_root: Path, matrix: Path) -> tuple[str, Path, int]:
    if output_root.is_symlink() or matrix.is_symlink():
        abort("formal output-root and matrix paths must not be symlinks")
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
        _mark_released_session_interrupted(lock_fd, output_root)
        session_id = uuid.uuid4().hex
        session_dir = output_root / ".sessions"
        if session_dir.is_symlink() or (session_dir.exists() and not session_dir.is_dir()):
            abort(f"formal session history path must be a regular directory: {session_dir}")
        session_dir.mkdir(parents=True, exist_ok=True)
        session_file = session_dir / f"{session_id}.json"
        lock_payload = json.dumps({
            "schema_version": 1,
            "baseline": "ezkl",
            "pid": os.getpid(),
            "session_id": session_id,
            "session_file": session_file.relative_to(ROOT).as_posix(),
        }, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        os.ftruncate(lock_fd, 0)
        os.lseek(lock_fd, 0, os.SEEK_SET)
        os.write(lock_fd, lock_payload)
        os.fsync(lock_fd)
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
    selected_device: dict[str, object],
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
        "selector_fixed_column_count", "fixed_column_count_post_compression",
        "permutation_copy_call_count", "permutation_successor_entry_count_upper",
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
            admission["schema_version"] != 3
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
        if cs["fixed_column_count_post_compression"] != (
            cs["num_fixed_columns"] + cs["selector_fixed_column_count"]
        ):
            abort(f"{dataset} native admission post-compression fixed-column count is inconsistent")
        expected_successor_entries = min(
            cs["num_permutation_columns"] * admission["domain_size"],
            2 * cs["permutation_copy_call_count"],
        )
        if cs["permutation_successor_entry_count_upper"] != expected_successor_entries:
            abort(f"{dataset} native admission compact permutation entry bound is inconsistent")
        memory = admission["native_memory_model"]
        memory_keys = {
            "schema", "proof_logrows", "proof_domain_size", "advice_column_phase_ids",
            "size_of_fr_bytes", "size_of_assigned_fr_bytes", "size_of_option_fr_bytes",
            "size_of_batch_invert_entry_bytes", "size_of_usize_bytes", "size_of_usize_pair_bytes",
            "size_of_g1_affine_bytes",
            "cuda_uuid", "cuda_free_bytes_at_admission", "cuda_total_memory_bytes",
            "selector_activation_bitmap_bytes", "fixed_polynomial_read_mode",
            "fixed_polynomial_read_chunk_fields_upper", "fixed_polynomial_read_raw_chunk_bytes_upper",
            "fixed_polynomial_read_decoded_chunk_bytes_upper", "fixed_polynomial_read_heap_scratch_bytes_upper",
        }
        if not isinstance(memory, dict) or set(memory) != memory_keys or memory.get("schema") != "ezkl-halo2-memory-model-v2":
            abort(f"{dataset} native admission has an incompatible source memory model")
        positive_memory_keys = {
            "proof_logrows", "proof_domain_size", "size_of_fr_bytes",
            "size_of_assigned_fr_bytes", "size_of_option_fr_bytes",
            "size_of_batch_invert_entry_bytes", "size_of_usize_bytes",
            "size_of_usize_pair_bytes", "size_of_g1_affine_bytes",
            "cuda_total_memory_bytes",
            "fixed_polynomial_read_chunk_fields_upper", "fixed_polynomial_read_raw_chunk_bytes_upper",
            "fixed_polynomial_read_decoded_chunk_bytes_upper", "fixed_polynomial_read_heap_scratch_bytes_upper",
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
            or memory["size_of_batch_invert_entry_bytes"] < memory["size_of_fr_bytes"]
            or memory["cuda_free_bytes_at_admission"] > memory["cuda_total_memory_bytes"]
            or not isinstance(memory["cuda_uuid"], str)
            or re.fullmatch(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", memory["cuda_uuid"]) is None
            or memory["cuda_uuid"] != selected_device.get("uuid")
        ):
            abort(f"{dataset} native admission memory model is internally inconsistent or targets another GPU")
        expected_bitmap = cs["num_selectors"] * ((admission["domain_size"] + 7) // 8)
        read_chunk_fields = memory["fixed_polynomial_read_chunk_fields_upper"]
        if (
            memory["selector_activation_bitmap_bytes"] != expected_bitmap
            or memory["fixed_polynomial_read_mode"] != "streamed_one_polynomial_chunked_v1"
            or read_chunk_fields != 16 * 1024
            or memory["fixed_polynomial_read_raw_chunk_bytes_upper"]
            != read_chunk_fields * memory["size_of_fr_bytes"]
            or memory["fixed_polynomial_read_decoded_chunk_bytes_upper"]
            != read_chunk_fields * memory["size_of_fr_bytes"]
            or memory["fixed_polynomial_read_heap_scratch_bytes_upper"]
            != memory["fixed_polynomial_read_raw_chunk_bytes_upper"]
            + memory["fixed_polynomial_read_decoded_chunk_bytes_upper"]
        ):
            abort(f"{dataset} native admission fixed-polynomial streaming bounds are inconsistent")
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


_CUB_QUERY_COUNTS = {
    "scalar_index_sort_pairs": 1 << 24,
    "sorted_bucket_run_length_encode": 1 << 24,
    "bucket_offsets_exclusive_scan": 1_048_561,
    "bucket_size_offset_sort_pairs_descending": 1_048_560,
    "bucket_size_index_sort_pairs_descending": 1_048_560,
    "large_bucket_prefix_inclusive_scan": 104_206,
}


def _validate_cub_workspace_query(
    value: object,
    *,
    source_sha256: str,
    backend_manifest_sha256: str,
    selected_device: dict[str, object],
    session_id: str,
) -> dict[str, object]:
    keys = {
        "schema", "session_id", "icicle_source_tree_sha256", "native_backend_manifest_sha256", "device_uuid",
        "free_memory_bytes", "total_memory_bytes", "integrated_device", "scalar_bits", "c", "chunk_scalars", "window_count", "M", "B",
        "H_max", "L_max", "cuda_runtime_version", "cuda_driver_version", "cuda_toolkit_version",
        "cub_version", "device_compute_capability", "native_type_sizes", "device_workspace_peak_bytes",
        "queries", "scratch_bytes_sum",
    }
    if not isinstance(value, dict) or set(value) != keys:
        abort("native CUB workspace query has an incompatible exact schema")
    expected_constants = {
        "schema": "ezkl.icicle.cub_msm_workspace_query.v2",
        "session_id": session_id,
        "icicle_source_tree_sha256": source_sha256,
        "native_backend_manifest_sha256": backend_manifest_sha256,
        "device_uuid": selected_device.get("uuid"),
        "scalar_bits": 254,
        "c": 16,
        "chunk_scalars": 1 << 20,
        "window_count": 16,
        "M": 1 << 24,
        "B": 1_048_560,
        "H_max": 1_048_560,
        "L_max": 104_206,
        "device_compute_capability": selected_device.get("compute_capability"),
    }
    for key, expected in expected_constants.items():
        if value.get(key) != expected:
            abort(f"native CUB workspace query identity/constant mismatch: {key}")
    for key in ("free_memory_bytes", "total_memory_bytes", "cuda_runtime_version", "cuda_driver_version", "cub_version", "device_workspace_peak_bytes", "scratch_bytes_sum"):
        item = value.get(key)
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            abort(f"native CUB workspace query has invalid {key}")
    if value["free_memory_bytes"] > value["total_memory_bytes"] or not isinstance(value.get("integrated_device"), bool):
        abort("native CUB workspace query has invalid live CUDA capacity values")
    type_sizes = value.get("native_type_sizes")
    if not isinstance(type_sizes, dict) or set(type_sizes) != {
        "scalar_bytes", "affine_bytes", "projective_bytes", "unsigned_bytes",
    } or any(
        isinstance(item, bool) or not isinstance(item, int) or item <= 0
        for item in type_sizes.values()
    ):
        abort("native CUB workspace query has invalid native sizeof values")
    toolkit = value.get("cuda_toolkit_version")
    if not isinstance(toolkit, str) or re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", toolkit) is None:
        abort("native CUB workspace query has invalid CUDA toolkit version")
    queries = value.get("queries")
    if not isinstance(queries, list) or len(queries) != len(_CUB_QUERY_COUNTS):
        abort("native CUB workspace query has an incomplete set of CUB operations")
    query_sum = 0
    for query, (expected_name, expected_count) in zip(queries, _CUB_QUERY_COUNTS.items(), strict=True):
        if not isinstance(query, dict) or set(query) != {"name", "num_items", "scratch_bytes"}:
            abort("native CUB workspace query contains a malformed operation")
        if query.get("name") != expected_name or query.get("num_items") != expected_count:
            abort(f"native CUB workspace query operation shape mismatch: {expected_name}")
        scratch = query.get("scratch_bytes")
        if isinstance(scratch, bool) or not isinstance(scratch, int) or scratch < 0:
            abort(f"native CUB workspace query has invalid scratch bytes: {expected_name}")
        query_sum += scratch
    if value["scratch_bytes_sum"] != query_sum:
        abort("native CUB workspace query scratch total does not equal its operation sum")
    return value


def _validate_cuda_memory_sample(
    value: object,
    *,
    session_id: str,
    selected_device: dict[str, object],
    workspace_query: dict[str, object],
) -> dict[str, object]:
    fields = {
        "schema", "session_id", "device_uuid", "free_memory_bytes", "total_memory_bytes",
        "integrated_device",
    }
    if not isinstance(value, dict) or set(value) != fields:
        abort("EZKL CUDA memory sample has an incompatible exact schema")
    if (
        value.get("schema") != "ezkl.icicle.device_memory_sample.v1"
        or value.get("session_id") != session_id
        or value.get("device_uuid") != selected_device.get("uuid")
        or value.get("integrated_device") != workspace_query.get("integrated_device")
        or value.get("total_memory_bytes") != workspace_query.get("total_memory_bytes")
    ):
        abort("EZKL CUDA memory sample does not match its session-selected device/workspace receipt")
    for key in ("free_memory_bytes", "total_memory_bytes"):
        item = value.get(key)
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            abort(f"EZKL CUDA memory sample has invalid {key}")
    if value["free_memory_bytes"] > value["total_memory_bytes"]:
        abort("EZKL CUDA memory sample free capacity exceeds total capacity")
    if not isinstance(value.get("integrated_device"), bool):
        abort("EZKL CUDA memory sample has invalid integrated-device identity")
    return value


def _gpu_query_environment(
    selected_device: dict[str, object],
    gpu_identity: dict[str, object],
    build_environment: dict[str, object],
) -> dict[str, str]:
    env = os.environ.copy()
    for name in ("LD_PRELOAD", "ICICLE_SMALL_K", "ICICLE_SMALL_K_FFT", "EZKL_FORMAL_RESOURCE_AUTH_SESSION_ID"):
        env.pop(name, None)
    env.update({
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": str(selected_device["uuid"]),
        "ICICLE_DEVICE_ID": "0",
        "ENABLE_ICICLE_GPU": "1",
        "EZKL_GAT_BACKEND": "gpu",
        "ICICLE_BACKEND_INSTALL_DIR": str(ROOT / "artifacts/build/icicle-cuda"),
        "ICICLE_NATIVE_LIB_DIR": str(ROOT / "artifacts/build/icicle-cuda/lib"),
        "ICICLE_NATIVE_MANIFEST_SHA256": str(gpu_identity["native_backend_manifest_sha256"]),
        "CUDA_ARCH": str(build_environment["cuda_architecture"]).removeprefix("sm_"),
    })
    return env


def _build_native_cub_workspace_query(
    session_dir: Path,
    session_id: str,
    runtime_identity: dict[str, object],
    selected_device: dict[str, object],
    build_environment: dict[str, object],
    gpu_identity: dict[str, object],
) -> dict[str, object]:
    source = require_file(ROOT / "tools/native_msm_workspace_query.cu", "native CUB workspace query source")
    source_sha256 = sha256(source)
    manifest_path = require_file(ROOT / "artifacts/build/icicle-cuda/native_backend_manifest.json", "native ICICLE manifest")
    backend_manifest = load_json(manifest_path, "native ICICLE manifest")
    icicle_source_sha256 = backend_manifest.get("native_source_sha256")
    if (
        not isinstance(icicle_source_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", icicle_source_sha256) is None
        or sha256(manifest_path) != gpu_identity.get("native_backend_manifest_sha256")
    ):
        abort("native CUB workspace query cannot bind the current ICICLE source/manifest identity")
    if selected_device.get("logical_device_id") != 0:
        abort("native CUB workspace query requires the formally selected CUDA logical device zero")
    arch = build_environment.get("cuda_architecture")
    nvcc = runtime_identity.get("nvcc_path")
    if not isinstance(arch, str) or re.fullmatch(r"sm_[0-9]+", arch) is None or not isinstance(nvcc, str):
        abort("native CUB workspace query lacks source-matched CUDA compiler/architecture identity")

    query_dir = session_dir / "capacity"
    if query_dir.exists():
        if query_dir.is_symlink() or not query_dir.is_dir() or any(query_dir.iterdir()):
            abort("native CUB workspace query directory must be fresh and empty")
    else:
        query_dir.mkdir(parents=True, exist_ok=False)
    binary_path = query_dir / "native_msm_workspace_query"
    result_path = query_dir / "native_msm_workspace_query.json"
    build_log_path = query_dir / "native_msm_workspace_query_build.log"
    if any(path.exists() or path.is_symlink() for path in (binary_path, result_path, build_log_path)):
        abort("native CUB workspace query refuses to overwrite an existing session artifact")
    env = _gpu_query_environment(selected_device, gpu_identity, build_environment)
    build_command = [
        nvcc, "-O2", "-std=c++17", f"-arch={arch}",
        "-I", local_cli(ROOT / "vendor/icicle/icicle/include", "ICICLE source headers"),
        local_cli(source, "CUB query source"),
        "-o", local_cli(binary_path, "CUB query binary"),
    ]
    print(f"[ezkl-formal] build session-bound CUB workspace query: {' '.join(build_command)}", flush=True)
    try:
        built = run_guarded_capture(
            build_command, cwd=ROOT, env=env, timeout=120,
        )
    except subprocess.TimeoutExpired as exc:
        build_log_path.write_text((exc.stdout or "") + (exc.stderr or ""), encoding="utf-8")
        abort("native CUB workspace query build exceeded its 120 second bound")
    build_log_path.write_text(built.stdout, encoding="utf-8")
    if built.returncode != 0 or not binary_path.is_file() or binary_path.is_symlink():
        abort(f"native CUB workspace query build failed with exit {built.returncode}; see {build_log_path}")

    command = [
        local_cli(binary_path, "CUB query binary"),
        "--session-id", session_id,
        "--icicle-source-tree-sha256", icicle_source_sha256,
        "--native-backend-manifest-sha256", str(gpu_identity["native_backend_manifest_sha256"]),
    ]
    print(f"[ezkl-formal] run session-bound CUB workspace query: {' '.join(command)}", flush=True)
    try:
        measured = run_guarded_capture(command, cwd=ROOT, env=env, timeout=120)
    except subprocess.TimeoutExpired:
        abort("native CUB workspace query exceeded its 120 second bound")
    if measured.returncode != 0:
        abort(f"native CUB workspace query failed with exit {measured.returncode}: {measured.stderr.strip()}")
    try:
        query = json.loads(measured.stdout)
    except json.JSONDecodeError as exc:
        abort(f"native CUB workspace query emitted malformed JSON: {exc}")
    query = _validate_cub_workspace_query(
        query,
        source_sha256=icicle_source_sha256,
        backend_manifest_sha256=str(gpu_identity["native_backend_manifest_sha256"]),
        selected_device=selected_device,
        session_id=session_id,
    )
    result_path.write_text(json.dumps(query, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    return {
        "schema": "ezkl.icicle.cub_msm_workspace_query_receipt.v2",
        "session_id": session_id,
        "source_path": source.relative_to(ROOT).as_posix(),
        "source_sha256": source_sha256,
        "binary_path": binary_path.relative_to(ROOT).as_posix(),
        "binary_sha256": sha256(binary_path),
        "build_log_path": build_log_path.relative_to(ROOT).as_posix(),
        "build_log_sha256": sha256(build_log_path),
        "result_path": result_path.relative_to(ROOT).as_posix(),
        "result_sha256": sha256(result_path),
        "native_backend_manifest_path": manifest_path.relative_to(ROOT).as_posix(),
        "native_backend_manifest_sha256": sha256(manifest_path),
        "icicle_source_tree_sha256": icicle_source_sha256,
        "query": query,
    }


def _validate_native_cub_workspace_receipt(
    receipt: object,
    session_dir: Path,
    selected_device: dict[str, object],
    gpu_identity: dict[str, object],
    session_id: str,
) -> dict[str, object]:
    fields = {
        "schema", "session_id", "source_path", "source_sha256", "binary_path", "binary_sha256", "build_log_path",
        "build_log_sha256", "result_path", "result_sha256", "native_backend_manifest_path",
        "native_backend_manifest_sha256", "icicle_source_tree_sha256", "query",
    }
    if not isinstance(receipt, dict) or set(receipt) != fields or receipt.get("schema") != "ezkl.icicle.cub_msm_workspace_query_receipt.v2":
        abort("EZKL session has no complete CUB workspace query source/binary receipt")
    if receipt.get("session_id") != session_id:
        abort("EZKL CUB workspace query receipt belongs to a different formal session")
    source = local_path(receipt.get("source_path"), "CUB query source")
    binary = local_path(receipt.get("binary_path"), "CUB query binary")
    build_log = local_path(receipt.get("build_log_path"), "CUB query build log")
    result_path = local_path(receipt.get("result_path"), "CUB query result")
    manifest_path = local_path(receipt.get("native_backend_manifest_path"), "native ICICLE manifest")
    try:
        binary.relative_to(session_dir)
        result_path.relative_to(session_dir)
        build_log.relative_to(session_dir)
    except ValueError:
        abort("session CUB query build artifacts are outside the authenticated session directory")
    for path, digest, label in (
        (source, receipt.get("source_sha256"), "CUB source"),
        (binary, receipt.get("binary_sha256"), "CUB binary"),
        (build_log, receipt.get("build_log_sha256"), "CUB build log"),
        (result_path, receipt.get("result_sha256"), "CUB JSON result"),
        (manifest_path, receipt.get("native_backend_manifest_sha256"), "ICICLE manifest"),
    ):
        if not isinstance(digest, str) or sha256(path) != digest:
            abort(f"EZKL session CUB workspace query {label} digest changed")
    if (
        source != ROOT / "tools/native_msm_workspace_query.cu"
        or receipt["native_backend_manifest_sha256"] != gpu_identity.get("native_backend_manifest_sha256")
    ):
        abort("EZKL session CUB workspace query is bound to a stale helper/backend source")
    backend_manifest = load_json(manifest_path, "native ICICLE manifest")
    if backend_manifest.get("native_source_sha256") != receipt.get("icicle_source_tree_sha256"):
        abort("EZKL session CUB workspace query ICICLE source digest changed")
    query_file = load_json(result_path, "CUB workspace query JSON")
    query = _validate_cub_workspace_query(
        query_file,
        source_sha256=str(receipt["icicle_source_tree_sha256"]),
        backend_manifest_sha256=str(receipt["native_backend_manifest_sha256"]),
        selected_device=selected_device,
        session_id=session_id,
    )
    if receipt.get("query") != query:
        abort("EZKL session CUB workspace query embedded result differs from its source-bound JSON")
    return receipt


def _sample_native_cuda_memory(
    receipt: dict[str, object],
    *,
    session_dir: Path,
    session_id: str,
    selected_device: dict[str, object],
    gpu_identity: dict[str, object],
    build_environment: dict[str, object],
) -> dict[str, object]:
    receipt = _validate_native_cub_workspace_receipt(
        receipt, session_dir, selected_device, gpu_identity, session_id
    )
    binary = local_path(receipt.get("binary_path"), "CUB query binary")
    env = _gpu_query_environment(selected_device, gpu_identity, build_environment)
    command = [local_cli(binary, "CUB query binary"), "--memory-only", "--session-id", session_id]
    try:
        measured = run_guarded_capture(command, cwd=ROOT, env=env, timeout=30)
    except subprocess.TimeoutExpired:
        abort("EZKL live CUDA memory recheck exceeded its 30 second bound")
    if measured.returncode != 0:
        abort(f"EZKL live CUDA memory recheck failed with exit {measured.returncode}: {measured.stderr.strip()}")
    try:
        raw = json.loads(measured.stdout)
    except json.JSONDecodeError as exc:
        abort(f"EZKL live CUDA memory recheck emitted malformed JSON: {exc}")
    workspace_query = receipt.get("query")
    sample = _validate_cuda_memory_sample(
        raw,
        session_id=session_id,
        selected_device=selected_device,
        workspace_query=workspace_query if isinstance(workspace_query, dict) else {},
    )
    sample["sampled_at_utc"] = datetime.now(timezone.utc).isoformat()
    return sample


def _effective_host_memory_available_bytes() -> tuple[int, dict[str, object]]:
    mem_available = None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, separator, value = line.partition(":")
            if separator and key == "MemAvailable":
                parts = value.split()
                if parts and parts[0].isdigit():
                    mem_available = int(parts[0]) * 1024
                    break
    except OSError as exc:
        abort(f"cannot sample host MemAvailable for EZKL capacity gate: {exc}")
    if mem_available is None:
        abort("host MemAvailable is absent or malformed; EZKL capacity gate cannot proceed")

    effective = mem_available
    accounting: dict[str, object] = {"mem_available_bytes": mem_available, "cgroup_limits": []}
    for base in _cgroup_memory_paths():
        try:
            maximum = (base / "memory.max").read_text().strip()
            current = int((base / "memory.current").read_text().strip())
        except (OSError, ValueError):
            continue
        if maximum == "max":
            continue
        if not maximum.isdigit():
            abort(f"malformed cgroup memory.max at {base}")
        available = max(0, int(maximum) - current)
        effective = min(effective, available)
        accounting["cgroup_limits"].append({
            "path": str(base), "limit_bytes": int(maximum), "current_bytes": current,
            "available_bytes": available,
        })
    accounting["effective_available_bytes"] = effective
    return effective, accounting


def _filesystem_available_bytes(path: Path) -> tuple[int, dict[str, int]]:
    try:
        stat = os.statvfs(path)
        device_id = int(os.stat(path).st_dev)
    except OSError as exc:
        abort(f"cannot sample filesystem capacity for EZKL formal path {path}: {exc}")
    available = int(stat.f_bavail) * int(stat.f_frsize)
    if available < 0:
        abort(f"filesystem returned a negative capacity for EZKL formal path {path}")
    return available, {"available_bytes": available, "device_id": device_id}


def _native_capacity_requirements(
    admission: dict[str, object],
    icicle_type_sizes: dict[str, object],
) -> dict[str, int]:
    cs = admission.get("constraint_system")
    memory = admission.get("native_memory_model")
    domain_size = admission.get("domain_size")
    if not isinstance(cs, dict) or not isinstance(memory, dict) or isinstance(domain_size, bool) or not isinstance(domain_size, int):
        abort("EZKL native admission lacks geometry required by the local capacity gate")
    fr = int(memory["size_of_fr_bytes"])
    assigned = int(memory["size_of_assigned_fr_bytes"])
    option_fr = int(memory["size_of_option_fr_bytes"])
    batch_invert_entry = int(memory["size_of_batch_invert_entry_bytes"])
    n = domain_size
    proof_domain = int(memory["proof_domain_size"])
    fixed_post = int(cs["fixed_column_count_post_compression"])
    fixed_uncompressed = int(cs["num_fixed_columns"])
    permutation = int(cs["num_permutation_columns"])
    advice = int(cs["num_advice_columns"])
    instances = int(cs["num_instance_columns"])
    degree = int(cs["degree"])
    icicle_scalar_bytes = int(icicle_type_sizes["scalar_bytes"])
    icicle_affine_bytes = int(icicle_type_sizes["affine_bytes"])
    if degree < 2:
        abort("EZKL native admission has invalid quotient degree for the local capacity gate")
    selectors = int(memory["selector_activation_bitmap_bytes"])
    read_scratch = int(memory["fixed_polynomial_read_heap_scratch_bytes_upper"])
    for label, number in (
        ("Fr", fr), ("Assigned<Fr>", assigned), ("Option<Fr>", option_fr),
        ("ICICLE scalar", icicle_scalar_bytes), ("ICICLE affine", icicle_affine_bytes),
    ):
        if number <= 0:
            abort(f"EZKL native admission has invalid sizeof({label})")
    if icicle_scalar_bytes != fr:
        abort("compiled ICICLE BN254 scalar size disagrees with native Halo2 Fr")
    # Source-derived known mapped setup allocations: fixed assignment maps,
    # fixed-value and coefficient maps, permutation polynomials, three
    # extended-domain selector polynomials retained by the PK, plus the live
    # l_blind map and one fixed-polynomial clone during keygen.
    fixed_assignment_maps = fixed_uncompressed * n * assigned
    setup_persisted_maps = (2 * fixed_post * n + permutation * n + 3 * proof_domain) * fr
    setup_transition_map = (proof_domain + n) * fr
    setup_mapped_peak = fixed_assignment_maps + setup_persisted_maps + setup_transition_map
    # The serialized PK keeps the same dominant fixed/l-selector payloads;
    # reserving the complete source-derived setup map payload covers that file
    # plus its bounded headers/commitment metadata and the separately saved VK.
    selector_vk_bytes = selectors
    commitment_vk_bytes = (fixed_post + permutation) * int(memory["size_of_g1_affine_bytes"])
    permutation_mapping_bytes = (
        8 + 8 + 10 * permutation
        + 16 * int(cs["permutation_successor_entry_count_upper"]) + 32
    )
    setup_key_artifact_upper = (
        setup_persisted_maps
        + selector_vk_bytes
        + commitment_vk_bytes
        + permutation_mapping_bytes
        + 8 + 12 + 2 * (4 + 4 * fixed_post)
    )
    quotient_coefficients = n * (degree - 1)
    proof_spill_known = (
        setup_persisted_maps
        + advice * n * fr
        + 2 * instances * n * fr
        + proof_domain * fr
        + quotient_coefficients * fr
    )
    # Native save_pk also emits a separate VK file; include its known selector
    # bitmaps and commitments in the retained-artifact reservation.
    setup_key_artifact_upper += selector_vk_bytes + commitment_vk_bytes
    # Source-confirmed heap allocations in keygen: all selector activation
    # bitmaps, one dense selector field vector, the PK denominator Option<Fr>
    # vector, and the streamed fixed-polynomial chunk buffers.
    selector_phase = selectors + n * fr + read_scratch
    # Native keygen supplies an exact-size iterator to the pinned batch
    # inversion implementation, so its prefix Vec reserves exactly N entries.
    # Count those native-sized (Fr, &mut Fr) entries alongside Option<Fr>.
    batch_invert_heap_upper = n * batch_invert_entry
    pk_phase = selectors + n * option_fr + batch_invert_heap_upper + read_scratch
    # `icicle.rs` materializes one scalar and one affine host Vec per <=2^20
    # MSM chunk. Their exact element sizes come from the compiled BN254 CUB
    # helper; this component is a distinct proof/setup MSM phase.
    icicle_msm_chunk_scalars = 1 << 20
    icicle_msm_staging = icicle_msm_chunk_scalars * (icicle_scalar_bytes + icicle_affine_bytes)
    # During permutation VK commitments, keygen retains the selector bitmap
    # Vecs, omega powers, and the current dense permutation column as well as
    # the ICICLE conversion Vecs.
    permutation_vk_phase = selectors + 2 * n * fr + icicle_msm_staging
    host_known = max(selector_phase, pk_phase, read_scratch, icicle_msm_staging, permutation_vk_phase)
    return {
        "fixed_assignment_mapped_bytes": fixed_assignment_maps,
        "setup_persisted_mapped_bytes": setup_persisted_maps,
        "setup_transition_mapped_bytes": setup_transition_map,
        "setup_disk_known_bytes": setup_mapped_peak,
        "setup_mapped_peak_bytes": setup_mapped_peak,
        "proof_spill_known_bytes": proof_spill_known,
        "setup_key_artifact_upper_bytes": setup_key_artifact_upper,
        "quotient_extended_backing_bytes": proof_domain * fr,
        "quotient_piece_bytes": quotient_coefficients * fr,
        "host_known_bytes": host_known,
        "selector_activation_bitmap_bytes": selectors,
        "selector_dense_field_vector_bytes": n * fr,
        "keygen_option_fr_vector_bytes": n * option_fr,
        "batch_invert_entry_bytes": batch_invert_entry,
        "batch_invert_heap_scratch_bytes_upper": batch_invert_heap_upper,
        "fixed_polynomial_read_heap_scratch_bytes": read_scratch,
        "icicle_msm_chunk_scalars": icicle_msm_chunk_scalars,
        "icicle_scalar_bytes": icicle_scalar_bytes,
        "icicle_affine_bytes": icicle_affine_bytes,
        "icicle_msm_host_staging_bytes": icicle_msm_staging,
        "permutation_vk_msm_host_known_bytes": permutation_vk_phase,
    }


def _check_dataset_capacity(
    dataset: str,
    requirements: dict[str, int],
    *,
    output_root: Path,
    remaining_known_disk_bytes: int,
    phase: str,
    device_sample: dict[str, object],
    workspace_query: dict[str, object],
) -> dict[str, object]:
    host_available, host_accounting = _effective_host_memory_available_bytes()
    disk_available, disk_accounting = _filesystem_available_bytes(output_root.parent)
    device_free = int(device_sample["free_memory_bytes"])
    device_peak = int(workspace_query["device_workspace_peak_bytes"])
    integrated = bool(workspace_query["integrated_device"])
    host_required = requirements["host_known_bytes"] + (device_peak if integrated else 0)
    if host_required > host_available:
        abort(
            f"{dataset} {phase} EZKL-known host allocation gate needs {host_required} bytes "
            f"(integrated device workspace included={integrated}), but only {host_available} are available"
        )
    if remaining_known_disk_bytes > disk_available:
        abort(
            f"{dataset} {phase} EZKL-known mapped/key artifact gate needs {remaining_known_disk_bytes} bytes "
            f"for this phase and the remaining formal chain, but only {disk_available} are available"
        )
    if device_peak > device_free:
        abort(
            f"{dataset} {phase} ICICLE MSM source-sized workspace gate needs {device_peak} CUDA bytes, "
            f"but only {device_free} are currently free on {device_sample['device_uuid']}"
        )
    return {
        "dataset": dataset,
        "phase": phase,
        "host_available_bytes": host_available,
        "host_required_known_bytes": host_required,
        "host_accounting": host_accounting,
        "disk_available_bytes": disk_available,
        "disk_required_known_bytes": remaining_known_disk_bytes,
        "disk_accounting": disk_accounting,
        "cuda_free_bytes": device_free,
        "cuda_required_workspace_bytes": device_peak,
        "cuda_total_bytes": device_sample["total_memory_bytes"],
        "cuda_device_uuid": device_sample["device_uuid"],
        "integrated_device": integrated,
        "coverage": "source-derived-known-allocation-gates; not a complete phase peak estimate",
        "sampled_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _recheck_local_phase_capacity(
    dataset: str,
    requirements: dict[str, int],
    *,
    output_root: Path,
    remaining_known_disk_bytes: int,
    phase: str,
    receipt: dict[str, object],
    session_dir: Path,
    session_id: str,
    selected_device: dict[str, object],
    gpu_identity: dict[str, object],
    build_environment: dict[str, object],
) -> dict[str, object]:
    sample = _sample_native_cuda_memory(
        receipt,
        session_dir=session_dir,
        session_id=session_id,
        selected_device=selected_device,
        gpu_identity=gpu_identity,
        build_environment=build_environment,
    )
    return _check_dataset_capacity(
        dataset,
        requirements,
        output_root=output_root,
        remaining_known_disk_bytes=remaining_known_disk_bytes,
        phase=phase,
        device_sample=sample,
        workspace_query=receipt["query"],
    )


def _admit_local_capacity(
    admissions: dict[str, dict[str, object]],
    *,
    output_root: Path,
    session_dir: Path,
    session_id: str,
    runtime_identity: dict[str, object],
    selected_device: dict[str, object],
    build_environment: dict[str, object],
    gpu_identity: dict[str, object],
) -> tuple[dict[str, object], dict[str, dict[str, int]], dict[str, object]]:
    receipt = _build_native_cub_workspace_query(
        session_dir,
        session_id,
        runtime_identity,
        selected_device,
        build_environment,
        gpu_identity,
    )
    receipt = _validate_native_cub_workspace_receipt(
        receipt, session_dir, selected_device, gpu_identity, session_id
    )
    query = receipt["query"]
    if not isinstance(query, dict):
        abort("EZKL CUB workspace receipt has no decoded query object")
    sizes = query["native_type_sizes"]
    for dataset, admission in admissions.items():
        model = admission["native_memory_model"]
        if (
            sizes["scalar_bytes"] != model["size_of_fr_bytes"]
            or model["cuda_total_memory_bytes"] != query["total_memory_bytes"]
        ):
            abort(f"{dataset} native admission scalar/device values disagree with compiled ICICLE identity")
    requirements = {
        dataset: _native_capacity_requirements(admission, sizes)
        for dataset, admission in admissions.items()
    }
    future_key_artifacts = sum(item["setup_key_artifact_upper_bytes"] for item in requirements.values())
    max_setup_map_peak = max(item["setup_mapped_peak_bytes"] for item in requirements.values())
    initial_disk_required = future_key_artifacts + max_setup_map_peak
    initial_sample = {
        "device_uuid": query["device_uuid"],
        "free_memory_bytes": query["free_memory_bytes"],
        "total_memory_bytes": query["total_memory_bytes"],
        "integrated_device": query["integrated_device"],
        "sampled_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    checks: list[dict[str, object]] = []
    for dataset in DATASETS:
        checks.append(_check_dataset_capacity(
            dataset,
            requirements[dataset],
            output_root=output_root,
            remaining_known_disk_bytes=initial_disk_required,
            phase="initial_five_dataset_admission",
            device_sample=initial_sample,
            workspace_query=query,
        ))
    return receipt, requirements, {
        "schema": "ezkl.local_known_capacity_gate.v1",
        "coverage": "source-derived-known-allocation-gates; not a complete phase peak estimate",
        "native_cub_workspace_query_sha256": receipt["result_sha256"],
        "session_id": session_id,
        "selected_device_uuid": selected_device["uuid"],
        "device_workspace_peak_bytes": query["device_workspace_peak_bytes"],
        "integrated_device": query["integrated_device"],
        "datasets": requirements,
        "initial_checks": checks,
    }


def _remaining_setup_disk_bytes(
    requirements: dict[str, dict[str, int]],
    dataset_index: int,
    *,
    include_current_transient: bool,
) -> int:
    remaining = DATASETS[dataset_index:]
    retained = sum(requirements[name]["setup_key_artifact_upper_bytes"] for name in remaining)
    if include_current_transient:
        retained += requirements[DATASETS[dataset_index]]["setup_mapped_peak_bytes"]
    return retained


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


def _formal_dataset_command_plan(
    binary: Path,
    srs: Path,
    output_root: Path,
    admissions: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    """Build the exact baseline-local setup/producer/prove command sequence."""
    plan: list[dict[str, object]] = []
    for dataset in DATASETS:
        output = output_root / dataset
        admission_dir = local_path(
            str(admissions[dataset]["prepared_dir"]), f"{dataset} admitted circuit directory"
        )
        setup = [
            local_cli(binary, "binary"),
            "gat-prepare",
            "--dataset", dataset,
            "--phase", "setup",
            "--resource-manifest", "resources/resource_manifest.json",
            "--output-dir", local_cli(output, "output-dir"),
            "--admission-dir", local_cli(admission_dir, "admitted circuit"),
            "--srs-path", local_cli(srs, "SRS"),
        ]
        if NATIVE_FORMAL_BACKEND == "gpu":
            setup.append("--gpu")
        producer = [
            "--dataset", dataset,
            "--resource-manifest", "resources/resource_manifest.json",
            "--setup-context", local_cli(output / "producer_context.json", "producer setup context"),
            "--output-dir", local_cli(output, "output-dir"),
            "--config", local_cli(ROOT / "resources/canonical/configs" / f"{dataset}_secure.cfg", "config"),
            "--statement", local_cli(output / "statement.json", "public statement"),
            "--public-output", local_cli(output / "public_output.json", "public output"),
            "--producer-manifest", local_cli(output / "producer_manifest.json", "producer manifest"),
        ]
        single = [
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
        ]
        plan.append({"dataset": dataset, "setup": setup, "producer": producer, "single": single})
    return plan


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
            _set_stage(session_file, "preflight_build_required")
        else:
            print("[ezkl-formal] PREFLIGHT_ONLY_READY: resources/SRS/GPU/build identities passed; no setup/prove was run", flush=True)
            _set_stage(session_file, "preflight_complete")
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
    admissions = native_admit_all_datasets(binary, srs, output_root, session_id, selected_device)
    _set_stage(session_file, "session_bound_icicle_capacity_query")
    session_dir = output_root / ".sessions" / session_id
    cub_receipt, capacity_requirements, capacity_profile = _admit_local_capacity(
        admissions,
        output_root=output_root,
        session_dir=session_dir,
        session_id=session_id,
        runtime_identity=runtime_identity,
        selected_device=selected_device,
        build_environment=build_environment,
        gpu_identity=gpu_identity,
    )
    session_document = json.loads(session_file.read_text())
    session_document["native_admissions"] = admissions
    session_document["native_cub_workspace_receipt"] = cub_receipt
    session_document["local_capacity_profile"] = capacity_profile
    session_document["capacity_checks"] = list(capacity_profile["initial_checks"])
    session_document["resource_manifest_sha256"] = resource_manifest_hash()
    session_document["srs_manifest_sha256"] = sha256(SRS_MANIFEST)
    session_document["source_tree_sha256"] = source_tree_hash()
    session_document["binary_sha256"] = sha256(binary)
    _write_session_record(session_file, session_document)
    cub_receipt = _validate_native_cub_workspace_receipt(
        cub_receipt, session_dir, selected_device, gpu_identity, session_id
    )
    archive_prior_outputs(output_root, matrix, session_id, session_file)

    for dataset_index, item in enumerate(_formal_dataset_command_plan(binary, srs, output_root, admissions)):
        dataset = str(item["dataset"])
        _set_stage(session_file, f"{dataset}_setup_prove_verify")
        setup_check = _recheck_local_phase_capacity(
            dataset,
            capacity_requirements[dataset],
            output_root=output_root,
            remaining_known_disk_bytes=_remaining_setup_disk_bytes(
                capacity_requirements, dataset_index, include_current_transient=True
            ),
            phase="setup",
            receipt=cub_receipt,
            session_dir=session_dir,
            session_id=session_id,
            selected_device=selected_device,
            gpu_identity=gpu_identity,
            build_environment=build_environment,
        )
        session_document = json.loads(session_file.read_text())
        session_document.setdefault("capacity_checks", []).append(setup_check)
        _write_session_record(session_file, session_document)
        output = prepare_output_dir(output_root, dataset)
        setup_ms = run(item["setup"], f"{dataset} native setup", gpu=NATIVE_FORMAL_BACKEND == "gpu")
        (output / "setup_timing.json").write_text(json.dumps({"duration_ms": setup_ms}) + "\n")
        invoke_python_entrypoint(
            "produce_public_statement",
            item["producer"],
            f"{dataset} independent public statement/Y producer",
            context=context,
            session_id=session_id,
        )
        proof_disk_required = (
            _remaining_setup_disk_bytes(
                capacity_requirements, dataset_index + 1, include_current_transient=False
            )
            + capacity_requirements[dataset]["proof_spill_known_bytes"]
        )
        prove_check = _recheck_local_phase_capacity(
            dataset,
            capacity_requirements[dataset],
            output_root=output_root,
            remaining_known_disk_bytes=proof_disk_required,
            phase="witness_prove_verify",
            receipt=cub_receipt,
            session_dir=session_dir,
            session_id=session_id,
            selected_device=selected_device,
            gpu_identity=gpu_identity,
            build_environment=build_environment,
        )
        session_document = json.loads(session_file.read_text())
        session_document.setdefault("capacity_checks", []).append(prove_check)
        _write_session_record(session_file, session_document)
        invoke_python_entrypoint(
            "run_single_dataset",
            item["single"],
            f"{dataset} final/witness/prove/independent verify",
            context=context,
            session_id=session_id,
        )

    _set_stage(session_file, "collect_five_results")
    try:
        finalization_capacity = check_finalization_capacity(
            output_root=output_root,
            matrix_path=matrix,
            session_file=session_file,
            session_id=session_id,
            source_tree_sha256=source_tree_hash(),
            resource_manifest_sha256=resource_manifest_hash(),
            phase="collector",
        )
    except FinalizationCapacityError as exc:
        abort(str(exc))
    session_document = json.loads(session_file.read_text())
    session_document["finalization_capacity"] = finalization_capacity
    _write_session_record(session_file, session_document)
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
    try:
        validation_capacity = check_finalization_capacity(
            output_root=output_root,
            matrix_path=matrix,
            session_file=session_file,
            session_id=session_id,
            source_tree_sha256=source_tree_hash(),
            resource_manifest_sha256=resource_manifest_hash(),
            phase="validator",
        )
    except FinalizationCapacityError as exc:
        abort(str(exc))
    session_document = json.loads(session_file.read_text())
    session_document["final_validation_capacity"] = validation_capacity
    _write_session_record(session_file, session_document)
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
    if not args._from_ezkl_entry:
        abort("start EZKL formal sessions with scripts/run_formal_matrix.py")
    output_root = formal_local_path(args.output_root, "output-root")
    matrix = formal_local_path(args.matrix, "matrix")
    binary_value = args.binary or f"target/{NATIVE_FORMAL_BACKEND}/release/ezkl"
    binary = formal_local_path(binary_value, "binary")
    args.output_root = output_root.relative_to(ROOT).as_posix()
    args.matrix = matrix.relative_to(ROOT).as_posix()
    args.binary = binary.relative_to(ROOT).as_posix()
    check_output_paths(output_root, matrix)
    session_id, session_file, lock_fd = begin_formal_session(output_root, matrix)
    try:
        guardian = ProcessGuardian.start(root=ROOT, session_id=session_id, session_file=session_file, lock_fd=lock_fd)
    except BaseException as exc:
        try:
            update_session_record(
                session_file,
                status="FAILED",
                stage="guardian_start",
                error=f"{type(exc).__name__}: {exc}",
            )
        except BaseException as record_exc:
            print(f"[ezkl-formal] could not record guardian startup failure: {record_exc}", file=sys.stderr, flush=True)
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
        raise
    try:
        result = _run_session(args, session_id=session_id, session_file=session_file)
        guardian.shutdown()
        if args.preflight_only:
            stage = str(json.loads(session_file.read_text()).get("stage", "preflight_complete"))
            status = "PREFLIGHT_BUILD_REQUIRED" if stage == "preflight_build_required" else "PREFLIGHT_ONLY_READY"
        else:
            status, stage = "FINAL_VALIDATION_OK", "final_validation_complete"
        update_session_record(session_file, status=status, stage=stage)
        session_document = json.loads(session_file.read_text())
        session_document["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_session_record(session_file, session_document)
        return result
    except BaseException as exc:
        if session_file.exists() and not session_file.is_symlink():
            try:
                update_session_record(
                    session_file,
                    status="FAILED",
                    stage=str(json.loads(session_file.read_text()).get("stage", "failed")),
                    error=f"{type(exc).__name__}: {exc}",
                )
            except BaseException as record_exc:
                print(f"[ezkl-formal] could not record failed session state: {record_exc}", file=sys.stderr, flush=True)
        try:
            guardian.shutdown()
        except BaseException as guardian_exc:
            print(f"[ezkl-formal] process guardian cleanup remains fail-closed: {guardian_exc}", file=sys.stderr, flush=True)
        raise
    finally:
        # Closing this duplicate does not unlock the shared OFD while the guardian
        # still owns it; never call LOCK_UN before its process-tree cleanup ends.
        os.close(lock_fd)


if __name__ == "__main__":
    try:
        raise SystemExit(main(parser().parse_args()))
    except (ContractError, OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        raise SystemExit(print_error(exc))
