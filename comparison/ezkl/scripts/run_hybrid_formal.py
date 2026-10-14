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
import contextlib
import fcntl
import importlib
import math
import json
import os
import platform
import queue
import re
import signal
import stat
import subprocess
import sys
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


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
    FORMAL_SESSION_ID,
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
    canonical_json_digest,
    validate_config,
    validate_native_srs_manifest,
    native_srs_contract,
    validate_gpu_environment,
    validate_resources,
    validate_formal_python_environment,
    normalize_formal_execution_environment,
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
    result.add_argument("--capacity-preflight-only", action="store_true", help="reuse the current formal GPU build, authenticate all five native Admissions, and stop before Setup")
    result.add_argument("--resume-session", help="resume this exact interrupted/failed formal session")
    result.add_argument("--_session-id", help=argparse.SUPPRESS)
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
            line.split(":", 1)[0]: int(line.split()[1]) * 1024 for line in meminfo if ":" in line and len(line.split()) >= 2 and line.split()[1].isdigit()
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
            profile["cgroup_memory_current_peak_bytes"] = max(int(profile.get("cgroup_memory_current_peak_bytes", 0)), current)
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


def run(command: list[str], label: str, *, gpu: bool = False, log_path: Path | None = None, memory_profile: dict[str, object] | None = None) -> float:
    print(f"[ezkl-formal] {label}: {' '.join(command)}", flush=True)
    log_stream = None
    if log_path is not None:
        if log_path.is_symlink() or (log_path.exists() and not log_path.is_file()):
            abort(f"formal command log path is not a regular file: {log_path}")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_stream = log_path.open("a", encoding="utf-8")
        log_stream.write(f"\n--- {datetime.now(timezone.utc).isoformat()} {label} ---\n")
        log_stream.flush()
        print(f"[ezkl-formal] command output log: {log_path}", flush=True)
    env = os.environ.copy()
    for key in ("ENABLE_ICICLE_GPU", "EZKL_GAT_BACKEND", "ICICLE_BACKEND_INSTALL_DIR", "ICICLE_NATIVE_LIB_DIR", "ICICLE_NATIVE_MANIFEST_SHA256", "CUDA_ARCH"):
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
        process = spawn_guarded_process(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    except BaseException as exc:
        if log_stream is not None:
            log_stream.write(
                f"\n--- {datetime.now(timezone.utc).isoformat()} command_exit_code=not_started "
                f"outcome=spawn_failure exception={type(exc).__name__} ---\n"
            )
            log_stream.flush()
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
        memory_profile.update(
            {
                "schema_version": 1,
                "process_pid": process.pid,
                "process_peak_rss_bytes": 0,
                "process_peak_vms_bytes": 0,
                "host_memory_total_bytes": 0,
                "host_mem_available_bytes_min": None,
                "effective_memory_limit_bytes": None,
                "effective_memory_limit_source": "host",
                "cgroup_memory_current_peak_bytes": 0,
            }
        )
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
            log_stream.write(f"\n--- {datetime.now(timezone.utc).isoformat()} command_exit_code={process.poll()} outcome=interrupted_or_runner_error ---\n")
            log_stream.flush()
            log_stream.close()
        raise
    monitor_stop.set()
    if monitor is not None:
        monitor.join(timeout=2)
        _sample_process_resources(process.pid, memory_profile)
    elapsed_ms = (time.monotonic() - started) * 1000.0
    if log_stream is not None:
        log_stream.write(
            f"\n--- {datetime.now(timezone.utc).isoformat()} command_exit_code={returncode} "
            f"wall_ms={elapsed_ms:.6f} outcome={'success' if returncode == 0 else 'failed'} ---\n"
        )
        log_stream.flush()
        log_stream.close()
    if returncode != 0:
        if terminated_leftovers:
            abort(f"{label} failed with exit code {returncode}; leftover process-group descendants were cleaned")
        abort(f"{label} failed with exit code {returncode}")
    if terminated_leftovers:
        abort(f"{label} left descendants after its leader exited; the group was cleaned")
    return elapsed_ms


def run_prerequisite(command: list[str], label: str, *, timeout_seconds: int, log_path: Path | None = None) -> None:
    print(f"[ezkl-formal] prerequisite {label} (timeout={timeout_seconds}s): {' '.join(command)}", flush=True)
    log_stream = None
    if log_path is not None:
        if log_path.is_symlink() or (log_path.exists() and not log_path.is_file()):
            abort(f"formal prerequisite log path is not a regular file: {log_path}")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_stream = log_path.open("a", encoding="utf-8")
        log_stream.write(f"\n--- {datetime.now(timezone.utc).isoformat()} prerequisite {label} ---\n")
        log_stream.flush()
        print(f"[ezkl-formal] independent build log: {log_path}", flush=True)
    try:
        process = spawn_guarded_process(
            command, cwd=ROOT, stdout=log_stream if log_stream is not None else None, stderr=subprocess.STDOUT if log_stream is not None else None
        )
    except BaseException as exc:
        if log_stream is not None:
            log_stream.write(
                f"\n--- {datetime.now(timezone.utc).isoformat()} prerequisite_exit_code=not_started "
                f"outcome=spawn_failure exception={type(exc).__name__} ---\n"
            )
            log_stream.flush()
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
            log_stream.write(f"\n--- {datetime.now(timezone.utc).isoformat()} prerequisite_exit_code={process.poll()} outcome=timeout ---\n")
            log_stream.flush()
            log_stream.close()
        abort(f"prerequisite {label} timed out after {timeout_seconds}s; process group was terminated")
    except KeyboardInterrupt:
        print(f"[ezkl-formal] interrupted during prerequisite {label}; terminating process group", file=sys.stderr, flush=True)
        _terminate_process_group(process)
        if log_stream is not None:
            log_stream.write(f"\n--- {datetime.now(timezone.utc).isoformat()} prerequisite_exit_code={process.poll()} outcome=interrupted ---\n")
            log_stream.flush()
            log_stream.close()
        raise
    except BaseException:
        print(f"[ezkl-formal] prerequisite {label} failed; terminating process group", file=sys.stderr, flush=True)
        _terminate_process_group(process)
        if log_stream is not None:
            log_stream.write(f"\n--- {datetime.now(timezone.utc).isoformat()} prerequisite_exit_code={process.poll()} outcome=runner_error ---\n")
            log_stream.flush()
            log_stream.close()
        raise
    terminated_leftovers = finish_guarded_process(process) if returncode == 0 else False
    if log_stream is not None:
        log_stream.write(
            f"\n--- {datetime.now(timezone.utc).isoformat()} prerequisite_exit_code={returncode} "
            f"outcome={'success' if returncode == 0 else 'failed'} ---\n"
        )
        log_stream.flush()
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


def run_resource_prepare_with_auth_handoff(command: list[str], *, session_id: str, timeout_seconds: int) -> dict[str, object]:
    """Stream the official auto-preparer and capture its one complete-auth result."""
    marker_prefix = "EZKL_FORMAL_RESOURCE_AUTH_V1="
    print(f"[ezkl-formal] prerequisite official resource preparation/authentication " f"(timeout={timeout_seconds}s): {' '.join(command)}", flush=True)
    env = os.environ.copy()
    env.pop("EZKL_FORMAL_RESOURCE_AUTH_SESSION_ID", None)
    env["EZKL_FORMAL_RESOURCE_AUTH_SESSION_ID"] = session_id
    process = spawn_guarded_process(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
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
                    parsed = json.loads(line[len(marker_prefix) :])
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


def _authenticate_resources_with_handoff(
    python: str,
    *,
    session_id: str,
    preflight_config_sha256s: tuple[tuple[str, str], ...],
    command: list[str],
    label: str,
) -> tuple[tuple[tuple[str, tuple[int, int, int, int, int]], ...], tuple[tuple[str, str], ...]]:
    """Run one complete resource authentication and accept its current-session handoff."""
    source_before = source_tree_hash()
    config_before = tuple(preflight_config_sha256s)
    if any(sha256(ROOT / "resources/canonical/configs" / f"{dataset}_secure.cfg") != digest for dataset, digest in config_before):
        abort(f"secure config changed before {label}")
    marker = run_resource_prepare_with_auth_handoff(command, session_id=session_id, timeout_seconds=21600)
    _preflight_dataset_configs()
    config_after = tuple((dataset, sha256(ROOT / "resources/canonical/configs" / f"{dataset}_secure.cfg")) for dataset in DATASETS)
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
    print(f"[ezkl-formal] resource-authentication=PASS files={len(identities_after)} handoff={label} current-session", flush=True)
    return identities_after, config_after


def _prepare_and_authenticate_resources(
    python: str, *, session_id: str, preflight_config_sha256s: tuple[tuple[str, str], ...]
) -> tuple[tuple[tuple[str, tuple[int, int, int, int, int]], ...], tuple[tuple[str, str], ...]]:
    prepare_script = ROOT / "scripts" / "prepare_gat_resources.py"
    require_file(prepare_script, "canonical resource preparation entrypoint")
    return _authenticate_resources_with_handoff(
        python,
        session_id=session_id,
        preflight_config_sha256s=preflight_config_sha256s,
        command=[python, "-u", str(prepare_script), "--formal-auto"],
        label="official-formal-auto",
    )


def _authenticate_current_resources(
    python: str, *, session_id: str, preflight_config_sha256s: tuple[tuple[str, str], ...]
) -> tuple[tuple[tuple[str, tuple[int, int, int, int, int]], ...], tuple[tuple[str, str], ...]]:
    validator = ROOT / "scripts" / "validate_resources.py"
    require_file(validator, "canonical resource validator")
    return _authenticate_resources_with_handoff(
        python,
        session_id=session_id,
        preflight_config_sha256s=preflight_config_sha256s,
        command=[python, "-u", str(validator), "--require-complete"],
        label="post-SRS-complete-validation",
    )


def _validate_config_srs_bindings(configs: dict[str, dict[str, str]], srs_manifest: dict[str, object]) -> None:
    entries = srs_manifest.get("entries")
    if not isinstance(entries, dict) or set(entries) != set(DATASETS):
        abort("authenticated SRS manifest does not contain exactly five dataset bindings")
    for dataset in DATASETS:
        entry = entries[dataset]
        values = configs[dataset]
        if not isinstance(entry, dict) or (
            values.get("srs_id"),
            values.get("srs_path"),
            values.get("trusted_srs_sha256"),
        ) != (entry.get("srs_id"), entry.get("path"), entry.get("sha256")):
            abort(f"{dataset} secure config SRS identity does not match authenticated native SRS manifest")


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


def reusable_build(binary: Path, backend: str, runtime_identity: dict[str, object], build_environment: dict[str, object] | None = None) -> bool:
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
        "backend": backend,
        "binary_path": binary.relative_to(ROOT).as_posix(),
        "binary_sha256": sha256(binary),
        "source_tree_sha256": source_tree_hash(),
        "resource_manifest_sha256": resource_manifest_hash(),
        "adapter_contract_id": contract["contract_id"],
        "adapter_contract_sha256": sha256(CONTRACT),
        "rustc": runtime_identity["rustc"],
        "cargo": runtime_identity["cargo"],
        "rustc_path": runtime_identity["rustc_path"],
        "cargo_path": runtime_identity["cargo_path"],
        "rustdoc": runtime_identity["rustdoc"],
        "rustdoc_path": runtime_identity["rustdoc_path"],
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


def _formal_gpu_environment(session_id: str) -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
    """Resolve the same strict host/device identities used by formal execution."""
    if FORMAL_SESSION_ID.fullmatch(session_id) is None:
        abort("formal GPU environment requires one canonical session id")
    runtime_identity = validate_formal_python_environment()
    selected_device = select_formal_cuda_device(runtime_identity["nvidia_smi_path"])
    os.environ["EZKL_FORMAL_SELECTED_CUDA_DEVICE"] = json.dumps(selected_device, sort_keys=True, separators=(",", ":"))
    build_environment = formal_build_environment_identity(runtime_identity, selected_device)
    gpu_identity = validate_gpu_environment()
    return runtime_identity, selected_device, build_environment, gpu_identity


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


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_session_record(path: Path, document: dict[str, object]) -> None:
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(document, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    _fsync_directory(path.parent)


def _load_resume_identity(session_file: Path) -> dict[str, object]:
    document = load_json(session_file, "resume session record")
    identity = document.get("resume_identity") if isinstance(document, dict) else None
    if not isinstance(identity, dict) or identity.get("schema") != "ezkl-formal-resume-identity-v1":
        abort("resume session has no supported immutable identity receipt")
    return identity


def _require_resume_identity_subset(expected: dict[str, object], current: dict[str, object], label: str) -> None:
    for key, value in current.items():
        if expected.get(key) != value:
            abort(f"resume {label} identity changed: {key}")


def _formal_resume_identity(
    *,
    runtime_identity: dict[str, object],
    selected_device: dict[str, object],
    build_environment: dict[str, object],
    gpu_identity: dict[str, object],
    preflight_config_sha256s: tuple[tuple[str, str], ...],
    srs: Path,
    binary: Path,
) -> dict[str, object]:
    build_receipt = ROOT / "artifacts" / "build" / NATIVE_FORMAL_BACKEND / "build_receipt.json"
    return {
        "schema": "ezkl-formal-resume-identity-v1",
        "source_tree_sha256": source_tree_hash(),
        "resource_manifest_sha256": resource_manifest_hash(),
        "config_sha256s": [[dataset, digest] for dataset, digest in preflight_config_sha256s],
        "srs_manifest_sha256": sha256(SRS_MANIFEST),
        "srs_path": srs.relative_to(ROOT).as_posix(),
        "srs_sha256": sha256(srs),
        "binary_path": binary.relative_to(ROOT).as_posix(),
        "binary_sha256": sha256(binary),
        "build_receipt_sha256": sha256(build_receipt),
        "build_environment": build_environment,
        "build_environment_sha256": build_environment["build_environment_sha256"],
        "runtime_identity": runtime_identity,
        "selected_cuda_device": selected_device,
        "gpu_identity": gpu_identity,
    }


def _validate_native_admission_domain(dataset: str, admission: dict[str, object], source_max_logrows: int) -> None:
    """Bound admission geometry against the trusted native SRS before shifting."""
    logrows = admission.get("logrows")
    max_logrows = admission.get("max_public_srs_logrows")
    domain_size = admission.get("domain_size")
    if (
        isinstance(logrows, bool)
        or not isinstance(logrows, int)
        or logrows <= 0
        or isinstance(max_logrows, bool)
        or not isinstance(max_logrows, int)
        or max_logrows <= 0
        or max_logrows != source_max_logrows
        or logrows > max_logrows
        or isinstance(domain_size, bool)
        or not isinstance(domain_size, int)
        or domain_size <= 0
    ):
        abort(f"{dataset} native admission logrows/domain exceed the authenticated native SRS contract")
    if domain_size != 1 << logrows:
        abort(f"{dataset} native admission domain does not match its bounded logrows")


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


def begin_formal_session(
    output_root: Path,
    matrix: Path,
    resume_session_id: str | None = None,
    requested_session_id: str | None = None,
) -> tuple[str, Path, int]:
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
        session_dir = output_root / ".sessions"
        if session_dir.is_symlink() or (session_dir.exists() and not session_dir.is_dir()):
            abort(f"formal session history path must be a regular directory: {session_dir}")
        session_dir.mkdir(parents=True, exist_ok=True)
        if resume_session_id is not None:
            if requested_session_id is not None:
                abort("a fresh internal session id cannot be combined with resume")
            if FORMAL_SESSION_ID.fullmatch(resume_session_id) is None:
                abort("resume session id must be one exact 32-character lowercase hexadecimal value")
            session_id = resume_session_id
            session_file = session_dir / f"{session_id}.json"
            if session_file.is_symlink() or not session_file.is_file():
                abort(f"resume session record is missing or symlinked: {session_file}")
            try:
                document = json.loads(session_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                abort(f"cannot read resume session receipt: {exc}")
            if (
                not isinstance(document, dict)
                or document.get("schema_version") != 1
                or document.get("baseline") != "ezkl"
                or document.get("session_id") != session_id
                or document.get("status") not in {"FAILED", "INTERRUPTED"}
                or document.get("output_root") != output_root.relative_to(ROOT).as_posix()
                or document.get("matrix") != matrix.relative_to(ROOT).as_posix()
                or not isinstance(document.get("resume_identity"), dict)
            ):
                abort("resume session identity/status/output paths are not eligible for exact-session recovery")
            document["status"] = "PREFLIGHT_STARTED"
            document["stage"] = "session_resume_started"
            document["resumed_at_utc"] = datetime.now(timezone.utc).isoformat()
            document["resume_count"] = int(document.get("resume_count", 0)) + 1
            document.pop("error", None)
            _write_session_record(session_file, document)
            print(f"[ezkl-formal] resuming exact interrupted session {session_id}", flush=True)
        else:
            session_id = requested_session_id or uuid.uuid4().hex
            if FORMAL_SESSION_ID.fullmatch(session_id) is None:
                abort("formal session id must be one exact 32-character lowercase hexadecimal value")
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
                    "output_root": output_root.relative_to(ROOT).as_posix(),
                    "matrix": matrix.relative_to(ROOT).as_posix(),
                },
            )
            print(f"[ezkl-formal] started formal session {session_id}; prior outputs remain intact until admission passes", flush=True)
        lock_payload = (
            json.dumps(
                {
                    "schema_version": 1,
                    "baseline": "ezkl",
                    "pid": os.getpid(),
                    "session_id": session_id,
                    "session_file": session_file.relative_to(ROOT).as_posix(),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            + b"\n"
        )
        os.ftruncate(lock_fd, 0)
        os.lseek(lock_fd, 0, os.SEEK_SET)
        os.write(lock_fd, lock_payload)
        os.fsync(lock_fd)
        return session_id, session_file, lock_fd
    except BaseException:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
        raise


def archive_prior_outputs(output_root: Path, matrix: Path, session_id: str, session_file: Path) -> None:
    """Archive prior outputs after the first dataset passes its exact capacity gate."""
    document = json.loads(session_file.read_text())
    if document.get("outputs_archived_for_session") is True:
        return
    for dataset in DATASETS:
        old_output = output_root / dataset
        if (old_output.exists() or old_output.is_symlink()) and (old_output.is_symlink() or not old_output.is_dir()):
            abort(f"prior formal output is not a regular directory: {old_output}")
    if (matrix.exists() or matrix.is_symlink()) and (matrix.is_symlink() or not matrix.is_file()):
        abort(f"prior formal matrix is not a regular file: {matrix}")
    archive_root = output_root / ".incomplete" / session_id / "previous"
    archive_root.mkdir(parents=True, exist_ok=True)
    archived_value = document.get("archived_previous_outputs", [])
    if not isinstance(archived_value, list) or any(not isinstance(value, str) for value in archived_value):
        abort("formal prior-output archive receipt is malformed")
    archived: list[str] = list(archived_value)
    for dataset in DATASETS:
        old_output = output_root / dataset
        destination = archive_root / "datasets" / dataset
        if destination.is_symlink() or (destination.exists() and not destination.is_dir()):
            abort(f"archived prior output is not a regular directory: {destination}")
        if old_output.exists() and destination.exists():
            abort(f"archive already contains a conflicting prior output: {destination}")
        if old_output.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            old_output.rename(destination)
            _fsync_directory(old_output.parent)
            _fsync_directory(destination.parent)
            print(f"[ezkl-formal] archived prior result directory: {old_output} -> {destination}", flush=True)
        if destination.exists() and str(destination.relative_to(ROOT)) not in archived:
            archived.append(str(destination.relative_to(ROOT)))
            document["archived_previous_outputs"] = archived
            _write_session_record(session_file, document)
    matrix_archive = archive_root / "matrix" / matrix.name
    if matrix_archive.is_symlink() or (matrix_archive.exists() and not matrix_archive.is_file()):
        abort(f"archived prior matrix is not a regular file: {matrix_archive}")
    if matrix.exists() and matrix_archive.exists():
        abort(f"archive already contains a conflicting prior matrix: {matrix_archive}")
    if matrix.exists():
        matrix_archive.parent.mkdir(parents=True, exist_ok=True)
        matrix.rename(matrix_archive)
        _fsync_directory(matrix.parent)
        _fsync_directory(matrix_archive.parent)
    if matrix_archive.exists() and str(matrix_archive.relative_to(ROOT)) not in archived:
        archived.append(str(matrix_archive.relative_to(ROOT)))
        print(f"[ezkl-formal] archived prior active matrix: {matrix} -> {matrix_archive}", flush=True)
        document["archived_previous_outputs"] = archived
        _write_session_record(session_file, document)
    document["archived_previous_outputs"] = archived
    document["archived_at_utc"] = datetime.now(timezone.utc).isoformat()
    document["outputs_archived_for_session"] = True
    _write_session_record(session_file, document)


def _quarantine_session_path(path: Path, output_root: Path, session_id: str, label: str) -> Path:
    if path.is_symlink():
        abort(f"refusing to quarantine a symlinked same-session stage: {path}")
    try:
        path.relative_to(output_root / ".sessions" / session_id)
    except ValueError:
        try:
            path.relative_to(output_root)
        except ValueError:
            abort(f"same-session quarantine target escapes output root: {path}")
    quarantine_root = output_root / ".incomplete" / session_id / "quarantine"
    quarantine_root.mkdir(parents=True, exist_ok=True)
    destination = quarantine_root / f"{label}-{time.time_ns()}"
    path.rename(destination)
    print(f"[ezkl-formal] quarantined incomplete same-session {label}: {path} -> {destination}", flush=True)
    return destination


def _validate_native_shplonk_shape(dataset: str, memory: dict[str, object], cs: dict[str, object], domain_size: int) -> None:
    """Validate SHPLONK shape counts, allowing zero only for absent shuffles."""

    shplonk_counts = memory.get("shplonk_rotation_set_commitment_counts")
    positive_count_fields = (
        "shplonk_query_polynomial_count",
        "shplonk_rotation_set_count",
        "shplonk_max_rotation_count",
        "shplonk_unique_opening_point_count",
        "shplonk_max_commitments_per_rotation_set",
        "shplonk_permutation_product_set_count",
        "shplonk_advice_query_polynomial_count",
        "shplonk_fixed_query_polynomial_count",
        "shplonk_lookup_argument_count",
        "shplonk_proof_instance_count",
        "shplonk_quotient_piece_count",
    )
    shuffle_count = memory.get("shplonk_shuffle_argument_count")
    if (
        not isinstance(shplonk_counts, list)
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in shplonk_counts)
        or any(
            isinstance(memory.get(name), bool)
            or not isinstance(memory.get(name), int)
            or memory[name] <= 0
            for name in positive_count_fields
        )
        or isinstance(shuffle_count, bool)
        or not isinstance(shuffle_count, int)
        or shuffle_count < 0
    ):
        abort(f"{dataset} native admission has malformed SHPLONK shape counts")

    degree = int(cs["degree"])
    permutation_columns = int(cs["num_permutation_columns"])
    lookup_arguments = int(cs["num_lookup_arguments"])
    shuffle_arguments = int(cs["num_shuffle_arguments"])
    shplonk_permutation_chunk = max(degree - 2, 1)
    expected_shplonk_permutation_sets = (permutation_columns + shplonk_permutation_chunk - 1) // shplonk_permutation_chunk
    expected_shplonk_queries = (
        int(memory["shplonk_advice_query_polynomial_count"])
        + int(memory["shplonk_fixed_query_polynomial_count"])
        + (expected_shplonk_permutation_sets + 2 * lookup_arguments + shuffle_arguments)
        * int(memory["shplonk_proof_instance_count"])
        + permutation_columns
        + (degree - 1)
        + 1
    )
    if (
        len(shplonk_counts) != memory["shplonk_rotation_set_count"]
        or sum(shplonk_counts) != memory["shplonk_query_polynomial_count"]
        or max(shplonk_counts, default=0) != memory["shplonk_max_commitments_per_rotation_set"]
        or memory["shplonk_query_polynomial_count"] != expected_shplonk_queries
        or memory["shplonk_permutation_product_set_count"] != expected_shplonk_permutation_sets
        or memory["shplonk_advice_query_polynomial_count"] > cs["num_advice_columns"]
        or memory["shplonk_fixed_query_polynomial_count"] > cs["fixed_column_count_post_compression"]
        or memory["shplonk_lookup_argument_count"] != lookup_arguments
        or shuffle_count != shuffle_arguments
        or memory["shplonk_proof_instance_count"] != 1
        or memory["shplonk_quotient_piece_count"] != degree - 1
        or memory["shplonk_max_rotation_count"] > memory["shplonk_unique_opening_point_count"]
        or memory["shplonk_unique_opening_point_count"] > domain_size
    ):
        abort(f"{dataset} native SHPLONK opening shape is inconsistent with Halo2 geometry")


def native_admit_all_datasets(
    binary: Path,
    srs: Path,
    output_root: Path,
    session_id: str,
    selected_device: dict[str, object],
    *,
    datasets: tuple[str, ...] = DATASETS,
    baseline_root: Path = ROOT,
    command_runner: Callable[[list[str], str], None] | None = None,
    reuse_receipt: object = None,
    receipt_callback: Callable[[str, dict[str, object]], None] | None = None,
) -> dict[str, dict[str, object]]:
    if not datasets or len(set(datasets)) != len(datasets) or any(dataset not in DATASETS for dataset in datasets):
        abort("native admission dataset selection is empty, duplicated, or unsupported")
    if baseline_root.is_symlink():
        abort("native admission baseline root is symlinked")
    baseline_root = baseline_root.resolve()
    manifest_path = baseline_root / "resources/resource_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        abort("native admission baseline root or resource manifest is missing or symlinked")
    manifest_sha256 = sha256(manifest_path)
    native_srs_max_logrows = native_srs_contract()[1]

    def cli_path(path: Path, label: str) -> str:
        try:
            return path.resolve(strict=False).relative_to(baseline_root).as_posix()
        except ValueError:
            abort(f"{label} is outside the native admission baseline: {path}")

    admission_root = output_root / ".sessions" / session_id / "admission"
    if reuse_receipt is not None and (
        not isinstance(reuse_receipt, dict)
        or any(dataset not in DATASETS or not isinstance(value, dict) for dataset, value in reuse_receipt.items())
    ):
        abort("saved native admission receipt map is malformed")
    if admission_root.is_symlink() or (admission_root.exists() and not admission_root.is_dir()):
        abort("native admission session directory is missing or symlinked")
    admission_root.mkdir(parents=True, exist_ok=True)
    admissions: dict[str, dict[str, object]] = {}
    expected_keys = {
        "schema_version",
        "relation_id",
        "dataset",
        "model_arch_id",
        "quant_cfg_id",
        "logrows",
        "domain_size",
        "max_public_srs_logrows",
        "num_rows",
        "total_assignments",
        "total_const_size",
        "total_dynamic_col_size",
        "max_dynamic_input_len",
        "num_dynamic_lookups",
        "num_shuffles",
        "module_sizes",
        "constraint_system",
        "scalar_domain_vector_bytes",
        "resource_manifest_sha256",
        "relation_identity",
        "settings_sha256",
        "native_memory_model",
    }
    cs_keys = {
        "degree",
        "minimum_rows",
        "num_fixed_columns",
        "num_advice_columns",
        "num_instance_columns",
        "num_selectors",
        "num_custom_gate_groups",
        "num_lookup_arguments",
        "num_shuffle_arguments",
        "num_permutation_columns",
        "num_blinding_factors",
        "selector_fixed_column_count_upper",
        "fixed_column_count_post_compression",
        "permutation_copy_call_count_upper",
        "permutation_successor_entry_count_upper",
        "num_lookup_query_polynomials",
        "max_lookup_query_polynomials_per_argument",
        "num_unique_lookup_table_expression_groups",
    }
    for dataset in datasets:
        admission_dir = admission_root / dataset
        prior = reuse_receipt.get(dataset) if isinstance(reuse_receipt, dict) else None
        reuse = prior is not None
        if reuse:
            expected_dir = str(admission_dir.relative_to(baseline_root))
            if (
                not isinstance(prior, dict)
                or prior.get("prepared_dir") != expected_dir
                or isinstance(prior.get("sha256"), bool)
                or not isinstance(prior.get("sha256"), str)
                or re.fullmatch(r"[0-9a-f]{64}", prior["sha256"]) is None
                or admission_dir.is_symlink()
                or not admission_dir.is_dir()
            ):
                abort(f"{dataset} saved native admission identity is malformed")
        else:
            if admission_dir.exists() or admission_dir.is_symlink():
                _quarantine_session_path(admission_dir, output_root, session_id, f"{dataset}-admission")
            admission_dir.mkdir()
            command = [
                cli_path(binary, "binary"),
                "gat-prepare",
                "--dataset",
                dataset,
                "--phase",
                "admission",
                "--resource-manifest",
                cli_path(manifest_path, "resource manifest"),
                "--output-dir",
                cli_path(admission_dir, "admission output"),
                "--srs-path",
                cli_path(srs, "SRS"),
                "--gpu",
            ]
            label = f"{dataset} native full-relation/capacity admission"
            if command_runner is None:
                run(command, label, gpu=True)
            else:
                command_runner(command, label)
        admission_path = admission_dir / "admission.json"
        if reuse and sha256(admission_path) != prior["sha256"]:
            abort(f"{dataset} saved native admission digest changed; refusing to reuse it")
        admission = load_json(admission_path, f"{dataset} native admission")
        if not isinstance(admission, dict) or set(admission) != expected_keys:
            abort(f"{dataset} native admission JSON has an incompatible schema")
        _validate_native_admission_domain(dataset, admission, native_srs_max_logrows)
        if (
            admission["schema_version"] != 7
            or admission["relation_id"] != "R_GAT-ezkl"
            or admission["dataset"] != dataset
        ):
            abort(f"{dataset} native admission failed relation/logrows/SRS checks")
        relation_identity = admission["relation_identity"]
        if (
            not isinstance(relation_identity, dict)
            or relation_identity.get("dataset") != dataset
            or relation_identity.get("resource_manifest_sha256") != manifest_sha256
            or admission["resource_manifest_sha256"] != manifest_sha256
            or not isinstance(admission["settings_sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", admission["settings_sha256"]) is None
        ):
            abort(f"{dataset} native admission does not bind its canonical resources and settings file")
        handoff_file = admission_dir / "settings.json"
        require_file(handoff_file, f"{dataset} admitted settings")
        if sha256(handoff_file) != admission["settings_sha256"]:
            abort(f"{dataset} native admission settings digest mismatch")
        cs = admission["constraint_system"]
        if not isinstance(cs, dict) or set(cs) != cs_keys or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in cs.values()):
            abort(f"{dataset} native admission has malformed exact Halo2 constraint-shape counts")
        if cs["minimum_rows"] <= 0 or admission["domain_size"] < cs["minimum_rows"]:
            abort(f"{dataset} native admission domain is below the current Halo2 minimum row requirement")
        if cs["selector_fixed_column_count_upper"] != cs["num_selectors"] or cs["fixed_column_count_post_compression"] != (cs["num_fixed_columns"] + cs["selector_fixed_column_count_upper"]):
            abort(f"{dataset} native admission selector/fixed-column upper bound is inconsistent")
        expected_permutation_cells = cs["num_permutation_columns"] * admission["domain_size"]
        if cs["permutation_copy_call_count_upper"] != expected_permutation_cells:
            abort(f"{dataset} native admission permutation synthesis upper bound is inconsistent")
        expected_successor_entries = min(expected_permutation_cells, 2 * cs["permutation_copy_call_count_upper"])
        if cs["permutation_successor_entry_count_upper"] != expected_successor_entries:
            abort(f"{dataset} native admission compact permutation entry bound is inconsistent")
        memory = admission["native_memory_model"]
        memory_keys = {
            "schema",
            "evidence_kind",
            "coverage",
            "native_target_arch",
            "uses_recursive_fft",
            "proof_logrows",
            "proof_domain_size",
            "recursive_fft_cached_domain_lengths",
            "recursive_fft_twiddle_field_elements",
            "fft_cache_supported_length_count",
            "fft_cache_entry_capacity",
            "fft_parallel_thread_count",
            "srs_fft_cache_supported_length_count",
            "srs_fft_cache_entry_capacity",
            "parallel_fft_lookup_table_elements",
            "size_of_fftdata_bytes",
            "size_of_fftdata_once_lock_bytes",
            "size_of_fftdata_cache_entry_bytes",
            "size_of_fftdata_cache_map_bytes",
            "size_of_arc_header_bytes",
            "custom_gate_dependency_columns",
            "lookup_dependency_columns",
            "lookup_query_polynomial_counts",
            "shuffle_dependency_columns",
            "advice_column_phase_ids",
            "size_of_fr_bytes",
            "size_of_assigned_fr_bytes",
            "size_of_option_fr_bytes",
            "size_of_batch_invert_entry_bytes",
            "size_of_usize_bytes",
            "size_of_usize_pair_bytes",
            "size_of_permutation_hashmap_header_bytes",
            "size_of_g1_affine_bytes",
            "size_of_g1_projective_bytes",
            "size_of_g2_affine_bytes",
            "size_of_fft_stage_bytes",
            "size_of_atomic_u64_bytes",
            "size_of_lookup_map_entry_bytes",
            "size_of_lookup_map_header_bytes",
            "size_of_vec_u8_bytes",
            "size_of_lookup_repr_bytes",
            "lookup_table_cardinality_upper_bounds",
            "size_of_vec_header_bytes",
            "size_of_polynomial_header_bytes",
            "size_of_mv_lookup_prepared_bytes",
            "size_of_mv_lookup_prepared_table_bytes",
            "size_of_mv_lookup_committed_bytes",
            "size_of_shplonk_commitment_extension_bytes",
            "size_of_shplonk_rotation_set_extension_bytes",
            "size_of_coeff_polynomial_header_bytes",
            "shplonk_rotation_set_commitment_counts",
            "shplonk_query_polynomial_count",
            "shplonk_rotation_set_count",
            "shplonk_max_rotation_count",
            "shplonk_unique_opening_point_count",
            "shplonk_max_commitments_per_rotation_set",
            "shplonk_permutation_product_set_count",
            "shplonk_advice_query_polynomial_count",
            "shplonk_fixed_query_polynomial_count",
            "shplonk_lookup_argument_count",
            "shplonk_shuffle_argument_count",
            "shplonk_proof_instance_count",
            "shplonk_quotient_piece_count",
            "graph_witness_scalar_element_count",
            "size_of_graph_witness_scalar_bytes",
            "size_of_gat_valtype_bytes",
            "gat_transient_witness_bytes_upper",
            "gat_transient_witness_evidence_kind",
            "gat_transient_witness_formula_id",
            "advice_event_record_bytes_upper",
            "advice_event_reservation_chunk_bytes",
            "advice_event_batch_size_upper",
            "advice_event_decoded_entry_bytes_upper",
            "advice_event_read_buffer_bytes_upper",
            "advice_event_column_metadata_bytes_upper",
            "advice_event_writer_buffer_bytes_upper",
            "advice_event_column_bytes_upper",
            "advice_event_latest_element_bytes",
            "advice_event_log_bound_evidence_kind",
            "cuda_uuid",
            "cuda_free_bytes_at_admission",
            "cuda_total_memory_bytes",
            "selector_activation_bitmap_bytes_upper",
            "selector_activation_storage_bytes_upper",
            "selector_activation_storage_kind",
            "fixed_polynomial_read_mode",
            "fixed_polynomial_read_chunk_fields_upper",
            "fixed_polynomial_read_raw_chunk_bytes_upper",
            "fixed_polynomial_read_decoded_chunk_bytes_upper",
            "fixed_polynomial_read_heap_scratch_bytes_upper",
            "native_pk_read_buffer_capacity_bytes",
            "native_successor_read_cache_bytes",
            "native_target_bitmap_fixed_cache_bytes",
        }
        if not isinstance(memory, dict) or set(memory) != memory_keys or memory.get("schema") != "ezkl-halo2-static-envelope-v11" or memory.get("evidence_kind") != "source_derived_upper_bounds_and_configured_geometry" or memory.get("coverage") != "static_pre_setup_envelope; exact_native_selector_and_permutation_observations_are_required_from_keygen_receipt":
            abort(f"{dataset} native admission has an incompatible source memory model")
        positive_memory_keys = {
            "proof_logrows",
            "proof_domain_size",
            "size_of_fr_bytes",
            "size_of_assigned_fr_bytes",
            "size_of_option_fr_bytes",
            "size_of_batch_invert_entry_bytes",
            "size_of_usize_bytes",
            "size_of_usize_pair_bytes",
            "size_of_permutation_hashmap_header_bytes",
            "size_of_g1_affine_bytes",
            "size_of_g1_projective_bytes",
            "size_of_g2_affine_bytes",
            "size_of_fft_stage_bytes",
            "size_of_atomic_u64_bytes",
            "size_of_lookup_map_entry_bytes",
            "size_of_lookup_map_header_bytes",
            "size_of_vec_u8_bytes",
            "size_of_lookup_repr_bytes",
            "size_of_graph_witness_scalar_bytes",
            "size_of_gat_valtype_bytes",
            "gat_transient_witness_bytes_upper",
            "advice_event_record_bytes_upper",
            "advice_event_reservation_chunk_bytes",
            "advice_event_batch_size_upper",
            "advice_event_decoded_entry_bytes_upper",
            "advice_event_read_buffer_bytes_upper",
            "advice_event_column_metadata_bytes_upper",
            "advice_event_writer_buffer_bytes_upper",
            "advice_event_column_bytes_upper",
            "advice_event_latest_element_bytes",
            "size_of_vec_header_bytes",
            "size_of_polynomial_header_bytes",
            "size_of_mv_lookup_prepared_bytes",
            "size_of_mv_lookup_prepared_table_bytes",
            "size_of_mv_lookup_committed_bytes",
            "size_of_shplonk_commitment_extension_bytes",
            "size_of_shplonk_rotation_set_extension_bytes",
            "size_of_coeff_polynomial_header_bytes",
            "size_of_fftdata_bytes",
            "size_of_fftdata_once_lock_bytes",
            "size_of_fftdata_cache_entry_bytes",
            "size_of_fftdata_cache_map_bytes",
            "fft_cache_supported_length_count",
            "fft_cache_entry_capacity",
            "fft_parallel_thread_count",
            "srs_fft_cache_supported_length_count",
            "srs_fft_cache_entry_capacity",
            "size_of_arc_header_bytes",
            "parallel_fft_lookup_table_elements",
            "cuda_total_memory_bytes",
            "fixed_polynomial_read_chunk_fields_upper",
            "fixed_polynomial_read_raw_chunk_bytes_upper",
            "fixed_polynomial_read_decoded_chunk_bytes_upper",
            "fixed_polynomial_read_heap_scratch_bytes_upper",
            "native_pk_read_buffer_capacity_bytes",
            "native_successor_read_cache_bytes",
            "native_target_bitmap_fixed_cache_bytes",
        }
        if any(isinstance(memory.get(key), bool) or not isinstance(memory.get(key), int) or memory[key] <= 0 for key in positive_memory_keys) or (
            isinstance(memory.get("cuda_free_bytes_at_admission"), bool)
            or not isinstance(memory.get("cuda_free_bytes_at_admission"), int)
            or memory["cuda_free_bytes_at_admission"] < 0
        ):
            abort(f"{dataset} native admission has non-positive or malformed source memory values")
        expected_scalar_domain_bytes = admission["domain_size"] * (
            cs["num_fixed_columns"] + cs["num_advice_columns"] + cs["num_instance_columns"]
        ) * memory["size_of_fr_bytes"]
        scalar_domain_bytes = admission["scalar_domain_vector_bytes"]
        if (
            isinstance(scalar_domain_bytes, bool)
            or not isinstance(scalar_domain_bytes, int)
            or scalar_domain_bytes <= 0
            or scalar_domain_bytes != expected_scalar_domain_bytes
        ):
            abort(f"{dataset} native admission scalar-domain resource byte count is inconsistent with its domain/columns")
        if (
            isinstance(memory.get("graph_witness_scalar_element_count"), bool)
            or not isinstance(memory.get("graph_witness_scalar_element_count"), int)
            or memory["graph_witness_scalar_element_count"] < 0
        ):
            abort(f"{dataset} native admission has a malformed graph-witness shape count")
        cardinality_bounds = memory["lookup_table_cardinality_upper_bounds"]
        if (
            not isinstance(cardinality_bounds, list)
            or len(cardinality_bounds) != cs["num_unique_lookup_table_expression_groups"]
            or any(isinstance(bound, bool) or not isinstance(bound, int) or bound <= 0 or bound > admission["domain_size"] for bound in cardinality_bounds)
            or cs["num_lookup_query_polynomials"] < cs["num_lookup_arguments"]
            or cs["max_lookup_query_polynomials_per_argument"] > cs["num_lookup_query_polynomials"]
            or cs["num_unique_lookup_table_expression_groups"] > cs["num_lookup_arguments"]
        ):
            abort(f"{dataset} native mv-lookup shape bounds are malformed or inconsistent")
        expected_proof_logrows = (admission["domain_size"] * (cs["degree"] - 1) - 1).bit_length()
        expected_recursive_fft = platform.machine() != "x86_64"
        expected_twiddle_elements = 2 * (memory["proof_domain_size"] - 1 + expected_proof_logrows) if expected_recursive_fft else 0
        expected_parallel_lut_elements = 1 << expected_proof_logrows if expected_proof_logrows < 10 else (1 << 10) + (1 << (expected_proof_logrows - 10))
        if (
            memory["proof_domain_size"] != 1 << expected_proof_logrows
            or memory["proof_logrows"] != expected_proof_logrows
            or memory["native_target_arch"] != platform.machine()
            or not isinstance(memory["uses_recursive_fft"], bool)
            or memory["uses_recursive_fft"] != expected_recursive_fft
            or memory["recursive_fft_twiddle_field_elements"] != expected_twiddle_elements
            or memory["recursive_fft_cached_domain_lengths"] != ([memory["proof_domain_size"]] if expected_recursive_fft else [])
            or memory["fft_cache_supported_length_count"] != expected_proof_logrows - admission["logrows"] + 1
            or memory["fft_cache_entry_capacity"] < memory["fft_cache_supported_length_count"]
            or memory["srs_fft_cache_supported_length_count"] != 1
            or memory["srs_fft_cache_entry_capacity"] < 1
            or memory["parallel_fft_lookup_table_elements"] != expected_parallel_lut_elements
            or memory["size_of_usize_pair_bytes"] != 2 * memory["size_of_usize_bytes"]
            or memory["size_of_lookup_map_entry_bytes"] != memory["size_of_vec_u8_bytes"] + memory["size_of_usize_bytes"]
            or memory["size_of_lookup_repr_bytes"] != memory["size_of_fr_bytes"]
            or memory["size_of_graph_witness_scalar_bytes"] != memory["size_of_fr_bytes"]
            or memory["gat_transient_witness_evidence_kind"] != "source-derived-conservative-live-tensor-envelope-v1"
            or memory["gat_transient_witness_formula_id"] != "whole-gat-retained-inputs-plus-max-live-head-phase-v1"
            or memory["advice_event_log_bound_evidence_kind"] != "cumulative_region_linear_cursor_upper_not_num_rows_or_proof_domain-v1"
            or memory["advice_event_record_bytes_upper"] != 9 + 2 * memory["size_of_lookup_repr_bytes"]
            or memory["advice_event_reservation_chunk_bytes"] != 1 << 20
            or memory["advice_event_batch_size_upper"] != 8192
            or memory["advice_event_decoded_entry_bytes_upper"] < memory["size_of_assigned_fr_bytes"]
            or memory["advice_event_writer_buffer_bytes_upper"] != 8192
            or memory["advice_event_column_bytes_upper"]
            < memory["advice_event_column_metadata_bytes_upper"]
            + memory["advice_event_record_bytes_upper"]
            + memory["advice_event_writer_buffer_bytes_upper"]
            or memory["advice_event_latest_element_bytes"] != memory["size_of_usize_bytes"]
            or memory["advice_event_read_buffer_bytes_upper"]
            != memory["advice_event_batch_size_upper"]
            * (
                memory["advice_event_record_bytes_upper"]
                + memory["advice_event_decoded_entry_bytes_upper"]
                + memory["size_of_option_fr_bytes"]
                + memory["size_of_batch_invert_entry_bytes"]
            )
            + 3 * memory["size_of_vec_header_bytes"]
            or memory["size_of_vec_header_bytes"] != memory["size_of_vec_u8_bytes"]
            or memory["size_of_option_fr_bytes"] < memory["size_of_fr_bytes"]
            or memory["size_of_batch_invert_entry_bytes"] < memory["size_of_fr_bytes"]
            or memory["size_of_fftdata_once_lock_bytes"] < memory["size_of_fftdata_bytes"]
            or memory["size_of_arc_header_bytes"] != 2 * memory["size_of_usize_bytes"]
            or memory["cuda_free_bytes_at_admission"] > memory["cuda_total_memory_bytes"]
            or not isinstance(memory["cuda_uuid"], str)
            or re.fullmatch(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", memory["cuda_uuid"]) is None
            or memory["cuda_uuid"] != selected_device.get("uuid")
        ):
            abort(f"{dataset} native admission memory model is internally inconsistent or targets another GPU")
        if (
            isinstance(memory["recursive_fft_twiddle_field_elements"], bool)
            or not isinstance(memory["recursive_fft_twiddle_field_elements"], int)
            or memory["recursive_fft_twiddle_field_elements"] < 0
            or not isinstance(memory["recursive_fft_cached_domain_lengths"], list)
        ):
            abort(f"{dataset} native admission has malformed recursive FFT cache accounting")
        for name, expected_count, allow_zero in (
            ("custom_gate_dependency_columns", cs["num_custom_gate_groups"], True),
            ("lookup_dependency_columns", cs["num_lookup_arguments"], True),
            ("lookup_query_polynomial_counts", cs["num_lookup_arguments"], False),
            ("shuffle_dependency_columns", cs["num_shuffle_arguments"], True),
        ):
            values = memory[name]
            if (
                not isinstance(values, list)
                or len(values) != expected_count
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < (0 if allow_zero else 1)
                    or value > (cs["fixed_column_count_post_compression"] + cs["num_advice_columns"] + cs["num_instance_columns"])
                    for value in values
                )
            ):
                abort(f"{dataset} native admission has malformed {name} source geometry")
        if (
            sum(memory["lookup_query_polynomial_counts"]) != cs["num_lookup_query_polynomials"]
            or max(memory["lookup_query_polynomial_counts"], default=0) != cs["max_lookup_query_polynomials_per_argument"]
        ):
            abort(f"{dataset} native admission lookup dependency/query counts do not match chunked arguments")
        _validate_native_shplonk_shape(dataset, memory, cs, admission["domain_size"])
        expected_bitmap = cs["num_selectors"] * ((admission["domain_size"] + 7) // 8)
        selector_storage = memory["selector_activation_storage_bytes_upper"]
        selector_storage_max = 2 * expected_bitmap + cs["num_selectors"] * 64
        read_chunk_fields = memory["fixed_polynomial_read_chunk_fields_upper"]
        if (
            memory["selector_activation_bitmap_bytes_upper"] != expected_bitmap
            or isinstance(selector_storage, bool)
            or not isinstance(selector_storage, int)
            or selector_storage != selector_storage_max
            or selector_storage > sys.maxsize
            or memory["selector_activation_storage_kind"] != "rle_bitmap_hybrid_v1"
            or memory["fixed_polynomial_read_mode"] != "streamed_one_polynomial_chunked_v1"
            or read_chunk_fields != 16 * 1024
            or memory["fixed_polynomial_read_raw_chunk_bytes_upper"] != read_chunk_fields * memory["size_of_fr_bytes"]
            or memory["fixed_polynomial_read_decoded_chunk_bytes_upper"] != read_chunk_fields * memory["size_of_fr_bytes"]
            or memory["fixed_polynomial_read_heap_scratch_bytes_upper"]
            != memory["fixed_polynomial_read_raw_chunk_bytes_upper"] + memory["fixed_polynomial_read_decoded_chunk_bytes_upper"]
        ):
            abort(f"{dataset} native admission fixed-polynomial streaming bounds are inconsistent")
        phase_ids = memory["advice_column_phase_ids"]
        if (
            not isinstance(phase_ids, list)
            or len(phase_ids) != cs["num_advice_columns"]
            or any(isinstance(phase, bool) or not isinstance(phase, int) or phase < 0 for phase in phase_ids)
        ):
            abort(f"{dataset} native admission advice phase layout is malformed")
        admission_receipt = {
            "path": str(admission_path.relative_to(baseline_root)),
            "sha256": sha256(admission_path),
            "prepared_dir": str(admission_dir.relative_to(baseline_root)),
            "settings_sha256": admission["settings_sha256"],
            "scalar_domain_vector_bytes": admission["scalar_domain_vector_bytes"],
            "total_assignments": admission["total_assignments"],
            "logrows": admission["logrows"],
            "domain_size": admission["domain_size"],
            "constraint_system": cs,
            "native_memory_model": memory,
            "relation_identity": relation_identity,
        }
        admissions[dataset] = admission_receipt
        if receipt_callback is not None:
            receipt_callback(dataset, admission_receipt)
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
    value: object, *, source_sha256: str, backend_manifest_sha256: str, selected_device: dict[str, object], session_id: str
) -> dict[str, object]:
    keys = {
        "schema",
        "session_id",
        "icicle_source_tree_sha256",
        "native_backend_manifest_sha256",
        "device_uuid",
        "free_memory_bytes",
        "total_memory_bytes",
        "integrated_device",
        "scalar_bits",
        "c",
        "chunk_scalars",
        "window_count",
        "M",
        "B",
        "H_max",
        "L_max",
        "cuda_runtime_version",
        "cuda_driver_version",
        "cuda_toolkit_version",
        "cub_version",
        "device_compute_capability",
        "native_type_sizes",
        "device_workspace_peak_bytes",
        "queries",
        "scratch_bytes_sum",
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
    for key in (
        "free_memory_bytes",
        "total_memory_bytes",
        "cuda_runtime_version",
        "cuda_driver_version",
        "cub_version",
        "device_workspace_peak_bytes",
        "scratch_bytes_sum",
    ):
        item = value.get(key)
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            abort(f"native CUB workspace query has invalid {key}")
    if value["free_memory_bytes"] > value["total_memory_bytes"] or not isinstance(value.get("integrated_device"), bool):
        abort("native CUB workspace query has invalid live CUDA capacity values")
    type_sizes = value.get("native_type_sizes")
    if (
        not isinstance(type_sizes, dict)
        or set(type_sizes) != {"scalar_bytes", "affine_bytes", "projective_bytes", "unsigned_bytes"}
        or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in type_sizes.values())
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
    value: object, *, session_id: str, selected_device: dict[str, object], workspace_query: dict[str, object]
) -> dict[str, object]:
    fields = {"schema", "session_id", "device_uuid", "free_memory_bytes", "total_memory_bytes", "integrated_device"}
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


def _gpu_query_environment(selected_device: dict[str, object], gpu_identity: dict[str, object], build_environment: dict[str, object]) -> dict[str, str]:
    env = os.environ.copy()
    for name in ("LD_PRELOAD", "ICICLE_SMALL_K", "ICICLE_SMALL_K_FFT", "EZKL_FORMAL_RESOURCE_AUTH_SESSION_ID"):
        env.pop(name, None)
    env.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": str(selected_device["uuid"]),
            "ICICLE_DEVICE_ID": "0",
            "ENABLE_ICICLE_GPU": "1",
            "EZKL_GAT_BACKEND": "gpu",
            "ICICLE_BACKEND_INSTALL_DIR": str(ROOT / "artifacts/build/icicle-cuda"),
            "ICICLE_NATIVE_LIB_DIR": str(ROOT / "artifacts/build/icicle-cuda/lib"),
            "ICICLE_NATIVE_MANIFEST_SHA256": str(gpu_identity["native_backend_manifest_sha256"]),
            "CUDA_ARCH": str(build_environment["cuda_architecture"]).removeprefix("sm_"),
        }
    )
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
        nvcc,
        "-O2",
        "-std=c++17",
        f"-arch={arch}",
        "-I",
        local_cli(ROOT / "vendor/icicle/icicle/include", "ICICLE source headers"),
        local_cli(source, "CUB query source"),
        "-o",
        local_cli(binary_path, "CUB query binary"),
    ]
    print(f"[ezkl-formal] build session-bound CUB workspace query: {' '.join(build_command)}", flush=True)
    try:
        built = run_guarded_capture(build_command, cwd=ROOT, env=env, timeout=120)
    except subprocess.TimeoutExpired as exc:
        build_log_path.write_text((exc.stdout or "") + (exc.stderr or ""), encoding="utf-8")
        abort("native CUB workspace query build exceeded its 120 second bound")
    build_log_path.write_text(built.stdout, encoding="utf-8")
    if built.returncode != 0 or not binary_path.is_file() or binary_path.is_symlink():
        abort(f"native CUB workspace query build failed with exit {built.returncode}; see {build_log_path}")

    command = [
        local_cli(binary_path, "CUB query binary"),
        "--session-id",
        session_id,
        "--icicle-source-tree-sha256",
        icicle_source_sha256,
        "--native-backend-manifest-sha256",
        str(gpu_identity["native_backend_manifest_sha256"]),
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
    receipt: object, session_dir: Path, selected_device: dict[str, object], gpu_identity: dict[str, object], session_id: str
) -> dict[str, object]:
    fields = {
        "schema",
        "session_id",
        "source_path",
        "source_sha256",
        "binary_path",
        "binary_sha256",
        "build_log_path",
        "build_log_sha256",
        "result_path",
        "result_sha256",
        "native_backend_manifest_path",
        "native_backend_manifest_sha256",
        "icicle_source_tree_sha256",
        "query",
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
    if source != ROOT / "tools/native_msm_workspace_query.cu" or receipt["native_backend_manifest_sha256"] != gpu_identity.get(
        "native_backend_manifest_sha256"
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
    receipt = _validate_native_cub_workspace_receipt(receipt, session_dir, selected_device, gpu_identity, session_id)
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
        raw, session_id=session_id, selected_device=selected_device, workspace_query=workspace_query if isinstance(workspace_query, dict) else {}
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
        accounting["cgroup_limits"].append({"path": str(base), "limit_bytes": int(maximum), "current_bytes": current, "available_bytes": available})
    accounting["effective_available_bytes"] = effective
    return effective, accounting


def _effective_host_memory_limit_bytes() -> tuple[int, dict[str, object]]:
    try:
        total = None
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, separator, value = line.partition(":")
            if separator and key == "MemTotal":
                pieces = value.split()
                if pieces and pieces[0].isdigit():
                    total = int(pieces[0]) * 1024
                break
    except OSError as exc:
        abort(f"cannot sample host MemTotal for EZKL reserve: {exc}")
    if total is None or total <= 0:
        abort("host MemTotal is absent or malformed; EZKL reserve cannot be calculated")
    limit = total
    cgroup_limits: list[dict[str, object]] = []
    for base in _cgroup_memory_paths():
        try:
            value = (base / "memory.max").read_text().strip()
        except OSError:
            continue
        if value == "max":
            continue
        if not value.isdigit() or int(value) <= 0:
            abort(f"malformed cgroup memory.max for EZKL reserve at {base}")
        value_bytes = int(value)
        limit = min(limit, value_bytes)
        cgroup_limits.append({"path": str(base), "limit_bytes": value_bytes})
    return limit, {"host_mem_total_bytes": total, "effective_memory_limit_bytes": limit, "cgroup_limits": cgroup_limits}


def _host_os_diagnostic_reserve_bytes() -> tuple[int, dict[str, object]]:
    effective_limit, accounting = _effective_host_memory_limit_bytes()
    reserve = max(2 * (1 << 30), (effective_limit * 5 + 99) // 100)
    return reserve, {**accounting, "os_diagnostic_reserve_bytes": reserve}


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


def _std_hashmap_bucket_count_for_capacity(entry_capacity: int) -> int:
    """Return std HashMap's source-derived bucket count for the formal targets."""
    if isinstance(entry_capacity, bool) or not isinstance(entry_capacity, int) or entry_capacity < 0:
        abort("EZKL permutation HashMap capacity is malformed")
    if entry_capacity == 0:
        return 0
    if entry_capacity < 4:
        return 4
    if entry_capacity < 8:
        return 8
    # Rust's std HashMap raw table leaves one eighth of buckets empty, rounds
    # bucket count up to a power of two, and rejects capacities whose
    # intermediate `capacity * 8` would overflow usize.
    if entry_capacity > sys.maxsize // 8:
        abort("EZKL permutation HashMap capacity exceeds the native usize growth bound")
    adjusted_capacity = entry_capacity * 8 // 7
    if adjusted_capacity <= 0:
        abort("EZKL permutation HashMap capacity cannot be represented")
    bucket_count = 1 << (adjusted_capacity - 1).bit_length()
    if bucket_count > sys.maxsize:
        abort("EZKL permutation HashMap bucket count exceeds the native usize range")
    return bucket_count


def _std_hashmap_table_bytes(bucket_count: int, entry_bytes: int, usize_bytes: int) -> int:
    """Upper-bound a HashMap raw-table allocation, including ctrl bytes/group trailer."""
    if bucket_count == 0:
        return 0
    if (
        isinstance(bucket_count, bool)
        or not isinstance(bucket_count, int)
        or bucket_count < 4
        or bucket_count & (bucket_count - 1)
        or isinstance(entry_bytes, bool)
        or not isinstance(entry_bytes, int)
        or entry_bytes <= 0
        or isinstance(usize_bytes, bool)
        or not isinstance(usize_bytes, int)
        or usize_bytes <= 0
    ):
        abort("EZKL permutation HashMap raw-table layout inputs are malformed")
    if bucket_count > sys.maxsize // entry_bytes:
        abort("EZKL permutation HashMap entry backing exceeds the native address range")
    entry_payload = bucket_count * entry_bytes
    # RawTable aligns its control-byte array after the entry array to at least
    # the 16-byte SIMD group width and appends one full group as a sentinel.
    # The formal x86_64/aarch64 builds use 16-byte groups.
    control_alignment = max(16, usize_bytes)
    if entry_payload > sys.maxsize - (control_alignment - 1):
        abort("EZKL permutation HashMap control offset exceeds the native address range")
    control_offset = (entry_payload + control_alignment - 1) & ~(control_alignment - 1)
    trailer_bytes = bucket_count + 16
    if control_offset > sys.maxsize - trailer_bytes:
        abort("EZKL permutation HashMap raw-table allocation exceeds the native address range")
    return control_offset + trailer_bytes


def _permutation_hashmap_heap_upper(
    *, mapping_entries: int, aux_entries: int, sizes_entries: int, entry_bytes: int, usize_bytes: int, map_header_bytes: int
) -> tuple[int, int, int, int]:
    """Bound the three std HashMaps and one old table during a resize."""
    entry_bounds = (mapping_entries, aux_entries, sizes_entries)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in entry_bounds):
        abort("EZKL permutation Assembly HashMap entry bound is malformed")
    if isinstance(map_header_bytes, bool) or not isinstance(map_header_bytes, int) or map_header_bytes <= 0:
        abort("EZKL permutation Assembly HashMap header size is invalid")
    buckets = [_std_hashmap_bucket_count_for_capacity(value) for value in entry_bounds]
    backing_bytes = sum(_std_hashmap_table_bytes(value, entry_bytes, usize_bytes) for value in buckets)
    # In each individual insertion only one map can resize. Count the largest
    # possible old table alongside all three final table upper bounds.
    prior_buckets = [value // 2 if value > 4 else 0 for value in buckets]
    rehash_old_bytes = max(
        (_std_hashmap_table_bytes(value, entry_bytes, usize_bytes) for value in prior_buckets),
        default=0,
    )
    headers_bytes = 3 * map_header_bytes
    rehash_transient_bytes = backing_bytes + rehash_old_bytes
    if headers_bytes > sys.maxsize - rehash_transient_bytes:
        abort("EZKL permutation Assembly HashMap heap bound exceeds the native address range")
    return headers_bytes + rehash_transient_bytes, backing_bytes, rehash_old_bytes, sum(buckets)


def _native_capacity_requirements(
    admission: dict[str, object],
    icicle_type_sizes: dict[str, object],
    *,
    native_census: dict[str, object] | None = None,
) -> dict[str, object]:
    cs = admission.get("constraint_system")
    memory = admission.get("native_memory_model")
    domain_size = admission.get("domain_size")
    if not isinstance(cs, dict) or not isinstance(memory, dict) or isinstance(domain_size, bool) or not isinstance(domain_size, int):
        abort("EZKL native admission lacks geometry required by the local capacity gate")
    fr = int(memory["size_of_fr_bytes"])
    assigned = int(memory["size_of_assigned_fr_bytes"])
    option_fr = int(memory["size_of_option_fr_bytes"])
    batch_invert_entry = int(memory["size_of_batch_invert_entry_bytes"])
    atomic_u64 = int(memory["size_of_atomic_u64_bytes"])
    map_entry = int(memory["size_of_lookup_map_entry_bytes"])
    map_header = int(memory["size_of_lookup_map_header_bytes"])
    repr_bytes = int(memory["size_of_lookup_repr_bytes"])
    vec_header = int(memory["size_of_vec_header_bytes"])
    polynomial_header = int(memory["size_of_polynomial_header_bytes"])
    prepared_header = int(memory["size_of_mv_lookup_prepared_bytes"])
    prepared_table_header = int(memory["size_of_mv_lookup_prepared_table_bytes"])
    committed_header = int(memory["size_of_mv_lookup_committed_bytes"])
    graph_witness_scalars = int(memory["graph_witness_scalar_element_count"])
    g1_affine = int(memory["size_of_g1_affine_bytes"])
    g1_projective = int(memory["size_of_g1_projective_bytes"])
    g2_affine = int(memory["size_of_g2_affine_bytes"])
    fft_stage = int(memory["size_of_fft_stage_bytes"])
    lookup_arguments = int(cs["num_lookup_arguments"])
    lookup_queries = int(cs["num_lookup_query_polynomials"])
    max_lookup_queries = int(cs["max_lookup_query_polynomials_per_argument"])
    table_group_count = int(cs["num_unique_lookup_table_expression_groups"])
    table_bounds = memory["lookup_table_cardinality_upper_bounds"]
    custom_gate_dependencies = memory["custom_gate_dependency_columns"]
    lookup_dependencies = memory["lookup_dependency_columns"]
    lookup_query_counts = memory["lookup_query_polynomial_counts"]
    shuffle_dependencies = memory["shuffle_dependency_columns"]
    n = domain_size
    total_assignments = admission.get("total_assignments")
    if (
        isinstance(total_assignments, bool)
        or not isinstance(total_assignments, int)
        or total_assignments <= 0
        or total_assignments > sys.maxsize
    ):
        abort("EZKL native admission has invalid cumulative assignment count for prover spill bounds")
    proof_domain = int(memory["proof_domain_size"])
    proof_logrows = int(memory["proof_logrows"])
    uses_recursive_fft = bool(memory["uses_recursive_fft"])
    fixed_post = int(cs["fixed_column_count_post_compression"])
    fixed_uncompressed = int(cs["num_fixed_columns"])
    permutation = int(cs["num_permutation_columns"])
    copy_calls_upper = cs.get("permutation_copy_call_count_upper")
    successor_entries_upper = cs.get("permutation_successor_entry_count_upper")
    if (
        isinstance(copy_calls_upper, bool)
        or not isinstance(copy_calls_upper, int)
        or copy_calls_upper < 0
        or copy_calls_upper > sys.maxsize
        or isinstance(successor_entries_upper, bool)
        or not isinstance(successor_entries_upper, int)
        or successor_entries_upper < 0
        or successor_entries_upper > sys.maxsize
    ):
        abort("EZKL native admission has malformed permutation-copy bounds for the local capacity gate")
    permutation_cells = permutation * n
    expected_successor_entries_upper = min(permutation_cells, 2 * copy_calls_upper)
    if permutation_cells > sys.maxsize or successor_entries_upper != expected_successor_entries_upper:
        abort("EZKL native admission permutation-copy bound is inconsistent with the local capacity gate")
    if expected_successor_entries_upper > sys.maxsize // 16:
        abort("EZKL native sparse successor-file ceiling exceeds the native address range")
    selector_activation_storage_ceiling = int(memory["selector_activation_storage_bytes_upper"])
    if native_census is None:
        copy_calls = 0
        successor_entries = 0
        selectors = 0
        permutation_cache_host = 0
        permutation_bitmap_host = 0
        permutation_data_backing = 0
        permutation_directory_backing = 0
        selector_activation_counts: list[int] | None = None
        fixed_assignment_counts: list[int] | None = None
    else:
        copy_calls = native_census.get("permutation_copy_calls")
        successor_entries = native_census.get("permutation_successor_edges")
        selectors = native_census.get("selector_activation_storage_peak_bytes")
        permutation_cache_host = native_census.get("permutation_cache_bytes")
        permutation_bitmap_host = native_census.get("permutation_allocation_bitmap_bytes")
        permutation_data_backing = native_census.get("permutation_data_allocated_bytes")
        permutation_directory_backing = native_census.get("permutation_directory_allocated_bytes")
        selector_activation_counts = native_census.get("selector_activation_counts")
        fixed_assignment_counts = native_census.get("fixed_assignment_callback_counts")
        for label, value in (
            ("permutation_copy_calls", copy_calls),
            ("permutation_successor_edges", successor_entries),
            ("selector_activation_storage_peak_bytes", selectors),
            ("permutation_cache_bytes", permutation_cache_host),
            ("permutation_allocation_bitmap_bytes", permutation_bitmap_host),
            ("permutation_data_allocated_bytes", permutation_data_backing),
            ("permutation_directory_allocated_bytes", permutation_directory_backing),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > sys.maxsize:
                abort(f"native keygen census has malformed {label}")
        if (
            not isinstance(selector_activation_counts, list)
            or len(selector_activation_counts) != int(cs["num_selectors"])
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > n for value in selector_activation_counts)
            or not isinstance(fixed_assignment_counts, list)
            or len(fixed_assignment_counts) != int(cs["num_fixed_columns"])
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > n for value in fixed_assignment_counts)
            or selectors > selector_activation_storage_ceiling
            or copy_calls > copy_calls_upper
            or successor_entries > expected_successor_entries_upper
            or successor_entries > min(permutation_cells, 2 * copy_calls)
        ):
            abort("native keygen census exceeds its admitted selector/permutation source bounds")
    successor_entries_for_files = expected_successor_entries_upper if native_census is None else successor_entries
    if successor_entries_for_files > sys.maxsize // 16:
        abort("native proving-key successor backing exceeds the native address range")
    native_successor_file_bytes = 16 * successor_entries_for_files
    native_successor_backing_reserved_bytes = ((native_successor_file_bytes + 4095) // 4096) * 4096
    if native_successor_backing_reserved_bytes > sys.maxsize:
        abort("native proving-key page-rounded successor backing exceeds the native address range")
    target_bitmap_virtual_bytes = (successor_entries_for_files + 7) // 8
    target_bitmap_page_bytes = ((target_bitmap_virtual_bytes + 4095) // 4096) * 4096
    target_bitmap_page_count = target_bitmap_page_bytes // 4096
    target_bitmap_allocation_bitmap_bytes = (target_bitmap_page_count + 7) // 8
    native_pk_read_backing_required = native_successor_backing_reserved_bytes + target_bitmap_page_bytes
    if native_pk_read_backing_required > sys.maxsize:
        abort("native proving-key read backing exceeds the native address range")
    permutation_hashmap_header = int(memory["size_of_permutation_hashmap_header_bytes"])
    advice = int(cs["num_advice_columns"])
    gat_transient_witness_bytes = int(memory["gat_transient_witness_bytes_upper"])
    advice_event_column_bytes = int(memory["advice_event_column_bytes_upper"])
    advice_event_batch_bytes = int(memory["advice_event_read_buffer_bytes_upper"])
    advice_event_record_bytes = int(memory["advice_event_record_bytes_upper"])
    advice_event_chunk_bytes = int(memory["advice_event_reservation_chunk_bytes"])
    latest_element_bytes = int(memory["advice_event_latest_element_bytes"])
    instances = int(cs["num_instance_columns"])
    degree = int(cs["degree"])
    fft_supported_lengths = int(memory["fft_cache_supported_length_count"])
    fft_cache_capacity = int(memory["fft_cache_entry_capacity"])
    srs_fft_supported_lengths = int(memory["srs_fft_cache_supported_length_count"])
    srs_fft_cache_capacity = int(memory["srs_fft_cache_entry_capacity"])
    fftdata_once_lock = int(memory["size_of_fftdata_once_lock_bytes"])
    fftdata_entry = int(memory["size_of_fftdata_cache_entry_bytes"])
    fftdata_map_header = int(memory["size_of_fftdata_cache_map_bytes"])
    arc_header = int(memory["size_of_arc_header_bytes"])
    parallel_threads = int(memory["fft_parallel_thread_count"])
    shplonk_query_polynomials = int(memory["shplonk_query_polynomial_count"])
    shplonk_rotation_sets = int(memory["shplonk_rotation_set_count"])
    shplonk_max_rotations = int(memory["shplonk_max_rotation_count"])
    shplonk_commitment_size = int(memory["size_of_shplonk_commitment_extension_bytes"])
    shplonk_rotation_set_size = int(memory["size_of_shplonk_rotation_set_extension_bytes"])
    coeff_polynomial_header = int(memory["size_of_coeff_polynomial_header_bytes"])
    parallel_lut_elements = int(memory["parallel_fft_lookup_table_elements"])
    icicle_scalar_bytes = int(icicle_type_sizes["scalar_bytes"])
    icicle_affine_bytes = int(icicle_type_sizes["affine_bytes"])
    if degree < 2:
        abort("EZKL native admission has invalid quotient degree for the local capacity gate")
    selector_activation_bitmap_ceiling = int(memory["selector_activation_bitmap_bytes_upper"])
    selector_census_storage = int(memory["selector_activation_storage_bytes_upper"])
    read_scratch = int(memory["fixed_polynomial_read_heap_scratch_bytes_upper"])
    for label, number in (
        ("Fr", fr),
        ("Assigned<Fr>", assigned),
        ("Option<Fr>", option_fr),
        ("ICICLE scalar", icicle_scalar_bytes),
        ("ICICLE affine", icicle_affine_bytes),
        ("AtomicU64", atomic_u64),
        ("lookup entry", map_entry),
        ("lookup map header", map_header),
        ("permutation HashMap header", permutation_hashmap_header),
        ("lookup repr", repr_bytes),
        ("Vec header", vec_header),
        ("Polynomial header", polynomial_header),
        ("Prepared header", prepared_header),
        ("PreparedTable header", prepared_table_header),
        ("Committed header", committed_header),
        ("G1 projective", g1_projective),
        ("G2 affine", g2_affine),
        ("FFT stage", fft_stage),
        ("SHPLONK query polynomials", shplonk_query_polynomials),
        ("SHPLONK rotation sets", shplonk_rotation_sets),
        ("SHPLONK rotations", shplonk_max_rotations),
        ("SHPLONK commitment extension", shplonk_commitment_size),
        ("SHPLONK rotation-set extension", shplonk_rotation_set_size),
        ("coefficient polynomial header", coeff_polynomial_header),
    ):
        if number <= 0:
            abort(f"EZKL native admission has invalid sizeof({label})")
    if icicle_scalar_bytes != fr:
        abort("compiled ICICLE BN254 scalar size disagrees with native Halo2 Fr")
    if (
        not isinstance(table_bounds, list)
        or len(table_bounds) != table_group_count
        or any(isinstance(bound, bool) or not isinstance(bound, int) or bound <= 0 or bound > n for bound in table_bounds)
        or lookup_queries < lookup_arguments
        or max_lookup_queries > lookup_queries
        or table_group_count > lookup_arguments
    ):
        abort("EZKL native admission has inconsistent source-bounded mv-lookup cardinalities")
    for label, values, expected_count, allow_zero in (
        ("custom gate dependency", custom_gate_dependencies, int(cs["num_custom_gate_groups"]), True),
        ("lookup dependency", lookup_dependencies, lookup_arguments, True),
        ("lookup query count", lookup_query_counts, lookup_arguments, False),
        ("shuffle dependency", shuffle_dependencies, int(cs["num_shuffle_arguments"]), True),
    ):
        if (
            not isinstance(values, list)
            or len(values) != expected_count
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < (0 if allow_zero else 1) or value > fixed_uncompressed + advice + instances
                for value in values
            )
        ):
            abort(f"EZKL native admission has malformed {label} counts")
    if (
        sum(lookup_query_counts) != lookup_queries
        or max(lookup_query_counts, default=0) != max_lookup_queries
        or not isinstance(memory["uses_recursive_fft"], bool)
        or parallel_threads <= 0
        or fft_supported_lengths <= 0
        or fft_cache_capacity < fft_supported_lengths
        or srs_fft_supported_lengths != 1
        or srs_fft_cache_capacity < 1
    ):
        abort("EZKL native admission has inconsistent evaluator/FFT resource metadata")
    shplonk_batch_width_upper = min(shplonk_query_polynomials, parallel_threads)
    shplonk_task_bytes = 2 * n * fr + 2 * coeff_polynomial_header
    shplonk_fixed_workspace_bytes = 4 * n * fr + 4 * coeff_polynomial_header
    shplonk_query_point_upper = shplonk_query_polynomials * shplonk_max_rotations
    shplonk_extension_host_upper = (
        shplonk_query_polynomials * shplonk_commitment_size
        + shplonk_rotation_sets * shplonk_rotation_set_size
        + 2 * shplonk_query_point_upper * fr
        + shplonk_rotation_sets * shplonk_max_rotations * fr
        + shplonk_query_point_upper * (2 * fr + int(memory["size_of_usize_bytes"]))
        + parallel_threads * shplonk_max_rotations * 2 * fr
    )
    shplonk_transient_spill_upper = (shplonk_batch_width_upper + 2) * n * fr

    def fft_cache_map_allocation(capacity: int) -> int:
        required_buckets = (capacity * 8 + 6) // 7
        bucket_count = max(4, 1 << (required_buckets - 1).bit_length())
        return fftdata_map_header + bucket_count * (fftdata_entry + 1) + 16

    def fft_cache_payload(length_count: int, map_capacity: int, transform_logrows: int, twiddles: int) -> int:
        return (
            fft_cache_map_allocation(map_capacity)
            + length_count * (fftdata_once_lock + arc_header)
            + transform_logrows * fft_stage
            + 2 * transform_logrows * vec_header
            + twiddles * fr
        )

    proof_fft_twiddle_elements = int(memory["recursive_fft_twiddle_field_elements"])
    proof_fft_cache_bytes = fft_cache_payload(fft_supported_lengths, fft_cache_capacity, proof_logrows, proof_fft_twiddle_elements if uses_recursive_fft else 0)
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
    # Serialized selector wire data is conservatively reserved at the dense
    # bitmap ceiling even though the native in-memory activation set is a
    # run/bitmap hybrid. These are file bytes, not resident-memory estimates.
    selector_vk_bytes = selector_activation_bitmap_ceiling
    commitment_vk_bytes = (fixed_post + permutation) * int(memory["size_of_g1_affine_bytes"])
    permutation_mapping_bytes = 8 + 8 + 10 * permutation + native_successor_backing_reserved_bytes + 32
    setup_key_artifact_upper = setup_persisted_maps + selector_vk_bytes + commitment_vk_bytes + permutation_mapping_bytes + 8 + 12 + 2 * (4 + 4 * fixed_post)
    quotient_coefficients = n * (degree - 1)
    advice_event_log_records_upper = total_assignments * advice_event_record_bytes
    advice_event_log_rounding_upper = advice * (advice_event_chunk_bytes - 1)
    advice_event_log_reserved_upper = advice_event_log_records_upper + advice_event_log_rounding_upper
    advice_latest_map_bytes = n * latest_element_bytes
    if any(
        value < 0 or value > sys.maxsize
        for value in (
            advice_event_log_records_upper,
            advice_event_log_rounding_upper,
            advice_event_log_reserved_upper,
            advice_latest_map_bytes,
        )
    ):
        abort("EZKL native advice-event spill or conversion-map source bound exceeds the native address range")
    lookup_query_spill_bytes = lookup_queries * n * fr
    lookup_m_spill_bytes = lookup_arguments * n * fr
    lookup_table_spill_bytes = table_group_count * n * fr
    lookup_phi_spill_bytes = lookup_arguments * n * fr
    lookup_spill_known = lookup_query_spill_bytes + lookup_m_spill_bytes + lookup_table_spill_bytes + lookup_phi_spill_bytes
    max_gate_dependencies = max(custom_gate_dependencies, default=0)
    max_lookup_dependencies = max(lookup_dependencies, default=0)
    max_shuffle_dependencies = max(shuffle_dependencies, default=0)
    max_lookup_input_count = max(lookup_query_counts, default=0)
    permutation_eval_coset_window = 5 if permutation else 0
    evaluation_coset_window = max(
        max_gate_dependencies,
        max_lookup_dependencies + 2 if lookup_arguments else 0,
        max_shuffle_dependencies + 1 if int(cs["num_shuffle_arguments"]) else 0,
        permutation_eval_coset_window,
    )
    evaluation_source_column_window = max(
        max_gate_dependencies,
        max_lookup_dependencies + 2 if lookup_arguments else 0,
        max_shuffle_dependencies + 1 if int(cs["num_shuffle_arguments"]) else 0,
        3 if permutation else 0,
    )
    evaluation_temporary_mapped_bytes = evaluation_coset_window * proof_domain * fr
    evaluation_source_hot_bytes = evaluation_source_column_window * n * fr
    # `evaluate_h` always keeps the three PK boundary-selector cosets and the
    # quotient accumulator mapped; only the current expression group's exact
    # source dependencies and their cosets are additionally touched.
    proof_eval_hot_mmap = 4 * proof_domain * fr + evaluation_temporary_mapped_bytes + evaluation_source_hot_bytes
    proof_spill_known = (
        setup_persisted_maps
        + advice * n * fr
        + 2 * instances * n * fr
        + proof_domain * fr
        + quotient_coefficients * fr
        + lookup_spill_known
        + advice_event_log_reserved_upper
        + advice_latest_map_bytes
        + evaluation_temporary_mapped_bytes
        + shplonk_transient_spill_upper
        + native_pk_read_backing_required
    )
    # The bounded native proving-key reader uses one lazily allocated
    # successor-store cache, one target-bitmap cache, its source-derived
    # allocation bitmap, and two fixed 4096-pair validation buffers. Native
    # cache sizes include their fixed page/index metadata and struct headers.
    native_pk_read_validation_buffers = 4 * 4096 * 16
    native_pk_read_stream_buffer = int(memory["native_pk_read_buffer_capacity_bytes"])
    native_pk_read_host_scratch = (
        int(memory["native_successor_read_cache_bytes"])
        + int(memory["native_target_bitmap_fixed_cache_bytes"])
        + target_bitmap_allocation_bitmap_bytes
        + native_pk_read_validation_buffers
        + native_pk_read_stream_buffer
    )
    # Native save_pk also emits a separate VK file; include its known selector
    # bitmaps and commitments in the retained-artifact reservation.
    setup_key_artifact_upper += selector_vk_bytes + commitment_vk_bytes
    # Formal native keygen now stores selector activations and permutation
    # state in bounded hybrid/sparse stores. The old HashMap and full successor
    # Vec estimates are not allocations in this path. Before the mandatory
    # native census these dynamic resident terms are intentionally unknown and
    # are enforced by the in-process allocation budget; after keygen they are
    # replaced with the exact source-reported values below.
    permutation_hashmap_heap_peak = 0
    permutation_hashmap_backing_bytes = 0
    permutation_hashmap_resize_old_bytes = 0
    permutation_hashmap_bucket_count = 0
    permutation_aux_entries = successor_entries
    permutation_sizes_entries = 0
    # NativePermutationStats.cache_bytes already includes its allocation
    # bitmap backing. Keep the separately reported bitmap field as a
    # decomposition only; adding it again would double-charge the shared
    # physical-memory pool and spuriously shrink the selector activation cap.
    permutation_mapping_host = permutation_cache_host
    selector_decode_bytes = n * fr
    selector_encoded_bytes = (n * max(1, degree.bit_length()) + 7) // 8
    if native_census is not None:
        native_preflight = native_census.get("preflight")
        if not isinstance(native_preflight, dict):
            abort("native keygen census lacks its exact native preflight geometry")
        selector_decode_bytes = native_preflight.get("one_selector_decode_bytes")
        selector_encoded_bytes = native_preflight.get("one_selector_encoded_bytes")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > sys.maxsize for value in (selector_decode_bytes, selector_encoded_bytes)):
            abort("native keygen census has malformed one-selector conversion buffers")
    selector_phase = (
        selectors
        + selector_decode_bytes
        + selector_encoded_bytes
        + read_scratch
        + permutation_mapping_host
    )
    # Native keygen supplies an exact-size iterator to the pinned batch
    # inversion implementation, so its prefix Vec reserves exactly N entries.
    # Count those native-sized (Fr, &mut Fr) entries alongside Option<Fr>.
    batch_invert_heap_upper = n * batch_invert_entry
    pk_phase = (
        selectors
        + n * option_fr
        + batch_invert_heap_upper
        + read_scratch
        + permutation_mapping_host
    )
    # `icicle.rs` materializes one scalar and one affine host Vec per <=2^20
    # MSM chunk. Their exact element sizes come from the compiled BN254 CUB
    # helper; this component is a distinct proof/setup MSM phase.
    icicle_msm_chunk_scalars = 1 << 20
    icicle_msm_staging = icicle_msm_chunk_scalars * (icicle_scalar_bytes + icicle_affine_bytes)
    # During permutation VK commitments, keygen retains the selector bitmap
    # Vecs, omega powers, and the current dense permutation column as well as
    # the ICICLE conversion Vecs.
    permutation_vk_phase = selectors + 2 * n * fr + icicle_msm_staging + permutation_mapping_host
    # Exact static reserve for later native keygen arrays, excluding activation
    # sets, one-selector conversion buffers, and bounded sparse-store caches,
    # which the in-process preflight accounts separately before allocation.
    setup_keygen_future_host_reserve = max(
        n * option_fr + batch_invert_heap_upper + read_scratch,
        2 * n * fr + icicle_msm_staging,
    )
    # Native SRS loading keeps the target G1 monomial vector while deriving
    # Lagrange points. On aarch64, recursive::fft copies its projective input
    # and allocates a second projective scratch vector, so the transform peak
    # has three projective N-vectors; the x86 parallel path has two. The
    # twiddle cache is the exact one-length SRS EvaluationDomain cache.
    srs_loaded = 2 * n * g1_affine + 2 * g2_affine
    srs_fft_twiddle_elements = 2 * (n - 1 + admission["logrows"])
    srs_fft_data = srs_fft_twiddle_elements * fr
    srs_fft_metadata = admission["logrows"] * fft_stage + 2 * admission["logrows"] * vec_header
    srs_fft_cache_metadata = fft_cache_payload(
        srs_fft_supported_lengths, srs_fft_cache_capacity, admission["logrows"], srs_fft_twiddle_elements if uses_recursive_fft else 0
    )
    srs_fft_twiddle_lut = parallel_lut_elements * fr
    if uses_recursive_fft:
        srs_fft_transform_peak = n * g1_affine + 3 * n * g1_projective + srs_fft_cache_metadata + 2 * g2_affine
        srs_fft_output_peak = 2 * n * g1_affine + n * g1_projective + srs_fft_cache_metadata + 2 * g2_affine
        srs_fft_cache_build_peak = n * g1_affine + n * g1_projective + srs_fft_cache_metadata + n * fr + 2 * g2_affine
    else:
        srs_fft_transform_peak = n * g1_affine + 2 * n * g1_projective + srs_fft_cache_metadata + srs_fft_twiddle_lut + 2 * g2_affine
        srs_fft_output_peak = 2 * n * g1_affine + n * g1_projective + srs_fft_cache_metadata + srs_fft_twiddle_lut + 2 * g2_affine
        srs_fft_cache_build_peak = n * g1_affine + n * g1_projective + srs_fft_cache_metadata + n * fr + srs_fft_twiddle_lut + 2 * g2_affine
    srs_downsize_peak = max(srs_fft_transform_peak, srs_fft_output_peak, srs_fft_cache_build_peak)
    # GAT Setup uses a witnessless circuit. Keep its declared tensor geometry
    # in the shape buffer term below, but do not charge the former dense Fr
    # witness copy that `GraphCircuit::new()` used to allocate.
    setup_witnessless_shape_values_upper = graph_witness_scalars * option_fr
    srs_setup_peak = srs_downsize_peak
    srs_loaded_and_witness = srs_loaded
    # While the proof owns the key and SRS, advice and public instance maps
    # remain mapped. The sparse permutation successor vector is heap-backed.
    selector_bitmap_host = selectors
    proof_nonprepare_heap = graph_witness_scalars * fr + selector_bitmap_host + permutation_mapping_host + srs_loaded
    # The three PK selector cosets are persistent in the loaded key. Other
    # proof polynomials are mmap-backed but are touched one exact evaluator
    # group at a time and are bounded by proof_eval_hot_mmap below.
    proof_nonprepare_hot_mmap = 3 * proof_domain * fr
    proof_nonprepare_resident = proof_nonprepare_heap + proof_nonprepare_hot_mmap
    # The native proof's assignment callback keeps one event column for each
    # advice column through circuit synthesis. The source-derived GAT live
    # tensor envelope is concurrent with this synthesis, while event decode
    # later has one latest-index map, one current field map (already covered
    # by A*N*Fr proof spill), and one bounded record/decoded/inversion batch.
    gat_proof_synthesis_host_peak = (
        proof_nonprepare_resident
        + gat_transient_witness_bytes
        + advice * advice_event_column_bytes
    )
    advice_conversion_mapped_host_upper = n * (latest_element_bytes + fr)
    advice_conversion_host_peak = (
        proof_nonprepare_resident
        + advice_event_batch_bytes
        + advice_conversion_mapped_host_upper
    )
    native_pk_read_host_peak = proof_nonprepare_resident + native_pk_read_host_scratch
    lookup_map_allocations: list[int] = []
    previous_bucket_growth: list[int] = []
    for upper in table_bounds:
        required_buckets = (upper * 8 + 6) // 7
        bucket_count = max(4, 1 << (required_buckets - 1).bit_length())
        map_allocation = map_header + bucket_count * (map_entry + 1) + 16 + upper * (repr_bytes + 32)
        lookup_map_allocations.append(map_allocation)
        previous_bucket_count = bucket_count // 2 if bucket_count > 4 else 0
        previous_bucket_growth.append(previous_bucket_count * (map_entry + 1) + (16 if previous_bucket_count else 0))
    lookup_maps_retained = sum(lookup_map_allocations)
    lookup_map_growth_peak = max(previous_bucket_growth, default=0)
    lookup_table_polynomials = lookup_table_spill_bytes
    lookup_metadata_retained = (
        table_group_count * (prepared_table_header + vec_header)
        + lookup_arguments * committed_header
        + lookup_arguments * prepared_header
        + lookup_queries * polynomial_header
        + 2 * vec_header
    )
    lookup_table_cache_retained_host = lookup_maps_retained + lookup_table_polynomials + lookup_metadata_retained
    # `Expression::evaluate` and the dense polynomial fold can have three
    # N-element field vectors live while one unique table group is built.
    lookup_table_build_peak = (3 * n * fr + lookup_map_growth_peak + (repr_bytes + 32)) if table_group_count else 0
    lookup_table_cache_peak_host = lookup_table_cache_retained_host + lookup_table_build_peak
    lookup_prepare_task_bytes = (
        (n * (max_lookup_queries * fr + atomic_u64 + 2 * fr) + max_lookup_queries * polynomial_header + prepared_header + 2 * vec_header)
        if lookup_arguments
        else 0
    )
    lookup_prepare_host_peak = proof_nonprepare_resident + lookup_table_cache_peak_host + lookup_prepare_task_bytes + icicle_msm_staging
    # Source lifetimes in commit_grand_sum: the largest heap phase is either
    # the accumulator + current input + BatchInvert's (Fr, &mut Fr) prefix,
    # or three dense vectors while constructing log_derivatives_diff. The
    # two accumulators are dropped before phi construction; phi is spilled.
    # Add only the touched input/m/phi mapping window per concurrent task.
    lookup_commit_heap_per_task = n * max(2 * fr + batch_invert_entry, 3 * fr)
    lookup_commit_active_map_per_task = 3 * n * fr
    lookup_commit_task_bytes = (lookup_commit_heap_per_task + lookup_commit_active_map_per_task + 5 * vec_header) if lookup_arguments else 0
    # Includes the one shared table-expression group mapping in addition to
    # the per-task input, m, and phi windows.
    lookup_commit_active_map_window = (4 * n * fr) if lookup_arguments else 0
    lookup_commit_host_peak = proof_nonprepare_resident + lookup_table_polynomials + lookup_metadata_retained + lookup_commit_task_bytes + icicle_msm_staging
    parallel_split_m = 1 << (parallel_threads.bit_length() - 1)
    parallel_task_elements = parallel_threads * parallel_split_m
    parallel_fft_transient = proof_domain * fr + parallel_lut_elements * fr + parallel_task_elements * fr
    recursive_fft_transient = 2 * proof_domain * fr if uses_recursive_fft else 0
    fft_cache_build_scratch = proof_domain * fr if uses_recursive_fft else 0
    proof_fft_transient = max(parallel_fft_transient, recursive_fft_transient, fft_cache_build_scratch)
    lookup_eval_workers = min(parallel_threads, max(1, (proof_domain + 4095) // 4096)) if lookup_arguments else 0
    lookup_eval_task_host_peak = (lookup_eval_workers * 4096 * max_lookup_input_count * (fr + batch_invert_entry)) if lookup_arguments else 0
    proof_eval_h_host_peak = proof_nonprepare_heap + proof_eval_hot_mmap + proof_fft_cache_bytes + max(proof_fft_transient, lookup_eval_task_host_peak)
    shplonk_host_peak = (
        proof_nonprepare_heap
        + proof_eval_hot_mmap
        + proof_fft_cache_bytes
        + lookup_table_cache_retained_host
        + icicle_msm_staging
        + shplonk_extension_host_upper
        + shplonk_fixed_workspace_bytes
        + shplonk_batch_width_upper * shplonk_task_bytes
    )
    proof_host_peak = max(
        lookup_prepare_host_peak,
        lookup_commit_host_peak,
        proof_eval_h_host_peak,
        shplonk_host_peak,
        gat_proof_synthesis_host_peak,
        advice_conversion_host_peak,
        native_pk_read_host_peak,
    )
    setup_fixed_assignment_dirty_host_upper = fixed_assignment_maps
    setup_keygen_host_peak = (
        srs_loaded_and_witness
        + setup_witnessless_shape_values_upper
        + setup_fixed_assignment_dirty_host_upper
        + gat_transient_witness_bytes
        + max(selector_phase, pk_phase, read_scratch, icicle_msm_staging, permutation_vk_phase)
    )
    host_known = max(setup_keygen_host_peak, srs_setup_peak, proof_host_peak)
    return {
        "fixed_assignment_mapped_bytes": fixed_assignment_maps,
        "setup_persisted_mapped_bytes": setup_persisted_maps,
        "setup_transition_mapped_bytes": setup_transition_map,
        "setup_disk_known_bytes": setup_mapped_peak,
        "setup_mapped_peak_bytes": setup_mapped_peak,
        "proof_spill_known_bytes": proof_spill_known,
        "advice_polynomial_mapped_bytes": advice * n * fr,
        "advice_event_log_records_upper_bytes": advice_event_log_records_upper,
        "advice_event_log_rounding_upper_bytes": advice_event_log_rounding_upper,
        "advice_event_log_reserved_upper_bytes": advice_event_log_reserved_upper,
        "advice_latest_index_mapped_bytes": advice_latest_map_bytes,
        "advice_conversion_current_field_map_upper_bytes": n * fr,
        "advice_event_column_resident_upper_bytes": advice * advice_event_column_bytes,
        "advice_event_decode_batch_resident_upper_bytes": advice_event_batch_bytes,
        "native_pk_read_successor_file_bytes": native_successor_file_bytes,
        "native_pk_read_successor_reserved_bytes": native_successor_backing_reserved_bytes,
        "native_pk_read_target_bitmap_virtual_bytes": target_bitmap_virtual_bytes,
        "native_pk_read_target_bitmap_backing_upper_bytes": target_bitmap_page_bytes,
        "native_pk_read_target_bitmap_allocation_bitmap_upper_bytes": target_bitmap_allocation_bitmap_bytes,
        "native_pk_read_backing_required_bytes": native_pk_read_backing_required,
        "native_pk_read_backing_evidence_kind": (
            "exact_successor_count_with_page_rounded_store_and_4KiB_target_page_upper"
            if native_census is not None
            else "source_successor_ceiling_with_page_rounded_store_and_4KiB_target_page_upper_census_pending"
        ),
        "native_pk_read_host_scratch_upper_bytes": native_pk_read_host_scratch,
        "native_pk_read_successor_cache_upper_bytes": int(memory["native_successor_read_cache_bytes"]),
        "native_pk_read_target_bitmap_fixed_cache_upper_bytes": int(memory["native_target_bitmap_fixed_cache_bytes"]),
        "native_pk_read_validation_buffers_upper_bytes": native_pk_read_validation_buffers,
        "native_pk_read_stream_buffer_upper_bytes": native_pk_read_stream_buffer,
        "lookup_query_spill_bytes": lookup_query_spill_bytes,
        "lookup_m_spill_bytes": lookup_m_spill_bytes,
        "lookup_table_spill_bytes": lookup_table_spill_bytes,
        "lookup_phi_spill_bytes": lookup_phi_spill_bytes,
        "lookup_spill_known_bytes": lookup_spill_known,
        "srs_loaded_host_bytes": srs_loaded,
        "srs_downsize_peak_host_bytes": srs_downsize_peak,
        "srs_fft_data_bytes": srs_fft_data,
        "srs_fft_metadata_bytes": srs_fft_metadata,
        "srs_fft_cache_bytes": srs_fft_cache_metadata,
        "srs_fft_transform_peak_bytes": srs_fft_transform_peak,
        "srs_fft_output_peak_bytes": srs_fft_output_peak,
        "srs_fft_cache_build_peak_bytes": srs_fft_cache_build_peak,
        "srs_fft_twiddle_lut_bytes_upper": srs_fft_twiddle_lut,
        "srs_setup_host_peak_bytes": srs_setup_peak,
        "setup_keygen_host_peak_bytes": setup_keygen_host_peak,
        "setup_keygen_future_host_reserve_bytes": setup_keygen_future_host_reserve,
        "setup_fixed_assignment_dirty_host_upper_bytes": setup_fixed_assignment_dirty_host_upper,
        "permutation_sparse_cache_host_bytes": permutation_cache_host if native_census is not None else None,
        "permutation_allocation_bitmap_host_bytes": permutation_bitmap_host if native_census is not None else None,
        "permutation_data_backing_observed_bytes": permutation_data_backing if native_census is not None else None,
        "permutation_directory_backing_observed_bytes": permutation_directory_backing if native_census is not None else None,
        "permutation_successor_entries_upper": expected_successor_entries_upper,
        "permutation_successor_entries_observed": successor_entries if native_census is not None else None,
        "permutation_successor_backing_file_upper_bytes": native_successor_backing_reserved_bytes,
        "permutation_aux_entries_upper": successor_entries if native_census is not None else None,
        "permutation_sizes_entries_upper": None,
        "permutation_mapping_host_bytes": permutation_mapping_host if native_census is not None else None,
        "selector_keygen_host_phase_bytes": selector_phase,
        "permutation_pk_host_phase_bytes": pk_phase,
        "proof_nonprepare_resident_host_bytes": proof_nonprepare_resident,
        "proof_nonprepare_heap_host_bytes": proof_nonprepare_heap,
        "proof_nonprepare_hot_mmap_host_bytes": proof_nonprepare_hot_mmap,
        "proof_evaluation_hot_mmap_host_bytes": proof_eval_hot_mmap,
        "proof_evaluation_coset_mapped_window_bytes": evaluation_temporary_mapped_bytes,
        "proof_evaluation_source_mapped_window_bytes": evaluation_source_hot_bytes,
        "proof_evaluation_fft_cache_host_bytes": proof_fft_cache_bytes,
        "proof_evaluation_fft_transient_host_bytes": proof_fft_transient,
        "proof_evaluation_lookup_task_host_bytes": lookup_eval_task_host_peak,
        "proof_evaluation_host_peak_bytes": proof_eval_h_host_peak,
        "shplonk_batch_width_upper": shplonk_batch_width_upper,
        "shplonk_task_host_bytes": shplonk_task_bytes,
        "shplonk_fixed_workspace_host_bytes": shplonk_fixed_workspace_bytes,
        "shplonk_extension_host_upper_bytes": shplonk_extension_host_upper,
        "shplonk_host_peak_bytes": shplonk_host_peak,
        "shplonk_transient_spill_upper_bytes": shplonk_transient_spill_upper,
        "lookup_map_retained_host_bytes": lookup_maps_retained,
        "lookup_map_growth_transient_host_bytes": lookup_map_growth_peak,
        "lookup_table_polynomial_mapped_bytes": lookup_table_polynomials,
        "lookup_metadata_retained_host_bytes": lookup_metadata_retained,
        "lookup_table_cache_retained_host_bytes": lookup_table_cache_retained_host,
        "lookup_table_build_transient_host_bytes": lookup_table_build_peak,
        "lookup_table_cache_peak_host_bytes": lookup_table_cache_peak_host,
        "lookup_prepare_task_host_bytes": lookup_prepare_task_bytes,
        "lookup_prepare_host_peak_bytes": lookup_prepare_host_peak,
        "lookup_commit_task_host_bytes": lookup_commit_task_bytes,
        "lookup_commit_active_map_window_bytes_upper": lookup_commit_active_map_window,
        "lookup_commit_host_peak_bytes": lookup_commit_host_peak,
        "proof_host_peak_bytes": proof_host_peak,
        "proof_gat_synthesis_host_peak_bytes": gat_proof_synthesis_host_peak,
        "proof_advice_conversion_mapped_host_upper_bytes": advice_conversion_mapped_host_upper,
        "proof_advice_conversion_host_peak_bytes": advice_conversion_host_peak,
        "proof_native_pk_read_host_peak_bytes": native_pk_read_host_peak,
        "lookup_map_bounds": [int(item) for item in table_bounds],
        "lookup_map_bucket_count_upper": [max(4, 1 << (((int(item) * 8 + 6) // 7 - 1).bit_length())) for item in table_bounds],
        "setup_key_artifact_upper_bytes": setup_key_artifact_upper,
        "quotient_extended_backing_bytes": proof_domain * fr,
        "quotient_piece_bytes": quotient_coefficients * fr,
        "setup_witnessless_shape_values_upper_bytes": setup_witnessless_shape_values_upper,
        "gat_transient_witness_bytes_upper": gat_transient_witness_bytes,
        "host_known_bytes": host_known,
        "selector_activation_storage_observed_bytes": selectors if native_census is not None else None,
        "selector_activation_storage_static_upper_bytes": selector_census_storage,
        "selector_activation_storage_evidence_kind": "exact_native_vk_pk_census" if native_census is not None else "static_ceiling_only_native_census_pending",
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
    disk_budget_components: dict[str, object] | None = None,
) -> dict[str, object]:
    host_available, host_accounting = _effective_host_memory_available_bytes()
    os_reserve, reserve_accounting = _host_os_diagnostic_reserve_bytes()
    disk_available, disk_accounting = _filesystem_available_bytes(output_root.parent)
    device_free = int(device_sample["free_memory_bytes"])
    device_peak = int(workspace_query["device_workspace_peak_bytes"])
    integrated = bool(workspace_query["integrated_device"])
    device_headroom = device_peak if integrated else 0
    host_required = requirements["host_known_bytes"] + device_headroom + os_reserve
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
        "host_os_diagnostic_reserve_bytes": os_reserve,
        "host_reserve_accounting": reserve_accounting,
        "host_unified_device_headroom_bytes": device_headroom,
        "disk_available_bytes": disk_available,
        "disk_required_known_bytes": remaining_known_disk_bytes,
        "disk_budget_components": disk_budget_components or {"total_required_bytes": remaining_known_disk_bytes},
        "disk_accounting": disk_accounting,
        "cuda_free_bytes": device_free,
        "cuda_required_workspace_bytes": device_peak,
        "cuda_total_bytes": device_sample["total_memory_bytes"],
        "cuda_device_uuid": device_sample["device_uuid"],
        "integrated_device": integrated,
        "coverage": "source-derived-known-allocation-gates; not a complete phase peak estimate",
        "sampled_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _authenticated_srs_copy_size_bytes(srs_path: Path, *, expected_size_bytes: object | None = None) -> int:
    """Return the checked size of an already-authenticated canonical SRS source."""
    try:
        canonical = Path(srs_path).resolve(strict=True)
        source_stat = canonical.stat()
    except (OSError, RuntimeError) as exc:
        abort(f"authenticated native SRS source cannot be resolved/stat'ed for capacity: {exc}")
    if not stat.S_ISREG(source_stat.st_mode):
        abort("authenticated native SRS source is not a regular file for capacity accounting")
    actual_size = source_stat.st_size
    if isinstance(actual_size, bool) or not isinstance(actual_size, int) or actual_size <= 0 or actual_size > sys.maxsize:
        abort("authenticated native SRS source has an invalid/unrepresentable size for capacity accounting")
    if expected_size_bytes is not None:
        if (
            isinstance(expected_size_bytes, bool)
            or not isinstance(expected_size_bytes, int)
            or expected_size_bytes <= 0
            or expected_size_bytes > sys.maxsize
            or expected_size_bytes != actual_size
        ):
            abort("authenticated native SRS source size differs from its validated manifest metadata")
    return actual_size


def _pending_srs_copy_disk_budget(
    srs_copy_size_bytes: int,
    pending_finalize_datasets: set[str],
    *,
    dataset_index: int = 0,
) -> dict[str, object]:
    """Account one full native final-prepare SRS copy per not-yet-finalized cell."""
    if (
        isinstance(srs_copy_size_bytes, bool)
        or not isinstance(srs_copy_size_bytes, int)
        or srs_copy_size_bytes <= 0
        or srs_copy_size_bytes > sys.maxsize
    ):
        abort("native SRS copy size is invalid/unrepresentable for capacity accounting")
    if isinstance(dataset_index, bool) or not isinstance(dataset_index, int) or not 0 <= dataset_index <= len(DATASETS):
        abort("pending native SRS copy dataset index is invalid")
    if not isinstance(pending_finalize_datasets, set) or not pending_finalize_datasets.issubset(set(DATASETS)):
        abort("pending final-prepare dataset selection does not match the formal dataset set")
    pending = [name for index, name in enumerate(DATASETS) if index >= dataset_index and name in pending_finalize_datasets]
    count = len(pending)
    if count and srs_copy_size_bytes > sys.maxsize // count:
        abort("pending native SRS copy budget exceeds the representable byte range")
    total = count * srs_copy_size_bytes
    return {
        "output_srs_copy_size_bytes": srs_copy_size_bytes,
        "pending_output_srs_copy_count": count,
        "pending_output_srs_copy_bytes": total,
        "pending_output_srs_copy_datasets": pending,
    }


def _write_native_keygen_budget_receipt(
    output: Path,
    *,
    dataset: str,
    session_id: str,
    admission: dict[str, object],
    resume_identity: dict[str, object],
    capacity: dict[str, object],
    setup_check: dict[str, object],
    workspace_query: dict[str, object],
) -> str:
    path = output / "native_keygen_budget.json"
    if path.exists() or path.is_symlink():
        abort(f"{dataset} native keygen budget refuses to overwrite an existing receipt")
    disk_available = setup_check.get("disk_available_bytes")
    setup_disk_required = setup_check.get("disk_required_known_bytes")
    proof_spill_reserve = capacity.get("proof_spill_known_bytes")
    future_host_reserve = capacity.get("setup_keygen_future_host_reserve_bytes")
    successor_file_upper = capacity.get("permutation_successor_backing_file_upper_bytes")
    gat_transient_witness = capacity.get("gat_transient_witness_bytes_upper")
    setup_keygen_host_peak = capacity.get("setup_keygen_host_peak_bytes")
    proof_gat_synthesis_host_peak = capacity.get("proof_gat_synthesis_host_peak_bytes")
    memory_model = admission.get("native_memory_model")
    admitted_gat_transient = memory_model.get("gat_transient_witness_bytes_upper") if isinstance(memory_model, dict) else None
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > sys.maxsize
        for value in (
            disk_available,
            setup_disk_required,
            proof_spill_reserve,
            future_host_reserve,
            successor_file_upper,
            gat_transient_witness,
            setup_keygen_host_peak,
            proof_gat_synthesis_host_peak,
        )
    ):
        abort(f"{dataset} native keygen budget inputs are malformed")
    if (
        isinstance(admitted_gat_transient, bool)
        or not isinstance(admitted_gat_transient, int)
        or gat_transient_witness != admitted_gat_transient
        or setup_keygen_host_peak < gat_transient_witness
        or proof_gat_synthesis_host_peak < gat_transient_witness
    ):
        abort(f"{dataset} native keygen budget GAT tensor reserve does not match the admitted capacity envelope")
    if setup_disk_required > disk_available:
        abort(
            f"{dataset} setup keygen peak needs {setup_disk_required} bytes, "
            f"but setup admission observed only {disk_available} bytes"
        )
    # The setup check already reserves the complete PK/VK artifacts, mapped
    # setup peak and pending SRS copy. The native sparse stores are a bounded
    # transient on top of that phase; future proof spill is separately checked
    # as a sequential phase and is never double-counted as a setup allocation.
    native_permutation_backing_budget = disk_available - setup_disk_required
    relation_identity = admission.get("relation_identity")
    if not isinstance(relation_identity, dict):
        abort(f"{dataset} native keygen budget lacks its admitted relation identity")
    value = {
        "schema": "ezkl.native-keygen-budget.v1",
        "session_id": session_id,
        "dataset": dataset,
        "admission_sha256": admission.get("sha256"),
        "settings_sha256": admission.get("settings_sha256"),
        "binary_sha256": resume_identity.get("binary_sha256"),
        "resource_manifest_sha256": relation_identity.get("resource_manifest_sha256"),
        "build_receipt_sha256": resume_identity.get("build_receipt_sha256"),
        "build_environment_sha256": resume_identity.get("build_environment_sha256"),
        "logrows": admission.get("logrows"),
        "domain_size": admission.get("domain_size"),
        "gat_transient_witness_bytes_upper": gat_transient_witness,
        "host_available_bytes": setup_check.get("host_available_bytes"),
        "host_os_diagnostic_reserve_bytes": setup_check.get("host_os_diagnostic_reserve_bytes"),
        "setup_keygen_future_host_reserve_bytes": future_host_reserve,
        "integrated_device": workspace_query.get("integrated_device"),
        "device_workspace_peak_bytes": workspace_query.get("device_workspace_peak_bytes"),
        "disk_available_bytes": disk_available,
        "setup_disk_required_known_bytes": setup_disk_required,
        "proof_spill_reserve_bytes": proof_spill_reserve,
        "disk_required_known_bytes": setup_disk_required,
        "permutation_successor_backing_file_upper_bytes": successor_file_upper,
        "native_permutation_backing_budget_bytes": native_permutation_backing_budget,
        "native_successor_read_cache_bytes": admission["native_memory_model"]["native_successor_read_cache_bytes"],
        "native_target_bitmap_fixed_cache_bytes": admission["native_memory_model"]["native_target_bitmap_fixed_cache_bytes"],
        "native_pk_read_buffer_capacity_bytes": admission["native_memory_model"]["native_pk_read_buffer_capacity_bytes"],
        "host_budget_scope": "one_shared_GB10_CPU_GPU_physical_pool; device_workspace_counted_once",
        "disk_budget_scope": "setup_and_future_proof_known_peaks_reserved; native_sparse_backing_has_finite_residual_cap",
    }
    _write_session_record(path, value)
    return sha256(path)


def _write_native_prove_budget_receipt(
    output: Path,
    *,
    dataset: str,
    session_id: str,
    admission: dict[str, object],
    resume_identity: dict[str, object],
    setup_receipt_sha256: str,
    native_keygen_budget_sha256: str,
    native_keygen_receipt_sha256: str,
    capacity: dict[str, object],
    prove_check: dict[str, object],
    workspace_query: dict[str, object],
) -> str:
    path = output / "native_prove_budget.json"
    if path.exists() or path.is_symlink():
        abort(f"{dataset} native prove budget refuses to overwrite an existing receipt")
    disk_available = prove_check.get("disk_available_bytes")
    disk_required = prove_check.get("disk_required_known_bytes")
    host_available = prove_check.get("host_available_bytes")
    host_required = prove_check.get("host_required_known_bytes")
    native_read_required = capacity.get("native_pk_read_backing_required_bytes")
    native_read_host_scratch = capacity.get("native_pk_read_host_scratch_upper_bytes")
    native_read_cache = capacity.get("native_pk_read_successor_cache_upper_bytes")
    target_bitmap_cache = capacity.get("native_pk_read_target_bitmap_fixed_cache_upper_bytes")
    target_bitmap_allocation = capacity.get("native_pk_read_target_bitmap_allocation_bitmap_upper_bytes")
    validation_buffers = capacity.get("native_pk_read_validation_buffers_upper_bytes")
    stream_buffer = capacity.get("native_pk_read_stream_buffer_upper_bytes")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > sys.maxsize
        for value in (
            disk_available,
            disk_required,
            host_available,
            host_required,
            native_read_required,
            native_read_host_scratch,
            native_read_cache,
            target_bitmap_cache,
            target_bitmap_allocation,
            validation_buffers,
            stream_buffer,
        )
    ):
        abort(f"{dataset} native prove budget has malformed host or disk accounting")
    if native_read_host_scratch != native_read_cache + target_bitmap_cache + target_bitmap_allocation + validation_buffers + stream_buffer:
        abort(f"{dataset} native PK reader host scratch does not equal its source-derived cache/buffer components")
    if host_required > host_available or native_read_host_scratch > host_required:
        abort(f"{dataset} native PK reader host scratch exceeds the checked shared-memory envelope")
    if native_read_required > disk_required:
        abort(f"{dataset} native PK read backing is missing from the checked proof disk peak")
    other_known = disk_required - native_read_required
    max_backing = min(native_read_required, disk_available - other_known) if disk_available >= other_known else 0
    if max_backing < native_read_required:
        abort(
            f"{dataset} native PK reader needs {native_read_required} source-bounded backing bytes, "
            f"but only {max_backing} remain after the other proof spill reservations"
        )
    required_identity = admission.get("relation_identity")
    if not isinstance(required_identity, dict):
        abort(f"{dataset} native prove budget lacks the admitted relation identity")
    value = {
        "schema": "ezkl.native-prove-budget.v1",
        "session_id": session_id,
        "dataset": dataset,
        "admission_sha256": admission.get("sha256"),
        "resource_manifest_sha256": required_identity.get("resource_manifest_sha256"),
        "build_receipt_sha256": resume_identity.get("build_receipt_sha256"),
        "build_environment_sha256": resume_identity.get("build_environment_sha256"),
        "binary_sha256": resume_identity.get("binary_sha256"),
        "settings_sha256": admission.get("settings_sha256"),
        "compiled_circuit_sha256": sha256(require_file(output / "compiled.circuit", f"{dataset} compiled circuit")),
        "vk_sha256": sha256(require_file(output / "vk.key", f"{dataset} verifying key")),
        "pk_sha256": sha256(require_file(output / "pk.key", f"{dataset} proving key")),
        "setup_receipt_sha256": setup_receipt_sha256,
        "native_keygen_budget_sha256": native_keygen_budget_sha256,
        "native_keygen_receipt_sha256": native_keygen_receipt_sha256,
        "disk_available_bytes": disk_available,
        "disk_required_known_bytes": disk_required,
        "native_pk_read_backing_required_bytes": native_read_required,
        "max_total_backing_bytes": max_backing,
        "native_pk_read_host_scratch_upper_bytes": native_read_host_scratch,
        "native_pk_read_successor_cache_upper_bytes": native_read_cache,
        "native_pk_read_target_bitmap_fixed_cache_upper_bytes": target_bitmap_cache,
        "native_pk_read_target_bitmap_allocation_bitmap_upper_bytes": target_bitmap_allocation,
        "native_pk_read_validation_buffers_upper_bytes": validation_buffers,
        "native_pk_read_stream_buffer_upper_bytes": stream_buffer,
        "host_available_bytes": host_available,
        "host_required_known_bytes": host_required,
        "integrated_device": workspace_query.get("integrated_device"),
        "device_workspace_peak_bytes": workspace_query.get("device_workspace_peak_bytes"),
        "backing_scope": "single_formal_pk_stream_read_and_target_validation; exact_successor_count_plus_4KiB_target_page_upper",
        "host_scope": "one_shared_GB10_CPU_GPU_physical_pool; device_workspace_counted_once",
    }
    _write_session_record(path, value)
    return sha256(path)


def _load_reusable_native_prove_budget_receipt(
    output: Path,
    *,
    dataset: str,
    session_id: str,
    admission: dict[str, object],
    resume_identity: dict[str, object],
    setup_receipt_sha256: str,
    native_keygen_budget_sha256: str,
    native_keygen_receipt_sha256: str,
    capacity: dict[str, object],
    workspace_query: dict[str, object],
) -> str | None:
    """Reuse only an exact session-bound prove budget; never rewrite a proven stage's input."""
    path = output / "native_prove_budget.json"
    if not path.exists() and not path.is_symlink():
        return None
    if path.is_symlink() or not path.is_file():
        return None
    try:
        budget = load_json(path, f"{dataset} native prove budget")
    except ContractError:
        return None
    relation = admission.get("relation_identity")
    if not isinstance(budget, dict) or not isinstance(relation, dict):
        return None
    expected = {
        "schema": "ezkl.native-prove-budget.v1",
        "session_id": session_id,
        "dataset": dataset,
        "admission_sha256": admission.get("sha256"),
        "resource_manifest_sha256": relation.get("resource_manifest_sha256"),
        "build_receipt_sha256": resume_identity.get("build_receipt_sha256"),
        "build_environment_sha256": resume_identity.get("build_environment_sha256"),
        "binary_sha256": resume_identity.get("binary_sha256"),
        "settings_sha256": admission.get("settings_sha256"),
        "compiled_circuit_sha256": sha256(output / "compiled.circuit") if (output / "compiled.circuit").is_file() else None,
        "vk_sha256": sha256(output / "vk.key") if (output / "vk.key").is_file() else None,
        "pk_sha256": sha256(output / "pk.key") if (output / "pk.key").is_file() else None,
        "setup_receipt_sha256": setup_receipt_sha256,
        "native_keygen_budget_sha256": native_keygen_budget_sha256,
        "native_keygen_receipt_sha256": native_keygen_receipt_sha256,
        "native_pk_read_backing_required_bytes": capacity.get("native_pk_read_backing_required_bytes"),
        "native_pk_read_host_scratch_upper_bytes": capacity.get("native_pk_read_host_scratch_upper_bytes"),
        "native_pk_read_successor_cache_upper_bytes": capacity.get("native_pk_read_successor_cache_upper_bytes"),
        "native_pk_read_target_bitmap_fixed_cache_upper_bytes": capacity.get("native_pk_read_target_bitmap_fixed_cache_upper_bytes"),
        "native_pk_read_target_bitmap_allocation_bitmap_upper_bytes": capacity.get("native_pk_read_target_bitmap_allocation_bitmap_upper_bytes"),
        "native_pk_read_validation_buffers_upper_bytes": capacity.get("native_pk_read_validation_buffers_upper_bytes"),
        "native_pk_read_stream_buffer_upper_bytes": capacity.get("native_pk_read_stream_buffer_upper_bytes"),
        "integrated_device": workspace_query.get("integrated_device"),
        "device_workspace_peak_bytes": workspace_query.get("device_workspace_peak_bytes"),
        "backing_scope": "single_formal_pk_stream_read_and_target_validation; exact_successor_count_plus_4KiB_target_page_upper",
        "host_scope": "one_shared_GB10_CPU_GPU_physical_pool; device_workspace_counted_once",
    }
    dynamic_fields = {
        "disk_available_bytes", "disk_required_known_bytes", "max_total_backing_bytes",
        "host_available_bytes", "host_required_known_bytes",
    }
    if set(budget) != set(expected) | dynamic_fields or any(budget.get(name) != value for name, value in expected.items()):
        return None
    integers = {
        name: budget.get(name)
        for name in (
            "disk_available_bytes", "disk_required_known_bytes", "max_total_backing_bytes",
            "host_available_bytes", "host_required_known_bytes", "native_pk_read_backing_required_bytes",
            "native_pk_read_host_scratch_upper_bytes", "native_pk_read_successor_cache_upper_bytes",
            "native_pk_read_target_bitmap_fixed_cache_upper_bytes", "native_pk_read_target_bitmap_allocation_bitmap_upper_bytes",
            "native_pk_read_validation_buffers_upper_bytes", "native_pk_read_stream_buffer_upper_bytes", "device_workspace_peak_bytes",
        )
    }
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > sys.maxsize for value in integers.values()):
        return None
    scratch = sum(
        int(integers[name])
        for name in (
            "native_pk_read_successor_cache_upper_bytes", "native_pk_read_target_bitmap_fixed_cache_upper_bytes",
            "native_pk_read_target_bitmap_allocation_bitmap_upper_bytes", "native_pk_read_validation_buffers_upper_bytes",
            "native_pk_read_stream_buffer_upper_bytes",
        )
    )
    disk_required = int(integers["disk_required_known_bytes"])
    disk_available = int(integers["disk_available_bytes"])
    native_read_required = int(integers["native_pk_read_backing_required_bytes"])
    other_known = disk_required - native_read_required
    expected_backing = min(native_read_required, disk_available - other_known) if other_known >= 0 and disk_available >= other_known else 0
    if (
        not isinstance(budget.get("integrated_device"), bool)
        or disk_available <= 0
        or int(integers["host_available_bytes"]) <= 0
        or disk_required > disk_available
        or native_read_required > disk_required
        or int(integers["max_total_backing_bytes"]) != expected_backing
        or int(integers["max_total_backing_bytes"]) < native_read_required
        or int(integers["host_required_known_bytes"]) > int(integers["host_available_bytes"])
        or int(integers["native_pk_read_host_scratch_upper_bytes"]) != scratch
        or int(integers["native_pk_read_host_scratch_upper_bytes"]) > int(integers["host_required_known_bytes"])
    ):
        return None
    return sha256(path)


def _disk_budget_components(
    *,
    pending_srs: dict[str, object],
    setup_key_artifact_upper_bytes: int = 0,
    setup_mapped_peak_bytes: int = 0,
    proof_spill_known_bytes: int = 0,
) -> dict[str, object]:
    numeric = {
        "future_setup_key_artifact_upper_bytes": setup_key_artifact_upper_bytes,
        "setup_mapped_peak_bytes": setup_mapped_peak_bytes,
        "proof_spill_known_bytes": proof_spill_known_bytes,
        "output_srs_copy_size_bytes": pending_srs["output_srs_copy_size_bytes"],
        "pending_output_srs_copy_count": pending_srs["pending_output_srs_copy_count"],
        "pending_output_srs_copy_bytes": pending_srs["pending_output_srs_copy_bytes"],
    }
    total = 0
    for name in (
        "future_setup_key_artifact_upper_bytes",
        "setup_mapped_peak_bytes",
        "proof_spill_known_bytes",
        "pending_output_srs_copy_bytes",
    ):
        value = numeric[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > sys.maxsize:
            abort(f"disk capacity component {name} is invalid/unrepresentable")
        if value > sys.maxsize - total:
            abort("known disk capacity budget exceeds the representable byte range")
        total += value
    for name in ("output_srs_copy_size_bytes", "pending_output_srs_copy_count"):
        value = numeric[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > sys.maxsize:
            abort(f"disk capacity component {name} is invalid/unrepresentable")
    return {**numeric, "pending_output_srs_copy_datasets": list(pending_srs["pending_output_srs_copy_datasets"]), "total_required_bytes": total}


def _native_prepare_budget(requirements: dict[str, int], *, initial_available_bytes: int, workspace_query: dict[str, object]) -> dict[str, int]:
    """Derive the native lookup task cap before native proof objects are loaded."""
    total_limit, _ = _effective_host_memory_limit_bytes()
    reserve, _ = _host_os_diagnostic_reserve_bytes()
    integrated = bool(workspace_query["integrated_device"])
    device_peak = int(workspace_query["device_workspace_peak_bytes"])
    device_headroom = device_peak if integrated else 0
    baseline_heap = requirements["proof_nonprepare_heap_host_bytes"]
    baseline_hot_mmap = requirements["proof_nonprepare_hot_mmap_host_bytes"]
    tablecache = requirements["lookup_table_cache_retained_host_bytes"]
    icicle_staging = requirements["icicle_msm_host_staging_bytes"]
    shplonk_extensions = requirements["shplonk_extension_host_upper_bytes"]
    task_envelope = max(
        0, initial_available_bytes - baseline_heap - baseline_hot_mmap - tablecache - shplonk_extensions - icicle_staging - device_headroom - reserve
    )
    minimum_task = max(
        requirements["lookup_prepare_task_host_bytes"],
        requirements["lookup_commit_task_host_bytes"],
        requirements["shplonk_fixed_workspace_host_bytes"] + requirements["shplonk_task_host_bytes"],
    )
    if minimum_task <= 0 or task_envelope < minimum_task:
        abort(
            "EZKL native lookup or SHPLONK cannot fit one source-bounded task after "
            f"reserving its loaded baseline, table cache, SHPLONK extensions, ICICLE staging, unified-device headroom, "
            f"and OS reserve: need {minimum_task} task bytes, have {task_envelope}"
        )
    return {
        "total_limit_bytes": total_limit,
        "initial_available_bytes": initial_available_bytes,
        "baseline_heap_bytes": baseline_heap,
        "baseline_hot_mmap_bytes": baseline_hot_mmap,
        "tablecache_bytes": tablecache,
        "shplonk_extensions_bytes": shplonk_extensions,
        "icicle_staging_bytes": icicle_staging,
        "device_headroom_bytes": device_headroom,
        "reserve_bytes": reserve,
        "incremental_envelope_bytes": max(0, initial_available_bytes - baseline_heap - baseline_hot_mmap - device_headroom - reserve),
        "future_phase_increment_bytes": tablecache + shplonk_extensions + icicle_staging + task_envelope,
        "minimum_task_bytes": minimum_task,
        "taskcap_bytes": task_envelope,
    }


def _validated_native_prepare_exports(
    budget: object, cardinality_bounds: object, *, domain_size: int, expected_lookup_groups: int
) -> tuple[str, str, str]:
    """Fail closed before exporting values consumed by native lookup preparation."""
    expected_keys = {
        "total_limit_bytes",
        "initial_available_bytes",
        "baseline_heap_bytes",
        "baseline_hot_mmap_bytes",
        "tablecache_bytes",
        "shplonk_extensions_bytes",
        "icicle_staging_bytes",
        "device_headroom_bytes",
        "reserve_bytes",
        "incremental_envelope_bytes",
        "future_phase_increment_bytes",
        "minimum_task_bytes",
        "taskcap_bytes",
    }
    if (
        not isinstance(budget, dict)
        or set(budget) != expected_keys
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in budget.values())
        or budget["total_limit_bytes"] <= 0
        or budget["initial_available_bytes"] <= 0
        or budget["minimum_task_bytes"] <= 0
        or budget["taskcap_bytes"] <= 0
        or budget["taskcap_bytes"] < budget["minimum_task_bytes"]
        or budget["taskcap_bytes"] > budget["initial_available_bytes"]
        or budget["initial_available_bytes"] > sys.maxsize
        or budget["taskcap_bytes"] > sys.maxsize
        or budget["device_headroom_bytes"] > sys.maxsize
    ):
        abort("native prepare budget contains invalid, zero, or unrepresentable exported byte counts")
    if (
        isinstance(domain_size, bool)
        or not isinstance(domain_size, int)
        or domain_size <= 0
        or domain_size > sys.maxsize
        or isinstance(expected_lookup_groups, bool)
        or not isinstance(expected_lookup_groups, int)
        or expected_lookup_groups <= 0
        or not isinstance(cardinality_bounds, list)
        or len(cardinality_bounds) != expected_lookup_groups
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 or value > domain_size or value > sys.maxsize for value in cardinality_bounds)
    ):
        abort("native lookup cardinality exports do not match the admitted source circuit/domain")
    return (
        str(budget["taskcap_bytes"]),
        str(budget["device_headroom_bytes"]),
        ",".join(str(value) for value in cardinality_bounds),
    )


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
    disk_budget_components: dict[str, object] | None = None,
) -> dict[str, object]:
    sample = _sample_native_cuda_memory(
        receipt, session_dir=session_dir, session_id=session_id, selected_device=selected_device, gpu_identity=gpu_identity, build_environment=build_environment
    )
    return _check_dataset_capacity(
        dataset,
        requirements,
        output_root=output_root,
        remaining_known_disk_bytes=remaining_known_disk_bytes,
        phase=phase,
        device_sample=sample,
        workspace_query=receipt["query"],
        disk_budget_components=disk_budget_components,
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
    srs_copy_size_bytes: int,
    active_datasets: set[str] | None = None,
    setup_required_datasets: set[str] | None = None,
    pending_finalize_datasets: set[str] | None = None,
    workspace_receipt: dict[str, object] | None = None,
) -> tuple[dict[str, object], dict[str, dict[str, int]], dict[str, object]]:
    receipt = workspace_receipt or _build_native_cub_workspace_query(session_dir, session_id, runtime_identity, selected_device, build_environment, gpu_identity)
    receipt = _validate_native_cub_workspace_receipt(receipt, session_dir, selected_device, gpu_identity, session_id)
    query = receipt["query"]
    if not isinstance(query, dict):
        abort("EZKL CUB workspace receipt has no decoded query object")
    sizes = query["native_type_sizes"]
    for dataset, admission in admissions.items():
        model = admission["native_memory_model"]
        if sizes["scalar_bytes"] != model["size_of_fr_bytes"] or model["cuda_total_memory_bytes"] != query["total_memory_bytes"]:
            abort(f"{dataset} native admission scalar/device values disagree with compiled ICICLE identity")
    requirements = {dataset: _native_capacity_requirements(admission, sizes) for dataset, admission in admissions.items()}
    active = set(requirements) if active_datasets is None else active_datasets
    setup_required = active if setup_required_datasets is None else setup_required_datasets
    pending_finalize = active if pending_finalize_datasets is None else pending_finalize_datasets
    if (
        not active.issubset(requirements)
        or not setup_required.issubset(active)
        or not pending_finalize.issubset(active)
        or not setup_required.issubset(pending_finalize)
    ):
        abort("resume capacity stage selections do not match the admitted datasets")
    future_key_artifacts = sum(requirements[name]["setup_key_artifact_upper_bytes"] for name in setup_required)
    max_setup_map_peak = max((requirements[name]["setup_mapped_peak_bytes"] for name in setup_required), default=0)
    max_future_proof_spill = max((requirements[name]["proof_spill_known_bytes"] for name in active), default=0)
    initial_srs_budget = _pending_srs_copy_disk_budget(srs_copy_size_bytes, pending_finalize)
    initial_disk_components = _disk_budget_components(
        pending_srs=initial_srs_budget,
        setup_key_artifact_upper_bytes=future_key_artifacts,
        setup_mapped_peak_bytes=max_setup_map_peak,
    )
    # Setup's transient maps and a later dataset's proof spill are separate
    # phases. Keep all retained key files and SRS copies, then reserve the
    # larger of those two transient peaks instead of summing unrelated phases.
    srs_pending_bytes = int(initial_srs_budget["pending_output_srs_copy_bytes"])
    initial_phase_peak = max(max_setup_map_peak, max_future_proof_spill)
    initial_disk_components["proof_spill_known_bytes"] = max_future_proof_spill
    initial_disk_components["total_required_bytes"] = future_key_artifacts + initial_phase_peak + srs_pending_bytes
    if initial_disk_components["total_required_bytes"] > sys.maxsize:
        abort("initial known setup/proof disk peak exceeds the native address range")
    initial_disk_required = int(initial_disk_components["total_required_bytes"])
    initial_sample = {
        "device_uuid": query["device_uuid"],
        "free_memory_bytes": query["free_memory_bytes"],
        "total_memory_bytes": query["total_memory_bytes"],
        "integrated_device": query["integrated_device"],
        "sampled_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    checks: list[dict[str, object]] = []
    for dataset in DATASETS:
        if dataset not in active:
            continue
        dataset_requirements = dict(requirements[dataset])
        if dataset not in setup_required:
            dataset_requirements["host_known_bytes"] = dataset_requirements["proof_host_peak_bytes"]
        checks.append(
            _check_dataset_capacity(
                dataset,
                dataset_requirements,
                output_root=output_root,
                remaining_known_disk_bytes=initial_disk_required,
                phase="dataset_capacity_admission",
                device_sample=initial_sample,
                workspace_query=query,
                disk_budget_components=initial_disk_components,
            )
        )
    return (
        receipt,
        requirements,
        {
            "schema": "ezkl.local_known_capacity_gate.v1",
            "coverage": "source-derived-known-allocation-gates; not a complete phase peak estimate",
            "native_cub_workspace_query_sha256": receipt["result_sha256"],
            "session_id": session_id,
            "selected_device_uuid": selected_device["uuid"],
            "device_workspace_peak_bytes": query["device_workspace_peak_bytes"],
            "integrated_device": query["integrated_device"],
            "datasets": requirements,
            "source_srs_copy_size_bytes": srs_copy_size_bytes,
            "pending_finalize_datasets": [name for name in DATASETS if name in pending_finalize],
            "initial_disk_budget_components": initial_disk_components,
            "initial_checks": checks,
        },
    )


def _remaining_setup_disk_bytes(
    requirements: dict[str, dict[str, int]],
    dataset_index: int,
    *,
    include_current_transient: bool,
    setup_required_datasets: set[str] | None = None,
) -> int:
    setup_required = set(DATASETS) if setup_required_datasets is None else setup_required_datasets
    remaining = [name for name in DATASETS[dataset_index:] if name in setup_required]
    retained = sum(requirements[name]["setup_key_artifact_upper_bytes"] for name in remaining)
    if include_current_transient and DATASETS[dataset_index] in setup_required:
        retained += requirements[DATASETS[dataset_index]]["setup_mapped_peak_bytes"]
    return retained


def _remaining_srs_copy_disk_budget(
    srs_copy_size_bytes: int,
    dataset_index: int,
    *,
    pending_finalize_datasets: set[str],
) -> dict[str, object]:
    return _pending_srs_copy_disk_budget(
        srs_copy_size_bytes,
        pending_finalize_datasets,
        dataset_index=dataset_index,
    )


def update_session_record(path: Path, *, status: str, stage: str, error: str | None = None) -> None:
    document = json.loads(path.read_text())
    document["status"] = status
    document["stage"] = stage
    document["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    if error is not None:
        document["error"] = error[:2000]
    _write_session_record(path, document)


class _Tee:
    def __init__(self, *streams: object) -> None:
        self.streams = streams

    def write(self, value: str) -> int:
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return False


def invoke_python_entrypoint(
    module_name: str,
    argv: list[str],
    label: str,
    *,
    context: object,
    session_id: str,
    log_path: Path | None = None,
    stage_receipt_hooks: dict[str, object] | None = None,
    producer_wall_ms: float | None = None,
) -> float:
    module = importlib.import_module(module_name)
    parsed = module.parser().parse_args(argv)
    print(f"[ezkl-formal] {label}: in-process authenticated session call", flush=True)
    log_stream = None
    if log_path is not None:
        if log_path.is_symlink() or (log_path.exists() and not log_path.is_file()):
            abort(f"formal command log path is not a regular file: {log_path}")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_stream = log_path.open("a", encoding="utf-8")
        log_stream.write(f"\n--- {datetime.now(timezone.utc).isoformat()} {label} ---\n")
        log_stream.flush()
    started = time.monotonic()
    try:
        if log_stream is None:
            kwargs: dict[str, object] = {"authenticated_context": context, "formal_session_id": session_id}
            if stage_receipt_hooks is not None:
                kwargs["stage_receipt_hooks"] = stage_receipt_hooks
            if producer_wall_ms is not None:
                kwargs["producer_wall_ms"] = producer_wall_ms
            result = module.main(parsed, **kwargs)
        else:
            with contextlib.redirect_stdout(_Tee(sys.stdout, log_stream)), contextlib.redirect_stderr(_Tee(sys.stderr, log_stream)):
                kwargs = {"authenticated_context": context, "formal_session_id": session_id}
                if stage_receipt_hooks is not None:
                    kwargs["stage_receipt_hooks"] = stage_receipt_hooks
                if producer_wall_ms is not None:
                    kwargs["producer_wall_ms"] = producer_wall_ms
                result = module.main(parsed, **kwargs)
    except BaseException as exc:
        if log_stream is not None:
            log_stream.write(
                f"\n--- {datetime.now(timezone.utc).isoformat()} {label} "
                f"outcome=runner_error exception={type(exc).__name__} ---\n"
            )
            log_stream.flush()
        raise
    finally:
        if log_stream is not None:
            log_stream.close()
    elapsed_ms = (time.monotonic() - started) * 1000.0
    if log_path is not None:
        with log_path.open("a", encoding="utf-8") as log_stream:
            log_stream.write(
                f"\n--- {datetime.now(timezone.utc).isoformat()} {label} return_code={result} "
                f"wall_ms={elapsed_ms:.6f} outcome={'success' if result == 0 else 'failed'} ---\n"
            )
    if result != 0:
        abort(f"{label} failed with return code {result}")
    return elapsed_ms


def _artifact_manifest_projection_sha256(path: Path) -> str:
    manifest = load_json(path, "artifact identity manifest during stage recovery")
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), list):
        abort(f"artifact identity manifest is malformed during stage recovery: {path}")
    projected = dict(manifest)
    projected["files"] = [
        record for record in manifest["files"]
        if not isinstance(record, dict) or record.get("path") not in {"proof.json", "native_pk_read_timing.json"}
    ]
    return canonical_json_digest(projected)


_FORMAL_STAGE_ORDER = ("producer", "final", "witness", "prove", "verify")
_FORMAL_FINALIZATION_STAGE_LOGS = {
    "producer": "logs/producer.log",
    "final": "logs/final.log",
    "witness": "logs/witness.log",
    "prove": "logs/prove.log",
    "verify": "logs/verify.log",
}
_FORMAL_STAGE_OUTPUTS = {
    "producer": ("statement.json", "public_output.json", "producer_manifest.json"),
    "final": ("artifact_manifest.json", "srs.kzg"),
    "witness": ("witness.json",),
    "prove": ("proof.json", "native_proof_metrics.json", "native_pk_read_timing.json", "gpu_evidence.json", "gpu_prover_record.json"),
    "verify": ("native_verify_metrics.json", "core_run_manifest.json", "result.json"),
}


def _stage_paths_digest_map(stage: str, paths: dict[str, Path]) -> dict[str, str]:
    digests: dict[str, str] = {}
    for name, path in paths.items():
        if stage == "final" and name == "artifact_manifest":
            if path.is_symlink() or not path.is_file():
                abort(f"{stage} stage receipt requires a regular artifact manifest: {path}")
            digests[name] = _artifact_manifest_projection_sha256(path)
        else:
            if path.is_symlink() or not path.is_file():
                abort(f"stage receipt requires a regular {stage} file {name}: {path}")
            digests[name] = sha256(path)
    return digests


def _stage_receipt_hooks(
    *,
    session_file: Path,
    session_dir: Path,
    output_root: Path,
    output: Path,
    dataset: str,
    session_id: str,
    resume_identity: dict[str, object],
    admission_sha256: str,
    setup_receipt_sha256: str,
    native_prove_budget_sha256: str,
) -> dict[str, Callable[..., object]]:
    resume_identity_sha256 = canonical_json_digest(resume_identity)

    def expected_base(stage: str) -> dict[str, object]:
        if stage not in _FORMAL_STAGE_ORDER:
            abort(f"unknown formal stage receipt: {stage}")
        base: dict[str, object] = {
            "schema": "ezkl.formal-stage-receipt.v1",
            "session_id": session_id,
            "dataset": dataset,
            "stage": stage,
            "resume_identity_sha256": resume_identity_sha256,
            "admission_sha256": admission_sha256,
            "setup_receipt_sha256": setup_receipt_sha256,
        }
        if stage in {"witness", "prove", "verify"}:
            base["native_prove_budget_sha256"] = native_prove_budget_sha256
        return base

    def current(stage: str) -> tuple[dict[str, object], dict[str, object]] | None:
        document = load_json(session_file, "formal session record")
        all_receipts = document.get("stage_receipts", {})
        dataset_receipts = all_receipts.get(dataset, {}) if isinstance(all_receipts, dict) else None
        anchor = dataset_receipts.get(stage) if isinstance(dataset_receipts, dict) else None
        if (
            not isinstance(anchor, dict)
            or set(anchor) != {"receipt", "sha256"}
            or not isinstance(anchor.get("receipt"), dict)
            or not isinstance(anchor.get("sha256"), str)
            or canonical_json_digest(anchor["receipt"]) != anchor["sha256"]
        ):
            return None
        receipt = anchor["receipt"]
        if any(receipt.get(name) != value for name, value in expected_base(stage).items()):
            return None
        return document, receipt

    def try_reuse(stage: str, inputs: dict[str, Path], outputs: dict[str, Path]) -> float | None:
        value = current(stage)
        if value is None:
            return None
        document, receipt = value
        try:
            input_digests = _stage_paths_digest_map("input", inputs)
            output_digests = _stage_paths_digest_map(stage, outputs)
        except (ContractError, OSError):
            return None
        if receipt.get("inputs_sha256") != input_digests or receipt.get("outputs_sha256") != output_digests:
            return None
        elapsed = receipt.get("elapsed_ms")
        if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed <= 0:
            return None
        if stage == "final":
            snapshot_sha = receipt.get("artifact_manifest_snapshot_sha256")
            snapshot = session_dir / "stage-artifacts" / dataset / "artifact_manifest.final.json"
            if (
                not isinstance(snapshot_sha, str)
                or snapshot.is_symlink()
                or not snapshot.is_file()
                or sha256(snapshot) != snapshot_sha
                or _artifact_manifest_projection_sha256(snapshot) != output_digests.get("artifact_manifest")
            ):
                return None
        return float(elapsed)

    def record(stage: str, inputs: dict[str, Path], outputs: dict[str, Path], elapsed_ms: float) -> None:
        if isinstance(elapsed_ms, bool) or not isinstance(elapsed_ms, (int, float)) or not math.isfinite(elapsed_ms) or elapsed_ms <= 0:
            abort(f"{dataset} {stage} stage produced an invalid wall timing")
        receipt: dict[str, object] = {
            **expected_base(stage),
            "inputs_sha256": _stage_paths_digest_map("input", inputs),
            "outputs_sha256": _stage_paths_digest_map(stage, outputs),
            "elapsed_ms": float(elapsed_ms),
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        if stage == "final":
            snapshot = session_dir / "stage-artifacts" / dataset / "artifact_manifest.final.json"
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            payload = (output / "artifact_manifest.json").read_bytes()
            temporary = snapshot.with_name(f".{snapshot.name}.{os.getpid()}.{time.time_ns()}.tmp")
            flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(temporary, flags, 0o600)
            try:
                with os.fdopen(descriptor, "wb", closefd=False) as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
            finally:
                os.close(descriptor)
            os.replace(temporary, snapshot)
            receipt["artifact_manifest_snapshot_sha256"] = sha256(snapshot)
        document = load_json(session_file, "formal session record")
        all_receipts = document.setdefault("stage_receipts", {})
        if not isinstance(all_receipts, dict):
            abort("formal stage receipt map is malformed")
        dataset_receipts = all_receipts.setdefault(dataset, {})
        if not isinstance(dataset_receipts, dict):
            abort(f"{dataset} stage receipt map is malformed")
        dataset_receipts[stage] = {"receipt": receipt, "sha256": canonical_json_digest(receipt)}
        _write_session_record(session_file, document)

    def invalidate_from(stage: str, reason: str) -> None:
        if stage not in _FORMAL_STAGE_ORDER:
            abort(f"cannot invalidate unknown formal stage: {stage}")
        first = _FORMAL_STAGE_ORDER.index(stage)
        affected = _FORMAL_STAGE_ORDER[first:]
        if stage in {"witness", "prove"} and current("final") is not None:
            restore_final_manifest()
        names = [name for selected in affected for name in _FORMAL_STAGE_OUTPUTS[selected]]
        existing = [output / name for name in dict.fromkeys(names) if (output / name).exists() or (output / name).is_symlink()]
        if existing:
            quarantine = output_root / ".incomplete" / session_id / "quarantine" / f"{dataset}-stages-{time.time_ns()}"
            quarantine.mkdir(parents=True, exist_ok=False)
            for path in existing:
                if path.is_symlink() or not path.is_file():
                    abort(f"refusing to quarantine non-regular {dataset} stage artifact: {path}")
                path.rename(quarantine / path.name)
        document = load_json(session_file, "formal session record")
        all_receipts = document.get("stage_receipts", {})
        dataset_receipts = all_receipts.get(dataset, {}) if isinstance(all_receipts, dict) else None
        if isinstance(dataset_receipts, dict):
            for selected in affected:
                dataset_receipts.pop(selected, None)
        _write_session_record(session_file, document)
        print(f"[ezkl-formal] invalidated {dataset} stages {','.join(affected)}: {reason}", flush=True)

    def restore_final_manifest() -> None:
        value = current("final")
        if value is None:
            return
        receipt = value[1]
        snapshot = session_dir / "stage-artifacts" / dataset / "artifact_manifest.final.json"
        snapshot_sha = receipt.get("artifact_manifest_snapshot_sha256")
        if (
            not isinstance(snapshot_sha, str)
            or snapshot.is_symlink()
            or not snapshot.is_file()
            or sha256(snapshot) != snapshot_sha
            or _artifact_manifest_projection_sha256(snapshot) != receipt.get("outputs_sha256", {}).get("artifact_manifest")
        ):
            abort(f"{dataset} final-stage artifact manifest snapshot failed authentication")
        destination = output / "artifact_manifest.json"
        if destination.is_symlink() or (destination.exists() and not destination.is_file()):
            abort(f"refusing to restore final-stage manifest over a non-regular path: {destination}")
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.{time.time_ns()}.restore")
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(temporary, flags, 0o600)
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as stream:
                stream.write(snapshot.read_bytes())
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            os.close(descriptor)
        os.replace(temporary, destination)

    return {
        "try_reuse": try_reuse,
        "record": record,
        "invalidate_from": invalidate_from,
        "restore_final_manifest": restore_final_manifest,
    }


_FORMAL_FINALIZATION_SCHEMA = "ezkl.formal-finalization-stage-receipt.v1"


def _latest_successful_log_wall_ms(path: Path, exit_field: str, label: str) -> float:
    if path.is_symlink() or not path.is_file():
        abort(f"{label} has no regular stage log: {path}")
    contents = path.read_text(encoding="utf-8", errors="replace")
    matches = list(re.finditer(
        rf"{re.escape(exit_field)}=([0-9]+) wall_ms=([0-9]+(?:\.[0-9]+)?) outcome=(success|failed)",
        contents,
    ))
    if not matches:
        abort(f"{label} log has no durable exit-code/wall-time footer")
    match = matches[-1]
    if match.group(1) != "0" or match.group(3) != "success":
        abort(f"{label} log's latest completed attempt did not exit successfully")
    elapsed = float(match.group(2))
    if not math.isfinite(elapsed) or elapsed <= 0:
        abort(f"{label} log footer has an invalid wall timing")
    return elapsed


def _finalization_result_identity(
    *,
    output_root: Path,
    session_dir: Path,
    session_file: Path,
    session_id: str,
) -> tuple[str, dict[str, object]]:
    """Bind the five independently verified results to their original stages and logs."""
    document = load_json(session_file, "formal session record before finalization")
    if not isinstance(document, dict) or document.get("session_id") != session_id:
        abort("finalization receipts require the exact active formal session")
    completed = document.get("completed_cells")
    admissions = document.get("native_admissions")
    setups = document.get("setup_reuse_receipts")
    all_stage_receipts = document.get("stage_receipts")
    resume_identity = document.get("resume_identity")
    if not all(isinstance(value, dict) for value in (completed, admissions, setups, all_stage_receipts, resume_identity)):
        abort("finalization receipt inputs lack complete session, Admission, setup, or stage anchors")
    resume_digest = canonical_json_digest(resume_identity)
    datasets: list[dict[str, object]] = []
    output_names = {
        "producer": {"statement": "statement.json", "public_output": "public_output.json", "producer_manifest": "producer_manifest.json"},
        "final": {"artifact_manifest": "artifact_manifest.json", "srs": "srs.kzg", "final_log": "logs/final.log"},
        "witness": {"witness": "witness.json", "witness_log": "logs/witness.log"},
        "prove": {"proof": "proof.json", "native_proof_metrics": "native_proof_metrics.json", "native_pk_read_timing": "native_pk_read_timing.json", "gpu_evidence": "gpu_evidence.json", "gpu_prover_record": "gpu_prover_record.json", "prove_log": "logs/prove.log"},
        "verify": {"native_verify_metrics": "native_verify_metrics.json", "core_run_manifest": "core_run_manifest.json", "verify_log": "logs/verify.log"},
    }
    for dataset in DATASETS:
        output = output_root / dataset
        if not _completed_cell_checkpoint(output, dataset, session_id, session_file):
            abort(f"{dataset} finalization inputs are not a durable independently verified result")
        result_path = output / "result.json"
        result = load_json(result_path, f"{dataset} finalization result")
        checkpoint = completed.get(dataset)
        admission = admissions.get(dataset)
        setup_digest = setups.get(dataset)
        dataset_receipts = all_stage_receipts.get(dataset)
        if (
            not isinstance(checkpoint, dict)
            or checkpoint.get("session_id") != session_id
            or checkpoint.get("status") != "VERIFY_OK"
            or checkpoint.get("result_sha256") != sha256(result_path)
            or not isinstance(admission, dict)
            or not isinstance(admission.get("sha256"), str)
            or not isinstance(admission.get("settings_sha256"), str)
            or result.get("settings_sha256") != admission.get("settings_sha256")
            or not isinstance(setup_digest, str)
            or not isinstance(dataset_receipts, dict)
        ):
            abort(f"{dataset} finalization inputs do not match the active VERIFY_OK, Admission, and setup anchors")
        setup_path = output / "setup_receipt.json"
        if setup_path.is_symlink() or not setup_path.is_file() or sha256(setup_path) != setup_digest:
            abort(f"{dataset} finalization setup receipt differs from its session anchor")
        admission_path = session_dir / "admission" / dataset / "admission.json"
        settings_path = session_dir / "admission" / dataset / "settings.json"
        if (
            admission_path.is_symlink() or not admission_path.is_file() or sha256(admission_path) != admission["sha256"]
            or settings_path.is_symlink() or not settings_path.is_file() or sha256(settings_path) != admission["settings_sha256"]
        ):
            abort(f"{dataset} finalization Admission/settings differ from the current session anchors")

        stage_identities: dict[str, object] = {}
        for stage in _FORMAL_STAGE_ORDER:
            anchor = dataset_receipts.get(stage)
            if (
                not isinstance(anchor, dict)
                or set(anchor) != {"receipt", "sha256"}
                or not isinstance(anchor.get("receipt"), dict)
                or not isinstance(anchor.get("sha256"), str)
                or canonical_json_digest(anchor["receipt"]) != anchor["sha256"]
            ):
                abort(f"{dataset} {stage} stage receipt is missing or unauthenticated before result collection")
            receipt = anchor["receipt"]
            elapsed = receipt.get("elapsed_ms")
            inputs = receipt.get("inputs_sha256")
            outputs = receipt.get("outputs_sha256")
            if (
                receipt.get("schema") != "ezkl.formal-stage-receipt.v1"
                or receipt.get("session_id") != session_id
                or receipt.get("dataset") != dataset
                or receipt.get("stage") != stage
                or receipt.get("resume_identity_sha256") != resume_digest
                or receipt.get("admission_sha256") != admission.get("sha256")
                or receipt.get("setup_receipt_sha256") != setup_digest
                or isinstance(elapsed, bool)
                or not isinstance(elapsed, (int, float))
                or not math.isfinite(elapsed)
                or elapsed <= 0
                or not isinstance(inputs, dict)
                or not inputs
                or any(not isinstance(name, str) or not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None for name, value in inputs.items())
                or not isinstance(outputs, dict)
            ):
                abort(f"{dataset} {stage} stage receipt has invalid identity, raw timing, or digest maps")
            expected_paths = {name: output / path for name, path in output_names[stage].items()}
            current_outputs = _stage_paths_digest_map(stage, expected_paths)
            if outputs != current_outputs:
                abort(f"{dataset} {stage} receipt outputs changed before five-result collection")
            log_exit_field = "return_code" if stage == "producer" else "phase_exit_code"
            logged_elapsed = _latest_successful_log_wall_ms(
                output / _FORMAL_FINALIZATION_STAGE_LOGS[stage], log_exit_field, f"{dataset} {stage}"
            )
            if round(logged_elapsed, 6) != round(float(elapsed), 6):
                abort(f"{dataset} {stage} log wall time differs from its session receipt")
            expected_timing = {
                "producer": ("public_statement_production", 6),
                "final": ("final_statement_binding", 6),
                "witness": ("witness_generation", 3),
                "prove": ("prove_command_wall_ms", 6),
                "verify": ("verify", 3),
            }[stage]
            timing_value = result.get("phase_timings_ms", {}).get(expected_timing[0]) if stage != "prove" else result.get(expected_timing[0])
            if (
                isinstance(timing_value, bool)
                or not isinstance(timing_value, (int, float))
                or not math.isfinite(timing_value)
                or round(float(elapsed), expected_timing[1]) != timing_value
            ):
                abort(f"{dataset} {stage} receipt timing differs from the original result timing")
            if stage == "verify" and result.get("independent_verifier_log_sha256") != outputs.get("verify_log"):
                abort(f"{dataset} independent verifier log differs from its successful Verify receipt")
            stage_identities[stage] = {
                "receipt_sha256": anchor["sha256"],
                "elapsed_ms": float(elapsed),
                "inputs_sha256": inputs,
                "outputs_sha256": outputs,
            }

        log_paths = {
            "native_admission": session_dir / "logs" / dataset / "native_admission.log",
            "native_setup": session_dir / "logs" / dataset / "native_setup.log",
            "producer": output / "logs" / "producer.log",
            "final": output / "logs" / "final.log",
            "witness": output / "logs" / "witness.log",
            "prove": output / "logs" / "prove.log",
            "verify": output / "logs" / "verify.log",
        }
        log_digests: dict[str, str] = {}
        for name, path in log_paths.items():
            if path.is_symlink() or not path.is_file():
                abort(f"{dataset} finalization is missing its original {name} stage log")
            log_digests[name] = sha256(path)
        if log_digests["verify"] != result.get("independent_verifier_log_sha256"):
            abort(f"{dataset} finalization Verify log no longer matches the verified result")
        setup_timing = load_json(output / "setup_timing.json", f"{dataset} native setup timing")
        native_setup_wall_ms = _latest_successful_log_wall_ms(
            log_paths["native_setup"], "command_exit_code", f"{dataset} native setup"
        )
        setup_elapsed = setup_timing.get("duration_ms") if isinstance(setup_timing, dict) else None
        reported_setup_elapsed = result.get("phase_timings_ms", {}).get("prepare_compile_setup")
        if (
            isinstance(setup_elapsed, bool) or not isinstance(setup_elapsed, (int, float))
            or not math.isfinite(setup_elapsed) or setup_elapsed <= 0
            or round(float(setup_elapsed), 6) != round(native_setup_wall_ms, 6)
            or round(float(setup_elapsed), 3) != reported_setup_elapsed
        ):
            abort(f"{dataset} native setup log/timing differs from the verified result")
        datasets.append({
            "dataset": dataset,
            "result_sha256": sha256(result_path),
            "proof_sha256": result.get("proof_sha256"),
            "checkpoint_sha256": canonical_json_digest(checkpoint),
            "admission_sha256": admission["sha256"],
            "setup_receipt_sha256": setup_digest,
            "stage_receipts": stage_identities,
            "stage_logs_sha256": log_digests,
            "phase_timings_ms": result.get("phase_timings_ms"),
        })
    identity: dict[str, object] = {
        "schema": "ezkl.formal-five-result-inputs.v1",
        "session_id": session_id,
        "source_tree_sha256": document.get("source_tree_sha256"),
        "resource_manifest_sha256": document.get("resource_manifest_sha256"),
        "resume_identity_sha256": resume_digest,
        "datasets": datasets,
    }
    return canonical_json_digest(identity), identity


def _finalization_receipt_anchor(document: dict[str, object], stage: str) -> dict[str, object] | None:
    all_receipts = document.get("finalization_receipts")
    anchor = all_receipts.get(stage) if isinstance(all_receipts, dict) else None
    if (
        not isinstance(anchor, dict)
        or set(anchor) != {"receipt", "sha256"}
        or not isinstance(anchor.get("receipt"), dict)
        or not isinstance(anchor.get("sha256"), str)
        or canonical_json_digest(anchor["receipt"]) != anchor["sha256"]
    ):
        return None
    return anchor


def _try_reuse_finalization_receipt(
    *,
    session_file: Path,
    session_id: str,
    stage: str,
    inputs_sha256: str,
    matrix: Path,
    log_path: Path,
    status: str,
    collector_receipt_sha256: str | None = None,
) -> tuple[float, str] | None:
    document = load_json(session_file, "formal session record for result-stage reuse")
    anchor = _finalization_receipt_anchor(document, stage)
    if anchor is None:
        return None
    receipt = anchor["receipt"]
    if (
        set(receipt) != {
            "schema", "session_id", "stage", "inputs_sha256", "matrix_path", "matrix_sha256",
            "log_sha256", "status", "elapsed_ms", "collector_receipt_sha256", "recorded_at_utc",
        }
        or receipt.get("schema") != _FORMAL_FINALIZATION_SCHEMA
        or receipt.get("session_id") != session_id
        or receipt.get("stage") != stage
        or receipt.get("inputs_sha256") != inputs_sha256
        or receipt.get("matrix_path") != matrix.relative_to(ROOT).as_posix()
        or receipt.get("status") != status
        or receipt.get("matrix_sha256") != (sha256(matrix) if matrix.is_file() and not matrix.is_symlink() else None)
        or receipt.get("log_sha256") != (sha256(log_path) if log_path.is_file() and not log_path.is_symlink() else None)
        or receipt.get("collector_receipt_sha256") != collector_receipt_sha256
    ):
        return None
    elapsed = receipt.get("elapsed_ms")
    if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed <= 0:
        return None
    try:
        logged_elapsed = _latest_successful_log_wall_ms(log_path, "command_exit_code", stage)
    except (ContractError, OSError):
        return None
    if round(logged_elapsed, 6) != round(float(elapsed), 6):
        return None
    return float(elapsed), str(anchor["sha256"])


def _clear_finalization_receipts(session_file: Path, stages: tuple[str, ...], reason: str) -> None:
    document = load_json(session_file, "formal session record before result-stage invalidation")
    all_receipts = document.setdefault("finalization_receipts", {})
    if not isinstance(all_receipts, dict):
        abort("formal finalization receipt map is malformed")
    changed = False
    for stage in stages:
        changed = all_receipts.pop(stage, None) is not None or changed
    if changed:
        _write_session_record(session_file, document)
    print(f"[ezkl-formal] invalidated finalization receipts {','.join(stages)}: {reason}", flush=True)


def _record_finalization_receipt(
    *,
    session_file: Path,
    session_id: str,
    stage: str,
    inputs_sha256: str,
    matrix: Path,
    log_path: Path,
    status: str,
    elapsed_ms: float,
    collector_receipt_sha256: str | None = None,
) -> str:
    if isinstance(elapsed_ms, bool) or not math.isfinite(elapsed_ms) or elapsed_ms <= 0:
        abort(f"{stage} stage produced an invalid raw wall timing")
    if round(_latest_successful_log_wall_ms(log_path, "command_exit_code", stage), 6) != round(float(elapsed_ms), 6):
        abort(f"{stage} stage log timing does not match its raw runner wall timing")
    _fsync_regular_file(matrix, f"{stage} result matrix")
    _fsync_regular_file(log_path, f"{stage} log")
    receipt: dict[str, object] = {
        "schema": _FORMAL_FINALIZATION_SCHEMA,
        "session_id": session_id,
        "stage": stage,
        "inputs_sha256": inputs_sha256,
        "matrix_path": matrix.relative_to(ROOT).as_posix(),
        "matrix_sha256": sha256(matrix),
        "log_sha256": sha256(log_path),
        "status": status,
        "elapsed_ms": float(elapsed_ms),
        "collector_receipt_sha256": collector_receipt_sha256,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    document = load_json(session_file, "formal session record after result-stage success")
    anchors = document.setdefault("finalization_receipts", {})
    if not isinstance(anchors, dict):
        abort("formal finalization receipt map is malformed")
    anchor_sha256 = canonical_json_digest(receipt)
    anchors[stage] = {"receipt": receipt, "sha256": anchor_sha256}
    _write_session_record(session_file, document)
    return anchor_sha256


def prepare_output_dir(output_root: Path, dataset: str) -> Path:
    output = output_root / dataset
    if output.exists() or output.is_symlink():
        abort(f"fresh formal session output path already exists: {output}")
    output.mkdir(parents=True)
    return output


_SETUP_REUSE_FILES = (
    "setup_timing.json",
    "settings.json",
    "compiled.circuit",
    "pk.key",
    "vk.key",
    "native_keygen_budget.json",
    "native_keygen_receipt.json",
    "gat_static_tables.json",
    "gat_integer_ledger.json",
    "circuit_ref.json",
    "producer_context.json",
)
_POST_SETUP_FILES = (
    "native_prove_budget.json",
    "statement.json",
    "public_output.json",
    "producer_manifest.json",
    "artifact_manifest.json",
    "srs.kzg",
    "witness.json",
    "proof.json",
    "native_proof_metrics.json",
    "native_pk_read_timing.json",
    "native_verify_metrics.json",
    "core_run_manifest.json",
    "gpu_evidence.json",
    "gpu_prover_record.json",
    "result.json",
)


def _load_native_keygen_receipt(
    output: Path,
    dataset: str,
    session_id: str,
    resume_identity: dict[str, object],
    admission: dict[str, object],
) -> tuple[dict[str, object], dict[str, object], str]:
    budget_path = output / "native_keygen_budget.json"
    receipt_path = output / "native_keygen_receipt.json"
    require_file(budget_path, f"{dataset} native keygen budget receipt")
    require_file(receipt_path, f"{dataset} native keygen census receipt")
    budget = load_json(budget_path, f"{dataset} native keygen budget receipt")
    receipt = load_json(receipt_path, f"{dataset} native keygen census receipt")
    if not isinstance(budget, dict) or not isinstance(receipt, dict):
        abort(f"{dataset} native keygen receipts must be JSON objects")
    relation_identity = admission.get("relation_identity")
    if not isinstance(relation_identity, dict):
        abort(f"{dataset} native admission has no authenticated relation identity")
    expected_budget = {
        "schema": "ezkl.native-keygen-budget.v1",
        "session_id": session_id,
        "dataset": dataset,
        "admission_sha256": admission.get("sha256"),
        "settings_sha256": admission.get("settings_sha256"),
        "binary_sha256": resume_identity.get("binary_sha256"),
        "resource_manifest_sha256": relation_identity.get("resource_manifest_sha256"),
        "build_receipt_sha256": resume_identity.get("build_receipt_sha256"),
        "build_environment_sha256": resume_identity.get("build_environment_sha256"),
        "logrows": admission.get("logrows"),
        "domain_size": admission.get("domain_size"),
    }
    for name, value in expected_budget.items():
        if budget.get(name) != value:
            abort(f"{dataset} native keygen budget identity mismatch: {name}")
    for name in (
        "host_available_bytes",
        "host_os_diagnostic_reserve_bytes",
        "setup_keygen_future_host_reserve_bytes",
        "device_workspace_peak_bytes",
        "disk_available_bytes",
        "setup_disk_required_known_bytes",
        "proof_spill_reserve_bytes",
        "disk_required_known_bytes",
        "permutation_successor_backing_file_upper_bytes",
        "native_permutation_backing_budget_bytes",
        "gat_transient_witness_bytes_upper",
    ):
        value = budget.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > sys.maxsize:
            abort(f"{dataset} native keygen budget has an invalid {name}")
    if budget["host_available_bytes"] <= 0 or budget["disk_available_bytes"] <= 0:
        abort(f"{dataset} native keygen budget has no admitted host or disk capacity")
    memory_model = admission.get("native_memory_model")
    admitted_gat_transient = memory_model.get("gat_transient_witness_bytes_upper") if isinstance(memory_model, dict) else None
    if (
        isinstance(admitted_gat_transient, bool)
        or not isinstance(admitted_gat_transient, int)
        or budget["gat_transient_witness_bytes_upper"] != admitted_gat_transient
    ):
        abort(f"{dataset} native keygen GAT workspace budget does not match its admitted memory model")
    if (
        budget["disk_required_known_bytes"] > budget["disk_available_bytes"]
        or budget["setup_disk_required_known_bytes"] != budget["disk_required_known_bytes"]
        or budget["native_permutation_backing_budget_bytes"]
        != budget["disk_available_bytes"] - budget["disk_required_known_bytes"]
        or not isinstance(budget.get("integrated_device"), bool)
    ):
        abort(f"{dataset} native keygen budget exceeds its admitted disk or device envelope")

    census = receipt.get("native_census")
    preflight = census.get("preflight") if isinstance(census, dict) else None
    cs = admission.get("constraint_system")
    if not isinstance(census, dict) or not isinstance(preflight, dict) or not isinstance(cs, dict):
        abort(f"{dataset} native keygen receipt lacks exact census geometry")
    for name, value in (
        ("logrows", admission.get("logrows")),
        ("domain_size", admission.get("domain_size")),
        ("fixed_columns", cs.get("num_fixed_columns")),
        ("advice_columns", cs.get("num_advice_columns")),
        ("instance_columns", cs.get("num_instance_columns")),
        ("permutation_columns", cs.get("num_permutation_columns")),
        ("selector_count", cs.get("num_selectors")),
        ("constraint_degree", cs.get("degree")),
    ):
        if preflight.get(name) != value:
            abort(f"{dataset} mandatory native keygen preflight changed admitted geometry: {name}")
    for name in (
        "compressed_selector_columns",
        "selector_activation_storage_current_bytes",
        "selector_activation_storage_peak_bytes",
        "permutation_copy_calls",
        "permutation_successor_edges",
        "permutation_nondefault_records",
        "permutation_data_pages_allocated",
        "permutation_directory_pages_allocated",
        "permutation_data_allocated_bytes",
        "permutation_directory_allocated_bytes",
        "permutation_cache_bytes",
        "permutation_allocation_bitmap_bytes",
    ):
        value = census.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > sys.maxsize:
            abort(f"{dataset} exact native keygen census has an invalid {name}")
    selector_counts = census.get("selector_activation_counts")
    fixed_counts = census.get("fixed_assignment_callback_counts")
    if (
        not isinstance(selector_counts, list)
        or len(selector_counts) != cs["num_selectors"]
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > admission["domain_size"] for value in selector_counts)
        or not isinstance(fixed_counts, list)
        or len(fixed_counts) != cs["num_fixed_columns"]
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > admission["domain_size"] for value in fixed_counts)
        or census["selector_activation_storage_current_bytes"] > census["selector_activation_storage_peak_bytes"]
        or census["selector_activation_storage_peak_bytes"] > admission["native_memory_model"]["selector_activation_storage_bytes_upper"]
        or census["compressed_selector_columns"] > cs["num_selectors"]
        or census["permutation_copy_calls"] > cs["permutation_copy_call_count_upper"]
        or census["permutation_successor_edges"] > min(
            cs["num_permutation_columns"] * admission["domain_size"],
            2 * census["permutation_copy_calls"],
        )
    ):
        abort(f"{dataset} exact native keygen census exceeds its source-derived shape ceiling")
    actual_backing = (
        census["permutation_data_allocated_bytes"]
        + census["permutation_directory_allocated_bytes"]
        + ((16 * census["permutation_successor_edges"] + 4095) // 4096) * 4096
    )
    source_components = (
        preflight.get("permutation_data_virtual_bytes"),
        preflight.get("permutation_directory_virtual_bytes"),
        budget["permutation_successor_backing_file_upper_bytes"],
    )
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > sys.maxsize for value in source_components):
        abort(f"{dataset} native preflight permutation backing bounds are invalid")
    source_backing_upper = sum(source_components)
    if (
        source_backing_upper > sys.maxsize
        or actual_backing > budget["native_permutation_backing_budget_bytes"]
        or actual_backing > source_backing_upper
    ):
        abort(f"{dataset} exact native permutation backing exceeds its authenticated setup budget")
    stage_caps = receipt.get("stage_budget_caps")
    if not isinstance(stage_caps, list) or len(stage_caps) != 2:
        abort(f"{dataset} native keygen receipt lacks separate VK/PK allocation caps")
    selector_caps: list[int] = []
    permutation_caps: list[int] = []
    selector_initial = preflight.get("selector_activation_initial_backing_bytes")
    if isinstance(selector_initial, bool) or not isinstance(selector_initial, int) or selector_initial < 0 or selector_initial > sys.maxsize:
        abort(f"{dataset} native preflight selector activation backing geometry is invalid")
    for index, stage in enumerate(("vk", "pk")):
        cap = stage_caps[index]
        if not isinstance(cap, dict) or cap.get("stage") != stage:
            abort(f"{dataset} native keygen allocation caps are not ordered VK then PK")
        selector_cap = cap.get("max_selector_activation_bytes")
        permutation_cap = cap.get("max_permutation_backing_bytes")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 or value >= 1 << 64 for value in (selector_cap, permutation_cap)):
            abort(f"{dataset} native {stage.upper()} keygen allocation cap is invalid")
        if selector_cap < selector_initial or selector_cap > budget["host_available_bytes"]:
            abort(f"{dataset} native {stage.upper()} selector cap is outside its shared-pool envelope")
        if permutation_cap > min(budget["native_permutation_backing_budget_bytes"], source_backing_upper):
            abort(f"{dataset} native {stage.upper()} permutation cap exceeds its source/disk envelope")
        selector_caps.append(selector_cap)
        permutation_caps.append(permutation_cap)
    if (
        census["selector_activation_storage_peak_bytes"] > min(selector_caps)
        or actual_backing > min(permutation_caps)
    ):
        abort(f"{dataset} exact keygen census exceeds the stricter of its VK/PK live allocation caps")

    expected_receipt = {
        "schema": "ezkl.native-keygen-receipt.v2",
        "session_id": session_id,
        "dataset": dataset,
        "relation_identity": relation_identity,
        "resource_manifest_sha256": relation_identity.get("resource_manifest_sha256"),
        "budget_receipt_sha256": sha256(budget_path),
        "settings_sha256": admission.get("settings_sha256"),
        "compiled_circuit_sha256": sha256(require_file(output / "compiled.circuit", f"{dataset} compiled circuit")),
        "vk_sha256": sha256(require_file(output / "vk.key", f"{dataset} verifying key")),
        "pk_sha256": sha256(require_file(output / "pk.key", f"{dataset} proving key")),
        "evidence_kind": "mandatory_vk_pk_native_keygen_synthesis_census",
        "coverage": "one_actual_vk_synthesis_and_matching_pk_synthesis; exact_selector_and_permutation_counts",
    }
    for name, value in expected_receipt.items():
        if receipt.get(name) != value:
            abort(f"{dataset} native keygen receipt identity mismatch: {name}")
    return budget, receipt, sha256(receipt_path)


def _setup_reuse_receipt(output: Path, dataset: str, session_id: str, resume_identity: dict[str, object], admission: dict[str, object]) -> dict[str, object]:
    setup_files: dict[str, str] = {}
    for name in _SETUP_REUSE_FILES:
        path = output / name
        require_file(path, f"{dataset} setup reuse artifact {name}")
        setup_files[name] = sha256(path)
    _, native_receipt, native_receipt_sha256 = _load_native_keygen_receipt(
        output, dataset, session_id, resume_identity, admission
    )
    if setup_files["settings.json"] != admission.get("settings_sha256") or setup_files["compiled.circuit"] != native_receipt.get("compiled_circuit_sha256"):
        abort(f"{dataset} setup artifacts do not match the exact native admission")
    return {
        "schema": "ezkl.setup-reuse-receipt.v1",
        "session_id": session_id,
        "dataset": dataset,
        "resume_identity_sha256": canonical_json_digest(resume_identity),
        "admission_sha256": admission.get("sha256"),
        "native_keygen_receipt_sha256": native_receipt_sha256,
        "setup_files": setup_files,
    }


def write_setup_reuse_receipt(
    output: Path, dataset: str, session_id: str, resume_identity: dict[str, object], admission: dict[str, object]
) -> str:
    for name in _SETUP_REUSE_FILES:
        _fsync_regular_file(output / name, f"{dataset} setup artifact {name}")
    receipt = _setup_reuse_receipt(output, dataset, session_id, resume_identity, admission)
    receipt_path = output / "setup_receipt.json"
    _write_session_record(receipt_path, receipt)
    return sha256(receipt_path)


def anchor_setup_reuse_receipt(session_file: Path, dataset: str, receipt_sha256: str) -> None:
    if len(receipt_sha256) != 64 or any(char not in "0123456789abcdef" for char in receipt_sha256):
        abort(f"{dataset} setup receipt digest is not canonical SHA-256")
    document = load_json(session_file, "formal session record")
    if not isinstance(document, dict):
        abort("formal session record is not an object")
    anchors = document.setdefault("setup_reuse_receipts", {})
    if not isinstance(anchors, dict):
        abort("formal session setup receipt anchors are malformed")
    existing = anchors.get(dataset)
    if existing is not None and existing != receipt_sha256:
        abort(f"{dataset} setup receipt anchor cannot be replaced within a session")
    anchors[dataset] = receipt_sha256
    _write_session_record(session_file, document)


def validate_setup_reuse_receipt(
    output: Path,
    dataset: str,
    session_id: str,
    resume_identity: dict[str, object],
    admission: dict[str, object],
    expected_receipt_sha256: object,
) -> bool:
    receipt_path = output / "setup_receipt.json"
    if (
        receipt_path.is_symlink()
        or not receipt_path.is_file()
        or not isinstance(expected_receipt_sha256, str)
        or len(expected_receipt_sha256) != 64
        or any(char not in "0123456789abcdef" for char in expected_receipt_sha256)
    ):
        return False
    try:
        if sha256(receipt_path) != expected_receipt_sha256:
            return False
        existing = load_json(receipt_path, f"{dataset} setup reuse receipt")
        expected = _setup_reuse_receipt(output, dataset, session_id, resume_identity, admission)
    except (ContractError, OSError, json.JSONDecodeError):
        return False
    if not isinstance(existing, dict) or existing != expected:
        return False
    return True


def validate_completed_cell(output: Path, session_id: str) -> bool:
    result_path = output / "result.json"
    if result_path.is_symlink() or not result_path.is_file():
        return False
    try:
        collector = importlib.import_module("collect_results")
        collector.validate_cell(result_path, expected_session_id=session_id)
    except (ContractError, OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        print(f"[ezkl-formal] completed-cell receipt rejected and will be rerun: {exc}", flush=True)
        return False
    return True


def _fsync_regular_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        abort(f"durable formal checkpoint requires a regular {label}: {path}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _completed_cell_checkpoint(output: Path, dataset: str, session_id: str, session_file: Path) -> bool:
    result_path = output / "result.json"
    if output.is_symlink() or not output.is_dir():
        abort(f"completed formal dataset output is not a regular directory: {output}")
    checkpoint_map = load_json(session_file, "formal session record").get("completed_cells", {})
    if not isinstance(checkpoint_map, dict) or any(name not in DATASETS for name in checkpoint_map):
        abort("formal completed-cell checkpoint map is malformed")
    existing = checkpoint_map.get(dataset)
    if existing is not None:
        expected_fields = {"schema", "dataset", "session_id", "status", "result_sha256", "proof_sha256", "checkpointed_at_utc"}
        if (
            not isinstance(existing, dict)
            or set(existing) != expected_fields
            or existing.get("schema") != "ezkl.formal-cell-checkpoint.v1"
            or existing.get("dataset") != dataset
            or existing.get("session_id") != session_id
            or existing.get("status") != "VERIFY_OK"
            or any(not isinstance(existing.get(name), str) or re.fullmatch(r"[0-9a-f]{64}", existing[name]) is None for name in ("result_sha256", "proof_sha256"))
            or not isinstance(existing.get("checkpointed_at_utc"), str)
            or not existing["checkpointed_at_utc"]
        ):
            abort(f"{dataset} durable result checkpoint is malformed")
        if result_path.is_symlink() or not result_path.is_file() or sha256(result_path) != existing["result_sha256"]:
            abort(f"{dataset} result changed after its durable session checkpoint")
    if not validate_completed_cell(output, session_id):
        if existing is not None:
            abort(f"{dataset} durable result checkpoint no longer validates")
        return False
    result = load_json(result_path, f"{dataset} completed formal result")
    proof_path = local_path(result.get("proof_path"), f"{dataset} completed proof")
    proof_sha256 = result.get("proof_sha256")
    if (
        proof_path.is_symlink()
        or not proof_path.is_file()
        or not isinstance(proof_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", proof_sha256) is None
        or sha256(proof_path) != proof_sha256
        or (existing is not None and existing["proof_sha256"] != proof_sha256)
    ):
        abort(f"{dataset} completed result does not bind a regular proof artifact")
    if existing is None:
        _fsync_regular_file(proof_path, "proof artifact")
        _fsync_regular_file(result_path, "result artifact")
        directory_fd = os.open(output, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        checkpoint_map[dataset] = {
            "schema": "ezkl.formal-cell-checkpoint.v1",
            "dataset": dataset,
            "session_id": session_id,
            "status": "VERIFY_OK",
            "result_sha256": sha256(result_path),
            "proof_sha256": proof_sha256,
            "checkpointed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        document = load_json(session_file, "formal session record")
        document["completed_cells"] = checkpoint_map
        _write_session_record(session_file, document)
        print(f"[ezkl-formal] durable VERIFIED result checkpoint: {dataset}", flush=True)
    return True


def quarantine_partial_production(output: Path, output_root: Path, session_id: str, dataset: str) -> None:
    derived = [output / name for name in _POST_SETUP_FILES]
    logs = output / "logs"
    if logs.exists() or logs.is_symlink():
        derived.append(logs)
    existing = [path for path in derived if path.exists() or path.is_symlink()]
    if not existing:
        return
    quarantine_root = output_root / ".incomplete" / session_id / "quarantine" / f"{dataset}-production-{time.time_ns()}"
    quarantine_root.mkdir(parents=True, exist_ok=False)
    for path in existing:
        if path.is_symlink():
            abort(f"refusing to quarantine symlinked partial producer output: {path}")
        path.rename(quarantine_root / path.name)
    print(f"[ezkl-formal] quarantined partial {dataset} producer/final/proof outputs while preserving verified setup", flush=True)


def _set_stage(session_file: Path, stage: str, *, active_dataset: str | None = None) -> None:
    document = load_json(session_file, "formal session record")
    document["status"] = "RUNNING"
    document["stage"] = stage
    document["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    if active_dataset is None:
        document.pop("active_dataset", None)
    else:
        document["active_dataset"] = active_dataset
    _write_session_record(session_file, document)


def _run_dataset_cells(
    datasets: tuple[str, ...],
    completed_datasets: set[str],
    dataset_failures: dict[str, dict[str, object]],
    dataset_failure_history: dict[str, list[dict[str, object]]],
    session_file: Path,
    execute_cell: Callable[[int, str], None],
    log_paths_for_dataset: Callable[[str], list[str]] | None = None,
) -> None:
    stale_failures = completed_datasets.intersection(dataset_failures)
    if stale_failures:
        for dataset in stale_failures:
            dataset_failures.pop(dataset, None)
        document = load_json(session_file, "formal session record")
        document["dataset_failures"] = dataset_failures
        document["dataset_failure_history"] = dataset_failure_history
        _write_session_record(session_file, document)
    for dataset_index, dataset in enumerate(datasets):
        if dataset in completed_datasets:
            print(f"[ezkl-formal] reuse completed {dataset} cell after independent result/artifact validation", flush=True)
            continue
        try:
            execute_cell(dataset_index, dataset)
        except Exception as exc:
            document = load_json(session_file, "formal session record")
            stage = document.get("stage")
            if document.get("active_dataset") != dataset or not isinstance(stage, str):
                stage = f"{dataset}_unknown"
            failure = {
                "stage": stage,
                "exception_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "log_paths": log_paths_for_dataset(dataset) if log_paths_for_dataset is not None else [],
            }
            dataset_failures[dataset] = failure
            dataset_failure_history.setdefault(dataset, []).append(failure)
            document["dataset_failures"] = dataset_failures
            document["dataset_failure_history"] = dataset_failure_history
            _write_session_record(session_file, document)
            print(f"[ezkl-formal] dataset failure recorded dataset={dataset} stage={failure['stage']} type={type(exc).__name__}: {exc}", flush=True)
        else:
            completed_datasets.add(dataset)
            if dataset in dataset_failures:
                dataset_failures.pop(dataset)
                document = load_json(session_file, "formal session record")
                document["dataset_failures"] = dataset_failures
                document["dataset_failure_history"] = dataset_failure_history
                _write_session_record(session_file, document)


def _dataset_failure_log_paths(output_root: Path, session_dir: Path, dataset: str) -> list[str]:
    """Return stable session logs and any existing phase logs for a failed cell."""
    candidates = [
        session_dir / "logs" / dataset / "native_admission.log",
        session_dir / "logs" / dataset / "native_setup.log",
    ]
    output_logs = output_root / dataset / "logs"
    if output_logs.is_dir() and not output_logs.is_symlink():
        for name in ("setup.log", "producer.log", "final.log", "witness.log", "prove.log", "verify.log"):
            candidates.append(output_logs / name)
    paths = [path for path in candidates if path.is_file() and not path.is_symlink()]
    return [path.relative_to(output_root).as_posix() for path in paths]


def _capacity_precheck_api():
    try:
        return importlib.import_module("resource_capacity_precheck")
    except ImportError as exc:
        abort(f"authenticated EZKL resource-capacity precheck is unavailable: {exc}")


def _record_capacity_source_identity(
    *,
    output_root: Path,
    session_dir: Path,
    session_file: Path,
    resume: bool,
    capacity_only: bool,
) -> dict[str, object]:
    api = _capacity_precheck_api()
    identity = api.capacity_source_identity()
    if not isinstance(identity, dict):
        abort("current EZKL capacity source identity is malformed")
    session_document = load_json(session_file, "formal session record")
    previous = session_document.get("capacity_source_identity")
    if resume and previous != identity:
        abort("resume capacity geometry/native-PK/config/resource identity changed or was never authenticated")
    if not resume:
        session_document["capacity_source_identity"] = identity
        _write_session_record(session_file, session_document)

    capacity_dir = session_dir / "capacity"
    if capacity_dir.is_symlink() or (capacity_dir.exists() and not capacity_dir.is_dir()):
        abort(f"formal capacity evidence path is not a regular directory: {capacity_dir}")
    capacity_dir.mkdir(parents=True, exist_ok=True)
    historical: dict[str, object] = {"status": "no_authenticated_historical_session"}
    old_session: Path | None = None
    try:
        old_session = api.latest_authenticated_historical_session(output_root)
    except ContractError as exc:
        historical = {"status": "historical_scan_unavailable", "error": str(exc)[:2000]}
    if old_session is not None:
        report_path = capacity_dir / "historical_resource_capacity_precheck.json"
        try:
            report = api.audit_historical_session(old_session, report_path, output_root=output_root)
            if not isinstance(report, dict):
                abort("historical EZKL capacity report is malformed")
            historical = {
                "status": "authenticated_historical_report",
                "session_path": old_session.relative_to(ROOT).as_posix(),
                "report_path": report_path.relative_to(ROOT).as_posix(),
                "report_sha256": sha256(report_path),
                "historical_source_match": report.get("historical_source_match") is True,
                "source_matched_disk_floor_failure": report.get("source_matched_disk_floor_failure") is True,
            }
        except ContractError as exc:
            historical = {
                "status": "historical_report_unavailable",
                "session_path": old_session.relative_to(ROOT).as_posix(),
                "error": str(exc)[:2000],
            }
    session_document = load_json(session_file, "formal session record")
    session_document["historical_capacity_evidence"] = historical
    _write_session_record(session_file, session_document)
    if (
        not capacity_only
        and not resume
        and historical.get("source_matched_disk_floor_failure") is True
    ):
        abort(
            "authenticated same-source historical capacity evidence shows a fixed-payload disk-floor failure; "
            "stopping before GPU build or any Setup"
        )
    return identity


def _run_five_admission_preflight(
    *,
    binary: Path,
    srs: Path,
    output_root: Path,
    session_id: str,
    session_file: Path,
    authenticated_context: object,
    selected_device: dict[str, object],
    saved_admissions: dict[str, dict[str, object]],
    capacity_only: bool,
) -> dict[str, dict[str, object]]:
    api = _capacity_precheck_api()
    admissions: dict[str, dict[str, object]] = {}
    failures: dict[str, dict[str, str]] = {}
    session_dir = output_root / ".sessions" / session_id
    logs_dir = session_dir / "logs"
    for dataset in DATASETS:
        log_path = logs_dir / dataset / "native_admission.log"
        try:
            selected = native_admit_all_datasets(
                binary,
                srs,
                output_root,
                session_id,
                selected_device,
                datasets=(dataset,),
                reuse_receipt=saved_admissions,
                receipt_callback=lambda name, receipt: _persist_native_admission_receipt(session_file, name, receipt),
                command_runner=lambda command, label, selected_log=log_path: run(
                    command,
                    label,
                    gpu=True,
                    log_path=selected_log,
                ),
            )
            admission = selected[dataset]
            admissions[dataset] = admission
            saved_admissions[dataset] = admission
        except Exception as exc:
            failures[dataset] = {"exception_type": type(exc).__name__, "error": str(exc)[:2000]}
            print(f"[ezkl-formal] Admission attempt failed dataset={dataset}; continuing all-five preflight: {type(exc).__name__}: {exc}", flush=True)

    records: list[dict[str, object]] = []
    audit_failures: dict[str, dict[str, str]] = {}
    for dataset in DATASETS:
        admission = admissions.get(dataset)
        if admission is None:
            records.append(
                api.audit_fresh_admission(
                    session_file,
                    dataset,
                    None,
                    None,
                    output_root=output_root,
                    authenticated_context=authenticated_context,
                )
            )
            continue
        prepared = local_path(str(admission.get("prepared_dir")), f"{dataset} fresh admission directory")
        admission_path = prepared / "admission.json"
        settings_path = prepared / "settings.json"
        try:
            record = api.audit_fresh_admission(
                session_file,
                dataset,
                admission_path,
                settings_path,
                output_root=output_root,
                authenticated_context=authenticated_context,
            )
            records.append(record)
        except Exception as exc:
            audit_failures[dataset] = {"exception_type": type(exc).__name__, "error": str(exc)[:2000]}
            print(f"[ezkl-formal] current capacity audit failed dataset={dataset}; continuing all-five audit: {type(exc).__name__}: {exc}", flush=True)
            records.append(
                api.audit_fresh_admission(
                    session_file,
                    dataset,
                    None,
                    None,
                    output_root=output_root,
                    authenticated_context=authenticated_context,
                )
            )

    capacity_dir = session_dir / "capacity"
    capacity_dir.mkdir(parents=True, exist_ok=True)
    report_path = capacity_dir / "resource_capacity_precheck.json"
    # The helper writes its diagnostic report before raising for any gate or
    # provenance failure. Never let a prior report turn that exception into a
    # Setup authorization.
    report = api.write_and_require_fresh_report(
        session_file,
        records,
        report_path,
        output_root=output_root,
        authenticated_context=authenticated_context,
    )

    session_document = load_json(session_file, "formal session record")
    session_document["capacity_admission_attempts"] = {
        "datasets": [dataset for dataset in DATASETS if dataset in admissions],
        "failures": failures,
        "audit_failures": audit_failures,
        "report_path": report_path.relative_to(ROOT).as_posix(),
        "report_sha256": sha256(report_path),
        "gate_status": report.get("gate_status"),
        "readiness": report.get("readiness", "NOT_ESTABLISHED"),
        "purpose": "all-five-native-admission-and-current-fixed-disk-floor-only; no setup or proof success implied",
    }
    _write_session_record(session_file, session_document)

    failure_items = [f"{dataset} Admission {entry['exception_type']}: {entry['error']}" for dataset, entry in failures.items()]
    failure_items.extend(f"{dataset} capacity audit {entry['exception_type']}: {entry['error']}" for dataset, entry in audit_failures.items())
    gate_status = report.get("gate_status")
    if failure_items or gate_status != "current_disk_floor_pass":
        summary = "; ".join(failure_items) if failure_items else f"capacity gate status={gate_status}"
        if capacity_only:
            print(f"[ezkl-formal] admission/capacity preflight failed before Setup; {summary}", file=sys.stderr, flush=True)
        abort(f"five-dataset admission/capacity preflight did not pass before any Setup: {summary}")
    print(f"[ezkl-formal] all-five current fixed-payload disk-floor preflight PASS; report={report_path.relative_to(ROOT)}", flush=True)
    return admissions


def _load_dataset_failure_state(
    document: dict[str, object], datasets: tuple[str, ...]
) -> tuple[dict[str, dict[str, object]], dict[str, list[dict[str, object]]]]:
    required_fields = ("stage", "exception_type", "error", "timestamp")

    def valid_record(record: object) -> bool:
        if not isinstance(record, dict):
            return False
        if any(not isinstance(record.get(field), str) for field in required_fields):
            return False
        if not record.get("stage") or not record.get("exception_type") or not record.get("timestamp"):
            return False
        if "traceback" in record and not isinstance(record["traceback"], str):
            return False
        log_paths = record.get("log_paths", [])
        return isinstance(log_paths, list) and all(isinstance(path, str) and path for path in log_paths)

    failures_value = document.get("dataset_failures", {})
    if not isinstance(failures_value, dict) or any(
        name not in datasets or not valid_record(record) for name, record in failures_value.items()
    ):
        abort("formal per-dataset failure record map is malformed")
    history_value = document.get("dataset_failure_history", {})
    if not isinstance(history_value, dict) or any(
        name not in datasets
        or not isinstance(records, list)
        or any(not valid_record(record) for record in records)
        for name, records in history_value.items()
    ):
        abort("formal per-dataset failure history is malformed")
    failures = {name: dict(record) for name, record in failures_value.items()}
    history = {name: [dict(record) for record in records] for name, records in history_value.items()}
    return failures, history


def _require_formal_dataset_completion(
    datasets: tuple[str, ...], completed_datasets: set[str], dataset_failures: dict[str, dict[str, object]]
) -> None:
    if dataset_failures:
        failure_summary = "; ".join(f"{dataset} [{record['stage']}] {record['exception_type']}: {record['error']}" for dataset, record in sorted(dataset_failures.items()))
        abort(f"formal dataset cells failed; skipping five-result collection and final validation: {failure_summary}")
    missing = [dataset for dataset in datasets if dataset not in completed_datasets]
    if missing:
        abort(f"formal dataset cells are incomplete; skipping five-result collection and final validation: {', '.join(missing)}")


def _persist_native_admission_receipt(session_file: Path, dataset: str, receipt: dict[str, object]) -> None:
    document = load_json(session_file, "formal session record")
    receipts = document.setdefault("native_admissions", {})
    if not isinstance(receipts, dict) or any(name not in DATASETS for name in receipts):
        abort("formal native admission receipt map is malformed")
    previous = receipts.get(dataset)
    if previous is not None and previous != receipt:
        abort(f"{dataset} native admission receipt cannot be replaced within this session")
    receipts[dataset] = receipt
    _write_session_record(session_file, document)


def _persist_dataset_capacity_profile(session_file: Path, profile: dict[str, object]) -> None:
    document = load_json(session_file, "formal session record")
    current = document.get("local_capacity_profile")
    if current is None:
        current = {
            "schema": profile["schema"],
            "coverage": profile["coverage"],
            "native_cub_workspace_query_sha256": profile["native_cub_workspace_query_sha256"],
            "session_id": profile["session_id"],
            "selected_device_uuid": profile["selected_device_uuid"],
            "device_workspace_peak_bytes": profile["device_workspace_peak_bytes"],
            "integrated_device": profile["integrated_device"],
            "datasets": {},
            "source_srs_copy_size_bytes": profile["source_srs_copy_size_bytes"],
            "pending_finalize_datasets": [],
            "initial_checks": [],
        }
    if not isinstance(current, dict) or current.get("session_id") != profile.get("session_id") or current.get(
        "native_cub_workspace_query_sha256"
    ) != profile.get("native_cub_workspace_query_sha256"):
        abort("formal local capacity profile changed its session or native workspace identity")
    datasets = current.get("datasets")
    initial_checks = current.get("initial_checks")
    if not isinstance(datasets, dict) or not isinstance(initial_checks, list):
        abort("formal local capacity profile is malformed")
    for dataset, requirement in profile["datasets"].items():
        previous = datasets.get(dataset)
        if previous is not None and previous != requirement:
            abort(f"{dataset} native capacity requirement changed within this session")
        datasets[dataset] = requirement
    current["pending_finalize_datasets"] = list(profile["pending_finalize_datasets"])
    current["source_srs_copy_size_bytes"] = profile["source_srs_copy_size_bytes"]
    current["initial_checks"] = [check for check in initial_checks if check.get("dataset") not in profile["datasets"]]
    current["initial_checks"].extend(profile["initial_checks"])
    document["local_capacity_profile"] = current
    document.setdefault("capacity_checks", []).extend(profile["initial_checks"])
    _write_session_record(session_file, document)


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
            f"[ezkl-formal] dataset preflight PASS: {dataset} config, dataset-scoped metadata, " "checkpoint manifest/tensors, graph files, and schema",
            flush=True,
        )
    return checked


def _authenticate_or_prepare_resources(
    python: str, *, allow_prepare: bool, session_id: str, preflight_config_sha256s: tuple[tuple[str, str], ...]
) -> tuple[tuple[tuple[str, tuple[int, int, int, int, int]], ...], tuple[tuple[str, str], ...]]:
    if allow_prepare:
        return _prepare_and_authenticate_resources(python, session_id=session_id, preflight_config_sha256s=tuple(preflight_config_sha256s))
    identities_before = resource_file_identities()
    validate_resources()
    identities_after = resource_file_identities()
    if identities_before != identities_after:
        abort("manifest-listed resource files changed during authentication; current session is invalid")
    return identities_after, tuple(preflight_config_sha256s)


def _authenticate_or_provision_srs(python: str, *, allow_provision: bool) -> dict[str, object]:
    try:
        srs_manifest = validate_native_srs_manifest()
        _validate_config_srs_bindings(_preflight_dataset_configs(), srs_manifest)
        return srs_manifest
    except ContractError as exc:
        print(f"[ezkl-formal] official SRS preflight needs repair: {exc}", flush=True)
        if not allow_provision:
            raise
    provisioner = ROOT / "scripts" / "provision_srs.py"
    require_file(provisioner, "official native SRS provisioner")
    run_prerequisite([python, "-u", str(provisioner)], "provision and authenticate official native get-srs material", timeout_seconds=21600)
    srs_manifest = validate_native_srs_manifest()
    _validate_config_srs_bindings(_preflight_dataset_configs(), srs_manifest)
    return srs_manifest


def _formal_dataset_command_plan(
    binary: Path, srs: Path, output_root: Path, admissions: dict[str, dict[str, object]], datasets: tuple[str, ...] = DATASETS
) -> list[dict[str, object]]:
    """Build the exact baseline-local setup/producer/prove command sequence."""
    plan: list[dict[str, object]] = []
    if not datasets or len(set(datasets)) != len(datasets) or any(dataset not in DATASETS for dataset in datasets):
        abort("formal command-plan dataset selection is empty, duplicated, or unsupported")
    for dataset in datasets:
        output = output_root / dataset
        admission_dir = local_path(str(admissions[dataset]["prepared_dir"]), f"{dataset} admitted circuit directory")
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
            local_cli(admission_dir, "admitted circuit"),
            "--srs-path",
            local_cli(srs, "SRS"),
        ]
        if NATIVE_FORMAL_BACKEND == "gpu":
            setup.append("--gpu")
        producer = [
            "--dataset",
            dataset,
            "--resource-manifest",
            "resources/resource_manifest.json",
            "--setup-context",
            local_cli(output / "producer_context.json", "producer setup context"),
            "--output-dir",
            local_cli(output, "output-dir"),
            "--config",
            local_cli(ROOT / "resources/canonical/configs" / f"{dataset}_secure.cfg", "config"),
            "--statement",
            local_cli(output / "statement.json", "public statement"),
            "--public-output",
            local_cli(output / "public_output.json", "public output"),
            "--producer-manifest",
            local_cli(output / "producer_manifest.json", "producer manifest"),
        ]
        single = [
            "--dataset",
            dataset,
            "--backend",
            NATIVE_FORMAL_BACKEND,
            "--execute-formal",
            "--reuse-setup",
            "--config",
            local_cli(ROOT / "resources/canonical/configs" / f"{dataset}_secure.cfg", "config"),
            "--public-statement",
            local_cli(output / "statement.json", "public statement"),
            "--public-output",
            local_cli(output / "public_output.json", "public output"),
            "--producer-manifest",
            local_cli(output / "producer_manifest.json", "producer manifest"),
            "--binary",
            local_cli(binary, "binary"),
            "--output-dir",
            local_cli(output, "output-dir"),
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
    os.environ["EZKL_FORMAL_SELECTED_CUDA_DEVICE"] = json.dumps(selected_device, sort_keys=True, separators=(",", ":"))
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
        "EZKL_FORMAL_SESSION_ID",
        "EZKL_NATIVE_PREPARE_BUDGET_BYTES",
        "EZKL_NATIVE_PREPARE_DEVICE_HEADROOM_BYTES",
        "EZKL_NATIVE_LOOKUP_CARDINALITY_BOUNDS",
    ):
        os.environ.pop(name, None)
    os.environ["EZKL_FORMAL_SESSION_ID"] = session_id

    _set_stage(session_file, "five_dataset_config_preflight")
    config_values = _preflight_dataset_configs()
    preflight_config_sha256s = tuple((dataset, sha256(ROOT / "resources/canonical/configs" / f"{dataset}_secure.cfg")) for dataset in DATASETS)
    saved_resume_identity = _load_resume_identity(session_file) if args.resume_session is not None else None
    if saved_resume_identity is not None:
        _require_resume_identity_subset(
            saved_resume_identity,
            {
                "source_tree_sha256": source_tree_hash(),
                "resource_manifest_sha256": resource_manifest_hash(),
                "config_sha256s": [[dataset, digest] for dataset, digest in preflight_config_sha256s],
            },
            "pre-resource",
        )
    _set_stage(session_file, "canonical_resource_authentication")
    authenticated_resource_file_identities, preflight_config_sha256s = _authenticate_or_prepare_resources(
        python,
        allow_prepare=not (args.preflight_only or args.capacity_preflight_only) and args.resume_session is None,
        session_id=session_id,
        preflight_config_sha256s=preflight_config_sha256s,
    )
    print("[ezkl-formal] authenticated complete five-dataset canonical resources", flush=True)

    _set_stage(session_file, "official_srs_authentication")
    srs_manifest = _authenticate_or_provision_srs(
        python,
        allow_provision=not (args.preflight_only or args.capacity_preflight_only) and args.resume_session is None,
    )
    entries = srs_manifest["entries"]
    shared_paths = {entry["path"] for entry in entries.values()}
    if len(shared_paths) != 1:
        abort("native SRS manifest does not expose one shared official cache")
    srs = ROOT / next(iter(shared_paths))
    if not srs.is_file() or srs.is_symlink():
        abort(f"shared official SRS is missing or symlinked: {srs}")
    srs_manifest_size = entries[DATASETS[0]]["size"]
    srs_copy_size_bytes = _authenticated_srs_copy_size_bytes(srs, expected_size_bytes=srs_manifest_size)
    print(f"[ezkl-formal] reuse authenticated official SRS: {local_cli(srs, 'SRS')}", flush=True)

    # SRS provisioning may atomically replace secure configs and the canonical
    # resource manifest. Seal only the identities authenticated after that
    # transition; resume uses the same validator and remains read-only.
    post_srs_config_sha256s = tuple(
        (dataset, sha256(ROOT / "resources/canonical/configs" / f"{dataset}_secure.cfg")) for dataset in DATASETS
    )
    authenticated_resource_file_identities, preflight_config_sha256s = _authenticate_current_resources(
        python,
        session_id=session_id,
        preflight_config_sha256s=post_srs_config_sha256s,
    )
    config_values = _preflight_dataset_configs()
    _validate_config_srs_bindings(config_values, srs_manifest)

    session_dir = output_root / ".sessions" / session_id
    _record_capacity_source_identity(
        output_root=output_root,
        session_dir=session_dir,
        session_file=session_file,
        resume=args.resume_session is not None,
        capacity_only=args.capacity_preflight_only,
    )

    _set_stage(session_file, "gpu_runtime_identity")
    gpu_identity: dict[str, object] | None = None
    if NATIVE_FORMAL_BACKEND == "gpu":
        try:
            gpu_identity = validate_gpu_environment()
            print("[ezkl-formal] native ICICLE/GPU runtime identity PASS", flush=True)
        except ContractError as exc:
            if args.resume_session is not None:
                abort(f"resume requires the already-authenticated GPU runtime and cannot provision it: {exc}")
            if args.preflight_only:
                print(f"[ezkl-formal] GPU_SETUP_REQUIRED: {exc}", flush=True)
            elif args.capacity_preflight_only:
                abort(f"capacity preflight requires the already-installed authenticated GPU runtime: {exc}")
            else:
                run_prerequisite(
                    [
                        "bash",
                        "setup-gpu.sh",
                        "--yes",
                        "--python",
                        python,
                        "--nvcc",
                        runtime_identity["nvcc_path"],
                        "--cmake",
                        runtime_identity["cmake_path"],
                        "--make",
                        runtime_identity["make_path"],
                        "--cc",
                        runtime_identity["cc_path"],
                        "--cxx",
                        runtime_identity["cxx_path"],
                        "--nvidia-smi",
                        runtime_identity["nvidia_smi_path"],
                        "--selected-device-json",
                        json.dumps(selected_device, sort_keys=True, separators=(",", ":")),
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

    if args.capacity_preflight_only and not build_reusable:
        abort("capacity-only preflight requires the existing source/resource/runtime-matched GPU build; rebuilding is forbidden in this mode")

    if not build_reusable:
        if args.resume_session is not None:
            abort("resume requires the exact prior source/resource/runtime-matched build; rebuilding is not allowed")
        if binary != ROOT / "target" / NATIVE_FORMAL_BACKEND / "release/ezkl":
            abort(f"stale custom --binary cannot be rebuilt; use target/{NATIVE_FORMAL_BACKEND}/release/ezkl")
        build_args = [
            "bash",
            local_cli(build, "build script"),
            f"--{NATIVE_FORMAL_BACKEND}",
            "--python",
            python,
            "--cargo",
            str(runtime_identity["cargo_path"]),
            "--rustc",
            str(runtime_identity["rustc_path"]),
            "--rustdoc",
            str(runtime_identity["rustdoc_path"]),
            "--cmake",
            str(runtime_identity["cmake_path"]),
            "--make",
            str(runtime_identity["make_path"]),
            "--nvcc",
            str(runtime_identity["nvcc_path"]),
            "--cc",
            str(runtime_identity["cc_path"]),
            "--cxx",
            str(runtime_identity["cxx_path"]),
            "--nvidia-smi",
            str(runtime_identity["nvidia_smi_path"]),
            "--selected-device-json",
            json.dumps(selected_device, sort_keys=True, separators=(",", ":")),
            "--build-environment-json",
            json.dumps(build_environment, sort_keys=True, separators=(",", ":")),
        ]
        run(build_args, "build source-matched official ezkl native binary", log_path=_build_log_path(session_id))
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

    resume_identity = _formal_resume_identity(
        runtime_identity=runtime_identity,
        selected_device=selected_device,
        build_environment=build_environment,
        gpu_identity=gpu_identity,
        preflight_config_sha256s=preflight_config_sha256s,
        srs=srs,
        binary=binary,
    )
    if saved_resume_identity is not None and saved_resume_identity != resume_identity:
        changed = sorted(key for key in set(saved_resume_identity) | set(resume_identity) if saved_resume_identity.get(key) != resume_identity.get(key))
        abort(f"resume immutable source/build/resource/config/SRS/GPU identity changed: {', '.join(changed)}")
    session_document = json.loads(session_file.read_text())
    session_document["resume_identity"] = resume_identity
    _write_session_record(session_file, session_document)

    session_document = load_json(session_file, "formal session record")
    saved_admissions = session_document.get("native_admissions", {})
    if not isinstance(saved_admissions, dict) or any(name not in DATASETS or not isinstance(value, dict) for name, value in saved_admissions.items()):
        abort("formal native admission receipt map is malformed")
    saved_anchors = session_document.get("setup_reuse_receipts", {})
    if not isinstance(saved_anchors, dict) or any(name not in DATASETS for name in saved_anchors):
        abort("formal setup receipt anchor map is malformed")
    checkpoint_map = session_document.get("completed_cells", {})
    if not isinstance(checkpoint_map, dict) or any(name not in DATASETS for name in checkpoint_map):
        abort("formal completed-cell checkpoint map is malformed")
    dataset_failures, dataset_failure_history = _load_dataset_failure_state(session_document, DATASETS)
    archived_value = session_document.get("outputs_archived_for_session", False)
    if not isinstance(archived_value, bool):
        abort("formal output archive state is malformed")
    completed_datasets: set[str] = set()
    if args.resume_session is not None and archived_value:
        for dataset in DATASETS:
            output = output_root / dataset
            if output.is_symlink():
                abort(f"resume output directory must not be a symlink: {output}")
            if dataset in checkpoint_map and not output.is_dir():
                abort(f"{dataset} durable result checkpoint has no active dataset directory")
            if output.is_dir() and _completed_cell_checkpoint(output, dataset, session_id, session_file):
                completed_datasets.add(dataset)
    _set_stage(session_file, "five_dataset_native_admission_preflight")
    admissions = _run_five_admission_preflight(
        binary=binary,
        srs=srs,
        output_root=output_root,
        session_id=session_id,
        session_file=session_file,
        authenticated_context=context,
        selected_device=selected_device,
        saved_admissions=saved_admissions,
        capacity_only=args.capacity_preflight_only,
    )
    if args.capacity_preflight_only:
        session_document = load_json(session_file, "formal session record")
        session_document["capacity_preflight_only"] = {
            "status": "ADMISSION_CAPACITY_ONLY",
            "setup_started": False,
            "proof_started": False,
            "result_status": "NOT_A_FORMAL_RESULT",
        }
        _write_session_record(session_file, session_document)
        _set_stage(session_file, "capacity_admission_preflight_complete")
        print("[ezkl-formal] capacity-only preflight complete; Setup/Keygen/Witness/Prove/Verify were not run", flush=True)
        return 0

    session_document = load_json(session_file, "formal session record")
    workspace_receipt_value = session_document.get("native_cub_workspace_receipt")
    if workspace_receipt_value is not None and not isinstance(workspace_receipt_value, dict):
        abort("saved native CUB workspace receipt is malformed")
    workspace_receipt = workspace_receipt_value if isinstance(workspace_receipt_value, dict) else None
    if workspace_receipt is not None:
        workspace_receipt = _validate_native_cub_workspace_receipt(workspace_receipt, session_dir, selected_device, gpu_identity, session_id)
    session_document["resource_manifest_sha256"] = resource_manifest_hash()
    session_document["srs_manifest_sha256"] = sha256(SRS_MANIFEST)
    session_document["source_tree_sha256"] = source_tree_hash()
    session_document["binary_sha256"] = sha256(binary)
    _write_session_record(session_file, session_document)

    def execute_cell(dataset_index: int, dataset: str) -> None:
        nonlocal workspace_receipt
        dataset_log_dir = session_dir / "logs" / dataset
        output = output_root / dataset
        admission = admissions[dataset]
        session_document = load_json(session_file, "formal session record")
        archived_value = session_document.get("outputs_archived_for_session", False)
        if not isinstance(archived_value, bool):
            abort("formal output archive state is malformed")
        setup_reused = False
        if archived_value and (output.is_symlink() or (output.exists() and not output.is_dir())):
            abort(f"resume output directory must be a regular directory: {output}")
        if args.resume_session is not None and archived_value and output.exists():
            setup_reused = validate_setup_reuse_receipt(
                output,
                dataset,
                session_id,
                resume_identity,
                admission,
                saved_anchors.get(dataset),
            )
        _set_stage(session_file, f"{dataset}_capacity_admission", active_dataset=dataset)
        if workspace_receipt is None:
            capacity_dir = session_dir / "capacity"
            if capacity_dir.exists() or capacity_dir.is_symlink():
                _quarantine_session_path(capacity_dir, output_root, session_id, "capacity")
        cub_receipt, capacity_requirements, capacity_profile = _admit_local_capacity(
            {dataset: admission},
            output_root=output_root,
            session_dir=session_dir,
            session_id=session_id,
            runtime_identity=runtime_identity,
            selected_device=selected_device,
            build_environment=build_environment,
            gpu_identity=gpu_identity,
            srs_copy_size_bytes=srs_copy_size_bytes,
            active_datasets={dataset},
            setup_required_datasets=set() if setup_reused else {dataset},
            pending_finalize_datasets={dataset},
            workspace_receipt=workspace_receipt,
        )
        workspace_receipt = _validate_native_cub_workspace_receipt(cub_receipt, session_dir, selected_device, gpu_identity, session_id)
        session_document = load_json(session_file, "formal session record")
        session_document["native_cub_workspace_receipt"] = workspace_receipt
        _write_session_record(session_file, session_document)
        _persist_dataset_capacity_profile(session_file, capacity_profile)
        session_document = load_json(session_file, "formal session record")
        archived_value = session_document.get("outputs_archived_for_session", False)
        if not archived_value:
            archive_prior_outputs(output_root, matrix, session_id, session_file)
        output = output_root / dataset
        if output.is_symlink():
            abort(f"formal output directory must not be a symlink: {output}")
        if output.exists() and not output.is_dir():
            abort(f"formal output path is not a regular directory: {output}")
        if output.is_dir():
            if not setup_reused:
                _quarantine_session_path(output, output_root, session_id, f"{dataset}-partial-cell")
        item = _formal_dataset_command_plan(binary, srs, output_root, admissions, datasets=(dataset,))[0]
        if not setup_reused:
            setup_disk_required = _remaining_setup_disk_bytes(
                capacity_requirements,
                dataset_index,
                include_current_transient=True,
                setup_required_datasets={dataset},
            )
            setup_key_bytes = _remaining_setup_disk_bytes(
                capacity_requirements,
                dataset_index,
                include_current_transient=False,
                setup_required_datasets={dataset},
            )
            setup_disk_components = _disk_budget_components(
                pending_srs=_remaining_srs_copy_disk_budget(srs_copy_size_bytes, dataset_index, pending_finalize_datasets={dataset}),
                setup_key_artifact_upper_bytes=setup_key_bytes,
                setup_mapped_peak_bytes=setup_disk_required - setup_key_bytes,
            )
            setup_check = _recheck_local_phase_capacity(
                dataset,
                capacity_requirements[dataset],
                output_root=output_root,
                remaining_known_disk_bytes=int(setup_disk_components["total_required_bytes"]),
                phase="setup",
                receipt=workspace_receipt,
                session_dir=session_dir,
                session_id=session_id,
                selected_device=selected_device,
                gpu_identity=gpu_identity,
                build_environment=build_environment,
                disk_budget_components=setup_disk_components,
            )
            session_document = load_json(session_file, "formal session record")
            session_document.setdefault("capacity_checks", []).append(setup_check)
            _write_session_record(session_file, session_document)
            output = prepare_output_dir(output_root, dataset)
            native_keygen_budget_sha256 = _write_native_keygen_budget_receipt(
                output,
                dataset=dataset,
                session_id=session_id,
                admission=admission,
                resume_identity=resume_identity,
                capacity=capacity_requirements[dataset],
                setup_check=setup_check,
                workspace_query=workspace_receipt["query"],
            )
            _set_stage(session_file, f"{dataset}_setup", active_dataset=dataset)
            setup_ms = run(
                item["setup"],
                f"{dataset} native setup",
                gpu=NATIVE_FORMAL_BACKEND == "gpu",
                log_path=dataset_log_dir / "native_setup.log",
            )
            (output / "setup_timing.json").write_text(json.dumps({"duration_ms": setup_ms}) + "\n")
            setup_receipt_sha256 = write_setup_reuse_receipt(output, dataset, session_id, resume_identity, admission)
            anchor_setup_reuse_receipt(session_file, dataset, setup_receipt_sha256)
        else:
            print(f"[ezkl-formal] reuse validated {dataset} native setup receipt and keys", flush=True)
            setup_receipt_sha256 = sha256(output / "setup_receipt.json")
        _native_keygen_budget, native_keygen_receipt, native_keygen_receipt_sha256 = _load_native_keygen_receipt(
            output, dataset, session_id, resume_identity, admission
        )
        native_census = native_keygen_receipt.get("native_census")
        if not isinstance(native_census, dict):
            abort(f"{dataset} authenticated setup lacks the mandatory native keygen census")
        capacity_requirements[dataset] = _native_capacity_requirements(
            admission,
            workspace_receipt["query"]["native_type_sizes"],
            native_census=native_census,
        )
        if (
            capacity_requirements[dataset].get("gat_transient_witness_bytes_upper")
            != _native_keygen_budget.get("gat_transient_witness_bytes_upper")
        ):
            abort(f"{dataset} authenticated keygen budget and recomputed capacity disagree on the GAT live-tensor reserve")
        capacity_requirements[dataset]["host_known_bytes"] = capacity_requirements[dataset]["proof_host_peak_bytes"]
        session_document = load_json(session_file, "formal session record")
        native_capacity = session_document.setdefault("native_keygen_capacity", {})
        if not isinstance(native_capacity, dict):
            abort("formal native keygen capacity evidence map is malformed")
        native_capacity[dataset] = {
            "native_keygen_budget_sha256": sha256(output / "native_keygen_budget.json"),
            "native_keygen_receipt_sha256": native_keygen_receipt_sha256,
            "coverage": "exact mandatory native VK/PK synthesis census; source-derived post-keygen capacity envelope",
            "requirements": capacity_requirements[dataset],
        }
        _write_session_record(session_file, session_document)
        proof_disk_components = _disk_budget_components(
            pending_srs=_remaining_srs_copy_disk_budget(srs_copy_size_bytes, dataset_index, pending_finalize_datasets={dataset}),
            proof_spill_known_bytes=capacity_requirements[dataset]["proof_spill_known_bytes"],
        )
        prove_check = _recheck_local_phase_capacity(
            dataset,
            capacity_requirements[dataset],
            output_root=output_root,
            remaining_known_disk_bytes=int(proof_disk_components["total_required_bytes"]),
            phase="witness_prove_verify",
            receipt=workspace_receipt,
            session_dir=session_dir,
            session_id=session_id,
            selected_device=selected_device,
            gpu_identity=gpu_identity,
            build_environment=build_environment,
            disk_budget_components=proof_disk_components,
        )
        native_prove_budget_sha256 = _load_reusable_native_prove_budget_receipt(
            output,
            dataset=dataset,
            session_id=session_id,
            admission=admission,
            resume_identity=resume_identity,
            setup_receipt_sha256=setup_receipt_sha256,
            native_keygen_budget_sha256=sha256(output / "native_keygen_budget.json"),
            native_keygen_receipt_sha256=native_keygen_receipt_sha256,
            capacity=capacity_requirements[dataset],
            workspace_query=workspace_receipt["query"],
        )
        if native_prove_budget_sha256 is None:
            native_prove_budget_path = output / "native_prove_budget.json"
            if native_prove_budget_path.exists() or native_prove_budget_path.is_symlink():
                _quarantine_session_path(native_prove_budget_path, output_root, session_id, f"{dataset}-stale-native-prove-budget")
            native_prove_budget_sha256 = _write_native_prove_budget_receipt(
                output,
                dataset=dataset,
                session_id=session_id,
                admission=admission,
                resume_identity=resume_identity,
                setup_receipt_sha256=setup_receipt_sha256,
                native_keygen_budget_sha256=sha256(output / "native_keygen_budget.json"),
                native_keygen_receipt_sha256=native_keygen_receipt_sha256,
                capacity=capacity_requirements[dataset],
                prove_check=prove_check,
                workspace_query=workspace_receipt["query"],
            )
        prove_check["native_prove_budget_sha256"] = native_prove_budget_sha256
        prepare_budget = _native_prepare_budget(
            capacity_requirements[dataset], initial_available_bytes=int(prove_check["host_available_bytes"]), workspace_query=workspace_receipt["query"]
        )
        prove_check["native_prepare_budget"] = prepare_budget
        session_document = load_json(session_file, "formal session record")
        session_document.setdefault("capacity_checks", []).append(prove_check)
        _write_session_record(session_file, session_document)
        stage_receipt_hooks = _stage_receipt_hooks(
            session_file=session_file,
            session_dir=session_dir,
            output_root=output_root,
            output=output,
            dataset=dataset,
            session_id=session_id,
            resume_identity=resume_identity,
            admission_sha256=str(admission.get("sha256")),
            setup_receipt_sha256=setup_receipt_sha256,
            native_prove_budget_sha256=native_prove_budget_sha256,
        )
        producer_inputs = {
            "setup_context": output / "producer_context.json",
            "settings": output / "settings.json",
            "secure_config": ROOT / "resources/canonical/configs" / f"{dataset}_secure.cfg",
            "resource_manifest": ROOT / "resources/resource_manifest.json",
            "native_admission": local_path(str(admission["path"]), f"{dataset} authenticated native admission"),
        }
        producer_outputs = {
            "statement": output / "statement.json",
            "public_output": output / "public_output.json",
            "producer_manifest": output / "producer_manifest.json",
        }
        producer_ms = stage_receipt_hooks["try_reuse"]("producer", producer_inputs, producer_outputs)
        if producer_ms is None:
            stage_receipt_hooks["invalidate_from"]("producer", "producer receipt or its input/output identities are missing or stale")
            producer_ms = invoke_python_entrypoint(
                "produce_public_statement",
                item["producer"],
                f"{dataset} independent public statement/Y producer",
                context=context,
                session_id=session_id,
                log_path=output / "logs" / "producer.log",
            )
        else:
            print(f"[ezkl-formal] reuse authenticated {dataset} public statement producer ({float(producer_ms):.3f} ms original wall)", flush=True)
        stage_receipt_hooks["record"]("producer", producer_inputs, producer_outputs, float(producer_ms))
        cardinality_bounds = admission["native_memory_model"]["lookup_table_cardinality_upper_bounds"]
        taskcap_env, device_headroom_env, cardinality_env = _validated_native_prepare_exports(
            prepare_budget,
            cardinality_bounds,
            domain_size=int(admission["domain_size"]),
            expected_lookup_groups=int(admission["constraint_system"]["num_unique_lookup_table_expression_groups"]),
        )
        os.environ["EZKL_NATIVE_PREPARE_BUDGET_BYTES"] = taskcap_env
        os.environ["EZKL_NATIVE_PREPARE_DEVICE_HEADROOM_BYTES"] = device_headroom_env
        os.environ["EZKL_NATIVE_LOOKUP_CARDINALITY_BOUNDS"] = cardinality_env
        try:
            _set_stage(session_file, f"{dataset}_witness_prove_independent_verify", active_dataset=dataset)
            invoke_python_entrypoint(
                "run_single_dataset",
                item["single"],
                f"{dataset} final/witness/prove/independent verify",
                context=context,
                session_id=session_id,
                stage_receipt_hooks=stage_receipt_hooks,
                producer_wall_ms=float(producer_ms),
            )
        finally:
            for name in ("EZKL_NATIVE_PREPARE_BUDGET_BYTES", "EZKL_NATIVE_PREPARE_DEVICE_HEADROOM_BYTES", "EZKL_NATIVE_LOOKUP_CARDINALITY_BOUNDS"):
                os.environ.pop(name, None)
        if not _completed_cell_checkpoint(output, dataset, session_id, session_file):
            abort(f"{dataset} did not produce an independently verified formal result")

    _run_dataset_cells(
        DATASETS,
        completed_datasets,
        dataset_failures,
        dataset_failure_history,
        session_file,
        execute_cell,
        log_paths_for_dataset=lambda dataset: _dataset_failure_log_paths(output_root, session_dir, dataset),
    )
    _require_formal_dataset_completion(DATASETS, completed_datasets, dataset_failures)

    final_document = load_json(session_file, "formal session record")
    final_checkpoints = final_document.get("completed_cells", {})
    if not isinstance(final_checkpoints, dict):
        abort("formal completed-cell checkpoint map is malformed before collection")
    for dataset in DATASETS:
        output = output_root / dataset
        result_path = output / "result.json"
        checkpoint = final_checkpoints.get(dataset)
        if (
            output.is_symlink()
            or not output.is_dir()
            or result_path.is_symlink()
            or not result_path.is_file()
            or not isinstance(checkpoint, dict)
            or checkpoint.get("session_id") != session_id
            or checkpoint.get("status") != "VERIFY_OK"
            or checkpoint.get("result_sha256") != sha256(result_path)
        ):
            abort(f"formal run has no independently verified durable result for {dataset}")

    result_inputs_sha256, _ = _finalization_result_identity(
        output_root=output_root,
        session_dir=session_dir,
        session_file=session_file,
        session_id=session_id,
    )
    collect_log = session_dir / "logs" / "collect.log"
    collect_reuse = _try_reuse_finalization_receipt(
        session_file=session_file,
        session_id=session_id,
        stage="collector",
        inputs_sha256=result_inputs_sha256,
        matrix=matrix,
        log_path=collect_log,
        status="COLLECTED_PENDING_FINAL_VALIDATION",
    )
    if collect_reuse is None:
        _clear_finalization_receipts(session_file, ("collector", "final_validator"), "collector receipt or one of its result/matrix/log identities is missing or stale")
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
        collector_ms = run(
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
            log_path=collect_log,
        )
        collector_receipt_sha256 = _record_finalization_receipt(
            session_file=session_file,
            session_id=session_id,
            stage="collector",
            inputs_sha256=result_inputs_sha256,
            matrix=matrix,
            log_path=collect_log,
            status="COLLECTED_PENDING_FINAL_VALIDATION",
            elapsed_ms=collector_ms,
        )
    else:
        collector_ms, collector_receipt_sha256 = collect_reuse
        print(f"[ezkl-formal] reuse authenticated five-result collection ({collector_ms:.3f} ms original wall)", flush=True)

    validator_inputs_sha256 = canonical_json_digest({
        "schema": "ezkl.formal-final-validator-inputs.v1",
        "five_result_inputs_sha256": result_inputs_sha256,
        "matrix_sha256": sha256(matrix),
        "collector_receipt_sha256": collector_receipt_sha256,
    })
    validate_log = session_dir / "logs" / "validate.log"
    validator_reuse = _try_reuse_finalization_receipt(
        session_file=session_file,
        session_id=session_id,
        stage="final_validator",
        inputs_sha256=validator_inputs_sha256,
        matrix=matrix,
        log_path=validate_log,
        status="FINAL_VALIDATION_OK",
        collector_receipt_sha256=collector_receipt_sha256,
    )
    if validator_reuse is None:
        _clear_finalization_receipts(session_file, ("final_validator",), "final-validator receipt or its verified collection/matrix/log identities is missing or stale")
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
        validation_ms = run(
            [python, local_cli(validate, "result validator"), "--matrix", local_cli(matrix, "matrix"), "--session-id", session_id],
            "validate five-dataset result matrix",
            log_path=validate_log,
        )
        _record_finalization_receipt(
            session_file=session_file,
            session_id=session_id,
            stage="final_validator",
            inputs_sha256=validator_inputs_sha256,
            matrix=matrix,
            log_path=validate_log,
            status="FINAL_VALIDATION_OK",
            elapsed_ms=validation_ms,
            collector_receipt_sha256=collector_receipt_sha256,
        )
    else:
        validation_ms, _ = validator_reuse
        print(f"[ezkl-formal] reuse authenticated final validation ({validation_ms:.3f} ms original wall)", flush=True)
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
    normalize_formal_execution_environment()
    if args.capacity_preflight_only and (args.preflight_only or args.resume_session is not None):
        abort("--capacity-preflight-only cannot be combined with --preflight-only or --resume-session")
    if args.resume_session is not None:
        if args.preflight_only:
            abort("--resume-session cannot be combined with --preflight-only")
        if FORMAL_SESSION_ID.fullmatch(args.resume_session) is None:
            abort("--resume-session must be one exact 32-character lowercase hexadecimal session id")
    output_root = formal_local_path(args.output_root, "output-root")
    matrix = formal_local_path(args.matrix, "matrix")
    binary_value = args.binary or f"target/{NATIVE_FORMAL_BACKEND}/release/ezkl"
    binary = formal_local_path(binary_value, "binary")
    args.output_root = output_root.relative_to(ROOT).as_posix()
    args.matrix = matrix.relative_to(ROOT).as_posix()
    args.binary = binary.relative_to(ROOT).as_posix()
    check_output_paths(output_root, matrix)
    session_id, session_file, lock_fd = begin_formal_session(output_root, matrix, args.resume_session, args._session_id)
    try:
        guardian = ProcessGuardian.start(root=ROOT, session_id=session_id, session_file=session_file, lock_fd=lock_fd)
    except BaseException as exc:
        try:
            update_session_record(session_file, status="FAILED", stage="guardian_start", error=f"{type(exc).__name__}: {exc}")
        except BaseException as record_exc:
            print(f"[ezkl-formal] could not record guardian startup failure: {record_exc}", file=sys.stderr, flush=True)
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
        raise
    try:
        result = _run_session(args, session_id=session_id, session_file=session_file)
        guardian.shutdown()
        if args.capacity_preflight_only:
            status, stage = "CAPACITY_PREFLIGHT_COMPLETE", "capacity_admission_preflight_complete"
        elif args.preflight_only:
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
                status = "INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "FAILED"
                update_session_record(
                    session_file, status=status, stage=str(json.loads(session_file.read_text()).get("stage", "failed")), error=f"{type(exc).__name__}: {exc}"
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
