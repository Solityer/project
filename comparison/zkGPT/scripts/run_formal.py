#!/usr/bin/env python3
"""Own the lock, session, and bounded stage process groups for formal zkGPT."""

from __future__ import annotations

import argparse
import ctypes
import fcntl
import hashlib
import json
import os
import re
import secrets
import signal
import stat
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from common import (
    ChainError,
    DATASETS,
    canonical_dataset_scope,
    atomic_write_json,
    atomic_write_text,
    baseline_root,
    binary_fingerprint,
    create_formal_session,
    ensure_directory,
    formal_final_path,
    formal_preflight_path,
    formal_run_lock,
    formal_session_status_path,
    formal_child_environment,
    load_json,
    install_process_signal_cleanup,
    local_path,
    relative_path,
    run_logged,
    run_directory,
    runtime_source_sha256,
    sha256_file,
    source_tree_sha256,
    validate_formal_cell,
    verify_build_provenance,
    build_binary,
    verifier_binary,
    require_formal_session,
    require_formal_resource_manifest,
)

from collect_results import _validate_preflight
from gat_zkgpt_experiment import run_cell
from validate_final_results import _reuse_owner_approved_cell, _validate_final_results

LOCAL_CAPACITY_PROFILE_SCHEMA = "zkgpt-local-capacity-profile-v2"
LOCAL_OWNER_SCHEMA = "zkgpt-local-formal-owner-v1"
CELL_INTERNAL_LOG_RESERVE_BYTES = 16 * 1024 * 1024
FINALIZATION_INTERNAL_DISK_RESERVE_BYTES = 64 * 1024 * 1024
HOST_ACCOUNTING_OVERHEAD_BYTES = 64 * 1024 * 1024
SESSION_ID_RE = re.compile(r"^[0-9a-f]{32}$")
OWNER_ENV_PREFIX = "ZKGPT_LOCAL_OWNER_"
OWNER_APPROVED_CELLS_SCHEMA = "zkgpt-owner-approved-cells-v1"
OWNER_APPROVED_CELLS_LIMIT_BYTES = 1024 * 1024
_OWNER_LOCK_FD_ENV = OWNER_ENV_PREFIX + "LOCK_FD"
_OWNER_TOKEN_ENV = OWNER_ENV_PREFIX + "TOKEN"
_OWNER_PID_ENV = OWNER_ENV_PREFIX + "PID"
_OWNER_START_ENV = OWNER_ENV_PREFIX + "START_TICKS"
_OWNER_STAGE_ENV = OWNER_ENV_PREFIX + "STAGE"
_OWNER_DATASET_ENV = OWNER_ENV_PREFIX + "DATASET"
_OWNER_STAGE_NONCE_ENV = OWNER_ENV_PREFIX + "STAGE_NONCE"
_OWNER_SESSION_ENV = OWNER_ENV_PREFIX + "SESSION_ID"
_OWNER_DATASET_SCOPE_ENV = OWNER_ENV_PREFIX + "DATASET_SCOPE"


def _write_status(
    root: Path, session_id: str, status: str, active_stage: str, completed: Sequence[str], stage_status: Dict[str, str], started_unix_ns: int, failure: str = ""
) -> None:
    atomic_write_json(
        formal_session_status_path(root, session_id),
        {
            "schema": "zkgpt-gat-formal-session-v1",
            "session_id": session_id,
            "status": status,
            "active_stage": active_stage,
            "completed_stages": list(completed),
            "stage_status": dict(stage_status),
            "started_unix_ns": started_unix_ns,
            "updated_unix_ns": time.time_ns(),
            "failure": failure,
        },
        root,
    )


def _session_dir(root: Path, value: str, *, allow_empty: bool) -> tuple[str, Path]:
    if not value:
        raise ChainError("internal formal phase requires --session-dir")
    supplied = Path(value)
    if not supplied.is_absolute():
        raise ChainError("internal formal --session-dir must be absolute")
    session_id = supplied.name
    if not SESSION_ID_RE.fullmatch(session_id):
        raise ChainError("internal formal session directory must end in a 32-hex root session id")
    expected = root / "results" / "formal" / session_id
    if str(supplied) != str(expected):
        raise ChainError(f"internal formal session path is not the canonical baseline-local path: {supplied}")
    if supplied.is_symlink():
        raise ChainError("internal formal session directory must not be a symlink")
    if supplied.exists() and not supplied.is_dir():
        raise ChainError("internal formal session path is not a directory")
    if not allow_empty and not supplied.is_dir():
        raise ChainError("internal formal session has not been prepared")
    return session_id, supplied


def _set_status(root: Path, session_id: str, status: str, active_stage: str, failure: str = "") -> None:
    path = formal_session_status_path(root, session_id)
    current = load_json(path, "formal session status")
    if not isinstance(current, dict) or current.get("session_id") != session_id:
        raise ChainError("formal session status identity is invalid")
    current.update({"status": status, "active_stage": active_stage, "updated_unix_ns": time.time_ns(), "failure": failure})
    atomic_write_json(path, current, root)


def _scratch_directory(root: Path, session_dir: Path) -> Path:
    scratch = local_path(root, "artifacts/private/native_scratch", "native scratch directory")
    try:
        scratch.relative_to(root.absolute())
    except ValueError as exc:
        raise ChainError("native scratch path escapes the zkGPT baseline") from exc
    ensure_directory(scratch, "native scratch directory", root)
    if scratch.is_symlink() or not scratch.is_dir():
        raise ChainError("native scratch path is aliased or not a directory")
    comparison_root = root.parent
    expected_device = os.stat(comparison_root, follow_symlinks=False).st_dev
    if os.stat(scratch, follow_symlinks=False).st_dev != expected_device:
        raise ChainError("native scratch directory is not on the verified comparison results filesystem")
    if os.stat(session_dir.parent.parent, follow_symlinks=False).st_dev != expected_device:
        raise ChainError("zkGPT formal session is not on the verified comparison results filesystem")
    os.environ["ZKGPT_NATIVE_SCRATCH_DIR"] = str(scratch)
    return scratch


def _run_logged_stage(root: Path, command: Sequence[str], logs: Path, name: str) -> tuple[Path, Path]:
    stdout_path = logs / f"{name}.stdout.log"
    stderr_path = logs / f"{name}.stderr.log"
    returncode = run_logged(command, root, stdout_path, stderr_path, name)
    if returncode != 0:
        raise ChainError(f"{name} exited with code {returncode}; see {stderr_path}")
    return stdout_path, stderr_path


def _tree_regular_bytes(path: Path) -> int:
    total = 0
    for current, directories, filenames in os.walk(path, followlinks=False):
        parent = Path(current)
        for name in directories:
            entry = parent / name
            if entry.is_symlink() or not entry.is_dir():
                raise ChainError(f"formal session contains an aliased/non-directory path: {entry}")
        for name in filenames:
            entry = parent / name
            info = entry.lstat()
            if entry.is_symlink() or not entry.is_file() or info.st_nlink != 1:
                raise ChainError(f"formal session contains an aliased/non-file path: {entry}")
            total += info.st_size
    return total


def _capacity_profile(session_id: str, session_dir: Path, preflight: Mapping[str, Any]) -> Dict[str, Any]:
    capacities = preflight.get("capacity")
    if not isinstance(capacities, dict) or set(capacities) != set(DATASETS):
        raise ChainError("zkGPT capacity matrix does not contain the five canonical datasets")
    dataset_scope = canonical_dataset_scope(preflight.get("dataset_scope"))
    datasets: Dict[str, Any] = {}
    for dataset in DATASETS:
        item = capacities[dataset]
        if not isinstance(item, dict):
            raise ChainError(f"zkGPT capacity admission is malformed for {dataset}")
        if item.get("status") == "BLOCKED":
            reason = item.get("reason")
            if not isinstance(reason, str) or not reason:
                raise ChainError(f"zkGPT blocked capacity has no reason for {dataset}")
            datasets[dataset] = {"status": "BLOCKED", "reason": reason}
            continue
        if (item.get("status") != "READY" or
                item.get("proof_path") != "formal_native_masked_lasso_hyrax+native_masked_layered_gkr" or
                item.get("full_relation_bound") is not True):
            raise ChainError(f"zkGPT full relation/capacity admission is not READY for {dataset}")
        scratch = item.get("estimated_native_scratch_bytes")
        disk = item.get("disk_requirement_bytes_upper_bound")
        retained = item.get("artifact_disk_bytes_upper_bound")
        host = item.get("estimated_peak_host_memory_bytes")
        wire = item.get("proof_artifact_wire_bytes_upper_bound")
        context = item.get("context_payload_bytes_bound")
        values = (scratch, disk, retained, host, wire, context)
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values):
            raise ChainError(f"zkGPT source-derived capacity fields are incomplete for {dataset}")
        if disk < scratch + retained:
            raise ChainError(f"zkGPT disk bound does not include scratch plus retained artifacts for {dataset}")
        # The capacity report covers the source-derived native live phases and
        # disk artifacts. The extra wire/context terms cover the exact-sized
        # in-memory canonical proof encoding and the bound verifier context.
        additional_disk = disk + CELL_INTERNAL_LOG_RESERVE_BYTES
        retained_upper = retained + CELL_INTERNAL_LOG_RESERVE_BYTES
        host_upper = host + wire + context + HOST_ACCOUNTING_OVERHEAD_BYTES
        datasets[dataset] = {
            "status": "READY",
            "additional_disk_peak_bytes": additional_disk,
            "cell_retained_artifact_bytes_upper_bound": retained_upper,
            "host_peak_bytes": host_upper,
            "device_peak_bytes": 0,
            "phase_coverage": ["setup", "witness", "commitment", "prove", "verify"],
        }
    fs_device = os.stat(session_dir.parent.parent, follow_symlinks=False).st_dev
    finalization_host = max((item["host_peak_bytes"] for item in datasets.values() if item.get("status") == "READY"), default=HOST_ACCOUNTING_OVERHEAD_BYTES)
    return {
        "schema": LOCAL_CAPACITY_PROFILE_SCHEMA,
        "baseline": "zkgpt",
        "session_id": session_id,
        "fs_device_id": str(fs_device),
        # Live statvfs admission happens after prepare, so existing session
        # files are already reflected in available bytes.
        "preexisting_baseline_artifact_bytes": 0,
        "dataset_scope": list(dataset_scope),
        "finalization_disk_bytes_upper_bound": FINALIZATION_INTERNAL_DISK_RESERVE_BYTES,
        "finalization_host_bytes_upper_bound": finalization_host,
        "memory_accounting_kind": "host-only",
        "device": None,
        "bound_methods": {"additional_disk_peak": "source_derived_upper_bound", "host_peak": "source_derived_upper_bound"},
        "datasets": datasets,
    }


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _owner_file(root: Path) -> Path:
    return local_path(root, "baseline-local/formal-owner.json", "zkGPT local formal owner record")


def _owner_lock_path(root: Path) -> Path:
    return local_path(root, "baseline-local/formal-run.lock", "zkGPT local formal lock")


def _proc_start_ticks(pid: int) -> int:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise ChainError("formal owner PID is invalid")
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    except OSError as exc:
        raise ChainError(f"formal owner process {pid} is no longer present") from exc
    close = raw.rfind(")")
    if close < 0:
        raise ChainError("formal owner process identity is malformed")
    fields = raw[close + 2 :].split()
    # The remainder begins at field 3 (state); starttime is field 22.
    try:
        return int(fields[19])
    except (IndexError, ValueError) as exc:
        raise ChainError("formal owner process start time is malformed") from exc


def _current_owner_record(root: Path) -> Dict[str, Any]:
    record = load_json(_owner_file(root), "zkGPT local formal owner record")
    if not isinstance(record, dict) or record.get("schema") != LOCAL_OWNER_SCHEMA or record.get("baseline") != "zkgpt":
        raise ChainError("zkGPT local formal owner record is malformed")
    return record


def _write_owner_record(root: Path, record: Dict[str, Any]) -> None:
    record["updated_unix_ns"] = time.time_ns()
    atomic_write_json(_owner_file(root), record, root)


def _update_owner_record(root: Path, **updates: Any) -> Dict[str, Any]:
    record = _current_owner_record(root)
    record.update(updates)
    _write_owner_record(root, record)
    return record


def _mark_session_terminal(root: Path, session_id: str, status: str, stage: str, failure: str) -> bool:
    if not SESSION_ID_RE.fullmatch(session_id):
        return False
    status_path = root / "results" / "formal" / session_id / "session_status.json"
    try:
        if status_path.is_file() and not status_path.is_symlink():
            _set_status(root, session_id, status, stage, failure)
        else:
            atomic_write_json(
                status_path,
                {
                    "schema": "zkgpt-gat-formal-session-v1",
                    "session_id": session_id,
                    "status": status,
                    "active_stage": stage,
                    "completed_stages": [],
                    "stage_status": {},
                    "started_unix_ns": time.time_ns(),
                    "updated_unix_ns": time.time_ns(),
                    "failure": failure,
                },
                root,
            )
        return True
    except Exception as exc:
        print(f"ZKGPT_SESSION_STATUS_WRITE_FAILED: {exc}", file=sys.stderr, flush=True)
        return False


def _attach_owner_to_session_status(root: Path, session_id: str) -> None:
    owner = _current_owner_record(root)
    status_path = root / "results" / "formal" / session_id / "session_status.json"
    status_value = load_json(status_path, "zkGPT session status")
    if not isinstance(status_value, dict) or status_value.get("session_id") != session_id:
        raise ChainError("zkGPT session status cannot be bound to its local owner")
    status_value["local_owner"] = {
        "schema": LOCAL_OWNER_SCHEMA,
        "session_id": session_id,
        "owner_pid": owner["owner_pid"],
        "owner_start_ticks": owner["owner_start_ticks"],
        "owner_token_sha256": owner["owner_token_sha256"],
        "lock_device": owner["lock_device"],
        "lock_inode": owner["lock_inode"],
    }
    atomic_write_json(status_path, status_value, root)


def _recover_stale_owner(root: Path) -> str:
    path = _owner_file(root)
    if not path.exists():
        return ""
    previous = _current_owner_record(root)
    previous_session = previous.get("session_id")
    if not isinstance(previous_session, str) or not SESSION_ID_RE.fullmatch(previous_session):
        if previous.get("status") == "RUNNING":
            raise ChainError("stale zkGPT owner record has no valid session identity")
        return ""
    previous_pid = previous.get("owner_pid")
    previous_start = previous.get("owner_start_ticks")
    if previous.get("status") == "RUNNING" and isinstance(previous_pid, int) and not isinstance(previous_pid, bool):
        proc_stat = Path(f"/proc/{previous_pid}/stat")
        if proc_stat.exists() and _proc_start_ticks(previous_pid) == previous_start:
            raise ChainError("prior zkGPT owner is still alive despite the local lock being available")
    if previous.get("status") == "RUNNING":
        terminal_status = "INTERRUPTED"
        failure = "stale local owner recovered after its baseline lock was released"
        stage = str(previous.get("active_stage") or "stale-owner-recovery")
    elif previous.get("status") in ("FAILED", "INTERRUPTED"):
        terminal_status = str(previous["status"])
        failure = str(previous.get("failure") or "local owner terminal state was recovered")
        stage = str(previous.get("active_stage") or "stale-owner-recovery")
    else:
        return ""
    if not _mark_session_terminal(root, previous_session, terminal_status, stage, failure):
        raise ChainError("could not persist the prior zkGPT session's terminal status")
    status_path = root / "results" / "formal" / previous_session / "session_status.json"
    persisted = load_json(status_path, "recovered zkGPT session status")
    if not isinstance(persisted, dict) or persisted.get("status") != terminal_status:
        raise ChainError("prior zkGPT session terminal status did not persist")
    if previous.get("status") == "RUNNING":
        previous.update({"status": "INTERRUPTED", "active_stage": "stale-owner-recovery", "failure": failure})
        _write_owner_record(root, previous)
    return previous_session


def _validate_owner_lease(root: Path, phase: str, session_id: str, dataset: str | None) -> Dict[str, Any]:
    record = _current_owner_record(root)
    try:
        owner_pid = int(os.environ[_OWNER_PID_ENV])
        owner_start = int(os.environ[_OWNER_START_ENV])
        lock_fd = int(os.environ[_OWNER_LOCK_FD_ENV])
    except (KeyError, ValueError) as exc:
        raise ChainError("private formal phase has no local owner lease") from exc
    token = os.environ.get(_OWNER_TOKEN_ENV, "")
    token_sha = hashlib.sha256(token.encode("ascii", errors="strict")).hexdigest() if token else ""
    expected_dataset = os.environ.get(_OWNER_DATASET_ENV) or None
    try:
        requested_scope = canonical_dataset_scope(json.loads(os.environ[_OWNER_DATASET_SCOPE_ENV]))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, ChainError) as exc:
        raise ChainError("private formal phase has no valid dataset scope lease") from exc
    if (
        record.get("status") != "RUNNING"
        or record.get("session_id") != session_id
        or record.get("owner_pid") != owner_pid
        or record.get("owner_start_ticks") != owner_start
        or record.get("owner_token_sha256") != token_sha
        or record.get("active_stage") != phase
        or record.get("active_dataset") != dataset
        or record.get("stage_nonce") != os.environ.get(_OWNER_STAGE_NONCE_ENV)
        or os.environ.get(_OWNER_STAGE_ENV) != phase
        or os.environ.get(_OWNER_SESSION_ENV) != session_id
        or expected_dataset != dataset
        or record.get("dataset_scope") != list(requested_scope)
    ):
        raise ChainError("private formal phase does not match the active zkGPT owner stage")
    if os.getppid() != owner_pid or _proc_start_ticks(owner_pid) != owner_start:
        raise ChainError("private formal phase is not a live direct child of its recorded owner")
    lock_path = _owner_lock_path(root)
    try:
        lock_info = os.fstat(lock_fd)
        path_info = os.stat(lock_path, follow_symlinks=False)
        if (
            not stat.S_ISREG(lock_info.st_mode)
            or lock_info.st_nlink != 1
            or (lock_info.st_dev, lock_info.st_ino) != (path_info.st_dev, path_info.st_ino)
            or record.get("lock_device") != lock_info.st_dev
            or record.get("lock_inode") != lock_info.st_ino
        ):
            raise ChainError("private phase did not inherit the active baseline-local lock inode")
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (OSError, ValueError) as exc:
        raise ChainError("private phase cannot validate the inherited baseline-local lock") from exc
    return record


def _read_owner_approved_cells_fd(root: Path, fd: int, session_id: str) -> Dict[str, Dict[str, str]]:
    if fd < 3:
        raise ChainError("owner approval input FD is invalid")
    try:
        info = os.fstat(fd)
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 0 or info.st_size < 1 or
                info.st_size > OWNER_APPROVED_CELLS_LIMIT_BYTES or (flags & os.O_ACCMODE) != os.O_RDONLY):
            raise ChainError("owner approval input must be a bounded anonymous read-only regular file")
        os.lseek(fd, 0, os.SEEK_SET)
        raw = bytearray()
        while len(raw) <= OWNER_APPROVED_CELLS_LIMIT_BYTES:
            chunk = os.read(fd, min(65536, OWNER_APPROVED_CELLS_LIMIT_BYTES + 1 - len(raw)))
            if not chunk:
                break
            raw.extend(chunk)
    except OSError as exc:
        raise ChainError("cannot read inherited owner approval input") from exc
    if len(raw) != info.st_size:
        raise ChainError("owner approval input length changed during read")
    try:
        payload = json.loads(bytes(raw).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ChainError("owner approval input is malformed") from exc
    if (not isinstance(payload, dict) or payload.get("schema") != OWNER_APPROVED_CELLS_SCHEMA or
            payload.get("session_id") != session_id):
        raise ChainError("owner approval input has a stale schema or session")
    cells = payload.get("cells")
    if not isinstance(cells, dict) or set(cells) != set(DATASETS):
        raise ChainError("owner approval input must contain exactly the five canonical datasets")
    required = {"dataset", "receipt_path", "receipt_sha256", "result_path", "result_sha256",
                "execution_path", "execution_sha256", "proof_path", "proof_sha256",
                "verification_report_path", "verification_report_sha256"}
    for dataset in DATASETS:
        entry = cells[dataset]
        if not isinstance(entry, dict) or set(entry) != required or entry.get("dataset") != dataset:
            raise ChainError(f"owner approval entry is malformed: {dataset}")
        if entry["receipt_path"] != f"results/formal/{session_id}/receipts/{dataset}.json":
            raise ChainError(f"owner approval receipt path is noncanonical: {dataset}")
        for key in required - {"dataset"}:
            value = entry[key]
            if not isinstance(value, str) or not value:
                raise ChainError(f"owner approval entry is missing {key}: {dataset}")
        for path_key in ("receipt_path", "result_path", "execution_path", "proof_path", "verification_report_path"):
            local_path(root, entry[path_key], "owner-approved cell artifact")
        for digest_key in ("receipt_sha256", "result_sha256", "execution_sha256", "proof_sha256", "verification_report_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", entry[digest_key]):
                raise ChainError(f"owner approval entry has an invalid {digest_key}: {dataset}")
    return cells


def _install_parent_death_watch(root: Path, session_id: str) -> None:
    owner_pid = int(os.environ[_OWNER_PID_ENV])
    owner_start = int(os.environ[_OWNER_START_ENV])

    def owner_died(signum: int, _frame: Any) -> None:
        try:
            status_written = _mark_session_terminal(root, session_id, "INTERRUPTED", "owner-death", "local owner exited before its private phase completed")
            if status_written:
                owner_record = _current_owner_record(root)
                if owner_record.get("session_id") == session_id and owner_record.get("owner_pid") == owner_pid:
                    owner_record.update(
                        {"status": "INTERRUPTED", "active_stage": "owner-death", "failure": "local owner exited before its private phase completed"}
                    )
                    _write_owner_record(root, owner_record)
        except Exception:
            pass
        signal.signal(signum, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            os.killpg(os.getpgrp(), signal.SIGTERM)
            time.sleep(0.5)
            os.killpg(os.getpgrp(), signal.SIGKILL)
        except OSError:
            os._exit(128 + signum)

    # Install handlers before registering PDEATHSIG so there is no interval
    # where the kernel can deliver SIGTERM with its default action.
    signal.signal(signal.SIGTERM, owner_died)
    signal.signal(signal.SIGINT, owner_died)
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        # Linux prctl(PR_SET_PDEATHSIG, SIGTERM) stops a private phase from
        # surviving a SIGKILL of its local owner.
        if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
            err = ctypes.get_errno()
            raise OSError(err, os.strerror(err))
    except (AttributeError, OSError) as exc:
        raise ChainError(f"cannot install local owner death watch: {exc}") from exc
    if os.getppid() != owner_pid or _proc_start_ticks(owner_pid) != owner_start:
        status_written = _mark_session_terminal(
            root, session_id, "INTERRUPTED", "owner-death", "local owner exited while installing the private phase death watch"
        )
        if status_written:
            try:
                owner_record = _current_owner_record(root)
                owner_record.update(
                    {"status": "INTERRUPTED", "active_stage": "owner-death", "failure": "local owner exited while installing the private phase death watch"}
                )
                _write_owner_record(root, owner_record)
            except Exception:
                pass
        raise ChainError("zkGPT owner died while installing the private phase death watch")


def _available_host_bytes() -> int:
    try:
        mem_available = None
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            key, _, value = line.partition(":")
            if key == "MemAvailable":
                mem_available = int(value.strip().split()[0]) * 1024
                break
        if mem_available is None:
            raise ValueError("MemAvailable field is missing")
        available = mem_available
    except (OSError, ValueError, IndexError) as exc:
        raise ChainError(f"cannot read live host memory headroom: {exc}") from exc
    try:
        membership: Dict[str, str] = {}
        for line in Path("/proc/self/cgroup").read_text(encoding="ascii").splitlines():
            hierarchy, controllers, path = line.split(":", 2)
            if hierarchy == "0" and not controllers:
                membership["cgroup2"] = path
            elif "memory" in controllers.split(","):
                membership["cgroup"] = path
        mount_lines = Path("/proc/self/mountinfo").read_text(encoding="ascii").splitlines()
    except OSError as exc:
        raise ChainError(f"cannot locate the current cgroup memory limit: {exc}") from exc

    def unescape_mount_field(value: str) -> str:
        return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match.group(1), 8)), value)

    cgroup_paths: List[tuple[Path, str]] = []
    for line in mount_lines:
        before, separator, after = line.partition(" - ")
        if not separator:
            continue
        left = before.split()
        right = after.split()
        if len(left) < 5 or not right:
            continue
        fs_type = right[0]
        if fs_type == "cgroup2":
            group_path = membership.get("cgroup2")
            limit_name, current_name = "memory.max", "memory.current"
            v1 = False
        elif fs_type == "cgroup" and "memory" in ",".join(right[2:]).split(","):
            group_path = membership.get("cgroup")
            limit_name, current_name = "memory.limit_in_bytes", "memory.usage_in_bytes"
            v1 = True
        else:
            continue
        if not group_path:
            continue
        mount_root = Path(unescape_mount_field(left[3]))
        mount_point = Path(unescape_mount_field(left[4]))
        member_path = Path(group_path)
        try:
            relative = member_path.relative_to(mount_root)
        except ValueError:
            continue
        leaf = mount_point / relative
        current_path = leaf
        while True:
            cgroup_paths.append((current_path, "v1" if v1 else "v2"))
            if current_path == mount_point:
                break
            try:
                current_path = current_path.parent
                current_path.relative_to(mount_point)
            except ValueError:
                break

    for directory, version in cgroup_paths:
        if version == "v2":
            limit_path, current_path = directory / "memory.max", directory / "memory.current"
        else:
            limit_path = directory / "memory.limit_in_bytes"
            current_path = directory / "memory.usage_in_bytes"
        if not limit_path.exists() and not current_path.exists():
            continue
        try:
            raw_limit = limit_path.read_text(encoding="ascii").strip()
            current = int(current_path.read_text(encoding="ascii").strip())
            if raw_limit == "max":
                continue
            limit = int(raw_limit)
            if version == "v1" and limit >= (1 << 60):
                continue
            available = min(available, max(0, limit - current))
        except (OSError, ValueError) as exc:
            raise ChainError(f"cannot read live cgroup memory headroom: {exc}") from exc
    return available


def _capacity_requirements(profile: Mapping[str, Any], pending_datasets: Sequence[str]) -> tuple[int, int]:
    datasets = profile.get("datasets")
    if not isinstance(datasets, dict) or set(datasets) != set(DATASETS):
        raise ChainError("zkGPT local capacity profile does not cover exactly five datasets")
    pending = list(pending_datasets)
    if any(dataset not in DATASETS for dataset in pending):
        raise ChainError("zkGPT capacity gate contains an unknown dataset")
    if pending != [dataset for dataset in DATASETS if dataset in set(pending)]:
        raise ChainError("zkGPT capacity gate dataset suffix is not in canonical order")
    final_disk = profile.get("finalization_disk_bytes_upper_bound")
    final_host = profile.get("finalization_host_bytes_upper_bound")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in (final_disk, final_host)):
        raise ChainError("zkGPT finalization capacity bounds are missing")
    retained = 0
    scratch_peaks: List[int] = []
    host_peaks: List[int] = []
    for dataset in pending:
        item = datasets[dataset]
        if not isinstance(item, dict):
            raise ChainError(f"zkGPT local capacity entry is malformed for {dataset}")
        if item.get("status") != "READY":
            reason = item.get("reason")
            raise ChainError(f"zkGPT dataset capacity is blocked for {dataset}: {reason if isinstance(reason, str) else 'missing reason'}")
        additional = item.get("additional_disk_peak_bytes")
        kept = item.get("cell_retained_artifact_bytes_upper_bound")
        host = item.get("host_peak_bytes")
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in (additional, kept, host)) or additional < kept:
            raise ChainError(f"zkGPT local capacity bounds are invalid for {dataset}")
        retained += kept
        scratch_peaks.append(additional - kept)
        host_peaks.append(host)
    if not pending and any(datasets[dataset].get("status") != "READY" for dataset in DATASETS):
        raise ChainError("zkGPT finalization capacity requires all five datasets to be READY")
    # Live free space already includes completed cells. Reserve only future
    # retained outputs, one cell's concurrent scratch, and finalization bytes.
    disk_required = retained + max(scratch_peaks, default=0) + final_disk
    host_required = max(host_peaks, default=final_host)
    return host_required, disk_required


def _capacity_gate(root: Path, session_id: str, session_dir: Path, profile: Mapping[str, Any], pending_datasets: Sequence[str], stage: str) -> Dict[str, int]:
    if (
        profile.get("schema") != LOCAL_CAPACITY_PROFILE_SCHEMA
        or profile.get("baseline") != "zkgpt"
        or profile.get("session_id") != session_id
        or profile.get("memory_accounting_kind") != "host-only"
        or profile.get("device") is not None
        or profile.get("preexisting_baseline_artifact_bytes") != 0
    ):
        raise ChainError("zkGPT local capacity profile identity is invalid")
    expected_device = os.stat(root.parent, follow_symlinks=False).st_dev
    expected_device_text = str(expected_device)
    if profile.get("fs_device_id") != expected_device_text:
        raise ChainError("zkGPT local capacity profile is bound to another results filesystem")
    scratch = _scratch_directory(root, session_dir)
    results_info = os.stat(session_dir.parent.parent, follow_symlinks=False)
    scratch_info = os.stat(scratch, follow_symlinks=False)
    if results_info.st_dev != expected_device or scratch_info.st_dev != expected_device:
        raise ChainError("zkGPT results and native scratch must share the admitted filesystem")
    host_required, disk_required = _capacity_requirements(profile, pending_datasets)
    host_available = _available_host_bytes()
    statvfs = os.statvfs(scratch)
    disk_available = statvfs.f_bavail * statvfs.f_frsize
    if host_available < host_required:
        raise ChainError(f"zkGPT live host headroom is below source-derived bound at {stage}: {host_available} < {host_required}")
    if disk_available < disk_required:
        raise ChainError(f"zkGPT live native scratch/filesystem headroom is below source-derived bound at {stage}: {disk_available} < {disk_required}")
    observation = {
        "stage": stage,
        "pending_datasets": list(pending_datasets),
        "host_available_bytes": host_available,
        "host_required_bytes": host_required,
        "disk_available_bytes": disk_available,
        "disk_required_bytes": disk_required,
        "fs_device_id": expected_device_text,
        "checked_unix_ns": time.time_ns(),
    }
    status_path = root / "results" / "formal" / session_id / "session_status.json"
    if status_path.is_file() and not status_path.is_symlink():
        status_value = load_json(status_path, "zkGPT session status")
        if not isinstance(status_value, dict) or status_value.get("session_id") != session_id:
            raise ChainError("zkGPT session status identity changed during capacity admission")
        observations = status_value.setdefault("local_capacity_rechecks", [])
        if not isinstance(observations, list):
            raise ChainError("zkGPT local capacity recheck history is malformed")
        observations.append(observation)
        atomic_write_json(status_path, status_value, root)
    return observation


def _load_prepared(root: Path, session_id: str, session_dir: Path) -> tuple[Dict[str, Any], Dict[str, Any], str, Dict[str, Any], Path, str]:
    os.environ["ZKGPT_FORMAL_SESSION_ID"] = session_id
    session_id, session, session_manifest_sha256 = require_formal_session(root)
    context_path = session_dir / "local_formal_context.json"
    context = load_json(context_path, "local formal session context")
    if not isinstance(context, dict) or context.get("schema") != "zkgpt-gat-local-formal-context-v1" or context.get("root_session_id") != session_id:
        raise ChainError("zkGPT local formal context is missing or bound to another session")
    try:
        session_scope = canonical_dataset_scope(session.get("dataset_scope"))
        context_scope = canonical_dataset_scope(context.get("dataset_scope"))
    except ChainError as exc:
        raise ChainError("zkGPT prepared local session scope is invalid") from exc
    if (
        context.get("session_manifest_sha256") != session_manifest_sha256
        or context_scope != session_scope
        or context.get("runtime_source_sha256") != runtime_source_sha256(root)
        or context.get("native_source_sha256") != source_tree_sha256(root)
    ):
        raise ChainError("zkGPT prepared local source/session context is stale")
    preflight_path = formal_preflight_path(root, session_id)
    preflight = load_json(preflight_path, "formal preflight")
    if not isinstance(preflight, dict):
        raise ChainError("zkGPT formal preflight is malformed")
    try:
        preflight_scope = canonical_dataset_scope(preflight.get("dataset_scope"))
    except ChainError as exc:
        raise ChainError("zkGPT formal preflight dataset scope is invalid") from exc
    if preflight_scope != session_scope:
        raise ChainError("zkGPT formal preflight dataset scope differs from its session")
    _resource_manifest, resource_manifest_path, resource_digest = require_formal_resource_manifest(
        root, session_id, session, session_manifest_sha256
    )
    resource_manifest_relative = relative_path(root, resource_manifest_path, "resource manifest")
    if (
        context.get("resource_manifest_path") != resource_manifest_relative
        or preflight.get("resource_manifest_path") != resource_manifest_relative
        or context.get("resource_manifest_sha256") != resource_digest
        or preflight.get("resource_manifest_sha256") != resource_digest
        or context.get("preflight_sha256") != sha256_file(preflight_path)
    ):
        raise ChainError("zkGPT prepared resource/preflight digest is stale")
    _validate_preflight(root, "cpu", preflight, resource_digest, session_id, session_manifest_sha256, session["native_source_sha256"], allow_partial=True)
    local_profile = context.get("local_capacity_profile")
    expected_profile = _capacity_profile(session_id, session_dir, preflight)
    if (
        not isinstance(local_profile, dict)
        or local_profile != expected_profile
        or context.get("local_capacity_profile_sha256") != _canonical_sha256(expected_profile)
    ):
        raise ChainError("zkGPT local capacity profile is missing, stale, or changed")
    source_digest = verify_build_provenance(root, "cpu")
    if context.get("source_tree_sha256") != source_digest:
        raise ChainError("zkGPT build provenance no longer matches its prepare receipt")
    prover = build_binary(root, "cpu")
    verifier = verifier_binary(root, "cpu")
    if context.get("prover_fingerprint") != binary_fingerprint(prover, "zkGPT prover") or context.get("verifier_fingerprint") != binary_fingerprint(
        verifier, "zkGPT verifier"
    ):
        raise ChainError("zkGPT source-matched binary changed after prepare")
    build_log_value = context.get("build_log_path")
    if not isinstance(build_log_value, str):
        raise ChainError("zkGPT prepare context has no build log path")
    build_log = local_path(root, build_log_value, "formal build log")
    if context.get("build_log_sha256") != sha256_file(build_log):
        raise ChainError("zkGPT formal build log changed after prepare")
    _scratch_directory(root, session_dir)
    return session, preflight, resource_digest, context, preflight_path, session_manifest_sha256


def _prepare_phase(root: Path, session_id: str, session_dir: Path, dataset_scope: Sequence[str]) -> Dict[str, Any]:
    if session_dir.exists() and any(session_dir.iterdir()):
        raise ChainError("zkGPT local formal session directory must be fresh and empty")
    sid, session, session_manifest_sha256 = create_formal_session(root, session_id, dataset_scope)
    os.environ["ZKGPT_FORMAL_SESSION_ID"] = sid
    _attach_owner_to_session_status(root, sid)
    _scratch_directory(root, session_dir)
    logs = session_dir / "logs"
    ensure_directory(logs, "formal session logs", root)
    _set_status(root, sid, "RUNNING", "resource-authentication")
    resource_command = [sys.executable, str(root / "scripts" / "prepare_resources.py"), "--check"]
    if tuple(dataset_scope) != DATASETS:
        resource_command.extend(["--datasets", *dataset_scope])
    _run_logged_stage(root, resource_command, logs, "resource-authentication")

    _set_status(root, sid, "RUNNING", "native-build")
    _run_logged_stage(root, [str(root / "build.sh"), "--cpu"], logs, "native-build")
    build_stdout = logs / "native-build.stdout.log"
    build_stderr = logs / "native-build.stderr.log"
    build_log = logs / "build.log"
    atomic_write_text(
        build_log,
        "=== build stdout ===\n"
        + build_stdout.read_text(encoding="utf-8", errors="replace")
        + "\n=== build stderr ===\n"
        + build_stderr.read_text(encoding="utf-8", errors="replace"),
        root,
    )
    build_source_digest = verify_build_provenance(root, "cpu")
    prover = build_binary(root, "cpu")
    verifier = verifier_binary(root, "cpu")
    build_fingerprints = {"prover": binary_fingerprint(prover, "zkGPT prover"), "verifier": binary_fingerprint(verifier, "zkGPT verifier")}

    preflight_stage = "five-dataset-preflight" if tuple(dataset_scope) == DATASETS else "requested-dataset-preflight"
    _set_status(root, sid, "RUNNING", preflight_stage)
    _run_logged_stage(root, [sys.executable, str(root / "scripts" / "formal_preflight.py")], logs, preflight_stage)
    preflight_path = formal_preflight_path(root, sid)
    preflight = load_json(preflight_path, "formal preflight")
    if (not isinstance(preflight, dict) or preflight.get("status") not in ("READY", "PARTIAL") or
            preflight.get("dataset_scope") != list(dataset_scope) or
            preflight.get("shared_preconditions_ready") is not True):
        raise ChainError("zkGPT shared source/resource/backend/session preconditions did not pass")
    _resource_manifest, resource_manifest_path, resource_digest = require_formal_resource_manifest(
        root, sid, session, session_manifest_sha256
    )
    resource_manifest_relative = relative_path(root, resource_manifest_path, "resource manifest")
    if (preflight.get("resource_manifest_path") != resource_manifest_relative
            or preflight.get("resource_manifest_sha256") != resource_digest):
        raise ChainError("zkGPT preflight resource manifest path/digest is stale")
    _validate_preflight(root, "cpu", preflight, resource_digest, sid, session_manifest_sha256, session["native_source_sha256"], allow_partial=True)
    profile = _capacity_profile(sid, session_dir, preflight)
    context = {
        "schema": "zkgpt-gat-local-formal-context-v1",
        "status": "PREPARED",
        "root_session_id": sid,
        "session_manifest_sha256": session_manifest_sha256,
        "runtime_source_sha256": session["runtime_source_sha256"],
        "native_source_sha256": session["native_source_sha256"],
        "source_tree_sha256": build_source_digest,
        "resource_manifest_path": resource_manifest_relative,
        "resource_manifest_sha256": resource_digest,
        "preflight_sha256": sha256_file(preflight_path),
        "prover_fingerprint": build_fingerprints["prover"],
        "verifier_fingerprint": build_fingerprints["verifier"],
        "build_log_path": relative_path(root, build_log, "build log"),
        "build_log_sha256": sha256_file(build_log),
        "datasets": list(DATASETS),
        "dataset_scope": list(dataset_scope),
        "local_capacity_profile": profile,
        "local_capacity_profile_sha256": _canonical_sha256(profile),
        "created_unix_ns": time.time_ns(),
    }
    atomic_write_json(session_dir / "local_formal_context.json", context, root)
    graph_datasets: Dict[str, Any] = {}
    capacity_matrix = preflight.get("capacity")
    if not isinstance(capacity_matrix, dict) or set(capacity_matrix) != set(DATASETS):
        raise ChainError("zkGPT graph identity source does not cover exactly five native capacity reports")
    for dataset in DATASETS:
        report = capacity_matrix[dataset]
        identity = report.get("canonical_graph_identity") if isinstance(report, dict) else None
        if isinstance(identity, dict) and report.get("status") == "READY":
            graph_datasets[dataset] = {"status": "READY", "identity": identity}
        elif isinstance(report, dict) and report.get("status") == "BLOCKED" and isinstance(report.get("reason"), str) and report["reason"]:
            graph_datasets[dataset] = {"status": "BLOCKED", "reason": report["reason"]}
        else:
            raise ChainError(f"zkGPT capacity report has no valid graph identity or capacity blocker for {dataset}")
    local_graph_identity = {"schema": "zkgpt-local-canonical-graph-set-v2", "datasets": graph_datasets}
    _set_status(root, sid, "RUNNING", "prepared")
    return {
        "phase": "prepare",
        "status": "PASS",
        "session_id": sid,
        "dataset_scope": list(dataset_scope),
        "session_dir": str(session_dir),
        "build_log_path": context["build_log_path"],
        "build_log_sha256": context["build_log_sha256"],
        "local_capacity_profile_sha256": context["local_capacity_profile_sha256"],
        "canonical_graph_identity": local_graph_identity,
    }


def _cell_phase(root: Path, session_id: str, session_dir: Path, dataset: str) -> Dict[str, Any]:
    session, preflight, resource_digest, context, preflight_path, session_manifest_sha256 = _load_prepared(root, session_id, session_dir)
    receipts = session_dir / "receipts"
    ensure_directory(receipts, "formal cell receipts", root)
    receipt_path = receipts / f"{dataset}.json"
    if receipt_path.exists():
        raise ChainError(f"zkGPT cell receipt already exists; refusing to rerun {dataset}")
    run_relative, run_path = run_directory(root, dataset, session_id)
    if any(run_path.iterdir()):
        raise ChainError(f"zkGPT formal cell has prior partial or stale output; refusing to overwrite {dataset}")
    _set_status(root, session_id, "RUNNING", f"cell:{dataset}")
    run_cell(root, dataset, "cpu", preflight, resource_digest, session_id, session, session_manifest_sha256, preflight_path)
    audited = validate_formal_cell(
        root, dataset, "cpu", preflight, resource_digest, session_id, session_manifest_sha256, session["native_source_sha256"], preflight_path
    )
    result_path = root / audited["result_path"]
    execution_path = result_path.parent / "execution.json"
    result = load_json(result_path, "native result")
    execution = load_json(execution_path, "execution record")
    proof_path = root / audited["artifact_path"]
    verify_report_path = root / audited["verifier_report_path"]
    receipt = {
        "schema": "zkgpt-gat-cell-receipt-v1",
        "status": "VERIFY_OK",
        "baseline": "zkgpt",
        "dataset": dataset,
        "session_id": session_id,
        "session_manifest_sha256": session_manifest_sha256,
        "runtime_source_sha256": context["runtime_source_sha256"],
        "native_source_sha256": context["native_source_sha256"],
        "source_tree_sha256": context["source_tree_sha256"],
        "resource_manifest_sha256": resource_digest,
        "preflight_sha256": sha256_file(preflight_path),
        "result_path": audited["result_path"],
        "result_sha256": sha256_file(result_path),
        "execution_path": str(execution_path.relative_to(root)),
        "execution_sha256": sha256_file(execution_path),
        "proof_path": audited["artifact_path"],
        "proof_sha256": audited["artifact_sha256"],
        "proof_size_bytes": audited["proof_size_bytes"],
        "verification_report_path": audited["verifier_report_path"],
        "verification_report_sha256": sha256_file(verify_report_path),
        "independent_verification": True,
        "paper_metrics": audited["paper_metrics"],
        "timing_ms": audited["timing_ms"],
        "native_backend": result["native_backend"],
        "prover_binary": context["prover_fingerprint"],
        "verifier_binary": context["verifier_fingerprint"],
        "created_unix_ns": time.time_ns(),
    }
    atomic_write_json(receipt_path, receipt, root)
    _set_status(root, session_id, "RUNNING", f"cell-complete:{dataset}")
    report = load_json(verify_report_path, "independent verifier report")
    metrics = audited["paper_metrics"]
    return {
        "phase": "cell",
        "status": "PASS",
        "dataset": dataset,
        "session_id": session_id,
        "verification_status": "VERIFY_OK",
        "commitment_ms": metrics["commitment_time_ms"],
        "prove_ms": metrics["prove_time_ms"],
        "verify_ms": metrics["verify_time_ms"],
        "proof_size_bytes": audited["proof_size_bytes"],
        "receipt_path": str(receipt_path.relative_to(root)),
        "receipt_sha256": sha256_file(receipt_path),
        "proof_path": str(proof_path.relative_to(root)),
        "proof_sha256": sha256_file(proof_path),
        "independent_verification": report.get("independent") is True,
        "verification_report_path": str(verify_report_path.relative_to(root)),
        "verification_report_sha256": sha256_file(verify_report_path),
    }


def _finalize_phase(root: Path, session_id: str, session_dir: Path, owner_approved_cells: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    session, preflight, resource_digest, context, preflight_path, session_manifest_sha256 = _load_prepared(root, session_id, session_dir)
    if not isinstance(owner_approved_cells, dict) or set(owner_approved_cells) != set(DATASETS):
        raise ChainError("finalize requires exactly five owner-approved cell receipts")
    receipts = session_dir / "receipts"
    timed_reports: Dict[str, str] = {}
    for dataset in DATASETS:
        receipt_path = receipts / f"{dataset}.json"
        receipt = load_json(receipt_path, f"zkGPT {dataset} cell receipt")
        approved = owner_approved_cells[dataset]
        expected_receipt_path = relative_path(root, receipt_path, "zkGPT cell receipt")
        if (
            not isinstance(receipt, dict)
            or receipt.get("status") != "VERIFY_OK"
            or receipt.get("session_id") != session_id
            or receipt.get("dataset") != dataset
            or approved.get("receipt_path") != expected_receipt_path
            or approved.get("receipt_sha256") != sha256_file(receipt_path)
        ):
            raise ChainError(f"zkGPT verified cell receipt is missing or stale: {dataset}")
        for key in ("result_path", "result_sha256", "execution_path", "execution_sha256", "proof_path", "proof_sha256",
                    "verification_report_path", "verification_report_sha256"):
            if receipt.get(key) != approved.get(key):
                raise ChainError(f"zkGPT owner-approved receipt binding changed for {dataset}: {key}")
        if approved["proof_sha256"] != sha256_file(local_path(root, approved["proof_path"], "zkGPT proof artifact")):
            raise ChainError(f"zkGPT proof artifact changed after cell receipt: {dataset}")
        if approved["verification_report_sha256"] != sha256_file(local_path(root, approved["verification_report_path"], "zkGPT timed independent-verifier report")):
            raise ChainError(f"zkGPT timed independent-verifier report changed after receipt: {dataset}")
        timed_reports[dataset] = approved["verification_report_sha256"]
    _set_status(root, session_id, "RUNNING", "collection")
    logs = session_dir / "logs"
    _run_logged_stage(root, [sys.executable, str(root / "scripts" / "collect_results.py")], logs, "collection")
    _set_status(root, session_id, "RUNNING", "final-validation")
    if _validate_final_results(root, owner_approved_cells=owner_approved_cells) != 0:
        raise ChainError("zkGPT owner final validation rejected the approved cell reports")
    for dataset, digest in timed_reports.items():
        report_path = local_path(root, owner_approved_cells[dataset]["verification_report_path"], "zkGPT timed independent-verifier report")
        if sha256_file(report_path) != digest:
            raise ChainError(f"final independent verifier overwrote the timed cell report: {dataset}")
    final_path = formal_final_path(root, session_id)
    final = load_json(final_path, "final validation result")
    if (
        not isinstance(final, dict)
        or final.get("status") != "VALIDATED"
        or final.get("cells_validated") != len(DATASETS)
        or final.get("session_id") != session_id
    ):
        raise ChainError("zkGPT final native validator did not validate exactly five current-session cells")
    accounting = final.get("verification_accounting")
    if (not isinstance(accounting, dict) or accounting.get("cell_stage_independent_verifier_invocations") != len(DATASETS) or
            accounting.get("final_stage_native_verifier_invocations") != 0 or
            accounting.get("independent_crypto_verifier_invocations_total") != len(DATASETS) or
            accounting.get("cell_reports_reused") is not True):
        raise ChainError("zkGPT final validation has inaccurate verifier reuse accounting")
    verification = final.get("final_native_verification")
    if not isinstance(verification, list) or len(verification) != len(DATASETS):
        raise ChainError("zkGPT final validation omitted approved cell verifier reports")
    for dataset, entry in zip(DATASETS, verification):
        approved = owner_approved_cells[dataset]
        if (not isinstance(entry, dict) or entry.get("dataset") != dataset or
                entry.get("verification_source") != "cell_stage_independent_report" or
                entry.get("reused_cell_report") is not True or entry.get("final_native_verifier_invocations") != 0 or
                entry.get("report_path") != approved["verification_report_path"] or
                entry.get("report_sha256") != approved["verification_report_sha256"]):
            raise ChainError(f"zkGPT final validation report is not bound to the approved cell report: {dataset}")
    _set_status(root, session_id, "COMPLETE", "")
    return {
        "phase": "finalize",
        "status": "PASS",
        "session_id": session_id,
        "final_validation_path": str(final_path.relative_to(root)),
        "final_validation_sha256": sha256_file(final_path),
        "validated_cells": list(DATASETS),
        "independent_verification": True,
        "reused_cell_reports": len(DATASETS),
        "final_native_verifier_invocations": 0,
        "paper_timing_semantics": final.get("paper_timing_semantics"),
    }


def _write_subset_validation(root: Path, session_id: str, session_dir: Path, dataset_scope: Sequence[str], owner_approved_cells: Mapping[str, Mapping[str, Any]], failed_datasets: Mapping[str, str]) -> Dict[str, Any]:
    scope = canonical_dataset_scope(dataset_scope)
    if (
        scope == DATASETS
        or set(owner_approved_cells) & set(failed_datasets)
        or (set(owner_approved_cells) | set(failed_datasets)) != set(scope)
    ):
        raise ChainError("subset validation outcomes do not match a strict proper dataset scope")
    session, preflight, resource_digest, context, preflight_path, session_manifest_sha256 = _load_prepared(root, session_id, session_dir)
    if session.get("dataset_scope") != list(scope) or preflight.get("dataset_scope") != list(scope):
        raise ChainError("subset validation scope differs from its prepared session")
    seen_artifact_paths: set[str] = set()
    seen_artifact_digests: set[str] = set()
    cells: List[Dict[str, Any]] = []
    for dataset in scope:
        approved = owner_approved_cells.get(dataset)
        if approved is None:
            cells.append({"dataset": dataset, "status": "FAILED", "reason": failed_datasets[dataset]})
            continue
        audited = validate_formal_cell(root, dataset, "cpu", preflight, resource_digest, session_id,
                                       session_manifest_sha256, session["native_source_sha256"], preflight_path,
                                       seen_artifact_paths, seen_artifact_digests)
        verification = _reuse_owner_approved_cell(root, session_id, session_manifest_sha256,
                                                  dataset, approved, audited)
        receipt_path = local_path(root, approved["receipt_path"], "subset cell receipt")
        receipt = load_json(receipt_path, "subset cell receipt")
        cells.append({
            "dataset": dataset,
            "status": "VERIFY_OK",
            "receipt_path": approved["receipt_path"],
            "receipt_sha256": approved["receipt_sha256"],
            "result_path": approved["result_path"],
            "result_sha256": approved["result_sha256"],
            "execution_path": approved["execution_path"],
            "execution_sha256": approved["execution_sha256"],
            "proof_path": approved["proof_path"],
            "proof_sha256": approved["proof_sha256"],
            "proof_size_bytes": audited["proof_size_bytes"],
            "verification_report_path": approved["verification_report_path"],
            "verification_report_sha256": approved["verification_report_sha256"],
            "independent_verification": receipt.get("independent_verification") is True,
            "verification_source": verification["verification_source"],
            "paper_metrics": audited["paper_metrics"],
            "timing_ms": audited["timing_ms"],
        })
    status = "SUBSET_FAILED" if failed_datasets else "SUBSET_COMPLETE"
    payload = {
        "schema": "zkgpt-gat-subset-validation-v1",
        "status": status,
        "validation_scope": "requested_dataset_subset",
        "dataset_scope": list(scope),
        "formal_dataset_universe": list(DATASETS),
        "full_five_complete": False,
        "formal_results_complete": False,
        "session_id": session_id,
        "session_manifest_sha256": session_manifest_sha256,
        "runtime_source_sha256": session["runtime_source_sha256"],
        "native_source_sha256": session["native_source_sha256"],
        "resource_manifest_path": context["resource_manifest_path"],
        "resource_manifest_sha256": resource_digest,
        "preflight_path": str(preflight_path.relative_to(root)),
        "preflight_sha256": sha256_file(preflight_path),
        "independent_verification": not failed_datasets and all(cell.get("independent_verification") is True for cell in cells),
        "cells": cells,
        "created_unix_ns": time.time_ns(),
    }
    subset_path = session_dir / "subset_validation.json"
    if subset_path.exists() or subset_path.is_symlink():
        raise ChainError("subset validation output already exists; refusing stale-result reuse")
    atomic_write_json(subset_path, payload, root)
    return {"path": str(subset_path.relative_to(root)), "sha256": sha256_file(subset_path), "status": status}


def _phase_marker(payload: Mapping[str, Any]) -> None:
    print("ZKGPT_LOCAL_FORMAL_STATE=" + json.dumps(payload, sort_keys=True, separators=(",", ":")), flush=True)


def _set_session_stage(root: Path, session_id: str, stage: str) -> None:
    status_path = root / "results" / "formal" / session_id / "session_status.json"
    if not status_path.is_file() or status_path.is_symlink():
        return
    status = load_json(status_path, "zkGPT session status")
    if not isinstance(status, dict) or status.get("session_id") != session_id:
        raise ChainError("zkGPT session status identity changed during local orchestration")
    status["active_stage"] = stage
    status["updated_unix_ns"] = time.time_ns()
    atomic_write_json(status_path, status, root)


def _record_session_stage(root: Path, session_id: str, stage: str) -> None:
    status_path = root / "results" / "formal" / session_id / "session_status.json"
    if not status_path.is_file() or status_path.is_symlink():
        raise ChainError("zkGPT local stage completion has no session status")
    status = load_json(status_path, "zkGPT session status")
    if not isinstance(status, dict) or status.get("session_id") != session_id:
        raise ChainError("zkGPT session status identity changed after a local phase")
    completed = status.setdefault("completed_stages", [])
    stage_status = status.setdefault("stage_status", {})
    if not isinstance(completed, list) or not isinstance(stage_status, dict):
        raise ChainError("zkGPT session stage history is malformed")
    if stage not in completed:
        completed.append(stage)
    stage_status[stage] = "PASS"
    status["updated_unix_ns"] = time.time_ns()
    atomic_write_json(status_path, status, root)


def _set_owner_control_stage(root: Path, session_id: str, stage: str, dataset: str | None = None) -> None:
    _update_owner_record(root, active_stage=stage, active_dataset=dataset, stage_nonce="", status="RUNNING")
    _set_session_stage(root, session_id, stage)


def _parse_stage_marker(stdout_path: Path, phase: str, session_id: str, dataset: str | None) -> Dict[str, Any]:
    marker_prefix = "ZKGPT_LOCAL_FORMAL_STATE="
    try:
        lines = [
            line[len(marker_prefix) :].strip()
            for line in stdout_path.read_text(encoding="utf-8", errors="replace").splitlines()
            if line.startswith(marker_prefix)
        ]
    except OSError as exc:
        raise ChainError(f"cannot read local formal phase marker: {exc}") from exc
    if len(lines) != 1:
        raise ChainError(f"zkGPT {phase} phase did not emit exactly one state marker")
    try:
        payload = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise ChainError(f"zkGPT {phase} phase state marker is malformed") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("phase") != phase
        or payload.get("status") != "PASS"
        or payload.get("session_id") != session_id
        or (phase == "cell" and payload.get("dataset") != dataset)
        or (phase != "cell" and payload.get("dataset") is not None)
    ):
        raise ChainError(f"zkGPT {phase} phase marker has the wrong local session or stage")
    return payload


def _launch_owner_child(root: Path, lock_stream: Any, command: Sequence[str], env: Mapping[str, str], stdout_path: Path, stderr_path: Path, label: str, extra_fds: Sequence[int] = ()) -> int:
    return run_logged(command, root, stdout_path, stderr_path, label, extra_env=env, isolate_process_group=True, pass_fds=(lock_stream.fileno(), *extra_fds))


def _run_owner_phase(
    root: Path, lock_stream: Any, session_id: str, session_dir: Path, phase: str, dataset: str | None, owner_token: str, owner_start_ticks: int, logs: Path, owner_approved_cells: Mapping[str, Mapping[str, Any]] | None = None
) -> Dict[str, Any]:
    dataset_scope = canonical_dataset_scope(_current_owner_record(root).get("dataset_scope"))
    nonce = secrets.token_hex(16)
    _update_owner_record(root, active_stage=phase, active_dataset=dataset, stage_nonce=nonce, status="RUNNING")
    _set_session_stage(root, session_id, f"phase:{phase}" if dataset is None else f"phase:{phase}:{dataset}")
    env = {
        _OWNER_LOCK_FD_ENV: str(lock_stream.fileno()),
        _OWNER_TOKEN_ENV: owner_token,
        _OWNER_PID_ENV: str(os.getpid()),
        _OWNER_START_ENV: str(owner_start_ticks),
        _OWNER_STAGE_ENV: phase,
        _OWNER_DATASET_ENV: dataset or "",
        _OWNER_STAGE_NONCE_ENV: nonce,
        _OWNER_SESSION_ENV: session_id,
        _OWNER_DATASET_SCOPE_ENV: json.dumps(list(dataset_scope), separators=(",", ":")),
        "ZKGPT_FORMAL_SESSION_ID": session_id,
    }
    command = [sys.executable, str(Path(__file__).resolve()), "--internal-phase", phase, "--session-dir", str(session_dir)]
    if dataset is not None:
        command.extend(["--dataset", dataset])
    if (phase == "finalize") != (owner_approved_cells is not None):
        raise ChainError("owner-approved cells are required only for finalize")
    approval_file = None
    approval_fd: int | None = None
    suffix = f"-{dataset}" if dataset else ""
    stdout_path = logs / f"{phase}{suffix}.stdout.log"
    stderr_path = logs / f"{phase}{suffix}.stderr.log"
    try:
        if owner_approved_cells is not None:
            if set(owner_approved_cells) != set(DATASETS):
                raise ChainError("owner cannot finalize without five approved cell receipts")
            approval_payload = json.dumps({"schema": OWNER_APPROVED_CELLS_SCHEMA, "session_id": session_id,
                                           "cells": owner_approved_cells}, sort_keys=True, separators=(",", ":")).encode("utf-8")
            if len(approval_payload) > OWNER_APPROVED_CELLS_LIMIT_BYTES:
                raise ChainError("owner-approved cell map exceeds its fixed handoff limit")
            approval_file = tempfile.TemporaryFile(mode="w+b")
            approval_file.write(approval_payload)
            approval_file.flush()
            approval_info = os.fstat(approval_file.fileno())
            if not stat.S_ISREG(approval_info.st_mode) or approval_info.st_nlink != 0:
                raise ChainError("owner approval handoff is not an anonymous regular file")
            approval_fd = os.open(f"/proc/self/fd/{approval_file.fileno()}", os.O_RDONLY)
            if fcntl.fcntl(approval_fd, fcntl.F_GETFL) & os.O_ACCMODE != os.O_RDONLY:
                raise ChainError("owner approval handoff is not read-only")
            os.lseek(approval_fd, 0, os.SEEK_SET)
            command.extend(["--owner-approved-cells-fd", str(approval_fd)])
        returncode = _launch_owner_child(root, lock_stream, command, env, stdout_path, stderr_path, f"local-{phase}{suffix}", () if approval_fd is None else (approval_fd,))
    finally:
        if approval_fd is not None:
            os.close(approval_fd)
        if approval_file is not None:
            approval_file.close()
    if returncode != 0:
        raise ChainError(f"zkGPT local {phase} phase exited with code {returncode}; see {stderr_path}")
    marker = _parse_stage_marker(stdout_path, phase, session_id, dataset)
    _record_session_stage(root, session_id, phase if dataset is None else f"cell:{dataset}")
    record = _current_owner_record(root)
    completed = record.get("completed_stages")
    if not isinstance(completed, list):
        raise ChainError("zkGPT local owner stage history is malformed")
    completed.append(phase if dataset is None else f"cell:{dataset}")
    _update_owner_record(root, active_stage="between-stages", active_dataset=None, stage_nonce="", completed_stages=completed)
    return marker


def _run_official(root: Path, dataset_scope: Sequence[str] = DATASETS) -> int:
    dataset_scope = canonical_dataset_scope(dataset_scope)
    formal_child_environment()
    with formal_run_lock(root) as lock_stream:
        install_process_signal_cleanup()
        recovered_session = _recover_stale_owner(root)
        session_id = secrets.token_hex(16)
        session_dir = root / "results" / "formal" / session_id
        if session_dir.exists() or session_dir.is_symlink():
            raise ChainError("fresh zkGPT local session id unexpectedly already exists")
        lock_info = os.fstat(lock_stream.fileno())
        if not stat.S_ISREG(lock_info.st_mode) or lock_info.st_nlink != 1:
            raise ChainError("zkGPT local formal lock is not a unique regular file")
        owner_token = secrets.token_hex(32)
        owner_start_ticks = _proc_start_ticks(os.getpid())
        owner_record = {
            "schema": LOCAL_OWNER_SCHEMA,
            "baseline": "zkgpt",
            "status": "RUNNING",
            "session_id": session_id,
            "owner_pid": os.getpid(),
            "owner_start_ticks": owner_start_ticks,
            "owner_token_sha256": hashlib.sha256(owner_token.encode("ascii")).hexdigest(),
            "lock_device": lock_info.st_dev,
            "lock_inode": lock_info.st_ino,
            "active_stage": "starting",
            "active_dataset": None,
            "stage_nonce": "",
            "dataset_scope": list(dataset_scope),
            "completed_stages": [],
            "recovered_previous_session_id": recovered_session or None,
            "started_unix_ns": time.time_ns(),
        }
        _write_owner_record(root, owner_record)
        os.environ["ZKGPT_FORMAL_SESSION_ID"] = session_id
        logs = root / "baseline-local" / "formal-owner-logs" / session_id
        try:
            ensure_directory(logs, "zkGPT local owner logs", root)
            prepare_marker = _run_owner_phase(root, lock_stream, session_id, session_dir, "prepare", None, owner_token, owner_start_ticks, logs)
            if prepare_marker.get("dataset_scope") != list(dataset_scope):
                raise ChainError("zkGPT prepare phase marker does not match the requested dataset scope")
            os.environ["ZKGPT_FORMAL_SESSION_ID"] = session_id
            _session, _preflight, _resource, context, _preflight_path, _manifest_sha = _load_prepared(root, session_id, session_dir)
            profile = context["local_capacity_profile"]
            if prepare_marker.get("session_dir") != str(session_dir) or prepare_marker.get("local_capacity_profile_sha256") != context.get(
                "local_capacity_profile_sha256"
            ):
                raise ChainError("zkGPT prepare phase marker does not match the source-bound local context")
            failed_datasets: Dict[str, str] = {}
            owner_approved_cells: Dict[str, Dict[str, str]] = {}
            for dataset in dataset_scope:
                try:
                    _set_owner_control_stage(root, session_id, f"capacity-admission:before-cell:{dataset}", dataset)
                    cell_capacity = profile["datasets"].get(dataset)
                    if not isinstance(cell_capacity, dict) or cell_capacity.get("status") != "READY":
                        reason = cell_capacity.get("reason") if isinstance(cell_capacity, dict) else None
                        raise ChainError(f"preflight capacity BLOCKED for {dataset}: {reason if isinstance(reason, str) else 'missing capacity profile'}")
                    _capacity_gate(root, session_id, session_dir, profile, (dataset,), f"before-cell:{dataset}")
                    cell_marker = _run_owner_phase(root, lock_stream, session_id, session_dir, "cell", dataset, owner_token, owner_start_ticks, logs)
                    receipt_path = session_dir / "receipts" / f"{dataset}.json"
                    receipt = load_json(receipt_path, f"zkGPT {dataset} cell receipt")
                    expected_receipt_path = relative_path(root, receipt_path, "zkGPT cell receipt")
                    if (
                        not isinstance(receipt, dict)
                        or receipt.get("status") != "VERIFY_OK"
                        or receipt.get("session_id") != session_id
                        or receipt.get("dataset") != dataset
                        or receipt.get("independent_verification") is not True
                        or cell_marker.get("receipt_path") != expected_receipt_path
                        or cell_marker.get("receipt_sha256") != sha256_file(receipt_path)
                        or cell_marker.get("proof_sha256") != receipt.get("proof_sha256")
                        or cell_marker.get("verification_report_path") != receipt.get("verification_report_path")
                        or cell_marker.get("verification_report_sha256") != receipt.get("verification_report_sha256")
                    ):
                        raise ChainError(f"zkGPT local runner did not receive a valid independent-verification receipt for {dataset}")
                    _run_relative, run_path = run_directory(root, dataset, session_id)
                    actual_retained = _tree_regular_bytes(run_path)
                    retained_bound = profile["datasets"][dataset]["cell_retained_artifact_bytes_upper_bound"]
                    if actual_retained > retained_bound:
                        raise ChainError(f"zkGPT retained output exceeds its source-derived bound for {dataset}: {actual_retained} > {retained_bound}")
                    owner_approved_cells[dataset] = {
                        "dataset": dataset,
                        "receipt_path": expected_receipt_path,
                        "receipt_sha256": cell_marker["receipt_sha256"],
                        "result_path": receipt["result_path"],
                        "result_sha256": receipt["result_sha256"],
                        "execution_path": receipt["execution_path"],
                        "execution_sha256": receipt["execution_sha256"],
                        "proof_path": receipt["proof_path"],
                        "proof_sha256": receipt["proof_sha256"],
                        "verification_report_path": receipt["verification_report_path"],
                        "verification_report_sha256": receipt["verification_report_sha256"],
                    }
                except Exception as exc:
                    reason = f"{type(exc).__name__}: {exc}"
                    failed_datasets[dataset] = reason
                    _set_status(root, session_id, "RUNNING", f"cell-failed:{dataset}")
                    status_path = formal_session_status_path(root, session_id)
                    status_value = load_json(status_path, "zkGPT session status")
                    if not isinstance(status_value, dict) or status_value.get("session_id") != session_id:
                        raise ChainError("zkGPT session identity changed while recording dataset failure")
                    stage_status = status_value.get("stage_status")
                    if not isinstance(stage_status, dict):
                        raise ChainError("zkGPT session stage status is malformed")
                    stage_status[f"cell:{dataset}"] = "FAILED"
                    dataset_failures = status_value.setdefault("dataset_failures", {})
                    if not isinstance(dataset_failures, dict):
                        raise ChainError("zkGPT dataset failure record is malformed")
                    dataset_failures[dataset] = reason
                    status_value["updated_unix_ns"] = time.time_ns()
                    atomic_write_json(status_path, status_value, root)
                    _update_owner_record(root, status="RUNNING", active_stage="between-stages", active_dataset=None, stage_nonce="")
                    print(f"ZKGPT_FORMAL_CELL_FAILED dataset={dataset} error={reason}", file=sys.stderr, flush=True)

            if failed_datasets:
                if dataset_scope != DATASETS:
                    subset_report = _write_subset_validation(
                        root, session_id, session_dir, dataset_scope, owner_approved_cells, failed_datasets
                    )
                    print(
                        f"ZKGPT_FORMAL_SUBSET_FAILED session_id={session_id} "
                        f"subset_validation={subset_report['path']} attempted={len(dataset_scope)}",
                        file=sys.stderr,
                        flush=True,
                    )
                raise ChainError(
                    f"zkGPT serial run attempted all {len(dataset_scope)} requested datasets; "
                    f"failed={len(failed_datasets)}; " + "; ".join(f"{dataset}: {reason}" for dataset, reason in failed_datasets.items())
                )
            if dataset_scope != DATASETS:
                subset_report = _write_subset_validation(
                    root, session_id, session_dir, dataset_scope, owner_approved_cells, failed_datasets
                )
                if subset_report.get("status") != "SUBSET_COMPLETE":
                    raise ChainError("zkGPT requested-scope validation did not complete successfully")
                _set_status(root, session_id, "SUBSET_COMPLETE", "subset-complete")
                _update_owner_record(
                    root,
                    status="SUBSET_COMPLETE",
                    active_stage="subset-complete",
                    active_dataset=None,
                    stage_nonce="",
                    failure="",
                    subset_validation_path=subset_report["path"],
                    subset_validation_sha256=subset_report["sha256"],
                )
                print(
                    f"ZKGPT_FORMAL_SUBSET_COMPLETE session_id={session_id} "
                    f"datasets={','.join(dataset_scope)} subset_validation={subset_report['path']} "
                    "full_five_complete=false",
                    flush=True,
                )
                return 0
            _set_owner_control_stage(root, session_id, "capacity-admission:before-finalization")
            _capacity_gate(root, session_id, session_dir, profile, [], "before-finalization")
            final_marker = _run_owner_phase(root, lock_stream, session_id, session_dir, "finalize", None, owner_token, owner_start_ticks, logs, owner_approved_cells=owner_approved_cells)
            final_path = formal_final_path(root, session_id)
            if final_marker.get("final_validation_path") != relative_path(root, final_path, "zkGPT final validation") or final_marker.get(
                "final_validation_sha256"
            ) != sha256_file(final_path):
                raise ChainError("zkGPT local finalization marker does not match the validated receipt")
            if final_marker.get("final_native_verifier_invocations") != 0 or final_marker.get("reused_cell_reports") != len(DATASETS):
                raise ChainError("zkGPT finalization did not reuse the five approved cell reports")
            final = load_json(final_path, "final validation result")
            final_verification = final.get("final_native_verification") if isinstance(final, dict) else None
            if not isinstance(final_verification, list) or len(final_verification) != len(DATASETS):
                raise ChainError("zkGPT final validation report has no per-cell verification accounting")
            for dataset, verified in zip(DATASETS, final_verification):
                approved = owner_approved_cells[dataset]
                if (not isinstance(verified, dict) or verified.get("dataset") != dataset or
                        verified.get("verification_source") != "cell_stage_independent_report" or
                        verified.get("reused_cell_report") is not True or
                        verified.get("final_native_verifier_invocations") != 0 or
                        verified.get("report_path") != approved["verification_report_path"] or
                        verified.get("report_sha256") != approved["verification_report_sha256"]):
                    raise ChainError(f"zkGPT final report is not bound to the owner-approved cell report: {dataset}")
            _update_owner_record(root, status="COMPLETE", active_stage="complete", active_dataset=None, stage_nonce="", failure="")
            print(f"ZKGPT_FORMAL_COMPLETE session_id={session_id} " f"final_validation=results/formal/{session_id}/final_validation.json", flush=True)
            return 0
        except KeyboardInterrupt:
            current_stage = str(_current_owner_record(root).get("active_stage") or "owner-interrupted")
            _mark_session_terminal(root, session_id, "INTERRUPTED", current_stage, "official zkGPT owner received an interrupt")
            try:
                _update_owner_record(
                    root,
                    status="INTERRUPTED",
                    active_stage="owner-interrupted",
                    active_dataset=None,
                    stage_nonce="",
                    failure="official zkGPT owner received an interrupt",
                )
            except Exception:
                pass
            raise
        except Exception as exc:
            subset_failed = dataset_scope != DATASETS and bool(locals().get("failed_datasets"))
            terminal_status = "SUBSET_FAILED" if subset_failed else "FAILED"
            try:
                failed_stage = str(_current_owner_record(root).get("active_stage") or "owner-failed")
            except Exception:
                failed_stage = "owner-failed"
            _mark_session_terminal(root, session_id, terminal_status, failed_stage, str(exc))
            try:
                _update_owner_record(root, status=terminal_status, active_stage="subset-failed" if subset_failed else "owner-failed", active_dataset=None, stage_nonce="", failure=str(exc))
            except Exception as status_exc:
                print(f"ZKGPT_OWNER_STATUS_WRITE_FAILED: {status_exc}", file=sys.stderr, flush=True)
            raise


def _run_internal_phase(root: Path, phase: str, session_dir_arg: str, dataset: str | None, owner_lease: Mapping[str, Any], owner_approved_cells: Mapping[str, Mapping[str, Any]] | None = None) -> int:
    session_id, session_dir = _session_dir(root, session_dir_arg, allow_empty=(phase == "prepare"))
    dataset_scope = canonical_dataset_scope(owner_lease.get("dataset_scope"))
    if phase == "cell" and dataset not in DATASETS:
        raise ChainError("cell phase requires one canonical --dataset")
    if phase == "cell" and dataset not in dataset_scope:
        raise ChainError("cell phase dataset is outside the immutable requested scope")
    if phase != "cell" and dataset is not None:
        raise ChainError("--dataset is only valid for a cell phase")
    if owner_lease.get("session_id") != session_id:
        raise ChainError("private phase owner lease is bound to another session")
    if (phase == "finalize") != (owner_approved_cells is not None):
        raise ChainError("owner-approved cells are required only for finalize")
    install_process_signal_cleanup()
    _install_parent_death_watch(root, session_id)
    if phase == "prepare":
        payload = _prepare_phase(root, session_id, session_dir, dataset_scope)
    elif phase == "cell":
        payload = _cell_phase(root, session_id, session_dir, dataset or "")
    elif phase == "finalize":
        payload = _finalize_phase(root, session_id, session_dir, owner_approved_cells or {})
    else:
        raise ChainError(f"unsupported internal phase: {phase}")
    _phase_marker(payload)
    return 0


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS)
    parser.add_argument("--internal-phase", choices=("prepare", "cell", "finalize"), help=argparse.SUPPRESS)
    parser.add_argument("--session-dir", help=argparse.SUPPRESS)
    parser.add_argument("--dataset", choices=DATASETS, help=argparse.SUPPRESS)
    parser.add_argument("--owner-approved-cells-fd", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(list(argv))
    root = baseline_root()
    if args.internal_phase is None:
        if args.session_dir or args.dataset or args.owner_approved_cells_fd is not None:
            print("private zkGPT phase arguments require --internal-phase", file=sys.stderr)
            return 2
        try:
            return _run_official(root, canonical_dataset_scope(args.datasets))
        except KeyboardInterrupt:
            print("ZKGPT_FORMAL_INTERRUPTED: local owner stopped its active stage group", file=sys.stderr, flush=True)
            return 130
        except Exception as exc:
            print(f"ZKGPT_FORMAL_FAILED: {exc}", file=sys.stderr, flush=True)
            return 2
    owner_lease: Mapping[str, Any] | None = None
    session_id = ""
    try:
        if not args.session_dir:
            raise ChainError("internal formal phase requires --session-dir")
        if args.datasets is not None:
            raise ChainError("--datasets is only valid for the official owner phase")
        session_id, _session_dir_path = _session_dir(root, args.session_dir, allow_empty=(args.internal_phase == "prepare"))
        if args.internal_phase == "cell" and args.dataset not in DATASETS:
            raise ChainError("cell phase requires one canonical --dataset")
        if args.internal_phase != "cell" and args.dataset is not None:
            raise ChainError("--dataset is only valid for a cell phase")
        if (args.internal_phase == "finalize") != (args.owner_approved_cells_fd is not None):
            raise ChainError("only the private finalize phase accepts its inherited owner approval FD")
        owner_lease = _validate_owner_lease(root, args.internal_phase, session_id, args.dataset)
        owner_approved_cells = None if args.owner_approved_cells_fd is None else _read_owner_approved_cells_fd(root, args.owner_approved_cells_fd, session_id)
        return _run_internal_phase(root, args.internal_phase, args.session_dir, args.dataset, owner_lease, owner_approved_cells)
    except KeyboardInterrupt:
        if owner_lease is not None:
            _mark_session_terminal(root, session_id, "INTERRUPTED", args.internal_phase or "", "private zkGPT phase was interrupted")
        print("ZKGPT_INTERNAL_PHASE_INTERRUPTED: owned stage group was stopped", file=sys.stderr, flush=True)
        return 130
    except Exception as exc:
        if owner_lease is not None and args.internal_phase != "cell":
            _mark_session_terminal(root, session_id, "FAILED", args.internal_phase or "", str(exc))
        print(f"ZKGPT_INTERNAL_PHASE_FAILED phase={args.internal_phase}: {exc}", file=sys.stderr, flush=True)
        return 2
    finally:
        if args.owner_approved_cells_fd is not None:
            try:
                os.close(args.owner_approved_cells_fd)
            except OSError:
                pass


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
