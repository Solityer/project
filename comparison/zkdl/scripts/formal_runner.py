#!/usr/bin/env python3
"""Own the baseline-local five-dataset zkDL formal lifecycle."""
from __future__ import annotations

import fcntl
import hashlib
import hmac
import json
import os
import re
import secrets
import signal
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import formal_phase
import formal_provenance

ROOT = Path(__file__).resolve().parent.parent
DATASETS = formal_phase.DATASETS
SESSION_RE = re.compile(r"formal_([0-9a-f]{32})\Z")
OWNER_TOKEN_ENV = "ZKDL_FORMAL_OWNER_TOKEN"
LOCK_FD_ENV = "ZKDL_FORMAL_LOCK_FD"


class RunnerError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _boot_id() -> str:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
    except OSError as exc:
        raise RunnerError("cannot identify the current boot for the local owner lease") from exc


def _write_json(path: Path, document: dict[str, Any]) -> None:
    if path.is_symlink() or path.parent.is_symlink() or not path.parent.is_dir():
        raise RunnerError(f"refusing to write local runner state through a symlink or missing directory: {path}")
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(document, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()


def _mark_stale_sessions(runs: Path) -> None:
    for session in runs.iterdir():
        match = SESSION_RE.fullmatch(session.name)
        if match is None or session.is_symlink() or not session.is_dir():
            continue
        state_path = session / "state.json"
        if state_path.is_symlink() or not state_path.is_file():
            continue
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (not isinstance(state, dict) or state.get("schema") != "zkdl_formal_runner_state_v1" or
                state.get("baseline") != "zkdl" or state.get("session_id") != match.group(1) or
                state.get("status") != "RUNNING"):
            continue
        state["interrupted_stage"] = state.get("stage", "unknown")
        state["status"] = "INTERRUPTED"
        state["stage"] = "interrupted"
        state["failure"] = "previous owner lease ended before the run recorded completion"
        state["finished_at_utc"] = _now()
        _write_json(state_path, state)
        print(f"zkdl_formal_stale_session=INTERRUPTED session={session.name}", flush=True)


def _acquire_local_lock(runs: Path) -> tuple[int, Path, os.stat_result]:
    if runs.is_symlink() or not runs.is_dir():
        raise RunnerError(f"zkDL runs directory is not a baseline-local directory: {runs}")
    lock_path = runs / ".formal.lock"
    lock_fd: int | None = None
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
        lock_stat = os.fstat(lock_fd)
        if not stat.S_ISREG(lock_stat.st_mode) or lock_path.is_symlink():
            raise RunnerError("zkDL formal lock is not a regular baseline-local file")
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        if lock_fd is not None:
            os.close(lock_fd)
        raise RunnerError("another zkDL formal run currently owns the baseline-local lock") from exc
    except OSError as exc:
        if lock_fd is not None:
            os.close(lock_fd)
        raise RunnerError(f"cannot acquire the zkDL baseline-local formal lock: {exc}") from exc
    except RunnerError:
        if lock_fd is not None:
            os.close(lock_fd)
        raise
    assert lock_fd is not None
    return lock_fd, lock_path, lock_stat


def _phase_marker(output: str, phase: str, dataset: str | None, session_id: str) -> dict[str, Any]:
    marker = None
    for line in reversed(output.splitlines()):
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and candidate.get("zkdl_phase") == phase:
            marker = candidate
            break
    if not isinstance(marker, dict) or marker.get("status") != "PASS":
        raise RunnerError(f"zkDL {phase} phase exited without its exact success marker")
    if phase in ("prepare", "finalize") and marker.get("session_id") != session_id:
        raise RunnerError(f"zkDL {phase} phase returned a mismatched session identity")
    if phase == "cell" and marker.get("dataset") != dataset:
        raise RunnerError(f"zkDL cell phase returned a mismatched dataset identity")
    return marker


def _marker_summary(phase: str, marker: dict[str, Any]) -> dict[str, Any]:
    keys = ("dataset", "prepare_receipt_path", "prepare_receipt_sha256", "receipt_path", "receipt_sha256",
            "final_validation_path", "final_validation_sha256", "proof_sha256", "verification_sha256")
    summary: dict[str, Any] = {"phase": phase, "status": "PASS"}
    summary.update({key: marker[key] for key in keys if key in marker})
    return summary


def _effective_host_memory_available() -> tuple[int, str]:
    system_available = 0
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            if line.startswith("MemAvailable:"):
                system_available = int(line.split()[1]) * 1024
                break
    except (OSError, ValueError, IndexError):
        pass
    if system_available <= 0:
        raise RunnerError("cannot determine live host MemAvailable")
    effective = system_available
    method = "system_memavailable"
    try:
        cgroup_lines = Path("/proc/self/cgroup").read_text(encoding="ascii").splitlines()
    except OSError:
        cgroup_lines = []
    v2_path = next((line.split(":", 2)[2] for line in cgroup_lines if line.startswith("0::")), None)
    if v2_path is not None:
        cgroup_root = Path("/sys/fs/cgroup")
        base = cgroup_root / v2_path.lstrip("/")
        for candidate in (base, *base.parents):
            try:
                candidate.relative_to(cgroup_root)
            except ValueError:
                continue
            try:
                limit_text = (candidate / "memory.max").read_text(encoding="ascii").strip()
                if limit_text == "max":
                    continue
                limit = int(limit_text)
                current = int((candidate / "memory.current").read_text(encoding="ascii").strip())
            except (OSError, ValueError):
                continue
            if limit > 0:
                effective = min(effective, max(0, limit - current))
                method = "min_system_and_cgroup_v2_available"
    else:
        memory_group = next((line.split(":", 2)[2] for line in cgroup_lines if ":memory:" in line), None)
        if memory_group is not None:
            cgroup_root = Path("/sys/fs/cgroup/memory")
            base = cgroup_root / memory_group.lstrip("/")
            for candidate in (base, *base.parents):
                try:
                    candidate.relative_to(cgroup_root)
                except ValueError:
                    continue
                try:
                    limit = int((candidate / "memory.limit_in_bytes").read_text(encoding="ascii").strip())
                    current = int((candidate / "memory.usage_in_bytes").read_text(encoding="ascii").strip())
                except (OSError, ValueError):
                    continue
                if 0 < limit < (1 << 60):
                    effective = min(effective, max(0, limit - current))
                    method = "min_system_and_cgroup_v1_available"
    return effective, method


def _session_file_bytes(session: Path) -> int:
    total = 0
    for path in session.rglob("*"):
        if path.is_symlink():
            raise RunnerError(f"zkDL session contains a symlink during capacity accounting: {path}")
        if path.is_file():
            size = path.stat(follow_symlinks=False).st_size
            if size < 0:
                raise RunnerError(f"zkDL session has an invalid file size: {path}")
            total += size
    return total


def _filesystem_device(path: Path, *, allow_missing: bool = False) -> int:
    cursor = path
    if path.is_symlink():
        raise RunnerError(f"zkDL formal filesystem path is a symlink: {path}")
    while not cursor.exists():
        if cursor == cursor.parent:
            raise RunnerError(f"cannot find an existing parent filesystem for {path}")
        cursor = cursor.parent
    if cursor.is_symlink():
        raise RunnerError(f"zkDL formal filesystem ancestor is a symlink: {cursor}")
    if not cursor.is_dir():
        if allow_missing:
            raise RunnerError(f"zkDL formal output filesystem parent is not a directory: {cursor}")
        raise RunnerError(f"zkDL formal filesystem path is not a directory: {cursor}")
    try:
        return cursor.stat(follow_symlinks=False).st_dev
    except OSError as exc:
        raise RunnerError(f"cannot inspect zkDL formal filesystem path {cursor}: {exc}") from exc


def _checked_capacity_int(value: Any, label: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if type(value) is not int or value < minimum or value > (1 << 64) - 1:
        raise RunnerError(f"zkDL capacity profile has an invalid {label}")
    return value


def _remaining_capacity_requirements(profile: dict[str, Any], completed: tuple[str, ...]) -> dict[str, int]:
    if profile.get("schema") != "zkdl_formal_capacity_profile_v3" or profile.get("baseline") != "zkdl":
        raise RunnerError("zkDL local capacity profile schema/baseline mismatch")
    if profile.get("backend") != "cpu" or profile.get("memory_accounting_kind") != "host-only" or profile.get("device") is not None:
        raise RunnerError("zkDL capacity profile does not match the source-bound CPU-only formal path")
    methods = profile.get("bound_methods")
    if (not isinstance(methods, dict) or
            methods.get("additional_disk_peak") != "source_derived_upper_bound" or
            methods.get("host_peak") != "source_derived_upper_bound" or
            methods.get("finalization_host_peak") != "srs_manifest_schema_python_heap_upper_bound"):
        raise RunnerError("zkDL capacity profile lacks source-derived host/disk bounds")
    if tuple(DATASETS[:len(completed)]) != completed:
        raise RunnerError("zkDL capacity recheck is not following the canonical dataset sequence")
    datasets = profile.get("datasets")
    if not isinstance(datasets, dict) or set(datasets) != set(DATASETS):
        raise RunnerError("zkDL capacity profile does not contain exactly the five canonical datasets")
    finalization = _checked_capacity_int(profile.get("finalization_disk_bytes_upper_bound"),
                                         "finalization disk upper bound", allow_zero=True)
    remaining = DATASETS[len(completed):]
    host_required = 0
    device_required = 0
    disk_required = finalization
    future_retained_prefix = 0
    for dataset in remaining:
        item = datasets.get(dataset)
        if not isinstance(item, dict):
            raise RunnerError(f"zkDL capacity profile row is malformed for {dataset}")
        if item.get("phase_coverage") != ["setup", "witness", "commitment", "prove", "verify"]:
            raise RunnerError(f"zkDL capacity profile phase coverage is malformed for {dataset}")
        route_source_domain = _checked_capacity_int(
            item.get("native_route_source_domain"), f"{dataset} route selector source domain")
        route_factor_elements = _checked_capacity_int(
            item.get("native_route_selector_factor_elements"),
            f"{dataset} route selector factor elements")
        if route_source_domain > (1 << 30) or route_factor_elements > route_source_domain + 1:
            raise RunnerError(f"zkDL route selector factor bound is invalid for {dataset}")
        host_required = max(host_required, _checked_capacity_int(item.get("host_peak_bytes"),
                                                                   f"{dataset} host peak"))
        device_peak = _checked_capacity_int(item.get("device_peak_bytes"), f"{dataset} CPU device peak", allow_zero=True)
        if device_peak != 0:
            raise RunnerError(f"zkDL CPU formal cell unexpectedly requires device capacity for {dataset}")
        device_required = max(device_required, device_peak)
        retained = _checked_capacity_int(item.get("cell_retained_artifact_bytes_upper_bound"),
                                         f"{dataset} retained artifact upper bound")
        additional = _checked_capacity_int(item.get("additional_disk_peak_bytes"),
                                           f"{dataset} additional disk peak")
        if additional < retained:
            raise RunnerError(f"zkDL capacity profile disk peak is smaller than retained output for {dataset}")
        # Each cell runs serially. Already-consumed cells are represented by
        # live available space; future cells need their retained prefix plus
        # one cell's additional scratch/retained peak at a time.
        candidate_disk = future_retained_prefix + additional + finalization
        if candidate_disk > (1 << 64) - 1:
            raise RunnerError("zkDL remaining disk requirement overflows u64")
        disk_required = max(disk_required, candidate_disk)
        future_retained_prefix += retained
        if future_retained_prefix > (1 << 64) - 1:
            raise RunnerError("zkDL retained prefix requirement overflows u64")
    disk_required = max(disk_required, future_retained_prefix + finalization)
    finalization_profile = profile.get("finalization_manifest_capacity")
    if (not isinstance(finalization_profile, dict) or
            finalization_profile.get("schema") != "zkdl_srs_manifest_finalization_heap_v1" or
            not isinstance(finalization_profile.get("datasets"), dict) or
            set(finalization_profile["datasets"]) != set(DATASETS)):
        raise RunnerError("zkDL finalization manifest capacity schema/dataset set mismatch")
    finalization_dataset_bounds: list[int] = []
    for dataset in DATASETS:
        item = finalization_profile["datasets"][dataset]
        if not isinstance(item, dict):
            raise RunnerError(f"zkDL finalization manifest capacity is malformed for {dataset}")
        family_count = _checked_capacity_int(item.get("family_count"), f"{dataset} SRS family count")
        segment_count = _checked_capacity_int(item.get("segment_count"), f"{dataset} SRS segment count")
        try:
            expected = formal_phase._srs_manifest_finalization_capacity(family_count, segment_count)
        except formal_phase.PhaseError as exc:
            raise RunnerError(f"zkDL finalization manifest bound is invalid for {dataset}: {exc}") from exc
        if item != expected:
            raise RunnerError(f"zkDL finalization manifest bound does not match its fixed-schema counts for {dataset}")
        finalization_dataset_bounds.append(expected["host_peak_bytes_upper_bound"])
    finalization_host = _checked_capacity_int(profile.get("finalization_host_peak_bytes_upper_bound"),
                                              "finalization host upper bound")
    if finalization_host != max(finalization_dataset_bounds):
        raise RunnerError("zkDL finalization host bound differs from its SRS manifest schema bounds")
    return {"host_bytes": host_required, "disk_bytes": disk_required,
            "device_bytes": device_required, "finalization_host_bytes": finalization_host}


def _prepare_capacity(marker: dict[str, Any], session: Path, session_id: str) -> dict[str, Any]:
    receipt_relative = f"runs/{session.name}/phase/prepare.json"
    build_relative = f"runs/{session.name}/build_manifest.json"
    if marker.get("prepare_receipt_path") != receipt_relative or marker.get("build_manifest_path") != build_relative:
        raise RunnerError("zkDL prepare phase returned a receipt/build path outside its local session")
    receipt_path = session / "phase" / "prepare.json"
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise RunnerError("zkDL prepare receipt is missing, non-regular or symlinked")
    receipt_digest = formal_phase.sha256_file(receipt_path)
    if marker.get("prepare_receipt_sha256") != receipt_digest:
        raise RunnerError("zkDL prepare phase marker does not authenticate its receipt")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerError("zkDL prepare receipt is unreadable") from exc
    if (not isinstance(receipt, dict) or receipt.get("schema") != "zkDL_formal_prepare_receipt_v2" or
            receipt.get("baseline") != "zkdl" or receipt.get("status") != "PASS" or
            receipt.get("session_id") != session_id):
        raise RunnerError("zkDL prepare receipt identity/status mismatch")
    build_path = session / "build_manifest.json"
    if build_path.is_symlink() or not build_path.is_file():
        raise RunnerError("zkDL session build manifest is missing, non-regular or symlinked")
    build_digest = formal_phase.sha256_file(build_path)
    if (receipt.get("build_manifest_path") != build_relative or
            receipt.get("build_manifest_sha256") != build_digest or
            marker.get("build_manifest_sha256") != build_digest):
        raise RunnerError("zkDL prepare receipt/marker does not authenticate the local build manifest")
    try:
        build = formal_provenance.validate_manifest(ROOT, build_path)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise RunnerError(f"zkDL source/build provenance changed before capacity admission: {exc}") from exc
    source_digest = build.get("source_digest")
    if (receipt.get("source_digest") != source_digest or marker.get("source_digest") != source_digest):
        raise RunnerError("zkDL prepare source identity differs from the validated build")
    if (receipt.get("graph_identity", {}).get("schema") != "zkdl_canonical_graph_set_v1" or
            marker.get("graph_identity") != receipt.get("graph_identity")):
        raise RunnerError("zkDL prepare receipt graph identity is missing or mismatched")
    profile = receipt.get("capacity_profile")
    if (not isinstance(profile, dict) or marker.get("capacity_profile") != profile or
            profile.get("session_id") != session_id or profile.get("baseline") != "zkdl"):
        raise RunnerError("zkDL prepare marker/receipt capacity profile identity mismatch")
    if profile.get("schema") != "zkdl_formal_capacity_profile_v3":
        raise RunnerError("zkDL prepare capacity profile is not baseline-local")
    native_admissions = receipt.get("native_admissions")
    if not isinstance(native_admissions, dict) or set(native_admissions) != set(DATASETS):
        raise RunnerError("zkDL prepare receipt does not contain exactly five native admissions")
    profile_datasets = profile.get("datasets")
    if not isinstance(profile_datasets, dict) or set(profile_datasets) != set(DATASETS):
        raise RunnerError("zkDL prepare capacity profile does not contain exactly five datasets")
    for dataset in DATASETS:
        admission = native_admissions[dataset]
        row = profile_datasets[dataset]
        if (not isinstance(admission, dict) or
                admission.get("schema") != "zkdl_native_relation_admission_v2" or
                admission.get("status") != "PASS" or admission.get("dataset") != dataset or
                not isinstance(row, dict)):
            raise RunnerError(f"zkDL native admission identity/schema mismatch for {dataset}")
        source_domain = _checked_capacity_int(
            admission.get("native_maximum_route_source_domain"),
            f"{dataset} native route source domain")
        factor_elements = _checked_capacity_int(
            admission.get("native_maximum_route_selector_factor_elements"),
            f"{dataset} native route selector factor elements")
        if (source_domain > (1 << 30) or factor_elements > source_domain + 1 or
                row.get("native_route_source_domain") != source_domain or
                row.get("native_route_selector_factor_elements") != factor_elements):
            raise RunnerError(f"zkDL native route selector factor profile mismatch for {dataset}")
    if type(profile.get("fs_device_id")) is not int or profile["fs_device_id"] != session.stat(follow_symlinks=False).st_dev:
        raise RunnerError("zkDL prepare capacity profile filesystem identity changed")
    if _session_file_bytes(session) < _checked_capacity_int(profile.get("preexisting_baseline_artifact_bytes"),
                                                            "preexisting artifact byte count", allow_zero=True):
        raise RunnerError("zkDL prepare session is smaller than its recorded preexisting artifact size")
    _remaining_capacity_requirements(profile, ())
    return profile


def _capacity_gate(profile: dict[str, Any], completed: tuple[str, ...], session: Path, session_id: str,
                   prepare_receipt_sha256: str, build_manifest_sha256: str) -> dict[str, Any]:
    requirements = _remaining_capacity_requirements(profile, completed)
    receipt_path = session / "phase" / "prepare.json"
    build_path = session / "build_manifest.json"
    if (receipt_path.is_symlink() or not receipt_path.is_file() or
            formal_phase.sha256_file(receipt_path) != prepare_receipt_sha256):
        raise RunnerError("zkDL prepare receipt changed after capacity profile authentication")
    if (build_path.is_symlink() or not build_path.is_file() or
            formal_phase.sha256_file(build_path) != build_manifest_sha256):
        raise RunnerError("zkDL local source/build receipt changed after capacity profile authentication")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        build = formal_provenance.validate_manifest(ROOT, build_path)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise RunnerError(f"zkDL per-cell capacity source/session recheck failed: {exc}") from exc
    if (receipt.get("session_id") != session_id or receipt.get("capacity_profile") != profile or
            receipt.get("build_manifest_sha256") != build_manifest_sha256 or
            receipt.get("source_digest") != build.get("source_digest")):
        raise RunnerError("zkDL per-cell capacity profile, source or build identity changed")
    expected_device = _checked_capacity_int(profile.get("fs_device_id"), "filesystem device id", allow_zero=True)
    for label, path, allow_missing in (
            ("runs", ROOT / "runs", False), ("session", session, False),
            ("results", ROOT / "results", True)):
        device = _filesystem_device(path, allow_missing=allow_missing)
        if device != expected_device:
            raise RunnerError(f"zkDL {label} path is on a different filesystem from the admitted session")
    try:
        statvfs = os.statvfs(session)
        disk_available = statvfs.f_bavail * statvfs.f_frsize
    except OSError as exc:
        raise RunnerError(f"cannot inspect live zkDL session filesystem capacity: {exc}") from exc
    host_available, host_method = _effective_host_memory_available()
    required_host = max(requirements["host_bytes"], requirements["finalization_host_bytes"])
    if host_available < required_host:
        raise RunnerError(f"zkDL host capacity blocked before stage: need={required_host} available={host_available} method={host_method}")
    if disk_available < requirements["disk_bytes"]:
        raise RunnerError(f"zkDL disk capacity blocked before stage: need={requirements['disk_bytes']} available={disk_available}")
    if profile.get("backend") != "cpu" or profile.get("device") is not None or requirements["device_bytes"] != 0:
        raise RunnerError("zkDL live device admission disagrees with the source-bound CPU-only formal path")
    return {
        "completed_datasets": list(completed),
        "required_host_bytes": required_host,
        "finalization_host_peak_bytes_upper_bound": requirements["finalization_host_bytes"],
        "available_host_bytes": host_available,
        "host_availability_method": host_method,
        "required_disk_bytes": requirements["disk_bytes"],
        "available_disk_bytes": disk_available,
        "required_device_bytes": requirements["device_bytes"],
        "backend": "cpu",
        "checked_at_utc": _now(),
    }


def _run_phase(lock_fd: int, token: str, session: Path, session_id: str, state: dict[str, Any],
               phase: str, dataset: str | None = None) -> dict[str, Any]:
    stage = f"cell:{dataset}" if phase == "cell" else phase
    state["stage"] = stage
    state["updated_at_utc"] = _now()
    _write_json(session / "state.json", state)
    command = [sys.executable, str(ROOT / "scripts" / "formal_phase.py"), "--internal-phase", phase,
               "--session-dir", str(session)]
    if dataset is not None:
        command.extend(("--dataset", dataset))
    env = {key: value for key, value in os.environ.items() if not key.startswith("COMPARISON_FORMAL_")}
    env[OWNER_TOKEN_ENV] = token
    env[LOCK_FD_ENV] = str(lock_fd)
    print(f"zkdl_formal_stage=START stage={stage} session={session_id}", flush=True)
    output = formal_phase._run(command, env=env, capture_output=True, pass_fds=(lock_fd,))
    marker = _phase_marker(output, phase, dataset, session_id)
    state.setdefault("completed_phases", []).append(_marker_summary(phase, marker))
    state["updated_at_utc"] = _now()
    _write_json(session / "state.json", state)
    print(f"zkdl_formal_stage=PASS stage={stage} session={session_id}", flush=True)
    return marker


def run() -> int:
    runs = ROOT / "runs"
    if runs.is_symlink():
        raise RunnerError(f"zkDL runs directory is unsafe: {runs}")
    if not runs.exists():
        runs.mkdir(mode=0o700)
    if not runs.is_dir():
        raise RunnerError(f"zkDL runs path is not a baseline-local directory: {runs}")
    lock_fd, _lock_path, lock_stat = _acquire_local_lock(runs)
    session: Path | None = None
    state: dict[str, Any] | None = None
    try:
        _mark_stale_sessions(runs)
        session_id = secrets.token_hex(16)
        session = runs / f"formal_{session_id}"
        session.mkdir(mode=0o700, exist_ok=False)
        token = secrets.token_hex(32)
        owner_pid = os.getpid()
        owner = {"pid": owner_pid, "boot_id": _boot_id(), "start_ticks": str(formal_phase._proc_start_ticks(owner_pid))}
        state = {
            "schema": "zkdl_formal_runner_state_v1",
            "baseline": "zkdl",
            "session_id": session_id,
            "session_dir": session.relative_to(ROOT).as_posix(),
            "status": "RUNNING",
            "stage": "prepare",
            "owner": owner,
            "lock": {"path": "runs/.formal.lock", "device": lock_stat.st_dev, "inode": lock_stat.st_ino},
            "internal_owner_token_sha256": hashlib.sha256(token.encode("ascii")).hexdigest(),
            "started_at_utc": _now(),
            "updated_at_utc": _now(),
            "completed_phases": [],
        }
        _write_json(session / "state.json", state)
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(signum, formal_phase._signal_handler)
        prepare_marker = _run_phase(lock_fd, token, session, session_id, state, "prepare")
        profile = _prepare_capacity(prepare_marker, session, session_id)
        state["capacity"] = {
            "profile_sha256": hashlib.sha256(json.dumps(profile, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "profile_schema": profile["schema"],
            "backend": profile["backend"],
            "prepare_receipt_sha256": prepare_marker["prepare_receipt_sha256"],
            "build_manifest_sha256": prepare_marker["build_manifest_sha256"],
            "source_digest": prepare_marker["source_digest"],
            "completed_retained_prefix_actual_bytes": 0,
            "completed_retained_prefix_artifact_upper_bytes": 0,
            "completed_cells": [],
        }
        _write_json(session / "state.json", state)
        completed: list[str] = []
        for dataset in DATASETS:
            capacity_check = _capacity_gate(profile, tuple(completed), session, session_id,
                                             prepare_marker["prepare_receipt_sha256"],
                                             prepare_marker["build_manifest_sha256"])
            state["capacity"]["last_live_check"] = capacity_check
            _write_json(session / "state.json", state)
            before_bytes = _session_file_bytes(session)
            _run_phase(lock_fd, token, session, session_id, state, "cell", dataset)
            after_bytes = _session_file_bytes(session)
            retained_actual = max(0, after_bytes - before_bytes)
            retained_upper = _checked_capacity_int(
                profile["datasets"][dataset].get("cell_retained_artifact_bytes_upper_bound"),
                f"{dataset} retained artifact upper bound")
            if retained_actual > retained_upper:
                raise RunnerError(f"zkDL {dataset} retained artifacts exceeded the admitted bound: actual={retained_actual} upper={retained_upper}")
            completed.append(dataset)
            capacity_state = state["capacity"]
            capacity_state["completed_cells"] = list(completed)
            capacity_state["completed_retained_prefix_actual_bytes"] += retained_actual
            capacity_state["completed_retained_prefix_artifact_upper_bytes"] += retained_upper
            capacity_state["last_cell_retained_actual_bytes"] = retained_actual
            capacity_state["last_cell_retained_upper_bytes"] = retained_upper
            _write_json(session / "state.json", state)
        capacity_check = _capacity_gate(profile, tuple(completed), session, session_id,
                                         prepare_marker["prepare_receipt_sha256"],
                                         prepare_marker["build_manifest_sha256"])
        state["capacity"]["last_live_check"] = capacity_check
        _write_json(session / "state.json", state)
        final_marker = _run_phase(lock_fd, token, session, session_id, state, "finalize")
        state["status"] = "PASS"
        state["stage"] = "complete"
        state["failure"] = ""
        state["final_validation_path"] = final_marker["final_validation_path"]
        state["final_validation_sha256"] = final_marker["final_validation_sha256"]
        state["finished_at_utc"] = _now()
        state["updated_at_utc"] = state["finished_at_utc"]
        _write_json(session / "state.json", state)
        print(f"zkdl_formal_run=PASS session={session_id} session_dir={session.relative_to(ROOT).as_posix()}", flush=True)
        return 0
    except KeyboardInterrupt as exc:
        if state is not None and session is not None:
            state["status"] = "INTERRUPTED"
            state["stage"] = "interrupted"
            state["failure"] = str(exc) or "formal run interrupted"
            state["finished_at_utc"] = _now()
            state["updated_at_utc"] = state["finished_at_utc"]
            _write_json(session / "state.json", state)
        print(f"zkdl_formal_run=INTERRUPTED error={exc}", file=sys.stderr, flush=True)
        return 130 if formal_phase.INTERRUPTED == signal.SIGINT else 143
    except Exception as exc:
        if state is not None and session is not None:
            state["status"] = "FAILED"
            state["failure"] = str(exc)
            state["finished_at_utc"] = _now()
            state["updated_at_utc"] = state["finished_at_utc"]
            _write_json(session / "state.json", state)
        print(f"zkdl_formal_run=FAILED error={exc}", file=sys.stderr, flush=True)
        return 2
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def main() -> int:
    if len(sys.argv) != 1:
        print("usage: bash scripts/run_five.sh", file=sys.stderr, flush=True)
        return 2
    try:
        return run()
    except RunnerError as exc:
        print(f"zkdl_formal_run=FAILED error={exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
