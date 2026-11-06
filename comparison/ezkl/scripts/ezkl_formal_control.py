#!/usr/bin/env python3
"""Detached controls for the official selected-dataset EZKL formal chain."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from formal_common import (
    DATASETS,
    FORMAL_SESSION_ID,
    ContractError,
    ROOT,
    abort,
    local_path,
    normalize_selected_datasets,
    print_error,
    selected_datasets_for_session,
)
from compatible_resume import (
    validate_capacity_admission_v2_controller_candidate,
    validate_failed_capacity_admission_resume_gate,
    validate_failed_configured_admission_resume_gate,
    validate_failed_admission_preflight_resume_gate,
    validate_failed_preflight_resume_gate,
)
from process_guardian import process_start_ticks

_SCHEMA = "ezkl.formal-controller.v1"
_SCRIPT = Path(__file__).resolve()
_COMPAT_SESSION_ID = "d3dc1ccb1cfa4a78829c306b835f300e"
_COMPAT_OUTPUT_ROOT = "runs/formal_ppi_ogbn_20261029_001"
_COMPAT_MATRIX = f"{_COMPAT_OUTPUT_ROOT}/formal_matrix.json"
_COMPAT_SELECTED_DATASETS = ("ppi", "ogbn_arxiv")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _classify_supervisor_exit(exit_code: int, *, stop_requested: bool) -> tuple[str, str]:
    """Return the log outcome and controller status for the observed child exit."""
    if exit_code == 0:
        return "success", "EXITED"
    if stop_requested and exit_code in {-signal.SIGTERM, 128 + signal.SIGTERM, 128 + signal.SIGINT}:
        return "stopped", "STOPPED"
    return "failed", "EXITED"


def _safe_local(value: str, label: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        abort(f"{label} must be a baseline-local relative path: {value}")
    cursor = ROOT
    for part in path.parts:
        if part in {"", "."}:
            continue
        cursor = cursor / part
        if cursor.is_symlink():
            abort(f"{label} path must not contain a symlink: {cursor}")
    return local_path(value, label)


def _compatible_source_review_reference(value: object, *, require_file: bool) -> str | None:
    """Return a canonical runs/check_logs reference for the approved review JSON."""
    if not isinstance(value, str) or not value:
        return None
    if not require_file:
        candidate = Path(value)
        if (
            candidate.is_absolute()
            or ".." in candidate.parts
            or candidate.as_posix() != value
            or not value.startswith("runs/check_logs/")
        ):
            return None
        return value
    selected = Path(value)
    if selected.is_absolute():
        try:
            value = selected.relative_to(ROOT).as_posix()
        except ValueError:
            return None
    if Path(value).is_absolute() or ".." in Path(value).parts:
        return None
    try:
        review_path = _safe_local(value, "compatible source review")
        relative = review_path.relative_to(ROOT).as_posix()
    except (ContractError, ValueError):
        return None
    if not relative.startswith("runs/check_logs/"):
        return None
    if require_file:
        if review_path.is_symlink() or not review_path.is_file():
            return None
        try:
            review_path.resolve(strict=True).relative_to((ROOT / "runs" / "check_logs").resolve(strict=True))
        except (OSError, ValueError):
            return None
    return relative


def _compatible_resume_options(
    args: argparse.Namespace,
    *,
    resume: bool,
    session_id: str,
    output_root: Path,
    matrix: Path,
    selected_datasets: tuple[str, ...],
    requested_datasets: tuple[str, ...] | None,
    session: dict[str, Any] | None,
) -> tuple[bool, str | None]:
    """Validate and normalize the one supported compatible detached-resume request."""
    prove_only = getattr(args, "compatible_prove_only", False)
    source_review_value = getattr(args, "compatible_source_review", None)
    if not isinstance(prove_only, bool):
        abort("--compatible-prove-only controller value must be boolean")
    if prove_only != (source_review_value is not None):
        abort("--compatible-prove-only and --compatible-source-review must be supplied together")
    if not prove_only:
        return False, None
    if not resume:
        abort("compatible prove-only is available only through resume")
    if requested_datasets is None or selected_datasets != _COMPAT_SELECTED_DATASETS:
        abort("compatible prove-only requires an explicit --datasets ppi ogbn_arxiv selection")
    if (
        session_id != _COMPAT_SESSION_ID
        or output_root.relative_to(ROOT).as_posix() != _COMPAT_OUTPUT_ROOT
        or matrix.relative_to(ROOT).as_posix() != _COMPAT_MATRIX
        or not isinstance(session, dict)
        or session.get("matrix") != _COMPAT_MATRIX
        or session.get("status") not in {"INTERRUPTED", "FAILED"}
    ):
        abort("compatible prove-only is bound to the exact unfinished baseline session, output root, and matrix")
    review_reference = _compatible_source_review_reference(source_review_value, require_file=True)
    if review_reference is None:
        abort("compatible source review must be a regular non-symlink file below runs/check_logs")
    completed = session.get("completed_cells")
    completed_ppi = False
    if completed not in (None, {}):
        checkpoint = completed.get("ppi") if isinstance(completed, dict) else None
        if (
            not isinstance(completed, dict)
            or set(completed) != {"ppi"}
            or not isinstance(checkpoint, dict)
            or set(checkpoint) != {
                "schema", "dataset", "session_id", "status", "result_sha256", "proof_sha256", "checkpointed_at_utc",
            }
            or checkpoint.get("schema") != "ezkl.formal-cell-checkpoint.v1"
            or checkpoint.get("dataset") != "ppi"
            or checkpoint.get("session_id") != session_id
            or checkpoint.get("status") != "VERIFY_OK"
            or any(
                not isinstance(checkpoint.get(name), str)
                or len(checkpoint[name]) != 64
                or any(character not in "0123456789abcdef" for character in checkpoint[name])
                for name in ("result_sha256", "proof_sha256")
            )
            or not isinstance(checkpoint.get("checkpointed_at_utc"), str)
            or not checkpoint["checkpointed_at_utc"]
        ):
            abort("compatible OGBN continuation requires exactly one authenticated-shape PPI VERIFY_OK checkpoint")
        request = session.get("compatible_resume_request")
        try:
            requested_review_sha256 = hashlib.sha256(_safe_local(review_reference, "compatible source review").read_bytes()).hexdigest()
        except OSError as exc:
            abort(f"completed-PPI continuation source review cannot be read: {exc}")
        if (
            not isinstance(request, dict)
            or request.get("session_id") != session_id
            or request.get("selected_datasets") != list(_COMPAT_SELECTED_DATASETS)
            or request.get("source_review_path") != review_reference
            or request.get("source_review_sha256") != requested_review_sha256
        ):
            abort("completed-PPI continuation must retain the exact compatible request and source-review bytes")
        completed_ppi = True
    ogbn_resume_stages = {
        "selected_dataset_native_admission_preflight",
        "selected_dataset_native_capacity_preflight",
        "ogbn_arxiv_configured_admission",
        "ogbn_arxiv_capacity_admission",
        "ogbn_arxiv_setup",
        "ogbn_arxiv_witness_prove_independent_verify",
    }
    if session.get("status") == "FAILED":
        if completed_ppi:
            if session.get("stage") not in ogbn_resume_stages:
                abort("completed-PPI continuation is limited to pending OGBN-Arxiv stages")
        elif session.get("stage") == "native_build_identity":
            validate_failed_preflight_resume_gate(session)
        elif session.get("stage") == "selected_dataset_native_capacity_preflight":
            validate_failed_admission_preflight_resume_gate(session)
        elif session.get("stage") == "ppi_configured_admission":
            validate_failed_configured_admission_resume_gate(session)
        elif session.get("stage") == "ppi_capacity_admission":
            if not validate_capacity_admission_v2_controller_candidate(
                _session_path(output_root, session_id), session, ROOT / review_reference,
            ):
                validate_failed_capacity_admission_resume_gate(session)
        else:
            abort("compatible prove-only resume is blocked after the approved preflight failure boundary")
    elif completed_ppi and session.get("stage") not in ogbn_resume_stages | {"ppi_witness_prove_independent_verify"}:
        abort("interrupted completed-PPI continuation is limited to its Verify boundary or pending OGBN-Arxiv stages")
    return True, review_reference


def _safe_mkdir(path: Path, label: str) -> None:
    for parent in (path, *path.parents):
        if parent == ROOT.parent:
            break
        if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
            abort(f"{label} path is not a regular directory: {parent}")
    path.mkdir(parents=True, exist_ok=True)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    _safe_mkdir(path.parent, "controller metadata parent")
    if path.is_symlink() or (path.exists() and not path.is_file()):
        abort(f"controller metadata path is not a regular file: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", closefd=False) as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _read_metadata(path: Path, session_id: str, output_root: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        abort(f"formal controller metadata is missing or symlinked: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        abort(f"cannot read formal controller metadata: {exc}")
    if (
        not isinstance(value, dict)
        or value.get("schema") != _SCHEMA
        or value.get("baseline") != "ezkl"
        or value.get("session_id") != session_id
        or value.get("output_root") != output_root.relative_to(ROOT).as_posix()
    ):
        abort("formal controller metadata does not match the requested session/output root")
    return value


def _proc_cmdline(pid: int) -> list[str] | None:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return None
    return [item.decode("utf-8", errors="replace") for item in raw.split(b"\0") if item]


def _proc_cwd(pid: int) -> str | None:
    try:
        return os.readlink(f"/proc/{pid}/cwd")
    except OSError:
        return None


def _proc_is_live(pid: int) -> bool:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return False
    close = raw.rfind(")")
    if close < 0:
        return False
    fields = raw[close + 2 :].split()
    return bool(fields) and fields[0] not in {"Z", "X"}


def _alive_exact(identity: object, *, expected_script: Path | None = None, session_id: str | None = None) -> bool:
    if not isinstance(identity, dict):
        return False
    pid, ticks, command = identity.get("pid"), identity.get("start_ticks"), identity.get("cmdline")
    if (
        isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0
        or isinstance(ticks, bool) or not isinstance(ticks, int) or ticks <= 0
        or not isinstance(command, list) or any(not isinstance(item, str) for item in command)
    ):
        return False
    try:
        if process_start_ticks(pid) != ticks or not _proc_is_live(pid):
            return False
    except (OSError, RuntimeError):
        return False
    actual = _proc_cmdline(pid)
    if actual is None:
        return False
    if expected_script is None:
        if actual != command:
            return False
    else:
        script = str(expected_script.resolve())
        if len(actual) < 2 or str(Path(actual[1]).resolve()) != script or actual != command:
            return False
        if session_id is not None and session_id not in actual:
            return False
    return _proc_cwd(pid) == str(ROOT.resolve())


def _send_exact_signal(identity: dict[str, Any], signum: int, *, expected_script: Path | None = None, session_id: str | None = None) -> bool:
    if not _alive_exact(identity, expected_script=expected_script, session_id=session_id):
        return False
    pid = int(identity["pid"])
    pidfd: int | None = None
    try:
        if hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"):
            pidfd = os.pidfd_open(pid, 0)
            if not _alive_exact(identity, expected_script=expected_script, session_id=session_id):
                return False
            signal.pidfd_send_signal(pidfd, signum, None, 0)
        else:
            if not _alive_exact(identity, expected_script=expected_script, session_id=session_id):
                return False
            os.kill(pid, signum)
        return True
    except ProcessLookupError:
        return False
    finally:
        if pidfd is not None:
            os.close(pidfd)


def _metadata_path(output_root: Path, session_id: str) -> Path:
    if FORMAL_SESSION_ID.fullmatch(session_id) is None:
        abort("session id must be one exact 32-character lowercase hexadecimal value")
    return output_root / ".sessions" / session_id / "controller.json"


def _session_path(output_root: Path, session_id: str) -> Path:
    return output_root / ".sessions" / f"{session_id}.json"


def _child_command(
    *, output_root: Path, matrix: Path, binary: Path, session_id: str | None, resume: bool, capacity_only: bool,
    selected_datasets: tuple[str, ...] = DATASETS,
    compatible_prove_only: bool = False,
    compatible_source_review: str | None = None,
) -> list[str]:
    if compatible_prove_only != (compatible_source_review is not None) or (compatible_prove_only and not resume):
        abort("compatible child flags require a paired compatible resume request")
    command = [
        sys.executable,
        str((ROOT / "scripts/run_formal_matrix.py").resolve()),
        "--execute-formal",
        "--binary", binary.relative_to(ROOT).as_posix(),
        "--output-root", output_root.relative_to(ROOT).as_posix(),
        "--matrix", matrix.relative_to(ROOT).as_posix(),
    ]
    if selected_datasets != DATASETS:
        command.extend(("--datasets", *selected_datasets))
    if capacity_only:
        command.append("--capacity-preflight-only")
    if resume:
        command.extend(("--resume-session", str(session_id)))
        if compatible_prove_only:
            command.append("--compatible-prove-only")
            command.extend(("--compatible-source-review", str(compatible_source_review)))
    elif session_id is not None:
        command.extend(("--_session-id", session_id))
    return command


def _formal_owner_command_variants(metadata: dict[str, Any], session_id: str) -> tuple[list[str], list[str]] | None:
    """Return the exact wrapper argv and its sole permitted execv replacement."""
    output_root = metadata.get("output_root")
    matrix = metadata.get("matrix")
    binary = metadata.get("binary")
    mode = metadata.get("mode")
    if (
        metadata.get("baseline") != "ezkl"
        or metadata.get("session_id") != session_id
        or not all(isinstance(value, str) and value and not Path(value).is_absolute() and ".." not in Path(value).parts
                   for value in (output_root, matrix, binary))
        or mode not in {"resume", "capacity_preflight_only", "five_dataset_formal", "subset_formal"}
    ):
        return None
    try:
        selected_datasets = normalize_selected_datasets(metadata.get("selected_datasets"))
    except ContractError:
        return None
    compatible_field_present = "compatible_prove_only" in metadata
    review_field_present = "compatible_source_review" in metadata
    compatible_prove_only = metadata.get("compatible_prove_only", False)
    compatible_source_review = metadata.get("compatible_source_review")
    if (
        compatible_field_present != review_field_present
        or not isinstance(compatible_prove_only, bool)
        or (compatible_prove_only != (compatible_source_review is not None))
    ):
        return None
    if compatible_prove_only:
        review_reference = _compatible_source_review_reference(compatible_source_review, require_file=False)
        if (
            mode != "resume"
            or session_id != _COMPAT_SESSION_ID
            or output_root != _COMPAT_OUTPUT_ROOT
            or matrix != _COMPAT_MATRIX
            or selected_datasets != _COMPAT_SELECTED_DATASETS
            or metadata.get("selected_datasets") != list(_COMPAT_SELECTED_DATASETS)
            or review_reference is None
            or review_reference != compatible_source_review
        ):
            return None
    wrapper = [
        sys.executable,
        str((ROOT / "scripts/run_formal_matrix.py").resolve()),
        "--execute-formal",
        "--binary", binary,
        "--output-root", output_root,
        "--matrix", matrix,
    ]
    if selected_datasets != DATASETS:
        wrapper.extend(("--datasets", *selected_datasets))
    if mode == "capacity_preflight_only":
        wrapper.append("--capacity-preflight-only")
    if mode == "resume":
        wrapper.extend(("--resume-session", session_id))
        if compatible_prove_only:
            wrapper.append("--compatible-prove-only")
            wrapper.extend(("--compatible-source-review", str(compatible_source_review)))
    else:
        wrapper.extend(("--_session-id", session_id))
    if metadata.get("child_command") != wrapper:
        return None

    formal_owner = [
        sys.executable,
        str((ROOT / "scripts/run_hybrid_formal.py").resolve()),
        "--binary", binary,
        "--output-root", output_root,
        "--matrix", matrix,
        "--_from-ezkl-entry",
    ]
    if selected_datasets != DATASETS:
        formal_owner.extend(("--datasets", *selected_datasets))
    if mode == "capacity_preflight_only":
        formal_owner.append("--capacity-preflight-only")
    if mode == "resume":
        formal_owner.extend(("--resume-session", session_id))
        if compatible_prove_only:
            formal_owner.append("--compatible-prove-only")
            formal_owner.extend(("--compatible-source-review", str(compatible_source_review)))
    else:
        formal_owner.extend(("--_session-id", session_id))
    return wrapper, formal_owner


def _formal_owner_matches(identity: object, metadata: dict[str, Any], session_id: str) -> bool:
    """Authenticate the runner PID across only the known run_formal_matrix execv."""
    if not isinstance(identity, dict):
        return False
    pid, ticks = identity.get("pid"), identity.get("start_ticks")
    variants = _formal_owner_command_variants(metadata, session_id)
    if (
        isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0
        or isinstance(ticks, bool) or not isinstance(ticks, int) or ticks <= 0
        or variants is None
        or identity.get("cmdline") != variants[0]
    ):
        return False
    try:
        if process_start_ticks(pid) != ticks or not _proc_is_live(pid):
            return False
    except (OSError, RuntimeError):
        return False
    actual = _proc_cmdline(pid)
    return actual in variants and _proc_cwd(pid) == str(ROOT.resolve())


def _signal_formal_owner(identity: object, metadata: dict[str, Any], session_id: str, signum: int) -> bool:
    """Signal the recorded owner after rechecking either exact allowed argv state."""
    for _attempt in range(2):
        if not _formal_owner_matches(identity, metadata, session_id):
            return False
        assert isinstance(identity, dict)
        actual = _proc_cmdline(int(identity["pid"]))
        if actual is None:
            return False
        if _send_exact_signal(
            {"pid": identity["pid"], "start_ticks": identity["start_ticks"], "cmdline": actual},
            signum,
            session_id=session_id,
        ):
            return True
    return False


def _write_launcher_metadata(
    path: Path, value: dict[str, Any], *, pid: int, start_ticks: int, command: list[str], status: str,
) -> None:
    value["control_process"] = {"pid": pid, "start_ticks": start_ticks, "cmdline": command}
    value["status"] = status
    value["updated_at_utc"] = _now()
    _atomic_json(path, value)


def _launch(args: argparse.Namespace, *, resume: bool) -> int:
    output_root = _safe_local(args.output_root, "output root")
    matrix = _safe_local(args.matrix, "matrix")
    binary = _safe_local(args.binary, "binary")
    requested_datasets = (
        normalize_selected_datasets(getattr(args, "datasets", None))
        if getattr(args, "datasets", None) is not None
        else None
    )
    resume_session: dict[str, Any] | None = None
    if resume:
        session_id = args.session_id
        session_path = _session_path(output_root, session_id)
        if session_path.is_symlink() or not session_path.is_file():
            abort(f"resume session is missing or symlinked: {session_path}")
        try:
            session = json.loads(session_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            abort(f"cannot read exact resume session: {exc}")
        resume_session = session if isinstance(session, dict) else None
        if (
            not isinstance(session, dict)
            or session.get("baseline") != "ezkl"
            or session.get("session_id") != session_id
            or session.get("output_root") != output_root.relative_to(ROOT).as_posix()
            or session.get("status") not in {"FAILED", "INTERRUPTED"}
        ):
            abort("resume requires this exact failed/interrupted EZKL session and output root")
        saved_datasets = selected_datasets_for_session(session)
        if requested_datasets is not None and requested_datasets != saved_datasets:
            abort("explicit resume --datasets differs from this immutable session selection")
        selected_datasets = saved_datasets
    else:
        session_id = uuid.uuid4().hex
        selected_datasets = requested_datasets or DATASETS

    compatible_prove_only, compatible_source_review = _compatible_resume_options(
        args,
        resume=resume,
        session_id=session_id,
        output_root=output_root,
        matrix=matrix,
        selected_datasets=selected_datasets,
        requested_datasets=requested_datasets,
        session=resume_session,
    )

    metadata_path = _metadata_path(output_root, session_id)
    log_path = output_root / ".sessions" / session_id / "logs" / "formal_session.log"
    _safe_mkdir(log_path.parent, "formal total log parent")
    if log_path.is_symlink() or (log_path.exists() and not log_path.is_file()):
        abort(f"formal total log is not a regular file: {log_path}")
    if metadata_path.exists() or metadata_path.is_symlink():
        previous = _read_metadata(metadata_path, session_id, output_root)
        if _alive_exact(previous.get("control_process"), expected_script=_SCRIPT, session_id=session_id):
            abort(f"exact EZKL session {session_id} already has a live detached controller")

    child = _child_command(
        output_root=output_root,
        matrix=matrix,
        binary=binary,
        session_id=session_id,
        resume=resume,
        capacity_only=bool(getattr(args, "capacity_preflight_only", False)),
        selected_datasets=selected_datasets,
        compatible_prove_only=compatible_prove_only,
        compatible_source_review=compatible_source_review,
    )
    supervisor_args = [
        sys.executable,
        str(_SCRIPT),
        "_supervise",
        "--session-id", session_id,
        "--output-root", output_root.relative_to(ROOT).as_posix(),
        "--metadata", metadata_path.relative_to(ROOT).as_posix(),
        "--log", log_path.relative_to(ROOT).as_posix(),
        "--child-json", json.dumps(child, separators=(",", ":")),
    ]
    history: list[dict[str, Any]] = []
    if metadata_path.is_file() and not metadata_path.is_symlink():
        try:
            previous = json.loads(metadata_path.read_text(encoding="utf-8"))
            if isinstance(previous, dict) and previous.get("schema") == _SCHEMA:
                old_history = previous.get("launch_history", [])
                if isinstance(old_history, list):
                    history.extend(item for item in old_history if isinstance(item, dict))
                history.append({
                    "status": previous.get("status"),
                    "started_at_utc": previous.get("started_at_utc"),
                    "finished_at_utc": previous.get("finished_at_utc"),
                    "exit_code": previous.get("exit_code"),
                    "control_process": previous.get("control_process"),
                })
        except (OSError, json.JSONDecodeError):
            pass
    metadata: dict[str, Any] = {
        "schema": _SCHEMA,
        "baseline": "ezkl",
        "session_id": session_id,
        "output_root": output_root.relative_to(ROOT).as_posix(),
        "matrix": matrix.relative_to(ROOT).as_posix(),
        "binary": binary.relative_to(ROOT).as_posix(),
        "mode": "resume" if resume else "capacity_preflight_only" if getattr(args, "capacity_preflight_only", False) else "five_dataset_formal" if selected_datasets == DATASETS else "subset_formal",
        "selected_datasets": list(selected_datasets),
        "child_command": child,
        "log_path": log_path.relative_to(ROOT).as_posix(),
        "started_at_utc": _now(),
        "launch_history": history,
    }
    if compatible_prove_only:
        metadata["compatible_prove_only"] = True
        metadata["compatible_source_review"] = compatible_source_review
    _atomic_json(metadata_path, metadata)
    log = log_path.open("a", encoding="utf-8")
    log.write(f"\n--- {_now()} detached formal controller launch session={session_id} mode={metadata['mode']} ---\n")
    log.flush()
    env = dict(os.environ)
    env.pop("LD_PRELOAD", None)
    try:
        supervisor = subprocess.Popen(
            supervisor_args,
            cwd=ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    except BaseException:
        log.close()
        metadata["status"] = "LAUNCH_FAILED"
        metadata["finished_at_utc"] = _now()
        _atomic_json(metadata_path, metadata)
        raise
    log.close()
    deadline = time.monotonic() + 5
    while True:
        try:
            ticks = process_start_ticks(supervisor.pid)
            break
        except (OSError, RuntimeError):
            if supervisor.poll() is not None or time.monotonic() >= deadline:
                metadata["status"] = "LAUNCH_FAILED"
                metadata["exit_code"] = supervisor.returncode
                metadata["finished_at_utc"] = _now()
                _atomic_json(metadata_path, metadata)
                abort("detached EZKL controller exited before its exact process identity could be recorded")
            time.sleep(0.05)
    _write_launcher_metadata(
        metadata_path,
        metadata,
        pid=supervisor.pid,
        start_ticks=ticks,
        command=supervisor_args,
        status="RUNNING",
    )
    early_exit = supervisor.poll()
    if early_exit is not None:
        latest = _read_metadata(metadata_path, session_id, output_root)
        if latest.get("status") in {"STARTING", "RUNNING", "STOPPING"}:
            latest["status"] = "EXITED"
            latest["exit_code"] = early_exit
            latest["finished_at_utc"] = _now()
            _atomic_json(metadata_path, latest)
        else:
            metadata = latest
        print(f"session_id={session_id}")
        print(f"controller={metadata.get('status')} exit_code={metadata.get('exit_code')}; inspect the session log and status")
        print(f"log={log_path.relative_to(ROOT)}")
        return 0
    print(f"session_id={session_id}")
    print(f"controller=RUNNING pid={supervisor.pid} start_ticks={ticks}")
    print(f"log={log_path.relative_to(ROOT)}")
    print("formal execution has been detached; this launch status is not a dataset/proof readiness result")
    return 0


def _supervise(args: argparse.Namespace) -> int:
    session_id = args.session_id
    output_root = _safe_local(args.output_root, "output root")
    metadata_path = _safe_local(args.metadata, "controller metadata")
    log_path = _safe_local(args.log, "formal total log")
    if FORMAL_SESSION_ID.fullmatch(session_id) is None or not isinstance(args.child_json, str):
        abort("detached supervisor has invalid session/command arguments")
    try:
        child_command = json.loads(args.child_json)
    except json.JSONDecodeError as exc:
        abort(f"detached supervisor command is malformed: {exc}")
    if not isinstance(child_command, list) or not child_command or any(not isinstance(item, str) for item in child_command):
        abort("detached supervisor command must be a nonempty string vector")
    metadata = _read_metadata(metadata_path, session_id, output_root)
    own_command = _proc_cmdline(os.getpid())
    if own_command is None:
        abort("cannot authenticate detached supervisor command line")
    started_deadline = time.monotonic() + 15
    while time.monotonic() < started_deadline:
        metadata = _read_metadata(metadata_path, session_id, output_root)
        identity = metadata.get("control_process")
        if isinstance(identity, dict) and identity.get("pid") == os.getpid() and identity.get("cmdline") == own_command:
            break
        time.sleep(0.05)
    else:
        abort("parent did not persist the detached controller PID/start-ticks/cmdline identity")
    if metadata.get("child_command") != child_command:
        abort("detached supervisor command differs from its authenticated controller metadata")

    stop_requested = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True

    previous_term = signal.signal(signal.SIGTERM, request_stop)
    previous_int = signal.signal(signal.SIGINT, request_stop)
    child: subprocess.Popen[bytes] | None = None
    child_ticks: int | None = None
    try:
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n--- {_now()} detached supervisor starting official selected-dataset runner ---\n")
            log.flush()
            if stop_requested:
                metadata["status"] = "STOPPED"
                metadata["finished_at_utc"] = _now()
                _atomic_json(metadata_path, metadata)
                return 0
            child_env = dict(os.environ)
            child_env.pop("LD_PRELOAD", None)
            child = subprocess.Popen(
                child_command,
                cwd=ROOT,
                env=child_env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
            child_started = time.monotonic()
            try:
                child_ticks = process_start_ticks(child.pid)
            except (OSError, RuntimeError) as exc:
                if child.poll() is None:
                    try:
                        child.send_signal(signal.SIGTERM)
                        child.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(child.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                        try:
                            child.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            try:
                                os.killpg(child.pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                            child.wait()
                exit_code = child.wait()
                metadata = _read_metadata(metadata_path, session_id, output_root)
                metadata["formal_owner_process"] = {
                    "pid": child.pid,
                    "start_ticks": None,
                    "cmdline": child_command,
                    "started_at_utc": _now(),
                    "identity_error": f"{type(exc).__name__}: {exc}"[:1000],
                }
                metadata["runner_exit_code"] = exit_code
                metadata["status"] = "CHILD_IDENTITY_UNAVAILABLE"
                metadata["exit_code"] = 1
                metadata["finished_at_utc"] = _now()
                _atomic_json(metadata_path, metadata)
                log.write(
                    f"\n--- {_now()} official_runner_exit_code={exit_code} "
                    "outcome=identity_unavailable; no running status claimed ---\n"
                )
                log.flush()
                return 1
            metadata = _read_metadata(metadata_path, session_id, output_root)
            metadata["formal_owner_process"] = {
                "pid": child.pid,
                "start_ticks": child_ticks,
                "cmdline": child_command,
                "started_at_utc": _now(),
            }
            metadata["status"] = "RUNNING"
            _atomic_json(metadata_path, metadata)
            while child.poll() is None:
                if stop_requested:
                    if (
                        not _signal_formal_owner(metadata.get("formal_owner_process"), metadata, session_id, signal.SIGTERM)
                        and child.poll() is None
                    ):
                        raise RuntimeError("formal owner identity changed before supervisor-forwarded stop")
                    metadata = _read_metadata(metadata_path, session_id, output_root)
                    metadata["status"] = "STOPPING"
                    metadata["stop_requested_at_utc"] = _now()
                    _atomic_json(metadata_path, metadata)
                    stop_requested = False
                time.sleep(0.25)
            exit_code = child.wait()
            child_elapsed_ms = (time.monotonic() - child_started) * 1000.0
            stop_was_requested = isinstance(metadata.get("stop_requested_at_utc"), str) and bool(
                metadata["stop_requested_at_utc"]
            )
            outcome, controller_status = _classify_supervisor_exit(
                exit_code, stop_requested=stop_was_requested
            )
            log.write(
                f"\n--- {_now()} official_runner_exit_code={exit_code} "
                f"wall_ms={child_elapsed_ms:.6f} outcome={outcome} ---\n"
            )
            log.flush()
        metadata = _read_metadata(metadata_path, session_id, output_root)
        metadata["status"] = controller_status
        metadata["exit_code"] = exit_code
        metadata["runner_elapsed_ms"] = child_elapsed_ms
        metadata["finished_at_utc"] = _now()
        _atomic_json(metadata_path, metadata)
        return exit_code
    except BaseException as exc:
        if child is not None and child.poll() is None and child_ticks is not None:
            owner = metadata.get("formal_owner_process")
            if owner is None:
                owner = {"pid": child.pid, "start_ticks": child_ticks, "cmdline": child_command}
                metadata["formal_owner_process"] = owner
            if _signal_formal_owner(owner, metadata, session_id, signal.SIGTERM):
                try:
                    child.wait()
                except BaseException:
                    pass
        try:
            metadata = _read_metadata(metadata_path, session_id, output_root)
            metadata["status"] = "SUPERVISOR_FAILED"
            metadata["error"] = f"{type(exc).__name__}: {exc}"[:2000]
            metadata["finished_at_utc"] = _now()
            _atomic_json(metadata_path, metadata)
        except BaseException:
            pass
        return 1
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGINT, previous_int)


def _read_session_status(output_root: Path, session_id: str) -> tuple[str, str, tuple[str, ...]]:
    path = _session_path(output_root, session_id)
    if path.is_symlink() or not path.is_file():
        return "SESSION_NOT_CREATED", "session file is not present", DATASETS
    try:
        session = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "SESSION_RECORD_INVALID", "cannot parse session file", DATASETS
    if (
        not isinstance(session, dict)
        or session.get("baseline") != "ezkl"
        or session.get("session_id") != session_id
        or session.get("output_root") != output_root.relative_to(ROOT).as_posix()
    ):
        return "SESSION_ANCHOR_INVALID", "session/output-root identity mismatch", DATASETS
    return (
        str(session.get("status", "UNKNOWN")),
        str(session.get("stage", "unknown")),
        selected_datasets_for_session(session),
    )


def _status(args: argparse.Namespace) -> int:
    output_root = _safe_local(args.output_root, "output root")
    metadata_path = _metadata_path(output_root, args.session_id)
    metadata = _read_metadata(metadata_path, args.session_id, output_root)
    control_live = _alive_exact(metadata.get("control_process"), expected_script=_SCRIPT, session_id=args.session_id)
    owner = metadata.get("formal_owner_process", metadata.get("child_process"))
    child_live = _formal_owner_matches(owner, metadata, args.session_id)
    session_status, stage, selected_datasets = _read_session_status(output_root, args.session_id)
    launcher_status = metadata.get("status", "UNKNOWN")
    if control_live:
        print(f"controller=RUNNING pid={metadata['control_process']['pid']} start_ticks={metadata['control_process']['start_ticks']}")
    elif child_live:
        print(f"controller=LOST formal_owner=RUNNING pid={owner['pid']} start_ticks={owner['start_ticks']}")
    else:
        if launcher_status in {"STARTING", "RUNNING", "STOPPING"}:
            launcher_status = "EXITED_WITHOUT_TERMINAL_METADATA"
        print(f"controller={launcher_status} exit_code={metadata.get('exit_code')}")
    print(f"session={session_status} stage={stage}")
    print(f"log={metadata.get('log_path')}")
    if selected_datasets == DATASETS:
        print("status is observational; FINAL_VALIDATION_OK plus five verified results represents a completed formal run")
    else:
        print(f"selected_datasets={','.join(selected_datasets)}")
        print("status is observational; SUBSET_COMPLETE plus all selected verified results represents a completed subset run")
    return 0


def _stop(args: argparse.Namespace) -> int:
    output_root = _safe_local(args.output_root, "output root")
    metadata_path = _metadata_path(output_root, args.session_id)
    metadata = _read_metadata(metadata_path, args.session_id, output_root)
    owner = metadata.get("formal_owner_process", metadata.get("child_process"))
    if owner is not None:
        if _formal_owner_matches(owner, metadata, args.session_id):
            pid = int(owner["pid"])
            ticks = int(owner["start_ticks"])
            latest = _read_metadata(metadata_path, args.session_id, output_root)
            if not _formal_owner_matches(latest.get("formal_owner_process", latest.get("child_process")), latest, args.session_id):
                abort("formal owner identity changed before stop request was recorded")
            latest_owner = latest.get("formal_owner_process", latest.get("child_process"))
            if not _signal_formal_owner(latest_owner, latest, args.session_id, signal.SIGTERM):
                current = _read_metadata(metadata_path, args.session_id, output_root)
                if _formal_owner_matches(current.get("formal_owner_process", current.get("child_process")), current, args.session_id):
                    abort("formal owner identity changed before exact-PID termination")
                try:
                    same_process = process_start_ticks(pid) == ticks and _proc_is_live(pid)
                except (OSError, RuntimeError):
                    same_process = False
                if same_process:
                    abort("formal owner PID remains live but no longer matches the bound argv transition")
            deadline = time.monotonic() + max(0, args.wait_seconds)
            while time.monotonic() < deadline:
                current = _read_metadata(metadata_path, args.session_id, output_root)
                control_live = _alive_exact(current.get("control_process"), expected_script=_SCRIPT, session_id=args.session_id)
                if not control_live and current.get("status") in {"STOPPED", "EXITED", "SUPERVISOR_FAILED", "CHILD_IDENTITY_UNAVAILABLE"}:
                    print(f"stop=COMPLETE formal_owner_pid={pid} exit_code={current.get('runner_exit_code', current.get('exit_code'))} session_id={args.session_id}")
                    return 0
                time.sleep(0.25)
            print(f"stop=REQUESTED formal_owner_pid={pid} start_ticks={ticks} session_id={args.session_id}; supervisor remains to record the runner exit code")
            return 0
        try:
            owner_pid = owner.get("pid") if isinstance(owner, dict) else None
            owner_ticks = owner.get("start_ticks") if isinstance(owner, dict) else None
            if isinstance(owner_pid, int) and not isinstance(owner_pid, bool) and isinstance(owner_ticks, int) and not isinstance(owner_ticks, bool):
                same_process = process_start_ticks(owner_pid) == owner_ticks and _proc_is_live(owner_pid)
            else:
                same_process = False
        except (OSError, RuntimeError):
            same_process = False
        if same_process:
            abort("recorded formal owner PID is live but its command/session identity does not match")
        print(f"stop=NOT_RUNNING formal_owner session_id={args.session_id}; controller is retaining its terminal status")
        return 0

    # The supervisor can be stopped only before it has recorded a formal owner.
    identity = metadata.get("control_process")
    if _alive_exact(identity, expected_script=_SCRIPT, session_id=args.session_id):
        if not _send_exact_signal(identity, signal.SIGTERM, expected_script=_SCRIPT, session_id=args.session_id):
            abort("supervisor identity changed before exact-PID termination")
        print(f"stop=REQUESTED supervisor_pid={identity['pid']} session_id={args.session_id}; it will prevent or forward launch and record any runner exit")
        return 0
    print(f"stop=NOT_RUNNING session_id={args.session_id}")
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    actions = result.add_subparsers(dest="action", required=True)
    start = actions.add_parser("start", help="start the selected canonical formal run in the background")
    start.add_argument("--output-root", default="runs/formal")
    start.add_argument("--matrix", default="results/formal_matrix.json")
    start.add_argument("--binary", default="target/gpu/release/ezkl")
    start.add_argument("--datasets", nargs="+", choices=DATASETS, default=None, help="canonical dataset selection; omitted means all five")
    start.add_argument("--capacity-preflight-only", action="store_true", help="run selected native Admissions and the fixed-payload disk-floor gate, then stop before Setup")
    resume = actions.add_parser("resume", help="resume one exact failed/interrupted formal session")
    resume.add_argument("--session-id", required=True)
    resume.add_argument("--output-root", default="runs/formal")
    resume.add_argument("--matrix", default="results/formal_matrix.json")
    resume.add_argument("--binary", default="target/gpu/release/ezkl")
    resume.add_argument("--datasets", nargs="+", choices=DATASETS, default=None, help="must match this session; omitted inherits its immutable selection")
    resume.add_argument("--compatible-prove-only", action="store_true", help="use the one approved prove-only compatible resume path")
    resume.add_argument("--compatible-source-review", help="exact source review JSON bound to the compatible resume")
    status = actions.add_parser("status", help="inspect one exact detached session")
    status.add_argument("--session-id", required=True)
    status.add_argument("--output-root", default="runs/formal")
    stop = actions.add_parser("stop", help="request guardian-aware termination of one exact session")
    stop.add_argument("--session-id", required=True)
    stop.add_argument("--output-root", default="runs/formal")
    stop.add_argument("--wait-seconds", type=int, default=60)
    supervisor = actions.add_parser("_supervise", help=argparse.SUPPRESS)
    supervisor.add_argument("--session-id", required=True)
    supervisor.add_argument("--output-root", required=True)
    supervisor.add_argument("--metadata", required=True)
    supervisor.add_argument("--log", required=True)
    supervisor.add_argument("--child-json", required=True)
    return result


def main(args: argparse.Namespace) -> int:
    if args.action == "start":
        return _launch(args, resume=False)
    if args.action == "resume":
        return _launch(args, resume=True)
    if args.action == "status":
        return _status(args)
    if args.action == "stop":
        if args.wait_seconds < 0 or args.wait_seconds > 3600:
            abort("--wait-seconds must be between 0 and 3600")
        return _stop(args)
    if args.action == "_supervise":
        return _supervise(args)
    abort("unsupported EZKL controller action")


if __name__ == "__main__":
    try:
        raise SystemExit(main(parser().parse_args()))
    except (ContractError, OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        raise SystemExit(print_error(exc))
