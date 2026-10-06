#!/usr/bin/env python3
"""Detached coordinator for the single official comparison experiment."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import ctypes
import ctypes.util
import selectors
import secrets
import signal
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


COMPARISON_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = COMPARISON_ROOT.parent
DATASETS = ("cora", "citeseer", "pubmed", "ppi", "ogbn_arxiv")
BASELINES = ("ezkl", "zkdl", "zkgpt")
EXECUTION_BASELINES = ("zkgpt", "ezkl", "zkdl")
SESSION_ID_RE = re.compile(r"^[0-9a-f]{32}$")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
ROOT_CELL_LOG_LIMIT = 16 * 1024 * 1024
ROOT_PREPARE_LOG_LIMIT = 128 * 1024 * 1024
ROOT_FINAL_LOG_LIMIT = 32 * 1024 * 1024
ROOT_RUNNER_LOG_RESERVE = 16 * 1024 * 1024
ROOT_CELL_LOG_RESERVE = len(DATASETS) * len(BASELINES) * (ROOT_CELL_LOG_LIMIT + 1024)
ROOT_FUTURE_LOG_RESERVE = ROOT_CELL_LOG_RESERVE + ROOT_FINAL_LOG_LIMIT + ROOT_RUNNER_LOG_RESERVE
ROOT_FINAL_DATA_RESERVE = 64 * 1024 * 1024
ROOT_RECEIPT_RESERVE = 64 * 1024 * 1024
ROOT_FORMAL_OUTPUT_RESERVE = ROOT_FUTURE_LOG_RESERVE + ROOT_FINAL_DATA_RESERVE + ROOT_RECEIPT_RESERVE
OWNER_TOKEN_ENV = "COMPARISON_FORMAL_OWNER_TOKEN"
ROOT_LOCK_FD_ENV = "COMPARISON_FORMAL_ROOT_LOCK_FD"
SESSION_ID_ENV = "COMPARISON_FORMAL_SESSION_ID"
ACTIVE_CHILD: subprocess.Popen[bytes] | None = None
ACTIVE_STAGE = ""


class RunnerError(RuntimeError):
    pass


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(_json_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_unaliased_path(path: Path, floor: Path, label: str) -> None:
    absolute = path.absolute()
    root = floor.absolute()
    try:
        relative = absolute.relative_to(root)
    except ValueError as exc:
        raise RunnerError(f"{label} escapes its checked root: {path}") from exc
    current = root
    for candidate in (current, *(root / Path(*relative.parts[:index]) for index in range(1, len(relative.parts) + 1))):
        if candidate.is_symlink():
            raise RunnerError(f"{label} contains an aliased path component: {candidate}")


def _directory_regular_bytes(root: Path) -> int:
    total = 0
    for current, directories, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in directories:
            path = current_path / name
            if path.is_symlink() or not path.is_dir():
                raise RunnerError(f"prepared session tree contains an aliased/non-directory path: {path}")
        for name in filenames:
            path = current_path / name
            info = path.lstat()
            if path.is_symlink() or not path.is_file():
                raise RunnerError(f"prepared session tree contains an aliased/non-file path: {path}")
            total += info.st_size
    return total


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerError(f"{label} is missing or invalid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RunnerError(f"{label} must be a JSON object: {path}")
    return value


def _read_boot_id() -> str:
    try:
        value = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
    except OSError:
        return "unavailable"
    return value if re.fullmatch(r"[0-9a-f-]{36}", value) else "unavailable"


def _proc_start_ticks(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        tail = raw[raw.rfind(")") + 2 :].split()
        return tail[19]
    except (OSError, IndexError):
        return "unavailable"


def _proc_snapshot(pid: int) -> dict[str, Any] | None:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        tail = raw[raw.rfind(")") + 2 :].split()
        return {
            "state": tail[0],
            "ppid": int(tail[1]),
            "pgid": int(tail[2]),
            "start_ticks": tail[19],
        }
    except (OSError, ValueError, IndexError):
        return None


def _process_has_owner_token(pid: int, token_sha256: str) -> bool:
    try:
        fields = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
    except OSError:
        return False


def _process_has_session_id(pid: int, session_id: str) -> bool:
    try:
        fields = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
    except OSError:
        return False
    prefix = f"{SESSION_ID_ENV}=".encode("ascii")
    value = next((field[len(prefix):] for field in fields if field.startswith(prefix)), None)
    return value == session_id.encode("ascii")
    prefix = f"{OWNER_TOKEN_ENV}=".encode("ascii")
    value = next((field[len(prefix):] for field in fields if field.startswith(prefix)), None)
    if value is None:
        return False
    try:
        return hashlib.sha256(value).hexdigest() == token_sha256
    except (ValueError, TypeError):
        return False


def _validate_root_lock_fd(lock_fd: int) -> None:
    lock_path = COMPARISON_ROOT / "runs" / ".formal.lock"
    if lock_path.is_symlink():
        raise RunnerError("formal global lock path is aliased")
    try:
        inherited = os.fstat(lock_fd)
        current = lock_path.stat(follow_symlinks=False)
        if (not os.path.samefile(f"/proc/self/fd/{lock_fd}", lock_path)
                or inherited.st_dev != current.st_dev or inherited.st_ino != current.st_ino):
            raise RunnerError("inherited descriptor is not the formal global lock")
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        raise RunnerError(f"cannot validate formal global lock descriptor: {exc}") from exc


def _record_owned_stage(session_root: Path, *, stage: str, pid: int, token_sha256: str,
                        session_id: str, root_owner: Mapping[str, Any]) -> None:
    state_path = session_root / "state.json"
    if not state_path.is_file() or state_path.is_symlink():
        return
    state = _read_json(state_path, "formal session state")
    if state.get("status") != "RUNNING" or state.get("stage") != stage:
        raise RunnerError("root session stage changed before its owned child was recorded")
    snapshot = _proc_snapshot(pid)
    if (snapshot is None or snapshot["pgid"] != pid
            or not _process_has_session_id(pid, session_id)):
        raise RunnerError(
            "formal stage supervisor identity mismatch "
            f"snapshot={snapshot} session={_process_has_session_id(pid, session_id)} expected_pgid={pid}"
        )
    state["owned_stage"] = {
        "kind": "stage-supervisor",
        "stage": stage,
        "pid": pid,
        "pgid": pid,
        "boot_id": _read_boot_id(),
        "start_ticks": snapshot["start_ticks"],
        "session_id": session_id,
        "owner_token_sha256": token_sha256,
        "root_owner": dict(root_owner),
    }
    atomic_json(state_path, state)


def _clear_owned_stage(session_root: Path, *, stage: str, pid: int) -> None:
    state_path = session_root / "state.json"
    if not state_path.is_file() or state_path.is_symlink():
        return
    try:
        state = _read_json(state_path, "formal session state")
    except RunnerError:
        return
    owned = state.get("owned_stage")
    if isinstance(owned, dict) and owned.get("stage") == stage and owned.get("pid") == pid:
        state.pop("owned_stage", None)
        atomic_json(state_path, state)


def _owned_stage_wrapper(parent_pid: int, stage: str, command: Sequence[str]) -> int:
    """Supervise one root-owned phase and reap even setsid descendants."""
    if parent_pid <= 0 or not command:
        return 2
    token = os.environ.get(OWNER_TOKEN_ENV, "")
    session_id = os.environ.get(SESSION_ID_ENV, "")
    lock_fd_text = os.environ.get(ROOT_LOCK_FD_ENV, "")
    if (not HEX64_RE.fullmatch(token)
            or not SESSION_ID_RE.fullmatch(session_id)
            or not lock_fd_text.isdigit()):
        return 2
    lock_fd = int(lock_fd_text)
    try:
        _validate_root_lock_fd(lock_fd)
        state_path = _session_root(session_id) / "state.json"
        state = _read_json(state_path, "root formal owner state")
        parent_cmd = [part for part in Path(f"/proc/{parent_pid}/cmdline").read_bytes().split(b"\0") if part]
        parent_stat = Path(f"/proc/{parent_pid}/stat").read_text(encoding="ascii")
        parent_tail = parent_stat[parent_stat.rfind(")") + 2:].split()
    except (OSError, RunnerError, ValueError, IndexError):
        return 2
    if (os.getppid() != parent_pid or state.get("status") != "RUNNING"
            or state.get("session_id") != session_id
            or state.get("internal_owner_token_sha256") != hashlib.sha256(token.encode("ascii")).hexdigest()
            or state.get("stage") != stage
            or not isinstance(state.get("lock"), dict)
            or state["lock"].get("pid") != parent_pid
            or state["lock"].get("boot_id") != _read_boot_id()
            or len(parent_tail) <= 19 or parent_tail[19] != state["lock"].get("start_ticks")
            or len(parent_cmd) < 4
            or Path(os.fsdecode(parent_cmd[1])).resolve() != (COMPARISON_ROOT / "formal_runner.py").resolve()
            or b"--_worker" not in parent_cmd or session_id.encode("ascii") not in parent_cmd):
        return 2
    return _supervise_owned_command(parent_pid, stage, command, token, session_id, lock_fd, state)


def _supervise_owned_command(parent_pid: int, stage: str, command: Sequence[str], token: str,
                             session_id: str, lock_fd: int, state: Mapping[str, Any]) -> int:
    """Run and reap one phase after its owner lease has been authenticated."""
    try:
        import ctypes
        libc = ctypes.CDLL(None, use_errno=True)
        # Adopt daemonized descendants if the phase leader dies. This lets the
        # supervisor clean children that deliberately create a new session.
        if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
            print(f"FORMAL_STAGE_SUPERVISOR_BLOCKED subreaper errno={ctypes.get_errno()}", file=sys.stderr, flush=True)
            return 2
        if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
            print(f"FORMAL_STAGE_SUPERVISOR_BLOCKED parent-death errno={ctypes.get_errno()}", file=sys.stderr, flush=True)
            return 2
    except (OSError, ImportError):
        return 2
    if os.getppid() != parent_pid:
        print("FORMAL_STAGE_SUPERVISOR_BLOCKED parent changed before setup", file=sys.stderr, flush=True)
        return 2
    token_sha256 = hashlib.sha256(token.encode("ascii")).hexdigest()
    root_owner = dict(state["lock"])
    session_root = _session_root(session_id)
    try:
        _record_owned_stage(
            session_root, stage=stage, pid=os.getpid(), token_sha256=token_sha256,
            session_id=session_id, root_owner=root_owner,
        )
    except (OSError, RunnerError) as exc:
        print(f"FORMAL_STAGE_SUPERVISOR_BLOCKED owner registration failed: {exc}", file=sys.stderr, flush=True)
        return 2

    # Publish supervisor readiness before the baseline phase can inspect its
    # owner lease. The atomic state write above is the launch barrier.
    received_parent_signal = False

    def on_parent_signal(_signum: int, _frame: Any) -> None:
        nonlocal received_parent_signal
        received_parent_signal = True

    signal.signal(signal.SIGTERM, on_parent_signal)
    signal.signal(signal.SIGINT, on_parent_signal)
    child: subprocess.Popen[bytes] | None = None
    return_code = 2
    try:
        child = subprocess.Popen(
            list(command), cwd=PROJECT_ROOT, env=os.environ,
            stdin=subprocess.DEVNULL, stdout=None, stderr=subprocess.STDOUT,
            close_fds=True, pass_fds=(lock_fd,),
        )
        while child.poll() is None and not received_parent_signal:
            time.sleep(0.05)
        if received_parent_signal and child.poll() is None:
            _signal_session_processes(session_id, exclude_pids={os.getpid(), parent_pid}, grace_seconds=1.0)
            try:
                child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                _signal_pid_if_session_owned(child.pid, session_id, signal.SIGKILL)
                child.wait()
        return_code = child.wait()
        # A phase leader may exit while native grandchildren remain in a
        # separate process group. Subreaper adoption plus the unique session
        # tag lets us stop and reap those exact descendants before releasing
        # the inherited global lock.
        _signal_session_processes(session_id, exclude_pids={os.getpid(), parent_pid}, grace_seconds=0.75)
        _reap_adopted_children()
        if received_parent_signal and return_code == 0:
            return_code = 128 + signal.SIGTERM
    except BaseException:
        _signal_session_processes(session_id, exclude_pids={os.getpid(), parent_pid}, grace_seconds=0.25)
        if child is not None and child.poll() is None:
            try:
                child.wait(timeout=1)
            except subprocess.TimeoutExpired:
                _signal_pid_if_session_owned(child.pid, session_id, signal.SIGKILL)
                child.wait()
        _reap_adopted_children()
        raise
    finally:
        _clear_owned_stage(session_root, stage=stage, pid=os.getpid())
    return return_code


def _sanitized_environment() -> dict[str, str]:
    # The shell launcher also clears loader and Python injection variables. Keep
    # only stable process basics; device selection remains owned by EZKL.
    return {
        "HOME": "/home/pzh",
        "PATH": "/home/pzh/.cargo/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "TZ": "UTC",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
    }


def _session_root(session_id: str) -> Path:
    if not SESSION_ID_RE.fullmatch(session_id):
        raise RunnerError("root session id is malformed")
    return COMPARISON_ROOT / "runs" / "formal" / session_id


def _baseline_session_dir(baseline: str, session_id: str) -> Path:
    if baseline == "ezkl":
        return _session_root(session_id) / "ezkl"
    if baseline == "zkdl":
        return COMPARISON_ROOT / "zkdl" / "runs" / f"formal_{session_id}"
    if baseline == "zkgpt":
        return COMPARISON_ROOT / "zkGPT" / "results" / "formal" / session_id
    raise RunnerError(f"unknown baseline: {baseline}")


def phase_command(baseline: str, phase: str, session_id: str, dataset: str | None = None,
                  *, diagnostic: bool = False) -> list[str]:
    session_dir = _baseline_session_dir(baseline, session_id)
    if baseline == "ezkl":
        command = [
            sys.executable,
            str(COMPARISON_ROOT / "ezkl" / "scripts" / "run_formal_matrix.py"),
            "--execute-formal", "--internal-phase", phase, "--session-dir", str(session_dir),
        ]
    elif baseline == "zkdl":
        command = [
            "/usr/bin/bash", str(COMPARISON_ROOT / "zkdl" / "scripts" / "run_five.sh"),
            "--internal-phase", phase, "--session-dir", str(session_dir),
        ]
    else:
        command = [
            sys.executable,
            str(COMPARISON_ROOT / "zkGPT" / "scripts" / "run_formal.py"),
            "--internal-phase", phase, "--session-dir", str(session_dir),
        ]
    if dataset is not None:
        command.extend(("--dataset", dataset))
    if diagnostic and baseline == "ezkl":
        diagnostic_root = Path("runs") / "diagnostic" / session_id / "formal"
        command.extend(("--output-root", diagnostic_root.as_posix(),
                        "--matrix", (Path("runs") / "diagnostic" / session_id / "formal_matrix.json").as_posix()))
    return command


def _write_state(root: Path, *, status: str, stage: str, exit_code: int | None,
                 failure: str = "", extra: Mapping[str, Any] | None = None) -> None:
    previous: dict[str, Any] = {}
    state_path = root / "state.json"
    if state_path.exists():
        try:
            previous = _read_json(state_path, "session state")
        except RunnerError:
            previous = {}
    payload = {
        "schema": "comparison-formal-session-v1",
        "session_id": root.name,
        "status": status,
        "stage": stage,
        "exit_code": exit_code,
        "failure": failure,
        "started_unix_ns": previous.get("started_unix_ns", time.time_ns()),
        "updated_unix_ns": time.time_ns(),
        "completed_stages": list(previous.get("completed_stages", [])),
        "cell_receipts": list(previous.get("cell_receipts", [])),
    }
    if extra:
        payload.update(extra)
    atomic_json(state_path, payload)


def _phase_marker(baseline: str, phase: str, output: str) -> dict[str, Any]:
    lines = output.splitlines()
    if baseline == "zkdl":
        for line in reversed(lines):
            if not line.startswith("{"):
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and value.get("zkdl_phase") == phase:
                if value.get("status") != "PASS":
                    raise RunnerError(f"zkDL {phase} marker did not report PASS")
                return value
        raise RunnerError(f"zkDL {phase} completed without its strict JSON phase marker")

    if baseline == "ezkl":
        prefix = f"EZKL_INTERNAL_PHASE_OK phase={phase} "
        for line in reversed(lines):
            if line.startswith(prefix):
                tail = line[len(prefix):]
                marker: dict[str, Any] = {"phase": phase, "status": "PASS"}
                if "{" in tail:
                    try:
                        data = json.loads(tail[tail.index("{"):])
                    except json.JSONDecodeError as exc:
                        raise RunnerError(f"EZKL {phase} marker JSON is malformed") from exc
                    if not isinstance(data, dict):
                        raise RunnerError(f"EZKL {phase} marker JSON is not an object")
                    marker.update(data)
                return marker
        raise RunnerError(f"EZKL {phase} completed without its strict phase marker")

    prefix = "ZKGPT_COORDINATOR_STATE="
    for line in reversed(lines):
        if line.startswith(prefix):
            try:
                value = json.loads(line[len(prefix):])
            except json.JSONDecodeError as exc:
                raise RunnerError(f"zkGPT {phase} marker JSON is malformed") from exc
            if not isinstance(value, dict) or value.get("phase") != phase or value.get("status") != "PASS":
                raise RunnerError(f"zkGPT {phase} marker does not match the completed phase")
            return value
    raise RunnerError(f"zkGPT {phase} completed without its strict JSON phase marker")


def _owned_process_group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _session_process_records(session_id: str, *, exclude_pids: set[int] | None = None) -> list[dict[str, Any]]:
    excluded = exclude_pids or set()
    matches: list[dict[str, Any]] = []
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid in excluded or not _process_has_session_id(pid, session_id):
            continue
        snapshot = _proc_snapshot(pid)
        if snapshot is not None and snapshot["state"] != "Z":
            matches.append({"pid": pid, **snapshot})
    return matches


def _signal_pid_if_session_owned(pid: int, session_id: str, signum: int,
                                 expected_start_ticks: str | None = None) -> bool:
    before = _proc_snapshot(pid)
    if (before is None or before["state"] == "Z"
            or (expected_start_ticks is not None and before["start_ticks"] != expected_start_ticks)
            or not _process_has_session_id(pid, session_id)):
        return False
    # Recheck identity immediately before signaling to avoid PID reuse races.
    current = _proc_snapshot(pid)
    if (current is None or current["start_ticks"] != before["start_ticks"]
            or not _process_has_session_id(pid, session_id)):
        return False
    try:
        os.kill(pid, signum)
        return True
    except ProcessLookupError:
        return False


def _signal_session_processes(session_id: str, *, exclude_pids: set[int] | None = None,
                              grace_seconds: float = 1.0) -> None:
    """Stop only processes carrying this exact random session environment tag."""
    excluded = exclude_pids or set()
    original = _session_process_records(session_id, exclude_pids=excluded)
    for record in original:
        _signal_pid_if_session_owned(int(record["pid"]), session_id, signal.SIGTERM,
                                     str(record["start_ticks"]))
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        if not _session_process_records(session_id, exclude_pids=excluded):
            return
        time.sleep(0.025)
    for record in _session_process_records(session_id, exclude_pids=excluded):
        _signal_pid_if_session_owned(int(record["pid"]), session_id, signal.SIGKILL,
                                     str(record["start_ticks"]))
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if not _session_process_records(session_id, exclude_pids=excluded):
            return
        time.sleep(0.025)
    survivors = _session_process_records(session_id, exclude_pids=excluded)
    if survivors:
        raise RunnerError(f"owned session processes survived cleanup: {[item['pid'] for item in survivors]}")


def _reap_adopted_children() -> None:
    while True:
        try:
            pid, _status = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if pid == 0:
            return


def _terminate_owned_process_group(pgid: int, grace_seconds: float = 0.5) -> None:
    """Stop only the process group created for one owned formal stage."""
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline and _owned_process_group_exists(pgid):
        time.sleep(0.02)
    if _owned_process_group_exists(pgid):
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _run_stage(command: Sequence[str], log_path: Path, stage: str,
               session_root: Path, log_limit_bytes: int,
               *, owner_token: str | None = None, lock_fd: int | None = None) -> tuple[int, str]:
    global ACTIVE_CHILD, ACTIVE_STAGE
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ACTIVE_STAGE = stage
    print(f"START {stage} log={log_path.relative_to(session_root)}", flush=True)
    started = time.time_ns()
    env = _sanitized_environment()
    token_sha256: str | None = None
    if owner_token is not None or lock_fd is not None:
        if owner_token is None or lock_fd is None or not re.fullmatch(r"[0-9a-f]{64}", owner_token):
            raise RunnerError("formal phase ownership requires both a valid token and inherited lock fd")
        _validate_root_lock_fd(lock_fd)
        session_id = os.environ.get(SESSION_ID_ENV, "")
        if not SESSION_ID_RE.fullmatch(session_id):
            raise RunnerError("formal phase ownership has no root session id")
        env[OWNER_TOKEN_ENV] = owner_token
        env[ROOT_LOCK_FD_ENV] = str(lock_fd)
        env[SESSION_ID_ENV] = session_id
        token_sha256 = hashlib.sha256(owner_token.encode("ascii")).hexdigest()
    log_limit_exceeded = False
    with log_path.open("ab", buffering=0) as log:
        header = f"\n=== START {stage} unix_ns={started} ===\n".encode()
        log.write(header)
        written = len(header)
        child_command = list(command)
        if owner_token is not None:
            child_command = [
                sys.executable, str(Path(__file__).resolve()), "--_stage-supervisor", str(os.getpid()), stage, "--",
                *command,
            ]
        child = subprocess.Popen(
            child_command, cwd=PROJECT_ROOT, env=env, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True,
            close_fds=True, pass_fds=(lock_fd,) if lock_fd is not None else (),
        )
        ACTIVE_CHILD = child
        assert child.stdout is not None
        selector = selectors.DefaultSelector()
        os.set_blocking(child.stdout.fileno(), False)
        selector.register(child.stdout, selectors.EVENT_READ)
        terminate_deadline: float | None = None
        exited_group_cleanup_started = False
        try:
            while selector.get_map():
                for key, _mask in selector.select(timeout=0.2):
                    try:
                        chunk = os.read(key.fd, 65536)
                    except BlockingIOError:
                        continue
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    remaining = max(0, log_limit_bytes - written)
                    if len(chunk) <= remaining:
                        log.write(chunk)
                        written += len(chunk)
                    else:
                        if remaining:
                            log.write(chunk[:remaining])
                            written += remaining
                        if not log_limit_exceeded:
                            message = (f"\nLOG_OUTPUT_LIMIT_EXCEEDED stage={stage} "
                                       f"limit_bytes={log_limit_bytes}; owned command group terminated\n").encode()
                            log.write(message)
                            log.flush()
                            log_limit_exceeded = True
                            terminate_deadline = time.monotonic() + 5
                            try:
                                os.killpg(child.pid, signal.SIGTERM)
                            except ProcessLookupError:
                                pass
                # A stage command owns its complete process group. If its
                # leader exits while descendants still hold the output pipe,
                # stop those descendants rather than waiting indefinitely or
                # allowing work to outlive the stage receipt.
                if child.poll() is not None and not exited_group_cleanup_started:
                    exited_group_cleanup_started = True
                    try:
                        os.killpg(child.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    terminate_deadline = time.monotonic() + 5
                if terminate_deadline is not None and time.monotonic() >= terminate_deadline:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    terminate_deadline = None
            return_code = child.wait()
            # Also catch descendants that deliberately closed or redirected
            # the inherited stage log before their leader returned.
            _terminate_owned_process_group(child.pid)
        finally:
            if child.poll() is None:
                _terminate_owned_process_group(child.pid)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    child.wait()
            selector.close()
            ACTIVE_CHILD = None
            child.stdout.close()
            if token_sha256 is not None:
                _clear_owned_stage(session_root, stage=stage, pid=child.pid)
        if log_limit_exceeded:
            return_code = 2
        log.write(f"\n=== END {stage} exit_code={return_code} ===\n".encode())
    output = log_path.read_text(encoding="utf-8", errors="replace")
    if return_code != 0:
        failure = f"{stage} exited with code {return_code}; log={log_path}"
        print(f"FAILED {failure}", file=sys.stderr, flush=True)
        return return_code if return_code > 0 else 2, output
    print(f"COMPLETE {stage}", flush=True)
    return 0, output


def spawn_detached_command(command: Sequence[str], log_path: Path, lock_fd: int,
                           env: Mapping[str, str] | None = None) -> subprocess.Popen[bytes]:
    """Start an owned session process while transferring the held flock fd."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fd = os.open(log_path, os.O_CREAT | os.O_WRONLY | os.O_APPEND | os.O_CLOEXEC, 0o600)
    try:
        return subprocess.Popen(
            list(command), cwd=PROJECT_ROOT, env=dict(env or _sanitized_environment()),
            stdin=subprocess.DEVNULL, stdout=log_fd, stderr=log_fd,
            start_new_session=True, pass_fds=(lock_fd,), close_fds=True,
        )
    finally:
        os.close(log_fd)


def _run_phase(baseline: str, phase: str, session_id: str,
               session_root: Path, dataset: str | None = None,
               log_path: Path | None = None, *, owner_token: str | None = None,
               lock_fd: int | None = None, diagnostic: bool = False) -> dict[str, Any]:
    if phase == "cell" and dataset not in DATASETS:
        raise RunnerError(f"cell phase has invalid dataset: {dataset}")
    if phase != "cell" and dataset is not None:
        raise RunnerError(f"dataset supplied to non-cell phase {phase}")
    command = phase_command(baseline, phase, session_id, dataset, diagnostic=diagnostic)
    if log_path is None:
        log_path = session_root / "logs" / "prepare" / f"{baseline}.log"
        if phase == "finalize":
            log_path = session_root / "logs" / "final-validation.log"
    stage = f"{phase}:{baseline}" + (f":{dataset}" if dataset else "")
    log_limit = ROOT_CELL_LOG_LIMIT if phase == "cell" else (
        ROOT_PREPARE_LOG_LIMIT if phase == "prepare" else ROOT_FINAL_LOG_LIMIT
    )
    status, output = _run_stage(command, log_path, stage, session_root, log_limit,
                                owner_token=owner_token, lock_fd=lock_fd)
    if status != 0:
        raise RunnerError(f"{stage} failed with exit code {status}; see {log_path}")
    marker = _phase_marker(baseline, phase, output)
    if phase == "prepare":
        build_log_path = marker.get("build_log_path")
        build_log_sha256 = marker.get("build_log_sha256")
        if build_log_path is None or build_log_sha256 is None:
            raise RunnerError(f"{baseline} prepare marker is missing its independent build-log identity")
        build_log = _resolve_baseline_path(baseline, build_log_path, "formal build log")
        if not isinstance(build_log_sha256, str) or sha256_file(build_log) != build_log_sha256:
            raise RunnerError(f"{baseline} formal build log digest mismatch")
    if phase == "cell" and marker.get("dataset") != dataset:
        raise RunnerError(f"{stage} returned a marker for another dataset")
    if phase == "finalize":
        if marker.get("status") not in ("PASS", "VALIDATED", "FINAL_VALIDATION_OK"):
            raise RunnerError(f"{baseline} final validator did not report success")
        final_path = (marker.get("final_validation_path") or marker.get("validation_path")
                      or marker.get("matrix_path") or marker.get("collection_path"))
        final_digest = (marker.get("final_validation_sha256") or marker.get("validation_sha256")
                        or marker.get("matrix_sha256") or marker.get("collection_sha256"))
        if final_path is None or final_digest is None:
            raise RunnerError(f"{baseline} finalize marker is missing final-validation artifact identity")
        resolved = _resolve_baseline_path(baseline, final_path, "final-validation artifact")
        if not isinstance(final_digest, str) or sha256_file(resolved) != final_digest:
            raise RunnerError(f"{baseline} final-validation artifact digest mismatch")
        marker["_validation_artifact_sha256"] = final_digest
    return marker


def _marker_metrics(marker: Mapping[str, Any], baseline: str, dataset: str) -> dict[str, Any]:
    if baseline == "ezkl":
        expected_status = marker.get("status") == "VERIFY_OK"
        fields = {
            "commitment_time_ms": ("commitment_time_ms", "commitment_ms"),
            "prove_time_ms": ("prove_time_ms", "prove_ms"),
            "verify_time_ms": ("verify_time_ms", "verify_ms"),
            "proof_size_bytes": ("proof_size_bytes",),
        }
    elif baseline == "zkdl":
        expected_status = marker.get("verification_status") == "VERIFIED" or marker.get("status") == "PASS"
        fields = {
            "commitment_time_ms": ("commitment_time_ms", "commitment_time_seconds"),
            "prove_time_ms": ("prove_time_ms", "prove_time_seconds"),
            "verify_time_ms": ("verify_time_ms", "verify_time_seconds"),
            "proof_size_bytes": ("proof_size_bytes",),
        }
    else:
        expected_status = marker.get("status") in ("VERIFY_OK", "PASS")
        fields = {
            "commitment_time_ms": ("commitment_time_ms", "commitment_ms"),
            "prove_time_ms": ("prove_time_ms", "prove_ms"),
            "verify_time_ms": ("verify_time_ms", "verify_ms"),
            "proof_size_bytes": ("proof_size_bytes",),
        }
    if not expected_status:
        raise RunnerError(f"{baseline}/{dataset} cell marker is not verified")
    if marker.get("dataset") != dataset:
        raise RunnerError(f"{baseline}/{dataset} cell marker dataset mismatch")
    if baseline == "ezkl" and marker.get("result_path") is None:
        raise RunnerError(f"EZKL {dataset} marker is missing its current result path")
    result: dict[str, Any] = {"baseline": baseline, "dataset": dataset}
    for output_key, candidates in fields.items():
        value = next((marker.get(key) for key in candidates if key in marker), None)
        if value is None:
            raise RunnerError(f"{baseline}/{dataset} cell marker is missing {output_key}")
        if output_key == "proof_size_bytes":
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise RunnerError(f"{baseline}/{dataset} has invalid proof size")
            result[output_key] = value
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise RunnerError(f"{baseline}/{dataset} has invalid {output_key}")
            source_key = next(key for key in candidates if key in marker)
            milliseconds = float(value) * (1000.0 if source_key.endswith("_seconds") else 1.0)
            if milliseconds <= 0:
                raise RunnerError(f"{baseline}/{dataset} has nonpositive {output_key}")
            result[output_key] = milliseconds
    return result


def _cell_marker_receipt(baseline: str, dataset: str, marker: Mapping[str, Any]) -> dict[str, Any]:
    metrics = _marker_metrics(marker, baseline, dataset)
    receipt: dict[str, Any] = {
        "schema": "comparison-cell-receipt-v1",
        "baseline": baseline,
        "dataset": dataset,
        "status": "VERIFY_OK",
        "metrics": metrics,
        "baseline_marker": dict(marker),
        "created_unix_ns": time.time_ns(),
    }
    return receipt


def _resolve_baseline_path(baseline: str, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RunnerError(f"{baseline} receipt has no {label}")
    path = Path(value)
    if not path.is_absolute():
        root = COMPARISON_ROOT / ("zkGPT" if baseline == "zkgpt" else baseline)
        path = root / path
    try:
        if path.is_symlink():
            raise RunnerError(f"{baseline} {label} must not be a symlink")
    except OSError as exc:
        raise RunnerError(f"cannot inspect {baseline} {label}: {exc}") from exc
    path = path.resolve(strict=True)
    base = (COMPARISON_ROOT / ("zkGPT" if baseline == "zkgpt" else baseline)).resolve()
    try:
        path.relative_to(base)
    except ValueError as exc:
        raise RunnerError(f"{baseline} {label} escapes its baseline root") from exc
    if not path.is_file():
        raise RunnerError(f"{baseline} {label} is not a regular file")
    return path


def _validate_cell_attestation(baseline: str, dataset: str,
                               marker: Mapping[str, Any]) -> None:
    if baseline == "ezkl":
        path_key, digest_key = "result_path", "result_sha256"
    elif baseline == "zkdl":
        path_key, digest_key = "receipt_path", "receipt_sha256"
    else:
        path_key, digest_key = "receipt_path", "receipt_sha256"
    path_value = marker.get(path_key)
    digest_value = marker.get(digest_key)
    # EZKL reports the signed/authenticated result directly. zkDL/zkGPT expose
    # a cell receipt that the root reopens and binds to the reported proof.
    if path_value is None or digest_value is None:
        if baseline == "ezkl":
            raise RunnerError(f"EZKL {dataset} marker is missing result artifact identity")
        raise RunnerError(f"{baseline} {dataset} marker is missing its session-bound receipt identity")
    path = _resolve_baseline_path(baseline, path_value, "cell result/receipt")
    if not isinstance(digest_value, str) or not re.fullmatch(r"[0-9a-f]{64}", digest_value):
        raise RunnerError(f"{baseline} {dataset} receipt digest is malformed")
    if sha256_file(path) != digest_value:
        raise RunnerError(f"{baseline} {dataset} receipt digest does not match its current file")
    artifact_key, artifact_digest_key = "proof_path", "proof_sha256"
    artifact_value, artifact_digest = marker.get(artifact_key), marker.get(artifact_digest_key)
    if artifact_value is None or artifact_digest is None:
        raise RunnerError(f"{baseline} {dataset} marker is missing proof artifact identity")
    artifact_path = _resolve_baseline_path(baseline, artifact_value, "proof artifact")
    if not isinstance(artifact_digest, str) or sha256_file(artifact_path) != artifact_digest:
        raise RunnerError(f"{baseline} {dataset} proof artifact digest does not match")
    if artifact_path.stat().st_size != marker.get("proof_size_bytes"):
        raise RunnerError(f"{baseline} {dataset} proof artifact size does not match its metric")
    if marker.get("independent") is not True and marker.get("independent_verification") not in (True, "1"):
        # The native result format carries VERIFY_OK; the phase adapter must
        # also attest the independent verifier rather than trusting status.
        raise RunnerError(f"{baseline} {dataset} marker lacks independent-verifier attestation")
    verification_path = (marker.get("verify_log_path") or marker.get("verification_path")
                         or marker.get("verification_report_path"))
    verification_digest = (marker.get("verify_log_sha256") or marker.get("verification_sha256")
                           or marker.get("verification_report_sha256"))
    if verification_path is None or verification_digest is None:
        raise RunnerError(f"{baseline} {dataset} marker is missing independent-verifier artifact identity")
    verify_artifact = _resolve_baseline_path(baseline, verification_path, "independent-verifier artifact")
    if not isinstance(verification_digest, str) or sha256_file(verify_artifact) != verification_digest:
        raise RunnerError(f"{baseline} {dataset} independent-verifier artifact digest mismatch")


def _validate_global_admission(markers: Mapping[str, Mapping[str, Any]],
                                session_id: str) -> dict[str, Any]:
    """Merge source-derived cell bounds against current shared resources."""
    profile_keys = {
        "schema", "baseline", "session_id", "fs_device_id",
        "preexisting_baseline_artifact_bytes", "finalization_disk_bytes_upper_bound",
        "bound_methods", "datasets", "memory_accounting_kind", "device",
    }


    required_phases = {"setup", "witness", "commitment", "prove", "verify"}
    accepted_methods = {"source_derived_upper_bound", "measured_full_phase"}
    profiles: dict[str, dict[str, Any]] = {}
    shared_device: dict[str, Any] | None = None
    root_device = str(os.stat(COMPARISON_ROOT, follow_symlinks=False).st_dev)
    fs = os.statvfs(COMPARISON_ROOT)
    available_disk = fs.f_bavail * fs.f_frsize
    for baseline in BASELINES:
        marker = markers.get(baseline)
        profile = marker.get("shared_capacity_profile") if isinstance(marker, Mapping) else None
        if not isinstance(profile, dict) or set(profile) != profile_keys:
            raise RunnerError(f"{baseline} prepare receipt has no complete shared-capacity profile")
        if (profile.get("schema") != "comparison-baseline-capacity-profile-v2" or
                profile.get("baseline") != baseline or profile.get("session_id") != session_id):
            raise RunnerError(f"{baseline} shared-capacity profile identity is stale")
        memory_kind = profile.get("memory_accounting_kind")
        device = profile.get("device")
        if baseline == "ezkl":
            if memory_kind != "unified-host-device":
                raise RunnerError("EZKL capacity profile must account for its unified host/device memory")
            if (not isinstance(device, dict) or set(device) != {"uuid", "total_memory_bytes"}
                    or not isinstance(device.get("uuid"), str)
                    or not re.fullmatch(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", device["uuid"])
                    or isinstance(device.get("total_memory_bytes"), bool)
                    or not isinstance(device.get("total_memory_bytes"), int)
                    or device["total_memory_bytes"] <= 0):
                raise RunnerError("EZKL capacity profile has no authenticated CUDA UUID and total memory")
            if shared_device is None:
                shared_device = dict(device)
            elif device != shared_device:
                raise RunnerError("EZKL capacity profiles disagree on the CUDA device identity")
        elif memory_kind != "host-only" or device is not None:
            raise RunnerError(f"{baseline} CPU capacity profile must use host-only accounting and device=null")
        if str(profile.get("fs_device_id")) != root_device:
            raise RunnerError(f"{baseline} session is not on the comparison results filesystem")
        session_dir = _baseline_session_dir(baseline, session_id)
        if session_dir.is_symlink() or not session_dir.is_dir():
            raise RunnerError(f"{baseline} prepared session directory is missing or aliased")
        if os.stat(session_dir, follow_symlinks=False).st_dev != os.stat(COMPARISON_ROOT, follow_symlinks=False).st_dev:
            raise RunnerError(f"{baseline} prepared session directory moved to another filesystem")
        actual_session_bytes = _directory_regular_bytes(session_dir)
        for number_key in ("preexisting_baseline_artifact_bytes", "finalization_disk_bytes_upper_bound"):
            value = profile.get(number_key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise RunnerError(f"{baseline} capacity profile has invalid {number_key}")
        if profile["preexisting_baseline_artifact_bytes"] < actual_session_bytes:
            raise RunnerError(f"{baseline} preexisting artifact audit understates its session tree")
        methods = profile.get("bound_methods")
        if not isinstance(methods, dict) or set(methods) != {"additional_disk_peak", "host_peak"}:
            raise RunnerError(f"{baseline} capacity profile has no explicit disk/host bound methods")
        if any(value not in accepted_methods for value in methods.values()):
            raise RunnerError(f"{baseline} capacity profile uses an unsupported peak method")
        cells = profile.get("datasets")
        if not isinstance(cells, dict) or set(cells) != set(DATASETS):
            raise RunnerError(f"{baseline} capacity profile does not cover exactly five datasets")
        for dataset in DATASETS:
            cell = cells[dataset]
            cell_keys = {
                "additional_disk_peak_bytes", "cell_retained_artifact_bytes_upper_bound",
                "host_peak_bytes", "device_peak_bytes", "phase_coverage",
            }
            if not isinstance(cell, dict) or set(cell) != cell_keys:
                raise RunnerError(f"{baseline}/{dataset} capacity profile fields are incomplete")
            for key in cell_keys - {"phase_coverage"}:
                value = cell[key]
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise RunnerError(f"{baseline}/{dataset} has invalid capacity value {key}")
            coverage = cell["phase_coverage"]
            if not isinstance(coverage, list) or not required_phases.issubset(set(coverage)):
                raise RunnerError(f"{baseline}/{dataset} peak bound omits a proof-chain phase")
            if cell["cell_retained_artifact_bytes_upper_bound"] > cell["additional_disk_peak_bytes"]:
                raise RunnerError(f"{baseline}/{dataset} retained output exceeds its additional disk peak")
            if memory_kind == "host-only" and cell["device_peak_bytes"] != 0:
                raise RunnerError(f"{baseline}/{dataset} host-only profile reports device memory")
            if memory_kind == "unified-host-device" and cell["host_peak_bytes"] < cell["device_peak_bytes"]:
                raise RunnerError(f"{baseline}/{dataset} combined host peak is below its device peak")
        profiles[baseline] = profile

    shared_graph_identity = _validate_shared_graph_identity(markers)
    shared_checkpoint_identity = _validate_shared_checkpoint_identity()
    available_host, host_method = _effective_host_memory_available()
    disk_prefix_retained = 0
    peak_active_disk = 0
    peak_cell_host = 0
    peak_cell_device = 0
    cell_budget: list[dict[str, Any]] = []
    for baseline in EXECUTION_BASELINES:
        profile = profiles[baseline]
        for dataset in DATASETS:
            cell = profile["datasets"][dataset]
            required_disk = disk_prefix_retained + cell["additional_disk_peak_bytes"]
            peak_active_disk = max(peak_active_disk, required_disk)
            peak_cell_host = max(peak_cell_host, cell["host_peak_bytes"])
            peak_cell_device = max(peak_cell_device, cell["device_peak_bytes"])
            cell_budget.append({
                "baseline": baseline,
                "dataset": dataset,
                "disk_required_bytes_from_prepare_end": required_disk,
                "additional_disk_peak_bytes": cell["additional_disk_peak_bytes"],
                "cell_retained_artifact_bytes_upper_bound": cell["cell_retained_artifact_bytes_upper_bound"],
                "disk_peak_method": profile["bound_methods"]["additional_disk_peak"],
                "host_peak_bytes": cell["host_peak_bytes"],
                "host_peak_method": profile["bound_methods"]["host_peak"],
                "device_peak_bytes": cell["device_peak_bytes"],
                "phase_coverage": list(cell["phase_coverage"]),
            })
            disk_prefix_retained += cell["cell_retained_artifact_bytes_upper_bound"]
    finalization_extra = sum(p["finalization_disk_bytes_upper_bound"] for p in profiles.values())
    final_disk_peak = disk_prefix_retained + finalization_extra + ROOT_FORMAL_OUTPUT_RESERVE
    required_disk = max(peak_active_disk + ROOT_FUTURE_LOG_RESERVE, final_disk_peak)
    if required_disk > available_disk:
        raise RunnerError(
            f"combined shared-filesystem admission blocked: required={required_disk} available={available_disk}"
        )
    if peak_cell_host > available_host:
        raise RunnerError(
            f"combined host-memory admission blocked: required={peak_cell_host} available={available_host} ({host_method})"
        )
    device_snapshot = None
    if shared_device is not None:
        device_snapshot = _cuda_device_memory_for_uuid(shared_device["uuid"])
        if device_snapshot["total_memory_bytes"] != shared_device["total_memory_bytes"]:
            raise RunnerError("EZKL CUDA total memory differs from the prepared device identity")
        if peak_cell_device > device_snapshot["free_memory_bytes"]:
            raise RunnerError(
                f"combined CUDA-memory admission blocked: required={peak_cell_device} "
                f"available={device_snapshot['free_memory_bytes']} uuid={shared_device['uuid']}"
            )
    return {
        "schema": "comparison-shared-capacity-admission-v1",
        "status": "READY",
        "session_id": session_id,
        "shared_graph_identity": shared_graph_identity,
        "shared_checkpoint_identity": shared_checkpoint_identity,
        "filesystem_device_id": root_device,
        "filesystem_available_after_prepare_bytes": available_disk,
        "required_peak_additional_disk_bytes": required_disk,
        "peak_active_cell_disk_bytes": peak_active_disk,
        "cumulative_cell_retained_upper_bound_bytes": disk_prefix_retained,
        "baseline_finalization_upper_bound_bytes": finalization_extra,
        "root_logs_and_results_reserve_bytes": ROOT_FORMAL_OUTPUT_RESERVE,
        "host_available_bytes": available_host,
        "host_availability_method": host_method,
        "required_peak_host_bytes": peak_cell_host,
        "cuda_device": shared_device,
        "cuda_free_bytes_at_admission": (device_snapshot["free_memory_bytes"] if device_snapshot else None),
        "required_peak_device_bytes": peak_cell_device,
        "preexisting_baseline_artifact_bytes_audit_only": {
            baseline: profiles[baseline]["preexisting_baseline_artifact_bytes"] for baseline in BASELINES
        },
        "cell_bounds": cell_budget,
        "admitted_at_unix_ns": time.time_ns(),
    }



def _validate_diagnostic_cell_capacity(baseline: str, dataset: str,
                                       marker: Mapping[str, Any], session_id: str) -> dict[str, Any]:
    """Admit one local diagnostic cell without claiming the 3x5 matrix READY."""
    profile = marker.get("shared_capacity_profile")
    expected_profile_keys = {
        "schema", "baseline", "session_id", "fs_device_id",
        "preexisting_baseline_artifact_bytes", "finalization_disk_bytes_upper_bound",
        "bound_methods", "datasets", "memory_accounting_kind", "device",
    }
    required_phases = {"setup", "witness", "commitment", "prove", "verify"}
    methods = {"source_derived_upper_bound", "measured_full_phase"}
    if not isinstance(profile, dict) or set(profile) != expected_profile_keys:
        raise RunnerError("diagnostic cell is blocked: prepare has no complete source/measured capacity profile")
    if (profile.get("schema") != "comparison-baseline-capacity-profile-v2"
            or profile.get("baseline") != baseline or profile.get("session_id") != session_id
            or str(profile.get("fs_device_id")) != str(os.stat(COMPARISON_ROOT, follow_symlinks=False).st_dev)):
        raise RunnerError("diagnostic capacity profile has stale session, baseline, or filesystem identity")
    memory_kind = profile.get("memory_accounting_kind")
    device = profile.get("device")
    if baseline == "ezkl":
        if (memory_kind != "unified-host-device" or not isinstance(device, dict)
                or set(device) != {"uuid", "total_memory_bytes"}
                or not isinstance(device.get("uuid"), str)
                or not re.fullmatch(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", device["uuid"])
                or isinstance(device.get("total_memory_bytes"), bool)
                or not isinstance(device.get("total_memory_bytes"), int)
                or device["total_memory_bytes"] <= 0):
            raise RunnerError("diagnostic EZKL profile lacks authenticated unified-device identity")
    elif memory_kind != "host-only" or device is not None:
        raise RunnerError("diagnostic CPU profile must use host-only accounting and device=null")
    method_record = profile.get("bound_methods")
    if (not isinstance(method_record, dict)
            or set(method_record) != {"additional_disk_peak", "host_peak"}
            or any(value not in methods for value in method_record.values())):
        raise RunnerError("diagnostic capacity profile has unsupported host/disk bound methods")
    datasets = profile.get("datasets")
    if not isinstance(datasets, dict) or set(datasets) != set(DATASETS):
        raise RunnerError("diagnostic capacity profile does not cover the five full datasets")
    cell = datasets.get(dataset)
    expected_cell_keys = {
        "additional_disk_peak_bytes", "cell_retained_artifact_bytes_upper_bound",
        "host_peak_bytes", "device_peak_bytes", "phase_coverage",
    }
    if not isinstance(cell, dict) or set(cell) != expected_cell_keys:
        raise RunnerError(f"diagnostic capacity profile has no complete {dataset} bound")
    for key in expected_cell_keys - {"phase_coverage"}:
        value = cell.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RunnerError(f"diagnostic {dataset} capacity field {key} is invalid")
    coverage = cell.get("phase_coverage")
    if not isinstance(coverage, list) or not required_phases.issubset(set(coverage)):
        raise RunnerError(f"diagnostic {dataset} source/measured bound omits a proof phase")
    if cell["cell_retained_artifact_bytes_upper_bound"] > cell["additional_disk_peak_bytes"]:
        raise RunnerError(f"diagnostic {dataset} retained files exceed the cell disk peak")
    if memory_kind == "host-only" and cell["device_peak_bytes"] != 0:
        raise RunnerError(f"diagnostic {dataset} CPU profile reports device memory")
    if memory_kind == "unified-host-device" and cell["host_peak_bytes"] < cell["device_peak_bytes"]:
        raise RunnerError(f"diagnostic {dataset} combined host peak is below its device peak")
    root_fs = os.statvfs(COMPARISON_ROOT)
    available_disk = root_fs.f_bavail * root_fs.f_frsize
    disk_required = cell["additional_disk_peak_bytes"] + ROOT_CELL_LOG_LIMIT + ROOT_RUNNER_LOG_RESERVE
    if disk_required > available_disk:
        raise RunnerError(
            f"diagnostic {baseline}/{dataset} disk admission blocked: required={disk_required} available={available_disk}"
        )
    available_host, host_method = _effective_host_memory_available()
    if cell["host_peak_bytes"] > available_host:
        raise RunnerError(
            f"diagnostic {baseline}/{dataset} host admission blocked: required={cell['host_peak_bytes']} "
            f"available={available_host} ({host_method})"
        )
    device_snapshot = None
    if memory_kind == "unified-host-device":
        device_snapshot = _cuda_device_memory_for_uuid(device["uuid"])
        if device_snapshot["total_memory_bytes"] != device["total_memory_bytes"]:
            raise RunnerError("diagnostic EZKL CUDA total memory differs from prepared device identity")
        if cell["device_peak_bytes"] > device_snapshot["free_memory_bytes"]:
            raise RunnerError(
                f"diagnostic {baseline}/{dataset} CUDA-memory admission blocked: "
                f"required={cell['device_peak_bytes']} available={device_snapshot['free_memory_bytes']}"
            )
    return {
        "status": "SINGLE_CELL_ADMITTED",
        "baseline": baseline,
        "dataset": dataset,
        "disk_required_bytes": disk_required,
        "disk_available_bytes": available_disk,
        "host_required_bytes": cell["host_peak_bytes"],
        "host_available_bytes": available_host,
        "host_availability_method": host_method,
        "device_required_bytes": cell["device_peak_bytes"],
        "device_available_bytes": (device_snapshot["free_memory_bytes"] if device_snapshot else 0),
        "device_uuid": (device["uuid"] if device_snapshot else None),
        "bound_methods": dict(method_record),
        "phase_coverage": list(coverage),
    }

def _source_key_values(path: Path) -> tuple[dict[str, str], str]:
    if path.is_symlink() or not path.is_file():
        raise RunnerError(f"canonical graph source is missing or aliased: {path}")
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise RunnerError(f"cannot read canonical graph source {path}: {exc}") from exc
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if "=" not in line:
            raise RunnerError(f"malformed key/value source {path}:{line_number}")
        key, value = (part.strip() for part in line.split("=", 1))
        if not key or key in values:
            raise RunnerError(f"empty or duplicate key in canonical graph source {path}:{line_number}")
        values[key] = value
    return values, hashlib.sha256(raw).hexdigest()


def _source_uint(value: str, label: str, *, positive: bool = False) -> int:
    if not re.fullmatch(r"[0-9]+", value):
        raise RunnerError(f"{label} is not an unsigned decimal integer")
    result = int(value)
    if result >= 1 << 64 or (positive and result == 0):
        raise RunnerError(f"{label} is outside the canonical u64 range")
    return result


def _comparison_graph_sha256(nodes: int, edges: int, graphs: int,
                             src: Sequence[int], dst: Sequence[int],
                             node_ptr: Sequence[int], edge_ptr: Sequence[int]) -> str:
    digest = hashlib.sha256()
    digest.update(b"comparison-canonical-graph-v1\0")
    for value in (nodes, edges, graphs):
        digest.update(struct.pack("<Q", value))
    for values in (src, dst, node_ptr, edge_ptr):
        digest.update(struct.pack("<Q", len(values)))
        for value in values:
            if value < 0 or value >= 1 << 64:
                raise RunnerError("canonical graph coordinate is outside the u64 range")
            digest.update(struct.pack("<Q", value))
    return digest.hexdigest()


def _expected_canonical_graph(dataset: str) -> tuple[dict[str, Any], dict[str, Any]]:
    config_path = PROJECT_ROOT / "configs" / f"{dataset}_secure.cfg"
    config, config_sha256 = _source_key_values(config_path)
    cache_dir = PROJECT_ROOT / "data" / "cache" / dataset
    meta, meta_sha256 = _source_key_values(cache_dir / "meta.cfg")
    if config.get("dataset") != dataset or meta.get("name") != dataset:
        raise RunnerError(f"canonical graph source names do not match {dataset}")
    nodes = _source_uint(meta.get("num_nodes", ""), f"{dataset} meta.num_nodes", positive=True)
    meta_graphs = _source_uint(meta.get("graph_count", "1"), f"{dataset} meta.graph_count", positive=True)
    if (config.get("self_loop_rule") != "per_node" or
            config.get("edge_sort_rule") != "edge_gid_then_dst_stable" or
            config.get("batching_rule") not in ("whole_graph_single", "multi_graph_batch") or
            config.get("input_preprocess_id") != meta.get("input_preprocess_id")):
        raise RunnerError(f"canonical graph config semantics do not match source metadata for {dataset}")
    deduplicate = config.get("deduplicate_edges", "true").lower()
    symmetrize = config.get("symmetrize_edges", "false").lower()
    if deduplicate not in ("true", "false") or symmetrize not in ("true", "false"):
        raise RunnerError(f"canonical graph edge policy is malformed for {dataset}")
    if deduplicate != "true" or symmetrize != "false":
        raise RunnerError(f"canonical graph edge policy is not the approved formal policy for {dataset}")

    node_ptr_path = cache_dir / "node_ptr.txt"
    source_digests: dict[str, Any] = {
        "config_sha256": config_sha256,
        "meta_sha256": meta_sha256,
        "edges_sha256": "",
        "node_ptr_sha256": None,
    }
    if node_ptr_path.is_symlink():
        raise RunnerError(f"canonical node_ptr source is missing or aliased: {node_ptr_path}")
    if node_ptr_path.exists():
        if not node_ptr_path.is_file():
            raise RunnerError(f"canonical node_ptr source is missing or aliased: {node_ptr_path}")
        try:
            node_ptr_text = node_ptr_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise RunnerError(f"cannot read canonical node_ptr source {node_ptr_path}: {exc}") from exc
        source_digests["node_ptr_sha256"] = sha256_file(node_ptr_path)
        node_ptr = [_source_uint(token, f"{dataset} node_ptr") for token in node_ptr_text.split()]
    else:
        node_ptr = [0, nodes]
    if len(node_ptr) < 2 or node_ptr[0] != 0 or node_ptr[-1] != nodes or any(a >= b for a, b in zip(node_ptr, node_ptr[1:])):
        raise RunnerError(f"canonical node_ptr boundaries are invalid for {dataset}")
    graph_count = len(node_ptr) - 1
    if graph_count != meta_graphs:
        raise RunnerError(f"canonical node_ptr graph count conflicts with metadata for {dataset}")
    batch_graphs = _source_uint(config.get("batch_graphs", "1"), f"{dataset} config.batch_graphs", positive=True)
    selected_count = min(batch_graphs, graph_count)
    if dataset == "ppi":
        if graph_count != 295 or batch_graphs != 295 or selected_count != graph_count:
            raise RunnerError("PPI canonical graph source must retain all 295 graph boundaries")
    elif graph_count != 1 or config.get("batching_rule") != "whole_graph_single" or selected_count != 1:
        raise RunnerError(f"whole-graph formal config does not select the complete graph for {dataset}")
    if nodes >= 1 << 24 or graph_count >= 1 << 16:
        raise RunnerError(f"canonical graph coordinates exceed the production dedup key range for {dataset}")

    edge_path = cache_dir / "edges.txt"
    if edge_path.is_symlink() or not edge_path.is_file():
        raise RunnerError(f"canonical edge source is missing or aliased: {edge_path}")
    source_digests["edges_sha256"] = sha256_file(edge_path)
    edges_by_graph: list[list[tuple[int, int]]] = [[] for _ in range(graph_count)]
    try:
        with edge_path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                fields = line.split()
                if not fields:
                    continue
                if len(fields) == 2:
                    graph_id, source, destination = "0", fields[0], fields[1]
                elif len(fields) == 3:
                    graph_id, source, destination = fields
                else:
                    raise RunnerError(f"malformed canonical edge source {edge_path}:{line_number}")
                gid = _source_uint(graph_id, f"{dataset} edge graph id")
                src_id = _source_uint(source, f"{dataset} edge source")
                dst_id = _source_uint(destination, f"{dataset} edge destination")
                if gid >= graph_count:
                    raise RunnerError(f"canonical edge graph id is out of range for {dataset}")
                if gid < selected_count:
                    edges_by_graph[gid].append((src_id, dst_id))
    except (OSError, UnicodeDecodeError) as exc:
        raise RunnerError(f"cannot read canonical edge source {edge_path}: {exc}") from exc

    normalized_src: list[int] = []
    normalized_dst: list[int] = []
    normalized_node_ptr = [0]
    normalized_edge_ptr = [0]
    for graph_id in range(selected_count):
        node_begin, node_end = node_ptr[graph_id], node_ptr[graph_id + 1]
        local_offset = normalized_node_ptr[-1]
        seen: set[tuple[int, int]] = set()
        graph_edges: list[tuple[int, int, int]] = []
        stable_index = 0
        for source, destination in edges_by_graph[graph_id]:
            if not (node_begin <= source < node_end and node_begin <= destination < node_end):
                continue
            local_src = local_offset + source - node_begin
            local_dst = local_offset + destination - node_begin
            pair = (local_src, local_dst)
            if pair in seen:
                continue
            seen.add(pair)
            graph_edges.append((local_src, local_dst, stable_index))
            stable_index += 1
        for local_node in range(local_offset, local_offset + node_end - node_begin):
            pair = (local_node, local_node)
            if pair in seen:
                continue
            seen.add(pair)
            graph_edges.append((local_node, local_node, stable_index))
            stable_index += 1
        graph_edges.sort(key=lambda edge: (edge[1], edge[2]))
        normalized_src.extend(edge[0] for edge in graph_edges)
        normalized_dst.extend(edge[1] for edge in graph_edges)
        normalized_node_ptr.append(local_offset + node_end - node_begin)
        normalized_edge_ptr.append(len(normalized_src))
    normalized_nodes = normalized_node_ptr[-1]
    normalized_edges = len(normalized_src)
    return ({
        "digest": _comparison_graph_sha256(normalized_nodes, normalized_edges, selected_count,
                                           normalized_src, normalized_dst,
                                           normalized_node_ptr, normalized_edge_ptr),
        "nodes": normalized_nodes,
        "edges": normalized_edges,
        "graphs": selected_count,
        "node_ptr": normalized_node_ptr,
        "edge_ptr": normalized_edge_ptr,
        "input_preprocess_id": config["input_preprocess_id"],
        "batching_rule": config["batching_rule"],
    }, source_digests)


def _validate_shared_graph_identity(markers: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    expected: dict[str, Any] = {}
    source_files: dict[str, Any] = {}
    for dataset in DATASETS:
        expected[dataset], source_files[dataset] = _expected_canonical_graph(dataset)
    expected_identity = {"schema": "comparison-canonical-graph-set-v1", "datasets": expected}
    row_keys = {"digest", "nodes", "edges", "graphs", "node_ptr", "edge_ptr", "input_preprocess_id", "batching_rule"}
    for baseline in BASELINES:
        marker = markers.get(baseline)
        identity = marker.get("shared_graph_identity") if isinstance(marker, Mapping) else None
        if not isinstance(identity, dict) or set(identity) != {"schema", "datasets"}:
            raise RunnerError(f"{baseline} prepare receipt has no strict shared graph identity")
        if identity.get("schema") != "comparison-canonical-graph-set-v1":
            raise RunnerError(f"{baseline} shared graph identity schema is unsupported")
        datasets = identity.get("datasets")
        if not isinstance(datasets, dict) or set(datasets) != set(DATASETS):
            raise RunnerError(f"{baseline} shared graph identity does not cover exactly five datasets")
        for dataset in DATASETS:
            row = datasets[dataset]
            if not isinstance(row, dict) or set(row) != row_keys:
                raise RunnerError(f"{baseline}/{dataset} shared graph identity fields are incomplete")
            if (not isinstance(row["digest"], str) or not re.fullmatch(r"[0-9a-f]{64}", row["digest"]) or
                    any(type(row[key]) is not int or row[key] < 0 for key in ("nodes", "edges", "graphs")) or
                    not isinstance(row["node_ptr"], list) or not isinstance(row["edge_ptr"], list) or
                    any(type(value) is not int or value < 0 for value in row["node_ptr"] + row["edge_ptr"]) or
                    not isinstance(row["input_preprocess_id"], str) or not isinstance(row["batching_rule"], str)):
                raise RunnerError(f"{baseline}/{dataset} shared graph identity contains invalid values")
            if row != expected[dataset]:
                raise RunnerError(f"{baseline}/{dataset} native graph identity differs from the authoritative project graph")
    return {**expected_identity, "authoritative_source_sha256": source_files}


def _canonical_checkpoint_source_identity(dataset: str) -> dict[str, Any]:
    bundle = PROJECT_ROOT / "artifacts" / "checkpoints" / f"{dataset}_gat"
    manifest_path = bundle / "manifest.json"
    tensors_npz = bundle / "tensors.npz"
    tensors_txt = bundle / "tensors.txt"
    for path in (manifest_path, tensors_npz, tensors_txt):
        _require_unaliased_path(path, PROJECT_ROOT, "canonical checkpoint source")
        if path.is_symlink() or not path.is_file():
            raise RunnerError(f"canonical checkpoint source is missing or aliased: {path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerError(f"canonical checkpoint manifest cannot be read for {dataset}: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("dataset") != dataset:
        raise RunnerError(f"canonical checkpoint manifest identifies another dataset: {dataset}")
    scripts_dir = str(PROJECT_ROOT / "scripts")
    inserted = scripts_dir not in sys.path
    if inserted:
        sys.path.insert(0, scripts_dir)
    try:
        try:
            from checkpoint_bundle import validate_checkpoint_manifest, validate_tensor_encodings
        except ImportError as exc:
            raise RunnerError(f"canonical checkpoint validation code is unavailable: {exc}") from exc
        try:
            validate_checkpoint_manifest(manifest)
            tensors = validate_tensor_encodings(manifest, tensors_npz, tensors_txt)
        except (OSError, ValueError, TypeError) as exc:
            raise RunnerError(f"canonical checkpoint source failed tensor validation for {dataset}: {exc}") from exc
    finally:
        if inserted:
            sys.path.remove(scripts_dir)

    tensor_index = manifest.get("tensor_index")
    if not isinstance(tensor_index, dict) or set(tensor_index) != set(tensors):
        raise RunnerError(f"canonical checkpoint tensor index is incomplete for {dataset}")
    digest = hashlib.sha256()
    digest.update(b"comparison-canonical-checkpoint-v1\0")
    digest.update(struct.pack("<Q", len(tensor_index)))
    for name in sorted(tensor_index, key=lambda item: item.encode("utf-8")):
        entry = tensor_index[name]
        tensor = tensors[name]
        if not isinstance(entry, dict) or not isinstance(name, str):
            raise RunnerError(f"canonical checkpoint tensor index row is malformed for {dataset}")
        dtype = entry.get("dtype")
        shape = entry.get("shape")
        if dtype != "float32" or not isinstance(shape, list) or tuple(shape) != tuple(tensor.shape):
            raise RunnerError(f"canonical checkpoint tensor dtype/shape changed for {dataset}/{name}")
        canonical_bytes = tensor.tobytes(order="C")
        raw_sha256 = hashlib.sha256(canonical_bytes).digest()
        encoded_name = name.encode("utf-8")
        encoded_dtype = dtype.encode("ascii")
        digest.update(struct.pack("<Q", len(encoded_name)))
        digest.update(encoded_name)
        digest.update(struct.pack("<Q", len(encoded_dtype)))
        digest.update(encoded_dtype)
        digest.update(struct.pack("<Q", len(shape)))
        for dimension in shape:
            if type(dimension) is not int or dimension <= 0 or dimension >= 1 << 64:
                raise RunnerError(f"canonical checkpoint tensor shape is invalid for {dataset}/{name}")
            digest.update(struct.pack("<Q", dimension))
        digest.update(raw_sha256)

    semantics = {}
    for field in ("model_arch_id", "model_param_id", "input_preprocess_id", "quant_cfg_id",
                  "quantization_scale_chain", "L", "K_out", "C", "d_in_profile",
                  "hidden_profile", "output_average_rule", "task_type", "report_unit", "tensor_count"):
        if field not in manifest:
            raise RunnerError(f"canonical checkpoint manifest omits {field} for {dataset}")
        semantics[field] = manifest[field]
    return {
        "canonical_source_digest": digest.hexdigest(),
        "tensor_count": len(tensor_index),
        "tensor_index_sha256": hashlib.sha256(_json_bytes(tensor_index)).hexdigest(),
        "semantics": semantics,
        "source_files_sha256": {
            "manifest": sha256_file(manifest_path),
            "tensors_npz": sha256_file(tensors_npz),
            "tensors_txt": sha256_file(tensors_txt),
        },
    }


def _checkpoint_contract(baseline: str, dataset: str) -> tuple[Path, str, Path]:
    if baseline == "ezkl":
        baseline_root = COMPARISON_ROOT / "ezkl"
        config = baseline_root / "resources" / "canonical" / "configs" / f"{dataset}_secure.cfg"
        bundle_relative = f"resources/canonical/checkpoints/{dataset}_gat"
    elif baseline == "zkdl":
        baseline_root = COMPARISON_ROOT / "zkdl"
        config = baseline_root / "configs" / f"{dataset}_secure.cfg"
        bundle_relative = f"models/checkpoints/{dataset}_gat"
    elif baseline == "zkgpt":
        baseline_root = COMPARISON_ROOT / "zkGPT"
        config = baseline_root / "baseline-local" / "configs" / f"{dataset}_secure.cfg"
        bundle_relative = f"baseline-local/checkpoints/{dataset}_gat"
    else:
        raise RunnerError(f"unknown baseline checkpoint contract: {baseline}")
    return config, bundle_relative, baseline_root


def _validate_one_checkpoint_source(baseline: str, dataset: str,
                                    canonical: Mapping[str, Any]) -> dict[str, Any]:
    config_path, bundle_relative, baseline_root = _checkpoint_contract(baseline, dataset)
    _require_unaliased_path(config_path, baseline_root, f"{baseline}/{dataset} config")
    config, _ = _source_key_values(config_path)
    if config.get("dataset") != dataset or config.get("checkpoint_bundle") != bundle_relative:
        raise RunnerError(f"{baseline}/{dataset} config does not select its authenticated checkpoint bundle")
    bundle = baseline_root / bundle_relative
    _require_unaliased_path(bundle, baseline_root, f"{baseline}/{dataset} checkpoint bundle")
    if bundle.is_symlink() or not bundle.is_dir():
        raise RunnerError(f"{baseline}/{dataset} checkpoint bundle is missing or aliased")
    for name in ("manifest.json", "tensors.npz", "tensors.txt"):
        path = bundle / name
        _require_unaliased_path(path, baseline_root, f"{baseline}/{dataset} checkpoint")
        if path.is_symlink() or not path.is_file():
            raise RunnerError(f"{baseline}/{dataset} checkpoint file is missing or aliased: {path}")

    try:
        manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerError(f"{baseline}/{dataset} checkpoint manifest is invalid: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("dataset") != dataset:
        raise RunnerError(f"{baseline}/{dataset} checkpoint manifest identifies another dataset")
    for key, value in canonical["semantics"].items():
        if manifest.get(key) != value:
            raise RunnerError(f"{baseline}/{dataset} checkpoint semantic field differs from canonical source: {key}")
    tensor_index = manifest.get("tensor_index")
    if (not isinstance(tensor_index, dict) or
            hashlib.sha256(_json_bytes(tensor_index)).hexdigest() != canonical["tensor_index_sha256"]):
        raise RunnerError(f"{baseline}/{dataset} checkpoint tensor index differs from canonical source")

    files = {
        "manifest": sha256_file(bundle / "manifest.json"),
        "tensors_npz": sha256_file(bundle / "tensors.npz"),
        "tensors_txt": sha256_file(bundle / "tensors.txt"),
    }
    expected_files = canonical["source_files_sha256"]
    for field in ("tensors_npz", "tensors_txt"):
        if files[field] != expected_files[field]:
            raise RunnerError(f"{baseline}/{dataset} authenticated tensor bytes differ from the canonical checkpoint")

    config_semantics = ("model_arch_id", "model_param_id", "input_preprocess_id", "quant_cfg_id",
                        "L", "K_out", "C", "d_in_profile", "task_type")
    for key in config_semantics:
        if key in config and key in canonical["semantics"]:
            value = canonical["semantics"][key]
            encoded = value if isinstance(value, str) else json.dumps(value, sort_keys=True, separators=(",", ":"))
            if config[key] != encoded:
                raise RunnerError(f"{baseline}/{dataset} config model semantics differ from canonical checkpoint: {key}")
    return {"bundle": bundle_relative, "manifest_sha256": files["manifest"],
            "tensors_npz_sha256": files["tensors_npz"], "tensors_txt_sha256": files["tensors_txt"]}


def _validate_shared_checkpoint_identity() -> dict[str, Any]:
    """Bind all native loaders to the same authenticated project checkpoint tensors."""
    datasets: dict[str, Any] = {}
    for dataset in DATASETS:
        canonical = _canonical_checkpoint_source_identity(dataset)
        baseline_files = {baseline: _validate_one_checkpoint_source(baseline, dataset, canonical)
                          for baseline in BASELINES}
        datasets[dataset] = {
            "canonical_source_digest": canonical["canonical_source_digest"],
            "tensor_index_sha256": canonical["tensor_index_sha256"],
            "tensor_count": canonical["tensor_count"],
            "semantics": canonical["semantics"],
            "source_files_sha256": canonical["source_files_sha256"],
            "baseline_resources": baseline_files,
        }
    return {"schema": "comparison-canonical-checkpoint-set-v1", "datasets": datasets}


def _recheck_checkpoint_source(admission: Mapping[str, Any], baseline: str, dataset: str) -> None:
    shared = admission.get("shared_checkpoint_identity")
    if not isinstance(shared, dict) or shared.get("schema") != "comparison-canonical-checkpoint-set-v1":
        raise RunnerError("global admission has no canonical checkpoint identity")
    datasets = shared.get("datasets")
    expected = datasets.get(dataset) if isinstance(datasets, dict) else None
    resources = expected.get("baseline_resources") if isinstance(expected, dict) else None
    expected_resource = resources.get(baseline) if isinstance(resources, dict) else None
    if not isinstance(expected_resource, dict):
        raise RunnerError(f"global admission has no checkpoint identity for {baseline}/{dataset}")
    config_path, bundle_relative, baseline_root = _checkpoint_contract(baseline, dataset)
    _require_unaliased_path(config_path, baseline_root, f"{baseline}/{dataset} config")
    config, _ = _source_key_values(config_path)
    if config.get("checkpoint_bundle") != bundle_relative:
        raise RunnerError(f"{baseline}/{dataset} checkpoint bundle config changed after global admission")
    canonical_bundle = PROJECT_ROOT / "artifacts" / "checkpoints" / f"{dataset}_gat"
    local_bundle = baseline_root / bundle_relative
    _require_unaliased_path(canonical_bundle, PROJECT_ROOT, f"canonical {dataset} checkpoint bundle")
    _require_unaliased_path(local_bundle, baseline_root, f"{baseline}/{dataset} checkpoint bundle")
    for bundle in (canonical_bundle, local_bundle):
        if bundle.is_symlink() or not bundle.is_dir():
            raise RunnerError(f"{baseline}/{dataset} checkpoint bundle moved or became aliased")
    for name, key in (("manifest.json", "manifest_sha256"),
                      ("tensors.npz", "tensors_npz_sha256"),
                      ("tensors.txt", "tensors_txt_sha256")):
        canonical_path = canonical_bundle / name
        local_path = local_bundle / name
        _require_unaliased_path(canonical_path, PROJECT_ROOT, f"canonical {dataset} checkpoint file")
        _require_unaliased_path(local_path, baseline_root, f"{baseline}/{dataset} checkpoint file")
        if (canonical_path.is_symlink() or local_path.is_symlink() or
                not canonical_path.is_file() or not local_path.is_file()):
            raise RunnerError(f"{baseline}/{dataset} checkpoint input is missing or aliased: {name}")
        local_sha = sha256_file(local_path)
        canonical_sha = expected["source_files_sha256"]["tensors_npz" if name == "tensors.npz" else
                                                        "tensors_txt" if name == "tensors.txt" else "manifest"]
        if sha256_file(canonical_path) != canonical_sha:
            raise RunnerError(f"canonical checkpoint source changed after global admission: {dataset}/{name}")
        local_expected = expected_resource[key]
        if local_sha != local_expected:
            raise RunnerError(f"{baseline}/{dataset} checkpoint resource changed after global admission: {name}")
        if name != "manifest.json" and local_sha != canonical_sha:
            raise RunnerError(f"{baseline}/{dataset} tensor bytes differ from canonical source after global admission")
    try:
        local_manifest = json.loads((local_bundle / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerError(f"{baseline}/{dataset} checkpoint manifest changed after global admission: {exc}") from exc
    if (not isinstance(local_manifest, dict) or
            hashlib.sha256(_json_bytes(local_manifest.get("tensor_index"))).hexdigest() != expected["tensor_index_sha256"] or
            any(local_manifest.get(key) != value for key, value in expected["semantics"].items())):
        raise RunnerError(f"{baseline}/{dataset} checkpoint manifest identity changed after global admission")


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
        raise RunnerError("cannot determine system MemAvailable")
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


def _cuda_device_memory_for_uuid(expected_uuid: str) -> dict[str, int | str]:
    """Query cudaMemGetInfo on the runtime device bound to an exact CUDA UUID."""
    runtime_candidates = (
        Path("/usr/local/cuda/lib64/libcudart.so"),
        Path("/usr/local/cuda/targets/aarch64-linux/lib/libcudart.so"),
        Path("/usr/local/cuda/targets/x86_64-linux/lib/libcudart.so"),
    )
    runtime_path = next((path for path in runtime_candidates if path.exists()), None)
    if runtime_path is None:
        raise RunnerError("cannot locate the pinned CUDA runtime for cudaMemGetInfo")
    try:
        runtime = ctypes.CDLL(str(runtime_path))
    except OSError as exc:
        raise RunnerError(f"cannot load the pinned CUDA runtime: {exc}") from exc
    driver_names = (ctypes.util.find_library("cuda"), "libcuda.so.1")
    driver = None
    for name in driver_names:
        if not name:
            continue
        try:
            driver = ctypes.CDLL(name)
            break
        except OSError:
            continue
    if driver is None:
        raise RunnerError("cannot load the CUDA driver to bind cudaMemGetInfo to a UUID")

    class _CudaUuid(ctypes.Structure):
        _fields_ = [("bytes", ctypes.c_ubyte * 16)]

    driver.cuInit.argtypes = [ctypes.c_uint]
    driver.cuInit.restype = ctypes.c_int
    driver.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    driver.cuDeviceGetCount.restype = ctypes.c_int
    driver.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    driver.cuDeviceGet.restype = ctypes.c_int
    uuid_function = getattr(driver, "cuDeviceGetUuid_v2", None) or getattr(driver, "cuDeviceGetUuid", None)
    if uuid_function is None:
        raise RunnerError("CUDA driver has no cuDeviceGetUuid API")
    uuid_function.argtypes = [ctypes.POINTER(_CudaUuid), ctypes.c_int]
    uuid_function.restype = ctypes.c_int

    runtime.cudaGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    runtime.cudaGetDeviceCount.restype = ctypes.c_int
    runtime.cudaGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
    runtime.cudaGetDevice.restype = ctypes.c_int
    runtime.cudaSetDevice.argtypes = [ctypes.c_int]
    runtime.cudaSetDevice.restype = ctypes.c_int
    runtime.cudaDeviceReset.argtypes = []
    runtime.cudaDeviceReset.restype = ctypes.c_int
    runtime.cudaMemGetInfo.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    runtime.cudaMemGetInfo.restype = ctypes.c_int

    def check_status(status: int, operation: str) -> None:
        if status != 0:
            raise RunnerError(f"{operation} failed with CUDA status {status}")

    check_status(driver.cuInit(0), "cuInit")
    driver_count = ctypes.c_int()
    check_status(driver.cuDeviceGetCount(ctypes.byref(driver_count)), "cuDeviceGetCount")
    runtime_count = ctypes.c_int()
    check_status(runtime.cudaGetDeviceCount(ctypes.byref(runtime_count)), "cudaGetDeviceCount")
    if driver_count.value <= 0 or runtime_count.value <= 0 or driver_count.value != runtime_count.value:
        raise RunnerError("CUDA driver/runtime device enumeration is empty or uses different ordinal visibility")

    selected_ordinal = None
    for ordinal in range(driver_count.value):
        device = ctypes.c_int()
        check_status(driver.cuDeviceGet(ctypes.byref(device), ordinal), "cuDeviceGet")
        uuid = _CudaUuid()
        check_status(uuid_function(ctypes.byref(uuid), device.value), "cuDeviceGetUuid")
        raw = bytes(uuid.bytes).hex()
        device_uuid = "GPU-" + "-".join((raw[:8], raw[8:12], raw[12:16], raw[16:20], raw[20:]))
        if device_uuid == expected_uuid:
            selected_ordinal = ordinal
            break
    if selected_ordinal is None:
        raise RunnerError(f"prepared CUDA UUID is not visible to the formal runner: {expected_uuid}")

    selected = False
    try:
        check_status(runtime.cudaSetDevice(selected_ordinal), "cudaSetDevice")
        selected = True
        free_memory = ctypes.c_size_t()
        total_memory = ctypes.c_size_t()
        check_status(
            runtime.cudaMemGetInfo(ctypes.byref(free_memory), ctypes.byref(total_memory)),
            "cudaMemGetInfo",
        )
    finally:
        if selected:
            reset_status = runtime.cudaDeviceReset()
            if reset_status != 0:
                raise RunnerError(f"cudaDeviceReset failed after capacity query: {reset_status}")
    if total_memory.value <= 0 or free_memory.value > total_memory.value:
        raise RunnerError("cudaMemGetInfo returned an invalid free/total byte count")
    return {
        "uuid": expected_uuid,
        "free_memory_bytes": int(free_memory.value),
        "total_memory_bytes": int(total_memory.value),
    }


def _recheck_remaining_capacity(admission: Mapping[str, Any], next_cell_index: int) -> None:
    """Recheck live free resources before every long proof cell starts."""
    cells = admission.get("cell_bounds")
    if not isinstance(cells, list) or not 0 <= next_cell_index < len(cells):
        raise RunnerError("cannot recheck capacity with an invalid cell cursor")
    fs = os.statvfs(COMPARISON_ROOT)
    available_disk = fs.f_bavail * fs.f_frsize
    prefix_retained = 0
    remaining_cell_peak = 0
    remaining_host_peak = 0
    remaining_device_peak = 0
    remaining_cells = cells[next_cell_index:]
    for cell in remaining_cells:
        additional = cell["additional_disk_peak_bytes"]
        retained = cell["cell_retained_artifact_bytes_upper_bound"]
        remaining_cell_peak = max(remaining_cell_peak, prefix_retained + additional)
        remaining_host_peak = max(remaining_host_peak, cell["host_peak_bytes"])
        remaining_device_peak = max(remaining_device_peak, cell["device_peak_bytes"])
        prefix_retained += retained
    finalization = int(admission["baseline_finalization_upper_bound_bytes"])
    future_cell_logs = len(remaining_cells) * (ROOT_CELL_LOG_LIMIT + 1024)
    future_final = (finalization + ROOT_FINAL_LOG_LIMIT + ROOT_RUNNER_LOG_RESERVE +
                    ROOT_FINAL_DATA_RESERVE + ROOT_RECEIPT_RESERVE)
    required_disk = max(remaining_cell_peak + future_cell_logs + future_final,
                        prefix_retained + future_final)
    if required_disk > available_disk:
        raise RunnerError(
            f"live shared-filesystem recheck blocked before cell "
            f"{remaining_cells[0]['baseline']}/{remaining_cells[0]['dataset']}: "
            f"required_remaining={required_disk} available={available_disk}"
        )
    available_host, host_method = _effective_host_memory_available()
    if remaining_host_peak > available_host:
        raise RunnerError(
            f"live host-memory recheck blocked before cell "
            f"{remaining_cells[0]['baseline']}/{remaining_cells[0]['dataset']}: "
            f"required_remaining={remaining_host_peak} available={available_host} ({host_method})"
        )
    if remaining_device_peak > 0:
        device = admission.get("cuda_device")
        if not isinstance(device, dict) or not isinstance(device.get("uuid"), str):
            raise RunnerError("live CUDA-memory recheck has no authenticated device identity")
        snapshot = _cuda_device_memory_for_uuid(device["uuid"])
        if snapshot["total_memory_bytes"] != device.get("total_memory_bytes"):
            raise RunnerError("live CUDA total memory differs from the admitted device identity")
        if remaining_device_peak > snapshot["free_memory_bytes"]:
            raise RunnerError(
                f"live CUDA-memory recheck blocked before cell "
                f"{remaining_cells[0]['baseline']}/{remaining_cells[0]['dataset']}: "
                f"required_remaining={remaining_device_peak} available={snapshot['free_memory_bytes']} "
                f"uuid={device['uuid']}"
            )


def _run_formal(session_id: str, inherited_lock_fd: int) -> int:
    root = _session_root(session_id)
    owner_token = os.environ.get(OWNER_TOKEN_ENV, "")
    if not re.fullmatch(r"[0-9a-f]{64}", owner_token):
        raise RunnerError("root worker has no per-session internal owner token")
    _validate_root_lock_fd(inherited_lock_fd)
    token_sha256 = hashlib.sha256(owner_token.encode("ascii")).hexdigest()
    os.environ[SESSION_ID_ENV] = session_id
    os.environ[ROOT_LOCK_FD_ENV] = str(inherited_lock_fd)
    lock_record = {
        "pid": os.getpid(), "boot_id": _read_boot_id(),
        "start_ticks": _proc_start_ticks(os.getpid()),
    }
    os.ftruncate(inherited_lock_fd, 0)
    os.lseek(inherited_lock_fd, 0, os.SEEK_SET)
    os.write(inherited_lock_fd, _json_bytes({**lock_record, "session_id": session_id}))
    os.fsync(inherited_lock_fd)
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    state_path = root / "state.json"
    state = {
        "schema": "comparison-formal-session-v1", "session_id": session_id,
        "status": "RUNNING", "stage": "startup", "exit_code": None,
        "failure": "", "started_unix_ns": time.time_ns(), "updated_unix_ns": time.time_ns(),
        "completed_stages": [], "cell_receipts": [],
        "lock": lock_record,
        "internal_owner_token_sha256": token_sha256,
    }
    atomic_json(state_path, state)
    (logs / "runner.log").touch(exist_ok=True)
    with (logs / "runner.log").open("a", encoding="utf-8", buffering=1) as runner_log:
        original_stdout, original_stderr = sys.stdout, sys.stderr
        sys.stdout = runner_log
        sys.stderr = runner_log
        try:
            print(f"FORMAL_SESSION={session_id}", flush=True)
            prepare_markers: dict[str, dict[str, Any]] = {}
            for baseline in EXECUTION_BASELINES:
                state.update({"stage": f"prepare:{baseline}", "updated_unix_ns": time.time_ns()})
                atomic_json(state_path, state)
                marker = _run_phase(baseline, "prepare", session_id, root,
                                    owner_token=owner_token, lock_fd=inherited_lock_fd)
                prepare_markers[baseline] = marker
                state["completed_stages"].append(f"prepare:{baseline}")
                atomic_json(state_path, state)

            state.update({"stage": "global-shared-capacity-admission", "updated_unix_ns": time.time_ns()})
            atomic_json(state_path, state)
            admission = _validate_global_admission(prepare_markers, session_id)
            admission_path = root / "shared_capacity_admission.json"
            atomic_json(admission_path, admission)
            admission_sha256 = sha256_file(admission_path)
            state["shared_capacity_admission_sha256"] = admission_sha256
            atomic_json(state_path, state)
            print("GLOBAL_SHARED_CAPACITY_ADMISSION=READY", flush=True)

            cell_rows: list[dict[str, Any]] = []
            cell_receipts: list[dict[str, Any]] = []
            cell_index = 0
            for baseline in EXECUTION_BASELINES:
                for dataset in DATASETS:
                    dataset_log = logs / "datasets" / dataset / f"{baseline}.log"
                    # One independent log per cell, created only when that
                    # exact baseline/dataset proof begins.
                    dataset_log.parent.mkdir(parents=True, exist_ok=True)
                    state.update({"stage": f"cell:{baseline}:{dataset}", "updated_unix_ns": time.time_ns()})
                    atomic_json(state_path, state)
                    _recheck_remaining_capacity(admission, cell_index)
                    _recheck_checkpoint_source(admission, baseline, dataset)
                    marker = _run_phase(baseline, "cell", session_id, root, dataset, dataset_log,
                                        owner_token=owner_token, lock_fd=inherited_lock_fd)
                    _validate_cell_attestation(baseline, dataset, marker)
                    receipt = _cell_marker_receipt(baseline, dataset, marker)
                    receipt_path = root / "receipts" / baseline / f"{dataset}.json"
                    atomic_json(receipt_path, receipt)
                    cell_receipts.append({"baseline": baseline, "dataset": dataset,
                                          "path": str(receipt_path.relative_to(root)),
                                          "sha256": sha256_file(receipt_path)})
                    cell_rows.append(receipt["metrics"])
                    state["cell_receipts"] = list(cell_receipts)
                    state["completed_stages"].append(f"cell:{baseline}:{dataset}")
                    atomic_json(state_path, state)
                    cell_index += 1

            final_markers: dict[str, dict[str, Any]] = {}
            with (logs / "final-validation.log").open("a", encoding="utf-8") as summary_log:
                summary_log.write(f"\n=== FINAL VALIDATION START session={session_id} ===\n")
                summary_log.flush()
            for baseline in EXECUTION_BASELINES:
                state.update({"stage": f"finalize:{baseline}", "updated_unix_ns": time.time_ns()})
                atomic_json(state_path, state)
                final_markers[baseline] = _run_phase(
                    baseline, "finalize", session_id, root,
                    owner_token=owner_token, lock_fd=inherited_lock_fd,
                )
                state["completed_stages"].append(f"finalize:{baseline}")
                atomic_json(state_path, state)

            _publish_final(session_id, root, cell_rows, cell_receipts, final_markers,
                           admission_sha256)
            with (logs / "final-validation.log").open("a", encoding="utf-8") as summary_log:
                summary_log.write("ROOT_AGGREGATE_VALIDATION=PASS cells=15\n")
                summary_log.flush()
            state.update({"status": "COMPLETE", "stage": "", "exit_code": 0,
                          "updated_unix_ns": time.time_ns(),
                          "final_validation": f"results/formal/{session_id}/final_validation.json"})
            atomic_json(state_path, state)
            print(f"FORMAL_RUN_COMPLETE session={session_id} cells=15", flush=True)
            return 0
        except BaseException as exc:
            status = "INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "FAILED"
            exit_code = 130 if status == "INTERRUPTED" else 2
            state.update({"status": status, "exit_code": exit_code,
                          "failure": f"{type(exc).__name__}: {exc}",
                          "updated_unix_ns": time.time_ns()})
            atomic_json(state_path, state)
            if str(state.get("stage", "")).startswith("finalize:"):
                with (logs / "final-validation.log").open("a", encoding="utf-8") as summary_log:
                    summary_log.write(f"ROOT_AGGREGATE_VALIDATION=FAILED error={type(exc).__name__}: {exc}\n")
                    summary_log.flush()
            print(f"FORMAL_RUN_{status}: {type(exc).__name__}: {exc}", flush=True)
            return exit_code
        finally:
            sys.stdout, sys.stderr = original_stdout, original_stderr
            try:
                os.close(inherited_lock_fd)
            except OSError:
                pass


def _run_diagnostic_phase(session_id: str, inherited_lock_fd: int, baseline: str,
                          phase: str, dataset: str | None) -> int:
    if baseline not in BASELINES or phase not in {"prepare", "cell"}:
        raise RunnerError("diagnostic worker accepts one baseline prepare or cell phase")
    if (phase == "cell") != (dataset is not None) or (dataset is not None and dataset not in DATASETS):
        raise RunnerError("diagnostic dataset does not match the requested phase")
    token = os.environ.get(OWNER_TOKEN_ENV, "")
    if not HEX64_RE.fullmatch(token):
        raise RunnerError("diagnostic worker has no per-session owner token")
    _validate_root_lock_fd(inherited_lock_fd)
    os.environ[SESSION_ID_ENV] = session_id
    os.environ[ROOT_LOCK_FD_ENV] = str(inherited_lock_fd)
    session_root = _session_root(session_id)
    state_path = session_root / "state.json"
    token_sha256 = hashlib.sha256(token.encode("ascii")).hexdigest()
    lock_record = {
        "pid": os.getpid(), "boot_id": _read_boot_id(),
        "start_ticks": _proc_start_ticks(os.getpid()),
    }
    os.ftruncate(inherited_lock_fd, 0)
    os.lseek(inherited_lock_fd, 0, os.SEEK_SET)
    os.write(inherited_lock_fd, _json_bytes({**lock_record, "session_id": session_id,
                                             "diagnostic": True}))
    os.fsync(inherited_lock_fd)
    if phase == "prepare":
        if state_path.exists() or state_path.is_symlink():
            raise RunnerError("diagnostic prepare session already has state")
        state: dict[str, Any] = {
            "schema": "comparison-formal-session-v1", "session_id": session_id,
            "status": "RUNNING", "stage": f"prepare:{baseline}", "exit_code": None,
            "failure": "", "started_unix_ns": time.time_ns(), "updated_unix_ns": time.time_ns(),
            "completed_stages": [], "cell_receipts": [], "lock": lock_record,
            "internal_owner_token_sha256": token_sha256,
            "diagnostic": {"schema": "comparison-formal-diagnostic-v1", "baseline": baseline,
                           "phase_receipts": {}},
        }
    else:
        state = _read_json(state_path, "existing diagnostic session")
        diagnostic = state.get("diagnostic")
        if (state.get("schema") != "comparison-formal-session-v1"
                or state.get("session_id") != session_id
                or state.get("status") != "DIAGNOSTIC_COMPLETE"
                or not isinstance(diagnostic, dict)
                or diagnostic.get("schema") != "comparison-formal-diagnostic-v1"
                or diagnostic.get("baseline") != baseline):
            raise RunnerError("diagnostic cell has no completed prepare for this baseline/session")
        phase_receipt = diagnostic.get("phase_receipts", {}).get(f"prepare:{baseline}")
        if not isinstance(phase_receipt, dict) or phase_receipt.get("status") != "PASS":
            raise RunnerError("diagnostic cell prepare receipt is absent or unsuccessful")
        prepare_path = session_root / str(phase_receipt.get("path", ""))
        if (prepare_path.is_symlink() or not prepare_path.is_file()
                or sha256_file(prepare_path) != phase_receipt.get("sha256")):
            raise RunnerError("diagnostic cell prepare receipt digest mismatch")
        prepare_marker = _read_json(prepare_path, "diagnostic prepare marker")
        capacity = _validate_diagnostic_cell_capacity(baseline, dataset or "", prepare_marker, session_id)
        diagnostic["cell_capacity_admission"] = capacity
        state.update({"status": "RUNNING", "stage": f"cell:{baseline}:{dataset}",
                      "exit_code": None, "failure": "", "updated_unix_ns": time.time_ns(),
                      "lock": lock_record, "internal_owner_token_sha256": token_sha256})
    atomic_json(state_path, state)
    logs = session_root / "logs" / "diagnostic"
    logs.mkdir(parents=True, exist_ok=True)
    label = f"{baseline}-{phase}" + (f"-{dataset}" if dataset else "")
    log_path = logs / f"{label}.log"
    try:
        marker = _run_phase(
            baseline, phase, session_id, session_root, dataset, log_path,
            owner_token=token, lock_fd=inherited_lock_fd, diagnostic=True,
        )
        if phase == "prepare":
            marker_path = session_root / "diagnostic" / f"prepare-{baseline}.json"
        else:
            _validate_cell_attestation(baseline, dataset or "", marker)
            marker_path = session_root / "diagnostic" / f"cell-{baseline}-{dataset}.json"
        atomic_json(marker_path, marker)
        diagnostic = state.setdefault("diagnostic", {})
        phase_key = f"{phase}:{baseline}" + (f":{dataset}" if dataset else "")
        diagnostic.setdefault("phase_receipts", {})[phase_key] = {
            "status": "PASS", "path": marker_path.relative_to(session_root).as_posix(),
            "sha256": sha256_file(marker_path), "log_path": str(log_path),
            "log_sha256": sha256_file(log_path),
        }
        state.update({"status": "DIAGNOSTIC_COMPLETE", "stage": "", "exit_code": 0,
                      "failure": "", "updated_unix_ns": time.time_ns(), "diagnostic": diagnostic})
        atomic_json(state_path, state)
        print(f"FORMAL_DIAGNOSTIC_COMPLETE session={session_id} phase={phase_key} log={log_path}", flush=True)
        return 0
    except BaseException as exc:
        state.update({"status": "DIAGNOSTIC_FAILED", "stage": f"{phase}:{baseline}" + (f":{dataset}" if dataset else ""),
                      "exit_code": 130 if isinstance(exc, KeyboardInterrupt) else 2,
                      "failure": f"{type(exc).__name__}: {exc}", "updated_unix_ns": time.time_ns()})
        state.setdefault("diagnostic", {})["failure_log_path"] = str(log_path)
        atomic_json(state_path, state)
        print(f"FORMAL_DIAGNOSTIC_FAILED: {type(exc).__name__}: {exc}; log={log_path}", file=sys.stderr, flush=True)
        return 130 if isinstance(exc, KeyboardInterrupt) else 2
    finally:
        try:
            os.close(inherited_lock_fd)
        except OSError:
            pass


def _start_diagnostic_phase(baseline: str, phase: str, dataset: str | None = None,
                            session_id: str | None = None) -> tuple[int, str]:
    """Run one hidden baseline phase through the same lock, worker and stage ABI."""
    if baseline not in BASELINES or phase not in {"prepare", "cell"}:
        raise RunnerError("diagnostic invocation accepts one baseline prepare or cell phase")
    if (phase == "cell") != (dataset is not None) or (dataset is not None and dataset not in DATASETS):
        raise RunnerError("diagnostic dataset does not match the requested phase")
    if phase == "prepare":
        if session_id is not None:
            raise RunnerError("diagnostic prepare creates its own fresh session id")
        session_id = secrets.token_hex(16)
    elif session_id is None or not SESSION_ID_RE.fullmatch(session_id):
        raise RunnerError("diagnostic cell requires the prepare session id")
    descriptor = _acquire_lock()
    child: subprocess.Popen[bytes] | None = None
    try:
        _recover_interrupted_sessions()
        session_root = _session_root(session_id)
        if phase == "prepare":
            session_root.mkdir(parents=True, exist_ok=False)
        else:
            if session_root.is_symlink() or not session_root.is_dir():
                raise RunnerError("diagnostic cell session directory is missing or aliased")
        log_path = session_root / "logs" / "diagnostic" / "launcher.log"
        command = [sys.executable, str(Path(__file__).resolve()), "--_worker", session_id,
                   "--_lock-fd", str(descriptor), "--_diagnostic-phase", baseline, phase]
        if dataset is not None:
            command.append(dataset)
        environment = _sanitized_environment()
        environment[OWNER_TOKEN_ENV] = secrets.token_hex(32)
        child = spawn_detached_command(command, log_path, descriptor, environment)
        print(f"FORMAL_DIAGNOSTIC_STARTED session={session_id} phase={phase}:{baseline}"
              + (f":{dataset}" if dataset else "") + f" log={log_path}", flush=True)
        os.close(descriptor)
        descriptor = -1
        return_code = child.wait()
        print(f"FORMAL_DIAGNOSTIC_SESSION={session_id} phase={phase}:{baseline}"
              + (f":{dataset}" if dataset else "") + f" exit_code={return_code} log={log_path}", flush=True)
        return return_code, session_id
    except BaseException:
        if child is not None and child.poll() is None:
            _terminate_owned_process_group(child.pid, grace_seconds=5.0)
            child.wait()
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _publish_final(session_id: str, session_root: Path, rows: Sequence[Mapping[str, Any]],
                   receipts: Sequence[Mapping[str, Any]],
                   final_markers: Mapping[str, Mapping[str, Any]],
                   shared_admission_sha256: str) -> None:
    root_state = _read_json(session_root / "state.json", "root session state")
    if root_state.get("diagnostic") is not None:
        raise RunnerError("diagnostic-only sessions cannot publish official final results")
    expected = {(baseline, dataset) for baseline in BASELINES for dataset in DATASETS}
    observed = {(row.get("baseline"), row.get("dataset")) for row in rows}
    if observed != expected or len(rows) != 15:
        raise RunnerError("final aggregate does not contain exactly one row for each 3x5 cell")
    if len(receipts) != 15:
        raise RunnerError("final aggregate does not have 15 session-bound cell receipts")
    receipt_pairs: set[tuple[str, str]] = set()
    for descriptor in receipts:
        if not isinstance(descriptor, Mapping):
            raise RunnerError("cell receipt index contains a malformed entry")
        baseline, dataset = descriptor.get("baseline"), descriptor.get("dataset")
        pair = (baseline, dataset)
        if pair not in expected or pair in receipt_pairs:
            raise RunnerError(f"cell receipt index has an invalid or duplicate cell: {pair}")
        receipt_pairs.add(pair)
        relative = descriptor.get("path")
        digest = descriptor.get("sha256")
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise RunnerError(f"cell receipt index has no path/digest for {pair}")
        receipt_path = (session_root / relative).resolve(strict=True)
        try:
            receipt_path.relative_to(session_root.resolve())
        except ValueError as exc:
            raise RunnerError(f"cell receipt escapes root session: {pair}") from exc
        if receipt_path.is_symlink() or not receipt_path.is_file() or sha256_file(receipt_path) != digest:
            raise RunnerError(f"cell receipt failed final digest check: {pair}")
        receipt = _read_json(receipt_path, f"cell receipt {pair}")
        if (receipt.get("schema") != "comparison-cell-receipt-v1" or
                receipt.get("baseline") != baseline or receipt.get("dataset") != dataset or
                receipt.get("status") != "VERIFY_OK"):
            raise RunnerError(f"cell receipt is not a verified current-session cell: {pair}")
        row = next((item for item in rows if item.get("baseline") == baseline and item.get("dataset") == dataset), None)
        if row is None or receipt.get("metrics") != dict(row):
            raise RunnerError(f"cell receipt metrics differ from the aggregate row: {pair}")
    if receipt_pairs != expected:
        raise RunnerError("final receipt index does not cover the exact 3x5 matrix")
    admission_path = session_root / "shared_capacity_admission.json"
    admission = _read_json(admission_path, "shared admission")
    if (admission.get("schema") != "comparison-shared-capacity-admission-v1" or
            admission.get("status") != "READY" or admission.get("session_id") != session_id or
            sha256_file(admission_path) != shared_admission_sha256):
        raise RunnerError("final aggregate is not bound to its unchanged shared admission receipt")
    if (not isinstance(admission.get("shared_graph_identity"), dict) or
            not isinstance(admission.get("shared_checkpoint_identity"), dict)):
        raise RunnerError("final aggregate shared admission omits graph or checkpoint identity")
    final_dir = COMPARISON_ROOT / "results" / "formal" / session_id
    if final_dir.exists():
        raise RunnerError(f"final output directory already exists: {final_dir}")
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{session_id}.", dir=final_dir.parent))
    try:
        ordered_rows = sorted(rows, key=lambda row: (BASELINES.index(str(row["baseline"])), DATASETS.index(str(row["dataset"]))))
        ordered_receipts = sorted(receipts, key=lambda row: (BASELINES.index(str(row["baseline"])), DATASETS.index(str(row["dataset"]))))
        matrix = {
            "schema": "comparison-formal-result-matrix-v1",
            "session_id": session_id,
            "datasets": list(DATASETS),
            "baselines": list(BASELINES),
            "cell_count": 15,
            "metrics": ["commitment_time_ms", "prove_time_ms", "verify_time_ms", "proof_size_bytes"],
            "cells": [dict(row) for row in ordered_rows],
        }
        validation = {
            "schema": "comparison-formal-final-validation-v1",
            "status": "VALIDATED",
            "session_id": session_id,
            "cell_count": 15,
            "independent_verification": "PASS",
            "baseline_final_validations": {
                baseline: {
                    "status": "VALIDATED",
                    "validation_artifact_sha256": final_markers[baseline].get("_validation_artifact_sha256"),
                }
                for baseline in BASELINES
            },
            "cell_receipts": ordered_receipts,
            "shared_capacity_admission_sha256": shared_admission_sha256,
            "matrix_path": f"results/formal/{session_id}/result_matrix.json",
            "matrix_sha256": hashlib.sha256(_json_bytes(matrix)).hexdigest(),
            "validated_at_unix_ns": time.time_ns(),
        }
        atomic_json(staging / "result_matrix.json", matrix)
        atomic_json(staging / "final_validation.json", validation)
        os.rename(staging, final_dir)
        directory_fd = os.open(final_dir.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        if staging.exists():
            for item in staging.iterdir():
                item.unlink()
            staging.rmdir()
        raise


def _owned_processes_for_token(token_sha256: str) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        snapshot = _proc_snapshot(pid)
        if snapshot is None or snapshot["state"] == "Z":
            continue
        if _process_has_owner_token(pid, token_sha256):
            matches.append({"pid": pid, **snapshot})
    return matches


def _session_processes_without_token(session_id: str) -> list[int]:
    marker = session_id.encode("ascii")
    matches: list[int] = []
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        except OSError:
            continue
        if marker not in raw:
            continue
        snapshot = _proc_snapshot(pid)
        if (snapshot is not None and snapshot["state"] != "Z"
                and not _process_has_session_id(pid, session_id)):
            matches.append(pid)
    return matches


def _signal_exact_owned_processes(token_sha256: str, *, grace_seconds: float = 3.0) -> None:
    original = _owned_processes_for_token(token_sha256)
    for record in original:
        pid = int(record["pid"])
        current = _proc_snapshot(pid)
        if (current is None or current["start_ticks"] != record["start_ticks"]
                or not _process_has_owner_token(pid, token_sha256)):
            continue
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        if not _owned_processes_for_token(token_sha256):
            return
        time.sleep(0.05)
    for record in _owned_processes_for_token(token_sha256):
        pid = int(record["pid"])
        current = _proc_snapshot(pid)
        if (current is None or current["start_ticks"] != record["start_ticks"]
                or not _process_has_owner_token(pid, token_sha256)):
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and _owned_processes_for_token(token_sha256):
        time.sleep(0.05)
    survivors = _owned_processes_for_token(token_sha256)
    if survivors:
        raise RunnerError(f"stale formal session still has owned processes after cleanup: {[p['pid'] for p in survivors]}")


def _remove_stale_final_staging(session_id: str) -> None:
    parent = COMPARISON_ROOT / "results" / "formal"
    if parent.is_symlink() or not parent.is_dir():
        return
    prefix = f".{session_id}."
    for candidate in parent.iterdir():
        if not candidate.name.startswith(prefix):
            continue
        if candidate.is_symlink() or not candidate.is_dir():
            raise RunnerError(f"stale finalization staging path is aliased: {candidate}")
        _directory_regular_bytes(candidate)
        shutil.rmtree(candidate)


def _recover_interrupted_sessions() -> None:
    """Fail closed on live owners; mark only proven dead-session work failed."""
    formal_root = COMPARISON_ROOT / "runs" / "formal"
    if formal_root.is_symlink() or not formal_root.is_dir():
        return
    current_boot_id = _read_boot_id()
    if current_boot_id == "unavailable":
        raise RunnerError("cannot recover interrupted sessions without a trustworthy kernel boot id")
    for session_root in sorted(formal_root.iterdir()):
        if session_root.is_symlink() or not session_root.is_dir() or not SESSION_ID_RE.fullmatch(session_root.name):
            continue
        state_path = session_root / "state.json"
        if state_path.is_symlink() or not state_path.is_file():
            continue
        state = _read_json(state_path, "previous formal session state")
        if state.get("schema") != "comparison-formal-session-v1" or state.get("session_id") != session_root.name:
            raise RunnerError(f"cannot recover malformed prior formal session: {session_root.name}")
        if state.get("status") != "RUNNING":
            continue

        owner = state.get("lock")
        if not isinstance(owner, dict):
            raise RunnerError(f"prior running session has no verifiable root owner: {session_root.name}")
        owner_pid = owner.get("pid")
        owner_boot = owner.get("boot_id")
        owner_ticks = owner.get("start_ticks")
        if (isinstance(owner_pid, bool) or not isinstance(owner_pid, int) or owner_pid <= 0
                or not isinstance(owner_boot, str) or not re.fullmatch(r"[0-9a-f-]{36}", owner_boot)
                or not isinstance(owner_ticks, str) or not owner_ticks.isdigit()):
            raise RunnerError(f"prior running session has malformed root owner identity: {session_root.name}")
        if owner_boot != current_boot_id:
            # PIDs from an earlier boot cannot still be running; no PID signal is safe or necessary.
            owner_alive = False
        else:
            owner_alive = _proc_start_ticks(owner_pid) == owner_ticks
        if owner_alive:
            raise RunnerError(f"prior session owner is still alive despite an available global lock: {session_root.name}")

        # The random non-secret session tag survives the phase adapter's
        # deliberate secret-token removal. It identifies only descendants of
        # this session; each PID is rechecked against /proc start ticks and
        # the exact environment field before it is signaled.
        _signal_session_processes(session_root.name)
        possible = _session_processes_without_token(session_root.name)
        if possible:
            raise RunnerError(
                f"prior session has command-line matches without its owner tag; refusing to signal them: {session_root.name} {possible}"
            )

        _remove_stale_final_staging(session_root.name)
        state["status"] = "FAILED"
        state["exit_code"] = 2
        state["failure"] = "recovered interrupted root session after exact owner/process cleanup; no formal success accepted"
        state["updated_unix_ns"] = time.time_ns()
        state.pop("owned_stage", None)
        atomic_json(state_path, state)


def _acquire_lock() -> int:
    lock_path = COMPARISON_ROOT / "runs" / ".formal.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_CLOEXEC, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(descriptor)
        raise RunnerError("a formal comparison session already holds the process lock") from exc
    return descriptor


def _start_detached() -> int:
    descriptor = _acquire_lock()
    session_id = secrets.token_hex(16)
    owner_token = secrets.token_hex(32)
    session_root = _session_root(session_id)
    session_created = False
    child: subprocess.Popen[bytes] | None = None
    try:
        _recover_interrupted_sessions()
        session_root.mkdir(parents=True, exist_ok=False)
        session_created = True
        child_log = session_root / "logs" / "launcher.log"
        child_log.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable, str(Path(__file__).resolve()), "--_worker",
            session_id, "--_lock-fd", str(descriptor),
        ]
        child_environment = _sanitized_environment()
        child_environment[OWNER_TOKEN_ENV] = owner_token
        child = spawn_detached_command(command, child_log, descriptor, child_environment)
        os.ftruncate(descriptor, 0)
        os.lseek(descriptor, 0, os.SEEK_SET)
        os.write(descriptor, _json_bytes({
            "pid": child.pid, "boot_id": _read_boot_id(),
            "start_ticks": _proc_start_ticks(child.pid), "session_id": session_id,
        }))
        os.fsync(descriptor)
        print(f"STARTED session={session_id} pid={child.pid} status={session_root / 'state.json'} log={child_log}", flush=True)
        return 0
    except BaseException as exc:
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=5)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                except ProcessLookupError:
                    pass
        if session_created:
            failure = f"{type(exc).__name__}: {exc}"
            try:
                (session_root / "logs").mkdir(parents=True, exist_ok=True)
                with (session_root / "logs" / "launcher.log").open("a", encoding="utf-8") as log:
                    log.write(f"FORMAL_RUN_LAUNCH_FAILED: {failure}\n")
                    log.flush()
                    os.fsync(log.fileno())
                atomic_json(session_root / "state.json", {
                    "schema": "comparison-formal-session-v1",
                    "session_id": session_id,
                    "status": "FAILED",
                    "stage": "launch",
                    "exit_code": 2,
                    "failure": failure,
                    "started_unix_ns": time.time_ns(),
                    "updated_unix_ns": time.time_ns(),
                    "completed_stages": [],
                    "cell_receipts": [],
                })
            except OSError:
                pass
        raise
    finally:
        os.close(descriptor)


def _signal_handler(signum: int, _frame: Any) -> None:
    global ACTIVE_CHILD
    child = ACTIVE_CHILD
    if child is not None:
        _terminate_owned_process_group(child.pid, grace_seconds=5.0)
        if child.poll() is None:
            try:
                child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()
    raise KeyboardInterrupt(f"received signal {signum} during {ACTIVE_STAGE or 'startup'}")


def _install_signals() -> None:
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGHUP, _signal_handler)


def _validate_official_shell_launch(nonce: str) -> None:
    supplied = os.environ.get("COMPARISON_FORMAL_LAUNCH_NONCE", "")
    if not re.fullmatch(r"[0-9a-f]{64}", nonce) or not secrets.compare_digest(nonce, supplied):
        raise RunnerError("formal runner launch requires the private invocation nonce from run_formal.sh")
    try:
        parent = os.getppid()
        parent_exe = Path(os.readlink(f"/proc/{parent}/exe")).name
        parent_args = [part for part in Path(f"/proc/{parent}/cmdline").read_bytes().split(b"\0") if part]
    except OSError as exc:
        raise RunnerError(f"cannot validate the official shell launcher: {exc}") from exc
    expected_script = (COMPARISON_ROOT / "run_formal.sh").resolve()
    if parent_exe != "bash" or expected_script.as_posix().encode("utf-8") not in parent_args:
        raise RunnerError("formal runner can only be launched by ./comparison/run_formal.sh")


def main(argv: Sequence[str]) -> int:
    if len(argv) == 2 and argv[0] == "--_launch":
        try:
            _validate_official_shell_launch(argv[1])
            return _start_detached()
        except RunnerError as exc:
            print(f"FORMAL_RUN_BLOCKED: {exc}", file=sys.stderr, flush=True)
            return 2
        except OSError as exc:
            print(f"FORMAL_RUN_LAUNCH_FAILED: {exc}", file=sys.stderr, flush=True)
            return 2
    if len(argv) >= 5 and argv[0] == "--_stage-supervisor" and argv[3] == "--":
        try:
            parent_pid = int(argv[1])
            return _owned_stage_wrapper(parent_pid, argv[2], argv[4:])
        except (ValueError, OSError, RunnerError) as exc:
            print(f"FORMAL_RUN_OWNED_STAGE_START_FAILED: {exc}", file=sys.stderr, flush=True)
            return 2
    if (len(argv) in (7, 8) and argv[0] == "--_worker" and argv[2] == "--_lock-fd"
            and argv[4] == "--_diagnostic-phase"):
        session_id = argv[1]
        try:
            descriptor = int(argv[3])
            _validate_root_lock_fd(descriptor)
            if not HEX64_RE.fullmatch(os.environ.get(OWNER_TOKEN_ENV, "")):
                raise RunnerError("diagnostic worker has no inherited internal owner token")
            if not SESSION_ID_RE.fullmatch(session_id) or _session_root(session_id).is_symlink():
                raise RunnerError("diagnostic worker session path is malformed or aliased")
            _install_signals()
            return _run_diagnostic_phase(
                session_id, descriptor, argv[5], argv[6], argv[7] if len(argv) == 8 else None,
            )
        except (ValueError, OSError, RunnerError) as exc:
            print(f"FORMAL_DIAGNOSTIC_WORKER_START_FAILED: {exc}", file=sys.stderr, flush=True)
            return 2
    if len(argv) == 4 and argv[0] == "--_worker" and argv[2] == "--_lock-fd":
        session_id = argv[1]
        try:
            descriptor = int(argv[3])
            os.fstat(descriptor)
            _validate_root_lock_fd(descriptor)
            if not re.fullmatch(r"[0-9a-f]{64}", os.environ.get(OWNER_TOKEN_ENV, "")):
                raise RunnerError("root worker has no inherited internal owner token")
            if not SESSION_ID_RE.fullmatch(session_id):
                raise RunnerError("worker session id is malformed")
            if _session_root(session_id).is_symlink():
                raise RunnerError("worker session directory is a symlink")
            _install_signals()
            return _run_formal(session_id, descriptor)
        except (ValueError, OSError, RunnerError) as exc:
            print(f"FORMAL_RUN_WORKER_START_FAILED: {exc}", file=sys.stderr, flush=True)
            return 2
    print("formal_runner.py is internal; start the comparison with ./comparison/run_formal.sh", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
