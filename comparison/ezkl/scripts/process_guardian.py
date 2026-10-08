#!/usr/bin/env python3
"""EZKL-local process-group guardian for one baseline-owned formal session.

The guardian inherits the exact flock file description owned by the formal
runner.  It supervises stage process groups and keeps that descriptor open
until the owner exits and every registered group has stopped.
"""

from __future__ import annotations

import json
import os
import selectors
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

GUARDIAN_SOCKET_ENV = "EZKL_FORMAL_PROCESS_GUARDIAN_SOCKET"
LOCK_FD_ENV = "EZKL_FORMAL_OWNER_LOCK_FD"
PROCESS_GROUP_ENV = "EZKL_FORMAL_PROCESS_GROUP"
_PDEATHSIG = 1
_ACTIVE_GUARDIAN: "ProcessGuardian | None" = None


def _proc_stat(pid: int) -> tuple[int, int, str] | None:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    close = raw.rfind(")")
    if close < 0:
        return None
    fields = raw[close + 2 :].split()
    if len(fields) <= 19:
        return None
    try:
        return int(fields[2]), int(fields[19]), fields[0]
    except ValueError:
        return None


def process_start_ticks(pid: int) -> int:
    stat = _proc_stat(pid)
    if stat is None:
        raise RuntimeError(f"cannot read process identity for pid {pid}")
    return stat[1]


def _group_members(pgid: int) -> list[tuple[int, str]]:
    members: list[tuple[int, str]] = []
    try:
        entries = os.scandir("/proc")
    except OSError as exc:
        raise RuntimeError(f"cannot inspect /proc for process-group cleanup: {exc}") from exc
    with entries:
        for entry in entries:
            if not entry.name.isdigit():
                continue
            stat = _proc_stat(int(entry.name))
            if stat is not None and stat[0] == pgid and stat[2] not in {"Z", "X"}:
                members.append((int(entry.name), stat[2]))
    return members


def _group_leader_matches(pgid: int, start_ticks: int) -> bool:
    stat = _proc_stat(pgid)
    return stat is not None and stat[0] == pgid and stat[1] == start_ticks and stat[2] not in {"Z", "X"}


def _signal_registered_group(pgid: int, leader_start_ticks: int, signum: int) -> None:
    members = _group_members(pgid)
    if not members:
        return
    # A process group cannot be reused while any member remains.  If its
    # original leader still exists, also require its start time to match.
    leader = _proc_stat(pgid)
    if leader is not None and leader[0] == pgid and leader[1] != leader_start_ticks:
        raise RuntimeError(f"refusing to signal reused process-group leader {pgid}")
    try:
        os.killpg(pgid, signum)
    except ProcessLookupError:
        return


def _stop_registered_group(pgid: int, leader_start_ticks: int) -> bool:
    """Stop the group and wait until it has no executing members."""
    members = _group_members(pgid)
    if not members:
        return False
    _signal_registered_group(pgid, leader_start_ticks, signal.SIGTERM)
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        if not _group_members(pgid):
            return True
        time.sleep(0.05)
    _signal_registered_group(pgid, leader_start_ticks, signal.SIGKILL)
    while _group_members(pgid):
        time.sleep(0.05)
    return True


def _owner_identity_matches(owner_pid: int, owner_start_ticks: int) -> bool:
    stat = _proc_stat(owner_pid)
    return stat is not None and stat[1] == owner_start_ticks and stat[2] not in {"Z", "X"}


def _cleanup_groups_failclosed(groups: dict[int, int]) -> None:
    while groups:
        for pgid, leader_start in list(groups.items()):
            try:
                _stop_registered_group(pgid, leader_start)
                if not _group_members(pgid):
                    groups.pop(pgid, None)
            except BaseException as exc:
                print(f"[ezkl-guardian] retaining formal lock while cleanup of process group {pgid} retries: {exc}", file=sys.stderr, flush=True)
        if groups:
            time.sleep(0.25)


def _local_group_for_process(process: subprocess.Popen[Any]) -> tuple[int, int] | None:
    pgid = getattr(process, "_ezkl_process_group", None)
    leader_start = getattr(process, "_ezkl_leader_start_ticks", None)
    if isinstance(pgid, int) and isinstance(leader_start, int):
        return pgid, leader_start
    if os.name == "posix" and process.pid > 0:
        try:
            if os.getpgid(process.pid) == process.pid:
                return process.pid, process_start_ticks(process.pid)
        except (OSError, RuntimeError):
            return None
    return None


def terminate_guarded_process_locally(process: subprocess.Popen[Any]) -> None:
    group = _local_group_for_process(process)
    if group is not None:
        pgid, leader_start = group
        while True:
            try:
                _stop_registered_group(pgid, leader_start)
                if not _group_members(pgid):
                    break
            except BaseException as exc:
                print(f"[ezkl-formal] retaining owner lock while local process-group cleanup retries: {exc}", file=sys.stderr, flush=True)
                time.sleep(0.25)
    elif process.poll() is None:
        process.terminate()
    if process.poll() is None:
        process.wait()


def check_guardian_health() -> None:
    socket_path = os.environ.get(GUARDIAN_SOCKET_ENV)
    if not socket_path:
        return
    if _ACTIVE_GUARDIAN is not None and _ACTIVE_GUARDIAN.process.poll() is not None:
        raise RuntimeError(f"EZKL process guardian exited unexpectedly ({_ACTIVE_GUARDIAN.process.returncode})")
    _guard_rpc(socket_path, {"op": "ping"})


def _write_interrupted_session(session_file: Path, session_id: str, owner_pid: int) -> None:
    try:
        if session_file.is_symlink() or not session_file.is_file():
            return
        document = json.loads(session_file.read_text(encoding="utf-8"))
        if (
            not isinstance(document, dict)
            or document.get("baseline") != "ezkl"
            or document.get("session_id") != session_id
            or document.get("status") not in {"PREFLIGHT_STARTED", "RUNNING"}
        ):
            return
        document["status"] = "INTERRUPTED"
        document["error"] = f"formal owner pid {owner_pid} exited before supervised process cleanup completed"
        document["interrupted_at_utc"] = datetime.now(timezone.utc).isoformat()
        temporary = session_file.with_name(f".{session_file.name}.{os.getpid()}.guardian.tmp")
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(temporary, flags, 0o600)
        try:
            payload = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
            with os.fdopen(fd, "wb", closefd=False) as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            os.close(fd)
        os.replace(temporary, session_file)
    except BaseException as exc:
        print(f"[ezkl-guardian] could not record interrupted session: {exc}", file=sys.stderr, flush=True)


def _guardian_main(
    *,
    owner_pid: int,
    owner_start_ticks: int,
    owner_pidfd: int,
    lock_fd: int,
    session_id: str,
    session_file: Path,
    socket_path: Path,
) -> int:
    if not hasattr(signal, "pidfd_send_signal") and not hasattr(os, "pidfd_open"):
        raise RuntimeError("Linux pidfd support is required for safe EZKL owner supervision")
    if not _owner_identity_matches(owner_pid, owner_start_ticks):
        owner_already_gone = True
    else:
        owner_already_gone = False

    if socket_path.exists() or socket_path.is_symlink():
        raise RuntimeError(f"guardian socket path already exists: {socket_path}")
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    listener.bind(str(socket_path))
    os.chmod(socket_path, 0o600)
    listener.listen(8)
    groups: dict[int, int] = {}
    selector = selectors.DefaultSelector()
    selector.register(listener, selectors.EVENT_READ, "listener")
    selector.register(owner_pidfd, selectors.EVENT_READ, "owner")
    owner_dead = owner_already_gone
    shutdown_requested = False

    def cleanup_groups() -> list[int]:
        cleaned: list[int] = []
        for pgid, leader_start in list(groups.items()):
            try:
                if _stop_registered_group(pgid, leader_start):
                    cleaned.append(pgid)
                if not _group_members(pgid):
                    groups.pop(pgid, None)
            except BaseException as exc:
                print(f"[ezkl-guardian] cleanup of process group {pgid} will be retried: {exc}", file=sys.stderr, flush=True)
        if groups:
            raise RuntimeError(f"process groups remain under supervision after cleanup attempt: {sorted(groups)}")
        return cleaned

    try:
        while not owner_dead and not shutdown_requested:
            if not _owner_identity_matches(owner_pid, owner_start_ticks):
                owner_dead = True
                break
            for key, _ in selector.select(timeout=0.1):
                if key.data == "owner":
                    owner_dead = True
                    break
                if key.data != "listener":
                    continue
                connection, _address = listener.accept()
                with connection:
                    try:
                        request = json.loads(connection.recv(65536).decode("utf-8"))
                        if not isinstance(request, dict) or not isinstance(request.get("op"), str):
                            raise ValueError("malformed guardian request")
                        operation = request["op"]
                        if operation == "ping":
                            response: dict[str, Any] = {"ok": True}
                        elif operation == "register":
                            if owner_dead or not _owner_identity_matches(owner_pid, owner_start_ticks):
                                raise RuntimeError("formal owner exited before process registration")
                            pgid = request.get("pgid")
                            leader_start = request.get("leader_start_ticks")
                            if (
                                isinstance(pgid, bool) or not isinstance(pgid, int) or pgid <= 0
                                or isinstance(leader_start, bool) or not isinstance(leader_start, int) or leader_start <= 0
                                or not _group_leader_matches(pgid, leader_start)
                            ):
                                raise RuntimeError("stage process identity changed before guardian registration")
                            groups[pgid] = leader_start
                            response = {"ok": True}
                        elif operation in {"finish", "terminate"}:
                            pgid = request.get("pgid")
                            leader_start = request.get("leader_start_ticks")
                            if not isinstance(pgid, int) or groups.get(pgid) != leader_start:
                                raise RuntimeError("stage process group is not registered to this EZKL session")
                            terminated = _stop_registered_group(pgid, leader_start) if operation == "terminate" else False
                            active = _group_members(pgid)
                            if operation == "finish" and active:
                                terminated = _stop_registered_group(pgid, leader_start)
                                active = _group_members(pgid)
                            if active:
                                raise RuntimeError(f"process group {pgid} remains active after cleanup: {active[:4]}")
                            groups.pop(pgid, None)
                            response = {"ok": True, "terminated_leftovers": terminated}
                        elif operation == "shutdown":
                            cleaned = cleanup_groups()
                            shutdown_requested = True
                            response = {"ok": True, "cleaned_groups": cleaned}
                        else:
                            raise ValueError(f"unknown guardian operation: {operation}")
                    except BaseException as exc:
                        response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                    connection.sendall(json.dumps(response, sort_keys=True).encode("utf-8"))
    finally:
        # A shutdown RPC can be the only ready selector event in the same
        # poll cycle where the owner dies.  Recheck the identity after leaving
        # the loop so that queued shutdown cannot leave a dead owner's session
        # marked RUNNING.  The writer itself only changes nonterminal states.
        if not owner_dead and not _owner_identity_matches(owner_pid, owner_start_ticks):
            owner_dead = True
        if owner_dead:
            _write_interrupted_session(session_file, session_id, owner_pid)
        _cleanup_groups_failclosed(groups)
        selector.close()
        listener.close()
        try:
            socket_path.unlink()
        except FileNotFoundError:
            pass
        os.close(owner_pidfd)
        os.close(lock_fd)
    return 0


class ProcessGuardian:
    def __init__(self, process: subprocess.Popen[bytes], socket_path: Path):
        self.process = process
        self.socket_path = socket_path
        self.closed = False
        self.groups: dict[int, int] = {}

    @classmethod
    def start(cls, *, root: Path, session_id: str, session_file: Path, lock_fd: int) -> "ProcessGuardian":
        global _ACTIVE_GUARDIAN
        if os.name != "posix" or not hasattr(os, "pidfd_open"):
            raise RuntimeError("EZKL formal SIGKILL supervision requires POSIX pidfd support")
        owner_pid = os.getpid()
        owner_start = process_start_ticks(owner_pid)
        os.environ.pop(PROCESS_GROUP_ENV, None)
        owner_pidfd = os.pidfd_open(owner_pid, 0)
        lock_directory = root / "runs"
        socket_path = lock_directory / f".ezkl-guardian-{session_id}.sock"
        try:
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--guardian",
                    str(owner_pid),
                    str(owner_start),
                    str(owner_pidfd),
                    str(lock_fd),
                    session_id,
                    str(session_file),
                    str(socket_path),
                ],
                cwd=root,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                pass_fds=(owner_pidfd, lock_fd),
            )
        except BaseException:
            os.close(owner_pidfd)
            raise
        os.close(owner_pidfd)
        guardian = cls(process, socket_path)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if process.poll() is not None:
                process.wait()
                raise RuntimeError(f"EZKL process guardian exited during startup ({process.returncode})")
            try:
                with guardian._connect(timeout=0.2) as connection:
                    connection.sendall(json.dumps({"op": "ping"}).encode())
                    response = json.loads(connection.recv(4096).decode())
                if response.get("ok"):
                    os.environ[GUARDIAN_SOCKET_ENV] = str(socket_path)
                    os.environ[LOCK_FD_ENV] = str(lock_fd)
                    _ACTIVE_GUARDIAN = guardian
                    return guardian
            except (OSError, ValueError):
                time.sleep(0.05)
        process.terminate()
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise RuntimeError("EZKL process guardian failed to become ready within 5 seconds")

    def _connect(self, *, timeout: float = 5.0) -> socket.socket:
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        connection.settimeout(timeout)
        try:
            connection.connect(str(self.socket_path))
        except BaseException:
            connection.close()
            raise
        return connection

    def request(self, request: dict[str, Any], *, timeout: float = 10.0) -> dict[str, Any]:
        if self.closed:
            raise RuntimeError("EZKL process guardian is already shut down")
        with self._connect(timeout=timeout) as connection:
            connection.sendall(json.dumps(request, sort_keys=True).encode("utf-8"))
            response = json.loads(connection.recv(65536).decode("utf-8"))
        if not isinstance(response, dict) or response.get("ok") is not True:
            raise RuntimeError(f"EZKL process guardian rejected request: {response}")
        return response

    def shutdown(self) -> None:
        global _ACTIVE_GUARDIAN
        if self.closed:
            return
        response: dict[str, Any] | None = None
        while response is None:
            try:
                response = self.request({"op": "shutdown"}, timeout=15.0)
            except BaseException as exc:
                if self.process.poll() is not None:
                    for pgid, leader_start in list(self.groups.items()):
                        terminate_guarded_process_locally(_GroupHandle(pgid, leader_start))
                        self.groups.pop(pgid, None)
                    raise RuntimeError("EZKL process guardian exited before confirming clean shutdown") from exc
                for pgid, leader_start in list(self.groups.items()):
                    terminate_guarded_process_locally(_GroupHandle(pgid, leader_start))
                    self.groups.pop(pgid, None)
                time.sleep(0.25)
        self.process.wait()
        self.closed = True
        os.environ.pop(GUARDIAN_SOCKET_ENV, None)
        os.environ.pop(LOCK_FD_ENV, None)
        os.environ.pop(PROCESS_GROUP_ENV, None)
        if _ACTIVE_GUARDIAN is self:
            _ACTIVE_GUARDIAN = None
        if response.get("cleaned_groups"):
            raise RuntimeError(f"EZKL process guardian found leftover groups at normal shutdown: {response['cleaned_groups']}")

    def remember_group(self, pgid: int, leader_start_ticks: int) -> None:
        self.groups[pgid] = leader_start_ticks

    def forget_group(self, pgid: int) -> None:
        self.groups.pop(pgid, None)


class _GroupHandle:
    def __init__(self, pgid: int, leader_start_ticks: int):
        self.pid = pgid
        self._ezkl_process_group = pgid
        self._ezkl_leader_start_ticks = leader_start_ticks
        self.returncode: int | None = 0

    def poll(self) -> int | None:
        return self.returncode

    def wait(self) -> int:
        return 0


def _guard_rpc(socket_path: str, request: dict[str, Any]) -> dict[str, Any]:
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    connection.settimeout(10.0)
    try:
        connection.connect(socket_path)
        connection.sendall(json.dumps(request, sort_keys=True).encode("utf-8"))
        response = json.loads(connection.recv(65536).decode("utf-8"))
    finally:
        connection.close()
    if not isinstance(response, dict) or response.get("ok") is not True:
        raise RuntimeError(f"EZKL process guardian request failed: {response}")
    return response


def spawn_guarded_process(command: list[str], **kwargs: Any) -> subprocess.Popen[Any]:
    """Start a stage in a fresh group, register before allowing command exec."""
    env = kwargs.get("env") or os.environ.copy()
    socket_path = env.get(GUARDIAN_SOCKET_ENV)
    inherited_group = env.get(PROCESS_GROUP_ENV)
    if inherited_group is not None:
        if not inherited_group.isdigit() or os.name != "posix" or os.getpgrp() != int(inherited_group):
            raise RuntimeError("EZKL nested stage has an invalid inherited process-group identity")
        process = subprocess.Popen(command, **{key: value for key, value in kwargs.items() if key != "start_new_session"}, start_new_session=False)
        process._ezkl_process_group = int(inherited_group)  # type: ignore[attr-defined]
        process._ezkl_leader_start_ticks = process_start_ticks(int(inherited_group))  # type: ignore[attr-defined]
        process._ezkl_shared_process_group = True  # type: ignore[attr-defined]
        return process
    if not socket_path:
        kwargs.setdefault("start_new_session", os.name == "posix")
        return subprocess.Popen(command, **kwargs)
    lock_text = env.get(LOCK_FD_ENV)
    if lock_text is None or not lock_text.isdigit():
        raise RuntimeError("formal stage has a guardian but no inherited flock descriptor")
    lock_fd = int(lock_text)
    os.fstat(lock_fd)
    gate_read, gate_write = os.pipe()
    parent_pid = os.getpid()
    parent_start = process_start_ticks(parent_pid)
    wrapper_command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--exec-stage",
        str(parent_pid),
        str(parent_start),
        str(gate_read),
        "--",
        *command,
    ]
    child_env = dict(env)
    # Only the runner, not arbitrary command line arguments, carries the gate.
    passed_fds = tuple(sorted({gate_read, lock_fd}))
    try:
        process = subprocess.Popen(
            wrapper_command,
            env=child_env,
            pass_fds=passed_fds,
            start_new_session=(os.name == "posix"),
            **{key: value for key, value in kwargs.items() if key not in {"env", "start_new_session", "pass_fds"}},
        )
    except BaseException:
        os.close(gate_read)
        os.close(gate_write)
        raise
    os.close(gate_read)
    try:
        leader_start = process_start_ticks(process.pid)
        _guard_rpc(socket_path, {
            "op": "register",
            "pgid": process.pid,
            "leader_start_ticks": leader_start,
        })
        if _ACTIVE_GUARDIAN is not None:
            _ACTIVE_GUARDIAN.remember_group(process.pid, leader_start)
        os.write(gate_write, b"G")
    except BaseException:
        terminate_guarded_process_locally(process)
        raise
    finally:
        os.close(gate_write)
    process._ezkl_process_group = process.pid  # type: ignore[attr-defined]
    process._ezkl_leader_start_ticks = leader_start  # type: ignore[attr-defined]
    return process


def finish_guarded_process(process: subprocess.Popen[Any], *, terminate: bool = False) -> bool:
    pgid = getattr(process, "_ezkl_process_group", None)
    leader_start = getattr(process, "_ezkl_leader_start_ticks", None)
    if getattr(process, "_ezkl_shared_process_group", False):
        if terminate:
            terminate_guarded_process_locally(process)
        return False
    socket_path = os.environ.get(GUARDIAN_SOCKET_ENV)
    if not isinstance(pgid, int) or not isinstance(leader_start, int) or not socket_path:
        return False
    try:
        response = _guard_rpc(socket_path, {
            "op": "terminate" if terminate else "finish",
            "pgid": pgid,
            "leader_start_ticks": leader_start,
        })
    except BaseException:
        terminate_guarded_process_locally(process)
        raise
    if _ACTIVE_GUARDIAN is not None:
        _ACTIVE_GUARDIAN.forget_group(pgid)
    return bool(response.get("terminated_leftovers"))


def run_guarded_capture(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    process = spawn_guarded_process(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    deadline = time.monotonic() + timeout
    try:
        while True:
            try:
                stdout, stderr = process.communicate(timeout=min(5.0, max(0.1, deadline - time.monotonic())))
                break
            except subprocess.TimeoutExpired:
                if time.monotonic() >= deadline:
                    raise
                check_guardian_health()
    except BaseException:
        try:
            finish_guarded_process(process, terminate=True)
        except BaseException:
            terminate_guarded_process_locally(process)
        try:
            process.communicate()
        except BaseException:
            pass
        raise
    terminated = finish_guarded_process(process)
    if terminated:
        raise RuntimeError("formal helper left descendants after its leader exited; the process group was cleaned")
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def _exec_stage(parent_pid: int, parent_start: int, gate_fd: int, command: list[str]) -> int:
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(_PDEATHSIG, signal.SIGTERM, 0, 0, 0) != 0:
        return 125
    if not _owner_identity_matches(parent_pid, parent_start):
        return 125
    try:
        token = os.read(gate_fd, 1)
    except OSError:
        return 125
    finally:
        os.close(gate_fd)
    if token != b"G" or not _owner_identity_matches(parent_pid, parent_start):
        return 125
    os.environ[PROCESS_GROUP_ENV] = str(os.getpgrp())
    os.execvpe(command[0], command, os.environ)
    return 127


def _main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[1] == "--exec-stage":
        try:
            parent_pid = int(argv[2])
            parent_start = int(argv[3])
            gate_fd = int(argv[4])
            if argv[5] != "--" or len(argv) < 7:
                return 125
            return _exec_stage(parent_pid, parent_start, gate_fd, argv[6:])
        except (IndexError, ValueError, OSError):
            return 125
    if len(argv) >= 2 and argv[1] == "--guardian":
        try:
            return _guardian_main(
                owner_pid=int(argv[2]),
                owner_start_ticks=int(argv[3]),
                owner_pidfd=int(argv[4]),
                lock_fd=int(argv[5]),
                session_id=argv[6],
                session_file=Path(argv[7]),
                socket_path=Path(argv[8]),
            )
        except BaseException as exc:
            print(f"[ezkl-guardian] fatal guardian error: {exc}", file=sys.stderr, flush=True)
            return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
