#!/usr/bin/env python3
"""Own one detached baseline-local zkDL five-dataset run."""
from __future__ import annotations

import argparse
import json
import os
import secrets
import signal
import stat
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
INTERRUPTED_SIGNAL: int | None = None
CHILD: subprocess.Popen[bytes] | None = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_atomic(path: Path, document: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
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


def _group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def _terminate_child_group(child: subprocess.Popen[bytes], signum: int) -> None:
    try:
        os.killpg(child.pid, signum)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if not _group_exists(child.pid):
            return
        time.sleep(0.05)
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _handle_signal(signum: int, _frame: object) -> None:
    global INTERRUPTED_SIGNAL
    INTERRUPTED_SIGNAL = signum
    child = CHILD
    if child is not None:
        _terminate_child_group(child, signal.SIGTERM)


def run_supervisor(command: list[str], cwd: Path, launch_dir: Path) -> int:
    global CHILD, INTERRUPTED_SIGNAL
    started_at = _now()
    child_returncode: int | None = None
    exit_code = 70
    status = "FAILED"
    failure = ""
    child_signal: int | None = None
    console_path = launch_dir / "console.log"
    console_fd: int | None = None
    try:
        if launch_dir.is_symlink() or not launch_dir.is_dir() or console_path.is_symlink():
            raise RuntimeError("launch directory or console path is missing or symlinked")
        console_fd = os.open(console_path, os.O_WRONLY | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0))
        if not stat.S_ISREG(os.fstat(console_fd).st_mode):
            raise RuntimeError("detached launch console is not a regular file")
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, _handle_signal)
        CHILD = subprocess.Popen(command, cwd=cwd, env={**os.environ, "PYTHONUNBUFFERED": "1"}, stdin=subprocess.DEVNULL,
                                 stdout=console_fd, stderr=subprocess.STDOUT, start_new_session=True)
        _write_json_atomic(launch_dir / "pid.json", {
            "schema": "zkdl_detached_launch_pid_v1", "supervisor_pid": os.getpid(),
            "supervisor_session_id": os.getsid(0), "runner_pid": CHILD.pid,
            "runner_process_group_id": CHILD.pid, "started_at_utc": started_at,
        })
        if INTERRUPTED_SIGNAL is not None:
            _terminate_child_group(CHILD, signal.SIGTERM)
        child_returncode = CHILD.wait()
        if INTERRUPTED_SIGNAL is not None:
            status = "INTERRUPTED"
            exit_code = 128 + INTERRUPTED_SIGNAL
            child_signal = INTERRUPTED_SIGNAL
        elif child_returncode < 0:
            status = "FAILED"
            child_signal = -child_returncode
            exit_code = 128 + child_signal
        else:
            exit_code = child_returncode
            status = "PASS" if exit_code == 0 else "FAILED"
    except BaseException as exc:
        failure = str(exc) or type(exc).__name__
        if CHILD is not None and CHILD.poll() is None:
            _terminate_child_group(CHILD, signal.SIGTERM)
            child_returncode = CHILD.wait()
        if INTERRUPTED_SIGNAL is not None:
            status = "INTERRUPTED"
            exit_code = 128 + INTERRUPTED_SIGNAL
            child_signal = INTERRUPTED_SIGNAL
        elif CHILD is not None and CHILD.returncode is not None and CHILD.returncode < 0:
            child_returncode = CHILD.returncode
            child_signal = -child_returncode
            exit_code = 128 + child_signal
        else:
            status = "FAILED"
            exit_code = 70
    finally:
        finished_at = _now()
        result = {"schema": "zkdl_detached_launch_exit_v1", "status": status, "exit_code": exit_code,
                  "child_returncode": child_returncode, "signal": child_signal, "failure": failure,
                  "supervisor_pid": os.getpid(), "runner_pid": CHILD.pid if CHILD is not None else None,
                  "started_at_utc": started_at, "finished_at_utc": finished_at}
        try:
            _write_json_atomic(launch_dir / "exit.json", result)
            print(json.dumps({"zkdl_detached_exit": status, "exit_code": exit_code,
                              "launch_dir": str(launch_dir)}, separators=(",", ":")), flush=True)
        except OSError as exc:
            print(f"zkdl-detached-supervisor=FAILED exit_artifact_error={exc}", file=sys.stderr, flush=True)
            exit_code = 70
        if console_fd is not None:
            os.close(console_fd)
        CHILD = None
    return exit_code


def _launch_dir(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        raise ValueError("launch directory must be an existing absolute non-symlink path")
    resolved = path.resolve(strict=True)
    if resolved.parent != RUNS.resolve(strict=True) or not resolved.name.startswith("launch_"):
        raise ValueError("launch directory must be directly under zkdl/runs")
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch-dir", required=True)
    args = parser.parse_args()
    try:
        launch_dir = _launch_dir(args.launch_dir)
    except (OSError, ValueError) as exc:
        print(f"zkdl-detached-supervisor=FAILED error={exc}", file=sys.stderr, flush=True)
        return 70
    return run_supervisor(["/bin/bash", str(ROOT / "scripts" / "run_five.sh")], ROOT, launch_dir)


if __name__ == "__main__":
    raise SystemExit(main())
