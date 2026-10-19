#!/usr/bin/env python3
"""Detach one zkGPT formal run and persist its actual child exit status."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

from common import DATASETS, ChainError, canonical_dataset_scope


LAUNCH_ID_RE = re.compile(r"^[0-9a-f]{32}$")
SIGNAL_NAMES = {
    signal.SIGINT: "SIGINT",
    signal.SIGTERM: "SIGTERM",
    signal.SIGHUP: "SIGHUP",
}
child: subprocess.Popen | None = None
received_signal: int | None = None


def atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def update_status(path: Path, state: Dict[str, Any], status: str,
                  **changes: Any) -> None:
    state.update(changes)
    state["status"] = status
    state["updated_unix_ns"] = time.time_ns()
    atomic_json(path, state)


def process_identity(pid: int) -> Dict[str, Any]:
    raw = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    close = raw.rfind(")")
    if close < 0:
        raise RuntimeError(f"process {pid} stat record is malformed")
    fields = raw[close + 2:].split()
    if len(fields) <= 19:
        raise RuntimeError(f"process {pid} stat record is truncated")
    argv = [os.fsdecode(value) for value in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0") if value]
    if not argv:
        raise RuntimeError(f"process {pid} command line is unavailable")
    return {
        "start_ticks": int(fields[19]),
        "process_group": int(fields[2]),
        "cwd": os.readlink(f"/proc/{pid}/cwd"),
        "executable": os.readlink(f"/proc/{pid}/exe"),
        "argv": argv,
    }


def forward_signal(signum: int, _frame: Any) -> None:
    global received_signal
    process = child
    if process is not None and process.poll() is not None:
        return
    received_signal = signum
    if process is not None:
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--launch-id", required=True)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS)
    args = parser.parse_args()
    if not LAUNCH_ID_RE.fullmatch(args.launch_id):
        raise SystemExit("invalid zkGPT detached launch ID")
    try:
        dataset_scope = canonical_dataset_scope(args.datasets)
    except ChainError as exc:
        raise SystemExit(str(exc)) from exc
    selector_explicit = args.datasets is not None
    if selector_explicit and list(args.datasets) != list(dataset_scope):
        raise SystemExit("detached supervisor dataset selector is not in canonical order")

    root = Path(args.root).resolve(strict=True)
    launch_root = root / "baseline-local" / "formal-detached"
    launch_dir = launch_root / args.launch_id
    if not launch_dir.is_dir() or launch_dir.is_symlink():
        raise SystemExit("zkGPT detached launch directory is not a private directory")
    runner = root / "scripts" / "run_formal.py"
    if not runner.is_file() or runner.is_symlink():
        raise SystemExit("zkGPT formal runner is missing or aliased")
    status_path = launch_dir / "status.json"
    supervisor = process_identity(os.getpid())
    state: Dict[str, Any] = {
        "schema": "zkgpt-gat-detached-launch-v2",
        "launch_id": args.launch_id,
        "status": "STARTING",
        "supervisor_pid": os.getpid(),
        "supervisor_start_ticks": supervisor["start_ticks"],
        "supervisor_process_group": supervisor["process_group"],
        "supervisor_cwd": supervisor["cwd"],
        "supervisor_executable": supervisor["executable"],
        "supervisor_argv": supervisor["argv"],
        "root_path": str(root),
        "runner_path": str(runner),
        "requested_datasets": list(dataset_scope),
        "datasets_selector_explicit": selector_explicit,
        "console_path": str(launch_dir / "console.log"),
        "status_path": str(status_path),
        "started_unix_ns": time.time_ns(),
        "updated_unix_ns": time.time_ns(),
        "finished_unix_ns": None,
        "child_pid": None,
        "child_start_ticks": None,
        "child_process_group": None,
        "child_cwd": None,
        "child_executable": None,
        "child_argv": None,
        "exit_code": None,
        "received_signal": None,
    }
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, forward_signal)
    atomic_json(status_path, state)

    global child
    child_command = [sys.executable, str(runner)]
    if selector_explicit:
        child_command.extend(["--datasets", *dataset_scope])
    child = subprocess.Popen(
        child_command, cwd=root, stdin=subprocess.DEVNULL,
        close_fds=True, start_new_session=True,
    )
    if received_signal is not None:
        try:
            os.killpg(child.pid, received_signal)
        except ProcessLookupError:
            pass
    try:
        child_info = process_identity(child.pid)
    except (OSError, RuntimeError, ValueError):
        exit_code = child.wait()
        terminal = "INTERRUPTED" if received_signal is not None else (
            ("COMPLETE" if dataset_scope == DATASETS else "SUBSET_COMPLETE")
            if exit_code == 0 else ("FAILED" if dataset_scope == DATASETS else "SUBSET_FAILED")
        )
        update_status(status_path, state, terminal, exit_code=exit_code,
                      received_signal=(SIGNAL_NAMES.get(received_signal)
                                       if received_signal is not None else None),
                      finished_unix_ns=time.time_ns())
        print(f"ZKGPT_FORMAL_CHILD_EXIT launch_id={args.launch_id} "
              f"exit_code={exit_code} status={terminal}", flush=True)
        return 0
    update_status(status_path, state, "RUNNING", child_pid=child.pid,
                  child_start_ticks=child_info["start_ticks"],
                  child_process_group=child_info["process_group"],
                  child_cwd=child_info["cwd"],
                  child_executable=child_info["executable"],
                  child_argv=child_info["argv"])
    print(f"ZKGPT_FORMAL_CHILD_STARTED launch_id={args.launch_id} child_pid={child.pid}",
          flush=True)
    exit_code = child.wait()
    if received_signal is not None:
        terminal = "INTERRUPTED"
    elif exit_code == 0:
        terminal = "COMPLETE" if dataset_scope == DATASETS else "SUBSET_COMPLETE"
    else:
        terminal = "FAILED" if dataset_scope == DATASETS else "SUBSET_FAILED"
    update_status(
        status_path, state, terminal, exit_code=exit_code,
        received_signal=(SIGNAL_NAMES.get(received_signal)
                         if received_signal is not None else None),
        finished_unix_ns=time.time_ns(),
    )
    print(f"ZKGPT_FORMAL_CHILD_EXIT launch_id={args.launch_id} "
          f"exit_code={exit_code} status={terminal}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
