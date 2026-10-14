#!/usr/bin/env python3
"""Safely stop one exact detached zkGPT formal launch."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict


LAUNCH_ID_RE = re.compile(r"^[0-9a-f]{32}$")
STATUS_SCHEMA = "zkgpt-gat-detached-launch-v2"
POINTER_SCHEMA = "zkgpt-gat-detached-launch-v1"
ACTIVE_STATES = ("STARTING", "RUNNING")
TERMINAL_STATES = ("COMPLETE", "FAILED", "INTERRUPTED")


class StopError(Exception):
    pass


def read_json(path: Path, description: str) -> Dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise StopError(f"{description} is missing or aliased: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StopError(f"{description} is unreadable: {exc}") from exc
    if not isinstance(value, dict):
        raise StopError(f"{description} is not a JSON object")
    return value


def process_identity(pid: int) -> Dict[str, Any]:
    if type(pid) is not int or pid <= 0:
        raise StopError("recorded supervisor PID is invalid")
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        close = raw.rfind(")")
        if close < 0:
            raise StopError("process stat record is malformed")
        fields = raw[close + 2:].split()
        if len(fields) <= 19:
            raise StopError("process stat record is truncated")
        argv = [os.fsdecode(value) for value in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0") if value]
        return {
            "start_ticks": int(fields[19]),
            "process_group": int(fields[2]),
            "cwd": os.readlink(f"/proc/{pid}/cwd"),
            "executable": os.readlink(f"/proc/{pid}/exe"),
            "argv": argv,
        }
    except (OSError, ValueError) as exc:
        raise StopError(f"process {pid} is no longer inspectable: {exc}") from exc


def launch_id_from_pointer(root: Path) -> str:
    pointer_path = root / "baseline-local" / "formal-detached" / "latest_launch.json"
    pointer = read_json(pointer_path, "latest launch pointer")
    if (set(pointer) != {"schema", "launch_id", "console_path", "status_path"} or
            pointer.get("schema") != POINTER_SCHEMA):
        raise StopError("latest launch pointer schema is unsupported")
    launch_id = pointer.get("launch_id")
    if not isinstance(launch_id, str) or not LAUNCH_ID_RE.fullmatch(launch_id):
        raise StopError("latest launch pointer has an invalid launch ID")
    expected_console = f"baseline-local/formal-detached/{launch_id}/console.log"
    expected_status = f"baseline-local/formal-detached/{launch_id}/status.json"
    if pointer.get("console_path") != expected_console or pointer.get("status_path") != expected_status:
        raise StopError("latest launch pointer paths do not match its launch ID")
    return launch_id


def validate_status(state: Dict[str, Any], root: Path, launch_id: str,
                    status_path: Path) -> str:
    launch_dir = root / "baseline-local" / "formal-detached" / launch_id
    supervisor_script = root / "scripts" / "formal_detached_supervisor.py"
    runner = root / "scripts" / "run_formal.py"
    if (state.get("schema") != STATUS_SCHEMA or state.get("launch_id") != launch_id or
            state.get("root_path") != str(root) or state.get("status_path") != str(status_path) or
            state.get("console_path") != str(launch_dir / "console.log") or
            state.get("runner_path") != str(runner)):
        raise StopError("detached launch status schema, ID, or root paths do not match")
    status = state.get("status")
    if status not in ACTIVE_STATES + TERMINAL_STATES:
        raise StopError("detached launch status is invalid")
    supervisor_pid = state.get("supervisor_pid")
    supervisor_ticks = state.get("supervisor_start_ticks")
    supervisor_argv = state.get("supervisor_argv")
    if (type(supervisor_pid) is not int or supervisor_pid <= 0 or
            type(supervisor_ticks) is not int or supervisor_ticks <= 0 or
            state.get("supervisor_process_group") != supervisor_pid or
            state.get("supervisor_cwd") != str(root) or
            not isinstance(state.get("supervisor_executable"), str) or
            not state["supervisor_executable"].startswith("/") or
            not isinstance(supervisor_argv, list) or len(supervisor_argv) < 2 or
            supervisor_argv[1:] != [str(supervisor_script), "--root", str(root), "--launch-id", launch_id]):
        raise StopError("detached launch supervisor identity metadata is invalid")
    child_pid = state.get("child_pid")
    if child_pid is not None:
        child_argv = state.get("child_argv")
        if (type(child_pid) is not int or child_pid <= 0 or
                type(state.get("child_start_ticks")) is not int or state["child_start_ticks"] <= 0 or
                state.get("child_process_group") != child_pid or
                state.get("child_cwd") != str(root) or
                not isinstance(state.get("child_executable"), str) or
                not state["child_executable"].startswith("/") or
                not isinstance(child_argv, list) or len(child_argv) != 2 or child_argv[1] != str(runner)):
            raise StopError("detached launch child identity metadata is invalid")
    elif status == "RUNNING":
        raise StopError("running detached launch has no child identity")
    return status


def validate_live_process(state: Dict[str, Any], role: str, pid: int) -> None:
    current = process_identity(pid)
    expected = {
        "start_ticks": state.get(f"{role}_start_ticks"),
        "process_group": state.get(f"{role}_process_group"),
        "cwd": state.get(f"{role}_cwd"),
        "executable": state.get(f"{role}_executable"),
        "argv": state.get(f"{role}_argv"),
    }
    if current != expected:
        raise StopError(f"live {role} PID does not match launch metadata")
    if current["process_group"] != pid:
        raise StopError(f"live {role} is not the expected process-group leader")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-id", help="exact 32-hex launch ID; defaults to latest_launch.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    launch_id = args.launch_id if args.launch_id is not None else launch_id_from_pointer(root)
    if not isinstance(launch_id, str) or not LAUNCH_ID_RE.fullmatch(launch_id):
        raise SystemExit("invalid zkGPT detached launch ID")
    launch_root = root / "baseline-local" / "formal-detached"
    launch_dir = launch_root / launch_id
    status_path = launch_dir / "status.json"
    if (launch_root.is_symlink() or launch_dir.is_symlink() or not launch_dir.is_dir() or
            launch_root.resolve() != launch_root or launch_dir.resolve() != launch_dir):
        raise SystemExit("zkGPT detached launch directory is missing or aliased")
    deadline = time.monotonic() + 10.0
    while not status_path.is_file() and time.monotonic() < deadline:
        time.sleep(0.1)
    state = read_json(status_path, "detached launch status")
    status = validate_status(state, root, launch_id, status_path)
    if status in TERMINAL_STATES:
        print(f"ZKGPT_FORMAL_STOP_ALREADY_TERMINAL launch_id={launch_id} status={status}")
        return 0
    supervisor_pid = state["supervisor_pid"]
    validate_live_process(state, "supervisor", supervisor_pid)
    if status == "RUNNING":
        validate_live_process(state, "child", state["child_pid"])
    current_state = read_json(status_path, "detached launch status")
    current_status = validate_status(current_state, root, launch_id, status_path)
    if current_status in TERMINAL_STATES:
        print(f"ZKGPT_FORMAL_STOP_ALREADY_TERMINAL launch_id={launch_id} status={current_status}")
        return 0
    if current_status == "RUNNING":
        validate_live_process(current_state, "child", current_state["child_pid"])
    if (current_state.get("supervisor_pid") != supervisor_pid or
            current_state.get("supervisor_start_ticks") != state.get("supervisor_start_ticks")):
        raise StopError("detached launch supervisor identity changed before stop")
    pidfd_open = getattr(os, "pidfd_open", None)
    pidfd_send_signal = getattr(signal, "pidfd_send_signal", None)
    if not callable(pidfd_open) or not callable(pidfd_send_signal):
        raise StopError("safe PID-bound SIGTERM requires Linux pidfd support")
    try:
        pidfd = pidfd_open(supervisor_pid, 0)
    except OSError as exc:
        raise StopError(f"could not open exact supervisor PID: {exc}") from exc
    try:
        validate_live_process(current_state, "supervisor", supervisor_pid)
        latest_state = read_json(status_path, "detached launch status")
        latest_status = validate_status(latest_state, root, launch_id, status_path)
        if latest_status in TERMINAL_STATES:
            print(f"ZKGPT_FORMAL_STOP_ALREADY_TERMINAL launch_id={launch_id} status={latest_status}")
            return 0
        if (latest_state.get("supervisor_pid") != supervisor_pid or
                latest_state.get("supervisor_start_ticks") != current_state.get("supervisor_start_ticks")):
            raise StopError("detached launch supervisor identity changed before signal")
        if latest_status == "RUNNING":
            validate_live_process(latest_state, "child", latest_state["child_pid"])
        pidfd_send_signal(pidfd, signal.SIGTERM, None, 0)
    except ProcessLookupError:
        latest_state = read_json(status_path, "detached launch status")
        latest_status = validate_status(latest_state, root, launch_id, status_path)
        if latest_status in TERMINAL_STATES:
            print(f"ZKGPT_FORMAL_STOP_ALREADY_TERMINAL launch_id={launch_id} status={latest_status}")
            return 0
        raise StopError("exact supervisor exited before SIGTERM could be delivered")
    except StopError:
        latest_state = read_json(status_path, "detached launch status")
        latest_status = validate_status(latest_state, root, launch_id, status_path)
        if latest_status in TERMINAL_STATES:
            print(f"ZKGPT_FORMAL_STOP_ALREADY_TERMINAL launch_id={launch_id} status={latest_status}")
            return 0
        raise
    except OSError as exc:
        raise StopError(f"PID-bound SIGTERM could not be delivered: {exc}") from exc
    finally:
        os.close(pidfd)
    print(f"ZKGPT_FORMAL_STOP_SIGNALLED launch_id={launch_id} supervisor_pid={supervisor_pid}")
    deadline = time.monotonic() + 120.0
    while time.monotonic() < deadline:
        latest_state = read_json(status_path, "detached launch status")
        latest_status = validate_status(latest_state, root, launch_id, status_path)
        if latest_status in TERMINAL_STATES:
            print(f"ZKGPT_FORMAL_STOP_COMPLETE launch_id={launch_id} status={latest_status}")
            return 0
        time.sleep(0.5)
    raise StopError(f"SIGTERM was sent, but launch did not reach a terminal state; inspect {status_path}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except StopError as exc:
        print(f"ZKGPT_FORMAL_STOP_REFUSED: {exc}", file=sys.stderr)
        raise SystemExit(2)
