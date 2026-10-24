#!/usr/bin/env python3
"""Follow the latest zkGPT formal launch and its active baseline-local stage log."""

from __future__ import annotations

import argparse
import errno
import json
import time
from pathlib import Path
from typing import Any

import stop_formal


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def tail(path: Path, count: int = 12) -> list[str]:
    try:
        with path.open("rb") as stream:
            stream.seek(0, 2)
            end = stream.tell()
            cursor = end
            newlines = 0
            while cursor > 0 and newlines <= count:
                step = min(8192, cursor)
                cursor -= step
                stream.seek(cursor)
                block = stream.read(step)
                newlines += block.count(b"\n")
            stream.seek(cursor)
            lines = stream.read(end - cursor).decode("utf-8", errors="replace").splitlines()
    except OSError:
        return []
    return lines[-count:]


def _live_process_health(state: dict[str, Any], role: str) -> tuple[str, str]:
    pid = state.get(f"{role}_pid")
    try:
        current = stop_formal.process_identity(pid)
    except stop_formal.StopError as exc:
        cause = exc.__cause__
        if isinstance(cause, OSError) and cause.errno in (errno.ENOENT, errno.ESRCH):
            return "NO_LIVE_OWNER", f"{role} process exited"
        return "OWNER_UNVERIFIED", f"cannot inspect {role} identity: {exc}"
    expected = {
        "start_ticks": state.get(f"{role}_start_ticks"),
        "process_group": state.get(f"{role}_process_group"),
        "cwd": state.get(f"{role}_cwd"),
        "executable": state.get(f"{role}_executable"),
        "argv": state.get(f"{role}_argv"),
    }
    if current != expected or current["process_group"] != pid:
        return "NO_LIVE_OWNER", f"{role} PID identity mismatch"
    return "LIVE", ""


def inspect_live_launch(root: Path, launch_id: str, status_path: Path) -> tuple[str, dict[str, Any] | None, str]:
    try:
        state = stop_formal.read_json(status_path, "detached launch status")
        launch_status = stop_formal.validate_status(state, root, launch_id, status_path)
    except stop_formal.StopError as exc:
        health = "OWNER_UNVERIFIED" if isinstance(exc.__cause__, PermissionError) else "OWNER_STATUS_INVALID"
        return health, None, str(exc)
    if launch_status in stop_formal.TERMINAL_STATES:
        return "TERMINAL", state, ""
    health, reason = _live_process_health(state, "supervisor")
    if health != "LIVE":
        if health == "NO_LIVE_OWNER":
            try:
                latest = stop_formal.read_json(status_path, "detached launch status")
                latest_status = stop_formal.validate_status(latest, root, launch_id, status_path)
            except stop_formal.StopError as exc:
                latest_health = "OWNER_UNVERIFIED" if isinstance(exc.__cause__, PermissionError) else "OWNER_STATUS_INVALID"
                return latest_health, state, str(exc)
            if latest_status in stop_formal.TERMINAL_STATES:
                return "TERMINAL", latest, ""
        return health, state, reason
    if launch_status == "RUNNING":
        health, reason = _live_process_health(state, "child")
        if health != "LIVE":
            try:
                latest = stop_formal.read_json(status_path, "detached launch status")
                latest_status = stop_formal.validate_status(latest, root, launch_id, status_path)
            except stop_formal.StopError as exc:
                health = "OWNER_UNVERIFIED" if isinstance(exc.__cause__, PermissionError) else "OWNER_STATUS_INVALID"
                return health, state, str(exc)
            if latest_status in stop_formal.TERMINAL_STATES:
                return "TERMINAL", latest, ""
            if latest_status != "RUNNING":
                return "OWNER_STATUS_INVALID", latest, "launch status changed unexpectedly while checking its child"
            supervisor_health, supervisor_reason = _live_process_health(latest, "supervisor")
            if supervisor_health != "LIVE":
                return supervisor_health, latest, supervisor_reason
            health, reason = _live_process_health(latest, "child")
            return health, latest, reason
    return "LIVE", state, ""


def session_for_launch(
    root: Path, launch_status: dict[str, Any] | None,
) -> tuple[str | None, dict[str, Any] | None, Path | None]:
    if launch_status is None:
        return None, None, None
    launch_started = launch_status.get("started_unix_ns")
    scope = launch_status.get("requested_datasets")
    if isinstance(launch_started, bool) or not isinstance(launch_started, int):
        return None, None, None
    try:
        session_dirs = list((root / "results" / "formal").iterdir())
    except OSError:
        return None, None, None
    candidates: list[tuple[int, str, dict[str, Any], Path]] = []
    for session_dir in session_dirs:
        if (not session_dir.is_dir() or session_dir.is_symlink() or
                len(session_dir.name) != 32):
            continue
        session_status = read_json(session_dir / "session_status.json")
        if not isinstance(session_status, dict) or session_status.get("session_id") != session_dir.name:
            continue
        started = session_status.get("started_unix_ns")
        if isinstance(started, bool) or not isinstance(started, int) or started < launch_started:
            continue
        if scope is not None and session_status.get("dataset_scope") != scope:
            continue
        candidates.append((started, session_dir.name, session_status, session_dir))
    if not candidates:
        return None, None, None
    _started, session_id, session_status, session_dir = max(candidates, key=lambda item: item[0])
    return session_id, session_status, session_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=float, default=5.0)
    args = parser.parse_args()
    if args.interval < 1.0 or args.interval > 60.0:
        parser.error("--interval must be between 1 and 60 seconds")

    root = Path(__file__).resolve().parent.parent
    pointer_path = root / "baseline-local" / "formal-detached" / "latest_launch.json"
    previous: tuple[Any, ...] | None = None
    while True:
        pointer = read_json(pointer_path)
        if pointer is None:
            print(f"zkGPT formal launch pointer is not ready: {pointer_path}", flush=True)
            return 1
        launch_id = pointer.get("launch_id")
        if not isinstance(launch_id, str) or len(launch_id) != 32:
            print("zkGPT latest launch pointer has an invalid identity", flush=True)
            return 1
        launch_dir = pointer_path.parent / launch_id
        owner_health, status, owner_reason = inspect_live_launch(root, launch_id, launch_dir / "status.json")
        session_id, session_status, session_dir = session_for_launch(root, status)
        active_stage = (session_status or {}).get("active_stage")
        active_dataset = (session_status or {}).get("active_dataset")
        formal_state = "INTERRUPTED" if owner_health == "NO_LIVE_OWNER" else (session_status or {}).get("status")
        if ((status or {}).get("status") in stop_formal.TERMINAL_STATES and
                formal_state == "RUNNING"):
            formal_state = "INTERRUPTED"
        dataset_scope = ((status or {}).get("requested_datasets") or
                         (session_status or {}).get("dataset_scope"))
        if isinstance(dataset_scope, list) and all(isinstance(item, str) for item in dataset_scope):
            scope_text = ",".join(dataset_scope)
        else:
            scope_text = "all5"
        active_logs: list[Path] = []
        if session_dir is not None:
            search_roots = [session_dir]
            if isinstance(active_dataset, str) and isinstance(session_id, str):
                search_roots.append(root / "baseline-local" / "runs" / "formal" / session_id / active_dataset)
            for search_root in search_roots:
                try:
                    active_logs.extend(
                        path for path in search_root.rglob("*.log")
                        if path.is_file() and not path.is_symlink()
                    )
                except OSError:
                    pass
        active_logs.sort(key=lambda item: item.stat().st_mtime_ns)
        selected_log = active_logs[-1] if active_logs else None
        selected_tail = tail(selected_log) if selected_log is not None else []
        console_path = launch_dir / "console.log"
        console_tail = tail(console_path, 4)
        snapshot = (
            launch_id,
            (status or {}).get("status"),
            (status or {}).get("exit_code"),
            session_id,
            formal_state,
            active_stage,
            active_dataset,
            scope_text,
            owner_health,
            owner_reason,
            str(selected_log) if selected_log else None,
            selected_tail[-1] if selected_tail else None,
            console_tail[-1] if console_tail else None,
        )
        if snapshot != previous:
            print(
                "ZKGPT_PROGRESS "
                f"launch_id={launch_id} launch_status={(status or {}).get('status', 'STARTING')} "
                f"exit_code={(status or {}).get('exit_code')} "
                f"datasets={scope_text} "
                f"formal_session={session_id or 'pending'} formal_status={formal_state or 'pending'} "
                f"live_owner={owner_health} stage={active_stage or 'pending'} dataset={active_dataset or '-'}",
                flush=True,
            )
            if selected_log is not None:
                print(f"ACTIVE_STAGE_LOG {selected_log}", flush=True)
                for line in selected_tail:
                    print(line, flush=True)
            for line in console_tail:
                print(f"CONSOLE {line}", flush=True)
            if owner_health != "LIVE" and owner_health != "TERMINAL":
                print(f"ZKGPT_PROGRESS_OWNER_CHECK {owner_health} reason={owner_reason}", flush=True)
            if formal_state == "INTERRUPTED" and owner_health == "TERMINAL":
                print(f"ZKGPT_PROGRESS_INTERRUPTED launch_id={launch_id} reason=launch ended while formal session remained RUNNING", flush=True)
            previous = snapshot
        if owner_health == "NO_LIVE_OWNER":
            print(f"ZKGPT_PROGRESS_INTERRUPTED launch_id={launch_id} reason={owner_reason}", flush=True)
            return 2
        if owner_health == "OWNER_UNVERIFIED" or owner_health == "OWNER_STATUS_INVALID":
            print(f"ZKGPT_PROGRESS_OWNER_UNVERIFIED launch_id={launch_id} reason={owner_reason}", flush=True)
            return 2
        if args.once or (status is not None and status.get("status") in
                         ("COMPLETE", "SUBSET_COMPLETE", "SUBSET_FAILED", "FAILED", "INTERRUPTED")):
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
