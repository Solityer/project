#!/usr/bin/env python3
"""Follow the latest zkGPT formal launch and its active baseline-local stage log."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


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
        status = read_json(launch_dir / "status.json")
        owner = read_json(root / "baseline-local" / "formal-owner.json")
        session_id = owner.get("session_id") if owner else None
        session_status = read_json(
            root / "results" / "formal" / str(session_id) / "session_status.json"
        ) if isinstance(session_id, str) else None
        active_stage = (owner or {}).get("active_stage") or (session_status or {}).get("active_stage")
        active_dataset = (owner or {}).get("active_dataset")
        formal_state = (owner or {}).get("status") or (session_status or {}).get("status")
        log_dir = (root / "baseline-local" / "formal-owner-logs" / session_id
                   if isinstance(session_id, str) else None)
        active_logs: list[Path] = []
        if log_dir is not None and log_dir.is_dir():
            active_logs = sorted(log_dir.glob("*.log"), key=lambda item: item.stat().st_mtime_ns)
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
            str(selected_log) if selected_log else None,
            selected_tail[-1] if selected_tail else None,
            console_tail[-1] if console_tail else None,
        )
        if snapshot != previous:
            print(
                "ZKGPT_PROGRESS "
                f"launch_id={launch_id} launch_status={(status or {}).get('status', 'STARTING')} "
                f"exit_code={(status or {}).get('exit_code')} "
                f"formal_session={session_id or 'pending'} formal_status={formal_state or 'pending'} "
                f"stage={active_stage or 'pending'} dataset={active_dataset or '-'}",
                flush=True,
            )
            if selected_log is not None:
                print(f"ACTIVE_STAGE_LOG {selected_log}", flush=True)
                for line in selected_tail:
                    print(line, flush=True)
            for line in console_tail:
                print(f"CONSOLE {line}", flush=True)
            previous = snapshot
        if args.once or (status is not None and status.get("status") in
                         ("COMPLETE", "FAILED", "INTERRUPTED")):
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
