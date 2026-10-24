#!/usr/bin/env python3
"""Inspect, follow, stop, and confirm one exact detached zkDL launch."""
from __future__ import annotations

import argparse
import json
import os
import select
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
DATASETS = ("cora", "citeseer", "pubmed", "ppi", "ogbn_arxiv")


class ControlError(RuntimeError):
    pass


def _json(path: Path) -> dict[str, Any] | None:
    if path.is_symlink() or not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _launch_dir(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        raise ControlError("launch directory must be the exact absolute non-symlink path printed by run_five_detached.sh")
    resolved = path.resolve(strict=True)
    if resolved.parent != RUNS.resolve(strict=True) or not resolved.name.startswith("launch_"):
        raise ControlError("launch directory must be a direct runs/launch_* child")
    return resolved


def _proc(pid: int) -> tuple[str, int, int, list[str]]:
    raw = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    end = raw.rfind(")")
    if end < 0:
        raise ControlError("cannot parse process stat")
    fields = raw[end + 2:].split()
    if len(fields) < 20:
        raise ControlError("process stat is truncated")
    cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    args = [arg.decode("utf-8", errors="replace") for arg in cmdline if arg]
    return fields[19], int(fields[2]), int(fields[3]), args


def _validate_supervisor(launch: Path, metadata: dict[str, Any]) -> int:
    try:
        pid = int(metadata["supervisor_pid"])
        pid_file = int((launch / "supervisor.pid").read_text(encoding="ascii").strip())
        start, pgid, sid, cmdline = _proc(pid)
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
    except (KeyError, OSError, ValueError, IndexError) as exc:
        raise ControlError(f"cannot validate this launch supervisor identity: {exc}") from exc
    script = str((ROOT / "scripts/detached_supervisor.py").resolve())
    if (pid != pid_file or metadata.get("schema") != "zkdl_detached_launch_pid_v1" or
            metadata.get("supervisor_boot_id") != boot_id or metadata.get("boot_id") != boot_id or
            metadata.get("supervisor_start_ticks") != start or metadata.get("supervisor_session_id") != sid or
            sid != pid or pgid != pid or script not in cmdline or "--launch-dir" not in cmdline or
            cmdline[cmdline.index("--launch-dir") + 1] != str(launch)):
        raise ControlError("supervisor PID was reused or no longer matches this exact detached launch")
    return pid


def _formal_session(launch: Path) -> tuple[Path, dict[str, Any]] | None:
    link = _json(launch / "formal_session.json")
    if link is None:
        return None
    try:
        session_id = link["session_id"]
        session = Path(link["session_dir"])
        if (link.get("schema") != "zkdl_detached_formal_session_v2" or
                link.get("launch_id") != launch.name or
                not isinstance(session_id, str) or session.name != f"formal_{session_id}" or
                not __import__("re").fullmatch(r"[0-9a-f]{32}", session_id) or
                session.parent != RUNS or session.is_symlink() or
                session.resolve(strict=True).parent != RUNS.resolve(strict=True)):
            raise ControlError("formal session mapping does not belong to this launch")
        session_state = _json(session / "state.json")
        if session_state is None or session_state.get("session_id") != session_id:
            raise ControlError("formal session state does not match launch mapping")
        return session, session_state
    except (KeyError, OSError, ValueError) as exc:
        if isinstance(exc, ControlError):
            raise
        raise ControlError(f"cannot validate formal session mapping: {exc}") from exc


def status(launch: Path) -> int:
    exit_doc = _json(launch / "exit.json")
    pid_doc = _json(launch / "pid.json")
    if exit_doc is not None:
        print(f"launch_status={exit_doc.get('status')} exit_code={exit_doc.get('exit_code')} launch_dir={launch}")
    elif pid_doc is None:
        print(f"launch_status=STARTING launch_dir={launch} (pid metadata not written yet)")
    else:
        try:
            _validate_supervisor(launch, pid_doc)
            print(f"launch_status=RUNNING supervisor_pid={pid_doc['supervisor_pid']} launch_dir={launch}")
        except ControlError as exc:
            print(f"launch_status=IDENTITY_MISMATCH detail={exc}")
            return 2
    mapped = _formal_session(launch)
    if mapped is None:
        print("formal_session=NOT_YET_MAPPED")
        return 0
    session, state = mapped
    print(f"formal_status={state.get('status')} stage={state.get('stage')} session_dir={session}")
    for dataset in DATASETS:
        row = state.get("dataset_status", {}).get(dataset, {})
        error = row.get("failure", "")
        print(f"dataset={dataset} status={row.get('status', 'UNKNOWN')} stage={row.get('stage', 'unknown')}" +
              (f" failure={error}" if error else ""))
    if state.get("failure"):
        print(f"failure={state['failure']}")
    return 0


def logs(launch: Path, selection: str) -> int:
    if selection == "overall":
        path = launch / "console.log"
    else:
        mapped = _formal_session(launch)
        if mapped is None:
            raise ControlError("formal session has not been mapped yet; use logs --dataset overall")
        session, _state = mapped
        path = (session / "logs" / f"{selection}.log" if selection in DATASETS or selection in ("admission", "build")
                else session / "logs" / f"{selection}.log")
    print(f"following_log={path}", flush=True)
    os.execvp("tail", ["tail", "--retry", "-n", "80", "-F", str(path)])
    return 0


def stop(launch: Path) -> int:
    if _json(launch / "exit.json") is not None:
        print(f"launch_status=ALREADY_FINISHED launch_dir={launch}")
        return 0
    metadata = _json(launch / "pid.json")
    if metadata is None:
        raise ControlError("supervisor PID metadata is not ready; stop was not sent")
    pid = _validate_supervisor(launch, metadata)
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        raise ControlError("this Python/Linux does not provide pidfd signaling; refusing an unsafe PID-based stop")
    try:
        pidfd = os.pidfd_open(pid, 0)
    except OSError as exc:
        raise ControlError(f"cannot open identity-bound supervisor pidfd: {exc}") from exc
    try:
        current = _validate_supervisor(launch, metadata)
        if current != pid:
            raise ControlError("supervisor identity changed before pidfd signal")
        try:
            signal.pidfd_send_signal(pidfd, signal.SIGTERM)
        except ProcessLookupError:
            if _json(launch / "exit.json") is not None:
                print(f"launch_status=ALREADY_FINISHED launch_dir={launch}")
                return 0
            raise ControlError("supervisor exited before the pidfd signal; no replacement PID was signaled")
        except OSError as exc:
            raise ControlError(f"pidfd signal failed without using a numeric PID fallback: {exc}") from exc
        print(f"stop_signal=SIGTERM supervisor_pid={pid} launch_dir={launch}", flush=True)
        poller = select.poll()
        poller.register(pidfd, select.POLLIN)
        if not poller.poll(20000):
            print("stop_status=STILL_STOPPING; rerun the same stop command to inspect this launch")
            return 1
    finally:
        os.close(pidfd)
    exit_doc = _json(launch / "exit.json")
    print(f"stop_status={exit_doc.get('status') if exit_doc else 'SUPERVISOR_EXITED'} launch_dir={launch}")
    return 0


def final(launch: Path) -> int:
    exit_doc = _json(launch / "exit.json")
    mapped = _formal_session(launch)
    if mapped is None:
        print(f"final_status={'RUNNING' if exit_doc is None else exit_doc.get('status')} reason=formal session not mapped")
        return 2
    session, state = mapped
    final_path = state.get("final_validation_path")
    expected_final_path = f"results/{session.name}/final_validation.json"
    if final_path and final_path != expected_final_path:
        print("final_status=FAILED reason=final receipt path is outside the exact session result directory")
        return 2
    receipt_path = ROOT / expected_final_path
    collection_path = ROOT / "results" / session.name / "collected.json"
    receipt = _json(receipt_path)
    collection = _json(collection_path)
    strict_line = "final_validation=PASS records=5 datasets=5 independent_verify=5"
    verified = False
    if receipt is not None and collection is not None and not receipt_path.is_symlink() and not collection_path.is_symlink():
        try:
            verified = (
                exit_doc is not None and exit_doc.get("status") == "PASS" and exit_doc.get("exit_code") == 0 and
                state.get("status") == "PASS" and receipt.get("schema") == "zkDL_formal_final_validation_receipt_v1" and
                receipt.get("status") == "PASS" and receipt.get("session_id") == state.get("session_id") and
                receipt.get("verified_cells") == list(DATASETS) and strict_line in receipt.get("validator_output", []) and
                collection.get("schema") == "zkDL_gat_result_collection_v3" and
                collection.get("session_id") == session.name and len(collection.get("rows", [])) == 5 and
                {row.get("dataset") for row in collection["rows"] if isinstance(row, dict)} == set(DATASETS)
            )
        except (OSError, TypeError, ValueError):
            verified = False
    if verified:
        print(f"final_status=PASS session_dir={session} collection={collection_path}")
        return 0
    print(f"final_status={state.get('status')} launch_status={exit_doc.get('status') if exit_doc else 'RUNNING'} "
          f"session_dir={session} failure={state.get('failure') or (receipt or {}).get('failure', 'strict final validation not complete')}")
    for dataset in DATASETS:
        row = state.get("dataset_status", {}).get(dataset, {})
        print(f"dataset={dataset} status={row.get('status', 'UNKNOWN')} stage={row.get('stage', 'unknown')} "
              f"failure={row.get('failure', '')}")
    return 2


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("status", "stop", "final"):
        command = sub.add_parser(action)
        command.add_argument("--launch-dir", required=True)
    log_parser = sub.add_parser("logs")
    log_parser.add_argument("--launch-dir", required=True)
    log_parser.add_argument("--dataset", choices=("overall", "admission", "build", *DATASETS), default="overall")
    args = parser.parse_args()
    try:
        launch = _launch_dir(args.launch_dir)
        if args.action == "status":
            return status(launch)
        if args.action == "logs":
            return logs(launch, args.dataset)
        if args.action == "stop":
            return stop(launch)
        return final(launch)
    except ControlError as exc:
        print(f"zkdl-detached-control=FAILED error={exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
