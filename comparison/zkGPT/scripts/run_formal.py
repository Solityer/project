#!/usr/bin/env python3
"""Own the lock, session, and bounded stage process groups for formal zkGPT."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

from common import (
    ChainError,
    atomic_write_json,
    baseline_root,
    create_formal_session,
    formal_run_lock,
    formal_session_status_path,
    install_process_signal_cleanup,
    invalidate_results,
    run_logged,
    ensure_directory,
)


def _write_status(root: Path, session_id: str, status: str, active_stage: str,
                  completed: Sequence[str], stage_status: Dict[str, str],
                  started_unix_ns: int, failure: str = "") -> None:
    atomic_write_json(formal_session_status_path(root, session_id), {
        "schema": "zkgpt-gat-formal-session-v1",
        "session_id": session_id,
        "status": status,
        "active_stage": active_stage,
        "completed_stages": list(completed),
        "stage_status": dict(stage_status),
        "started_unix_ns": started_unix_ns,
        "updated_unix_ns": time.time_ns(),
        "failure": failure,
    }, root)


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(list(argv))
    root = baseline_root()
    session_id = ""
    session_started = 0
    completed: List[str] = []
    stage_status: Dict[str, str] = {}
    active_stage = ""
    try:
        with formal_run_lock(root):
            invalidate_results(root, ("formal_preflight.json", "collected_results.json", "final_validation.json"))
            session_id, session, _session_manifest_sha256 = create_formal_session(root)
            os.environ["ZKGPT_FORMAL_SESSION_ID"] = session_id
            install_process_signal_cleanup()
            session_started = int(session["started_unix_ns"])
            active_stage = "session-created"
            steps = (
                ("resource-authentication", [sys.executable, str(root / "scripts" / "prepare_resources.py"), "--check"]),
                ("native-build", [str(root / "build.sh"), "--cpu"]),
                ("five-dataset-preflight", [sys.executable, str(root / "scripts" / "formal_preflight.py")]),
                ("five-dataset-execution", [sys.executable, str(root / "scripts" / "run_five_datasets.py")]),
                ("collection", [sys.executable, str(root / "scripts" / "collect_results.py")]),
                ("final-validation", [sys.executable, str(root / "scripts" / "validate_final_results.py")]),
            )
            logs = root / "results" / "formal" / session_id / "logs"
            ensure_directory(logs, "formal session logs", root)
            _write_status(root, session_id, "RUNNING", active_stage, completed, stage_status, session_started)
            print(f"FORMAL_SESSION={session_id}", flush=True)
            for stage_name, command in steps:
                active_stage = stage_name
                stage_status[stage_name] = "RUNNING"
                _write_status(root, session_id, "RUNNING", active_stage, completed, stage_status, session_started)
                print(f"START {stage_name}", flush=True)
                log_base = logs / stage_name
                try:
                    returncode = run_logged(
                        command, root, log_base.with_suffix(".stdout.log"),
                        log_base.with_suffix(".stderr.log"), stage_name,
                        extra_env={
                            "ZKGPT_FORMAL_SESSION_ID": session_id,
                            "ZKGPT_OWNED_PROCESS_GROUP": "1",
                        },
                        isolate_process_group=True,
                    )
                except KeyboardInterrupt:
                    stage_status[stage_name] = "INTERRUPTED"
                    _write_status(root, session_id, "INTERRUPTED", stage_name,
                                  completed, stage_status, session_started,
                                  "received SIGINT/SIGTERM; owned stage group was terminated and reaped")
                    raise
                if returncode != 0:
                    stage_status[stage_name] = "FAILED"
                    _write_status(root, session_id, "FAILED", stage_name, completed,
                                  stage_status, session_started,
                                  f"stage exited with code {returncode}")
                    print(f"FAILED {stage_name} exit_code={returncode}", file=sys.stderr, flush=True)
                    return returncode if returncode > 0 else 2
                stage_status[stage_name] = "COMPLETE"
                completed.append(stage_name)
                active_stage = ""
                _write_status(root, session_id, "RUNNING", "", completed, stage_status, session_started)
                print(f"COMPLETE {stage_name}", flush=True)
            _write_status(root, session_id, "COMPLETE", "", completed, stage_status, session_started)
            print(f"FINAL_VALIDATION_OK session={session_id} results=results/formal/{session_id}/final_validation.json", flush=True)
            return 0
    except KeyboardInterrupt:
        if session_id:
            if active_stage and stage_status.get(active_stage) == "RUNNING":
                stage_status[active_stage] = "INTERRUPTED"
            try:
                _write_status(
                    root, session_id, "INTERRUPTED", active_stage, completed,
                    stage_status, session_started,
                    "received SIGINT/SIGTERM; formal chain stopped before completion",
                )
            except OSError as exc:
                print(f"FORMAL_STATUS_WRITE_FAILED: {exc}", file=sys.stderr, flush=True)
        return 130
    except Exception as exc:
        if session_id:
            if active_stage and stage_status.get(active_stage) == "RUNNING":
                stage_status[active_stage] = "FAILED"
            try:
                _write_status(
                    root, session_id, "FAILED", active_stage, completed,
                    stage_status, session_started, str(exc),
                )
            except OSError as status_exc:
                print(f"FORMAL_STATUS_WRITE_FAILED: {status_exc}", file=sys.stderr, flush=True)
        print(f"FORMAL_RUN_BLOCKED: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
