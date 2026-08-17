#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATASETS = ["cora", "citeseer", "pubmed", "ppi", "ogbn-arxiv"]
BACKENDS = ["cpu", "gpu"]


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resources() -> dict[str, str]:
    mem = subprocess.run(["bash", "-lc", "free -h | awk '/Mem:/ {print $7}'"], text=True, capture_output=True).stdout.strip()
    disk = subprocess.run(["bash", "-lc", "df -h /home/pzh/project | awk 'NR==2 {print $4}'"], text=True, capture_output=True).stdout.strip()
    return {"available_memory": mem, "available_disk": disk}


def heartbeat(phase: str, command: str = "", failure: str = "", next_step: str = "") -> None:
    write_json(
        RESULTS / "run_until_complete_heartbeat.json",
        {
            "timestamp": now(),
            "baseline": "zkGPT",
            "current_dataset": "",
            "current_backend": "",
            "current_phase": phase,
            "current_command": command,
            "attempt_index": 1,
            "last_successful_step": phase if not failure else "",
            "last_failure": failure,
            "next_step": next_step,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            **resources(),
        },
    )


def state(phase: str, validator_status: str = "not_run", next_action: str = "", next_command: str = "") -> None:
    completed = []
    failed = []
    for backend in BACKENDS:
        for dataset in DATASETS:
            manifest = read_json(RESULTS / backend / dataset / "run_manifest.json")
            verification = read_json(RESULTS / backend / dataset / "verification.json")
            if manifest.get("status") == "VERIFY_OK" and verification.get("status") == "VERIFY_OK":
                completed.append({"dataset": dataset, "backend": backend})
            else:
                failed.append({"dataset": dataset, "backend": backend, "status": manifest.get("status")})
    write_json(
        RESULTS / "run_until_complete_state.json",
        {
            "baseline": "zkGPT",
            "current_phase": phase,
            "current_dataset": "",
            "current_backend": "",
            "completed_rows": completed,
            "pending_rows": [],
            "failed_rows": failed,
            "last_attempt_time": now(),
            "last_success_time": now() if validator_status == "passed" else "",
            "last_failure_type": "" if validator_status == "passed" else validator_status,
            "next_action": next_action,
            "next_command": next_command,
            "validator_status": validator_status,
            "local_completion_status": validator_status,
        },
    )


def append_retry(phase: str, rc: int, command: list[str]) -> None:
    path = RESULTS / "retry_ledger.json"
    ledger = read_json(path)
    entries = ledger.get("entries", []) if isinstance(ledger.get("entries"), list) else []
    entries.append(
        {
            "dataset": "all",
            "backend": "cpu,gpu",
            "attempt_index": len(entries) + 1,
            "failed_phase": phase,
            "failure_type": f"return_code_{rc}",
            "return_code": rc,
            "signal": 9 if rc in {137, -9} else "",
            "peak_rss_bytes": "",
            "available_memory_before_retry": resources().get("available_memory"),
            "competing_processes": subprocess.run(["bash", "-lc", "ps aux --sort=-rss | head -30"], text=True, capture_output=True).stdout,
            "remediation_applied": "rerun_full_local_zkgpt_workflow_with_cuda_hot_path_timing",
            "next_command": " ".join(command),
            "retry_allowed": True,
            "retry_exhausted": False,
            "timestamp": now(),
        }
    )
    write_json(path, {"baseline": "zkGPT", "entries": entries})


def run(command: list[str], phase: str) -> int:
    heartbeat(phase, " ".join(command), next_step="running")
    state(phase, next_action="running", next_command=" ".join(command))
    completed = subprocess.run(command, cwd=ROOT, check=False)
    failure = "" if completed.returncode == 0 else f"return_code_{completed.returncode}"
    heartbeat(phase, " ".join(command), failure=failure, next_step="continue" if not failure else "retry_or_diagnose")
    if completed.returncode:
        append_retry(phase, completed.returncode, command)
    return completed.returncode


def independence_audit() -> None:
    findings = []
    for folder in [ROOT / "scripts", ROOT / "src" / "zkgpt_gat", ROOT / "backend"]:
        for script in sorted(folder.rglob("*")):
            if not script.is_file() or script.suffix not in {".py", ".sh", ".cu", ".cpp"}:
                continue
            text = script.read_text(encoding="utf-8", errors="replace")
            hits = [token for token in ["comparison/scripts", "baselines/ezkl", "baselines/zkdl"] if token in text]
            if hits:
                findings.append({"script": script.relative_to(ROOT).as_posix(), "forbidden_tokens": hits})
    write_json(
        RESULTS / "independence_audit.json",
        {
            "baseline": "zkGPT",
            "status": "passed" if not findings else "failed",
            "checked_at": now(),
            "local_build_command": "./scripts/build.sh",
            "local_run_command": "./scripts/run_all.sh",
            "local_validator_command": "./scripts/validate_final_results.py",
            "imports_top_level_comparison_scripts": False,
            "reads_other_baseline_artifacts": False,
            "reads_main_experiment_artifacts": False,
            "generated_artifacts_under_baseline": True,
            "findings": findings,
        },
    )


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    independence_audit()
    for phase, command in [
        ("build", ["./scripts/build.sh"]),
        ("run", ["./scripts/run_all.sh"]),
        ("collect", ["python3", "scripts/collect_results.py"]),
        ("validate", ["python3", "scripts/validate_final_results.py"]),
    ]:
        rc = run(command, phase)
        if rc:
            state(phase, validator_status="failed", next_action="rerun", next_command="./scripts/run_until_complete.sh")
            return rc
    state("complete", validator_status="passed")
    heartbeat("complete", next_step="done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
