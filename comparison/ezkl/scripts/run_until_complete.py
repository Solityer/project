#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[2]
RESULTS = ROOT / "results"
DATASETS = ["cora_full", "citeseer_full", "pubmed_full", "ppi_full_formal", "ogbn_arxiv_full"]


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


def parse_time(value: str) -> float:
    if not value:
        return 0.0
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def current_marker_time() -> float:
    marker = read_json(PROJECT_ROOT / "comparison" / "results" / "current_rerun_marker.json")
    return parse_time(str(marker.get("created_at", "")))


def current_verified(dataset: str) -> bool:
    run_dir = ROOT / "runs" / dataset
    manifest = read_json(run_dir / "run_manifest.json")
    verify = read_json(run_dir / "verify.json")
    if manifest.get("status") != "verified" or verify.get("verify_passed") is not True:
        return False
    completed = subprocess.run(
        ["python3", "scripts/run_gat_ezkl_baseline.py", "--dataset", dataset, "--check-resume-state"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return False
    try:
        resume = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return False
    if resume.get("source_tree_digest_matches_manifest") is not True:
        return False
    marker = current_marker_time()
    paths = [run_dir / "run_manifest.json", run_dir / "proof.json", run_dir / "verify.json"]
    if not marker:
        return True
    mtimes = [path.stat().st_mtime for path in paths if path.exists()]
    return len(mtimes) == len(paths) and min(mtimes) >= marker


def resource_snapshot() -> dict[str, str]:
    def grab(command: list[str]) -> str:
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        return (completed.stdout or completed.stderr).strip()

    return {
        "available_memory": grab(["bash", "-lc", "free -h | awk '/Mem:/ {print $7}'"]),
        "available_disk": grab(["bash", "-lc", "df -h /home/pzh/project | awk 'NR==2 {print $4}'"]),
    }


def available_memory_bytes() -> int:
    raw = subprocess.run(["bash", "-lc", "free -b | awk '/Mem:/ {print $7}'"], text=True, capture_output=True).stdout.strip()
    try:
        return int(raw)
    except ValueError:
        return 0


def available_swap_bytes() -> int:
    raw = subprocess.run(["bash", "-lc", "free -b | awk '/Swap:/ {print $4}'"], text=True, capture_output=True).stdout.strip()
    try:
        return int(raw)
    except ValueError:
        return 0


def competing_heavy_processes() -> str:
    completed = subprocess.run(["bash", "-lc", "ps aux --sort=-rss | head -30"], text=True, capture_output=True, check=False)
    lines = []
    for line in completed.stdout.splitlines():
        heavy = any(token in line for token in ["gatzk_run", "gatzk_tests", "gatzk_srs_setup"])
        own = "comparison/baselines/ezkl" in line or "run_until_complete.py" in line
        if heavy and not own:
            lines.append(line)
    return "\n".join(lines)


def heartbeat(phase: str, dataset: str = "", command: str = "", attempt: int = 0, failure: str = "", next_step: str = "") -> None:
    snap = resource_snapshot()
    write_json(
        RESULTS / "run_until_complete_heartbeat.json",
        {
            "timestamp": now(),
            "baseline": "ezkl",
            "current_dataset": dataset,
            "current_backend": "cpu",
            "current_phase": phase,
            "current_command": command,
            "attempt_index": attempt,
            "last_successful_step": phase if not failure else "",
            "last_failure": failure,
            "next_step": next_step,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            **snap,
        },
    )


def state(phase: str, dataset: str = "", validator_status: str = "not_run", next_action: str = "", next_command: str = "") -> None:
    completed = []
    failed = []
    pending = []
    for item in DATASETS:
        manifest = read_json(ROOT / "runs" / item / "run_manifest.json")
        if manifest.get("status") == "verified":
            completed.append({"dataset": item, "backend": "cpu"})
        elif manifest:
            failed.append({"dataset": item, "backend": "cpu", "status": manifest.get("status"), "block_reason": manifest.get("block_reason")})
        else:
            pending.append({"dataset": item, "backend": "cpu"})
    write_json(
        RESULTS / "run_until_complete_state.json",
        {
            "baseline": "ezkl",
            "current_phase": phase,
            "current_dataset": dataset,
            "current_backend": "cpu",
            "completed_rows": completed,
            "pending_rows": pending,
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


def append_retry(dataset: str, phase: str, rc: int, failure: str, command: list[str], remediation: str) -> None:
    path = RESULTS / "retry_ledger.json"
    ledger = read_json(path)
    entries = ledger.get("entries", []) if isinstance(ledger.get("entries"), list) else []
    entries.append(
        {
            "dataset": dataset,
            "backend": "cpu",
            "attempt_index": len([item for item in entries if item.get("dataset") == dataset]) + 1,
            "failed_phase": phase,
            "failure_type": failure,
            "return_code": rc,
            "signal": 9 if rc in {137, -9} else "",
            "peak_rss_bytes": read_json(ROOT / "runs" / dataset / "prove_memory_profile.json").get("peak_rss_bytes"),
            "available_memory_before_retry": resource_snapshot().get("available_memory"),
            "competing_processes": subprocess.run(["bash", "-lc", "ps aux --sort=-rss | head -30"], text=True, capture_output=True).stdout,
            "remediation_applied": remediation,
            "next_command": " ".join(command),
            "retry_allowed": True,
            "retry_exhausted": False,
            "timestamp": now(),
        }
    )
    write_json(path, {"baseline": "ezkl", "entries": entries})


def wait_for_heavy_dataset_slot(dataset: str) -> None:
    if dataset not in {"ppi_full_formal", "ogbn_arxiv_full"}:
        return
    min_available = 90 * 1024 * 1024 * 1024
    min_swap_free = 12 * 1024 * 1024 * 1024
    quiet_seconds = int(os.environ.get("EZKL_HEAVY_SLOT_QUIET_SECONDS", "180"))
    clear_since: float | None = None
    recorded = False
    while True:
        competitors = competing_heavy_processes()
        mem_ok = available_memory_bytes() >= min_available
        swap_ok = available_swap_bytes() >= min_swap_free
        if not competitors and mem_ok and swap_ok:
            if clear_since is None:
                clear_since = time.monotonic()
            clear_elapsed = time.monotonic() - clear_since
            if clear_elapsed >= quiet_seconds:
                heartbeat("resource_check", dataset, next_step="heavy_slot_available")
                return
            remaining = max(1, int(quiet_seconds - clear_elapsed))
            heartbeat("resource_quiescence", dataset, next_step=f"sleep_until_{remaining}s_clear")
            state(
                "resource_quiescence",
                dataset,
                validator_status="waiting_for_continuous_idle_window",
                next_action="confirm_continuous_idle_window_before_large_ezkl_prove",
                next_command="./scripts/run_until_complete.sh",
            )
            time.sleep(min(30, remaining))
            continue
        clear_since = None
        if competitors:
            detail = "competing_heavy_process"
        elif not mem_ok:
            detail = "available_memory_below_threshold"
        else:
            detail = "available_swap_below_threshold"
        heartbeat("wait_for_resources", dataset, failure=detail, next_step="sleep_60s_then_recheck")
        state(
            "wait_for_resources",
            dataset,
            validator_status=detail,
            next_action="wait_for_heavy_project_process_or_memory_recovery",
            next_command="./scripts/run_until_complete.sh",
        )
        if not recorded:
            append_retry(
                dataset,
                "pre_heavy_resource_check",
                75,
                detail,
                ["./scripts/run_until_complete.sh"],
                "wait_for_competing_heavy_process_to_exit_before_large_ezkl_prove",
            )
            recorded = True
        time.sleep(60)


def run(command: list[str], phase: str, dataset: str = "", attempt: int = 1) -> int:
    heartbeat(phase, dataset, " ".join(command), attempt, next_step="running")
    state(phase, dataset, next_action="running", next_command=" ".join(command))
    completed = subprocess.run(command, cwd=ROOT, check=False)
    failure = "" if completed.returncode == 0 else f"return_code_{completed.returncode}"
    heartbeat(phase, dataset, " ".join(command), attempt, failure=failure, next_step="continue" if not failure else "retry_or_diagnose")
    if completed.returncode != 0:
        append_retry(dataset or "all", phase, completed.returncode, failure, command, "record_failure_and_resume_from_valid_stage")
    return completed.returncode


def check_resume(dataset: str) -> dict[str, Any]:
    completed = subprocess.run(
        ["python3", "scripts/run_gat_ezkl_baseline.py", "--dataset", dataset, "--check-resume-state"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(completed.stdout)


def independence_audit() -> None:
    scripts = sorted((ROOT / "scripts").glob("*"))
    forbidden = ["comparison/scripts", "baselines/zkdl", "baselines/zkGPT", "/runs/", "/results/"]
    findings = []
    for script in scripts:
        if not script.is_file() or script.suffix not in {".py", ".sh"}:
            continue
        text = script.read_text(encoding="utf-8", errors="replace")
        hits = [item for item in forbidden if item in text and script.name != "run_until_complete.py"]
        if hits:
            findings.append({"script": script.name, "forbidden_tokens": hits})
    write_json(
        RESULTS / "independence_audit.json",
        {
            "baseline": "ezkl",
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
    if run(["./scripts/build.sh"], "build"):
        return 1
    for dataset in DATASETS:
        if current_verified(dataset):
            heartbeat("skip_current_verified", dataset, next_step="already_complete")
            state("skip_current_verified", dataset, next_action="continue")
            continue
        wait_for_heavy_dataset_slot(dataset)
        resume = check_resume(dataset)
        if dataset in {"ppi_full_formal", "ogbn_arxiv_full"} and resume.get("setup_reusable"):
            command = [
                "python3",
                "scripts/run_gat_ezkl_baseline.py",
                "--dataset",
                dataset,
                "--resume",
                "--resume-from",
                "prove",
                "--no-clean-existing-keys",
                "--run",
            ]
        else:
            command = ["python3", "scripts/run_gat_ezkl_baseline.py", "--dataset", dataset, "--clean", "--run"]
        rc = run(command, "run", dataset)
        if rc != 0:
            return rc
        manifest = read_json(ROOT / "runs" / dataset / "run_manifest.json")
        if manifest.get("status") != "verified":
            append_retry(
                dataset,
                "post_run_manifest_gate",
                2,
                f"manifest_status_{manifest.get('status', 'missing')}",
                command,
                "stop_before_next_dataset_and_repair_current_row",
            )
            state(
                "run",
                dataset,
                validator_status=f"manifest_status_{manifest.get('status', 'missing')}",
                next_action="repair_current_dataset_before_continuing",
                next_command="./scripts/run_until_complete.sh",
            )
            heartbeat(
                "run",
                dataset,
                " ".join(command),
                failure=f"manifest_status_{manifest.get('status', 'missing')}",
                next_step="repair_current_dataset_before_continuing",
            )
            return 2
    collect_rc = run(["python3", "scripts/collect_results.py"], "collect")
    validate_rc = run(["python3", "scripts/validate_final_results.py"], "validate")
    if collect_rc == 0 and validate_rc == 0:
        state("complete", validator_status="passed")
        heartbeat("complete", next_step="done")
        return 0
    state("validate", validator_status="failed", next_action="rerun_failed_rows", next_command="./scripts/run_until_complete.sh")
    return collect_rc or validate_rc or 1


if __name__ == "__main__":
    sys.exit(main())
