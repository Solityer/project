#!/usr/bin/env python3
"""Run selected zkGPT datasets serially under one local run lock."""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from common import (
    ChainError,
    DATASETS,
    FINAL_SCHEMA,
    PAPER_TIMING_SEMANTICS,
    atomic_write_json,
    atomic_write_text,
    baseline_root,
    canonical_dataset_scope,
    create_formal_session,
    ensure_directory,
    formal_final_path,
    formal_preflight_path,
    formal_results_directory,
    formal_run_lock,
    formal_session_status_path,
    install_process_signal_cleanup,
    load_json,
    local_path,
    relative_path,
    run_logged,
    run_directory,
)
from gat_zkgpt_experiment import run_cell


LOCAL_CAPACITY_PROFILE_SCHEMA = "zkgpt-local-capacity-profile-v3"
CELL_INTERNAL_LOG_RESERVE_BYTES = 16 * 1024 * 1024
HOST_ACCOUNTING_OVERHEAD_BYTES = 64 * 1024 * 1024


def _set_status(
    root: Path, session_id: str, status: str, active_stage: str,
    failure: str = "", dataset: str | None = None,
) -> None:
    path = formal_session_status_path(root, session_id)
    current = load_json(path, "formal session status")
    if not isinstance(current, dict) or current.get("session_id") != session_id:
        raise ChainError("formal session status identity is invalid")
    current.update({
        "status": status,
        "active_stage": active_stage,
        "active_dataset": dataset,
        "updated_unix_ns": time.time_ns(),
        "failure": failure,
    })
    atomic_write_json(path, current, root)
    print(f"ZKGPT_FORMAL_STAGE session_id={session_id} stage={active_stage} dataset={dataset or '-'} status={status}", flush=True)


def _record_completed_stage(root: Path, session_id: str, stage: str) -> None:
    path = formal_session_status_path(root, session_id)
    current = load_json(path, "formal session status")
    if not isinstance(current, dict) or current.get("session_id") != session_id:
        raise ChainError("formal session status identity is invalid")
    completed = current.setdefault("completed_stages", [])
    stage_status = current.setdefault("stage_status", {})
    if not isinstance(completed, list) or not isinstance(stage_status, dict):
        raise ChainError("formal session stage history is malformed")
    if stage not in completed:
        completed.append(stage)
    stage_status[stage] = "PASS"
    current["updated_unix_ns"] = time.time_ns()
    atomic_write_json(path, current, root)


def _scratch_directory(root: Path, session_dir: Path) -> Path:
    scratch = local_path(root, "artifacts/private/native_scratch", "native scratch directory")
    try:
        scratch.relative_to(root.absolute())
    except ValueError as exc:
        raise ChainError("native scratch path escapes the zkGPT baseline") from exc
    ensure_directory(scratch, "native scratch directory", root)
    if scratch.is_symlink() or not scratch.is_dir():
        raise ChainError("native scratch path is aliased or not a directory")
    expected_device = os.stat(root.parent, follow_symlinks=False).st_dev
    if (os.stat(scratch, follow_symlinks=False).st_dev != expected_device or
            os.stat(session_dir.parent.parent, follow_symlinks=False).st_dev != expected_device):
        raise ChainError("formal results and native scratch must share the baseline filesystem")
    os.environ["ZKGPT_NATIVE_SCRATCH_DIR"] = str(scratch)
    return scratch


def _run_logged_stage(root: Path, command: Sequence[str], logs: Path, name: str) -> Tuple[Path, Path]:
    stdout_path = logs / f"{name}.stdout.log"
    stderr_path = logs / f"{name}.stderr.log"
    returncode = run_logged(command, root, stdout_path, stderr_path, name)
    if returncode != 0:
        raise ChainError(f"{name} exited with code {returncode}; see {stderr_path}")
    return stdout_path, stderr_path


def _tree_regular_bytes(path: Path) -> int:
    total = 0
    for current, directories, filenames in os.walk(path, followlinks=False):
        parent = Path(current)
        for name in directories:
            entry = parent / name
            if entry.is_symlink() or not entry.is_dir():
                raise ChainError(f"formal run contains an aliased/non-directory path: {entry}")
        for name in filenames:
            entry = parent / name
            info = entry.lstat()
            if entry.is_symlink() or not entry.is_file() or info.st_nlink != 1:
                raise ChainError(f"formal run contains an aliased/non-file path: {entry}")
            total += info.st_size
    return total


def _capacity_profile(session_id: str, session_dir: Path, preflight: Mapping[str, Any]) -> Dict[str, Any]:
    capacities = preflight.get("capacity")
    if not isinstance(capacities, dict) or set(capacities) != set(DATASETS):
        raise ChainError("zkGPT capacity matrix does not contain the five canonical datasets")
    datasets: Dict[str, Any] = {}
    for dataset in DATASETS:
        item = capacities[dataset]
        if not isinstance(item, dict):
            raise ChainError(f"zkGPT capacity admission is malformed for {dataset}")
        if item.get("status") == "BLOCKED":
            reason = item.get("reason")
            datasets[dataset] = {"status": "BLOCKED", "reason": reason if isinstance(reason, str) else "capacity blocked"}
            continue
        if (item.get("status") != "READY" or
                item.get("proof_path") != "formal_native_masked_lasso_hyrax+native_masked_layered_gkr" or
                item.get("full_relation_bound") is not True):
            raise ChainError(f"zkGPT full relation/capacity admission is not READY for {dataset}")
        scratch = item.get("estimated_native_scratch_bytes")
        disk = item.get("disk_requirement_bytes_upper_bound")
        retained = item.get("artifact_disk_bytes_upper_bound")
        host = item.get("estimated_peak_host_memory_bytes")
        wire = item.get("proof_artifact_wire_bytes_upper_bound")
        context = item.get("context_payload_bytes_bound")
        values = (scratch, disk, retained, host, wire, context)
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values):
            raise ChainError(f"zkGPT source-derived capacity fields are incomplete for {dataset}")
        if disk < scratch + retained:
            raise ChainError(f"zkGPT disk bound does not include scratch plus retained output for {dataset}")
        datasets[dataset] = {
            "status": "READY",
            "additional_disk_peak_bytes": disk + CELL_INTERNAL_LOG_RESERVE_BYTES,
            "retained_bytes_upper_bound": retained + CELL_INTERNAL_LOG_RESERVE_BYTES,
            "host_peak_bytes": host + wire + context + HOST_ACCOUNTING_OVERHEAD_BYTES,
        }
    fs_device = os.stat(session_dir.parent.parent, follow_symlinks=False).st_dev
    return {
        "schema": LOCAL_CAPACITY_PROFILE_SCHEMA,
        "session_id": session_id,
        "fs_device_id": str(fs_device),
        "datasets": datasets,
    }


def _available_host_bytes() -> int:
    try:
        mem_available = None
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            key, _, value = line.partition(":")
            if key == "MemAvailable":
                mem_available = int(value.strip().split()[0]) * 1024
                break
        if mem_available is None:
            raise ValueError("MemAvailable field is missing")
        available = mem_available
        membership: Dict[str, str] = {}
        for line in Path("/proc/self/cgroup").read_text(encoding="ascii").splitlines():
            hierarchy, controllers, path = line.split(":", 2)
            if hierarchy == "0" and not controllers:
                membership["cgroup2"] = path
            elif "memory" in controllers.split(","):
                membership["cgroup"] = path
        mount_lines = Path("/proc/self/mountinfo").read_text(encoding="ascii").splitlines()
    except (OSError, ValueError, IndexError) as exc:
        raise ChainError(f"cannot read live host memory headroom: {exc}") from exc

    def unescape_mount_field(value: str) -> str:
        return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match.group(1), 8)), value)

    cgroup_paths: List[Tuple[Path, str]] = []
    for line in mount_lines:
        before, separator, after = line.partition(" - ")
        if not separator:
            continue
        left = before.split()
        right = after.split()
        if len(left) < 5 or not right:
            continue
        fs_type = right[0]
        if fs_type == "cgroup2":
            group_path = membership.get("cgroup2")
            limit_name, current_name = "memory.max", "memory.current"
            version = "v2"
        elif fs_type == "cgroup" and "memory" in ",".join(right[2:]).split(","):
            group_path = membership.get("cgroup")
            limit_name, current_name = "memory.limit_in_bytes", "memory.usage_in_bytes"
            version = "v1"
        else:
            continue
        if not group_path:
            continue
        mount_root = Path(unescape_mount_field(left[3]))
        mount_point = Path(unescape_mount_field(left[4]))
        try:
            relative = Path(group_path).relative_to(mount_root)
        except ValueError:
            continue
        current_path = mount_point / relative
        while True:
            cgroup_paths.append((current_path, version))
            if current_path == mount_point:
                break
            try:
                current_path = current_path.parent
                current_path.relative_to(mount_point)
            except ValueError:
                break

    for directory, version in cgroup_paths:
        limit_path = directory / ("memory.max" if version == "v2" else "memory.limit_in_bytes")
        current_path = directory / ("memory.current" if version == "v2" else "memory.usage_in_bytes")
        if not limit_path.exists() and not current_path.exists():
            continue
        try:
            raw_limit = limit_path.read_text(encoding="ascii").strip()
            current = int(current_path.read_text(encoding="ascii").strip())
            if raw_limit == "max":
                continue
            limit = int(raw_limit)
            if version == "v1" and limit >= (1 << 60):
                continue
            available = min(available, max(0, limit - current))
        except (OSError, ValueError) as exc:
            raise ChainError(f"cannot read live cgroup memory headroom: {exc}") from exc
    return available


def _capacity_gate(root: Path, session_id: str, session_dir: Path,
                   profile: Mapping[str, Any], dataset: str) -> Dict[str, int]:
    if profile.get("schema") != LOCAL_CAPACITY_PROFILE_SCHEMA or profile.get("session_id") != session_id:
        raise ChainError("zkGPT local capacity profile identity is invalid")
    datasets = profile.get("datasets")
    item = datasets.get(dataset) if isinstance(datasets, dict) else None
    if not isinstance(item, dict) or item.get("status") != "READY":
        reason = item.get("reason") if isinstance(item, dict) else None
        raise ChainError(f"zkGPT capacity is blocked for {dataset}: {reason or 'missing capacity entry'}")
    expected_device = os.stat(root.parent, follow_symlinks=False).st_dev
    scratch = _scratch_directory(root, session_dir)
    if profile.get("fs_device_id") != str(expected_device):
        raise ChainError("zkGPT capacity profile refers to another filesystem")
    host_required = item["host_peak_bytes"]
    disk_required = item["additional_disk_peak_bytes"]
    host_available = _available_host_bytes()
    statvfs = os.statvfs(scratch)
    disk_available = statvfs.f_bavail * statvfs.f_frsize
    if host_available < host_required:
        raise ChainError(f"zkGPT live host headroom is below the bound for {dataset}: {host_available} < {host_required}")
    if disk_available < disk_required:
        raise ChainError(f"zkGPT live disk headroom is below the bound for {dataset}: {disk_available} < {disk_required}")
    observation = {
        "dataset": dataset,
        "host_available_bytes": host_available,
        "host_required_bytes": host_required,
        "disk_available_bytes": disk_available,
        "disk_required_bytes": disk_required,
        "checked_unix_ns": time.time_ns(),
    }
    path = formal_session_status_path(root, session_id)
    current = load_json(path, "formal session status")
    observations = current.setdefault("local_capacity_checks", [])
    if not isinstance(observations, list):
        raise ChainError("formal session capacity history is malformed")
    observations.append(observation)
    atomic_write_json(path, current, root)
    return observation


def _prepare_phase(root: Path, dataset_scope: Sequence[str], session_id: str) -> Tuple[Path, Dict[str, Any], Dict[str, Any], Path]:
    session_id, session = create_formal_session(root, session_id, dataset_scope)
    session_dir = formal_results_directory(root, session_id)
    os.environ["ZKGPT_FORMAL_SESSION_ID"] = session_id
    _scratch_directory(root, session_dir)
    logs = session_dir / "logs"
    ensure_directory(logs, "formal session logs", root)

    _set_status(root, session_id, "RUNNING", "native-build")
    _run_logged_stage(root, [str(root / "build.sh"), "--cpu"], logs, "native-build")
    preflight_stage = "five-dataset-capacity" if tuple(dataset_scope) == DATASETS else "requested-dataset-capacity"
    _set_status(root, session_id, "RUNNING", preflight_stage)
    _run_logged_stage(root, [sys.executable, str(root / "scripts" / "formal_preflight.py")], logs, preflight_stage)
    preflight_path = formal_preflight_path(root, session_id)
    preflight = load_json(preflight_path, "formal preflight")
    if (not isinstance(preflight, dict) or preflight.get("status") not in ("READY", "PARTIAL") or
            preflight.get("dataset_scope") != list(dataset_scope) or
            preflight.get("shared_preconditions_ready") is not True):
        raise ChainError("zkGPT shared build/backend preconditions did not pass")
    profile = _capacity_profile(session_id, session_dir, preflight)
    _record_completed_stage(root, session_id, "prepare")
    return session_dir, preflight, profile, preflight_path


def _cell_phase(root: Path, session_id: str, session_dir: Path,
                preflight: Mapping[str, Any], profile: Mapping[str, Any], dataset: str) -> Dict[str, Any]:
    _set_status(root, session_id, "RUNNING", f"cell:{dataset}", dataset=dataset)
    _capacity_gate(root, session_id, session_dir, profile, dataset)
    result = run_cell(root, dataset, "cpu", preflight, session_id)
    _run_relative, run_path = run_directory(root, dataset, session_id)
    retained = _tree_regular_bytes(run_path)
    bound = profile["datasets"][dataset]["retained_bytes_upper_bound"]
    if retained > bound:
        raise ChainError(f"zkGPT retained output exceeds the capacity bound for {dataset}: {retained} > {bound}")
    _record_completed_stage(root, session_id, f"cell:{dataset}")
    _set_status(root, session_id, "RUNNING", f"cell-complete:{dataset}", dataset=dataset)
    return result


def _finalize_phase(root: Path, session_id: str, session_dir: Path,
                    dataset_scope: Sequence[str], cells: Mapping[str, Mapping[str, Any]],
                    failures: Mapping[str, str]) -> Tuple[Path, Dict[str, Any]]:
    scope = canonical_dataset_scope(dataset_scope)
    rows = [dict(cells[dataset]) if dataset in cells else
            {"dataset": dataset, "backend": "cpu", "status": "FAILED", "reason": failures[dataset]}
            for dataset in scope]
    complete = scope == DATASETS and not failures and len(cells) == len(DATASETS)
    any_failure = bool(failures)
    if scope == DATASETS:
        status = "VALIDATED" if complete else "FAILED"
        path = formal_final_path(root, session_id)
        payload = {
            "schema": FINAL_SCHEMA,
            "status": status,
            "baseline": "zkGPT",
            "session_id": session_id,
            "paper_timing_semantics": PAPER_TIMING_SEMANTICS,
            "matrix": {"datasets": list(DATASETS), "cell_count": len(DATASETS)},
            "cells_validated": len(cells),
            "cells": rows,
            "full_five_complete": complete,
            "proof_scope": "full_gat",
            "privacy_profile": "masked_full",
            "independent_verifier": complete,
            "verification_accounting": {
                "cell_stage_independent_verifier_invocations": len(cells),
                "final_stage_native_verifier_invocations": 0,
                "independent_crypto_verifier_invocations_total": len(cells),
            },
            "created_unix_ns": time.time_ns(),
        }
    else:
        status = "SUBSET_FAILED" if any_failure else "SUBSET_COMPLETE"
        path = session_dir / "subset_validation.json"
        payload = {
            "schema": "zkgpt-gat-subset-validation-v2",
            "status": status,
            "validation_scope": "requested_dataset_subset",
            "dataset_scope": list(scope),
            "formal_dataset_universe": list(DATASETS),
            "full_five_complete": False,
            "formal_results_complete": False,
            "session_id": session_id,
            "paper_timing_semantics": PAPER_TIMING_SEMANTICS,
            "independent_verification": not any_failure and len(cells) == len(scope),
            "cells": rows,
            "created_unix_ns": time.time_ns(),
        }
    if path.exists() or path.is_symlink():
        raise ChainError("formal result output already exists; refusing stale-result reuse")
    atomic_write_json(path, payload, root)
    return path, payload


def _run_official(root: Path, dataset_scope: Sequence[str] = DATASETS) -> int:
    dataset_scope = canonical_dataset_scope(dataset_scope)
    with formal_run_lock(root):
        install_process_signal_cleanup()
        session_id = secrets.token_hex(16)
        try:
            session_dir, preflight, profile, _preflight_path = _prepare_phase(root, dataset_scope, session_id)
            cells: Dict[str, Dict[str, Any]] = {}
            failures: Dict[str, str] = {}
            for dataset in dataset_scope:
                try:
                    cells[dataset] = _cell_phase(root, session_id, session_dir,
                                                 preflight, profile, dataset)
                except Exception as exc:
                    reason = f"{type(exc).__name__}: {exc}"
                    failures[dataset] = reason
                    status_path = formal_session_status_path(root, session_id)
                    status = load_json(status_path, "formal session status")
                    dataset_failures = status.setdefault("dataset_failures", {})
                    if not isinstance(dataset_failures, dict):
                        raise ChainError("formal dataset failure history is malformed")
                    dataset_failures[dataset] = reason
                    status["stage_status"] = status.get("stage_status", {})
                    if isinstance(status["stage_status"], dict):
                        status["stage_status"][f"cell:{dataset}"] = "FAILED"
                    status["updated_unix_ns"] = time.time_ns()
                    atomic_write_json(status_path, status, root)
                    print(f"ZKGPT_FORMAL_CELL_FAILED dataset={dataset} error={reason}", file=sys.stderr, flush=True)
            _set_status(root, session_id, "RUNNING", "result-aggregation")
            result_path, result = _finalize_phase(root, session_id, session_dir, dataset_scope, cells, failures)
            if dataset_scope != DATASETS:
                terminal = "SUBSET_FAILED" if failures else "SUBSET_COMPLETE"
                _set_status(root, session_id, terminal, terminal.lower())
                print(f"ZKGPT_FORMAL_{terminal} session_id={session_id} datasets={','.join(dataset_scope)} result={relative_path(root, result_path, 'subset result')} full_five_complete=false", flush=True)
                return 2 if failures else 0
            if failures:
                _set_status(root, session_id, "FAILED", "failed", "; ".join(f"{key}: {value}" for key, value in failures.items()))
                print(f"ZKGPT_FORMAL_FAILED session_id={session_id} result={relative_path(root, result_path, 'formal result')} failed={','.join(failures)}", file=sys.stderr, flush=True)
                return 2
            if result.get("status") != "VALIDATED" or result.get("cells_validated") != len(DATASETS):
                raise ChainError("formal result aggregation did not validate exactly five successful datasets")
            _set_status(root, session_id, "COMPLETE", "complete")
            print(f"ZKGPT_FORMAL_COMPLETE session_id={session_id} final_validation={relative_path(root, result_path, 'formal result')}", flush=True)
            return 0
        except KeyboardInterrupt:
            if session_id:
                _set_status(root, session_id, "INTERRUPTED", "interrupted", "formal owner received an interrupt")
            raise
        except Exception as exc:
            if session_id:
                _set_status(root, session_id, "FAILED", "failed", str(exc))
            raise


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS)
    args = parser.parse_args(list(argv))
    try:
        return _run_official(baseline_root(), canonical_dataset_scope(args.datasets))
    except KeyboardInterrupt:
        print("ZKGPT_FORMAL_INTERRUPTED: active native stage group stopped", file=sys.stderr, flush=True)
        return 130
    except Exception as exc:
        print(f"ZKGPT_FORMAL_FAILED: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
