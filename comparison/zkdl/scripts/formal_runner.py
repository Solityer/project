#!/usr/bin/env python3
"""Own the baseline-local five-dataset zkDL formal lifecycle."""
from __future__ import annotations

import json
import os
import re
import secrets
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import formal_phase

ROOT = Path(__file__).resolve().parent.parent
DATASETS = formal_phase.DATASETS
DETACHED_LAUNCH_ENV = "ZKDL_DETACHED_LAUNCH_DIR"


class RunnerError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, document: dict[str, Any]) -> None:
    if path.is_symlink() or path.parent.is_symlink() or not path.parent.is_dir():
        raise RunnerError(f"refusing to write local runner state through a symlink or missing directory: {path}")
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
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


def _write_detached_session_link(session: Path, session_id: str) -> None:
    raw = os.environ.get(DETACHED_LAUNCH_ENV)
    if not raw:
        return
    launch_dir = Path(raw)
    if not launch_dir.is_absolute() or launch_dir.is_symlink() or not launch_dir.is_dir():
        raise RunnerError("detached launch directory identity is missing or unsafe")
    resolved = launch_dir.resolve(strict=True)
    if resolved.parent != (ROOT / "runs").resolve(strict=True) or not resolved.name.startswith("launch_"):
        raise RunnerError("detached launch directory is outside baseline-local runs")
    _write_json(resolved / "formal_session.json", {
        "schema": "zkdl_detached_formal_session_v2", "launch_id": resolved.name,
        "session_id": session_id, "session_dir": str(session),
    })


def _build_and_publish_session(runs: Path, session_id: str) -> Path:
    """Build the formal binaries once and publish their log with a new session."""
    if re.fullmatch(r"[0-9a-f]{32}", session_id) is None:
        raise RunnerError("formal build stage requires a canonical session id")
    if runs.is_symlink() or not runs.is_dir() or runs.resolve(strict=True) != ROOT / "runs":
        raise RunnerError(f"zkDL runs directory is unavailable: {runs}")
    stage = runs / f".build_stage_{session_id}"
    session = runs / f"formal_{session_id}"
    if stage.exists() or session.exists():
        raise RunnerError("formal build stage or session path already exists")
    stage.mkdir(mode=0o700)
    (stage / "logs").mkdir(mode=0o700)
    build_log = stage / "logs" / "build.log"
    build_env = dict(os.environ)
    try:
        with build_log.open("xb", buffering=0) as log:
            subprocess.run(["make", "baseline-local-build"], cwd=ROOT, env=build_env,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
            log.flush()
            os.fsync(log.fileno())
        for name in ("gat_runner", "gat_resource_setup"):
            binary = ROOT / name
            if not binary.is_file():
                raise RunnerError(f"formal build did not produce {binary}")
    except subprocess.CalledProcessError as exc:
        raise RunnerError(f"baseline-local build failed with exit {exc.returncode}; see {build_log}") from exc
    os.rename(stage, session)
    return session

def _record_dataset_failure(session: Path, state: dict[str, Any], dataset: str, stage: str,
                            failure: str, *, status: str = "FAILED", replace: bool = False) -> None:
    current = state.setdefault("dataset_status", {}).get(dataset, {})
    if not replace and current.get("status") in ("FAILED", "BLOCKED"):
        return
    row = {"dataset": dataset, "status": status, "stage": stage, "failure": failure[:2048]}
    state["dataset_status"][dataset] = row
    phase_dir = session / "phase"
    phase_dir.mkdir(exist_ok=True)
    _write_json(phase_dir / f"{dataset}_failure.json", {
        "schema": "zkdl_formal_dataset_failure_v1", "baseline": "zkdl",
        "session_id": state["session_id"], **row, "recorded_at_utc": _now(),
    })
    log_path = session / "logs" / f"{dataset}.log"
    if log_path.parent.is_dir():
        marker = json.dumps({"dataset": dataset, "status": status, "stage": stage,
                             "failure": failure[:2048]}, sort_keys=True, separators=(",", ":"))
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(marker + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    state["updated_at_utc"] = _now()
    _write_json(session / "state.json", state)


def _phase_marker(output: str, phase: str, dataset: str | None, session_id: str) -> dict[str, Any]:
    marker = None
    for line in reversed(output.splitlines()):
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and candidate.get("zkdl_phase") == phase:
            marker = candidate
            break
    allowed_statuses = {"PASS", "PARTIAL", "FAILED"} if phase == "prepare" else ({"PASS", "FAILED"} if phase == "finalize" else {"PASS"})
    if not isinstance(marker, dict) or marker.get("status") not in allowed_statuses:
        raise RunnerError(f"zkDL {phase} phase exited without its exact outcome marker")
    if phase in ("prepare", "finalize") and marker.get("session_id") != session_id:
        raise RunnerError(f"zkDL {phase} phase returned a mismatched session identity")
    if phase == "cell" and marker.get("dataset") != dataset:
        raise RunnerError(f"zkDL cell phase returned a mismatched dataset identity")
    return marker


def _marker_summary(phase: str, marker: dict[str, Any]) -> dict[str, Any]:
    keys = ("dataset", "prepare_receipt_path", "receipt_path", "final_validation_path",
            "proof_path", "proof_size_bytes", "verification_path")
    summary: dict[str, Any] = {"phase": phase, "status": marker.get("status")}
    summary.update({key: marker[key] for key in keys if key in marker})
    return summary


def _effective_host_memory_available() -> tuple[int, str]:
    system_available = 0
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            if line.startswith("MemAvailable:"):
                system_available = int(line.split()[1]) * 1024
                break
    except (OSError, ValueError, IndexError):
        pass
    if system_available <= 0:
        raise RunnerError("cannot determine live host MemAvailable")
    effective = system_available
    method = "system_memavailable"
    try:
        cgroup_lines = Path("/proc/self/cgroup").read_text(encoding="ascii").splitlines()
    except OSError:
        cgroup_lines = []
    v2_path = next((line.split(":", 2)[2] for line in cgroup_lines if line.startswith("0::")), None)
    if v2_path is not None:
        cgroup_root = Path("/sys/fs/cgroup")
        base = cgroup_root / v2_path.lstrip("/")
        for candidate in (base, *base.parents):
            try:
                candidate.relative_to(cgroup_root)
            except ValueError:
                continue
            try:
                limit_text = (candidate / "memory.max").read_text(encoding="ascii").strip()
                if limit_text == "max":
                    continue
                limit = int(limit_text)
                current = int((candidate / "memory.current").read_text(encoding="ascii").strip())
            except (OSError, ValueError):
                continue
            if limit > 0:
                effective = min(effective, max(0, limit - current))
                method = "min_system_and_cgroup_v2_available"
    else:
        memory_group = next((line.split(":", 2)[2] for line in cgroup_lines if ":memory:" in line), None)
        if memory_group is not None:
            cgroup_root = Path("/sys/fs/cgroup/memory")
            base = cgroup_root / memory_group.lstrip("/")
            for candidate in (base, *base.parents):
                try:
                    candidate.relative_to(cgroup_root)
                except ValueError:
                    continue
                try:
                    limit = int((candidate / "memory.limit_in_bytes").read_text(encoding="ascii").strip())
                    current = int((candidate / "memory.usage_in_bytes").read_text(encoding="ascii").strip())
                except (OSError, ValueError):
                    continue
                if 0 < limit < (1 << 60):
                    effective = min(effective, max(0, limit - current))
                    method = "min_system_and_cgroup_v1_available"
    return effective, method


def _filesystem_device(path: Path, *, allow_missing: bool = False) -> int:
    cursor = path
    if path.is_symlink():
        raise RunnerError(f"zkDL formal filesystem path is a symlink: {path}")
    while not cursor.exists():
        if cursor == cursor.parent:
            raise RunnerError(f"cannot find an existing parent filesystem for {path}")
        cursor = cursor.parent
    if cursor.is_symlink():
        raise RunnerError(f"zkDL formal filesystem ancestor is a symlink: {cursor}")
    if not cursor.is_dir():
        if allow_missing:
            raise RunnerError(f"zkDL formal output filesystem parent is not a directory: {cursor}")
        raise RunnerError(f"zkDL formal filesystem path is not a directory: {cursor}")
    try:
        return cursor.stat(follow_symlinks=False).st_dev
    except OSError as exc:
        raise RunnerError(f"cannot inspect zkDL formal filesystem path {cursor}: {exc}") from exc


def _checked_capacity_int(value: Any, label: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if type(value) is not int or value < minimum or value > (1 << 64) - 1:
        raise RunnerError(f"zkDL capacity profile has an invalid {label}")
    return value


def _remaining_capacity_requirements(profile: dict[str, Any], completed: tuple[str, ...],
                                    pending: tuple[str, ...] | None = None) -> dict[str, int]:
    legacy_cpu = profile.get("schema") == "zkdl_formal_capacity_profile_v3"
    if profile.get("schema") not in ("zkdl_formal_capacity_profile_v3", "zkdl_formal_capacity_profile_v4") or profile.get("baseline") != "zkdl":
        raise RunnerError("zkDL local capacity profile schema/baseline mismatch")
    device = profile.get("device")
    if legacy_cpu:
        if profile.get("backend") != "cpu" or profile.get("memory_accounting_kind") != "host-only" or device is not None:
            raise RunnerError("legacy zkDL analysis capacity profile is not CPU-only")
    elif (profile.get("backend") != "cpu" or profile.get("gpu_offload_stage") != "hyrax_ipa_fold" or
          profile.get("memory_accounting_kind") != "shared_host_cuda" or not isinstance(device, dict) or
          not isinstance(device.get("name"), str) or not device["name"].strip() or
          type(device.get("ordinal")) is not int or device["ordinal"] < 0 or
          not isinstance(device.get("uuid"), str) or not device["uuid"].strip()):
        raise RunnerError("zkDL capacity profile does not match the source-bound CPU plus Hyrax GPU offload path")
    methods = profile.get("bound_methods")
    if (not isinstance(methods, dict) or
            methods.get("additional_disk_peak") != "source_derived_upper_bound" or
            methods.get("host_peak") != "source_derived_upper_bound" or
            methods.get("finalization_host_peak") != "srs_manifest_schema_python_heap_upper_bound"):
        raise RunnerError("zkDL capacity profile lacks source-derived host/disk bounds")
    if legacy_cpu:
        if "gpu_workspace" in methods or "shared_host_cuda" in methods:
            raise RunnerError("legacy CPU-only capacity profile contains GPU accounting fields")
    else:
        # NVIDIA GB10 exposes the host and GPU through one UMA DRAM pool.
        # Its CUDA free-memory view omits reclaimable host page cache, so use
        # the OS/cgroup pool budget for the combined host+workspace bound and
        # reserve CUDA free-memory checks for actual cudaMalloc workspace.
        uses_shared_host_pool = device["name"] == "NVIDIA GB10"
        expected_shared_method = (
            "shared_uma_host_available_ge_host_plus_gpu_workspace_cuda_free_ge_workspace"
            if uses_shared_host_pool else
            "shared_pool_min_host_cuda_ge_host_plus_gpu_workspace")
        if (methods.get("gpu_workspace") != "authenticated_basis_144B_plus_max_opening_704P_plus_352" or
                methods.get("shared_host_cuda") != expected_shared_method):
            raise RunnerError("zkDL capacity profile lacks source-derived shared host/CUDA bounds")
    datasets = profile.get("datasets")
    if not isinstance(datasets, dict) or not datasets or not set(datasets).issubset(DATASETS):
        raise RunnerError("zkDL capacity profile does not contain a canonical admitted subset")
    admitted_order = tuple(ds for ds in DATASETS if ds in datasets)
    if any(dataset not in datasets for dataset in completed):
        raise RunnerError("zkDL completed cells are absent from the admitted capacity profile")
    if tuple(ds for ds in DATASETS if ds in completed) != completed:
        raise RunnerError("zkDL completed cell order is not canonical")
    remaining = tuple(ds for ds in admitted_order if ds not in completed) if pending is None else pending
    if any(ds not in datasets for ds in remaining) or tuple(ds for ds in DATASETS if ds in remaining) != remaining:
        raise RunnerError("zkDL capacity recheck is not following the canonical admitted dataset sequence")
    finalization = _checked_capacity_int(profile.get("finalization_disk_bytes_upper_bound"),
                                         "finalization disk upper bound", allow_zero=True)
    host_required = 0
    device_required = 0
    disk_required = finalization
    future_retained_prefix = 0
    for dataset in remaining:
        item = datasets.get(dataset)
        if not isinstance(item, dict):
            raise RunnerError(f"zkDL capacity profile row is malformed for {dataset}")
        if item.get("phase_coverage") != ["setup", "witness", "commitment", "prove", "verify"]:
            raise RunnerError(f"zkDL capacity profile phase coverage is malformed for {dataset}")
        route_source_domain = _checked_capacity_int(
            item.get("native_route_source_domain"), f"{dataset} route selector source domain")
        route_factor_elements = _checked_capacity_int(
            item.get("native_route_selector_factor_elements"), f"{dataset} route selector factor elements")
        if route_source_domain > (1 << 30) or route_factor_elements > route_source_domain + 1:
            raise RunnerError(f"zkDL route selector factor bound is invalid for {dataset}")
        host_peak = _checked_capacity_int(item.get("host_peak_bytes"), f"{dataset} host peak")
        if legacy_cpu:
            device_peak = _checked_capacity_int(item.get("device_peak_bytes"), f"{dataset} legacy CPU device peak", allow_zero=True)
            if device_peak != 0:
                raise RunnerError(f"legacy CPU-only capacity profile unexpectedly requires device memory for {dataset}")
            host_required = max(host_required, host_peak)
        else:
            basis = _checked_capacity_int(item.get("gpu_authenticated_basis_size"), f"{dataset} authenticated GPU basis")
            padded = _checked_capacity_int(item.get("gpu_max_opening_padded_generator_size"), f"{dataset} maximum GPU opening")
            workspace = _checked_capacity_int(item.get("gpu_workspace_required_bytes"), f"{dataset} GPU workspace")
            maximum_padded = 1 << basis.bit_length()
            expected_workspace = 144 * basis + 704 * padded + 352
            if (basis != (1 << 18) or padded > maximum_padded or padded & (padded - 1) or
                    expected_workspace > (1 << 64) - 1 or workspace != expected_workspace or
                    item.get("device_peak_bytes") != workspace):
                raise RunnerError(f"zkDL GPU workspace bound is not source-derived for {dataset}")
            prepared_free = _checked_capacity_int(item.get("gpu_cuda_free_bytes_at_prepare"),
                                                  f"{dataset} prepare CUDA free bytes", allow_zero=True)
            prepared_total = _checked_capacity_int(item.get("gpu_cuda_total_bytes_at_prepare"),
                                                   f"{dataset} prepare CUDA total bytes")
            if prepared_free < workspace or prepared_total < prepared_free:
                raise RunnerError(f"zkDL GPU workspace exceeds prepared CUDA capacity for {dataset}")
            host_with_gpu = host_peak + workspace
            if host_with_gpu > (1 << 64) - 1:
                raise RunnerError(f"zkDL shared host/CUDA memory bound overflows for {dataset}")
            host_required = max(host_required, host_with_gpu)
            device_peak = workspace
        device_required = max(device_required, device_peak)
        retained = _checked_capacity_int(item.get("cell_retained_artifact_bytes_upper_bound"),
                                         f"{dataset} retained artifact upper bound")
        additional = _checked_capacity_int(item.get("additional_disk_peak_bytes"),
                                           f"{dataset} additional disk peak")
        if additional < retained:
            raise RunnerError(f"zkDL capacity profile disk peak is smaller than retained output for {dataset}")
        candidate_disk = future_retained_prefix + additional + finalization
        if candidate_disk > (1 << 64) - 1:
            raise RunnerError("zkDL remaining disk requirement overflows u64")
        disk_required = max(disk_required, candidate_disk)
        future_retained_prefix += retained
        if future_retained_prefix > (1 << 64) - 1:
            raise RunnerError("zkDL retained prefix requirement overflows u64")
    disk_required = max(disk_required, future_retained_prefix + finalization)
    finalization_profile = profile.get("finalization_manifest_capacity")
    if (not isinstance(finalization_profile, dict) or
            finalization_profile.get("schema") != "zkdl_srs_manifest_finalization_heap_v1" or
            not isinstance(finalization_profile.get("datasets"), dict) or
            set(finalization_profile["datasets"]) != set(datasets)):
        raise RunnerError("zkDL finalization manifest capacity schema/admitted dataset set mismatch")
    finalization_dataset_bounds: list[int] = []
    for dataset in admitted_order:
        item = finalization_profile["datasets"][dataset]
        if not isinstance(item, dict):
            raise RunnerError(f"zkDL finalization manifest capacity is malformed for {dataset}")
        family_count = _checked_capacity_int(item.get("family_count"), f"{dataset} SRS family count")
        segment_count = _checked_capacity_int(item.get("segment_count"), f"{dataset} SRS segment count")
        try:
            expected = formal_phase._srs_manifest_finalization_capacity(family_count, segment_count)
        except formal_phase.PhaseError as exc:
            raise RunnerError(f"zkDL finalization manifest bound is invalid for {dataset}: {exc}") from exc
        if item != expected:
            raise RunnerError(f"zkDL finalization manifest bound does not match its fixed-schema counts for {dataset}")
        finalization_dataset_bounds.append(expected["host_peak_bytes_upper_bound"])
    finalization_host = _checked_capacity_int(profile.get("finalization_host_peak_bytes_upper_bound"),
                                              "finalization host upper bound")
    if finalization_dataset_bounds and finalization_host != max(finalization_dataset_bounds):
        raise RunnerError("zkDL finalization host bound differs from its SRS manifest schema bounds")
    if not finalization_dataset_bounds and finalization_host != 0:
        raise RunnerError("zkDL empty capacity subset has a nonzero finalization host bound")
    return {"host_bytes": host_required, "disk_bytes": disk_required,
            "device_bytes": device_required, "finalization_host_bytes": finalization_host}

def _prepare_capacity(marker: dict[str, Any], session: Path, session_id: str) -> dict[str, Any]:
    receipt_path = session / "phase" / "prepare.json"
    if not receipt_path.is_file():
        raise RunnerError("zkDL prepare receipt is missing")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerError("zkDL prepare receipt is unreadable") from exc
    if (not isinstance(receipt, dict) or receipt.get("schema") != "zkDL_formal_prepare_receipt_v3" or
            receipt.get("baseline") != "zkdl" or receipt.get("status") not in ("PASS", "PARTIAL") or
            marker.get("status") != receipt.get("status") or receipt.get("session_id") != session_id):
        raise RunnerError("zkDL prepare receipt status mismatch")
    dataset_status = receipt.get("dataset_status")
    if not isinstance(dataset_status, dict) or set(dataset_status) != set(DATASETS):
        raise RunnerError("zkDL prepare outcome does not enumerate the five datasets")
    passing = tuple(dataset for dataset in DATASETS
                    if isinstance(dataset_status.get(dataset), dict) and dataset_status[dataset].get("status") == "PASS")
    if not passing:
        raise RunnerError("zkDL prepare receipt has no dataset that passed resource admission")
    preflight = receipt.get("preflight_report")
    datasets = preflight.get("datasets") if isinstance(preflight, dict) else None
    if not isinstance(datasets, dict) or not set(passing).issubset(datasets):
        raise RunnerError("zkDL prepare receipt lacks per-dataset resource plans")
    for dataset in passing:
        row = datasets[dataset]
        if (not isinstance(row, dict) or row.get("dataset") != dataset or row.get("status") != "PASS" or
                not isinstance(row.get("resource_prerequisites"), dict)):
            raise RunnerError(f"zkDL resource plan is missing for {dataset}")
    profile = receipt.get("capacity_profile")
    if (not isinstance(profile, dict) or profile.get("session_id") != session_id or
            profile.get("baseline") != "zkdl" or profile.get("schema") != "zkdl_formal_capacity_profile_v4" or
            profile.get("gpu_offload_stage") != "hyrax_ipa_fold"):
        raise RunnerError("zkDL prepare receipt lacks the required capacity profile")
    native_admissions = receipt.get("native_admissions")
    profile_datasets = profile.get("datasets")
    if (not isinstance(native_admissions, dict) or set(native_admissions) != set(passing) or
            not isinstance(profile_datasets, dict) or set(profile_datasets) != set(passing)):
        raise RunnerError("zkDL prepare receipt/profile does not match the admitted dataset subset")
    for dataset in passing:
        admission = native_admissions[dataset]
        row = profile_datasets[dataset]
        if (not isinstance(admission, dict) or admission.get("schema") != "zkdl_native_relation_admission_v2" or
                admission.get("status") != "PASS" or admission.get("dataset") != dataset or not isinstance(row, dict)):
            raise RunnerError(f"zkDL native resource admission failed for {dataset}")
        source_domain = _checked_capacity_int(admission.get("native_maximum_route_source_domain"),
                                              f"{dataset} native route source domain")
        factor_elements = _checked_capacity_int(admission.get("native_maximum_route_selector_factor_elements"),
                                                f"{dataset} native route selector factor elements")
        histogram = admission.get("native_hyrax_opening_segment_domain_histogram")
        resource = datasets[dataset]["resource_prerequisites"]
        srs_plan = resource.get("srs_plan", {})
        if (source_domain == 0 or source_domain > (1 << 30) or factor_elements == 0 or
                factor_elements > source_domain + 1 or row.get("native_route_source_domain") != source_domain or
                row.get("native_route_selector_factor_elements") != factor_elements or
                not isinstance(histogram, list) or not histogram or
                row.get("gpu_authenticated_basis_size") != srs_plan.get("basis_domain") or
                row.get("gpu_max_opening_padded_generator_size") !=
                max(bucket.get("padded_generator_size", 0) for bucket in histogram if isinstance(bucket, dict))):
            raise RunnerError(f"zkDL capacity profile differs from the actual resource plan for {dataset}")
    if type(profile.get("fs_device_id")) is not int or profile["fs_device_id"] != session.stat(follow_symlinks=False).st_dev:
        raise RunnerError("zkDL capacity profile filesystem does not match the session")
    _remaining_capacity_requirements(profile, (), ())
    return profile

def _bounded_subprocess_text(value: Any, limit: int = 4096) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if not isinstance(value, str) or not value:
        return ""
    if len(value) > limit:
        return f"{value[:limit]}...[truncated {len(value) - limit} characters]"
    return value


def _gpu_offload_device_info(session: Path, dataset: str, expected_device: dict[str, Any]) -> dict[str, int | str]:
    config = formal_phase.ROOT / "configs" / f"{session.name}_{dataset}.cfg"
    binary = ROOT / "gat_runner"
    result = None
    try:
        result = subprocess.run([str(binary), "gpu-offload-device-info", "--config", str(config)], cwd=ROOT,
                                capture_output=True, text=True, errors="replace", timeout=15, check=True)
        candidate = next((json.loads(line) for line in reversed(result.stdout.splitlines())
                          if line.lstrip().startswith("{")), None)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        stderr = _bounded_subprocess_text(getattr(exc, "stderr", None))
        stdout = _bounded_subprocess_text(getattr(exc, "stdout", None))
        if not stdout and result is not None:
            stdout = _bounded_subprocess_text(result.stdout)
        captured = []
        if stderr:
            captured.append(f"stderr={stderr}")
        if stdout:
            captured.append(f"stdout={stdout}")
        detail = f"; captured {'; '.join(captured)}" if captured else ""
        raise RunnerError(f"cannot refresh live zkDL CUDA device capacity: {exc}{detail}") from exc
    if (not isinstance(candidate, dict) or candidate.get("schema") != "zkdl_gpu_offload_device_info_v1" or
            candidate.get("status") != "PASS" or candidate.get("gpu_device_name") != expected_device["name"] or
            candidate.get("gpu_device_ordinal") != expected_device["ordinal"] or
            candidate.get("gpu_device_uuid") != expected_device["uuid"]):
        raise RunnerError("live zkDL CUDA device identity differs from the admitted device")
    free = _checked_capacity_int(candidate.get("gpu_cuda_free_bytes"), "live CUDA free bytes", allow_zero=True)
    total = _checked_capacity_int(candidate.get("gpu_cuda_total_bytes"), "live CUDA total bytes")
    if free > total:
        raise RunnerError("live zkDL CUDA free memory exceeds total device capacity")
    return {"name": candidate["gpu_device_name"], "ordinal": candidate["gpu_device_ordinal"],
            "uuid": candidate["gpu_device_uuid"], "free_bytes": free, "total_bytes": total}

def _capacity_gate(profile: dict[str, Any], completed: tuple[str, ...], session: Path,
                   pending: tuple[str, ...] | None = None) -> dict[str, Any]:
    requirements = _remaining_capacity_requirements(profile, completed, pending)
    expected_device = _checked_capacity_int(profile.get("fs_device_id"), "filesystem device id", allow_zero=True)
    for label, path, allow_missing in (
            ("runs", ROOT / "runs", False), ("session", session, False),
            ("results", ROOT / "results", True)):
        device = _filesystem_device(path, allow_missing=allow_missing)
        if device != expected_device:
            raise RunnerError(f"zkDL {label} path is on a different filesystem from the admitted session")
    try:
        statvfs = os.statvfs(session)
        disk_available = statvfs.f_bavail * statvfs.f_frsize
    except OSError as exc:
        raise RunnerError(f"cannot inspect live zkDL session filesystem capacity: {exc}") from exc
    host_available, host_method = _effective_host_memory_available()
    required_host = max(requirements["host_bytes"], requirements["finalization_host_bytes"])
    if host_available < required_host:
        raise RunnerError(f"zkDL host capacity blocked before stage: need={required_host} available={host_available} method={host_method}")
    if disk_available < requirements["disk_bytes"]:
        raise RunnerError(f"zkDL disk capacity blocked before stage: need={requirements['disk_bytes']} available={disk_available}")
    live_device = None
    active = tuple(pending) if pending is not None else tuple(dataset for dataset in DATASETS
                                                             if dataset in profile["datasets"] and dataset not in completed)
    uses_shared_host_pool = (profile.get("memory_accounting_kind") == "shared_host_cuda" and
                             isinstance(profile.get("device"), dict) and
                             profile["device"].get("name") == "NVIDIA GB10")
    if active:
        live_device = _gpu_offload_device_info(session, active[0], profile["device"])
        if requirements["device_bytes"] > live_device["free_bytes"]:
            raise RunnerError("zkDL GPU workspace blocked before stage: "
                              f"need={requirements['device_bytes']} available={live_device['free_bytes']}")
        if not uses_shared_host_pool:
            shared_available = min(host_available, live_device["free_bytes"])
            shared_required = max(required_host, requirements["finalization_host_bytes"])
            if shared_available < shared_required:
                raise RunnerError(f"zkDL shared host/CUDA memory blocked before stage: need={shared_required} "
                                  f"available={shared_available}")
    if profile.get("backend") != "cpu" or profile.get("gpu_offload_stage") != "hyrax_ipa_fold" or not isinstance(profile.get("device"), dict):
        raise RunnerError("zkDL live device admission disagrees with the source-bound CPU plus Hyrax GPU path")
    return {
        "completed_datasets": list(completed),
        "pending_datasets": list(pending) if pending is not None else None,
        "required_host_bytes": required_host,
        "finalization_host_peak_bytes_upper_bound": requirements["finalization_host_bytes"],
        "available_host_bytes": host_available,
        "host_availability_method": host_method,
        "required_disk_bytes": requirements["disk_bytes"],
        "available_disk_bytes": disk_available,
        "required_device_bytes": requirements["device_bytes"],
        "available_device_bytes": live_device["free_bytes"] if live_device else 0,
        "available_shared_memory_bytes": (host_available if uses_shared_host_pool or live_device is None
                                          else min(host_available, live_device["free_bytes"])),
        "gpu_offload_stage": profile["gpu_offload_stage"],
        "gpu_device": profile["device"],
        "memory_accounting_kind": profile["memory_accounting_kind"],
        "backend": "cpu",
        "checked_at_utc": _now(),
    }


def _run_phase(session: Path, session_id: str, state: dict[str, Any],
               phase: str, dataset: str | None = None) -> dict[str, Any]:
    stage = f"cell:{dataset}" if phase == "cell" else phase
    state["stage"] = stage
    state["updated_at_utc"] = _now()
    _write_json(session / "state.json", state)
    command = [sys.executable, str(ROOT / "scripts" / "formal_phase.py"), "--internal-phase", phase,
               "--session-dir", str(session)]
    if dataset is not None:
        command.extend(("--dataset", dataset))
    env = dict(os.environ)
    print(f"zkdl_formal_stage=START stage={stage} session={session_id}", flush=True)
    try:
        output = formal_phase._run(command, env=env, capture_output=True, capture_mode="phase_marker")
    except formal_phase.PhaseError as exc:
        if dataset is not None:
            formal_phase._append_log(session / "logs" / f"{dataset}.log",
                                     json.dumps({"phase": phase, "dataset": dataset, "status": "FAILED",
                                                 "error": str(exc)}, sort_keys=True, separators=(",", ":")))
        raise RunnerError(str(exc)) from exc
    marker = _phase_marker(output, phase, dataset, session_id)
    state.setdefault("completed_phases", []).append(_marker_summary(phase, marker))
    if phase == "prepare" and isinstance(marker.get("dataset_status"), dict):
        for item_dataset, outcome in marker["dataset_status"].items():
            if item_dataset in state.get("dataset_status", {}) and isinstance(outcome, dict):
                normalized = dict(outcome)
                if normalized.get("status") == "PASS":
                    normalized["status"] = "READY"
                    normalized["stage"] = "admitted"
                state["dataset_status"][item_dataset].update(normalized)
    elif phase == "finalize" and isinstance(marker.get("dataset_status"), dict):
        for item_dataset, outcome in marker["dataset_status"].items():
            if item_dataset in state.get("dataset_status", {}) and isinstance(outcome, dict):
                state["dataset_status"][item_dataset].update(outcome)
    elif phase == "cell" and dataset is not None and marker.get("status") == "PASS":
        state["dataset_status"][dataset].update({"status": "PASS", "stage": "complete", "failure": ""})
    state["updated_at_utc"] = _now()
    _write_json(session / "state.json", state)
    print(f"zkdl_formal_stage={marker['status']} stage={stage} session={session_id}", flush=True)
    return marker


def _validate_final_pass(marker: dict[str, Any], session: Path, session_id: str) -> str:
    expected_receipt = f"results/{session.name}/final_validation.json"
    expected_collection = f"results/{session.name}/collected.json"
    if marker.get("final_validation_path") != expected_receipt or marker.get("collection_path") != expected_collection:
        raise RunnerError("zkDL final results point outside this session result directory")
    receipt_path = ROOT / expected_receipt
    collection_path = ROOT / expected_collection
    if not receipt_path.is_file() or not collection_path.is_file():
        raise RunnerError("zkDL final validation receipt or collection is missing")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    collection = json.loads(collection_path.read_text(encoding="utf-8"))
    strict_marker = "final_validation=PASS records=5 datasets=5 independent_verify=5"
    rows = collection.get("rows", [])
    if (receipt.get("schema") != "zkDL_formal_final_validation_receipt_v1" or
            receipt.get("status") != "PASS" or receipt.get("session_id") != session_id or
            receipt.get("verified_cells") != list(DATASETS) or strict_marker not in receipt.get("validator_output", []) or
            receipt.get("collection_path") != expected_collection or
            collection.get("schema") != "zkDL_gat_result_collection_v3" or
            collection.get("session_id") != session.name or not isinstance(rows, list) or
            {row.get("dataset") for row in rows if isinstance(row, dict)} != set(DATASETS) or
            len(rows) != len(DATASETS)):
        raise RunnerError("zkDL final results do not contain five independently verified datasets")
    return expected_receipt

def _write_interrupted_summary(session: Path, state: dict[str, Any], failure: str) -> None:
    session_id = state["session_id"]
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        row = state["dataset_status"][dataset]
        receipt_path = session / "phase" / f"{dataset}.json"
        if receipt_path.is_file() and not receipt_path.is_symlink():
            try:
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                if (receipt.get("schema") == "zkDL_formal_cell_receipt_v2" and
                        receipt.get("session_id") == session_id and receipt.get("dataset") == dataset and
                        receipt.get("status") == "PASS" and receipt.get("independent_verification") is True):
                    rows.append({"dataset": dataset, "status": "PASS", "stage": "independently_verified",
                                 "result": {key: receipt.get(key) for key in (
                                     "statement_id", "proof_path", "proof_size_bytes", "verification_path",
                                     "commitment_time_seconds", "prove_time_seconds", "verify_time_seconds", "metrics")},
                                 "cell_receipt_path": receipt_path.relative_to(ROOT).as_posix()})
                    row.update({"status": "PASS", "stage": "independently_verified", "failure": ""})
                    failure_path = session / "phase" / f"{dataset}_failure.json"
                    if failure_path.is_file() and not failure_path.is_symlink():
                        failure_path.unlink()
                    continue
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass
        current_status = row.get("status")
        if current_status in ("PENDING", "READY"):
            row.update({"status": "BLOCKED", "stage": "interrupted",
                        "failure": "not started because the formal run was interrupted"})
        rows.append({"dataset": dataset, "status": row.get("status", "BLOCKED"),
                     "stage": row.get("stage", "unknown"),
                     "failure": row.get("failure") or failure})
    results_dir = ROOT / "results" / session.name
    if results_dir.exists() and any(results_dir.iterdir()):
        return
    results_dir.mkdir(parents=True, exist_ok=True)
    collection_path = results_dir / "collected.json"
    summary = {"schema": "zkDL_gat_formal_partial_result_summary_v1", "mode": "formal_interrupted",
               "baseline": "zkdl", "session_id": session_id, "runs_path": session.relative_to(ROOT).as_posix(),
               "overall_status": "INTERRUPTED", "strict_pass": False, "failure": failure[:2048],
               "datasets": rows}
    _write_json(collection_path, summary)
    validation = {"schema": "zkDL_formal_final_validation_receipt_v1", "baseline": "zkdl",
                  "session_id": session_id, "status": "INTERRUPTED", "strict_pass": False,
                  "failure": failure[:2048], "collection_path": collection_path.relative_to(ROOT).as_posix(),
                  "dataset_status": {row["dataset"]: row for row in rows}}
    _write_json(results_dir / "final_validation.json", validation)


def run() -> int:
    runs = ROOT / "runs"
    if runs.is_symlink():
        raise RunnerError(f"zkDL runs directory is unsafe: {runs}")
    if not runs.exists():
        runs.mkdir(mode=0o700)
    if not runs.is_dir():
        raise RunnerError(f"zkDL runs path is not a baseline-local directory: {runs}")
    session: Path | None = None
    state: dict[str, Any] | None = None
    formal_phase.SIGNAL_GRACE_SECONDS = 10.0
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, formal_phase._signal_handler)
    try:
        session_id = secrets.token_hex(16)
        session = runs / f"formal_{session_id}"
        print(f"zkdl_formal_stage=START stage=baseline_local_build session={session_id}", flush=True)
        session = _build_and_publish_session(runs, session_id)
        print(f"zkdl_formal_stage=PASS stage=baseline_local_build session={session_id}", flush=True)
        state = {
            "schema": "zkdl_formal_runner_state_v1", "baseline": "zkdl", "session_id": session_id,
            "session_dir": session.relative_to(ROOT).as_posix(), "status": "RUNNING", "stage": "prepare",
            "started_at_utc": _now(), "updated_at_utc": _now(), "completed_phases": [],
            "dataset_status": {dataset: {"dataset": dataset, "status": "PENDING", "stage": "pending", "failure": ""}
                               for dataset in DATASETS},
        }
        _write_json(session / "state.json", state)
        _write_detached_session_link(session, session_id)
        global_failure = ""
        profile: dict[str, Any] | None = None
        prepare_marker: dict[str, Any] | None = None
        try:
            prepare_marker = _run_phase(session, session_id, state, "prepare")
            if prepare_marker.get("status") == "FAILED":
                global_failure = prepare_marker.get("failure") or "shared preparation failed before dataset admission"
            else:
                try:
                    profile = _prepare_capacity(prepare_marker, session, session_id)
                except Exception as exc:
                    global_failure = f"shared resource capacity validation failed: {type(exc).__name__}: {exc}"
                    for dataset in DATASETS:
                        if state["dataset_status"][dataset].get("status") == "READY":
                            _record_dataset_failure(session, state, dataset, "prepare_identity", global_failure,
                                                    status="BLOCKED")
                if profile is not None:
                    state["capacity"] = {
                        "profile_schema": profile["schema"], "backend": profile["backend"],
                        "gpu_offload_stage": profile["gpu_offload_stage"], "gpu_device": profile["device"],
                        "completed_cells": [],
                    }
                    _write_json(session / "state.json", state)
                    completed: list[str] = []
                    for dataset in DATASETS:
                        if state["dataset_status"][dataset].get("status") != "READY":
                            continue
                        if dataset not in profile["datasets"]:
                            _record_dataset_failure(session, state, dataset, "capacity_profile",
                                                    "dataset was admitted without its exact capacity profile", status="BLOCKED")
                            continue
                        try:
                            capacity_check = _capacity_gate(profile, tuple(completed), session, pending=(dataset,))
                        except Exception as exc:
                            failure = f"{type(exc).__name__}: {exc}"
                            _record_dataset_failure(session, state, dataset, "capacity_admission", failure)
                            continue
                        state["capacity"]["last_live_check"] = capacity_check
                        _write_json(session / "state.json", state)
                        try:
                            _run_phase(session, session_id, state, "cell", dataset)
                        except Exception as exc:
                            failure = f"{type(exc).__name__}: {exc}"
                            _record_dataset_failure(session, state, dataset, "cell", failure)
                            break
                        completed.append(dataset)
                        capacity_state = state["capacity"]
                        capacity_state["completed_cells"] = list(completed)
                        _write_json(session / "state.json", state)
                    try:
                        final_capacity = _capacity_gate(profile, tuple(completed), session, pending=())
                        state["capacity"]["finalization_live_check"] = final_capacity
                    except Exception as exc:
                        global_failure = f"finalization capacity admission failed: {type(exc).__name__}: {exc}"
                    _write_json(session / "state.json", state)
        except Exception as exc:
            global_failure = global_failure or f"shared preparation/orchestration failure: {type(exc).__name__}: {exc}"
            for dataset in DATASETS:
                if state["dataset_status"][dataset].get("status") in ("PENDING", "READY"):
                    _record_dataset_failure(session, state, dataset, "orchestration", global_failure, status="BLOCKED")

        if global_failure:
            state["global_failure"] = global_failure[:2048]
        state["stage"] = "finalize"
        state["updated_at_utc"] = _now()
        _write_json(session / "state.json", state)
        try:
            final_marker = _run_phase(session, session_id, state, "finalize")
            if final_marker.get("status") == "PASS":
                final_path = _validate_final_pass(final_marker, session, session_id)
                state["final_validation_path"] = final_path
                if global_failure or any(state["dataset_status"][dataset].get("status") != "PASS" for dataset in DATASETS):
                    raise RunnerError("strict final validation reported PASS despite incomplete orchestration state")
                state["status"] = "PASS"
                state["stage"] = "complete"
                state["failure"] = ""
                state["finished_at_utc"] = _now()
                state["updated_at_utc"] = state["finished_at_utc"]
                _write_json(session / "state.json", state)
                print(f"zkdl_formal_run=PASS session={session_id} session_dir={session.relative_to(ROOT).as_posix()}", flush=True)
                return 0
            state["status"] = "FAILED"
            state["stage"] = "complete"
            state["failure"] = final_marker.get("failure") or global_failure or "strict five-dataset final validation did not pass"
            state["final_validation_path"] = final_marker.get("final_validation_path", "")
        except Exception as exc:
            state["status"] = "FAILED"
            state["stage"] = "complete"
            state["failure"] = f"{global_failure + '; ' if global_failure else ''}finalization failed: {type(exc).__name__}: {exc}"
        state["finished_at_utc"] = _now()
        state["updated_at_utc"] = state["finished_at_utc"]
        _write_json(session / "state.json", state)
        print(f"zkdl_formal_run=FAILED session={session_id} failure={state['failure']}", file=sys.stderr, flush=True)
        return 2
    except KeyboardInterrupt as exc:
        if state is not None and session is not None:
            interruption = str(exc) or "formal run interrupted"
            stage = str(state.get("stage", "unknown"))
            active_dataset = stage.split(":", 1)[1] if stage.startswith("cell:") else None
            if active_dataset in DATASETS and state["dataset_status"][active_dataset].get("status") == "READY":
                _record_dataset_failure(session, state, active_dataset, "interrupted", interruption,
                                        status="INTERRUPTED", replace=True)
            for dataset in DATASETS:
                if state["dataset_status"][dataset].get("status") in ("PENDING", "READY"):
                    _record_dataset_failure(session, state, dataset, "interrupted",
                                            "not started because the formal run was interrupted", status="BLOCKED")
            state["global_failure"] = interruption[:2048]
            _write_interrupted_summary(session, state, interruption)
            state["status"] = "INTERRUPTED"
            state["stage"] = "interrupted"
            state["failure"] = interruption
            state["finished_at_utc"] = _now()
            state["updated_at_utc"] = state["finished_at_utc"]
            _write_json(session / "state.json", state)
        print(f"zkdl_formal_run=INTERRUPTED error={exc}", file=sys.stderr, flush=True)
        return 143 if formal_phase.INTERRUPTED in (signal.SIGTERM, signal.SIGHUP) else 130
    except Exception as exc:
        if state is not None and session is not None:
            state["status"] = "FAILED"
            state["failure"] = str(exc)
            state["finished_at_utc"] = _now()
            state["updated_at_utc"] = state["finished_at_utc"]
            _write_json(session / "state.json", state)
        print(f"zkdl_formal_run=FAILED error={exc}", file=sys.stderr, flush=True)
        return 2
def main() -> int:
    if len(sys.argv) != 1:
        print("usage: bash scripts/run_five.sh", file=sys.stderr, flush=True)
        return 2
    try:
        return run()
    except RunnerError as exc:
        print(f"zkdl_formal_run=FAILED error={exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
