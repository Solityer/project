#!/usr/bin/env python3
"""Support the source-matched, session-scoped GAT-ZKML ablation runner."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any


# Loading the canonical validators below must not alter the source tree after
# build provenance has been captured: their import otherwise creates
# scripts/__pycache__ entries that change the inventory hash.
sys.dont_write_bytecode = True


VARIANTS = {"wo_feature_folding", "wo_shared_neighborhood"}
DATASETS = ("cora", "citeseer", "pubmed", "ppi", "ogbn_arxiv")
HEX_256 = re.compile(r"^[0-9a-f]{64}$")
MATRIX_FORMAT = "zkgat_ablation_gpu_matrix_session_v1"
TIMING_SCHEMA = "online_trace_commit_prove_protocol_verify_v2"
GPU_RUNTIME_BACKEND = "cuda_hotspots"
METRIC_FIELDS = ("commitment_time_ms", "prove_time_ms", "verify_time_ms", "proof_size_bytes")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def atomic_write_text(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_json(path: pathlib.Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def parse_config(path: pathlib.Path) -> tuple[dict[str, str], list[str]]:
    require(path.is_file() and not path.is_symlink(), f"config must be a regular file: {path}")
    values: dict[str, str] = {}
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    for line_number, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        require("=" in line, f"malformed config line {line_number} in {path}")
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        require(key not in values, f"duplicate config key {key!r} in {path}")
        values[key] = value
    return values, lines


def safe_project_relative(value: str, field: str) -> str:
    path = pathlib.PurePosixPath(value)
    require(value and not path.is_absolute() and ".." not in path.parts, f"{field} must be a safe project-relative path")
    return path.as_posix()


def write_variant_config(
    base: pathlib.Path,
    output: pathlib.Path,
    variant: str,
    export_dir: str,
    registry_path: str,
    registry_digest: str | None,
) -> None:
    require(variant in VARIANTS, f"unsupported ablation variant: {variant}")
    export_dir = safe_project_relative(export_dir, "export_dir")
    registry_path = safe_project_relative(registry_path, "verifier_registry_path")
    if registry_digest is not None:
        require(HEX_256.fullmatch(registry_digest) is not None, "verifier_registry_digest must be lowercase SHA-256")

    values, lines = parse_config(base)
    require(values.get("dataset") in DATASETS, f"unsupported base dataset in {base}")
    updates = {
        "protocol_variant": variant,
        "export_dir": export_dir,
        "verifier_registry_path": registry_path,
    }
    if registry_digest is not None:
        updates["verifier_registry_digest"] = registry_digest

    output_lines: list[str] = []
    seen: set[str] = set()
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            output_lines.append(line)
            continue
        key = line.split("=", 1)[0].strip()
        if key == "verifier_registry_digest" and registry_digest is None:
            continue
        if key in updates:
            indent = line[: len(line) - len(line.lstrip())]
            output_lines.append(f"{indent}{key} = {updates[key]}\n")
            seen.add(key)
        else:
            output_lines.append(line)

    for key in ("protocol_variant", "export_dir", "verifier_registry_path", "verifier_registry_digest"):
        if key in updates and key not in seen:
            output_lines.append(f"{key} = {updates[key]}\n")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text("".join(output_lines), encoding="utf-8")
    os.replace(temporary, output)

    rendered, _ = parse_config(output)
    require(rendered.get("protocol_variant") == variant, "generated config lost protocol_variant")
    require(rendered.get("export_dir") == export_dir, "generated config lost session export_dir")
    require(rendered.get("verifier_registry_path") == registry_path, "generated config lost registry path")
    require(rendered.get("verifier_registry_digest") == registry_digest, "generated config has an unexpected registry digest")
    for key, value in values.items():
        if key not in {"protocol_variant", "export_dir", "verifier_registry_path", "verifier_registry_digest"}:
            require(rendered.get(key) == value, f"generated config unexpectedly changed base field {key}")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"JSON artifact has duplicate field {key!r}")
        result[key] = value
    return result


def load_json(path: pathlib.Path, context: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"{context} must be a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_pairs)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"cannot read {context} as strict JSON: {exc}") from exc
    require(isinstance(value, dict), f"{context} must be a JSON object")
    return value


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project_path(project_root: pathlib.Path, value: str) -> pathlib.Path:
    path = pathlib.Path(value)
    return path if path.is_absolute() else project_root / path


def _parse_profile(value: str, field: str, context: str, separator: str) -> list[int]:
    stripped = value.strip()
    require(stripped.startswith("[") and stripped.endswith("]"), f"{context}: {field} is malformed")
    pieces = [piece.strip() for piece in stripped[1:-1].split(",")]
    require(pieces and all(pieces), f"{context}: {field} is malformed")
    result: list[int] = []
    for piece in pieces:
        numbers = piece.split(separator)
        require(len(numbers) == (2 if separator == "x" else 1), f"{context}: {field} is malformed")
        require(all(number.isdigit() and int(number) > 0 for number in numbers), f"{context}: {field} must contain positive dimensions")
        result.extend(int(number) for number in numbers)
    return result


def _parse_small_key_values(path: pathlib.Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        require("=" in line, f"malformed metadata line {line_number} in {path}")
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        require(key not in values, f"duplicate metadata key {key!r} in {path}")
        values[key] = value
    return values


def startup_inventory(project_root: pathlib.Path) -> dict[str, Any]:
    """Collect cheap path/metadata facts only; full SRS authentication stays per-cell."""
    project_root = project_root.resolve()
    global_inventory: dict[str, Any] = {
        "cuda": {"probe": "nvidia-smi_query_only_no_kernel", "status": "unknown"},
        "build_tree": str(project_root / "build_cuda"),
        "inventory_scope": "paths_checkpoint_bundle_srs_manifest_dataset_meta_and_shape_only",
        "srs_full_authentication": "deferred_to_each_cell_register_and_inspect",
        "issues": [],
        "recorded_at": utc_now(),
    }
    query_tool = shutil.which("nvidia-smi")
    if query_tool is None:
        global_inventory["cuda"].update({"status": "query_tool_missing", "path": None})
    else:
        try:
            probe = subprocess.run(
                [query_tool, "--query-gpu=name,driver_version,memory.total,memory.free", "--format=csv,noheader"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=5,
                check=False,
            )
            lines = [line.strip() for line in probe.stdout.splitlines() if line.strip()]
            global_inventory["cuda"].update({
                "status": "visible" if probe.returncode == 0 and lines else "not_visible_or_probe_failed",
                "path": query_tool,
                "query_rc": probe.returncode,
                "devices": lines,
                "stderr": probe.stderr.strip(),
                "memory_n_a_is_accepted": True,
                "kernel_invoked": False,
            })
        except (OSError, subprocess.TimeoutExpired) as exc:
            global_inventory["cuda"].update({"status": "probe_error", "path": query_tool, "error": str(exc)})

    datasets: dict[str, Any] = {}
    for dataset in DATASETS:
        entry: dict[str, Any] = {"dataset": dataset, "status": "unknown", "issues": [], "warnings": []}
        config_path = project_root / "configs" / f"{dataset}_secure.cfg"
        entry["config_path"] = str(config_path)
        try:
            values, _ = parse_config(config_path)
            require(values.get("dataset") == dataset, f"config declares dataset={values.get('dataset')!r}")
            entry["base_config_sha256"] = sha256_file(config_path)
            entry["shape"] = {
                "L": int(values["L"]),
                "hidden_profile": values["hidden_profile"],
                "hidden_dimensions": _parse_profile(values["hidden_profile"], "hidden_profile", dataset, "x"),
                "d_in_profile": values["d_in_profile"],
                "input_dimensions": _parse_profile(values["d_in_profile"], "d_in_profile", dataset, "x")
                if "x" in values["d_in_profile"] else [int(item.strip()) for item in values["d_in_profile"].strip()[1:-1].split(",")],
                "K_out": int(values["K_out"]),
                "num_classes": int(values["num_classes"]),
            }
            require(entry["shape"]["L"] >= 1 and entry["shape"]["K_out"] >= 1 and entry["shape"]["num_classes"] >= 1,
                    f"{dataset}: configured model shape contains a nonpositive dimension")

            cache_root = _project_path(project_root, values["cache_root"])
            dataset_cache = cache_root / dataset
            entry["data_cache"] = str(dataset_cache)
            if not dataset_cache.is_dir():
                entry["issues"].append(f"dataset cache directory is missing: {dataset_cache}")
            else:
                meta_path = dataset_cache / "meta.cfg"
                required_files = [dataset_cache / "edges.txt"]
                feature_candidates = [dataset_cache / "features.txt", dataset_cache / "features.npy"]
                label_candidates = [dataset_cache / "labels.txt", dataset_cache / "labels_multi.npy", dataset_cache / "labels_multi.txt"]
                absent = [str(path) for path in required_files if not path.is_file() or path.is_symlink()]
                if not any(path.is_file() and not path.is_symlink() for path in feature_candidates):
                    absent.append(f"{dataset_cache}/features.(txt|npy)")
                if not any(path.is_file() and not path.is_symlink() for path in label_candidates):
                    absent.append(f"{dataset_cache}/labels.(txt|npy)")
                if not meta_path.is_file() or meta_path.is_symlink():
                    absent.append(str(meta_path))
                if absent:
                    entry["issues"].extend(f"required cached dataset input missing: {path}" for path in absent)
                else:
                    meta = _parse_small_key_values(meta_path)
                    entry["data_meta"] = meta
                    require(meta.get("name") == dataset, f"{dataset}: data meta names a different dataset")
                    require(int(meta["num_features"]) == entry["shape"]["input_dimensions"][0],
                            f"{dataset}: data meta feature width differs from d_in_profile")
                    require(int(meta["num_classes"]) == entry["shape"]["num_classes"],
                            f"{dataset}: data meta class count differs from config")
                    require(int(meta["num_nodes"]) > 0, f"{dataset}: data meta has no nodes")

            bundle_path = _project_path(project_root, values["checkpoint_bundle"])
            entry["checkpoint_bundle"] = str(bundle_path)
            bundle_files = ("manifest.json", "tensors.txt", "tensors.npz", "best_model.pt")
            if not bundle_path.is_dir() or bundle_path.is_symlink():
                entry["issues"].append(f"checkpoint bundle directory is missing or not regular: {bundle_path}")
            else:
                missing = [name for name in bundle_files if not (bundle_path / name).is_file() or (bundle_path / name).is_symlink()]
                if missing:
                    entry["issues"].append(f"checkpoint bundle files are missing: {', '.join(missing)}")
                else:
                    checkpoint_manifest = load_json(bundle_path / "manifest.json", f"{dataset} checkpoint manifest")
                    entry["checkpoint_bundle_manifest"] = {
                        key: checkpoint_manifest.get(key)
                        for key in ("dataset", "model_arch_id", "model_param_id", "quant_cfg_id", "input_preprocess_id", "L", "K_out")
                    }
                    for key in ("dataset", "model_arch_id", "model_param_id", "quant_cfg_id", "input_preprocess_id", "L", "K_out"):
                        expected = int(values[key]) if key in {"L", "K_out"} else values.get(key)
                        require(checkpoint_manifest.get(key) == expected,
                                f"{dataset}: checkpoint manifest {key} does not match base config")
                    entry["checkpoint_file_sizes"] = {name: (bundle_path / name).stat().st_size for name in bundle_files}

            srs_path = _project_path(project_root, values["srs_path"])
            manifest_path = srs_path / "srs_manifest.json"
            entry["srs_path"] = str(srs_path)
            if not manifest_path.is_file() or manifest_path.is_symlink():
                entry["issues"].append(f"SRS manifest is missing or not regular: {manifest_path}")
            else:
                manifest = load_json(manifest_path, f"{dataset} SRS manifest")
                degree_bound = manifest.get("degree_bound")
                require(manifest.get("srs_id") == values.get("srs_id"), f"{dataset}: SRS manifest id differs from config")
                require(manifest.get("srs_type") == values.get("srs_type") == "trusted_srs",
                        f"{dataset}: configured SRS is not the canonical trusted SRS type")
                require(isinstance(degree_bound, int) and degree_bound > 0, f"{dataset}: SRS degree_bound is malformed")
                require(HEX_256.fullmatch(str(manifest.get("manifest_hash", ""))) is not None,
                        f"{dataset}: SRS manifest_hash is malformed")
                require(HEX_256.fullmatch(str(manifest.get("protocol_schema_id", ""))) is not None,
                        f"{dataset}: SRS protocol_schema_id is malformed")
                entry["srs_metadata"] = {
                    "srs_id": manifest["srs_id"],
                    "srs_type": manifest["srs_type"],
                    "degree_bound": degree_bound,
                    "manifest_hash": manifest["manifest_hash"],
                    "protocol_schema_id": manifest["protocol_schema_id"],
                    "domain_sizes": manifest.get("domain_sizes", {}),
                    "full_authentication": "deferred_to_each_cell_register_and_inspect",
                }
            entry["status"] = "needs_attention" if entry["issues"] else "metadata_inventory_ok"
        except Exception as exc:
            entry["issues"].append(str(exc))
            entry["status"] = "needs_attention"
        datasets[dataset] = entry

    global_inventory["datasets"] = datasets
    global_inventory["global_issues"] = global_inventory.get("issues", [])
    global_inventory["status"] = "metadata_inventory_complete"
    return global_inventory


def _variant_list(value: str) -> list[str]:
    variants = [item for item in value.split(",") if item]
    require(variants and all(item in VARIANTS for item in variants), "variants must be a nonempty list of supported ablation variants")
    require(len(set(variants)) == len(variants), "duplicate protocol variant")
    return variants


def _cell_key(variant: str, dataset: str) -> str:
    require(variant in VARIANTS, f"unsupported protocol variant: {variant}")
    require(dataset in DATASETS, f"unsupported dataset: {dataset}")
    return f"{variant}/{dataset}"


def _read_matrix(session_root: pathlib.Path) -> dict[str, Any]:
    matrix = load_json(session_root / "matrix_summary.json", "matrix summary")
    require(matrix.get("format") == MATRIX_FORMAT, "unsupported matrix summary format")
    return matrix


def _refresh_matrix_files(session_root: pathlib.Path, matrix: dict[str, Any]) -> None:
    cells = matrix["cells"]
    terminal_worker = matrix.get("worker_status") in {"complete", "finished_with_failures"}
    if terminal_worker:
        for cell in cells.values():
            status = cell.get("status", "pending")
            missing_verified_marker = status == "success" and cell.get("verification_status") != "VERIFY_OK"
            if status == "running" or status not in {"pending", "running", "success", "failed"} or missing_verified_marker:
                previous_reason = str(cell.get("reason", "")).strip()
                if missing_verified_marker:
                    incomplete_reason = "runner_state_incomplete: success state lacks its validated VERIFY_OK marker"
                else:
                    incomplete_reason = f"runner_state_incomplete: worker finalized while the cell was {status}"
                cell["status"] = "failed"
                cell["reason"] = f"{previous_reason}; {incomplete_reason}" if previous_reason else incomplete_reason
                cell["finished_at"] = cell.get("finished_at") or utc_now()
                cell["metrics"] = {}
                cell.pop("verification_status", None)
                cell.pop("evidence_path", None)
            elif status == "pending" and not cell.get("reason"):
                cell["reason"] = "runner_state_incomplete: worker finalized before the cell was attempted"
    for cell in cells.values():
        if cell.get("status") == "failed":
            cell["metrics"] = {}
            cell.pop("verification_status", None)
            cell.pop("evidence_path", None)

    statuses = [entry.get("status", "pending") for entry in cells.values()]
    matrix["total"] = len(cells)
    matrix["attempted"] = sum(status != "pending" for status in statuses)
    matrix["succeeded"] = sum(status == "success" for status in statuses)
    matrix["failed"] = sum(status == "failed" for status in statuses)
    matrix["pending"] = sum(status == "pending" for status in statuses)
    matrix["running"] = sum(status == "running" for status in statuses)
    matrix["unresolved"] = sum(status not in {"success", "failed", "pending", "running"} for status in statuses)

    if matrix.get("worker_status") == "complete":
        cells_all_success = (matrix["succeeded"] == matrix["total"] and matrix["failed"] == 0 and
                            matrix["pending"] == 0 and matrix["running"] == 0 and matrix["unresolved"] == 0)
        if cells_all_success and not matrix.get("session_failure_reason") and matrix.get("session_failure_rc") is None:
            matrix["status"] = "complete"
        else:
            matrix["worker_status"] = "finished_with_failures"
            if not matrix.get("session_failure_reason"):
                if cells_all_success:
                    matrix["session_failure_reason"] = "worker requested complete while a session failure return code was recorded"
                else:
                    matrix["session_failure_reason"] = "worker requested complete with cells that were not all successful"
            matrix["status"] = "finished_with_failures"
    elif matrix.get("worker_status") == "finished_with_failures":
        if matrix["succeeded"] == matrix["total"] and not matrix.get("session_failure_reason"):
            matrix["session_failure_reason"] = "worker reported finished_with_failures although every cell is successful"
        matrix["status"] = "finished_with_failures"
    elif matrix.get("worker_status") == "starting":
        matrix["status"] = "starting"
    else:
        matrix["status"] = "running"

    summary_path = session_root / "status.tsv"
    columns = ("variant", "dataset", "status", "verification_status", "phase", "command_rc", "logger_rc", "validation_rc",
               "commitment_time_ms", "prove_time_ms", "verify_time_ms", "proof_size_bytes", "reason", "log")
    rows: list[dict[str, str]] = []
    for variant in matrix["variants"]:
        for dataset in matrix["datasets"]:
            cell = cells[_cell_key(variant, dataset)]
            metrics = cell.get("metrics") or {} if cell.get("status") == "success" else {}
            rows.append({
                "variant": variant,
                "dataset": dataset,
                "status": str(cell.get("status", "pending")),
                "verification_status": cell.get("verification_status", "") if cell.get("status") == "success" else "",
                "phase": str(cell.get("phase", "not_started")),
                "command_rc": "" if cell.get("command_rc") is None else str(cell.get("command_rc")),
                "logger_rc": "" if cell.get("logger_rc") is None else str(cell.get("logger_rc")),
                "validation_rc": "" if cell.get("validation_rc") is None else str(cell.get("validation_rc")),
                "commitment_time_ms": str(metrics.get("commitment_time_ms", "")),
                "prove_time_ms": str(metrics.get("prove_time_ms", "")),
                "verify_time_ms": str(metrics.get("verify_time_ms", "")),
                "proof_size_bytes": str(metrics.get("proof_size_bytes", "")),
                "reason": str(cell.get("reason", "")).replace("\t", " ").replace("\n", " "),
                "log": str(cell.get("log", "")),
            })
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = summary_path.with_name(summary_path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, summary_path)

    successes = [cell for cell in cells.values() if cell.get("status") == "success"]
    failures = [cell for cell in cells.values() if cell.get("status") == "failed"]
    atomic_write_text(session_root / "successes.jsonl", "".join(json.dumps(cell, sort_keys=True) + "\n" for cell in successes))
    atomic_write_text(session_root / "failures.jsonl", "".join(json.dumps(cell, sort_keys=True) + "\n" for cell in failures))

    for name, entries in (("successes.csv", successes), ("failures.csv", failures)):
        path = session_root / name
        tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
        with tmp.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            for cell in entries:
                metrics = cell.get("metrics") or {} if cell.get("status") == "success" else {}
                writer.writerow({
                    "variant": cell.get("variant", ""),
                    "dataset": cell.get("dataset", ""),
                    "status": cell.get("status", ""),
                    "verification_status": cell.get("verification_status", "") if cell.get("status") == "success" else "",
                    "phase": cell.get("phase", ""),
                    "command_rc": "" if cell.get("command_rc") is None else cell.get("command_rc"),
                    "logger_rc": "" if cell.get("logger_rc") is None else cell.get("logger_rc"),
                    "validation_rc": "" if cell.get("validation_rc") is None else cell.get("validation_rc"),
                    "commitment_time_ms": metrics.get("commitment_time_ms", ""),
                    "prove_time_ms": metrics.get("prove_time_ms", ""),
                    "verify_time_ms": metrics.get("verify_time_ms", ""),
                    "proof_size_bytes": metrics.get("proof_size_bytes", ""),
                    "reason": str(cell.get("reason", "")).replace("\n", " "),
                    "log": cell.get("log", ""),
                })
        os.replace(tmp, path)

    # Commit the canonical state last. Viewers read this file and must not see
    # a success transition until every sidecar representing it was written.
    atomic_write_json(session_root / "matrix_summary.json", matrix)


def init_matrix(args: argparse.Namespace) -> None:
    session_root = pathlib.Path(args.session_root).resolve()
    require(session_root.is_dir() and not session_root.is_symlink(), f"session reservation must be a regular directory: {session_root}")
    existing = list(session_root.iterdir())
    allowed_reservation = {"global.log"}
    require(all(path.name in allowed_reservation and path.is_file() and not path.is_symlink() and path.stat().st_size == 0
                for path in existing),
            f"refusing to initialize a reused or nonempty session directory: {session_root}")
    variants = _variant_list(args.variants)
    cells: dict[str, Any] = {}
    for variant in variants:
        for dataset in DATASETS:
            key = _cell_key(variant, dataset)
            cells[key] = {
                "variant": variant,
                "dataset": dataset,
                "status": "pending",
                "phase": "not_started",
                "reason": "",
                "command_rc": None,
                "logger_rc": None,
                "validation_rc": None,
                "stage_rcs": {},
                "registry_digest": "",
                "metrics": {},
                "log": "",
                "started_at": "",
                "finished_at": "",
            }
    matrix = {
        "format": MATRIX_FORMAT,
        "session_id": session_root.name,
        "session_root": str(session_root),
        "created_at": utc_now(),
        "finished_at": "",
        "worker_pid": args.worker_pid,
        "worker_pgid": _safe_process_group(args.worker_pid),
        "worker_sid": _safe_session_id(args.worker_pid),
        "worker_status": "starting",
        "status": "starting",
        "session_failure_reason": "",
        "session_failure_rc": None,
        "phase": "startup",
        "variants": variants,
        "datasets": list(DATASETS),
        "backend": "gpu",
        "current_cell": None,
        "inventory": None,
        "build": {"status": "pending", "command_rc": None, "logger_rc": None, "reason": "", "log": ""},
        "cells": cells,
    }
    _refresh_matrix_files(session_root, matrix)
    atomic_write_json(session_root / "worker.json", {
        "pid": args.worker_pid,
        "process_group_id": matrix["worker_pgid"],
        "session_id": matrix["worker_sid"],
        "recorded_at": utc_now(),
    })
    current = {"active": False, "variant": "", "dataset": "", "phase": "startup", "log": "", "updated_at": utc_now()}
    atomic_write_json(session_root / "current_cell.json", current)
    print(f"MATRIX_INITIALIZED session={session_root} variants={','.join(variants)} cells={len(cells)}")


def _safe_process_group(pid: int) -> int | None:
    try:
        return os.getpgid(pid)
    except OSError:
        return None


def _safe_session_id(pid: int) -> int | None:
    try:
        return os.getsid(pid)
    except OSError:
        return None


def update_matrix(args: argparse.Namespace) -> None:
    session_root = pathlib.Path(args.session_root).resolve()
    matrix = _read_matrix(session_root)
    if args.worker_status is not None:
        require(args.worker_status in {"starting", "running", "complete", "finished_with_failures"}, "invalid worker status")
        matrix["worker_status"] = args.worker_status
        if args.worker_status in {"complete", "finished_with_failures"}:
            matrix["finished_at"] = utc_now()
    if args.phase is not None:
        matrix["phase"] = args.phase
    if args.session_failure_reason is not None:
        matrix["session_failure_reason"] = args.session_failure_reason.strip()
    if args.session_failure_rc is not None:
        matrix["session_failure_rc"] = _parse_optional_rc(args.session_failure_rc)
    if args.inventory_path is not None:
        matrix["inventory"] = load_json(pathlib.Path(args.inventory_path), "startup inventory")
    if args.build_status is not None:
        require(args.build_status in {"running", "passed", "failed", "skipped"}, "invalid build status")
        matrix["build"] = {
            "status": args.build_status,
            "command_rc": _parse_optional_rc(args.command_rc),
            "logger_rc": _parse_optional_rc(args.logger_rc),
            "reason": args.reason or "",
            "log": args.log or "",
            "updated_at": utc_now(),
        }
    if args.current_active is not None:
        active = args.current_active == "true"
        current = {
            "active": active,
            "variant": args.variant or "",
            "dataset": args.dataset or "",
            "phase": args.phase or matrix.get("phase", ""),
            "log": args.log or "",
            "updated_at": utc_now(),
        }
        if active:
            matrix["current_cell"] = current
        else:
            matrix["current_cell"] = None
        atomic_write_json(session_root / "current_cell.json", current)
    if args.variant is not None and args.dataset is not None:
        key = _cell_key(args.variant, args.dataset)
        require(key in matrix["cells"], f"cell is not part of this session: {key}")
        cell = matrix["cells"][key]
        if args.cell_status is not None:
            require(args.cell_status in {"pending", "running", "success", "failed"}, "invalid cell status")
            if args.cell_status == "success":
                require(cell.get("status") == "running", "cell must be running before it can become successful")
            cell["status"] = args.cell_status
            if args.cell_status == "running" and not cell.get("started_at"):
                cell["started_at"] = utc_now()
            if args.cell_status in {"success", "failed"}:
                cell["finished_at"] = utc_now()
        if args.cell_phase is not None:
            cell["phase"] = args.cell_phase
        if args.reason is not None:
            cell["reason"] = args.reason
        if args.cell_log is not None:
            cell["log"] = args.cell_log
        if args.registry_digest is not None:
            cell["registry_digest"] = args.registry_digest
        if args.command_rc is not None:
            cell["command_rc"] = _parse_optional_rc(args.command_rc)
        if args.logger_rc is not None:
            cell["logger_rc"] = _parse_optional_rc(args.logger_rc)
        if args.validation_rc is not None:
            cell["validation_rc"] = _parse_optional_rc(args.validation_rc)
        if args.stage is not None:
            cell["stage_rcs"][args.stage] = {
                "command_rc": _parse_optional_rc(args.stage_command_rc),
                "cell_logger_rc": _parse_optional_rc(args.cell_logger_rc),
                "global_logger_rc": _parse_optional_rc(args.global_logger_rc),
                "status": args.stage_status or ("passed" if _parse_optional_rc(args.stage_command_rc) == 0 else "failed"),
                "reason": args.stage_reason or "",
                "updated_at": utc_now(),
            }
        if args.metrics_json is not None:
            metrics = load_json(pathlib.Path(args.metrics_json), "validated metrics").get("metrics")
            require(isinstance(metrics, dict), "validated metrics file has no metrics object")
            cell["metrics"] = metrics
        if args.evidence_path is not None:
            cell["evidence_path"] = args.evidence_path
        if args.proof_path is not None:
            cell["proof_path"] = args.proof_path
        if cell.get("status") == "success":
            _validate_success_transition(session_root, cell, args)
            cell["verification_status"] = "VERIFY_OK"
        elif args.cell_status == "failed":
            cell["metrics"] = {}
            cell.pop("verification_status", None)
            cell.pop("evidence_path", None)
    _refresh_matrix_files(session_root, matrix)
    print(f"MATRIX_UPDATED status={matrix['status']} session={session_root}")


def _parse_optional_rc(value: str | None) -> int | None:
    if value is None or value == "null":
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"return code must be an integer or null: {value!r}") from exc


def _require_local_session_file(session_root: pathlib.Path, value: str, subdir: str, context: str) -> pathlib.Path:
    path = pathlib.Path(value)
    if not path.is_absolute():
        path = session_root / path
    require(not path.is_symlink(), f"{context} must not be a symlink")
    resolved = path.resolve()
    root = (session_root / subdir).resolve()
    require(resolved.is_relative_to(root), f"{context} must stay inside this session's {subdir} directory")
    require(resolved.is_file(), f"{context} is missing or not a regular file")
    return resolved


def _validate_success_transition(session_root: pathlib.Path, cell: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    require(cell.get("command_rc") == 0, "success transition requires command_rc=0")
    require(cell.get("logger_rc") == 0, "success transition requires logger_rc=0")
    require(cell.get("validation_rc") == 0, "success transition requires validation_rc=0")
    require(args.evidence_path is not None, "success transition requires a validated evidence receipt")
    require(args.metrics_json is not None, "success transition requires validated metrics")
    require(args.proof_path is not None, "success transition requires a proof artifact path")

    evidence_path = _require_local_session_file(session_root, args.evidence_path, "evidence", "validated evidence receipt")
    metrics_path = _require_local_session_file(session_root, args.metrics_json, "evidence", "validated metrics receipt")
    proof_path = _require_local_session_file(session_root, args.proof_path, "results", "proof artifact")
    evidence = load_json(evidence_path, "validated formal evidence receipt")
    metrics_doc = load_json(metrics_path, "validated formal metrics receipt")
    metrics = metrics_doc.get("metrics")
    require(isinstance(metrics, dict) and set(metrics) == set(METRIC_FIELDS), "validated metrics receipt does not have exactly the four success metrics")
    require(metrics_doc.get("proof_path") == str(proof_path), "metrics receipt proof path differs from the selected proof")
    require(evidence.get("metrics") == metrics, "evidence receipt metrics differ from the validated metrics file")
    require(evidence.get("protocol_variant") == cell["variant"] and evidence.get("dataset") == cell["dataset"] and
            evidence.get("backend") == "gpu" and evidence.get("status") == "VERIFY_OK",
            "validated evidence receipt belongs to a different cell or backend")
    registry_digest = cell.get("registry_digest", "")
    require(HEX_256.fullmatch(str(registry_digest)) is not None and evidence.get("registry_digest") == registry_digest,
            "validated evidence receipt does not match this cell's pinned registry")
    require(evidence.get("proof_hash") == sha256_file(proof_path) and proof_path.stat().st_size > 0,
            "validated evidence receipt proof hash does not match the nonempty proof file")
    try:
        proof_size = int(metrics["proof_size_bytes"])
    except (TypeError, ValueError) as exc:
        raise ValueError("validated proof_size_bytes is not an integer") from exc
    require(proof_size > 0 and proof_size == proof_path.stat().st_size, "validated proof_size_bytes differs from the proof file size")
    for name in ("commitment_time_ms", "prove_time_ms", "verify_time_ms"):
        try:
            metric_value = float(metrics[name])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"validated {name} is not numeric") from exc
        require(math.isfinite(metric_value) and metric_value >= 0.0, f"validated {name} must be finite and nonnegative")

    run_dir_value = evidence.get("run_dir")
    require(isinstance(run_dir_value, str) and run_dir_value, "evidence receipt has no run directory")
    run_dir = pathlib.Path(run_dir_value).resolve()
    results_root = (session_root / "results").resolve()
    require(run_dir.is_relative_to(results_root), "evidence receipt run directory is outside this session's results")
    require(pathlib.Path(metrics_doc.get("run_dir", "")).resolve() == run_dir,
            "metrics receipt run directory differs from evidence")
    require(proof_path == (run_dir / "proof.bin").resolve(), "selected proof path is not the evidence run's proof.bin")

    config_path = session_root / "configs" / cell["variant"] / f"{cell['dataset']}.cfg"
    registry_path = session_root / "registries" / cell["variant"] / f"{cell['dataset']}.registry"
    inspection_path = session_root / "preflight" / cell["variant"] / f"{cell['dataset']}.registry.json"
    require(config_path.is_file() and not config_path.is_symlink(), "same-cell pinned config is missing")
    require(evidence.get("config_sha256") == sha256_file(config_path), "evidence config hash differs from this session's pinned config")
    require(registry_path.is_file() and not registry_path.is_symlink() and sha256_file(registry_path) == registry_digest,
            "same-cell registry artifact does not match the pinned digest")
    inspection = load_json(inspection_path, "same-cell registry inspection")
    require(inspection.get("protocol_variant") == cell["variant"] and inspection.get("dataset") == cell["dataset"] and
            inspection.get("artifact_sha256") == registry_digest and inspection.get("registry_digest") == registry_digest,
            "same-cell registry inspection identity differs from the evidence")

    run_manifest_path = run_dir / "run_manifest.json"
    require(sha256_file(run_manifest_path) == evidence.get("run_manifest_sha256"), "evidence run manifest hash mismatch")
    run_manifest = load_json(run_manifest_path, "evidence run manifest")
    require(run_manifest.get("protocol_name") == cell["variant"] and run_manifest.get("dataset") == cell["dataset"] and
            run_manifest.get("backend") == "gpu" and run_manifest.get("status") == "VERIFY_OK" and
            run_manifest.get("verifier_registry_digest") == registry_digest and run_manifest.get("trusted_registry_digest") == registry_digest,
            "evidence run manifest identity differs from this cell")
    report_path = run_dir / str(run_manifest.get("verifier_report_path", ""))
    require(report_path.is_file() and not report_path.is_symlink(), "evidence verifier report is missing")
    require(sha256_file(report_path) == evidence.get("verifier_report_sha256") == run_manifest.get("verifier_report_hash"),
            "evidence verifier report hash mismatch")
    require(run_manifest.get("proof_hash") == evidence.get("proof_hash"), "evidence run manifest proof hash mismatch")
    return {"metrics": metrics, "evidence_path": str(evidence_path), "proof_path": str(proof_path), "run_dir": str(run_dir)}


def latest_session_root(project_root: pathlib.Path) -> pathlib.Path:
    pointer = project_root / "runs" / "ablation_sessions" / "gpu_matrix" / "latest"
    require(pointer.is_file() and not pointer.is_symlink(), f"no latest GPU ablation session pointer: {pointer}")
    session_id = pointer.read_text(encoding="utf-8").strip()
    require(re.fullmatch(r"[A-Za-z0-9_.-]+", session_id) is not None, "latest session pointer is malformed")
    parent = pointer.parent.resolve()
    session_root = parent / session_id
    require(session_root.is_dir() and not session_root.is_symlink(), f"latest session directory is missing or not regular: {session_root}")
    require(session_root.resolve().parent == parent, "latest session escaped the canonical GPU matrix root")
    matrix_path = session_root / "matrix_summary.json"
    if matrix_path.is_file() and not matrix_path.is_symlink():
        matrix = _read_matrix(session_root)
        require(matrix.get("session_id") == session_id, "latest pointer ID does not match matrix session_id")
    return session_root


def _confined_current_log(
    session_root: pathlib.Path,
    project_root: pathlib.Path,
    value: str,
    variant: str,
    dataset: str,
    recorded_cell_log: str,
) -> pathlib.Path:
    require(value != "", "current cell pointer has no log path")
    require(variant in VARIANTS and dataset in DATASETS, "current cell pointer has an invalid variant or dataset")
    allowed_cell_logs = {
        f"logs/{variant}_{dataset}.log",
        f"{variant}_{dataset}.failure.log",
    }
    require(recorded_cell_log in allowed_cell_logs,
            "selected cell log is neither its canonical log nor its exact failure fallback")

    project_root = project_root.resolve()
    session_root = session_root.resolve()
    try:
        session_relative = session_root.relative_to(project_root)
    except ValueError as exc:
        raise ValueError("current cell session is outside the project root") from exc
    expected_pointer = session_relative / recorded_cell_log
    candidate = pathlib.Path(value)
    if candidate.is_absolute():
        require(candidate.absolute() == (project_root / expected_pointer).absolute(),
                "current cell log path does not match the selected cell's recorded log")
        candidate = candidate.absolute()
    else:
        require(pathlib.PurePosixPath(value).as_posix() == expected_pointer.as_posix(),
                "current cell log path does not match the selected cell's recorded log")
        candidate = project_root / expected_pointer

    expected = session_root / recorded_cell_log
    require(candidate.absolute() == expected.absolute(), "current cell log path does not match the session-relative cell log")
    current = session_root
    for component in pathlib.PurePosixPath(recorded_cell_log).parts:
        current = current / component
        require(not current.is_symlink(), "current cell log path contains a symlink")
    resolved = candidate.resolve()
    require(resolved == expected.resolve() and resolved.is_relative_to(session_root),
            "current cell log path resolves outside this session")
    require(resolved.is_file(), "current cell log is missing or not a regular file")
    return resolved


def print_matrix_summary(project_root: pathlib.Path, failures_only: bool = False) -> None:
    session_root = latest_session_root(project_root)
    matrix = _read_matrix(session_root)
    print(f"MATRIX_SUMMARY status={matrix['status']} session_id={matrix['session_id']} session={session_root} "
          f"worker_pid={matrix.get('worker_pid')} phase={matrix.get('phase')} total={matrix['total']} attempted={matrix['attempted']} "
          f"succeeded={matrix['succeeded']} failed={matrix['failed']} pending={matrix['pending']} "
          f"running={matrix.get('running', 0)} unresolved={matrix.get('unresolved', 0)}")
    if matrix.get("session_failure_reason") or matrix.get("session_failure_rc") is not None:
        print(f"MATRIX_SESSION_FAILURE reason={matrix.get('session_failure_reason') or 'unspecified'} "
              f"rc={matrix.get('session_failure_rc')}")
    current = matrix.get("current_cell")
    if current:
        print(f"MATRIX_CURRENT variant={current.get('variant')} dataset={current.get('dataset')} phase={current.get('phase')} log={current.get('log')}")
    for variant in matrix["variants"]:
        for dataset in matrix["datasets"]:
            cell = matrix["cells"][_cell_key(variant, dataset)]
            status = cell.get("status", "pending")
            if failures_only and status not in {"failed", "pending"}:
                continue
            if status == "success":
                metrics = cell.get("metrics", {})
                print(f"SUCCESS variant={variant} dataset={dataset} verification_status={cell.get('verification_status', '')} "
                      f"commitment_time_ms={metrics.get('commitment_time_ms')} "
                      f"prove_time_ms={metrics.get('prove_time_ms')} verify_time_ms={metrics.get('verify_time_ms')} "
                      f"proof_size_bytes={metrics.get('proof_size_bytes')} registry_digest={cell.get('registry_digest')} "
                      f"proof={cell.get('proof_path')} log={cell.get('log')}")
            elif status == "failed":
                print(f"FAILURE variant={variant} dataset={dataset} phase={cell.get('phase')} reason={cell.get('reason')} "
                      f"command_rc={cell.get('command_rc')} logger_rc={cell.get('logger_rc')} validation_rc={cell.get('validation_rc')} "
                      f"log={cell.get('log')}")
            elif status == "pending" and failures_only:
                print(f"UNATTEMPTED variant={variant} dataset={dataset} phase={cell.get('phase')} reason={cell.get('reason')} log={cell.get('log')}")
            elif not failures_only:
                print(f"{status.upper()} variant={variant} dataset={dataset} phase={cell.get('phase')} log={cell.get('log')}")
    if failures_only and matrix["failed"] == 0:
        print("MATRIX_FAILURES count=0")


def follow_global_log(project_root: pathlib.Path) -> None:
    session_root = latest_session_root(project_root)
    log_path = session_root / "global.log"
    position = 0
    while True:
        if log_path.is_file() and not log_path.is_symlink():
            with log_path.open("rb") as stream:
                stream.seek(position)
                chunk = stream.read()
                position = stream.tell()
            if chunk:
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
        matrix_path = session_root / "matrix_summary.json"
        if matrix_path.is_file() and not matrix_path.is_symlink():
            matrix = _read_matrix(session_root)
            require(matrix.get("session_id") == session_root.name, "latest pointer ID does not match matrix session_id")
            if matrix.get("status") in {"complete", "finished_with_failures"}:
                print(f"\nMATRIX_FOLLOW_COMPLETE status={matrix['status']} session={session_root}")
                return
        time.sleep(0.25)


def follow_current_log(project_root: pathlib.Path) -> None:
    session_root = latest_session_root(project_root)
    pointer_path = session_root / "current_cell.json"
    current_log: pathlib.Path | None = None
    position = 0
    last_pointer_stamp = ""
    while True:
        matrix_path = session_root / "matrix_summary.json"
        if not matrix_path.is_file() or matrix_path.is_symlink():
            time.sleep(0.25)
            continue
        matrix = _read_matrix(session_root)
        require(matrix.get("session_id") == session_root.name, "latest pointer ID does not match matrix session_id")
        if matrix.get("status") in {"complete", "finished_with_failures"}:
            if current_log is not None and current_log.is_file():
                with current_log.open("rb") as stream:
                    stream.seek(position)
                    pending = stream.read()
                    if pending:
                        sys.stdout.buffer.write(pending)
                        sys.stdout.buffer.flush()
            print(f"\nMATRIX_FOLLOW_CURRENT_COMPLETE status={matrix['status']} last_cell={last_pointer_stamp or 'none'}")
            return

        pointer: dict[str, Any] | None = None
        if pointer_path.is_file() and not pointer_path.is_symlink():
            try:
                pointer = load_json(pointer_path, "current cell pointer")
            except ValueError:
                pointer = None
        if pointer and pointer.get("active") and pointer.get("log"):
            variant = str(pointer.get("variant", ""))
            dataset = str(pointer.get("dataset", ""))
            current = matrix.get("current_cell")
            if (not isinstance(current, dict) or current.get("active") is not True or
                    current.get("variant") != variant or current.get("dataset") != dataset or
                    current.get("log") != pointer.get("log")):
                time.sleep(0.25)
                continue
            key = _cell_key(variant, dataset)
            cell = matrix["cells"].get(key)
            if not isinstance(cell, dict) or cell.get("status") != "running":
                time.sleep(0.25)
                continue
            next_log = _confined_current_log(
                session_root, project_root, str(pointer["log"]), variant, dataset, str(cell.get("log", ""))
            )
            if current_log != next_log:
                if current_log is not None and current_log.is_file():
                    with current_log.open("rb") as stream:
                        stream.seek(position)
                        pending = stream.read()
                        if pending:
                            sys.stdout.buffer.write(pending)
                            sys.stdout.buffer.flush()
                current_log, position = next_log, 0
                stamp = f"{pointer.get('variant')}/{pointer.get('dataset')}"
                print(f"===== FOLLOW_CURRENT {stamp} phase={pointer.get('phase')} log={current_log} =====", flush=True)
                last_pointer_stamp = stamp
            if current_log.is_file() and not current_log.is_symlink():
                with current_log.open("rb") as stream:
                    stream.seek(position)
                    chunk = stream.read()
                    position = stream.tell()
                if chunk:
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.buffer.flush()
        time.sleep(0.25)


def check_registry(args: argparse.Namespace) -> None:
    project_root = pathlib.Path(args.project_root).resolve()
    config, _ = parse_config(pathlib.Path(args.config))
    inspection = load_json(pathlib.Path(args.inspection), "registry inspection")
    digest = args.registry_digest
    require(HEX_256.fullmatch(digest) is not None, "registry digest must be lowercase SHA-256")
    registry_path = pathlib.Path(args.registry).resolve()
    configured_registry = pathlib.Path(config.get("verifier_registry_path", ""))
    if not configured_registry.is_absolute():
        configured_registry = project_root / configured_registry
    require(configured_registry.resolve() == registry_path, "variant config points at a different verifier registry path")
    require(config.get("verifier_registry_digest") == digest, "variant config does not pin the inspected registry digest")
    require(sha256_file(registry_path) == digest, "registry artifact SHA-256 differs from pinned digest")
    require(inspection.get("artifact_sha256") == digest and inspection.get("registry_digest") == digest,
            "registry inspection does not authenticate the pinned registry digest")
    require(inspection.get("protocol_variant") == args.variant, "registry inspection protocol_variant mismatch")
    require(inspection.get("security_budget_passed") is True, "registry SecurityBudgetLedger preflight failed")
    srs_root = pathlib.Path(config.get("srs_path", ""))
    if not srs_root.is_absolute():
        srs_root = project_root / srs_root
    srs_manifest_path = srs_root / "srs_manifest.json"
    srs_manifest = load_json(srs_manifest_path, "configured SRS manifest")
    require(srs_manifest.get("srs_id") == config.get("srs_id") == inspection.get("srs_id"),
            "registry inspection SRS id differs from the canonical config and SRS manifest")
    try:
        raw_srs_manifest = srs_manifest_path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read configured SRS manifest bytes: {exc}") from exc
    manifest_hash_marker = b'  "manifest_hash"'
    manifest_hash_position = raw_srs_manifest.rfind(manifest_hash_marker)
    require(manifest_hash_position >= 0,
            "configured SRS manifest is missing its canonical manifest_hash field")
    require(hashlib.sha256(raw_srs_manifest[:manifest_hash_position]).hexdigest() == srs_manifest.get("manifest_hash"),
            "configured SRS manifest canonical hash field is invalid")
    require(hashlib.sha256(raw_srs_manifest).hexdigest() == inspection.get("srs_manifest_hash"),
            "registry inspection SRS manifest full-file hash differs from the configured SRS")
    require(srs_manifest.get("protocol_schema_id") == inspection.get("protocol_schema_id"),
            "registry inspection changed the static protocol schema bound by the configured SRS")
    if "srs_degree_bound" in inspection:
        require(str(srs_manifest.get("degree_bound")) == str(inspection.get("srs_degree_bound")),
                "registry inspection SRS degree bound differs from the configured SRS")
    for config_key, registry_key in (
        ("dataset", "dataset"),
        ("model_arch_id", "model_arch_id"),
        ("model_param_id", "model_param_id"),
        ("quant_cfg_id", "quant_cfg_id"),
        ("static_table_id", "static_table_id"),
        ("srs_id", "srs_id"),
    ):
        if registry_key in inspection:
            require(inspection.get(registry_key) == config.get(config_key), f"registry inspection {registry_key} does not match config")
        else:
            raise ValueError(f"registry inspection is missing {registry_key}")
    require(HEX_256.fullmatch(str(inspection.get("protocol_schema_id", ""))) is not None,
            "registry inspection protocol_schema_id is malformed")
    require(HEX_256.fullmatch(str(inspection.get("instance_schema_hash", ""))) is not None,
            "registry inspection instance_schema_hash is malformed")


def _load_build_provenance_module(project_root: pathlib.Path):
    path = project_root / "tools" / "build_provenance.py"
    spec = importlib.util.spec_from_file_location("gatzk_ablation_build_provenance", path)
    require(spec is not None and spec.loader is not None, f"cannot load build provenance helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _truth(value: Any) -> bool:
    return value is True or value == "true"


def _load_aggregate_validator(project_root: pathlib.Path):
    path = project_root / "scripts" / "aggregate_secure_results.py"
    spec = importlib.util.spec_from_file_location("gatzk_ablation_single_cell_validator", path)
    require(spec is not None and spec.loader is not None, f"cannot load canonical single-cell validation functions: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _native_success_line(path: pathlib.Path, dataset: str, variant: str, expected: dict[str, Any], context: str) -> None:
    require(path.is_file() and not path.is_symlink(), f"{context}: native run log is missing")
    matches: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith(f"dataset={dataset} "):
            continue
        fields: dict[str, str] = {}
        for token in line.split():
            if "=" in token:
                key, value = token.split("=", 1)
                fields[key] = value
        if (fields.get("backend") == "gpu" and fields.get("protocol_variant") == variant and
                fields.get("status") == "VERIFY_OK"):
            matches.append(fields)
    require(len(matches) == 1, f"{context}: expected exactly one matching native VERIFY_OK line, got {len(matches)}")
    fields = matches[0]
    for name in METRIC_FIELDS:
        require(name in fields, f"{context}: native VERIFY_OK line is missing {name}")
        if name == "proof_size_bytes":
            try:
                reported = int(fields[name])
            except ValueError as exc:
                raise ValueError(f"{context}: native {name} is not an integer") from exc
            require(reported == expected[name], f"{context}: native proof size differs from proof.bin")
        else:
            try:
                reported = float(fields[name])
            except ValueError as exc:
                raise ValueError(f"{context}: native {name} is not numeric") from exc
            require(math.isfinite(reported) and abs(reported - expected[name]) <= max(0.00051, abs(expected[name]) * 1e-7),
                    f"{context}: native {name} differs from the run manifest")


def validate_cell(args: argparse.Namespace) -> None:
    project_root = pathlib.Path(args.project_root).resolve()
    run_dir = pathlib.Path(args.run_dir).resolve()
    config_path = pathlib.Path(args.config).resolve()
    expected_binary = pathlib.Path(args.binary).resolve()
    require(args.variant in VARIANTS, f"unsupported ablation variant: {args.variant}")
    require(args.dataset in DATASETS, f"unsupported dataset: {args.dataset}")
    require(args.backend in {"cpu", "gpu"}, f"unsupported backend: {args.backend}")
    manifest = load_json(run_dir / "run_manifest.json", "run manifest")
    proof_path = run_dir / "proof.bin"
    require(proof_path.is_file() and not proof_path.is_symlink() and proof_path.stat().st_size > 0,
            "formal run has no nonempty regular proof.bin")
    proof_hash = sha256_file(proof_path)
    require(manifest.get("proof_hash") == proof_hash, "run manifest proof_hash mismatch")
    require(manifest.get("protocol_name") == args.variant, "run manifest protocol_name does not match the selected variant")
    require(manifest.get("dataset") == args.dataset, "run manifest dataset mismatch")
    require(manifest.get("backend") == args.backend, "run manifest backend mismatch")
    require(manifest.get("benchmark_mode") == "single", "run manifest is not a single formal run")
    require(_truth(manifest.get("is_full_dataset")) and _truth(manifest.get("full_dataset")),
            "run manifest is not a full-dataset run")
    require(pathlib.Path(str(manifest.get("config", ""))).resolve() == config_path, "run manifest config path mismatch")
    require(sha256_file(config_path) == manifest.get("config_sha256"), "run manifest config SHA-256 mismatch")
    config, _ = parse_config(config_path)
    require(config.get("dataset") == args.dataset and config.get("protocol_variant") == args.variant,
            "pinned formal config does not match dataset and selected variant")
    require(config.get("verifier_registry_digest") == args.registry_digest, "formal config registry digest mismatch")
    require(manifest.get("verifier_registry_digest") == args.registry_digest and
            manifest.get("trusted_registry_digest") == args.registry_digest, "run manifest registry binding mismatch")
    registry_path = pathlib.Path(args.registry)
    inspection_path = pathlib.Path(args.inspection)
    check_registry(argparse.Namespace(
        project_root=str(project_root), config=str(config_path), variant=args.variant,
        registry=str(registry_path), registry_digest=args.registry_digest, inspection=str(inspection_path),
    ))
    inspection = load_json(inspection_path, "registry inspection")
    registration = {
        "registry_digest": args.registry_digest,
        "security_budget_ledger_digest": inspection.get("security_budget_ledger_digest"),
        "security_budget_exact": inspection.get("security_budget_exact"),
        "public_input_digest": inspection.get("public_input_digest"),
        "topology_digest": inspection.get("topology_digest"),
    }

    aggregate = _load_aggregate_validator(project_root)
    context = f"gpu/{args.dataset} ({run_dir})"
    aggregate.require_canonical_timing_schema(manifest, context)
    aggregate.check_formal_acceptance_state(manifest, context)
    provenance = aggregate.check_source_provenance(project_root, run_dir, manifest, context)
    aggregate.check_verifier_report(run_dir, proof_path, manifest, provenance, registration, context)
    require(manifest.get("backend") == "gpu", "GPU matrix cell manifest backend is not gpu")
    require(manifest.get("compute_backend_name") == GPU_RUNTIME_BACKEND, "GPU matrix cell did not use cuda_hotspots")
    require(aggregate.as_bool(manifest.get("gpu_backend_requested"), "gpu_backend_requested", context),
            "GPU matrix cell did not request a GPU backend")
    require(aggregate.as_int(manifest.get("gpu_kernel_invocation_count"), "gpu_kernel_invocation_count", context) > 0,
            "GPU matrix cell reports no CUDA kernel invocation")
    require(aggregate.as_bool(manifest.get("gpu_runtime_present"), "gpu_runtime_present", context) or
            aggregate.as_bool(manifest.get("gpu_runtime_loaded"), "gpu_runtime_loaded", context),
            "GPU matrix cell has no loaded/present CUDA runtime evidence")
    aggregate.nonempty_string(manifest.get("gpu_device"), "gpu_device", context)
    aggregate.nonempty_string(manifest.get("gpu_acceleration_status"), "gpu_acceleration_status", context)

    metrics: dict[str, Any] = {}
    for name in ("commitment_time_ms", "prove_time_ms", "verify_time_ms"):
        value = aggregate.as_float(manifest.get(name), name, context)
        require(value >= 0.0, f"{context}: {name} must not be negative")
        metrics[name] = value
    proof_size = aggregate.as_int(manifest.get("proof_size_bytes"), "proof_size_bytes", context)
    require(proof_size > 0 and proof_size == proof_path.stat().st_size,
            f"{context}: proof_size_bytes does not equal the nonempty proof.bin size")
    metrics["proof_size_bytes"] = proof_size
    _native_success_line(pathlib.Path(args.result_log), args.dataset, args.variant, metrics, context)

    require(pathlib.Path(str(manifest.get("binary_path", ""))).resolve() == expected_binary,
            "run manifest binary path differs from the selected backend executable")
    require(provenance["binary_hash"] == sha256_file(expected_binary), "formal run binary hash does not match the selected executable")
    if args.receipt_path:
        receipt_path = pathlib.Path(args.receipt_path)
        require(not receipt_path.exists() and not receipt_path.is_symlink(),
                f"formal evidence receipt already exists: {receipt_path}")
        evidence = {
            "backend": args.backend,
            "config_sha256": manifest["config_sha256"],
            "dataset": args.dataset,
            "metrics": metrics,
            "proof_hash": proof_hash,
            "protocol_variant": args.variant,
            "registry_digest": args.registry_digest,
            "run_dir": str(run_dir),
            "run_manifest_sha256": sha256_file(run_dir / "run_manifest.json"),
            "verifier_report_sha256": sha256_file(run_dir / str(manifest.get("verifier_report_path"))),
            "status": "VERIFY_OK",
        }
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = receipt_path.with_name(receipt_path.name + ".tmp")
        temporary.write_text(json.dumps(evidence, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
        os.replace(temporary, receipt_path)
    print(f"FORMAL_EVIDENCE_OK dataset={args.dataset} backend={args.backend} protocol_variant={args.variant} "
          f"registry_digest={args.registry_digest} commitment_time_ms={metrics['commitment_time_ms']} "
          f"prove_time_ms={metrics['prove_time_ms']} verify_time_ms={metrics['verify_time_ms']} "
          f"proof_size_bytes={metrics['proof_size_bytes']} run_dir={run_dir}")
    if args.metrics_output:
        atomic_write_json(pathlib.Path(args.metrics_output), {"metrics": metrics, "proof_path": str(proof_path), "run_dir": str(run_dir)})


def write_session_index(args: argparse.Namespace) -> None:
    project_root = pathlib.Path(args.project_root).resolve()
    session_root = pathlib.Path(args.session_root).resolve()
    require(args.variant in VARIANTS, f"unsupported ablation variant: {args.variant}")
    backends = tuple(args.backends.split(","))
    require(backends and all(backend in {"cpu", "gpu"} for backend in backends), "invalid session backend list")
    require(len(set(backends)) == len(backends), "duplicate backend in session backend list")

    datasets: dict[str, Any] = {}
    registrations_complete = True
    for dataset in DATASETS:
        base_path = project_root / "configs" / f"{dataset}_secure.cfg"
        config_path = session_root / "configs" / f"{dataset}.cfg"
        registry_path = session_root / "registries" / f"{dataset}.registry"
        inspection_path = session_root / "preflight" / f"{dataset}.registry.json"
        base_values: dict[str, str] = {}
        if base_path.is_file() and not base_path.is_symlink():
            base_values, _ = parse_config(base_path)
        else:
            registrations_complete = False
        config_values: dict[str, str] = {}
        if config_path.is_file() and not config_path.is_symlink():
            config_values, _ = parse_config(config_path)
        else:
            registrations_complete = False
        inspection: dict[str, Any] = {}
        srs_manifest: dict[str, Any] = {}
        if config_values:
            srs_root = pathlib.Path(config_values.get("srs_path", ""))
            if not srs_root.is_absolute():
                srs_root = project_root / srs_root
            srs_manifest_path = srs_root / "srs_manifest.json"
            if srs_manifest_path.is_file() and not srs_manifest_path.is_symlink():
                srs_manifest = load_json(srs_manifest_path, f"{dataset} configured SRS manifest")
        if registry_path.is_file() and not registry_path.is_symlink() and inspection_path.is_file() and not inspection_path.is_symlink():
            inspection = load_json(inspection_path, f"{dataset} registry inspection")
            digest = sha256_file(registry_path)
            require(inspection.get("artifact_sha256") == digest and inspection.get("registry_digest") == digest,
                    f"{dataset} session registry digest does not match inspection")
            require(inspection.get("protocol_variant") == args.variant, f"{dataset} registry has the wrong protocol variant")
            require(config_values.get("protocol_variant") == args.variant and config_values.get("verifier_registry_digest") == digest,
                    f"{dataset} pinned config does not match the inspected registry")
            require(inspection.get("dataset") == dataset, f"{dataset} registry dataset mismatch")
        else:
            registrations_complete = False
            digest = ""

        generated_config_sha256 = sha256_file(config_path) if config_path.is_file() and not config_path.is_symlink() else ""
        runs: dict[str, Any] = {}
        for backend in backends:
            evidence_path = session_root / "evidence" / f"{backend}_{dataset}.json"
            if evidence_path.is_file() and not evidence_path.is_symlink():
                evidence = load_json(evidence_path, f"{backend}/{dataset} formal evidence receipt")
                require(evidence.get("protocol_variant") == args.variant and evidence.get("dataset") == dataset and
                        evidence.get("backend") == backend and evidence.get("registry_digest") == digest,
                        f"{backend}/{dataset} evidence receipt does not match this session")
                require(evidence.get("status") == "VERIFY_OK" and
                        evidence.get("config_sha256") == generated_config_sha256,
                        f"{backend}/{dataset} evidence receipt does not match the pinned config")
                run_dir = pathlib.Path(str(evidence.get("run_dir", ""))).resolve()
                results_root = (session_root / "results").resolve()
                require(run_dir.is_relative_to(results_root), f"{backend}/{dataset} evidence points outside this session results directory")
                run_manifest_path = run_dir / "run_manifest.json"
                proof_path = run_dir / "proof.bin"
                require(sha256_file(run_manifest_path) == evidence.get("run_manifest_sha256"),
                        f"{backend}/{dataset} evidence run manifest hash mismatch")
                require(sha256_file(proof_path) == evidence.get("proof_hash"), f"{backend}/{dataset} evidence proof hash mismatch")
                run_manifest = load_json(run_manifest_path, f"{backend}/{dataset} evidence run manifest")
                require(run_manifest.get("protocol_name") == args.variant and run_manifest.get("dataset") == dataset and
                        run_manifest.get("backend") == backend and run_manifest.get("status") == "VERIFY_OK" and
                        run_manifest.get("verifier_registry_digest") == digest and run_manifest.get("trusted_registry_digest") == digest,
                        f"{backend}/{dataset} evidence run manifest identity mismatch")
                runs[backend] = evidence
            else:
                runs[backend] = {"status": "not_verified"}
        datasets[dataset] = {
            "base_config": str(base_path),
            "base_config_sha256": sha256_file(base_path) if base_path.is_file() and not base_path.is_symlink() else "",
            "generated_config": str(config_path),
            "generated_config_sha256": generated_config_sha256,
            "registry_path": str(registry_path),
            "registry_sha256": digest,
            "protocol_variant": inspection.get("protocol_variant", ""),
            "registry_id": inspection.get("registry_id", ""),
            "protocol_schema_id": inspection.get("protocol_schema_id", ""),
            "instance_schema_hash": inspection.get("instance_schema_hash", ""),
            "srs_id": inspection.get("srs_id", ""),
            "srs_manifest_hash": inspection.get("srs_manifest_hash", ""),
            "srs_degree_bound": inspection.get("srs_degree_bound", srs_manifest.get("degree_bound", "")),
            "security_budget_ledger_digest": inspection.get("security_budget_ledger_digest", ""),
            "runs": runs,
        }

    all_runs_verified = registrations_complete and all(
        runs.get(backend, {}).get("status") == "VERIFY_OK"
        for dataset_entry in datasets.values()
        for backend, runs in dataset_entry["runs"].items()
    )
    status = "complete" if all_runs_verified else ("formal_runs_in_progress" if registrations_complete else "preflight_in_progress")
    index = {
        "format": "zkgat_ablation_session_index_v1",
        "session_id": session_root.name,
        "protocol_variant": args.variant,
        "backends": list(backends),
        "status": status,
        "preflight_complete": registrations_complete,
        "datasets": datasets,
    }
    output = session_root / "ablation_session_index.json"
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(f"SESSION_INDEX status={status} path={output}")


def _read_integer(path: pathlib.Path) -> int | None:
    try:
        value = path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError):
        return None
    if value == "max":
        return None
    try:
        result = int(value)
    except ValueError:
        return None
    return result if result > 0 else None


def effective_available_memory() -> int:
    available: list[int] = []
    for line in pathlib.Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        if line.startswith("MemAvailable:"):
            fields = line.split()
            require(len(fields) >= 2 and fields[1].isdigit(), "cannot parse MemAvailable from /proc/meminfo")
            available.append(int(fields[1]) * 1024)
            break
    require(available, "MemAvailable is absent from /proc/meminfo")

    memberships: list[tuple[str, str]] = []
    try:
        for line in pathlib.Path("/proc/self/cgroup").read_text(encoding="ascii").splitlines():
            fields = line.split(":", 2)
            if len(fields) == 3:
                memberships.append((fields[1], fields[2].lstrip("/")))
    except (OSError, UnicodeError):
        memberships = []
    candidates: set[pathlib.Path] = {pathlib.Path("/sys/fs/cgroup")}
    for controllers, relative in memberships:
        if not controllers:
            base = pathlib.Path("/sys/fs/cgroup") / relative
            while base != pathlib.Path("/"):
                candidates.add(base)
                base = base.parent
        elif "memory" in controllers.split(","):
            for root in (pathlib.Path("/sys/fs/cgroup/memory"), pathlib.Path("/sys/fs/cgroup")):
                base = root / relative
                while base != pathlib.Path("/"):
                    candidates.add(base)
                    base = base.parent
    for directory in candidates:
        limit = _read_integer(directory / "memory.max")
        current = _read_integer(directory / "memory.current") or 0
        if limit is None:
            limit = _read_integer(directory / "memory.limit_in_bytes")
            current = _read_integer(directory / "memory.usage_in_bytes") or current
        if limit is not None:
            available.append(max(0, limit - current))
    return min(available)


def print_build_jobs(_args: argparse.Namespace) -> None:
    available = effective_available_memory()
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    gib = 1024**3
    reserve = max(2 * gib, min(8 * gib, available // 4))
    per_compile_job = 6 * gib
    memory_limited_jobs = max(1, (available - reserve) // per_compile_job)
    print(max(1, min(cpus, memory_limited_jobs, 4)))


def check_build(args: argparse.Namespace) -> None:
    project_root = pathlib.Path(args.project_root).resolve()
    build_dir = pathlib.Path(args.build_dir).resolve()
    cache_path = build_dir / "CMakeCache.txt"
    require(cache_path.is_file() and not cache_path.is_symlink(), f"missing regular CMake cache: {cache_path}")
    cache: dict[str, str] = {}
    for line in cache_path.read_text(encoding="utf-8").splitlines():
        if ":" in line and "=" in line and not line.startswith("//"):
            key, value = line.split("=", 1)[0].split(":", 1)[0], line.split("=", 1)[1]
            cache[key] = value
    expected_cuda = "ON" if args.cuda else "OFF"
    require(cache.get("CMAKE_BUILD_TYPE") == "Release", f"{build_dir} is not a Release build")
    require(cache.get("GATZK_ENABLE_CUDA_BACKEND") == expected_cuda,
            f"{build_dir} CUDA backend does not match requested build type")
    source_manifest_path = build_dir / "generated" / "gatzk" / "source_manifest.json"
    generated = load_json(source_manifest_path, "generated source manifest")
    provenance = _load_build_provenance_module(project_root)
    current = provenance.source_manifest_document(project_root)
    require(generated.get("source_tree_hash") == current.get("source_tree_hash"),
            f"{build_dir} was not configured against the current source tree")
    targets = ("gatzk_run", "gatzk_registry_register", "gatzk_registry_inspect")
    for target in targets:
        executable = build_dir / target
        require(executable.is_file() and not executable.is_symlink() and os.access(executable, os.X_OK),
                f"missing source-matched executable {executable}")
    print(f"SOURCE_MATCHED_BUILD_OK cuda={expected_cuda.lower()} source_tree_hash={current['source_tree_hash']} build_dir={build_dir}")


def write_inventory(args: argparse.Namespace) -> None:
    inventory = startup_inventory(pathlib.Path(args.project_root))
    atomic_write_json(pathlib.Path(args.output), inventory)
    issues = sum(len(entry.get("issues", [])) for entry in inventory["datasets"].values())
    print(f"STARTUP_INVENTORY_RECORDED datasets={len(inventory['datasets'])} dataset_issues={issues} "
          f"cuda_probe={inventory['cuda'].get('status', 'unknown')} scope={inventory['inventory_scope']} output={args.output}")


def check_inventory_dataset(args: argparse.Namespace) -> None:
    inventory = load_json(pathlib.Path(args.input), "startup inventory")
    dataset = args.dataset
    require(dataset in inventory.get("datasets", {}), f"dataset is absent from startup inventory: {dataset}")
    entry = inventory["datasets"][dataset]
    for warning in entry.get("warnings", []):
        print(f"INVENTORY_WARNING dataset={dataset} detail={warning}")
    issues = entry.get("issues", [])
    for issue in issues:
        print(f"INVENTORY_FAILURE dataset={dataset} detail={issue}")
    if issues:
        raise ValueError(f"startup metadata inventory has {len(issues)} blocker(s) for {dataset}")
    print(f"INVENTORY_DATASET_OK dataset={dataset} status={entry.get('status')}")


def extract_registry_digest(args: argparse.Namespace) -> None:
    path = pathlib.Path(args.log)
    require(path.is_file() and not path.is_symlink(), f"registration log must be a regular file: {path}")
    digests = re.findall(r"candidate_registry_digest=([0-9a-f]{64})(?:\s|$)", path.read_text(encoding="utf-8", errors="replace"))
    require(len(digests) == 1, f"registration must emit exactly one candidate registry digest, got {len(digests)}")
    print(digests[0])


def extract_inspection(args: argparse.Namespace) -> None:
    source = pathlib.Path(args.input)
    require(source.is_file() and not source.is_symlink(), f"inspector capture must be a regular file: {source}")
    candidates: list[dict[str, Any]] = []
    for line in source.read_text(encoding="utf-8", errors="strict").splitlines():
        try:
            value = json.loads(line, object_pairs_hook=_reject_duplicate_pairs)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(value, dict):
            candidates.append(value)
    require(len(candidates) == 1, f"inspector command must emit exactly one JSON object, got {len(candidates)}")
    atomic_write_json(pathlib.Path(args.output), candidates[0])
    print(f"REGISTRY_INSPECTION_CAPTURED output={args.output}")


def print_session_view(args: argparse.Namespace) -> None:
    project_root = pathlib.Path(args.project_root).resolve()
    if args.mode == "summary":
        print_matrix_summary(project_root)
    elif args.mode == "failures":
        print_matrix_summary(project_root, failures_only=True)
    elif args.mode == "follow":
        follow_global_log(project_root)
    elif args.mode == "follow-current":
        follow_current_log(project_root)
    else:
        raise ValueError(f"unknown matrix view: {args.mode}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    config = commands.add_parser("write-config")
    config.add_argument("--base", required=True, type=pathlib.Path)
    config.add_argument("--output", required=True, type=pathlib.Path)
    config.add_argument("--variant", required=True)
    config.add_argument("--export-dir", required=True)
    config.add_argument("--registry-path", required=True)
    config.add_argument("--registry-digest")

    registry = commands.add_parser("check-registry")
    registry.add_argument("--config", required=True)
    registry.add_argument("--project-root", required=True)
    registry.add_argument("--variant", required=True)
    registry.add_argument("--registry", required=True)
    registry.add_argument("--registry-digest", required=True)
    registry.add_argument("--inspection", required=True)

    cell = commands.add_parser("validate-cell")
    cell.add_argument("--project-root", required=True)
    cell.add_argument("--run-dir", required=True)
    cell.add_argument("--config", required=True)
    cell.add_argument("--variant", required=True)
    cell.add_argument("--dataset", required=True)
    cell.add_argument("--backend", required=True)
    cell.add_argument("--registry-digest", required=True)
    cell.add_argument("--registry", required=True)
    cell.add_argument("--inspection", required=True)
    cell.add_argument("--binary", required=True)
    cell.add_argument("--result-log", required=True)
    cell.add_argument("--receipt-path")
    cell.add_argument("--metrics-output")

    inventory = commands.add_parser("inventory")
    inventory.add_argument("--project-root", required=True)
    inventory.add_argument("--output", required=True)

    inventory_check = commands.add_parser("inventory-check")
    inventory_check.add_argument("--input", required=True)
    inventory_check.add_argument("--dataset", required=True, choices=DATASETS)

    registry_digest = commands.add_parser("extract-registry-digest")
    registry_digest.add_argument("--log", required=True)

    extract = commands.add_parser("extract-inspection")
    extract.add_argument("--input", required=True)
    extract.add_argument("--output", required=True)

    init = commands.add_parser("matrix-init")
    init.add_argument("--session-root", required=True)
    init.add_argument("--variants", required=True)
    init.add_argument("--worker-pid", required=True, type=int)

    update = commands.add_parser("matrix-update")
    update.add_argument("--session-root", required=True)
    update.add_argument("--worker-status", choices=("starting", "running", "complete", "finished_with_failures"))
    update.add_argument("--phase")
    update.add_argument("--session-failure-reason")
    update.add_argument("--session-failure-rc")
    update.add_argument("--inventory-path")
    update.add_argument("--build-status", choices=("running", "passed", "failed", "skipped"))
    update.add_argument("--variant")
    update.add_argument("--dataset")
    update.add_argument("--cell-status", choices=("pending", "running", "success", "failed"))
    update.add_argument("--cell-phase")
    update.add_argument("--reason")
    update.add_argument("--log")
    update.add_argument("--cell-log")
    update.add_argument("--registry-digest")
    update.add_argument("--command-rc")
    update.add_argument("--logger-rc")
    update.add_argument("--validation-rc")
    update.add_argument("--stage")
    update.add_argument("--stage-status")
    update.add_argument("--stage-command-rc")
    update.add_argument("--cell-logger-rc")
    update.add_argument("--global-logger-rc")
    update.add_argument("--stage-reason")
    update.add_argument("--metrics-json")
    update.add_argument("--evidence-path")
    update.add_argument("--proof-path")
    update.add_argument("--current-active", choices=("true", "false"))

    view = commands.add_parser("view")
    view.add_argument("--project-root", required=True)
    view.add_argument("--mode", required=True, choices=("summary", "failures", "follow", "follow-current"))

    build = commands.add_parser("check-build")
    build.add_argument("--project-root", required=True)
    build.add_argument("--build-dir", required=True)
    build.add_argument("--cuda", action="store_true")

    commands.add_parser("build-jobs")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "write-config":
            write_variant_config(args.base, args.output, args.variant, args.export_dir, args.registry_path, args.registry_digest)
        elif args.command == "check-registry":
            check_registry(args)
        elif args.command == "validate-cell":
            validate_cell(args)
        elif args.command == "inventory":
            write_inventory(args)
        elif args.command == "inventory-check":
            check_inventory_dataset(args)
        elif args.command == "extract-registry-digest":
            extract_registry_digest(args)
        elif args.command == "extract-inspection":
            extract_inspection(args)
        elif args.command == "matrix-init":
            init_matrix(args)
        elif args.command == "matrix-update":
            update_matrix(args)
        elif args.command == "view":
            print_session_view(args)
        elif args.command == "check-build":
            check_build(args)
        elif args.command == "build-jobs":
            print_build_jobs(args)
        else:
            raise ValueError(f"unknown helper command: {args.command}")
        return 0
    except Exception as error:
        print(f"ablation-entrypoint: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
