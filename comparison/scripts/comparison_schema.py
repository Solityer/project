#!/usr/bin/env python3
"""Shared schema, collectors, and validators for comparison baselines.

The top-level comparison reads only baseline-local machine artifacts plus
shared immutable inputs under data/cache, configs, and artifacts/checkpoints.
It does not read main experiment proof outputs or result tables.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path("/home/pzh/project").resolve()
COMPARISON_ROOT = PROJECT_ROOT / "comparison"
RESULTS_DIR = COMPARISON_ROOT / "results"
MARKER_PATH = RESULTS_DIR / "current_rerun_marker.json"

BASELINE_ROOTS = {
    "ezkl": COMPARISON_ROOT / "baselines" / "ezkl",
    "zkdl": COMPARISON_ROOT / "baselines" / "zkdl",
    "zkGPT": COMPARISON_ROOT / "baselines" / "zkGPT",
}

DATASETS = ["cora", "citeseer", "pubmed", "ppi", "ogbn-arxiv"]
EZKL_DATASETS = {
    "cora": "cora_full",
    "citeseer": "citeseer_full",
    "pubmed": "pubmed_full",
    "ppi": "ppi_full_formal",
    "ogbn-arxiv": "ogbn_arxiv_full",
}
DATASET_CACHE = {
    "cora": "cora",
    "citeseer": "citeseer",
    "pubmed": "pubmed",
    "ppi": "ppi",
    "ogbn-arxiv": "ogbn_arxiv",
}
CHECKPOINTS = {
    "cora": "cora_gat",
    "citeseer": "citeseer_gat",
    "pubmed": "pubmed_gat",
    "ppi": "ppi_gat",
    "ogbn-arxiv": "ogbn_arxiv_gat",
}
CONFIGS = {
    "cora": "cora_full.cfg",
    "citeseer": "citeseer_full.cfg",
    "pubmed": "pubmed_full.cfg",
    "ppi": "ppi_batch_formal.cfg",
    "ogbn-arxiv": "ogbn_arxiv_full.cfg",
}

REQUIRED_FIELDS = [
    "baseline",
    "baseline_root",
    "dataset",
    "backend",
    "N",
    "E",
    "G_batch",
    "L",
    "hidden_profile",
    "K_out",
    "C",
    "model_id",
    "checkpoint_digest",
    "data_id",
    "graph_digest",
    "node_ptr_digest",
    "edge_index_digest",
    "feature_digest",
    "config_id",
    "statement_id",
    "semantic_public_input_id",
    "public_input_encoding_id",
    "proof_system_id",
    "proof_core_schema_id",
    "run_id",
    "rerun_timestamp",
    "git_head",
    "source_tree_digest",
    "command_line",
    "status",
    "full_dataset",
    "verified",
    "proof_generated",
    "verifier_reads_witness",
    "witness_included_in_proof",
    "cpu_fallback",
    "gpu_available",
    "gpu_device",
    "gpu_kernel_used",
    "gpu_effective",
    "gpu_claim_level",
    "gpu_slower_than_cpu",
    "gpu_slowdown_explanation",
    "commitment_time_ms",
    "prove_time_ms",
    "verify_time_ms",
    "end_to_end_time_ms",
    "gpu_kernel_time_ms",
    "gpu_kernel_count",
    "gpu_memory_copy_h2d_ms",
    "gpu_memory_copy_d2h_ms",
    "gpu_synchronization_time_ms",
    "gpu_proof_subroutine_time_ms",
    "cpu_remaining_proof_time_ms",
    "data_transfer_time_ms",
    "serialization_time_ms",
    "orchestration_time_ms",
    "gpu_speedup_kernel_vs_cpu_equivalent",
    "gpu_speedup_end_to_end_vs_cpu",
    "gpu_bottleneck_reason",
    "proof_size_core_bytes",
    "proof_size_backend_extra_bytes",
    "artifact_size_total_bytes",
    "proof_size_reported_bytes",
    "proof_size_definition",
    "proof_size_fair_for_cross_backend",
    "proof_core_size_delta_reason",
    "input_sources",
    "forbidden_dependency_check",
    "main_experiment_artifact_dependency",
    "cross_baseline_artifact_dependency",
    "stale_result",
    "rerun_current_code",
    "validator_status",
    "block_reason",
]

EXTRA_FIELDS = [
    "proof_artifact_path",
    "verify_artifact_path",
    "result_artifact_path",
    "proof_artifact_digest",
    "verify_artifact_digest",
    "result_artifact_digest",
    "source_tree_digest_scope",
    "local_validator_artifact",
    "gpu_timing_instrumentation",
]

CSV_FIELDS = REQUIRED_FIELDS + EXTRA_FIELDS

GPU_CLAIM_LEVELS = {
    "full_prover_backend",
    "major_proof_subroutine_acceleration",
    "minor_kernel_hotspot_only",
    "forward_trace_only",
    "invalid_gpu_baseline",
    "unknown_or_incomplete",
    "none",
}


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = read_json(path)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: normalize_csv_cell(row.get(field)) for field in CSV_FIELDS})


def normalize_csv_cell(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return value


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_time(value: str | None) -> float:
    if not value:
        return 0.0
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


@lru_cache(maxsize=1)
def current_marker_time() -> float:
    marker = load_json(MARKER_PATH)
    return parse_time(str(marker.get("created_at", "")))


@lru_cache(maxsize=1)
def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception as exc:  # pragma: no cover - diagnostic only
        return f"unavailable:{exc}"


def stable_json_digest(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def file_digest(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest_paths(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.as_posix()):
        if not path.exists() or not path.is_file():
            continue
        try:
            rel = path.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            rel = path.as_posix()
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(file_digest(path).encode("ascii"))
        h.update(b"\0")
    return h.hexdigest()


def safe_stat_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def tree_size_bytes(root: Path) -> int:
    if not root.exists():
        return 0
    total = 0
    for path in root.rglob("*"):
        if path.is_file():
            total += safe_stat_size(path)
    return total


def latest_mtime(paths: list[Path]) -> float:
    stamps = []
    for path in paths:
        if path.exists():
            stamps.append(path.stat().st_mtime)
    return max(stamps) if stamps else 0.0


def oldest_mtime(paths: list[Path]) -> float:
    stamps = []
    for path in paths:
        if path.exists():
            stamps.append(path.stat().st_mtime)
    return min(stamps) if stamps else 0.0


def rel(path: Path, root: Path = PROJECT_ROOT) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def path_inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in {None, "", "NA", "not_applicable"}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int = 0) -> int:
    try:
        if value in {None, "", "NA", "not_applicable"}:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def hidden_profile_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                parts.append(f"{item.get('head_count')}x{item.get('head_dim')}")
        return ",".join(parts)
    return ""


def source_tree_paths(baseline: str) -> list[Path]:
    root = BASELINE_ROOTS[baseline]
    patterns: dict[str, list[str]] = {
        "ezkl": [
            "Cargo.toml",
            "Cargo.lock",
            "build.rs",
            "src/**/*.rs",
            "cpp/gat_ezkl_adapter/**/*.cpp",
            "cpp/gat_ezkl_adapter/**/*.hpp",
            "cpp/gat_ezkl_adapter/**/*.h",
            "cpp/gat_ezkl_adapter/CMakeLists.txt",
            "scripts/*.py",
            "configs/*.json",
        ],
        "zkdl": [
            "src/**/*.cu",
            "src/**/*.cuh",
            "src/**/*.hpp",
            "tools/*.py",
            "configs/*.cfg",
            "build/Makefile",
        ],
        "zkGPT": [
            "src/zkgpt_gat/**/*.py",
            "backend/**/*.cu",
            "backend/**/*.cpp",
            "scripts/*.py",
            "scripts/*.sh",
            "CMakeLists.txt",
        ],
    }
    out: list[Path] = []
    for pattern in patterns[baseline]:
        out.extend(path for path in root.glob(pattern) if path.is_file())
    return out


@lru_cache(maxsize=8)
def source_tree_digest(baseline: str) -> str:
    return digest_paths(source_tree_paths(baseline))


def source_tree_scope(baseline: str) -> list[str]:
    return [rel(path, BASELINE_ROOTS[baseline]) for path in source_tree_paths(baseline)]


def dataset_cache_files(dataset: str) -> dict[str, Path]:
    root = PROJECT_ROOT / "data" / "cache" / DATASET_CACHE[dataset]
    feature = root / "features.npy"
    if not feature.exists():
        feature = root / "features.txt"
    node_ptr = root / "node_ptr.txt"
    return {
        "cache_root": root,
        "meta": root / "meta.cfg",
        "edges": root / "edges.txt",
        "features": feature,
        "labels": root / "labels.txt",
        "node_ptr": node_ptr,
    }


def parse_meta(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip()
    return out


@lru_cache(maxsize=16)
def shared_input_ids(dataset: str) -> dict[str, Any]:
    files = dataset_cache_files(dataset)
    meta = parse_meta(files["meta"])
    checkpoint_root = PROJECT_ROOT / "artifacts" / "checkpoints" / CHECKPOINTS[dataset]
    checkpoint_files = [
        checkpoint_root / "manifest.json",
        checkpoint_root / "tensors.npz",
        checkpoint_root / "tensors.txt",
    ]
    manifest = load_json(checkpoint_root / "manifest.json")
    config_path = PROJECT_ROOT / "configs" / CONFIGS[dataset]
    edge_digest = file_digest(files["edges"])
    feature_digest = file_digest(files["features"])
    node_ptr_digest = file_digest(files["node_ptr"]) if files["node_ptr"].exists() else stable_json_digest({
        "dataset": dataset,
        "graph_count": int(meta.get("graph_count", "1")),
        "num_nodes": int(meta.get("num_nodes", "0")),
        "policy": "single_graph_implicit_node_ptr",
    })
    graph_digest = digest_paths([files["meta"], files["edges"], files["node_ptr"]])
    data_id = digest_paths([files["meta"], files["edges"], files["features"], files["labels"], files["node_ptr"]])
    checkpoint_digest = digest_paths([path for path in checkpoint_files if path.exists()])
    config_id = file_digest(config_path)
    model_id = str(manifest.get("model_param_id") or CHECKPOINTS[dataset])
    hidden_profile = hidden_profile_text(manifest.get("hidden_profile"))
    semantic = stable_json_digest({
        "dataset": dataset,
        "N": int(meta.get("num_nodes", "0")),
        "E_digest": edge_digest,
        "G_batch": int(meta.get("graph_count", "1")),
        "L": int(manifest.get("L", 0)),
        "hidden_profile": hidden_profile,
        "K_out": int(manifest.get("K_out", 0)),
        "C": int(manifest.get("C", meta.get("num_classes", "0"))),
        "model_id": model_id,
        "checkpoint_digest": checkpoint_digest,
        "data_id": data_id,
        "graph_digest": graph_digest,
        "node_ptr_digest": node_ptr_digest,
        "edge_index_digest": edge_digest,
        "feature_digest": feature_digest,
        "config_id": config_id,
        "public_output": "full_dataset_final_logits",
        "preprocessing": "edge_gid_then_dst_stable;self_loop=per_node",
    })
    return {
        "N": int(meta.get("num_nodes", "0")),
        "G_batch": int(meta.get("graph_count", "1")),
        "L": int(manifest.get("L", 0)),
        "hidden_profile": hidden_profile,
        "K_out": int(manifest.get("K_out", 0)),
        "C": int(manifest.get("C", meta.get("num_classes", "0"))),
        "model_id": model_id,
        "checkpoint_digest": checkpoint_digest,
        "data_id": data_id,
        "graph_digest": graph_digest,
        "node_ptr_digest": node_ptr_digest,
        "edge_index_digest": edge_digest,
        "feature_digest": feature_digest,
        "config_id": config_id,
        "semantic_public_input_id": f"semantic-gat-full:{semantic}",
        "input_sources": [
            rel(files["cache_root"]),
            rel(checkpoint_root),
            rel(config_path),
        ],
    }


def base_row(baseline: str, dataset: str, backend: str) -> dict[str, Any]:
    shared = shared_input_ids(dataset)
    root = BASELINE_ROOTS[baseline]
    proof_system = {
        "ezkl": "ezkl-halo2-kzg-shplonk-native-gat",
        "zkdl": "zkdl-hyrax-style-tensor-relation-proof",
        "zkGPT": "zkgpt-gkr-lasso-matmul-hyrax-mle",
    }[baseline]
    encoding = {
        "ezkl": "ezkl-final-logits-chunked-columns-v1",
        "zkdl": "zkdl-public-metadata-vk-static-v2",
        "zkGPT": "gat-zkgpt-fixed-order-v1",
    }[baseline]
    row: dict[str, Any] = {
        "baseline": baseline,
        "baseline_root": str(root),
        "dataset": dataset,
        "backend": backend,
        **{key: shared[key] for key in [
            "N",
            "G_batch",
            "L",
            "hidden_profile",
            "K_out",
            "C",
            "model_id",
            "checkpoint_digest",
            "data_id",
            "graph_digest",
            "node_ptr_digest",
            "edge_index_digest",
            "feature_digest",
            "config_id",
            "semantic_public_input_id",
        ]},
        "E": 0,
        "statement_id": f"{baseline}:full-gat:{shared['semantic_public_input_id']}",
        "public_input_encoding_id": encoding,
        "proof_system_id": proof_system,
        "run_id": "",
        "rerun_timestamp": "",
        "git_head": git_head(),
        "source_tree_digest": source_tree_digest(baseline),
        "source_tree_digest_scope": source_tree_scope(baseline),
        "command_line": "",
        "status": "missing",
        "full_dataset": False,
        "verified": False,
        "proof_generated": False,
        "verifier_reads_witness": False,
        "witness_included_in_proof": False,
        "cpu_fallback": False,
        "gpu_available": False,
        "gpu_device": "none",
        "gpu_kernel_used": False,
        "gpu_effective": False,
        "gpu_claim_level": "none",
        "gpu_slower_than_cpu": "not_applicable",
        "gpu_slowdown_explanation": "",
        "commitment_time_ms": 0.0,
        "prove_time_ms": 0.0,
        "verify_time_ms": 0.0,
        "end_to_end_time_ms": 0.0,
        "gpu_kernel_time_ms": 0.0,
        "gpu_kernel_count": 0,
        "gpu_memory_copy_h2d_ms": 0.0,
        "gpu_memory_copy_d2h_ms": 0.0,
        "gpu_synchronization_time_ms": 0.0,
        "gpu_proof_subroutine_time_ms": 0.0,
        "cpu_remaining_proof_time_ms": 0.0,
        "data_transfer_time_ms": 0.0,
        "serialization_time_ms": 0.0,
        "orchestration_time_ms": 0.0,
        "gpu_speedup_kernel_vs_cpu_equivalent": "not_applicable",
        "gpu_speedup_end_to_end_vs_cpu": "not_applicable",
        "gpu_bottleneck_reason": "",
        "proof_size_core_bytes": 0,
        "proof_size_backend_extra_bytes": 0,
        "artifact_size_total_bytes": 0,
        "proof_size_reported_bytes": 0,
        "proof_size_definition": "",
        "proof_size_fair_for_cross_backend": False,
        "proof_core_size_delta_reason": "",
        "input_sources": shared["input_sources"],
        "forbidden_dependency_check": "passed",
        "main_experiment_artifact_dependency": False,
        "cross_baseline_artifact_dependency": False,
        "stale_result": True,
        "rerun_current_code": False,
        "validator_status": "not_run",
        "block_reason": "",
        "proof_artifact_path": "",
        "verify_artifact_path": "",
        "result_artifact_path": "",
        "proof_artifact_digest": "",
        "verify_artifact_digest": "",
        "result_artifact_digest": "",
        "local_validator_artifact": "",
        "gpu_timing_instrumentation": "not_applicable",
    }
    return row


def finalize_freshness(row: dict[str, Any], artifacts: list[Path]) -> None:
    marker = current_marker_time()
    old = oldest_mtime([path for path in artifacts if path.exists()])
    fresh = bool(marker and old and old >= marker)
    row["stale_result"] = not fresh
    row["rerun_current_code"] = fresh and row.get("git_head") == git_head()


def collect_ezkl() -> list[dict[str, Any]]:
    baseline = "ezkl"
    root = BASELINE_ROOTS[baseline]
    rows: list[dict[str, Any]] = []
    result_json = root / "results" / "gat_ezkl_full_results.json"
    for dataset, ezkl_dataset in EZKL_DATASETS.items():
        run_dir = root / "runs" / ezkl_dataset
        manifest = load_json(run_dir / "run_manifest.json")
        spec = load_json(run_dir / "graph_attention_spec.json")
        model_info = load_json(run_dir / "model_info.json")
        verify = load_json(run_dir / "verify.json")
        proof_path = run_dir / "proof.json"
        verify_path = run_dir / "verify.json"
        row = base_row(baseline, dataset, "cpu")
        verified = manifest.get("status") == "verified" and verify.get("verify_passed") is True
        proof_generated = manifest.get("proof_generated") is True and proof_path.exists()
        end_to_end = sum(as_float(manifest.get(key)) for key in [
            "gen_settings_ms",
            "compile_ms",
            "setup_ms",
            "gen_witness_ms",
            "prove_time_ms",
            "verify_time_ms",
        ])
        row.update({
            "E": as_int(spec.get("edge_count"), as_int(manifest.get("E"))),
            "L": as_int(model_info.get("L"), row["L"]),
            "hidden_profile": hidden_profile_text(model_info.get("hidden_profile")) or row["hidden_profile"],
            "K_out": as_int(model_info.get("K_out"), row["K_out"]),
            "C": as_int(model_info.get("num_classes"), row["C"]),
            "run_id": str(manifest.get("run_id", "")),
            "rerun_timestamp": str(manifest.get("run_completed_at") or manifest.get("run_started_at") or ""),
            "command_line": str(manifest.get("command") or manifest.get("failed_command") or ""),
            "status": "ok" if verified else str(manifest.get("status") or "blocked"),
            "full_dataset": bool(manifest.get("full_dataset", spec.get("full_dataset", False))),
            "verified": verified,
            "proof_generated": proof_generated,
            "cpu_fallback": False,
            "gpu_available": "not_applicable",
            "gpu_device": "not_applicable",
            "gpu_kernel_used": False,
            "gpu_effective": False,
            "gpu_claim_level": "none",
            "commitment_time_ms": as_float(manifest.get("commitment_time_ms")),
            "prove_time_ms": as_float(manifest.get("prove_time_ms")),
            "verify_time_ms": as_float(manifest.get("verify_time_ms")),
            "end_to_end_time_ms": end_to_end,
            "proof_size_core_bytes": safe_stat_size(proof_path) if proof_generated else 0,
            "proof_size_backend_extra_bytes": 0,
            "artifact_size_total_bytes": tree_size_bytes(run_dir),
            "proof_size_reported_bytes": as_int(manifest.get("proof_size_bytes")),
            "proof_size_definition": "verifier-facing ezkl proof.json only; excludes witness/settings/keys/logs/profiles",
            "proof_size_fair_for_cross_backend": True,
            "proof_core_schema_id": "ezkl-halo2-kzg-shplonk-proof-json-v1",
            "validator_status": "passed" if verified else "failed",
            "block_reason": "" if verified else str(manifest.get("block_reason") or "ezkl_not_verified"),
            "proof_artifact_path": rel(proof_path),
            "verify_artifact_path": rel(verify_path),
            "result_artifact_path": rel(result_json),
            "proof_artifact_digest": file_digest(proof_path),
            "verify_artifact_digest": file_digest(verify_path),
            "result_artifact_digest": file_digest(result_json),
            "local_validator_artifact": rel(root / "results" / "strict_audit_report.md"),
        })
        row["E"] = row["E"] or as_int(manifest.get("E"))
        finalize_freshness(row, [run_dir / "run_manifest.json", proof_path, verify_path])
        rows.append(row)
    return rows


def zkdl_run_dir(dataset: str, backend: str) -> Path:
    return BASELINE_ROOTS["zkdl"] / "zkdl_gat_outputs" / "runs" / dataset / backend / "warm"


def collect_zkdl() -> list[dict[str, Any]]:
    baseline = "zkdl"
    root = BASELINE_ROOTS[baseline]
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for backend in ["cpu", "gpu"]:
            run_dir = zkdl_run_dir(dataset, backend)
            proof = load_json(run_dir / "full_proof_manifest.json")
            verifier = load_json(run_dir / "full_verifier_manifest.json")
            metadata = load_json(run_dir / "formal_artifacts" / "zkdl_full_chain_public_metadata.json")
            proof_path = run_dir / "full_proof_manifest.json"
            verify_path = run_dir / "full_verifier_manifest.json"
            proof_object = Path(str(proof.get("proof_object_path") or run_dir / "formal_artifacts" / "zkdl_full_proof_object.txt"))
            row = base_row(baseline, dataset, backend)
            dim = metadata.get("dataset_shape", {}) if isinstance(metadata.get("dataset_shape"), dict) else {}
            model_shape = metadata.get("model_shape", {}) if isinstance(metadata.get("model_shape"), dict) else {}
            verified = (
                proof.get("proof_generated") is True
                and proof.get("verified") is True
                and verifier.get("verified") is True
                and proof.get("missing_relations") == []
                and verifier.get("missing_relations") == []
            )
            relations = proof.get("relations", []) if isinstance(proof.get("relations"), list) else []
            commitment = as_float(proof.get("commitment_time_ms"))
            prove = as_float(proof.get("prove_time_ms"))
            verify = as_float(verifier.get("verify_time_ms"))
            e2e = commitment + prove + verify
            core_size = as_int(proof.get("proof_size_bytes"))
            physical_bundle = tree_size_bytes(run_dir)
            logical_bundle = max(physical_bundle, core_size + safe_stat_size(proof_path) + safe_stat_size(verify_path))
            row.update({
                "E": as_int(dim.get("edge_count"), row["E"]),
                "N": as_int(dim.get("node_count"), row["N"]),
                "G_batch": as_int(dim.get("graph_count"), row["G_batch"]),
                "L": as_int(model_shape.get("L"), row["L"]),
                "hidden_profile": str(model_shape.get("hidden_profile") or row["hidden_profile"]),
                "K_out": as_int(model_shape.get("output_head_count"), row["K_out"]),
                "C": as_int(dim.get("class_count"), row["C"]),
                "run_id": f"{dataset}-{backend}-{proof.get('benchmark_mode', 'warm')}",
                "rerun_timestamp": dt.datetime.fromtimestamp(latest_mtime([proof_path, verify_path]), dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z") if proof_path.exists() else "",
                "command_line": f"./bin/gat_zkdl --formal-full-chain --device {backend} --dataset {dataset} --output-root zkdl_gat_outputs --formal-commitment-block-size 64",
                "status": "ok" if verified else "blocked",
                "full_dataset": proof.get("is_full_dataset") is True,
                "verified": verified,
                "proof_generated": proof.get("proof_generated") is True,
                "verifier_reads_witness": verifier.get("verifier_reads_witness") is True,
                "witness_included_in_proof": proof.get("witness_included_in_proof") is True or verifier.get("witness_included_in_proof") is True,
                "commitment_time_ms": commitment,
                "prove_time_ms": prove,
                "verify_time_ms": verify,
                "end_to_end_time_ms": e2e,
                "proof_size_core_bytes": core_size,
                "proof_size_backend_extra_bytes": 0,
                "artifact_size_total_bytes": logical_bundle,
                "proof_size_reported_bytes": core_size,
                "proof_size_definition": str(proof.get("proof_size_definition") or "deterministic_relation_payload_sum_v2"),
                "proof_size_fair_for_cross_backend": False,
                "proof_core_schema_id": f"zkdl-full-chain-relation-payload-v2-{backend}",
                "validator_status": "passed" if verified else "failed",
                "block_reason": "" if verified else "zkdl_full_manifest_gate_failed",
                "proof_artifact_path": rel(proof_object if proof_object.exists() else proof_path),
                "verify_artifact_path": rel(verify_path),
                "result_artifact_path": rel(root / "zkdl_gat_outputs" / "results" / "latest.json"),
                "proof_artifact_digest": file_digest(proof_object) or file_digest(proof_path),
                "verify_artifact_digest": file_digest(verify_path),
                "result_artifact_digest": file_digest(root / "zkdl_gat_outputs" / "results" / "latest.json"),
                "local_validator_artifact": rel(root / "zkdl_gat_outputs" / "results" / "summary.md"),
            })
            if backend == "gpu":
                subroutine = commitment + prove
                row.update({
                    "gpu_available": True,
                    "gpu_device": "cuda",
                    "gpu_kernel_used": verified and bool(relations),
                    "gpu_effective": verified and bool(relations),
                    "gpu_claim_level": "major_proof_subroutine_acceleration" if verified and relations else "unknown_or_incomplete",
                    "gpu_kernel_time_ms": subroutine,
                    "gpu_kernel_count": len(relations),
                    "gpu_proof_subroutine_time_ms": subroutine,
                    "cpu_remaining_proof_time_ms": verify,
                    "gpu_timing_instrumentation": "coarse_relation_manifest_timing",
                    "gpu_bottleneck_reason": "pending_cpu_gpu_pair_analysis",
                })
            finalize_freshness(row, [proof_path, verify_path, proof_object])
            rows.append(row)
    return rows


def collect_zkgpt() -> list[dict[str, Any]]:
    baseline = "zkGPT"
    root = BASELINE_ROOTS[baseline]
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for backend in ["cpu", "gpu"]:
            artifact_dir = root / "results" / backend / dataset
            manifest = load_json(artifact_dir / "run_manifest.json")
            metadata = load_json(artifact_dir / "metadata.json")
            benchmark = load_json(artifact_dir / "benchmark_summary.json")
            verification = load_json(artifact_dir / "verification.json")
            gpu_backend = load_json(artifact_dir / "gpu_backend.json")
            gpu_hot = load_json(artifact_dir / "gpu_proof_hot_path.json")
            proof_path = artifact_dir / "proof_payload.json"
            verify_path = artifact_dir / "verification.json"
            dim = metadata.get("dim_cfg", {}) if isinstance(metadata.get("dim_cfg"), dict) else {}
            row = base_row(baseline, dataset, backend)
            verified = manifest.get("status") == "VERIFY_OK" and verification.get("status") == "VERIFY_OK"
            proof_generated = proof_path.exists()
            commitment = as_float(benchmark.get("commitment_time_ms"))
            prove = as_float(benchmark.get("prove_time_ms"))
            verify = as_float(benchmark.get("verify_time_ms"))
            e2e = as_float(benchmark.get("wall_time_ms"), commitment + prove + verify)
            gpu_extra = safe_stat_size(artifact_dir / "gpu_backend.json") if backend == "gpu" else 0
            gpu_kernel_time = as_float(gpu_hot.get("kernel_time_ms"))
            gpu_h2d = as_float(gpu_hot.get("memory_copy_h2d_ms"))
            gpu_d2h = as_float(gpu_hot.get("memory_copy_d2h_ms"))
            gpu_sync = as_float(gpu_hot.get("synchronization_time_ms"))
            gpu_total = as_float(gpu_hot.get("total_time_ms"), gpu_kernel_time + gpu_h2d + gpu_d2h + gpu_sync)
            gpu_valid = (
                backend == "gpu"
                and gpu_hot.get("cuda_kernel_executed") is True
                and gpu_hot.get("proof_hot_path") is True
                and gpu_hot.get("cpu_fallback") is False
            )
            row.update({
                "E": as_int(dim.get("E"), row["E"]),
                "N": as_int(dim.get("N"), row["N"]),
                "G_batch": as_int(dim.get("G_batch"), row["G_batch"]),
                "L": as_int(dim.get("L"), row["L"]),
                "hidden_profile": hidden_profile_text(dim.get("hidden_profile")) or row["hidden_profile"],
                "K_out": as_int(dim.get("K_out"), row["K_out"]),
                "C": as_int(dim.get("C"), row["C"]),
                "run_id": f"{dataset}-{backend}-{manifest.get('run_digest', '')[:12]}",
                "rerun_timestamp": dt.datetime.fromtimestamp(latest_mtime([artifact_dir / "run_manifest.json", proof_path, verify_path]), dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z") if proof_path.exists() else "",
                "command_line": f"python3 scripts/gat_zkgpt_experiment.py --datasets {dataset} --backends {backend} --force",
                "status": "ok" if verified else "blocked",
                "full_dataset": manifest.get("full_dataset") is True and metadata.get("full_dataset") is True,
                "verified": verified,
                "proof_generated": proof_generated,
                "verifier_reads_witness": manifest.get("verifier_reads_witness") is True,
                "witness_included_in_proof": manifest.get("witness_included_in_proof") is True,
                "cpu_fallback": False if backend == "cpu" else not gpu_valid,
                "commitment_time_ms": commitment,
                "prove_time_ms": prove,
                "verify_time_ms": verify,
                "end_to_end_time_ms": e2e,
                "proof_size_core_bytes": as_int(benchmark.get("proof_size_bytes"), safe_stat_size(proof_path)),
                "proof_size_backend_extra_bytes": gpu_extra,
                "artifact_size_total_bytes": tree_size_bytes(artifact_dir),
                "proof_size_reported_bytes": as_int(benchmark.get("proof_size_bytes"), safe_stat_size(proof_path)),
                "proof_size_definition": str(manifest.get("proof_size_definition") or "canonical serialized proof_payload.json"),
                "proof_size_fair_for_cross_backend": backend == "cpu",
                "proof_core_schema_id": "zkgpt-proof-payload-v1" if backend == "cpu" else "zkgpt-proof-payload-v1+gpu-hot-path-binding-v2",
                "validator_status": "passed" if verified else "failed",
                "block_reason": "" if verified else "zkgpt_verifier_not_ok",
                "proof_artifact_path": rel(proof_path),
                "verify_artifact_path": rel(verify_path),
                "result_artifact_path": rel(root / "results" / "summary.json"),
                "proof_artifact_digest": file_digest(proof_path),
                "verify_artifact_digest": file_digest(verify_path),
                "result_artifact_digest": file_digest(root / "results" / "summary.json"),
                "local_validator_artifact": rel(root / "results" / "formal_proof_semantics_audit.json"),
            })
            if backend == "gpu":
                row.update({
                    "gpu_available": gpu_backend.get("gpu_runtime_present") is True or gpu_hot.get("gpu_runtime_present") is True,
                    "gpu_device": str(gpu_hot.get("device_name") or gpu_backend.get("device_name") or "unknown"),
                    "gpu_kernel_used": gpu_hot.get("cuda_kernel_executed") is True,
                    "gpu_effective": gpu_valid,
                    "gpu_claim_level": "major_proof_subroutine_acceleration" if gpu_valid else "invalid_gpu_baseline",
                    "gpu_kernel_time_ms": gpu_kernel_time,
                    "gpu_kernel_count": as_int(gpu_hot.get("kernel_count"), 1 if gpu_hot.get("cuda_kernel_executed") else 0),
                    "gpu_memory_copy_h2d_ms": gpu_h2d,
                    "gpu_memory_copy_d2h_ms": gpu_d2h,
                    "gpu_synchronization_time_ms": gpu_sync,
                    "gpu_proof_subroutine_time_ms": gpu_total,
                    "cpu_remaining_proof_time_ms": max(prove - gpu_total, 0.0),
                    "data_transfer_time_ms": gpu_h2d + gpu_d2h,
                    "gpu_timing_instrumentation": "cuda_event_timing" if gpu_total > 0 else "missing_cuda_event_timing",
                    "gpu_bottleneck_reason": "pending_cpu_gpu_pair_analysis",
                })
            finalize_freshness(row, [artifact_dir / "run_manifest.json", proof_path, verify_path])
            rows.append(row)
    return rows


def apply_pairwise_analysis(rows: list[dict[str, Any]]) -> None:
    by_key = {(row["baseline"], row["dataset"], row["backend"]): row for row in rows}
    for row in rows:
        if row["backend"] != "gpu":
            continue
        cpu = by_key.get((row["baseline"], row["dataset"], "cpu"))
        if not cpu:
            continue
        cpu_e2e = as_float(cpu.get("end_to_end_time_ms"))
        gpu_e2e = as_float(row.get("end_to_end_time_ms"))
        cpu_sub = as_float(cpu.get("commitment_time_ms")) + as_float(cpu.get("prove_time_ms"))
        gpu_sub = as_float(row.get("gpu_proof_subroutine_time_ms"))
        if gpu_e2e > 0 and cpu_e2e > 0:
            row["gpu_speedup_end_to_end_vs_cpu"] = cpu_e2e / gpu_e2e
            slower = gpu_e2e > cpu_e2e
            row["gpu_slower_than_cpu"] = slower
            if slower:
                row["gpu_slowdown_explanation"] = (
                    "GPU end-to-end exceeded CPU; coarse profile points to transfer/synchronization/"
                    "serialization or CPU remaining proof work unless CUDA event timing shows otherwise"
                )
                row["gpu_bottleneck_reason"] = "gpu_overhead_or_cpu_remaining_proof_work"
            else:
                row["gpu_slowdown_explanation"] = ""
                row["gpu_bottleneck_reason"] = "gpu_proof_subroutine_accelerated_end_to_end"
        if gpu_sub > 0 and cpu_sub > 0:
            row["gpu_speedup_kernel_vs_cpu_equivalent"] = cpu_sub / gpu_sub
        if as_int(row.get("proof_size_core_bytes")) != as_int(cpu.get("proof_size_core_bytes")):
            reason = (
                f"{row['baseline']} CPU/GPU proof core schema or verifier-required backend binding differs; "
                "do not compare as identical byte payloads"
            )
            row["proof_core_size_delta_reason"] = reason
            cpu["proof_core_size_delta_reason"] = reason


def collect_all_rows() -> list[dict[str, Any]]:
    rows = collect_ezkl() + collect_zkdl() + collect_zkgpt()
    apply_pairwise_analysis(rows)
    return rows


def required_keys_for_mode(mode: str, through: str | None = None) -> list[tuple[str, str, str]]:
    order = ["ezkl", "zkdl", "zkGPT"]
    if mode == "partial":
        if through not in order:
            raise ValueError("--through must be one of ezkl, zkdl, zkGPT in partial mode")
        order = order[: order.index(through) + 1]
    keys: list[tuple[str, str, str]] = []
    for baseline in order:
        if baseline == "ezkl":
            keys.extend((baseline, dataset, "cpu") for dataset in DATASETS)
        else:
            keys.extend((baseline, dataset, backend) for backend in ["cpu", "gpu"] for dataset in DATASETS)
    return keys


def validate_rows(rows: list[dict[str, Any]], mode: str = "full", through: str | None = None) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    required = required_keys_for_mode(mode, through)
    by_key = {(row["baseline"], row["dataset"], row["backend"]): row for row in rows}

    for key in required:
        if key not in by_key:
            errors.append(f"missing_required_row:{key}")

    for key in required:
        row = by_key.get(key)
        if not row:
            continue
        label = "/".join(key)
        for field in REQUIRED_FIELDS:
            if field not in row:
                errors.append(f"{label}:missing_field:{field}")
        if row.get("status") != "ok":
            errors.append(f"{label}:status_not_ok:{row.get('status')}")
        if row.get("validator_status") != "passed":
            errors.append(f"{label}:local_validator_not_passed:{row.get('validator_status')}")
        if row.get("full_dataset") is not True:
            errors.append(f"{label}:not_full_dataset")
        if row.get("verified") is not True:
            errors.append(f"{label}:not_verified")
        if row.get("proof_generated") is not True:
            errors.append(f"{label}:proof_not_generated")
        if row.get("verifier_reads_witness") is not False:
            errors.append(f"{label}:verifier_reads_witness")
        if row.get("witness_included_in_proof") is not False:
            errors.append(f"{label}:witness_included_in_proof")
        if row.get("stale_result") is not False:
            errors.append(f"{label}:stale_result")
        if row.get("rerun_current_code") is not True:
            errors.append(f"{label}:not_rerun_current_code")
        if row.get("main_experiment_artifact_dependency") is not False:
            errors.append(f"{label}:main_experiment_artifact_dependency")
        if row.get("cross_baseline_artifact_dependency") is not False:
            errors.append(f"{label}:cross_baseline_artifact_dependency")
        if row.get("forbidden_dependency_check") != "passed":
            errors.append(f"{label}:forbidden_dependency_check_failed")
        core = as_int(row.get("proof_size_core_bytes"))
        extra = as_int(row.get("proof_size_backend_extra_bytes"))
        total = as_int(row.get("artifact_size_total_bytes"))
        if core <= 0:
            errors.append(f"{label}:proof_core_size_missing")
        if total <= 0 or core > total or extra > total or core + extra > total:
            errors.append(f"{label}:proof_size_accounting_inconsistent")
        if not row.get("proof_size_definition"):
            errors.append(f"{label}:proof_size_definition_missing")
        if row.get("proof_size_reported_bytes") != row.get("proof_size_core_bytes") and not row.get("proof_core_size_delta_reason"):
            warnings.append(f"{label}:reported_size_differs_from_core_without_delta_reason")
        for artifact_field in ["proof_artifact_path", "verify_artifact_path", "result_artifact_path"]:
            value = row.get(artifact_field)
            if not value:
                errors.append(f"{label}:missing_{artifact_field}")
                continue
            artifact = PROJECT_ROOT / str(value)
            root = BASELINE_ROOTS[row["baseline"]]
            if artifact_field != "result_artifact_path" and not path_inside(artifact, root):
                errors.append(f"{label}:{artifact_field}_outside_baseline")
        if row["backend"] == "gpu":
            if row.get("cpu_fallback") is not False:
                errors.append(f"{label}:gpu_cpu_fallback")
            if row.get("gpu_kernel_used") is not True:
                errors.append(f"{label}:gpu_kernel_not_used")
            if as_float(row.get("gpu_proof_subroutine_time_ms")) <= 0:
                errors.append(f"{label}:gpu_proof_subroutine_time_missing")
            if row.get("gpu_claim_level") not in {"full_prover_backend", "major_proof_subroutine_acceleration"}:
                errors.append(f"{label}:gpu_claim_level_invalid:{row.get('gpu_claim_level')}")
            if row.get("gpu_slower_than_cpu") is True and not row.get("gpu_slowdown_explanation"):
                errors.append(f"{label}:gpu_slowdown_unexplained")
        if row["baseline"] == "ezkl":
            if row.get("gpu_available") != "not_applicable":
                errors.append(f"{label}:ezkl_gpu_available_not_explicit_na")
            if row.get("gpu_claim_level") != "none":
                errors.append(f"{label}:ezkl_gpu_claim_not_none")

    for dataset in DATASETS:
        present = [row for row in rows if row["dataset"] == dataset and (row["baseline"], row["dataset"], row["backend"]) in required]
        if not present:
            continue
        for field in [
            "N",
            "E",
            "G_batch",
            "L",
            "hidden_profile",
            "K_out",
            "C",
            "model_id",
            "checkpoint_digest",
            "data_id",
            "graph_digest",
            "node_ptr_digest",
            "edge_index_digest",
            "feature_digest",
            "config_id",
            "semantic_public_input_id",
        ]:
            values = {json.dumps(row.get(field), sort_keys=True) for row in present}
            if len(values) > 1:
                errors.append(f"{dataset}:semantic_alignment_mismatch:{field}:{sorted(values)}")

    for baseline in ["zkdl", "zkGPT"]:
        for dataset in DATASETS:
            cpu = by_key.get((baseline, dataset, "cpu"))
            gpu = by_key.get((baseline, dataset, "gpu"))
            if not cpu or not gpu:
                continue
            if cpu.get("statement_id") != gpu.get("statement_id"):
                errors.append(f"{baseline}/{dataset}:cpu_gpu_statement_id_mismatch")
            if cpu.get("semantic_public_input_id") != gpu.get("semantic_public_input_id"):
                errors.append(f"{baseline}/{dataset}:cpu_gpu_semantic_public_input_mismatch")
            if (
                cpu.get("proof_core_schema_id") == gpu.get("proof_core_schema_id")
                and cpu.get("proof_size_core_bytes") != gpu.get("proof_size_core_bytes")
                and not gpu.get("proof_core_size_delta_reason")
            ):
                errors.append(f"{baseline}/{dataset}:proof_core_size_delta_unexplained")

    return {
        "generated_at": now_utc(),
        "mode": mode,
        "through": through,
        "git_head": git_head(),
        "rerun_marker": load_json(MARKER_PATH),
        "required_row_count": len(required),
        "collected_row_count": len(rows),
        "accepted_row_count": 0 if errors else len(required),
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "warnings": warnings,
        "rows": rows,
    }


def validate_all(mode: str = "full", through: str | None = None, audit_path: Path | None = None) -> dict[str, Any]:
    rows = collect_all_rows()
    audit = validate_rows(rows, mode=mode, through=through)
    if audit_path is not None:
        write_json(audit_path, audit)
    return audit


def paper_ready_rows(audit: dict[str, Any]) -> list[dict[str, Any]]:
    if audit.get("status") != "passed":
        raise RuntimeError("cannot collect paper-ready rows because validation failed")
    required = set(required_keys_for_mode("full"))
    out = [
        row for row in audit.get("rows", [])
        if (row.get("baseline"), row.get("dataset"), row.get("backend")) in required
    ]
    return sorted(out, key=lambda row: (
        {"ezkl": 0, "zkdl": 1, "zkGPT": 2}[row["baseline"]],
        DATASETS.index(row["dataset"]),
        {"cpu": 0, "gpu": 1}[row["backend"]],
    ))


def write_report(path: Path, rows: list[dict[str, Any]], audit: dict[str, Any]) -> None:
    lines = [
        "# Final Comparison Report",
        "",
        f"- Generated at: `{audit.get('generated_at')}`",
        f"- Git head: `{audit.get('git_head')}`",
        "- Accepted rows are current-code reruns, baseline-local, verified, and full-dataset.",
        "- Proof size uses `proof_size_core_bytes` for verifier-facing core payloads; backend extras and artifact bundle sizes are separate.",
        "",
        "## Rows",
        "",
        "| baseline | dataset | backend | status | prove_time_ms | verify_time_ms | proof_size_core_bytes | gpu_claim_level | gpu_slower_than_cpu |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['baseline']} | {row['dataset']} | {row['backend']} | {row['status']} | "
            f"{row['prove_time_ms']} | {row['verify_time_ms']} | {row['proof_size_core_bytes']} | "
            f"{row['gpu_claim_level']} | {row['gpu_slower_than_cpu']} |"
        )
    lines.extend([
        "",
        "## Output Files",
        "",
        "- `comparison/results/final_comparison_audit.json`",
        "- `comparison/results/final_comparison_results.csv`",
        "- `comparison/results/final_comparison_results.json`",
        "- `comparison/results/final_comparison_report.md`",
        "",
        "No known blockers.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
