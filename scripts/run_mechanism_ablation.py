#!/usr/bin/env python3
"""Run and audit the fixed real-data CUDA zkGAT mechanism experiment.

Formal inputs are prepared by the native loader/checkpoint/quantized-forward
path. The runner refuses synthetic/custom workloads and records a separate
proof plus strict CPU verifier receipt for every Original/Control cell.
"""

from __future__ import annotations

import argparse
import atexit
import csv
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import re
import shlex
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, BinaryIO


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runs" / "mechanism_ablation_real"
MECHANISMS = ("feature_folding", "shared_neighborhood")
VARIANTS = ("original", "control")
METRIC_FIELDS = (
    "commitment_time_ms",
    "prove_time_ms",
    "verify_time_ms",
    "proof_size_bytes",
)
HEX_256 = re.compile(r"^[0-9a-f]{64}$")
CANONICAL_UINT = re.compile(r"^(0|[1-9][0-9]*)$")
POWER_OF_TWO_LOWER_BOUND = re.compile(r"^2\^([0-9]+)$")
RUN_FORMAT = "zkgat_mechanism_result_v1"
TIMING_SCHEMA = "mechanism_commitment_disjoint_verify_parse_accept_v2"
REAL_INPUT_SCHEMA = "zkgat_real_mechanism_input_v1"
EXPERIMENT_REPETITIONS = 3
FORMAL_WIDTH = 8
DATASETS: dict[str, dict[str, Any]] = {
    "cora": {
        "config": "configs/cora_secure.cfg", "srs": "artifacts/srs/cora_formal_srs",
        "srs_id": "cora_formal_srs", "N": 2708, "E": 13264, "n_domain": 4096,
        "e_domain": 32768, "arch": "gat_family_L3_hid8x8-8x8_kout2_c7",
        "params": "cora_planetoid_seed11", "preprocess": "row_normalize",
    },
    "citeseer": {
        "config": "configs/citeseer_secure.cfg", "srs": "artifacts/srs/citeseer_formal_srs",
        "srs_id": "citeseer_formal_srs", "N": 3327, "E": 12431, "n_domain": 4096,
        "e_domain": 32768, "arch": "gat_family_L3_hid8x8-8x8_kout2_c6",
        "params": "citeseer_planetoid_seed11", "preprocess": "row_normalize",
    },
    "ppi": {
        "config": "configs/ppi_secure.cfg", "srs": "artifacts/srs/ppi_formal_srs",
        "srs_id": "ppi_formal_srs", "n_domain": 65536, "e_domain": 8388608,
        "arch": "gat_family_L3_hid8x8-8x8_kout2_c121",
        "params": "ppi_gat_seed11", "preprocess": "row_normalize",
        "component_shapes": {0: (1546, 17699), 9: (1377, 16229)},
        "components": {1: (1546, 17699, (0,)), 2: (2923, 33928, (0, 9))},
    },
}
PPI_COMPONENT_RULE = "ascending canonical component ID with N>=3 and at least one node having two distinct off-self incoming neighbors"
FORMAL_CONFIGS: tuple[dict[str, Any], ...] = (
    {"mechanism": "feature_folding", "dataset": "cora", "width": 1, "graphs": 1, "components": None},
    {"mechanism": "feature_folding", "dataset": "cora", "width": 2, "graphs": 1, "components": None},
    {"mechanism": "feature_folding", "dataset": "cora", "width": 4, "graphs": 1, "components": None},
    {"mechanism": "feature_folding", "dataset": "cora", "width": 8, "graphs": 1, "components": None},
    {"mechanism": "feature_folding", "dataset": "citeseer", "width": 8, "graphs": 1, "components": None},
    {"mechanism": "feature_folding", "dataset": "ppi", "width": 8, "graphs": 1, "components": (0,)},
    {"mechanism": "feature_folding", "dataset": "ppi", "width": 8, "graphs": 2, "components": (0, 9)},
    {"mechanism": "shared_neighborhood", "dataset": "cora", "width": 8, "graphs": 1, "components": None},
    {"mechanism": "shared_neighborhood", "dataset": "citeseer", "width": 8, "graphs": 1, "components": None},
    {"mechanism": "shared_neighborhood", "dataset": "ppi", "width": 8, "graphs": 1, "components": (0,)},
    {"mechanism": "shared_neighborhood", "dataset": "ppi", "width": 8, "graphs": 2, "components": (0, 9)},
)
if len(FORMAL_CONFIGS) != 11 or len({(x["mechanism"], x["dataset"], x["width"], x["graphs"]) for x in FORMAL_CONFIGS}) != 11:
    raise RuntimeError("internal formal mechanism matrix must contain exactly 11 unique configurations")
TIMING_SEMANTICS = {
    "commitment_time_ms": "cold native KZG commitment work only, including first-use static-base preparation invoked inside commitment; Taylor-shift coefficient materialization is excluded",
    "prove_time_ms": "native online proof work, Taylor-shift coefficient materialization, and canonical proof encoding, excluding KZG commitment work and artifact file I/O",
    "verify_time_ms": "independent verifier parsing through accept/reject",
    "proof_size_bytes": "actual canonical serialized mechanism proof bytes",
}
ACTIVE_SECRET_FILES: set[pathlib.Path] = set()
ACTIVE_CHILDREN: set[int] = set()
_INTERRUPT_SIGNAL: int | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def cleanup_secret_files() -> None:
    for path in list(ACTIVE_SECRET_FILES):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
    ACTIVE_SECRET_FILES.clear()


atexit.register(cleanup_secret_files)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def atomic_write_text(path: pathlib.Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_json(path: pathlib.Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def emit(message: str, session_log: pathlib.Path | None = None) -> None:
    print(message, flush=True)
    if session_log is not None:
        with session_log.open("a", encoding="utf-8") as stream:
            stream.write(f"{utc_now()} {message}\n")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"JSON has duplicate field {key!r}")
        result[key] = value
    return result


def load_json(path: pathlib.Path, context: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"{context} is missing or not a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_pairs)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"cannot parse {context}: {exc}") from exc
    require(isinstance(value, dict), f"{context} must be a JSON object")
    return value


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_contains_bytes(path: pathlib.Path, needle: bytes) -> bool:
    overlap = b""
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            block = overlap + chunk
            if needle in block:
                return True
            overlap = block[-(len(needle) - 1):] if len(needle) > 1 else b""
    return False


def load_build_provenance_module() -> Any:
    path = PROJECT_ROOT / "tools" / "build_provenance.py"
    spec = importlib.util.spec_from_file_location("mechanism_runner_build_provenance", path)
    require(spec is not None and spec.loader is not None, f"cannot load build provenance helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git_output(*args: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout


def source_snapshot() -> dict[str, Any]:
    provenance = load_build_provenance_module()
    manifest = provenance.source_manifest_document(PROJECT_ROOT)
    diff = git_output("diff", "--binary", "HEAD")
    status = git_output("status", "--porcelain=v1", "--untracked-files=all")
    return {
        "git_head": git_output("rev-parse", "HEAD").decode("ascii").strip(),
        "git_diff_sha256": sha256_bytes(diff),
        "git_status_sha256": sha256_bytes(status),
        "source_inventory_id": manifest["inventory_id"],
        "source_tree_hash": manifest["source_tree_hash"],
        "source_file_count": manifest["file_count"],
        "source_manifest": manifest,
    }


def integer_arg(value: str) -> int:
    try:
        parsed = int(value, 10)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def resolve_cuda_binary(value: pathlib.Path | None) -> pathlib.Path:
    if value is not None:
        path = value.expanduser().resolve()
    else:
        path = (PROJECT_ROOT / "build_cuda" / "gatzk_mechanism").resolve()
    require(path.is_file() and os.access(path, os.X_OK), f"mechanism executable is missing or not executable: {path}")
    return path


def build_snapshot(binary: pathlib.Path, backend: str) -> dict[str, Any]:
    build_root = binary.parent
    cache_path = build_root / "CMakeCache.txt"
    source_manifest_path = build_root / "generated" / "gatzk" / "source_manifest.json"
    require(cache_path.is_file(), f"build cache is missing: {cache_path}")
    require(source_manifest_path.is_file(), f"build source manifest is missing: {source_manifest_path}")
    cache_bytes = cache_path.read_bytes()
    cache_text = cache_bytes.decode("utf-8", errors="replace")
    cache_values: dict[str, str] = {}
    for line in cache_text.splitlines():
        if line and not line.startswith("//") and not line.startswith("#") and ":" in line and "=" in line:
            key, value = line.split("=", 1)
            cache_values[key.split(":", 1)[0]] = value
    manifest = load_json(source_manifest_path, "build source manifest")
    if backend == "cuda":
        require(cache_values.get("GATZK_ENABLE_CUDA_BACKEND") == "ON", "selected build was not configured with CUDA support")
    return {
        "build_root": str(build_root),
        "cmake_cache_sha256": sha256_bytes(cache_bytes),
        "cmake_cache_size_bytes": len(cache_bytes),
        "cmake_generator": cache_values.get("CMAKE_GENERATOR"),
        "build_type": cache_values.get("CMAKE_BUILD_TYPE"),
        "cxx_compiler": cache_values.get("CMAKE_CXX_COMPILER"),
        "cxx_compiler_id": cache_values.get("CMAKE_CXX_COMPILER_ID"),
        "cuda_enabled": cache_values.get("GATZK_ENABLE_CUDA_BACKEND"),
        "binary_path": str(binary),
        "binary_sha256": sha256_file(binary),
        "binary_size_bytes": binary.stat().st_size,
        "source_manifest_path": str(source_manifest_path),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "built_source_tree_hash": manifest.get("source_tree_hash"),
        "source_manifest": manifest,
    }


def srs_snapshot(path: pathlib.Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    require(resolved.is_dir() and not path.is_symlink(), f"--srs must be a regular SRS directory: {path}")
    manifest_path = resolved / "srs_manifest.json"
    require(manifest_path.is_file() and not manifest_path.is_symlink(), f"SRS manifest is missing: {manifest_path}")
    manifest = load_json(manifest_path, "SRS manifest")
    raw = manifest_path.read_bytes()
    require(manifest.get("manifest_format") == "zkgat_srs_manifest", "selected SRS has an unsupported manifest format")
    require(manifest.get("srs_type") == "trusted_srs", "selected SRS is not registered as a trusted SRS")
    require(manifest.get("security_level") == "120-bit", "selected SRS does not declare the 120-bit security target")
    require(manifest.get("srs_id"), "selected SRS manifest has no id")
    require(manifest.get("protocol_schema_id"), "selected SRS manifest has no protocol schema identity")
    ceremony_hash = manifest.get("ceremony_transcript_hash")
    require(isinstance(ceremony_hash, str) and HEX_256.fullmatch(ceremony_hash),
            "selected SRS has no trusted ceremony transcript digest")
    require(manifest.get("setup_transcript_hash") == ceremony_hash,
            "selected SRS setup transcript does not match its trusted ceremony transcript")
    require(isinstance(manifest.get("manifest_hash"), str) and HEX_256.fullmatch(manifest["manifest_hash"]),
            "selected SRS manifest has no canonical manifest digest")
    for key in ("pk_coef_path", "pk_coef_hash", "vk_path", "vk_hash", "srs_audit_report_path", "srs_audit_report_hash"):
        require(key in manifest, f"selected SRS manifest is missing required trusted material field {key}")
    for path_key, hash_key in (("pk_coef_path", "pk_coef_hash"), ("vk_path", "vk_hash"),
                               ("srs_audit_report_path", "srs_audit_report_hash")):
        item = resolved / str(manifest[path_key])
        canonical_item = item.resolve()
        require(canonical_item.is_relative_to(resolved),
                f"SRS trusted material escapes the registered directory: {item}")
        require(item.is_file() and not item.is_symlink(), f"SRS trusted material is missing or not regular: {item}")
        require(isinstance(manifest[hash_key], str) and HEX_256.fullmatch(manifest[hash_key]),
                f"SRS trusted material has an invalid digest: {hash_key}")
        require(sha256_file(canonical_item) == manifest[hash_key],
                f"SRS trusted material digest does not match its registered bytes: {item}")
    lagrange_paths = manifest.get("pk_lagrange_paths")
    lagrange_hashes = manifest.get("pk_lagrange_hashes")
    require(isinstance(lagrange_paths, dict) and isinstance(lagrange_hashes, dict),
            "selected SRS manifest has no Lagrange material registry")
    domains = manifest.get("domain_sizes")
    generators = manifest.get("domain_generators")
    require(isinstance(domains, dict) and domains, "selected SRS manifest has no registered domain-size catalog")
    require(isinstance(generators, dict) and set(generators) == set(domains),
            "selected SRS domain generator catalog is incomplete or inconsistent")
    require(set(lagrange_paths) == set(domains) and set(lagrange_hashes) == set(domains),
            "selected SRS Lagrange files do not cover the complete registered domain catalog")
    for domain, domain_size in domains.items():
        require(isinstance(domain, str) and domain and type(domain_size) is int and domain_size > 0,
                f"selected SRS has an invalid registered domain: {domain!r}")
        require(domain_size & (domain_size - 1) == 0,
                f"selected SRS domain {domain} is not a power of two")
        require(isinstance(generators[domain], str) and HEX_256.fullmatch(generators[domain]),
                f"selected SRS domain {domain} has no trusted generator identity")
        item = resolved / str(lagrange_paths[domain])
        canonical_item = item.resolve()
        require(canonical_item.is_relative_to(resolved),
                f"SRS {domain} Lagrange material escapes the registered directory: {item}")
        require(item.is_file() and not item.is_symlink(), f"SRS {domain} Lagrange material is missing or not regular: {item}")
        require(isinstance(lagrange_hashes[domain], str) and HEX_256.fullmatch(lagrange_hashes[domain]),
                f"SRS {domain} Lagrange material digest is invalid")
        require(sha256_file(canonical_item) == lagrange_hashes[domain],
                f"SRS {domain} Lagrange material digest does not match its registered bytes")
    envelope = manifest.get("max_instance_envelope")
    require(isinstance(envelope, dict), "selected SRS manifest has no trusted instance envelope")
    degree_bound = manifest.get("degree_bound")
    require(type(degree_bound) is int and degree_bound > 0, "selected SRS manifest has an invalid degree bound")
    return {
        "path": str(resolved),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_bytes(raw),
        "manifest_size_bytes": len(raw),
        "manifest_srs_id": manifest.get("srs_id"),
        "manifest_protocol_schema_id": manifest.get("protocol_schema_id"),
        "manifest_degree_bound": degree_bound,
        "manifest_manifest_hash": manifest.get("manifest_hash"),
        "srs_type": manifest.get("srs_type"),
        "security_level": manifest.get("security_level"),
        "ceremony_transcript_hash": ceremony_hash,
        "domain_sizes": domains,
        "max_instance_envelope": envelope,
        "manifest": manifest,
    }


def dataset_paths(dataset: str) -> tuple[pathlib.Path, pathlib.Path]:
    require(dataset in DATASETS, f"unknown formal dataset: {dataset}")
    entry = DATASETS[dataset]
    return PROJECT_ROOT / entry["config"], PROJECT_ROOT / entry["srs"]


def read_config(path: pathlib.Path) -> dict[str, str]:
    require(path.is_file() and not path.is_symlink(), f"formal config is missing or not regular: {path}")
    values: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        require("=" in line, f"malformed formal config row at {path}:{line_number}")
        key, value = (part.strip() for part in line.split("=", 1))
        require(key not in values, f"duplicate formal config key {key!r} at {path}:{line_number}")
        values[key] = value
    return values


def config_snapshot(dataset: str) -> dict[str, Any]:
    config_path, srs_path = dataset_paths(dataset)
    config = read_config(config_path)
    expected = DATASETS[dataset]
    require(config.get("dataset") == dataset, f"{config_path} does not declare dataset={dataset}")
    require(config.get("srs_type") == "trusted_srs", f"{config_path} does not select a trusted SRS")
    require(config.get("srs_id") == expected["srs_id"], f"{config_path} selects the wrong registered SRS id")
    require((PROJECT_ROOT / config.get("srs_path", "")).resolve() == srs_path.resolve(),
            f"{config_path} SRS path differs from the canonical registered SRS path")
    require(config.get("model_arch_id") == expected["arch"], f"{config_path} architecture id differs from the formal registry")
    require(config.get("model_param_id") == expected["params"], f"{config_path} checkpoint identity differs from the formal registry")
    require(config.get("input_preprocess_id") == expected["preprocess"],
            f"{config_path} preprocessing id differs from the formal registry")
    checkpoint_value = config.get("checkpoint_bundle")
    require(isinstance(checkpoint_value, str) and checkpoint_value, f"{config_path} has no checkpoint bundle path")
    checkpoint_path = (PROJECT_ROOT / checkpoint_value).resolve()
    require(checkpoint_path.is_dir() and not checkpoint_path.is_symlink(),
            f"canonical checkpoint bundle is missing or not regular: {checkpoint_path}")
    return {
        "dataset": dataset,
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "config_values": config,
        "checkpoint_path": str(checkpoint_path),
        "srs": srs_snapshot(srs_path),
    }


def resolve_evidence_file(value: str, label: str) -> pathlib.Path:
    candidate = pathlib.Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    require(candidate.is_file() and not candidate.is_symlink(), f"{label} is missing or not a regular file: {candidate}")
    return candidate.resolve()


def validate_hash(value: Any, label: str) -> str:
    require(isinstance(value, str) and HEX_256.fullmatch(value), f"{label} must be a lowercase SHA-256 digest")
    return value


def canonical_inventory_sha256(entries: list[dict[str, str]]) -> str:
    ordered = sorted(entries, key=lambda item: item["path"])
    digest = hashlib.sha256()
    for item in ordered:
        digest.update(item["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(item["sha256"].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def recursive_file_inventory(roots: list[pathlib.Path], label: str) -> list[dict[str, str]]:
    """Hash the canonical regular-file inventory under roots, resolving symlinks."""
    files: dict[str, pathlib.Path] = {}
    visited_directories: set[str] = set()

    def visit(directory: pathlib.Path) -> None:
        canonical = directory.resolve(strict=True)
        require(canonical.is_dir(), f"{label} inventory root is not a directory: {directory}")
        key = str(canonical)
        if key in visited_directories:
            return
        visited_directories.add(key)
        try:
            children = sorted(canonical.iterdir(), key=lambda item: item.name)
        except OSError as exc:
            raise ValueError(f"cannot enumerate {label} inventory directory {canonical}: {exc}") from exc
        for child in children:
            target = child.resolve(strict=True)
            if target.is_dir():
                visit(target)
            elif target.is_file():
                files[str(target)] = target
            else:
                raise ValueError(f"{label} inventory contains a non-regular filesystem entry: {child}")

    for root in roots:
        require(root.exists(), f"{label} inventory root is missing: {root}")
        visit(root)
    return [{"path": path, "sha256": sha256_file(files[path])} for path in sorted(files)]


def validate_real_metadata(
    metadata: dict[str, Any], *, dataset: str, components: tuple[int, ...] | None,
    bundle_path: pathlib.Path, metadata_path: pathlib.Path, config_info: dict[str, Any],
) -> dict[str, Any]:
    """Validate that the serialized real bundle came from the fixed formal loader path."""
    expected = DATASETS[dataset]
    require(metadata.get("schema") == REAL_INPUT_SCHEMA, "real-input metadata schema is not supported")
    require(metadata.get("dataset") == dataset, "real-input metadata dataset identity differs from its config")
    require(pathlib.Path(str(metadata.get("config_path", ""))).resolve() == pathlib.Path(config_info["config_path"]),
            "real-input metadata was prepared from a different formal config")
    require(validate_hash(metadata.get("config_sha256"), "real-input config_sha256") == config_info["config_sha256"],
            "real-input metadata config hash differs from the canonical formal config")
    config_values = config_info["config_values"]
    for field in ("input_preprocess_id", "model_arch_id", "model_param_id", "quant_cfg_id", "static_table_id"):
        require(isinstance(metadata.get(field), str) and metadata[field] == config_values.get(field),
                f"real-input metadata {field} differs from the formal config")
    require(metadata.get("layer") == 0 and metadata.get("head") == 0,
            "real mechanism input must be formal hidden layer 0, attention head 0")
    require(metadata.get("formal_head_width") == FORMAL_WIDTH,
            "real mechanism source head must retain its formal width 8")

    checkpoint = metadata.get("checkpoint_bundle")
    require(isinstance(checkpoint, dict), "real-input metadata has no checkpoint bundle identity")
    require(pathlib.Path(str(checkpoint.get("path", ""))).resolve() == pathlib.Path(config_info["checkpoint_path"]),
            "real-input metadata checkpoint bundle path differs from the formal config")
    validate_hash(checkpoint.get("sha256"), "checkpoint bundle sha256")
    checkpoint_files = checkpoint.get("files")
    require(isinstance(checkpoint_files, list) and checkpoint_files, "checkpoint bundle metadata has no file-level digest inventory")
    checkpoint_file_paths: set[str] = set()
    checkpoint_records: list[dict[str, str]] = []
    for item in checkpoint_files:
        require(isinstance(item, dict), "malformed checkpoint file identity")
        file_path = resolve_evidence_file(str(item.get("path", "")), "checkpoint file")
        digest = validate_hash(item.get("sha256"), "checkpoint file sha256")
        require(sha256_file(file_path) == digest, f"checkpoint file hash mismatch: {file_path}")
        require(file_path.is_relative_to(pathlib.Path(config_info["checkpoint_path"])),
                f"checkpoint file escapes the canonical checkpoint bundle: {file_path}")
        checkpoint_file_paths.add(str(file_path))
        checkpoint_records.append({"path": str(file_path), "sha256": digest})
    require(checkpoint_file_paths, "checkpoint bundle digest inventory is empty")
    require(len(checkpoint_file_paths) == len(checkpoint_records), "checkpoint digest inventory contains duplicate files")
    require(checkpoint_records == sorted(checkpoint_records, key=lambda item: item["path"]),
            "checkpoint file inventory is not in canonical path order")
    checkpoint_expected_records = recursive_file_inventory(
        [pathlib.Path(config_info["checkpoint_path"])], "checkpoint",
    )
    require(checkpoint_records == checkpoint_expected_records,
            "checkpoint file inventory does not cover the complete canonical checkpoint bundle")
    checkpoint_bundle_sha256 = canonical_inventory_sha256(checkpoint_records)
    require(checkpoint["sha256"] == checkpoint_bundle_sha256,
            "checkpoint bundle SHA-256 differs from the canonical checkpoint file inventory")
    checkpoint_root = pathlib.Path(config_info["checkpoint_path"]).resolve()
    checkpoint = {
        **checkpoint,
        "path": str(checkpoint_root),
        "sha256": checkpoint_bundle_sha256,
        "files": checkpoint_records,
    }

    data_sources = metadata.get("data_sources")
    require(isinstance(data_sources, list) and data_sources, "real-input metadata has no data-source digest inventory")
    data_source_records: list[dict[str, str]] = []
    for item in data_sources:
        require(isinstance(item, dict), "malformed real dataset source identity")
        file_path = resolve_evidence_file(str(item.get("path", "")), "real dataset source")
        digest = validate_hash(item.get("sha256"), "real data source sha256")
        require(sha256_file(file_path) == digest, f"real dataset source hash mismatch: {file_path}")
        data_source_records.append({"path": str(file_path), "sha256": digest})
    require(len({item["path"] for item in data_source_records}) == len(data_source_records),
            "real dataset source inventory contains duplicate canonical paths")
    require(data_source_records == sorted(data_source_records, key=lambda item: item["path"]),
            "real dataset source inventory is not in canonical path order")
    config_values = config_info["config_values"]
    source_roots = sorted({
        (PROJECT_ROOT / config_values[key]).resolve()
        for key in ("data_root", "cache_root")
        if isinstance(config_values.get(key), str) and config_values[key]
    }, key=str)
    require(len(source_roots) == 2,
            "formal config must identify both data_root and cache_root for source inventory validation")
    expected_data_sources = recursive_file_inventory(source_roots, "real dataset source")
    require(data_source_records == expected_data_sources,
            "real dataset source inventory does not cover the complete configured data/cache roots")
    source_manifest_sha256 = canonical_inventory_sha256(data_source_records)

    bundle = metadata.get("bundle")
    require(isinstance(bundle, dict), "real-input metadata has no bundle receipt")
    bundle_resolved = pathlib.Path(str(bundle.get("path", ""))).resolve()
    require(bundle_resolved == bundle_path.resolve(), "real-input metadata points at a different bundle path")
    bundle_sha256 = validate_hash(bundle.get("sha256"), "real input bundle sha256")
    require(bundle_path.is_file() and not bundle_path.is_symlink(), "prepared real-input bundle is missing or not regular")
    require(sha256_file(bundle_path) == bundle_sha256, "prepared real-input bundle hash differs from its metadata")
    witness_sha256 = validate_hash(metadata.get("witness_sha256"), "real witness sha256")

    actual = metadata.get("actual")
    require(isinstance(actual, dict), "real-input metadata has no actual graph size")
    graphs = len(components) if dataset == "ppi" and components else 1
    if dataset == "ppi":
        expected_nodes, expected_edges, ids = expected["components"][graphs]
    else:
        expected_nodes, expected_edges, ids = expected["N"], expected["E"], (0,)
    require({key: actual.get(key) for key in ("N", "E", "G")} ==
            {"N": expected_nodes, "E": expected_edges, "G": graphs},
            f"actual {dataset} graph counts {actual!r} differ from the pre-registered full graph/component selection")
    selected = metadata.get("selected_components")
    require(isinstance(selected, list) and len(selected) == graphs,
            "real-input metadata selected-component count differs from actual G")
    node_ptr = metadata.get("node_ptr")
    edge_ptr = metadata.get("edge_ptr")
    require(isinstance(node_ptr, list) and isinstance(edge_ptr, list), "real-input metadata omits graph boundary arrays")
    require(len(node_ptr) == graphs + 1 and len(edge_ptr) == graphs + 1,
            "real-input graph boundary arrays do not match actual G")
    require(all(type(v) is int for v in node_ptr + edge_ptr), "real-input graph boundaries must be integer offsets")
    require(node_ptr[0] == 0 and edge_ptr[0] == 0 and node_ptr[-1] == expected_nodes and edge_ptr[-1] == expected_edges,
            "real-input graph boundaries do not span the complete selected graph")
    computed_node_ptr = [0]
    computed_edge_ptr = [0]
    seen_ids: list[int] = []
    selected_records: list[dict[str, Any]] = []
    for index, record in enumerate(selected):
        require(isinstance(record, dict), "malformed selected component metadata")
        component_id = record.get("canonical_id")
        require(type(component_id) is int and component_id == ids[index],
                "selected component IDs differ from the fixed canonical component rule")
        component_nodes, component_edges = record.get("N"), record.get("E")
        require(type(component_nodes) is int and component_nodes >= 3,
                f"selected component {component_id} has an invalid node count")
        require(type(component_edges) is int and component_edges > 0,
                f"selected component {component_id} has an invalid edge count")
        if dataset == "ppi":
            require((component_nodes, component_edges) == expected["component_shapes"][component_id],
                    f"PPI component {component_id} counts differ from the fixed complete-component identity")
        node_begin, node_end = record.get("absolute_node_begin"), record.get("absolute_node_end")
        require(type(node_begin) is int and type(node_end) is int and node_begin >= 0 and node_end - node_begin == component_nodes,
                f"selected component {component_id} has an invalid original node boundary")
        stats = record.get("nonself_neighbor_stats")
        require(isinstance(stats, dict), f"selected component {component_id} has no non-self neighbor statistics")
        eligible = stats.get("eligible_nodes")
        minimum, median, maximum = stats.get("min"), stats.get("median"), stats.get("max")
        require(type(eligible) is int and eligible >= 0, f"selected component {component_id} has invalid eligible-node count")
        require(type(minimum) is int and minimum >= 0 and type(maximum) is int and maximum >= minimum,
                f"selected component {component_id} has invalid neighbor degree extrema")
        require(type(median) in (int, float) and math.isfinite(float(median)) and minimum <= median <= maximum,
                f"selected component {component_id} has invalid median neighbor count")
        if dataset == "ppi":
            require(eligible >= 1 and maximum >= 2,
                    f"PPI component {component_id} does not meet the nontrivial-neighborhood selection rule")
        computed_node_ptr.append(computed_node_ptr[-1] + component_nodes)
        computed_edge_ptr.append(computed_edge_ptr[-1] + component_edges)
        seen_ids.append(component_id)
        selected_records.append(record)
    require(node_ptr == computed_node_ptr and edge_ptr == computed_edge_ptr,
            "real-input graph boundary arrays disagree with selected component N/E")
    require(sum(item["N"] for item in selected_records) == expected_nodes and
            sum(item["E"] for item in selected_records) == expected_edges,
            "selected component counts do not sum to actual graph N/E")
    if dataset == "ppi":
        require(tuple(seen_ids) == tuple(ids), "PPI component ordering is not canonical ascending component ID")
    else:
        require(seen_ids == [0] and selected_records[0]["absolute_node_begin"] == 0 and
                selected_records[0]["absolute_node_end"] == expected_nodes,
                "Cora/Citeseer input must retain the complete original graph")

    scales = metadata.get("scales")
    require(isinstance(scales, dict), "real-input metadata has no official quantization scales")
    scale_names = ("S_input", "S_weight", "S_hidden", "S_exp", "S_delta")
    require(all(name in scales for name in scale_names), "real-input metadata omits a formal quantization scale")
    for name in scale_names:
        value = scales[name]
        require(type(value) in (int, float) and math.isfinite(float(value)) and value > 0,
                f"real-input metadata has invalid quantization scale {name}")

    srs_expectation = metadata.get("srs_expectation")
    require(isinstance(srs_expectation, dict), "real-input metadata has no SRS domain/capacity expectation")
    require(srs_expectation.get("dataset_id") == dataset,
            "real-input metadata SRS expectation names a different dataset")
    require(srs_expectation.get("srs_id") == expected["srs_id"],
            "real-input metadata expects a different registered SRS id")
    require(srs_expectation.get("n_domain") == expected["n_domain"] and
            srs_expectation.get("e_domain") == expected["e_domain"],
            "real-input metadata requested unregistered N/E domains")
    required_coefficients = 2 * expected["e_domain"] + 14
    require(srs_expectation.get("required_coefficient_count") == required_coefficients,
            "real-input coefficient prefix requirement differs from the registered mechanism shape")
    srs_manifest = config_info["srs"]["manifest"]
    require(srs_manifest["domain_sizes"].get("N") == expected["n_domain"] and
            srs_manifest["domain_sizes"].get("edge") == expected["e_domain"],
            "trusted SRS does not register the exact required N/E domains")
    require(srs_manifest["degree_bound"] + 1 >= required_coefficients,
            "trusted SRS coefficient prefix is too short for the required ledger/work domains")
    envelope = srs_manifest["max_instance_envelope"]
    require(envelope.get("max_nodes", 0) >= expected_nodes and envelope.get("max_edges", 0) >= expected_edges and
            envelope.get("max_graphs", 0) >= graphs,
            "trusted SRS maximum instance envelope does not include the selected full graph workload")

    return {
        "schema": REAL_INPUT_SCHEMA, "dataset": dataset,
        "config_path": config_info["config_path"], "config_sha256": config_info["config_sha256"],
        "input_preprocess_id": metadata["input_preprocess_id"], "model_arch_id": metadata["model_arch_id"],
        "model_param_id": metadata["model_param_id"], "quant_cfg_id": metadata["quant_cfg_id"],
        "static_table_id": metadata["static_table_id"], "checkpoint_bundle": checkpoint,
        "checkpoint_bundle_sha256": checkpoint_bundle_sha256,
        "checkpoint_root": str(checkpoint_root),
        "data_sources": data_source_records, "bundle_path": str(bundle_path.resolve()),
        "bundle_sha256": bundle_sha256, "metadata_path": str(metadata_path.resolve()),
        "metadata_sha256": sha256_file(metadata_path),
        "witness_sha256": witness_sha256, "layer": 0, "head": 0,
        "formal_head_width": FORMAL_WIDTH, "actual": {"N": expected_nodes, "E": expected_edges, "G": graphs},
        "components": tuple(seen_ids), "selected_components": selected_records,
        "node_ptr": node_ptr, "edge_ptr": edge_ptr, "scales": scales,
        "srs_expectation": srs_expectation,
        "source_manifest_sha256": source_manifest_sha256,
        "data_identity_sha256": source_manifest_sha256,
        "source_roots": [str(path) for path in source_roots],
    }


def cuda_snapshot(device: int) -> dict[str, Any]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,name,driver_version", "--format=csv,noheader,nounits"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    require(result.returncode == 0, f"cannot identify CUDA devices with nvidia-smi: {result.stderr.strip()}")
    devices: list[dict[str, str]] = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", 3)]
        require(len(fields) == 4, f"malformed nvidia-smi device row: {line!r}")
        devices.append({"index": fields[0], "uuid": fields[1], "name": fields[2], "driver_version": fields[3]})
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or visible.strip() == "":
        selected_token = str(device)
    else:
        tokens = [token.strip() for token in visible.split(",")]
        require(device < len(tokens), f"CUDA device index {device} is outside CUDA_VISIBLE_DEVICES={visible!r}")
        selected_token = tokens[device]
    selected = next(
        (
            item for item in devices
            if item["index"] == selected_token
            or item["uuid"] == selected_token
            or item["uuid"].startswith(selected_token)
        ),
        None,
    )
    require(selected is not None, f"selected CUDA device token {selected_token!r} is not present in nvidia-smi inventory")
    return {
        "backend": "cuda",
        "selected_device_index": device,
        "selected_device_token": selected_token,
        "selected_device": selected,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "inventory": devices,
    }


def assert_no_external_cuda_compute(device_snapshot: dict[str, Any]) -> None:
    selected = device_snapshot.get("selected_device")
    require(isinstance(selected, dict) and isinstance(selected.get("uuid"), str),
            "selected CUDA device has no stable UUID for compute-activity checks")
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader,nounits"],
        check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    require(result.returncode == 0,
            f"cannot inspect external CUDA compute activity: {result.stderr.strip()}")
    active: list[dict[str, str]] = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        require(len(fields) == 3, f"malformed CUDA compute-process row: {line!r}")
        if fields[1] == selected["uuid"]:
            active.append({"pid": fields[0], "gpu_uuid": fields[1], "used_memory": fields[2]})
    require(not active,
            "selected CUDA device already has external compute processes; refusing to share the formal experiment GPU: "
            + json.dumps(active, sort_keys=True))


def acquire_cuda_lock(device: int) -> BinaryIO:
    DEFAULT_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock_path = DEFAULT_OUTPUT_ROOT / f".cuda-device-{device}.lock"
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(lock_path, flags, 0o600)
    stream = os.fdopen(fd, "r+b", buffering=0)
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        stream.close()
        raise ValueError(f"CUDA device {device} is locked by another mechanism experiment runner") from exc
    stream.seek(0)
    stream.truncate()
    stream.write(f"pid={os.getpid()} started_utc={utc_now()}\n".encode("ascii"))
    os.fsync(stream.fileno())
    return stream


def safe_output_root(value: pathlib.Path) -> pathlib.Path:
    allowed_root = DEFAULT_OUTPUT_ROOT.resolve()
    candidate = value.expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    require(not candidate.is_symlink(), "output root cannot be a symlink")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(allowed_root)
    except ValueError as exc:
        raise ValueError(f"output root must be inside the isolated {allowed_root} tree") from exc
    forbidden = (PROJECT_ROOT / "results" / "paper_secure").resolve()
    require(resolved != forbidden and forbidden not in resolved.parents and resolved not in forbidden.parents,
            "mechanism output cannot overlap results/paper_secure")
    if resolved != allowed_root:
        require(not resolved.exists(), "custom output root must be new; existing run directories are immutable")
    return resolved


def create_session_dir(output_root: pathlib.Path) -> tuple[str, pathlib.Path]:
    output_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for _ in range(8):
        session_id = f"mechanism_{timestamp}_{uuid.uuid4().hex[:12]}"
        session_root = output_root / session_id
        try:
            session_root.mkdir(mode=0o700)
            return session_id, session_root
        except FileExistsError:
            continue
    raise ValueError(f"cannot reserve unique session directory under {output_root}")


def validate_security_receipt(security: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate each variant's own exact budget and extract shared policy fields.

    The detailed challenge, opening-game, and identity counts are deliberately
    retained per variant. They are evidence for each ledger, not pair-equality
    fields: the mechanism changes those counts by design.
    """
    require(type(security.get("target_bits")) is int and security["target_bits"] == 120,
            "mechanism result does not use the required 120-bit security target")
    require(security.get("budget_exact") is True, "mechanism security budget is not exact")
    require(security.get("bound_kind") == "conservative_exact", "mechanism security budget is not conservatively exact")
    ledger_hash = security.get("budget_ledger_sha256")
    require(isinstance(ledger_hash, str) and HEX_256.fullmatch(ledger_hash), "invalid security budget ledger digest")

    assumptions = security.get("assumptions")
    require(isinstance(assumptions, dict), "mechanism security receipt has no assumption profile")
    computational_bits = security.get("computational_assumption_bits")
    kzg_bits = assumptions.get("kzg_assumption_bits")
    require(type(computational_bits) is int and computational_bits >= 120,
            "mechanism security receipt has an invalid computational assumption strength")
    require(type(kzg_bits) is int and kzg_bits >= 120,
            "mechanism security receipt has an invalid KZG assumption strength")
    kzg_profile_id = assumptions.get("kzg_assumption_profile_id")
    registry_profile_id = assumptions.get("registry_assumption_id")
    require(isinstance(kzg_profile_id, str) and kzg_profile_id,
            "mechanism security receipt has no KZG assumption profile identity")
    require(isinstance(registry_profile_id, str) and registry_profile_id,
            "mechanism security receipt has no registry/Fiat-Shamir assumption profile identity")
    fs_query_budget_max = security.get("fiat_shamir_query_budget_max")
    require(type(fs_query_budget_max) is int and fs_query_budget_max == (1 << 20),
            "mechanism security receipt has no supported 2^20 Fiat-Shamir query budget")
    fs_profile_id = assumptions.get("fiat_shamir_assumption_profile_id")
    require(isinstance(fs_profile_id, str) and fs_profile_id,
            "mechanism security receipt has no Fiat-Shamir assumption profile identity")
    fs_security_bits = assumptions.get("fiat_shamir_security_bits")
    fs_ro_terms = assumptions.get("fiat_shamir_random_oracle_terms_128bit")
    require(type(fs_security_bits) is int and fs_security_bits == 128,
            "mechanism security receipt has an invalid Fiat-Shamir assumption strength")
    require(type(fs_ro_terms) is int and fs_ro_terms == 1,
            "mechanism security receipt has an invalid random-oracle assumption term count")
    fs_challenge_count = assumptions.get("fiat_shamir_challenge_count")
    fs_inventory = security.get("fiat_shamir_challenge_inventory")
    require(type(fs_challenge_count) is int and fs_challenge_count > 0,
            "mechanism security receipt has no variant-specific Fiat-Shamir challenge count")
    require(isinstance(fs_inventory, list) and fs_challenge_count == len(fs_inventory),
            "Fiat-Shamir challenge count does not match the variant-specific challenge inventory")
    kzg_game_count = assumptions.get("kzg_point_group_games")
    kzg_inventory = security.get("kzg_point_group_game_inventory")
    require(type(kzg_game_count) is int and kzg_game_count > 0,
            "mechanism security receipt has no variant-specific KZG game count")
    require(isinstance(kzg_inventory, list) and kzg_game_count == len(kzg_inventory),
            "KZG point-group game count does not match the variant-specific game inventory")
    field_modulus = security.get("field_modulus_decimal")
    require(isinstance(field_modulus, str) and CANONICAL_UINT.fullmatch(field_modulus) and int(field_modulus) > 0,
            "mechanism security receipt has an invalid field modulus")
    field_denominator = security.get("field_denominator_lower_bound")
    require(isinstance(field_denominator, str) and POWER_OF_TWO_LOWER_BOUND.fullmatch(field_denominator),
            "mechanism security receipt has no valid field denominator lower bound")

    total_error = security.get("total_error_bound")
    require(isinstance(total_error, dict), "mechanism security receipt has no total error bound")
    require(total_error.get("meets_target") is True, "mechanism security budget did not meet its target")
    require(total_error.get("denominator_lower_bound") == field_denominator,
            "total security bound uses a different denominator from the field policy")
    numerator_text = total_error.get("numerator")
    threshold_text = total_error.get("target_numerator_threshold")
    require(isinstance(numerator_text, str) and CANONICAL_UINT.fullmatch(numerator_text),
            "mechanism security receipt has an invalid total error numerator")
    require(isinstance(threshold_text, str) and CANONICAL_UINT.fullmatch(threshold_text),
            "mechanism security receipt has an invalid target numerator threshold")
    numerator, threshold = int(numerator_text), int(threshold_text)
    denominator_match = POWER_OF_TWO_LOWER_BOUND.fullmatch(field_denominator)
    assert denominator_match is not None
    denominator_bits = int(denominator_match.group(1))
    require(str(denominator_bits) == denominator_match.group(1),
            "mechanism field denominator lower bound is not canonical")
    require(denominator_bits >= 120 and threshold == (1 << (denominator_bits - 120)),
            "mechanism security target threshold does not match the declared denominator and 120-bit target")
    require(numerator <= threshold, "mechanism security error numerator exceeds the 120-bit target threshold")

    # These are the common proof-policy settings. Keep the complete ledger and
    # variant-specific counts separately; never equate them across variants.
    policy = {
        "target_bits": 120,
        "budget_exact": True,
        "bound_kind": "conservative_exact",
        "field_modulus_decimal": field_modulus,
        "field_denominator_lower_bound": field_denominator,
        "computational_assumption_bits": computational_bits,
        "kzg_assumption_bits": kzg_bits,
        "kzg_assumption_profile_id": kzg_profile_id,
        "fiat_shamir_query_budget_max": fs_query_budget_max,
        "fiat_shamir_assumption_profile_id": fs_profile_id,
        "fiat_shamir_security_bits": fs_security_bits,
        "fiat_shamir_random_oracle_terms_128bit": fs_ro_terms,
        "registry_assumption_id": registry_profile_id,
    }
    budget = {
        "valid": True,
        "ledger_sha256": ledger_hash,
        "total_error_bound": total_error,
        "fiat_shamir_challenge_count": fs_challenge_count,
        "kzg_point_group_games": kzg_game_count,
        "total_128bit_terms": assumptions.get("total_128bit_terms"),
        "identity_count": security.get("identity_count"),
    }
    return policy, budget


def validate_cuda_backend_receipt(result: dict[str, Any], *, device: int) -> dict[str, Any]:
    """Read CUDA attestations from the native v1 receipt's nested backend object."""
    backend = result.get("backend")
    require(isinstance(backend, dict), "prover result has no native backend receipt")
    require(backend.get("actual_cuda") is True,
            "CUDA was requested but the native prover did not attest actual CUDA execution")
    require(backend.get("cuda_ntt_forced") is True,
            "real CUDA proof did not attest the mechanism-only forced NTT dispatch policy")
    require(backend.get("device") == device, "prover result CUDA device differs from requested device")
    require(isinstance(backend.get("device_name"), str) and backend["device_name"],
            "prover result has no CUDA device name")
    kernel_delta = backend.get("cuda_kernel_invocations_delta")
    require(type(kernel_delta) is int and kernel_delta >= 0,
            "prover result has no valid native CUDA kernel invocation delta")
    stats_delta = backend.get("cuda_stats_delta")
    require(isinstance(stats_delta, dict), "prover result has no native CUDA statistics delta")
    require(all(isinstance(name, str) and type(value) is int and value >= 0
                for name, value in stats_delta.items()),
            "prover result has malformed native CUDA statistics deltas")
    return {
        "device_name": backend["device_name"],
        "cuda_kernel_invocations_delta": kernel_delta,
        "cuda_stats_delta": stats_delta,
        "cuda_ntt_forced": backend["cuda_ntt_forced"],
    }


def validate_run_result(
    result: dict[str, Any],
    *,
    mechanism: str,
    variant: str,
    dataset: str,
    width: int,
    device: int,
    real_input: dict[str, Any],
    proof_path: pathlib.Path,
    statement_path: pathlib.Path,
    proof_bytes: int,
    expected_srs: dict[str, Any],
) -> dict[str, Any]:
    require(result.get("format") == RUN_FORMAT, f"unexpected prover result format: {result.get('format')!r}")
    require(result.get("timing_schema") == TIMING_SCHEMA,
            f"unexpected prover timing schema: {result.get('timing_schema')!r}")
    require(result.get("mechanism") == mechanism, "prover result mechanism does not match requested mechanism")
    require(result.get("variant") == variant, "prover result variant does not match requested variant")
    require(result.get("dataset") == dataset, "prover result dataset differs from requested real input")
    input_receipt = result.get("real_input")
    require(isinstance(input_receipt, dict), "prover result has no real mechanism-input receipt")
    require(input_receipt.get("schema") == REAL_INPUT_SCHEMA, "prover result real-input schema is invalid")
    require(input_receipt.get("dataset") == dataset, "prover receipt dataset differs from the prepared input")
    require(input_receipt.get("bundle_sha256") == real_input["bundle_sha256"],
            "prover consumed a different real-input bundle")
    require(input_receipt.get("metadata_sha256") == real_input["metadata_sha256"],
            "prover statement did not bind the exact prepared metadata file")
    require(input_receipt.get("checkpoint_bundle_sha256") == real_input["checkpoint_bundle_sha256"],
            "prover statement did not bind the canonical checkpoint bundle")
    require(input_receipt.get("source_manifest_sha256") == real_input["source_manifest_sha256"],
            "prover statement did not bind the canonical real-data source inventory")
    require(input_receipt.get("input_preprocess_id") == real_input["input_preprocess_id"],
            "prover statement preprocessing identity differs from the formal loader")
    require(input_receipt.get("actual") == real_input["actual"],
            "prover statement graph N/E/G differs from the loader-derived real input")
    require(input_receipt.get("witness_sha256") == real_input["witness_sha256"],
            "prover consumed a different real finite-field witness")
    pair_digest = validate_hash(input_receipt.get("pair_input_commitment_sha256"),
                                "real-input pair commitment digest")
    require(input_receipt.get("layer") == 0 and input_receipt.get("head") == 0,
            "prover did not bind formal hidden layer 0, attention head 0")
    require(input_receipt.get("formal_head_width") == FORMAL_WIDTH and input_receipt.get("selected_width") == width,
            "prover did not bind the width-8 formal head and requested coordinate prefix")
    backend_receipt = validate_cuda_backend_receipt(result, device=device)
    cuda_kernel_delta = backend_receipt["cuda_kernel_invocations_delta"]
    cuda_stats_delta = backend_receipt["cuda_stats_delta"]
    require(cuda_kernel_delta > 0,
            "CUDA was requested but the timed mechanism proof has zero native CUDA invocations")
    warmup_time = result.get("cuda_warmup_time_ms")
    warmup_kernels = result.get("cuda_warmup_kernel_invocations")
    require(type(warmup_time) in (int, float) and math.isfinite(float(warmup_time)) and warmup_time >= 0,
            "prover receipt has no valid untimed CUDA warmup duration")
    require(type(warmup_kernels) is int and warmup_kernels > 0,
            "prover receipt has no actual CUDA invocation evidence for its untimed warmup")
    for field in ("ntt_calls", "intt_calls", "msm_single_calls", "msm_batch_calls"):
        require(type(cuda_stats_delta.get(field)) is int and cuda_stats_delta[field] >= 0,
                f"prover receipt has no valid CUDA counter {field}")
    require(cuda_stats_delta["ntt_calls"] > 0 and cuda_stats_delta["intt_calls"] > 0 and
            (cuda_stats_delta["msm_single_calls"] > 0 or cuda_stats_delta["msm_batch_calls"] > 0),
            "timed mechanism proof lacks native CUDA NTT/INTT or MSM evidence")

    security = result.get("security")
    require(isinstance(security, dict), "prover result has no security budget receipt")
    if "security_budget_passed" in security:
        require(security.get("security_budget_passed") is True, "mechanism security budget did not pass")
    security_policy, security_budget = validate_security_receipt(security)

    srs = result.get("srs")
    require(isinstance(srs, dict), "prover result has no SRS identity")
    require(isinstance(srs.get("id"), str) and srs["id"], "prover result has no SRS id")
    require(isinstance(srs.get("manifest_sha256"), str) and HEX_256.fullmatch(srs["manifest_sha256"]),
            "prover result has an invalid SRS manifest digest")
    require(isinstance(srs.get("protocol_schema_id"), str) and srs["protocol_schema_id"],
            "prover result has no protocol schema identity")
    require(srs.get("id") == expected_srs.get("manifest_srs_id"), "prover loaded a different SRS id than the selected manifest")
    require(srs.get("manifest_sha256") == expected_srs.get("manifest_sha256"),
            "prover loaded a different SRS manifest than the selected path")

    statement_digest = result.get("statement_sha256")
    require(isinstance(statement_digest, str) and HEX_256.fullmatch(statement_digest),
            "prover result has no valid public statement digest")
    require(statement_path.is_file() and sha256_file(statement_path) == statement_digest,
            "public statement bytes do not match the prover result digest")

    proof = result.get("proof")
    require(isinstance(proof, dict), "prover result has no proof artifact receipt")
    reported_path = proof.get("path")
    require(isinstance(reported_path, str) and pathlib.Path(reported_path).resolve() == proof_path.resolve(),
            "prover result proof path differs from the isolated cell artifact path")
    reported_proof_digest = proof.get("sha256")
    require(isinstance(reported_proof_digest, str) and HEX_256.fullmatch(reported_proof_digest),
            "prover result has an invalid proof digest")
    require(proof_path.is_file() and not proof_path.is_symlink(), "serialized proof artifact is missing")
    actual_proof_digest = sha256_file(proof_path)
    actual_proof_size = proof_path.stat().st_size
    require(actual_proof_digest == reported_proof_digest, "serialized proof bytes differ from prover receipt")
    require(type(proof.get("size_bytes")) is int and proof["size_bytes"] == actual_proof_size == proof_bytes,
            "proof_size_bytes is not the actual canonical serialized artifact length")

    metrics = result.get("metrics")
    require(isinstance(metrics, dict), "prover result has no metrics")
    for name in ("commitment_time_ms", "prove_time_ms"):
        value = metrics.get(name)
        require(type(value) in (int, float) and math.isfinite(float(value)) and value >= 0,
                f"invalid {name}")
    return {
        "security": security,
        "security_policy": security_policy,
        "security_budget": security_budget,
        "cuda_kernel_invocations_delta": cuda_kernel_delta,
        "cuda_stats_delta": cuda_stats_delta,
        "cuda_ntt_forced": backend_receipt["cuda_ntt_forced"],
        "srs": srs,
        "pair_input_commitment_sha256": pair_digest,
        "statement_sha256": statement_digest,
        "proof_sha256": actual_proof_digest,
        "proof_size_bytes": actual_proof_size,
        "commitment_time_ms": float(metrics["commitment_time_ms"]),
        "prove_time_ms": float(metrics["prove_time_ms"]),
        "protocol_schema_id": srs["protocol_schema_id"],
        "real_input": input_receipt,
        "dataset": dataset,
        "device_name": backend_receipt["device_name"],
        "warmup_time_ms": float(warmup_time), "warmup_kernel_invocations": warmup_kernels,
    }


def validate_verify_result(
    result: dict[str, Any],
    *,
    proof_sha256: str,
    statement_sha256: str,
    srs: dict[str, Any],
) -> dict[str, float | int]:
    require(result.get("format") == RUN_FORMAT, f"unexpected verifier result format: {result.get('format')!r}")
    require(result.get("timing_schema") == TIMING_SCHEMA,
            f"unexpected verifier timing schema: {result.get('timing_schema')!r}")
    require(result.get("status") == "VERIFY_OK", f"strict independent verifier status is {result.get('status')!r}")
    require(result.get("proof_sha256") == proof_sha256, "verifier consumed a different proof artifact")
    require(result.get("statement_sha256") == statement_sha256, "verifier consumed a different public statement")
    verify_srs = result.get("srs")
    require(isinstance(verify_srs, dict), "verifier result has no SRS identity")
    require(verify_srs.get("id") == srs.get("id"), "prover and verifier SRS ids differ")
    require(verify_srs.get("manifest_sha256") == srs.get("manifest_sha256"),
            "prover and verifier SRS manifest digests differ")
    require(result.get("verifier_backend") == "cpu", "independent verifier was not the strict CPU verifier")
    cuda_kernel_delta = result.get("verifier_cuda_kernel_invocations_delta")
    require(type(cuda_kernel_delta) is int and cuda_kernel_delta == 0,
            "strict CPU verifier has a nonzero or invalid CUDA kernel invocation delta")
    metrics = result.get("metrics")
    require(isinstance(metrics, dict), "verifier result has no verification metric")
    value = metrics.get("verify_time_ms")
    require(type(value) in (int, float) and math.isfinite(float(value)) and value >= 0,
            "invalid verify_time_ms")
    return {
        "verify_time_ms": float(value),
        "verifier_cuda_kernel_invocations_delta": cuda_kernel_delta,
    }


def run_process(
    command: list[str],
    log_path: pathlib.Path,
    timeout_seconds: float,
    label: str,
    redaction_secret: bytes | None = None,
) -> dict[str, Any]:
    start_utc = utc_now()
    started = time.monotonic()
    return_code: int | None = None
    timed_out = False
    pid: int | None = None
    secret_leaked = False
    output_error: list[BaseException] = []
    with log_path.open("wb") as log:
        rendered: list[str] = []
        redact_next = False
        for value in command:
            if redact_next:
                rendered.append("<private-seed-file>")
                redact_next = False
            else:
                rendered.append(value)
                redact_next = value == "--witness-seed-file"
        log.write(f"stage={label}\nstart_utc={start_utc}\ncommand={shlex.join(rendered)}\n".encode("utf-8"))
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        pid = process.pid
        ACTIVE_CHILDREN.add(process.pid)

        def drain_output() -> None:
            nonlocal secret_leaked
            assert process.stdout is not None
            pending = b""
            try:
                while True:
                    chunk = process.stdout.read(65536)
                    if not chunk:
                        break
                    data = pending + chunk
                    if redaction_secret and redaction_secret in data:
                        secret_leaked = True
                        data = data.replace(redaction_secret, b"*" * len(redaction_secret))
                    if redaction_secret:
                        emit_length = max(0, len(data) - len(redaction_secret) + 1)
                        log.write(data[:emit_length])
                        pending = data[emit_length:]
                    else:
                        log.write(data)
                if pending:
                    if redaction_secret and redaction_secret in pending:
                        secret_leaked = True
                        pending = pending.replace(redaction_secret, b"*" * len(redaction_secret))
                    log.write(pending)
                log.flush()
            except BaseException as exc:
                output_error.append(exc)

        reader = threading.Thread(target=drain_output, daemon=True)
        reader.start()
        deadline = time.monotonic() + timeout_seconds if timeout_seconds > 0 else None
        try:
            while return_code is None:
                wait_for = 1.0
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        timed_out = True
                        break
                    wait_for = min(wait_for, remaining)
                try:
                    return_code = process.wait(timeout=wait_for)
                except subprocess.TimeoutExpired:
                    if _INTERRUPT_SIGNAL is not None:
                        break
                    continue
            if timed_out or _INTERRUPT_SIGNAL is not None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    return_code = process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    return_code = process.wait()
        finally:
            ACTIVE_CHILDREN.discard(process.pid)
        reader.join(timeout=10)
        if reader.is_alive():
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            reader.join()
        if output_error:
            raise OSError(f"failed to capture {label} output: {output_error[0]}")
        end_utc = utc_now()
        elapsed = time.monotonic() - started
        log.write(
            f"end_utc={end_utc}\nprocess_pid={pid}\nreturn_code={return_code}\n"
            f"timed_out={str(timed_out).lower()}\nwall_time_seconds={elapsed:.6f}\n"
            .encode("utf-8")
        )
    return {
        "pid": pid,
        "start_utc": start_utc,
        "end_utc": end_utc,
        "return_code": return_code,
        "timed_out": timed_out,
        "interrupted": _INTERRUPT_SIGNAL is not None,
        "secret_leaked": secret_leaked,
        "wall_time_seconds": elapsed,
        "log": str(log_path),
    }


def matrix_config_id(spec: dict[str, Any]) -> str:
    component_tag = "full" if spec["components"] is None else "components-" + "-".join(str(x) for x in spec["components"])
    mechanism_tag = "ff" if spec["mechanism"] == "feature_folding" else "sn"
    return f"{mechanism_tag}_{spec['dataset']}_{component_tag}_d{spec['width']}_g{spec['graphs']}"


def bundle_key(dataset: str, components: tuple[int, ...] | None) -> str:
    if components is None:
        return f"{dataset}_full_graph"
    return f"{dataset}_components_" + "_".join(str(x) for x in components)


def unique_bundle_specs() -> list[tuple[str, str, tuple[int, ...] | None]]:
    found: dict[tuple[str, tuple[int, ...] | None], None] = {}
    for spec in FORMAL_CONFIGS:
        found[(spec["dataset"], spec["components"])] = None
    return [(bundle_key(ds, comps), ds, comps) for ds, comps in found]


def spec_for_cell(dataset: str, mechanism: str, width: int, graphs: int, components: tuple[int, ...] | None) -> dict[str, Any]:
    matches = [item for item in FORMAL_CONFIGS if item == {
        "mechanism": mechanism, "dataset": dataset, "width": width, "graphs": graphs, "components": components,
    }]
    require(len(matches) == 1, "cell does not map to exactly one pre-registered configuration")
    return matches[0]


def empty_summary(session_id: str, session_root: pathlib.Path) -> dict[str, Any]:
    return {
        "format": "zkgat_real_mechanism_experiment_v1",
        "session_id": session_id,
        "session_root": str(session_root),
        "created_utc": utc_now(),
        "session_status": "running",
        "backend": "cuda",
        "selected_layer": 0,
        "selected_head": 0,
        "formal_head_width": FORMAL_WIDTH,
        "feature_folding_width_semantics": "D is the prefix of the same real width-8 hidden feature from formal layer 0/head 0; D<8 is a controlled mechanism width, not a complete formal head calculation",
        "ppi_component_rule": PPI_COMPONENT_RULE,
        "mechanisms": list(MECHANISMS),
        "variants": list(VARIANTS),
        "repetitions": EXPERIMENT_REPETITIONS,
        "expected_config_count": len(FORMAL_CONFIGS),
        "expected_pair_count": len(FORMAL_CONFIGS) * EXPERIMENT_REPETITIONS,
        "expected_cell_count": len(FORMAL_CONFIGS) * EXPERIMENT_REPETITIONS * len(VARIANTS),
        "formal_configs": [dict(item, config_id=matrix_config_id(item)) for item in FORMAL_CONFIGS],
        "prepared_inputs": [],
        "cells": [],
        "pairs": [],
        "groups": [],
        "cell_count": 0,
        "verified_cells": 0,
        "complete_pairs": 0,
        "failure_reason": None,
        "timing_semantics": TIMING_SEMANTICS,
    }


def update_summary_counts(document: dict[str, Any], cells: list[dict[str, Any]]) -> None:
    pairs = summarize_real_pairs(cells)
    groups = summarize_real_groups(pairs)
    document["cells"] = cells
    document["pairs"] = pairs
    document["groups"] = groups
    document["cell_count"] = len(cells)
    document["verified_cells"] = sum(cell.get("verification_status") == "VERIFY_OK" for cell in cells)
    document["complete_pairs"] = sum(pair.get("status") == "complete" for pair in pairs)


def publish_progress(session_root: pathlib.Path, document: dict[str, Any], cells: list[dict[str, Any]]) -> None:
    update_summary_counts(document, cells)
    atomic_write_json(session_root / "summary.json", document)
    atomic_write_json(session_root / "status.json", {
        "session_id": document["session_id"], "session_status": document["session_status"],
        "updated_utc": utc_now(), "cell_count": document["cell_count"],
        "verified_cells": document["verified_cells"], "complete_pairs": document["complete_pairs"],
        "expected_cell_count": document["expected_cell_count"], "failure_reason": document.get("failure_reason"),
    })


def prepare_real_bundle(
    *, binary: pathlib.Path, session_root: pathlib.Path, dataset: str,
    components: tuple[int, ...] | None, config_info: dict[str, Any], timeout_seconds: int,
    session_log: pathlib.Path,
) -> dict[str, Any]:
    key = bundle_key(dataset, components)
    input_dir = session_root / "real_inputs" / key
    input_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    bundle_path = input_dir / "real_input.bin"
    metadata_path = input_dir / "metadata.json"
    log_path = input_dir / "prepare.log"
    config_path = pathlib.Path(config_info["config_path"])
    component_text = ",".join(str(value) for value in (components if components is not None else (0,)))
    command = [
        str(binary), "prepare-real", "--config", str(config_path), "--components", component_text,
        "--real-input-out", str(bundle_path), "--metadata-out", str(metadata_path),
    ]
    status_path = input_dir / "prepare_status.json"
    record: dict[str, Any] = {
        "bundle_key": key, "dataset": dataset, "selected_components": list(components or (0,)),
        "config_path": str(config_path), "config_sha256": config_info["config_sha256"],
        "command_log": str(log_path), "exit_code": None, "wall_time_seconds": None,
        "input_bundle_path": str(bundle_path), "metadata_path": str(metadata_path),
        "prepare_status_path": str(status_path), "prepare_status": "preparing",
    }
    atomic_write_json(status_path, record)
    emit(f"PREPARE_START dataset={dataset} components={component_text} output={input_dir}", session_log)
    try:
        process = run_process(command, log_path, timeout_seconds, "prepare-real")
        record.update({
            "exit_code": process["return_code"], "wall_time_seconds": process["wall_time_seconds"],
            "prepare_timed_out": process["timed_out"], "prepare_process_pid": process["pid"],
        })
        if process["timed_out"]:
            raise ValueError(f"prepare-real exceeded the {timeout_seconds}s configured preflight limit for {key}")
        require(process["return_code"] == 0,
                f"prepare-real failed for {key} with exit code {process['return_code']}")
        metadata = load_json(metadata_path, f"{key} real-input metadata")
        validated = validate_real_metadata(
            metadata, dataset=dataset, components=components, bundle_path=bundle_path,
            metadata_path=metadata_path, config_info=config_info,
        )
        record.update({
            "metadata": validated,
            "metadata_sha256": validated["metadata_sha256"],
            "bundle_sha256": validated["bundle_sha256"],
            "witness_sha256": validated["witness_sha256"],
            "actual": validated["actual"], "components": list(validated["components"]),
            "node_ptr": validated["node_ptr"], "edge_ptr": validated["edge_ptr"],
            "prepare_status": "prepared",
        })
        emit(
            f"PREPARE_END dataset={dataset} components={component_text} N={validated['actual']['N']} "
            f"E={validated['actual']['E']} G={validated['actual']['G']} bundle_sha256={validated['bundle_sha256']}",
            session_log,
        )
        return record
    except BaseException as exc:
        record["prepare_status"] = "failed"
        record["failure_reason"] = str(exc) or type(exc).__name__
        for path, key_name in ((log_path, "command_log_sha256"),
                               (bundle_path, "partial_bundle_sha256"),
                               (metadata_path, "partial_metadata_sha256")):
            if path.is_file() and not path.is_symlink():
                record[key_name] = sha256_file(path)
        atomic_write_json(status_path, record)
        raise
    finally:
        if record.get("prepare_status") == "prepared":
            record["command_log_sha256"] = sha256_file(log_path)
            atomic_write_json(status_path, record)


def check_expected_spec(spec: dict[str, Any], input_record: dict[str, Any]) -> dict[str, int]:
    actual = input_record["actual"]
    require(spec["width"] <= FORMAL_WIDTH, "selected mechanism width exceeds the formal source head width")
    require(spec["graphs"] == actual["G"], "configuration G differs from loader-derived graph count")
    return {"N": actual["N"], "E": actual["E"], "D": spec["width"], "G": actual["G"]}


def write_private_mask_seed(path: pathlib.Path, used_tokens: set[bytes]) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    for _ in range(16):
        token = os.urandom(32).hex().encode("ascii")
        if token not in used_tokens:
            used_tokens.add(token)
            break
    else:
        raise ValueError("could not generate a fresh per-cell private mask seed")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    ACTIVE_SECRET_FILES.add(path)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(token + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        ACTIVE_SECRET_FILES.discard(path)
        raise
    require(path.stat().st_mode & 0o777 == 0o600, "private mask seed file permissions are not 0600")
    return token


def remaining_timeout(started: float, timeout_seconds: int, label: str) -> float:
    if timeout_seconds == 0:
        return 0.0
    remaining = timeout_seconds - (time.monotonic() - started)
    require(remaining > 0, f"{label} exceeded the {timeout_seconds}s per-cell time limit")
    return remaining


def build_real_mechanism_run_command(
    *, binary: pathlib.Path, spec: dict[str, Any], variant: str, input_record: dict[str, Any], device: int,
    bundle_path: pathlib.Path, metadata_path: pathlib.Path, srs_path: pathlib.Path,
    seed_path: pathlib.Path, proof_path: pathlib.Path, statement_path: pathlib.Path,
    result_path: pathlib.Path,
) -> list[str]:
    actual = input_record["actual"]
    require(all(type(actual.get(field)) is int and actual[field] > 0 for field in ("N", "E", "G")),
            "real mechanism command requires positive loader-derived N/E/G")
    return [
        str(binary), "run", "--mechanism", spec["mechanism"], "--variant", variant,
        "--nodes", str(actual["N"]), "--edges", str(actual["E"]), "--width", str(spec["width"]),
        "--graphs", str(actual["G"]), "--seed", "0", "--backend", "cuda", "--device", str(device),
        "--real-input", str(bundle_path), "--real-input-metadata", str(metadata_path),
        "--srs", str(srs_path), "--witness-seed-file", str(seed_path),
        "--proof-out", str(proof_path), "--statement-out", str(statement_path),
        "--result-json", str(result_path),
    ]


def run_real_cell(
    *, binary: pathlib.Path, session_root: pathlib.Path, spec: dict[str, Any],
    input_record: dict[str, Any], srs_info: dict[str, Any], device: int,
    timeout_seconds: int, pair_index: int, repetition: int, variant: str,
    used_seed_tokens: set[bytes], device_snapshot: dict[str, Any],
    session_log: pathlib.Path,
) -> dict[str, Any]:
    cfgid = matrix_config_id(spec)
    cell_root = session_root / spec["mechanism"] / cfgid / f"rep_{repetition:03d}" / variant
    cell_root.mkdir(parents=True, exist_ok=False, mode=0o700)
    bundle_path = pathlib.Path(input_record["input_bundle_path"])
    metadata_path = pathlib.Path(input_record["metadata_path"])
    seed_path = cell_root / ".private_mask_seed.witness_seed"
    proof_path = cell_root / "proof.bin"
    statement_path = cell_root / "public_statement.json"
    prover_result_path = cell_root / "prover_result.json"
    verifier_result_path = cell_root / "verifier_result.json"
    prover_log = cell_root / "prover.log"
    verifier_log = cell_root / "verifier.log"
    graph = input_record["actual"]
    common: dict[str, Any] = {
        "mechanism": spec["mechanism"], "variant": variant, "dataset": spec["dataset"],
        "config_id": cfgid, "config_path": input_record["metadata"]["config_path"],
        "config_sha256": input_record["metadata"]["config_sha256"],
        "N": graph["N"], "E": graph["E"], "D": spec["width"], "G": graph["G"],
        "layer": 0, "head": 0, "formal_head_width": FORMAL_WIDTH,
        "component_ids": list(input_record["components"]),
        "node_ptr": input_record["node_ptr"], "edge_ptr": input_record["edge_ptr"],
        "nonself_neighbor_stats": [item["nonself_neighbor_stats"] for item in input_record["metadata"]["selected_components"]],
        "input_bundle_path": str(bundle_path), "input_bundle_sha256": input_record["bundle_sha256"],
        "input_metadata_path": str(metadata_path), "input_metadata_sha256": input_record["metadata_sha256"],
        "input_witness_sha256": input_record["witness_sha256"],
        "data_identity_sha256": input_record["metadata"]["data_identity_sha256"],
        "checkpoint_bundle": input_record["metadata"]["checkpoint_bundle"],
        "data_sources": input_record["metadata"]["data_sources"],
        "srs_id": srs_info["manifest_srs_id"], "srs_manifest_sha256": srs_info["manifest_sha256"],
        "backend": "cuda", "device": device,
        "cell_status": "failed", "pair_status": "incomplete", "verification_status": "NOT_RUN",
        "proof_path": str(proof_path), "statement_path": str(statement_path),
        "prover_result_path": str(prover_result_path), "verifier_result_path": str(verifier_result_path),
        "prover_log": str(prover_log), "verifier_log": str(verifier_log),
        "prover_exit_code": None, "verifier_exit_code": None,
        "prover_process_pid": None, "verifier_process_pid": None, "failure_reason": None,
        "pair_order_index": pair_index, "repetition": repetition,
    }
    seed_bytes = write_private_mask_seed(seed_path, used_seed_tokens)
    common["private_mask_seed_file_mode"] = "0600"
    common["private_mask_seed_unique_in_session"] = True
    run_command = build_real_mechanism_run_command(
        binary=binary, spec=spec, variant=variant, input_record=input_record, device=device,
        bundle_path=bundle_path, metadata_path=metadata_path, srs_path=pathlib.Path(srs_info["path"]),
        seed_path=seed_path, proof_path=proof_path, statement_path=statement_path,
        result_path=prover_result_path,
    )
    started = time.monotonic()
    try:
        require(_INTERRUPT_SIGNAL is None, "formal mechanism run received a stop signal")
        assert_no_external_cuda_compute(device_snapshot)
        require(len(seed_bytes) == 64, "private mask seed has an unexpected encoding")
        prover_proc = run_process(run_command, prover_log, remaining_timeout(started, timeout_seconds, "prover"),
                                  "real-mechanism-prover", seed_bytes)
        common["prover_exit_code"] = prover_proc["return_code"]
        common["prover_process_pid"] = prover_proc["pid"]
        common["prover_wall_time_seconds"] = prover_proc["wall_time_seconds"]
        if prover_proc["timed_out"]:
            raise ValueError(f"prover exceeded the {timeout_seconds}s per-cell limit")
        require(not prover_proc["secret_leaked"], "native prover attempted to write its private mask seed to the log")
        require(not prover_proc["interrupted"], "native prover was interrupted by a stop signal")
        require(prover_proc["return_code"] == 0, f"prover exited with code {prover_proc['return_code']}")
        for output_path in (prover_result_path, statement_path, proof_path):
            if output_path.is_file():
                require(not file_contains_bytes(output_path, seed_bytes), f"private mask seed appeared in {output_path.name}")
        prover_result = load_json(prover_result_path, "real mechanism prover result")
        require(proof_path.is_file() and not proof_path.is_symlink(), "prover did not serialize a regular proof artifact")
        proof_size = proof_path.stat().st_size
        receipt = validate_run_result(
            prover_result, mechanism=spec["mechanism"], variant=variant,
            dataset=spec["dataset"], width=spec["width"], device=device,
            real_input=input_record["metadata"], proof_path=proof_path,
            statement_path=statement_path, proof_bytes=proof_size, expected_srs=srs_info,
        )
        require(receipt["real_input"].get("metadata_sha256", input_record["metadata_sha256"]) == input_record["metadata_sha256"],
                "prover receipt does not bind the prepared real-input metadata hash")
        common.update({
            "pair_input_commitment_sha256": receipt["pair_input_commitment_sha256"],
            "statement_sha256": receipt["statement_sha256"], "statement_size_bytes": statement_path.stat().st_size,
            "statement_file_sha256": sha256_file(statement_path),
            "proof_sha256": receipt["proof_sha256"], "proof_size_bytes": receipt["proof_size_bytes"],
            "prover_result_sha256": sha256_file(prover_result_path),
            "security": receipt["security"], "security_policy": receipt["security_policy"],
            "security_budget": receipt["security_budget"],
            "security_budget_ledger_sha256": receipt["security_budget"]["ledger_sha256"],
            "security_budget_valid": receipt["security_budget"]["valid"],
            "security_target_bits": receipt["security_policy"]["target_bits"],
            "security_bound_kind": receipt["security_policy"]["bound_kind"],
            "security_error_numerator": receipt["security_budget"]["total_error_bound"]["numerator"],
            "security_target_numerator_threshold": receipt["security_budget"]["total_error_bound"]["target_numerator_threshold"],
            "security_fiat_shamir_query_budget_max": receipt["security_policy"]["fiat_shamir_query_budget_max"],
            "security_fiat_shamir_challenge_count": receipt["security_budget"]["fiat_shamir_challenge_count"],
            "security_kzg_point_group_games": receipt["security_budget"]["kzg_point_group_games"],
            "security_total_128bit_terms": receipt["security_budget"]["total_128bit_terms"],
            "security_identity_count": receipt["security_budget"]["identity_count"],
            "protocol_schema_id": receipt["protocol_schema_id"],
            "cuda_actual": True, "cuda_device_name": receipt["device_name"],
            "cuda_kernel_invocations_delta": receipt["cuda_kernel_invocations_delta"],
            "cuda_stats_delta": receipt["cuda_stats_delta"],
            "cuda_ntt_forced": receipt["cuda_ntt_forced"],
            "cuda_warmup_time_ms": receipt["warmup_time_ms"],
            "cuda_warmup_kernel_invocations": receipt["warmup_kernel_invocations"],
            "commitment_time_ms": receipt["commitment_time_ms"],
            "prove_time_ms": receipt["prove_time_ms"],
        })
        verify_command = [
            str(binary), "verify", "--proof", str(proof_path), "--statement", str(statement_path),
            "--srs", srs_info["path"], "--result-json", str(verifier_result_path),
        ]
        verifier_proc = run_process(verify_command, verifier_log,
                                   remaining_timeout(started, timeout_seconds, "prover and independent verification"),
                                   "independent-cpu-verifier")
        common["verifier_exit_code"] = verifier_proc["return_code"]
        common["verifier_process_pid"] = verifier_proc["pid"]
        common["verify_wall_time_seconds"] = verifier_proc["wall_time_seconds"]
        if verifier_proc["timed_out"]:
            raise ValueError(f"independent verifier exceeded the {timeout_seconds}s per-cell limit")
        require(not verifier_proc["interrupted"], "independent verifier was interrupted by a stop signal")
        require(verifier_proc["return_code"] == 0,
                f"independent verifier exited with code {verifier_proc['return_code']}")
        verify_result = load_json(verifier_result_path, "independent CPU verifier result")
        require(not file_contains_bytes(verifier_result_path, seed_bytes), "private mask seed appeared in verifier output")
        verifier_receipt = validate_verify_result(
            verify_result, proof_sha256=receipt["proof_sha256"],
            statement_sha256=receipt["statement_sha256"], srs=receipt["srs"],
        )
        require(common["prover_process_pid"] != common["verifier_process_pid"],
                "proof and verification did not run in separate processes")
        common.update({
            "verify_time_ms": verifier_receipt["verify_time_ms"],
            "verifier_cuda_kernel_invocations_delta": verifier_receipt["verifier_cuda_kernel_invocations_delta"],
            "verifier_result_sha256": sha256_file(verifier_result_path),
            "verification_status": "VERIFY_OK", "cell_status": "complete",
        })
    except (OSError, ValueError, subprocess.SubprocessError, KeyboardInterrupt) as exc:
        common["failure_reason"] = str(exc) or "interrupted"
        common["cell_status"] = "failed"
        if verifier_result_path.exists():
            try:
                diagnostic = load_json(verifier_result_path, "verifier diagnostic")
                if diagnostic.get("status") == "VERIFY_OK":
                    common["verification_status"] = "VERIFY_OK"
            except ValueError:
                pass
        for output_path, key in ((proof_path, "partial_proof_sha256"),
                                 (statement_path, "partial_statement_sha256"),
                                 (prover_result_path, "partial_prover_result_sha256"),
                                 (verifier_result_path, "partial_verifier_result_sha256")):
            if output_path.is_file() and not output_path.is_symlink():
                common[key] = sha256_file(output_path)
    finally:
        seed_path.unlink(missing_ok=True)
        ACTIVE_SECRET_FILES.discard(seed_path)
    common["mask_seed_file_removed"] = not seed_path.exists()
    for path, field in ((prover_log, "prover_log_sha256"), (verifier_log, "verifier_log_sha256")):
        if path.is_file() and not path.is_symlink():
            common[field] = sha256_file(path)
    common["cell_wall_time_seconds"] = time.monotonic() - started
    cell_path = cell_root / "cell.json"
    common["cell_json_path"] = str(cell_path)
    atomic_write_json(cell_path, common)
    emit(
        f"CELL_END config={cfgid} rep={repetition} variant={variant} status={common['cell_status']} "
        f"verify={common['verification_status']} exit={common['prover_exit_code']}/{common['verifier_exit_code']}",
        session_log,
    )
    return common


def mark_real_pair(cells: list[dict[str, Any]], config_id: str, repetition: int) -> None:
    pair_cells = [cell for cell in cells if cell["config_id"] == config_id and cell["repetition"] == repetition]
    by_variant = {cell["variant"]: cell for cell in pair_cells}
    original, control = by_variant.get("original"), by_variant.get("control")
    pair_ok = bool(
        original and control
        and original.get("cell_status") == "complete" and control.get("cell_status") == "complete"
        and original.get("verification_status") == "VERIFY_OK" and control.get("verification_status") == "VERIFY_OK"
        and original.get("pair_input_commitment_sha256") == control.get("pair_input_commitment_sha256")
        and original.get("input_bundle_sha256") == control.get("input_bundle_sha256")
        and original.get("input_metadata_sha256") == control.get("input_metadata_sha256")
        and original.get("input_witness_sha256") == control.get("input_witness_sha256")
        and original.get("N") == control.get("N") and original.get("E") == control.get("E")
        and original.get("D") == control.get("D") and original.get("G") == control.get("G")
        and original.get("component_ids") == control.get("component_ids")
        and original.get("srs_id") == control.get("srs_id")
        and original.get("srs_manifest_sha256") == control.get("srs_manifest_sha256")
        and original.get("protocol_schema_id") == control.get("protocol_schema_id")
        and original.get("backend") == control.get("backend")
        and original.get("device") == control.get("device")
        and original.get("security_budget_valid") is True and control.get("security_budget_valid") is True
        and original.get("security_policy") == control.get("security_policy")
    )
    for cell in pair_cells:
        cell["pair_status"] = "paired" if pair_ok else "invalid_pair"
        if not pair_ok and cell.get("failure_reason") is None:
            cell["failure_reason"] = "paired real input, relation digest, SRS, CUDA device, or security policy differs"
        atomic_write_json(pathlib.Path(cell["cell_json_path"]), cell)


def require_repetition_input_stability(cells: list[dict[str, Any]], config_id: str) -> None:
    selected = [cell for cell in cells if cell["config_id"] == config_id and cell.get("pair_status") == "paired"]
    identities = {
        (cell.get("N"), cell.get("E"), cell.get("D"), cell.get("G"),
         tuple(cell.get("component_ids", [])), cell.get("input_bundle_sha256"),
         cell.get("input_metadata_sha256"), cell.get("input_witness_sha256"),
         cell.get("data_identity_sha256"), cell.get("pair_input_commitment_sha256"),
         cell.get("srs_id"), cell.get("srs_manifest_sha256"), cell.get("security_policy"))
        for cell in selected
    }
    require(len(identities) <= 1,
            f"real input, relation digest, SRS, or security policy changed between repetitions for {config_id}")


def summarize_real_pairs(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expected_config = {matrix_config_id(spec): spec for spec in FORMAL_CONFIGS}
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for cell in cells:
        grouped.setdefault((cell["config_id"], cell["repetition"]), {})[cell["variant"]] = cell
    pairs: list[dict[str, Any]] = []
    for (config_id, repetition), variants in sorted(grouped.items()):
        spec = expected_config.get(config_id)
        require(spec is not None, f"result contains an unregistered configuration {config_id}")
        original, control = variants.get("original"), variants.get("control")
        valid = bool(original and control and original.get("pair_status") == "paired" and control.get("pair_status") == "paired")
        pair: dict[str, Any] = {
            "config_id": config_id, "mechanism": spec["mechanism"], "dataset": spec["dataset"],
            "N": original["N"] if original else (control["N"] if control else None),
            "E": original["E"] if original else (control["E"] if control else None),
            "D": original["D"] if original else (control["D"] if control else spec["width"]),
            "G": original["G"] if original else (control["G"] if control else spec["graphs"]),
            "component_ids": (original.get("component_ids") if original else
                              (control.get("component_ids") if control else list(spec["components"] or (0,)))),
            "repetition": repetition, "status": "complete" if valid else "incomplete",
            "original": original, "control": control,
        }
        if valid:
            metrics: dict[str, Any] = {}
            for name in METRIC_FIELDS:
                original_value, control_value = float(original[name]), float(control[name])
                relative = None if control_value == 0 else (original_value - control_value) * 100.0 / control_value
                ratio = None if original_value == 0 else control_value / original_value
                metrics[name] = {
                    "original": original_value, "control": control_value,
                    "original_minus_control": original_value - control_value,
                    "relative_original_minus_control_over_control_percent": relative,
                    "control_over_original": ratio,
                }
            pair["metrics"] = metrics
        pairs.append(pair)
    return pairs


def _sample(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": sum(values) / len(values) if values else None,
        "sample_sd": statistics.stdev(values) if len(values) >= 2 else None,
        "n": len(values),
    }


def summarize_real_groups(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for spec in FORMAL_CONFIGS:
        cfgid = matrix_config_id(spec)
        selected = [pair for pair in pairs if pair["config_id"] == cfgid]
        complete = [pair for pair in selected if pair["status"] == "complete"]
        metrics: dict[str, Any] = {}
        for name in METRIC_FIELDS:
            original_values = [float(pair["metrics"][name]["original"]) for pair in complete]
            control_values = [float(pair["metrics"][name]["control"]) for pair in complete]
            differences = [float(pair["metrics"][name]["original_minus_control"]) for pair in complete]
            relative_values = [
                float(pair["metrics"][name]["relative_original_minus_control_over_control_percent"])
                for pair in complete
                if pair["metrics"][name]["relative_original_minus_control_over_control_percent"] is not None
            ]
            ratios = [float(pair["metrics"][name]["control_over_original"]) for pair in complete
                      if pair["metrics"][name]["control_over_original"] is not None]
            metrics[name] = {
                "original": _sample(original_values), "control": _sample(control_values),
                "paired_original_minus_control": _sample(differences),
                "relative_original_minus_control_over_control_percent": _sample(relative_values),
                "control_over_original_advantage": _sample(ratios),
            }
        source_pair = selected[0] if selected else None
        source_cell = None if source_pair is None else (source_pair.get("original") or source_pair.get("control"))
        n = source_cell.get("N") if source_cell else None
        e = source_cell.get("E") if source_cell else None
        graphs = source_cell.get("G") if source_cell else spec["graphs"]
        result.append({
            "config_id": cfgid, "mechanism": spec["mechanism"], "dataset": spec["dataset"],
            "N": n, "E": e, "D": source_cell.get("D") if source_cell else spec["width"], "G": graphs,
            "component_ids": (source_cell.get("component_ids") if source_cell else list(spec["components"] or (0,))),
            "requested_repetitions": EXPERIMENT_REPETITIONS,
            "complete_repetitions": len(complete), "metrics": metrics,
        })
    return result


def write_cells_csv(path: pathlib.Path, cells: list[dict[str, Any]]) -> None:
    fields = [
        "mechanism", "variant", "dataset", "config_id", "N", "E", "D", "G", "component_ids",
        "node_ptr", "edge_ptr", "nonself_neighbor_stats", "repetition", "pair_order_index",
        "backend", "device", "device_name", "cell_status", "pair_status", "verification_status",
        *METRIC_FIELDS, "pair_input_commitment_sha256", "input_bundle_sha256", "input_metadata_sha256",
        "input_witness_sha256", "data_identity_sha256", "config_path", "config_sha256", "checkpoint_bundle",
        "data_sources", "srs_id", "srs_manifest_sha256", "protocol_schema_id", "security_budget_ledger_sha256",
        "security_budget_valid", "security_target_bits", "security_error_numerator",
        "security_target_numerator_threshold", "security_fiat_shamir_challenge_count",
        "security_kzg_point_group_games", "cuda_actual", "cuda_device_name", "cuda_ntt_forced", "cuda_kernel_invocations_delta",
        "cuda_stats_delta", "cuda_warmup_time_ms", "cuda_warmup_kernel_invocations",
        "proof_path", "proof_sha256", "statement_path", "statement_sha256", "prover_result_path",
        "prover_result_sha256", "verifier_result_path", "verifier_result_sha256",
        "prover_exit_code", "verifier_exit_code", "prover_log", "verifier_log", "failure_reason",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for cell in cells:
            row = dict(cell)
            for field in ("component_ids", "node_ptr", "edge_ptr", "nonself_neighbor_stats", "checkpoint_bundle", "data_sources", "cuda_stats_delta"):
                if field in row:
                    row[field] = json.dumps(row[field], sort_keys=True, separators=(",", ":"))
            writer.writerow({field: row.get(field, "") for field in fields})


def _paper_number(value: Any, name: str) -> str:
    if value is None:
        return "—"
    if name == "proof_size_bytes":
        return f"{float(value):.2f}"
    return f"{float(value):.6f}"


def write_mechanism_paper_markdown(path: pathlib.Path, document: dict[str, Any], mechanism: str) -> None:
    title = "Sparse Node-to-Edge Feature Verification" if mechanism == "feature_folding" else "Shared Neighborhood Verification"
    lines = [
        f"# {title} — formal real-data CUDA mechanism results", "",
        f"Session: `{document['session_id']}`; status: **{document['session_status']}**; backend: **CUDA**.",
        f"Independent verifier: CPU, separate process. Accepted formal proof cells: {document['verified_cells']}/{document['expected_cell_count']}.",
        "Layer/head source: formal hidden layer 0, attention head 0, width 8. Feature Folding D<8 uses the first D coordinates of the same real width-8 hidden feature as a controlled mechanism width.",
        "PPI components use the fixed selection rule recorded in provenance: " + PPI_COMPONENT_RULE + ".",
        "Commitment and Prove are disjoint native proof partitions; Verify covers independent verifier parsing through acceptance; proof size is the serialized mechanism proof file length.",
        "Paired difference is Original − Control; relative change is (Original − Control) / Control; advantage ratio is Control / Original. Mean and sample SD use complete verified pairs only.", "",
        "| Dataset | N | E | D | G | Component IDs | Metric | Original mean | Original sample SD | Control mean | Control sample SD | Paired O−C mean | Paired O−C SD | Relative (O−C)/C % mean | Relative sample SD | C/O mean | C/O sample SD | Repetitions |",
        "|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in document["groups"]:
        if row["mechanism"] != mechanism:
            continue
        comp = ",".join(str(x) for x in row["component_ids"])
        for metric_name in METRIC_FIELDS:
            values = row["metrics"][metric_name]
            lines.append(
                f"| {row['dataset']} | {row['N']} | {row['E']} | {row['D']} | {row['G']} | {comp} | `{metric_name}` | "
                f"{_paper_number(values['original']['mean'], metric_name)} | {_paper_number(values['original']['sample_sd'], metric_name)} | "
                f"{_paper_number(values['control']['mean'], metric_name)} | {_paper_number(values['control']['sample_sd'], metric_name)} | "
                f"{_paper_number(values['paired_original_minus_control']['mean'], metric_name)} | "
                f"{_paper_number(values['paired_original_minus_control']['sample_sd'], metric_name)} | "
                f"{_paper_number(values['relative_original_minus_control_over_control_percent']['mean'], metric_name)} | "
                f"{_paper_number(values['relative_original_minus_control_over_control_percent']['sample_sd'], metric_name)} | "
                f"{_paper_number(values['control_over_original_advantage']['mean'], metric_name)} | "
                f"{_paper_number(values['control_over_original_advantage']['sample_sd'], metric_name)} | "
                f"{row['complete_repetitions']}/{row['requested_repetitions']} |"
            )
    lines.extend(["", "## Proof cells", "", "| Config | Rep | Variant | Verification | Pair | Commitment ms | Prove ms | Verify ms | Proof bytes | CUDA device | Proof kernels | Warmup kernels | Input bundle SHA-256 | Metadata SHA-256 | Relation SHA-256 | Statement SHA-256 | Prover receipt SHA-256 | Proof SHA-256 | Verifier receipt SHA-256 |", "|---|---:|---|---|---|---:|---:|---:|---:|---|---:|---:|---|---|---|---|---|---|---|"])
    for cell in document["cells"]:
        if cell["mechanism"] != mechanism:
            continue
        lines.append(
            f"| {cell['config_id']} | {cell['repetition']} | {cell['variant']} | {cell['verification_status']} | {cell['pair_status']} | "
            f"{_paper_number(cell.get('commitment_time_ms'), 'commitment_time_ms')} | "
            f"{_paper_number(cell.get('prove_time_ms'), 'prove_time_ms')} | "
            f"{_paper_number(cell.get('verify_time_ms'), 'verify_time_ms')} | "
            f"{_paper_number(cell.get('proof_size_bytes'), 'proof_size_bytes')} | "
            f"{cell.get('cuda_device_name', '—')} | {cell.get('cuda_kernel_invocations_delta', '—')} | "
            f"{cell.get('cuda_warmup_kernel_invocations', '—')} | `{cell.get('input_bundle_sha256', '—')}` | "
            f"`{cell.get('input_metadata_sha256', '—')}` | `{cell.get('pair_input_commitment_sha256', '—')}` | "
            f"`{cell.get('statement_sha256', '—')}` | `{cell.get('prover_result_sha256', '—')}` | "
            f"`{cell.get('proof_sha256', '—')}` | `{cell.get('verifier_result_sha256', '—')}` |"
        )
    lines.extend([
        "", "## Data and provenance", "",
        f"- Source tree SHA-256: `{document.get('provenance', {}).get('source', {}).get('source_tree_hash', '—')}`",
        f"- Source diff SHA-256: `{document.get('provenance', {}).get('source', {}).get('git_diff_sha256', '—')}`",
        f"- Source inventory ID: `{document.get('provenance', {}).get('source', {}).get('source_inventory_id', '—')}`",
        f"- CUDA binary SHA-256: `{document.get('provenance', {}).get('build', {}).get('binary_sha256', '—')}`; CMake cache SHA-256 `{document.get('provenance', {}).get('build', {}).get('cmake_cache_sha256', '—')}`; build source manifest SHA-256 `{document.get('provenance', {}).get('build', {}).get('source_manifest_sha256', '—')}`.",
        f"- CUDA device record: `{document.get('provenance', {}).get('device', {}).get('selected_device', {})}`; formal CUDA proofs attested: `{document.get('verified_cells', 0)}`.",
    ])
    datasets = sorted({row["dataset"] for row in document["groups"] if row["mechanism"] == mechanism})
    for dataset in datasets:
        info = document.get("provenance", {}).get("datasets", {}).get(dataset, {})
        lines.append(f"- `{dataset}` config SHA-256 `{info.get('config_sha256', '—')}`, checkpoint bundle `{info.get('checkpoint_path', '—')}`.")
        srs = info.get("srs", {})
        lines.append(f"  - Checkpoint bundle path and each member digest are listed in `summary.json`/`cells.csv`; formal checkpoint bundle SHA-256 values are recorded per prepared input.")
        lines.append(f"  - SRS `{srs.get('manifest_srs_id', '—')}` trusted manifest SHA-256 `{srs.get('manifest_sha256', '—')}`, registered domains `{srs.get('domain_sizes', {})}`.")
    lines.extend(["", "### Prepared input hashes", ""])
    for bundle in document.get("prepared_inputs", []):
        meta = bundle.get("metadata", {})
        if meta.get("dataset") not in datasets:
            continue
        lines.append(
            f"- `{bundle['bundle_key']}` N/E/G `{meta.get('actual', {}).get('N', '—')}/{meta.get('actual', {}).get('E', '—')}/{meta.get('actual', {}).get('G', '—')}`; "
            f"components `{bundle.get('components', [])}`; bundle SHA-256 `{bundle.get('bundle_sha256', '—')}`; "
            f"metadata SHA-256 `{bundle.get('metadata_sha256', '—')}`; witness SHA-256 `{bundle.get('witness_sha256', '—')}`; "
            f"checkpoint bundle SHA-256 `{meta.get('checkpoint_bundle_sha256', '—')}`; source manifest SHA-256 `{meta.get('source_manifest_sha256', '—')}`."
        )
    lines.append(f"- Output directory: `{document['session_root']}`")
    atomic_write_text(path, "\n".join(lines) + "\n")


def write_paper_csv(path: pathlib.Path, groups: list[dict[str, Any]], mechanism: str) -> None:
    fields = ["mechanism", "dataset", "N", "E", "D", "G", "component_ids", "complete_repetitions", "requested_repetitions", "metric",
              "original_mean", "original_sample_sd", "control_mean", "control_sample_sd", "paired_original_minus_control_mean",
              "paired_original_minus_control_sample_sd", "relative_original_minus_control_over_control_percent_mean",
              "relative_original_minus_control_over_control_percent_sample_sd", "control_over_original_mean", "control_over_original_sample_sd"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for group in groups:
            if group["mechanism"] != mechanism:
                continue
            for metric_name, values in group["metrics"].items():
                writer.writerow({
                    "mechanism": mechanism, "dataset": group["dataset"], "N": group["N"], "E": group["E"],
                    "D": group["D"], "G": group["G"],
                    "component_ids": ",".join(str(x) for x in group["component_ids"]),
                    "complete_repetitions": group["complete_repetitions"], "requested_repetitions": group["requested_repetitions"],
                    "metric": metric_name, "original_mean": values["original"]["mean"],
                    "original_sample_sd": values["original"]["sample_sd"], "control_mean": values["control"]["mean"],
                    "control_sample_sd": values["control"]["sample_sd"],
                    "paired_original_minus_control_mean": values["paired_original_minus_control"]["mean"],
                    "paired_original_minus_control_sample_sd": values["paired_original_minus_control"]["sample_sd"],
                    "relative_original_minus_control_over_control_percent_mean": values["relative_original_minus_control_over_control_percent"]["mean"],
                    "relative_original_minus_control_over_control_percent_sample_sd": values["relative_original_minus_control_over_control_percent"]["sample_sd"],
                    "control_over_original_mean": values["control_over_original_advantage"]["mean"],
                    "control_over_original_sample_sd": values["control_over_original_advantage"]["sample_sd"],
                })


def generate_reports(session_root: pathlib.Path, document: dict[str, Any]) -> dict[str, str]:
    write_cells_csv(session_root / "cells.csv", document["cells"])
    paths: dict[str, str] = {}
    for mechanism, stem in (("feature_folding", "feature_folding"), ("shared_neighborhood", "shared_neighborhood")):
        md = session_root / f"{stem}_summary.md"
        csv_path = session_root / f"{stem}_summary.csv"
        write_mechanism_paper_markdown(md, document, mechanism)
        write_paper_csv(csv_path, document["groups"], mechanism)
        paths[f"{mechanism}_markdown"] = str(md)
        paths[f"{mechanism}_csv"] = str(csv_path)
    document["result_files"] = paths
    atomic_write_json(session_root / "summary.json", document)
    atomic_write_json(session_root / "results_manifest.json", {
        "session_id": document["session_id"], "session_status": document["session_status"],
        "files": {key: {"path": value, "sha256": sha256_file(pathlib.Path(value))}
                  for key, value in paths.items()},
        "cells_csv": {"path": str(session_root / "cells.csv"), "sha256": sha256_file(session_root / "cells.csv")},
        "summary_json": {"path": str(session_root / "summary.json"), "sha256": sha256_file(session_root / "summary.json")},
    })
    return paths


def revalidate_input_sources(document: dict[str, Any]) -> None:
    for bundle in document.get("prepared_inputs", []):
        meta = bundle.get("metadata")
        require(isinstance(meta, dict), "prepared input summary is missing its provenance record")
        bundle_path = pathlib.Path(bundle["input_bundle_path"])
        metadata_path = pathlib.Path(bundle["metadata_path"])
        require(sha256_file(bundle_path) == bundle["bundle_sha256"], f"real input bundle changed: {bundle_path}")
        require(sha256_file(metadata_path) == bundle["metadata_sha256"], f"real input metadata changed: {metadata_path}")
        checkpoint = meta["checkpoint_bundle"]
        checkpoint_inventory = recursive_file_inventory([pathlib.Path(meta["checkpoint_root"])], "checkpoint")
        require(checkpoint_inventory == checkpoint["files"],
                f"checkpoint bundle inventory changed during the session: {meta['checkpoint_root']}")
        data_inventory = recursive_file_inventory([pathlib.Path(path) for path in meta["source_roots"]],
                                                  "real dataset source")
        require(data_inventory == meta["data_sources"],
                f"real dataset source inventory changed during the session: {meta['dataset']}")
        require(canonical_inventory_sha256(data_inventory) == meta["source_manifest_sha256"],
                f"real dataset source manifest digest changed during the session: {meta['dataset']}")


def formal_schedule() -> list[tuple[dict[str, Any], int]]:
    schedule: list[tuple[dict[str, Any], int]] = []
    for repetition in range(1, EXPERIMENT_REPETITIONS + 1):
        shift = ((repetition - 1) * 4) % len(FORMAL_CONFIGS)
        ordered = list(FORMAL_CONFIGS[shift:]) + list(FORMAL_CONFIGS[:shift])
        schedule.extend((spec, repetition) for spec in ordered)
    return schedule


def execute_formal(args: argparse.Namespace) -> int:
    require(0 <= args.timeout_seconds <= 599, "--timeout-seconds must be 0 (unlimited) or between 1 and 599")
    require(0 <= args.prepare_timeout_seconds <= 599,
            "--prepare-timeout-seconds must be 0 (unlimited) or between 1 and 599")
    output_root = safe_output_root(DEFAULT_OUTPUT_ROOT)
    session_id, session_root = create_session_dir(output_root)
    os.umask(0o077)
    document = empty_summary(session_id, session_root)
    cells: list[dict[str, Any]] = []
    prepared: list[dict[str, Any]] = []
    lock_stream: BinaryIO | None = None
    session_log = session_root / "session.log"
    atomic_write_json(session_root / "status.json", {
        "session_id": session_id, "session_status": "preflight", "expected_cell_count": 66,
        "started_utc": document["created_utc"], "timeout_seconds": args.timeout_seconds,
    })
    atomic_write_json(session_root / "summary.json", document)
    emit(f"MECHANISM_SESSION_STARTED session_id={session_id} output={session_root}", session_log)
    try:
        device = args.device
        require(type(device) is int and device >= 0, "CUDA formal experiment requires a non-negative --device")
        lock_stream = acquire_cuda_lock(device)
        device_snapshot = cuda_snapshot(device)
        assert_no_external_cuda_compute(device_snapshot)
        binary = resolve_cuda_binary(args.binary)
        source_start = source_snapshot()
        build = build_snapshot(binary, "cuda")
        require(build.get("built_source_tree_hash") == source_start["source_tree_hash"],
                "CUDA mechanism binary was not built from the current source inventory")
        dataset_info = {name: config_snapshot(name) for name in ("cora", "citeseer", "ppi")}
        for dataset, info in dataset_info.items():
            expected = DATASETS[dataset]
            manifest = info["srs"]["manifest"]
            require(info["srs"]["manifest_srs_id"] == expected["srs_id"], f"{dataset} SRS id is not canonical")
            require(info["srs"]["domain_sizes"].get("N") == expected["n_domain"] and
                    info["srs"]["domain_sizes"].get("edge") == expected["e_domain"],
                    f"{dataset} trusted SRS lacks the exact registered mechanism domains")
            required_coefficient_count = 2 * expected["e_domain"] + 14
            require(info["srs"]["manifest_degree_bound"] + 1 >= required_coefficient_count,
                    f"{dataset} trusted SRS coefficient prefix is too short for the mechanism ledger requirement")
            envelope = info["srs"]["max_instance_envelope"]
            if dataset == "ppi":
                max_nodes, max_edges, max_graphs = 2923, 33928, 2
            else:
                max_nodes, max_edges, max_graphs = expected["N"], expected["E"], 1
            require(envelope.get("max_nodes", 0) >= max_nodes and envelope.get("max_edges", 0) >= max_edges and
                    envelope.get("max_graphs", 0) >= max_graphs,
                    f"{dataset} trusted SRS instance envelope cannot contain every configured graph")
        document["provenance"] = {
            "source": {key: value for key, value in source_start.items() if key != "source_manifest"},
            "build": {key: value for key, value in build.items() if key != "source_manifest"},
            "device": device_snapshot,
            "datasets": {
                name: {
                    "config_path": info["config_path"], "config_sha256": info["config_sha256"],
                    "checkpoint_path": info["checkpoint_path"],
                    "srs": {key: value for key, value in info["srs"].items() if key != "manifest"},
                    "srs_manifest": info["srs"]["manifest"],
                } for name, info in dataset_info.items()
            },
        }
        atomic_write_json(session_root / "source_manifest.json", source_start["source_manifest"])
        atomic_write_json(session_root / "build_source_manifest.json", build["source_manifest"])
        atomic_write_json(session_root / "configuration_and_srs_manifest.json", document["provenance"]["datasets"])
        document.update({
            "binary_path": str(binary), "srs_by_dataset": {name: info["srs"] for name, info in dataset_info.items()},
            "timeout_seconds_per_cell": args.timeout_seconds,
            "prepare_timeout_seconds": args.prepare_timeout_seconds,
            "pair_order_schedule": "global alternating Original-Control / Control-Original; configuration order rotates by repetition",
            "process_warmup": "each prover process performs an untimed CUDA NTT/INTT plus ephemeral non-SRS non-stable MSM warmup before proof counters; actual SRS/KZG bases remain cold until commitment timing",
        })
        prepared_by_key: dict[str, dict[str, Any]] = {}
        for key, dataset, components in unique_bundle_specs():
            assert_no_external_cuda_compute(device_snapshot)
            try:
                record = prepare_real_bundle(
                    binary=binary, session_root=session_root, dataset=dataset, components=components,
                    config_info=dataset_info[dataset], timeout_seconds=args.prepare_timeout_seconds,
                    session_log=session_log,
                )
            except (OSError, ValueError, subprocess.SubprocessError, KeyboardInterrupt):
                failed_status_path = session_root / "real_inputs" / key / "prepare_status.json"
                if failed_status_path.is_file():
                    prepared.append(load_json(failed_status_path, f"{key} prepare failure record"))
                    document["prepared_inputs"] = list(prepared)
                    publish_progress(session_root, document, cells)
                raise
            prepared.append(record)
            document["prepared_inputs"] = list(prepared)
            prepared_by_key[key] = record
            publish_progress(session_root, document, cells)
        for spec in FORMAL_CONFIGS:
            input_record = prepared_by_key[bundle_key(spec["dataset"], spec["components"])]
            check_expected_spec(spec, input_record)
        document["session_status"] = "running_proofs"
        publish_progress(session_root, document, cells)
        used_seed_tokens: set[bytes] = set()
        pair_index = 0
        for spec, repetition in formal_schedule():
            cfgid = matrix_config_id(spec)
            input_record = prepared_by_key[bundle_key(spec["dataset"], spec["components"])]
            variants = list(VARIANTS)
            if pair_index % 2:
                variants.reverse()
            emit(
                f"PAIR_START config={cfgid} repetition={repetition} order={','.join(variants)} "
                f"N={input_record['actual']['N']} E={input_record['actual']['E']} D={spec['width']} G={input_record['actual']['G']}",
                session_log,
            )
            for variant in variants:
                require(_INTERRUPT_SIGNAL is None, "formal mechanism session received a stop signal")
                cell = run_real_cell(
                    binary=binary, session_root=session_root, spec=spec, input_record=input_record,
                    srs_info=dataset_info[spec["dataset"]]["srs"], device=args.device,
                    timeout_seconds=args.timeout_seconds, pair_index=pair_index, repetition=repetition,
                    variant=variant, used_seed_tokens=used_seed_tokens,
                    device_snapshot=device_snapshot, session_log=session_log,
                )
                cells.append(cell)
                if cell["cell_status"] != "complete":
                    mark_real_pair(cells, cfgid, repetition)
                    document["failure_reason"] = cell.get("failure_reason") or f"formal cell failed: {cfgid} rep {repetition} {variant}"
                    document["stopping_cell"] = {"config_id": cfgid, "repetition": repetition, "variant": variant}
                    publish_progress(session_root, document, cells)
                    raise ValueError(document["failure_reason"])
                publish_progress(session_root, document, cells)
            mark_real_pair(cells, cfgid, repetition)
            try:
                require_repetition_input_stability(cells, cfgid)
            except ValueError as exc:
                for prior_cell in cells:
                    if prior_cell["config_id"] == cfgid and prior_cell["pair_status"] == "paired":
                        prior_cell["pair_status"] = "invalid_repeat_identity"
                        prior_cell["failure_reason"] = str(exc)
                        atomic_write_json(pathlib.Path(prior_cell["cell_json_path"]), prior_cell)
                document["failure_reason"] = str(exc)
                document["stopping_cell"] = {"config_id": cfgid, "repetition": repetition}
                publish_progress(session_root, document, cells)
                raise
            publish_progress(session_root, document, cells)
            if any(cell.get("pair_status") != "paired" for cell in cells if cell["config_id"] == cfgid and cell["repetition"] == repetition):
                document["failure_reason"] = f"Original/Control pair failed identity checks: {cfgid} repetition {repetition}"
                document["stopping_cell"] = {"config_id": cfgid, "repetition": repetition}
                raise ValueError(document["failure_reason"])
            pair_index += 1

        source_end = source_snapshot()
        build_end = build_snapshot(binary, "cuda")
        configs_end = {name: config_snapshot(name) for name in dataset_info}
        revalidate_input_sources(document)
        stable = (
            source_start["source_tree_hash"] == source_end["source_tree_hash"]
            and source_start["git_diff_sha256"] == source_end["git_diff_sha256"]
            and source_start["git_status_sha256"] == source_end["git_status_sha256"]
            and build["binary_sha256"] == build_end["binary_sha256"]
            and build["cmake_cache_sha256"] == build_end["cmake_cache_sha256"]
            and build["source_manifest_sha256"] == build_end["source_manifest_sha256"]
            and all(dataset_info[name]["config_sha256"] == configs_end[name]["config_sha256"] and
                    dataset_info[name]["srs"]["manifest_sha256"] == configs_end[name]["srs"]["manifest_sha256"]
                    for name in dataset_info)
        )
        document["provenance_stable"] = stable
        document["provenance_end"] = {
            "source_tree_hash": source_end["source_tree_hash"], "git_diff_sha256": source_end["git_diff_sha256"],
            "git_status_sha256": source_end["git_status_sha256"], "binary_sha256": build_end["binary_sha256"],
            "cmake_cache_sha256": build_end["cmake_cache_sha256"],
            "source_manifest_sha256": build_end["source_manifest_sha256"],
            "srs_manifest_sha256": {name: configs_end[name]["srs"]["manifest_sha256"] for name in configs_end},
        }
        require(stable, "source/config/build/SRS/data/checkpoint/input provenance changed during the formal session")
        require(len(cells) == document["expected_cell_count"], "formal schedule ended without exactly 66 attempted proof cells")
        require(_INTERRUPT_SIGNAL is None, "formal mechanism session received a stop signal before finalization")
        require(document["verified_cells"] == document["expected_cell_count"] and
                document["complete_pairs"] == document["expected_pair_count"],
                "formal session did not produce 66 independently verified cells and 33 complete pairs")
        document["session_status"] = "complete"
    except (OSError, ValueError, subprocess.SubprocessError, KeyboardInterrupt) as exc:
        if document.get("failure_reason") is None:
            document["failure_reason"] = str(exc) or "interrupted"
        document["session_status"] = "incomplete"
    finally:
        document["prepared_inputs"] = prepared
        update_summary_counts(document, cells)
        if document.get("session_status") != "complete":
            document["session_status"] = "incomplete"
        publish_progress(session_root, document, cells)
        generate_reports(session_root, document)
        emit(
            f"MECHANISM_SESSION_END session_id={session_id} status={document['session_status']} "
            f"cells={len(cells)}/{document['expected_cell_count']} verified={document['verified_cells']} "
            f"pairs={document['complete_pairs']}/{document['expected_pair_count']} summary={session_root / 'summary.json'}",
            session_log,
        )
        if lock_stream is not None:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
            lock_stream.close()
    return 0 if document.get("session_status") == "complete" else 1


def expected_cell_keys() -> set[tuple[str, int, str]]:
    return {
        (matrix_config_id(spec), repetition, variant)
        for spec in FORMAL_CONFIGS
        for repetition in range(1, EXPERIMENT_REPETITIONS + 1)
        for variant in VARIANTS
    }


def check_regular_digest(path_text: str, expected_digest: str, label: str) -> None:
    path = pathlib.Path(path_text)
    require(path.is_file() and not path.is_symlink(), f"{label} artifact is missing or not regular: {path}")
    require(sha256_file(path) == expected_digest, f"{label} SHA-256 mismatch: {path}")


def audit_session(args: argparse.Namespace) -> int:
    session_root = args.session.expanduser().resolve()
    require(session_root.is_dir() and not args.session.is_symlink(), "session path is missing, not a directory, or a symlink")
    document = load_json(session_root / "summary.json", "mechanism summary")
    require(document.get("format") == "zkgat_real_mechanism_experiment_v1", "session summary is not a real-data mechanism run")
    require(document.get("session_status") == "complete", "formal session is incomplete; no formal result audit can pass")
    require(document.get("cell_count") == 66 and document.get("verified_cells") == 66 and document.get("complete_pairs") == 33,
            "session summary does not contain 66 independent verifications and 33 paired repetitions")
    require(document.get("provenance_stable") is True, "formal session did not close with stable source/build/data provenance")
    source_expected = document.get("provenance", {}).get("source", {})
    source_current = source_snapshot()
    require(source_current["source_tree_hash"] == source_expected.get("source_tree_hash") and
            source_current["git_diff_sha256"] == source_expected.get("git_diff_sha256") and
            source_current["git_status_sha256"] == source_expected.get("git_status_sha256"),
            "source tree or working diff changed since the formal run")
    binary = pathlib.Path(document["binary_path"])
    build_expected = document.get("provenance", {}).get("build", {})
    build_current = build_snapshot(binary, "cuda")
    require(build_current["binary_sha256"] == build_expected.get("binary_sha256") and
            build_current["cmake_cache_sha256"] == build_expected.get("cmake_cache_sha256") and
            build_current["source_manifest_sha256"] == build_expected.get("source_manifest_sha256"),
            "CUDA build identity changed since the formal run")
    dataset_current = {name: config_snapshot(name) for name in ("cora", "citeseer", "ppi")}
    for name, current in dataset_current.items():
        recorded = document["provenance"]["datasets"][name]
        require(current["config_sha256"] == recorded["config_sha256"] and
                current["srs"]["manifest_sha256"] == recorded["srs"]["manifest_sha256"],
                f"{name} config or trusted SRS manifest changed since the formal run")
    revalidate_input_sources(document)
    cells = document.get("cells")
    require(isinstance(cells, list) and len(cells) == 66, "session summary cell inventory is incomplete")
    keys: set[tuple[str, int, str]] = set()
    for cell in cells:
        key = (cell.get("config_id"), cell.get("repetition"), cell.get("variant"))
        require(key not in keys, f"session summary contains duplicate cell {key}")
        keys.add(key)
        require(cell.get("cell_status") == "complete" and cell.get("pair_status") == "paired" and
                cell.get("verification_status") == "VERIFY_OK", f"cell is not formally accepted: {key}")
        require(cell.get("backend") == "cuda" and cell.get("cuda_actual") is True and
                cell.get("cuda_kernel_invocations_delta", 0) > 0,
                f"cell lacks actual CUDA proof evidence: {key}")
        require(cell.get("mask_seed_file_removed") is True and cell.get("private_mask_seed_file_mode") == "0600",
                f"cell private mask seed was not handled securely: {key}")
        for field in ("proof_path", "proof_sha256", "statement_path", "statement_file_sha256",
                      "prover_result_path", "prover_result_sha256", "verifier_result_path", "verifier_result_sha256",
                      "prover_log", "prover_log_sha256", "verifier_log", "verifier_log_sha256", "cell_json_path"):
            require(isinstance(cell.get(field), str) and cell[field], f"cell {key} has no {field}")
        check_regular_digest(cell["proof_path"], cell["proof_sha256"], "proof")
        check_regular_digest(cell["statement_path"], cell["statement_file_sha256"], "statement")
        check_regular_digest(cell["prover_result_path"], cell["prover_result_sha256"], "prover receipt")
        check_regular_digest(cell["verifier_result_path"], cell["verifier_result_sha256"], "independent verifier receipt")
        check_regular_digest(cell["prover_log"], cell["prover_log_sha256"], "prover log")
        check_regular_digest(cell["verifier_log"], cell["verifier_log_sha256"], "verifier log")
        cell_doc = load_json(pathlib.Path(cell["cell_json_path"]), "cell record")
        require(cell_doc == cell, f"cell record differs from the collected summary: {key}")
    require(keys == expected_cell_keys(), "session summary does not contain the exact fixed 11 x 2 x 3 configuration matrix")
    for bundle in document.get("prepared_inputs", []):
        check_regular_digest(bundle["input_bundle_path"], bundle["bundle_sha256"], "real input bundle")
        check_regular_digest(bundle["metadata_path"], bundle["metadata_sha256"], "real input metadata")
        metadata = load_json(pathlib.Path(bundle["metadata_path"]), "real-input metadata")
        require(metadata.get("schema") == REAL_INPUT_SCHEMA and metadata.get("bundle", {}).get("sha256") == bundle["bundle_sha256"],
                f"prepared real-input metadata is inconsistent: {bundle['bundle_key']}")
    if args.reverify:
        binary = pathlib.Path(document["binary_path"])
        require(binary.is_file() and not binary.is_symlink(), f"session prover/verifier binary is missing: {binary}")
        audit_root = session_root / "independent_reaudit"
        audit_root.mkdir(mode=0o700, exist_ok=True)
        audit_cells: list[dict[str, Any]] = []
        for index, cell in enumerate(cells, start=1):
            result_path = audit_root / f"verify_{index:03d}.json"
            log_path = audit_root / f"verify_{index:03d}.log"
            cmd = [str(binary), "verify", "--proof", cell["proof_path"], "--statement", cell["statement_path"],
                   "--srs", document["provenance"]["datasets"][cell["dataset"]]["srs"]["path"],
                   "--result-json", str(result_path)]
            proc = run_process(cmd, log_path, 0.0, "post-session-independent-cpu-verifier")
            require(proc["return_code"] == 0 and not proc["interrupted"],
                    f"post-session independent verifier failed for {(cell['config_id'], cell['repetition'], cell['variant'])}")
            receipt = load_json(result_path, "post-session independent verifier receipt")
            prover_result = load_json(pathlib.Path(cell["prover_result_path"]), "prover result")
            verify_srs = prover_result.get("srs")
            require(isinstance(verify_srs, dict), "prover result has no SRS receipt during re-audit")
            checked = validate_verify_result(receipt, proof_sha256=cell["proof_sha256"],
                                             statement_sha256=cell["statement_sha256"], srs=verify_srs)
            audit_cells.append({
                "config_id": cell["config_id"], "repetition": cell["repetition"], "variant": cell["variant"],
                "status": "VERIFY_OK", "verify_time_ms": checked["verify_time_ms"],
                "receipt_path": str(result_path), "receipt_sha256": sha256_file(result_path),
                "log_path": str(log_path), "exit_code": proc["return_code"],
            })
        require(len(audit_cells) == 66, "post-session verifier re-audit did not cover all 66 proofs")
    else:
        audit_cells = [{"config_id": cell["config_id"], "repetition": cell["repetition"],
                        "variant": cell["variant"], "status": "RECEIPT_VERIFIED"}
                       for cell in cells]
    audit = {
        "format": "zkgat_real_mechanism_result_audit_v1", "session_id": document["session_id"],
        "created_utc": utc_now(), "status": "PASS", "reverified_independently": bool(args.reverify),
        "summary_sha256": sha256_file(session_root / "summary.json"),
        "cell_count": len(audit_cells), "verified_cells": sum(item["status"] == "VERIFY_OK" for item in audit_cells)
        if args.reverify else 66, "cells": audit_cells,
    }
    atomic_write_json(session_root / "audit.json", audit)
    print(f"RESULT_AUDIT_PASS session={session_root} cells={len(audit_cells)} reverified={str(args.reverify).lower()}")
    return 0


def regenerate_summary(args: argparse.Namespace) -> int:
    session_root = args.session.expanduser().resolve()
    require(session_root.is_dir() and not args.session.is_symlink(), "session path is missing, not a directory, or a symlink")
    document = load_json(session_root / "summary.json", "mechanism summary")
    audit = load_json(session_root / "audit.json", "mechanism result audit")
    require(audit.get("status") == "PASS" and audit.get("session_id") == document.get("session_id"),
            "results can be regenerated only after a passing audit for this session")
    require(audit.get("summary_sha256") == sha256_file(session_root / "summary.json"),
            "summary changed after result audit")
    require(document.get("session_status") == "complete" and document.get("cell_count") == 66,
            "cannot create formal tables from an incomplete experiment")
    generate_reports(session_root, document)
    print(f"PAPER_TABLES_REGENERATED session={session_root} feature_folding={session_root / 'feature_folding_summary.md'} "
          f"shared_neighborhood={session_root / 'shared_neighborhood_summary.md'}")
    return 0


def launch_formal(args: argparse.Namespace) -> int:
    require(0 <= args.timeout_seconds <= 599 and 0 <= args.prepare_timeout_seconds <= 599,
            "formal runner timeouts must be 0 (unlimited) or at most 599 seconds")
    output_root = safe_output_root(DEFAULT_OUTPUT_ROOT)
    launch_root = output_root / "launchers"
    launch_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    launch_dir = launch_root / f"cuda_{stamp}_{uuid.uuid4().hex[:10]}"
    launch_dir.mkdir(mode=0o700)
    log_path = launch_dir / "runner.log"
    command = [sys.executable, str(pathlib.Path(__file__).resolve()), "run", "--device", str(args.device),
               "--timeout-seconds", str(args.timeout_seconds), "--prepare-timeout-seconds", str(args.prepare_timeout_seconds)]
    if args.binary is not None:
        command.extend(["--binary", str(args.binary.expanduser().resolve())])
    if args.timeout_seconds == 0:
        command.extend([])
    with log_path.open("ab", buffering=0) as log:
        child = subprocess.Popen(command, cwd=PROJECT_ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True, close_fds=True, umask=0o077)
    atomic_write_text(launch_dir / "launcher.pid", f"{child.pid}\n")
    atomic_write_json(launch_dir / "launcher.json", {
        "format": "zkgat_mechanism_detached_launcher_v1", "pid": child.pid,
        "process_group": child.pid, "created_utc": utc_now(), "command": command,
        "log_path": str(log_path), "runner_output_root": str(output_root),
    })
    print(f"LAUNCHER_STARTED pid={child.pid} launcher_dir={launch_dir} log={log_path}")
    return 0


def install_signal_handlers() -> None:
    def handle(signum: int, _frame: Any) -> None:
        global _INTERRUPT_SIGNAL
        _INTERRUPT_SIGNAL = signum
        for pid in list(ACTIVE_CHILDREN):
            try:
                os.killpg(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, handle)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the fixed 11-configuration real-data CUDA zkGAT mechanism experiment.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="prepare all real inputs and run the 66 formal CUDA proofs")
    run.add_argument("--binary", type=pathlib.Path, help="CUDA gatzk_mechanism binary; defaults to build_cuda/gatzk_mechanism")
    run.add_argument("--device", type=integer_arg, required=True, help="exclusive CUDA device index")
    run.add_argument("--timeout-seconds", type=integer_arg, default=0,
                     help="per-cell prove+verify timeout; 0 means unlimited for the official run")
    run.add_argument("--prepare-timeout-seconds", type=integer_arg, default=0,
                     help="real-input preparation timeout; 0 means unlimited")

    launch = subparsers.add_parser("launch", help="start the complete CUDA run in a detached SSH-resilient session")
    launch.add_argument("--binary", type=pathlib.Path, help="CUDA gatzk_mechanism binary; defaults to build_cuda/gatzk_mechanism")
    launch.add_argument("--device", type=integer_arg, required=True, help="exclusive CUDA device index")
    launch.add_argument("--timeout-seconds", type=integer_arg, default=0,
                        help="per-cell prove+verify timeout; 0 means unlimited for the official run")
    launch.add_argument("--prepare-timeout-seconds", type=integer_arg, default=0,
                        help="real-input preparation timeout; 0 means unlimited")

    audit = subparsers.add_parser("audit", help="audit exact cell/artifact/hash receipts and optionally reverify all proofs")
    audit.add_argument("session", type=pathlib.Path, help="session directory printed by the runner")
    audit.add_argument("--reverify", action="store_true", help="run the independent CPU verifier again for all 66 proofs")

    summarize = subparsers.add_parser("summarize", help="regenerate the two paper tables after a passing audit")
    summarize.add_argument("session", type=pathlib.Path, help="session directory printed by the runner")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    install_signal_handlers()
    try:
        if args.command == "run":
            return execute_formal(args)
        if args.command == "launch":
            return launch_formal(args)
        if args.command == "audit":
            return audit_session(args)
        if args.command == "summarize":
            return regenerate_summary(args)
        raise ValueError(f"unsupported command: {args.command}")
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"mechanism-runner: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
