#!/usr/bin/env python3
"""Shared fail-closed contracts for the formal zkDL execution chain.

This module is production runner infrastructure, not a test adapter.  It keeps
all paths, resources, manifests, and result artifacts inside this baseline and
uses a single exact cell layout for both the runner and the independent
collector/validator.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any, Iterable


BASELINE_ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("cora", "citeseer", "pubmed", "ppi", "ogbn-arxiv")
DEVICES = ("cpu", "gpu")
CONFIGS = {
    "cora": "cora_full.cfg",
    "citeseer": "citeseer_full.cfg",
    "pubmed": "pubmed_full.cfg",
    "ppi": "ppi_full.cfg",
    "ogbn-arxiv": "ogbn_arxiv_full.cfg",
}
PROTOCOL_PATH = BASELINE_ROOT / "GAT-ZKML-zkDL.md"
RESOURCE_MANIFEST_PATH = BASELINE_ROOT / "resources" / "resource_manifest.json"
RESOURCE_MANIFEST_SCHEMA = "zkDL_gat_resource_manifest_v1"
CELL_SCHEMA = "zkDL_gat_cell_success_manifest_v1"
CELL_FILES = frozenset(
    {
        "formal_artifacts/preflight.json",
        "formal_artifacts/canonical_relation.json",
        "formal_artifacts/statement.bin",
        "formal_artifacts/witness.private.bin",
        "formal_artifacts/proof.bin",
        "proof_manifest.json",
        "verifier_manifest.json",
        "result.json",
        "success_manifest.json",
    }
)
CELL_DIRS = frozenset({"formal_artifacts"})


def _relative(path: Path, root: Path = BASELINE_ROOT) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def ensure_contained(path: Path, *, root: Path = BASELINE_ROOT, label: str = "path") -> Path:
    """Resolve *path* and reject absolute escapes and symlink components."""
    root_resolved = root.resolve()
    raw_candidate = path if path.is_absolute() else root_resolved / path
    # Check the lexical path before resolving it.  A symlink whose target is
    # still inside the baseline is forbidden as well: artifacts must have one
    # unambiguous owner and may not alias another cell/resource.
    raw_candidate = Path(os.path.abspath(str(raw_candidate)))
    try:
        raw_parts = raw_candidate.relative_to(root_resolved).parts
    except ValueError as error:
        raise ValueError(f"{label} escapes baseline root: {path}") from error
    current = root_resolved
    for part in raw_parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{label} contains a symlink component: {current}")
    candidate = raw_candidate.resolve(strict=False)
    if candidate != root_resolved and root_resolved not in candidate.parents:
        raise ValueError(f"{label} must stay inside {root_resolved}: {path}")
    return candidate


def ensure_regular_file(path: Path, *, label: str, allow_missing: bool = False) -> Path:
    path = ensure_contained(path, label=label)
    if not path.exists():
        if allow_missing:
            return path
        raise FileNotFoundError(f"missing {label}: {path}")
    st = path.lstat()
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
        raise ValueError(f"{label} must be a regular non-symlink file: {path}")
    if st.st_nlink != 1:
        raise ValueError(f"{label} must not be hard-linked: {path}")
    return path


def ensure_directory(path: Path, *, label: str, allow_missing: bool = False) -> Path:
    path = ensure_contained(path, label=label)
    if not path.exists():
        if allow_missing:
            return path
        raise FileNotFoundError(f"missing {label}: {path}")
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{label} must be a real directory: {path}")
    return path


def sha256_file(path: Path) -> str:
    path = ensure_regular_file(path, label="hashed file")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def atomic_write_bytes(path: Path, payload: bytes, *, mode: int = 0o644) -> None:
    path = ensure_contained(path, label="output")
    path.parent.mkdir(parents=True, exist_ok=True)
    ensure_directory(path.parent, label="output parent")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def atomic_write_json(path: Path, value: Any, *, mode: int = 0o644) -> None:
    atomic_write_bytes(path, canonical_json(value), mode=mode)


def read_json(path: Path, *, label: str = "JSON") -> dict[str, Any]:
    ensure_regular_file(path, label=label)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def config_path(dataset: str) -> Path:
    if dataset not in CONFIGS:
        raise ValueError(f"unknown dataset: {dataset}")
    return ensure_regular_file(
        BASELINE_ROOT / "configs" / CONFIGS[dataset], label=f"{dataset} config"
    )


def parse_config(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def _manifest_digest(manifest: dict[str, Any]) -> str:
    material = dict(manifest)
    material.pop("manifest_id", None)
    return f"sha256:{sha256_bytes(canonical_json(material))}"


def load_resource_manifest(*, require_staged: bool = False) -> tuple[dict[str, Any], str]:
    manifest = read_json(RESOURCE_MANIFEST_PATH, label="resource manifest")
    if manifest.get("schema") != RESOURCE_MANIFEST_SCHEMA:
        raise ValueError("unsupported zkDL resource manifest schema")
    if manifest.get("baseline") != "zkdl":
        raise ValueError("resource manifest baseline mismatch")
    if manifest.get("manifest_id") != _manifest_digest(manifest):
        raise ValueError("resource manifest digest mismatch")
    datasets = manifest.get("datasets")
    if not isinstance(datasets, dict) or set(datasets) != set(DATASETS):
        raise ValueError("resource manifest must enumerate exactly five datasets")
    for dataset in DATASETS:
        entry = datasets[dataset]
        if not isinstance(entry, dict):
            raise ValueError(f"resource manifest entry is not an object: {dataset}")
        for key in ("cache", "checkpoint_bundle"):
            raw = entry.get(key)
            if not isinstance(raw, str) or not raw or Path(raw).is_absolute():
                raise ValueError(f"invalid {dataset} {key} resource path")
            ensure_contained(BASELINE_ROOT / raw, label=f"{dataset} {key}")
        if require_staged:
            cache = ensure_directory(
                BASELINE_ROOT / entry["cache"], label=f"{dataset} cache"
            )
            checkpoint = ensure_directory(
                BASELINE_ROOT / entry["checkpoint_bundle"], label=f"{dataset} checkpoint"
            )
            ensure_regular_file(cache / "meta.cfg", label=f"{dataset} cache metadata")
            ensure_regular_file(cache / "edges.txt", label=f"{dataset} edge list")
            if not (cache / "features.txt").is_file() and not (cache / "features.npy").is_file():
                raise FileNotFoundError(f"{dataset} cache has no features.txt/features.npy")
            ensure_regular_file(checkpoint / "manifest.json", label=f"{dataset} model manifest")
            ensure_regular_file(checkpoint / "tensors.txt", label=f"{dataset} model tensors")
    return manifest, str(manifest["manifest_id"])


def validate_config_resources(dataset: str, config: dict[str, str], manifest: dict[str, Any]) -> None:
    entry = manifest["datasets"][dataset]
    if config.get("dataset") != dataset:
        raise ValueError(f"config dataset mismatch: {dataset}")
    for key in ("cache", "checkpoint_bundle"):
        if config.get(key) != entry[key]:
            raise ValueError(f"config/resource manifest mismatch for {dataset}: {key}")
        ensure_contained(BASELINE_ROOT / config[key], label=f"{dataset} {key}")


def resource_inventory_digest(dataset: str, manifest: dict[str, Any]) -> str:
    """Digest staged resource identities without exposing private model data."""
    entry = manifest["datasets"][dataset]
    records: list[dict[str, Any]] = []
    for key in ("cache", "checkpoint_bundle"):
        root = ensure_directory(BASELINE_ROOT / entry[key], label=f"{dataset} {key}")
        for path in sorted(root.rglob("*")):
            if path.is_dir():
                if path.is_symlink():
                    raise ValueError(f"resource directory is a symlink: {path}")
                continue
            ensure_regular_file(path, label=f"{dataset} resource")
            records.append(
                {
                    "path": _relative(path),
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return f"sha256:{sha256_bytes(canonical_json(records))}"


def cell_path(output_root: Path, dataset: str, device: str) -> Path:
    if dataset not in DATASETS or device not in DEVICES:
        raise ValueError(f"invalid cell: {dataset}/{device}")
    output_root = ensure_contained(output_root, label="runs root")
    if output_root == BASELINE_ROOT:
        raise ValueError("runs root must be a baseline-local child directory")
    if output_root.exists() and output_root.is_symlink():
        raise ValueError("runs root may not be a symlink")
    return ensure_contained(output_root / dataset / device / "warm", label="run cell")


def prepare_cell(run: Path, *, replace: bool = False) -> None:
    run = ensure_contained(run, label="run cell")
    if run.exists():
        if run.is_symlink() or not run.is_dir():
            raise ValueError(f"run cell is not a real directory: {run}")
        if any(run.iterdir()):
            if not replace:
                raise FileExistsError(
                    f"run cell already contains artifacts; pass --replace explicitly: {run}"
                )
            # The target is a single validated dataset/device cell, never a
            # workspace or recursive baseline root.
            for child in run.iterdir():
                if child.is_symlink() or child.is_dir():
                    if child.is_symlink():
                        child.unlink()
                    else:
                        import shutil

                        shutil.rmtree(child)
                else:
                    child.unlink()
    else:
        run.mkdir(parents=True, exist_ok=False)
    ensure_directory(run, label="run cell")


def artifact_inventory(run: Path, *, require_complete: bool = False) -> dict[str, dict[str, int | str]]:
    """Return a strict cell inventory and reject symlink/hardlink/special files."""
    run = ensure_directory(run, label="run cell")
    files: dict[str, dict[str, int | str]] = {}
    directories: set[str] = set()
    for path in sorted(run.rglob("*")):
        rel = path.relative_to(run).as_posix()
        if path.is_symlink():
            raise ValueError(f"run artifact symlink is forbidden: {rel}")
        if path.is_dir():
            directories.add(rel)
            continue
        ensure_regular_file(path, label=f"run artifact {rel}")
        if rel in files:
            raise ValueError(f"duplicate run artifact: {rel}")
        files[rel] = {"size": path.stat().st_size, "sha256": sha256_file(path)}
    if not directories.issubset(CELL_DIRS):
        extra = sorted(directories - CELL_DIRS)
        raise ValueError(f"unexpected run artifact directories: {extra}")
    if require_complete:
        missing = sorted(CELL_FILES - files.keys())
        extra = sorted(files.keys() - CELL_FILES)
        if missing or extra:
            raise ValueError(f"cell artifact inventory mismatch missing={missing} extra={extra}")
    return files


def relative_manifest_path(raw: Any, *, run: Path, label: str) -> Path:
    if not isinstance(raw, str) or not raw or Path(raw).is_absolute():
        raise ValueError(f"{label} must be a relative path")
    path = ensure_contained(BASELINE_ROOT / raw, label=label)
    try:
        path.relative_to(run)
    except ValueError as error:
        raise ValueError(f"{label} escapes run cell: {raw}") from error
    return path


def protocol_digest() -> str:
    return f"sha256:{sha256_file(PROTOCOL_PATH)}"


def config_digest(path: Path) -> str:
    return f"sha256:{sha256_file(path)}"
