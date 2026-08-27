#!/usr/bin/env python3
"""Baseline-local resource and output containment policy for ezkl.

This module only checks names, paths, and file presence.  It deliberately
does not open dataset, checkpoint, SRS, or public-output payloads.  The native
adapter remains the authority for their authenticated contents.
"""

from __future__ import annotations

import json
import os
import stat
import hashlib
from pathlib import Path
from typing import Any, Iterable

from formal_source_manifest import checked_path, lexical_baseline_root


ROOT = lexical_baseline_root(Path(__file__).absolute().parent.parent)
RESOURCE_SCHEMA = "ezkl_formal_resource_manifest_v1"
BASELINE = "ezkl"
FORMAL_ROUTE = "ezkl_gat_native_compressed_csr_attention_full"
FORMAL_SOURCE_SRS_MAX_K = 28
CEREMONY_ROOT = Path("resources") / "ceremony"
CEREMONY_SRS = CEREMONY_ROOT / "source.srs"
CEREMONY_DESCRIPTOR = CEREMONY_ROOT / "ceremony.descriptor.json"
DATASET_ORDER = (
    "cora_full",
    "citeseer_full",
    "pubmed_full",
    "ppi_full_formal",
    "ogbn_arxiv_full",
)
DATASET_SLUG = {
    "cora_full": "cora",
    "citeseer_full": "citeseer",
    "pubmed_full": "pubmed",
    "ppi_full_formal": "ppi",
    "ogbn_arxiv_full": "ogbn_arxiv",
}
DATASET_FILES = ("meta.cfg", "edges.txt")
CHECKPOINT_FILES = ("manifest.json", "tensors.txt")
STATIC_DESCRIPTOR = Path("configs") / "formal_operation_registry.json"

# The adapter and ezkl commands must leave exactly this top-level file set in
# a committed result cell.  Temporary stage directories and command logs live
# outside the cell, under results/runs_tmp and results/logs respectively.
#
# These files contain private model/witness material.  They are deliberately
# not part of the committed inventory: the native prover may consume them
# only while the per-session temporary stage exists, and the runner removes
# them before moving the public cell into results/runs.
PRIVATE_ARTIFACTS = frozenset({"model_parameters.i64.txt", "witness.json"})
ADAPTER_ARTIFACTS = frozenset(
    {
        "adapter_manifest.json",
        "backend_aux_table_spec.json",
        "backend_aux_tables.i64.bin",
        "canonical_forward.json",
        "checkpoint_hash.txt",
        "csr_segment_layout.bin",
        "csr_segment_layout.json",
        "formal_main_tables.i64.bin",
        "formal_quant_config.json",
        "formal_table_spec.json",
        "graph_attention_spec.json",
        "graph_hash.txt",
        "integer_bound_ledger.json",
        "metadata.json",
        "model_registration.json",
        "node_features.i64.bin",
        "security_profile.json",
        "shape_hash.txt",
    }
)
PROOF_ARTIFACTS = frozenset(
    {
        "circuit_manifest.json",
        "network.compiled",
        "pk.key",
        "pk.srs",
        "proof.json",
        "public_statement.json",
        "settings.json",
        "verify.json",
        "vk.key",
        "vk.srs",
    }
)
CELL_ARTIFACTS = ADAPTER_ARTIFACTS | PROOF_ARTIFACTS | frozenset({"run_manifest.json"})
RESULT_BACKENDS = ("cpu", "cuda")


def baseline_root(root: Path | str = ROOT) -> Path:
    return lexical_baseline_root(root)


def is_inside(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def resolve_local(
    root: Path | str,
    value: Path | str,
    label: str,
    *,
    allow_missing: bool = True,
) -> Path:
    return checked_path(baseline_root(root), value, label, allow_missing=allow_missing)


def dataset_slug(dataset: str) -> str:
    try:
        return DATASET_SLUG[str(dataset)]
    except KeyError as exc:
        raise ValueError(f"unsupported formal dataset: {dataset}") from exc


def dataset_paths(root: Path | str, dataset: str) -> dict[str, Path]:
    root_path = baseline_root(root)
    slug = dataset_slug(dataset)
    return {
        "config": resolve_local(root_path, Path("configs") / f"{slug}_full.cfg", "config"),
        "dataset_root": resolve_local(root_path, Path("data/cache") / slug, "dataset_root"),
        "checkpoint": resolve_local(
            root_path, Path("artifacts/checkpoints") / f"{slug}_gat", "checkpoint"
        ),
        "public_root": resolve_local(root_path, Path("resources/public_y"), "public_y root"),
    }


def manifest_payload(root: Path | str = ROOT) -> dict[str, Any]:
    root_path = baseline_root(root)
    datasets: dict[str, Any] = {}
    for dataset in DATASET_ORDER:
        paths = dataset_paths(root_path, dataset)
        datasets[dataset] = {
            "config": str(paths["config"].relative_to(root_path)),
            "dataset_root": str(paths["dataset_root"].relative_to(root_path)),
            "checkpoint": str(paths["checkpoint"].relative_to(root_path)),
            "required_dataset_files": list(DATASET_FILES)
            + (["node_ptr.txt"] if dataset == "ppi_full_formal" else []),
            "required_checkpoint_files": list(CHECKPOINT_FILES),
            "status": "awaiting_user_staged_resources",
        }
    return {
        "schema": RESOURCE_SCHEMA,
        "baseline": BASELINE,
        "route_id": FORMAL_ROUTE,
        "root_policy": "dataset_and_checkpoint_resources_must_be_staged_below_baseline",
        "dataset_execution_policy": "prepare_and_preflight_never_load_dataset_or_checkpoint_contents",
        "datasets": datasets,
        "static_contract": {
            "operation_registry": str(STATIC_DESCRIPTOR),
            "required_configs": [
                str(Path("configs") / f"{dataset_slug(name)}_full.cfg")
                for name in DATASET_ORDER
            ],
        },
        "setup_resources": {
            "ceremony_srs": str(CEREMONY_SRS),
            "ceremony_descriptor": str(CEREMONY_DESCRIPTOR),
            "source_srs_max_k": FORMAL_SOURCE_SRS_MAX_K,
            "required_files": [str(CEREMONY_DESCRIPTOR), str(CEREMONY_SRS)],
            "status": "awaiting_user_staged_resources",
        },
        "output_policy": {
            "cell_root": "results/runs/<backend>/<dataset>",
            "temporary_root": "results/runs_tmp",
            "logs_root": "results/logs",
            "exact_cell_inventory": sorted(CELL_ARTIFACTS),
        },
    }


def _read_manifest(root: Path) -> dict[str, Any]:
    path = resolve_local(root, "resources/resource_manifest.json", "resource manifest")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read resource manifest: {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != RESOURCE_SCHEMA:
        raise ValueError("resource manifest schema mismatch")
    if payload.get("baseline") != BASELINE or payload.get("route_id") != FORMAL_ROUTE:
        raise ValueError("resource manifest baseline/route identity mismatch")
    if set(payload.get("datasets", {})) != set(DATASET_ORDER):
        raise ValueError("resource manifest dataset coverage mismatch")
    expected = manifest_payload(root)
    if payload.get("setup_resources") != expected["setup_resources"]:
        raise ValueError("resource manifest ceremony/setup resource contract mismatch")
    if payload.get("static_contract") != expected["static_contract"]:
        raise ValueError("resource manifest static contract mismatch")
    if payload.get("output_policy") != expected["output_policy"]:
        raise ValueError("resource manifest output policy mismatch")
    for dataset in DATASET_ORDER:
        if payload["datasets"].get(dataset) != expected["datasets"][dataset]:
            raise ValueError(f"resource manifest path/requirements mismatch: {dataset}")
    return payload


def read_resource_manifest(root: Path | str = ROOT) -> dict[str, Any]:
    return _read_manifest(baseline_root(root))


def check_resource_layout(
    root: Path | str = ROOT,
    datasets: Iterable[str] = DATASET_ORDER,
    *,
    require_staged: bool = False,
) -> dict[str, Any]:
    """Check resource names and presence without reading resource contents."""

    root_path = baseline_root(root)
    manifest = _read_manifest(root_path)
    selected = tuple(dict.fromkeys(str(item) for item in datasets))
    unknown = sorted(set(selected) - set(DATASET_ORDER))
    if unknown:
        raise ValueError(f"unsupported formal dataset(s): {unknown}")
    missing: list[str] = []
    configs: list[str] = []
    for dataset in selected:
        paths = dataset_paths(root_path, dataset)
        configs.append(str(paths["config"].relative_to(root_path)))
        if not _is_private_regular_file(paths["config"]):
            missing.append(str(paths["config"]))
        if not require_staged:
            continue
        for name in DATASET_FILES:
            path = paths["dataset_root"] / name
            if not _is_private_regular_file(path):
                missing.append(str(path))
        if dataset == "ppi_full_formal":
            path = paths["dataset_root"] / "node_ptr.txt"
            if not _is_private_regular_file(path):
                missing.append(str(path))
        for name in CHECKPOINT_FILES:
            path = paths["checkpoint"] / name
            if not _is_private_regular_file(path):
                missing.append(str(path))
    if require_staged:
        ceremony_srs = resolve_local(root_path, CEREMONY_SRS, "ceremony SRS")
        ceremony_descriptor = resolve_local(
            root_path, CEREMONY_DESCRIPTOR, "ceremony descriptor"
        )
        for path in (ceremony_srs, ceremony_descriptor):
            if not _is_private_regular_file(path):
                missing.append(str(path))
    if missing:
        raise FileNotFoundError("missing baseline-local staged resources:\n" + "\n".join(missing))
    return {
        "schema": "ezkl_formal_resource_check_v1",
        "baseline": BASELINE,
        "route_id": FORMAL_ROUTE,
        "require_staged": require_staged,
        "datasets": list(selected),
        "configs": configs,
        "status": "staged" if require_staged else "layout_ready",
        "manifest": str((root_path / "resources/resource_manifest.json").relative_to(root_path)),
        "contents_read": False,
        "resource_manifest_digest": _sha256(root_path / "resources/resource_manifest.json"),
    }


def _is_private_regular_file(path: Path) -> bool:
    """Return whether *path* is a single-link regular file without following links."""

    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise ValueError(f"cannot inspect staged resource: {path}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"staged resource is a symlink: {path}")
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"staged resource is not a regular file: {path}")
    if metadata.st_nlink != 1:
        raise ValueError(f"staged resource is hard-linked: {path}")
    return True


def validate_ceremony_material(root: Path | str = ROOT) -> dict[str, Any]:
    """Authenticate the canonical baseline-local ceremony descriptor and SRS.

    This deliberately hashes only the trusted SRS bytes; it does not parse the
    SRS or load any dataset/checkpoint contents.  Native setup remains the
    authority for the cryptographic SRS structure and exact source ``k``.
    """

    root_path = baseline_root(root)
    descriptor_path = resolve_local(root_path, CEREMONY_DESCRIPTOR, "ceremony descriptor")
    srs_path = resolve_local(root_path, CEREMONY_SRS, "ceremony SRS")
    if not _is_private_regular_file(descriptor_path):
        raise FileNotFoundError(f"missing baseline-local ceremony descriptor: {descriptor_path}")
    if not _is_private_regular_file(srs_path):
        raise FileNotFoundError(f"missing baseline-local ceremony SRS: {srs_path}")
    try:
        descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read ceremony descriptor: {descriptor_path}: {exc}") from exc
    if not isinstance(descriptor, dict):
        raise ValueError("ceremony descriptor must be a JSON object")
    if descriptor.get("descriptor_kind") != "ezkl_trusted_ceremony_srs_v1":
        raise ValueError("ceremony descriptor kind is not registered for formal ezkl")
    ceremony_id = descriptor.get("ceremony_id")
    if not isinstance(ceremony_id, str) or not ceremony_id:
        raise ValueError("ceremony descriptor lacks a stable ceremony_id")
    if descriptor.get("source_srs_max_k") != FORMAL_SOURCE_SRS_MAX_K:
        raise ValueError("formal ezkl requires ceremony source_srs_max_k exactly 28")
    declared_digest = descriptor.get("source_srs_sha256")
    if (
        not isinstance(declared_digest, str)
        or len(declared_digest) != len("sha256:") + 64
        or not declared_digest.startswith("sha256:")
        or any(ch not in "0123456789abcdef" for ch in declared_digest[7:])
    ):
        raise ValueError("ceremony descriptor lacks a valid source_srs_sha256")
    actual_digest = _sha256(srs_path)
    if declared_digest != actual_digest:
        raise ValueError(
            f"trusted ceremony SRS digest mismatch: expected {declared_digest}, got {actual_digest}"
        )
    return {
        "descriptor_kind": descriptor["descriptor_kind"],
        "ceremony_id": ceremony_id,
        "source_srs_sha256": actual_digest,
        "source_srs_max_k": FORMAL_SOURCE_SRS_MAX_K,
        "ceremony_descriptor_path": str(descriptor_path),
        "source_srs_path": str(srs_path),
        "ceremony_descriptor_sha256": _sha256(descriptor_path),
    }


def validate_results_tree(
    root: Path | str = ROOT,
    *,
    require_complete: bool = False,
) -> set[tuple[str, str]]:
    """Validate the complete recursive ``results/runs`` namespace.

    Unknown backends/datasets, nested files/directories, links, hard links,
    and special entries are rejected even for a partial collection.  Missing
    cells are allowed only when ``require_complete`` is false.
    """

    root_path = baseline_root(root)
    expected = {(dataset, backend) for dataset in DATASET_ORDER for backend in RESULT_BACKENDS}
    runs_path = root_path / "results" / "runs"
    try:
        runs_path = resolve_local(root_path, Path("results") / "runs", "formal results root", allow_missing=True)
    except (FileNotFoundError, ValueError) as exc:
        if isinstance(exc, FileNotFoundError) and not require_complete:
            return set()
        raise ValueError(str(exc)) from exc
    if not runs_path.exists():
        if require_complete:
            raise ValueError(f"formal results root is missing: {runs_path}")
        return set()
    if not runs_path.is_dir():
        raise ValueError(f"formal results root is not a directory: {runs_path}")
    found: set[tuple[str, str]] = set()
    for backend_entry in runs_path.iterdir():
        metadata = backend_entry.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"formal results root contains a symlink: {backend_entry}")
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"formal results root contains a non-directory: {backend_entry}")
        if backend_entry.name not in RESULT_BACKENDS:
            raise ValueError(f"unknown formal result backend directory: {backend_entry.name}")
        for dataset_entry in backend_entry.iterdir():
            dataset_metadata = dataset_entry.lstat()
            if stat.S_ISLNK(dataset_metadata.st_mode):
                raise ValueError(f"formal result backend contains a symlink: {dataset_entry}")
            if not stat.S_ISDIR(dataset_metadata.st_mode):
                raise ValueError(f"formal result backend contains a non-directory: {dataset_entry}")
            if dataset_entry.name not in DATASET_ORDER:
                raise ValueError(f"unknown formal result dataset directory: {dataset_entry.name}")
            validate_cell_inventory(dataset_entry)
            found.add((dataset_entry.name, backend_entry.name))
    missing = sorted(expected - found)
    if require_complete and missing:
        raise ValueError("missing required formal result cells: " + ", ".join(
            f"{backend}/{dataset}" for dataset, backend in missing
        ))
    return found


def canonical_cell_dir(root: Path | str, backend: str, dataset: str) -> Path:
    if backend not in {"cpu", "cuda"}:
        raise ValueError(f"unsupported backend: {backend}")
    if dataset not in DATASET_ORDER:
        raise ValueError(f"unsupported formal dataset: {dataset}")
    root_path = baseline_root(root)
    return resolve_local(root_path, Path("results/runs") / backend / dataset, "result cell")


def validate_cell_inventory(cell_dir: Path | str, *, require_manifest: bool = True) -> None:
    raw_cell = Path(cell_dir).expanduser()
    anchor = Path(raw_cell.anchor or "/")
    try:
        cell = checked_path(anchor, raw_cell, "result cell")
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError(str(exc)) from exc
    if not cell.is_dir():
        raise ValueError(f"result cell is not a directory: {cell}")
    actual: set[str] = set()
    for entry in cell.iterdir():
        metadata = entry.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"result cell contains a symlink artifact: {entry.name}")
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"result cell contains non-regular artifact: {entry.name}")
        if metadata.st_nlink != 1:
            raise ValueError(f"result cell contains hard-linked artifact: {entry.name}")
        actual.add(entry.name)
    expected = CELL_ARTIFACTS if require_manifest else (ADAPTER_ARTIFACTS | frozenset({"run_manifest.json"}))
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(f"result cell exact inventory mismatch missing={missing} extra={extra}")


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    """Write a JSON result atomically while rejecting unsafe destinations."""

    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    atomic_write_bytes(Path(path), encoded)


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Atomically write bytes to a checked baseline-local destination."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    anchor = Path(path.anchor or "/")
    checked_path(anchor, path.parent, "result parent")
    checked_path(anchor, path, "result destination", allow_missing=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    checked_path(anchor, temporary, "result temporary destination", allow_missing=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def atomic_write_text(path: Path, payload: str) -> None:
    """Atomically write UTF-8 text to a checked destination."""

    atomic_write_bytes(Path(path), payload.encode("utf-8"))
