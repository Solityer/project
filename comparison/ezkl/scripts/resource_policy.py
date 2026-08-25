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
from pathlib import Path
from typing import Any, Iterable

from formal_source_manifest import checked_path, lexical_baseline_root


ROOT = lexical_baseline_root(Path(__file__).absolute().parent.parent)
RESOURCE_SCHEMA = "ezkl_formal_resource_manifest_v1"
BASELINE = "ezkl"
FORMAL_ROUTE = "ezkl_gat_native_compressed_csr_attention_full"
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
        "model_parameters.i64.txt",
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
        "witness.json",
    }
)
CELL_ARTIFACTS = ADAPTER_ARTIFACTS | PROOF_ARTIFACTS | frozenset({"run_manifest.json"})


def baseline_root(root: Path | str = ROOT) -> Path:
    return lexical_baseline_root(root)


def is_inside(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def resolve_local(root: Path | str, value: Path | str, label: str) -> Path:
    return checked_path(baseline_root(root), value, label, allow_missing=True)


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
    selected = tuple(dict.fromkeys(dataset_slug(item) and str(item) for item in datasets))
    unknown = sorted(set(selected) - set(DATASET_ORDER))
    if unknown:
        raise ValueError(f"unsupported formal dataset(s): {unknown}")
    missing: list[str] = []
    configs: list[str] = []
    for dataset in selected:
        paths = dataset_paths(root_path, dataset)
        configs.append(str(paths["config"].relative_to(root_path)))
        if not paths["config"].is_file():
            missing.append(str(paths["config"]))
        if not require_staged:
            continue
        for name in DATASET_FILES:
            if not (paths["dataset_root"] / name).is_file():
                missing.append(str(paths["dataset_root"] / name))
        if dataset == "ppi_full_formal" and not (paths["dataset_root"] / "node_ptr.txt").is_file():
            missing.append(str(paths["dataset_root"] / "node_ptr.txt"))
        for name in CHECKPOINT_FILES:
            if not (paths["checkpoint"] / name).is_file():
                missing.append(str(paths["checkpoint"] / name))
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
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()
